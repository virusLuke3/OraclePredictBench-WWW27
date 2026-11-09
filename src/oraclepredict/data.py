"""Independent replay of published UMA events, history, and chronological splits.

Raw outcomes are available only to preprocessing/evaluation. Inference consumes
the separately exported allowlist table, never this raw construction layer.
"""
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from .common import ROOT, RAW, PROCESSED, dump, utcnow

FORBIDDEN = ['log_bond_raw','log_reward_plus1_raw','reward_to_bond_ratio','adapter_version']
NATIVE_FEATURES = ['log_final_fee_raw','liveness_hours','proposal_latency_hours','ancillary_bytes','proposed_price_class']
HISTORY_FEATURES = [f'{r}_prior_{s}' for r in ['proposer','requester'] for s in ['completed','dispute_rate','rejection_rate']]
FEATURES = NATIVE_FEATURES + HISTORY_FEATURES

def priceclass(value):
    return {0:'binary_zero',10**18:'binary_one',5*10**17:'indeterminate_half',-(2**255):'early_expiration'}.get(int(value),'other')

def ordered_key(row, prefix):
    return (row[f'{prefix}_time'].value,int(row[f'{prefix}_block']),int(row[f'{prefix}_tx_index']),int(row[f'{prefix}_log_index']))

def replay_histories(population, queries):
    """Only prior settlements, ordered by (UTC, block, tx, log), within Polygon."""
    prior=population.sort_values(['settlement_time','settlement_block','settlement_tx_index','settlement_log_index'])
    completed=list(prior.to_dict('records')); j=0; counts=defaultdict(lambda:[0,0,0]); result=[]
    for q in queries.sort_values(['proposal_time','proposal_block','proposal_tx_index','proposal_log_index']).to_dict('records'):
        qkey=ordered_key(q,'proposal')
        while j<len(completed) and ordered_key(completed[j],'settlement')<qkey:
            r=completed[j]; disputed=int(pd.notna(r['dispute_tx']))
            rejected=int(disputed and int(r['resolved_price_raw'])!=int(r['proposed_price_raw']))
            for role in ['proposer','requester']:
                s=counts[(role,str(r[role]).lower())];s[0]+=1;s[1]+=disputed;s[2]+=rejected
            j+=1
        rec={'sample_id':q['oo_request_id']}
        for role in ['proposer','requester']:
            n,d,y=counts[(role,str(q[role]).lower())]
            rec.update({f'{role}_prior_completed':n,f'{role}_prior_dispute_rate':d/n if n else np.nan,f'{role}_prior_rejection_rate':y/n if n else np.nan})
        result.append(rec)
    return pd.DataFrame(result).set_index('sample_id')

def build_split(manifest, val_origin, test_origin):
    m=manifest.copy()
    m['legacy_split']=np.where(m.proposal_time<val_origin,'train',np.where(m.proposal_time<test_origin,'validation','test'))
    crossing=set(m.groupby('group_id').legacy_split.nunique().loc[lambda s:s>1].index)
    def reasons(r):
        out=[]
        if r.group_id in crossing:out.append('cross_boundary_question')
        if r.legacy_split=='train' and r.label_available_at>=val_origin:out.append('train_label_immature')
        if r.legacy_split=='validation' and r.label_available_at>=test_origin:out.append('validation_label_immature')
        return ';'.join(out)
    m['exclusion_reason']=m.apply(reasons,axis=1)
    m['split']=m.legacy_split.where(m.exclusion_reason.eq(''),'excluded')
    return m

def main():
    PROCESSED.mkdir(parents=True,exist_ok=True)
    task=pd.read_parquet(RAW/'data/prediction/uma_prediction_tasks.parquet').set_index('sample_id',drop=False)
    native=pd.read_parquet(RAW/'data/native/polygon_uma_request_rounds.parquet')
    events=pd.read_parquet(RAW/'data/native/polygon_oov2_events.parquet')
    old=pd.read_parquet(ROOT/'Data/Queried_Data/data/uma_decision_splits.parquet')
    assert task.index.is_unique and native.oo_request_id.is_unique
    joined=native.copy()
    for event,prefix in [('RequestPrice','request'),('ProposePrice','proposal'),('Settle','settlement')]:
        ev=events.loc[events.event.eq(event)].set_index('oo_request_id')
        assert ev.index.is_unique
        rename={'event_time_utc':prefix+'_time','source_block':prefix+'_block','transaction_index':prefix+'_tx_index','log_index':prefix+'_log_index','source_tx':prefix+'_source_tx'}
        joined=joined.rename(columns={v:v+'_published' for v in rename.values() if v in joined.columns})
        joined=joined.merge(ev[list(rename)].rename(columns=rename),on='oo_request_id',how='left',validate='one_to_one')
    population=joined.loc[joined.sample_tier.eq('primary')&joined.proposal_time.notna()&joined.settlement_time.notna()].copy()
    query=joined.set_index('oo_request_id').loc[task.index].rename_axis('oo_request_id').reset_index()
    assert query.question_id.notna().all()
    rec=query.set_index('oo_request_id')
    for left,right in [('decision_time_utc','proposal_time'),('request_event_time_utc','request_time'),('settlement_time_outcome_only','settlement_time')]:
        assert (task[left]==rec[right]).all(),left
    reconstructed=pd.DataFrame(index=task.index)
    request_events=events[events.event.eq('RequestPrice')].set_index('oo_request_id').loc[task.index]
    proposal_events=events[events.event.eq('ProposePrice')].set_index('oo_request_id').loc[task.index]
    settlement_events=events[events.event.eq('Settle')].set_index('oo_request_id').loc[task.index]
    reconstructed['log_final_fee_raw']=request_events.final_fee_raw.map(lambda v:math.log1p(int(v)))
    reconstructed['liveness_hours']=(proposal_events.expiration_time_utc-proposal_events.event_time_utc).dt.total_seconds()/3600
    reconstructed['proposal_latency_hours']=(proposal_events.event_time_utc-request_events.event_time_utc).dt.total_seconds()/3600
    reconstructed['ancillary_bytes']=request_events.ancillary_data_hex.map(lambda s:len(bytes.fromhex(s.removeprefix('0x'))))
    reconstructed['proposed_price_class']=proposal_events.proposed_price_raw.map(priceclass)
    history=replay_histories(population,query)
    reconstructed=reconstructed.join(history)
    checks=[]
    for f in FEATURES:
        a,b=task[f],reconstructed[f]
        match=(a.eq(b)|a.isna()&b.isna()) if f=='proposed_price_class' else pd.Series(np.isclose(a.astype(float),b.astype(float),equal_nan=True),index=a.index)
        checks.append({'feature':f,'kept':True,'mismatches':int((~match).sum()),'status':'replayed_from_pinned_native_events' if f in NATIVE_FEATURES else 'independently_replayed_past_settlements','proof':'native RequestPrice/ProposePrice + event UTC/order' if f in NATIVE_FEATURES else 'same-role retained-primary population; settlement order strictly before proposal','missing_rule':'null for no historical denominator' if '_rate' in f else 'train-only median/onehot'})
        assert match.all(),f
    for f in FORBIDDEN+['requester_is_polymarket_adapter']:
        checks.append({'feature':f,'kept':False,'mismatches':None,'status':'excluded','proof':'historical parameter binding missing' if f in FORBIDDEN else 'historical membership not independently bound; constant type feature omitted','missing_rule':'not supplied through any inference route'})
    pd.DataFrame(checks).to_csv(ROOT/'metadata/feature_availability_audit.csv',index=False)
    # Independently verify raw integer label identity with the pinned Settle event.
    y=np.array([int(int(r)!=int(p)) for r,p in zip(settlement_events.resolved_price_raw,proposal_events.proposed_price_raw)])
    assert np.array_equal(y,task.proposal_rejected_by_protocol.to_numpy())
    manifest=pd.DataFrame({'sample_id':task.index,'group_id':'polygon:oov2:'+rec.question_id.astype(str),'proposal_time':rec.proposal_time,'label_available_at':rec.settlement_time}).reset_index(drop=True)
    old_times=task[['sample_id','decision_time_utc']].reset_index(drop=True).merge(old[['sample_id','split']],on='sample_id',validate='one_to_one')
    val_origin=old_times.loc[old_times.split.eq('validation'),'decision_time_utc'].min()
    test_origin=old_times.loc[old_times.split.eq('test'),'decision_time_utc'].min()
    manifest=build_split(manifest,val_origin,test_origin)
    manifest.to_parquet(ROOT/'metadata/split_manifest.parquet',index=False)
    manifest.to_csv(ROOT/'metadata/split_manifest.csv',index=False)
    groups=manifest.groupby('group_id').agg(requests=('sample_id','size'),first_proposal=('proposal_time','min'),last_proposal=('proposal_time','max'),legacy_partitions=('legacy_split',lambda s:','.join(sorted(set(s)))),new_partitions=('split',lambda s:','.join(sorted(set(s)))))
    groups.to_csv(ROOT/'metadata/group_audit.csv')
    features=reconstructed.copy();features.insert(0,'sample_id',features.index);features=features.reset_index(drop=True)
    features.to_parquet(PROCESSED/'features.parquet',index=False)
    labels=pd.DataFrame({'sample_id':task.index,'label':y,'resolution_class':task.resolution_class_outcome_only}).reset_index(drop=True)
    for s in ['train','validation','test']:
        ids=manifest.loc[manifest.split.eq(s),'sample_id']
        labels[labels.sample_id.isin(ids)].to_parquet(PROCESSED/f'{s}_labels.parquet',index=False)
    summary=[]
    for s,d in manifest.merge(labels,on='sample_id').groupby('split',sort=False):
        summary.append({'split':s,'N':len(d),'G':d.group_id.nunique(),'positive':int(d.label.sum()),'first_proposal':str(d.proposal_time.min()),'last_proposal':str(d.proposal_time.max()),'latest_label':str(d.label_available_at.max())})
    pd.DataFrame(summary).to_csv(ROOT/'metadata/sample_flow.csv',index=False)
    # Select pilot before model fitting, from validation only, evenly across time.
    val=manifest[manifest.split.eq('validation')].sort_values(['proposal_time','sample_id'])
    pilot=val.iloc[np.linspace(0,len(val)-1,min(32,len(val))).round().astype(int)]
    pilot.to_csv(ROOT/'metadata/pilot_manifest.csv',index=False)
    split_spec={'validation_origin':str(val_origin),'test_origin':str(test_origin),'rule':'purge entire cross-boundary question, then exclude labels not settled before next original boundary','legacy_public_test_reused':True,'label_availability':'conservative Settle event time','group_rule':'polygon:oov2:question_id; no independently verified wider event mapping','frozen_weights':True,'history_population':len(population)}
    dump(ROOT/'configs/task.json',{'task':'conditional eventual protocol integer disagreement','retrospective_cohort':True,'independent_truth':False,'features':FEATURES,'excluded':FORBIDDEN+['requester_is_polymarket_adapter'],'split':split_spec,'fallback':'training_class_prior','main_metric':'binary_brier','main_comparison':'A2 minus validation-selected B*','prediction_intervals':'not applicable to binary probability outputs','created_at':utcnow()})
    crossing=groups[groups.legacy_partitions.str.contains(',')]
    crossing.to_csv(ROOT/'metadata/crossing_questions.csv')
    audit={'executed_at':utcnow(),'published_task_N':len(task),'published_G':query.question_id.nunique(),'history_population_N':len(population),'feature_matches':{r['feature']:r['mismatches'] for r in checks if r['kept']},'label_mismatches':int(np.sum(y!=task.proposal_rejected_by_protocol.to_numpy())),'crossing_questions':len(crossing),'exclusion_reasons':manifest.exclusion_reason.value_counts().to_dict(),'partitions':summary,'limitations':['Validation replays published native decoded events, not independent blockchain RPC receipts.','Cohort membership/link/transfer reconciliation relies on publisher QC.','Question is minimum dependency unit; wider event dependence not excluded.','Old test cohort was public and previously studied.','Four unbound parameter features and unverified adapter membership excluded.']}
    dump(ROOT/'metadata/audit.json',audit)
    print(pd.DataFrame(summary).to_string(index=False)); print('11 feature values independently match published inputs; no model performance evaluated.')

if __name__=='__main__':main()
