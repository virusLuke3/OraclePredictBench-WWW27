"""Case-weighted paired cluster inference, with repeated losses kept within case."""
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from .common import ROOT,PROCESSED,dump,digest,utcnow

def cluster_bootstrap(loss, groups, draws=9999,seed=20261102):
    """Returns joint draws for all methods; group copies remain separate in SE."""
    loss=np.asarray(loss,float)
    if loss.ndim==1:loss=loss[:,None]
    codes,unique=pd.factorize(np.asarray(groups),sort=True);G=len(unique);n=np.bincount(codes).astype(float)
    sums=np.zeros((G,loss.shape[1]));np.add.at(sums,codes,loss)
    indices=np.random.default_rng(seed).integers(0,G,size=(draws,G))
    sizes=n[indices];den=sizes.sum(axis=1);totals=sums[indices];means=totals.sum(axis=1)/den[:,None]
    return {'means':means,'indices':indices,'N':den,'sizes':sizes,'sums':sums,'n':n,'group_labels':unique,'G':G,'N_original':len(loss),'point':loss.mean(axis=0)}

def paired_test(draws,left,right):
    delta=float(draws['point'][left]-draws['point'][right]);G=draws['G'];N=draws['N_original']
    sums=draws['sums'][:,left]-draws['sums'][:,right];n=draws['n']
    sampled=draws['means'][:,left]-draws['means'][:,right]
    pct=np.quantile(sampled,[.025,.975])
    out={'delta':delta,'ci_low':float(pct[0]),'ci_high':float(pct[1]),'G':G,'p_raw':None,'studentized_ci_low':None,'studentized_ci_high':None}
    if G<2:return {**out,'inference_status':'insufficient_groups'}
    se=np.sqrt(G/(G-1)*np.square(sums-n*delta).sum()/N**2)
    residual=sums[draws['indices']]-draws['sizes']*sampled[:,None]
    boot_se=np.sqrt(G/(G-1)*np.square(residual).sum(axis=1)/draws['N']**2)
    degenerate=(boot_se<=1e-15)|~np.isfinite(boot_se)
    out.update(se=float(se),degenerate_draws=int(degenerate.sum()))
    if se<=1e-15 or degenerate.any() or G<30:return {**out,'inference_status':'exploratory_no_p'}
    t=(sampled-delta)/boot_se;q=np.quantile(t,[.025,.975])
    out.update(p_raw=float((1+(np.abs(t)>=abs(delta/se)).sum())/(len(t)+1)),studentized_ci_low=float(delta-q[1]*se),studentized_ci_high=float(delta-q[0]*se),inference_status='approximate_conditional_group_inference')
    return out

def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[None]*len(values);running=0
    for rank,i in enumerate(order):running=max(running,(len(values)-rank)*values[i]);out[i]=min(1.,running)
    return out

def calendar_supergroups(manifest,days):
    """Coalesce calendar intervals touched by a question without splitting it."""
    bins=(pd.to_datetime(manifest.proposal_time,utc=True).astype('int64')//(days*86400*10**9)).to_numpy()
    t=pd.DataFrame({'group':manifest.group_id.to_numpy(),'bin':bins});span=t.groupby('group').bin.agg(['min','max']).sort_values('min')
    assigned={};block=-1;end=None
    for group,row in span.iterrows():
        if end is None or row['min']>end:block+=1;end=row['max']
        else:end=max(end,row['max'])
        assigned[group]=block
    return manifest.group_id.map(assigned).to_numpy()

def main():
    cfg=json.loads((ROOT/'configs/statistics.json').read_text());selection=json.loads((ROOT/'metadata/selected_model.json').read_text())
    base=pd.read_parquet(ROOT/'runs/baseline_test.parquet');agent=pd.read_parquet(ROOT/'runs/agents_test.parquet')
    predictions=pd.concat([base,agent],ignore_index=True)
    labels=pd.read_parquet(PROCESSED/'test_labels.parquet');manifest=pd.read_parquet(ROOT/'metadata/split_manifest.parquet').query("split=='test'").sort_values(['proposal_time','sample_id'])
    ids=manifest.sample_id.tolist();methods=sorted(predictions.method.unique());N=len(ids)
    for m,d in predictions.groupby('method'):
        assert set(d.sample_id)==set(ids),m
        assert d.groupby('sample_id').size().nunique()==1,m
        assert not d.duplicated(['sample_id','repeat']).any(),m
    predictions=predictions.merge(labels,on='sample_id',validate='many_to_one')
    predictions['loss']=(predictions.p_used-predictions.label)**2
    eps=cfg['log_loss_epsilon'];p=predictions.p_used.clip(eps,1-eps)
    predictions['log_loss']=-(predictions.label*np.log(p)+(1-predictions.label)*np.log(1-p))
    loss=predictions.groupby(['sample_id','method']).loss.mean().unstack().loc[ids,methods]
    boot=cluster_bootstrap(loss.to_numpy(),manifest.group_id,draws=cfg['bootstrap_draws'],seed=cfg['bootstrap_seed'])
    bstar=next(m for m,model in selection['methods'].items() if model==selection['best_model']);bidx=methods.index(bstar)
    np.savez_compressed(ROOT/'Results/Source_Data/bootstrap_indices.npz',indices=boot['indices'],N=boot['N'],group_labels=np.asarray(boot['group_labels'],dtype=str))
    bd=pd.DataFrame(boot['means'],columns=methods);bd.insert(0,'actual_N',boot['N']);bd.insert(0,'draw',np.arange(len(bd)));bd.to_parquet(ROOT/'Results/Source_Data/bootstrap_results.parquet',index=False)
    rows=[]
    for j,m in enumerate(methods):
        d=predictions[predictions.method.eq(m)];ci=np.quantile(boot['means'][:,j],[.025,.975]);comparison=paired_test(boot,j,bidx)
        secondary=[]
        for _,r in d.groupby('repeat'):
            secondary.append({'AUROC':roc_auc_score(r.label,r.p_used) if r.label.nunique()==2 else np.nan,'AveragePrecision':average_precision_score(r.label,r.p_used) if r.label.nunique()==2 else np.nan,'balanced_accuracy':balanced_accuracy_score(r.label,r.p_used>=.5)})
        rows.append({'method':m,'N':N,'G':boot['G'],'repeats':d.repeat.nunique(),'brier':float(loss[m].mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'delta_vs_best':comparison['delta'],'delta_ci_low':comparison['ci_low'],'delta_ci_high':comparison['ci_high'],'log_loss':d.log_loss.mean(),**pd.DataFrame(secondary).mean().to_dict(),'latency_p50_ms':d.total_latency_ms.median(),'latency_p95_ms':d.total_latency_ms.quantile(.95),'fallback_rate':d.fallback_used.mean(),'mean_input_tokens':d.input_tokens.mean() if 'input_tokens' in d else None,'mean_output_tokens':d.output_tokens.mean() if 'output_tokens' in d else None,'mean_llm_calls':d.llm_calls.mean(),'mean_tool_calls':d.tool_calls.mean(),'cost_usd':None,'cost_status':'hardware time, no dollar tariff assumed'})
        if m in {'D0','A1','A2','D_match'}:
            complete_usage=int(d.usage_complete.eq(True).sum())
            rows[-1].update(token_usage_complete_outputs=complete_usage,token_usage_incomplete_outputs=len(d)-complete_usage,mean_token_scope='Only outputs with complete provider usage; all outputs remain in score and latency denominators')
        else:
            rows[-1].update(token_usage_complete_outputs=None,token_usage_incomplete_outputs=None,mean_token_scope='Not applicable: no LLM calls')
    results=pd.DataFrame(rows);results.to_csv(ROOT/'Results/Tables/main_results.csv',index=False)
    comps=[]
    for a,b,kind in [('A2',bstar,'primary')]+[(a,b,'secondary') for a,b in cfg['secondary_family']]:
        comps.append({'left':a,'right':b,'family':kind,**paired_test(boot,methods.index(a),methods.index(b))})
    sec=[i for i,v in enumerate(comps) if v['family']=='secondary']
    if all(comps[i]['p_raw'] is not None for i in sec):
        for i,p_adj in zip(sec,holm([comps[i]['p_raw'] for i in sec])):comps[i]['p_holm']=p_adj
    pd.DataFrame(comps).to_csv(ROOT/'Results/Tables/comparisons.csv',index=False)
    sensitivity=[]
    for days in cfg['calendar_days']:
        groups=calendar_supergroups(manifest,days);cb=cluster_bootstrap(loss.to_numpy(),groups,cfg['bootstrap_draws'],cfg['bootstrap_seed'])
        for a,b,kind in [('A2',bstar,'primary')]+[(a,b,'secondary') for a,b in cfg['secondary_family']]:
            sensitivity.append({'calendar_days':days,'left':a,'right':b,**paired_test(cb,methods.index(a),methods.index(b))})
    pd.DataFrame(sensitivity).to_csv(ROOT/'Results/Tables/calendar_sensitivity.csv',index=False)
    groupweighted=loss.assign(group=manifest.group_id.to_numpy()).groupby('group').mean().mean();groupweighted.rename('group_weighted_brier').to_csv(ROOT/'Results/Tables/group_weighted.csv')
    diagnostic=manifest[['sample_id','group_id','proposal_time']].merge(labels,on='sample_id').merge(pd.read_parquet(PROCESSED/'features.parquet')[['sample_id','proposer_prior_completed']],on='sample_id').set_index('sample_id').loc[ids]
    diagnostic['period']=np.where(np.arange(N)<N//2,'first_half','second_half');diagnostic['history']=np.where(diagnostic.proposer_prior_completed.eq(0),'no_prior','has_prior')
    stability=[]
    for column in ['period','history','resolution_class']:
        for value,d in diagnostic.groupby(column):
            for m in methods:stability.append({'stratum':column,'value':value,'method':m,'N':len(d),'G':d.group_id.nunique(),'positive':int(d.label.sum()),'brier':float(loss.loc[d.index,m].mean())})
    pd.DataFrame(stability).to_csv(ROOT/'Results/Tables/stability.csv',index=False)
    predictions.groupby(['method','repeat']).agg(N=('sample_id','size'),brier=('loss','mean'),fallback=('fallback_used','mean')).to_csv(ROOT/'Results/Tables/run_stability.csv')
    predictions.to_parquet(ROOT/'Results/Source_Data/predictions_scored.parquet',index=False)
    probs=predictions.groupby(['sample_id','method']).p_used.agg(['mean','std']);probs.to_csv(ROOT/'Results/Tables/probability_stability.csv')
    group_sizes=manifest.groupby('group_id').size()
    summary={'N':N,'G':boot['G'],'positive':int(labels.label.sum()),'best_baseline_family':bstar,'best_baseline_model':selection['best_model'],'bootstrap_draws':cfg['bootstrap_draws'],'max_group_size':int(group_sizes.max()),'group_sizes':group_sizes.value_counts().sort_index().to_dict(),'primary_comparison':comps[0],'inference_scope':cfg['interpretation'],'calendar_blocks':{str(d):int(calendar_supergroups(manifest,d).max()+1) for d in cfg['calendar_days']}}
    summary['provenance']={'evaluated_at':utcnow(),'protocol_sha256':digest(ROOT/'metadata/frozen_protocol.json'),'agent_predictions_sha256':digest(ROOT/'runs/agents_test.parquet'),'baseline_predictions_sha256':digest(ROOT/'runs/baseline_test.parquet'),'test_labels_sha256':digest(PROCESSED/'test_labels.parquet'),'statistics_config_sha256':digest(ROOT/'configs/statistics.json')}
    dump(ROOT/'Results/Source_Data/statistical_summary.json',summary);print(results[['method','brier','ci_low','ci_high','delta_vs_best','fallback_rate']].to_string(index=False));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
