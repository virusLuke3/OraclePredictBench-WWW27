import json
import numpy as np
import pandas as pd
import pytest
from oraclepredict.common import ROOT,PROCESSED
from oraclepredict.data import FEATURES,FORBIDDEN,build_split,replay_histories,priceclass
from oraclepredict.metrics import binary_brier,BRIER
from oraclepredict.tools import EvidenceStore,validate_probability
from oraclepredict.baselines import FrozenPredictor,load_partition
from oraclepredict.evaluate import cluster_bootstrap,holm,calendar_supergroups
from oraclepredict.agents import usage

def test_autogluon_cannot_fit_validation_only_categories_or_variation(tmp_path):
    from autogluon.tabular import TabularPredictor
    from oraclepredict.baselines import training_feature_generator
    train=pd.DataFrame({'kind':['a','b','c']*8,'constant':[1]*24,'label':[0,1]*12})
    val=pd.DataFrame({'kind':['validation_only']*6,'constant':[10,20,30,40,50,60],'label':[0,1]*3})
    generator=training_feature_generator(train.drop(columns='label'))
    predictor=TabularPredictor(label='label',path=str(tmp_path/'ag'),verbosity=0).fit(train_data=train,tuning_data=val,feature_generator=generator,hyperparameters={'RF':{'n_estimators':2}},fit_weighted_ensemble=False,calibrate=False,calibrate_decision_threshold=False,num_cpus=1)
    transformed=predictor.transform_features(val.drop(columns='label'))
    assert 'constant' not in transformed.columns
    assert transformed['kind'].isna().all()
    assert set(generator.features_in)=={'kind'}

def test_real_provider_nullable_usage_details():
    result=usage([{'response':{'usage':{'prompt_tokens':25,'completion_tokens':10,'prompt_tokens_details':None,'reasoning_tokens':0}}}])
    assert result['cached_input_tokens'] is None and result['provider_reported_reasoning_tokens']==0
    assert result['input_tokens']==25
    missing=usage([{'error':'timeout'}]);assert missing['input_tokens'] is None and not missing['usage_complete']

def test_raw_integer_special_values():
    assert priceclass(-(2**255))=='early_expiration'
    assert priceclass(5*10**17)=='indeterminate_half'
    assert int(str(10**18))!=int(str(10**18+1))

def test_brier_probability_semantics():
    assert binary_brier([0,1],[0,1])==0
    assert binary_brier([0,1],[1,0])==1
    assert binary_brier([0,1],[.5,.5])==.25
    assert BRIER([0,1],np.array([.5,.5]))==-.25
    assert binary_brier([0,1],[[.7,.3],[.2,.8]])==pytest.approx(.065)
    for p in [[np.nan,.2],[1.1,.2]]:
        with pytest.raises(ValueError):binary_brier([0,1],p)

def history_row(sid,pday,sday,res='0',tx=0):
    return {'oo_request_id':sid,'proposal_time':pd.Timestamp(pday,tz='UTC'),'settlement_time':pd.Timestamp(sday,tz='UTC'),'proposal_block':1,'settlement_block':1,'proposal_tx_index':tx,'settlement_tx_index':tx,'proposal_log_index':1,'settlement_log_index':2,'proposer':'actor','requester':'requester','dispute_tx':'d','proposed_price_raw':'0','resolved_price_raw':res}

def test_current_outcome_and_future_rows_cannot_change_snapshot():
    prior=history_row('old','2024-01-01','2024-01-02')
    current=history_row('current','2024-01-03','2024-01-05','1')
    f=pd.DataFrame([prior,current]);q=pd.DataFrame([current])
    a=replay_histories(f,q)
    f.loc[f.oo_request_id.eq('current'),'resolved_price_raw']='0'
    f=pd.concat([f,pd.DataFrame([history_row('future','2024-01-06','2024-01-07','1')])])
    b=replay_histories(f,q)
    pd.testing.assert_frame_equal(a,b)
    assert a.loc['current','proposer_prior_completed']==1
    assert a.loc['current','proposer_prior_rejection_rate']==0
    empty=replay_histories(pd.DataFrame([current]),q)
    assert pd.isna(empty.loc['current','proposer_prior_rejection_rate'])

def test_purge_and_maturity_are_enforced():
    dates=pd.to_datetime(['2024-01-01','2024-02-02','2024-01-03','2024-03-02'],utc=True)
    m=pd.DataFrame({'sample_id':list('abcd'),'group_id':['cross','cross','late','test'],'proposal_time':dates,'label_available_at':pd.to_datetime(['2024-01-05','2024-02-03','2024-02-04','2024-03-03'],utc=True)})
    out=build_split(m,pd.Timestamp('2024-02-01',tz='UTC'),pd.Timestamp('2024-03-01',tz='UTC'))
    assert out.split.tolist()==['excluded','excluded','excluded','test']
    actual=pd.read_parquet(ROOT/'metadata/split_manifest.parquet');cfg=json.loads((ROOT/'configs/task.json').read_text())['split']
    retained=actual[actual.split.ne('excluded')]
    assert retained.groupby('group_id').split.nunique().max()==1
    assert (retained.loc[retained.split.eq('train'),'label_available_at']<pd.Timestamp(cfg['validation_origin'])).all()
    assert (retained.loc[retained.split.eq('validation'),'label_available_at']<pd.Timestamp(cfg['test_origin'])).all()

def test_tool_input_identity_cutoff_and_predictor_isolation():
    x=load_partition('validation');manifest=pd.read_parquet(ROOT/'metadata/split_manifest.parquet').set_index('sample_id');p=FrozenPredictor()
    store=EvidenceStore(x,manifest.proposal_time,p,p.selection['training_prior']);sid=x.index[0];s=store.session(sid)
    packet=s.packet;assert set(packet['features'])==set(FEATURES)
    assert not set(FORBIDDEN)&set(packet['features'])
    with pytest.raises(ValueError):s.execute('get_proposal',{'cutoff':'2099-01-01'})
    with pytest.raises(ValueError):s.execute('predict_tabular',{'case_id':'other'})
    with pytest.raises(ValueError):s.execute('get_outcome',{})
    with pytest.raises(ValueError):load_partition('test',labels=True)
    payload=s.execute('predict_tabular',{})
    assert payload['result']['p_reject']==pytest.approx(p.predict(x.iloc[:1])[0])
    with pytest.raises(ValueError):validate_probability({'p_reject':.5,'evidence_ids':['future']},{'input'})

def test_bootstrap_real_denominator_joint_draws_and_group_copies():
    loss=np.array([[0.,.2],[0.,.2],[1.,.6]])
    b=cluster_bootstrap(loss,['A','A','B'],draws=100,seed=1)
    assert set(b['N'])=={2.,3.,4.}
    for k,index in enumerate(b['indices']):
        values=np.concatenate([loss[:2] if i==0 else loss[2:] for i in index])
        assert np.allclose(b['means'][k],values.mean(axis=0))
    assert np.mean((np.array([0.,1.])-0)**2)==.5 # repeat-mean loss, not squared mean p=.25
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])

def test_calendar_blocks_never_split_question():
    m=pd.DataFrame({'group_id':['A','A','B','C'],'proposal_time':pd.to_datetime(['2024-01-01','2024-02-01','2024-01-15','2024-03-01'],utc=True)})
    block=calendar_supergroups(m,7)
    assert block[0]==block[1]==block[2] and block[3]!=block[0]
