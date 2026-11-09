"""Explicit temporal validation, small audited search, and fixed fitted artifacts."""
import argparse, importlib.metadata, json, platform, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from .common import ROOT, PROCESSED, dump, utcnow
from .data import FEATURES
from .metrics import BRIER,binary_brier

def load_partition(split, labels=False):
    manifest=pd.read_parquet(ROOT/'metadata/split_manifest.parquet')
    ids=manifest.loc[manifest.split.eq(split),'sample_id']
    frame=pd.read_parquet(PROCESSED/'features.parquet').set_index('sample_id').loc[ids]
    assert list(frame.columns)==FEATURES
    if not labels:return frame
    if split=='test':raise ValueError('Training module cannot load test labels')
    y=pd.read_parquet(PROCESSED/f'{split}_labels.parquet').set_index('sample_id').loc[ids,'label']
    return frame,y

def training_feature_generator(train_x):
    generator=AutoMLPipelineFeatureGenerator()
    generator.fit_transform(train_x.copy())
    assert generator.is_fit()
    return generator

def train():
    directory=ROOT/'models';directory.mkdir(exist_ok=True)
    if (directory/'selection.json').exists():raise RuntimeError('Frozen model selection exists; use a separate run directory to retrain')
    train_x,train_y=load_partition('train',True);val_x,val_y=load_partition('validation',True)
    cfg={'seed':17,'num_cpus':8,'budget_seconds':3600,'lr_C':[0.01,0.1,1,10],'rf_depth':[4,8],'rf_leaf':[3,10],'gbm_leaves':[7,15],'gbm_min_leaf':[10,30],'gbm_max_rounds':1000,'calibration':False,'bagging':False,'stacking':False,'refit':False,'feature_generator_fit':'explicit training inputs only; already-fitted generator passed to AutoGluon'}
    dump(ROOT/'configs/baseline.json',cfg)
    num=[f for f in FEATURES if f!='proposed_price_class'];cat=['proposed_price_class']
    prior=float(train_y.mean()); records=[]; valpred=[]; start=time.perf_counter()
    records.append({'model':'Prior','family':'Prior','validation_brier':binary_brier(val_y,np.full(len(val_y),prior)),'fit_seconds':0.,'validation_prediction_seconds':0.,'status':'success'})
    valpred.append(pd.DataFrame({'sample_id':val_x.index,'model':'Prior','p':prior}))
    for C in cfg['lr_C']:
        name=f'LR_C{C:g}'
        processor=ColumnTransformer([('numeric',Pipeline([('missing',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
        model=Pipeline([('features',processor),('classifier',LogisticRegression(C=C,solver='lbfgs',max_iter=3000,random_state=17))])
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            model.fit(train_x,train_y)
        elapsed=time.perf_counter()-t;t=time.perf_counter();p=model.predict_proba(val_x)[:,list(model.classes_).index(1)];predtime=time.perf_counter()-t
        joblib.dump(model,directory/f'{name}.joblib')
        records.append({'model':name,'family':'LR','validation_brier':binary_brier(val_y,p),'fit_seconds':elapsed,'validation_prediction_seconds':predtime,'status':'success','warnings':';'.join(str(w.message) for w in caught),'iterations':int(model[-1].n_iter_[0])})
        valpred.append(pd.DataFrame({'sample_id':val_x.index,'model':name,'p':p}))
    rf=[dict(n_estimators=500,max_depth=d,min_samples_leaf=l,max_features='sqrt',bootstrap=True,criterion='gini',random_state=17,ag_args={'name_suffix':f'_d{d}_l{l}'}) for d in cfg['rf_depth'] for l in cfg['rf_leaf']]
    gbm=[dict(objective='binary',boosting_type='gbdt',learning_rate=0.03,num_leaves=n,min_data_in_leaf=l,num_boost_round=1000,seed=17,ag_args={'name_suffix':f'_n{n}_l{l}'}) for n in cfg['gbm_leaves'] for l in cfg['gbm_min_leaf']]
    dump(ROOT/'metadata/planned_candidates.json',{'RF':rf,'GBM':gbm,'LR':cfg['lr_C'],'Prior':True})
    tx=train_x.reset_index(drop=True).assign(label=train_y.to_numpy());vx=val_x.reset_index(drop=True).assign(label=val_y.to_numpy())
    generator=training_feature_generator(train_x.reset_index(drop=True))
    pd.DataFrame({'sample_id':train_x.index,'fit_role':'feature_generator_train_only'}).to_csv(ROOT/'metadata/preprocessor_fit_manifest.csv',index=False)
    predictor=TabularPredictor(label='label',problem_type='binary',positive_class=1,eval_metric=BRIER,path=str(directory/'autogluon'),log_to_file=True).fit(train_data=tx,tuning_data=vx,feature_generator=generator,hyperparameters={'RF':rf,'GBM':gbm},time_limit=3600,auto_stack=False,num_bag_folds=0,num_stack_levels=0,dynamic_stacking=False,fit_weighted_ensemble=True,refit_full=False,calibrate=False,calibrate_decision_threshold=False,fit_strategy='sequential',num_cpus=8,num_gpus=0)
    lb=predictor.leaderboard(extra_info=True,silent=True);lb.to_csv(ROOT/'metadata/leaderboard_autogluon.csv',index=False)
    # Save actual serializable defaults/fit metadata rather than claiming presets.
    info=predictor.info();(ROOT/'metadata/model_info.json').write_text(json.dumps(info,indent=2,default=str)+'\n')
    actual={};weights={}
    for name in predictor.model_names():
        actual[name]=predictor.model_hyperparameters(name,output_format='all')
        t=time.perf_counter();p=predictor.predict_proba(val_x,model=name)[1].to_numpy();predtime=time.perf_counter()-t
        row=lb.set_index('model').loc[name]
        family='Ensemble' if name.startswith('Weighted') else ('RF' if name.startswith('RandomForest') else 'GBM')
        records.append({'model':name,'family':family,'validation_brier':binary_brier(val_y,p),'fit_seconds':float(row.fit_time),'validation_prediction_seconds':predtime,'status':'success'})
        assert np.isclose(-float(row.score_val),binary_brier(val_y,p),atol=1e-7)
        valpred.append(pd.DataFrame({'sample_id':val_x.index,'model':name,'p':p}))
        if family=='Ensemble':
            # Version-pinned private adapter, independently checked below.
            model=predictor._trainer.load_model(name);w=model._get_model_weights();weights[name]=w
            assert all(v>=0 for v in w.values()) and np.isclose(sum(w.values()),1)
            rebuilt=sum(float(v)*predictor.predict_proba(val_x,model=k)[1].to_numpy() for k,v in w.items())
            assert np.allclose(rebuilt,p,atol=1e-6)
    (ROOT/'metadata/hyperparameters_all.json').write_text(json.dumps(actual,indent=2,default=str)+'\n')
    dump(ROOT/'metadata/ensemble_weights.json',weights)
    try: failures=predictor.model_failures()
    except AttributeError:failures=pd.DataFrame()
    failures.to_csv(ROOT/'metadata/model_failures.csv',index=False)
    leaderboard=pd.DataFrame(records).sort_values(['validation_brier','validation_prediction_seconds','model']);leaderboard.to_csv(ROOT/'metadata/leaderboard_validation.csv',index=False)
    pd.concat(valpred,ignore_index=True).to_parquet(ROOT/'runs/predictions_validation.parquet',index=False)
    selected=leaderboard.groupby('family',sort=False).head(1)
    methods={r.family:r.model for r in selected.itertuples()};best=leaderboard.iloc[0].model
    selection={'selected_at':utcnow(),'best_model':best,'best_family':leaderboard.iloc[0].family,'validation_brier':float(leaderboard.iloc[0].validation_brier),'training_prior':prior,'methods':methods,'selection_rule':'minimum validation Brier; ties by measured validation prediction latency then model name','training_search_seconds':time.perf_counter()-start,'seed':17,'model_repeats':1}
    dump(directory/'selection.json',selection);dump(ROOT/'metadata/selected_model.json',selection)
    dump(ROOT/'metadata/environment.json',{'python':platform.python_version(),'platform':platform.platform(),'packages':{m:importlib.metadata.version(m) for m in ['autogluon.tabular','scikit-learn','lightgbm','pandas','numpy','openai']},'cpus':8,'gpu_for_baselines':False})
    print(leaderboard[['model','family','validation_brier']].to_string(index=False));print('Selected B*:',best)

class FrozenPredictor:
    def __init__(self):
        self.selection=json.loads((ROOT/'models/selection.json').read_text());self.ag=None;self.cache={}
    def predict(self,frame,model=None):
        if list(frame.columns)!=FEATURES:raise ValueError('Input is not exactly approved allowlist')
        model=model or self.selection['best_model']
        if model=='Prior':return np.full(len(frame),self.selection['training_prior'])
        if model.startswith('LR_'):
            if model not in self.cache:self.cache[model]=joblib.load(ROOT/'models'/f'{model}.joblib')
            m=self.cache[model];return m.predict_proba(frame)[:,list(m.classes_).index(1)]
        if self.ag is None:self.ag=TabularPredictor.load(str(ROOT/'models/autogluon'))
        return self.ag.predict_proba(frame,model=model)[1].to_numpy()

def predict_partition(split):
    if split=='test' and not (ROOT/'metadata/frozen_protocol.json').exists():raise RuntimeError('Freeze before test inference')
    frame=load_partition(split);p=FrozenPredictor();records=[]
    for family,model in p.selection['methods'].items():
        # Comparable warm, sequential one-request latency; initialization excluded and disclosed.
        p.predict(frame.iloc[:1],model)
        for sid in frame.index:
            t=time.perf_counter();start=utcnow()
            try:prob=float(p.predict(frame.loc[[sid]],model)[0]);error=None;fallback=False
            except Exception as e:prob=p.selection['training_prior'];error=type(e).__name__;fallback=True
            records.append(dict(sample_id=sid,method=family,model=model,split=split,repeat=0,p_used=prob,fallback_used=fallback,error_type=error,total_latency_ms=1000*(time.perf_counter()-t),start_utc=start,end_utc=utcnow(),llm_calls=0,tool_calls=0,cost_usd=None,cost_status='local CPU time; no rental tariff assumed'))
    pd.DataFrame(records).to_parquet(ROOT/f'runs/baseline_{split}.parquet',index=False)
    print('Predicted',split,len(records),'method-case rows; no test labels loaded.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['train','predict']);parser.add_argument('--split',default='validation');args=parser.parse_args()
    train() if args.action=='train' else predict_partition(args.split)
