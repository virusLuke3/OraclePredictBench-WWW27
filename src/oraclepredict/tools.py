"""Minimal read-only service: no labels, files, free-form code, or networking."""
import hashlib, json, operator
import numpy as np
import pandas as pd
from .data import FEATURES

class EvidenceStore:
    def __init__(self, features, cutoffs, predictor, prior):
        if list(features.columns)!=FEATURES:raise ValueError('Non-allowlisted input')
        self.features=features.copy();self.cutoffs=cutoffs.copy();self.predictor=predictor;self.prior=prior
    def snapshot(self,case_id):
        row=self.features.loc[case_id]
        values={k:None if pd.isna(v) else (v.item() if hasattr(v,'item') else v) for k,v in row.items()}
        return {'evidence_id':'input','as_of':str(self.cutoffs.loc[case_id]),'features':values,'training_prior':self.prior}
    def session(self,case_id):return ToolSession(self,case_id)

class ToolSession:
    def __init__(self,store,case_id):self.store=store;self.case_id=case_id;self.packet=store.snapshot(case_id);self.payloads=[]
    def execute(self,name,args):
        if not isinstance(args,dict):raise ValueError('Arguments must be an object')
        allowed={'get_proposal':set(),'get_actor_history':{'role'},'predict_tabular':set(),'calculate':{'left','right','op'},'validate_evidence':{'evidence_ids'}}
        if name not in allowed or set(args)-allowed[name]:raise ValueError('Unknown tool or prohibited argument')
        if name=='get_proposal':result={'features':self.packet['features']}
        elif name=='get_actor_history':
            role=args.get('role')
            if role not in ['proposer','requester']:raise ValueError('Invalid role')
            result={k:v for k,v in self.packet['features'].items() if k.startswith(role+'_prior_')}
        elif name=='predict_tabular':
            p=float(self.store.predictor.predict(self.store.features.loc[[self.case_id]])[0]);result={'model_id':self.store.predictor.selection['best_model'],'p_reject':p,'training_only_fit':True,'validation_selected':True}
        elif name=='calculate':
            vals={**self.packet['features'],'training_prior':self.store.prior}
            a,b=vals[args['left']],vals[args['right']]
            if not isinstance(a,(int,float)) or not isinstance(b,(int,float)):raise ValueError('Unavailable numeric input')
            ops={'add':operator.add,'subtract':operator.sub,'multiply':operator.mul,'divide':operator.truediv}
            val=ops[args['op']](a,b)
            if not np.isfinite(val):raise ValueError('Nonfinite calculation')
            result={'value':val,'operands':args}
        else:
            known={'input',*[r['evidence_id'] for r in self.payloads]};ids=args.get('evidence_ids',[])
            if not isinstance(ids,list):raise ValueError('IDs must be a list')
            result={'valid':all(x in known for x in ids),'cutoff_enforced_by_server':True}
        record={'evidence_id':f'tool_{len(self.payloads)+1}','tool':name,'as_of':self.packet['as_of'],'result':result}
        record['sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest();self.payloads.append(record)
        return record

def validate_probability(obj,allowed_ids):
    if not isinstance(obj,dict):raise ValueError('Output must be object')
    p=obj.get('p_reject')
    if isinstance(p,bool) or not isinstance(p,(int,float)) or not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
    ids=obj.get('evidence_ids',[])
    if not isinstance(ids,list) or any(not isinstance(x,str) or x not in allowed_ids for x in ids):raise ValueError('Invalid evidence reference')
    return float(p)
