"""Real local-LLM experiment; raw HTTP calls retained, no simulated predictions."""
import argparse, concurrent.futures, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from .common import ROOT, PROCESSED, utcnow, dump, digest
from .baselines import FrozenPredictor
from .tools import EvidenceStore,validate_probability

PROMPTS={p.stem:p.read_text() for p in (ROOT/'prompts').glob('*.txt')}

class Caller:
    def __init__(self,cfg,seed,call_log_dir=None):self.cfg=cfg;self.seed=seed;self.calls=[];self.call_log_dir=call_log_dir
    def save_call(self,record):
        self.calls.append(record)
        if self.call_log_dir is not None:
            import uuid
            dump(self.call_log_dir/f'{uuid.uuid4().hex}.json',record)
    def call(self,role,contents,history=None):
        messages=[{'role':'system','content':PROMPTS['common']+'\n'+PROMPTS[role]},{'role':'user','content':json.dumps(contents,allow_nan=False)}]
        if history:messages+=history
        payload={'model':self.cfg['model_id'],'messages':messages,'temperature':self.cfg['temperature'],'top_p':self.cfg['top_p'],'top_k':self.cfg['top_k'],'max_tokens':self.cfg['max_tokens'],'seed':self.seed,'chat_template_kwargs':{'enable_thinking':self.cfg['enable_thinking']},'response_format':{'type':'json_object'}}
        base=os.getenv(self.cfg['base_url_env'],self.cfg['base_url_default']);last=None
        for attempt in range(self.cfg['transport_retries']+1):
            t=time.perf_counter();start=utcnow();raw=None
            try:
                response=requests.post(base+'/chat/completions',json=payload,timeout=self.cfg['timeout_seconds']);raw=response.text
                response.raise_for_status();j=response.json()
                self.save_call({'role':role,'attempt':attempt,'start_utc':start,'end_utc':utcnow(),'latency_ms':1000*(time.perf_counter()-t),'request':payload,'response':j,'error':None})
                return json.loads(j['choices'][0]['message']['content'])
            except (requests.RequestException,) as e:
                last=e;self.save_call({'role':role,'attempt':attempt,'start_utc':start,'end_utc':utcnow(),'latency_ms':1000*(time.perf_counter()-t),'request':payload,'raw_response':raw,'error':type(e).__name__})
                if attempt<self.cfg['transport_retries']:time.sleep(1)
        raise last

def usage(calls):
    tokens={'input_tokens':0,'output_tokens':0,'cached_input_tokens':0};unknown_cached=False;reasoning=[];complete=True
    for c in calls:
        u=c.get('response',{}).get('usage') or {};complete=complete and bool(u)
        tokens['input_tokens']+=int(u.get('prompt_tokens',0));tokens['output_tokens']+=int(u.get('completion_tokens',0))
        cached=(u.get('prompt_tokens_details') or {}).get('cached_tokens')
        if cached is None:unknown_cached=True
        else:tokens['cached_input_tokens']+=int(cached)
        reasoning.append(u.get('reasoning_tokens',(u.get('completion_tokens_details') or {}).get('reasoning_tokens')))
    if unknown_cached:tokens['cached_input_tokens']=None
    tokens['provider_reported_reasoning_tokens']=sum(reasoning) if reasoning and all(v is not None for v in reasoning) else None
    tokens['usage_complete']=complete
    if not complete:
        tokens['known_input_tokens']=tokens['input_tokens'];tokens['known_output_tokens']=tokens['output_tokens'];tokens['input_tokens']=None;tokens['output_tokens']=None
    return tokens

def run_case(store,sid,repeat,cfg,run_dir):
    output=run_dir/f'{sid}_{repeat}.json'
    if output.exists():
        existing=json.loads(output.read_text())
        if cfg.get('protocol_sha256') and existing.get('protocol_sha256')!=cfg['protocol_sha256']:raise RuntimeError('Resume protocol identity mismatch')
        return existing['records']
    packet=store.snapshot(sid);seed=cfg['seeds'][repeat];session=store.session(sid);trace=[];records=[];all_calls=[]
    call_dir=run_dir/'calls'/f'{sid}_{repeat}'
    def finish(method,obj,calls,elapsed,error=None,tools=0,extra=None):
        try:
            if error:raise ValueError(error)
            p=validate_probability(obj,{'input',*[x['evidence_id'] for x in session.payloads]});fallback=False
        except Exception as e:p=store.prior;fallback=True;error=error or str(e)
        rec={'sample_id':sid,'method':method,'model':cfg['model_id'],'repeat':repeat,'p_used':p,'raw_output':obj,'fallback_used':fallback,'error_type':error,'total_latency_ms':elapsed,'llm_calls':len(calls),'tool_calls':tools,'retry_count':sum(c['attempt']>0 for c in calls),'model_latency_ms':sum(c['latency_ms'] for c in calls),'cost_usd':None,'cost_status':'self-hosted; hardware time reported, no dollar tariff assumed',**usage(calls),**(extra or {})}
        records.append(rec);return rec
    d=Caller(cfg,seed,call_dir);t=time.perf_counter();obj=None;err=None
    try:obj=d.call('D0',packet)
    except Exception as e:err=type(e).__name__+': '+str(e)
    finish('D0',obj,d.calls,1000*(time.perf_counter()-t),err);all_calls+=d.calls
    a=Caller(cfg,seed,call_dir);t=time.perf_counter();draft=None;err=None;history=[];tool_ms=0
    try:
        for k in range(cfg['tool_calls_max']+1):
            response=a.call('A1',packet,history)
            if response.get('action')=='final' or ('p_reject' in response and 'action' not in response):draft=response;break
            if k==cfg['tool_calls_max']:raise ValueError('Tool budget exhausted without final output')
            t_tool=time.perf_counter();payload=session.execute(response.get('action'),response.get('arguments',{}));tool_ms+=1000*(time.perf_counter()-t_tool)
            trace.append({'action':response,'payload':payload})
            history.extend([{'role':'assistant','content':json.dumps(response)},{'role':'user','content':json.dumps({'tool_result':payload,'remaining_tool_calls':cfg['tool_calls_max']-k-1})}])
    except Exception as e:err=type(e).__name__+': '+str(e)
    a1=finish('A1',draft,a.calls,1000*(time.perf_counter()-t),err,len(session.payloads),{'tool_latency_ms':tool_ms});all_calls+=a.calls
    # A2 shares the exact planning prefix with A1; system-equivalent cost includes it.
    v=Caller(cfg,seed,call_dir);t=time.perf_counter();final=draft;findings=None;verr=err;revised=False
    try:
        if err:raise ValueError('Planner failed; unified fallback')
        contents={'packet':packet,'tool_payloads':session.payloads,'draft':draft}
        findings=v.call('verifier',contents)
        if not isinstance(findings.get('needs_revision'),bool) or not isinstance(findings.get('errors'),list):raise ValueError('Invalid verifier schema')
        if findings['needs_revision']:
            if not findings['errors']:raise ValueError('Revision requested without located errors')
            allowed={'input',*[x['evidence_id'] for x in session.payloads]}
            if any(e.get('evidence_id') not in allowed for e in findings['errors']):raise ValueError('Verifier cites unknown evidence')
            final=v.call('revise',{**contents,'findings':findings});revised=True
    except Exception as e:verr=type(e).__name__+': '+str(e)
    finish('A2',final,a.calls+v.calls,a1['total_latency_ms']+1000*(time.perf_counter()-t),verr,len(session.payloads),{'revised':revised,'shared_A1_prefix':True,'tool_latency_ms':tool_ms});all_calls+=v.calls
    d=Caller(cfg,seed,call_dir);t=time.perf_counter();obj=None;err=None
    try:obj=d.call('D_match',{'packet':packet,'tool_payloads':session.payloads})
    except Exception as e:err=type(e).__name__+': '+str(e)
    # Main system cost includes obtaining its evidence; marginal one-pass cost also retained.
    marginal=1000*(time.perf_counter()-t)
    finish('D_match',obj,a.calls+d.calls,a1['total_latency_ms']+marginal,err,len(session.payloads),{'single_pass_latency_ms':marginal,'shared_A1_evidence':True});all_calls+=d.calls
    result={'protocol_sha256':cfg.get('protocol_sha256'),'sample_id':sid,'repeat':repeat,'seed':seed,'packet':packet,'trace':trace,'verifier':findings,'records':records,'actual_calls':all_calls,'actual_usage':usage(all_calls),'finished_at':utcnow()}
    dump(output,result);print(f'{sid[:12]} repeat={repeat} completed; fallbacks={sum(r["fallback_used"] for r in records)}',flush=True)
    return records

def run(split,repeats):
    cfg=json.loads((ROOT/'configs/llm.json').read_text());manifest=pd.read_parquet(ROOT/'metadata/split_manifest.parquet')
    if split=='test':
        from .freeze import verify_frozen
        verify_frozen()
        cfg['protocol_sha256']=digest(ROOT/'metadata/frozen_protocol.json')
    ids=pd.read_csv(ROOT/'metadata/pilot_manifest.csv').sample_id.tolist() if split=='pilot' else manifest.loc[manifest.split.eq(split),'sample_id'].tolist()
    frame=pd.read_parquet(PROCESSED/'features.parquet').set_index('sample_id');pred=FrozenPredictor()
    pred.predict(frame.loc[[ids[0]]]) # initialize before concurrent tool inference
    store=EvidenceStore(frame,manifest.set_index('sample_id').proposal_time,pred,pred.selection['training_prior'])
    run_dir=ROOT/'runs'/f'agents_{split}';run_dir.mkdir(exist_ok=True);start=utcnow();t=time.perf_counter();records=[]
    jobs=[(sid,r) for sid in ids for r in range(repeats)]
    if cfg.get('submission_order')=='repeat_major':jobs=[(sid,r) for r in range(repeats) for sid in ids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['concurrency']) as ex:
        futures=[ex.submit(run_case,store,sid,r,cfg,run_dir) for sid,r in jobs]
        for f in concurrent.futures.as_completed(futures):records.extend(f.result())
    df=pd.DataFrame(records);df['raw_output']=df.raw_output.map(json.dumps);df['split']=split;df.to_parquet(ROOT/'runs'/f'agents_{split}.parquet',index=False)
    dump(ROOT/'runs'/f'agents_{split}_execution.json',{'start':start,'end':utcnow(),'elapsed_seconds':time.perf_counter()-t,'N':len(ids),'repeats':repeats,'rows':len(df),'cfg':cfg,'resumable':True,'note':'A1 prefix reused in A2 and D_match; ledger counts physical calls once; method costs include complete evidence acquisition.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='pilot',choices=['pilot','validation','test']);p.add_argument('--repeats',type=int,default=1);a=p.parse_args();run(a.split,a.repeats)
