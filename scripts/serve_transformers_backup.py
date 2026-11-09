"""Optional text-only backup for the SAME pinned model; disclose runtime changes.

This is not the original SGLang runtime. It must pass a new validation pilot and
receive its own recorded protocol before contributing a new formal run.
"""
import asyncio,json,os,queue,threading,time,uuid
from collections import defaultdict
from concurrent.futures import Future
from pathlib import Path

import torch
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText,AutoTokenizer,set_seed
# lm-format-enforcer 0.11.3 imports the pre-Transformers-5 export location.
# Keep this compatibility alias task-local; do not modify installed packages.
import transformers.tokenization_utils as tokenizer_compat
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
tokenizer_compat.PreTrainedTokenizerBase=PreTrainedTokenizerBase
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn,build_token_enforcer_tokenizer_data
import uvicorn

MODEL_PATH=Path(os.environ.get('ORACLEPREDICT_MODEL_PATH','/mnt/hdd22t/oraclepredictbench-runtime/Qwen3.8-27B'))
MODEL_ID='Qwen3.8-27B'
jobs=queue.Queue();ready=threading.Event();failure=[];app=FastAPI()

def worker():
    try:
        torch.set_num_threads(8)
        os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD','1')
        # FLA requires contiguous inputs; preserve values while avoiding stride-dependent kernels.
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_ops
        def contiguous_kernel(fn):
            def call(*args,**kwargs):
                args=tuple(v.contiguous() if isinstance(v,torch.Tensor) else v for v in args)
                kwargs={k:v.contiguous() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
                return fn(*args,**kwargs)
            return call
        qwen_ops.torch_chunk_gated_delta_rule=contiguous_kernel(qwen_ops.torch_chunk_gated_delta_rule)
        qwen_ops.torch_recurrent_gated_delta_rule=contiguous_kernel(qwen_ops.torch_recurrent_gated_delta_rule)
        tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,padding_side='left')
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        tokenizer_data=build_token_enforcer_tokenizer_data(tokenizer)
        model=AutoModelForImageTextToText.from_pretrained(MODEL_PATH,local_files_only=True,dtype=torch.bfloat16,device_map='balanced',max_memory={i:'22GiB' for i in range(4)},attn_implementation='sdpa').eval()
        print('MODEL_READY',json.dumps(model.hf_device_map),flush=True);ready.set()
    except Exception as e:
        failure.append(repr(e));ready.set();raise
    while True:
        first=jobs.get();batch=[first]
        time.sleep(.025)
        while len(batch)<8:
            try:batch.append(jobs.get_nowait())
            except queue.Empty:break
        # Native HF sampling uses a shared generator. Group equal requested seeds;
        # batch position can affect draws, as disclosed in the runtime metadata.
        grouped=defaultdict(list)
        for payload,future in batch:
            key=(payload.get('seed',17),payload.get('temperature',.7),payload.get('top_p',.95),payload.get('top_k',20),payload.get('max_tokens',512))
            grouped[key].append((payload,future))
        for (seed,temperature,top_p,top_k,max_tokens),items in grouped.items():
            try:
                strings=[tokenizer.apply_chat_template(p['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False) for p,_ in items]
                encoded=tokenizer(strings,return_tensors='pt',padding=True).to(next(model.parameters()).device)
                if encoded.input_ids.shape[1]+max_tokens>8192:raise ValueError('Context budget exceeded')
                constraint=build_transformers_prefix_allowed_tokens_fn(tokenizer_data,JsonSchemaParser({'type':'object'}))
                set_seed(seed)
                with torch.inference_mode():
                    generated=model.generate(**encoded,max_new_tokens=max_tokens,do_sample=True,temperature=temperature,top_p=top_p,top_k=top_k,prefix_allowed_tokens_fn=constraint)
                tail=generated[:,encoded.input_ids.shape[1]:]
                eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
                for row,(payload,future) in enumerate(items):
                    ids=tail[row].tolist();end=next((k+1 for k,t in enumerate(ids) if t in eos),len(ids));ids=ids[:end]
                    content=tokenizer.decode(ids,skip_special_tokens=True).strip()
                    n_in=int(encoded.attention_mask[row].sum());n_out=len(ids)
                    result={'id':'chatcmpl-'+uuid.uuid4().hex,'object':'chat.completion','created':int(time.time()),'model':MODEL_ID,'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop' if any(token in eos for token in ids) else 'length'}],'usage':{'prompt_tokens':n_in,'completion_tokens':n_out,'total_tokens':n_in+n_out,'prompt_tokens_details':{'cached_tokens':0},'reasoning_tokens':0},'runtime':{'engine':'transformers','batch_size':len(items),'requested_seed':seed,'batch_position_affects_sampling':True,'prefix_cache':False}}
                    if not future.cancelled():future.set_result(result)
            except Exception as e:
                print('GENERATION_ERROR',repr(e),flush=True)
                for _,future in items:
                    if not future.cancelled():future.set_exception(e)

@app.get('/v1/models')
def models():
    if not ready.is_set():raise HTTPException(503,'Model loading')
    if failure:raise HTTPException(503,failure[0])
    return {'object':'list','data':[{'id':MODEL_ID,'object':'model','owned_by':'local'}]}

@app.post('/v1/chat/completions')
async def completions(payload:dict):
    if not ready.is_set() or failure:raise HTTPException(503,'Model unavailable')
    if payload.get('model')!=MODEL_ID:raise HTTPException(400,'Wrong model')
    future=Future();jobs.put((payload,future))
    try:return await asyncio.wrap_future(future)
    except Exception as e:raise HTTPException(500,str(e)) from e

if __name__=='__main__':
    threading.Thread(target=worker,daemon=True).start()
    uvicorn.run(app,host='127.0.0.1',port=31028,log_level='warning')
