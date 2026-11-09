"""November revision: paired predictor-access ablation, with fresh control runs.

All inference inputs are proposal-time features and the training prior. Outcome
tables are read only by the separate analysis script. Earlier frozen runs remain
untouched. Control prompts and decoding match the original experiment exactly.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from .agents import Caller, usage
from .baselines import FrozenPredictor
from .common import ROOT, PROCESSED, dump, digest, utcnow
from .freeze import verify_frozen
from .tools import EvidenceStore, ToolSession, validate_probability

REV = ROOT / 'review/revision_20261106'
RUN = ROOT / 'runs/revision_20261106'
CONFIG = ROOT / 'configs/revision_20261106/ablation.json'
PROMPT_DIR = ROOT / 'prompts/revision_20261106'
ARMS = ('with_predictor', 'without_predictor')


class NoPredictorSession(ToolSession):
    def execute(self, name, args):
        if name == 'predict_tabular':
            raise ValueError('Predictor access disabled in this experimental arm')
        return super().execute(name, args)


class NoPredictorStore(EvidenceStore):
    def __init__(self, features, cutoffs, prior):
        super().__init__(features, cutoffs, None, prior)

    def session(self, case_id):
        return NoPredictorSession(self, case_id)


class ArmCaller(Caller):
    def __init__(self, cfg, seed, call_log_dir, arm):
        super().__init__(cfg, seed, call_log_dir)
        self.prompts = {p.stem: p.read_text() for p in (ROOT/'prompts').glob('*.txt')}
        if arm == 'without_predictor':
            self.prompts['A1'] = (PROMPT_DIR/'A1_without_predictor.txt').read_text()

    def call(self, role, contents, history=None):
        messages = [
            {'role':'system', 'content':self.prompts['common']+'\n'+self.prompts[role]},
            {'role':'user', 'content':json.dumps(contents, allow_nan=False)},
        ] + (history or [])
        payload = dict(model=self.cfg['model_id'], messages=messages,
            temperature=self.cfg['temperature'], top_p=self.cfg['top_p'], top_k=self.cfg['top_k'],
            max_tokens=self.cfg['max_tokens'], seed=self.seed,
            chat_template_kwargs={'enable_thinking':self.cfg['enable_thinking']}, response_format={'type':'json_object'})
        base = os.getenv(self.cfg['base_url_env'], self.cfg['base_url_default'])
        for attempt in range(self.cfg['transport_retries']+1):
            start = utcnow(); tick = time.perf_counter(); raw = None
            try:
                response = requests.post(base+'/chat/completions', json=payload, timeout=self.cfg['timeout_seconds'])
                raw = response.text; response.raise_for_status(); data = response.json()
                self.save_call(dict(role=role, attempt=attempt, start_utc=start, end_utc=utcnow(),
                    latency_ms=1000*(time.perf_counter()-tick), request=payload, response=data, error=None))
                return json.loads(data['choices'][0]['message']['content'])
            except requests.RequestException as error:
                self.save_call(dict(role=role, attempt=attempt, start_utc=start, end_utc=utcnow(),
                    latency_ms=1000*(time.perf_counter()-tick), request=payload, raw_response=raw, error=type(error).__name__))
                if attempt == self.cfg['transport_retries']:
                    raise
                time.sleep(1)


def run_case(store, sid, repeat, arm, cfg, directory, protocol_hash):
    target = directory/f'{sid}_{repeat}_{arm}.json'
    if target.exists():
        saved = json.loads(target.read_text())
        if saved['protocol_sha256'] != protocol_hash:
            raise RuntimeError('Cannot reuse a case from a different protocol')
        return saved['records']
    start = utcnow(); prep_tick = time.perf_counter()
    packet = store.snapshot(sid); session = store.session(sid)
    preparation_ms = 1000*(time.perf_counter()-prep_tick)
    records = []; trace = []; call_dir = directory/'calls'/f'{sid}_{repeat}_{arm}'
    seed = cfg['seeds'][repeat]

    def finish(method, obj, calls, elapsed, error=None, **extra):
        try:
            if error:
                raise ValueError(error)
            probability = validate_probability(obj, {'input', *[x['evidence_id'] for x in session.payloads]})
            fallback = False
        except Exception as exception:
            probability = store.prior; fallback = True; error = error or str(exception)
        record = dict(sample_id=sid, repeat=repeat, seed=seed, arm=arm, method=method,
            p_used=probability, raw_output=obj, fallback_used=fallback, error_type=error,
            total_latency_ms=elapsed, preparation_ms=preparation_ms, llm_calls=len(calls),
            tool_calls=len(session.payloads), retry_count=sum(c['attempt']>0 for c in calls),
            model_latency_ms=sum(c['latency_ms'] for c in calls), cost_usd=None,
            cost_status='Warm system with packet preparation, tool calls, verification, revision and retries; initialization separate',
            **usage(calls), **extra)
        records.append(record)
        return record

    planner = ArmCaller(cfg, seed, call_dir, arm)
    tick = time.perf_counter(); draft = None; error = None; history = []; tool_ms = 0
    try:
        for k in range(cfg['tool_calls_max']+1):
            response = planner.call('A1', packet, history)
            if response.get('action') == 'final' or ('p_reject' in response and 'action' not in response):
                draft = response
                break
            if k == cfg['tool_calls_max']:
                raise ValueError('Tool budget exhausted without final output')
            tool_tick = time.perf_counter()
            payload = session.execute(response.get('action'), response.get('arguments',{}))
            tool_ms += 1000*(time.perf_counter()-tool_tick)
            trace.append({'action':response, 'payload':payload})
            history.extend([{'role':'assistant','content':json.dumps(response)},
                {'role':'user','content':json.dumps({'tool_result':payload,'remaining_tool_calls':cfg['tool_calls_max']-k-1})}])
    except Exception as exception:
        error = type(exception).__name__+': '+str(exception)
    suffix = 'withB' if arm == 'with_predictor' else 'noB'
    a1 = finish('A1_'+suffix, draft, planner.calls, preparation_ms+1000*(time.perf_counter()-tick),
                error, tool_latency_ms=tool_ms)
    verifier = ArmCaller(cfg, seed, call_dir, arm)
    tick = time.perf_counter(); final = draft; findings = None; verify_error = error; revised = False
    try:
        if error:
            raise ValueError('Planner failed; unified fallback')
        contents = {'packet':packet, 'tool_payloads':session.payloads, 'draft':draft}
        findings = verifier.call('verifier', contents)
        if not isinstance(findings.get('needs_revision'),bool) or not isinstance(findings.get('errors'),list):
            raise ValueError('Invalid verifier schema')
        if findings['needs_revision']:
            if not findings['errors']:
                raise ValueError('Revision requested without located errors')
            allowed = {'input', *[x['evidence_id'] for x in session.payloads]}
            if any(e.get('evidence_id') not in allowed for e in findings['errors']):
                raise ValueError('Verifier cites unknown evidence')
            final = verifier.call('revise', {**contents,'findings':findings}); revised = True
    except Exception as exception:
        verify_error = type(exception).__name__+': '+str(exception)
    finish('A2_'+suffix, final, planner.calls+verifier.calls,
        a1['total_latency_ms']+1000*(time.perf_counter()-tick), verify_error,
        tool_latency_ms=tool_ms, revised=revised, shared_A1_prefix=True)
    calls = planner.calls+verifier.calls
    dump(target, dict(protocol_sha256=protocol_hash, sample_id=sid, repeat=repeat, seed=seed, arm=arm,
        start_utc=start, finished_at=utcnow(), packet=packet, trace=trace, verifier=findings,
        records=records, actual_calls=calls, actual_usage=usage(calls)))
    print(f'{sid[:12]} repeat={repeat} arm={arm} outputs=2 fallbacks={sum(r["fallback_used"] for r in records)}',flush=True)
    return records


def protocol_files():
    original = json.loads((ROOT/'metadata/frozen_protocol.json').read_text())
    names = set(original['hashes'])-{'src/oraclepredict/evaluate.py','src/oraclepredict/plots.py'}
    names.update(['src/oraclepredict/revision_ablation.py','configs/revision_20261106/ablation.json',
                  'prompts/revision_20261106/A1_without_predictor.txt','tests/test_revision_ablation.py'])
    return sorted(names)


def freeze():
    verify_frozen()
    path = REV/'ablation_frozen_protocol.json'
    if path.exists():
        raise RuntimeError('Preserve the existing revision protocol; do not overwrite')
    pilot = pd.read_parquet(RUN/'agents_pilot.parquet')
    assert pilot.sample_id.nunique() == 32 and len(pilot) == 128
    assert set(pilot.method) == {'A1_withB','A2_withB','A1_noB','A2_noB'}
    execution = json.loads((RUN/'agents_pilot_execution.json').read_text())
    assert execution['completed']
    cfg = json.loads(CONFIG.read_text())
    dump(path, dict(frozen_at=utcnow(), hashes={n:digest(ROOT/n) for n in protocol_files()},
        config=cfg, pilot_summary={'N':32,'outputs':128,'fallback_outputs':int(pilot.fallback_used.sum())},
        test_previously_seen=True, interpretation='Exploratory paired access intervention; no new untouched cohort',
        control='Fresh same-runtime with-predictor A1/A2, interleaved with without-predictor arms',
        comparisons=cfg['comparisons'], inference='9999 paired question-cluster percentile intervals; no new confirmatory p-value claims'))
    print('Frozen revision protocol',digest(path))


def run(split, repeats):
    verify_frozen()
    cfg = json.loads(CONFIG.read_text())
    if split == 'test':
        path = REV/'ablation_frozen_protocol.json'
        protocol = json.loads(path.read_text())
        for name, expected in protocol['hashes'].items():
            if digest(ROOT/name) != expected:
                raise RuntimeError('Revision input changed after freeze: '+name)
        protocol_hash = digest(path)
    else:
        protocol_hash = hashlib.sha256(json.dumps({n:digest(ROOT/n) for n in protocol_files()},sort_keys=True).encode()).hexdigest()
    init_tick = time.perf_counter(); started = utcnow()
    manifest = pd.read_parquet(ROOT/'metadata/split_manifest.parquet',columns=['sample_id','split','proposal_time'])
    ids = (pd.read_csv(ROOT/'metadata/pilot_manifest.csv').sample_id.tolist() if split=='pilot'
           else manifest.loc[manifest.split.eq('test'),'sample_id'].tolist())
    features = pd.read_parquet(PROCESSED/'features.parquet').set_index('sample_id')
    predictor = FrozenPredictor(); predictor.predict(features.loc[[ids[0]]])
    cutoffs = manifest.set_index('sample_id').proposal_time; prior = predictor.selection['training_prior']
    stores = {'with_predictor':EvidenceStore(features,cutoffs,predictor,prior),
              'without_predictor':NoPredictorStore(features,cutoffs,prior)}
    initialization_seconds = time.perf_counter()-init_tick
    directory = RUN/f'agents_{split}'; directory.mkdir(parents=True,exist_ok=True)
    jobs = []
    for repeat in range(repeats):
        for sid in ids:
            order = ARMS if int(hashlib.sha256(f'{sid}:{repeat}'.encode()).hexdigest(),16)%2 == 0 else ARMS[::-1]
            jobs.extend((sid,repeat,arm) for arm in order)
    tick = time.perf_counter(); records = []; run_start = utcnow()
    with ThreadPoolExecutor(max_workers=cfg['concurrency']) as executor:
        futures = [executor.submit(run_case,stores[arm],sid,repeat,arm,cfg,directory,protocol_hash) for sid,repeat,arm in jobs]
        for future in as_completed(futures):
            records.extend(future.result())
    frame = pd.DataFrame(records); frame['raw_output'] = frame.raw_output.map(json.dumps)
    frame.to_parquet(RUN/f'agents_{split}.parquet',index=False)
    elapsed = time.perf_counter()-tick
    dump(RUN/f'agents_{split}_execution.json',dict(initialization_started=started,start=run_start,end=utcnow(),
        initialization_seconds=initialization_seconds,elapsed_seconds=elapsed,N=len(ids),repeats=repeats,
        arms=list(ARMS),rows=len(frame),completed=len(frame)==len(ids)*repeats*4,
        protocol_sha256=protocol_hash,config=cfg,allocated_GPU_hours=4*elapsed/3600,
        note='Fresh controls. A2 reuses its own arm A1 prefix; physical calls count shared prefixes once. Model cold start excluded.'))
    print('COMPLETED',split,len(frame),'outputs',elapsed,'seconds',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['pilot','freeze','test'])
    args = parser.parse_args()
    if args.action == 'freeze':freeze()
    else:run('pilot' if args.action=='pilot' else 'test',1 if args.action=='pilot' else 3)
