"""Recompute the paper from saved predictions and fail on reference mismatches."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    reference = json.loads((ROOT/'provenance/result_reference.json').read_text())
    logs = ROOT/'validation'; logs.mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONPATH':str(ROOT/'src'), 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'}
    trace_counts = {}
    for directory, predictions, outputs in [
        ('runs/agents_test','runs/agents_test.parquet',1884),
        ('runs/revision_20261106/agents_test','runs/revision_20261106/agents_test.parquet',1884),
    ]:
        expected = pd.read_parquet(ROOT/predictions).set_index(['sample_id','repeat','method'])
        assert len(expected) == outputs and expected.index.is_unique
        seen = set(); calls = 0; fallback = 0
        for path in sorted((ROOT/directory).glob('*.json')):
            case = json.loads(path.read_text()); calls += len(case['actual_calls'])
            for row in case['records']:
                key = (case['sample_id'],case['repeat'],row['method'])
                assert key not in seen; seen.add(key)
                actual = expected.loc[key]
                assert abs(float(actual.p_used)-float(row['p_used'])) < 1e-14, key
                assert bool(actual.fallback_used) == bool(row['fallback_used']), key
                fallback += bool(row['fallback_used'])
        assert seen == set(expected.index)
        trace_counts[directory] = {'cases':len(list((ROOT/directory).glob('*.json'))),'outputs':len(seen),'physical_calls':calls,'fallback_outputs':fallback}
    assert trace_counts['runs/agents_test'] == dict(cases=471,outputs=1884,physical_calls=2854,fallback_outputs=0)
    assert trace_counts['runs/revision_20261106/agents_test'] == dict(cases=942,outputs=1884,physical_calls=3463,fallback_outputs=5)
    bootstrap_files = ['Results/Source_Data/bootstrap_indices.npz','Results/revision_20261106/bootstrap_indices.npz']
    draws = {}
    for name in bootstrap_files:
        with np.load(ROOT/name) as f: draws[name]={k:f[k].copy() for k in f.files}
    pictures = {name:np.asarray(Image.open(ROOT/f'paper/figures/{name}.png')).copy() for name in ['method','results']}
    commands = [[sys.executable,'-m','oraclepredict.evaluate'],[sys.executable,'scripts/analyze_revision.py'],
                [sys.executable,'scripts/plot_revision.py'],[sys.executable,'scripts/write_revision_results.py']]
    with (logs/'replay.log').open('w') as log:
        for cmd in commands:
            print('Running', ' '.join(cmd[1:]), flush=True)
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for name, expected in reference['exact_files'].items():
        assert sha(ROOT/name)==expected, f'Regenerated result differs from publication: {name}'
    for name, fields in draws.items():
        with np.load(ROOT/name) as f:
            for key, value in fields.items(): assert np.array_equal(f[key],value), (name,key)
    for name, expected in pictures.items():
        assert np.array_equal(expected,np.asarray(Image.open(ROOT/f'paper/figures/{name}.png'))), name
    assert not (ROOT/'review/revision_20261106/registration_proposal.json').exists(), 'Internal author document generated'
    report = dict(passed=True,checked_at=datetime.now(timezone.utc).isoformat(),
                  traces=trace_counts,identical_recomputed_files=list(reference['exact_files']),
                  bootstrap_draws_per_batch=9999,bootstrap_indices_identical=True,figure_pixels_identical=True,
                  model_inference_executed=False,latencies_remeasured=False,
                  scope='Saved predictions and recorded latencies only; no GPU, API key or model call.')
    (logs/'replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':main()
