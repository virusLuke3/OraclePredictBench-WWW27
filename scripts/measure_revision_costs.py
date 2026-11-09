"""Measure shared offline preparation and warm CPU service boundaries.

Never overwrite frozen inputs or fitted models. Rebuilt inputs are compared in a
temporary directory. CPU runs re-use all frozen weights; no outcomes are loaded.
"""
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from oraclepredict import data
from oraclepredict.common import ROOT, PROCESSED, dump, digest, utcnow
from oraclepredict.baselines import FrozenPredictor
from oraclepredict.tools import EvidenceStore

OUT=ROOT/'Results/revision_20261106'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    offline_started=utcnow()
    with tempfile.TemporaryDirectory(prefix='oracle_revision_cost_') as directory:
        temp=Path(directory)
        for name in ['metadata','configs','Data']:(temp/name).mkdir(parents=True,exist_ok=True)
        (temp/'Data/Queried_Data').symlink_to(ROOT/'Data/Queried_Data',target_is_directory=True)
        old_root,old_processed=data.ROOT,data.PROCESSED
        data.ROOT=temp;data.PROCESSED=temp/'Data/Processed_Data'
        try:
            tick=time.perf_counter();data.main();elapsed=time.perf_counter()-tick
        finally:
            data.ROOT=old_root;data.PROCESSED=old_processed
        matches={}
        for relative in ['Data/Processed_Data/features.parquet','metadata/split_manifest.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(temp/relative),pd.read_parquet(ROOT/relative))
            matches[relative]=True
        offline=dict(started_at=offline_started,elapsed_seconds=elapsed,exact_frame_matches=matches,
            scope='Read cached released events, replay all histories/features, build splits and label tables, write artifacts; no download or model fitting.',
            allocation='Shared offline cost, not multiplied by method or charged as per-request latency.',
            code_sha256=digest(ROOT/'src/oraclepredict/data.py'))
    dump(OUT/'offline_preparation_cost.json',offline)
    initialization=time.perf_counter()
    manifest=pd.read_parquet(ROOT/'metadata/split_manifest.parquet',columns=['sample_id','split','proposal_time'])
    ids=manifest.loc[manifest.split.eq('test'),'sample_id'].tolist()
    features=pd.read_parquet(PROCESSED/'features.parquet').set_index('sample_id')
    predictor=FrozenPredictor()
    methods=predictor.selection['methods']
    for model in methods.values():predictor.predict(features.loc[[ids[0]]],model)
    store=EvidenceStore(features,manifest.set_index('sample_id').proposal_time,predictor,predictor.selection['training_prior'])
    initialization_seconds=time.perf_counter()-initialization
    # Comparison is prediction-only; this file contains no reference outcomes.
    original=pd.read_parquet(ROOT/'runs/baseline_test.parquet').set_index(['sample_id','method']).p_used
    rows=[];start=utcnow();tick=time.perf_counter()
    for repeat in range(3):
        for sid in ids:
            for method,model in methods.items():
                start_case=time.perf_counter()
                packet=store.snapshot(sid)
                prep=1000*(time.perf_counter()-start_case)
                frame=store.features.loc[[sid]]
                prediction=float(predictor.predict(frame,model)[0])
                total=1000*(time.perf_counter()-start_case)
                assert abs(prediction-original.loc[(sid,method)])<1e-12
                rows.append(dict(sample_id=sid,method=method,repeat=repeat,p_used=prediction,
                                 preparation_ms=prep,total_latency_ms=total))
    f=pd.DataFrame(rows);f.to_csv(OUT/'cpu_latency_replay.csv',index=False)
    summary=f.groupby('method').agg(N=('sample_id','nunique'),timed_outputs=('sample_id','size'),
        p50_ms=('total_latency_ms','median'),p95_ms=('total_latency_ms',lambda s:s.quantile(.95)),
        preparation_p50_ms=('preparation_ms','median')).reset_index()
    summary.to_csv(OUT/'cpu_latency_summary.csv',index=False)
    dump(OUT/'cpu_latency_execution.json',dict(start=start,end=utcnow(),elapsed_seconds=time.perf_counter()-tick,
        initialization_seconds=initialization_seconds,predictions_unchanged=True,rows=len(f),
        hardware='Local CPU, warm sequential one-request calls; concurrent GPU inference recorded separately',
        thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        scope='From cached admissible feature table: packet construction, feature lookup/preprocessing, prediction. Offline replay and model load separate.',
        code_sha256=digest(Path(__file__))))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
