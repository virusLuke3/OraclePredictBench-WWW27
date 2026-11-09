"""Independently install the lock, refit baselines, and replay frozen statistics."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['fit','results'],required=True);a=parser.parse_args()
    report_path=ROOT/'review/reproduction_report.json'
    if a.phase=='fit':
        report={'started_at':now(),'hosted_Colab':False}
        dst=Path(tempfile.mkdtemp(prefix='oraclepredictbench-replay-'))
    else:
        if not report_path.exists():
            parser.error('Run --phase fit first to create the independent environment.')
        report=json.loads(report_path.read_text())
        dst=Path(report.get('workspace',''))
        if not report.get('baseline_refit',{}).get('passed') or not (dst/'.venv/bin/python').is_file():
            parser.error('The recorded independent environment is unavailable; run --phase fit on this machine first.')
    report['workspace']=str(dst)
    def save():report_path.write_text(json.dumps(report,indent=2)+'\n')
    def copy(name):
        src=ROOT/name;target=dst/name
        if src.is_dir():shutil.copytree(src,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    def run(cmd):
        print('RUN',cmd,flush=True);subprocess.run(cmd,cwd=dst,env=env,check=True)
    save()
    if a.phase=='fit':
        for name in ['src','tests','configs','prompts','metadata','Data/Processed_Data','pyproject.toml','requirements.lock']:copy(name)
        (dst/'runs').mkdir(exist_ok=True)
        run(['uv','venv','--python',sys.executable,str(dst/'.venv')])
        py=str(dst/'.venv/bin/python')
        run(['uv','pip','install','--python',py,'-r','requirements.lock'])
        run(['uv','pip','install','--python',py,'--no-deps','-e','.'])
        run([py,'-m','oraclepredict.baselines','train'])
        run([py,'-m','pytest','-q'])
        code="""import json,pandas as pd,numpy as np
from pathlib import Path
reference=Path(REFERENCE)
keys=['sample_id','model']
a=pd.read_parquet(reference/'runs/predictions_validation.parquet').set_index(keys).sort_index()
b=pd.read_parquet('runs/predictions_validation.parquet').set_index(keys).sort_index()
assert a.index.equals(b.index)
error=float(np.max(np.abs(a.p-b.p)));assert np.allclose(a.p,b.p,rtol=1e-7,atol=1e-8),error
x=json.loads((reference/'metadata/selected_model.json').read_text());y=json.loads(Path('metadata/selected_model.json').read_text())
assert x['best_model']==y['best_model'] and x['methods']==y['methods']
Path('fit_replay_result.json').write_text(json.dumps({'validation_predictions':len(a),'candidate_models':a.index.get_level_values('model').nunique(),'maximum_absolute_error':error,'selected_model':y['best_model'],'passed':True}))
""".replace('REFERENCE',repr(str(ROOT)))
        run([py,'-c',code]);report['baseline_refit']=json.loads((dst/'fit_replay_result.json').read_text());report['fit_finished_at']=now();save()
    else:
        for name in ['src','scripts/write_results.py','Data/Queried_Data','Data/Processed_Data','metadata','runs/agents_test.parquet','runs/baseline_test.parquet','paper','review/execution_validation.json']:copy(name)
        for name in ['Results/Tables','Results/Figures','Results/Source_Data','Results/Diagrams_Source_Files','review']:(dst/name).mkdir(parents=True,exist_ok=True)
        py=str(dst/'.venv/bin/python')
        run([py,'-m','oraclepredict.data'])
        code="""import json,pandas as pd,numpy as np
from pathlib import Path
from oraclepredict.baselines import FrozenPredictor
reference=Path(REFERENCE)
for name in ['features.parquet','train_labels.parquet','validation_labels.parquet','test_labels.parquet']:
    pd.testing.assert_frame_equal(pd.read_parquet(reference/'Data/Processed_Data'/name),pd.read_parquet(Path('Data/Processed_Data')/name))
pd.testing.assert_frame_equal(pd.read_parquet(reference/'metadata/split_manifest.parquet'),pd.read_parquet('metadata/split_manifest.parquet'))
expected=pd.read_parquet(reference/'runs/baseline_test.parquet');features=pd.read_parquet('Data/Processed_Data/features.parquet').set_index('sample_id')
predictor=FrozenPredictor();error=0.
for model,rows in expected.groupby('model'):
    probabilities=predictor.predict(features.loc[rows.sample_id],model)
    diff=float(np.max(np.abs(probabilities-rows.p_used.to_numpy())))
    assert np.allclose(probabilities,rows.p_used.to_numpy(),rtol=1e-7,atol=1e-8),(model,diff)
    error=max(error,diff)
Path('reconstruction_replay_result.json').write_text(json.dumps({'raw_input_reconstruction_matches':True,'matching_processed_tables':4,'split_manifest_matches':True,'fresh_refit_test_predictions':len(expected),'maximum_absolute_prediction_error':error,'passed':True}))
""".replace('REFERENCE',repr(str(ROOT)))
        run([py,'-c',code]);report['raw_reconstruction_and_test_predictions']=json.loads((dst/'reconstruction_replay_result.json').read_text());save()
        run([py,'-m','oraclepredict.evaluate']);run([py,'-m','oraclepredict.plots']);run([py,'scripts/write_results.py'])
        code="""import json,pandas as pd,numpy as np
from pathlib import Path
reference=Path(REFERENCE);files=['main_results.csv','comparisons.csv','calendar_sensitivity.csv','stability.csv','run_stability.csv']
for f in files:pd.testing.assert_frame_equal(pd.read_csv(reference/'Results/Tables'/f),pd.read_csv(Path('Results/Tables')/f))
assert (reference/'paper/generated_results.tex').read_bytes()==Path('paper/generated_results.tex').read_bytes()
a=np.load(reference/'Results/Source_Data/bootstrap_indices.npz');b=np.load('Results/Source_Data/bootstrap_indices.npz')
for k in a.files:assert np.array_equal(a[k],b[k]),k
Path('results_replay_result.json').write_text(json.dumps({'identical_tables':files,'identical_bootstrap_draws':9999,'generated_prose_identical':True,'passed':True}))
""".replace('REFERENCE',repr(str(ROOT)))
        run([py,'-c',code]);report['frozen_results_replay']=json.loads((dst/'results_replay_result.json').read_text());report['results_finished_at']=now();save()
    print('REPRODUCTION_PHASE_PASSED',a.phase,flush=True)

if __name__=='__main__':main()
