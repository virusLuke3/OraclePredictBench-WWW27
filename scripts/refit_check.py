"""Train all baseline candidates in a separate directory and check predictions."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
workspace=Path(tempfile.mkdtemp(prefix='oracle-baseline-refit-'))/'run'
subprocess.run([sys.executable,'scripts/prepare_new_run.py','--output',str(workspace)],cwd=ROOT,check=True)
env={**os.environ,'PYTHONPATH':str(workspace/'src'),'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'}
logs=ROOT/'validation';logs.mkdir(exist_ok=True)
with (logs/'refit.log').open('w') as log:
    subprocess.run([sys.executable,'-m','oraclepredict.baselines','train'],cwd=workspace,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
a=pd.read_parquet(ROOT/'runs/predictions_validation.parquet').set_index(['sample_id','model']).sort_index()
b=pd.read_parquet(workspace/'runs/predictions_validation.parquet').set_index(['sample_id','model']).sort_index()
assert a.index.equals(b.index)
error=float(np.max(np.abs(a.p-b.p)));assert np.allclose(a.p,b.p,rtol=1e-7,atol=1e-8),error
original=json.loads((ROOT/'metadata/selected_model.json').read_text());new=json.loads((workspace/'metadata/selected_model.json').read_text())
assert original['best_model']==new['best_model'] and original['methods']==new['methods']
# Check test probabilities using the newly fitted models without reading labels.
code="""
import json,numpy as np,pandas as pd
from pathlib import Path
from oraclepredict.baselines import FrozenPredictor
from oraclepredict.common import ROOT
reference=Path(REFERENCE)
r=pd.read_parquet(reference/'runs/baseline_test.parquet')
x=pd.read_parquet(ROOT/'Data/Processed_Data/features.parquet').set_index('sample_id')
p=FrozenPredictor();error=0
for model,f in r.groupby('model'):
    values=p.predict(x.loc[f.sample_id],model)
    error=max(error,float(np.max(np.abs(values-f.p_used))))
    assert np.allclose(values,f.p_used,rtol=1e-7,atol=1e-8),(model,error)
Path('test_refit_check.json').write_text(json.dumps({'predictions':len(r),'maximum_absolute_error':error}))
""".replace('REFERENCE',repr(str(ROOT)))
with (logs/'refit.log').open('a') as log:
    subprocess.run([sys.executable,'-c',code],cwd=workspace,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
report=dict(passed=True,candidate_models=int(a.index.get_level_values('model').nunique()),validation_predictions=len(a),validation_maximum_absolute_error=error,selected_model=new['best_model'],test=json.loads((workspace/'test_refit_check.json').read_text()),LLM_inference_executed=False)
(logs/'refit_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
print('Separate refit workspace retained:',workspace)
