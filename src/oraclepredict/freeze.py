import json,subprocess
from pathlib import Path
import pandas as pd
from .common import ROOT,dump,digest,utcnow,code_commit

def verify_frozen():
    record=json.loads((ROOT/'metadata/frozen_protocol.json').read_text())
    # Reporting code may evolve; inference inputs and fitted artifacts must match.
    reporting={'src/oraclepredict/evaluate.py','src/oraclepredict/plots.py'}
    mismatch=[p for p,h in record['hashes'].items() if p not in reporting and digest(ROOT/p)!=h]
    if mismatch:raise RuntimeError('Frozen inference inputs changed: '+str(mismatch))

def main():
    target=ROOT/'metadata/frozen_protocol.json'
    if target.exists():raise RuntimeError('Protocol already frozen; never silently overwrite')
    pilot=pd.read_parquet(ROOT/'runs/agents_pilot.parquet')
    assert pilot.sample_id.nunique()==32 and set(pilot.method)=={'D0','A1','A2','D_match'}
    assert (ROOT/'metadata/model_snapshot_sha256.txt').exists()
    files=[]
    for directory,glob in [('configs','*.json'),('prompts','*.txt'),('src/oraclepredict','*.py')]:files+=list((ROOT/directory).glob(glob))
    files += [ROOT/'metadata/split_manifest.parquet',ROOT/'metadata/pilot_manifest.csv',ROOT/'metadata/selected_model.json',ROOT/'metadata/model_snapshot_sha256.txt',ROOT/'metadata/model_identity.json',ROOT/'requirements.lock',ROOT/'metadata/data_manifest.json',ROOT/'Data/Processed_Data/features.parquet']
    files += [ROOT/'metadata/preprocessor_fit_manifest.csv']
    for name in ['metadata/backup_runtime_identity.json','metadata/backup_runtime_requirements.txt','metadata/backup_snapshot_verification.json','scripts/serve_transformers_backup.py','review/preprocessor_correction.json']:
        if (ROOT/name).exists():files.append(ROOT/name)
    model_files=list((ROOT/'models').rglob('*'));files += [p for p in model_files if p.is_file()]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    dump(target,{'frozen_at':utcnow(),'code_commit':code_commit(),'hashes':hashes,'pilot_N':32,'test_evaluated_in_this_run':False,'historical_public_test_exposure_disclosed':True,'earlier_outage_attempt_scored':(ROOT/'metadata/frozen_protocol_attempt01.json').exists(),'registration':'prespecified local freeze, not public preregistration; corrected rerun is not an untouched test','main_comparison':'A2 minus validation-selected B*','notes':'Inference/model/prompt changes after freeze require explicit deviation record; analysis/report rendering may evolve without changing locked predictions.'})
    print('Frozen',len(hashes),'artifacts at',target)

if __name__=='__main__':main()
