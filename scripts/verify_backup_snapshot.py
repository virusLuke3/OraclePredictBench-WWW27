"""Verify the backup weight bytes against the recorded immutable HF revision."""
import hashlib,json,os
from pathlib import Path
from oraclepredict.common import ROOT,dump,utcnow

def main():
    directory=Path(os.environ.get('ORACLEPREDICT_MODEL_PATH','/mnt/hdd22t/oraclepredictbench-runtime/Qwen3.8-27B'))
    reference=json.loads((ROOT/'metadata/model_artifact_reference.json').read_text())
    checked=[]
    for entry in reference['shards']:
        p=directory/entry['rfilename'];actual=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        match=actual==entry['lfs']['sha256'] and p.stat().st_size==entry['size']
        checked.append({'file':p.name,'bytes':p.stat().st_size,'sha256':actual,'match':match})
        assert match,p.name
        print('SHA256_MATCH',p.name,flush=True)
    auxiliary=[{'file':p.name,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in {'.json','.jinja','.txt'}]
    dump(ROOT/'metadata/backup_snapshot_verification.json',{'verified_at':utcnow(),'repo':reference['repo'],'revision':reference['revision'],'weights':checked,'auxiliary':auxiliary,'all_weights_match':True,'transport':'Weights downloaded via the official Qwen ModelScope transport mirror, then matched byte-for-byte against pinned Hugging Face LFS SHA256 values. Auxiliary files fetched from the pinned HF revision. A floating mirror revision is not used as model identity.'})
    print('BACKUP_SNAPSHOT_SHA256_VERIFIED',flush=True)

if __name__=='__main__':main()
