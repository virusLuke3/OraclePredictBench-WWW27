from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT/'Data/Queried_Data/atlas/releases/atlas-utc-20261029'
PROCESSED = ROOT/'Data/Processed_Data'

def utcnow(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonable(v):
    if isinstance(v, (np.integer, np.floating, np.bool_)): return v.item()
    if isinstance(v, (Path, datetime)): return str(v)
    if isinstance(v, np.ndarray): return v.tolist()
    raise TypeError(type(v).__name__)
def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=jsonable,allow_nan=False)+'\n')
def code_commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
