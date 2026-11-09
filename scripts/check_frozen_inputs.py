"""Check every included inference input against the original scientific freeze."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
for name in ['metadata/frozen_protocol.json','review/revision_20261106/ablation_frozen_protocol.json']:
    record=json.loads((ROOT/name).read_text()); checked=0
    for filename,expected in record['hashes'].items():
        if name.startswith('metadata/') and filename in {'src/oraclepredict/evaluate.py','src/oraclepredict/plots.py'}:continue
        p=ROOT/filename
        if not p.exists():raise SystemExit(f'Missing frozen input: {filename}. See docs/REPRODUCE.md for refitting in a separate workspace.')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,filename
        checked+=1
    print(name,checked,'frozen inputs match')
