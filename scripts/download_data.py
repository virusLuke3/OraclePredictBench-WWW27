"""Download immutable public inputs; verify original handoff hashes independently."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib, json, time, urllib.request

ROOT = Path(__file__).resolve().parents[1]
def main():
    reference = json.loads((ROOT/'metadata/public_input_reference.json').read_text())
    repo, sha = reference['repo_id'], reference['resolved_sha']
    entries = [e for e in reference['files'] if not any(x in e['repo_path'] for x in ['decision_time_web', 'decision_time_price', '/daily/'])]
    # Add published native event records for independently replaying history clocks.
    prefix = 'atlas/releases/atlas-utc-20261029/'
    entries += [{'repo_path':prefix+p} for p in ['metadata/table_catalog.json','data/native/polygon_oov2_events.parquet']]
    def fetch(e):
        dest=ROOT/'Data/Queried_Data'/e['repo_path']; dest.parent.mkdir(parents=True, exist_ok=True)
        url=f'https://huggingface.co/datasets/{repo}/resolve/{sha}/{e["repo_path"]}'
        if not dest.exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url,timeout=60) as r: content=r.read()
                    dest.write_bytes(content); break
                except Exception:
                    if attempt==2: raise
                    time.sleep(1+attempt)
        b=dest.read_bytes(); digest=hashlib.sha256(b).hexdigest()
        if 'sha256' in e: assert digest==e['sha256'],e['repo_path']
        if 'size_bytes' in e: assert len(b)==e['size_bytes'],e['repo_path']
        record=dict(repo_id=repo,requested_revision=sha,resolved_commit=sha,repo_path=e['repo_path'],size_bytes=len(b),sha256=digest,url=url,checked_at=datetime.now(timezone.utc).isoformat(),handoff_hash_verified='sha256' in e)
        print(json.dumps({'file':e['repo_path'],'bytes':len(b),'sha256':digest}),flush=True)
        return record
    records=list(ThreadPoolExecutor(max_workers=4).map(fetch,entries))
    (ROOT/'metadata').mkdir(exist_ok=True)
    (ROOT/'metadata/data_manifest.json').write_text(json.dumps(dict(repo_id=repo,resolved_commit=sha,license='CC-BY-4.0 for project-created data; third-party source rights retained',files=records),indent=2)+'\n')
if __name__=='__main__': main()
