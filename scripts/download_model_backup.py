"""Acquire the same pinned public weights to a task-owned optional backup cache."""
import concurrent.futures,hashlib,json,time,requests,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=Path(os.environ.get('ORACLEPREDICT_MODEL_PATH','/mnt/hdd22t/oraclepredictbench-runtime/Qwen3.8-27B'))
def main():
    identity=json.loads((ROOT/'metadata/model_identity.json').read_text());repo=identity['repo_id'];revision=identity['public_revision']
    hashes={x['file']:x['sha256'] for x in identity['checked_weights']}
    info=requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}',params={'blobs':'true'},timeout=30).json()
    entries=[x for x in info['siblings'] if x['rfilename'].endswith(('.safetensors','.json','.jinja','.txt'))]
    DEST.mkdir(parents=True,exist_ok=True)
    def fetch(e):
        session=requests.Session();session.trust_env=False
        name=e['rfilename'];p=DEST/name;expected=hashes.get(name) or (e.get('lfs') or {}).get('sha256')
        # Completed files were atomically renamed only after their hash matched.
        # The separate full snapshot verifier checks all files before serving.
        if p.exists():return name
        temp=p.with_suffix(p.suffix+'.partial');p.parent.mkdir(parents=True,exist_ok=True)
        size=e.get('size') or (e.get('lfs') or {}).get('size')
        offset=temp.stat().st_size if temp.exists() else 0
        if size and offset>size:raise ValueError('Oversized partial download')
        def get_range(start,end):
            session=requests.Session();session.trust_env=False
            for attempt in range(5):
                try:
                    base=f'https://modelscope.cn/models/{repo}/resolve/master/{name}' if name.endswith('.safetensors') else f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
                    url=base+f'?oraclepredict_range={start}-{end}-{attempt}'
                    headers={'Range':f'bytes={start}-{end}'} if size else {}
                    r=session.get(url,headers=headers,timeout=(20,90));r.raise_for_status()
                    if size:
                        assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')
                        assert len(r.content)==end-start+1
                    return r.content
                except Exception as exc:
                    print('retry',name,start,attempt,type(exc).__name__,flush=True)
                    if attempt==4:raise
                    time.sleep(1)
        if not size:
            temp.write_bytes(get_range(0,None))
        else:
            # Bound memory to four 64-MiB chunks per file. Append in order so
            # interruption leaves a valid resumable prefix, never sparse holes.
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ranges:
                while offset<size:
                    starts=range(offset,min(offset+4*64*1024*1024,size),64*1024*1024)
                    futures=[ranges.submit(get_range,start,min(start+64*1024*1024,size)-1) for start in starts]
                    for future in futures:
                        block=future.result()
                        with temp.open('ab') as f:f.write(block)
                        offset+=len(block)
        if expected:assert hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()==expected,name
        if size:assert temp.stat().st_size==size
        temp.replace(p);print('verified',name,p.stat().st_size,flush=True);return name
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(fetch,entries))
    print('DOWNLOAD_COMPLETE_RUN_SNAPSHOT_VERIFIER_BEFORE_SERVING',DEST,flush=True)
if __name__=='__main__':main()
