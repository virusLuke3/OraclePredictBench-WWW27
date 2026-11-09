"""Export a review ZIP without Git history, author paperwork or fitted binaries."""
import argparse,hashlib,json,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TEXT={'.py','.json','.csv','.md','.txt','.yaml','.yml','.toml','.tex','.bib','.svg','.log','.cff','.lock'}
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();out=a.output.expanduser().resolve()
if out.exists():parser.error('Output already exists; choose a new path.')
# Track only files reviewed for the repository, never arbitrary workspace files.
names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0')
redactions=[];excluded=[];manifest={};out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in sorted(filter(None,names)):
        p=Path(name)
        if p.parts[0] in {'models','.github'} or name in {'provenance/source_export.json'}:
            excluded.append(name);continue
        data=(ROOT/p).read_bytes();before=hashlib.sha256(data).hexdigest()
        if p.suffix in TEXT or p.name in {'README.md','Makefile','LICENSE','NOTICE','.gitignore'}:
            text=data.decode('utf-8')
            text=re.sub(r'/home/[A-Za-z0-9_.-]+/', '/home/reviewer/',text)
            # This strips authenticated origin URLs if they appear in local receipts.
            text=re.sub(r'https://github\.com/[^/\s"\']+/OraclePredictBench[^\s"\')>]*', '[working repository omitted]',text)
            data=text.encode('utf-8')
        digest=hashlib.sha256(data).hexdigest()
        if digest!=before:redactions.append(dict(path=name,original_sha256=before,review_sha256=digest,reason='personal filesystem path or working repository URL'))
        z.writestr(name,data);manifest[name]={'sha256':digest,'bytes':len(data)}
    notice={'scope':'Content-prepared review export; this ZIP does not create an anonymous public URL. Published third-party citations and license notices are retained.',
            'exclusions':'Git history, CI metadata, fitted binary models (may contain personal paths), original export inventory. Raw per-request model responses and predictions are retained.',
            'frozen_hashes':'Historical freezes are preserved. Omitted fitted models cannot be checked with check-frozen; refit in a separate workspace. Text redactions are listed explicitly.',
            'redactions':redactions,'excluded_files':excluded}
    z.writestr('ANONYMIZATION.json',json.dumps(notice,indent=2)+'\n')
    z.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print(json.dumps({'archive':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(manifest),'text_redactions':len(redactions),'excluded_files':len(excluded)},indent=2))
