"""Create a separate empty run workspace; never delete or overwrite supplied runs."""
import argparse
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();dst=a.output.expanduser().resolve()
if dst.exists():p.error('Output already exists; choose a new directory.')
if dst.is_relative_to(ROOT):p.error('Output must be outside the reference repository.')
dst.mkdir(parents=True)
for name in ['src','tests','configs','prompts','metadata','Data','scripts','paper','docs','requirements.lock','requirements-replay.lock','pyproject.toml','Makefile','LICENSE']:
    source=ROOT/name;target=dst/name
    if source.is_dir():shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
    else:shutil.copy2(source,target)
# Preserve historical protocols for comparison; do not treat them as a new freeze.
for name in ['frozen_protocol.json','frozen_protocol_attempt01.json','backup_runtime_identity.json','backup_snapshot_verification.json','environment.json']:
    q=dst/'metadata'/name
    if q.exists():
        target=dst/'reference_metadata'/name;target.parent.mkdir(exist_ok=True);q.rename(target)
for directory in ['runs','runs/revision_20261106','review/revision_20261106','Results/Tables','Results/Source_Data','Results/Figures','Results/revision_20261106']:
    (dst/directory).mkdir(parents=True,exist_ok=True)
print(dst)
print('Empty run workspace created. Refit baselines, validate the actual LLM runtime, run a pilot, then freeze before test inference.')
