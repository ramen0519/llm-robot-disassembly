"""Preserve pre-existing runs and source files without following links."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_luna_experiment import safe,read,write
OUT=ROOT/'runs/sequence_iterative_prepare_20261110'


def checksum(path):
    value=hashlib.sha256()
    with safe(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def inventory(root):
    result={}
    for folder,dirs,names in os.walk(safe(root),followlinks=False):
        for name in list(dirs):
            path=Path(folder)/name;info=path.lstat()
            if path==OUT or stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&1024:
                dirs.remove(name)
        for name in names:
            path=Path(folder)/name;info=path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&1024:continue
            result[path.relative_to(ROOT).as_posix()]=checksum(path)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if sys.argv[1]=='before':
        source={p.relative_to(ROOT).as_posix():checksum(p) for directory in ('worker','orchestrator','tools')
                for p in (ROOT/directory).glob('*.py') if 'iterative' not in p.name and 'checkpoint' not in p.name}
        write(OUT/'existing_source_before.json',source)
        prior=inventory(ROOT/'runs');write(OUT/'protected_before.json',prior)
        print(json.dumps({'protected_runs':len(prior),'legacy_sources':len(source)}))
    else:
        runs=read(OUT/'protected_before.json');sources=read(OUT/'existing_source_before.json')
        changed=[p for p,h in runs.items() if not safe(ROOT/p).is_file() or checksum(ROOT/p)!=h]
        source_changed=[p for p,h in sources.items() if not safe(ROOT/p).is_file() or checksum(ROOT/p)!=h]
        result=dict(protected_runs=len(runs),legacy_sources=len(sources),changed=changed,
                    source_changed=source_changed,preserved=not changed and not source_changed)
        write(OUT/'preservation.json',result);print(json.dumps(result))


if __name__=='__main__':main()
