"""Install only run-local CAD/vision dependencies from pinned public artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1] / '.runtime'
REV = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'

def fetch(url, dest, expected=None):
    if not dest.exists():
        request = urllib.request.Request(url, headers={'User-Agent':'disassembly-research/1'})
        with urllib.request.urlopen(request, timeout=90) as response, dest.with_suffix('.part').open('wb') as out:
            import shutil
            shutil.copyfileobj(response, out)
        dest.with_suffix('.part').rename(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError('download hash mismatch')
    print(dest.name, digest, flush=True)
    return digest

def ocp():
    target = ROOT / 'ocp'
    target.mkdir(exist_ok=True)
    data = json.load(urllib.request.urlopen('https://pypi.org/pypi/cadquery-ocp-novtk/json', timeout=20))
    candidates = [(v,f) for v,files in data['releases'].items() if v=='7.9.3.1.1' for f in files if 'cp310-' in f['filename'] and 'manylinux' in f['filename'] and 'x86_64' in f['filename'] and not f['yanked']]
    if not candidates:
        raise ValueError('no supported Linux Python 3.10 wheel')
    version, wheel = sorted(candidates, key=lambda item:item[1]['upload_time'])[-1]
    for name, item in [('ocp', wheel)]:
        dest = ROOT / item['filename']
        fetch(item['url'],dest,item['digests']['sha256'])
        with zipfile.ZipFile(dest) as archive:
            archive.extractall(target)
    print(version,flush=True)

def vision():
    target = ROOT / 'vision'
    target.mkdir(exist_ok=True)
    for name in ['config.json','preprocessor_config.json','model.safetensors']:
        url = 'https://huggingface.co/facebook/dinov2-small/resolve/'+REV+'/'+name
        expected={'config.json':'1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d',
                  'preprocessor_config.json':'14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828',
                  'model.safetensors':'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'}
        fetch(url,target/name,expected[name])
    target = ROOT/'vf'
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--only-binary=:all:','--target',str(target),
        'transformers==4.39.3','safetensors==0.4.5','tokenizers==0.15.2','huggingface-hub==0.23.5','regex==2024.9.11','fsspec==2024.6.1','filelock==3.14.0','pyyaml==6.0.2','packaging==24.0','typing-extensions==4.12.2'],check=True)
if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    (ROOT/'tmp').mkdir(exist_ok=True)
    if sys.argv[1]=='ocp': ocp()
    elif sys.argv[1]=='vision': vision()
