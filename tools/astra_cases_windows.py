"""Run the same Linux default entry, using registered Windows credentials only in live mode."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/astra_cases_v1.json').read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--check-model',action='store_true')
    p.add_argument('--credential-root',type=Path,default=Path('C:/dev/disassembly_codex_ready_v2'))
    p.add_argument('argv',nargs=argparse.REMAINDER)
    args=p.parse_args()
    argv=args.argv[1:] if args.argv[:1]==['--'] else args.argv
    if not args.check_model and not argv: p.error('provide run arguments after --')
    if ROOT.drive.upper()!='F:': raise ValueError('use the F research checkout')
    linux='/mnt/f/'+ROOT.as_posix()[3:]
    import local_api_key
    from contextlib import nullcontext
    options=argparse.ArgumentParser(add_help=False)
    options.add_argument('--mode',default='astra')
    selected,_=options.parse_known_args(argv)
    paid=selected.mode=='astra' and not any(x in argv for x in ('-h','--help'))
    for idx,value in enumerate(argv):
        if len(value)>2 and value[1:3] in (':/',':'+chr(92)):
            if value[0].lower() not in ('c','f'): raise ValueError('unsupported volume')
            argv[idx]='/mnt/'+value[0].lower()+'/'+value[3:].replace(chr(92),'/')
    with local_api_key.key_environment(args.credential_root) if args.check_model or paid else nullcontext():
        if args.check_model:
            key=os.environ.get('OPENAI_API_KEY')
            if not key: raise RuntimeError('no configured credential')
            request=urllib.request.Request('https://api.openai.com/v1/models/'+CFG['model'],headers={'Authorization':'Bearer '+key})
            try:
                with urllib.request.urlopen(request,timeout=15) as response:
                    result=json.load(response)
                print(json.dumps(dict(model_id=result.get('id'),configured=CFG['model'],available=result.get('id')==CFG['model'],
                                      generation_calls=0,project_data_sent=False)))
                return 0 if result.get('id')==CFG['model'] else 1
            except urllib.error.HTTPError as exc:
                print(json.dumps(dict(model_id=CFG['model'],metadata_http=exc.code,generation_calls=0)))
                return 1
        env=os.environ.copy()
        if paid: env['WSLENV']=':'.join(x for x in [env.get('WSLENV',''),'OPENAI_API_KEY/u'] if x)
        runtime=linux+'/.runtime/'
        cmd=['wsl','-d','ASAP-Ubuntu','-u','root','--cd',linux,'--','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
            'MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','MPLBACKEND=Agg','MPLCONFIGDIR='+runtime+'mpl',
            'XDG_CACHE_HOME='+runtime+'cache','TMPDIR='+runtime+'tmp','/opt/asap-research/envs/asap/bin/python','-B','-u',
            'tools/run_sequence_native_experiment.py']+argv
        return subprocess.run(cmd,env=env).returncode
if __name__=='__main__':
    sys.exit(main())
