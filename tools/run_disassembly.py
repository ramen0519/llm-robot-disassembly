"""Default T0–T5 entry point. Preparations and paid execution are separate modes."""
import argparse
from decimal import Decimal
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from caseplanner import CASE_COLLECTION
from caseplanner.io import location, read, write


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--assembly',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--unit',choices=['mm','cm','m','inch'])
    p.add_argument('--assembled-obj',action='store_true')
    p.add_argument('--base')
    p.add_argument('--mode',choices=['astra','fixture','prepare'],default='astra')
    p.add_argument('--asap-root',default='/opt/asap-research/ASAP')
    p.add_argument('--store', default=str(ROOT / '.cases' / CASE_COLLECTION),
                   help='Case database for the current collection; defaults to the fresh research store.')
    p.add_argument('--vision',default=str(ROOT/'.runtime/vision'))
    p.add_argument('--config',default=str(ROOT/'configs/astra_cases_v1.json'))
    p.add_argument('--exclude-family',action='append',default=[])
    p.add_argument('--family',required=True)
    p.add_argument('--evaluation',action='store_true')
    p.add_argument('--memory-off',action='store_true')
    p.add_argument('--frozen-index',help='reuse a previously recorded eligible index for paired evaluations')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-attempts',type=int,default=16)
    p.add_argument('--max-path',type=int,default=96)
    p.add_argument('--max-stability',type=int,default=64)
    p.add_argument('--max-poses',type=int,default=3)
    p.add_argument('--max-grippers',type=int,default=3)
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--max-calls',type=int,default=0)
    p.add_argument('--max-usd',default='0')
    p.add_argument('--rank-attempts',type=int,choices=[1,2],default=2)
    p.add_argument('--proximity-mm',type=float,default=.1)
    p.add_argument('--fixture-order',nargs='*',default=[])
    return p


def main(argv=None, *, test_transport=None):
    args=parser().parse_args(argv)
    for key in ('input','run','asap_root','store','vision','config'):
        setattr(args,key,location(getattr(args,key)))
    if args.frozen_index: args.frozen_index=location(args.frozen_index)
    if str(ROOT).startswith('/mnt/f/disassembly_research/'):
        for target in (args.run,args.store):
            if Path('/mnt/f/disassembly_research') not in target.parents:
                raise ValueError('this installation stores runs and case libraries on F')
    if args.mode=='astra' and (args.max_calls<1 or not Decimal(args.max_usd).is_finite() or Decimal(args.max_usd)<=0):
        raise ValueError('live Astra requires explicit positive call and cost caps')
    if args.mode!='fixture' and args.fixture_order:
        raise ValueError('fixture order only allowed in fixture mode')
    if min(args.max_attempts,args.max_path,args.max_stability,args.max_poses,args.max_grippers,args.seconds)<=0:
        raise ValueError('budgets must be positive')
    args.run.mkdir(parents=True,exist_ok=False)
    import time
    from caseplanner.io import sha
    import subprocess
    tracked=[ROOT/'tools/run_disassembly.py',ROOT/'tools/run_sequence_native_experiment.py',args.config]
    tracked+=sorted((ROOT/'caseplanner').glob('*.py'))+[ROOT/'caseplanner/raster.c']
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    write(args.run/'code_versions.json',dict(base_commit=head,files={str(p.relative_to(ROOT)) if ROOT in p.parents else p.name:sha(p) for p in tracked}))
    from caseplanner.memory import Library
    library=None
    if args.mode!='prepare':
        excluded=set(args.exclude_family)|({args.family} if args.evaluation else set())
        if test_transport is not None:
            if args.mode!='astra': raise ValueError('test transport is a Responses contract double')
            args.store=args.store/'offline_provider_test_only'
        library=Library(args.store,excluded,not args.memory_off,args.frozen_index,
                        domain='offline_contract' if test_transport is not None else 'research')
        snapshot = library.snapshot()
        snapshot.update(exclude_families=sorted(excluded), evaluation=args.evaluation, memory_off=args.memory_off)
        write(args.run/'frozen_index.json', snapshot)
    start=time.monotonic()
    write(args.run/'settings.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    phase='input'
    try:
        from caseplanner.geometry import prepare
        from caseplanner.images import Scene, CAMERA
        geom=prepare(args.input,args.run/'input',args.assembly,args.unit,args.assembled_obj,args.proximity_mm)
        scene=Scene(args.run/'input')
        aliases={p['source_part_id']:p['part_id'] for p in geom['parts']}
        if args.base is not None: args.base=aliases.get(args.base,args.base)
        if args.fixture_order: args.fixture_order=[aliases.get(p,p) for p in args.fixture_order]
        if args.base is not None and args.base not in scene.meshes:
            raise ValueError('fixed base missing')
        scene.render(scene.assembled,args.run/'input/top.png')
        scene.render(scene.assembled,args.run/'input/oblique.png',direction=(1,1,1))
        write(args.run/'input/views.json',dict(camera=CAMERA,frame='CAD assembly',top=[0,0,1],oblique=[1,1,1],
            projection='orthographic',sources='placed source meshes',prepare_seconds=time.monotonic()-start))
        if args.mode=='prepare':
            write(args.run/'result.json',dict(status='PREPARED',parts=len(geom['parts']),physics=0,api=0))
            return 0
        phase='runtime'
        from caseplanner.run import execute
        return execute(args,scene,library,test_transport)
    except Exception as exc:
        import traceback
        write(args.run/'error.json',dict(status='ERROR',stage=phase,outcome='input_invalid' if phase=='input' and isinstance(exc,(ValueError,FileNotFoundError,KeyError)) else 'execution_error',kind=type(exc).__name__,detail=str(exc),trace=traceback.format_exc()))
        raise
if __name__=='__main__':
    sys.exit(main())
