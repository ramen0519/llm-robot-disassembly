"""Unchanged ASAP decisions with opt-in resource accounting and read-only cache."""
import os
os.environ['OMP_NUM_THREADS']='1'
import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'worker'))
from remote_rank_client import checked_path
from evaluation_budget import PhysicsBudget,EvaluationLimit
from replay_path_extension import readonly_sdf_cache
from transfer_protocol import validate_manifest,verify_input


def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser()
    for name in ('asap-root','assembly-dir','output-dir','evaluation-manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--unmetered-regression',action='store_true')
    args=p.parse_args()
    root=checked_path(args.asap_root,False);assembly=checked_path(args.assembly_dir,False);out=checked_path(args.output_dir)
    manifest=json.loads(checked_path(args.evaluation_manifest).read_text())
    for k,v in manifest['settings'].items():setattr(args,k,v)
    args.seed=0;args.base_part=manifest.get('development_base_part')
    validate_manifest(args.evaluation_manifest,str(assembly),args)
    before=verify_input(assembly,manifest)
    if args.unmetered_regression and manifest['purpose']!='development':raise ValueError('unmetered_only_development')
    out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],universal_newlines=True).strip()
    if commit!='3727e716b57e0b508c4e9ecd9fefa6953e8f72da':raise ValueError('wrong_ASAP_commit')
    subprocess.run(['git','-C',str(root),'diff','--exit-code','HEAD','--'],check=True)
    sys.path.insert(0,str(root))
    import numpy as np
    import plan_sequence.physics_planner as physics
    from plan_sequence.generator.heuristic import HeuristicsOutsidenessGenerator
    from plan_sequence.planner.dfs import DFSSequencePlanner
    from plan_sequence.planner.base import NumpyEncoder
    guard=PhysicsBudget(manifest['path_limit'],manifest['stability_limit'],args.timeout)
    attempts=[];active={};error=None;stats={};depth=0
    with readonly_sdf_cache(physics), guard.installed(physics) if not args.unmetered_regression else nullcontext():
        generator=HeuristicsOutsidenessGenerator(str(root/'assets'),str(assembly),base_part=args.base_part,save_sdf=True)
        planner=DFSSequencePlanner(generator,1,save_sdf=True);planner.seed(0)
        original=planner._simulate
        original_update=planner._update_tree
        def simulate(*a,**kw):
            if not args.unmetered_regression:
                guard.check()
                if len(attempts)>=args.budget:
                    guard.stop_reason='composite_budget';raise EvaluationLimit('composite_budget')
            started=time.monotonic();before_counts=dict(guard.counts)
            row={'part_move':a[0] if a else kw.get('part_move'),'parts_rest':a[1] if len(a)>1 else kw.get('parts_rest'),
                 'feasible':False,'returned':False}
            attempts.append(row)
            try:
                value=original(*a,**kw)
                row.update(returned=True,raw_sim_info=json.loads(json.dumps(value,cls=NumpyEncoder)),feasible=bool(value['feasible']))
                if not args.unmetered_regression:guard.check()
                return value
            except Exception as e:
                row['error_type']=type(e).__name__;raise
            finally:
                row.update(seconds=time.monotonic()-started,counts={k:guard.counts[k]-before_counts[k] for k in before_counts})
                with (out/'attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps({'attempt':len(attempts),'part':row['part_move'],'feasible':row['feasible'],'counts':guard.counts}),flush=True)
        def update(*a,**kw):
            if not args.unmetered_regression:guard.check()
            return original_update(*a,**kw)
        planner._simulate=simulate;planner._update_tree=update
        tree=planner.plan(budget=args.budget,max_grippers=args.max_gripper,max_poses=args.max_pose,
                          pose_reuse=0,early_term=True,timeout=guard.remaining(),debug=0)
        stats=planner.get_stats(tree)
        feasible=[node for node,data in tree.nodes(data=True) if data.get('n_gripper') is not None]
        depth=max((len(planner.parts)-len(node) for node in feasible),default=0)
        planner.log(tree,stats,str(out/'native'))
    verify_input(assembly,manifest)
    success=bool(stats.get('success')) and guard.remaining()>0
    result=dict(status='PASS' if success else 'UNKNOWN',success=success,native_stats=stats,
        max_depth=depth,part_count=len(planner.parts),attempts=attempts,actual_physics=True,
        source_commit=commit,decision_policy='original_dfs_heur_out',base_part=args.base_part,
        budget=guard.report(),input_unchanged=True,unmetered_regression=args.unmetered_regression,
        physics_settings={k:getattr(physics,k) for k in ('FORCE_MAG','FRAME_SKIP','MAX_TIME','CONTACT_EPS','POS_FAR_THRESHOLD','POS_NEAR_THRESHOLD','NEAR_STEP','MAX_STEP','CHECK_FREQ','COL_TH_PATH','COL_TH_STABLE','MIN_SEP')})
    write(out/'native_result.json',result)
    print(json.dumps({'status':result['status'],'max_depth':depth,'physics':guard.counts,'stop':stats.get('stop_msg')}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
