"""Opt-in WSL entry point for a fixed-remainder, path-only order proposal.

The search uses native heur-out, DFS, pose generation, six-direction RedMax
path checks and the native two-part terminal choice. Stability is evaluated
only after the selected path chain is saved. No ranking service is used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from path_only_sequence_proposal import PROFILE, build_proposal
from proposal_support_policy import parse_support_cap, effective_support_cap
from proposal_support_score import score_plan
from proposal_stability import assess_step
from remote_rank_client import checked_path, IPC_ROOT
from run_sequence_native import input_snapshot, network_disabled, positive_int, positive_seconds, sha, source_snapshot, write_new
from sequence_native import snapshot
from evaluation_budget import PhysicsBudget, EvaluationLimit
from replay_path_extension import readonly_sdf_cache

ASAP = Path('/opt/asap-research/ASAP')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assembly-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--budget', type=positive_int, default=32, help='Maximum path evaluations, including the terminal removal.')
    p.add_argument('--max-poses', type=positive_int, default=128)
    p.add_argument('--plan-seconds', type=positive_seconds, default=420)
    p.add_argument('--audit-seconds', type=positive_seconds, default=240)
    p.add_argument('--support-cap', type=parse_support_cap, default='auto', metavar='auto|N',
                   help='Post-plan audit: auto allows up to all remaining parts at each step (default); N sets a fixed ceiling.')
    return p.parse_args(argv)


def audit_selected_steps(proposal, *, support_cap, audit_seconds, asset_folder, assembly_dir,
                         check_stable, emit):
    """Apply one audit deadline to the saved poses; each step starts with zero supports."""
    policy = parse_support_cap(support_cap)
    deadline = time.monotonic() + audit_seconds
    for step in proposal['steps']:
        cap = effective_support_cap(policy, len(step['child']))
        left = deadline - time.monotonic()
        if left <= 0:
            observation = dict(no_support_status='NOT_RUN', up_to_cap_status='NOT_RUN', reason='audit_deadline',
                               held_parts=None, trials=[], all_remaining_fixed=None)
        else:
            observation = assess_step(step, asset_folder=asset_folder, assembly_dir=assembly_dir,
                                      support_cap=cap, check_stable=check_stable, wall_time_s=left)
        observation['effective_support_cap'] = cap
        step['stability'] = snapshot(observation)
        emit('stability_result', attempt_id=step['attempt_id'], part_move=step['part_move'], **step['stability'])


def run(args):
    assembly = checked_path(args.assembly_dir, run_only=False)
    if not any(bound in assembly.parents for bound in (ASAP/'assets', IPC_ROOT)):
        raise ValueError('use a dedicated assembly under ASAP/assets or the project runs directory')
    out = checked_path(args.output_dir)
    if out == assembly or assembly in out.parents or out in assembly.parents:
        raise ValueError('input and output must be distinct, non-nested directories')
    if not 0 <= args.seed < 2**32:
        raise ValueError('invalid seed')
    args.support_cap = parse_support_cap(args.support_cap)
    out.mkdir(parents=True, exist_ok=False)
    verified_input = input_snapshot(assembly)
    sources = source_snapshot(ASAP)
    sys.path.insert(0, str(ASAP))
    import numpy as np
    import redmax_py
    import plan_sequence.feasibility_check as fc
    import plan_sequence.physics_planner as physics
    from plan_sequence.generator.heuristic import HeuristicsOutsidenessGenerator
    from plan_sequence.planner.dfs import DFSSequencePlanner

    def emit(kind, **fields):
        row = snapshot(dict(kind=kind, **fields))
        with (out/'events.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    asset = str(ASAP/'assets')
    write_new(out/'settings.json', dict(
        profile=PROFILE, assembly=str(assembly), parameters={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        api_calls=0, generator='heur-out', planner='dfs', base_part=None, pose_reuse=0,
        terminal='original_mass_rule', cache_policy='read_only_existing_SDF', input_sha256=verified_input,
        asap_commit=subprocess.check_output(['git','-C',str(ASAP),'rev-parse','HEAD'],text=True).strip(),
        asap_path_sources={k:v for k,v in sources.items() if k in (
            'plan_sequence/planner/base.py','plan_sequence/planner/dfs.py','plan_sequence/physics_planner.py',
            'plan_sequence/feasibility_check.py','plan_sequence/sim_string.py','plan_sequence/stable_pose.py')},
        redmax_binary_sha256=sha(Path(redmax_py.__file__)),
        runner_sha256={n:sha(Path(__file__).parent/n) for n in
                       ('run_path_only_proposal.py','path_only_sequence_proposal.py','proposal_stability.py',
                        'proposal_support_score.py','proposal_support_policy.py')},
        native_physics_constants={k:getattr(physics,k) for k in
            ('FORCE_MAG','FRAME_SKIP','MAX_TIME','POS_FAR_THRESHOLD','MAX_STEP','CONTACT_EPS','MIN_SEP','COL_TH_PATH','COL_TH_STABLE')}))
    guard = PhysicsBudget(path_limit=6*args.budget, stability_limit=1, seconds=args.plan_seconds)
    attempts = []
    class ProposalDFS(DFSSequencePlanner):
        def _simulate(self, part_move, parts_rest, parts_removed, pose, max_grippers, timeout=None,
                      grasp_planner=None, optimizer=None, debug=0, render=False):
            if grasp_planner is not None:
                raise ValueError('robot execution is outside the path proposal profile')
            guard.check()
            if len(attempts) >= args.budget:
                guard.stop_reason='composite_budget'
                raise EvaluationLimit(guard.stop_reason)
            attempt_id=len(attempts)+1
            parent=sorted([str(part_move)]+list(parts_rest))
            emit('path_start', attempt_id=attempt_id, parent=parent, part_move=part_move)
            action, path=fc.check_assemblable(asset, str(assembly), parts_rest, part_move, pose=pose,
                                              save_sdf=True, return_path=True)
            passed=action is not None
            saved=None
            if passed:
                values=np.asarray(path)
                if values.ndim!=2 or not len(values) or not np.isfinite(values).all():
                    raise ValueError('native success returned an invalid trajectory')
                filename='path_{:04d}.npy'.format(attempt_id)
                with (out/filename).open('xb') as f:
                    np.save(f,values,allow_pickle=False)
                saved=dict(file=filename,sha256=sha(out/filename),samples=len(values))
            ended=guard.clock()
            accepted=passed and ended <= guard.deadline
            row=snapshot(dict(attempt_id=attempt_id, parent=parent, child=list(parts_rest),
                              part_move=part_move, pose=pose, action=action, trajectory=saved,
                              path_pass=passed, accepted_for_transition=accepted,
                              finished_s=ended-guard.started, deadline_exceeded=ended > guard.deadline,
                              path_fixed_parts=list(parts_rest), terminal=len(parts_rest)==1))
            attempts.append(row)
            emit('path_result',**row)
            # Native DFS requires a list in feasible edges for its gripper bookkeeping.
            # These are the parts immobilized during the path check; no support test occurred here.
            return dict(feasible=accepted, part_move=part_move, pose=pose, base_part=None,
                        parts_fix=list(parts_rest) if accepted else None, action=action, grasp=None,
                        proposal_attempt=attempt_id)

    with readonly_sdf_cache(physics):
        with guard.installed(physics):
            generator=HeuristicsOutsidenessGenerator(asset,str(assembly),base_part=None,save_sdf=True)
            planner=ProposalDFS(generator,num_proc=1,save_sdf=True)
            planner.seed(args.seed)
            tree=planner.plan(budget=args.budget,max_grippers=len(planner.parts),max_poses=args.max_poses,
                              pose_reuse=0,early_term=True,timeout=guard.remaining())
        edges=[]
        for parent,child,data in tree.edges(data=True):
            info=data['sim_info']
            edge=attempts[info['proposal_attempt']-1]
            if (set(parent)!=set(edge['parent']) or set(child)!=set(edge['child']) or
                    bool(info['feasible'])!=edge['accepted_for_transition']):
                raise ValueError('tree edge differs from verified transition')
            edges.append(edge)
        proposal=build_proposal(planner.parts,sorted(edges,key=lambda e:e['attempt_id']))
        proposal.update(candidate_policy='heur-out', planner_stop=planner.stop_msg,
                        budget_stop=guard.stop_reason, api_calls=0, initial_parts=planner.parts,
                        execution_status='ERROR' if planner.stop_msg=='exception' and not guard.stop_reason else 'OK',
                        path_physics=guard.report())
        # Upstream can stop when it first reaches two parts; only an accepted 2->1 edge completes our chain.
        write_new(out/'proposal.json',proposal)
        emit('candidate_saved',complete=proposal['complete'],order=proposal['disassembly_order'],
             remaining=proposal['remaining_parts'],path_checks=guard.counts['path'],stop=planner.stop_msg)
        def check_stability(asset_folder,assembly_dir,fixed,free,pose,timeout):
            return fc.check_stable(asset_folder,assembly_dir,fixed,free,
                                   pose=None if pose is None else np.asarray(pose),
                                   save_sdf=True,allow_gap=False,timeout=timeout)
        audit_selected_steps(proposal, support_cap=args.support_cap, audit_seconds=args.audit_seconds,
                             asset_folder=asset, assembly_dir=str(assembly), check_stable=check_stability, emit=emit)
    proposal['input_unchanged']=input_snapshot(assembly)==verified_input
    proposal['asap_sources_unchanged']=source_snapshot(ASAP)==sources
    if not proposal['input_unchanged'] or not proposal['asap_sources_unchanged']:
        raise RuntimeError('input/source preservation check failed')
    proposal['settings_sha256'] = sha(out/'settings.json')
    proposal['support_score'] = score_plan(proposal, support_cap=args.support_cap)
    write_new(out/'result.json',proposal)
    return 0 if proposal['execution_status']=='OK' else 1


if __name__=='__main__':
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    with network_disabled():
        raise SystemExit(run(parse_args()))
