"""Original ASAP DFS/physics with external part ordering only; no worker API."""
import os
os.environ.update(OMP_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import urllib.request

from evaluation_budget import PhysicsBudget, EvaluationLimit
from independent_features import describe_task
from rank_hook import EventLog
from remote_rank_client import RemoteRankClient, checked_path
from replay_path_extension import readonly_sdf_cache
from sequence_native import (PROFILE, TERMINAL_POLICY, DecisionHalted, SequenceNativeSession,
    connected_path, installed_sequence, snapshot, tree_snapshot)
from task_memory import task_context


COMMIT = '3727e716b57e0b508c4e9ecd9fefa6953e8f72da'


def positive_int(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('positive integer required')
    return n


def positive_seconds(value):
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError('positive finite seconds required')
    return n


def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog='Use fasteners --help for the offline joint planning stage.')
    p.add_argument('--asap-root', type=Path, required=True)
    p.add_argument('--assembly-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--base-part', default=None)
    p.add_argument('--budget', type=positive_int, default=256)
    p.add_argument('--path-limit', type=positive_int, default=256)
    p.add_argument('--stability-limit', type=positive_int, default=256)
    p.add_argument('--timeout', type=positive_seconds, default=3600)
    p.add_argument('--response-timeout', type=positive_seconds, default=120)
    p.add_argument('--max-gripper', type=positive_int, default=3)
    p.add_argument('--max-pose', type=positive_int, default=128)
    p.add_argument('--prepare-only', action='store_true', help='Read CAD; no ranking or physics')
    add_fastener_args(p, required=False)
    return p


def sha(path):
    return hashlib.sha256(checked_path(path, run_only=False).read_bytes()).hexdigest()


def write_new(path, value):
    with checked_path(path).open('x', encoding='utf-8') as f:
        json.dump(snapshot(value), f, indent=2, allow_nan=False)


def files(folder):
    for p in checked_path(folder, run_only=False).iterdir():
        p = checked_path(p, run_only=False)
        if p.is_dir():
            yield from files(p)
        elif p.is_file():
            yield p


def source_snapshot(root):
    return {str(p.relative_to(root)): sha(p) for directory in
            ('assets', 'plan_sequence', 'plan_path', 'plan_robot', 'utils')
            for p in files(root/directory) if p.suffix == '.py'}


def input_snapshot(directory):
    return {p.relative_to(directory).as_posix(): sha(p) for p in files(directory)}


def initialize_output(directory):
    """Allow the host to precreate empty IPC directories, never reuse old work."""
    out = checked_path(directory)
    if out.exists():
        for entry in out.iterdir():
            entry = checked_path(entry)
            if entry.name != 'ipc' or not entry.is_dir():
                raise ValueError('output_contains_previous_or_unexpected_content')
            for child in entry.iterdir():
                child = checked_path(child)
                if child.name not in ('requests', 'responses') or not child.is_dir():
                    raise ValueError('output_contains_previous_or_unexpected_IPC_content')
                if any(child.iterdir()):
                    raise ValueError('output_contains_previous_IPC_messages')
    else:
        out.mkdir(parents=True, exist_ok=False)
    return out


@contextmanager
def network_disabled():
    def denied(*args, **kwargs):
        raise RuntimeError('NETWORK_NOT_ALLOWED_IN_SEQUENCE_WORKER')
    connect, opener = socket.socket.connect, urllib.request.OpenerDirector.open
    socket.socket.connect = denied
    urllib.request.OpenerDirector.open = denied
    try:
        yield
    finally:
        socket.socket.connect, urllib.request.OpenerDirector.open = connect, opener


def add_fastener_args(p, required):
    p.add_argument('--joint', required=required, type=Path, help='Declared connection and CAD occurrences')
    p.add_argument('--bom', required=required, type=Path, help='STEP placement inventory')
    p.add_argument('--events', type=Path, help='Ordered planning observations; omitted for initial state')


def read_joint_json(path):
    return json.loads(checked_path(Path.cwd() / path, run_only=False).read_text(encoding='utf-8-sig'))


def joint_stage(args):
    from fastener_sequence import BoltedJoint, plan_joint
    if not args.joint or not args.bom:
        raise ValueError('joint and BOM must be supplied together')
    bom = read_joint_json(args.bom)
    joint = BoltedJoint.from_bom(read_joint_json(args.joint), bom)
    events = (read_joint_json(args.events) if args.events else
              dict(joint_id=joint.id, source_step_sha256=joint.step_hash, events=[]))
    return joint, bom, plan_joint(joint, events)


def run_fastener_stage(argv):
    """Evaluate explicit joint operations before geometry, without loading ASAP or a model."""
    p = argparse.ArgumentParser(description='Propose separate removals from bolt event history.')
    add_fastener_args(p, required=True)
    _, _, result = joint_stage(p.parse_args(argv))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main():
    if sys.argv[1:2] == ['fasteners']:
        return run_fastener_stage(sys.argv[2:])
    args = parser().parse_args()
    # Native DFS keys nodes only by remaining parts. Do not start it from a
    # joined pair or with a bolt released but still in the hole.
    stage = None
    if args.joint or args.bom or args.events:
        stage = joint_stage(args)
        if not stage[2]['candidate_requests']:
            raise ValueError('joint still blocks separate removal; native physics not started')
    if (not 0 <= args.seed < 2**32 or not args.run_id or len(args.run_id) > 256
            or args.response_timeout > 120):
        raise ValueError('invalid_seed_run_id_or_response_timeout')
    root = checked_path(args.asap_root, run_only=False)
    assembly = checked_path(args.assembly_dir, run_only=False)
    out = initialize_output(args.output_dir)
    guard = PhysicsBudget(args.path_limit, args.stability_limit, args.timeout)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                     universal_newlines=True).strip()
    if commit != COMMIT:
        raise ValueError('unsupported_ASAP_source_revision')
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    sources, inputs = source_snapshot(root), input_snapshot(assembly)
    if stage:
        from fastener_sequence import bind_released_scene
        joint, bom, report = stage
        mapped_id = bind_released_scene(joint, bom, read_joint_json(assembly/'fastener_occurrences.json'),
                                        inputs, args.base_part)
        write_new(out/'fastener_handoff.json', dict(plan=report, eligible_native_part=mapped_id))
    write_new(out/'source_before.json', sources)
    write_new(out/'input_before.json', inputs)
    config = dict(profile=PROFILE, planner='dfs', generator='external_sequence',
        seed=args.seed, base_part=args.base_part, budget=args.budget,
        path_limit=args.path_limit, stability_limit=args.stability_limit,
        timeout=args.timeout, response_timeout=args.response_timeout,
        max_gripper=args.max_gripper, max_pose=args.max_pose, pose_reuse=0,
        num_proc=1, early_term=True, terminal_policy=TERMINAL_POLICY,
        automatic_retries=0, candidate_pruning=False, physics_verdict_reuse=False,
        decision_failure_policy='halt_no_canonical_or_heuristic_fallback',
        full_robot_execution=False, worker_API_calls=0)
    write_new(out/'configuration.json', config)
    settings = dict(budget=args.budget, max_grippers=args.max_gripper,
        max_poses=args.max_pose, pose_reuse=0, early_term=True, timeout=args.timeout,
        path_limit=args.path_limit, stability_limit=args.stability_limit,
        plan_grasp=False, plan_arm=False, gripper_type='robotiq-140', gripper_scale=.4,
        optimizer='L-BFGS-B', seed=args.seed, save_sdf=True, num_proc=1,
        sequence_profile=PROFILE, terminal_policy=TERMINAL_POLICY)
    sys.path.insert(0, str(root))
    with network_disabled():
        import numpy as np
        import redmax_py
        import plan_sequence.physics_planner as physics
        from plan_sequence.generator.base import Generator
        from plan_sequence.generator import generators
        from plan_sequence.planner.base import SequencePlanner
        from plan_sequence.planner.dfs import DFSSequencePlanner
        features = describe_task(root, assembly, args.base_part, settings)
        write_new(out/'features.json', features)
        write_new(out/'context.json', task_context(features))
        hash_paths = dict(physics_source_sha256=root/'plan_sequence/physics_planner.py',
            sim_string_sha256=root/'plan_sequence/sim_string.py',
            assets_load_sha256=root/'assets/load.py', assets_transform_sha256=root/'assets/transform.py',
            plan_path_run_connect_sha256=root/'plan_path/run_connect.py',
            redmax_binary_sha256=Path(redmax_py.__file__),
            readonly_sdf_sha256=Path(__file__).parent/'replay_path_extension.py',
            feasibility_search_sha256=root/'plan_sequence/feasibility_check.py',
            stable_pose_sha256=root/'plan_sequence/stable_pose.py',
            planner_base_sha256=root/'plan_sequence/planner/base.py',
            planner_dfs_sha256=root/'plan_sequence/planner/dfs.py')
        binding = {k: sha(p) for k, p in hash_paths.items()}
        constants = {k: getattr(physics, k) for k in ('FORCE_MAG', 'FRAME_SKIP', 'MAX_TIME',
            'CONTACT_EPS', 'POS_FAR_THRESHOLD', 'POS_NEAR_THRESHOLD', 'NEAR_STEP', 'MAX_STEP',
            'CHECK_FREQ', 'COL_TH_PATH', 'COL_TH_STABLE', 'MIN_SEP')}
        write_new(out/'physics_binding.json', dict(source_hashes=binding, constants=constants,
            SDF_cache_files=sum(name.endswith('.sdf') for name in inputs),
            constructor_save_sdf=True, xml_load_sdf=True, xml_save_sdf=False,
            path_success_cache=False, stable_pose_policy='original_trimesh_probability_order',
            support_policy='original_greedy_parts_fall', direction_policy='original_six_axis_search'))
        if args.prepare_only:
            assert input_snapshot(assembly) == inputs and source_snapshot(root) == sources
            write_new(out/'preparation.json', dict(status='PASS', mode='CAD_PREPARATION_ONLY',
                parts=len(features['parts']), physics_calls=0, rank_requests=0, API_calls=0,
                GNN_calls=0, elapsed_seconds=guard.report()['elapsed_seconds'], input_unchanged=True,
                sources_unchanged=True))
            print(json.dumps(dict(status='PREPARATION_PASS', parts=len(features['parts']), API=0, physics=0)), flush=True)
            return 0

        def rng_digest():
            payload = dict(python=repr(random.getstate()), numpy=repr(np.random.get_state()))
            if 'torch' in sys.modules:
                payload['torch'] = sys.modules['torch'].get_rng_state().tolist()
            return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()

        original_methods = {'simulate': SequencePlanner._simulate, 'update_tree': SequencePlanner._update_tree,
            'path_check': physics.MultiPartPathPlanner.check_success,
            'stability_check': physics.MultiPartStabilityPlanner.check_success}
        method_hashes = {k: hashlib.sha256(inspect.getsource(v).encode('utf-8')).hexdigest()
                         for k, v in original_methods.items()}
        client = RemoteRankClient(out/'ipc', out/'remote_audit.jsonl', timeout_seconds=args.response_timeout)
        log = EventLog(out/'events.jsonl', args.run_id)
        session = SequenceNativeSession(log, client, features, guard, args.budget, rng_digest, binding)
        tree, planner, stats, error = None, None, {}, None
        evaluation_finished = None
        try:
            session.emit('run_start', configuration=config, source_commit=commit,
                source_methods_sha256=method_hashes, actual_physics=True,
                candidate_policy='external_complete_permutation', native_order_in_prompt=False)
            with readonly_sdf_cache(physics):
                with guard.installed(physics):
                    with installed_sequence(Generator, generators, SequencePlanner, session) as Ranked:
                        guard.check()
                        generator = Ranked(str(root/'assets'), str(assembly), base_part=args.base_part, save_sdf=True)
                        planner = DFSSequencePlanner(generator, 1, save_sdf=True)
                        planner.seed(args.seed)
                        guard.check()
                        tree = planner.plan(budget=args.budget, max_grippers=args.max_gripper,
                            max_poses=args.max_pose, pose_reuse=0, early_term=True,
                            timeout=guard.remaining(), debug=0)
                        evaluation_finished = session.elapsed()
                        stats = planner.get_stats(tree)
                        planner.log(tree, stats, str(out/'native'))
                        if stats.get('stop_msg') == 'exception' and not session.halt_reason and not guard.stop_reason:
                            error = 'NativeOrObserverException'
        except (DecisionHalted, EvaluationLimit):
            tree = tree if tree is not None else session.last_tree
        except Exception as exc:
            error = type(exc).__name__
            session.emit('execution_error', error_type=error)
            tree = tree if tree is not None else session.last_tree
        finally:
            if evaluation_finished is None:
                evaluation_finished = session.elapsed()
            client.close()
        restored = (SequencePlanner._simulate is original_methods['simulate'] and
            SequencePlanner._update_tree is original_methods['update_tree'] and
            physics.MultiPartPathPlanner.check_success is original_methods['path_check'] and
            physics.MultiPartStabilityPlanner.check_success is original_methods['stability_check'])
        input_ok, source_ok = input_snapshot(assembly) == inputs, source_snapshot(root) == sources
        if not all((restored, input_ok, source_ok)):
            error = 'PreservationOrRestorationFailure'
        parts = planner.parts if planner is not None else sorted(features['parts'])
        chain = connected_path(tree, parts)
        success = bool(chain['success'] and not error and not session.halt_reason and session.completed_in_budget())
        selected = chain['complete_chain'] if success else []
        if tree is not None:
            write_new(out/'tree.json', tree_snapshot(tree))
        result = dict(schema_version=1, profile=PROFILE, run_id=args.run_id,
            status='PASS' if success else 'ERROR' if error else 'UNKNOWN', success=success,
            actual_physics=True, API_calls=0, api_in_worker=False,
            rank_request_count=session.generation_count, rank_decisions=session.rank_decisions,
            decision_halt_reason=session.halt_reason, error=error,
            stop_reason=session.halt_reason or guard.stop_reason or stats.get('stop_msg'),
            native_stats=stats, connected=chain, selected_plan=selected,
            attempts=session.attempts, transitions=session.transitions,
            episode_feedback=session.feedback, budget=guard.report(),
            source_commit=commit, configuration=config, physics_binding=binding,
            physics_constants=constants, source_methods_sha256=method_hashes,
            source_methods_restored=restored, sources_unchanged=source_ok, input_unchanged=input_ok,
            heuristic_tripwire_calls=session.tripwire_calls, terminal_policy=TERMINAL_POLICY,
            complete_permutation=True, hard_pruning=False, lower_level_policy='original_ASAP',
            completion_elapsed_seconds=session.completion_elapsed_seconds,
            evaluation_finished_elapsed_seconds=evaluation_finished,
            postprocessing_elapsed_seconds=session.elapsed()-evaluation_finished,
            elapsed_seconds=session.elapsed(), full_robot_execution=False)
        write_new(out/'worker_result.json', result)
        write_new(out/'native_result.json', result)
        session.emit('run_end', status=result['status'], success=success,
            max_depth=chain['max_depth'], rank_requests=session.generation_count,
            counts=guard.counts, stop_reason=result['stop_reason'])
        log.close()
        print(json.dumps(dict(status=result['status'], success=success,
            max_depth=chain['max_depth'], counts=guard.counts,
            rank_requests=session.generation_count, stop_reason=result['stop_reason'])), flush=True)
        return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
