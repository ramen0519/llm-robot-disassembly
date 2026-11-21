"""Actual ASAP worker for explicit LLM/fixture action selection; no network/key.

Extends the ASAP DFS adapter for action retries and terminal-two decisions.
It does not run the upstream heuristic, mass terminal chooser, or greedy holds.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from contextlib import nullcontext

PROJECT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT))

from remote_rank_client import checked_path
from explicit_action_ipc import ExplicitActionClient
from independent_features import describe_task, filter_features
from task_memory import task_context
from run_rank_hook import COMMIT
from explicit_request_budget import (bounded_memory_request, capacity_probe,
                                     RequestCoreTooLarge)
from pose_catalog import POLICIES, PoseCatalogError, select_pose_catalog
from worker.accepted_pose_carry import POLICIES as TRANSITION_POLICIES


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def digest_files(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('asap-root', 'assembly-dir', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--base-part', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--budget', type=int, default=16)
    p.add_argument('--state-action-budget', type=int, default=12)
    p.add_argument('--max-gripper', type=int, default=3)
    p.add_argument('--max-pose', type=int, default=3)
    p.add_argument('--pose-catalog-policy', choices=POLICIES, default='legacy_truncate_v1')
    p.add_argument('--pose-transition-policy', choices=TRANSITION_POLICIES, default='legacy_v1')
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--response-timeout', type=int, default=90)
    p.add_argument('--frozen-memory', type=Path)
    p.add_argument('--memory-policy', choices=('legacy_v1', 'state_diverse_v1', 'state_balanced_v2', 'transfer_v1', 'assembly_balanced_v1'), default='legacy_v1')
    p.add_argument('--evaluation-manifest', type=Path)
    p.add_argument('--memory-max-records', type=int, default=4)
    p.add_argument('--fallback-policy', choices=('legacy_v1', 'path_stage_v1'), default='legacy_v1')
    p.add_argument('--path-selection-policy', choices=('legacy_v1', 'path_probe_round_v1'), default='legacy_v1')
    p.add_argument('--decision-failure-policy', choices=('legacy_v1', 'halt_v1'), default='legacy_v1')
    p.add_argument('--stability-diagnostics', action='store_true')
    p.add_argument('--physics-scene-policy', choices=('fresh_v1', 'path_scene_reuse_v1'),
                   default='fresh_v1')
    p.add_argument('--request-preflight-only', action='store_true',
                   help='Generate real CAD/pose requests and synthetic payload capacity checks; no API/physics.')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    root = checked_path(args.asap_root, run_only=False)
    assembly = checked_path(args.assembly_dir, run_only=False)
    output = checked_path(args.output_dir)
    evaluation = budget_guard = None
    if args.evaluation_manifest:
        from transfer_protocol import validate_manifest, verify_input
        from evaluation_budget import PhysicsBudget
        evaluation = validate_manifest(checked_path(args.evaluation_manifest), str(assembly), args)
        verify_input(assembly, evaluation)
        budget_guard = PhysicsBudget(evaluation['path_limit'], evaluation['stability_limit'], args.timeout)
    if any(v <= 0 for v in (args.budget, args.state_action_budget, args.max_gripper, args.max_pose, args.timeout)):
        raise ValueError('positive_budget_required')
    if not 0 <= args.seed < 2**32:
        raise ValueError('invalid_seed')
    if not 0 <= args.memory_max_records <= 12:
        raise ValueError('invalid_retrieval_limit')
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    if commit != COMMIT:
        raise ValueError('unsupported_upstream_commit')
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import trimesh
    from assets.load import load_part_ids
    from assets.transform import transform_pt_by_matrix
    from plan_sequence.stable_pose import get_combined_mesh
    from orchestrator.explicit_action_memory import FrozenActionMemory, load_snapshot

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    initial_files = digest_files(assembly)
    write(output / 'input_before.json', initial_files)
    settings = {'seed': args.seed, 'budget': args.budget, 'max_grippers': args.max_gripper,
                'max_poses': args.max_pose, 'state_action_budget': args.state_action_budget,
                'timeout': args.timeout, 'save_sdf': True, 'read_only_sdf': True,
                'plan_grasp': False, 'plan_arm': False, 'num_proc': 1,
                'search': 'extended_ASAP_DFS_explicit_actions_v1',
                'pose_policy': 'geometry_enumeration_hash_order_no_probability_scores',
                'support_budget': 'additional_supports_plus_mover_excluding_fixed_base',
                'terminal_policy': 'same_explicit_selection_until_one_part'}
    if args.pose_catalog_policy != 'legacy_truncate_v1':
        settings['pose_catalog_policy'] = args.pose_catalog_policy
    if args.pose_transition_policy != 'legacy_v1':
        settings['pose_transition_policy'] = args.pose_transition_policy
    if args.decision_failure_policy != 'legacy_v1':
        settings['decision_failure_policy'] = args.decision_failure_policy
    if args.physics_scene_policy != 'fresh_v1':
        settings['physics_scene_policy'] = args.physics_scene_policy
    features = describe_task(root, assembly, args.base_part, settings)
    context = task_context(features)
    write(output / 'features.json', features)
    write(output / 'context.json', context)
    event_count = [0]
    def emit(event=None, **fields):
        event_count[0] += 1
        item = dict(event or {}, **fields) if isinstance(event, dict) else dict(kind=event, **fields)
        item.setdefault('event_id', args.run_id+':e'+str(event_count[0]))
        item['run_id'] = args.run_id
        with (output / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False)+'\n')
    poses = {}
    pose_audit = []
    def pose_provider(remaining):
        key = tuple(sorted(remaining))
        if key in poses:
            return deepcopy(poses[key])
        started = time.monotonic()
        if args.base_part is not None:
            raw = [np.eye(4)]
            description = 'fixed_base_identity_pose'
        else:
            mesh = get_combined_mesh(str(assembly), list(key))
            raw, _probabilities_not_used = trimesh.poses.compute_stable_poses(mesh, n_samples=1)
            raw = [mat.copy() for mat in raw]
            for mat in raw:
                mat[:2, 3] -= transform_pt_by_matrix(mesh.centroid, mat)[:2]
            description = 'trimesh_geometric_candidates_no_probabilities_to_selector'
        unique = {}
        for mat in raw:
            matrix = np.round(mat, 10).tolist()
            fingerprint = hashlib.sha256(json.dumps(matrix, separators=(',', ':')).encode()).hexdigest()
            unique[fingerprint] = {'pose_id': 'pose_'+fingerprint[:16], 'matrix': matrix}
        try:
            selected = select_pose_catalog(unique, args.max_pose, args.pose_catalog_policy,
                                           evaluation=bool(evaluation))
        except PoseCatalogError as error:
            # Record the state that failed before it can enter the state cache.
            # Never replace an incomplete catalog with a guessed feasible subset.
            failure = dict(error.audit, remaining_parts=list(key), status='ERROR',
                offered_count=0, max_pose=args.max_pose, policy=description,
                delivery_policy=args.pose_catalog_policy,
                seconds=time.monotonic()-started, physically_verified=False)
            pose_audit.append(failure)
            emit('pose_catalog_error', **failure)
            raise
        poses[key] = selected
        pose_audit.append({'remaining_parts': list(key), 'generated_count': len(unique),
                          'offered_count': len(selected), 'max_pose': args.max_pose,
                          'candidate_truncation': len(unique) > len(selected),
                          'policy': description, 'catalog': selected,
                          'delivery_policy': args.pose_catalog_policy,
                          'seconds': time.monotonic()-started, 'physically_verified': False})
        return deepcopy(selected)

    frozen = None
    memory_before = None
    if args.frozen_memory:
        memory_path = checked_path(args.frozen_memory)
        memory_before = hashlib.sha256(memory_path.read_bytes()).hexdigest()
        frozen = FrozenActionMemory(load_snapshot(memory_path))

    def enrich(request):
        request = deepcopy(request)
        request['geometry'] = filter_features(features, request['remaining_parts'])
        try:
            # Fail before retrieval/IPC if mandatory geometry/history is too big.
            bounded_memory_request(request, [])
            refs = [] if frozen is None else frozen.retrieve(request,
                max_records=args.memory_max_records, policy=args.memory_policy)
            request, audit = bounded_memory_request(request, refs,
                require_parent_pairs=args.memory_policy in ('state_diverse_v1', 'state_balanced_v2', 'transfer_v1', 'assembly_balanced_v1'))
        except RequestCoreTooLarge as error:
            audit = error.audit
            audit.update(memory_policy=args.memory_policy, memory_max_records=args.memory_max_records)
            with (output / 'request_payload_audit.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(audit, sort_keys=True, allow_nan=False)+'\n')
            emit('request_payload_error', **audit)
            raise
        audit.update(memory_policy=args.memory_policy, memory_max_records=args.memory_max_records)
        with (output / 'request_payload_audit.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(audit, sort_keys=True, allow_nan=False)+'\n')
        emit('request_payload_budget', **audit)
        return request

    def record_preservation():
        final_files = digest_files(assembly)
        unchanged = initial_files == final_files
        memory_unchanged = not args.frozen_memory or hashlib.sha256(args.frozen_memory.read_bytes()).hexdigest() == memory_before
        write(output / 'preservation.json', {'input_unchanged': unchanged, 'memory_unchanged': memory_unchanged,
                                            'file_count': len(initial_files), 'after': final_files})
        write(output / 'pose_catalogs.json', pose_audit)
        if not unchanged or not memory_unchanged:
            raise RuntimeError('input_or_memory_changed')

    if args.request_preflight_only:
        # This branch precedes IPC construction, planner imports and verification.
        try:
            parts = sorted(load_part_ids(str(assembly)))
            core = dict(schema_version=1, protocol='explicit_action_v1', run_id=args.run_id,
                request_id=args.run_id+':preflight', state_id='preflight:initial',
                remaining_parts=parts, candidate_ids=[p for p in parts if p != args.base_part],
                base_part=args.base_part, max_grippers=args.max_gripper,
                pose_catalog=pose_provider(parts), attempted_action_ids=[], attempted_actions=[],
                task_context=context, evidence_ids=[], episode_feedback=[], memory_references=[], geometry={})
            if args.path_selection_policy == 'path_probe_round_v1':
                core['path_stage_observations'] = []
            initial = enrich(core)
            write(output / 'preflight_initial_request.json', initial)
            stress, audit = capacity_probe(initial, attempted_count=60, episode_count=12,
                                           memory_count=args.memory_max_records)
            write(output / 'preflight_synthetic_capacity_request.json', stress)
            report = dict(status='PASS', actual_cad_features=True, actual_geometric_poses=True,
                actual_physics=False, api_calls=0, physics_calls=0,
                synthetic_attempts_and_feedback=True, capacity=audit,
                memory_policy=args.memory_policy, memory_max_records=args.memory_max_records)
            write(output / 'request_preflight_report.json', report)
            emit('request_preflight_end', **report)
        finally:
            record_preservation()
        return 0

    from plan_sequence.planner.dfs import DFSSequencePlanner
    from plan_sequence.planner.base import SequencePlanner
    from explicit_action_physics import execute_action
    from explicit_action_planner import make_explicit_action_planner
    client = ExplicitActionClient(output / 'ipc', args.response_timeout)
    scene_session = None

    def verify(local_context, choice, action_output):
        merged = dict(local_context, asap_root=str(root), assembly_dir=str(assembly),
                      seed=args.seed, budget=args.budget, task_context=context,
                      remaining_wall_seconds=local_context.get('time_remaining_seconds', args.timeout))
        if args.stability_diagnostics:
            merged['stability_diagnostics'] = True
        if scene_session is not None:
            merged['physics_scene_policy'] = args.physics_scene_policy
            return execute_action(merged, choice, action_output, scene_session=scene_session)
        return execute_action(merged, choice, action_output)

    Planner = make_explicit_action_planner(DFSSequencePlanner, SequencePlanner)
    def decide(request):
        remaining = planner._remaining_time()
        client.timeout_seconds = min(args.response_timeout, remaining) if remaining is not None else args.response_timeout
        return client(request)

    planner = Planner(parts=load_part_ids(str(assembly)), decision_fn=decide,
                      verifier_fn=verify, pose_provider=pose_provider, run_id=args.run_id,
                      base_part=args.base_part, max_grippers=args.max_gripper,
                      task_context=context, geometry=features, event_sink=emit,
                      request_enricher=enrich, fallback_policy=args.fallback_policy,
                      path_selection_policy=args.path_selection_policy,
                      decision_failure_policy=args.decision_failure_policy,
                      pose_transition_policy=args.pose_transition_policy, max_pose=args.max_pose)
    emit('run_start', source_commit=commit, heuristic_or_gnn=False,
         terminal_two_parts='same_explicit_action_selection', settings=settings,
         memory_policy=args.memory_policy, memory_max_records=args.memory_max_records,
         fallback_policy=args.fallback_policy, decision_failure_policy=args.decision_failure_policy)
    try:
        if args.physics_scene_policy != 'fresh_v1':
            from path_scene_reuse import PathSceneSession
            scene_session = PathSceneSession()
        import plan_sequence.physics_planner as physics_module
        with budget_guard.installed(physics_module) if budget_guard else nullcontext():
            tree = planner.plan(budget=args.budget, state_action_budget=args.state_action_budget,
                                timeout=budget_guard.remaining() if budget_guard else args.timeout,
                                output_dir=output / 'planner')
        result = planner.report()
        result['native_stats'] = planner.get_stats(tree)
        result.update(actual_physics=True, api_in_worker=False, source_commit=commit,
                      pose_generation_audit=pose_audit)
        if scene_session is not None:
            result['path_scene_session'] = scene_session.summary()
        write(output / 'worker_result.json', result)
    finally:
        try:
            if scene_session is not None:
                scene_session.close()
                write(output / 'path_scene_session.json', scene_session.summary())
        finally:
            record_preservation()
            if budget_guard:
                write(output / 'evaluation_budget.json', budget_guard.report())
    emit('run_end', status=result.get('status'), actual_physics=True)
    return 1 if result.get('status') == 'ERROR' else 0


if __name__ == '__main__':
    raise SystemExit(main())
