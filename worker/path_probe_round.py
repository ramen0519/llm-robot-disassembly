"""Temporary path-stage scheduling over recorded current-episode observations.

No simulator, API, filesystem access, physics cache, or feasibility verdict.
A failed path's still-untried support variants are deferred while a different
registered path has never been probed in this state. The variants become active
again when that initial sweep ends. Candidate parts and the full catalog remain.
"""
from copy import deepcopy

try:
    from .explicit_action_contract import (DIRECTIONS, SKILLS, canonical, digest,
        action_id, enumerate_actions, canonical_response, validate_choice,
        validate_request, validate_response)
except ImportError:
    from explicit_action_contract import (DIRECTIONS, SKILLS, canonical, digest,
        action_id, enumerate_actions, canonical_response, validate_choice,
        validate_request, validate_response)


PATH_SELECTION_POLICIES = ('legacy_v1', 'path_probe_round_v1')
PATH_ROUND_POLICY = 'path_probe_round_v1'


def _semantic_key(choice, pose_matrix):
    return 'probe-path:' + digest({'part_id': choice['part_id'],
        'pose_matrix': pose_matrix, 'skill_id': choice['skill_id'],
        'direction_id': choice['direction_id'],
        'second_direction_id': choice['second_direction_id']})


def semantic_path_key(request, choice):
    """A compact within-request group key, explicitly not an exact cache key.

    The partition carries a separate scope digest. This key alone must never
    establish identity across tasks, states, runtime settings, or RNG states.
    """
    validate_choice(request, choice)
    pose = next(p for p in request['pose_catalog'] if p['pose_id'] == choice['pose_id'])
    return _semantic_key(choice, pose['matrix'])


def partition_path_round(request):
    """Return lossless active/deferred path groups without enumerating supports.

    Full action-space coverage is checked by the provider's rectangle partition.
    A path's catalog count is independent of supports because the empty support
    set is legal for every candidate. Aliases with identical pose matrices count
    as one path. Only observed actual-backend nonlate stage results affect defer.
    """
    request = validate_request(request)
    if 'path_stage_observations' not in request:
        raise ValueError('path_stage_observations_required')
    poses = {p['pose_id']: p['matrix'] for p in request['pose_catalog']}
    attempted, failed, succeeded, ambiguous, prototypes = set(), set(), set(), set(), {}
    eligible, ignored = 0, 0
    for row, observation in zip(request['attempted_actions'], request['path_stage_observations']):
        choice = {key: deepcopy(value) for key, value in row.items() if key != 'action_id'}
        choice['evidence_ids'] = []
        key = _semantic_key(choice, poses[choice['pose_id']])
        attempted.add(key)
        if key not in prototypes:
            prototype = deepcopy(choice)
            prototype['support_ids'] = []
            prototypes[key] = prototype
        if (observation['backend'] != 'ASAP_RedMax'
                or observation['deadline_exceeded']
                or observation['status'] not in ('PASS', 'UNKNOWN', 'REJECTED')
                or type(observation['path_success']) is not bool
                or (observation['status'] == 'PASS' and observation['path_success'] is not True)):
            ignored += 1
            ambiguous.add(key)
            continue
        eligible += 1
        (succeeded if observation['path_success'] else failed).add(key)
    # Registered schedules: six single forces and four perpendicular turns
    # after each first force. Count from the actual registered contract.
    schedule_count = sum(1 if skill == 'single_force_v1' else sum(
        sum(a*b for a, b in zip(DIRECTIONS[first], DIRECTIONS[second])) == 0
        for second in DIRECTIONS) for skill in SKILLS for first in DIRECTIONS)
    all_count = len(request['candidate_ids']) * len({canonical(p) for p in poses.values()}) * schedule_count
    if len(attempted) > all_count:
        raise ValueError('path_round_catalog_scope_mismatch')
    untried_count = all_count - len(attempted)
    deferred = (failed - succeeded - ambiguous) if untried_count > 0 else set()
    scope = {'task_context': request['task_context'],
        'remaining_parts': sorted(request['remaining_parts']), 'base_part': request['base_part'],
        'max_grippers': request['max_grippers'],
        'pose_matrices': sorted({canonical(p) for p in poses.values()}),
        'protocol': request['protocol']}
    audit = dict(policy=PATH_ROUND_POLICY, scope_digest=digest(scope),
        round_index=0 if untried_count > 0 else 1,
        phase='initial_path_sweep' if untried_count > 0 else 'remaining_support_actions',
        all_path_count=all_count, attempted_path_count=len(attempted),
        untried_path_count=untried_count, deferred_path_count=len(deferred),
        active_catalog_path_count=all_count-len(deferred),
        observation_count=len(request['path_stage_observations']),
        eligible_observation_count=eligible, ignored_observation_count=ignored,
        observed_failed_path_count=len(failed), observed_success_path_count=len(succeeded),
        ambiguous_path_count=len(ambiguous),
        deferral_active=bool(deferred),
        release_reason='untried_paths_exhausted' if untried_count == 0 else None,
        source_event_ids_are_citation_permissions=False,
        candidate_pruning=False, permanent_action_pruning=False,
        physics_cache_used=False, physics_verdict_reused=False)
    return dict(audit, round_audit=deepcopy(audit),
        attempted_path_keys=sorted(attempted), observed_failed_path_keys=sorted(failed),
        observed_success_path_keys=sorted(succeeded), ambiguous_path_keys=sorted(ambiguous),
        deferred_path_keys=sorted(deferred),
        deferred_paths=[dict(path_key=key, choice=deepcopy(prototypes[key]))
                        for key in sorted(deferred)])


def validate_path_round_response(request, response):
    """Admission guard only; a deferred action is not a physical rejection."""
    partition = partition_path_round(request)
    response = validate_response(request, response)
    if (response['choice'] is not None and
            semantic_path_key(request, response['choice']) in set(partition['deferred_path_keys'])):
        raise ValueError('action_deferred_for_probe_round')
    return response


def canonical_path_round_response(request, reason, fallback_policy='path_stage_v1'):
    """Labelled deterministic fallback constrained to the current active round."""
    partition = partition_path_round(request)
    deferred = set(partition['deferred_path_keys'])
    response = canonical_response(request, reason, policy=fallback_policy)
    if (response['choice'] is not None and
            semantic_path_key(request, response['choice']) in deferred):
        choice = next((choice for choice in enumerate_actions(request, policy=fallback_policy)
            if semantic_path_key(request, choice) not in deferred), None)
        if choice is None:
            # Every deferred state with novelty has at least one legal empty
            # support action. Reaching this branch is a programming error.
            raise ValueError('path_round_active_space_mismatch')
        response['choice'] = choice
        response['action_id'] = action_id(request, choice)
        response['candidate_order'] = [choice['part_id']] + [part for part in
            sorted(request['candidate_ids']) if part != choice['part_id']]
        response['metadata'].pop('path_signature', None)
        response['metadata'].pop('path_stage_selection', None)
        response['metadata']['round_filtered_fallback'] = True
    response['metadata'].update(path_selection_policy=PATH_ROUND_POLICY,
        path_probe_round=deepcopy(partition['round_audit']),
        semantic_path_key=semantic_path_key(request, response['choice'])
            if response['choice'] is not None else None)
    return validate_path_round_response(request, response)
