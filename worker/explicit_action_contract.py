"""Python 3.8 contract for one explicitly selected, bounded physics attempt.

This module enumerates syntax, not feasibility. It never imports an ASAP
heuristic, removes a candidate because of failure, or owns the DFS tree.
"""
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math
import re

PROTOCOL = 'explicit_action_v1'
MAX_POSE_CATALOG = 128
DIRECTIONS = {'xp': [1, 0, 0], 'xn': [-1, 0, 0], 'yp': [0, 1, 0],
              'yn': [0, -1, 0], 'zp': [0, 0, 1], 'zn': [0, 0, -1]}
SKILLS = ('single_force_v1', 'turn_force_two_leg_v1')
FALLBACK_POLICIES = ('legacy_v1', 'path_stage_v1')
CHOICE_KEYS = {'part_id', 'pose_id', 'skill_id', 'direction_id',
               'second_direction_id', 'support_ids', 'evidence_ids'}
REQUEST_KEYS = {'schema_version', 'protocol', 'run_id', 'request_id', 'state_id',
    'remaining_parts', 'candidate_ids', 'base_part', 'max_grippers', 'pose_catalog',
    'attempted_action_ids', 'task_context', 'evidence_ids', 'episode_feedback',
    'memory_references', 'geometry'}
OPTIONAL_REQUEST_KEYS = {'attempted_actions', 'path_stage_observations', 'virtual_search_restore', 'accepted_pose_carry'}
ATTEMPTED_ACTION_KEYS = (CHOICE_KEYS - {'evidence_ids'}) | {'action_id'}
PATH_OBSERVATION_KEYS = {'action_id', 'path_success', 'status', 'deadline_exceeded',
                         'backend', 'source_event_id'}
RESPONSE_KEYS = {'schema_version', 'protocol', 'request_id', 'state_id',
    'candidate_order', 'choice', 'action_id', 'fallback_reason', 'metadata'}
FORBIDDEN_KEYS = {'native_order', 'heuristic_order', 'heuristic_score', 'heuristic_scores',
    'heur_out_order', 'heur_out_score', 'gnn_order', 'gnn_scores', 'gnn_score',
    'recommended_order', 'fallback_order', 'deterministic_fallback_order', 'outsideness'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _ids(value, label, limit=4096):
    if (type(value) is not list or len(value) > limit
            or any(type(item) is not str or not item or len(item) > 512 for item in value)
            or len(value) != len(set(value))):
        raise ValueError('invalid_' + label)


def _hidden_priors(value):
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or key.lower().replace('-', '_') in FORBIDDEN_KEYS:
                raise ValueError('hidden_priority_information')
            _hidden_priors(item)
    elif type(value) is list:
        for item in value:
            _hidden_priors(item)


def validate_matrix(matrix):
    """Accept finite proper rigid transforms; no reflection/scale/shear."""
    if (type(matrix) is not list or len(matrix) != 4
            or any(type(row) is not list or len(row) != 4 for row in matrix)
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   for row in matrix for x in row)):
        raise ValueError('invalid_pose_matrix')
    if any(abs(matrix[3][i] - (1 if i == 3 else 0)) > 1e-6 for i in range(4)):
        raise ValueError('invalid_homogeneous_pose')
    for i in range(3):
        for j in range(3):
            if abs(sum(matrix[k][i] * matrix[k][j] for k in range(3)) - int(i == j)) > 1e-5:
                raise ValueError('pose_rotation_not_orthonormal')
    a = matrix
    det = (a[0][0] * (a[1][1]*a[2][2]-a[1][2]*a[2][1])
           - a[0][1] * (a[1][0]*a[2][2]-a[1][2]*a[2][0])
           + a[0][2] * (a[1][0]*a[2][1]-a[1][1]*a[2][0]))
    if abs(det - 1) > 1e-5:
        raise ValueError('pose_rotation_not_proper')


def validate_request(request):
    if (type(request) is not dict or not REQUEST_KEYS.issubset(request)
            or not set(request).issubset(REQUEST_KEYS | OPTIONAL_REQUEST_KEYS)):
        raise ValueError('invalid_explicit_request_fields')
    if type(request['schema_version']) is not int or request['schema_version'] != 1 or request['protocol'] != PROTOCOL:
        raise ValueError('invalid_explicit_protocol')
    for key in ('run_id', 'request_id', 'state_id'):
        if type(request[key]) is not str or not 1 <= len(request[key]) <= 512:
            raise ValueError('invalid_request_identity')
    _ids(request['remaining_parts'], 'remaining_parts', 256)
    _ids(request['candidate_ids'], 'candidate_ids', 256)
    parts, base = request['remaining_parts'], request['base_part']
    if len(parts) < 2 or (base is not None and (type(base) is not str or base not in parts)):
        raise ValueError('invalid_remaining_or_base')
    if set(request['candidate_ids']) != set(parts) - ({base} if base is not None else set()):
        raise ValueError('incomplete_candidate_set')
    if type(request['max_grippers']) is not int or not 1 <= request['max_grippers'] <= 256:
        raise ValueError('invalid_gripper_budget')
    catalog = request['pose_catalog']
    if type(catalog) is not list or not 1 <= len(catalog) <= MAX_POSE_CATALOG:
        raise ValueError('invalid_pose_catalog')
    pose_ids = []
    for pose in catalog:
        if type(pose) is not dict or not {'pose_id', 'matrix'}.issubset(pose) or not set(pose).issubset({'pose_id', 'matrix', 'geometry'}):
            raise ValueError('invalid_pose_catalog_entry')
        if type(pose['pose_id']) is not str or not pose['pose_id'] or len(pose['pose_id']) > 128:
            raise ValueError('invalid_pose_id')
        validate_matrix(pose['matrix'])
        pose_ids.append(pose['pose_id'])
    if len(set(pose_ids)) != len(pose_ids):
        raise ValueError('duplicate_pose_id')
    _ids(request['attempted_action_ids'], 'attempted_action_ids', 10000)
    if any(re.fullmatch(r'action:[0-9a-f]{64}', x) is None for x in request['attempted_action_ids']):
        raise ValueError('invalid_attempted_action_id')
    _ids(request['evidence_ids'], 'evidence_ids', 128)
    if type(request['task_context']) is not dict or not request['task_context'] or type(request['geometry']) is not dict:
        raise ValueError('invalid_context_or_geometry')
    for key in ('episode_feedback', 'memory_references'):
        if type(request[key]) is not list or len(request[key]) > 12 or any(type(x) is not dict for x in request[key]):
            raise ValueError('unbounded_' + key)
    supplied = set()
    for item in request['memory_references']:
        identity = item.get('evidence_id')
        if type(identity) is not str or not identity or identity in supplied:
            raise ValueError('invalid_memory_reference_identity')
        if (item.get('current_part_id_mapping') is not None or item.get('hard_pruning') is not False
                or item.get('physics_reuse_allowed') is not False
                or type(item.get('reference_only')) is not bool):
            raise ValueError('unsafe_memory_reference')
        supplied.add(identity)
    for item in request['episode_feedback']:
        if item.get('status') not in ('PASS', 'REJECTED', 'UNKNOWN', 'ERROR'):
            raise ValueError('invalid_episode_status')
        identity = item.get('evidence_id')
        if type(identity) is not str or not identity:
            raise ValueError('invalid_episode_identity')
        supplied.add(identity)
    if supplied != set(request['evidence_ids']):
        raise ValueError('evidence_ids_must_match_supplied_records')
    if 'attempted_actions' in request:
        rows = request['attempted_actions']
        if type(rows) is not list or len(rows) > 10000:
            raise ValueError('invalid_attempted_actions')
        identities = []
        for row in rows:
            if type(row) is not dict or set(row) != ATTEMPTED_ACTION_KEYS:
                raise ValueError('invalid_attempted_action_fields')
            # Reconstruct only syntax and exact current-condition identity.
            # An attempted row is not an evidence citation or failure verdict.
            choice = {key: deepcopy(value) for key, value in row.items()
                      if key != 'action_id'}
            choice['evidence_ids'] = []
            expected = action_id(request, choice)
            if row['action_id'] != expected:
                raise ValueError('attempted_action_scope_mismatch')
            identities.append(expected)
        if (len(identities) != len(set(identities))
                or identities != request['attempted_action_ids']):
            raise ValueError('attempted_actions_must_match_ids')
    if 'path_stage_observations' in request:
        rows = request['path_stage_observations']
        if ('attempted_actions' not in request or type(rows) is not list
                or len(rows) > 10000):
            raise ValueError('invalid_path_stage_observations')
        identities, events = [], []
        for row in rows:
            if type(row) is not dict or set(row) != PATH_OBSERVATION_KEYS:
                raise ValueError('invalid_path_stage_observation_fields')
            if (row['path_success'] is not None and type(row['path_success']) is not bool
                    or row['status'] not in ('PASS', 'UNKNOWN', 'REJECTED', 'ERROR')
                    or type(row['deadline_exceeded']) is not bool
                    or (row['backend'] is not None and
                        (type(row['backend']) is not str or not 1 <= len(row['backend']) <= 128))
                    or type(row['source_event_id']) is not str
                    or not 1 <= len(row['source_event_id']) <= 512
                    or type(row['action_id']) is not str):
                raise ValueError('invalid_path_stage_observation_value')
            # These IDs identify recorded executions, not permission to cite
            # every old event in the bounded model evidence list.
            identities.append(row['action_id'])
            events.append(row['source_event_id'])
        if identities != request['attempted_action_ids'] or len(events) != len(set(events)):
            raise ValueError('path_stage_observations_must_match_attempts')
    if 'virtual_search_restore' in request:
        from worker.virtual_search_restore import validate as validate_restore
        validate_restore(request)
    if 'accepted_pose_carry' in request:
        from worker.accepted_pose_carry import validate as validate_pose_carry
        validate_pose_carry(request)
    _hidden_priors(request)
    if len(canonical(request).encode('utf-8')) > 196608:
        raise ValueError('request_too_large')
    result = deepcopy(request)
    result['candidate_ids'] = sorted(result['candidate_ids'])
    result['remaining_parts'] = sorted(result['remaining_parts'])
    return result


def validate_choice(request, choice):
    if type(choice) is not dict or set(choice) != CHOICE_KEYS:
        raise ValueError('invalid_action_choice_fields')
    if any(type(choice[key]) is not str for key in ('part_id', 'pose_id', 'skill_id', 'direction_id')):
        raise ValueError('invalid_choice_id_type')
    if choice['second_direction_id'] is not None and type(choice['second_direction_id']) is not str:
        raise ValueError('invalid_second_direction_type')
    if choice['part_id'] not in request['candidate_ids']:
        raise ValueError('invalid_selected_part')
    if choice['pose_id'] not in [p['pose_id'] for p in request['pose_catalog']]:
        raise ValueError('unknown_pose_id')
    if choice['skill_id'] not in SKILLS or choice['direction_id'] not in DIRECTIONS:
        raise ValueError('unknown_registered_skill_or_direction')
    second = choice['second_direction_id']
    if choice['skill_id'] == 'single_force_v1':
        if second is not None:
            raise ValueError('single_force_has_no_second_direction')
    elif second not in DIRECTIONS or sum(a*b for a, b in zip(DIRECTIONS[choice['direction_id']], DIRECTIONS[second])) != 0:
        raise ValueError('two_leg_requires_noncollinear_axes')
    _ids(choice['support_ids'], 'support_ids', 256)
    supports = set(choice['support_ids'])
    if (not supports.issubset(set(request['remaining_parts']) - {choice['part_id'], request['base_part']})
            or len(supports) > request['max_grippers'] - 1):
        raise ValueError('illegal_support_set_or_budget')
    _ids(choice['evidence_ids'], 'cited_evidence_ids', 128)
    if not set(choice['evidence_ids']).issubset(request['evidence_ids']):
        raise ValueError('unknown_evidence_id')
    return deepcopy(choice)


def action_scope(request, choice):
    """An exact conditional attempt scope; not a physics cache key or verdict."""
    validate_choice(request, choice)
    pose = next(p for p in request['pose_catalog'] if p['pose_id'] == choice['pose_id'])
    action = {k: deepcopy(v) for k, v in choice.items() if k not in ('evidence_ids', 'pose_id')}
    action['support_ids'] = sorted(action['support_ids'])
    return {'task_context': deepcopy(request['task_context']),
            'remaining_parts': sorted(request['remaining_parts']), 'base_part': request['base_part'],
            'max_grippers': request['max_grippers'], 'pose_matrix': deepcopy(pose['matrix']),
            'action': action, 'protocol': PROTOCOL}


def action_id(request, choice):
    return 'action:' + digest(action_scope(request, choice))


def path_signature(request, choice):
    """Group declared path conditions for ordering, never for physics reuse.

    The path executor fixes all other parts. Additional supports affect the
    later stability stage, so they do not distinguish this *ordering* group.
    Actual runtime/RNG/deadline evidence is deliberately not inferred here.
    """
    scope = action_scope(request, choice)
    del scope['action']['support_ids']
    return 'path-stage:' + digest(scope)


def _support_choices(request, prototype):
    supports = sorted(set(request['remaining_parts']) -
                      {prototype['part_id'], request['base_part']})
    for size in range(min(len(supports), request['max_grippers'] - 1) + 1):
        for support in combinations(supports, size):
            choice = deepcopy(prototype)
            choice['support_ids'] = list(support)
            choice['evidence_ids'] = []
            yield choice


def _path_prototypes(request):
    """Canonical part round-robin, without a geometric recommendation."""
    seen = set()
    for pose in sorted(request['pose_catalog'], key=lambda x: x['pose_id']):
        for skill in SKILLS:
            for direction in DIRECTIONS:
                seconds = [None] if skill == 'single_force_v1' else [d for d in DIRECTIONS
                    if sum(a*b for a, b in zip(DIRECTIONS[direction], DIRECTIONS[d])) == 0]
                for second in seconds:
                    for part in sorted(request['candidate_ids']):
                        choice = {'part_id': part, 'pose_id': pose['pose_id'], 'skill_id': skill,
                            'direction_id': direction, 'second_direction_id': second,
                            'support_ids': [], 'evidence_ids': []}
                        signature = path_signature(request, choice)
                        # Pose aliases with the same matrix describe the same
                        # exact physical action identity in the existing contract.
                        if signature not in seen:
                            seen.add(signature)
                            yield signature, choice


def _attempted_path_state(request):
    complete = not request['attempted_action_ids'] or 'attempted_actions' in request
    choices = []
    for row in request.get('attempted_actions', []):
        choice = {key: deepcopy(value) for key, value in row.items() if key != 'action_id'}
        choice['evidence_ids'] = []
        choices.append(choice)
    signatures = [path_signature(request, choice) for choice in choices]
    return complete, choices, signatures


def _support_retry(request, choices, signatures):
    """Give one immediate support alternative after scoped path/stability facts.

    A second consecutive attempt of the same path ends this priority slot.
    Older success records and Memory never create a current physics verdict.
    Missing/contradictory feedback simply supplies no support priority.
    """
    if not choices or (len(signatures) > 1 and signatures[-1] == signatures[-2]):
        return None
    latest = choices[-1]
    identifier = request['attempted_action_ids'][-1]
    matches = [item for item in request['episode_feedback'] if item.get('action_id') == identifier]
    if len(matches) != 1:
        return None
    item = matches[0]
    if (item.get('status') not in ('UNKNOWN', 'REJECTED')
            or item.get('path_success') is not True
            or item.get('stability_success') is not False
            or item.get('accepted_for_transition') is not False
            or item.get('deadline_exceeded') is not False):
        return None
    remaining = item.get('remaining_parts')
    if (type(remaining) is not list or any(type(part) is not str for part in remaining)
            or sorted(remaining) != sorted(request['remaining_parts'])):
        return None
    pose = next(p for p in request['pose_catalog'] if p['pose_id'] == latest['pose_id'])
    try:
        validate_matrix(item.get('selected_pose_matrix'))
    except (ValueError, TypeError):
        return None
    if item['selected_pose_matrix'] != pose['matrix']:
        return None
    reported = deepcopy(item.get('choice'))
    if type(reported) is not dict or set(reported) != CHOICE_KEYS:
        return None
    # Past citations may have left the recent window; they are not physical
    # parameters. Validate every actual action field against the current scope.
    reported['evidence_ids'] = []
    try:
        if action_id(request, reported) != identifier:
            return None
    except (ValueError, TypeError):
        return None
    attempted = set(request['attempted_action_ids'])
    return next((choice for choice in _support_choices(request, latest)
                 if action_id(request, choice) not in attempted), None)


def _path_stage_actions(request, include_attempted=False):
    complete, choices, signatures = _attempted_path_state(request)
    if not complete:
        # Legacy requests without readable history cannot establish novelty.
        for choice in enumerate_actions(request, include_attempted=include_attempted):
            yield choice
        return
    attempted = set() if include_attempted else set(request['attempted_action_ids'])
    tried_paths = set(signatures)
    emitted = set()
    retry = _support_retry(request, choices, signatures)
    if retry is not None:
        identifier = action_id(request, retry)
        if identifier not in attempted:
            emitted.add(identifier)
            yield retry
    # First expose one empty-support representative of each untried path.
    # Stability is still checked by the real executor for every yielded action.
    for signature, choice in _path_prototypes(request):
        identifier = action_id(request, choice)
        if signature not in tried_paths and identifier not in attempted:
            emitted.add(identifier)
            yield choice
    # Failures only affect priority. Every remaining legal support variant is
    # still reachable; no path/group/part is pruned and no verdict is reused.
    for _signature, prototype in _path_prototypes(request):
        for choice in _support_choices(request, prototype):
            identifier = action_id(request, choice)
            if identifier not in attempted and identifier not in emitted:
                yield choice


def enumerate_actions(request, include_attempted=False, policy='legacy_v1'):
    """Lazy deterministic enumeration, no geometric preference or failed-part pruning."""
    if type(policy) is not str or policy not in FALLBACK_POLICIES:
        raise ValueError('unknown_fallback_policy')
    request = validate_request(request)
    if policy == 'path_stage_v1':
        for choice in _path_stage_actions(request, include_attempted=include_attempted):
            yield choice
        return
    attempted = set() if include_attempted else set(request['attempted_action_ids'])
    for part in sorted(request['candidate_ids']):
        supports = sorted(set(request['remaining_parts']) - {part, request['base_part']})
        for pose in sorted(request['pose_catalog'], key=lambda x: x['pose_id']):
            for skill in SKILLS:
                for direction in DIRECTIONS:
                    seconds = [None] if skill == 'single_force_v1' else [d for d in DIRECTIONS
                        if sum(a*b for a, b in zip(DIRECTIONS[direction], DIRECTIONS[d])) == 0]
                    for second in seconds:
                        for size in range(min(len(supports), request['max_grippers'] - 1) + 1):
                            for support in combinations(supports, size):
                                choice = {'part_id': part, 'pose_id': pose['pose_id'], 'skill_id': skill,
                                    'direction_id': direction, 'second_direction_id': second,
                                    'support_ids': list(support), 'evidence_ids': []}
                                if action_id(request, choice) not in attempted:
                                    yield choice


def canonical_response(request, reason, policy='legacy_v1'):
    request = validate_request(request)
    choice = next(enumerate_actions(request, policy=policy), None)
    order = sorted(request['candidate_ids'])
    if choice is not None:
        order.remove(choice['part_id'])
        order.insert(0, choice['part_id'])
    response = {'schema_version': 1, 'protocol': PROTOCOL, 'request_id': request['request_id'],
        'state_id': request['state_id'], 'candidate_order': order, 'choice': choice,
        'action_id': action_id(request, choice) if choice is not None else None,
        'fallback_reason': reason if reason else ('action_space_exhausted' if choice is None else None),
        'metadata': {'selection_source': 'deterministic_fallback', 'llm_live': False,
            'attempted': False, 'exhausted': choice is None, 'candidate_pruning': False,
            'heuristic_order_used': False, 'physics_bypass': False}}
    if policy == 'path_stage_v1':
        complete, choices, signatures = _attempted_path_state(request)
        tried = set(signatures)
        novelty = complete and any(signature not in tried
            for signature, _prototype in _path_prototypes(request))
        retry = _support_retry(request, choices, signatures) if complete else None
        signature = path_signature(request, choice) if choice is not None else None
        selection = ('exhausted' if choice is None else
                     'legacy_missing_readable_attempts' if not complete else
                     'support_retry_after_path_success' if retry is not None and
                     action_id(request, retry) == response['action_id'] else
                     'untried_path' if signature not in tried else 'deferred_support')
        response['metadata'].update(fallback_policy=policy, path_signature=signature,
            path_stage_selection=selection, path_history_complete=complete,
            path_novelty_available=bool(novelty))
    return response


def validate_response(request, response):
    request = validate_request(request)
    if type(response) is not dict or set(response) != RESPONSE_KEYS:
        raise ValueError('invalid_explicit_response_fields')
    if (type(response['schema_version']) is not int or response['schema_version'] != 1
            or response['protocol'] != PROTOCOL or response['request_id'] != request['request_id']
            or response['state_id'] != request['state_id']):
        raise ValueError('stale_or_invalid_response_identity')
    _ids(response['candidate_order'], 'candidate_order', 256)
    if set(response['candidate_order']) != set(request['candidate_ids']):
        raise ValueError('invalid_candidate_permutation')
    if type(response['metadata']) is not dict or (response['fallback_reason'] is not None and type(response['fallback_reason']) is not str):
        raise ValueError('invalid_response_metadata')
    try:
        from .decision_halt import is_halt_response, validate_halt_fields
    except ImportError:
        from decision_halt import is_halt_response, validate_halt_fields
    if is_halt_response(response):
        validate_halt_fields(response)
    elif response['choice'] is None:
        if response['action_id'] is not None or next(enumerate_actions(request), None) is not None:
            raise ValueError('premature_action_exhaustion')
    else:
        validate_choice(request, response['choice'])
        expected = action_id(request, response['choice'])
        if response['candidate_order'][0] != response['choice']['part_id']:
            raise ValueError('selected_part_must_be_first_candidate')
        if response['action_id'] != expected:
            raise ValueError('action_id_mismatch')
        if expected in request['attempted_action_ids']:
            raise ValueError('action_already_attempted')
    canonical(response)
    return deepcopy(response)
