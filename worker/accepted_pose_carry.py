"""Carry an accepted part-level removal's exact transform, never a new verdict.

The worker owns provenance. Catalogs may grow, but never recenter, replace an
existing matrix, reset action history or duplicate a state budget. State IDs
remain identities of the initial geometric node, not hashes of mutable catalogs.
"""
from copy import deepcopy
import re

FIELD = 'accepted_pose_carry'
POLICY = 'accepted_pose_carry_v1'
POLICIES = ('legacy_v1', POLICY)
SCOPE = 'accepted_parent_transform_not_robot_motion'
PROOF_KEYS = {'event_id', 'action_id', 'source_request_id', 'source_state_id',
    'run_id', 'parent_parts', 'remaining_parts', 'choice', 'pose_matrix',
    'task_context_sha256', 'input_files_sha256', 'raw_result_sha256', 'verdict'}
VERDICT = dict(status='PASS', accepted_for_transition=True, deadline_exceeded=False,
    path_success=True, stability_success=True, backend='ASAP_RedMax',
    task_context_input_and_run_conditions_checked=True, input_files_unchanged=True,
    source_files_unchanged=True)


def _require(ok):
    if not ok:
        raise ValueError('invalid_accepted_pose_carry')


def _clean(choice):
    value = deepcopy(choice)
    _require(type(value) is dict)
    value['evidence_ids'] = []
    return value


def _source_request(proof, request):
    return dict(task_context=request['task_context'], remaining_parts=proof['parent_parts'],
        candidate_ids=[p for p in proof['parent_parts'] if p != request['base_part']],
        base_part=request['base_part'], max_grippers=request['max_grippers'], evidence_ids=[],
        pose_catalog=[dict(pose_id=proof['choice']['pose_id'], matrix=proof['pose_matrix'])])


def _proof(proof, request):
    from worker.explicit_action_contract import action_id, digest, validate_matrix
    _require(type(proof) is dict and set(proof) == PROOF_KEYS)
    _require(proof['run_id'] == request['run_id'] and proof['remaining_parts'] == request['remaining_parts'])
    _require(type(proof['parent_parts']) is list and all(type(p) is str for p in proof['parent_parts'])
        and len(set(proof['parent_parts'])) == len(proof['parent_parts']))
    _require(type(proof['choice']) is dict and proof['choice'].get('evidence_ids') == [])
    _require(proof['remaining_parts'] == [p for p in proof['parent_parts'] if p != proof['choice'].get('part_id')]
        and len(proof['parent_parts']) == len(proof['remaining_parts']) + 1)
    _require(proof['task_context_sha256'] == digest(request['task_context']))
    hashes = request['task_context'].get('task', {}).get('input_sha256')
    _require(type(hashes) is dict and bool(hashes) and proof['input_files_sha256'] == digest(hashes))
    for name in ('event_id', 'source_request_id', 'source_state_id', 'raw_result_sha256'):
        _require(type(proof[name]) is str and 0 < len(proof[name]) <= 512)
    _require(re.fullmatch(r'[0-9a-f]{64}',proof['raw_result_sha256']) is not None)
    _require(proof['event_id'].startswith(request['run_id'] + ':attempt:')
        and proof['source_request_id'].startswith(request['run_id'] + ':request:'))
    _require(proof['source_state_id'] != request['state_id'])
    _require(proof['verdict'] == VERDICT and all(type(proof['verdict'][k]) is type(v) for k,v in VERDICT.items()))
    validate_matrix(proof['pose_matrix'])
    _require(action_id(_source_request(proof, request), proof['choice']) == proof['action_id'])
    return proof


def snapshot(edge, *, run_id, task_context, base_part, max_grippers, remaining_parts):
    """Validate the complete original observation before making compact context."""
    from worker.explicit_action_contract import digest
    _require(type(edge) is dict and type(edge.get('context')) is dict and type(edge.get('raw')) is dict)
    context, raw = edge['context'], edge['raw']
    _require(context.get('run_id') == run_id and context.get('task_context') == task_context
        and context.get('base_part') == base_part and context.get('max_grippers') == max_grippers
        and context.get('remaining_parts') == edge.get('remaining_parts')
        and context.get('parts_rest') == remaining_parts
        and context.get('state_id') == edge.get('state_id')
        and context.get('request_id') == edge.get('request_id')
        and context.get('action_id') == edge.get('action_id'))
    _require(edge.get('status') == 'PASS' and edge.get('accepted_for_transition') is True
        and edge.get('deadline_exceeded') is False and raw.get('status') == 'PASS'
        and raw.get('feasible') is True)
    _require(all(raw.get(k) == v and type(raw.get(k)) is type(v) for k,v in VERDICT.items()
        if k not in ('accepted_for_transition', 'deadline_exceeded')))
    hashes = task_context.get('task', {}).get('input_sha256')
    _require(type(hashes) is dict and bool(hashes) and raw.get('input_hashes') == hashes)
    choice = _clean(edge.get('choice'))
    _require(_clean(raw.get('chosen')) == choice and raw.get('pose') == context.get('pose_matrix')
        and raw.get('part_move') == choice['part_id'] and raw.get('parts_fix') == choice['support_ids'])
    executed = raw.get('executed')
    _require(type(executed) is dict and executed.get('pose') == context['pose_matrix']
        and executed.get('pose_id') == choice['pose_id'] and executed.get('part_id') == choice['part_id']
        and executed.get('support_ids') == choice['support_ids']
        and executed.get('path_fixed_parts') == remaining_parts)
    scope = raw.get('scope')
    _require(type(scope) is dict and scope.get('task_context') == task_context
        and scope.get('assembled_parts') == edge['remaining_parts']
        and scope.get('parts_rest') == remaining_parts and scope.get('part_move') == choice['part_id']
        and scope.get('pose') == context['pose_matrix'] and scope.get('base_part') == base_part
        and _clean(scope.get('selected_action')) == choice)
    proof = dict(event_id=edge['event_id'], action_id=edge['action_id'],
        source_request_id=edge['request_id'], source_state_id=edge['state_id'], run_id=run_id,
        parent_parts=deepcopy(edge['remaining_parts']), remaining_parts=list(remaining_parts),
        choice=choice, pose_matrix=deepcopy(context['pose_matrix']),
        task_context_sha256=digest(task_context), input_files_sha256=digest(hashes),
        raw_result_sha256=digest(raw), verdict=deepcopy(VERDICT))
    _proof(proof, dict(run_id=run_id, state_id='not_the_parent_state', task_context=task_context,
        remaining_parts=list(remaining_parts), base_part=base_part, max_grippers=max_grippers))
    return proof


def append_to_state(state, proof, limit=128):
    """Append a candidate only; keep prior attempts, state identity and budgets."""
    from worker.explicit_action_contract import MAX_POSE_CATALOG, digest, validate_matrix
    _require(type(limit) is int and 1 <= limit <= MAX_POSE_CATALOG)
    validate_matrix(proof['pose_matrix'])
    catalog = state['pose_catalog']
    matches = [p for p in catalog if p['matrix'] == proof['pose_matrix']]
    _require(len(matches) <= 1)
    sources = state.setdefault('pose_carry_sources', {})
    state.setdefault('pose_catalog_revision', 0)
    if matches:
        identity = matches[0]['pose_id']
    else:
        if len(catalog) >= limit:
            raise ValueError('accepted_pose_catalog_overflow_no_truncation')
        identity = 'carry_' + digest(proof['pose_matrix'])[:24]
        _require(all(p['pose_id'] != identity for p in catalog))
        catalog.append(dict(pose_id=identity, matrix=deepcopy(proof['pose_matrix'])))
        sources[identity] = deepcopy(proof)
        state['pose_catalog_revision'] += 1
    previous = state.get('active_pose_carry')
    if previous is None or previous['source']['event_id'] != proof['event_id']:
        state['pose_carry_entry_attempt_count'] = len(state.get('attempted_action_ids', []))
        state['pose_carry_activation_revision'] = state.get('pose_carry_activation_revision', 0) + 1
        # A fresh forward entry supersedes an earlier backtrack snapshot for
        # this node. Historical attempts remain, including their shared budget.
        state.pop('virtual_restore_snapshot', None)
    state['active_pose_carry'] = dict(pose_id=identity, source=deepcopy(proof))
    return identity


def _binding(request, provenance):
    from worker.explicit_action_contract import digest
    value = {k: request.get(k) for k in ('run_id', 'request_id', 'state_id', 'task_context',
        'remaining_parts', 'candidate_ids', 'base_part', 'max_grippers', 'attempted_action_ids',
        'attempted_actions', 'path_stage_observations')}
    value['pose_catalog'] = [{k:p[k] for k in ('pose_id','matrix')} for p in request['pose_catalog']]
    value['provenance'] = {k:v for k,v in provenance.items() if k != 'request_binding_sha256'}
    return digest(value)


def bind(request, state):
    value = dict(version=POLICY, scope=SCOPE, owner='explicit_action_dfs',
        catalog_revision=state['pose_catalog_revision'],
        catalog_sources=deepcopy(state['pose_carry_sources']), active=deepcopy(state['active_pose_carry']),
        entry_attempt_count=state['pose_carry_entry_attempt_count'],
        activation_revision=state['pose_carry_activation_revision'],
        physics_success_implied=False, requires_fresh_verification=True,
        state_identity='initial_geometric_node_append_only_pose_catalog')
    value['request_binding_sha256'] = _binding(request, value)
    return value


def _event_number(identity, run_id):
    prefix = run_id + ':attempt:'
    _require(type(identity) is str and identity.startswith(prefix))
    suffix = identity[len(prefix):]
    _require(suffix.isdigit() and str(int(suffix)) == suffix and int(suffix) > 0)
    return int(suffix)


def validate(request):
    """Require parent scope and exact matrix; visible observations must agree."""
    if FIELD not in request:
        return None
    value = request[FIELD]
    _require(type(value) is dict and set(value) == {'version','scope','owner','request_binding_sha256',
        'catalog_revision','catalog_sources','active','physics_success_implied',
        'requires_fresh_verification','state_identity','entry_attempt_count','activation_revision'})
    _require(value['version'] == POLICY and value['scope'] == SCOPE and value['owner'] == 'explicit_action_dfs'
        and value['physics_success_implied'] is False and value['requires_fresh_verification'] is True
        and value['state_identity'] == 'initial_geometric_node_append_only_pose_catalog'
        and value['request_binding_sha256'] == _binding(request, value))
    _require(type(value['catalog_sources']) is dict and type(value['catalog_revision']) is int
        and 0 <= value['catalog_revision'] == len(value['catalog_sources']) <= 128)
    _require(type(value['active']) is dict and set(value['active']) == {'pose_id','source'})
    _require(type(value['activation_revision']) is int and value['activation_revision'] > 0
        and type(value['entry_attempt_count']) is int
        and 0 <= value['entry_attempt_count'] <= len(request['attempted_action_ids'])
        and 'path_stage_observations' in request)
    entry_number = _event_number(value['active']['source']['event_id'], request['run_id'])
    numbers = [_event_number(row['source_event_id'],request['run_id']) for row in request['path_stage_observations']]
    _require(numbers == sorted(set(numbers)) and entry_number not in numbers
        and value['entry_attempt_count'] == sum(n < entry_number for n in numbers))
    records = list(value['catalog_sources'].items()) + [(value['active']['pose_id'], value['active']['source'])]
    for identity, source in records:
        _proof(source, request)
        matches = [p for p in request['pose_catalog'] if p['pose_id'] == identity]
        _require(len(matches) == 1 and matches[0]['matrix'] == source['pose_matrix'])
        for row in request['episode_feedback']:
            if row.get('evidence_id') == source['event_id'] or row.get('action_id') == source['action_id']:
                _require(row.get('evidence_id') == source['event_id'] and row.get('action_id') == source['action_id']
                    and row.get('remaining_parts') == source['parent_parts']
                    and _clean(row.get('choice')) == source['choice']
                    and row.get('selected_pose_matrix') == source['pose_matrix']
                    and row.get('status') == 'PASS' and row.get('accepted_for_transition') is True
                    and row.get('deadline_exceeded') is False and row.get('path_success') is True
                    and row.get('stability_success') is True)
    return deepcopy(value)


def entry_reference(request):
    """Fresh accepted entry owns current pose until the first new node attempt.

    Historical attempts do not own a newly entered node's current command.
    A budget remains shared; no history or failure evidence is removed.
    """
    value = validate(request)
    if value is None or len(request['attempted_action_ids']) != value['entry_attempt_count']:
        return None
    source = value['active']['source']
    return dict(scope='orientation_only_after_transition',
        pose_carry_activation_revision=value['activation_revision'],
        remaining_parts=deepcopy(source['parent_parts']), action_id=source['action_id'],
        evidence_id=source['event_id'], choice=deepcopy(source['choice']),
        selected_pose_matrix=deepcopy(source['pose_matrix']))


def resolve_pose(request, reference):
    """Exact pose for a bound current reference; None never authorizes fallback."""
    from worker.virtual_search_restore import restored_reference
    from worker.explicit_action_contract import action_id
    value = validate(request)
    if value is None or type(reference) is not dict:
        return None
    matrix = reference.get('selected_pose_matrix', reference.get('pose_matrix'))
    entering = entry_reference(request)
    restored = None if entering is not None else restored_reference(request)
    bound = restored is not None and reference == restored
    if entering is not None:
        source = value['active']['source']
        bound = (reference.get('evidence_id') == source['event_id']
            and reference.get('action_id') == source['action_id']
            and reference.get('remaining_parts') == source['parent_parts']
            and matrix == source['pose_matrix'] and type(reference.get('choice')) is dict
            and _clean(reference['choice']) == source['choice'])
    elif request['attempted_action_ids']:
        last = request['attempted_action_ids'][-1]
        rows = [r for r in request['episode_feedback'] if r.get('action_id') == last]
        if len(rows) == 1:
            row = rows[0]
            try:
                bound = bound or (reference.get('evidence_id') == row.get('evidence_id')
                    and reference.get('action_id') == last and reference.get('remaining_parts') == request['remaining_parts']
                    and matrix == row.get('selected_pose_matrix') and _clean(reference.get('choice')) == _clean(row.get('choice'))
                    and action_id(request, _clean(row['choice'])) == last)
            except (KeyError, TypeError, ValueError):
                return None
    if not bound:
        return None
    matches = [p['pose_id'] for p in request['pose_catalog'] if p['matrix'] == matrix]
    return matches[0] if len(matches) == 1 else None
