"""Build an isolated recovery test from actual, preserved task observations.

A case is a declared state replay, not a newly discovered full disassembly plan.
Only a fresh physical trigger is forwarded as current-episode feedback.
"""
from copy import deepcopy
import math

from worker.explicit_action_contract import action_id, validate_request


def _seconds(value, label, *, positive=False, wall_limit=45):
    if type(wall_limit) not in (int,float) or not math.isfinite(wall_limit) or not 0 < wall_limit <= 600:
        raise ValueError('invalid_explicit_case_wall_limit')
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or positive and not 0 < value <= wall_limit):
        raise ValueError('invalid_' + label)
    return value


def _same_ids(actual, expected):
    return (type(actual) is list and all(type(x) is str for x in actual)
            and len(actual) == len(set(actual)) and sorted(actual) == sorted(expected))


def matches_diagnostic_phase(raw, expected_phase):
    """Narrow an already bound failure to an observed checker stage.

    Legacy manifests omit this field. Initial disconnection is a zero-step
    contact-graph observation, never evidence of falling or displacement.
    """
    if expected_phase is None:
        return True
    if expected_phase not in ('initial_disconnected', 'dynamic_failure'):
        raise ValueError('unsupported_expected_diagnostic_phase')
    from worker.stability_diagnostics import project_stability_diagnostics
    try:
        source = raw.get('stability_diagnostics') or {}
        diagnostic = project_stability_diagnostics(source)
    except (ValueError, TypeError, KeyError):
        return False
    if (not diagnostic or diagnostic['phase'] != expected_phase
            or diagnostic['check_success_returned'] is not True
            or diagnostic['returned_success'] is not False
            or source.get('instrumentation_errors')):
        return False
    if expected_phase == 'initial_disconnected':
        parts = diagnostic['initial_disconnected_parts']
        movers = (raw.get('executed') or {}).get('stability_moving_parts')
        return (diagnostic['physics_steps'] == 0 and not diagnostic['dynamic_failure_parts']
                and bool(parts) and diagnostic['returned_parts'] == parts
                and diagnostic['returned_parts_semantics'] == 'initial_contact_graph_degree_zero'
                and _same_ids((raw.get('raw_upstream') or {}).get('parts_fall'), parts)
                and type(movers) is list and set(parts).issubset(movers))
    return diagnostic['physics_steps'] > 0 and bool(diagnostic['dynamic_failure_parts'])


def is_reproduced_trigger(raw, trigger_kind, wall_seconds=30, *, expected_diagnostic_phase=None):
    """Check the observed failure stage; this is not a physical feasibility test.

    Scope/provenance must additionally pass ``validate_trigger_scope``. A
    timeout, worker error, or an uncalled checker never supplies a failure
    trigger. Old manifest spellings remain accepted without changing them.
    """
    kinds = {'path_failure': 'path_failed', 'path_failed': 'path_failed',
             'stability_failure': 'stability_failed', 'stability_failed': 'stability_failed'}
    if trigger_kind not in kinds:
        raise ValueError('unsupported_trigger_kind')
    _seconds(wall_seconds, 'case_wall_seconds', positive=True)
    elapsed = _seconds(raw.get('elapsed_seconds'), 'trigger_elapsed_seconds')
    upstream = raw.get('raw_upstream') or {}
    if (raw.get('backend') != 'ASAP_RedMax'
            or raw.get('status') not in ('UNKNOWN', 'REJECTED')
            or elapsed > wall_seconds or raw.get('path_check_calls') != 1
            or raw.get('path_success') is not upstream.get('path_success')
            or raw.get('stability_success') is not upstream.get('stability_success')
            or not matches_diagnostic_phase(raw, expected_diagnostic_phase)):
        return False
    if kinds[trigger_kind] == 'path_failed':
        return (raw.get('path_success') is False and raw.get('stability_success') is None
                and raw.get('stability_check_calls') == 0)
    return (raw.get('path_success') is True and raw.get('stability_success') is False
            and raw.get('stability_check_calls') == 1
            and raw.get('stability_mode') == 'one_exact_selected_support_configuration')


def validate_trigger_scope(request, raw, clean_choice):
    """Bind a worker result to the declared current task, not merely its IDs.

    Artifact hashes and fresh-directory ownership are checked by the runner and
    registry; this check alone does not establish that a file is newly executed.
    """
    pose = next(p['matrix'] for p in request['pose_catalog']
                if p['pose_id'] == clean_choice['pose_id'])
    scope = raw.get('scope') or {}
    executed = raw.get('executed') or {}
    task_context = request['task_context']
    task = task_context.get('task') or {}
    initial = task_context.get('initial_parts')
    remaining = request['remaining_parts']
    rest = [p for p in remaining if p != clean_choice['part_id']]
    fixed = ([] if request['base_part'] is None else [request['base_part']]) + clean_choice['support_ids']
    policy = scope.get('support_policy') or {}
    expected_hashes = task.get('input_sha256')
    input_hashes = raw.get('input_hashes')
    if (raw.get('protocol') != 'explicit_action_physics_v1'
            or raw.get('chosen') != clean_choice or raw.get('backend') != 'ASAP_RedMax'
            or raw.get('input_files_unchanged') is not True
            or raw.get('source_files_unchanged') is not True
            or raw.get('task_context_input_and_run_conditions_checked') is not True
            or scope.get('task_context') != task_context
            or scope.get('selected_action') != clean_choice
            or scope.get('base_part') != request['base_part']
            or scope.get('part_move') != clean_choice['part_id']
            or not _same_ids(scope.get('assembled_parts'), remaining)
            or not _same_ids(scope.get('parts_rest'), rest)
            or scope.get('pose') != pose or raw.get('pose') != pose
            or scope.get('pose_id') != clean_choice['pose_id']
            or executed.get('pose') != pose or executed.get('part_id') != clean_choice['part_id']
            or executed.get('pose_id') != clean_choice['pose_id']
            or executed.get('skill_id') != clean_choice['skill_id']
            or not _same_ids(executed.get('support_ids'), clean_choice['support_ids'])
            or not _same_ids(executed.get('path_fixed_parts'), rest)
            or not _same_ids(executed.get('stability_fixed_parts'), fixed)
            or policy.get('base_part') != request['base_part']
            or policy.get('max_grippers') != request['max_grippers']
            or not _same_ids(policy.get('selected_support_ids'), clean_choice['support_ids'])
            or type(initial) is not list or not set(remaining).issubset(initial)
            or not _same_ids(scope.get('parts_removed'), set(initial) - set(remaining))
            or type(expected_hashes) is not dict or not expected_hashes
            or type(input_hashes) is not dict
            or any(input_hashes.get(k) != v for k, v in expected_hashes.items())
            or not raw.get('source_hashes')
            or scope.get('physics', {}).get('source_and_binary_hashes') != raw['source_hashes']):
        raise ValueError('fresh_actual_trigger_scope_mismatch')
    settings = task.get('solver_settings') or {}
    if (raw.get('seed') != settings.get('seed') or scope.get('seed') != settings.get('seed')
            or scope.get('budget') != settings.get('budget')):
        raise ValueError('fresh_actual_trigger_settings_mismatch')
    upstream = raw.get('raw_upstream') or {}
    if (raw.get('status') not in ('PASS', 'UNKNOWN', 'REJECTED', 'ERROR')
            or raw.get('path_check_calls') != 1
            or type(raw.get('path_success')) not in (bool, type(None))
            or raw.get('path_success') is not upstream.get('path_success')
            or type(raw.get('stability_success')) not in (bool, type(None))
            or raw.get('stability_success') is not None and raw.get('path_success') is not True
            or raw.get('status') == 'PASS' and (raw.get('path_success') is not True
                                                or raw.get('stability_success') is not True)):
        raise ValueError('fresh_actual_trigger_stage_mismatch')
    if raw.get('stability_success') is not upstream.get('stability_success'):
        if not (raw.get('stability_mode') == 'all_remaining_rest_explicitly_fixed_no_checker_call'
                and raw.get('stability_success') is True and upstream.get('stability_success') is None
                and raw.get('stability_check_calls') == 0
                and executed.get('stability_moving_parts') == []):
            raise ValueError('fresh_actual_trigger_stage_mismatch')
    boundary = raw.get('stability_mode') == 'all_remaining_rest_explicitly_fixed_no_checker_call'
    movers = set(rest) - set(fixed)
    if boundary:
        if (raw.get('path_success') is not True or raw.get('stability_success') is not True
                or raw.get('stability_check_calls') != 0
                or upstream.get('stability_success') is not None or movers
                or executed.get('stability_moving_parts') != []):
            raise ValueError('fresh_actual_trigger_stability_boundary_mismatch')
    elif raw.get('stability_success') is not None:
        if (raw.get('stability_check_calls') != 1
                or raw.get('stability_mode') != 'one_exact_selected_support_configuration'
                or not _same_ids(executed.get('stability_moving_parts'), movers)):
            raise ValueError('fresh_actual_trigger_stability_checker_mismatch')
    elif (raw.get('path_success') is not True and raw.get('stability_check_calls') != 0):
        raise ValueError('fresh_actual_trigger_stability_stage_order_mismatch')


def request_after_trigger(template, raw, choice, *, run_id, event_id, wall_seconds=30, wall_limit=45,
                          source_choice=None):
    _seconds(wall_seconds, 'case_wall_seconds', positive=True, wall_limit=wall_limit)
    elapsed = _seconds(raw.get('elapsed_seconds'), 'trigger_elapsed_seconds')
    request = validate_request(template)
    # A recovery validation is a new isolated request, not a worker DFS visit.
    # Validate old restoration provenance first, then omit its old authority.
    request.pop('virtual_search_restore', None)
    # This isolated validation request is not the old worker's child-state visit.
    # Keep the exact candidate transforms, but never transfer its carry authority.
    request.pop('accepted_pose_carry', None)
    request.update(run_id=run_id, request_id=run_id + ':recovery',
                   attempted_action_ids=[], attempted_actions=[],
                   path_stage_observations=[], episode_feedback=[],
                   memory_references=[], evidence_ids=[])
    clean_choice = deepcopy(choice)
    clean_choice['evidence_ids'] = []
    if source_choice is not None:
        # A source-bound historical observation may cite the old episode. Check
        # that exact original action before dropping citation IDs in the new
        # isolated context. Physical fields and immutable raw evidence stay exact.
        original = deepcopy(source_choice)
        if (type(original.get('evidence_ids')) is not list
                or any(type(value) is not str or value not in template['evidence_ids']
                       for value in original['evidence_ids'])):
            raise ValueError('source_trigger_citation_mismatch')
        projected = deepcopy(original)
        projected['evidence_ids'] = []
        if projected != clean_choice:
            raise ValueError('source_trigger_choice_projection_mismatch')
        validate_trigger_scope(validate_request(template), raw, original)
    request = validate_request(request)
    identity = action_id(request, clean_choice)
    late = elapsed > wall_seconds
    pose = next(p['matrix'] for p in request['pose_catalog']
                if p['pose_id'] == choice['pose_id'])
    if source_choice is None:
        validate_trigger_scope(request, raw, clean_choice)
    request['attempted_action_ids'] = [identity]
    request['attempted_actions'] = [dict(
        {k: v for k, v in clean_choice.items() if k != 'evidence_ids'}, action_id=identity)]
    request['path_stage_observations'] = [dict(action_id=identity,
        path_success=raw['path_success'], status=raw['status'], deadline_exceeded=late,
        backend=raw['backend'], source_event_id=event_id)]
    feedback = dict(evidence_id=event_id, status=raw['status'], action_id=identity,
        remaining_parts=request['remaining_parts'], choice=clean_choice,
        decision_source='declared_fixture', selected_pose_matrix=pose,
        path_success=raw['path_success'], stability_success=raw['stability_success'],
        measured_endpoint=deepcopy(raw.get('endpoint_observation')),
        raw_stability_parts=deepcopy(raw.get('raw_upstream', {}).get('parts_fall')),
        accepted_for_transition=False, deadline_exceeded=late)
    if raw.get('stability_diagnostics') is not None:
        from worker.stability_diagnostics import project_stability_diagnostics
        feedback['stability_diagnostics'] = project_stability_diagnostics(raw['stability_diagnostics'])
    request['episode_feedback'] = [feedback]
    request['evidence_ids'] = [event_id]
    return validate_request(request)


def physical_context(template, request, choice, *, event_id, seconds=30, wall_limit=45,
                     preserve_stability_policy=False):
    _seconds(seconds, 'case_wall_seconds', positive=True, wall_limit=wall_limit)
    request = validate_request(request)
    if template.get('task_context') != request['task_context']:
        raise ValueError('case_task_context_mismatch')
    if (not _same_ids(template.get('remaining_parts'), request['remaining_parts'])
            or template.get('base_part') != request['base_part']
            or template.get('max_grippers') != request['max_grippers']):
        raise ValueError('case_declared_state_mismatch')
    initial = request['task_context'].get('initial_parts')
    removed = template.get('removed_parts', template.get('parts_removed', []))
    if (type(initial) is not list or not set(request['remaining_parts']).issubset(initial)
            or not _same_ids(removed, set(initial) - set(request['remaining_parts']))):
        raise ValueError('case_removed_state_mismatch')
    settings = request['task_context'].get('task', {}).get('solver_settings', {})
    if template.get('seed') != settings.get('seed') or template.get('budget') != settings.get('budget'):
        raise ValueError('case_solver_settings_mismatch')
    context = deepcopy(template)
    context.update(remaining_parts=deepcopy(request['remaining_parts']),
                   pose_catalog=deepcopy(request['pose_catalog']),
                   base_part=request['base_part'], max_grippers=request['max_grippers'],
                   event_id=event_id, action_id=action_id(request, choice),
                   remaining_wall_seconds=seconds, stability_timeout=seconds)
    # These redundant legacy display fields must describe the new action too.
    context['parts_rest'] = [p for p in request['remaining_parts'] if p != choice['part_id']]
    context['pose_matrix'] = next(p['matrix'] for p in request['pose_catalog']
                                  if p['pose_id'] == choice['pose_id'])
    context.pop('time_remaining_seconds', None)
    if preserve_stability_policy:
        if 'stability_timeout' in template:
            context['stability_timeout'] = template['stability_timeout']
        else:
            context.pop('stability_timeout', None)
    return context
