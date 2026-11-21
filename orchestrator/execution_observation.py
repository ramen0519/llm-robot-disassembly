"""Planner acceptance is separate from a raw physical result (Python 3.8).

Missing historical values stay unknown. This metadata never authorizes a new
transition, skips physics, or turns a timeout into a physical failure cause.
"""
from copy import deepcopy
from math import isfinite

KEYS = {'version', 'source', 'accepted_for_transition', 'deadline_exceeded',
        'attempt_elapsed_seconds', 'physics_elapsed_seconds',
        'remaining_wall_seconds', 'allowance_source'}


def validate_execution(value, status=None):
    if (type(value) is not dict or set(value) != KEYS
            or value['version'] != 'planner_execution_v1'
            or value['source'] not in ('planner_attempt', 'validation_attempt')
            or value['allowance_source'] not in (None, 'context.time_remaining_seconds',
                                                'context.remaining_wall_seconds')):
        raise ValueError('invalid_execution_observation')
    for key in ('accepted_for_transition', 'deadline_exceeded'):
        if value[key] is not None and type(value[key]) is not bool:
            raise ValueError('invalid_execution_flag')
    for key in ('attempt_elapsed_seconds', 'physics_elapsed_seconds', 'remaining_wall_seconds'):
        n = value[key]
        if n is not None and (type(n) not in (int, float) or not isfinite(n) or n < 0):
            raise ValueError('invalid_execution_seconds')
    if (value['remaining_wall_seconds'] is None) != (value['allowance_source'] is None):
        raise ValueError('execution_allowance_source_missing')
    if value['accepted_for_transition'] is True and (
            status != 'PASS' or value['deadline_exceeded'] is not False):
        raise ValueError('accepted_transition_requires_nonlate_physical_pass')
    return deepcopy(value)


def from_attempt(attempt):
    context = attempt.get('context') or {}
    field = next((k for k in ('time_remaining_seconds', 'remaining_wall_seconds')
                  if context.get(k) is not None), None)
    return validate_execution(dict(version='planner_execution_v1', source='planner_attempt',
        accepted_for_transition=attempt.get('accepted_for_transition'),
        deadline_exceeded=attempt.get('deadline_exceeded'),
        attempt_elapsed_seconds=attempt.get('elapsed_seconds'),
        physics_elapsed_seconds=attempt['raw'].get('elapsed_seconds'),
        remaining_wall_seconds=context.get(field) if field else None,
        allowance_source='context.' + field if field else None), attempt['status'])


def action_result(attempt, evidence_ref):
    raw = attempt['raw']
    return dict(status=attempt['status'], path_success=raw.get('path_success'),
        stability_success=raw.get('stability_success'),
        verifier_origin='actual_ASAP_RedMax' if raw.get('backend') == 'ASAP_RedMax' else 'executor_ERROR',
        raw=deepcopy(raw), evidence_ref=evidence_ref, execution=from_attempt(attempt))


def project_execution(result):
    """Only add new fields when metadata exists; legacy projections stay exact."""
    if 'execution' not in result:
        return {}
    value = validate_execution(result['execution'], result.get('status'))
    return dict(execution=value, accepted_for_transition=value['accepted_for_transition'],
                deadline_exceeded=value['deadline_exceeded'])


def export_episode_memory(report, output, evidence_ref, *, allow_test_root=False):
    """One export path for the actual episode workflow and offline replays."""
    from pathlib import Path
    from orchestrator.explicit_action_memory import append_action_observation, freeze_memory
    output = Path(output)
    decisions = {d['request']['request_id']: d for d in report['decisions']}
    failed, records = {}, []
    for attempt in report['attempts']:
        decision = decisions[attempt['request_id']]
        if (attempt['choice'] != decision['response']['choice']
                or attempt['state_id'] != decision['request']['state_id']):
            raise ValueError('episode_memory_decision_binding_mismatch')
        key = (attempt['state_id'], attempt['choice']['part_id'])
        record = append_action_observation(output/'action_observations.jsonl',
            decision['request'], decision['response'], action_result(attempt, evidence_ref),
            parent_failure_evidence_id=failed.get(key), allow_test_root=allow_test_root)
        records.append(record)
        if attempt['status'] in ('UNKNOWN', 'REJECTED'):
            failed[key] = record['evidence_id']
    freeze_memory(output/'action_observations.jsonl', output/'action_memory_frozen.json',
                  allow_test_root=allow_test_root)
    return records
