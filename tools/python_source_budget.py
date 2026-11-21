"""Opt-in exact recorded allowance; preserve independent original outputs."""
from pathlib import Path
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read, sha
from tools.python_recorded_clock import recorded_seconds
from worker.skill_case_budget import validate_source_context
from copy import deepcopy


OBSERVATION_CONTRACT = 'source_bound_v1'


def observation_binding(prepared):
    """Bind source outcome, original clock and a separate fresh-case allowance."""
    bound = source_budget(prepared)
    spec = prepared['spec']
    report = read(checked(spec['report'], True))
    attempt = report['attempts'][spec['attempt_index'] - 1]
    directory = checked(prepared['source_path'], True)
    raw_path = directory/'result.json'
    physical = read(directory/'request.json')
    if (attempt['raw'] != prepared['raw'] or read(raw_path) != prepared['raw']
            or attempt['choice'] != physical['choice']
            or attempt['request_id'] != prepared['request']['request_id']
            or attempt['state_id'] != prepared['request']['state_id']
            or attempt.get('accepted_for_transition') is not False
            or attempt['status'] not in ('UNKNOWN', 'REJECTED')
            or attempt['status'] != prepared['raw']['status']):
        raise ValueError('source_observation_identity_or_outcome_mismatch')
    from orchestrator.execution_observation import from_attempt
    return dict(version=OBSERVATION_CONTRACT, original_execution=from_attempt(attempt),
        original_allowance_seconds=bound['seconds'],
        source_report_sha256=prepared['source_report_sha256'],
        source_request_sha256=bound['source_request_sha256'],
        source_result_sha256=sha(raw_path),
        fresh_validation_budget=bound,
        saved_observation_is_new_trial=False)


def observation_request(prepared, run_id, event_id):
    from orchestrator.skill_case import request_after_trigger
    from worker.explicit_action_contract import validate_request
    bound = observation_binding(prepared)
    seconds = bound['original_allowance_seconds']
    request = request_after_trigger(prepared['request'], prepared['raw'], prepared['choice'],
        run_id=run_id, event_id=event_id, wall_seconds=seconds, wall_limit=seconds,
        source_choice=prepared['raw']['chosen'])
    execution = bound['original_execution']
    request['episode_feedback'][-1].update(
        accepted_for_transition=execution['accepted_for_transition'],
        deadline_exceeded=execution['deadline_exceeded'], execution=deepcopy(execution))
    request['path_stage_observations'][-1]['deadline_exceeded'] = execution['deadline_exceeded']
    return validate_request(request), bound


def source_budget(prepared):
    seconds = recorded_seconds(prepared)
    path = checked(Path(prepared['source_path'])/'request.json', True)
    source = read(path)['context']
    if validate_source_context(prepared['context'], source) != seconds:
        raise ValueError('recorded_source_allowance_mismatch')
    return dict(version='recorded_source_budget_v1', seconds=seconds,
                source_request_path=str(path), source_request_sha256=sha(path),
                source_report_sha256=prepared['source_report_sha256'],
                stability_policy_preserved=True, initialization_included=True)


def validate_binding(context, binding):
    path = checked(binding['source_request_path'], True)
    if (binding.get('version') != 'recorded_source_budget_v1'
            or binding.get('stability_policy_preserved') is not True
            or binding.get('initialization_included') is not True
            or sha(path) != binding['source_request_sha256']
            or validate_source_context(context, read(path)['context']) != binding['seconds']):
        raise ValueError('recorded_source_budget_changed')
    return binding['seconds']
