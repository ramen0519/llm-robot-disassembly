"""API/physics-free development gate, using only predeclared debug inputs."""
from copy import deepcopy

from orchestrator.lesson_runtime import extension_for
from orchestrator.python_skill_program import step_program, build_context
from orchestrator.python_input_contract import input_description, DESCRIPTION
from worker.explicit_action_contract import action_id, digest


def debug_rows(packet):
    rows = packet.get('validation_feedback') or packet.get('context', {}).get('python_debug_requests', [])
    if not rows or len(rows) > 8 or any(r.get('phase') != 'debug' for r in rows):
        raise ValueError('python_preflight_requires_bounded_debug_only')
    if any('current_observation_request' not in r for r in rows):
        raise ValueError('python_preflight_missing_debug_input')
    return rows


def generation_contract(packet):
    examples = []
    for row in preflight_rows(packet):
        context, _ = build_context(row['current_observation_request'], extension_for())
        examples.append(dict(case_id=row['case_id'], phase='debug',
            purpose=row.get('purpose', 'source_debug'), **input_description(context)))
    return dict(description=DESCRIPTION, examples=examples,
                admission_inputs_supplied=False, diagnostics_not_physical_evidence=True)


def preflight_rows(packet):
    """Explicit development applicability probes, separate from admission."""
    rows = debug_rows(packet)
    extra = packet.get('context', {}).get('python_applicability_requests', [])
    if extra:
        if (type(extra) is not list or len(extra) > 4 or any(
                r.get('phase') != 'debug' or r.get('purpose') != 'development_applicability_NOT_evaluation'
                or 'current_observation_request' not in r for r in extra)):
            raise ValueError('declared_development_applicability_only')
        rows = rows + extra
    if len(rows) > 8 or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('bounded_unique_preflight_cases_required')
    return rows


def audit_program(program, rows):
    """Same observed request, not a cross-task cache or candidate pruning rule."""
    checked = debug_rows(dict(validation_feedback=rows))
    checks = []
    for row in checked:
        req = row['current_observation_request']
        result = step_program(program, req, extension=extension_for(), diagnose=True)
        errors = []
        diagnostic = result.get('input_audit')
        if diagnostic is None:
            errors.append('diagnostic_execution_unavailable')
        elif not diagnostic['accepted']:
            errors.append('unknown_input_field')
        response = result.get('response')
        if response is None:
            errors.append('no_valid_debug_repair')
        recovery = row.get('recovery')
        repeated = False
        # An observed failed action on THIS exact saved input is a development
        # regression, not proof the action is impossible. PASS/LATE/ERROR and
        # unmeasured outcomes never activate this gate.
        measured_failure = (row.get('outcome') in ('UNKNOWN', 'REJECTED')
            and type(recovery) is dict and recovery.get('backend') == 'ASAP_RedMax'
            and recovery.get('deadline_exceeded') is not True
            and (recovery.get('path_success') is False or
                 (recovery.get('path_success') is True and recovery.get('stability_success') is False)))
        if response is not None and measured_failure:
            repeated = action_id(req, recovery['chosen']) == response['action_id']
            if repeated:
                errors.append('unchanged_failed_debug_action')
        checks.append(dict(case_id=row['case_id'], phase='debug',
            request_sha256=digest(req), context_sha256=digest(build_context(req, extension_for())[0]),
            validation_record_sha256=row.get('validation_record_sha256'),
            observed_failed_action=deepcopy(recovery.get('chosen')) if measured_failure else None,
            proposed_action=deepcopy(response['choice']) if response else None,
            proposed_action_id=response['action_id'] if response else None,
            repeated_failed_action=repeated, input_audit=diagnostic,
            runtime_reason=result.get('not_applicable_reason'),
            runtime_error=result.get('error_code'), errors=errors, accepted=not errors))
    return dict(version='python_debug_preflight_v1', program_sha256=digest(program),
        accepted=all(c['accepted'] for c in checks), checks=checks,
        api_calls=0, physical_calls=0, new_physical_success=False,
        candidate_pruning=False, coverage='executed_debug_branches_only')
