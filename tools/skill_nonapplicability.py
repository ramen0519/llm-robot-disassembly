"""Replay actual healthy training states locally; never manufacture a failure.

This is a negative contract gate, not new physics or repair success. The saved
decision request is used verbatim, including the state after previous removals.
"""
from copy import deepcopy
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read, sha
from worker.task_memory import digest
from orchestrator.lesson_runtime import extension_for
from orchestrator.python_skill_program import build_context, step_program


def post_pass_state(report, attempt_index):
    """Bind an actual pre-action request to the preceding accepted removal.

    Its own later action may fail. That outcome is not used to call this a
    no-trigger state; earlier failures are preserved and scoped by the real VM.
    """
    if type(attempt_index) is not int or not 2 <= attempt_index <= len(report['attempts']):
        raise ValueError('post_pass_predecessor_required')
    prior, current = report['attempts'][attempt_index-2:attempt_index]
    raw = prior['raw']
    if (prior.get('status') != 'PASS' or prior.get('accepted_for_transition') is not True
            or prior.get('deadline_exceeded') is not False or raw.get('backend') != 'ASAP_RedMax'
            or raw.get('path_success') is not True or raw.get('stability_success') is not True):
        raise ValueError('measured_accepted_predecessor_required')
    matches = [d['request'] for d in report['decisions'] if d['request']['request_id'] == current['request_id']]
    if len(matches) != 1:
        raise ValueError('unique_post_pass_request_required')
    req = matches[0]
    expected = set(prior['remaining_parts']) - {prior['choice']['part_id']}
    observed = [f for f in req['episode_feedback'] if f.get('action_id') == prior['action_id']]
    if (set(req['remaining_parts']) != expected or current['remaining_parts'] != req['remaining_parts']
            or req['state_id'] != current['state_id'] or len(observed) != 1
            or observed[0].get('evidence_id') != prior['event_id']
            or observed[0].get('remaining_parts') != prior['remaining_parts']
            or observed[0].get('choice') != prior['choice']
            or observed[0].get('status') != 'PASS' or observed[0].get('accepted_for_transition') is not True
            or observed[0].get('deadline_exceeded') is not False
            or observed[0].get('path_success') is not True or observed[0].get('stability_success') is not True):
        raise ValueError('post_pass_state_transition_mismatch')
    context, _ = build_context(req, extension_for())
    if context.get('failure') is not None or context.get('last') is not None:
        raise ValueError('post_pass_state_has_current_failure')
    return dict(version='observed_post_pass_state_v1', predecessor_attempt_index=attempt_index-1,
        predecessor_event_id=prior['event_id'], predecessor_raw_sha256=digest(raw),
        request_sha256=digest(req), context_sha256=digest(context),
        following_action_outcome_used=False, whole_episode_success_required=False)


def bind_cases(manifest):
    state_roles = manifest.get('multi_task_schema') == 'development_states_v3'
    if manifest.get('multi_task_schema') not in ('development_roles_v2', 'development_states_v3'):
        if manifest.get('nonapplicability_cases'):
            raise ValueError('nonapplicability_requires_explicit_roles_v2')
        return []
    from tools.multitask_skill_evidence import manifest_roles, validate_roles
    from tools.run_python_skill_cycle import python_generation_packet
    from tools.run_skill_discovery import source_case
    split_path = checked(manifest['split']['path'], True)
    if sha(split_path) != manifest['split']['sha256']:
        raise ValueError('split_changed')
    split = read(split_path)
    train = [r['id'] for r in split['training']]
    validate_roles(train, [r['id'] for r in split['evaluation']])
    _, _, healthy = manifest_roles(manifest, train)
    rows = []
    identities = set()
    for ref in manifest['nonapplicability_cases']:
        if ref['task_id'] not in healthy:
            raise ValueError('healthy_case_not_declared_training_task')
        path = checked(ref['manifest_path'], True)
        if sha(path) != ref['manifest_sha256']:
            raise ValueError('healthy_source_manifest_changed')
        child = read(path)
        if (child.get('experience_task') != ref['task_id'] or 'experience_sources' in child
                or child.get('python_failure_scope') or len(child['cases']) != 1):
            raise ValueError('bound_healthy_source_required')
        # Existing binding checks every actual trace, raw artifact and frozen
        # memory identity. A PASS label in a caller-created request is not proof.
        packet = python_generation_packet(child, 'healthy-contract:' + ref['case_id'])
        report_path = checked(child['experience_report'], True)
        if sha(report_path) != ref['report_sha256']:
            raise ValueError('healthy_report_changed')
        spec = child['cases'][0]
        if (checked(spec['report'], True) != report_path or spec['source_task'] != ref['task_id']
                or set(spec) & {'pose_yaw_degrees', 'trigger_override'}):
            raise ValueError('unmodified_actual_healthy_state_required')
        prepared = source_case(spec)
        report = read(report_path)
        attempt = report['attempts'][spec['attempt_index'] - 1]
        raw, req = prepared['raw'], prepared['request']
        if state_roles:
            provenance = post_pass_state(report, spec['attempt_index'])
            if ref.get('state_provenance') != provenance or digest(req) != provenance['request_sha256']:
                raise ValueError('healthy_state_provenance_changed')
        elif (attempt.get('status') != 'PASS' or attempt.get('accepted_for_transition') is not True
                or attempt.get('deadline_exceeded') is not False
                or raw.get('status') != 'PASS' or raw.get('path_success') is not True
                or raw.get('stability_success') is not True
                or any(r.get('status') != 'PASS' or r.get('accepted_for_transition') is not True
                       or r.get('deadline_exceeded') is not False for r in req['episode_feedback'])):
            raise ValueError('measured_healthy_PASS_state_required')
        if digest(req) != ref['request_sha256']:
            raise ValueError('healthy_request_changed')
        context, _ = build_context(req, extension_for())
        if context.get('failure') is not None or context.get('last') is not None:
            raise ValueError('healthy_state_must_have_no_scoped_failure')
        identity = (ref['report_sha256'], req['request_id'])
        if identity in identities:
            raise ValueError('duplicate_healthy_observation')
        identities.add(identity)
        rows.append(dict(case_id=ref['case_id'], task_id=ref['task_id'],
            request=deepcopy(req), request_sha256=digest(req), context_sha256=digest(context),
            source_packet_sha256=digest(packet), report_sha256=ref['report_sha256'],
            source_event_id=provenance['predecessor_event_id'] if state_roles else attempt['event_id'],
            evidence_class=('actual_saved_POST_PASS_state_NOT_new_physics' if state_roles else
                            'actual_saved_PASS_state_NOT_new_physics')))
    return rows


def audit(program, rows):
    """Use the exact restricted interpreter/extension used by deployed skills."""
    if not rows:
        raise ValueError('bound_healthy_rows_required')
    checks = []
    for row in rows:
        req = row['request']
        context, _ = build_context(req, extension_for())
        if (row['request_sha256'] != digest(req) or row['context_sha256'] != digest(context)
                or context.get('failure') is not None):
            raise ValueError('healthy_runtime_binding_changed')
        value = step_program(program, req, extension=extension_for(), diagnose=True)
        accepted = (value.get('response') is None and value.get('not_applicable_reason') == 'no_trigger'
            and value.get('diagnostic_choice') is None and not value.get('diagnostic_runtime_error')
            and not value.get('error_code') and value.get('input_audit', {}).get('accepted') is True)
        checks.append(dict(case_id=row['case_id'], source_event_id=row['source_event_id'],
            request_sha256=digest(req), context_sha256=digest(context),
            accepted=accepted, runtime=value))
    return dict(version='actual_healthy_nonapplicability_v1', accepted=all(c['accepted'] for c in checks),
        program_sha256=digest(program), checks=checks, api_calls=0, physical_calls=0,
        new_physical_success=False, generalization_claim=False)
