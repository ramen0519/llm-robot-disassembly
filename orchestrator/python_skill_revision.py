"""Opt-in, API-free revision contract. A changed prediction is NOT physics evidence."""
from copy import deepcopy

from orchestrator.python_skill_preflight import audit_program, generation_contract, preflight_rows
from orchestrator.skill_discovery import _object
from worker.explicit_action_contract import action_id, digest

FLAG = 'python_revision_contract_version'
RELATIONS = ('changed_from_failed_repair', 'same_as_failed_repair', 'no_failed_repair', 'abstain')
ACTION_FIELDS = ('part_id', 'pose_id', 'skill_id', 'direction_id', 'second_direction_id', 'support_ids')
INSTRUCTIONS = """
Revision contract v1: distinguish THREE observations:
(1) runtime_input_example.last is the ORIGINAL trigger. It is not the previous
generated repair. The host will NOT replace this input to make your code work.
(2) revision_cases.previous_repair is what the previous Python actually tried
and what the verifier observed. Fix the branch invoked on the ORIGINAL input,
not a branch that only fires after the failed repair. Changing source text alone
does not change its physical action. A legal action is not a successful action.
(3) top-level evidence contains later observations, including PASS when present.
Several changed fields do not establish causality. Do not copy source part IDs.
Use the compact runtime schema and bound examples, whose nested fields are real.
For EACH declared debug/applicability case, trace your NEW program and return a
behavior_checks entry predicting its physical action (without citations) or null.
State its relationship to the measured failed previous repair. The host executes
the same code in the bounded interpreter and rejects prediction/code mismatches
and unchanged failed repairs BEFORE physics. If no justified alternative exists,
abstain honestly; that is not registration success. This is a development check,
not permission to hard-code example IDs. Admission inputs/outcomes remain hidden.
No invented ctx.failed_recovery field, no native exec, no extra model retry.
"""


def enabled(packet):
    ctx = packet.get('context', {})
    if FLAG not in ctx:
        return False
    if type(ctx[FLAG]) is not int or ctx[FLAG] not in (1, 2, 3, 4) or ctx.get('lesson_stage') != 'program':
        raise ValueError('unsupported_python_revision_contract')
    return True


def physical_choice(choice):
    return None if choice is None else {k: deepcopy(choice[k]) for k in ACTION_FIELDS}


def measured_failure(row):
    r = row.get('recovery')
    return (row.get('outcome') in ('UNKNOWN', 'REJECTED') and type(r) is dict
        and r.get('backend') == 'ASAP_RedMax' and r.get('deadline_exceeded') is not True
        and (r.get('path_success') is False or
             (r.get('path_success') is True and r.get('stability_success') is False)))


def behavior_schema(packet):
    rows = preflight_rows(packet)
    text = {'type': 'string', 'minLength': 1, 'maxLength': 256}
    action = _object({k: deepcopy(text) for k in ACTION_FIELDS[:4]} | {
        'second_direction_id': {'type': ['string', 'null']},
        'support_ids': {'type': 'array', 'items': text, 'maxItems': 256}})
    return dict(type='array', minItems=len(rows), maxItems=len(rows), items=_object({
        'case_id': {'type': 'string', 'enum': [r['case_id'] for r in rows]},
        'predicted_action': {'anyOf': [action, {'type': 'null'}]},
        'relationship': {'type': 'string', 'enum': list(RELATIONS)}}))


def validate_predictions(packet, predictions):
    rows = {r['case_id']: r for r in preflight_rows(packet)}
    if type(predictions) is not list or len(predictions) != len(rows):
        raise ValueError('revision_prediction_cases_incomplete')
    seen = set()
    for p in predictions:
        if type(p) is not dict or set(p) != {'case_id', 'predicted_action', 'relationship'}:
            raise ValueError('invalid_revision_prediction_fields')
        name = p['case_id']
        if type(name) is not str or name not in rows or name in seen or p['relationship'] not in RELATIONS:
            raise ValueError('invalid_revision_prediction_case_or_relationship')
        seen.add(name)
        choice = p['predicted_action']
        if choice is not None:
            if type(choice) is not dict or set(choice) != set(ACTION_FIELDS):
                raise ValueError('invalid_revision_predicted_action')
            action_id(rows[name]['current_observation_request'], dict(choice, evidence_ids=[]))
    return deepcopy(predictions)


def compact_view(packet, projected):
    """Lossless raw files stay local; preserve all evidence and exact nested examples once."""
    rows = preflight_rows(packet)
    contract = generation_contract(packet)
    schemas, cases = {}, []
    diagnostics = {r['case_id']: r for r in packet.get('context', {}).get(
        'development_preflight_feedback', {}).get('checks', [])}
    for row, example in zip(rows, contract['examples']):
        schema = example.pop('input_schema', None)
        # input_description names its schema explicitly; avoid silent omissions.
        if schema is None:
            schema = example.pop('runtime_input_schema')
        key = digest(schema)
        schemas[key] = schema
        example['runtime_input_schema_ref'] = key
        old = row.get('recovery')
        previous = None if old is None else {k: deepcopy(old.get(k)) for k in (
            'backend', 'status', 'chosen', 'path_success', 'stability_success', 'deadline_exceeded',
            'stability_diagnostics', 'endpoint_observation', 'blocker', 'all_paths_impossible')}
        diagnostic = diagnostics.get(row['case_id'])
        cases.append(dict(case_id=row['case_id'], request_sha256=digest(row['current_observation_request']),
            input_reference='runtime_contract.examples with this case_id',
            original_trigger_reference='runtime_input_example.last',
            previous_repair=previous, previous_repair_outcome=row.get('outcome'),
            measured_failed_previous_repair=measured_failure(row),
            validation_record_sha256=row.get('validation_record_sha256'),
            previous_preflight=None if diagnostic is None else {k: deepcopy(diagnostic.get(k)) for k in (
                'proposed_action', 'errors', 'runtime_reason', 'runtime_error', 'repeated_failed_action',
                'input_audit', 'prediction', 'actual_relationship', 'prediction_matches_execution')},
            physical_cause_established=False))
        if 'preflight_abstention_feedback' in packet.get('context', {}):
            from orchestrator.python_abstention import feedback_for_case
            abstention = feedback_for_case(packet, row)
            if abstention is not None:
                if cases[-1]['previous_preflight'] is None:
                    raise ValueError('abstention_requires_existing_preflight')
                cases[-1]['previous_preflight']['abstention_diagnostic'] = abstention
    contract['schemas'] = schemas
    if packet.get('context', {}).get('support_development_experiences'):
        from orchestrator.support_learning import share_runtime_examples
        contract = share_runtime_examples(contract)
    # Redundant raw requests/lesson facts/episode memory are already represented
    # by the bound examples and ALL top-level evidence; never select just PASS.
    result = {k: deepcopy(projected[k]) for k in (
        'run_id', 'request_id', 'state_id', 'operation', 'evidence', 'existing_skills', 'previous_program') if k in projected}
    lesson = projected['context']['lesson']
    result['context'] = {FLAG: 1, 'lesson_stage': 'program',
        'lesson': {k: deepcopy(lesson[k]) for k in ('content_sha256', 'claim', 'claims_are_hypotheses', 'generalization')},
        'runtime_contract': contract, 'revision_cases': cases,
        'source_ids_are_not_target_ids': True, 'admission_inputs_supplied': False,
        'new_physical_evidence': False, 'raw_packet_sha256': digest(packet)}
    for key in ('prior_knowledge', 'prior_generation_rejection', 'automatically_extracted_same_condition_pairs',
                'multi_source_provenance', 'multi_task_learning', 'multi_source_initial_contact_transitions',
                'support_development_experiences', 'support_development_preflight', 'recovery_protocol'):
        if key in projected['context']:
            result['context'][key] = deepcopy(projected['context'][key])
    return result


def audit_revision(program, packet, predictions):
    """Run bounded VM, compare claims, preserve the existing repeat/failure gate."""
    if packet.get('context', {}).get(FLAG) == 4:
        from orchestrator.python_separation_grounding import audit_separation_revision
        return audit_separation_revision(program, packet)
    if packet.get('context', {}).get(FLAG) == 3:
        from orchestrator.python_direction_grounding import audit_direction_revision
        return audit_direction_revision(program, packet)
    if packet.get('context', {}).get(FLAG) == 2:
        from orchestrator.python_execution_revision import audit_execution_revision
        return audit_execution_revision(program, packet)
    rows = preflight_rows(packet)
    result = audit_program(program, rows)
    result['version'] = 'python_revision_preflight_v1'
    result['predictions_sha256'] = digest(predictions)
    try:
        validated = validate_predictions(packet, predictions)
    except (ValueError, TypeError, KeyError):
        result.update(accepted=False, prediction_error='invalid_revision_predictions')
        return result
    by_id = {p['case_id']: p for p in validated}
    for row, check in zip(rows, result['checks']):
        prediction = by_id[row['case_id']]
        chosen = prediction['predicted_action']
        req = row['current_observation_request']
        predicted_id = None if chosen is None else action_id(req, dict(chosen, evidence_ids=[]))
        actual_id = check['proposed_action_id']
        relationship = ('abstain' if actual_id is None else
            'no_failed_repair' if not measured_failure(row) else
            'same_as_failed_repair' if check['repeated_failed_action'] else 'changed_from_failed_repair')
        check.update(prediction=deepcopy(prediction), actual_relationship=relationship,
                     prediction_matches_execution=predicted_id == actual_id)
        if predicted_id != actual_id:
            check['errors'].append('predicted_action_differs_from_execution')
        if prediction['relationship'] != relationship:
            check['errors'].append('predicted_relationship_differs_from_execution')
        check['accepted'] = not check['errors']
    result['accepted'] = all(c['accepted'] for c in result['checks'])
    return result
