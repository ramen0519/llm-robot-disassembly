"""Opt-in v2: code is the action authority; observed failures stay conditional."""
from orchestrator.python_skill_preflight import audit_program, preflight_rows
from worker.explicit_action_contract import action_scope, path_signature, digest

INSTRUCTIONS = """
Execution revision v2. Generate a NEW executable hypothesis from the measured
development outcomes, runtime geometry and candidate lesson. The host intentionally
omits the old source to avoid cosmetic copying; its observed behavior is supplied.
First distinguish the ORIGINAL trigger in runtime_input_example.last from the
FAILED GENERATED REPAIR in revision_cases.previous_repair. Your function will be
called on the ORIGINAL runtime input, not a fabricated ctx.failed_repair object.
Use the supplied facts to change the actual decision algorithm on that input.
Renaming variables, extracting helpers or merely describing a different choice
does not repair it. An arbitrary first different axis is not physical evidence.
For measured PATH failure, a support-only change reuses the same path test:
this verifier fixes ALL other remaining parts during path testing. Compute a
different mover/pose/primitive/force trajectory from evidence, or abstain. This
does not prove the old path universally impossible. For STABILITY failure after
a passing path, support changes can be meaningful and remain allowed.
Return only program and its ordinary card/identity fields in the schema. Do not
predict a second copy of the action or return behavior_checks. The HOST executes
your code on the declared inputs and computes the action differences. Do not
hard-code debug IDs, fabricate physics, or treat a changed action as a success.
Keep the program and card concise. All acceptance still requires fresh physics.
"""


def execution_view(packet, projected):
    from orchestrator.python_skill_revision import compact_view, measured_failure
    result = compact_view(packet, projected)
    previous = result.pop('previous_program', None)
    context = result['context']
    context.update(python_revision_contract_version=2,
        previous_program_sha256=digest(previous) if previous else None,
        previous_source_supplied=False, action_authority='host_execution_of_generated_program',
        development_only=True, hard_pruning=False,
        path_verifier_contract=dict(all_other_remaining_parts_fixed=True,
            support_only_change_does_not_change_path_test=True))
    for row, case in zip(preflight_rows(packet), context['revision_cases']):
        recovery = row.get('recovery') or {}
        case['revision_requirement'] = (
            'change_path_conditions_or_abstain' if measured_failure(row) and recovery.get('path_success') is False
            else 'change_physical_action_or_abstain' if measured_failure(row)
            else 'preserve_observed_facts_no_unobserved_failure_claim')
        diagnostic = case.get('previous_preflight')
        if diagnostic:
            if diagnostic.get('abstention_diagnostic') is not None and not measured_failure(row):
                case.update(revision_stage='preflight_abstention_without_physical_attempt',
                    revision_requirement='address_executed_abstention_condition_with_justified_legal_alternative_or_keep_scope_limit')
            for key in ('prediction', 'prediction_matches_execution'):
                diagnostic.pop(key, None)
            diagnostic['errors'] = [e for e in diagnostic.get('errors', []) if not e.startswith('predicted_')]
    return result


def audit_execution_revision(program, packet):
    from orchestrator.python_skill_revision import measured_failure
    rows = preflight_rows(packet)
    result = audit_program(program, rows)
    result.update(version='python_execution_revision_preflight_v2',
        action_authority='host_execution', model_predictions_required=False)
    for row, check in zip(rows, result['checks']):
        request, proposed = row['current_observation_request'], check['proposed_action']
        old = row.get('recovery') or {}
        same_path = None
        changed = []
        if measured_failure(row) and proposed is not None:
            before, after = action_scope(request, old['chosen']), action_scope(request, proposed)
            changed = [k for k in before['action'] if before['action'][k] != after['action'].get(k)]
            if before['pose_matrix'] != after['pose_matrix']:
                changed.append('pose_matrix')
            if old.get('path_success') is False:
                same_path = path_signature(request, old['chosen']) == path_signature(request, proposed)
                if same_path:
                    check['errors'].append('unchanged_failed_path_conditions')
        check.update(changed_physical_fields=changed, same_failed_path_conditions=same_path,
                     accepted=not check['errors'])
    result['accepted'] = all(c['accepted'] for c in result['checks'])
    return result
