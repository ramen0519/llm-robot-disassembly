"""Fact-only repair review for planning; never selects an action or runs physics."""
from copy import deepcopy

from orchestrator.skill_program import _scoped_last
from orchestrator.recovery_support_guard import check_recovery_supports
from worker.explicit_action_contract import validate_request
from worker.stability_diagnostics import project_stability_diagnostics

POLICY = 'repair_evidence_v2'

INSTRUCTIONS = """
repair_review separates a CURRENT executable skill proposal from authorization
for a FUTURE failure. To use a proposal now, put mode=registered_skill in the FIRST
step, with its skill ID and proposed mover. Merely adding an ID to
recovery_skill_ids does NOT execute it now; it permits later checks after a new
failure. Review every supplied skill in skill_review: execute_now, defer, or
unavailable. A current proposal may be chosen or deferred; do not call it
unavailable. With no current proposal, report unavailable, though future recovery
authorization is still allowed. Give one short decision justification, not a
reasoning transcript. A review is a model self-report, not proof of attention.
No skill is preferred or mandatory. Keep all candidates. A scoped source PASS
and a current executable proposal do not predict physical success here.
For stability failures inspect stability_repair: actual support sets, measured
dynamic failures versus initial contact disconnection, and remaining capacity.
Supporting every observed failed body may exceed capacity or release an existing
support. These are capacity facts, not proof that an action is impossible. You
may explicitly choose new supports, a different catalog pose or another mover.
An automatic recovery cannot release supports under preserve_existing_v1; a NEW
explicit plan action or explicitly selected plan skill can choose replacements.
support_change_assessment lists which existing supports would be released and
which measured bodies would remain unheld. When mover or pose changes, the old
measurements are reference-only. Holding those bodies is a hypothesis, not a
guarantee: releasing another support may cause a new failure. When all observed
bodies exceed capacity, a skill may abstain; that does not rule out another
support subset, pose or mover. Do not equate raw returned representatives with
the complete measured dynamic-failure list. Consult each observation separately.
Changing only force or path does not alter the same post-removal stability
configuration. Prior failures stay conditional; do not invent a root cause,
forbid a part, merge incompatible observations, or turn missing diagnostics into
success. Path-only skills do not acquire stability repair capability from being
listed as recovery. Each proposed action still requires fresh path and stability.
"""


def _stability_row(request, last):
    if not last:
        return None
    f, choice = last['feedback'], last['choice']
    if f.get('path_success') is not True or f.get('stability_success') is not False:
        return None
    diagnostic = None
    try:
        d = project_stability_diagnostics(f.get('stability_diagnostics'))
        if (d and d.get('check_success_returned') is True and d.get('returned_success') is False
                and not (f.get('stability_diagnostics') or {}).get('instrumentation_errors')):
            diagnostic = d
    except (ValueError, TypeError, KeyError):
        pass
    dynamic, disconnected = [], []
    if diagnostic and diagnostic['phase'] == 'dynamic_failure':
        items = diagnostic['dynamic_failure_parts']
        # Reject inconsistent attribution as a whole, not a fabricated partial diagnosis.
        if last['unstable'] and {x['part_id'] for x in items} == last['unstable']:
            dynamic = deepcopy(items)
        else:
            diagnostic = None
    if diagnostic and diagnostic['phase'] == 'initial_disconnected':
        disconnected = sorted(last['initial_disconnected'])
    observed = {x['part_id'] for x in dynamic} | set(disconnected)
    supports, capacity = set(choice['support_ids']), request['max_grippers'] - 1
    return dict(evidence_id=f['evidence_id'], choice=deepcopy(choice),
        pose_matrix=deepcopy(last['pose_matrix']), remaining_parts=list(request['remaining_parts']),
        path_success=True, stability_success=False,
        diagnostic_phase=diagnostic['phase'] if diagnostic else None,
        dynamic_failure_parts=dynamic, initial_disconnected_parts=disconnected,
        diagnostic_available=diagnostic is not None,
        unknown_cause=True, blocker=None, support_capacity=capacity,
        unused_support_slots=max(0, capacity-len(supports)),
        observed_unheld_parts=sorted(observed),
        all_observed_fit_as_supports=len(observed) <= capacity if observed else None,
        preserve_existing_union=sorted(supports | observed),
        preserve_existing_union_exceeds_capacity=len(supports | observed) > capacity if observed else None,
        support_target_recommendation=None, failure_does_not_prune=True)


def stability_repair(request):
    request = validate_request(request)
    rows = []
    for index, observed in enumerate(request['episode_feedback']):
        prefix = deepcopy(request)
        prefix['episode_feedback'] = prefix['episode_feedback'][:index+1]
        identity = observed.get('action_id')
        matches = [a for a in request['attempted_actions'] if a['action_id'] == identity]
        stages = [a for a in request['path_stage_observations'] if a['action_id'] == identity]
        if len(matches) != 1 or len(stages) != 1:
            continue
        # Project the original matching provenance; never synthesize a PASS or a new action.
        prefix['attempted_actions'], prefix['path_stage_observations'] = matches, stages
        row = _stability_row(request, _scoped_last(prefix))
        if row:
            rows.append(row)
    current = _stability_row(request, _scoped_last(request))
    return dict(current=current, same_state_observations=rows[-8:],
        configuration_fields=['remaining_minus_mover', 'pose_matrix', 'base_part', 'support_set'],
        force_alone_changes_stability_configuration=False,
        observations_are_separate_not_union_constraint=True,
        no_physics_reuse=True, hard_pruning=False)


def repair_review(request, cards, support_policy):
    request = validate_request(request)
    last = _scoped_last(request)
    current_failure = None
    reference = None
    if last:
        f = last['feedback']
        kind = ('path_failed' if f['path_success'] is False else
                'stability_failed' if f['path_success'] is True and f['stability_success'] is False else None)
        if kind:
            current_failure = dict(kind=kind, evidence_id=f['evidence_id'], choice=deepcopy(last['choice']))
            reference = dict(choice=last['choice'], remaining_parts=request['remaining_parts'], evidence_id=f['evidence_id'])
    rows = []
    for card in cards:
        choice = card.get('currently_proposed_choice')
        guard = (check_recovery_supports(reference, choice, request)
                 if choice and support_policy == 'preserve_existing_v1' else None)
        rows.append(dict(skill_id=card['skill_id'], current_proposal_available=choice is not None,
            currently_proposed_choice=deepcopy(choice), abstention_reason=card.get('applicability'),
            authorized_now_by='first_registered_skill_step',
            future_authorization_by='recovery_skill_ids_after_a_new_failure',
            automatic_recovery_support_guard=guard,
            support_change_assessment=support_change_assessment(request, choice),
            physical_success_known=False))
    return dict(current_failure=current_failure, skills=rows,
        selection_forced=False, ordering_is_not_preference=True,
        review_is_model_self_report=True, future_applicability_unknown=True)


def compact_escalation(escalation):
    """Avoid recirculating full past prompts/lessons through nested event logs."""
    result = deepcopy(escalation)
    keys = ('kind', 'sequence', 'reason', 'role', 'status', 'accepted', 'evidence_id',
            'action_id', 'plan_id', 'skill_id', 'not_applicable_reason', 'triggered', 'runtime_feedback',
            'allowed', 'released_support_ids', 'added_support_ids', 'union_support_ids',
            'union_exceeds_capacity', 'support_capacity', 'requires_new_explicit_plan')
    result['prior_events'] = [{k: deepcopy(e[k]) for k in keys if k in e}
                              for e in escalation.get('prior_events', [])]
    result['prior_event_projection'] = 'decision_and_guard_facts_only_full_journal_retained'
    return result


def support_change_assessment(request, choice):
    """Explain proposed support changes; never choose, authorize or verify them."""
    current = stability_repair(request)['current']
    if current is None or choice is None:
        return None
    pose = next(p['matrix'] for p in request['pose_catalog'] if p['pose_id'] == choice['pose_id'])
    old = current['choice']
    previous, proposed = set(old['support_ids']), set(choice['support_ids'])
    observed = set(current['observed_unheld_parts'])
    same_mover_pose = old['part_id'] == choice['part_id'] and current['pose_matrix'] == pose
    return dict(source_evidence_id=current['evidence_id'],
        same_mover_and_pose=same_mover_pose, prior_observations_reference_only=not same_mover_pose,
        released_support_ids=sorted(previous-proposed), added_support_ids=sorted(proposed-previous),
        observed_body_ids=sorted(observed),
        observed_bodies_held_by_proposal=sorted(observed & proposed),
        observed_bodies_remaining_unheld=sorted(observed-proposed-{choice['part_id'], request['base_part']}),
        support_capacity=current['support_capacity'], proposed_support_count=len(proposed),
        support_capacity_respected=len(proposed) <= current['support_capacity'],
        newly_released_support_stability='UNKNOWN',
        fresh_path_and_stability_required=True, physical_success_known=False,
        selection_forced=False, hard_pruning=False)


def stability_choice_comparison(request, choice):
    """Diagnostic only: changing force alone leaves this checker's configuration."""
    facts = stability_repair(request)
    current = facts['current']
    if current is None or choice is None:
        return None
    pose = next(p['matrix'] for p in request['pose_catalog'] if p['pose_id'] == choice['pose_id'])
    old = current['choice']
    changed = [key for key, equal in (
        ('mover', old['part_id'] == choice['part_id']),
        ('pose', current['pose_matrix'] == pose),
        ('supports', set(old['support_ids']) == set(choice['support_ids']))) if not equal]
    return dict(source_evidence_id=current['evidence_id'], changed_configuration_fields=changed,
        same_failed_stability_configuration=not changed,
        support_change_assessment=support_change_assessment(request, choice),
        next_physics_result=None, action_rewritten=False, hard_pruning=False)
