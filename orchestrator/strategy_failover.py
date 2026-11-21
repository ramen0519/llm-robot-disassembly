"""Explicit model-selected alternatives after bound stability failures.

No feasibility inference or state transition. Failures defer one model-chosen
part only within the current conditional plan; no part is removed or pruned.
"""
from copy import deepcopy

from worker.explicit_action_contract import digest

LEGACY = 'legacy_v1'
POLICY = 'model_ordered_failover_v1'
POLICIES = (LEGACY, POLICY)
MAX_ALTERNATIVES_PER_STEP = 3
MAX_ALTERNATIVES_PER_PLAN = 4


def validate_policy(value):
    if value not in POLICIES:
        raise ValueError('unknown_strategy_failure_policy')
    return value


def state_key(request):
    return (request['run_id'], request['state_id'], tuple(request['remaining_parts']))


def observe(request, ledger):
    """Accumulate unique current-state facts across replans, never ERROR/late.

    The shared candidate guard joins current attempted actions, raw path-stage
    observations, feedback identity, exact pose/choice, backend and deadline.
    Historical memory and unbound explanations cannot increment this ledger.
    """
    from orchestrator.strategy_candidates import _observations
    key = state_key(request)
    current = ledger.setdefault(key, {})
    for item in _observations(request):
        row = item['feedback']
        if row.get('path_success') is not True or row.get('stability_success') is not False:
            continue
        record = dict(evidence_id=row['evidence_id'], action_id=row['action_id'],
            part_id=item['choice']['part_id'], remaining_parts=list(request['remaining_parts']),
            choice=deepcopy(item['choice']), selected_pose_matrix=deepcopy(item['pose_matrix']),
            status=row['status'], accepted_for_transition=False, deadline_exceeded=False,
            path_success=True, stability_success=False)
        identity = record['evidence_id']
        if identity in current and current[identity] != record:
            raise ValueError('strategy_stability_evidence_conflict')
        current[identity] = record
    # Only the latest *bound* current failure can authorize switching now.
    latest_action = request['attempted_action_ids'][-1] if request['attempted_action_ids'] else None
    latest = next((row for row in current.values() if row['action_id'] == latest_action), None)
    # Recheck current evidence; an old ledger row alone cannot authorize an
    # ERROR/late/altered latest report or an observation that has disappeared.
    fresh_ids = {item['feedback']['evidence_id'] for item in _observations(request)
                 if item['feedback'].get('path_success') is True
                 and item['feedback'].get('stability_success') is False}
    if latest is not None and latest['evidence_id'] not in fresh_ids:
        latest = None
    if request.get('accepted_pose_carry') is not None:
        from worker.accepted_pose_carry import validate
        carry = validate(request)
        current_visit_actions = request['attempted_action_ids'][carry['entry_attempt_count']:]
        if latest is not None and latest['action_id'] not in current_visit_actions:
            latest = None
    return current, deepcopy(latest)


def summary(request, ledger):
    records, latest = observe(request, ledger)
    rows = []
    for part in sorted(request['candidate_ids']):
        facts = [row for row in records.values() if row['part_id'] == part]
        rows.append(dict(part_id=part, bound_stability_failures=len(facts),
            evidence_ids=[row['evidence_id'] for row in facts]))
    return dict(policy=POLICY, state_id=request['state_id'], remaining_parts=request['remaining_parts'],
        counts_persist_across_replanning=True, parts=rows,
        latest_bound_failure=latest, failed_part_is_impossible=False,
        candidate_pruning=False, new_state_required_for_success=True,
        max_alternatives_per_step=MAX_ALTERNATIVES_PER_STEP,
        max_alternatives_per_plan=MAX_ALTERNATIVES_PER_PLAN)


def trigger(request, ledger, part_id, threshold):
    records, latest = observe(request, ledger)
    facts = [row for row in records.values() if row['part_id'] == part_id]
    if (latest is None or latest['part_id'] != part_id or len(facts) < threshold):
        return None
    return dict(policy=POLICY, source_part_id=part_id, after_failures=threshold,
        observed_failures=len(facts), source_evidence_ids=[r['evidence_id'] for r in facts],
        source_action_ids=[r['action_id'] for r in facts],
        latest_evidence_id=latest['evidence_id'], state_id=request['state_id'],
        remaining_parts=list(request['remaining_parts']), failed_part_retained=True,
        candidate_pruning=False, physical_success_implied=False,
        requires_fresh_path_and_stability=True)


def branch_identity(plan_id, step_index, alternative_index, trigger_record):
    return 'strategy-branch:' + digest(dict(plan_id=plan_id, step_index=step_index,
        alternative_index=alternative_index, trigger=trigger_record))
