"""Bounded factual failure/later-PASS pairs from preserved actual planner traces.

This is a deterministic observation projection, not a policy, diagnosis, skill
template, or new physical experiment. All part/support labels remain source IDs.
"""
from copy import deepcopy
import math

from orchestrator.skill_case import validate_trigger_scope
from worker.explicit_action_contract import canonical, digest


MAX_PAIRS = 16
MAX_ATTEMPTS = 10000
DELTA_FIELDS = ("skill_id", "direction_id", "second_direction_id", "support_ids", "pose_matrix")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _project_attempt(attempt, index):
    if type(attempt) is not dict:
        raise ValueError("attempt_not_object")
    context, raw, choice = attempt.get("context"), attempt.get("raw"), attempt.get("choice")
    if any(type(value) is not dict for value in (context, raw, choice)):
        raise ValueError("missing_attempt_context")
    event = attempt.get("event_id")
    if type(event) is not str or not event or len(event) > 512:
        raise ValueError("invalid_event_identity")
    allowance = context.get("remaining_wall_seconds", context.get("time_remaining_seconds"))
    if (attempt.get("deadline_exceeded") is not False or not _number(raw.get("elapsed_seconds"))
            or not _number(allowance) or raw["elapsed_seconds"] > allowance):
        raise ValueError("late_or_unverified_deadline")
    if attempt.get("remaining_parts") != context.get("remaining_parts"):
        raise ValueError("attempt_remaining_mismatch")
    request = {key: context.get(key) for key in (
        "remaining_parts", "base_part", "max_grippers", "pose_catalog", "task_context")}
    validate_trigger_scope(request, raw, choice)
    if attempt.get("status") != raw.get("status"):
        raise ValueError("attempt_raw_status_mismatch")
    if (attempt["status"] == "PASS" and attempt.get("accepted_for_transition") is True
            and raw.get("path_success") is True and raw.get("stability_success") is True
            and raw.get("feasible") is True):
        outcome = "accepted_pass"
    elif (attempt["status"] in ("UNKNOWN", "REJECTED")
            and attempt.get("accepted_for_transition") is False
            and (raw.get("path_success") is False or
                 raw.get("path_success") is True and raw.get("stability_success") is False)):
        outcome = "observed_stage_failure"
    else:
        raise ValueError("no_eligible_failure_or_accepted_pass")
    task = context["task_context"]["task"]
    state_scope = dict(task_context=context["task_context"], input_hashes=raw["input_hashes"],
        source_hashes=raw["source_hashes"], remaining_parts=sorted(attempt["remaining_parts"]),
        base_part=context["base_part"], max_grippers=context["max_grippers"],
        seed=task["solver_settings"]["seed"], budget=task["solver_settings"]["budget"],
        part_id=choice["part_id"])
    action = {field: deepcopy(choice[field]) for field in DELTA_FIELDS if field != "pose_matrix"}
    action["support_ids"] = sorted(action["support_ids"])
    action["pose_matrix"] = deepcopy(context["pose_matrix"])
    # The existing boundary validates the actual pose; also bind the projection.
    if action["pose_matrix"] != raw["scope"]["pose"]:
        raise ValueError("projected_pose_mismatch")
    controller = raw.get("controller")
    controller = controller if type(controller) is dict else {}
    counts = {name: controller.get(name) if type(controller.get(name)) is int
              and controller[name] >= 0 else None for name in ("switch_count", "physics_steps")}
    actions = controller.get("actions")
    if type(actions) is not list or len(actions) > 2:
        actions = None
    else:
        for row in actions:
            if (type(row) is not dict or type(row.get("step")) is not int or row["step"] < 0
                    or type(row.get("action")) is not list or len(row["action"]) != 3
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in row["action"])):
                actions = None
                break
    observed_controller = dict(reported_switch_count=counts["switch_count"],
        reported_physics_steps=counts["physics_steps"],
        reported_force_actions=deepcopy(actions),
        skill_fully_executed=raw.get("skill_fully_executed")
            if type(raw.get("skill_fully_executed")) is bool else None)
    return dict(index=index, event_id=event, outcome=outcome, scope_key=digest(state_scope),
                source_task=task["assembly_id"], source_part_id=choice["part_id"],
                action=action, controller=observed_controller,
                execution=dict(rng_before=raw.get("rng_before"), remaining_wall_seconds=allowance,
                    effective_path_timeout_seconds=raw.get("effective_path_timeout_seconds")
                        if _number(raw.get("effective_path_timeout_seconds")) else None),
                path_success=raw["path_success"], stability_success=raw["stability_success"])


def summarize_recovery_pairs(report, *, max_pairs=MAX_PAIRS):
    """Return chronological same-condition failure -> later accepted-PASS pairs.

    Every eligible predecessor is counted, with the first ``max_pairs`` emitted
    in success-index then failure-index order. No best-outcome filtering, advice,
    causal explanation, cross-task correspondence, or controller-effect claim.
    Pose differences are allowed and explicitly emitted as a changed condition.
    """
    if type(max_pairs) is not int or not 1 <= max_pairs <= MAX_PAIRS:
        raise ValueError("invalid_pair_limit")
    if (type(report) is not dict or type(report.get("attempts")) is not list
            or len(report["attempts"]) > MAX_ATTEMPTS):
        raise ValueError("invalid_or_unbounded_planner_trace")
    failures, seen, pairs, excluded = {}, set(), [], []
    total, eligible, excluded_count = 0, 0, 0
    for index, attempt in enumerate(report["attempts"]):
        try:
            item = _project_attempt(attempt, index)
            if item["event_id"] in seen:
                raise ValueError("duplicate_event_identity")
            seen.add(item["event_id"])
        except (ValueError, TypeError, KeyError, StopIteration) as error:
            excluded_count += 1
            if len(excluded) < 32:
                excluded.append(dict(attempt_index=index, reason=str(error)))
            continue
        eligible += 1
        if item["outcome"] == "observed_stage_failure":
            failures.setdefault(item["scope_key"], []).append(item)
            continue
        predecessors = failures.get(item["scope_key"], [])
        total += len(predecessors)
        for before in predecessors[:max(0, max_pairs-len(pairs))]:
            changed = [field for field in DELTA_FIELDS if before["action"][field] != item["action"][field]]
            pairs.append(dict(failure_evidence_id=before["event_id"], success_evidence_id=item["event_id"],
                failure_attempt_index=before["index"], success_attempt_index=item["index"],
                source_task=item["source_task"], source_part_id=item["source_part_id"],
                same_state=True, same_part=True,
                same_pose=before["action"]["pose_matrix"] == item["action"]["pose_matrix"],
                changed_choice_fields=changed,
                before=deepcopy(before["action"]), after=deepcopy(item["action"]),
                before_outcome=dict(path_success=before["path_success"], stability_success=before["stability_success"]),
                after_outcome=dict(path_success=True, stability_success=True, accepted_for_transition=True),
                before_controller=deepcopy(before["controller"]), after_controller=deepcopy(item["controller"]),
                before_execution=deepcopy(before["execution"]), after_execution=deepcopy(item["execution"]),
                same_rng_state=(before["execution"]["rng_before"] == item["execution"]["rng_before"])
                    if before["execution"]["rng_before"] is not None
                    and item["execution"]["rng_before"] is not None else None,
                reference_only=True, current_part_id_mapping=None, physics_reuse_allowed=False,
                failure_cause=None, causal_effect_established=False, generalization_established=False,
                prescription=None))
    result = dict(schema_version=1, kind="observed_same_state_failure_later_pass_pairs",
        pairs=pairs, total_pair_count=total, emitted_pair_count=len(pairs),
        omitted_pair_count=total-len(pairs), truncated=total > len(pairs),
        trace_attempt_count=len(report["attempts"]), eligible_attempt_count=eligible,
        excluded_attempt_count=excluded_count, excluded_attempts=excluded,
        source_part_ids_are_current_bindings=False,
        state_definition="same task/input/solver/remaining/base/gripper/part; pose changes reported explicitly",
        new_physical_calls=0, new_api_calls=0, recommendation_generated=False)
    canonical(result)
    return result
