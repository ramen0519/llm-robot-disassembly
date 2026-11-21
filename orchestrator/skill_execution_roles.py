"""State-bound support authorization facts, separate from physical validation.

These assessments never grant execution authority. An explicit-action physics
PASS does not demonstrate that automatic recovery may release existing supports.
"""
from copy import deepcopy

from orchestrator.recovery_support_guard import check_recovery_supports
from orchestrator.skill_program import _scoped_last
from worker.explicit_action_contract import validate_choice, validate_request
from worker.task_memory import digest

VERSION = "skill_execution_roles_v1"


def assess_execution_roles(request, choice, reference=None):
    """Apply the online support guard to one validated current-state proposal.

    Only the latest bound, nonlate actual failure can supply existing supports.
    A caller reference must identify that same observation, not source memory.
    """
    bound = None
    reason = None
    proposal_hash = None
    try:
        current = validate_request(request)
        proposed = validate_choice(current, choice)
        proposal_hash = digest(proposed)
        last = _scoped_last(current)
        if last is not None:
            feedback = last["feedback"]
            failed = (feedback.get("path_success") is False or
                      feedback.get("path_success") is True and
                      feedback.get("stability_success") is False)
            if (failed and feedback.get("status") in ("UNKNOWN", "REJECTED")
                    and feedback.get("accepted_for_transition") is False):
                bound = dict(choice=deepcopy(last["choice"]),
                    remaining_parts=list(current["remaining_parts"]),
                    evidence_id=feedback["evidence_id"])
        if reference is not None:
            clean = deepcopy(reference.get("choice")) if type(reference) is dict else None
            if type(clean) is dict:
                clean["evidence_ids"] = []
            if (bound is None or clean != bound["choice"]
                    or reference.get("remaining_parts") is None
                    or sorted(reference["remaining_parts"]) != bound["remaining_parts"]
                    or reference.get("evidence_id") != bound["evidence_id"]):
                bound = None
                reason = "recovery_support_reference_not_current_failure"
        guard = check_recovery_supports(bound, proposed, current)
    except (ValueError, TypeError, KeyError, AttributeError):
        guard = check_recovery_supports(None, {}, {"remaining_parts": []})
        reason = "invalid_or_missing_current_proposal_context"
    if reason:
        guard.update(allowed=False, reason=reason, requires_new_explicit_plan=True)
    return dict(version=VERSION, assessment_scope="current_state_proposal",
        validation_mode="explicit_action_physics",
        request_state_id=request.get("state_id") if type(request) is dict else None,
        proposal_sha256=proposal_hash,
        automatic_recovery=dict(allowed=guard["allowed"], guard=guard,
            authorized=False, automatic_execution_verified=False),
        explicit_plan=dict(requires_new_plan=True, requires_fresh_physics=True,
            authorized=False, automatic_execution_verified=False,
            explicit_plan_execution_verified=False),
        current_physical_success_known=False,
        physical_pass_is_runtime_authority=False)


def sequence_execution_roles(records):
    """All steps, including earlier failures, must preserve automatic supports."""
    result = deepcopy(records[-1]["execution_roles"])
    result["assessment_scope"] = "bounded_recovery_sequence"
    automatic = result["automatic_recovery"]
    automatic["guard"] = None
    automatic["step_guards"] = [deepcopy(r["execution_roles"]["automatic_recovery"]["guard"])
                                 for r in records]
    automatic["allowed"] = all(r["execution_roles"]["automatic_recovery"]["allowed"]
                                for r in records)
    return result


def summarize_execution_roles(records):
    """Describe only recorded source cases; never infer legacy authorization."""
    evaluated, unrecorded, explicit_pass, automatic_pass, explicit_only = [], [], [], [], []
    for record in records:
        case = record["case"]
        scope = dict(case_id=case["case_id"], phase=case["phase"], source_task=case["source_task"],
            condition_hash=record["condition_hash"], validation_record_sha256=record["record_sha256"],
            fixture=record["fixture_mode"], physical_outcome=record["outcome"])
        roles = record.get("execution_roles")
        if roles is None:
            unrecorded.append(scope)
            continue
        if roles.get("version") != VERSION:
            raise ValueError("unsupported_skill_execution_roles")
        evaluated.append(scope)
        if (record["outcome"] == "PASS" and record["natural_trigger"]
                and record["program_execution_checked"]):
            explicit_pass.append(scope)
            if roles["automatic_recovery"]["allowed"] is True:
                automatic_pass.append(scope)
            else:
                explicit_only.append(scope)
    return dict(version=VERSION, status="RECORDED" if evaluated else "NOT_RECORDED",
        scope="source_validation_conditions_only", evaluated_cases=evaluated,
        unrecorded_cases=unrecorded,
        explicit_plan=dict(physical_pass_scope=explicit_pass, requires_new_plan=True,
            requires_fresh_physics=True, authorized=False, automatic_execution_verified=False,
            explicit_plan_execution_verified=False),
        automatic_recovery=dict(guard_compatible_pass_scope=automatic_pass,
            explicit_plan_only_pass_scope=explicit_only, authorized=False,
            requires_current_state_guard=True, automatic_execution_verified=False),
        physical_pass_is_runtime_authority=False)
