"""Scoped execution-guard evidence, kept separate from physical observations."""
from copy import deepcopy

from orchestrator.conditional_plan import action_for_step, task_scope
from worker.explicit_action_contract import digest, validate_request

KIND = "plan_pose_guard_v1"


def make_guard(plan, index, request):
    request = validate_request(request)
    try:
        action_for_step(plan, index, request)
    except ValueError as error:
        if str(error) != "planned_orientation_unavailable":
            raise ValueError("unsupported_or_unbound_plan_guard") from None
    else:
        raise ValueError("pose_guard_not_present")
    step = plan["steps"][index]
    body = dict(schema_version=1, kind=KIND, reason="planned_orientation_unavailable",
        request_id=request["request_id"], state_id=request["state_id"],
        task_scope=task_scope(request), remaining_parts=deepcopy(request["remaining_parts"]),
        catalog_sha256=digest(request["pose_catalog"]), plan_id=plan["plan_id"], step_index=index,
        reference_rotation=deepcopy(plan["source_poses"][step["pose_reference"]]),
        intended=dict(part_id=step["part_id"], skill_id=step["primitive"],
            direction_id=step["direction"], second_direction_id=step["second_direction"],
            support_ids=deepcopy(step["support_ids"])),
        prior_pose_policy=plan["pose_policy_by_step"][index],
        physical_attempt_executed=False, path_success=None, stability_success=None,
        blocker=None, all_paths_impossible=False, physics_reuse_allowed=False, hard_pruning=False)
    body["evidence_id"] = "plan-guard:" + digest(body)
    return body


def validate_guard(guard, plan, index, request):
    if guard != make_guard(plan, index, request):
        raise ValueError("stale_or_tampered_plan_guard")
    return deepcopy(guard)


def guard_packet(guard, request, run_id):
    """No successful target action or implementation is supplied to the model."""
    return dict(run_id=run_id, request_id=run_id + ":generate", state_id=request["state_id"],
        operation="synthesize", evidence=[deepcopy(guard)], existing_skills=[],
        context=dict(failure_scope=KIND, evidence_is_physical_failure=False,
            development_observation={k: deepcopy(request[k]) for k in (
                "candidate_ids", "remaining_parts", "base_part", "max_grippers",
                "pose_catalog", "geometry")},
            goal="Propose one freshly verifiable removal after the observed plan pose guard.",
            successful_target_action_supplied=False, source_part_ids_are_roles_not_constants=True))
