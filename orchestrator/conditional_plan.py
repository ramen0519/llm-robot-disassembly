"""Finite conditional action plans. Predictions never own or mutate DFS state."""
from copy import deepcopy
import math

from worker.explicit_action_contract import (
    CHOICE_KEYS, DIRECTIONS, SKILLS, PROTOCOL, action_id, digest,
    validate_choice, validate_request, validate_response,
)
from worker.path_probe_round import validate_path_round_response

PLAN_KEYS = {"request_id", "state_id", "candidate_order", "steps", "recovery_skill_ids"}
STEP_KEYS = {"expected_remaining", "part_id", "mode", "pose_reference", "primitive",
             "direction", "second_direction", "support_ids", "evidence_ids", "registered_skill_id"}


def task_scope(request):
    # CAD features contain only the CURRENT remaining bodies. Their membership
    # legitimately changes after removal; immutable mesh/solver identity is in
    # task_context. Individual remaining-body features are checked separately.
    return digest({k: request[k] for k in ("run_id", "task_context", "base_part", "max_grippers")})


def verify_geometry(source_parts, request):
    for part, features in request["geometry"].get("parts", {}).items():
        if part not in source_parts or source_parts[part] != features:
            raise ValueError("remaining_geometry_features_changed")


def rotation(pose):
    return [row[:3] for row in pose["matrix"][:3]]


def match_pose(reference, request):
    """Use a unique current catalog pose with the planned world orientation.

    Ground translation is recomputed by ASAP's catalog. This rule is explicit
    in the plan contract; no guessed future pose or nearest orientation is used.
    """
    matches = [p for p in request["pose_catalog"] if all(
        abs(p["matrix"][i][j] - reference[i][j]) <= 1e-6 for i in range(3) for j in range(3))]
    if len(matches) != 1:
        raise ValueError("planned_orientation_unavailable_or_ambiguous")
    return matches[0]["pose_id"]


def _ids(value, permitted, label):
    if (type(value) is not list or any(type(v) is not str for v in value)
            or len(set(value)) != len(value) or not set(value).issubset(permitted)):
        raise ValueError("invalid_plan_" + label)


def schema(request, library_ids, max_steps=64):
    parts = sorted(request["remaining_parts"])
    nullable = lambda values: {"anyOf": [{"type": "string", "enum": list(values)}, {"type": "null"}]}
    ids = lambda values: {"type": "array", "items": {"type": "string", **({"enum": list(values)} if values else {})},
                          "maxItems": len(values)}
    props = {
        "expected_remaining": ids(parts), "part_id": {"type": "string", "enum": request["candidate_ids"]},
        "mode": {"type": "string", "enum": ["action", "registered_skill"]},
        "pose_reference": nullable(p["pose_id"] for p in request["pose_catalog"]),
        "primitive": nullable(SKILLS), "direction": nullable(DIRECTIONS),
        "second_direction": nullable(DIRECTIONS), "support_ids": ids(parts),
        "evidence_ids": ids(request["evidence_ids"]),
        "registered_skill_id": nullable(library_ids) if library_ids else {"type": "null"},
    }
    top = {"request_id": {"type": "string", "enum": [request["request_id"]]},
        "state_id": {"type": "string", "enum": [request["state_id"]]},
        "candidate_order": ids(request["candidate_ids"]),
        "steps": {"type": "array", "minItems": 1, "maxItems": min(max_steps, len(parts) - 1),
            "items": {"type": "object", "additionalProperties": False,
                      "properties": props, "required": sorted(props)}},
        "recovery_skill_ids": ids(library_ids)}
    return {"type": "object", "additionalProperties": False, "properties": top, "required": sorted(top)}


def validate_plan(value, request, library_ids, max_steps=64):
    request = validate_request(request)
    if (type(value) is not dict or set(value) != PLAN_KEYS
            or value["request_id"] != request["request_id"] or value["state_id"] != request["state_id"]):
        raise ValueError("stale_or_invalid_plan_identity")
    _ids(value["candidate_order"], request["candidate_ids"], "candidate_order")
    if set(value["candidate_order"]) != set(request["candidate_ids"]):
        raise ValueError("incomplete_plan_candidate_order")
    _ids(value["recovery_skill_ids"], library_ids, "recovery_skill_ids")
    steps = value["steps"]
    if type(steps) is not list or not 1 <= len(steps) <= min(max_steps, len(request["remaining_parts"]) - 1):
        raise ValueError("invalid_plan_length")
    remaining = set(request["remaining_parts"])
    pose_ids = {p["pose_id"] for p in request["pose_catalog"]}
    for index, step in enumerate(steps):
        if type(step) is not dict or set(step) != STEP_KEYS:
            raise ValueError("invalid_plan_step_fields")
        _ids(step["expected_remaining"], remaining, "remaining")
        if set(step["expected_remaining"]) != remaining:
            raise ValueError("plan_remaining_transition_mismatch")
        if step["part_id"] not in remaining - {request["base_part"]}:
            raise ValueError("invalid_planned_mover")
        _ids(step["support_ids"], remaining - {step["part_id"], request["base_part"]}, "supports")
        _ids(step["evidence_ids"], request["evidence_ids"], "evidence")
        if len(step["support_ids"]) > request["max_grippers"] - 1:
            raise ValueError("plan_support_capacity")
        if step["mode"] == "registered_skill":
            if (step["registered_skill_id"] not in library_ids
                    or any(step[k] is not None for k in ("pose_reference", "primitive", "direction", "second_direction"))
                    or step["support_ids"]):
                raise ValueError("invalid_registered_skill_step")
        elif step["mode"] == "action":
            if step["registered_skill_id"] is not None or step["pose_reference"] not in pose_ids:
                raise ValueError("invalid_plan_pose_or_program")
            # A syntax-only reduced request does not predict future physics.
            fake = deepcopy(request)
            fake.update(remaining_parts=sorted(remaining), candidate_ids=sorted(remaining - {request["base_part"]}),
                        attempted_action_ids=[], attempted_actions=[], path_stage_observations=[])
            choice = dict(part_id=step["part_id"], pose_id=step["pose_reference"],
                skill_id=step["primitive"], direction_id=step["direction"],
                second_direction_id=step["second_direction"], support_ids=step["support_ids"],
                evidence_ids=step["evidence_ids"])
            validate_choice(fake, choice)
            if index == 0:
                response_for(request, choice, value["candidate_order"], {})
        else:
            raise ValueError("unknown_plan_step_mode")
        remaining.remove(step["part_id"])
    bound = deepcopy(value)
    bound.update(plan_id="plan:" + digest(value), task_scope=task_scope(request),
        source_poses={p["pose_id"]: rotation(p) for p in request["pose_catalog"]},
        source_geometry_parts=deepcopy(request["geometry"].get("parts", {})),
        first_request_sha256=digest(request), pose_rule="unique_current_orientation_translation_from_catalog",
        physics_success_implied=False)
    return bound


def response_for(request, choice, order, metadata):
    order = [p for p in order if p in request["candidate_ids"]]
    order = [choice["part_id"]] + [p for p in order if p != choice["part_id"]]
    value = dict(schema_version=1, protocol=PROTOCOL, request_id=request["request_id"],
        state_id=request["state_id"], candidate_order=order, choice=deepcopy(choice),
        action_id=action_id(request, choice), fallback_reason=None,
        metadata=dict(selection_source="conditional_plan", llm_live=False, attempted=False,
            candidate_pruning=False, heuristic_order_used=False, physics_bypass=False,
            physics_verdict_reused=False, **metadata))
    return validate_path_round_response(request, validate_response(request, value))


def action_for_step(plan, index, request):
    if task_scope(request) != plan["task_scope"]:
        raise ValueError("plan_task_scope_changed")
    verify_geometry(plan["source_geometry_parts"], request)
    step = plan["steps"][index]
    if set(request["remaining_parts"]) != set(step["expected_remaining"]):
        raise ValueError("plan_remaining_guard_failed")
    if step["mode"] != "action":
        raise ValueError("registered_program_step_requires_interpreter")
    reference = plan["source_poses"][step["pose_reference"]]
    binding = {}
    if plan.get("plan_contract") in ("conditional_plan_v3", "conditional_plan_v4"):
        from orchestrator.plan_pose_binding import resolve_pose
        pose_id, detail = resolve_pose(reference, request, plan["pose_policy_by_step"][index])
        binding["plan_pose_binding"] = detail
    else:
        pose_id = match_pose(reference, request)
    choice = dict(part_id=step["part_id"],
        pose_id=pose_id,
        skill_id=step["primitive"], direction_id=step["direction"],
        second_direction_id=step["second_direction"], support_ids=list(step["support_ids"]),
        evidence_ids=[e for e in step["evidence_ids"] if e in request["evidence_ids"]])
    return response_for(request, choice, plan["candidate_order"], {
        "plan_id": plan["plan_id"], "plan_step": index, "plan_execution_without_api": True,
        "expired_plan_citation_ids": [e for e in step["evidence_ids"] if e not in request["evidence_ids"]],
        "generated_program_executed": False, "generated_skill_id": None, **binding})
