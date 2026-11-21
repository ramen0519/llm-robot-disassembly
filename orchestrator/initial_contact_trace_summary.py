"""Factual initial-contact failure/later different-mover PASS observations.

No simulator, API, file access, ranking recommendation, or new CAD calculation
is performed. Source IDs remain labels in recorded development experience.
"""
from collections import Counter
from copy import deepcopy

from orchestrator.skill_trace_summary import _project_attempt
from worker.explicit_action_contract import canonical, digest, validate_request, validate_response
from worker.stability_diagnostics import project_stability_diagnostics


ACTION_FIELDS = ("part_id", "skill_id", "direction_id", "second_direction_id", "support_ids")
GEOMETRY_FIELDS = ("bounds", "bounds_center", "center_mass", "extents", "volume", "watertight", "validity")
MAX_GEOMETRY_CONTEXTS = 16
MAX_GEOMETRY_PARTS = 64


def _bound_observation(attempt, index, decisions):
    projected = _project_attempt(attempt, index)
    decision = decisions.get(attempt.get("request_id"))
    if decision is None:
        raise ValueError("missing_recorded_decision")
    request = validate_request(decision["request"])
    response = validate_response(request, decision["response"])
    context, raw = attempt["context"], attempt["raw"]
    if (response["choice"] != attempt["choice"] or response["action_id"] != attempt["action_id"]
            or request["state_id"] != attempt["state_id"]
            or any(context.get(k) != request[k] for k in (
                "run_id", "request_id", "state_id", "task_context", "remaining_parts", "base_part", "max_grippers"))
            or context.get("action_id") != response["action_id"]):
        raise ValueError("recorded_decision_attempt_binding_mismatch")
    geometry = request["geometry"]
    task_context = request["task_context"]
    recorded_parts = geometry.get("parts")
    source_parts = task_context.get("geometry_features")
    if (geometry.get("task") != task_context["task"]
            or type(recorded_parts) is not dict or type(source_parts) is not dict
            or set(recorded_parts) != set(request["remaining_parts"])
            or any(source_parts.get(part) != feature for part, feature in recorded_parts.items())):
        raise ValueError("recorded_cad_feature_binding_mismatch")
    pose = next(p["matrix"] for p in request["pose_catalog"] if p["pose_id"] == attempt["choice"]["pose_id"])
    if context["pose_matrix"] != pose or raw["scope"]["pose"] != pose:
        raise ValueError("recorded_pose_binding_mismatch")
    diagnostic = project_stability_diagnostics(raw.get("stability_diagnostics"))
    initial = diagnostic is not None and diagnostic["phase"] == "initial_disconnected"
    if initial and (diagnostic["physics_steps"] != 0
            or diagnostic["check_success_returned"] is not True
            or raw["stability_diagnostics"].get("instrumentation_errors") != []
            or diagnostic["returned_parts"] != raw["raw_upstream"].get("parts_fall")
            or not set(diagnostic["initial_disconnected_parts"]).issubset(raw["executed"]["stability_moving_parts"])):
        raise ValueError("unbound_initial_contact_diagnostic")
    scope = {"task_context": task_context, "input_hashes": raw["input_hashes"],
             "source_hashes": raw["source_hashes"], "remaining_parts": request["remaining_parts"],
             "base_part": request["base_part"], "max_grippers": request["max_grippers"], "pose_matrix": pose}
    parts = geometry["parts"]
    if len(parts) > MAX_GEOMETRY_PARTS:
        cad = {"status": "OMITTED_PART_LIMIT", "part_count": len(parts), "parts": None}
    else:
        cad = {"status": "RECORDED_FEATURES_ONLY", "part_count": len(parts),
               "parts": {part: {k: deepcopy(value[k]) for k in GEOMETRY_FIELDS if k in value}
                         for part, value in sorted(parts.items())}}
    cad.update(source_request_id=request["request_id"],
        source_fields=["decisions[].request.geometry.parts", "decisions[].request.task_context.geometry_features"],
        bound_to_attempt_context=True, source_geometry_sha256=digest(geometry),
        input_sha256=deepcopy(task_context["task"]["input_sha256"]),
        frame=geometry.get("relations", {}).get("frame"), units=geometry.get("relations", {}).get("units"),
        newly_calculated_geometry=False, physical_material_density_inferred=False)
    geometry_key = digest({k: v for k, v in cad.items() if k != "source_request_id"})
    return {"attempt_index": index, "evidence_id": attempt["event_id"],
        "request_id": request["request_id"], "state_id": request["state_id"],
        "source_task": task_context["task"]["assembly_id"], "scope_key": digest(scope),
        "choice": {k: deepcopy(attempt["choice"][k]) for k in ACTION_FIELDS},
        "pose_id": attempt["choice"]["pose_id"], "pose_matrix": deepcopy(pose),
        "remaining_parts": deepcopy(request["remaining_parts"]), "base_part": request["base_part"],
        "max_grippers": request["max_grippers"], "geometry_context_id": geometry_key,
        "outcome": projected["outcome"], "initial_disconnected_failure": bool(initial and projected["outcome"] == "observed_stage_failure"),
        "diagnostic_phase": diagnostic["phase"] if diagnostic else None,
        "initial_disconnected_parts": diagnostic["initial_disconnected_parts"] if initial else [],
        "path_success": raw["path_success"], "stability_success": raw["stability_success"],
        "rng_before": raw.get("rng_before"), "remaining_wall_seconds": projected["execution"]["remaining_wall_seconds"],
        "effective_path_timeout_seconds": projected["execution"]["effective_path_timeout_seconds"]}, cad


def summarize_initial_contact_transitions(report, *, max_pairs=16, max_attempts=10000):
    """First later PASS at identical recorded state/pose, with a different mover.

    A changed mover, direction or support is a confound, not a causal repair
    result. Every source attempt and every unmatched initial failure is counted.
    """
    if (type(max_pairs) is not int or not 1 <= max_pairs <= 32
            or type(max_attempts) is not int or not 1 <= max_attempts <= 10000):
        raise ValueError("invalid_initial_contact_summary_limit")
    if (type(report) is not dict or type(report.get("attempts")) is not list
            or len(report["attempts"]) > max_attempts or type(report.get("decisions")) is not list):
        raise ValueError("invalid_initial_contact_source_trace")
    decisions = {}
    for decision in report["decisions"]:
        identifier = decision["request"]["request_id"]
        if identifier in decisions:
            raise ValueError("duplicate_recorded_request")
        decisions[identifier] = decision
    observations, excluded, geometry_contexts, seen = [], [], {}, set()
    phase_counts = Counter((a.get("raw", {}).get("stability_diagnostics") or {}).get("phase", "NOT_OBSERVED")
                           for a in report["attempts"])
    for index, attempt in enumerate(report["attempts"], 1):
        try:
            item, cad = _bound_observation(attempt, index, decisions)
            if item["evidence_id"] in seen:
                raise ValueError("duplicate_recorded_event")
            seen.add(item["evidence_id"])
        except (ValueError, TypeError, KeyError, StopIteration) as error:
            excluded.append({"attempt_index": index, "reason": str(error)[:160]})
            continue
        if item["geometry_context_id"] not in geometry_contexts and len(geometry_contexts) < MAX_GEOMETRY_CONTEXTS:
            geometry_contexts[item["geometry_context_id"]] = cad
        item["geometry_emitted"] = item["geometry_context_id"] in geometry_contexts
        observations.append(item)
    pairs, unmatched, total = [], [], 0
    for before in observations:
        if not before["initial_disconnected_failure"]:
            continue
        after = next((a for a in observations if a["attempt_index"] > before["attempt_index"]
            and a["scope_key"] == before["scope_key"] and a["outcome"] == "accepted_pass"), None)
        if after is None:
            unmatched.append({"attempt_index": before["attempt_index"], "evidence_id": before["evidence_id"],
                              "reason": "no_later_pass_at_identical_recorded_state_pose"})
            continue
        if after["choice"]["part_id"] == before["choice"]["part_id"]:
            unmatched.append({"attempt_index": before["attempt_index"], "evidence_id": before["evidence_id"],
                              "reason": "first_later_pass_uses_same_mover_not_this_projection"})
            continue
        total += 1
        if len(pairs) >= max_pairs:
            continue
        changed = [key for key in ACTION_FIELDS if before["choice"][key] != after["choice"][key]]
        pairs.append({"failure_attempt_index": before["attempt_index"], "success_attempt_index": after["attempt_index"],
            "failure_evidence_id": before["evidence_id"], "success_evidence_id": after["evidence_id"],
            "source_task": before["source_task"], "same_recorded_state_and_pose": True,
            "same_remaining_parts": True, "same_base_and_gripper_budget": True,
            "before": deepcopy(before), "after": deepcopy(after), "changed_choice_fields": changed,
            "success_mover_in_observed_initial_disconnected_set": after["choice"]["part_id"] in before["initial_disconnected_parts"],
            "same_rng_state": before["rng_before"] == after["rng_before"] if before["rng_before"] is not None and after["rng_before"] is not None else None,
            "different_mover_and_possible_direction_or_support_are_confounders": True,
            "failure_cause": None, "causal_effect_established": False, "generalization_established": False,
            "reference_only": True, "current_part_id_mapping": None, "physics_reuse_allowed": False,
            "recommended_action": None})
    result = {"schema_version": 1, "kind": "observed_initial_contact_failure_then_different_mover_pass",
        "trace_attempt_count": len(report["attempts"]), "phase_denominator": dict(phase_counts),
        "eligible_observation_count": len(observations), "excluded_attempt_count": len(excluded),
        "excluded_attempts": excluded, "initial_failure_count": sum(o["initial_disconnected_failure"] for o in observations),
        "total_pair_count": total, "emitted_pair_count": len(pairs), "omitted_pair_count": total - len(pairs),
        "pairs": pairs, "unmatched_initial_failures": unmatched, "observations": observations,
        "geometry_contexts": geometry_contexts,
        "limits": {"max_attempts": max_attempts, "max_pairs": max_pairs,
                   "max_geometry_contexts": MAX_GEOMETRY_CONTEXTS, "max_geometry_parts": MAX_GEOMETRY_PARTS},
        "selection_rule": "chronological initial failure then first later accepted PASS with identical task/input/solver/remaining/base/gripper/pose; different mover is explicitly a changed variable",
        "new_api_calls": 0, "new_physical_calls": 0, "new_geometry_calculations": 0,
        "source_ids_are_current_bindings": False, "prescriptive_policy_generated": False}
    canonical(result)
    return result
