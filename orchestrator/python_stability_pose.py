"""Source-frame evidence and synthetic checks for stability recovery Python.

These functions perform no physics, API requests, candidate pruning or skill
admission. A coordinate contract is not a proof of path or stability success.
"""
from copy import deepcopy
from orchestrator.python_skill_pose import transform_vector, multiply, cube_rotations
from orchestrator.skill_program import _scoped_last, step_program
from worker.explicit_action_contract import DIRECTIONS, action_id, digest, validate_request, validate_matrix


def nearest_axis(vector):
    return max(DIRECTIONS, key=lambda name: sum(a*b for a, b in zip(DIRECTIONS[name], vector)))


def support_preservation_scope(request):
    """Current role/capacity facts; never a physical success or source-ID binding."""
    request = validate_request(request)
    last = _scoped_last(request)
    if (len(request["remaining_parts"]) <= 2 or not last or last["feedback"].get("path_success") is not True
            or last["feedback"].get("stability_success") is not False
            or not last["initial_disconnected"]):
        return None
    mover = last["choice"]["part_id"]
    supports = [p for p in request["remaining_parts"] if p not in (mover, request["base_part"])]
    capacity = request["max_grippers"] - 1
    if len(supports) > capacity or not supports:
        return None
    return dict(mover=mover, all_other_nonbase_parts=supports, support_capacity=capacity,
        previous_path_choice=deepcopy(last["choice"]),
        evidence_id=last["feedback"]["evidence_id"],
        fresh_path_verification_required=True, physical_success_implied=False,
        stability_boundary="no free remaining body if all these supports are fixed",
        scope="more than two remaining parts; legacy two-part branch is separate")


def stability_source_contract(packet):
    """Use the first chronological multi-body, single-force observed transition."""
    pairs = packet["context"]["initial_contact_observed_transitions"]["pairs"]
    eligible = [p for p in pairs if p["same_recorded_state_and_pose"] is True
        and p["after"]["path_success"] is True and p["after"]["stability_success"] is True
        and len(p["before"]["remaining_parts"]) > 2
        and p["after"]["choice"]["skill_id"] == "single_force_v1"
        and p["before"]["choice"]["part_id"] in p["after"]["choice"]["support_ids"]]
    if not eligible:
        raise ValueError("stability_pose_requires_bound_multibody_success")
    pair = min(eligible, key=lambda p: p["success_attempt_index"])
    pose = pair["after"]["pose_matrix"]
    validate_matrix(pose)
    vector = DIRECTIONS[pair["after"]["choice"]["direction_id"]]
    return dict(version="stability_source_frame_v1",
        source_success_evidence_id=pair["success_evidence_id"],
        source_failure_evidence_id=pair["failure_evidence_id"],
        source_pose_matrix=deepcopy(pose), source_world_vector=list(vector),
        source_cad_vector=transform_vector(pose, vector, True),
        application_scope="current initial-disconnected failure with more than two remaining parts",
        selection_rule="first chronological eligible development pair, not admission outcomes",
        source_part_ids_are_not_bindings=True, physics_success_implied=False,
        support_change_does_not_certify_path=True)


def synthetic_stability_request(template, rotation):
    """Counterfactual coordinate fixture; never treated as new physical evidence."""
    template = validate_request(template)
    last = _scoped_last(template)
    if (not last or not last["initial_disconnected"] or len(template["remaining_parts"]) <= 2
            or last["feedback"]["path_success"] is not True
            or last["feedback"]["stability_success"] is not False):
        raise ValueError("initial_multibody_stability_template_required")
    validate_matrix(rotation)
    req = deepcopy(template)
    matrix = multiply(rotation, last["pose_matrix"])
    pose_id = "pose_" + digest(matrix)[:16]
    event = "synthetic-stability-pose:" + digest(matrix)
    choice = deepcopy(last["choice"])
    choice.update(pose_id=pose_id, evidence_ids=[])
    for key in ("direction_id", "second_direction_id"):
        if choice[key] is not None:
            choice[key] = nearest_axis(transform_vector(rotation, DIRECTIONS[choice[key]]))
    req.update(run_id=event, request_id=event, state_id=event,
        pose_catalog=[dict(pose_id=pose_id, matrix=matrix)],
        attempted_action_ids=[], attempted_actions=[], path_stage_observations=[],
        episode_feedback=[], memory_references=[], evidence_ids=[])
    identity = action_id(req, choice)
    feedback = deepcopy(last["feedback"])
    feedback.update(evidence_id=event, action_id=identity, choice=choice, selected_pose_matrix=matrix)
    req.update(attempted_action_ids=[identity],
        attempted_actions=[dict(action_id=identity, **{k: v for k, v in choice.items() if k != "evidence_ids"})],
        path_stage_observations=[dict(action_id=identity, path_success=True, status="UNKNOWN",
            deadline_exceeded=False, backend="ASAP_RedMax", source_event_id=event)],
        episode_feedback=[feedback], evidence_ids=[event])
    return validate_request(req)


def check_stability_pose_contract(program, template, contract):
    rows = []
    for rotation in cube_rotations():
        req = synthetic_stability_request(template, rotation)
        matrix = req["pose_catalog"][0]["matrix"]
        expected = nearest_axis(transform_vector(matrix, contract["source_cad_vector"]))
        step = step_program(program, req)
        choice = (step.get("response") or {}).get("choice")
        support = (support_preservation_scope(req)
                   if contract.get("preserve_path_when_all_remaining_fit") else None)
        if support:
            old = support["previous_path_choice"]
            passed = bool(choice and all(choice[k] == old[k] for k in
                ("part_id", "pose_id", "skill_id", "direction_id", "second_direction_id"))
                and set(choice["support_ids"]) == set(support["all_other_nonbase_parts"]))
        else:
            passed = bool(choice and choice["pose_id"] == req["pose_catalog"][0]["pose_id"]
                and choice["skill_id"] == "single_force_v1" and choice["direction_id"] == expected)
        rows.append(dict(rotation=rotation, expected_direction=expected, actual_choice=choice,
                         support_preservation=support, passed=passed, reason=step.get("not_applicable_reason")))
    return dict(kind="synthetic_stability_coordinates_NOT_PHYSICS", synthetic=True,
        physical_calls=0, api_calls=0, checked=len(rows), passed=sum(r["passed"] for r in rows),
        accepted=all(r["passed"] for r in rows), rows=rows,
        source_success_evidence_id=contract["source_success_evidence_id"])


def failed_debug_requirements(packet):
    """Observations and executable change requirements, never target answers."""
    rows = []
    for item in packet.get("validation_feedback", []):
        if item.get("phase") != "debug":
            raise ValueError("revision_gate_accepts_debug_only")
        raw = item.get("recovery")
        if (not raw or raw.get("status") not in ("UNKNOWN", "REJECTED")
                or raw.get("path_success") is not True or raw.get("stability_success") is not False
                or raw.get("stability_check_calls") != 1 or raw.get("elapsed_seconds", 999) > 30):
            continue
        diagnostic = raw.get("stability_diagnostics") or {}
        if diagnostic.get("phase") not in ("dynamic_failure", "initial_disconnected"):
            continue
        choice = raw["chosen"]
        geometry = item["current_observation_request"]["geometry"]["parts"]
        current = _scoped_last(item["current_observation_request"])
        current_phase = ("initial_disconnected" if current and current["initial_disconnected"]
                         else "dynamic_failure" if current and current["unstable"] else None)
        free = [p for p in item["current_observation_request"]["remaining_parts"]
                if p != choice["part_id"] and p not in choice["support_ids"]
                and p != item["current_observation_request"]["base_part"]]
        finite = [p for p in free if type(geometry.get(p, {}).get("volume")) in (int, float)
                  and geometry[p].get("validity", {}).get("volume_finite") is True]
        ordered = sorted(finite, key=lambda p: geometry[p]["volume"])
        features = [dict(source_part_id=p, volume=geometry[p]["volume"],
                         ascending_volume_rank=ordered.index(p) + 1,
                         among_unheld_remaining_parts=len(ordered))
                    for p in ordered if p in {r["part_id"] for r in diagnostic.get("dynamic_failure_parts", [])}]
        rows.append(dict(case_id=item["case_id"],
            failed_recovery_choice=deepcopy(choice), failed_pose_matrix=deepcopy(raw["pose"]),
            diagnostic=deepcopy(diagnostic),
            branch_invoked_before_recovery=current_phase,
            failure_after_recovery=diagnostic["phase"],
            observed_failed_unheld_part_features=features,
            volume_rank_is_observation_not_mass_or_cause=True,
            observed_parts_geometry={p: {k: deepcopy(v[k]) for k in ("extents", "volume") if k in v}
                                     for p, v in geometry.items()},
            required_change="mover or pose or support set; force alone leaves this checker configuration unchanged",
            runtime_future_feedback_not_available=True, literal_source_part_ids_forbidden=True,
            next_action_recommendation=None, physical_cause=None))
    return rows


def check_stability_revision(program, packet):
    requirements = failed_debug_requirements(packet)
    feedback = {r["case_id"]: r for r in packet.get("validation_feedback", [])}
    rows = []
    for requirement in requirements:
        item = feedback[requirement["case_id"]]
        response = step_program(program, item["current_observation_request"]).get("response")
        old = requirement["failed_recovery_choice"]
        choice = response["choice"] if response else None
        pose = (next(p["matrix"] for p in item["current_observation_request"]["pose_catalog"]
                     if p["pose_id"] == choice["pose_id"]) if choice else None)
        same = bool(choice and choice["part_id"] == old["part_id"]
                    and pose == requirement["failed_pose_matrix"]
                    and set(choice["support_ids"]) == set(old["support_ids"]))
        rows.append(dict(case_id=item["case_id"], repeated_failed_stability_configuration=same,
            proposed_choice=choice, abstained=choice is None, passed=not same))
    for item in packet.get("validation_feedback", []):
        if not item.get("retain_successful_probe"):
            continue
        raw = item["recovery"]
        response = step_program(program, item["current_observation_request"]).get("response")
        actual = response["choice"] if response else None
        expected = raw["chosen"]
        same = bool(actual and all(actual[k] == expected[k] for k in
            ("part_id", "pose_id", "skill_id", "direction_id", "second_direction_id"))
            and set(actual["support_ids"]) == set(expected["support_ids"]))
        rows.append(dict(case_id=item["case_id"], successful_development_probe_preserved=same,
            proposed_choice=actual, passed=same))
    return dict(kind="offline_debug_revision_NOT_PHYSICS", rows=rows,
        accepted=all(r["passed"] for r in rows), physics_calls=0, api_calls=0,
        runtime_candidate_pruning=False, success_implied=False)
