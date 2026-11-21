"""Data-only source/current-frame checks; never a physical feasibility test."""
from copy import deepcopy
from itertools import permutations, product

from worker.explicit_action_contract import (
    DIRECTIONS, action_id, digest, validate_matrix, validate_request,
)
from orchestrator.skill_program import _scoped_last, step_program


def transform_vector(matrix, vector, transpose=False):
    return [sum((matrix[j][i] if transpose else matrix[i][j]) * vector[j]
                for j in range(3)) for i in range(3)]


def multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def closest_axis(vector):
    ranked = sorted(((sum(v[i] * vector[i] for i in range(3)), name)
                     for name, v in DIRECTIONS.items()), reverse=True)
    if ranked[0][0] < 0.999999:
        raise ValueError("pose_contract_requires_axis_aligned_vectors")
    return ranked[0][1]


def source_contract(packet):
    pairs = packet["context"].get("automatically_extracted_same_condition_pairs", {}).get("pairs", [])
    eligible = [pair for pair in pairs if pair.get("same_state") is True
                and pair.get("same_part") is True and pair.get("after_outcome", {}).get(
                    "accepted_for_transition") is True
                and pair.get("after", {}).get("skill_id") == "turn_force_two_leg_v1"]
    if len(eligible) != 1:
        raise ValueError("pose_contract_requires_one_bound_source_pair")
    pair = eligible[0]
    after = pair["after"]
    validate_matrix(after["pose_matrix"])
    vectors = [DIRECTIONS[after[key]] for key in ("direction_id", "second_direction_id")]
    return dict(version="source_frame_equivariance_v1",
        source_success_evidence_id=pair["success_evidence_id"],
        source_pose_matrix=deepcopy(after["pose_matrix"]),
        source_world_vectors=deepcopy(vectors),
        source_cad_vectors=[transform_vector(after["pose_matrix"], v, True) for v in vectors],
        rule="Freeze source pose/vectors from this evidence. ctx.last.pose_matrix is CURRENT, never source.",
        prior_bug="Using current rotation for both source inverse and target cancels adaptation.",
        runtime_absent_fields=["failed_recovery_choice"],
        physics_success_implied=False)


def cube_rotations():
    """24 proper axis-aligned rotations; no target assembly information."""
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            m = [[int(axes[i] == j) * signs[i] for j in range(3)] + [0] for i in range(3)]
            m.append([0, 0, 0, 1])
            try:
                validate_matrix(m)
            except ValueError:
                continue
            yield m


def synthetic_pose_request(template, matrix):
    """Labelled contract fixture, NOT a fresh actual verifier observation."""
    last = _scoped_last(validate_request(template))
    if last is None or last["feedback"].get("path_success") is not False:
        raise ValueError("pose_contract_requires_bound_path_failure")
    req = deepcopy(template)
    event = "synthetic-pose-contract:" + digest(matrix)
    pose_id = "pose_" + digest(matrix)[:16]
    old = deepcopy(last["choice"])
    # Rotate the failed probe from its original CAD frame as well.
    cad = transform_vector(last["pose_matrix"], DIRECTIONS[old["direction_id"]], True)
    old.update(pose_id=pose_id, direction_id=closest_axis(transform_vector(matrix, cad)), evidence_ids=[])
    req.update(run_id=event, request_id=event, state_id=event,
        pose_catalog=[dict(pose_id=pose_id, matrix=deepcopy(matrix))],
        attempted_action_ids=[], attempted_actions=[], path_stage_observations=[],
        episode_feedback=[], memory_references=[], evidence_ids=[])
    identifier = action_id(req, old)
    req["attempted_action_ids"] = [identifier]
    req["attempted_actions"] = [dict(action_id=identifier,
        **{key: value for key, value in old.items() if key != "evidence_ids"})]
    # The interpreter requires this label even in offline contract fixtures.
    # The enclosing report explicitly records synthetic=True and physics_calls=0.
    req["path_stage_observations"] = [dict(action_id=identifier, path_success=False,
        status="UNKNOWN", deadline_exceeded=False, backend="ASAP_RedMax", source_event_id=event)]
    req["episode_feedback"] = [dict(evidence_id=event, action_id=identifier, choice=old,
        remaining_parts=req["remaining_parts"], selected_pose_matrix=deepcopy(matrix),
        path_success=False, stability_success=None, status="UNKNOWN",
        accepted_for_transition=False, deadline_exceeded=False)]
    req["evidence_ids"] = [event]
    return validate_request(req)


def check_pose_contract(program, template, contract):
    rows = []
    for matrix in cube_rotations():
        req = synthetic_pose_request(template, matrix)
        expected = [closest_axis(transform_vector(matrix, v)) for v in contract["source_cad_vectors"]]
        step = step_program(program, req)
        response = step.get("response")
        actual = response["choice"] if response else None
        passed = bool(actual and actual["skill_id"] == "turn_force_two_leg_v1"
            and actual["part_id"] == req["episode_feedback"][0]["choice"]["part_id"]
            and actual["pose_id"] == req["pose_catalog"][0]["pose_id"]
            and [actual["direction_id"], actual["second_direction_id"]] == expected)
        rows.append(dict(matrix=matrix, expected_directions=expected,
            actual_choice=actual, passed=passed, abstention=step.get("not_applicable_reason")))
    return dict(kind="synthetic_coordinate_contract_NOT_PHYSICS", synthetic=True,
        api_calls=0, physical_calls=0, checked=len(rows), passed=sum(r["passed"] for r in rows),
        accepted=all(r["passed"] for r in rows), rows=rows,
        source_success_evidence_id=contract["source_success_evidence_id"])


def apply_declared_yaw(prepared, degrees):
    rotations = {
        90: [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        180: [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        270: [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    if type(degrees) is not int or degrees not in rotations:
        raise ValueError("declared_yaw_must_be_90_180_270")
    value = deepcopy(prepared)
    rotation = rotations[degrees]
    mapping = {}
    for pose in value["request"]["pose_catalog"]:
        matrix = multiply(rotation, pose["matrix"])
        validate_matrix(matrix)
        new_id = "pose_" + digest(matrix)[:16]
        mapping[pose["pose_id"]] = new_id
        pose.update(pose_id=new_id, matrix=matrix)
    choice = value["choice"]
    choice["pose_id"] = mapping[choice["pose_id"]]
    for key in ("direction_id", "second_direction_id"):
        if choice[key] is not None:
            choice[key] = closest_axis(transform_vector(rotation, DIRECTIONS[choice[key]]))
    value["request"].update(attempted_action_ids=[], attempted_actions=[], path_stage_observations=[],
                            episode_feedback=[], memory_references=[], evidence_ids=[])
    value["context"]["pose_catalog"] = deepcopy(value["request"]["pose_catalog"])
    value["context"]["pose_matrix"] = next(p["matrix"] for p in value["request"]["pose_catalog"]
                                          if p["pose_id"] == choice["pose_id"])
    validate_request(value["request"])
    return value
