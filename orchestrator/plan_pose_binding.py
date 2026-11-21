"""Model-authorized, bounded pose rebinding; never a feasibility verdict."""
from copy import deepcopy
import math

from worker.explicit_action_contract import validate_matrix

EXACT = "exact_orientation_v1"
BOUNDED = "nearest_within_5deg_world_axes_v1"
POLICIES = (EXACT, BOUNDED)
MAX_ANGLE_DEGREES = 5.0
TIE_DEGREES = 1e-6


def rotation_angle(reference, matrix):
    trace = sum(reference[i][j] * matrix[i][j] for i in range(3) for j in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


def resolve_pose(reference, request, policy):
    if policy not in POLICIES:
        raise ValueError("unknown_plan_pose_policy")
    if (type(reference) is not list or len(reference) != 3
            or any(type(row) is not list or len(row) != 3 for row in reference)):
        raise ValueError("invalid_plan_reference_rotation")
    validate_matrix([list(row) + [0] for row in reference] + [[0, 0, 0, 1]])
    catalog = request["pose_catalog"]
    for row in catalog:
        validate_matrix(row["matrix"])
    exact = [row for row in catalog if all(
        abs(row["matrix"][i][j] - reference[i][j]) <= 1e-6
        for i in range(3) for j in range(3))]
    if len(exact) > 1:
        raise ValueError("planned_orientation_ambiguous")
    if exact:
        selected, kind = exact[0], "exact"
    elif policy == EXACT:
        raise ValueError("planned_orientation_unavailable")
    else:
        scores = sorted((rotation_angle(reference, row["matrix"]), row["pose_id"], row)
                        for row in catalog)
        if not scores or scores[0][0] > MAX_ANGLE_DEGREES + 1e-9:
            raise ValueError("planned_pose_adjustment_exceeds_limit")
        if len(scores) > 1 and abs(scores[1][0] - scores[0][0]) <= TIE_DEGREES:
            raise ValueError("planned_pose_adjustment_ambiguous")
        selected, kind = scores[0][2], "bounded_rebind"
    audit = dict(policy=policy, match_kind=kind, reference_rotation=deepcopy(reference),
        selected_pose_id=selected["pose_id"], selected_pose_matrix=deepcopy(selected["matrix"]),
        rotation_delta_degrees=rotation_angle(reference, selected["matrix"]),
        angle_limit_degrees=MAX_ANGLE_DEGREES if policy == BOUNDED else None,
        comparison_anchor="original_plan_reference_not_previous_rebind",
        force_frame="world_axes_unchanged", support_ids_unchanged=True,
        current_catalog_only=True, fresh_path_and_stability_required=True,
        physics_success_implied=False, pose_transition_validated=False)
    return selected["pose_id"], audit
