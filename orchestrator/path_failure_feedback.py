"""Fact-only path evidence for planning and registered proposal review.

Endpoint gaps describe one observed terminal state, not blockers or global
constraints. A matching failed observation never removes a candidate or skill.
"""
from copy import deepcopy
import math

POLICIES = ("legacy_v1", "path_evidence_v1", "repair_evidence_v2")


def endpoint_brief(value, mover, remaining):
    if value is None:
        return None
    if (type(value) is not dict or value.get("scope") != "selected_trajectory_terminal_state_only"
            or value.get("all_paths_impossible") is not False or value.get("blocker") is not None):
        return None
    minimum = value.get("min_sep")
    if type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum < 0:
        return None
    pairs = value.get("pairs")
    if type(pairs) is not list:
        return None
    observed, unsatisfied, seen = [], [], set()
    for pair in pairs:
        if type(pair) is not dict:
            return None
        part, distance = pair.get("rest_part_id"), pair.get("min_distance")
        if (pair.get("moving_part_id") != mover or part not in set(remaining) - {mover}
                or part in seen or type(distance) not in (int, float) or not math.isfinite(distance)
                or type(pair.get("separation_unsatisfied")) is not bool
                or pair["separation_unsatisfied"] != (distance < minimum)):
            return None
        seen.add(part)
        observed.append(dict(part_id=part, min_distance=distance, separation_unsatisfied=distance < minimum))
        if distance < minimum:
            unsatisfied.append(part)
    return dict(scope=value["scope"], min_sep=minimum, pairs=observed,
        observed_separation_unsatisfied_ids=unsatisfied, measured_pair_count=len(pairs),
        expected_other_parts=len(set(remaining)-{mover}),
        termination_reason=value.get("termination_reason") if value.get("termination_reason") in
            ("success", "unknown") else "unknown",
        blocker=None, all_paths_impossible=False, hard_pruning=False,
        evidence_sha256=value.get("evidence_sha256"))


def observations(request):
    for kind, collection in (("current_episode", request["episode_feedback"]),
                             ("frozen_memory", request["memory_references"])):
        for row in collection:
            frozen = kind == "frozen_memory"
            choice = row.get("source_choice" if frozen else "choice")
            remaining = row.get("source_remaining_parts" if frozen else "remaining_parts")
            if type(choice) is not dict or type(remaining) is not list:
                continue
            yield kind, row, choice, remaining


def failure_evidence(request):
    """Compact source facts, including failures of the alternative itself."""
    rows = []
    for kind, row, choice, remaining in observations(request):
        if row.get("path_success") is not False:
            continue
        raw = row.get("measured_endpoint") if kind == "frozen_memory" else row.get("endpoint_observation")
        # Current episode feedback uses measured_endpoint, just like memory.
        if raw is None:
            raw = row.get("measured_endpoint")
        brief = endpoint_brief(raw, choice["part_id"], remaining)
        rows.append(dict(evidence_id=row["evidence_id"], source_kind=kind, choice=deepcopy(choice),
            source_remaining_parts=list(remaining),
            source_pose_matrix=deepcopy(row.get("source_pose_matrix" if kind == "frozen_memory"
                                                else "selected_pose_matrix")),
            path_success=False, stability_success=row.get("stability_success"),
            endpoint=brief, endpoint_available=brief is not None, status=row.get("status"),
            current_attempt_verified=False, physics_reuse_allowed=False))
    return dict(version="path_evidence_v1", observations=rows,
        path_fixed_bodies="all other remaining bodies, independent of chosen supports",
        support_effect_scope="post-removal stability only; does not change the same path-check configuration",
        endpoint_gap_scope="terminal distances only, not a separating path or blocker proof",
        failed_recovery_is_new_evidence=True, candidate_pruning=False,
        action_recommendation=None, missing_measurement_is_not_success=True)


def proposal_history(request, choice):
    """Same CAD/state/pose/action comparisons; never an admissibility guard."""
    if choice is None:
        return []
    pose = next(p["matrix"] for p in request["pose_catalog"] if p["pose_id"] == choice["pose_id"])
    result = []
    from orchestrator.explicit_action_provider_v5 import same_mesh_id_correspondence
    for kind, row, source, remaining in observations(request):
        if kind == "frozen_memory":
            proof = same_mesh_id_correspondence(request, row)
            if not proof or set(proof["parts"]) != set(request["remaining_parts"]):
                continue
        if set(remaining) != set(request["remaining_parts"]):
            continue
        source_pose = row.get("source_pose_matrix" if kind == "frozen_memory" else "selected_pose_matrix")
        if source_pose != pose or any(source[k] != choice[k] for k in
                                     ("part_id", "skill_id", "direction_id", "second_direction_id")):
            continue
        result.append(dict(evidence_id=row["evidence_id"], source_kind=kind, status=row.get("status"),
            path_success=row.get("path_success"), stability_success=row.get("stability_success"),
            source_support_ids=source["support_ids"],
            same_support_set=set(source["support_ids"]) == set(choice["support_ids"]),
            solver_seed_budget_equivalence=row.get("same_conditions") if kind == "frozen_memory" else None,
            current_result_guaranteed=False, disables_proposal=False))
    return result
