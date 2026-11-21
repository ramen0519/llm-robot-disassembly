"""Geometry-scoped cross-task references, never transferable physics verdicts.

This pure Python 3.8 module consumes validated frozen observation records.  It
does not open files, run a verifier, choose a removal order, or change candidates.
Its distances are explicitly an uncalibrated retrieval metric, not confidence.
"""
import math
from typing import Any, Iterable, Optional

try:
    from .task_memory import (_copy, _normal_records, _validate_context,
                              canonical, digest)
except ImportError:
    from task_memory import (_copy, _normal_records, _validate_context,
                             canonical, digest)


VERSION = "geometry_reference_v1"
_STAGE_KEYS = ("path_found", "support_search_succeeded", "stability_passed",
               "path_search_attempted", "support_search_attempted")


def _number(value: Any) -> Optional[float]:
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _bounds(value: Any) -> Optional[list]:
    if type(value) is not dict:
        return None
    raw = value.get("bounds")
    if (type(raw) is not list or len(raw) != 2
            or any(type(row) is not list or len(row) != 3 for row in raw)):
        return None
    result = [[_number(item) for item in row] for row in raw]
    if any(item is None for row in result for item in row):
        return None
    extents = [result[1][axis] - result[0][axis] for axis in range(3)]
    if any(not math.isfinite(v) or v < 0 for v in extents):
        return None
    claimed = value.get("extents")
    if claimed is not None:
        if type(claimed) is not list or len(claimed) != 3:
            return None
        numbers = [_number(v) for v in claimed]
        if any(v is None or not math.isclose(v, extents[i], rel_tol=1e-8,
                                              abs_tol=1e-10)
               for i, v in enumerate(numbers)):
            return None
    return result


def geometry_descriptors(context: dict) -> dict:
    """Describe initial CAD geometry; missing/degenerate bounds are not guessed.

    Features are scale/translation normalized using the complete initial
    assembly.  Relative position and relative extents remain axis dependent;
    arbitrary rotational invariance, material identity and contacts are absent.
    Bounds-center is used instead of an uncertain mesh center of mass.
    """
    _validate_context(context)
    geometry = context["geometry_features"]
    valid = {part: _bounds(geometry[part]) for part in context["initial_parts"]}
    invalid = [part for part, bounds in valid.items() if bounds is None]
    # A partially described assembly would change normalization unpredictably.
    if invalid:
        return {"parts": {}, "unavailable_reason": "incomplete_or_invalid_initial_bounds",
                "invalid_parts": invalid}
    low = [min(bounds[0][axis] for bounds in valid.values()) for axis in range(3)]
    high = [max(bounds[1][axis] for bounds in valid.values()) for axis in range(3)]
    spans = [high[axis] - low[axis] for axis in range(3)]
    if any(not math.isfinite(v) for v in spans) or max(spans) <= 0:
        return {"parts": {}, "unavailable_reason": "degenerate_initial_assembly_bounds",
                "invalid_parts": []}
    parts = {}
    for part, bounds in valid.items():
        extents = [bounds[1][axis] - bounds[0][axis] for axis in range(3)]
        largest = max(extents)
        if largest <= 0:
            continue
        center = [bounds[0][axis] / 2 + bounds[1][axis] / 2 for axis in range(3)]
        normalized_center = [(center[axis] - low[axis]) / spans[axis]
                             if spans[axis] > 0 else 0.0 for axis in range(3)]
        relative_extents = [extents[axis] / spans[axis] if spans[axis] > 0 else 0.0
                            for axis in range(3)]
        box_volume = extents[0] * extents[1] * extents[2]
        feature = geometry[part]
        volume = _number(feature.get("volume"))
        validity = feature.get("validity", {})
        reliable = (type(validity) is dict and feature.get("watertight") is True
                    and validity.get("center_mass_status") == "uniform_density_estimate"
                    and validity.get("winding_consistent") is True)
        fill_ratio = None
        if (reliable and volume is not None and volume > 0
                and math.isfinite(box_volume) and box_volume > 0
                and volume <= box_volume * (1 + 1e-8)):
            fill_ratio = min(1.0, volume / box_volume)
        parts[part] = {
            "aspect_ratios_sorted": [v / largest for v in sorted(extents)],
            "relative_extents_axes": relative_extents,
            "normalized_bounds_center_axes": normalized_center,
            "volume_fill_ratio": fill_ratio,
            "zero_extent_axes": [axis for axis, value in enumerate(extents) if value == 0],
            "units": "dimensionless_initial_assembly_relative",
            "coordinate_frame": "initial_final_CAD_frame_not_current_stable_pose",
        }
    return {"parts": parts, "unavailable_reason": None, "invalid_parts": [],
            "zero_span_axes": [axis for axis, value in enumerate(spans) if value == 0]}


def _distance(source: dict, target: dict) -> dict:
    components = {}
    for key in ("aspect_ratios_sorted", "relative_extents_axes",
                "normalized_bounds_center_axes"):
        components[key] = sum(abs(a - b) for a, b in zip(source[key], target[key])) / 3
    if source["volume_fill_ratio"] is not None and target["volume_fill_ratio"] is not None:
        components["volume_fill_ratio"] = abs(source["volume_fill_ratio"] - target["volume_fill_ratio"])
    weights = {"aspect_ratios_sorted": 0.4, "relative_extents_axes": 0.3,
               "normalized_bounds_center_axes": 0.2, "volume_fill_ratio": 0.1}
    total = sum(components[key] * weights[key] for key in components)
    distance = total / sum(weights[key] for key in components)
    return {"geometry_distance": distance, "geometry_similarity": 1.0 / (1.0 + distance),
            "component_absolute_distances": components,
            "metric_version": VERSION, "similarity_is_probability": False,
            "calibrated_confidence": None,
            "ranking_or_removal_recommendation": None}


def _mesh_hashes(context: dict) -> Optional[list]:
    values = context["task"]["input_sha256"]
    if type(values) is not dict:
        return None
    result = sorted(value for name, value in values.items() if name.lower().endswith(".obj"))
    return result or None


def _shape_fingerprint(descriptors: dict) -> str:
    # ID-independent multiset.  This compares summaries, not mesh equivalence.
    return digest(sorted(canonical(value) for value in descriptors["parts"].values()))


def _scope(source: dict, current: dict, source_geometry: dict, current_geometry: dict) -> dict:
    source_meshes, current_meshes = _mesh_hashes(source), _mesh_hashes(current)
    if source_meshes is None or current_meshes is None:
        same_meshes = None
        kind = "mesh_identity_unresolved"
    else:
        same_meshes = source_meshes == current_meshes
        kind = ("same_mesh_family_different_conditions" if same_meshes
                else "different_mesh_inputs")
    return {"kind": kind, "same_task_context": canonical(source) == canonical(current),
            "same_mesh_hash_multiset": same_meshes,
            "mesh_input_bytes_differ": None if same_meshes is None else not same_meshes,
            "geometry_descriptor_multiset_differs": _shape_fingerprint(source_geometry) != _shape_fingerprint(current_geometry),
            "mesh_shape_equivalence_proven": False,
            "same_solver_settings": source["task"]["solver_settings"] == current["task"]["solver_settings"],
            "same_goal": source["task"]["goal"] == current["task"]["goal"],
            "same_base_part_id_is_correspondence_evidence": False,
            "same_numeric_part_id_is_correspondence_evidence": False}


def _source_facts(record: dict) -> dict:
    raw = record["raw_result"] if type(record["raw_result"]) is dict else {}
    stages = record.get("stage_feedback")
    stages = {key: stages[key] for key in _STAGE_KEYS
              if type(stages.get(key)) is bool} if type(stages) is dict else {}
    request = record["request"]
    return {"source_status": record["status"],
            "source_feasible": raw.get("feasible"),
            "source_action": _copy(raw.get("action")),
            "source_support": _copy(raw.get("parts_fix")),
            "source_pose": _copy(request["pose"]),
            "source_assembled_parts": sorted([request["part_move"]] + request["parts_rest"]),
            "source_parts_removed": sorted(request["parts_removed"]),
            "source_request_settings": {key: _copy(request[key]) for key in
                                        ("max_grippers", "timeout", "optimizer", "debug", "render")},
            "stage_feedback": stages or None,
            "error_type": record.get("error_type"),
            "blocker": None, "path": None,
            "scope": "source_bounded_verifier_attempt_only"}


def propose(context: dict, assembled_parts: Iterable[str], candidate_ids: Iterable[str],
            records: Iterable[dict], max_hints: int = 12, enabled: bool = True) -> dict:
    """Return bounded tentative geometry correspondences with immutable evidence.

    The output contains the complete candidate set in canonical transport order.
    It cannot be used as a physics cache or a candidate filter.  Observation
    status is not used in the retrieval distance.  All source/current candidate
    pairs compete equally; matching ID strings have zero semantic weight.
    Exact-context observations remain the responsibility of task_memory.
    """
    _validate_context(context)
    assembled, candidates = list(assembled_parts), list(candidate_ids)
    base = context["task"]["base_part"]
    if (any(type(value) is not str or not value for value in assembled + candidates)
            or len(set(assembled)) != len(assembled) or len(set(candidates)) != len(candidates)
            or not set(assembled).issubset(context["initial_parts"])
            or (base is not None and base not in assembled)
            or set(candidates) != set(assembled) - ({base} if base is not None else set())
            or type(max_hints) is not int or not 0 <= max_hints <= 100
            or type(enabled) is not bool):
        raise ValueError("invalid transfer state, full candidate set, limit or mode")
    result = {"schema_version": 1, "kind": "tentative_cross_task_geometry_references",
              "enabled": enabled, "metric_version": VERSION,
              "current_context_sha256": digest(context),
              "current_assembled_parts": sorted(assembled),
              "available_candidate_ids": sorted(candidates),
              "tentative_correspondences": [], "evidence_ids": [],
              "constraints": [], "allow_hard_pruning": False,
              "physics_reuse_allowed": False, "current_attempt_verified": False,
              "candidate_order_recommendation": None,
              "limitations": ["uncalibrated_geometry_retrieval_not_physics",
                              "axis_dependent_initial_CAD_positions_not_stable_pose",
                              "similar_shape_does_not_prove_same_contact_or_removal",
                              "no_source_ID_action_or_support_transferred_as_current_facts"],
              "diagnostics": {"source_record_count": 0, "pair_count": 0,
                              "same_context_skipped": 0, "legacy_without_geometry_skipped": 0,
                              "geometry_unavailable_skipped": 0, "truncated": False},
              "source_records_sha256": None}
    if not enabled:
        return result
    normalized = _normal_records(records)
    result["source_records_sha256"] = digest(normalized)
    result["diagnostics"]["source_record_count"] = len(normalized)
    current_geometry = geometry_descriptors(context)
    if not current_geometry["parts"]:
        result["diagnostics"]["current_geometry_unavailable"] = current_geometry["unavailable_reason"]
        return result
    pairs, descriptor_cache = [], {}
    for record in normalized:
        if record.get("imported_legacy") is True:
            result["diagnostics"]["legacy_without_geometry_skipped"] += 1
            continue
        source_context = record["context"]
        if canonical(source_context) == canonical(context):
            result["diagnostics"]["same_context_skipped"] += 1
            continue
        context_id = digest(source_context)
        if context_id not in descriptor_cache:
            descriptor_cache[context_id] = geometry_descriptors(source_context)
        source_geometry = descriptor_cache[context_id]
        source_part = record["request"]["part_move"]
        if source_part not in source_geometry["parts"]:
            result["diagnostics"]["geometry_unavailable_skipped"] += 1
            continue
        scope = _scope(source_context, context, source_geometry, current_geometry)
        for current_part in sorted(candidates):
            if current_part not in current_geometry["parts"]:
                continue
            source_descriptor = source_geometry["parts"][source_part]
            target_descriptor = current_geometry["parts"][current_part]
            metrics = _distance(source_descriptor, target_descriptor)
            source_task = source_context["task"]
            pairs.append({
                "hint_id": digest({"source_record_id": record["record_id"],
                                   "source_context": context_id,
                                   "current_context": digest(context),
                                   "current_candidate_id": current_part}),
                "source_evidence_id": record["record_id"],
                "source_part_id": source_part, "current_candidate_id": current_part,
                "mapping_status": "tentative_geometry_correspondence",
                "current_attempt_verified": False, "current_physical_facts": None,
                "source_geometry_descriptor": _copy(source_descriptor),
                "current_geometry_descriptor": _copy(target_descriptor),
                "similarity": metrics, "task_scope": _copy(scope),
                "source_observation": _source_facts(record),
                "source_provenance": {"record_id": record["record_id"],
                                      "context_sha256": context_id,
                                      "assembly_id": source_task["assembly_id"],
                                      "input_sha256": _copy(source_task["input_sha256"]),
                                      "source_commit": source_task["source_commit"],
                                      "solver_settings_sha256": digest(source_task["solver_settings"]),
                                      "goal": _copy(source_task["goal"]),
                                      "log": _copy(record["source"])},
                "constraints": [], "allow_hard_pruning": False,
                "interpretation": "source_observation_as_reference_only_reverify_current_candidate"})
    pairs.sort(key=lambda value: (value["similarity"]["geometry_distance"],
                                 value["source_evidence_id"], value["current_candidate_id"]))
    result["diagnostics"]["pair_count"] = len(pairs)
    result["diagnostics"]["truncated"] = len(pairs) > max_hints
    result["tentative_correspondences"] = pairs[:max_hints]
    result["evidence_ids"] = sorted({value["source_evidence_id"] for value in pairs[:max_hints]})
    return result
