"""Read-only endpoint evidence for the pinned ASAP force path planner.

This observer does not step, reset, reposition, or decide feasibility.  Distances
are RedMax's sampled, collision-threshold-adjusted SDF distances.  An unmet
clearance predicate is not proof of collision, a blocker, or impossibility.
Importing this module needs neither ASAP nor NumPy (Python 3.8 compatible).
"""
from contextlib import contextmanager
from functools import wraps
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
import time
from typing import Any, Iterator


FRAME = "world_translation_xyz_euler_xyz"


def _number(value: Any, label: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError("invalid_" + label)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("invalid_" + label)
    if not math.isfinite(result):
        raise ValueError("nonfinite_" + label)
    return result


def _copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError):
        raise ValueError("scope_must_be_finite_json")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\x00\n\r"):
        raise ValueError("invalid_" + label)
    return value


def _vector(value: Any) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError("expected_world_xyz_euler_xyz")
    return [_number(item, "path_coordinate") for item in value]


def _thresholds(value: Any) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError("missing_observed_collision_thresholds")
    result = {}
    for part_id, threshold in value.items():
        _text(part_id, "threshold_part_id")
        result[part_id] = _number(threshold, "collision_threshold")
        if result[part_id] < 0:
            raise ValueError("negative_collision_threshold")
    return result


@contextmanager
def observed_collision_thresholds(planner_class_or_module: Any) -> Iterator[None]:
    """Temporarily observe constructor-calculated thresholds without changing them.

    Pass MultiPartPathPlanner itself, or its physics_planner module.  The exact
    original return object is passed through; only a detached JSON-like copy is
    attached to the new planner instance.  The original method is restored even
    on exceptions.  This process-local instrumentation is for a serial worker.
    """
    cls = getattr(planner_class_or_module, "MultiPartPathPlanner", planner_class_or_module)
    original = cls._compute_col_th_dict

    @wraps(original)
    def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
        raw = original(self, *args, **kwargs)
        self._evidence_col_th = _thresholds(raw)
        return raw

    cls._compute_col_th_dict = observed
    try:
        yield
    finally:
        cls._compute_col_th_dict = original


def collect_endpoint(planner: Any, raw_success: bool, path: Any, scope: dict,
                     event_id: str, trajectory_ref: str,
                     trajectory_sha256: str) -> dict:
    """Measure the current endpoint immediately after one check_success call.

    The caller must supply the actual complete task/solver/action scope and an
    already-saved trajectory's hash.  No file is opened here.  `physics.min_sep`
    must hold an explicitly overridden effective min_sep; otherwise the native
    planner.min_sep is used.  The caller retains responsibility for recording
    seed, budgets, mesh/pose/support identity, source/binary/cache fingerprints.

    A mismatched path/current simulator endpoint or inconsistent distance vs.
    native contact predicate is rejected, never silently turned into evidence.
    Geometry-query counts include the two distances inside body_in_contact.
    """
    started = time.perf_counter()
    if type(raw_success) is not bool:
        raise ValueError("raw_success_must_be_bool")
    _text(event_id, "event_id")
    _text(trajectory_ref, "trajectory_ref")
    ref = trajectory_ref.replace("\\", "/")
    if (PureWindowsPath(trajectory_ref).drive or PurePosixPath(ref).is_absolute()
            or ".." in ref.split("/") or ":" in ref or "//" in ref
            or ref in (".", "")):
        raise ValueError("trajectory_ref_must_be_relative_no_traversal")
    if not isinstance(trajectory_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", trajectory_sha256):
        raise ValueError("invalid_trajectory_sha256")
    if not isinstance(scope, dict):
        raise ValueError("scope_must_be_dict")
    copied_scope = _copy(scope)
    physics = copied_scope.setdefault("physics", {})
    if not isinstance(physics, dict):
        raise ValueError("scope_physics_must_be_dict")
    if hasattr(path, "tolist"):
        path = path.tolist()
    if not isinstance(path, (list, tuple)) or not path:
        raise ValueError("nonempty_trajectory_required")
    # Validate all samples, not just the last one referenced in the event.
    states = [_vector(q) for q in path]
    terminal_q = states[-1]
    part_move = _text(planner.part_move, "moving_part_id")
    parts_rest = list(planner.parts_fix)
    for part in parts_rest:
        _text(part, "rest_part_id")
    if (not parts_rest or len(parts_rest) != len(set(parts_rest))
            or part_move in parts_rest):
        raise ValueError("invalid_rest_part_ids")
    if "part_move" in copied_scope and copied_scope["part_move"] != part_move:
        raise ValueError("scope_moving_part_mismatch")
    if "parts_rest" in copied_scope and copied_scope["parts_rest"] != parts_rest:
        raise ValueError("scope_rest_parts_mismatch")
    thresholds = _thresholds(getattr(planner, "_evidence_col_th", None))
    if not set([part_move] + parts_rest).issubset(thresholds):
        raise ValueError("incomplete_collision_thresholds")
    if "col_th_dict" in physics and physics["col_th_dict"] != thresholds:
        raise ValueError("scope_collision_threshold_mismatch")
    physics["col_th_dict"] = thresholds
    # DFS removal history is not the actual obstacle population of this path
    # simulator.  Native check_assemblable constructs it with removed=[] even
    # when SequencePlanner's history is nonempty.  Keep both meanings separate.
    actual_removed = list(planner.parts_removed)
    for part in actual_removed:
        _text(part, "removed_part_id")
    if (len(actual_removed) != len(set(actual_removed))
            or set(actual_removed).intersection([part_move] + parts_rest)):
        raise ValueError("invalid_path_simulation_removed_parts")
    if ("parts_removed_in_path_simulation" in physics
            and physics["parts_removed_in_path_simulation"] != actual_removed):
        raise ValueError("scope_path_simulation_removed_parts_mismatch")
    physics["parts_removed_in_path_simulation"] = actual_removed
    min_sep = _number(physics.get("min_sep", planner.min_sep), "min_sep")
    if min_sep < 0:
        raise ValueError("negative_min_sep")
    physics["min_sep"] = min_sep
    if "frame" in copied_scope and copied_scope["frame"] != FRAME:
        raise ValueError("scope_coordinate_frame_mismatch")
    copied_scope["frame"] = FRAME

    body_move = "part" + part_move
    actual_terminal = _vector(planner.sim.get_joint_qm(body_move))
    if actual_terminal != terminal_q:
        raise ValueError("trajectory_endpoint_not_current_simulator_state")

    pairs = []
    for rest_id in parts_rest:
        body_rest = "part" + rest_id
        d_rm = _number(planner.sim.get_body_distance(body_rest, body_move), "distance_rest_move")
        d_mr = _number(planner.sim.get_body_distance(body_move, body_rest), "distance_move_rest")
        raw_contact = planner.sim.body_in_contact(body_rest, body_move, min_sep)
        if type(raw_contact) is not bool:
            raise ValueError("contact_predicate_must_be_bool")
        d_min = min(d_rm, d_mr)
        if raw_contact != (d_min < min_sep):
            raise ValueError("distance_contact_predicate_disagreement")
        pairs.append({"rest_part_id": rest_id, "moving_part_id": part_move,
                      "d_rest_move": d_rm, "d_move_rest": d_mr,
                      "min_distance": d_min, "separation_unsatisfied": raw_contact})

    contacts = list(planner.sim.get_contact_bodies(body_move))
    for body_name in contacts:
        _text(body_name, "contact_body_name")
    if len(contacts) != len(set(contacts)):
        raise ValueError("duplicate_contact_history_body")
    if raw_success and any(pair["separation_unsatisfied"] for pair in pairs):
        raise ValueError("successful_path_violates_endpoint_separation")
    elapsed = _number(time.perf_counter() - started, "measurement_seconds")
    if elapsed < 0:
        raise ValueError("negative_measurement_seconds")
    return {"schema_version": 1, "event_id": event_id,
            "kind": "path_endpoint_measurement", "scope": copied_scope,
            "raw_success": raw_success,
            "trajectory": {"sha256": trajectory_sha256, "relative_path": ref,
                           "states": len(states), "frame": FRAME},
            "terminal_q": terminal_q,
            "measurements": {"min_sep": min_sep, "pairs": pairs,
                             "contact_history": contacts,
                             "geometry_queries": 4 * len(pairs),
                             "measurement_seconds": elapsed},
            "blocker": None,
            "termination_reason": "success" if raw_success else "unknown"}
