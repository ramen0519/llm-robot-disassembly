"""Execute one selected action with ASAP physics, without hidden option search.

Python 3.8 compatible.  Importing this module does not import ASAP or NumPy.
``execute_action`` never owns a DFS state or invokes a candidate generator.
The path abstraction fixes every other remaining part, as pinned ASAP does;
the separate gravity test fixes ONLY the selected supports plus mandatory base.
Neither grasp accessibility nor physical transitions between poses are tested.

Invalid requests raise before importing a solver or creating an output folder.
Valid requests create a NEW folder and return PASS / UNKNOWN / ERROR, retaining
raw simulator results.  A failed sampled attempt never proves impossibility.
"""
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional


AXES = {"xp": [1, 0, 0], "xn": [-1, 0, 0], "yp": [0, 1, 0],
        "yn": [0, -1, 0], "zp": [0, 0, 1], "zn": [0, 0, -1]}
SKILLS = ("single_force_v1", "turn_force_two_leg_v1")
SWITCH_STEP = 50
CHOICE_FIELDS = {"part_id", "pose_id", "skill_id", "direction_id",
                 "second_direction_id", "support_ids"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + "\n")


def _ids(value: Any, name: str, empty: bool = True) -> list:
    if (type(value) is not list or (not empty and not value)
            or any(type(item) is not str or not item or
                   any(c in item for c in "\x00\r\n/\\") for item in value)
            or len(set(value)) != len(value)):
        raise ValueError("invalid_" + name)
    return list(value)


def _finite(value: Any, name: str, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid_" + name)
    if value < 0 or (positive and value == 0):
        raise ValueError("invalid_" + name)
    return float(value)


def _pose(value: Any) -> Any:
    if value is None:
        return None  # Explicit original/input pose, not a search request.
    if (type(value) is not list or len(value) != 4
            or any(type(row) is not list or len(row) != 4 for row in value)
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   for row in value for v in row)):
        raise ValueError("pose_must_be_finite_4x4_or_null")
    if any(abs(a - b) > 1e-8 for a, b in zip(value[3], [0, 0, 0, 1])):
        raise ValueError("pose_must_be_rigid_transform")
    for i in range(3):
        for j in range(3):
            if abs(sum(value[k][i] * value[k][j] for k in range(3))
                   - (1 if i == j else 0)) > 1e-6:
                raise ValueError("pose_rotation_must_be_orthonormal")
    r = value
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    if abs(det - 1) > 1e-6:
        raise ValueError("pose_rotation_must_preserve_handedness")
    return deepcopy(value)


def validate_action(context: dict, choice: dict) -> dict:
    """Validate selector output again at the physical boundary (no file I/O)."""
    if type(context) is not dict or type(choice) is not dict:
        raise ValueError("context_and_choice_must_be_objects")
    _canonical(context)
    _canonical(choice)
    if not CHOICE_FIELDS.issubset(choice) or set(choice) - CHOICE_FIELDS - {"evidence_ids"}:
        raise ValueError("invalid_choice_fields")
    remaining = _ids(context.get("remaining_parts"), "remaining_parts", empty=False)
    if len(remaining) < 2:
        raise ValueError("removal_requires_at_least_two_remaining_parts")
    base = context.get("base_part")
    if base is not None and base not in remaining:
        raise ValueError("base_must_be_remaining")
    part = choice["part_id"]
    if type(part) is not str or part not in remaining or part == base:
        raise ValueError("chosen_part_not_removable")
    support = _ids(choice["support_ids"], "support_ids")
    if any(item not in remaining or item in (part, base) for item in support):
        raise ValueError("supports_must_be_other_remaining_parts_excluding_base")
    max_grippers = context.get("max_grippers")
    if (type(max_grippers) is not int or max_grippers < 1
            or len(support) + 1 > max_grippers):
        raise ValueError("selected_supports_exceed_gripper_budget")
    if (type(choice["skill_id"]) is not str or choice["skill_id"] not in SKILLS
            or type(choice["direction_id"]) is not str or choice["direction_id"] not in AXES):
        raise ValueError("unregistered_skill_or_direction")
    second = choice["second_direction_id"]
    if choice["skill_id"] == "single_force_v1":
        if second is not None:
            raise ValueError("single_force_requires_null_second_direction")
    elif (type(second) is not str or second not in AXES or sum(a * b for a, b in
          zip(AXES[choice["direction_id"]], AXES[second])) != 0):
        raise ValueError("two_leg_requires_perpendicular_registered_direction")
    _ids(choice.get("evidence_ids", []), "evidence_ids")
    catalog = context.get("pose_catalog")
    if type(catalog) is not list or not catalog:
        raise ValueError("pose_catalog_required")
    poses = {}
    for entry in catalog:
        if (type(entry) is not dict or not {"pose_id", "matrix"}.issubset(entry)
                or type(entry["pose_id"]) is not str or not entry["pose_id"]
                or entry["pose_id"] in poses):
            raise ValueError("invalid_pose_catalog")
        poses[entry["pose_id"]] = _pose(entry["matrix"])
    if type(choice["pose_id"]) is not str or choice["pose_id"] not in poses:
        raise ValueError("unknown_pose_id")
    seed = context.get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid_seed")
    _finite(context.get("budget", 1), "budget", positive=True)
    for name in ("stability_timeout", "remaining_wall_seconds"):
        if context.get(name) is not None:
            _finite(context[name], name)
    if 'stability_diagnostics' in context and type(context['stability_diagnostics']) is not bool:
        raise ValueError('stability_diagnostics_must_be_boolean')
    removed = _ids(context.get("removed_parts", context.get("parts_removed", [])), "removed_parts")
    if set(removed).intersection(remaining):
        raise ValueError("removed_parts_overlap_remaining")
    rest = [item for item in remaining if item != part]
    fixed = ([] if base is None else [base]) + support
    return {"remaining": remaining, "rest": rest, "base": base,
            "fixed": fixed, "stability_movers": [p for p in rest if p not in fixed],
            "pose": poses[choice["pose_id"]], "removed": removed,
            "seed": seed, "max_grippers": max_grippers}


def _load_runtime(asap_root: Path) -> Any:
    """Only the actual executor imports the legacy Python/ASAP environment."""
    if str(asap_root) not in sys.path:
        sys.path.insert(0, str(asap_root))
    import numpy as np
    import redmax_py
    physics = importlib.import_module("plan_sequence.physics_planner")
    sim_string = importlib.import_module("plan_sequence.sim_string")
    # A cached import from another checkout must not silently verify this run.
    for module in (physics, sim_string):
        Path(module.__file__).absolute().relative_to(asap_root)
    from path_repair import execute_single, execute_two_leg
    source_files = {
        "redmax_binary_sha256": Path(redmax_py.__file__),
        "physics_source_sha256": Path(physics.__file__),
        "sim_string_sha256": Path(sim_string.__file__),
        "controller_sha256": Path(__file__).with_name("path_repair.py"),
        "observer_sha256": Path(__file__).with_name("path_evidence.py"),
        "explicit_executor_sha256": Path(__file__),
    }
    return SimpleNamespace(np=np, physics=physics, execute_single=execute_single,
                           execute_two_leg=execute_two_leg, source_files=source_files)


def _asset_snapshot(assembly: Path) -> dict:
    from remote_rank_client import checked_path
    result = {}
    for item in sorted(assembly.iterdir()):
        if item.suffix.lower() in (".obj", ".json", ".sdf"):
            safe = checked_path(item, run_only=False)
            if not safe.is_file():
                raise ValueError("assembly_inputs_must_be_regular_files")
            result[safe.name] = _sha(safe)
    return result


def _rng_digest(np: Any) -> str:
    state = np.random.get_state()
    converted = [item.tolist() if hasattr(item, "tolist") else item for item in state]
    return hashlib.sha256(_canonical({"python": random.getstate(), "numpy": converted}).encode()).hexdigest()


def _remaining(deadline: Optional[float]) -> Optional[float]:
    return None if deadline is None else max(0., deadline - time.perf_counter())


def _reuse_source_files(root):
    # Construction depends on these modules as well as the native verifier.
    # Resolve only the explicitly loaded checkout, never enumerate other drives.
    from remote_rank_client import checked_path
    sources = {'scene_session_sha256': Path(__file__).with_name('path_scene_reuse.py'),
               'readonly_sdf_sha256': Path(__file__).with_name('replay_path_extension.py')}
    for name in ('plan_path.run_connect', 'assets.load', 'assets.transform'):
        module = importlib.import_module(name)
        path = checked_path(module.__file__, run_only=False)
        path.relative_to(root)
        sources[name.replace('.', '_') + '_sha256'] = path
    return sources


def _scene_key(root, assembly, selected, choice, hashes, inputs, runtime):
    cls = runtime.physics.MultiPartPathPlanner
    return dict(version=1, asap_root=str(root), assembly_dir=str(assembly),
        source_hashes=hashes, input_hashes=inputs, part_move=choice['part_id'],
        path_fixed_parts=selected['rest'], path_removed_parts=[],
        remaining_parts=selected['remaining'], removed_ledger=selected['removed'],
        pose=selected['pose'], base_part=selected['base'], seed=selected['seed'],
        # Direction/support are intentionally absent: direction is applied AFTER
        # reset; supports belong to a separately constructed stability scene.
        constructor=dict(save_sdf=True, xml_load_sdf=True, xml_save_sdf=False),
        native_defaults={name: getattr(cls, name) for name in
            ('frame_skip', 'max_time', 'min_sep', 'contact_eps', 'col_th_base')})


def endpoint_observation(event: Optional[dict], evidence_sha256: Optional[str]) -> Optional[dict]:
    """Project an already collected endpoint, without measurement or inference.

    The normal caller supplies the validated ``collect_endpoint`` event. Missing
    facts remain null; a small distance is never renamed collision or blocker.
    No simulator, filesystem, threshold evaluation, or additional query is used.
    """
    if type(event) is not dict:
        return None
    measured = event.get("measurements")
    measured = measured if type(measured) is dict else {}

    def number(value: Any) -> Optional[float]:
        return value if type(value) in (int, float) and math.isfinite(value) else None

    pairs = None
    if type(measured.get("pairs")) is list:
        pairs = []
        for pair in measured["pairs"]:
            if type(pair) is not dict:
                continue
            pairs.append({
                "moving_part_id": pair.get("moving_part_id") if type(pair.get("moving_part_id")) is str else None,
                "rest_part_id": pair.get("rest_part_id") if type(pair.get("rest_part_id")) is str else None,
                "min_distance": number(pair.get("min_distance")),
                "separation_unsatisfied": pair.get("separation_unsatisfied")
                    if type(pair.get("separation_unsatisfied")) is bool else None,
            })
    reason = event.get("termination_reason")
    valid_sha = (type(evidence_sha256) is str and len(evidence_sha256) == 64
                 and all(c in "0123456789abcdef" for c in evidence_sha256))
    return {"scope": "selected_trajectory_terminal_state_only",
            "min_sep": number(measured.get("min_sep")), "pairs": pairs,
            "termination_reason": reason if reason in ("success", "unknown") else None,
            "evidence_sha256": evidence_sha256 if valid_sha else None,
            "blocker": None, "all_paths_impossible": False}


def _scope(context: dict, choice: dict, selected: dict, runtime: Any, planner: Any,
           trajectory: dict, source_hashes: dict, before: dict, rng: str,
           stability_timeout: Optional[float]) -> dict:
    return {
        "task_context": deepcopy(context.get("task_context")),
        "assembled_parts": selected["remaining"], "part_move": choice["part_id"],
        "parts_rest": selected["rest"], "parts_removed": selected["removed"],
        "base_part": selected["base"], "pose": selected["pose"],
        "pose_id": choice["pose_id"], "selected_action": deepcopy(choice),
        "support_policy": {"stability_function": "upstream_MultiPartStabilityPlanner_check_success",
                           "base_part": selected["base"], "selected_support_ids": choice["support_ids"],
                           "actual_parts_fix": selected["fixed"],
                           "actual_parts_move": selected["stability_movers"],
                           "max_grippers": selected["max_grippers"],
                           "base_consumes_gripper": False, "timeout": stability_timeout,
                           "allow_gap": False, "support_search": False},
        "action_schedule": trajectory["controller"]["actions"],
        "physics": {"min_sep": float(planner.min_sep), "force_mag": float(planner.force_mag),
                    "frame_skip": int(planner.frame_skip), "max_time": float(planner.max_time),
                    "timestep": float(planner.sim.options.h),
                    "parts_removed_in_path_simulation": list(planner.parts_removed),
                    "col_th_dict": deepcopy(planner._evidence_col_th),
                    "sdf_cache_sha256": {key: value for key, value in before.items() if key.lower().endswith(".sdf")},
                    "source_and_binary_hashes": source_hashes},
        "solver_fingerprint": hashlib.sha256(_canonical(source_hashes).encode()).hexdigest(),
        "seed": selected["seed"], "budget": context.get("budget", 1),
        "rng_before": rng, "initial_q": trajectory["geometry"]["start_q"],
        "collector_version": "explicit_action_endpoint_v1",
    }


def _run(context: dict, choice: dict, selected: dict, root: Path, assembly: Path,
         output: Path, result: dict, deadline: Optional[float], scene_session=None) -> None:
    from path_evidence import collect_endpoint, observed_collision_thresholds
    from replay_path_extension import readonly_sdf_cache

    remaining = _remaining(deadline)
    if remaining is not None and remaining <= 0:
        result.update(status="UNKNOWN", reason="action_wall_budget_exhausted_before_runtime")
        return
    runtime = _load_runtime(root)
    if scene_session is not None:
        runtime.source_files = dict(runtime.source_files, **_reuse_source_files(root))
    if context.get('stability_diagnostics', False):
        runtime.source_files = dict(runtime.source_files,
            stability_observer_sha256=Path(__file__).with_name('stability_diagnostics.py'))
    source_hashes = {name: _sha(path) for name, path in runtime.source_files.items()}
    inputs_before = _asset_snapshot(assembly)
    result["input_hashes"] = inputs_before
    result["source_hashes"] = source_hashes
    actual_ids = {Path(name).stem for name in inputs_before if name.lower().endswith(".obj")}
    if not set(selected["remaining"]).issubset(actual_ids):
        raise ValueError("selected_parts_missing_from_assembly_objs")
    if context.get("task_context") is not None:
        from task_memory import _validate_context
        task_context = context["task_context"]
        _validate_context(task_context)
        task = task_context["task"]
        mesh_and_config = {name: digest for name, digest in inputs_before.items()
                           if not name.lower().endswith(".sdf")}
        if (task["input_sha256"] != mesh_and_config
                or set(task_context["initial_parts"]) != actual_ids
                or set(selected["remaining"]) | set(selected["removed"]) != actual_ids
                or task["base_part"] != selected["base"]
                or task["solver_settings"].get("seed") != selected["seed"]
                or task["solver_settings"].get('physics_scene_policy', 'fresh_v1')
                    != context.get('physics_scene_policy', 'fresh_v1')
                or task["solver_settings"].get("budget") != context.get("budget", 1)):
            raise ValueError("task_context_does_not_match_actual_input_or_run_conditions")
        result["task_context_input_and_run_conditions_checked"] = True
    pose = None if selected["pose"] is None else runtime.np.array(selected["pose"], dtype=float)
    result["rng_before"] = _rng_digest(runtime.np)
    result["executed"] = {"part_id": choice["part_id"], "pose_id": choice["pose_id"],
                          "pose": selected["pose"], "support_ids": choice["support_ids"],
                          "path_fixed_parts": selected["rest"],
                          "stability_fixed_parts": selected["fixed"],
                          "stability_moving_parts": selected["stability_movers"],
                          "force_actions": [], "skill_id": choice["skill_id"]}
    lease = None

    def construct_path():
        with readonly_sdf_cache(runtime.physics) as cache_io:
            with observed_collision_thresholds(runtime.physics.MultiPartPathPlanner):
                planner = runtime.physics.MultiPartPathPlanner(
                    str(root / "assets"), str(assembly), selected["rest"], choice["part_id"],
                    parts_removed=[], pose=pose, save_sdf=True)
        return planner, cache_io

    try:
        path_started = time.perf_counter()
        if scene_session is None:
            planner, cache_io = construct_path()
            result["path_cache_io"] = cache_io
            result["path_constructor_seconds"] = time.perf_counter() - path_started
        else:
            lease = scene_session.acquire(
                _scene_key(root, assembly, selected, choice, source_hashes, inputs_before, runtime),
                construct_path, lambda: _rng_digest(runtime.np))
            planner = lease.planner
            result['path_scene_reuse'] = lease.audit
            result['path_cache_io'] = [] if lease.audit['cache_hit'] else lease.audit['construction_cache_io']
            result['path_constructor_seconds'] = lease.audit['constructor_seconds']
            result['path_scene_acquire_seconds'] = lease.audit['acquire_seconds']
        result["native_path_timeout_seconds"] = float(planner.max_time)
        remaining = _remaining(deadline)
        if remaining is not None and remaining <= 0:
            result.update(status="UNKNOWN", reason="action_wall_budget_exhausted_after_constructor")
            return
        if remaining is not None:
            planner.max_time = min(planner.max_time, remaining)
        result["effective_path_timeout_seconds"] = float(planner.max_time)
        started = time.perf_counter()
        result["path_check_calls"] += 1
        control_options = {} if lease is None else {'reset_observer': lease.after_reset}
        if choice["skill_id"] == "single_force_v1":
            trajectory = runtime.execute_single(planner, AXES[choice["direction_id"]], **control_options)
        else:
            result["two_leg_controller_calls"] += 1
            trajectory = runtime.execute_two_leg(planner, AXES[choice["direction_id"]],
                                                AXES[choice["second_direction_id"]], SWITCH_STEP, **control_options)
        result["path_seconds"] = time.perf_counter() - started
        result["path_success"] = trajectory["raw_success"]
        result["raw_upstream"]["path_success"] = trajectory["raw_success"]
        result["controller"] = trajectory["controller"]
        result["action_extension"] = {
            "protocol": "explicit_action_physics_v1", "choice": deepcopy(choice),
            "controller": trajectory["controller"],
            "legacy_single_action_replayer_supported": choice["skill_id"] == "single_force_v1",
            "pose_transition_validated": False, "grasp_accessibility_validated": False,
        }
        result["geometry"] = trajectory["geometry"]
        result["path_physics_steps"] = trajectory["controller"]["physics_steps"]
        actual_actions = trajectory["controller"]["actions"][:1]
        if trajectory["controller"]["switch_count"]:
            actual_actions += trajectory["controller"]["actions"][1:]
        result["executed"]["force_actions"] = actual_actions
        result["skill_fully_executed"] = (choice["skill_id"] == "single_force_v1"
                                          or trajectory["controller"]["switch_count"] == 1)
        result["repair_geometry_verified"] = bool(choice["skill_id"] == "turn_force_two_leg_v1"
            and trajectory["controller"]["switch_count"] == 1 and trajectory["geometry"]["noncollinear"])
        trajectory_file = output / "trajectory.json"
        _write_new(trajectory_file, trajectory)
        result["trajectory_file"] = "trajectory.json"
        result["trajectory_sha256"] = _sha(trajectory_file)
        stability_timeout = context.get("stability_timeout")
        remaining = _remaining(deadline)
        if remaining is not None:
            stability_timeout = remaining if stability_timeout is None else min(stability_timeout, remaining)
        scope = _scope(context, choice, selected, runtime, planner, trajectory,
                       source_hashes, inputs_before, result["rng_before"], stability_timeout)
        event_id = context.get("event_id", "explicit:" + hashlib.sha256(_canonical({
            "output_dir": str(output), "choice": choice,
            "remaining_parts": selected["remaining"], "source_hashes": source_hashes,
            "input_hashes": inputs_before,
        }).encode()).hexdigest())
        event = collect_endpoint(planner, trajectory["raw_success"], trajectory["path"], scope,
                                 event_id, "trajectory.json", result["trajectory_sha256"])
        evidence_file = output / "endpoint_evidence.json"
        _write_new(evidence_file, event)
        result["evidence_file"] = "endpoint_evidence.json"
        result["evidence_sha256"] = _sha(evidence_file)
        result["endpoint_observation"] = endpoint_observation(event, result["evidence_sha256"])
        result["scope"] = event["scope"]
        result["geometry_queries"] = event["measurements"]["geometry_queries"]
        records = []
        if context.get("task_context") is not None:
            from constraint_memory import records_from_evidence
            records = records_from_evidence(context["task_context"], [event],
                                           str(evidence_file), result["evidence_sha256"])
            result["constraint_memory_status"] = "MEASURED"
        _write_new(output / "constraint_records.json", records)
        result["constraint_records_file"] = "constraint_records.json"
        result["constraint_record_count"] = len(records)
        if not trajectory["raw_success"]:
            result.update(status="UNKNOWN", reason="selected_path_attempt_not_successful")
            return
        if not selected["stability_movers"]:
            result["stability_success"] = True
            result["stability_mode"] = "all_remaining_rest_explicitly_fixed_no_checker_call"
            # This is a boundary condition, NOT a simulated stability success.
            result["raw_upstream"]["stability_success"] = None
        else:
            remaining = _remaining(deadline)
            if remaining is not None:
                stability_timeout = remaining if context.get("stability_timeout") is None else min(
                    context["stability_timeout"], remaining)
            if stability_timeout is not None and stability_timeout <= 0:
                result.update(status="UNKNOWN", reason="action_wall_budget_exhausted_before_stability")
                return
            started = time.perf_counter()
            with readonly_sdf_cache(runtime.physics) as cache_io:
                checker = runtime.physics.MultiPartStabilityPlanner(
                    str(root / "assets"), str(assembly), selected["fixed"], selected["stability_movers"],
                    pose=pose, save_sdf=True, allow_gap=False)
            result["stability_cache_io"] = cache_io
            # Constructor time is also deducted before entering the gravity test.
            remaining = _remaining(deadline)
            if remaining is not None:
                stability_timeout = remaining if context.get("stability_timeout") is None else min(
                    context["stability_timeout"], remaining)
            result["effective_stability_timeout_seconds"] = stability_timeout
            if stability_timeout is not None and stability_timeout <= 0:
                result.update(status="UNKNOWN", reason="action_wall_budget_exhausted_after_stability_constructor")
                return
            result["stability_check_calls"] += 1
            result["internal_stability_evaluations"] += 1
            if context.get('stability_diagnostics', False):
                from stability_diagnostics import observe_stability
                with observe_stability(runtime.physics) as observation:
                    result['stability_diagnostics'] = observation.data
                    stable, raw_parts = checker.check_success(timeout=stability_timeout)
                    observation.returned(stable, raw_parts)
            else:
                stable, raw_parts = checker.check_success(timeout=stability_timeout)
            if type(stable).__name__ not in ("bool", "bool_"):
                raise ValueError("stability_result_must_be_boolean")
            stable = bool(stable)
            if raw_parts is not None:
                if type(raw_parts) not in (list, tuple):
                    raise ValueError("stability_parts_must_be_list_or_null")
                raw_parts = _ids(list(raw_parts), "upstream_stability_parts")
                if not set(raw_parts).issubset(selected["stability_movers"]):
                    raise ValueError("stability_returned_unknown_part")
            result["stability_success"] = stable
            result["raw_upstream"]["stability_success"] = stable
            result["raw_upstream"]["parts_fall"] = raw_parts
            result["stability_seconds"] = time.perf_counter() - started
            result["stability_mode"] = "one_exact_selected_support_configuration"
        result["feasible"] = result["path_success"] and result["stability_success"]
        result["status"] = "PASS" if result["feasible"] else "UNKNOWN"
        result["reason"] = "selected_conditions_feasible" if result["feasible"] else "selected_stability_attempt_not_successful"
    finally:
        reusable = False
        try:
            result["input_files_unchanged"] = _asset_snapshot(assembly) == inputs_before
            result["source_files_unchanged"] = all(_sha(path) == source_hashes[name]
                                                   for name, path in runtime.source_files.items())
            if not result["input_files_unchanged"] or not result["source_files_unchanged"]:
                result.update(status="ERROR", feasible=False, reason="input_or_solver_changed_during_execution")
            remaining = _remaining(deadline)
            reusable = (result['status'] in ('PASS', 'UNKNOWN')
                        and 'budget_exhausted' not in result.get('reason', '')
                        and (remaining is None or remaining > 0))
        finally:
            if lease is not None:
                lease.finish(reusable)


def execute_action(context: Dict[str, Any], choice: Dict[str, Any], output_dir: Any, scene_session=None) -> dict:
    """Execute the selected part/pose/world-force/supports exactly once.

    Required context: asap_root, assembly_dir, remaining_parts, base_part,
    max_grippers, pose_catalog=[{pose_id,matrix}]. Optional: seed, budget,
    removed_parts, task_context, event_id, stability_timeout,
    remaining_wall_seconds. The caller seeds its run; this function preserves
    RNG progression and records the pre-action RNG digest, rather than reseeding.

    ``support_ids`` excludes a mandatory fixed base. Their count plus ONE moving
    gripper must fit max_grippers. This explicit convention differs from native
    ASAP's internal max_fix check when a mandatory base is present; it is logged.
    All state samples are saved, including failed paths. No paid API is called.
    """
    try:
        selected = validate_action(context, choice)
        policy = context.get('physics_scene_policy', 'fresh_v1')
        if policy not in ('fresh_v1', 'path_scene_reuse_v1'):
            raise ValueError('unknown_physics_scene_policy')
        if (scene_session is None) != (policy == 'fresh_v1'):
            raise ValueError('physics_scene_policy_session_mismatch')
        if scene_session is not None:
            from path_scene_reuse import PathSceneSession
            if not isinstance(scene_session, PathSceneSession):
                raise ValueError('invalid_physics_scene_session')
    except Exception:
        if scene_session is not None:
            scene_session.invalidate('invalid_action_request')
        raise
    context = deepcopy(context)
    if "remaining_wall_seconds" not in context and "time_remaining_seconds" in context:
        context["remaining_wall_seconds"] = _finite(context["time_remaining_seconds"], "time_remaining_seconds")
    from remote_rank_client import checked_path
    root = checked_path(context.get("asap_root", ""), run_only=False)
    assembly = checked_path(context.get("assembly_dir", ""), run_only=False)
    output = checked_path(output_dir, run_only=False)
    if not root.is_dir() or not assembly.is_dir() or not output.parent.is_dir():
        raise ValueError("solver_assembly_and_output_parent_must_exist")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    allowance = context.get("remaining_wall_seconds")
    deadline = None if allowance is None else started + allowance
    result = {
        "schema_version": 1, "protocol": "explicit_action_physics_v1", "status": "ERROR",
        "feasible": False, "path_success": None, "stability_success": None,
        "chosen": deepcopy(choice), "executed": None,
        "action": AXES[choice["direction_id"]][:], "part_move": choice["part_id"],
        "parts_fix": choice["support_ids"][:], "pose": selected["pose"], "grasp": None,
        "legacy_single_action_replayer_supported": choice["skill_id"] == "single_force_v1",
        "backend": "ASAP_RedMax", "api_calls": 0,
        "path_check_calls": 0, "stability_check_calls": 0,
        "two_leg_controller_calls": 0, "internal_stability_evaluations": 0,
        "geometry_queries": 0, "path_physics_steps": 0,
        "geometry_query_count_scope": "endpoint_observer_only_excludes_native_geometry_operations",
        "check_counter_semantics": "dispatch_attempts_including_calls_that_raise",
        "raw_upstream": {"path_success": None, "stability_success": None, "parts_fall": None},
        "endpoint_observation": None,
        "hidden_direction_search_calls": 0, "hidden_pose_search_calls": 0,
        "hidden_support_search_calls": 0, "dfs_state_mutations": 0,
        "blocker": None, "all_paths_impossible": False, "hard_pruning": False,
        "grasp_accessibility_validated": False, "pose_transition_validated": False,
        "removed_part_transport_validated": False,
        "stability_raw_parts_semantics": "upstream_combines_initial_disconnection_and_observed_falling_no_cause_inference",
        "path_abstraction": "all_other_remaining_parts_fixed_as_upstream",
        "support_budget_policy": "moving_gripper_plus_selected_supports_base_exempt",
        "native_base_budget_semantics_matched": selected["base"] is None,
        "constraint_memory_status": "NOT_REQUESTED_NO_TASK_CONTEXT",
        "constraint_record_count": 0, "skill_fully_executed": False,
        "repair_geometry_verified": False,
        "seed": selected["seed"], "rng_reseeded_by_executor": False,
        "point_sampling": "no_explicit_point_sampling_in_executor_upstream_constructor_preserved",
        "deadline_semantics": "soft_phase_checks_and_native_timeouts_constructor_not_interruptible",
        "constructor_save_sdf_argument": True, "xml_load_sdf": True, "xml_save_sdf": False,
    }
    _write_new(output / "request.json", {"context": context, "choice": choice})
    if context.get('stability_diagnostics', False):
        # No checker call (e.g. failed path or explicit all-fixed rest) is null.
        result['stability_diagnostics'] = None
    try:
        _run(context, choice, selected, root, assembly, output, result, deadline, scene_session)
    except Exception as error:
        if scene_session is not None:
            scene_session.invalidate('executor_error')
        result.update(status="ERROR", feasible=False, error_type=type(error).__name__, error=str(error))
    if scene_session is not None:
        result['physics_scene_policy'] = policy
        if 'budget_exhausted' in result.get('reason', ''):
            scene_session.invalidate('action_deadline_expired')
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_new(output / "result.json", result)
    return result
