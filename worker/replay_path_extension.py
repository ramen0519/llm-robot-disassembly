"""Re-simulate saved two-force path extensions, then upstream stability (Py3.8).

This is an actual-physics replay command, not a renderer or unit-test fixture.
It never uses the legacy single-action replayer and never changes saved runs.
Only a new --output JSON is written. Existing SDF caches are loaded but never
saved; mesh, pose, physical constants, source, binary and controller are checked.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ATOL = 1e-9
RTOL = 1e-7
EXPECTED_SOLVER_KEYS = {"redmax_binary_sha256", "physics_source_sha256",
                        "sim_string_sha256", "controller_sha256", "observer_sha256"}
SDF_BUILDERS = ("get_contact_sim_string", "get_path_sim_string", "get_stability_sim_string")


@contextmanager
def readonly_sdf_cache(physics_module):
    """Keep original cache loading while disabling only XML save_sdf flags.

    The upstream save_sdf argument controls both loading and saving. Setting it
    False rebuilds SDFs and can change numerical contact thresholds. Callers of
    this context pass save_sdf=True; these serial, process-local wrappers retain
    load_sdf="true" and change only save_sdf="true" to "false". No source file,
    force, contact, ground, solver, or stability parameter is changed.
    """
    originals = {name: getattr(physics_module, name) for name in SDF_BUILDERS}
    if any(not callable(builder) or getattr(builder, "_readonly_sdf_wrapper", False)
           for builder in originals.values()):
        raise ValueError("invalid_or_nested_readonly_sdf_builder")
    audit = []

    def wrapper_for(name, builder):
        @wraps(builder)
        def wrapped(*args, **kwargs):
            original_xml = builder(*args, **kwargs)
            if type(original_xml) is not str:
                raise ValueError("simulation_builder_did_not_return_xml")
            original_tree = ET.fromstring(original_xml)
            bodies = [body for body in original_tree.iter("body")
                      if body.get("type") in ("SDF", "MultiSDF")]
            if (not bodies or any(body.get("load_sdf") != "true" or body.get("save_sdf") != "true"
                                  for body in bodies)):
                raise ValueError("readonly_replay_requires_original_cache_loading_enabled")
            # These exact strings are emitted by the pinned upstream builder.
            # Preserve all XML byte-for-byte except the save flag itself.
            modified_xml = original_xml.replace('save_sdf="true"', 'save_sdf="false"')
            modified_xml = modified_xml.replace("save_sdf='true'", "save_sdf='false'")
            modified_tree = ET.fromstring(modified_xml)
            modified_bodies = [body for body in modified_tree.iter("body")
                               if body.get("type") in ("SDF", "MultiSDF")]
            if (len(modified_bodies) != len(bodies)
                    or any(body.get("load_sdf") != "true" or body.get("save_sdf") != "false"
                           for body in modified_bodies)):
                raise ValueError("readonly_cache_xml_conversion_failed")
            for old, new in zip(bodies, modified_bodies):
                old_attributes, new_attributes = dict(old.attrib), dict(new.attrib)
                old_attributes.pop("save_sdf")
                new_attributes.pop("save_sdf")
                if old_attributes != new_attributes:
                    raise ValueError("readonly_cache_changed_physics_attributes")
            audit.append({"builder": name, "sdf_body_count": len(bodies),
                          "xml_load_sdf": True, "xml_save_sdf": False,
                          "only_save_flag_changed": True,
                          "original_xml_sha256": hashlib.sha256(original_xml.encode()).hexdigest(),
                          "readonly_xml_sha256": hashlib.sha256(modified_xml.encode()).hexdigest()})
            return modified_xml
        wrapped._readonly_sdf_wrapper = True
        return wrapped

    try:
        for name, builder in originals.items():
            setattr(physics_module, name, wrapper_for(name, builder))
        yield audit
    finally:
        for name, builder in originals.items():
            setattr(physics_module, name, builder)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def read_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"),
                        object_pairs_hook=_unique_object,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    canonical(result)
    return result


def relative_trajectory(run_dir, relative):
    """Validate lexical paths before even constructing or opening a target."""
    from remote_rank_client import checked_path
    if not isinstance(relative, str) or not relative:
        raise ValueError("trajectory_file_required")
    normalized = relative.replace("\\", "/")
    if (PureWindowsPath(relative).drive or normalized.startswith("/")
            or ":" in normalized or "\x00" in normalized
            or any(part in ("", ".", "..") for part in normalized.split("/"))):
        raise ValueError("trajectory_must_be_relative_without_traversal")
    return checked_path(Path(run_dir) / normalized)


def _axis(value):
    if (type(value) is not list or len(value) != 3
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)
            or sum(abs(item) for item in value) != 1
            or any(item not in (-1, 0, 1) for item in value)):
        raise ValueError("saved_action_must_be_unit_axis")
    return value


def validate_extension_contract(outcome, saved):
    """Pure contract validation.  No inference of physics from these booleans."""
    if type(outcome) is not dict or type(saved) is not dict:
        raise ValueError("outcome_and_trajectory_must_be_objects")
    if (outcome.get("feasible") is not True or outcome.get("path_success") is not True
            or outcome.get("stability_success") is not True or saved.get("raw_success") is not True):
        raise ValueError("replay_requires_recorded_feasible_extension")
    controller = saved.get("controller")
    if type(controller) is not dict or controller != outcome.get("controller"):
        raise ValueError("saved_controller_differs_from_outcome")
    if (controller.get("skill_id") != "turn_force_two_leg_v1"
            or type(controller.get("version")) is not int or controller["version"] != 1
            or type(controller.get("reset_count")) is not int or controller["reset_count"] != 1
            or type(controller.get("switch_count")) is not int or controller["switch_count"] != 1
            or controller.get("intermediate_resets") != 0):
        raise ValueError("unsupported_or_incomplete_two_leg_controller")
    actions = controller.get("actions")
    if (type(actions) is not list or len(actions) != 2
            or any(type(item) is not dict or set(item) != {"step", "action"} for item in actions)
            or type(actions[0]["step"]) is not int or actions[0]["step"] != 0
            or type(actions[1]["step"]) is not int or not 1 <= actions[1]["step"] <= 1000000):
        raise ValueError("invalid_two_leg_schedule")
    first, second = _axis(actions[0]["action"]), _axis(actions[1]["action"])
    if sum(a * b for a, b in zip(first, second)) != 0:
        raise ValueError("saved_directions_must_be_perpendicular")
    scope = outcome.get("scope", {})
    if scope.get("action_schedule") != actions:
        raise ValueError("scope_schedule_differs_from_trajectory")
    if (type(controller.get("physics_steps")) is not int
            or controller["physics_steps"] <= actions[1]["step"]
            or controller.get("switch_q") is None):
        raise ValueError("recorded_switch_not_executed")
    if saved.get("geometry", {}).get("noncollinear") is not True:
        raise ValueError("recorded_trajectory_not_noncollinear")
    if outcome.get("geometry") != saved.get("geometry"):
        raise ValueError("saved_geometry_differs_from_outcome")
    if outcome.get("legacy_single_action_replayer_supported") is not False:
        raise ValueError("legacy_replayer_contract_missing")
    return first, second, actions[1]["step"]


def bind_selected_plan(outcomes, normalized_tree, sequence, initial_parts):
    """Bind manifest outcomes to the actual chosen edges, not other successes."""
    if (type(sequence) is not list or len(sequence) != len(initial_parts) - 1
            or len(sequence) != len(set(sequence)) or not set(sequence).issubset(initial_parts)):
        raise ValueError("invalid_selected_sequence")
    edges = {(tuple(parent), tuple(child)): attributes
             for parent, child, attributes in normalized_tree["edges"]}
    selected, parent = [], tuple(initial_parts)
    matched = set()
    for index, part in enumerate(sequence):
        child = tuple(item for item in parent if item != part)
        raw = edges.get((parent, child), {}).get("sim_info", {})
        if raw.get("feasible") is not True or raw.get("part_move") != part:
            raise ValueError("selected_sequence_edge_not_feasible")
        extension = raw.get("path_extension")
        if extension is not None:
            matches = [i for i, candidate in enumerate(outcomes)
                       if canonical(candidate) == canonical(extension)]
            if len(matches) != 1 or matches[0] in matched:
                raise ValueError("selected_tree_extension_missing_duplicate_or_mismatched_in_manifest")
            candidate = outcomes[matches[0]]
            if (candidate.get("feasible") is not True or not candidate.get("parent_call_id")
                    or not candidate.get("request_id") or not candidate.get("trajectory_sha256")
                    or candidate.get("scope", {}).get("assembled_parts") != list(parent)
                    or candidate.get("scope", {}).get("parts_rest") != list(child)
                    or candidate.get("scope", {}).get("part_move") != part
                    or candidate.get("parts_fix") != raw.get("parts_fix")):
                raise ValueError("selected_tree_extension_scope_or_identity_mismatch")
            matched.add(matches[0])
            selected.append({"outcome": candidate, "selected_plan_edge": True,
                             "sequence_index": index, "parent": list(parent), "child": list(child),
                             "parent_call_id": candidate["parent_call_id"],
                             "extension_sha256": hashlib.sha256(canonical(extension).encode()).hexdigest()})
        parent = child
    ignored = [{"request_id": candidate.get("request_id"), "parent_call_id": candidate.get("parent_call_id"),
                "selected_plan_edge": False, "status": "NOT_SELECTED_NOT_REPLAYED"}
               for i, candidate in enumerate(outcomes)
               if candidate.get("feasible") is True and i not in matched]
    if not selected:
        raise ValueError("no_path_extension_in_selected_plan")
    return selected, ignored


def compare_trajectories(saved, replayed, initial_q):
    """Compare actual arrays and endpoints; caller determines actual provenance."""
    import numpy as np
    first = np.asarray(saved["path"], dtype=float)
    second = np.asarray(replayed["path"], dtype=float)
    if (first.ndim != 2 or second.ndim != 2 or first.shape[1:] != (6,)
            or second.shape[1:] != (6,) or not len(first) or not len(second)
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("saved_or_replayed_path_invalid")
    shape_match = first.shape == second.shape
    close = bool(shape_match and np.allclose(first, second, rtol=RTOL, atol=ATOL))
    saved_hash = hashlib.sha256(canonical(saved["path"]).encode()).hexdigest()
    replay_hash = hashlib.sha256(canonical(replayed["path"]).encode()).hexdigest()
    expected_start = np.asarray(initial_q, dtype=float)
    controller = replayed["controller"]
    switch_q = np.asarray(controller["switch_q"], dtype=float) if controller["switch_q"] is not None else None
    start_matches = bool(expected_start.shape == (6,) and
                         np.allclose(second[0], expected_start, rtol=RTOL, atol=ATOL) and
                         np.allclose(second[0], replayed["geometry"]["start_q"], rtol=RTOL, atol=ATOL))
    endpoint_matches = bool(np.allclose(second[-1], replayed["geometry"]["end_q"], rtol=RTOL, atol=ATOL))
    switch_in_path = bool(switch_q is not None and switch_q.shape == (6,) and
                         np.any(np.all(np.isclose(second, switch_q, rtol=RTOL, atol=ATOL), axis=1)))
    jumps = np.linalg.norm(np.diff(second[:, :3], axis=0), axis=1)
    return {"saved_path_hash": saved_hash, "replayed_path_hash": replay_hash,
            "path_hash_equal": saved_hash == replay_hash, "path_shape_equal": shape_match,
            "path_allclose": close, "rtol": RTOL, "atol": ATOL,
            "maximum_absolute_path_difference": float(np.abs(first - second).max()) if shape_match else None,
            "start_matches_scoped_initial_state": start_matches,
            "end_matches_measured_final_state": endpoint_matches,
            "switch_state_present_in_path": switch_in_path,
            "maximum_recorded_translation_step": float(jumps.max()) if len(jumps) else 0.,
            "continuous_single_simulation": bool(controller["reset_count"] == 1 and
                                                 controller["intermediate_resets"] == 0),
            "switch_count": controller["switch_count"],
            "noncollinear": replayed["geometry"]["noncollinear"]}


def _solver_hashes():
    import redmax_py
    import plan_sequence.physics_planner as physics
    import plan_sequence.sim_string as sim_string
    return {"redmax_binary_sha256": sha(redmax_py.__file__),
            "physics_source_sha256": sha(physics.__file__),
            "sim_string_sha256": sha(sim_string.__file__),
            "controller_sha256": sha(Path(__file__).with_name("path_repair.py")),
            "observer_sha256": sha(Path(__file__).with_name("path_evidence.py"))}


def replay_all(root, assembly, run_dir, trusted_local_tree=False):
    from remote_rank_client import checked_path
    from independent_features import describe_task, _source_hashes
    from task_memory import task_context
    from constraint_memory import validate_scope

    if trusted_local_tree is not True:
        raise ValueError("explicit_trusted_local_tree_required")
    root, assembly = checked_path(root, run_only=False), checked_path(assembly, run_only=False)
    run_dir = checked_path(run_dir)
    outcome_file = checked_path(run_dir / "path_outcomes.json")
    manifest = read_json(outcome_file)
    outcomes = manifest.get("outcomes")
    if type(outcomes) is not list:
        raise ValueError("missing_path_outcomes")
    feasible = [item for item in outcomes if isinstance(item, dict) and item.get("feasible") is True]
    if not feasible:
        raise ValueError("no_feasible_path_extensions_to_replay")
    parsed_file = checked_path(run_dir / "parsed_result.json")
    parsed = read_json(parsed_file)
    if parsed.get("status") != "PASS" or parsed.get("plan_success") is not True:
        raise ValueError("selected_plan_must_have_passed_artifact_validation")
    seed = feasible[0]["scope"]["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("invalid_saved_seed")
    native = checked_path(run_dir / "native" / "dfs-independent" / ("s" + str(seed)) / assembly.name)
    checked_path(native / "tree.pkl")
    from compare_rank_hook import tree_data
    from validate_assembly import validate_stats, tree_fingerprint
    normalized_tree = tree_data(native)  # Explicit trust covers only our own local run.
    stats = read_json(checked_path(native / "stats.json"))
    initial_context = feasible[0]["scope"]["task_context"]
    validation = validate_stats(stats, initial_context["initial_parts"],
                                initial_context["task"]["base_part"],
                                initial_context["task"]["solver_settings"]["max_grippers"], normalized_tree)
    if (validation["status"] != "PASS" or stats != parsed["raw_stats"]
            or tree_fingerprint(normalized_tree) != parsed["tree_fingerprint_sha256"]):
        raise ValueError("selected_tree_stats_or_fingerprint_differ_from_validation")
    selected, ignored = bind_selected_plan(outcomes, normalized_tree, stats["sequence"],
                                          initial_context["initial_parts"])
    report = {"schema_version": 1, "status": "RUNNING", "actual_physics": True,
              "api_calls": 0, "source_outcomes_sha256": sha(outcome_file),
              "feasible_extension_count": len(feasible), "replays": [],
              "selected_plan_extension_count": len(selected), "nonselected_feasible_outcomes": ignored,
              "trusted_local_tree": True, "selected_sequence": stats["sequence"],
              "tree_file_sha256": sha(native / "tree.pkl"), "parsed_result_sha256": sha(parsed_file),
              "legacy_single_action_replayer_supported": False,
              "replayer": "complete_saved_force_schedule",
              "constructor_save_sdf_argument": True, "xml_load_sdf": True, "xml_save_sdf": False,
              "cache_policy": "load_existing_SDF_disable_only_XML_save_flag",
              "performance_comparison": "NOT_RUN"}
    sys.path.insert(0, str(root))
    import numpy as np
    import plan_sequence.physics_planner as physics_module
    from plan_sequence.physics_planner import MultiPartPathPlanner
    from plan_sequence.feasibility_check import get_stable_plan_1pose_serial
    from path_repair import execute_two_leg
    from path_evidence import observed_collision_thresholds

    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--"],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    actual_solver = _solver_hashes()
    inputs_before = _source_hashes(assembly)
    def cache_hashes():
        return {path.name: sha(checked_path(path, run_only=False))
                for path in sorted(assembly.iterdir()) if path.suffix.lower() == ".sdf"}
    cache_before = cache_hashes()
    for binding in selected:
        outcome = binding["outcome"]
        entry = {"request_id": outcome.get("request_id"), "status": "ERROR",
                 "path_check_calls": 0, "stability_check_calls": 0,
                 "plan_binding": {key: value for key, value in binding.items() if key != "outcome"},
                 "selected_plan_edge": True}
        report["replays"].append(entry)
        try:
            scope = outcome["scope"]
            validate_scope(scope)
            expected_context = scope["task_context"]
            actual_context = task_context(describe_task(root, assembly, scope["base_part"],
                                         expected_context["task"]["solver_settings"]))
            if canonical(actual_context) != canonical(expected_context):
                raise ValueError("replay_task_mesh_pose_settings_or_geometry_mismatch")
            entry["task_context_equal"] = True
            expected_solver = scope["physics"]["source_and_binary_hashes"]
            if set(expected_solver) != EXPECTED_SOLVER_KEYS or actual_solver != expected_solver:
                raise ValueError("replay_source_binary_controller_or_observer_hash_mismatch")
            fingerprint = hashlib.sha256(json.dumps(actual_solver, sort_keys=True).encode()).hexdigest()
            if fingerprint != scope["solver_fingerprint"]:
                raise ValueError("replay_solver_fingerprint_mismatch")
            entry["source_binary_controller_hashes_equal"] = True
            if scope["physics"].get("sdf_cache_sha256") != cache_before:
                raise ValueError("source_SDF_cache_hash_mismatch")
            entry["source_sdf_cache_hashes_equal"] = True
            trajectory_file = relative_trajectory(run_dir, outcome["trajectory_file"])
            if sha(trajectory_file) != outcome["trajectory_sha256"]:
                raise ValueError("saved_trajectory_file_hash_mismatch")
            saved = read_json(trajectory_file)
            first, second, switch_step = validate_extension_contract(outcome, saved)
            support = scope["support_policy"]
            if not isinstance(support, dict) or "timeout" not in support:
                raise ValueError("scope_missing_actual_stability_timeout")
            if support.get("stability_function") != "upstream_get_stable_plan_1pose_serial":
                raise ValueError("unsupported_stability_policy")
            pose = None if scope["pose"] is None else np.asarray(scope["pose"], dtype=float)
            physics = scope["physics"]
            with readonly_sdf_cache(physics_module) as path_cache_io:
                with observed_collision_thresholds(MultiPartPathPlanner):
                    planner = MultiPartPathPlanner(str(root / "assets"), str(assembly), scope["parts_rest"],
                        scope["part_move"], parts_removed=physics["parts_removed_in_path_simulation"],
                        pose=pose, force_mag=physics["force_mag"], save_sdf=True)
            entry["path_cache_io"] = path_cache_io
            entry["replayed_collision_thresholds"] = planner._evidence_col_th
            if (planner.frame_skip != physics["frame_skip"] or planner.max_time != physics["max_time"]
                    or planner.min_sep != physics["min_sep"]
                    or planner._evidence_col_th != physics["col_th_dict"]):
                raise ValueError("replay_physics_or_collision_threshold_mismatch")
            started = time.perf_counter()
            entry["path_check_calls"] += 1
            replayed = execute_two_leg(planner, first, second, switch_step,
                                      min_sep=saved["controller"]["min_sep_argument"])
            entry["path_seconds"] = time.perf_counter() - started
            entry["path_success"] = replayed["raw_success"]
            entry["comparison"] = compare_trajectories(saved, replayed, scope["initial_q"])
            entry["replayed_trajectory"] = replayed
            max_grippers = support["max_grippers"]
            if max_grippers is not None and (type(max_grippers) is not int or max_grippers <= 0):
                raise ValueError("invalid_saved_gripper_budget")
            if replayed["raw_success"]:
                started = time.perf_counter()
                entry["stability_check_calls"] += 1
                with readonly_sdf_cache(physics_module) as stability_cache_io:
                    parts_fix, evaluations = get_stable_plan_1pose_serial(
                        str(root / "assets"), str(assembly), scope["parts_rest"], scope["base_part"],
                        pose, max_fix=None if max_grippers is None else max_grippers - 1,
                        save_sdf=True, timeout=support["timeout"], allow_gap=support.get("allow_gap", False),
                        debug=0, render=False, return_count=True)
                entry["stability_cache_io"] = stability_cache_io
                entry["stability_seconds"] = time.perf_counter() - started
                entry["internal_stability_evaluations"] = evaluations
                entry["stability_success"] = parts_fix is not None
                entry["replayed_parts_fix"] = parts_fix
                entry["support_set_equal"] = parts_fix == outcome["parts_fix"]
            else:
                entry.update(stability_success=False, support_set_equal=False,
                             internal_stability_evaluations=0)
            checks = entry["comparison"]
            required = ("path_allclose", "start_matches_scoped_initial_state",
                        "end_matches_measured_final_state", "switch_state_present_in_path",
                        "continuous_single_simulation", "noncollinear")
            passed = (entry["path_success"] and entry["stability_success"] and
                      entry["support_set_equal"] and checks["switch_count"] == 1 and
                      all(checks[key] for key in required))
            entry["status"] = "PASS" if passed else "FAIL"
        except Exception as error:
            entry["error_type"] = type(error).__name__
            entry["error"] = str(error)
    report["input_files_unchanged"] = _source_hashes(assembly) == inputs_before
    report["sdf_cache_unchanged"] = cache_hashes() == cache_before
    report["source_and_binary_unchanged"] = _solver_hashes() == actual_solver
    report["saved_outcomes_unchanged"] = sha(outcome_file) == report["source_outcomes_sha256"]
    report["saved_selected_plan_unchanged"] = (sha(native / "tree.pkl") == report["tree_file_sha256"]
                                              and sha(parsed_file) == report["parsed_result_sha256"])
    report["status"] = "PASS" if (all(item["status"] == "PASS" for item in report["replays"])
        and report["input_files_unchanged"] and report["source_and_binary_unchanged"]
        and report["saved_outcomes_unchanged"] and report["sdf_cache_unchanged"]
        and report["saved_selected_plan_unchanged"]) else "FAIL"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root", required=True, type=Path)
    parser.add_argument("--assembly-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trusted-local-tree", action="store_true",
                        help="Required: only load tree.pkl generated by your own local run")
    args = parser.parse_args()
    from remote_rank_client import checked_path
    output = checked_path(args.output)
    if output.exists():
        raise FileExistsError("replay_output_must_be_new")
    if not output.parent.is_dir():
        raise ValueError("replay_output_parent_must_exist")
    started = time.perf_counter()
    try:
        report = replay_all(args.asap_root, args.assembly_dir, args.run_dir, args.trusted_local_tree)
    except Exception as error:
        report = {"schema_version": 1, "status": "ERROR", "api_calls": 0,
                  "error_type": type(error).__name__, "error": str(error),
                  "legacy_single_action_replayer_supported": False}
    report["elapsed_seconds"] = time.perf_counter() - started
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "output": str(output), "api_calls": 0}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
