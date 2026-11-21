"""Freeze cross-checked path-repair evidence; always require fresh physics.

Only selected-plan extensions with a successful independent physics replay can
be registered. This module reads trusted local run artifacts; it performs no
API call, physics execution, arbitrary code generation, or pickle loading.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath

from worker.constraint_memory import load_snapshot as load_constraints, validate_scope, validate_event
from worker.task_memory import (_safe_path, _lexical_path, _validate_context,
                                canonical, digest, _sha)
from worker.replay_path_extension import validate_extension_contract


POLICY = "frozen_scope_references_only"
SKILL = "turn_force_two_leg_v1"
LIFECYCLE = "validated_for_recorded_scope_only"
VALIDATION = {"path": "PASS", "stability": "PASS", "fresh_replay": "PASS", "selected_plan_edge": True}
_CONDITION_KEYS = ("task_context", "assembled_parts", "part_move", "parts_rest", "parts_removed",
                   "base_part", "pose", "support_policy", "physics", "solver_fingerprint",
                   "seed", "budget", "initial_q")
_RECORD_KEYS = {"schema_version", "skill_evidence_id", "skill_id", "controller_sha256",
                "source_run_id", "source_request_id", "source_parent_call_id", "failed_constraint_ids",
                "source_scope", "source_scope_sha256", "validated_schedule", "source_trajectory_sha256",
                "validation", "source_files", "lifecycle", "generalized", "force_new_physics",
                "hard_pruning", "reuse_verdict", "record_sha256"}
_SNAPSHOT_KEYS = {"schema_version", "policy", "records", "record_count", "content_sha256",
                  "hard_pruning", "reuse_verdict", "force_new_physics"}


def _path(value, allow_test_root=False):
    # Translate only the known C-drive representation after lexical exclusion.
    raw = _lexical_path(value)
    normalized = raw.replace("\\", "/")
    if os.name == "nt" and normalized.lower().startswith("/mnt/c/"):
        raw = "C:/" + normalized[7:]
    elif os.name != "nt" and PureWindowsPath(raw).drive.lower() == "c:":
        raw = "/mnt/c/" + normalized[3:]
    return _safe_path(raw, allow_test_root=allow_test_root)


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical(value)
    return value


def _relative(root, relative, allow_test_root=False):
    if (type(relative) is not str or not relative or PureWindowsPath(relative).drive
            or relative.startswith(("/", "\\")) or ":" in relative
            or any(item in ("", ".", "..") for item in relative.replace("\\", "/").split("/"))):
        raise ValueError("skill_provenance_requires_safe_relative_path")
    return _path(root / relative, allow_test_root)


def _file_entry(path):
    return {"path": str(path), "sha256": _sha_file(path)}


def _identity(record):
    return {key: record[key] for key in ("skill_id", "controller_sha256", "source_run_id",
            "source_request_id", "source_parent_call_id", "failed_constraint_ids",
            "source_scope_sha256", "validated_schedule", "source_trajectory_sha256")}


def _validate_record(record, check_sources=False, allow_test_root=False):
    if type(record) is not dict or set(record) != _RECORD_KEYS:
        raise ValueError("invalid_skill_evidence_record_fields")
    if (type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["skill_id"] != SKILL or record["lifecycle"] != LIFECYCLE
            or record["generalized"] is not False or record["force_new_physics"] is not True
            or record["hard_pruning"] is not False or record["reuse_verdict"] is not False
            or record["validation"] != VALIDATION
            or any(not _sha(record[key]) for key in ("controller_sha256", "source_scope_sha256",
                                                     "source_trajectory_sha256", "record_sha256"))):
        raise ValueError("unverified_or_generalized_skill_claim")
    validate_scope(record["source_scope"])
    if (record["source_scope_sha256"] != digest(record["source_scope"])
            or record["validated_schedule"] != record["source_scope"]["action_schedule"]
            or record["controller_sha256"] != record["source_scope"]["physics"]["source_and_binary_hashes"]["controller_sha256"]):
        raise ValueError("skill_scope_schedule_or_controller_mismatch")
    ids = record["failed_constraint_ids"]
    if (type(ids) is not list or not ids or ids != sorted(set(ids))
            or any(type(item) is not str or not item.startswith("constraint:") for item in ids)
            or any(type(record[key]) is not str or not record[key] for key in
                   ("source_run_id", "source_request_id", "source_parent_call_id"))):
        raise ValueError("invalid_skill_source_identity")
    if record["skill_evidence_id"] != "skill-evidence:" + digest(_identity(record)):
        raise ValueError("skill_evidence_ID_mismatch")
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    if record["record_sha256"] != digest(body):
        raise ValueError("skill_record_hash_mismatch")
    files = record["source_files"]
    required = {"result", "parsed_result", "path_outcomes", "constraint_memory", "replay", "tree",
                "context", "decision_audit", "events", "successful_trajectory", "failed_trajectory",
                "failed_evidence", "successful_evidence", "controller_code"}
    if type(files) is not dict or set(files) != required:
        raise ValueError("skill_source_proof_files_incomplete")
    for entry in files.values():
        if type(entry) is not dict or set(entry) != {"path", "sha256"} or not _sha(entry["sha256"]):
            raise ValueError("invalid_skill_provenance_hash")
        path = _path(entry["path"], allow_test_root)
        if check_sources and _sha_file(path) != entry["sha256"]:
            raise ValueError("frozen_skill_source_file_changed")


def _validate_snapshot(snapshot, check_sources=False, allow_test_root=False):
    if (type(snapshot) is not dict or set(snapshot) != _SNAPSHOT_KEYS
            or type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1
            or snapshot["policy"] != POLICY or type(snapshot["records"]) is not list
            or type(snapshot["record_count"]) is not int
            or snapshot["record_count"] != len(snapshot["records"])
            or not snapshot["records"]
            or snapshot["hard_pruning"] is not False or snapshot["reuse_verdict"] is not False
            or snapshot["force_new_physics"] is not True
            or snapshot["content_sha256"] != digest(snapshot["records"])):
        raise ValueError("invalid_frozen_skill_snapshot")
    ids = []
    for record in snapshot["records"]:
        _validate_record(record, check_sources, allow_test_root)
        ids.append(record["skill_evidence_id"])
    if ids != sorted(set(ids)):
        raise ValueError("duplicate_or_unsorted_skill_evidence")


def _same_conditions(first, second):
    return [key for key in _CONDITION_KEYS if canonical(first.get(key)) != canonical(second.get(key))]


def _paths_close(first, second):
    return (type(first) is list and type(second) is list and bool(first)
            and len(first) == len(second)
            and all(type(a) is list and type(b) is list and len(a) == len(b) == 6
                    and all(type(x) in (int, float) and type(y) in (int, float)
                            and math.isfinite(x) and math.isfinite(y)
                            and math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-9)
                            for x, y in zip(a, b)) for a, b in zip(first, second)))


def freeze_validated_skills(run_dir, replay_path, new_output, allow_test_root=False):
    """Cross-check complete local evidence, then exclusively create a snapshot."""
    run_dir, replay_path, new_output = [_path(path, allow_test_root)
                                        for path in (run_dir, replay_path, new_output)]
    if new_output.exists():
        raise FileExistsError("skill_snapshot_output_must_be_new")
    paths = {key: _relative(run_dir, name, allow_test_root) for key, name in {
        "result": "result.json", "parsed_result": "parsed_result.json", "path_outcomes": "path_outcomes.json",
        "constraint_memory": "constraint_memory.json", "context": "context.json",
        "decision_audit": "remote_audit.jsonl", "events": "events.jsonl",
        "controller_code": "implementation/worker/path_repair.py"}.items()}
    paths["replay"] = replay_path
    result, parsed, outcomes, context, replay = [_json(paths[key]) for key in
        ("result", "parsed_result", "path_outcomes", "context", "replay")]
    constraints = load_constraints(paths["constraint_memory"], allow_test_root=allow_test_root)
    _validate_context(context)
    if (result.get("status") != "PASS" or result.get("plan_success") is not True
            or result.get("verifier_backend") != "actual_ASAP_RedMax"
            or result.get("model_decision_fixture") is not False
            or parsed.get("status") != "PASS" or parsed.get("validation_ok") is not True
            or parsed.get("plan_success") is not True or parsed.get("trusted_local_tree") is not True
            or replay.get("status") != "PASS" or replay.get("actual_physics") is not True
            or replay.get("trusted_local_tree") is not True):
        raise ValueError("registration_requires_actual_success_and_selected_plan_replay")
    preserved = ("input_files_unchanged", "sdf_cache_unchanged", "source_and_binary_unchanged",
                 "saved_outcomes_unchanged", "saved_selected_plan_unchanged")
    if not all(replay.get(key) is True for key in preserved):
        raise ValueError("replay_did_not_preserve_inputs_and_source")
    sequence = parsed["raw_stats"]["sequence"]
    if result.get("sequence") != sequence or replay.get("selected_sequence") != sequence:
        raise ValueError("source_and_replay_selected_sequence_mismatch")
    seed = context["task"]["solver_settings"]["seed"]
    assembly_name = context["task"]["assembly_id"].replace("\\", "/").split("/")[-1]
    paths["tree"] = _relative(run_dir, "native/dfs-independent/s{}/{}/tree.pkl".format(seed, assembly_name), allow_test_root)
    hashes = {key: _file_entry(path) for key, path in paths.items()}
    for source, field in (("path_outcomes", "source_outcomes_sha256"),
                          ("parsed_result", "parsed_result_sha256"), ("tree", "tree_file_sha256")):
        if replay.get(field) != hashes[source]["sha256"]:
            raise ValueError("replay_bound_source_hash_mismatch")
    audit = [json.loads(line) for line in paths["decision_audit"].read_text(encoding="utf-8").splitlines() if line]
    events = [json.loads(line) for line in paths["events"].read_text(encoding="utf-8").splitlines() if line]
    run_id = result["run_id"]
    if not any(event.get("kind") == "run_start" and event.get("run_id") == run_id for event in events):
        raise ValueError("source_run_identity_missing")
    evidence_files = {}
    for file in sorted(_relative(run_dir, "paths", allow_test_root).glob("*_evidence.json")):
        file = _path(file, allow_test_root)
        value = _json(file)
        validate_event(value, context)
        if value["event_id"] in evidence_files:
            raise ValueError("duplicate_evidence_event_ID")
        evidence_files[value["event_id"]] = (value, file)
    records = []
    for check in replay.get("replays", []):
        if check.get("selected_plan_edge") is not True:
            continue
        if (check.get("status") != "PASS" or check.get("path_success") is not True
                or check.get("stability_success") is not True or check.get("support_set_equal") is not True
                or check.get("source_binary_controller_hashes_equal") is not True
                or check.get("source_sdf_cache_hashes_equal") is not True or check.get("task_context_equal") is not True):
            raise ValueError("selected_repair_replay_not_fully_verified")
        matches = [item for item in outcomes["outcomes"] if item.get("request_id") == check.get("request_id")]
        if len(matches) != 1:
            raise ValueError("replayed_request_missing_or_duplicate")
        outcome = matches[0]
        scope = outcome["scope"]
        validate_scope(scope, context)
        if not outcome["request_id"].startswith(run_id + ":"):
            raise ValueError("repair_request_not_in_source_run")
        bind = check["plan_binding"]
        index = bind["sequence_index"]
        if type(index) is not int or not 0 <= index < len(sequence):
            raise ValueError("invalid_selected_plan_index")
        parent = [part for part in context["initial_parts"] if part not in sequence[:index]]
        child = [part for part in parent if part != sequence[index]]
        if (bind.get("selected_plan_edge") is not True or bind["parent"] != parent or bind["child"] != child
                or bind["parent_call_id"] != outcome["parent_call_id"]
                or bind["extension_sha256"] != digest(outcome)
                or scope["assembled_parts"] != parent or scope["parts_rest"] != child
                or scope["part_move"] != sequence[index]):
            raise ValueError("repair_not_bound_to_selected_plan_edge")
        trajectory_path = _relative(run_dir, outcome["trajectory_file"], allow_test_root)
        trajectory = _json(trajectory_path)
        validate_extension_contract(outcome, trajectory)
        if _sha_file(trajectory_path) != outcome["trajectory_sha256"]:
            raise ValueError("successful_trajectory_hash_mismatch")
        fresh = check["replayed_trajectory"]
        validate_extension_contract(dict(outcome, controller=fresh["controller"], geometry=fresh["geometry"]), fresh)
        if not _paths_close(trajectory["path"], fresh["path"]):
            raise ValueError("fresh_replay_path_differs_from_selected_trajectory")
        comparison = check["comparison"]
        if (comparison.get("saved_path_hash") != digest(trajectory["path"])
                or comparison.get("replayed_path_hash") != digest(fresh["path"])
                or any(comparison.get(key) is not True for key in ("path_allclose", "path_shape_equal",
                    "start_matches_scoped_initial_state", "end_matches_measured_final_state",
                    "switch_state_present_in_path", "continuous_single_simulation", "noncollinear"))
                or comparison.get("switch_count") != 1):
            raise ValueError("replay_trajectory_comparison_not_verified")
        if scope["physics"]["source_and_binary_hashes"]["controller_sha256"] != hashes["controller_code"]["sha256"]:
            raise ValueError("archived_controller_code_hash_mismatch")
        failed_event, failed_event_file = evidence_files[outcome["failed_evidence_id"]]
        passed_event, passed_event_file = evidence_files[outcome["repaired_evidence_id"]]
        if (failed_event["raw_success"] is not False or passed_event["raw_success"] is not True
                or canonical(passed_event["scope"]) != canonical(scope)
                or _same_conditions(failed_event["scope"], scope)
                or failed_event["scope"]["action_schedule"] != [scope["action_schedule"][0]]):
            raise ValueError("failed_and_repaired_measurement_scope_mismatch")
        for event, relative, expected_sha in ((failed_event, outcome["failed_path"], failed_event["trajectory"]["sha256"]),
                                             (passed_event, outcome["trajectory_file"], outcome["trajectory_sha256"])):
            if event["trajectory"]["relative_path"] != relative or event["trajectory"]["sha256"] != expected_sha:
                raise ValueError("measurement_trajectory_provenance_mismatch")
            if _sha_file(_relative(run_dir, relative, allow_test_root)) != expected_sha:
                raise ValueError("measurement_trajectory_file_changed")
        failure_records = [record for record in constraints["records"]
                           if record["evidence_event_id"] == outcome["failed_evidence_id"]]
        if not failure_records:
            raise ValueError("successful_skill_missing_measured_failure_constraints")
        for constraint in failure_records:
            if (canonical(constraint["evidence"]) != canonical(failed_event)
                    or constraint["source"]["sha256"] != _sha_file(failed_event_file)
                    or _path(constraint["source"]["ref"], allow_test_root) != failed_event_file):
                raise ValueError("failed_constraint_provenance_mismatch")
        ids = sorted(record["record_id"] for record in failure_records)
        accepted = [entry for entry in audit if entry.get("request_id") == outcome["request_id"]
                    and entry.get("outcome") == "accepted" and entry.get("fallback_reason") is None]
        if len(accepted) != 1:
            raise ValueError("source_path_decision_not_uniquely_accepted")
        metadata = accepted[0]["response_metadata"]
        if (metadata.get("llm_live") is not True or metadata.get("attempted") is not True
                or metadata.get("path_skill") != outcome["decision"]
                or not set(outcome["decision"]["used_constraint_ids"]).issubset(ids)):
            raise ValueError("source_skill_decision_not_verified_live_choice")
        files = dict(hashes)
        files.update(successful_trajectory=_file_entry(trajectory_path),
                     failed_trajectory=_file_entry(_relative(run_dir, outcome["failed_path"], allow_test_root)),
                     failed_evidence=_file_entry(failed_event_file), successful_evidence=_file_entry(passed_event_file))
        record = {"schema_version": 1, "skill_id": SKILL, "controller_sha256": hashes["controller_code"]["sha256"],
                  "source_run_id": run_id, "source_request_id": outcome["request_id"],
                  "source_parent_call_id": outcome["parent_call_id"], "failed_constraint_ids": ids,
                  "source_scope": deepcopy(scope), "source_scope_sha256": digest(scope),
                  "validated_schedule": deepcopy(scope["action_schedule"]),
                  "source_trajectory_sha256": outcome["trajectory_sha256"], "validation": deepcopy(VALIDATION),
                  "source_files": files, "lifecycle": LIFECYCLE, "generalized": False,
                  "force_new_physics": True, "hard_pruning": False, "reuse_verdict": False}
        record["skill_evidence_id"] = "skill-evidence:" + digest(_identity(record))
        record["record_sha256"] = digest(record)
        records.append(record)
    records.sort(key=lambda value: value["skill_evidence_id"])
    snapshot = {"schema_version": 1, "policy": POLICY, "records": records, "record_count": len(records),
                "content_sha256": digest(records), "hard_pruning": False, "reuse_verdict": False,
                "force_new_physics": True}
    _validate_snapshot(snapshot, check_sources=True, allow_test_root=allow_test_root)
    with new_output.open("x", encoding="utf-8") as stream:
        stream.write(canonical(snapshot) + "\n")
    return snapshot


def load(path, allow_test_root=False):
    """Verify the frozen snapshot and all referenced source-file hashes."""
    snapshot = _json(_path(path, allow_test_root))
    _validate_snapshot(snapshot, check_sources=True, allow_test_root=allow_test_root)
    return snapshot


def matching_cards(snapshot, current):
    """Return scoped historical references; never reuse an old physical verdict.

    current may be the path-choice features, a complete measured scope, or a
    task context. Incomplete conditions are labelled, never treated as exact.
    Cross-assembly transfer is deliberately not implemented in this library.
    """
    _validate_snapshot(snapshot)
    if type(current) is not dict:
        raise ValueError("current_skill_context_must_be_object")
    scope, expected_controller, moving = None, None, None
    if "memory_context" in current:
        context = current["memory_context"]
        moving = current.get("moving_part_id")
        expected_controller = current.get("registered_skill", {}).get("code_sha256")
        candidates = [ref["source_scope"] for ref in current.get("constraint_references", [])
                      if ref.get("origin") == "current_episode_measurement"]
        if candidates:
            if any(canonical(value) != canonical(candidates[0]) for value in candidates):
                raise ValueError("conflicting_current_failure_scopes")
            scope = candidates[0]
    elif "task_context" in current:
        scope, context = current, current["task_context"]
        moving = current.get("part_move")
        expected_controller = current.get("physics", {}).get("source_and_binary_hashes", {}).get("controller_sha256")
    else:
        context = current
    _validate_context(context)
    if scope is not None:
        validate_scope(scope, context)
        if moving is not None and scope["part_move"] != moving:
            raise ValueError("current_skill_moving_part_mismatch")
    cards = []
    for record in snapshot["records"]:
        source = record["source_scope"]
        # Matching source IDs is legitimate only within an identical CAD input.
        identity = ("assembly_id", "input_sha256", "source_commit", "goal")
        if any(canonical(source["task_context"]["task"][key]) != canonical(context["task"][key]) for key in identity):
            continue
        if moving is not None and source["part_move"] != moving:
            continue
        if expected_controller is not None and expected_controller != record["controller_sha256"]:
            continue
        mismatches = ["current_scope_not_supplied"] if scope is None else _same_conditions(source, scope)
        if expected_controller is None:
            mismatches.append("current_controller_not_supplied")
        observed_schedule = None if scope is None else deepcopy(scope["action_schedule"])
        cards.append({key: deepcopy(record[key]) for key in (
            "skill_evidence_id", "skill_id", "controller_sha256", "source_run_id", "source_request_id",
            "failed_constraint_ids", "source_scope", "source_scope_sha256", "validated_schedule",
            "source_trajectory_sha256", "validation", "lifecycle", "generalized", "force_new_physics",
            "hard_pruning", "reuse_verdict")})
        cards[-1].update(reference_only=True, current_attempt_verified=False,
            applicability="source_conditions_reference" if mismatches else "same_task_state_conditions_reference",
            condition_mismatches=mismatches, current_observed_action_schedule=observed_schedule,
            action_schedule_matches_current_observation=observed_schedule == record["validated_schedule"])
    return cards
