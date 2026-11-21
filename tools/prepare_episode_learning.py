"""Select development experience from a completed, frozen actual ASAP episode.

No API or physics runs here. The oldest eligible failed attempt is selected
before seeing any new candidate. Admission specifications are copied unchanged;
their outcomes are not supplied as synthesis evidence. Stored observations are
conditional evidence, never a source-to-target part mapping or hard constraint.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tools"))

from run_integrated_repair import checked
from run_explicit_action_experiment import audit_report
from orchestrator.explicit_action_memory import FrozenActionMemory
from orchestrator.skill_case import validate_trigger_scope
from orchestrator.skill_validation import _actual_attempt
from worker.explicit_action_contract import action_scope, validate_request, validate_response
from worker.task_memory import _safe_path, digest

LOCAL_SOURCES = {
    "explicit_executor_sha256": "worker/explicit_action_physics.py",
    "controller_sha256": "worker/path_repair.py",
    "observer_sha256": "worker/path_evidence.py",
    "stability_observer_sha256": "worker/stability_diagnostics.py",
}
BYTE_LIMIT = 32 * 1024 * 1024


def _path(value, *, allow_test_root=False, runs_only=True):
    if allow_test_root:
        try:
            return _safe_path(value, True)
        except ValueError:
            # The exception does not permit traversal/non-C paths: checked
            # independently rejects them before any filesystem access.
            return checked(value, runs_only)
    return checked(value, runs_only)


def _load(path):
    with path.open("rb") as stream:
        raw = stream.read(BYTE_LIMIT + 1)
    if len(raw) > BYTE_LIMIT:
        raise ValueError("episode_artifact_byte_limit")
    value = json.loads(raw.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError("episode_artifact_object_required")
    return value


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _project_reference(path, allow_test_root):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        if allow_test_root:
            return str(path)
        raise ValueError("episode_report_outside_project")


def prepare_episode_learning(episode_dir, admission_manifest_path, output_dir, *,
                             allow_test_root=False):
    """Validate saved evidence, then create one new manifest or explicit stop.

    ``allow_test_root`` is only a Python test seam and is never exposed by CLI.
    The source episode may have failed to finish its plan; its recorded physical
    attempts and frozen memory still have to pass every integrity check.
    """
    def safe(value, runs_only=True):
        return _path(value, allow_test_root=allow_test_root, runs_only=runs_only)

    episode, output = safe(episode_dir), safe(output_dir)
    template_path = safe(admission_manifest_path, False)
    if output.exists():
        raise FileExistsError("new_learning_output_required")
    originals = {}

    def load(path, runs_only=True):
        path = safe(path, runs_only)
        value = _load(path)
        originals[str(path)] = _sha(path)
        return value

    worker_path = episode / "worker_result.json"
    report_path = episode / "planner/planner_report.json"
    worker, planner = load(worker_path), load(report_path)
    if any(worker.get(key) != value for key, value in planner.items()):
        raise ValueError("worker_planner_report_mismatch")
    if worker.get("actual_physics") is not True or worker.get("api_in_worker") is not False:
        raise ValueError("actual_asap_full_episode_required")
    calculated = audit_report(worker)
    stored_audit = load(episode / "audit.json")
    if (calculated.get("validation_ok") is not True
            or stored_audit.get("validation_ok") is not True
            or stored_audit.get("errors") != []
            or any(key not in calculated or calculated[key] != value
                   for key, value in stored_audit.items())):
        raise ValueError("episode_audit_failed_or_mismatched")
    host = load(episode / "host_preservation.json")
    preservation = load(episode / "preservation.json")
    if (host.get("source_unchanged") is not True or host.get("memory_unchanged") is not True
            or preservation.get("input_unchanged") is not True
            or preservation.get("memory_unchanged") is not True):
        raise ValueError("episode_source_input_or_memory_not_preserved")
    inputs = load(episode / "input_before.json")
    if preservation.get("after") != inputs:
        raise ValueError("episode_input_hashes_changed")
    source_hashes = load(episode / "source_before.json")
    if not source_hashes:
        raise ValueError("missing_preserved_source_hashes")
    # Verify the archived source bytes, not today's edited source tree.
    for relative, expected in source_hashes.items():
        archived = safe(episode / "implementation" / relative)
        if _sha(archived) != expected:
            raise ValueError("archived_episode_source_hash_mismatch")
        originals[str(archived)] = expected

    memory_path = episode / "action_memory_frozen.json"
    snapshot = FrozenActionMemory(load(memory_path)).snapshot
    records = snapshot["records"]
    attempts, decisions = worker["attempts"], worker["decisions"]
    if len(records) != len(attempts):
        raise ValueError("episode_memory_attempt_count_mismatch")
    by_request = {d["request"]["request_id"]: d for d in decisions}
    memory_by_request = {r["source_request_id"]: r for r in records}
    if len(memory_by_request) != len(records):
        raise ValueError("duplicate_memory_request")
    task_ids, considered, selected = set(), [], None
    history_sources = []
    for index, attempt in enumerate(attempts, 1):
        decision = by_request[attempt["request_id"]]
        request = validate_request(decision["request"])
        response = validate_response(request, decision["response"])
        task = request["task_context"]["task"]
        task_ids.add(task["assembly_id"])
        if request["run_id"] != worker["run_id"]:
            raise ValueError("episode_request_run_mismatch")
        if any(inputs.get(k) != v for k, v in task["input_sha256"].items()):
            raise ValueError("episode_task_input_hash_mismatch")
        record = memory_by_request.get(attempt["request_id"])
        if (record is None or record["source_run_id"] != request["run_id"]
                or record["source_state_id"] != request["state_id"]
                or record["source_action_id"] != response["action_id"]
                or record["choice"] != response["choice"]
                or record["scope"] != action_scope(request, response["choice"])):
            raise ValueError("episode_memory_decision_binding_mismatch")
        observed = record["result"]
        if (observed["raw"] != attempt["raw"] or observed["status"] != attempt["status"]
                or observed["path_success"] is not attempt["raw"].get("path_success")
                or observed["stability_success"] is not attempt["raw"].get("stability_success")
                or safe(ROOT / observed["evidence_ref"]) != safe(report_path)):
            raise ValueError("episode_memory_raw_binding_mismatch")
        directory = safe(episode / "planner" / ("attempt_%06d" % index))
        physical, raw = load(directory / "request.json"), load(directory / "result.json")
        if raw != attempt["raw"] or physical.get("choice") != attempt["choice"]:
            raise ValueError("episode_raw_file_binding_mismatch")
        context = physical.get("context", {})
        if (context.get("task_context") != request["task_context"]
                or context.get("request_id") != request["request_id"]
                or context.get("run_id") != request["run_id"]
                or context.get("state_id") != request["state_id"]
                or context.get("action_id") != response["action_id"]
                or context.get("remaining_parts") != request["remaining_parts"]):
            raise ValueError("episode_physical_request_binding_mismatch")
        item = {"attempt_index": index, "event_id": attempt["event_id"],
                "memory_evidence_id": record["evidence_id"],
                "memory_record_sha256": record["record_sha256"],
                "status": attempt["status"], "eligible": False,
                "selected": False, "reason": None,
                "raw_result_path": str(directory / "result.json"),
                "raw_result_sha256": originals[str(directory / "result.json")]}
        if raw.get("backend") != "ASAP_RedMax":
            item["reason"] = "not_actual_asap_observation"
        elif observed["verifier_origin"] != "actual_ASAP_RedMax":
            raise ValueError("episode_memory_verifier_origin_mismatch")
        elif raw.get("path_check_calls") != 1 or type(raw.get("path_success")) is not bool:
            item["reason"] = "no_known_physical_path_outcome"
        else:
            validate_trigger_scope(request, raw, physical["choice"])
            checked_attempt = _actual_attempt(directory, {
                "input_hashes": raw["input_hashes"], "source_hashes": raw["source_hashes"]},
                allow_test_root, False)
            for ref in checked_attempt["refs"]:
                originals[ref["path"]] = ref["sha256"]
            if any(inputs.get(k) != v for k, v in raw["input_hashes"].items()):
                raise ValueError("episode_actual_input_hash_mismatch")
            for key, relative in LOCAL_SOURCES.items():
                if key in raw["source_hashes"] and raw["source_hashes"][key] != source_hashes.get(relative):
                    raise ValueError("episode_archived_controller_binding_mismatch")
            history_sources.append(deepcopy(raw["source_hashes"]))
            allowance = context.get("remaining_wall_seconds")
            if (type(allowance) not in (int, float) or not math.isfinite(allowance)
                    or allowance <= 0 or type(attempt.get("deadline_exceeded")) is not bool):
                raise ValueError("episode_deadline_missing_or_invalid")
            if checked_attempt["late"] or attempt["deadline_exceeded"]:
                item["reason"] = "deadline_exceeded_observation_not_eligible"
            elif attempt["accepted_for_transition"]:
                item["reason"] = "accepted_action_not_a_failure"
            elif attempt["status"] not in ("UNKNOWN", "REJECTED"):
                item["reason"] = "not_a_known_failed_attempt"
            elif raw["path_success"] is False:
                item.update(eligible=True, reason="observed_path_failure", trigger_kind="path_failed")
            elif raw.get("stability_success") is False:
                item.update(eligible=True, reason="observed_stability_failure", trigger_kind="stability_failed")
            else:
                item["reason"] = "no_known_failure_stage"
        if item["eligible"] and selected is None:
            item["selected"] = True
            selected = item
        considered.append(item)
    if len(task_ids) > 1:
        raise ValueError("episode_contains_multiple_tasks")

    template = load(template_path, False)
    admissions = [deepcopy(c) for c in template.get("cases", []) if c.get("phase") == "admission"]
    if not admissions or len(admissions) > 5 or len({c["case_id"] for c in admissions}) != len(admissions):
        raise ValueError("predeclared_admission_cases_required")
    admission_inputs = []
    for case in admissions:
        source_report = safe(ROOT / case["report"])
        source = load(source_report)
        index = case.get("attempt_index")
        if type(index) is not int or not 1 <= index <= len(source["attempts"]):
            raise ValueError("invalid_admission_attempt_index")
        directory = safe(source_report.parent / ("attempt_%06d" % index))
        references = {}
        for name in ("request.json", "result.json", "trajectory.json", "endpoint_evidence.json"):
            path = safe(directory / name)
            load(path)
            references[str(path)] = originals[str(path)]
        admission_inputs.append({"case_id": case["case_id"], "report_path": str(source_report),
            "report_sha256": originals[str(source_report)], "artifact_sha256": references})

    future_sources = []
    for sources in history_sources:
        current = deepcopy(sources)
        for key, relative in LOCAL_SOURCES.items():
            if key in current:
                current[key] = _sha(safe(ROOT / relative, False))
        if current not in future_sources:
            future_sources.append(current)
    if any(_sha(safe(path, False)) != expected for path, expected in originals.items()):
        raise ValueError("episode_changed_during_preparation")
    status = "PREPARED" if selected else "NO_ELIGIBLE_EXPERIENCE"
    manifest_path = output / "manifest.json" if selected else None
    result = {"schema_version": 1, "status": status,
        "selection_policy": "oldest_eligible_failed_attempt_in_recorded_order_v1",
        "episode_dir": str(episode), "source_run_id": worker["run_id"],
        "experience_task": next(iter(task_ids)) if task_ids else None,
        "experience_report": _project_reference(report_path, allow_test_root),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "selected": selected, "considered": considered,
        "memory_path": str(memory_path), "memory_file_sha256": originals[str(memory_path)],
        "memory_content_sha256": snapshot["content_sha256"], "memory_record_count": len(records),
        "admission_manifest_path": str(template_path),
        "admission_manifest_sha256": originals[str(template_path)],
        "admission_inputs": admission_inputs, "admission_cases_unchanged": True,
        "historical_source_hashes": history_sources,
        "expected_current_validation_source_hashes": future_sources,
        "original_artifact_sha256": originals, "original_files_unchanged": True,
        "development_only": True, "target_is_held_out": False,
        "source_to_target_part_mapping": None, "hard_pruning": False,
        "new_api_calls": 0, "new_physical_calls": 0,
        "evidence_class": "saved_actual_asap_artifacts_checked_not_new_physics",
        "performance_comparison": "NOT_RUN"}
    output.mkdir(parents=True, exist_ok=False)
    if selected:
        debug = {"case_id": "episode_debug_%06d" % selected["attempt_index"], "phase": "debug",
            "source_task": result["experience_task"], "report": result["experience_report"],
            "attempt_index": selected["attempt_index"], "trigger_kind": selected["trigger_kind"]}
        manifest = {"schema_version": 1, "purpose": "automatic episode development, not held-out evaluation",
            "experience_task": result["experience_task"], "experience_report": result["experience_report"],
            "experience_memory": _project_reference(memory_path, allow_test_root),
            "selection_rationale": result["selection_policy"], "cases": [debug] + admissions,
            "episode_learning_audit": str(output / "preparation.json"),
            "frozen_episode_memory": {"path": str(memory_path),
                "file_sha256": result["memory_file_sha256"], "content_sha256": snapshot["content_sha256"]}}
        _json_new(manifest_path, manifest)
    _json_new(output / "preparation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--admission-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = prepare_episode_learning(args.episode_dir, args.admission_manifest, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "manifest_path", "experience_task",
        "new_api_calls", "new_physical_calls", "original_files_unchanged")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
