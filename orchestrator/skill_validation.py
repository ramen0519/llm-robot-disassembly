"""Evidence checks for generated procedures; never a replacement physics judge.

Only new, bound ASAP worker artifacts can support scoped registration. Fixture
mode is explicit and remains fixture evidence even if a JSON says ``PASS``.
These are integrity checks in a trusted local runner, not remote attestation.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

from worker.task_memory import _safe_path, canonical, digest

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
PHASES = ("debug", "admission", "transfer")


def _sha(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _hashes(value: Any, label: str) -> None:
    if (type(value) is not dict or not value
            or any(type(k) is not str or not k or not _sha(v) for k, v in value.items())):
        raise ValueError("missing_or_invalid_" + label)


def read_artifact(path: Any, allow_test_root: bool = False) -> Dict[str, Any]:
    """Check lexical drive/link boundaries before any file access."""
    safe = _safe_path(path, allow_test_root)
    with safe.open("rb") as stream:
        raw = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("skill_artifact_byte_limit")
    value = json.loads(raw.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError("skill_artifact_must_be_object")
    canonical(value)
    return {"path": str(safe), "sha256": hashlib.sha256(raw).hexdigest(), "value": value}


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _actual_attempt(directory: Any, case: dict, allow_test_root: bool,
                    fixture: bool) -> dict:
    directory = _safe_path(directory, allow_test_root)
    request = read_artifact(directory / "request.json", allow_test_root)
    result = read_artifact(directory / "result.json", allow_test_root)
    raw, req = result["value"], request["value"]
    if case.get('source_budget') is not None:
        binding=case['source_budget']
        source=read_artifact(binding['source_request_path'],allow_test_root)
        from worker.skill_case_budget import validate_source_context
        seconds=validate_source_context(req['context'],source['value']['context'])
        if (source['sha256']!=binding['source_request_sha256']
                or seconds!=binding['seconds'] or seconds!=case.get('case_wall_seconds')):
            raise ValueError('validation_source_budget_mismatch')
    if raw.get("protocol") != "explicit_action_physics_v1":
        raise ValueError("unsupported_physics_protocol")
    allowed_backends = ("ASAP_RedMax", "fixture_not_actual_physics") if fixture else ("ASAP_RedMax",)
    if raw.get("backend") not in allowed_backends:
        raise ValueError("actual_backend_required_or_fixture_mismatch")
    if not fixture and (raw.get("fixture") or req.get("fixture")):
        raise ValueError("fixture_cannot_support_actual_registration")
    if raw.get("chosen") != req.get("choice") or type(req.get("context")) is not dict:
        raise ValueError("worker_choice_binding_mismatch")
    from worker.explicit_action_physics import validate_action
    physical_scope = validate_action(req["context"], req["choice"])
    if raw.get("status") not in ("PASS", "UNKNOWN", "REJECTED", "ERROR"):
        raise ValueError("invalid_physical_status")
    _hashes(raw.get("input_hashes"), "actual_input_hashes")
    _hashes(raw.get("source_hashes"), "actual_source_hashes")
    if raw["input_hashes"] != case["input_hashes"]:
        raise ValueError("stale_or_mismatched_input")
    if raw["source_hashes"] != case["source_hashes"]:
        raise ValueError("stale_or_mismatched_solver")
    if raw.get("input_files_unchanged") is not True or raw.get("source_files_unchanged") is not True:
        raise ValueError("input_or_solver_not_preserved")
    if raw.get("path_check_calls") not in (0, 1) or raw.get("stability_check_calls") not in (0, 1):
        raise ValueError("unexpected_physics_call_count")
    upstream = raw.get("raw_upstream")
    if type(upstream) is not dict:
        raise ValueError("raw_upstream_required")
    if raw.get("path_success") is not upstream.get("path_success"):
        raise ValueError("path_status_disagrees_with_raw")
    if type(raw.get("path_success")) not in (bool, type(None)):
        raise ValueError("invalid_raw_path_success")
    refs = [request, result]
    endpoint = None
    if raw.get("path_check_calls") == 1 and raw.get("path_success") is not None:
        if raw.get("trajectory_file") != "trajectory.json" or raw.get("evidence_file") != "endpoint_evidence.json":
            raise ValueError("missing_actual_path_artifacts")
        trajectory = read_artifact(directory / "trajectory.json", allow_test_root)
        endpoint = read_artifact(directory / "endpoint_evidence.json", allow_test_root)
        if (trajectory["sha256"] != raw.get("trajectory_sha256")
                or endpoint["sha256"] != raw.get("evidence_sha256")
                or endpoint["value"].get("trajectory", {}).get("sha256") != trajectory["sha256"]):
            raise ValueError("path_evidence_hash_mismatch")
        if (trajectory["value"].get("raw_success") is not raw["path_success"]
                or endpoint["value"].get("raw_success") is not raw["path_success"]
                or endpoint["value"].get("kind") != "path_endpoint_measurement"):
            raise ValueError("path_evidence_status_mismatch")
        scope = endpoint["value"].get("scope", {})
        if (scope.get("selected_action") != req["choice"]
                or scope.get("physics", {}).get("source_and_binary_hashes") != raw["source_hashes"]):
            raise ValueError("path_evidence_scope_mismatch")
        expected_scope = {"task_context": req["context"].get("task_context"),
            "assembled_parts": physical_scope["remaining"], "part_move": req["choice"]["part_id"],
            "parts_rest": physical_scope["rest"], "parts_removed": physical_scope["removed"],
            "base_part": physical_scope["base"], "pose": physical_scope["pose"],
            "pose_id": req["choice"]["pose_id"], "seed": physical_scope["seed"]}
        if any(scope.get(key) != expected for key, expected in expected_scope.items()):
            raise ValueError("path_evidence_physical_context_mismatch")
        support = scope.get("support_policy") or {}
        if (support.get("selected_support_ids") != req["choice"]["support_ids"]
                or support.get("actual_parts_fix") != physical_scope["fixed"]
                or support.get("actual_parts_move") != physical_scope["stability_movers"]):
            raise ValueError("path_evidence_support_scope_mismatch")
        if (type(trajectory["value"].get("path")) is not list or not trajectory["value"]["path"]
                or not _number(trajectory["value"].get("controller", {}).get("physics_steps"))
                or type(endpoint["value"].get("measurements", {}).get("pairs")) is not list
                or type(endpoint["value"].get("terminal_q")) is not list
                or len(endpoint["value"]["terminal_q"]) != 6):
            raise ValueError("missing_actual_path_measurements")
        refs.extend((trajectory, endpoint))
    boundary = raw.get("stability_mode") == "all_remaining_rest_explicitly_fixed_no_checker_call"
    if boundary:
        executed = raw.get("executed") or {}
        if (raw.get("path_success") is not True or raw.get("stability_success") is not True
                or raw.get("stability_check_calls") != 0
                or upstream.get("stability_success") is not None
                or executed.get("stability_moving_parts") != []
                or physical_scope["stability_movers"] != []):
            raise ValueError("invalid_all_rest_fixed_boundary")
    elif raw.get("stability_success") is not None:
        if (raw.get("path_success") is not True or raw.get("stability_check_calls") != 1
                or type(raw["stability_success"]) is not bool
                or raw["stability_success"] is not upstream.get("stability_success")
                or raw.get("stability_mode") != "one_exact_selected_support_configuration"
                or type(raw.get("executed", {}).get("stability_moving_parts")) is not list):
            raise ValueError("missing_actual_stability_details")
    if raw["status"] == "PASS" and (raw.get("path_check_calls") != 1
            or raw.get("path_success") is not True or raw.get("stability_success") is not True
            or raw.get("feasible") is not True):
        raise ValueError("fake_pass_or_missing_stage")
    elapsed, allowance = raw.get("elapsed_seconds"), req["context"].get("remaining_wall_seconds")
    if not _number(elapsed) or (allowance is not None and not _number(allowance)):
        raise ValueError("missing_elapsed_or_invalid_deadline")
    late = allowance is not None and elapsed > allowance
    return {"request": request, "result": result, "refs": refs, "late": late,
            "stability_evidence_kind": "explicit_all_rest_fixed_boundary" if boundary else
                ("actual_gravity_checker" if raw.get("stability_check_calls") == 1 else "NOT_RUN")}


def _condition(attempt: dict, source_task: str) -> dict:
    """Seed alone cannot turn the same deterministic condition into a new case."""
    req = attempt["request"]["value"]
    context = req["context"]
    choice = {k: deepcopy(v) for k, v in req["choice"].items() if k not in ("evidence_ids", "pose_id")}
    pose = next((p.get("matrix") for p in context.get("pose_catalog", [])
                 if p.get("pose_id") == req["choice"].get("pose_id")), None)
    return {"input_hashes": attempt["result"]["value"]["input_hashes"],
            "remaining_parts": context.get("remaining_parts"), "base_part": context.get("base_part"),
            "max_grippers": context.get("max_grippers"), "pose_matrix": pose,
            "trigger_choice": choice}


def record_validation(case: dict, artifacts: dict, expected_program_hash: str,
                      allow_test_root: bool = False, fixture: bool = False,
                      program: Optional[dict] = None, *, _classification_version: int = 4) -> dict:
    """Read and bind raw worker outputs; caller success flags are never accepted.

    Case fields: case_id, phase, source_task, program_hash, input_hashes,
    source_hashes, trigger_kind. Artifact fields: attempt_dir, selection_path,
    trigger_attempt_dir. Selection is the runner's immutable binding artifact.
    A false predicate yields NOT_TRIGGERED; a true predicate with no legal
    proposal yields NOT_APPLICABLE. Neither classification is a physical PASS.
    """
    if _classification_version not in (2, 3, 4):
        raise ValueError('unsupported_validation_classification')
    if 'recovery_sequence_path' in artifacts:
        from orchestrator.recovery_sequence import record_sequence
        return record_sequence(case, artifacts, expected_program_hash, allow_test_root, fixture, program,
                               classification_version=_classification_version)
    if (type(case) is not dict or type(artifacts) is not dict
            or not _sha(expected_program_hash) or case.get("program_hash") != expected_program_hash):
        raise ValueError("candidate_program_hash_mismatch")
    for key in ("case_id", "source_task"):
        if type(case.get(key)) is not str or not case[key] or len(case[key]) > 256:
            raise ValueError("missing_" + key)
    if case.get("phase") not in PHASES:
        raise ValueError("invalid_validation_phase")
    _hashes(case.get("input_hashes"), "expected_input_hashes")
    _hashes(case.get("source_hashes"), "expected_source_hashes")
    if case.get("trigger_kind") not in ("path_failed", "stability_failed"):
        raise ValueError("unsupported_trigger_kind")
    expected_phase = case.get('expected_diagnostic_phase')
    if expected_phase is not None and (expected_phase not in ('initial_disconnected', 'dynamic_failure')
            or case['trigger_kind'] != 'stability_failed'):
        raise ValueError('invalid_expected_diagnostic_phase')
    selection = read_artifact(artifacts["selection_path"], allow_test_root)
    selected = selection["value"]
    if (selected.get("program_hash") != expected_program_hash
            or selected.get("source_task") != case["source_task"]
            or selected.get("case_id") != case["case_id"]):
        raise ValueError("selection_program_or_case_binding_mismatch")
    verification_class = "fixture_not_actual_physics" if fixture else "actual_asap_artifacts_checked"
    record = {"schema_version": 1, "classification_version": _classification_version,
              "case": deepcopy(case), "artifacts": deepcopy(artifacts),
              "verification_class": verification_class, "fixture_mode": fixture, "source_refs": [],
              "program_hash": expected_program_hash, "natural_trigger": False,
              "outcome": "NOT_TRIGGERED", "path_calls": 0, "stability_calls": 0,
              "physics_reuse_allowed": False, "hard_pruning": False}
    refs = [selection]
    replayed = None
    program_request = None
    assessed_choice = None
    if program is not None:
        from orchestrator.skill_program import program_hash, step_program
        if program_hash(program) != expected_program_hash:
            raise ValueError("candidate_program_hash_mismatch")
        program_request = read_artifact(selected["program_request_path"], allow_test_root)
        if program_request["sha256"] != selected.get("program_request_sha256"):
            raise ValueError("program_request_hash_mismatch")
        from orchestrator.lesson_runtime import checked_execution, execute_bound
        extension = checked_execution(program, program_request["value"], selected, case)
        replayed = execute_bound(program, program_request["value"], extension)
        record["program"] = deepcopy(program)
        refs.append(program_request)
    record["program_execution_checked"] = replayed is not None
    record["program_predicate_triggered"] = replayed.get("triggered") if replayed else None
    if selected.get("program_status") == "NOT_APPLICABLE":
        if artifacts.get("attempt_dir") is not None:
            raise ValueError("not_triggered_cannot_claim_physics_attempt")
        if replayed is not None and replayed.get("response") is not None:
            raise ValueError("program_replay_disagrees_with_not_applicable")
        record["condition_hash"] = digest({"source_task": case["source_task"],
                                            "input_hashes": case["input_hashes"], "trigger": None})
        record["program_triggered"] = replayed.get("triggered") if replayed else None
        if record["program_predicate_triggered"] is True:
            record["outcome"] = "NOT_APPLICABLE"
        record["not_applicable_reason"] = replayed.get("not_applicable_reason") if replayed else None
        if artifacts.get("trigger_attempt_dir") is not None:
            trigger = _actual_attempt(artifacts["trigger_attempt_dir"], case, allow_test_root, fixture)
            prior = trigger["result"]["value"]
            if selected.get("trigger_result_sha256") != trigger["result"]["sha256"]:
                raise ValueError("trigger_selection_binding_mismatch")
            record.update(trigger_path_calls=prior["path_check_calls"],
                          trigger_stability_calls=prior["stability_check_calls"],
                          physics_backend=prior["backend"],
                          condition_hash=digest(_condition(trigger, case["source_task"])))
            if _classification_version >= 3:
                # A physically present trigger and a false program predicate
                # are different facts. Keep both so a failed/abstaining
                # predeclared admission condition cannot disappear.
                record['natural_trigger'] = _natural_trigger(prior, case['trigger_kind'], trigger['late'],
                                                            expected_phase=expected_phase)
            if prior["backend"] == "ASAP_RedMax":
                record["verification_class"] = "actual_asap_artifacts_checked"
            refs.extend(trigger["refs"])
    else:
        if selected.get("program_status") != "ACTION":
            raise ValueError("invalid_program_status")
        attempt = _actual_attempt(artifacts["attempt_dir"], case, allow_test_root, fixture)
        req, raw = attempt["request"]["value"], attempt["result"]["value"]
        assessed_choice = req["choice"]
        record["physics_backend"] = raw["backend"]
        if raw["backend"] == "ASAP_RedMax":
            record["verification_class"] = "actual_asap_artifacts_checked"
        if replayed is not None and (replayed.get("response") or {}).get("choice") != req["choice"]:
            raise ValueError("program_replay_action_mismatch")
        if (selected.get("worker_request_sha256") != attempt["request"]["sha256"]
                or selected.get("worker_result_sha256") != attempt["result"]["sha256"]
                or selected.get("action_sha256") != digest(req["choice"])):
            raise ValueError("attempt_selection_binding_mismatch")
        trigger = _actual_attempt(artifacts["trigger_attempt_dir"], case, allow_test_root, fixture)
        prior, prior_req = trigger["result"]["value"], trigger["request"]["value"]
        if prior["backend"] != raw["backend"]:
            raise ValueError("trigger_recovery_backend_mismatch")
        if trigger["result"]["path"] == attempt["result"]["path"]:
            raise ValueError("trigger_and_recovery_must_be_distinct_attempts")
        if selected.get("trigger_result_sha256") != trigger["result"]["sha256"]:
            raise ValueError("trigger_selection_binding_mismatch")
        for field in ("remaining_parts", "removed_parts", "base_part", "max_grippers"):
            if req["context"].get(field) != prior_req["context"].get(field):
                raise ValueError("trigger_and_recovery_state_mismatch")
        if req["choice"] == prior_req["choice"]:
            raise ValueError("recovery_requires_changed_action")
        record["recovery_kind"] = ("same_part" if req["choice"].get("part_id") ==
                                    prior_req["choice"].get("part_id") else "alternative_candidate")
        if program is not None:
            current = program_request["value"]
            for field in ("remaining_parts", "base_part", "max_grippers", "pose_catalog"):
                if current.get(field) != req["context"].get(field):
                    raise ValueError("program_request_physics_scope_mismatch")
            feedback = [f for f in current.get("episode_feedback", [])
                        if f.get("evidence_id") == replayed.get("scoped_feedback_evidence_id")]
            expected_feedback_choice = deepcopy(prior_req['choice'])
            if case.get('recovery_protocol'):
                expected_feedback_choice['evidence_ids'] = []
            if (len(feedback) != 1 or feedback[0].get("choice") != expected_feedback_choice
                    or feedback[0].get("path_success") is not prior.get("path_success")
                    or feedback[0].get("stability_success") is not prior.get("stability_success")
                    or feedback[0].get("remaining_parts") != prior_req["context"].get("remaining_parts")):
                raise ValueError("program_feedback_not_bound_to_natural_trigger")
            if expected_phase is not None:
                from worker.stability_diagnostics import project_stability_diagnostics
                if (feedback[0].get('stability_diagnostics') !=
                        project_stability_diagnostics(prior.get('stability_diagnostics'))
                        or feedback[0].get('raw_stability_parts') !=
                        (prior.get('raw_upstream') or {}).get('parts_fall')):
                    raise ValueError('program_diagnostic_not_bound_to_natural_trigger')
        triggered = _natural_trigger(prior, case['trigger_kind'], trigger['late'],
                                     legacy=_classification_version < 3, expected_phase=expected_phase)
        if not triggered:
            raise ValueError("natural_physical_failure_trigger_required")
        record.update(natural_trigger=True, outcome="LATE" if attempt["late"] else raw["status"],
                      path_calls=raw["path_check_calls"], stability_calls=raw["stability_check_calls"],
                      elapsed_seconds=raw["elapsed_seconds"],
                      stability_evidence_kind=attempt["stability_evidence_kind"],
                      condition_hash=digest(_condition(trigger, case["source_task"])),
                      trigger_path_calls=prior["path_check_calls"],
                      trigger_stability_calls=prior["stability_check_calls"])
        refs.extend(attempt["refs"] + trigger["refs"])
    if _classification_version >= 4:
        from orchestrator.skill_execution_roles import assess_execution_roles
        record["execution_roles"] = assess_execution_roles(
            program_request["value"] if program_request is not None else {}, assessed_choice)
    record["source_refs"] = [{"path": ref["path"], "sha256": ref["sha256"]} for ref in refs]
    record["record_sha256"] = digest(record)
    return record


def _natural_trigger(raw: dict, kind: str, late: bool, *, legacy: bool = False,
                     expected_phase=None) -> bool:
    from orchestrator.skill_case import matches_diagnostic_phase
    observed = (raw.get('path_check_calls') == 1 and raw.get('path_success') is False
                if kind == 'path_failed' else
                raw.get('path_success') is True and raw.get('stability_check_calls') == 1
                and raw.get('stability_success') is False)
    status_ok = raw.get('status') != 'PASS' if legacy else raw.get('status') in ('UNKNOWN', 'REJECTED')
    return bool(observed and status_ok and not late and matches_diagnostic_phase(raw, expected_phase))


def check_evidence(record: dict, allow_test_root: bool = False) -> dict:
    """Re-read evidence at admission/freeze, rejecting post-validation tampering."""
    if (type(record) is not dict or record.get("record_sha256") !=
            digest({k: v for k, v in record.items() if k != "record_sha256"})):
        raise ValueError("validation_record_hash_mismatch")
    for ref in record["source_refs"]:
        if read_artifact(ref["path"], allow_test_root)["sha256"] != ref["sha256"]:
            raise ValueError("validation_evidence_changed")
    fixture = record["fixture_mode"]
    replay = record_validation(record["case"], record["artifacts"], record["program_hash"],
                               allow_test_root=allow_test_root, fixture=fixture,
                               program=record.get("program"),
                               _classification_version=record.get('classification_version', 2))
    if "classification_version" not in record:
        # Read-only compatibility for pre-v2 records. Their old classification
        # is retained as history; new runs always distinguish predicate/no-action.
        replay.pop("classification_version", None)
        replay.pop("program_predicate_triggered", None)
        if replay["outcome"] == "NOT_APPLICABLE":
            replay["outcome"] = "NOT_TRIGGERED"
        replay["record_sha256"] = digest({k: v for k, v in replay.items() if k != "record_sha256"})
    if replay != record:
        raise ValueError("validation_evidence_replay_mismatch")
    return deepcopy(record)
