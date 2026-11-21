"""Same frozen library format, with explicit nonphysical-trigger validation."""
import ast
import hashlib
from copy import deepcopy
from pathlib import Path

from orchestrator.pose_guard_recovery import KIND, validate_guard
from orchestrator.pose_guard_runtime import evaluate, step_guard_program
from orchestrator.python_skill_program import validate_program
from orchestrator.skill_registry import SkillRegistry, _new_json, _utc
from orchestrator.skill_validation import read_artifact, _actual_attempt
from worker.explicit_action_contract import digest, validate_response


def contract_probe(program, bundle):
    request, guard = bundle["request"], bundle["guard"]
    plan, index = bundle["plan"], bundle["index"]
    actual = step_guard_program(program, request, guard, plan, index)
    try:
        absent = evaluate(program, request, None, plan, index)["choice"]
        absent_ok = absent is None
    except (ValueError, KeyError, TypeError, RecursionError):
        absent_ok = False
    response = actual["response"]
    duplicate_ok = invalid_ok = False
    if response:
        duplicate = deepcopy(request)
        duplicate["attempted_action_ids"].append(response["action_id"])
        duplicate.setdefault("attempted_actions", []).append(dict(
            action_id=response["action_id"],
            **{k: v for k, v in response["choice"].items() if k != "evidence_ids"}))
        duplicate.setdefault("path_stage_observations", []).append(dict(
            action_id=response["action_id"], path_success=None, status="UNKNOWN",
            deadline_exceeded=False, backend="contract_probe_no_physics",
            source_event_id="contract_probe_no_physics"))
        next_action = step_guard_program(program, duplicate, guard, plan, index)
        duplicate_ok = (next_action["response"] or {}).get("action_id") != response["action_id"]
        invalid = deepcopy(response)
        invalid["choice"]["part_id"] = "__not_current__"
        try:
            validate_response(request, invalid)
        except ValueError:
            invalid_ok = True
    literals = {n.value for n in ast.walk(ast.parse(program["source"]))
                if isinstance(n, ast.Constant) and type(n.value) is str}
    forbidden = set(request["remaining_parts"]) | {p["pose_id"] for p in request["pose_catalog"]}
    return dict(static_contract=True, not_applicable=absent_ok,
        duplicate_action=duplicate_ok, invalid_output_rejected=invalid_ok,
        observation_sensitivity=bool(response) and absent_ok,
        no_literal_source_ids=not bool(literals & forbidden),
        physical_calls=0, api_calls=0)


def record_guard_validation(case, artifacts, program, allow_test_root=False, fixture=False):
    if case.get("trigger_kind") != KIND or case["program_hash"] != digest(validate_program(program)):
        raise ValueError("invalid_guard_validation_case")
    if case["phase"] not in ("debug", "admission"):
        raise ValueError("invalid_guard_validation_phase")
    bundle_ref = read_artifact(artifacts["guard_bundle_path"], allow_test_root)
    bundle = bundle_ref["value"]
    request, guard = bundle["request"], bundle["guard"]
    validate_guard(guard, bundle["plan"], bundle["index"], request)
    if case["guard_bundle_sha256"] != bundle_ref["sha256"]:
        raise ValueError("guard_bundle_changed_after_precommit")
    source_ref = read_artifact(bundle["source_failure_path"], allow_test_root)
    if source_ref["sha256"] != bundle["source_failure_sha256"]:
        raise ValueError("guard_source_failure_changed")
    # Replay the original recorded guard too; the following fallback's physics
    # result is never substituted for the unexecuted planned action.
    source = source_ref["value"]
    from orchestrator.pose_guard_case import validate_replay
    validate_replay(bundle, source)
    from orchestrator.pose_guard_recovery import make_guard
    source_guard = make_guard(source["plan"], source["step_index"], source["request"])
    if source_guard["reason"] != source["guard_event"]["reason"]:
        raise ValueError("historical_guard_source_mismatch")
    refs = [bundle_ref, source_ref]
    root = Path(bundle["source_failure_path"]).absolute().parents[2]
    for name, expected in source["source_paths"].items():
        if Path(name).is_absolute() or ":" in name or ".." in Path(name).parts:
            raise ValueError("unsafe_guard_source_path")
        from worker.task_memory import _safe_path
        safe = _safe_path(root / name, allow_test_root)
        with safe.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != expected:
            raise ValueError("historical_guard_artifact_changed")
        refs.append(dict(path=str(safe), sha256=actual_hash))
    result = step_guard_program(program, request, guard, bundle["plan"], bundle["index"])
    selection_ref = read_artifact(artifacts["selection_path"], allow_test_root)
    selected = selection_ref["value"]
    if (selected["program_hash"] != case["program_hash"]
            or selected["guard_bundle_sha256"] != bundle_ref["sha256"]
            or selected["response"] != result["response"]):
        raise ValueError("guard_program_selection_replay_mismatch")
    refs.append(selection_ref)
    outcome, path_calls, stability_calls, seconds = "NOT_APPLICABLE", 0, 0, 0
    recovery_kind = None
    if result["response"] is not None:
        if artifacts["attempt_dir"] != case["expected_attempt_dir"]:
            raise ValueError("guard_fresh_attempt_directory_mismatch")
        attempted = _actual_attempt(artifacts["attempt_dir"], case, allow_test_root, fixture)
        worker_request, raw = attempted["request"]["value"], attempted["result"]["value"]
        if worker_request["choice"] != result["response"]["choice"]:
            raise ValueError("guard_physical_action_mismatch")
        for key in ("remaining_parts", "base_part", "max_grippers", "pose_catalog", "task_context"):
            if worker_request["context"][key] != request[key]:
                raise ValueError("guard_physics_context_mismatch")
        if raw["chosen"]["evidence_ids"]:
            raise ValueError("guard_cannot_be_physical_evidence_citation")
        outcome = "LATE" if attempted["late"] else raw["status"]
        path_calls, stability_calls, seconds = raw["path_check_calls"], raw["stability_check_calls"], raw["elapsed_seconds"]
        recovery_kind = ("same_planned_mover" if raw["chosen"]["part_id"] == guard["intended"]["part_id"]
                         else "alternative_removal_not_original_action_feasibility")
        refs += attempted["refs"]
    elif artifacts.get("attempt_dir") is not None:
        raise ValueError("abstained_guard_program_cannot_claim_physics")
    condition = dict(input_hashes=case["input_hashes"], remaining_parts=sorted(request["remaining_parts"]),
        base=request["base_part"], max_grippers=request["max_grippers"],
        poses=[p["matrix"] for p in request["pose_catalog"]],
        reference_rotation=guard["reference_rotation"], intended=guard["intended"])
    record = dict(schema_version=1, classification_version="plan_guard_v1", case=deepcopy(case),
        artifacts=deepcopy(artifacts), program=deepcopy(program), program_hash=case["program_hash"],
        verification_class=("fixture_not_actual_physics" if fixture else
            "actual_asap_artifacts_checked" if path_calls else "no_physical_check_completed"),
        fixture_mode=fixture, natural_trigger=True, natural_trigger_kind=KIND,
        trigger_physics_calls=0, trigger_is_physical_failure=False, program_execution_checked=True,
        outcome=outcome, path_calls=path_calls, stability_calls=stability_calls,
        elapsed_seconds=seconds, recovery_kind=recovery_kind, condition_hash=digest(condition),
        physics_reuse_allowed=False, hard_pruning=False,
        source_refs=[dict(path=r["path"], sha256=r["sha256"]) for r in refs])
    record["record_sha256"] = digest(record)
    return record


def check_guard_evidence(record, allow_test_root=False):
    if record["record_sha256"] != digest({k: v for k, v in record.items() if k != "record_sha256"}):
        raise ValueError("guard_validation_record_changed")
    replay = record_guard_validation(record["case"], record["artifacts"], record["program"],
                                    allow_test_root, record["fixture_mode"])
    if replay != record:
        raise ValueError("guard_validation_replay_changed")
    return record


class PoseGuardSkillRegistry(SkillRegistry):
    """Registration is scoped to declared guarded state/pose conditions."""
    def add_validation(self, candidate_id, case, artifacts):
        candidate = self.candidate(candidate_id)
        manifests = self._manifests(candidate_id)
        if case not in manifests["debug"]["cases"] + manifests["admission"]["cases"]:
            raise ValueError("guard_case_not_in_frozen_manifest")
        record = record_guard_validation(case, artifacts, candidate["program"],
                                        self.allow_test_root, self.fixture)
        _new_json(self._path(candidate_id, "validations") / (digest(case["case_id"]) + ".json"),
                  record, self.allow_test_root)
        return record

    def validations(self, candidate_id):
        folder = self._path(candidate_id, "validations")
        return [check_guard_evidence(read_artifact(p, self.allow_test_root)["value"], self.allow_test_root)
                for p in sorted(folder.glob("*.json"))] if folder.exists() else []

    def admit(self, candidate_id):
        candidate, manifests = self.candidate(candidate_id), self._manifests(candidate_id)
        records = self.validations(candidate_id)
        expected = {r["case_id"] for phase in ("debug", "admission") for r in manifests[phase]["cases"]}
        debug = [r for r in records if r["case"]["phase"] == "debug"]
        admission = [r for r in records if r["case"]["phase"] == "admission"]
        reasons = []
        if {r["case"]["case_id"] for r in records} != expected:
            reasons.append("manifest_cases_missing_or_extra")
        if not any(r["outcome"] == "PASS" for r in debug):
            reasons.append("no_debug_guard_recovery")
        if len({r["condition_hash"] for r in admission if r["outcome"] == "PASS"}) < 2:
            reasons.append("fewer_than_two_distinct_admission_conditions")
        if {r["condition_hash"] for r in debug} & {r["condition_hash"] for r in admission}:
            reasons.append("debug_admission_condition_leakage")
        if any(r["outcome"] != "PASS" for r in admission):
            reasons.append("admission_condition_failed_or_abstained")
        probes = []
        for record in records:
            if record["outcome"] == "PASS":
                bundle = read_artifact(record["artifacts"]["guard_bundle_path"], self.allow_test_root)["value"]
                probes.append(contract_probe(candidate["program"], bundle))
        keys = ("static_contract", "not_applicable", "duplicate_action", "invalid_output_rejected",
                "observation_sensitivity", "no_literal_source_ids")
        if not probes or any(not all(p[k] for k in keys) for p in probes):
            reasons.append("guard_program_contract_failed")
        if not self.fixture and candidate["generation"]["origin"] != "live_llm":
            reasons.append("not_live_llm_generated")
        decision = dict(schema_version=1, candidate_id=candidate_id, program_hash=candidate_id,
            lifecycle=("fixture_registered_scoped" if self.fixture else "registered_scoped") if not reasons else
                      ("provisional" if any(r["outcome"] == "PASS" for r in debug) else "proposed"),
            admitted=not reasons, reasons=reasons, fixture=self.fixture, contract_checks=probes,
            validation_family=KIND, manifest_sha256=manifests["manifest_sha256"],
            validation_record_sha256=[r["record_sha256"] for r in records],
            validation_scope=[r["case"] for r in admission if r["outcome"] == "PASS"],
            all_outcomes=[dict(case_id=r["case"]["case_id"], phase=r["case"]["phase"],
                outcome=r["outcome"], condition_hash=r["condition_hash"]) for r in records],
            useful_for_scope="NOT_RUN", transfer_observed=False, generalization_claim=False,
            created_utc=_utc())
        decision["decision_sha256"] = digest(decision)
        _new_json(self._path(candidate_id, "admission.json"), decision, self.allow_test_root)
        self._event(candidate_id, "admission_decided", dict(admitted=not reasons,
                    validation_family=KIND, decision_sha256=decision["decision_sha256"]))
        return decision
