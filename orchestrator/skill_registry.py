"""Immutable generated-procedure versions and evidence-scoped admission.

Technical admission is deliberately separate from usefulness against a control.
The latter is NOT_RUN here; registration never promises improved performance.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Optional
import uuid

from worker.task_memory import _safe_path, canonical, digest
from orchestrator.skill_validation import check_evidence, read_artifact, record_validation
from orchestrator.skill_execution_roles import summarize_execution_roles


def _execution_roles(decision, records):
    summary = summarize_execution_roles(records)
    recorded = decision.get("execution_roles")
    if recorded is not None and recorded != summary:
        raise ValueError("admission_execution_roles_mismatch")
    if recorded is None and any(type(r.get("classification_version")) is int
                               and r["classification_version"] >= 4 for r in records):
        raise ValueError("admission_execution_roles_missing")
    return summary


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_json(path: Any, value: dict, allow_test_root: bool = False) -> None:
    path = _safe_path(path, allow_test_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _identity(value: Any) -> str:
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("candidate_id_must_be_full_program_hash")
    return value


def _manifest(value: dict, phase: str) -> dict:
    if (type(value) is not dict or value.get("phase") != phase
            or type(value.get("cases")) is not list or not value["cases"]):
        raise ValueError("invalid_" + phase + "_manifest")
    ids = []
    for case in value["cases"]:
        if (type(case) is not dict or case.get("phase") != phase
                or type(case.get("case_id")) is not str or not case["case_id"]):
            raise ValueError("invalid_manifest_case")
        ids.append(case["case_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_manifest_case_id")
    return deepcopy(value)


def _contract_probe(program: dict, request: dict, extension=None) -> dict:
    """Exercise the candidate itself, not caller-supplied success booleans."""
    from orchestrator.skill_program import step_program, validate_program
    from worker.explicit_action_contract import validate_response
    validate_program(program)
    from orchestrator.lesson_runtime import execute_bound
    actual = execute_bound(program, request, extension)
    without_feedback = deepcopy(request)
    without_feedback["episode_feedback"] = []
    without_feedback["evidence_ids"] = [r["evidence_id"] for r in without_feedback["memory_references"]]
    absent = execute_bound(program, without_feedback, extension)

    def physical_choice(result):
        choice = (result.get("response") or {}).get("choice")
        return None if choice is None else {k: v for k, v in choice.items() if k != "evidence_ids"}

    response = actual.get("response")
    duplicate_ok = invalid_rejected = False
    if response is not None:
        duplicate = deepcopy(request)
        duplicate["attempted_action_ids"].append(response["action_id"])
        duplicate.setdefault("attempted_actions", []).append(dict(action_id=response["action_id"],
            **{k: deepcopy(v) for k, v in response["choice"].items() if k != "evidence_ids"}))
        duplicate.setdefault("path_stage_observations", []).append(dict(action_id=response["action_id"],
            path_success=None, status="UNKNOWN", deadline_exceeded=False,
            backend="contract_probe_no_physics", source_event_id="contract_probe_no_physics"))
        retry = execute_bound(program, duplicate, extension)
        duplicate_ok = (retry.get("response") or {}).get("action_id") != response["action_id"]
        invalid = deepcopy(response)
        invalid["choice"]["part_id"] = "__not_a_current_part__"
        try:
            validate_response(request, invalid)
        except ValueError:
            invalid_rejected = True
    return {"static_contract": True, "not_applicable": absent.get("response") is None,
            "observation_sensitivity": physical_choice(actual) != physical_choice(absent),
            "duplicate_action": duplicate_ok, "invalid_output_rejected": invalid_rejected,
            "physical_calls": 0, "api_calls": 0,
            "actual_output_sha256": digest(actual), "without_feedback_output_sha256": digest(absent)}


class SkillRegistry:
    """A run-local single writer; every artifact/event is created exactly once."""

    def __init__(self, root: Any, allow_test_root: bool = False, fixture: bool = False):
        self.allow_test_root, self.fixture = allow_test_root, fixture
        self.root = _safe_path(root, allow_test_root)
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = self.root / "registry.json"
        if metadata.exists():
            old = read_artifact(metadata, allow_test_root)["value"]
            if old.get("schema_version") != 1 or old.get("fixture") is not fixture:
                raise ValueError("registry_mode_or_version_mismatch")
        else:
            _new_json(metadata, {"schema_version": 1, "fixture": fixture,
                      "created_utc": _utc(), "purpose": "scoped_generated_skill_registration",
                      "physics_reuse_allowed": False, "performance_effect": "NOT_RUN"}, allow_test_root)

    def _path(self, candidate_id: str, name: str) -> Path:
        return _safe_path(self.root / "candidates" / _identity(candidate_id) / name,
                          self.allow_test_root)

    def _event(self, candidate_id: str, kind: str, fields: dict) -> dict:
        value = dict(schema_version=1, candidate_id=candidate_id, kind=kind,
                     created_utc=_utc(), **deepcopy(fields))
        value["event_sha256"] = digest(value)
        _new_json(self._path(candidate_id, "events") / (uuid.uuid4().hex + ".json"),
                  value, self.allow_test_root)
        return value

    def candidate(self, candidate_id: str) -> dict:
        from orchestrator.skill_program import program_hash, validate_program
        value = read_artifact(self._path(candidate_id, "candidate.json"), self.allow_test_root)["value"]
        if (value.get("candidate_id") != candidate_id or value.get("program_hash") != candidate_id
                or program_hash(validate_program(value["program"])) != candidate_id
                or value.get("record_sha256") != digest({k: v for k, v in value.items()
                                                        if k != "record_sha256"})):
            raise ValueError("candidate_artifact_hash_mismatch")
        if value["generation"]["origin"] == "live_llm":
            generation = value["generation"]
            response = read_artifact(generation["provider_response_path"], self.allow_test_root)
            if response["sha256"] != generation["provider_response_sha256"]:
                raise ValueError("generation_response_changed")
        return value

    def propose(self, program: dict, card: dict, generation: dict) -> dict:
        from orchestrator.skill_program import program_hash, validate_program
        program = validate_program(program)
        identity = program_hash(program)
        if type(card) is not dict or type(card.get("description")) is not str or not card["description"]:
            raise ValueError("skill_card_description_required")
        if type(generation) is not dict or generation.get("origin") not in (
                "live_llm", "fixture_authored", "offline_fixture", "manual_authored"):
            raise ValueError("explicit_generation_origin_required")
        if generation["origin"] == "live_llm":
            response = read_artifact(generation["provider_response_path"], self.allow_test_root)
            if response["sha256"] != generation.get("provider_response_sha256"):
                raise ValueError("generation_response_hash_mismatch")
            if not generation.get("model") or not generation.get("generation_id"):
                raise ValueError("live_generation_identity_required")
            provider = response["value"]
            live_metadata = provider.get("metadata") or {}
            if (provider.get("status") != "proposed" or provider.get("origin") != "actual_llm"
                    or not (live_metadata.get("llm_live") is True
                            or live_metadata.get("original_llm_live") is True)
                    or program_hash(provider.get("program")) != identity):
                raise ValueError("actual_provider_generation_required")
        candidate_path = self._path(identity, "candidate.json")
        if candidate_path.exists():
            prior = self.candidate(identity)
            self._event(identity, "duplicate_program_proposal", {"generation": generation,
                        "counts_as_new_procedure": False})
            return prior
        candidate = {"schema_version": 1, "candidate_id": identity, "program_hash": identity,
                     "program": program, "card": deepcopy(card), "generation": deepcopy(generation),
                     "lifecycle": "proposed", "fixture_registry": self.fixture,
                     "created_utc": _utc(), "generalization_claim": False,
                     "performance_effect": "NOT_RUN"}
        candidate["record_sha256"] = digest(candidate)
        _new_json(candidate_path, candidate, self.allow_test_root)
        self._event(identity, "proposed", {"program_hash": identity})
        return candidate

    def set_manifests(self, candidate_id: str, debug_manifest: dict, admission_manifest: dict) -> dict:
        candidate = self.candidate(candidate_id)
        debug = _manifest(debug_manifest, "debug")
        admission = _manifest(admission_manifest, "admission")
        if set(c["case_id"] for c in debug["cases"]) & set(c["case_id"] for c in admission["cases"]):
            raise ValueError("debug_and_admission_case_ids_overlap")
        if any(c.get("program_hash") != candidate_id for c in debug["cases"] + admission["cases"]):
            raise ValueError("manifest_program_hash_mismatch")
        from orchestrator.lesson_runtime import CONTRACT
        version = candidate['card'].get('runtime_contract')
        if version not in (None, CONTRACT) or any(c.get('runtime_contract') != version
                for c in debug['cases'] + admission['cases']):
            raise ValueError('manifest_runtime_contract_mismatch')
        protocol = candidate['card'].get('recovery_protocol')
        if protocol is not None:
            from orchestrator.recovery_sequence import validate_protocol
            validate_protocol(protocol)
        if any(c.get('recovery_protocol') != protocol for c in debug['cases'] + admission['cases']):
            raise ValueError('manifest_recovery_protocol_mismatch')
        value = {"debug": debug, "admission": admission, "created_utc": _utc()}
        value["manifest_sha256"] = digest(value)
        _new_json(self._path(candidate_id, "manifests.json"), value, self.allow_test_root)
        self._event(candidate_id, "manifests_frozen", {"manifest_sha256": value["manifest_sha256"]})
        return value

    def _manifests(self, candidate_id: str) -> dict:
        value = read_artifact(self._path(candidate_id, "manifests.json"), self.allow_test_root)["value"]
        if value.get("manifest_sha256") != digest({k: v for k, v in value.items() if k != "manifest_sha256"}):
            raise ValueError("manifest_hash_mismatch")
        _manifest(value["debug"], "debug"); _manifest(value["admission"], "admission")
        return value

    def add_validation(self, candidate_id: str, case: dict, artifacts: dict) -> dict:
        candidate = self.candidate(candidate_id)
        manifests = self._manifests(candidate_id)
        planned = manifests["debug"]["cases"] + manifests["admission"]["cases"]
        if case not in planned:
            raise ValueError("validation_case_not_in_frozen_manifest")
        if case.get('recovery_protocol') and 'recovery_sequence_path' not in artifacts:
            raise ValueError('declared_recovery_sequence_evidence_required')
        record = record_validation(case, artifacts, candidate_id,
                    allow_test_root=self.allow_test_root, fixture=self.fixture,
                    program=candidate["program"])
        filename = digest({"case_id": case["case_id"]}) + ".json"
        _new_json(self._path(candidate_id, "validations") / filename, record, self.allow_test_root)
        if case["phase"] == "debug" and record["outcome"] == "PASS" and record["natural_trigger"]:
            self._event(candidate_id, "debug_validated", {"record_sha256": record["record_sha256"],
                        "lifecycle": "provisional", "generalization_claim": False})
        return record

    def validations(self, candidate_id: str) -> list:
        directory = self._path(candidate_id, "validations")
        if not directory.exists():
            return []
        return [check_evidence(read_artifact(p, self.allow_test_root)["value"], self.allow_test_root)
                for p in sorted(directory.glob("*.json"))]

    def admit(self, candidate_id: str, debug_manifest: Optional[dict] = None,
              admission_manifest: Optional[dict] = None, contract_checks: Optional[dict] = None) -> dict:
        candidate = self.candidate(candidate_id)
        manifests = self._manifests(candidate_id)
        if ((debug_manifest is not None and debug_manifest != manifests["debug"])
                or (admission_manifest is not None and admission_manifest != manifests["admission"])):
            raise ValueError("admission_manifest_changed_after_freeze")
        records = self.validations(candidate_id)
        expected = {c["case_id"] for p in ("debug", "admission") for c in manifests[p]["cases"]}
        actual = {r["case"]["case_id"] for r in records}
        debug = [r for r in records if r["case"]["phase"] == "debug"]
        admission = [r for r in records if r["case"]["phase"] == "admission"]
        good = lambda r: r["outcome"] == "PASS" and r["natural_trigger"] and r["program_execution_checked"]
        debug_pass = [r for r in debug if good(r)]
        admission_pass = [r for r in admission if good(r)]
        reasons = []
        if actual != expected:
            reasons.append("manifest_cases_missing_or_extra")
        if not debug_pass:
            reasons.append("no_natural_debug_recovery")
        if len({r["condition_hash"] for r in admission_pass}) < 2:
            reasons.append("fewer_than_two_distinct_admission_conditions")
        if {r["condition_hash"] for r in debug} & {r["condition_hash"] for r in admission}:
            reasons.append("debug_admission_physical_condition_leakage")
        # Every planned failure remains visible. A failing triggered admission
        # condition prevents claiming the entire predeclared scope succeeded.
        if any(r["natural_trigger"] and r["outcome"] != "PASS" for r in admission):
            reasons.append("triggered_admission_condition_failed")
        required_checks = ("static_contract", "not_applicable", "duplicate_action",
                           "observation_sensitivity", "invalid_output_rejected")
        probes = []
        for record in debug_pass + admission_pass:
            selection = read_artifact(record["artifacts"]["selection_path"], self.allow_test_root)["value"]
            request = read_artifact(selection["program_request_path"], self.allow_test_root)["value"]
            from orchestrator.lesson_runtime import checked_execution
            extension = checked_execution(candidate["program"], request, selection, record["case"])
            probes.append(dict(case_id=record["case"]["case_id"],
                               **_contract_probe(candidate["program"], request, extension)))
        if not probes or any(not all(p.get(k) is True for k in required_checks) for p in probes):
            reasons.append("contract_gate_incomplete")
        if not self.fixture and candidate["generation"]["origin"] != "live_llm":
            reasons.append("not_live_llm_generated")
        if not self.fixture and any(r["verification_class"] != "actual_asap_artifacts_checked" for r in records):
            reasons.append("fixture_evidence_cannot_enter_actual_registry")
        lifecycle = ("fixture_registered_scoped" if self.fixture else "registered_scoped") if not reasons else (
                    "provisional" if debug_pass else "proposed")
        decision = {"schema_version": 1, "candidate_id": candidate_id, "program_hash": candidate_id,
                    "lifecycle": lifecycle, "admitted": not reasons, "reasons": reasons,
                    "fixture": self.fixture, "contract_checks": probes,
                    "caller_checks_not_authoritative": deepcopy(contract_checks),
                    "manifest_sha256": manifests["manifest_sha256"],
                    "validation_record_sha256": [r["record_sha256"] for r in records],
                    "validation_scope": [r["case"] for r in admission_pass],
                    "execution_roles": summarize_execution_roles(records),
                    "all_outcomes": [{"case_id": r["case"]["case_id"], "phase": r["case"]["phase"],
                        "outcome": r["outcome"], "condition_hash": r["condition_hash"]} for r in records],
                    "useful_for_scope": "NOT_RUN", "transfer_observed": False,
                    "generalization_claim": False, "created_utc": _utc()}
        decision["decision_sha256"] = digest(decision)
        _new_json(self._path(candidate_id, "admission.json"), decision, self.allow_test_root)
        self._event(candidate_id, "admission_decided", {"decision_sha256": decision["decision_sha256"],
                    "lifecycle": lifecycle})
        return decision

    def freeze(self, path: Any) -> dict:
        entries = []
        directory = _safe_path(self.root / "candidates", self.allow_test_root)
        for folder in sorted(directory.iterdir()) if directory.exists() else []:
            folder = _safe_path(folder, self.allow_test_root)
            if not folder.is_dir():
                continue
            candidate = self.candidate(folder.name)
            admission_path = self._path(folder.name, "admission.json")
            if not admission_path.exists():
                continue
            decision = read_artifact(admission_path, self.allow_test_root)["value"]
            if decision.get("decision_sha256") != digest({k: v for k, v in decision.items() if k != "decision_sha256"}):
                raise ValueError("admission_decision_hash_mismatch")
            records = self.validations(folder.name)
            if decision["validation_record_sha256"] != [r["record_sha256"] for r in records]:
                raise ValueError("admission_validation_set_changed")
            execution_roles = _execution_roles(decision, records)
            if decision["admitted"]:
                entries.append({"candidate": candidate, "admission": decision,
                                "validations": records, "execution_roles": execution_roles})
        snapshot = {"schema_version": 1, "kind": "frozen_generated_skills_v1", "entries": entries,
                    "fixture": self.fixture, "usable_for_actual_planning": not self.fixture,
                    "physics_reuse_allowed": False, "hard_pruning": False,
                    "created_utc": _utc(), "performance_effect": "NOT_RUN"}
        snapshot["content_sha256"] = digest(snapshot)
        _new_json(path, snapshot, self.allow_test_root)
        return snapshot

    def retrieve(self, snapshot: dict, context: Optional[dict] = None) -> list:
        return retrieve(snapshot, context, allow_fixture=self.fixture)


def retrieve(snapshot: dict, context: Optional[dict] = None, allow_fixture: bool = False) -> list:
    """Return source-scoped programs, never mapped target IDs or cached PASS."""
    from orchestrator.skill_program import program_hash
    if (type(snapshot) is not dict or snapshot.get("kind") != "frozen_generated_skills_v1"
            or snapshot.get("content_sha256") != digest({k: v for k, v in snapshot.items() if k != "content_sha256"})):
        raise ValueError("frozen_registry_hash_mismatch")
    if snapshot.get("fixture") and not allow_fixture:
        raise ValueError("fixture_snapshot_not_for_actual_planning")
    if (type(snapshot.get("fixture")) is not bool
            or snapshot.get("usable_for_actual_planning") is not (not snapshot["fixture"])
            or snapshot.get("physics_reuse_allowed") is not False
            or snapshot.get("hard_pruning") is not False):
        raise ValueError("unsafe_frozen_registry_policy")
    found = []
    for item in snapshot["entries"]:
        candidate, decision = item["candidate"], item["admission"]
        expected_lifecycle = "fixture_registered_scoped" if snapshot["fixture"] else "registered_scoped"
        if (decision.get("admitted") is not True or decision.get("lifecycle") != expected_lifecycle
                or candidate.get("candidate_id") != candidate.get("program_hash")
                or decision.get("candidate_id") != candidate.get("candidate_id")
                or decision.get("program_hash") != candidate.get("program_hash")
                or program_hash(candidate["program"]) != candidate["program_hash"]
                or candidate.get("record_sha256") != digest({k: v for k, v in candidate.items() if k != "record_sha256"})
                or decision.get("decision_sha256") != digest({k: v for k, v in decision.items() if k != "decision_sha256"})):
            raise ValueError("invalid_frozen_skill_entry")
        validations = item.get("validations", [])
        from orchestrator.lesson_runtime import CONTRACT
        version = candidate['card'].get('runtime_contract')
        if version not in (None, CONTRACT) or any(r['case'].get('runtime_contract') != version for r in validations):
            raise ValueError('frozen_runtime_contract_mismatch')
        protocol = candidate['card'].get('recovery_protocol')
        if protocol is not None:
            from orchestrator.recovery_sequence import validate_protocol
            validate_protocol(protocol)
        if any(r['case'].get('recovery_protocol') != protocol
               or (protocol is not None and r.get('recovery_sequence', {}).get('protocol') != protocol)
               for r in validations):
            raise ValueError('frozen_recovery_protocol_mismatch')
        if decision["validation_record_sha256"] != [r.get("record_sha256") for r in validations]:
            raise ValueError("frozen_validation_set_mismatch")
        if any(r.get("record_sha256") != digest({k: v for k, v in r.items() if k != "record_sha256"})
               for r in validations):
            raise ValueError("frozen_validation_record_tampered")
        execution_roles = _execution_roles(decision, validations)
        if "execution_roles" in item and item["execution_roles"] != execution_roles:
            raise ValueError("frozen_execution_roles_mismatch")
        if not snapshot["fixture"] and (candidate["generation"]["origin"] != "live_llm"
                or any(r["verification_class"] != "actual_asap_artifacts_checked" for r in validations)):
            raise ValueError("fixture_entry_cannot_enter_actual_snapshot")
        # Admission PASS scope is not the full set of tasks used in development.
        # Keep every debug outcome visible, including failures/not-applicable.
        # Derive this view from already hash-checked records; frozen bytes stay
        # untouched and source IDs remain source labels rather than target IDs.
        development_scope = [dict(deepcopy(record["case"]), outcome=record["outcome"],
                                  validation_record_sha256=record["record_sha256"])
                             for record in validations if record["case"].get("phase") == "debug"]
        found.append({"strategy_id": candidate["candidate_id"], "program_hash": candidate["program_hash"],
                      "program": deepcopy(candidate["program"]), "card": deepcopy(candidate["card"]),
                      "lifecycle": decision["lifecycle"], "validation_scope": deepcopy(decision["validation_scope"]),
                      "development_scope": development_scope,
                      "execution_roles": execution_roles,
                      "current_task_context": deepcopy(context), "current_part_id_mapping": None,
                      "current_attempt_verified": False, "requires_fresh_physics": True,
                      "physics_reuse_allowed": False, "hard_pruning": False,
                      "fixture": snapshot["fixture"], "useful_for_scope": "NOT_RUN"})
    return found
