"""Fixture-only generated registry tests; registration is not real discovery."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_skill_validation import fixture_case, save
from test_skill_program import program, choice
from orchestrator.skill_registry import SkillRegistry, retrieve
from orchestrator.skill_program import program_hash


class SkillRegistryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="skill_registry_fixture_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.registry = SkillRegistry(self.root / "registry", True, fixture=True)
        self.policy = program()
        self.identity = program_hash(self.policy)
        self.card = {"description": "Synthetic conditional recovery, never an actual finding"}
        self.generation = {"origin": "offline_fixture", "generation_id": "unit_fixture"}
        self.registry.propose(self.policy, self.card, self.generation)

    def prepare(self, *, duplicate=False, failure=False, not_triggered=False):
        bundles = [fixture_case(self.root, "debug", "debug", "a"),
                   fixture_case(self.root, "admission1", "admission", "b"),
                   fixture_case(self.root, "admission2", "admission", "b" if duplicate else "c",
                                success=not failure)]
        if not_triggered:
            bundles.append(fixture_case(self.root, "not_triggered", "admission", "d", not_triggered=True))
        self.registry.set_manifests(self.identity, {"phase": "debug", "cases": [bundles[0][0]]},
            {"phase": "admission", "cases": [b[0] for b in bundles[1:]]})
        for case, artifacts, _ in bundles:
            self.registry.add_validation(self.identity, case, artifacts)
        return bundles

    def test_scoped_fixture_admission_and_frozen_reuse_never_actual(self):
        self.prepare(not_triggered=True)
        result = self.registry.admit(self.identity)
        self.assertTrue(result["admitted"])
        self.assertEqual(result["lifecycle"], "fixture_registered_scoped")
        self.assertEqual(result["useful_for_scope"], "NOT_RUN")
        self.assertIn("NOT_TRIGGERED", [r["outcome"] for r in result["all_outcomes"]])
        frozen = self.registry.freeze(self.root / "snapshot.json")
        with self.assertRaisesRegex(ValueError, "fixture_snapshot"):
            retrieve(frozen)
        rows = retrieve(frozen, {"new_task": "different"}, allow_fixture=True)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["current_part_id_mapping"])
        self.assertTrue(rows[0]["requires_fresh_physics"])
        self.assertFalse(rows[0]["current_attempt_verified"])

    def test_duplicate_condition_does_not_count_as_two(self):
        self.prepare(duplicate=True)
        result = self.registry.admit(self.identity)
        self.assertFalse(result["admitted"])
        self.assertIn("fewer_than_two_distinct_admission_conditions", result["reasons"])
        self.assertEqual(result["lifecycle"], "provisional")

    def test_explicit_physical_pass_registration_does_not_grant_automatic_role(self):
        bundles = [fixture_case(self.root, name, phase, name,
                    trigger_choice=choice(support_ids=["b"])) for name, phase in
                   (("debug", "debug"), ("admission1", "admission"), ("admission2", "admission"))]
        self.registry.set_manifests(self.identity, {"phase": "debug", "cases": [bundles[0][0]]},
            {"phase": "admission", "cases": [b[0] for b in bundles[1:]]})
        for case, artifacts, _ in bundles:
            self.registry.add_validation(self.identity, case, artifacts)
        decision = self.registry.admit(self.identity)
        self.assertTrue(decision["admitted"])
        roles = decision["execution_roles"]
        self.assertEqual(len(roles["explicit_plan"]["physical_pass_scope"]), 3)
        self.assertEqual(roles["automatic_recovery"]["guard_compatible_pass_scope"], [])
        self.assertEqual(len(roles["automatic_recovery"]["explicit_plan_only_pass_scope"]), 3)
        self.assertFalse(roles["automatic_recovery"]["authorized"])
        frozen = self.registry.freeze(self.root / "explicit_only.json")
        self.assertEqual(retrieve(frozen, allow_fixture=True)[0]["execution_roles"], roles)

    def test_legacy_snapshot_exposes_unrecorded_roles_without_mutation(self):
        from worker.task_memory import digest
        self.prepare(); self.registry.admit(self.identity)
        frozen = self.registry.freeze(self.root / "legacy_roles.json")
        entry = frozen["entries"][0]
        entry.pop("execution_roles")
        for record in entry["validations"]:
            record["classification_version"] = 3
            record.pop("execution_roles")
            record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
        decision = entry["admission"]
        decision.pop("execution_roles")
        decision["validation_record_sha256"] = [r["record_sha256"] for r in entry["validations"]]
        decision["decision_sha256"] = digest({k: v for k, v in decision.items() if k != "decision_sha256"})
        frozen["content_sha256"] = digest({k: v for k, v in frozen.items() if k != "content_sha256"})
        before = deepcopy(frozen)
        roles = retrieve(frozen, allow_fixture=True)[0]["execution_roles"]
        self.assertEqual(roles["status"], "NOT_RECORDED")
        self.assertEqual(len(roles["unrecorded_cases"]), 3)
        self.assertEqual(roles["automatic_recovery"]["guard_compatible_pass_scope"], [])
        self.assertFalse(roles["automatic_recovery"]["authorized"])
        self.assertEqual(frozen, before)

    def test_frozen_role_scope_must_match_hash_checked_validation_records(self):
        from worker.task_memory import digest
        self.prepare(); self.registry.admit(self.identity)
        frozen = self.registry.freeze(self.root / "role_scope_tamper.json")
        frozen["entries"][0]["execution_roles"]["automatic_recovery"]["authorized"] = True
        frozen["content_sha256"] = digest({k: v for k, v in frozen.items() if k != "content_sha256"})
        with self.assertRaisesRegex(ValueError, "frozen_execution_roles_mismatch"):
            retrieve(frozen, allow_fixture=True)

    def test_retrieval_exposes_all_debug_outcomes_without_mutating_snapshot(self):
        bundles = [fixture_case(self.root, "debug_pass", "debug", "a"),
                   fixture_case(self.root, "debug_failure", "debug", "d", success=False),
                   fixture_case(self.root, "admission1", "admission", "b"),
                   fixture_case(self.root, "admission2", "admission", "c")]
        self.registry.set_manifests(self.identity,
            {"phase": "debug", "cases": [case for case, _, _ in bundles[:2]]},
            {"phase": "admission", "cases": [case for case, _, _ in bundles[2:]]})
        for case, artifacts, _ in bundles:
            self.registry.add_validation(self.identity, case, artifacts)
        self.assertTrue(self.registry.admit(self.identity)["admitted"])
        frozen = self.registry.freeze(self.root / "development_snapshot.json")
        before = deepcopy(frozen)
        row = retrieve(frozen, allow_fixture=True)[0]
        self.assertEqual(frozen, before)
        self.assertEqual({case["case_id"] for case in row["development_scope"]},
                         {"debug_pass", "debug_failure"})
        self.assertEqual({case["outcome"] for case in row["development_scope"]}, {"PASS", "UNKNOWN"})
        self.assertTrue(all(case["validation_record_sha256"] for case in row["development_scope"]))
        self.assertEqual({case["phase"] for case in row["validation_scope"]}, {"admission"})

    def test_failure_kept_and_prevents_full_scope_admission(self):
        self.prepare(failure=True)
        result = self.registry.admit(self.identity)
        self.assertFalse(result["admitted"])
        self.assertIn("triggered_admission_condition_failed", result["reasons"])
        self.assertEqual(len(result["all_outcomes"]), 3)
        self.assertEqual(self.registry.freeze(self.root / "empty.json")["entries"], [])

    def test_missing_cases_not_posthoc_success_selection(self):
        bundles = self.prepare()
        # A fresh registry's fixed manifest contains cases not yet executed.
        other = SkillRegistry(self.root / "other", True, fixture=True)
        other.propose(self.policy, self.card, self.generation)
        other.set_manifests(self.identity, {"phase": "debug", "cases": [bundles[0][0]]},
                           {"phase": "admission", "cases": [b[0] for b in bundles[1:]]})
        other.add_validation(self.identity, bundles[0][0], bundles[0][1])
        result = other.admit(self.identity)
        self.assertIn("manifest_cases_missing_or_extra", result["reasons"])

    def test_manifest_and_validation_and_snapshot_cannot_be_overwritten(self):
        bundles = self.prepare()
        with self.assertRaises(FileExistsError):
            self.registry.add_validation(self.identity, bundles[0][0], bundles[0][1])
        manifests = self.registry._manifests(self.identity)
        with self.assertRaises(FileExistsError):
            self.registry.set_manifests(self.identity, manifests["debug"], manifests["admission"])
        self.registry.admit(self.identity)
        self.registry.freeze(self.root / "snapshot.json")
        with self.assertRaises(FileExistsError):
            self.registry.freeze(self.root / "snapshot.json")

    def test_name_changes_do_not_duplicate_program(self):
        first = self.registry.candidate(self.identity)
        second = self.registry.propose(self.policy, {"description": "new name"}, self.generation)
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.root / "registry" / "candidates").iterdir())), 1)

    def test_program_tamper_rejected(self):
        candidate = self.registry.candidate(self.identity)
        candidate["program"]["rules"][0]["select"]["direction"] = "xp"
        save(self.root / "registry" / "candidates" / self.identity / "candidate.json", candidate)
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            self.registry.candidate(self.identity)

    def test_frozen_protocol_cannot_be_added_without_bound_sequence_records(self):
        from worker.task_memory import digest
        self.prepare();self.registry.admit(self.identity)
        frozen=self.registry.freeze(self.root/'protocol_snapshot.json')
        candidate=frozen['entries'][0]['candidate']
        candidate['card']['recovery_protocol']=dict(version='bounded_support_recovery_v1',
            max_recovery_attempts=3,max_total_seconds=2000,action_scope='support_only')
        candidate['record_sha256']=digest({k:v for k,v in candidate.items() if k!='record_sha256'})
        frozen['content_sha256']=digest({k:v for k,v in frozen.items() if k!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'frozen_recovery_protocol_mismatch'):
            retrieve(frozen,allow_fixture=True)

    def test_fixture_generation_not_accepted_as_live_provider(self):
        response = {"status": "proposed", "origin": "injected_transport", "metadata": {"llm_live": False},
                    "program": self.policy}
        response_path = self.root / "fake_provider.json"
        sha = save(response_path, response)
        reg = SkillRegistry(self.root / "actual", True)
        with self.assertRaisesRegex(ValueError, "actual_provider"):
            reg.propose(self.policy, self.card, {"origin": "live_llm", "model": "gpt-6-luna",
                "generation_id": "fake", "provider_response_path": str(response_path), "provider_response_sha256": sha})

    def test_e_drive_registry_root_rejected(self):
        with self.assertRaisesRegex(ValueError, "exclude non"):
            SkillRegistry("E:/not_accessed")

    def test_debug_admission_physical_condition_overlap_rejected(self):
        bundles = [fixture_case(self.root, "debug", "debug", "a"),
                   fixture_case(self.root, "admission1", "admission", "a"),
                   fixture_case(self.root, "admission2", "admission", "b")]
        self.registry.set_manifests(self.identity, {"phase": "debug", "cases": [bundles[0][0]]},
            {"phase": "admission", "cases": [b[0] for b in bundles[1:]]})
        for case, artifacts, _ in bundles:
            self.registry.add_validation(self.identity, case, artifacts)
        result = self.registry.admit(self.identity)
        self.assertIn("debug_admission_physical_condition_leakage", result["reasons"])

    def test_posthoc_case_and_changed_manifest_rejected(self):
        bundles = self.prepare()
        case = deepcopy(bundles[0][0])
        case["case_id"] = "unplanned"
        with self.assertRaisesRegex(ValueError, "frozen_manifest"):
            self.registry.add_validation(self.identity, case, bundles[0][1])
        with self.assertRaisesRegex(ValueError, "changed_after_freeze"):
            self.registry.admit(self.identity, {"phase": "debug", "cases": []})

    def test_bool_claims_do_not_replace_executed_contract_probes(self):
        self.prepare()
        decision = self.registry.admit(self.identity, contract_checks={"static_contract": False})
        self.assertTrue(decision["admitted"])
        self.assertTrue(all(row["static_contract"] for row in decision["contract_checks"]))
        self.assertEqual(decision["caller_checks_not_authoritative"], {"static_contract": False})


if __name__ == "__main__":
    unittest.main()
