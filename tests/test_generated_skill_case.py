"""Transfer-runner wiring fixtures: simulated worker replies, NO actual physics.

Existing request/memory contracts run locally. The physical harness and scoped
worker-result checker are explicitly injected, so PASS here is only unit-test
success and cannot serve as research admission or real transfer evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_generated_skill_case as runner
from tests.test_skill_program import append_observation, choice, request
from worker.explicit_action_contract import validate_request


def write_fixture(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream)


class FixtureService:
    selector_decisions = []

    def __init__(self, generated=True, no_action=False):
        self.generated = generated
        self.no_action = no_action
        self.calls = 0

    def handle(self, req):
        self.calls += 1
        selected = deepcopy(req["episode_feedback"][-1]["choice"])
        selected["direction_id"] = "zp"  # Explicit test fixture only.
        result = runner._observation_response(req, selected)
        result["metadata"].update(generated_program_executed=self.generated,
            generated_skill_id="fixture-program" if self.generated else None,
            generated_selector_response={"status": "selected"} if self.generated else None,
            generated_skill_fallback_reason=None if self.generated else "no_applicable_registered_skill")
        if self.no_action:
            result.update(choice=None, action_id=None)
            result["metadata"]["exhausted"] = True
        return result


class GeneratedSkillCaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="generated_transfer_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.template = validate_request(request())
        self.trigger_choice = choice()
        self.prepared = {"request": self.template, "context": {}, "choice": self.trigger_choice,
            "expected_source_hashes": {"fixture_source": "a"*64},
            "spec": {"source_task": "target_fixture"}}
        self.snapshot = {"fixture": True}
        self.trigger_pass = False
        self.trigger_stability_failure = False
        self.recovery_pass = True
        self.recovery_elapsed = 0.1
        self.raw_calls = []
        self.start_patch(patch.object(runner, "write", side_effect=write_fixture))
        self.start_patch(patch.object(runner, "physical_context", side_effect=lambda *a, **k: {
            "event_id": k["event_id"], "remaining_wall_seconds": k["seconds"]}))
        self.start_patch(patch.object(runner, "actual_action", side_effect=self.actual_fixture))
        self.start_patch(patch.object(runner, "request_after_trigger", side_effect=self.after_fixture))
        self.start_patch(patch.object(runner, "validate_trigger_scope", side_effect=self.scope_fixture))
        self.retrieval = self.start_patch(patch.object(runner, "retrieve", return_value=[{
            "strategy_id": "fixture-program", "validation_scope": [
                {"source_task": "source_fixture", "input_hashes": {'a.obj': 'b'*64}}]}]))
        self.network = self.start_patch(patch("urllib.request.OpenerDirector.open",
            side_effect=AssertionError("real_network_forbidden")))
        self.addCleanup(self.network.assert_not_called)

    def start_patch(self, value):
        result = value.start()
        self.addCleanup(value.stop)
        return result

    def actual_fixture(self, packet, folder):
        self.raw_calls.append(deepcopy(packet))
        trigger = len(self.raw_calls) == 1
        passed = self.trigger_pass if trigger else self.recovery_pass
        raw = {"status": "PASS" if passed else "UNKNOWN", "backend": "ASAP_RedMax",
            "path_success": passed, "stability_success": True if passed else None,
            "chosen": deepcopy(packet["choice"]), "source_hashes": {"fixture_source": "a"*64},
            "elapsed_seconds": 0.1 if trigger else self.recovery_elapsed,
            "path_check_calls": 1, "stability_check_calls": int(passed),
            "unit_fixture_only": True}
        raw['input_hashes'] = {'a.obj': 'a'*64}
        if trigger and self.trigger_stability_failure:
            raw.update(path_success=True, stability_success=False, stability_check_calls=1,
                       stability_mode='one_exact_selected_support_configuration')
        raw['raw_upstream'] = dict(path_success=raw['path_success'], stability_success=raw['stability_success'])
        write_fixture(folder / "physics/result.json", raw)
        return raw

    def after_fixture(self, template, raw, selected, *, run_id, event_id, wall_seconds):
        req = deepcopy(template)
        req.update(run_id=run_id, request_id=run_id + ":recovery")
        append_observation(req, selected, path=raw['path_success'], stability=raw['stability_success'])
        return validate_request(req)

    @staticmethod
    def scope_fixture(req, raw, selected):
        if raw["chosen"] != selected:
            raise ValueError("fixture_chosen_mismatch")

    def run_case(self, service=None):
        return runner.run_transfer_case(self.prepared, self.snapshot, service or FixtureService(),
            self.output, "fixture-transfer-run", allow_test_root=True)

    def test_trigger_then_unforced_service_then_recovery_and_memory_freeze(self):
        service = FixtureService()
        report = self.run_case(service)
        self.assertEqual(service.calls, 1)
        self.assertEqual(len(self.raw_calls), 2)
        self.assertEqual(self.raw_calls[0]["choice"]["direction_id"], "xp")
        self.assertEqual(self.raw_calls[1]["choice"]["direction_id"], "zp")
        self.assertTrue(report["generated_reuse_executed"])
        self.assertTrue(report["recovery_accepted"])
        self.assertTrue(report["cross_task_reuse_observed"])
        self.assertFalse(report["learned_skill_reuse_observed"])
        self.assertEqual(report["full_plan_test"], "NOT_RUN")
        self.assertEqual(report["performance_comparison"], "NOT_RUN")
        self.assertEqual(report["memory_export"]["status"], "FROZEN")
        self.assertEqual(report["memory_export"]["record_count"], 2)
        memory = json.loads((self.output / "memory_frozen.json").read_text())
        self.assertEqual(memory["records"][1]["parent_failure_evidence_id"], memory["records"][0]["evidence_id"])
        self.assertFalse(memory["physics_reuse_allowed"])

    def test_recovery_failure_is_completed_workflow_but_not_success(self):
        self.recovery_pass = False
        report = self.run_case()
        self.assertEqual(report["status"], "RECOVERY_NOT_ACCEPTED")
        self.assertTrue(report["workflow_completed"])
        self.assertTrue(report["generated_reuse_executed"])
        self.assertFalse(report["recovery_accepted"])
        self.assertFalse(report["generated_reuse_pass"])
        self.assertEqual(report["memory_export"]["record_count"], 2)

    def test_late_physical_pass_is_not_accepted(self):
        self.recovery_elapsed = 30.001
        report = self.run_case()
        self.assertEqual(report["recovery"]["status"], "PASS")
        self.assertFalse(report["recovery_accepted"])
        self.assertEqual(report["status"], "RECOVERY_NOT_ACCEPTED")

    def test_unreproduced_trigger_skips_all_selection_and_recovery(self):
        self.trigger_pass = True
        service = FixtureService()
        report = self.run_case(service)
        self.assertEqual(service.calls, 0)
        self.assertEqual(len(self.raw_calls), 1)
        self.assertEqual(report["status"], "TRIGGER_NOT_REPRODUCED")
        self.assertFalse(report["generated_reuse_executed"])
        self.assertEqual(report["memory_export"]["record_count"], 1)

    def test_original_actor_fallback_success_is_not_generated_skill_reuse(self):
        self.snapshot["fixture"] = False
        report = self.run_case(FixtureService(generated=False))
        self.assertTrue(report["recovery_accepted"])
        self.assertFalse(report["generated_reuse_executed"])
        self.assertFalse(report["cross_task_reuse_observed"])
        self.assertFalse(report["learned_skill_reuse_observed"])
        self.assertFalse(report["learned_program_executed"])
        self.assertEqual(report["generated_skill_fallback_reason"], "no_applicable_registered_skill")

    def test_target_used_only_in_debug_is_not_cross_task_reuse(self):
        self.snapshot["fixture"] = False  # Unit metadata fixture, not real learned evidence.
        self.retrieval.return_value[0]["development_scope"] = [
            {"source_task": "target_fixture", "phase": "debug", "outcome": "UNKNOWN"}]
        report = self.run_case()
        self.assertTrue(report["generated_reuse_executed"])
        self.assertTrue(report["learned_program_executed"])
        self.assertTrue(report["target_used_in_development"])
        self.assertEqual(report["generated_development_tasks"], ["target_fixture"])
        self.assertEqual(report["generated_source_tasks"], ["source_fixture", "target_fixture"])
        self.assertFalse(report["cross_task_reuse_observed"])
        self.assertFalse(report["cross_task_reuse_pass"])
        self.assertFalse(report["learned_skill_reuse_observed"])
        self.assertTrue(report["generated_reuse_pass"])

    def test_nonfixture_new_task_still_reports_cross_task_execution(self):
        self.snapshot["fixture"] = False
        self.retrieval.return_value[0]["development_scope"] = [
            {"source_task": "development_fixture", "input_hashes": {'a.obj': 'c'*64}}]
        report = self.run_case()
        self.assertTrue(report["learned_program_executed"])
        self.assertTrue(report["cross_task_reuse_pass"])
        self.assertFalse(report["target_used_in_development"])

    def test_stability_failure_invokes_unforced_selector_and_fresh_recovery(self):
        self.prepared['spec']['trigger_kind'] = 'stability_failed'
        self.trigger_stability_failure = True
        service = FixtureService()
        report = self.run_case(service)
        self.assertEqual(service.calls, 1)
        self.assertTrue(report['trigger_reproduced'])
        self.assertTrue(report['generated_reuse_executed'])
        self.assertTrue(report['recovery_accepted'])
        self.assertEqual(report['stability_check_calls'], 2)
        self.assertEqual(report['memory_export']['record_count'], 2)

    def test_different_declared_failure_stage_skips_selection(self):
        self.prepared['spec']['trigger_kind'] = 'stability_failed'
        service = FixtureService()
        report = self.run_case(service)
        self.assertEqual(service.calls, 0)
        self.assertFalse(report['trigger_reproduced'])

    def test_same_geometry_alias_in_debug_cannot_claim_cross_task(self):
        self.retrieval.return_value[0]['development_scope'] = [
            {'source_task': 'another_name', 'input_hashes': {'a.obj': 'a'*64}}]
        report = self.run_case()
        self.assertTrue(report['generated_reuse_pass'])
        self.assertTrue(report['target_used_in_development'])
        self.assertFalse(report['cross_task_reuse_observed'])
        self.assertEqual(report['task_identity_relation'], 'SAME_OR_PREVIOUSLY_USED')

    def test_missing_geometry_evidence_prevents_cross_task_claim(self):
        self.retrieval.return_value[0]['validation_scope'][0].pop('input_hashes')
        report = self.run_case()
        self.assertTrue(report['generated_reuse_executed'])
        self.assertFalse(report['cross_task_reuse_observed'])
        self.assertEqual(report['task_identity_relation'], 'UNKNOWN')

    def test_sdf_cache_or_part_filename_changes_do_not_establish_new_geometry(self):
        rows = [{'validation_scope': [{'source_task': 'alias',
            'input_hashes': {'renamed.obj': 'a'*64, 'renamed.sdf': 'b'*64}}]}]
        result = runner.classify_task_reuse('target', {'a.obj': 'a'*64, 'a.sdf': 'c'*64}, rows)
        self.assertFalse(result['cross_task_eligible'])
        self.assertTrue(result['target_used_for_skill_validation'])
        self.assertFalse(result['task_identity_evidence']['part_id_mapping_inferred'])

    def test_case_spec_cli_and_default_are_explicit(self):
        parsed = runner.parser().parse_args(['--run-dir', 'run', '--skill-library', 'library',
                                           '--case-spec', 'case.json'])
        self.assertEqual(parsed.case_spec, Path('case.json'))
        self.assertEqual(runner.load_case_spec(), runner.DEFAULT_SOURCE)
        spec = deepcopy(runner.DEFAULT_SOURCE)
        spec['trigger_kind'] = 'stability_failed'
        with patch.object(runner, 'checked', return_value=Path('unit_fixture.json')), \
                patch.object(runner, 'read', return_value=spec):
            self.assertEqual(runner.load_case_spec(Path('unit_fixture.json')), spec)
            spec['phase'] = 'admission'
            with self.assertRaisesRegex(ValueError, 'invalid_declared_transfer_case'):
                runner.load_case_spec(Path('unit_fixture.json'))

    def test_unknown_selected_program_and_no_action_cannot_claim_learned_execution(self):
        self.snapshot["fixture"] = False
        self.retrieval.return_value = []
        report = self.run_case()
        self.assertFalse(report["learned_program_executed"])
        self.assertFalse(report["generated_reuse_executed"])
        self.assertFalse(report["cross_task_reuse_pass"])

    def test_no_action_never_claims_learned_execution(self):
        self.snapshot["fixture"] = False
        # This synthetic state still has actions. The existing contract rejects
        # premature exhaustion before any recovery or success report is made.
        with self.assertRaisesRegex(ValueError, "premature_action_exhaustion"):
            self.run_case(FixtureService(no_action=True))
        self.assertEqual(len(self.raw_calls), 1)
        self.assertFalse((self.output / "case_report.json").exists())

    def test_memory_export_failure_is_explicit_not_silent_success(self):
        with patch.object(runner, "append_action_observation", side_effect=ValueError("fixture_export_failure")):
            report = self.run_case()
        self.assertEqual(report["memory_export"]["status"], "NOT_RUN")
        self.assertEqual([e["stage"] for e in report["memory_export"]["errors"]], ["trigger", "recovery"])
        self.assertTrue(report["recovery_accepted"])

    def test_live_mode_cannot_start_without_explicit_budget_authorization(self):
        with patch.object(runner, "source_case") as source:
            with self.assertRaisesRegex(ValueError, "explicit_approved_config"):
                runner.main(["--run-dir", "C:/dev/disassembly_codex_ready_v2/runs/unit_not_created",
                    "--skill-library", "C:/dev/disassembly_codex_ready_v2/runs/nonexistent.json", "--mode", "llm"])
        source.assert_not_called()

    def test_fixture_library_cannot_be_used_for_live_selection(self):
        with self.assertRaisesRegex(ValueError, "offline_wiring_only"):
            runner.main(["--run-dir", "C:/dev/disassembly_codex_ready_v2/runs/unit_not_created",
                "--skill-library", "C:/dev/disassembly_codex_ready_v2/runs/nonexistent.json",
                "--mode", "llm", "--allow-live-api", "--llm-config", "C:/dev/nonexistent.json",
                "--ledger", "C:/dev/nonexistent.sqlite", "--allow-fixture-library"])


if __name__ == "__main__":
    unittest.main()
