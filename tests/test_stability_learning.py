"""Saved-artifact and classification checks, not new physics or model trials."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prepare_stability_learning as preparation

SOURCE = ROOT / "runs/automatic_learning_cycle_20261019/live_01/next_episode"


class StabilitySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((SOURCE / "planner/planner_report.json").read_text(encoding="utf-8"))
        cls.physicals = [json.loads((SOURCE / "planner" / ("attempt_%06d" % i) / "request.json").read_text(encoding="utf-8"))
                         for i in range(1, len(cls.report["attempts"]) + 1)]
        cls.considered = [{"attempt_index": i, "memory_evidence_id": "unit-classification-only",
            "raw_result_path": str(SOURCE / "planner" / ("attempt_%06d" % i) / "result.json"),
            "raw_result_sha256": preparation.episode_preparation._sha(SOURCE / "planner" / ("attempt_%06d" % i) / "result.json"),
            "eligible": not a["accepted_for_transition"],
            "reason": "accepted_action_not_a_failure" if a["accepted_for_transition"] else "observed_failure"}
            for i, a in enumerate(cls.report["attempts"], 1)]

    def classify(self, attempt=None, checked=None, physical=None):
        return preparation.classify_condition(attempt or deepcopy(self.report["attempts"][1]),
            checked or deepcopy(self.considered[1]), physical or deepcopy(self.physicals[1]))

    def test_chronology_selects_two_four_five_and_retains_all_failures(self):
        rows, selected = preparation.select_conditions(self.report["attempts"], self.considered, self.physicals)
        self.assertEqual([r["attempt_index"] for r in selected], [2, 4, 5])
        self.assertEqual([r["phase"] for r in selected], ["debug", "admission", "admission"])
        self.assertEqual(len({r["condition_hash"] for r in selected}), 3)
        self.assertEqual(sum(r["stability_failure_observed"] for r in rows), 9)
        self.assertEqual(sum(r["reason"] == "initial_disconnection_outside_dynamic_skill_scope" for r in rows), 6)
        self.assertEqual(len(rows), 12)
        self.assertEqual(selected[0]["condition_hash"], "da72b9df8dcea1686be17be7703a955af3fce732be7ed67901b92a0e366f8283")

    def test_dynamic_parts_must_be_observed_not_raw_parts_fall_guess(self):
        attempt = deepcopy(self.report["attempts"][1])
        attempt["raw"]["stability_diagnostics"] = None
        self.assertEqual(self.classify(attempt)["reason"], "missing_stability_diagnostic")

    def test_bad_diagnostic_rejected_without_inventing_observation(self):
        attempt = deepcopy(self.report["attempts"][1])
        attempt["raw"]["stability_diagnostics"]["dynamic_failure_parts"][0]["displacement"] = .1
        row = self.classify(attempt)
        self.assertFalse(row["eligible"])
        self.assertEqual(row["reason"], "invalid_stability_diagnostic")

    def test_zero_physics_or_instrumentation_errors_excluded(self):
        for key, value in (("physics_steps", 0), ("instrumentation_errors", ["ValueError"])):
            with self.subTest(key=key):
                attempt = deepcopy(self.report["attempts"][1])
                attempt["raw"]["stability_diagnostics"][key] = value
                self.assertEqual(self.classify(attempt)["reason"], "incomplete_dynamic_observation")

    def test_checker_call_required_and_late_integrity_not_overridden(self):
        attempt = deepcopy(self.report["attempts"][1])
        attempt["raw"]["stability_check_calls"] = 0
        self.assertEqual(self.classify(attempt)["reason"], "missing_actual_stability_checker")
        checked = dict(self.considered[1], eligible=False, reason="deadline_exceeded_observation_not_eligible")
        self.assertEqual(self.classify(checked=checked)["reason"], checked["reason"])

    def test_support_budget_and_current_candidate_scope_enforced(self):
        physical = deepcopy(self.physicals[1])
        physical["context"]["max_grippers"] = 1
        self.assertEqual(self.classify(physical=physical)["reason"], "observed_dynamic_parts_exceed_declared_support_budget")
        attempt = deepcopy(self.report["attempts"][1])
        attempt["remaining_parts"].remove("4")
        self.assertEqual(self.classify(attempt)["reason"], "observed_dynamic_parts_not_current_support_candidates")

    def test_duplicate_physical_condition_is_not_new_admission(self):
        attempts = [self.report["attempts"][1]] * 4
        considered = [dict(self.considered[1], attempt_index=i) for i in range(1, 5)]
        physicals = [self.physicals[1]] * 4
        rows, selected = preparation.select_conditions(attempts, considered, physicals)
        self.assertEqual(len(selected), 1)
        self.assertEqual([r["reason"] for r in rows[1:]], ["duplicate_physical_trigger_condition"] * 3)

    def test_input_length_mismatch_stops(self):
        with self.assertRaisesRegex(ValueError, "input_count"):
            preparation.select_conditions(self.report["attempts"], [], self.physicals)


class StabilityPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stability_artifact_checks_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).absolute()
        self.output = self.root / "preparation"

    def prepare(self):
        return preparation.prepare_stability_learning(SOURCE, self.output, allow_test_root=True)

    def test_full_artifact_integrity_then_manifest_without_any_new_calls(self):
        before = preparation.episode_preparation._sha(SOURCE / "action_memory_frozen.json")
        with patch("subprocess.Popen", side_effect=AssertionError("No subprocess allowed")):
            result = self.prepare()
        self.assertEqual(result["status"], "PREPARED")
        self.assertEqual(result["selection_counts"], dict(all_attempts=12, all_stability_failures=9,
            initial_disconnection=6, dynamic_failure=3, eligible_distinct_conditions=3, selected_conditions=3))
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual([c["attempt_index"] for c in manifest["cases"]], [2, 4, 5])
        self.assertEqual([c["trigger_kind"] for c in manifest["cases"]], ["stability_failed"] * 3)
        self.assertTrue(all("trigger_override" not in c for c in manifest["cases"]))
        self.assertEqual(manifest["synthesis_failure_scope"], "dynamic_stability_failed")
        self.assertEqual(manifest["experience_report"], "runs/automatic_learning_cycle_20261019/live_01/next_episode/planner/planner_report.json")
        self.assertEqual(manifest["frozen_episode_memory"]["file_sha256"], before)
        self.assertEqual(preparation.episode_preparation._sha(SOURCE / "action_memory_frozen.json"), before)
        self.assertEqual(result["new_api_calls"], 0)
        self.assertEqual(result["new_physical_calls"], 0)
        self.assertFalse(result["generic_integrity_manifest_used_for_experiment"])
        self.assertFalse(result["target_is_held_out"])
        self.assertFalse(result["fresh_admission_results_available_to_synthesis"])
        for path, expected in result["original_artifact_sha256"].items():
            self.assertEqual(preparation.episode_preparation._sha(Path(path)), expected)

    def test_insufficient_conditions_emits_stop_without_manifest(self):
        real_select = preparation.select_conditions
        def insufficient(*args):
            rows, selected = real_select(*args)
            return rows, selected[:1]
        with patch.object(preparation, "select_conditions", side_effect=insufficient):
            result = self.prepare()
        self.assertEqual(result["status"], "INSUFFICIENT_DISTINCT_DYNAMIC_FAILURES")
        self.assertIsNone(result["manifest_path"])
        self.assertFalse((self.output / "manifest.json").exists())

    def test_source_integrity_error_stops_before_selecting(self):
        with patch.object(preparation.episode_preparation, "prepare_episode_learning", side_effect=ValueError("integrity rejected")), \
                patch.object(preparation, "select_conditions", side_effect=AssertionError("Must not select")):
            with self.assertRaisesRegex(ValueError, "integrity rejected"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_excluded_paths_rejected_before_any_artifact_read(self):
        for excluded in ("E:/must-not-access", "/mnt/e/must-not-access", "runs/../bad"):
            for source, output in ((excluded, self.output), (SOURCE, excluded)):
                with self.subTest(source=source, output=output), \
                        patch.object(preparation.episode_preparation, "_load", side_effect=AssertionError("No read")):
                    with self.assertRaises(ValueError):
                        preparation.prepare_stability_learning(source, output)

    def test_cli_does_not_expose_test_path_escape(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as raised:
            preparation.main(["--episode-dir", str(SOURCE), "--output-dir", str(self.output), "--allow-test-root"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
