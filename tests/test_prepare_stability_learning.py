"""Initial-contact preparation checks using saved artifacts, not fresh physics."""
from copy import deepcopy
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
LATEST = ROOT / "runs/stability_skill_learning_20261019/live_02/next_episode"


def saved_inputs(source):
    report = json.loads((source / "planner/planner_report.json").read_text(encoding="utf-8"))
    physicals, considered = [], []
    for i, attempt in enumerate(report["attempts"], 1):
        directory = source / "planner" / ("attempt_%06d" % i)
        physicals.append(json.loads((directory / "request.json").read_text(encoding="utf-8")))
        considered.append({"attempt_index": i, "eligible": not attempt["accepted_for_transition"],
            "reason": "accepted_action_not_a_failure" if attempt["accepted_for_transition"] else "observed_failure",
            "memory_evidence_id": "pure-classification-test-only", "raw_result_path": str(directory / "result.json"),
            "raw_result_sha256": preparation.episode_preparation._sha(directory / "result.json")})
    return report, physicals, considered


class InitialContactClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.physicals, cls.considered = saved_inputs(SOURCE)

    def classify(self, attempt=None, physical=None, checked=None):
        return preparation.classify_condition(attempt or deepcopy(self.report["attempts"][0]),
            checked or deepcopy(self.considered[0]), physical or deepcopy(self.physicals[0]),
            failure_scope="initial_disconnected")

    def test_first_three_and_complete_failure_denominator(self):
        rows, selected = preparation.select_conditions(self.report["attempts"], self.considered,
            self.physicals, failure_scope="initial_disconnected")
        self.assertEqual([r["attempt_index"] for r in selected], [1, 3, 7])
        self.assertEqual([r["phase"] for r in selected], ["debug", "admission", "admission"])
        self.assertEqual(len({r["condition_hash"] for r in selected}), 3)
        self.assertEqual(len(rows), 12)
        self.assertEqual(sum(r["stability_failure_observed"] for r in rows), 9)
        self.assertEqual(sum(r["diagnostic_phase"] == "initial_disconnected" for r in rows), 6)
        self.assertEqual(sum(r["eligible"] for r in rows), 6)
        self.assertEqual(sum(r["selected"] for r in rows), 3)
        self.assertTrue(all(r["disconnected_count_exceeds_support_budget"] for r in selected))
        self.assertTrue(all(r["excluded_for_support_budget"] is False for r in selected))

    def test_more_than_support_budget_is_kept_as_observed_difficulty(self):
        row = self.classify()
        self.assertTrue(row["eligible"])
        self.assertEqual(row["observed_initial_disconnected_parts"], ["1", "2", "3", "4"])
        self.assertEqual(row["available_support_budget"], 2)
        self.assertTrue(row["disconnected_count_exceeds_support_budget"])
        self.assertFalse(row["dynamic_fall_observed"])
        self.assertFalse(row["all_alternative_actions_impossible"])

    def test_two_disconnected_parts_also_kept_without_selecting_easy_case_first(self):
        row = self.classify(self.report["attempts"][10], self.physicals[10], self.considered[10])
        self.assertTrue(row["eligible"])
        self.assertFalse(row["disconnected_count_exceeds_support_budget"])

    def test_dynamic_failure_is_not_reclassified_as_initial(self):
        row = self.classify(self.report["attempts"][1], self.physicals[1], self.considered[1])
        self.assertFalse(row["eligible"])
        self.assertEqual(row["reason"], "non_initial_contact_stability_failure")

    def test_nonzero_physics_and_instrumentation_errors_excluded(self):
        for key, value in (("physics_steps", 1), ("instrumentation_errors", ["ValueError"]),
                           ("instrumentation_errors", None), ("check_success_returned", False)):
            with self.subTest(key=key, value=value):
                attempt = deepcopy(self.report["attempts"][0])
                attempt["raw"]["stability_diagnostics"][key] = value
                row = self.classify(attempt)
                self.assertFalse(row["eligible"])
                self.assertEqual(row["reason"], "incomplete_initial_contact_observation")

    def test_upstream_returned_ids_must_match_projected_diagnostic(self):
        attempt = deepcopy(self.report["attempts"][0])
        attempt["raw"]["raw_upstream"]["parts_fall"] = ["1"]
        self.assertEqual(self.classify(attempt)["reason"], "initial_contact_returned_parts_binding_mismatch")

    def test_reported_disconnected_ids_must_be_current_free_parts(self):
        for mutation in ("moving", "support", "base"):
            with self.subTest(mutation=mutation):
                attempt = deepcopy(self.report["attempts"][0])
                physical = deepcopy(self.physicals[0])
                if mutation == "moving":
                    attempt["raw"]["executed"]["stability_moving_parts"] = ["1"]
                elif mutation == "support":
                    physical["choice"]["support_ids"] = ["1"]
                else:
                    physical["context"]["base_part"] = "1"
                row = self.classify(attempt, physical)
                self.assertFalse(row["eligible"])
                self.assertEqual(row["reason"], "initial_contact_parts_not_current_free_parts")

    def test_observed_contact_graph_must_support_zero_degree_report(self):
        for graph, reason in ((None, "initial_contact_graph_evidence_invalid"),
                              ({"nodes": ["1", "2", "3", "4"], "edges": [["1", "2"]]},
                               "initial_contact_graph_returned_parts_mismatch")):
            with self.subTest(graph=graph):
                attempt = deepcopy(self.report["attempts"][0])
                attempt["raw"]["stability_diagnostics"]["initial_contact_graph"] = graph
                row = self.classify(attempt)
                self.assertFalse(row["eligible"])
                self.assertEqual(row["reason"], reason)

    def test_late_and_missing_diagnostic_not_revived(self):
        checked = dict(self.considered[0], eligible=False, reason="deadline_exceeded_observation_not_eligible")
        self.assertEqual(self.classify(checked=checked)["reason"], checked["reason"])
        attempt = deepcopy(self.report["attempts"][0])
        attempt["raw"]["stability_diagnostics"] = None
        self.assertEqual(self.classify(attempt)["reason"], "missing_stability_diagnostic")

    def test_condition_deduplication_ignores_alias_or_seed(self):
        attempts = [deepcopy(self.report["attempts"][0]) for _ in range(3)]
        physicals = [deepcopy(self.physicals[0]) for _ in range(3)]
        physicals[1]["context"]["seed"] = 1234
        physicals[2]["context"]["seed"] = 999
        checked = [dict(self.considered[0], attempt_index=i) for i in range(1, 4)]
        rows, selected = preparation.select_conditions(attempts, checked, physicals,
            failure_scope="initial_disconnected")
        self.assertEqual(len(selected), 1)
        self.assertEqual([r["reason"] for r in rows[1:]], ["duplicate_physical_trigger_condition"] * 2)

    def test_default_dynamic_policy_still_selects_two_four_five(self):
        rows, selected = preparation.select_conditions(self.report["attempts"], self.considered, self.physicals)
        self.assertEqual([r["attempt_index"] for r in selected], [2, 4, 5])
        self.assertEqual(sum(r["eligible"] for r in rows), 3)


class InitialContactPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="initial_contact_artifact_checks_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).absolute()
        self.output = self.root / "prepared"

    def prepare(self, source=SOURCE):
        return preparation.prepare_stability_learning(source, self.output,
            failure_scope="initial_disconnected", allow_test_root=True)

    def test_full_integrity_manifest_original_memory_and_expected_phase(self):
        before = preparation.episode_preparation._sha(SOURCE / "action_memory_frozen.json")
        with patch("subprocess.Popen", side_effect=AssertionError("No simulation or API subprocess")):
            result = self.prepare()
        self.assertEqual(result["status"], "PREPARED")
        self.assertEqual(result["failure_scope"], "initial_disconnected")
        self.assertEqual(result["selection_policy"], preparation.INITIAL_POLICY)
        self.assertEqual(result["selection_counts"], dict(all_attempts=12, all_stability_failures=9,
            initial_disconnection=6, dynamic_failure=3, eligible_distinct_conditions=6, selected_conditions=3))
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["synthesis_failure_scope"], "initial_disconnected")
        self.assertEqual([c["attempt_index"] for c in manifest["cases"]], [1, 3, 7])
        self.assertTrue(all(c["trigger_kind"] == "stability_failed" for c in manifest["cases"]))
        self.assertTrue(all(c["expected_diagnostic_phase"] == "initial_disconnected" for c in manifest["cases"]))
        self.assertTrue(all("trigger_override" not in c for c in manifest["cases"]))
        self.assertEqual(manifest["frozen_episode_memory"]["file_sha256"], before)
        self.assertFalse(any("support_budget" in s for s in manifest["scope_exclusions"]))
        self.assertEqual(result["new_api_calls"], 0)
        self.assertEqual(result["new_physical_calls"], 0)
        self.assertFalse(result["target_is_held_out"])
        for path, expected in result["original_artifact_sha256"].items():
            self.assertEqual(preparation.episode_preparation._sha(Path(path)), expected)

    def test_latest_episode_two_conditions_cannot_be_promoted_to_three(self):
        result = self.prepare(LATEST)
        self.assertEqual(result["status"], "INSUFFICIENT_DISTINCT_INITIAL_CONTACT_FAILURES")
        self.assertEqual([r["attempt_index"] for r in result["selected"]], [2, 3])
        self.assertIsNone(result["manifest_path"])
        self.assertFalse((self.output / "manifest.json").exists())

    def test_unknown_scope_rejected_before_any_filesystem_read(self):
        with patch.object(preparation.episode_preparation, "_load", side_effect=AssertionError("No read")):
            with self.assertRaisesRegex(ValueError, "unsupported_stability_learning_failure_scope"):
                preparation.prepare_stability_learning(SOURCE, self.output, failure_scope="invented_phase")

    def test_forbidden_paths_rejected_before_artifact_access(self):
        for excluded in ("E:/forbidden", "/mnt/e/forbidden", "runs/../forbidden"):
            with self.subTest(path=excluded), \
                    patch.object(preparation.episode_preparation, "_load", side_effect=AssertionError("No read")):
                with self.assertRaises(ValueError):
                    preparation.prepare_stability_learning(excluded, self.output,
                        failure_scope="initial_disconnected")

    def test_exclusive_output_preserves_existing_files(self):
        self.output.mkdir()
        sentinel = self.output / "preserve.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
