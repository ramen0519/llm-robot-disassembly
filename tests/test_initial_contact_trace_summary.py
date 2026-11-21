"""Read-only saved-trace projections; no fresh physics or LLM claims."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from orchestrator.initial_contact_trace_summary import summarize_initial_contact_transitions

ROOT = Path(__file__).absolute().parents[1]
SOURCE = ROOT / "runs/automatic_learning_cycle_20261019/live_01/next_episode/planner/planner_report.json"


class InitialContactTraceSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(SOURCE.read_text(encoding="utf-8"))

    def test_chronological_same_pose_different_mover_pairs_and_denominators(self):
        result = summarize_initial_contact_transitions(self.source)
        self.assertEqual(result["trace_attempt_count"], 12)
        self.assertEqual(result["eligible_observation_count"], 12)
        self.assertEqual(result["excluded_attempts"], [])
        self.assertEqual(result["phase_denominator"], {"initial_disconnected": 6, "dynamic_failure": 3,
                                                     "passed": 2, "NOT_OBSERVED": 1})
        self.assertEqual(result["initial_failure_count"], 6)
        self.assertEqual([(p["failure_attempt_index"], p["success_attempt_index"]) for p in result["pairs"]],
                         [(1, 6), (7, 10), (8, 10), (9, 10)])
        self.assertEqual([p["attempt_index"] for p in result["unmatched_initial_failures"]], [3, 11])
        self.assertEqual(result["pairs"][0]["changed_choice_fields"],
                         ["part_id", "skill_id", "direction_id", "second_direction_id"])
        for pair in result["pairs"]:
            self.assertTrue(pair["success_mover_in_observed_initial_disconnected_set"])
            self.assertFalse(pair["causal_effect_established"])
            self.assertFalse(pair["generalization_established"])
            self.assertIsNone(pair["recommended_action"])
            self.assertIsNone(pair["current_part_id_mapping"])

    def test_geometry_is_original_recorded_cad_values_not_recomputed_rank(self):
        result = summarize_initial_contact_transitions(self.source)
        self.assertEqual(len(result["geometry_contexts"]), 3)
        for item in result["observations"]:
            cad = result["geometry_contexts"][item["geometry_context_id"]]
            request = next(d["request"] for d in self.source["decisions"]
                           if d["request"]["request_id"] == item["request_id"])
            self.assertEqual(cad["parts"], request["geometry"]["parts"])
            self.assertEqual(cad["units"], request["geometry"]["relations"]["units"])
            self.assertFalse(cad["newly_calculated_geometry"])
        self.assertEqual(result["new_geometry_calculations"], 0)
        self.assertEqual(result["new_physical_calls"], 0)
        self.assertEqual(result["new_api_calls"], 0)

    def test_mismatched_cad_request_excluded_without_discarding_denominator(self):
        report = deepcopy(self.source)
        report["decisions"][0]["request"]["geometry"]["parts"]["1"]["volume"] = 900
        result = summarize_initial_contact_transitions(report)
        self.assertEqual(result["trace_attempt_count"], 12)
        self.assertEqual(result["excluded_attempt_count"], 1)
        self.assertEqual(result["excluded_attempts"][0]["reason"], "recorded_cad_feature_binding_mismatch")
        self.assertFalse(any(p["failure_attempt_index"] == 1 for p in result["pairs"]))

    def test_decision_action_identity_mismatch_rejected(self):
        report = deepcopy(self.source)
        report["attempts"][0]["action_id"] = "not-the-recorded-action"
        result = summarize_initial_contact_transitions(report)
        self.assertEqual(result["excluded_attempts"][0]["attempt_index"], 1)
        self.assertFalse(any(p["failure_attempt_index"] == 1 for p in result["pairs"]))

    def test_late_failure_cannot_become_recovery_evidence(self):
        report = deepcopy(self.source)
        report["attempts"][0]["deadline_exceeded"] = True
        result = summarize_initial_contact_transitions(report)
        self.assertEqual(result["excluded_attempts"][0]["reason"], "late_or_unverified_deadline")
        self.assertEqual(result["total_pair_count"], 3)

    def test_initial_contact_branch_cannot_contain_dynamic_steps(self):
        report = deepcopy(self.source)
        report["attempts"][0]["raw"]["stability_diagnostics"]["physics_steps"] = 2
        result = summarize_initial_contact_transitions(report)
        self.assertEqual(result["excluded_attempts"][0]["reason"], "unbound_initial_contact_diagnostic")

    def test_missing_decision_remains_excluded_not_synthetic(self):
        report = deepcopy(self.source)
        report["decisions"].pop(0)
        result = summarize_initial_contact_transitions(report)
        self.assertEqual(result["excluded_attempts"][0]["reason"], "missing_recorded_decision")

    def test_duplicate_request_is_not_silently_overwritten(self):
        report = deepcopy(self.source)
        report["decisions"].append(deepcopy(report["decisions"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate_recorded_request"):
            summarize_initial_contact_transitions(report)

    def test_pair_limit_keeps_total_and_all_observation_denominators(self):
        result = summarize_initial_contact_transitions(self.source, max_pairs=1)
        self.assertEqual(result["total_pair_count"], 4)
        self.assertEqual(result["emitted_pair_count"], 1)
        self.assertEqual(result["omitted_pair_count"], 3)
        self.assertEqual(len(result["observations"]), 12)

    def test_invalid_limits_rejected(self):
        for limit in (0, 33, True):
            with self.assertRaises(ValueError):
                summarize_initial_contact_transitions(self.source, max_pairs=limit)
        with self.assertRaises(ValueError):
            summarize_initial_contact_transitions(self.source, max_attempts=11)

    def test_empty_trace_has_no_fabricated_recovery(self):
        result = summarize_initial_contact_transitions({"attempts": [], "decisions": []})
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["total_pair_count"], 0)
        self.assertFalse(result["prescriptive_policy_generated"])

    def test_source_input_is_unchanged(self):
        report = deepcopy(self.source)
        summarize_initial_contact_transitions(report)
        self.assertEqual(report, self.source)


if __name__ == "__main__":
    unittest.main()
