"""Offline generation contracts; fake transport is not actual model/physics."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.python_skill_discovery import PythonSkillDiscoveryService, INSTRUCTIONS
from orchestrator.python_skill_evidence import generation_view
from worker.stability_diagnostics import StabilityObservation
from tests.test_skill_discovery import synthesis, config
from tools.run_python_skill_cycle import python_generation_packet


class PythonStabilityGenerationTests(unittest.TestCase):
    def test_compact_transition_projection_retains_all_pairs_without_causal_claim(self):
        from orchestrator.initial_contact_trace_summary import summarize_initial_contact_transitions
        source = Path(__file__).absolute().parents[1] / (
            "runs/python_skill_stability_reuse_20261020/live_full_01/planner/planner_report.json")
        summary = summarize_initial_contact_transitions(json.loads(source.read_text(encoding="utf-8")))
        original = deepcopy(summary)
        packet = dict(context=dict(python_failure_scope="remaining_body_stability_failed",
                                   initial_contact_observed_transitions=summary))
        projected = generation_view(packet)["context"]["initial_contact_observed_transitions"]
        self.assertEqual(summary, original)
        self.assertEqual(len(projected["pairs"]), len(summary["pairs"]))
        self.assertEqual(projected["trace_attempt_count"], summary["trace_attempt_count"])
        self.assertNotIn("geometry_contexts", projected)
        for raw, row in zip(summary["pairs"], projected["pairs"]):
            self.assertEqual(row["after"]["choice"], raw["after"]["choice"])
            self.assertEqual(row["observed_role_relationships"]["after_support_contains_before_mover"],
                             raw["before"]["choice"]["part_id"] in raw["after"]["choice"]["support_ids"])
            self.assertFalse(row["causal_effect_established"])

    def test_stability_prompt_does_not_inherit_path_vector_prescription(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PythonSkillDiscoveryService(config(enabled=False, max_calls=0, max_usd=0),
                                                   Path(directory) / "unused.sqlite")
            packet = synthesis()
            legacy = service.preview(packet)["body"]
            packet["context"]["python_failure_scope"] = "remaining_body_stability_failed"
            body = service.preview(packet)["body"]
            self.assertEqual(legacy["instructions"], INSTRUCTIONS)
            self.assertIn("initial_disconnected", body["instructions"])
            self.assertIn("NOT physical success", body["instructions"])
            self.assertNotIn("preserving that recovery's successful ordered", body["instructions"])
            self.assertEqual(body["text"], legacy["text"])
            packet["context"]["python_failure_scope"] = "invented"
            with self.assertRaisesRegex(ValueError, "unsupported_python_failure_scope"):
                service.preview(packet)

    def test_refinement_retains_measured_diagnostic_without_invented_cause(self):
        diagnostic = StabilityObservation()
        diagnostic.checker_count = 1
        diagnostic.data["initial_disconnected_parts"] = ["observed"]
        diagnostic.returned(False, ["observed"])
        packet = dict(context={}, validation_feedback=[dict(phase="debug",
            original_trigger=dict(stability_diagnostics=diagnostic.data,
                                  stability_mode="one_exact_selected_support_configuration"),
            recovery=None)])
        original = deepcopy(packet)
        projected = generation_view(packet)
        self.assertEqual(packet, original)
        raw = projected["validation_feedback"][0]["original_trigger"]
        self.assertEqual(raw["stability_diagnostics"]["phase"], "initial_disconnected")
        self.assertNotIn("failure_cause", raw)
        packet["validation_feedback"][0]["phase"] = "admission"
        with self.assertRaisesRegex(ValueError, "debug_feedback_only"):
            generation_view(packet)
        packet["validation_feedback"][0]["phase"] = "debug"
        diagnostic.data["extra_physics_calls"] = 1
        with self.assertRaises(ValueError):
            generation_view(packet)

    def test_unmeasured_or_missing_memory_rejected_before_generation(self):
        manifest = dict(python_failure_scope="remaining_body_stability_failed")
        with self.assertRaisesRegex(ValueError, "requires_bound_memory"):
            python_generation_packet(manifest, "fixture")
        manifest["experience_memory"] = "fixture-only"
        with patch("tools.run_python_skill_cycle.synthesis_packet",
                   return_value=dict(context={}, evidence=[])):
            with self.assertRaisesRegex(ValueError, "requires_measured_failure"):
                python_generation_packet(manifest, "fixture")

    def test_bound_measured_rows_select_new_scope_without_program_or_admission(self):
        row = dict(evidence_id="fixture", physically_checked_stability_failure=True,
            path_success=True, stability_success=False, deadline_exceeded=False,
            stability_diagnostics=dict(phase="dynamic_failure"))
        manifest = dict(python_failure_scope="remaining_body_stability_failed", experience_memory="fixture",
                        experience_report="development-only.json", cases=[dict(report="admission-secret")])
        with patch("tools.run_python_skill_cycle.synthesis_packet",
                   return_value=dict(context={}, evidence=[row])), patch(
                       "tools.run_python_skill_cycle.read", return_value={"source": "bound-development"}
                   ) as read_source, patch("tools.run_python_skill_cycle.checked", side_effect=lambda p, _: p), patch(
                       "tools.run_python_skill_cycle.summarize_initial_contact_transitions",
                       return_value={"causal_effect_established": False}
                   ) as summarize:
            packet = python_generation_packet(manifest, "fixture")
        self.assertEqual(read_source.call_count, 1)
        self.assertEqual(read_source.call_args.args[0].name, "development-only.json")
        summarize.assert_called_once_with({"source": "bound-development"})
        self.assertEqual(packet["context"]["measured_stability_evidence_ids"], ["fixture"])
        self.assertFalse(packet["context"]["admission_results_exposed"])
        self.assertNotIn("program", packet)


if __name__ == "__main__":
    unittest.main()
