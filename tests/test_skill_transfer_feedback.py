"""Promotion integrity tests using saved logs; no new API or physics calls.

Successful tests create only temporary evidence copies. Mutations are injected
into the JSON reader, never into the original run artifacts.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prepare_skill_transfer_feedback as promotion
import run_skill_discovery as discovery

RUN = ROOT / "runs/skill_refinement_20261019"
TRANSFER = RUN / "transfer_live_01"
GENERATION = RUN / "live_03"


class SkillTransferFeedbackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="skill_transfer_promotion_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "new_bundle"
        self.manifest = self.output / "adaptive_manifest.json"
        original_checked = promotion.checked

        def checked(path, runs_only=False):
            # All original evidence remains inside the existing C workspace;
            # only this owned temporary root receives the test exception.
            path = Path(path)
            if path.is_absolute() and self.root in path.parents:
                resolved = path.resolve()
                if self.root not in resolved.parents:
                    raise ValueError("temporary_path_escape")
                return resolved
            return original_checked(path, runs_only)

        self.checked = checked
        guard = patch.object(promotion, "checked", side_effect=checked)
        guard.start()
        self.addCleanup(guard.stop)

    def prepare(self):
        return promotion.prepare_transfer_feedback(TRANSFER, GENERATION, self.output, self.manifest)

    def reject_mutation(self, path, mutate, expected=ValueError):
        original = promotion._load
        target = path.absolute()

        def load(candidate):
            value = original(candidate)
            if Path(candidate).absolute() == target:
                value = deepcopy(value)
                mutate(value)
            return value

        with patch.object(promotion, "_load", side_effect=load):
            with self.assertRaises(expected):
                self.prepare()
        self.assertFalse(self.output.exists(), "validation must precede output creation")

    def test_explicit_promotion_preserves_bytes_and_is_compatible_with_refinement(self):
        audit = self.prepare()
        self.assertTrue(audit["original_files_unchanged"])
        self.assertTrue(audit["adaptive_development"])
        self.assertFalse(audit["promoted_target_is_held_out"])
        self.assertEqual(audit["new_api_calls"], 0)
        self.assertEqual(audit["new_physical_calls"], 0)
        for record in audit["copies"]:
            self.assertEqual(Path(record["source_path"]).read_bytes(),
                             Path(record["bundle_path"]).read_bytes())
        manifest = promotion._load(self.manifest)
        old_manifest = promotion._load(GENERATION / "precommitted_cases.json")
        self.assertEqual(manifest["experience_report"], old_manifest["experience_report"])
        self.assertEqual(manifest["cases"][1:], [case for case in old_manifest["cases"]
                                               if case["phase"] == "admission"])
        spec = manifest["cases"][0]
        self.assertEqual(spec["source_task"], "07009")
        self.assertEqual(spec["phase"], "debug")
        self.assertEqual(spec["trigger_kind"], "path_failed")
        self.assertEqual(spec["attempt_index"], 3)
        step = promotion._load(self.output / "cases" / spec["case_id"] / "program_step.json")
        self.assertFalse(step["new_physical_execution"])
        self.assertTrue(step["replayed_from_saved_observations"])
        with patch.object(discovery, "checked", side_effect=self.checked), \
                patch.object(discovery, "read", side_effect=promotion._load), \
                patch.object(discovery, "sha", side_effect=promotion._sha):
            feedback = discovery.build_refinement_feedback(self.output,
                promotion._load(self.output / "result.json"),
                promotion._load(self.output / "precommitted_cases.json"))
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["phase"], "debug")
        self.assertEqual(feedback[0]["recovery"]["status"], "UNKNOWN")
        self.assertFalse(feedback[0]["recovery"]["path_success"])
        self.assertTrue(feedback[0]["recovery"]["bound_actual_scope_checked"])
        self.assertEqual(feedback[0]["interpreter_trace"]["response"]["choice"],
                         promotion._load(TRANSFER / "response.json")["choice"])

    def test_fixture_or_nonlive_proposal_is_rejected(self):
        self.reject_mutation(GENERATION / "provider_proposal.json",
                             lambda value: value.update(origin="injected_transport"))

    def test_fixture_transfer_is_rejected(self):
        self.reject_mutation(TRANSFER / "case_report.json",
                             lambda value: value.update(fixture_library=True))

    def test_different_generation_library_is_rejected(self):
        self.reject_mutation(TRANSFER / "precommitted_case.json",
                             lambda value: value.update(library_file_sha256="0" * 64))

    def test_unpreserved_source_or_library_is_rejected(self):
        self.reject_mutation(TRANSFER / "preservation.json",
                             lambda value: value.update(source_unchanged=False))

    def test_nonlive_selector_cannot_support_promotion(self):
        self.reject_mutation(TRANSFER / "selector_decisions.json",
                             lambda values: values[0]["response"].update(origin="injected_transport"))

    def test_stale_or_mock_worker_result_is_rejected(self):
        self.reject_mutation(TRANSFER / "recovery/physics/result.json",
                             lambda value: value.update(backend="fixture_not_actual_physics"))

    def test_changed_response_action_is_rejected(self):
        self.reject_mutation(TRANSFER / "response.json",
                             lambda value: value["choice"].update(second_direction_id="xp"))

    def test_late_failure_is_not_eligible_for_development_feedback(self):
        self.reject_mutation(TRANSFER / "recovery/physics/result.json",
                             lambda value: value.update(elapsed_seconds=31))

    def test_changed_trajectory_hash_is_rejected(self):
        self.reject_mutation(TRANSFER / "recovery/physics/result.json",
                             lambda value: value.update(trajectory_sha256="0" * 64))

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "untouched.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "new_run_required"):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_manifest_cannot_escape_new_bundle(self):
        with self.assertRaisesRegex(ValueError, "new_manifest_must_be_inside_new_run"):
            promotion.prepare_transfer_feedback(TRANSFER, GENERATION, self.output,
                                                self.root / "outside.json")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
