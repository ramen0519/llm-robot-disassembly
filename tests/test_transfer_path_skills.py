"""Synthetic cross-task retrieval contracts; API=0 and actual physics=0."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from orchestrator import transfer_path_skills as transfer
from worker.task_memory import canonical, digest
from test_path_skill_memory import SyntheticRun


class TransferPathSkillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_transfer_skill_", dir=ROOT / "runs")
        self.addCleanup(self.temp.cleanup)
        self.fixture = SyntheticRun(Path(self.temp.name))
        self.snapshot = self.fixture.freeze()
        self.target = deepcopy(self.fixture.context)
        self.target["task"].update(assembly_id="synthetic-other-assembly")
        self.target["task"]["input_sha256"]["a.obj"] = "9" * 64
        self.target["geometry_features"]["a"] = {"extents": [9, 8, 7], "volume": 10}
        self.current_scope = deepcopy(self.fixture.failed_scope)
        self.current_scope["task_context"] = deepcopy(self.target)

    def features(self):
        return {"memory_context": deepcopy(self.target), "moving_part_id": "a",
                "registered_skill": {"skill_id": transfer.source_memory.SKILL,
                                     "code_sha256": self.fixture.code_sha},
                "constraint_references": [{"origin": "current_episode_measurement",
                                           "source_scope": deepcopy(self.current_scope)}],
                "first_axis": "+X", "allowed_second_axes": ["+Y", "-Y", "+Z", "-Z"], "switch_step": 50}

    def card(self, current=None):
        cards = transfer.matching_cards(self.snapshot, self.features() if current is None else current)
        self.assertEqual(len(cards), 1)
        return cards[0]

    def test_load_retains_existing_snapshot_and_checks_source_hashes(self):
        self.assertEqual(transfer.load(self.fixture.output), self.snapshot)
        self.fixture.tree.write_bytes(b"changed synthetic tree")
        with self.assertRaisesRegex(ValueError, "frozen_skill_source_file_changed"):
            transfer.load(self.fixture.output)

    def test_cross_assembly_card_is_reference_not_current_verdict(self):
        card = self.card()
        self.assertEqual(card["applicability"], "cross_assembly_procedure_reference")
        self.assertTrue(card["reference_only"])
        self.assertTrue(card["force_new_physics"])
        for key in ("current_attempt_verified", "generalized", "hard_pruning", "reuse_verdict",
                    "source_validation_applies_to_target"):
            self.assertFalse(card[key])
        self.assertEqual(card["validation"]["path"], "PASS")
        self.assertEqual(card["transfer_semantics"]["target_path_verdict"], "UNKNOWN")
        self.assertEqual(card["transfer_semantics"]["target_stability_verdict"], "UNKNOWN")

    def test_source_and_target_geometry_remain_separate(self):
        card = self.card()
        self.assertEqual(card["source_geometry"]["observations"], {})
        self.assertEqual(card["target_context"]["moving_geometry_observations"], {"extents": [9, 8, 7], "volume": 10})
        self.assertNotEqual(card["source_scope"]["task_context"]["task"]["assembly_id"],
                            card["target_context"]["task"]["assembly_id"])
        self.assertIsNone(card["current_part_id_mapping"])
        self.assertIsNone(card["transfer_semantics"]["current_part_id_mapping"])

    def test_same_textual_part_id_does_not_establish_correspondence(self):
        card = self.card()
        self.assertEqual(card["source_geometry"]["source_part_id"], "a")
        self.assertEqual(card["target_context"]["moving_part_id"], "a")
        self.assertIsNone(card["current_part_id_mapping"])
        self.assertIn("unverified_part_correspondence", card["condition_mismatches"])

    def test_source_axes_not_transferred_or_recommended(self):
        card = self.card()
        semantics = card["transfer_semantics"]
        self.assertEqual(semantics["source_axis_names"], ["+X", "+Z"])
        self.assertEqual(card["validated_schedule"], self.fixture.actions)
        for key in ("axis_mapping", "coordinate_transform", "unit_scale_mapping",
                    "axis_labels_have_same_geometric_role", "target_selected_schedule", "target_axis_recommendation"):
            self.assertIsNone(semantics[key])
        self.assertEqual(semantics["source_schedule_role"], "source_validation_only_not_target_instruction")
        self.assertFalse(semantics["trajectory_copy_allowed"])
        self.assertFalse(semantics["past_pass_reused"])

    def test_source_constraint_ids_do_not_become_target_constraints(self):
        card = self.card()
        self.assertEqual(card["failed_constraint_ids"], self.fixture.constraint_ids)
        self.assertFalse(card["transfer_semantics"]["source_failed_constraint_ids_are_target_constraints"])
        self.assertFalse(card["transfer_semantics"]["target_generalization_verified"])

    def test_source_scope_and_provenance_hashes_are_retained(self):
        card = self.card()
        original = self.snapshot["records"][0]
        for key in ("skill_evidence_id", "controller_sha256", "source_scope", "source_scope_sha256",
                    "source_trajectory_sha256", "validated_schedule"):
            self.assertEqual(card[key], original[key])
        self.assertEqual(card["source_provenance"]["record_sha256"], original["record_sha256"])
        self.assertEqual(card["source_provenance"]["snapshot_content_sha256"], self.snapshot["content_sha256"])
        self.assertEqual(card["source_provenance"]["file_sha256"],
                         {key: value["sha256"] for key, value in original["source_files"].items()})
        self.assertNotIn("source_files", card)

    def test_same_assembly_delegated_to_original_matcher(self):
        self.assertEqual(transfer.matching_cards(self.snapshot, self.fixture.context), [])
        changed = deepcopy(self.fixture.context)
        changed["task"]["input_sha256"]["a.obj"] = "9" * 64
        self.assertEqual(transfer.matching_cards(self.snapshot, changed), [])

    def test_controller_mismatch_excludes_reference(self):
        current = self.features()
        current["registered_skill"]["code_sha256"] = "8" * 64
        current["constraint_references"][0]["source_scope"]["physics"]["source_and_binary_hashes"]["controller_sha256"] = "8" * 64
        self.assertEqual(transfer.matching_cards(self.snapshot, current), [])

    def test_missing_controller_only_yields_incomplete_reference(self):
        card = self.card(self.target)
        self.assertIn("current_controller_not_supplied", card["condition_mismatches"])
        self.assertIn("current_scope_not_supplied", card["condition_mismatches"])
        self.assertIsNone(card["target_context"]["registered_controller_sha256"])
        self.assertIsNone(card["target_context"]["moving_part_id"])
        self.assertFalse(card["current_attempt_verified"])

    def test_candidate_features_support_no_selected_mover(self):
        candidate = {"memory_context": deepcopy(self.target),
                     "cad": {"task": deepcopy(self.target["task"]),
                             "parts": deepcopy(self.target["geometry_features"])}}
        card = self.card(candidate)
        self.assertEqual(card["target_context"]["assembled_parts"], ["a", "b", "base"])
        self.assertIsNone(card["target_context"]["moving_part_id"])
        self.assertIsNone(card["target_context"]["observed_action_schedule"])
        self.assertIsNone(card["target_context"]["first_axis"])

    def test_full_measured_scope_supported(self):
        card = self.card(self.current_scope)
        self.assertEqual(card["target_context"]["moving_part_id"], "a")
        self.assertEqual(card["target_context"]["scope_sha256"], digest(self.current_scope))
        self.assertEqual(card["current_observed_action_schedule"], self.current_scope["action_schedule"])
        self.assertFalse(card["action_schedule_matches_current_observation"])

    def test_different_pose_physics_and_units_not_assumed_equivalent(self):
        current = self.features()
        measured = current["constraint_references"][0]["source_scope"]
        measured["pose"] = [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        measured["physics"]["force_mag"] = 20
        measured["frame"] = "synthetic_different_recorded_frame"
        card = self.card(current)
        self.assertIn("scope_pose", card["condition_mismatches"])
        self.assertIn("scope_physics", card["condition_mismatches"])
        self.assertNotEqual(card["source_geometry"]["path_state_frame"], card["target_context"]["path_state_frame"])
        self.assertIsNone(card["transfer_semantics"]["coordinate_transform"])

    def test_changed_target_domain_does_not_copy_source_axis(self):
        current = self.features()
        current.update(first_axis="+Y", allowed_second_axes=["+X", "-X", "+Z", "-Z"], switch_step=60)
        current["constraint_references"][0]["source_scope"]["action_schedule"] = [{"step": 0, "action": [0, 1, 0]}]
        card = self.card(current)
        self.assertEqual(card["target_context"]["first_axis"], "+Y")
        self.assertEqual(card["target_context"]["switch_step"], 60)
        self.assertEqual(card["transfer_semantics"]["source_switch_step"], 50)
        self.assertEqual(card["validated_schedule"][0]["action"], [1, 0, 0])
        self.assertIsNone(card["transfer_semantics"]["target_selected_schedule"])

    def test_cards_and_inputs_not_mutated(self):
        current = self.features()
        source_before, current_before = canonical(self.snapshot), canonical(current)
        card = self.card(current)
        card["target_context"]["task"]["assembly_id"] = "edited-copy"
        card["source_scope"]["part_move"] = "edited-copy"
        self.assertEqual(canonical(self.snapshot), source_before)
        self.assertEqual(canonical(current), current_before)

    def test_limit_and_invalid_limit_types(self):
        self.assertEqual(transfer.matching_cards(self.snapshot, self.features(), max_records=0), [])
        self.assertEqual(len(transfer.matching_cards(self.snapshot, self.features(), max_records=1)), 1)
        for value in (True, False, -1, 5, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                transfer.matching_cards(self.snapshot, self.features(), max_records=value)

    def test_invalid_target_object_or_context_rejected(self):
        for current in (None, [], {}, {"memory_context": {}}, {"memory_context": self.target, "moving_part_id": "missing"},
                        {"memory_context": self.target, "moving_part_id": "base"}):
            with self.subTest(current=current), self.assertRaises(ValueError):
                transfer.matching_cards(self.snapshot, current)

    def test_registry_and_measured_code_must_agree(self):
        current = self.features()
        current["registered_skill"]["code_sha256"] = "8" * 64
        with self.assertRaisesRegex(ValueError, "target_controller_disagrees_with_measured_scope"):
            self.card(current)

    def test_invalid_registry_and_constraint_reference_rejected(self):
        for change in ({"registered_skill": []}, {"registered_skill": {"skill_id": "other", "code_sha256": "0" * 64}},
                       {"constraint_references": None}, {"constraint_references": [None]}):
            current = self.features()
            current.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.card(current)

    def test_conflicting_scope_and_moving_id_rejected(self):
        current = self.features()
        current["moving_part_id"] = "b"
        with self.assertRaisesRegex(ValueError, "target_moving_part_disagrees_with_scope"):
            self.card(current)
        current = self.features()
        another = deepcopy(current["constraint_references"][0])
        another["source_scope"]["budget"] += 1
        current["constraint_references"].append(another)
        with self.assertRaisesRegex(ValueError, "conflicting_current_failure_scopes"):
            self.card(current)

    def test_CAD_context_disagreement_rejected(self):
        current = self.features()
        current["cad"] = {"task": deepcopy(self.target["task"]), "parts": {"a": {"fabricated": 1}}}
        with self.assertRaisesRegex(ValueError, "target_CAD_disagrees_with_context"):
            self.card(current)
        current["cad"]["parts"] = {"a": deepcopy(self.target["geometry_features"]["a"])}
        with self.assertRaisesRegex(ValueError, "target_CAD_disagrees_with_current_state"):
            self.card(current)

    def test_invalid_action_domain_rejected(self):
        for change in ({"first_axis": "foo"}, {"allowed_second_axes": []}, {"allowed_second_axes": ["+X"]},
                       {"allowed_second_axes": ["+Z", "+Z"]}, {"switch_step": True}, {"switch_step": 0},
                       {"switch_step": None}):
            current = self.features()
            current.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "target_action_domain_invalid"):
                self.card(current)

    def test_initial_action_must_match_measured_scope(self):
        current = self.features()
        current["constraint_references"][0]["source_scope"]["action_schedule"][0]["action"] = [0, 0, 1]
        with self.assertRaisesRegex(ValueError, "target_initial_axis_disagrees_with_measured_attempt"):
            self.card(current)

    def test_invalid_source_schedule_rejected(self):
        for schedule in ([], [{"step": 0, "action": [1, 0, 0]}],
                         [{"step": 0, "action": [True, 0, 0]}, {"step": 50, "action": [0, 0, 1]}],
                         [{"step": 0, "action": [1, 0, 0]}, {"step": 50, "action": [-1, 0, 0]}],
                         [{"step": 0, "action": [1, 0, 0]}, {"step": True, "action": [0, 0, 1]}]):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                transfer._source_schedule(schedule)

    def test_snapshot_tamper_rejected(self):
        self.snapshot["records"][0]["generalized"] = True
        with self.assertRaises(ValueError):
            transfer.matching_cards(self.snapshot, self.features())

    def test_retrieval_performs_no_API_physics_or_source_file_reads(self):
        with patch("subprocess.run", side_effect=AssertionError("no physics subprocess")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("no API")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("load already verified source files")):
            self.card()

    def test_forbidden_load_paths_rejected_before_IO(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("no filesystem access")), \
             patch.object(Path, "open", side_effect=AssertionError("no filesystem access")):
            for path in ("E:/never-access", "/mnt/e/never-access", "C:/dev/../never-access"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    transfer.load(path)


if __name__ == "__main__":
    unittest.main()
