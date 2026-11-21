"""Read-only historical evidence plus synthetic fixtures, no API or fresh physics."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from orchestrator.python_stability_pose import support_preservation_scope, check_stability_pose_contract
from orchestrator.skill_program import step_program
from tools.python_support_fixture import PROGRAM
from tools.run_python_skill_cycle import (
    python_generation_packet, registered_parent_packet, contract_rejection_feedback,
)

ROOT = Path(__file__).absolute().parents[1]
RUN = ROOT / "runs/python_stability_threepart_20261020"
OLD = ROOT / "runs/python_stability_pose_20261020"

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

class SupportPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = read(RUN / "old_skill_replay/cases/debug_three_parts/program_request.json")
        cls.old = read(OLD / "generation_07/provider_proposal.json")["program"]

    def test_current_three_part_roles_preserve_path_and_need_fresh_verification(self):
        before = deepcopy(self.request)
        scope = support_preservation_scope(self.request)
        self.assertEqual(scope["all_other_nonbase_parts"], ["2", "3"])
        self.assertTrue(scope["fresh_path_verification_required"])
        self.assertFalse(scope["physical_success_implied"])
        result = step_program(PROGRAM, self.request)["response"]["choice"]
        prior = scope["previous_path_choice"]
        for key in ("part_id", "pose_id", "skill_id", "direction_id", "second_direction_id"):
            self.assertEqual(result[key], prior[key])
        self.assertEqual(self.request, before)
        with self.assertRaisesRegex(ValueError, "attempted_action_scope_mismatch"):
            support_preservation_scope(dict(self.request, max_grippers=2))

    def test_old_program_fails_new_branch_contract_and_human_fixture_is_equivariant(self):
        contract = dict(source_cad_vector=[0, 0, 1], source_success_evidence_id="synthetic-contract",
                        preserve_path_when_all_remaining_fit=True)
        old = check_stability_pose_contract(self.old, self.request, contract)
        new = check_stability_pose_contract(PROGRAM, self.request, contract)
        self.assertEqual(old["passed"], 0)
        self.assertEqual(new["passed"], 24)
        self.assertEqual(new["physical_calls"], 0)

    def test_absent_feedback_and_excess_support_capacity_abstain(self):
        req = deepcopy(self.request)
        req.update(episode_feedback=[], memory_references=[], evidence_ids=[])
        self.assertIsNone(support_preservation_scope(req))
        self.assertIsNone(step_program(PROGRAM, req)["response"])
        large = read(OLD / "generation_07/cases/debug_former_admission_000001/program_request.json")
        self.assertIsNone(support_preservation_scope(large))
        self.assertIsNone(step_program(PROGRAM, large)["response"])
        terminal = read(OLD / "generation_07/cases/debug_000006/program_request.json")
        self.assertIsNone(support_preservation_scope(terminal))

    def test_registered_parent_binds_debug_only_without_inheriting_authorization(self):
        manifest = read(RUN / "manifest.json")
        packet = python_generation_packet(manifest, "support-offline-test")
        result = registered_parent_packet(packet, manifest)
        self.assertEqual(len(result["validation_feedback"]), 5)
        self.assertTrue(all(r["phase"] == "debug" for r in result["validation_feedback"]))
        self.assertFalse(result["context"]["registered_parent"]["authorization_inherited"])
        bad = deepcopy(manifest)
        bad["registered_parent"]["result_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "result_hash_mismatch"):
            registered_parent_packet(packet, bad)
        bad = deepcopy(manifest)
        bad["registered_parent"]["cases"][0] = "admission_yaw_90"
        with self.assertRaisesRegex(ValueError, "admission_not_generation_input"):
            registered_parent_packet(packet, bad)

    def test_coordinate_rejection_feedback_is_offline_not_admission_or_authorization(self):
        feedback = contract_rejection_feedback(RUN / "generation_01", RUN / "api_budget.sqlite")
        self.assertEqual(feedback["kind"], "prior_synthetic_coordinate_rejection_NOT_PHYSICS")
        self.assertEqual(feedback["physical_calls"], 0)
        self.assertFalse(feedback["admission_feedback"])
        self.assertLess(feedback["passed"], feedback["checked"])
        with self.assertRaisesRegex(ValueError, "bound_live_contract_rejection_required"):
            contract_rejection_feedback(RUN / "generation_01", RUN / "wrong.sqlite")

if __name__ == "__main__":
    unittest.main()
