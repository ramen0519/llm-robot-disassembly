"""Read-only evidence and synthetic coordinate tests; no API/fresh physics."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from orchestrator.python_stability_pose import (
    stability_source_contract, synthetic_stability_request, check_stability_pose_contract,
    failed_debug_requirements, check_stability_revision,
)
from tools.run_python_skill_cycle import (
    python_generation_packet, development_parent_packet, refinement_packet, add_development_probes,
)
from orchestrator.python_skill_discovery import PythonSkillDiscoveryService
from orchestrator.skill_discovery import MAX_LOCAL_EVIDENCE_BYTES
from orchestrator.python_skill_evidence import generation_view

ROOT = Path(__file__).absolute().parents[1]
RUN = ROOT / "runs/python_stability_pose_20261020"
OLD = ROOT / "runs/python_stability_generation_20261020/generation_05"
LEDGER = ROOT / "runs/python_skill_full_07009_20261020/api_budget.sqlite"

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

HANDWRITTEN_CONTRACT_FIXTURE = dict(language="python_policy_v1", source="""
def propose(ctx):
    last=ctx["last"]
    if last is None or not last["initial_disconnected"]:
        return None
    old=last["choice"]
    m=last["pose_matrix"]
    best=None
    score=-2
    for name in ctx["directions"]:
        v=ctx["directions"][name]
        dot=v[0]*m[0][2]+v[1]*m[1][2]+v[2]*m[2][2]
        if dot>score:
            best=name
            score=dot
    for part in ctx["candidate_ids"]:
        if part!=old["part_id"]:
            c={"part_id":part,"pose_id":old["pose_id"],"skill_id":"single_force_v1",
               "direction_id":best,"second_direction_id":None,
               "support_ids":[old["part_id"]],"evidence_ids":[last["feedback"]["evidence_id"]]}
            if eligible(c):
                return c
    return None
""")

class StabilityPoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read(RUN / "manifest.json")
        cls.packet = python_generation_packet(cls.manifest, "offline-contract-test")
        cls.template = read(OLD / "cases/debug_000003/program_request.json")

    def test_factual_source_direction_and_no_mutation(self):
        before = deepcopy(self.packet)
        contract = stability_source_contract(self.packet)
        self.assertEqual(contract["source_cad_vector"], [0, 0, 1])
        self.assertFalse(contract["physics_success_implied"])
        self.assertEqual(before, self.packet)

    def test_fixture_passes_24_but_old_llm_program_does_not(self):
        contract = stability_source_contract(self.packet)
        audit = check_stability_pose_contract(HANDWRITTEN_CONTRACT_FIXTURE, self.template, contract)
        self.assertEqual(audit["passed"], 24)
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["physical_calls"], 0)
        prior = read(OLD / "provider_proposal.json")["program"]
        old = check_stability_pose_contract(prior, self.template, contract)
        self.assertFalse(old["accepted"])
        self.assertEqual(old["passed"], 0)

    def test_missing_or_path_only_feedback_rejected(self):
        req = deepcopy(self.template)
        req["episode_feedback"] = []
        req["evidence_ids"] = []
        with self.assertRaisesRegex(ValueError, "initial_multibody"):
            synthetic_stability_request(req, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    def test_parent_reclassification_is_explicit_and_new_admissions_absent(self):
        packet = development_parent_packet(self.packet, self.manifest, live=True, ledger=LEDGER)
        self.assertEqual(len(packet["validation_feedback"]), 4)
        self.assertEqual({r["phase"] for r in packet["validation_feedback"]}, {"debug"})
        self.assertEqual(sum(r["previous_phase"] == "admission" for r in packet["validation_feedback"]), 1)
        encoded = json.dumps(packet)
        self.assertNotIn("admission_yaw_90", encoded)
        self.assertNotIn("admission_yaw_180", encoded)
        manifest = deepcopy(self.manifest)
        manifest["development_parent"]["former_admission_is_now_development"] = False
        with self.assertRaisesRegex(ValueError, "reclassification"):
            development_parent_packet(self.packet, manifest, live=True, ledger=LEDGER)

    def test_old_condition_cannot_be_called_new_admission(self):
        manifest = deepcopy(self.manifest)
        manifest["cases"][-1].pop("pose_yaw_degrees")
        with self.assertRaisesRegex(ValueError, "separate"):
            development_parent_packet(self.packet, manifest, live=True, ledger=LEDGER)

    def test_larger_local_binding_has_bounded_provider_projection(self):
        packet = development_parent_packet(self.packet, self.manifest, live=True, ledger=LEDGER)
        packet["context"]["stability_coordinate_contract"] = stability_source_contract(packet)
        service = PythonSkillDiscoveryService(read(LEDGER.parent / "approved_live_config.json"), LEDGER)
        preview = service.preview(packet)
        self.assertLess(preview["body_bytes"], 262144)
        self.assertIn("650 output tokens", preview["body"]["instructions"])
        self.assertEqual(preview["body"]["max_output_tokens"], 1024)
        geometry = json.loads(preview["body"]["input"][0]["content"])["validation_feedback"][0][
            "current_observation_request"]["geometry"]
        self.assertEqual(set(geometry["parts"]), set(packet["validation_feedback"][0][
            "current_observation_request"]["geometry"]["parts"]))
        self.assertFalse(geometry["physics_success_implied"])
        packet["context"]["oversize"] = "x" * MAX_LOCAL_EVIDENCE_BYTES
        with self.assertRaisesRegex(ValueError, "packet_too_large"):
            service.preview(packet)

    def test_repeated_failed_stability_configuration_rejected_before_physics(self):
        parent = RUN / "generation_03"
        packet = refinement_packet(self.packet, parent, self.manifest, live=True, ledger=LEDGER)
        requirements = failed_debug_requirements(packet)
        self.assertEqual([r["case_id"] for r in requirements], ["debug_former_admission_000001"])
        self.assertEqual(requirements[0]["branch_invoked_before_recovery"], "initial_disconnected")
        self.assertEqual(requirements[0]["failure_after_recovery"], "dynamic_failure")
        self.assertEqual(requirements[0]["observed_failed_unheld_part_features"][0]["ascending_volume_rank"], 1)
        program = read(parent / "provider_proposal.json")["program"]
        audit = check_stability_revision(program, packet)
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["physics_calls"], 0)
        self.assertTrue(audit["rows"][0]["repeated_failed_stability_configuration"])
        packet["validation_feedback"][0]["phase"] = "admission"
        with self.assertRaisesRegex(ValueError, "debug_only"):
            failed_debug_requirements(packet)

    def test_all_probe_outcomes_bound_and_success_retention_required(self):
        packet = refinement_packet(self.packet, RUN / "generation_03", self.manifest, live=True, ledger=LEDGER)
        augmented = add_development_probes(packet, RUN / "support_probes")
        probes = augmented["validation_feedback"][-3:]
        self.assertEqual([r["outcome"] for r in probes], ["UNKNOWN", "UNKNOWN", "PASS"])
        self.assertTrue(probes[-1]["retain_successful_probe"])
        self.assertIn("human", probes[-1]["origin"])
        program = read(RUN / "generation_05/provider_proposal.json")["program"]
        audit = check_stability_revision(program, augmented)
        self.assertFalse(audit["accepted"])
        self.assertFalse(audit["rows"][-1]["successful_development_probe_preserved"])
        packet["validation_feedback"] = packet["validation_feedback"][:1]
        with self.assertRaisesRegex(ValueError, "declared_debug"):
            add_development_probes(packet, RUN / "support_probes")

if __name__ == "__main__":
    unittest.main()
