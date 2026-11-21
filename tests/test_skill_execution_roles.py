"""Authorization facts only: synthetic observations, no API or physics."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_skill_program import request, choice, append_observation
from orchestrator.skill_execution_roles import assess_execution_roles


class ExecutionRoleTests(unittest.TestCase):
    def test_replacement_is_explicit_plan_only_without_changing_action(self):
        req = append_observation(request(), choice(support_ids=["b"]))
        proposed = choice(direction_id="zp")
        before = deepcopy((req, proposed))
        roles = assess_execution_roles(req, proposed)
        guard = roles["automatic_recovery"]["guard"]
        self.assertFalse(roles["automatic_recovery"]["allowed"])
        self.assertEqual(guard["reason"], "recovery_support_release_requires_new_plan")
        self.assertEqual(guard["released_support_ids"], ["b"])
        self.assertTrue(roles["explicit_plan"]["requires_new_plan"])
        self.assertFalse(roles["explicit_plan"]["authorized"])
        self.assertFalse(roles["automatic_recovery"]["automatic_execution_verified"])
        self.assertEqual((req, proposed), before)

    def test_preserved_supports_are_guard_compatible_not_execution_verified(self):
        req = append_observation(request(), choice(support_ids=["b"]))
        roles = assess_execution_roles(req, choice(direction_id="zp", support_ids=["b"]))
        self.assertTrue(roles["automatic_recovery"]["allowed"])
        self.assertFalse(roles["automatic_recovery"]["authorized"])
        self.assertFalse(roles["current_physical_success_known"])

    def test_supplied_reference_must_match_latest_bound_actual_failure(self):
        req = append_observation(request())
        feedback = req["episode_feedback"][-1]
        ref = dict(choice=deepcopy(feedback["choice"]), remaining_parts=req["remaining_parts"],
                   evidence_id=feedback["evidence_id"])
        self.assertTrue(assess_execution_roles(req, choice(direction_id="zp"), ref)
                        ["automatic_recovery"]["allowed"])
        ref["evidence_id"] = "old-memory-observation"
        result = assess_execution_roles(req, choice(direction_id="zp"), ref)
        self.assertFalse(result["automatic_recovery"]["allowed"])
        self.assertEqual(result["automatic_recovery"]["guard"]["reason"],
                         "recovery_support_reference_not_current_failure")

    def test_unknown_stale_success_and_malformed_context_never_authorize(self):
        for req in (request(), {}, append_observation(request(), path=True, stability=True,
                     status="PASS", accepted=True), append_observation(request(), late=True)):
            self.assertFalse(assess_execution_roles(req, choice())["automatic_recovery"]["allowed"])
        req = append_observation(request())
        req["path_stage_observations"][-1]["backend"] = "fixture_not_actual_physics"
        self.assertFalse(assess_execution_roles(req, choice())["automatic_recovery"]["allowed"])

    def test_illegal_proposed_support_capacity_is_not_guard_pass(self):
        req = append_observation(request())
        self.assertFalse(assess_execution_roles(req, choice(support_ids=["a", "b"]))
                         ["automatic_recovery"]["allowed"])


if __name__ == "__main__":
    unittest.main()
