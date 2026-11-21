"""Offline evidence/transport-contract regressions; no HTTP or new physics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.path_failure_feedback import endpoint_brief, failure_evidence, proposal_history
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.explicit_action_memory import FrozenActionMemory
from orchestrator.explicit_action_provider_v5 import restore_request_v5
from tests.test_hierarchical_planner import policy, request, plan_value, feedback
from orchestrator.conditional_plan import validate_plan

ROOT = Path(__file__).absolute().parents[1]
EP = ROOT / "runs/01256_frozen_experience_20261021/episode_01"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


class PathFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        net = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network_forbidden"))
        self.net = net.start()
        self.addCleanup(net.stop)
        self.addCleanup(self.net.assert_not_called)

    def actual_request(self, i=8):
        return read(EP / "hierarchy_decisions.json")[i]["request"]

    def test_endpoint_scope_retained_without_turning_gaps_into_blockers(self):
        r = self.actual_request()
        old = deepcopy(r)
        facts = failure_evidence(r)
        current = [x for x in facts["observations"] if x["source_kind"] == "current_episode"]
        self.assertEqual(len(current), 2)
        for row in current:
            self.assertEqual(set(row["endpoint"]["observed_separation_unsatisfied_ids"]), {"12", "2"})
            self.assertIsNone(row["endpoint"]["blocker"])
            self.assertEqual(row["endpoint"]["termination_reason"], "unknown")
            self.assertFalse(row["endpoint"]["all_paths_impossible"])
        self.assertEqual(r, old)
        self.assertEqual(current[1]["choice"]["skill_id"], "turn_force_two_leg_v1")

    def test_bad_or_missing_endpoint_never_fabricates_diagnosis(self):
        r = self.actual_request()
        row = r["episode_feedback"][-1]
        value = row["measured_endpoint"]
        for modify in (lambda v: v.update(scope="all_paths"),
                       lambda v: v.update(blocker="2"),
                       lambda v: v.update(min_sep=float("nan")),
                       lambda v: v["pairs"][0].update(moving_part_id="invented"),
                       lambda v: v["pairs"][0].update(min_distance=float("inf")),
                       lambda v: v["pairs"][0].update(separation_unsatisfied=True)):
            v = deepcopy(value); modify(v)
            self.assertIsNone(endpoint_brief(v, "13", row["remaining_parts"]))
        self.assertIsNone(endpoint_brief(None, "13", row["remaining_parts"]))
        for e in r["episode_feedback"]:
            e.pop("measured_endpoint", None)
        current = [v for v in failure_evidence(r)["observations"] if v["source_kind"] == "current_episode"]
        self.assertTrue(all(v["endpoint"] is None for v in current))

    def test_same_proposed_recovery_failure_exposed_but_not_disabled(self):
        r = self.actual_request(7)
        memory = FrozenActionMemory(read(ROOT / "runs/01256_frozen_experience_20261021/memory_after_cumulative_frozen.json"))
        r["memory_references"] = memory.retrieve(r, 8, "assembly_balanced_v1")
        choice = read(EP / "worker_result.json")["attempts"][7]["choice"]
        before = deepcopy((r, choice))
        matches = proposal_history(r, choice)
        self.assertTrue(any(v["path_success"] is False and v["same_support_set"] for v in matches))
        self.assertTrue(all(v["disables_proposal"] is False for v in matches))
        self.assertEqual((r, choice), before)

    def test_other_pose_or_state_cannot_be_same_proposal_history(self):
        r = self.actual_request()
        choice = read(EP / "worker_result.json")["attempts"][7]["choice"]
        self.assertTrue(proposal_history(r, choice))
        for e in r["episode_feedback"]:
            e["remaining_parts"] = e["remaining_parts"] + ["not_current"]
        self.assertEqual(proposal_history(r, choice), [])
        self.assertEqual(proposal_history(r, None), [])

    def test_foreign_mesh_does_not_bind_source_part_ids(self):
        r = self.actual_request(7)
        memory = FrozenActionMemory(read(ROOT / "runs/01256_frozen_experience_20261021/memory_after_cumulative_frozen.json"))
        r["memory_references"] = memory.retrieve(r, 8, "assembly_balanced_v1")
        r["episode_feedback"] = []
        for row in r["memory_references"]:
            hashes = row["source_task_context"]["task"]["input_sha256"]
            for name in list(hashes):
                if name.endswith(".obj"):
                    hashes[name] = "0"*64
        choice = read(EP / "worker_result.json")["attempts"][7]["choice"]
        self.assertEqual(proposal_history(r, choice), [])

    def test_provider_opt_in_keeps_original_request_and_default_body(self):
        cfg = policy()["roles"]["planner"]
        r = self.actual_request()
        base = dict(request=r, skills=[], escalation={}, max_steps=8, plan_provider_contract="conditional_plan_v2")
        for selected in ("legacy_v1", "path_evidence_v1"):
            svc = SolPlanService(cfg, Path(self.tmp.name)/(selected+".sqlite"), Mock(),
                provider_contract="conditional_plan_v2", planning_feedback_policy=selected)
            env = deepcopy(base)
            if selected != "legacy_v1":
                env["planning_feedback_policy"] = selected
            body = svc._body(env)
            packet = json.loads(body["input"][0]["content"])
            self.assertEqual(restore_request_v5(packet["observation"]), r)
            self.assertEqual("path_failure_evidence" in packet, selected != "legacy_v1")
            self.assertLess(len(json.dumps(body).encode()), 262144)
            if selected != "legacy_v1":
                with self.assertRaisesRegex(ValueError, "policy_mismatch"):
                    svc._body(base)

    def test_registered_code_is_unchanged_and_prior_failure_is_on_card(self):
        r = self.actual_request(7)
        memory = FrozenActionMemory(read(ROOT / "runs/01256_frozen_experience_20261021/memory_after_cumulative_frozen.json"))
        r["memory_references"] = memory.retrieve(r, 8, "assembly_balanced_v1")
        r["evidence_ids"] = sorted({e["evidence_id"] for key in ("memory_references", "episode_feedback") for e in r[key]})
        library = read(ROOT / "runs/python_stability_threepart_20261020/reuse_skills_frozen.json")
        before = deepcopy(library)
        svc = HierarchicalDecisionService(Mock(), Mock(), library, policy()["episode"],
            planning_feedback_policy="path_evidence_v1")
        cards = svc._cards(r, time.monotonic()+30)
        target = next(c for c in cards if c["skill_id"].startswith("7f849f8b"))
        self.assertIsNotNone(target["currently_proposed_choice"])
        self.assertTrue(any(e["path_success"] is False for e in target["proposed_action_observations"]))
        self.assertEqual(library, before)

    def test_replan_receives_remaining_call_allowance_and_never_exceeds_limit(self):
        planner = Mock()
        planner.plan.side_effect = lambda r, *a, **kw: dict(status="planned",
            plan=validate_plan(plan_value(r, direction="xn" if r["attempted_action_ids"] else "xp"), r, []),
            metadata={"llm_live":False, "model_returned":"fixture-sol"})
        limits = policy()["episode"]; limits["scout_calls"] = 0
        with patch("orchestrator.hierarchical_service.retrieve", return_value=[]):
            svc = HierarchicalDecisionService(Mock(), planner, {}, limits, planning_feedback_policy="path_evidence_v1")
        r = request(); a = svc.handle(r)
        self.assertEqual(planner.plan.call_args.args[2]["planning_allowance"]["calls_after_this"], 1)
        self.assertFalse(planner.plan.call_args.args[2]["planning_allowance"]["calls_beyond_limit_authorized"])
        r2 = feedback(r, a, False); b = svc.handle(r2)
        self.assertEqual(planner.plan.call_args.args[2]["planning_allowance"]["calls_after_this"], 0)
        svc.handle(feedback(r2, b, False))
        self.assertEqual(planner.plan.call_count, 2)


if __name__ == "__main__":
    unittest.main()
