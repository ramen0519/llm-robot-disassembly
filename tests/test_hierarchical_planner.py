"""Injected provider/feedback fixtures, never claims real physics or model quality."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from tests.test_explicit_action_contract import request as basic_request, attach_attempts
from tests.test_explicit_action_service import provider
from orchestrator.llm_ranker import DEFAULTS
from orchestrator.conditional_plan import validate_plan, action_for_step
from orchestrator.hierarchy_budget import HierarchyBudget, validate_policy, MODELS
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from worker.explicit_action_contract import canonical_response, validate_request


def request():
    return dict(basic_request(), attempted_actions=[], path_stage_observations=[])


def policy(enabled=False):
    roles = {}
    for role, model in MODELS.items():
        roles[role] = dict(DEFAULTS, enabled=True, model=model, max_calls=4,
            max_usd="1", input_usd_per_million="2" if role == "planner" else ".1",
            output_usd_per_million="10" if role == "planner" else ".5",
            max_output_tokens=4096, reasoning_effort="low" if role == "planner" else "none")
    return dict(schema_version=1, enabled=enabled, max_calls=12, max_usd="3", roles=roles,
        episode=dict(scout_calls=1, planner_calls=2, fallback_calls=0, max_plan_steps=64, recovery_per_step=1))


def plan_value(r, parts=None, direction="xp"):
    parts = parts or sorted(r["candidate_ids"])
    remaining = set(r["remaining_parts"])
    steps = []
    for part in parts:
        steps.append(dict(expected_remaining=sorted(remaining), part_id=part, mode="action",
            pose_reference=r["pose_catalog"][0]["pose_id"], primitive="single_force_v1",
            direction=direction, second_direction=None, support_ids=[], evidence_ids=[], registered_skill_id=None))
        remaining.remove(part)
    return dict(request_id=r["request_id"], state_id=r["state_id"],
        candidate_order=sorted(r["candidate_ids"]), steps=steps, recovery_skill_ids=[])


def feedback(r, response, success=True):
    new = deepcopy(r)
    new["request_id"] += ":next"
    new["state_id"] += ":next" if success else ""
    new["episode_feedback"].append(dict(evidence_id="e:"+new["request_id"],
        action_id=response["action_id"], status="PASS" if success else "UNKNOWN",
        remaining_parts=list(r["remaining_parts"]), choice=response["choice"],
        path_success=success, stability_success=True if success else None, accepted_for_transition=success,
        deadline_exceeded=False, selected_pose_matrix=deepcopy(next(
            p["matrix"] for p in r["pose_catalog"] if p["pose_id"]==response["choice"]["pose_id"]))))
    new["evidence_ids"].append(new["episode_feedback"][-1]["evidence_id"])
    if success:
        part = response["choice"]["part_id"]
        new["remaining_parts"].remove(part)
        new["candidate_ids"].remove(part)
        new["attempted_action_ids"] = []
        new["attempted_actions"] = []
        new["path_stage_observations"] = []
    else:
        attach_attempts(new, [response["choice"]])
        new["path_stage_observations"] = [dict(action_id=response["action_id"],
            path_success=False, status="UNKNOWN", deadline_exceeded=False,
            backend="ASAP_RedMax", source_event_id=new["episode_feedback"][-1]["evidence_id"])]
    return validate_request(new)


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hierarchy_unit_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, OPENAI_API_KEY="fixture-not-a-real-key")
        env.start(); self.addCleanup(env.stop)
        net = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network_forbidden"))
        self.network = net.start(); self.addCleanup(net.stop)
        self.addCleanup(self.network.assert_not_called)

    def test_plan_runs_next_step_without_owning_search(self):
        r = request(); original = deepcopy(r)
        p = validate_plan(plan_value(r), r, [])
        a = action_for_step(p, 0, r)
        self.assertEqual(r, original)
        n = feedback(r, a)
        self.assertEqual(action_for_step(p, 1, n)["choice"]["part_id"], "2")
        with self.assertRaisesRegex(ValueError, "remaining_guard"):
            action_for_step(p, 1, r)

    def test_orientation_requires_unique_match_and_uses_current_translation(self):
        r = request(); p = validate_plan(plan_value(r), r, [])
        r["pose_catalog"][0]["pose_id"] = "new"
        r["pose_catalog"][0]["matrix"][2][3] = 9
        self.assertEqual(action_for_step(p, 0, r)["choice"]["pose_id"], "new")
        r["pose_catalog"].append(dict(pose_id="duplicate", matrix=deepcopy(r["pose_catalog"][0]["matrix"])))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            action_for_step(p, 0, r)

    def test_rejects_bad_plan_id_supports_transition_and_omitted_candidates(self):
        for mutate in (lambda p:p.update(request_id="stale"),
                       lambda p:p.update(candidate_order=["1"]),
                       lambda p:p["steps"][0].update(support_ids=["base"]),
                       lambda p:p["steps"][1].update(expected_remaining=["1","base"]),
                       lambda p:p["steps"][0].update(evidence_ids=["invented"])):
            p = plan_value(request()); mutate(p)
            with self.assertRaises(ValueError): validate_plan(p, request(), [])

    def test_physics_config_change_invalidates_plan(self):
        r = request(); p = validate_plan(plan_value(r), r, [])
        r["task_context"]["solver_settings"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "scope_changed"): action_for_step(p, 0, r)

    def test_geometry_membership_changes_after_removal_but_features_cannot(self):
        r = request(); p = validate_plan(plan_value(r), r, [])
        n = feedback(r, action_for_step(p, 0, r))
        del n["geometry"]["parts"]["1"]
        action_for_step(p, 1, n)
        n["geometry"]["parts"]["2"]["extents"][0] += 1
        with self.assertRaisesRegex(ValueError, "geometry_features"):
            action_for_step(p, 1, n)

    def test_sol_provider_contract_caching_and_no_retry(self):
        cfg = policy()["roles"]["planner"]
        raw = provider(plan_value(request())); raw["model"] = cfg["model"]
        transport = Mock(return_value=raw)
        b = HierarchyBudget(self.root/"global.sqlite", policy())
        svc = SolPlanService(cfg, self.root/"sol.sqlite", b.transport("planner", transport))
        result = svc.plan(request(), [], {}, 64, 30)
        self.assertEqual(result["status"], "planned", result)
        self.assertEqual(svc.plan(request(), [], {}, 64, 30)["status"], "planned")
        self.assertEqual(transport.call_count, 1)
        self.assertFalse(result["metadata"]["llm_live"])
        self.assertEqual(b.summary()["live_api_attempts"], 0)
        self.assertEqual(b.summary()["roles"]["planner"], 1)

    def test_unknown_transport_is_never_resent_and_policy_is_immutable(self):
        b = HierarchyBudget(self.root/"global.sqlite", policy())
        cfg = policy()["roles"]["planner"]
        svc = SolPlanService(cfg, self.root/"sol.sqlite")
        body = svc._body(dict(request=request(), skills=[], escalation={}, max_steps=64))
        send = Mock(side_effect=TimeoutError())
        transport = b.transport("planner", send)
        with self.assertRaises(TimeoutError): transport(cfg["base_url"]+"/responses", {}, body, 10)
        with self.assertRaisesRegex(ValueError, "no_resend"): transport(cfg["base_url"]+"/responses", {}, body, 10)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(b.summary()["pending_or_unknown"], 1)
        changed = policy(); changed["max_calls"] += 1
        with self.assertRaisesRegex(ValueError, "immutable"): HierarchyBudget(b.path, changed)

    def test_role_budget_cannot_consume_planner_allocation(self):
        p = policy(); p["roles"]["scout"]["max_calls"] = 1
        b = HierarchyBudget(self.root/"global.sqlite", p)
        cfg = p["roles"]["scout"]
        body = dict(model=cfg["model"], store=False, reasoning={"effort":"none"}, max_output_tokens=16)
        t = b.transport("scout", Mock(return_value={}))
        t(cfg["base_url"]+"/responses", {}, body, 10)
        body["instructions"] = "next"
        with self.assertRaisesRegex(ValueError, "cost_limit"): t(cfg["base_url"]+"/responses", {}, body, 10)
        self.assertEqual(b.summary()["roles"]["planner"], 0)

    def service(self, planner, scout_calls=0):
        scout = Mock()
        scout.timeout_seconds = 30
        scout.handle.side_effect = lambda r: canonical_response(r, None)
        p = policy()["episode"]; p["scout_calls"] = scout_calls
        with patch("orchestrator.hierarchical_service.retrieve", return_value=[]):
            svc = HierarchicalDecisionService(scout, planner, {}, p)
        return svc

    def planner(self):
        planner = Mock()
        planner.plan.side_effect = lambda r, *a, **kw: dict(status="planned",
            plan=validate_plan(plan_value(r, direction="xn" if r["attempted_action_ids"] else "xp"), r, []),
            metadata={"llm_live":False,"model_returned":"fixture-sol","call_id":"fixture"})
        return planner

    def test_code_continues_after_pass_and_replans_after_failure(self):
        planner = self.planner(); svc = self.service(planner)
        r = request(); a = svc.handle(r)
        self.assertEqual(a["metadata"]["hierarchy_role"], "plan_action")
        self.assertEqual(svc.handle(deepcopy(r)), a)
        n = feedback(r, a); b = svc.handle(n)
        self.assertEqual(planner.plan.call_count, 1)
        n2 = feedback(n, b, False); c = svc.handle(n2)
        self.assertEqual(planner.plan.call_count, 2)
        self.assertNotEqual(c["action_id"], b["action_id"])
        self.assertEqual(c["metadata"]["hierarchy_role"], "plan_action")
        n3 = feedback(n2, c, False); d = svc.handle(n3)
        self.assertEqual(planner.plan.call_count, 2)
        self.assertEqual(d["metadata"]["hierarchy_role"], "deterministic_fallback")

    def test_scout_pass_then_sol_and_missing_feedback_replans(self):
        planner = self.planner(); svc = self.service(planner, 1)
        r = request(); a = svc.handle(r)
        self.assertEqual(a["metadata"]["hierarchy_role"], "scout")
        n = feedback(r, a); b = svc.handle(n)
        self.assertEqual(planner.plan.call_count, 1)
        changed = deepcopy(n); changed["request_id"] += ":unknown"
        svc.handle(changed)
        self.assertEqual(planner.plan.call_count, 2)
        self.assertTrue(any(e.get("reason")=="missing_or_ambiguous_verifier_feedback" for e in svc.events))

    def test_request_conflict_and_mutated_library_rejected(self):
        svc = self.service(self.planner()); r = request(); svc.handle(r)
        changed = deepcopy(r); changed["state_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "id_conflict"): svc.handle(changed)
        changed["request_id"] += "2"; svc.snapshot["changed"] = True
        with self.assertRaisesRegex(ValueError, "library_changed"): svc.handle(changed)

    def test_sol_authorized_registered_program_runs_only_after_failure(self):
        from tests.test_generated_skill_service import row
        entry = row()
        def reply(r, *a, **kw):
            p = plan_value(r)
            p["recovery_skill_ids"] = [entry["strategy_id"]]
            return dict(status="planned", plan=validate_plan(p,r,[entry["strategy_id"]]),
                        metadata={"llm_live":False})
        planner = Mock(); planner.plan.side_effect = reply
        scout = Mock(); scout.timeout_seconds=30
        limits = policy()["episode"]; limits["scout_calls"] = 0
        with patch("orchestrator.hierarchical_service.retrieve", return_value=[entry]):
            service = HierarchicalDecisionService(scout,planner,{},limits)
        r = request(); a = service.handle(r)
        n = feedback(r,a,False); b = service.handle(n)
        self.assertEqual(b["metadata"]["hierarchy_role"],"recovery")
        self.assertEqual(b["metadata"]["generated_skill_id"],entry["strategy_id"])
        self.assertEqual(b["choice"]["direction_id"],"zp")
        self.assertEqual(planner.plan.call_count,1)
        n2=feedback(n,b); c=service.handle(n2)
        self.assertEqual(c["choice"]["part_id"],"2")
        self.assertEqual(planner.plan.call_count,1)


if __name__ == "__main__":
    unittest.main()
