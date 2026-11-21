"""Offline contract/feedback regressions; provider responses are injected."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.conditional_plan import action_for_step, validate_plan
from orchestrator.conditional_plan_v3 import compile_plan, schema_v3, CONTRACT
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.plan_pose_binding import BOUNDED, EXACT, resolve_pose
from tests.test_hierarchical_planner import request, plan_value, feedback, policy
from tests.test_explicit_action_service import provider

ROOT = Path(__file__).absolute().parents[1]
OLD = ROOT / "runs/01256_part13_repair_20261021/episode_02"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def wire(plan, pose_policy=BOUNDED):
    value = {k:deepcopy(plan[k]) for k in ("request_id","state_id","candidate_order","steps","recovery_skill_ids")}
    for step in value["steps"]:
        step.pop("expected_remaining", None)
        step["pose_policy"] = pose_policy if step["mode"] == "action" else None
    return value


def tilted(degrees, identity="tilted", translation=7):
    theta = math.radians(degrees); c,s = math.cos(theta),math.sin(theta)
    return dict(pose_id=identity, matrix=[[1,0,0,0],[0,c,-s,0],[0,s,c,translation],[0,0,0,1]])


class PoseContinuityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict("os.environ", OPENAI_API_KEY="OFFLINE_NOT_A_REAL_KEY")
        env.start(); self.addCleanup(env.stop)
        net = patch("urllib.request.OpenerDirector.open",side_effect=AssertionError("network_forbidden"))
        self.net=net.start(); self.addCleanup(net.stop); self.addCleanup(self.net.assert_not_called)

    def test_actual_failed_state_rebinds_without_changing_force_or_supports(self):
        decisions = read(OLD/"hierarchy_decisions.json")
        prior = read(OLD/"hierarchy_plans.json")[-1]["result"]["plan"]
        origin = next(d["request"] for d in decisions if d["request"]["request_id"] == prior["request_id"])
        current = decisions[12]["request"]
        original = deepcopy((current,prior))
        with self.assertRaisesRegex(ValueError,"orientation_unavailable"):
            action_for_step(prior,3,current)
        bound = compile_plan(wire(prior),origin,prior["recovery_skill_ids"],16)
        response = action_for_step(bound,3,current)
        c, meta = response["choice"], response["metadata"]["plan_pose_binding"]
        self.assertEqual((c["part_id"],c["direction_id"],c["support_ids"]),("7","xn",["0","2"]))
        self.assertEqual(c["pose_id"],"pose_1fd3d1a735bf1c99")
        self.assertAlmostEqual(meta["rotation_delta_degrees"],3.411682,places=5)
        self.assertEqual(meta["match_kind"],"bounded_rebind")
        self.assertTrue(meta["fresh_path_and_stability_required"])
        self.assertFalse(meta["pose_transition_validated"])
        self.assertEqual((current,prior),original)

    def test_exact_default_and_explicit_exact_do_not_silently_adapt(self):
        r=request(); value=plan_value(r)
        v1=validate_plan(value,r,[]); v3=compile_plan(wire(value,EXACT),r,[])
        altered=deepcopy(r); altered["pose_catalog"]=[tilted(3)]
        for p in (v1,v3):
            with self.assertRaisesRegex(ValueError,"orientation_unavailable"):
                action_for_step(p,0,altered)

    def test_bounded_selects_nearest_independent_of_order_and_keeps_ground_translation(self):
        r=request(); ref=[[1,0,0],[0,1,0],[0,0,1]]
        r["pose_catalog"]=[tilted(4,"far",12),tilted(2,"near",3)]
        identity, audit=resolve_pose(ref,r,BOUNDED)
        self.assertEqual(identity,"near")
        self.assertEqual(audit["selected_pose_matrix"][2][3],3)
        r["pose_catalog"].reverse()
        self.assertEqual(resolve_pose(ref,r,BOUNDED)[0],"near")

    def test_bound_and_ties_reject_without_id_tiebreak(self):
        r=request(); ref=[[1,0,0],[0,1,0],[0,0,1]]
        for catalog, error in (([tilted(5.01)],"exceeds_limit"),
            ([tilted(3,"a"),tilted(-3,"b")],"ambiguous"),
            ([tilted(0,"a"),tilted(0,"b")],"ambiguous")):
            r["pose_catalog"]=catalog
            with self.assertRaisesRegex(ValueError,error): resolve_pose(ref,r,BOUNDED)
        r["pose_catalog"]=[tilted(5)]
        self.assertEqual(resolve_pose(ref,r,BOUNDED)[0],"tilted")

    def test_original_anchor_prevents_incremental_drift(self):
        r=request(); p=compile_plan(wire(plan_value(r)),r,[])
        r["pose_catalog"]=[tilted(4)]
        response=action_for_step(p,0,r)
        next_req=feedback(r,response)
        next_req["pose_catalog"]=[tilted(8)]
        with self.assertRaisesRegex(ValueError,"exceeds_limit"):
            action_for_step(p,1,next_req)

    def test_rejects_nonrigid_or_nonfinite_orientation(self):
        r=request(); ref=[[1,0,0],[0,1,0],[0,0,1]]
        for bad in (float("nan"),2,True):
            r["pose_catalog"]=[tilted(2)]
            r["pose_catalog"][0]["matrix"][0][0]=bad
            with self.assertRaises(ValueError): resolve_pose(ref,r,BOUNDED)
        with self.assertRaises(ValueError): resolve_pose([[1,0],[0,1]],r,BOUNDED)

    def test_contract_requires_explicit_policy_and_registered_steps_have_null(self):
        r=request(); value=wire(plan_value(r))
        schema=schema_v3(r,[])["properties"]["steps"]["items"]
        self.assertIn("pose_policy",schema["required"])
        for mutate in (lambda p:p["steps"][0].pop("pose_policy"),
                       lambda p:p["steps"][0].update(pose_policy="unlimited"),
                       lambda p:p["steps"][0].update(pose_policy=None)):
            bad=deepcopy(value); mutate(bad)
            with self.assertRaises(ValueError): compile_plan(bad,r,[])
        skill=deepcopy(value)
        skill["steps"][0].update(mode="registered_skill",registered_skill_id="skill1",
            pose_reference=None,primitive=None,direction=None,second_direction=None,support_ids=[],pose_policy=None)
        compile_plan(skill,r,["skill1"])
        skill["steps"][0]["pose_policy"]=BOUNDED
        with self.assertRaisesRegex(ValueError,"pose_policy"): compile_plan(skill,r,["skill1"])

    def test_unchanged_state_geometry_solver_and_duplicate_action_guards(self):
        r=request(); p=compile_plan(wire(plan_value(r)),r,[])
        for mutate, error in (
            (lambda n:n["task_context"]["solver_settings"].update(seed=42),"scope_changed"),
            (lambda n:n["geometry"]["parts"]["1"]["extents"].__setitem__(0,999),"geometry_features"),
            (lambda n:n.update(remaining_parts=["base","2"]),"remaining_guard")):
            bad=deepcopy(r);mutate(bad)
            with self.assertRaisesRegex(ValueError,error):action_for_step(p,0,bad)
        a=action_for_step(p,0,r)
        failed=feedback(r,a,False)
        with self.assertRaises(ValueError):action_for_step(p,0,failed)

    def test_provider_v3_roundtrip_legacy_is_separate_and_no_retry(self):
        r=request(); cfg=policy()["roles"]["planner"]
        raw=provider(wire(plan_value(r)));raw["model"]=cfg["model"]
        send=Mock(return_value=raw)
        service=SolPlanService(cfg,Path(self.tmp.name)/"sol.sqlite",send,provider_contract=CONTRACT)
        result=service.plan(r,[],{},16,30)
        self.assertEqual(result["status"],"planned",result)
        body=send.call_args.args[2]
        self.assertIn("WORLD AXES",body["instructions"])
        self.assertNotIn("one current catalog pose with the same world orientation",body["instructions"])
        self.assertEqual(service.plan(r,[],{},16,30)["status"],"planned")
        self.assertEqual(send.call_count,1)
        invalid=wire(plan_value(r));invalid["steps"][0]["pose_policy"]="unlimited"
        badraw=provider(invalid);badraw["model"]=cfg["model"]
        badsend=Mock(return_value=badraw)
        svc=SolPlanService(cfg,Path(self.tmp.name)/"bad.sqlite",badsend,provider_contract=CONTRACT)
        self.assertEqual(svc.plan(r,[],{},16,30)["status"],"rejected")
        self.assertEqual(badsend.call_count,1)

    def test_service_rebind_advances_only_after_pass_and_replans_after_failure(self):
        planner=Mock()
        planner.plan.side_effect=lambda r,*a,**kw:dict(status="planned",
            plan=compile_plan(wire(plan_value(r)),r,[]),metadata={"llm_live":False})
        scout=Mock();scout.timeout_seconds=30
        limits=policy()["episode"];limits["scout_calls"]=0
        with patch("orchestrator.hierarchical_service.retrieve",return_value=[]):
            svc=HierarchicalDecisionService(scout,planner,{},limits)
        r=request(); first=svc.handle(r)
        next_req=feedback(r,first)
        next_req["pose_catalog"]=[tilted(3)]
        second=svc.handle(next_req)
        self.assertEqual(planner.plan.call_count,1)
        self.assertEqual(second["metadata"]["hierarchy_role"],"plan_action")
        self.assertEqual(second["metadata"]["plan_pose_binding"]["match_kind"],"bounded_rebind")
        failed=feedback(next_req,second,False)
        # A fresh plan must be requested; the old cursor cannot advance on UNKNOWN.
        planner.plan.side_effect=lambda *a,**kw:dict(status="rejected",fallback_reason="fixture_no_plan",metadata={})
        svc.handle(failed)
        self.assertEqual(planner.plan.call_count,2)
        self.assertTrue(any(e["kind"]=="plan_invalidated" for e in svc.events))

    def test_pose_guard_replan_names_execution_reason_not_a_physics_failure(self):
        planner=Mock(); planner.provider_contract=CONTRACT
        planner.plan.side_effect=lambda r,*a,**kw:dict(status="planned",
            plan=compile_plan(wire(plan_value(r)),r,[]),metadata={"llm_live":False})
        scout=Mock();scout.timeout_seconds=30
        limits=policy()["episode"];limits["scout_calls"]=0
        with patch("orchestrator.hierarchical_service.retrieve",return_value=[]):
            svc=HierarchicalDecisionService(scout,planner,{},limits)
        r=request(); first=svc.handle(r)
        next_req=feedback(r,first); next_req["pose_catalog"]=[tilted(12)]
        svc.handle(next_req)
        escalation=planner.plan.call_args.args[2]
        self.assertEqual(escalation["reason"],"planned_pose_adjustment_exceeds_limit")
        physics=[e for e in svc.events if e["kind"]=="physics_feedback"]
        self.assertEqual([e["status"] for e in physics],["PASS"])


if __name__=="__main__":
    unittest.main()
