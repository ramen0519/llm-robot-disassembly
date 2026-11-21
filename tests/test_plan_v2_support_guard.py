"""Offline compiler/provider/recovery tests. All provider feedback is injected."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.test_hierarchical_planner import request, policy, plan_value, feedback
from tests.test_explicit_action_service import provider
from orchestrator.conditional_plan import validate_plan, action_for_step
from orchestrator.conditional_plan_v2 import compile_plan, schema_v2, CONTRACT
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.recovery_support_guard import check_recovery_supports
from orchestrator.explicit_action_provider_v5 import restore_request_v5

ROOT = Path(__file__).absolute().parents[1]
LIVE = ROOT / "runs/hierarchy_01256_all_poses_20261021/episode_01"
LIBRARY = ROOT / "runs/python_stability_threepart_20261020/reuse_skills_frozen.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def wire(plan):
    value = {k:deepcopy(plan[k]) for k in
             ("request_id","state_id","candidate_order","steps","recovery_skill_ids")}
    for step in value["steps"]:
        step.pop("expected_remaining")
    return value


class PlanV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        env = patch.dict(os.environ, OPENAI_API_KEY="OFFLINE_NOT_A_REAL_KEY")
        env.start(); self.addCleanup(env.stop)
        network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network_forbidden"))
        self.network = network.start(); self.addCleanup(network.stop)
        self.addCleanup(self.network.assert_not_called)

    def test_compiler_derives_sets_without_mutation_or_physics_assumption(self):
        r = request(); v = wire(plan_value(r)); original = deepcopy((r,v))
        info = {}; p = compile_plan(v, r, [], diagnostics=info)
        self.assertEqual((r,v), original)
        self.assertEqual(p["steps"][0]["expected_remaining"], sorted(r["remaining_parts"]))
        self.assertEqual(p["steps"][1]["expected_remaining"],
                         sorted(set(r["remaining_parts"])-{p["steps"][0]["part_id"]}))
        self.assertFalse(p["physics_success_implied"])
        first = action_for_step(p,0,r)
        with self.assertRaisesRegex(ValueError,"remaining_guard"):
            action_for_step(p,1,r)
        second = action_for_step(p,1,feedback(r,first))
        self.assertNotEqual(first["choice"]["part_id"],second["choice"]["part_id"])

    def test_schema_removes_duplicate_state_field_but_keeps_strict_action_fields(self):
        step = schema_v2(request(),[])["properties"]["steps"]["items"]
        self.assertNotIn("expected_remaining",step["properties"])
        self.assertEqual(set(step["required"]),set(step["properties"]))
        self.assertFalse(step["additionalProperties"])
        with self.assertRaisesRegex(ValueError,"invalid_v2_step_fields"):
            compile_plan(plan_value(request()),request(),[])

    def test_invalid_v1_remaining_still_rejected_instead_of_silently_repaired(self):
        r=request(); v=plan_value(r)
        v["steps"][1]["expected_remaining"] = []
        with self.assertRaisesRegex(ValueError,"remaining_transition_mismatch"):
            validate_plan(v,r,[])
        with self.assertRaisesRegex(ValueError,"invalid_v2_step_fields"):
            compile_plan(v,r,[])

    def test_invalid_actions_and_future_supports_still_rejected_with_step_diagnostic(self):
        r=request()
        mutations = [
            (lambda v:v["steps"][1].update(part_id=v["steps"][0]["part_id"]),1),
            (lambda v:v["steps"][1].update(support_ids=[v["steps"][0]["part_id"]]),1),
            (lambda v:v["steps"][0].update(pose_reference="invented"),0),
            (lambda v:v.update(candidate_order=[]),0),
            (lambda v:v["steps"][0].update(evidence_ids=["invented"]),0)]
        for mutate,index in mutations:
            with self.subTest(index=index, mutation=mutate):
                v=wire(plan_value(r)); mutate(v); info={}
                with self.assertRaises(ValueError):
                    compile_plan(v,r,[],diagnostics=info)
                self.assertEqual(info["step_index"],index)
                self.assertEqual(len(info["wire_value_sha256"]),64)

    def test_actual_saved_sol_plan_compiles_to_same_choices_and_guards(self):
        p=next(p["result"]["plan"] for p in read(LIVE/"hierarchy_plans.json")
               if p["result"]["status"]=="planned")
        r=next(d["request"] for d in read(LIVE/"hierarchy_decisions.json")
               if d["request"]["request_id"]==p["request_id"])
        v=compile_plan(wire(p),r,p["recovery_skill_ids"],8)
        self.assertEqual(v,p)

    def test_provider_codec_cache_contract_binding_and_no_retry(self):
        r=request(); cfg=policy()["roles"]["planner"]
        raw=provider(wire(plan_value(r))); raw["model"]=cfg["model"]
        transport=Mock(return_value=raw)
        svc=SolPlanService(cfg,self.path/"ledger.sqlite",transport,provider_contract=CONTRACT)
        a=svc.plan(r,[],{},8,30)
        self.assertEqual(a["status"],"planned",a)
        self.assertEqual(svc.plan(r,[],{},8,30)["status"],"planned")
        self.assertEqual(transport.call_count,1)
        body=transport.call_args.args[2]
        packet=json.loads(body["input"][0]["content"])
        self.assertEqual(restore_request_v5(packet["observation"]),r)
        self.assertEqual(packet["execution_contract"],CONTRACT)
        self.assertNotIn("expected_remaining",body["text"]["format"]["schema"]["properties"]["steps"]["items"]["properties"])
        legacy=SolPlanService(cfg,self.path/"ledger.sqlite",transport)
        self.assertEqual(legacy.plan(r,[],{},8,30)["fallback_reason"],"request_id_conflict")
        self.assertEqual(transport.call_count,1)

    def test_bad_response_is_rejected_and_diagnostic_survives_cache(self):
        r=request(); v=wire(plan_value(r)); v["steps"][1]["support_ids"]=[v["steps"][0]["part_id"]]
        raw=provider(v); cfg=policy()["roles"]["planner"]; raw["model"]=cfg["model"]
        send=Mock(return_value=raw)
        svc=SolPlanService(cfg,self.path/"bad.sqlite",send,provider_contract=CONTRACT)
        a=svc.plan(r,[],{},8,30)
        self.assertEqual(a["status"],"rejected")
        self.assertEqual(a["metadata"]["plan_contract_diagnostics"]["step_index"],1)
        self.assertEqual(a["metadata"]["plan_contract_error"],"invalid_plan_supports")
        self.assertEqual(svc.plan(r,[],{},8,30)["metadata"]["plan_contract_diagnostics"],
                         a["metadata"]["plan_contract_diagnostics"])
        self.assertEqual(send.call_count,1)

    def test_unknown_strings_are_not_copied_into_diagnostics(self):
        r=request(); v=wire(plan_value(r)); v["steps"][0]["part_id"]="secret-untrusted-text"
        info={}
        with self.assertRaises(ValueError):compile_plan(v,r,[],diagnostics=info)
        self.assertNotIn("secret-untrusted-text",json.dumps(info))


class RecoveryGuardTests(unittest.TestCase):
    def reference(self):
        return dict(choice=dict(support_ids=["0"]),remaining_parts=["0","1","5","8","10"],
                    evidence_id="actual-failure",path_success=True,stability_success=False)

    def r(self):
        return dict(remaining_parts=["0","1","5","8","10"],max_grippers=3)

    def test_replacement_is_refused_not_rewritten_or_called_impossible(self):
        c=dict(support_ids=["10","5"]); original=deepcopy(c)
        result=check_recovery_supports(self.reference(),c,self.r())
        self.assertFalse(result["allowed"])
        self.assertEqual(result["released_support_ids"],["0"])
        self.assertTrue(result["union_exceeds_capacity"])
        self.assertEqual(c,original)
        self.assertIsNone(result["physical_verdict"])
        self.assertFalse(result["hard_pruning"])

    def test_same_support_addition_and_path_recovery_are_allowed(self):
        for supports in (["0"],["0","10"]):
            ref=self.reference()
            self.assertTrue(check_recovery_supports(ref,dict(support_ids=supports),self.r())["allowed"])
            ref.update(path_success=False,stability_success=None)
            self.assertTrue(check_recovery_supports(ref,dict(support_ids=supports),self.r())["allowed"])
        self.assertFalse(check_recovery_supports(None,dict(support_ids=[]),self.r())["allowed"])

    def test_actual_registered_program_proposal_is_blocked_and_replanning_gets_reason(self):
        from orchestrator.explicit_action_service import ExplicitActionService
        decisions=read(LIVE/"hierarchy_decisions.json")
        initial=deepcopy(next(d["request"] for d in decisions
                             if d["response"]["metadata"]["hierarchy_role"]=="plan_action"))
        failed=deepcopy(next(d["request"] for d in decisions
                            if d["response"]["metadata"]["hierarchy_role"]=="recovery"))
        old=next(p["result"]["plan"] for p in read(LIVE/"hierarchy_plans.json")
                 if p["result"]["status"]=="planned")
        limits=policy()["episode"]; limits.update(scout_calls=0,max_plan_steps=8)
        seen=[]
        def plan_reply(r,cards,escalation,**kwargs):
            seen.append(deepcopy(escalation))
            if len(seen)==1:
                plan=compile_plan(wire(old),r,old["recovery_skill_ids"],8)
            else:
                value=wire(old); value.update(request_id=r["request_id"],state_id=r["state_id"])
                value["steps"]=value["steps"][:1]; value["steps"][0]["support_ids"]=["0","1"]
                plan=compile_plan(value,r,old["recovery_skill_ids"],8)
            return dict(status="planned",plan=plan,metadata={"llm_live":False})
        planner=Mock(); planner.plan.side_effect=plan_reply
        with tempfile.TemporaryDirectory() as folder:
            scout=ExplicitActionService({},Path(folder)/"scout.sqlite",mode="deterministic")
            svc=HierarchicalDecisionService(scout,planner,read(LIBRARY),limits,
                    recovery_support_policy="preserve_existing_v1")
            original=deepcopy((initial,failed))
            a=svc.handle(initial)
            self.assertEqual(a["choice"]["support_ids"],["0"])
            b=svc.handle(failed)
            self.assertEqual((initial,failed),original)
            self.assertEqual(b["metadata"]["hierarchy_role"],"plan_action")
            self.assertEqual(b["choice"]["support_ids"],["0","1"])
            guards=[e for e in svc.events if e["kind"]=="recovery_support_guard"]
            self.assertEqual(len(guards),1)
            self.assertEqual(guards[0]["proposed_support_ids"],["10","5"])
            self.assertFalse(guards[0]["allowed"])
            self.assertTrue(any(e["kind"]=="recovery_support_guard" for e in seen[1]["prior_events"]))
            self.assertEqual(seen[1]["recovery_support_policy"],"preserve_existing_v1")
            self.assertEqual(planner.plan.call_count,2)
            self.assertFalse(any(d["response"]["metadata"]["hierarchy_role"]=="recovery" for d in svc.decisions))
            svc.close()


if __name__ == "__main__":
    unittest.main()
