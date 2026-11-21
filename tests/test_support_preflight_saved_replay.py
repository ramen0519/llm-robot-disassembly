"""Actual stored failure + injected model replies, never new physics or network."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.hierarchy_budget import HierarchyBudget
from orchestrator.strategy_candidates import build_anchor, resolve_reference
from orchestrator.strategy_failover import POLICY
from orchestrator.strategy_plan import compile_plan, wire_step
from orchestrator.strategy_planner import StrategyPlanService
from orchestrator.strategy_service import StrategyDecisionService
from worker.decision_halt import is_halt_response
from worker.explicit_action_contract import digest, validate_choice

ROOT=Path(__file__).absolute().parents[1]
SOURCE=ROOT/'runs/repeat_00161_prepare_20261110/trial_3'


def saved_failure():
    decisions=json.loads((SOURCE/'episode/hierarchy_decisions.json').read_text(encoding='utf-8'))
    plans=json.loads((SOURCE/'episode/hierarchy_plans.json').read_text(encoding='utf-8'))
    req=decisions[-1]['request']
    bound=plans[-1]['result']['plan']
    value={name:deepcopy(bound[name]) for name in
           ('request_id','state_id','candidate_order','recovery_skill_ids')}
    value['steps']=[wire_step(step,POLICY) for step in bound['steps']]
    return req,value


class SupportPreflightSavedReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='support_replay_fixture_')
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        env=patch.dict('os.environ',OPENAI_API_KEY='fixture-only-never-real')
        env.start();self.addCleanup(env.stop)
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.network=net.start();self.addCleanup(net.stop)
        self.addCleanup(self.network.assert_not_called)

    def make_service(self, *, correct_after_first=True, plan_calls=3):
        req,bad=saved_failure()
        policy=json.loads((SOURCE/'disabled_policy.json').read_text(encoding='utf-8'))
        policy['enabled']=False  # the hierarchy transport is explicitly injected
        self.shared=HierarchyBudget(self.folder/'hierarchy.sqlite',policy,
                                    transport_failure_policy='latch_on_error_v1')
        self.bodies=[]
        def transport(url,headers,body,timeout):
            self.bodies.append(deepcopy(body))
            answer=deepcopy(bad)
            if correct_after_first and len(self.bodies)>1:
                answer['steps'][0]['support_policy']='replace_once_v1'
            return dict(id='INJECTED-support-'+str(len(self.bodies)),model='gpt-6.1-sol',
                status='completed',error=None,
                output=[dict(type='message',role='assistant',status='completed',phase='final_answer',
                    content=[dict(type='output_text',text=json.dumps(answer))])],
                usage=dict(input_tokens=100,output_tokens=100,total_tokens=200))
        planner=StrategyPlanService(policy['roles']['planner'],self.folder/'planner.sqlite',
            self.shared.transport('planner',backend=transport),role='planner',strategy_failure_policy=POLICY)
        scout=StrategyPlanService(policy['roles']['scout'],self.folder/'scout.sqlite',
            self.shared.transport('scout',backend=Mock(side_effect=AssertionError('NO_SCOUT'))),
            role='scout',strategy_failure_policy=POLICY)
        limits=dict(policy['episode'],scout_calls=0,planner_calls=plan_calls,fallback_calls=0)
        snapshot=dict(schema_version=1,kind='frozen_generated_skills_v1',entries=[],
            fixture=False,usable_for_actual_planning=True,physics_reuse_allowed=False,hard_pruning=False)
        snapshot['content_sha256']=digest(snapshot)
        service=StrategyDecisionService(scout,planner,snapshot,limits,timeout_seconds=120,
            recovery_support_policy='preserve_existing_v1',planning_feedback_policy='repair_evidence_v2',
            strategy_failure_policy=POLICY,transport_guard=self.shared.transport_blocked)
        return service,req

    def test_original_plan_rejected_before_any_physical_proposal(self):
        req,wire=saved_failure();original=deepcopy(req)
        self.assertEqual(wire['steps'][0]['part_id'],'8')
        self.assertEqual(req['episode_feedback'][-1]['choice']['support_ids'],['1','8'])
        diagnostics={}
        with self.assertRaisesRegex(ValueError,'strategy_first_step_support_conflict'):
            compile_plan(wire,req,[],8,diagnostics,orientation_reference=resolve_reference(req),
                         strategy_failure_policy=POLICY)
        self.assertIn('support_preflight',diagnostics)
        self.assertEqual(req,original)

    def test_explicit_corrected_permission_is_legal_but_not_physics_success(self):
        req,wire=saved_failure();original=deepcopy(req)
        wire['steps'][0]['support_policy']='replace_once_v1'
        plan=compile_plan(wire,req,[],8,orientation_reference=resolve_reference(req),
                          strategy_failure_policy=POLICY)
        anchor=build_anchor(req,plan['steps'][0],reference=resolve_reference(req))
        self.assertIsNotNone(anchor['anchor'])
        self.assertNotIn('8',anchor['anchor']['support_ids'])
        self.assertFalse(plan['physics_success_implied'])
        self.assertEqual(req,original)

    def test_real_provider_and_both_ledgers_reject_then_new_plan_same_worker_id(self):
        service,req=self.make_service();before=digest(req)
        result=service.handle(req)
        self.assertFalse(is_halt_response(result),result)
        self.assertEqual(result['choice']['part_id'],'8')
        self.assertNotIn('8',result['choice']['support_ids'])
        validate_choice(req,result['choice'])
        self.assertEqual(result['request_id'],req['request_id'])
        self.assertEqual(service.planner_calls,2)
        self.assertEqual(len(self.bodies),2)
        self.assertEqual(digest(req),before)
        self.assertNotEqual(self.bodies[0],self.bodies[1])
        second=json.loads(self.bodies[1]['input'][0]['content'])
        self.assertIn('strategy_first_step_support_conflict',json.dumps(second))
        self.assertFalse(service.planner_decisions[0]['result']['metadata']['llm_live'])
        self.assertEqual(service.planner_decisions[0]['result']['metadata']['plan_contract_error'],
                         'strategy_first_step_support_conflict')
        from contextlib import closing
        with closing(sqlite3.connect(self.folder/'planner.sqlite')) as db:
            rows=db.execute('SELECT request_id,reserved_call FROM rank_requests').fetchall()
        self.assertEqual(len(rows),2)
        self.assertEqual(len({r[0] for r in rows}),2)
        self.assertEqual(sum(r[1] for r in rows),2)
        summary=self.shared.summary()
        self.assertEqual(summary['live_api_attempts'],0)
        self.assertEqual(summary['injected_provider_attempts'],2)
        self.assertEqual(summary['automatic_http_retries'],0)
        self.assertEqual(service.handle(deepcopy(req)),result)
        self.assertEqual(len(self.bodies),2)

    def test_same_invalid_selection_ends_at_budget_without_state_change(self):
        service,req=self.make_service(correct_after_first=False,plan_calls=2)
        before=digest(req)
        result=service.handle(req)
        self.assertTrue(is_halt_response(result),result)
        self.assertEqual(len(self.bodies),2)
        self.assertEqual(service.planner_calls,2)
        self.assertEqual(self.shared.summary()['injected_provider_attempts'],2)
        self.assertEqual(digest(req),before)
        self.assertFalse(any(e['kind']=='strategy_proposal_issued' for e in service.events))


if __name__=='__main__':unittest.main()
