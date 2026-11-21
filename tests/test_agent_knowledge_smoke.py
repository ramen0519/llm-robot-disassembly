"""Saved-input runner with injected HTTP; never live API or new physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_agent_knowledge_smoke as smoke
from orchestrator.hierarchy_budget import HierarchyBudget
from orchestrator.lesson_memory import load_bundle
from orchestrator.explicit_action_provider_v5 import restore_request_v5,encode_provider_decision_v5
from worker.path_probe_round import canonical_path_round_response
from tests.test_hierarchical_planner import plan_value
from tests.test_planning_repair_review import reviewed


class KnowledgeSmokeTests(unittest.TestCase):
    def setUp(self):
        self.folder=Path(tempfile.mkdtemp(prefix='agent_smoke_OFFLINE_',dir=smoke.ROOT/'runs'))
        self.request=smoke.read(smoke.REQUEST)
        self.data=load_bundle(smoke.SOURCE/'knowledge_bundle.json')
        self.policy=smoke.make_policy();self.policy['enabled']=False
        self.env=patch.dict(os.environ,OPENAI_API_KEY='OFFLINE_NOT_A_KEY');self.env.start();self.addCleanup(self.env.stop)
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('HTTP_FORBIDDEN'))
        mock=net.start();self.addCleanup(net.stop);self.addCleanup(mock.assert_not_called)

    def backend(self,url,headers,body,timeout):
        p=json.loads(body['input'][0]['content'])
        if body['text']['format']['name']=='knowledge_tool_requests':
            value=dict(request_id=p['request_id'],state_id=p['state_id'],finish=True,reason='Offline decline fixture',operations=[])
        elif body['model']=='gpt-6-luna':
            req=restore_request_v5({k:p[k] for k in ('projection_version','request','shared_source_contexts')})
            value=encode_provider_decision_v5(req,canonical_path_round_response(req,None,fallback_policy='path_stage_v1'))
        else:
            req=restore_request_v5(p['observation'])
            # A disassembly plan leaves one terminal part; do not remove all N.
            value=reviewed(plan_value(req,parts=sorted(req['candidate_ids'])[:-1]),p['registered_skills'])
        return dict(id='OFFLINE',model=body['model'],status='completed',error=None,
            output=[dict(type='message',role='assistant',status='completed',content=[dict(type='output_text',text=json.dumps(value))])])

    def test_both_roles_use_existing_public_services_without_physics(self):
        budget=HierarchyBudget(self.folder/'global.sqlite',self.policy,transport_failure_policy='latch_on_error_v1')
        result=smoke.run_roles(self.folder,budget,self.request,self.data,dict(scout=self.backend,planner=self.backend))
        self.assertEqual(set(result),{'scout','planner'})
        self.assertEqual(result['planner']['plan_status'],'planned')
        self.assertEqual(result['scout']['fallback_reason'],None)
        self.assertEqual(budget.summary()['transport_attempts'],4)
        self.assertEqual(budget.summary()['live_api_attempts'],0)
        self.assertTrue(all(v['physics_attempts']==0 for v in result.values()))

    def test_connection_error_stops_other_role_without_retry(self):
        budget=HierarchyBudget(self.folder/'global.sqlite',self.policy,transport_failure_policy='latch_on_error_v1')
        calls=[]
        def broken(*args):calls.append(1);raise TimeoutError()
        result=smoke.run_roles(self.folder,budget,self.request,self.data,dict(scout=broken,planner=self.backend))
        self.assertEqual(result['planner']['status'],'NOT_RUN_TRANSPORT_CIRCUIT_OPEN')
        self.assertEqual(calls,[1]);self.assertEqual(budget.summary()['transport_attempts'],1)

    def test_policy_exact_six_call_one_dollar_and_no_learning(self):
        policy=smoke.make_policy()
        self.assertEqual(policy['max_calls'],6);self.assertEqual(policy['max_usd'],'1.00')
        self.assertEqual(policy['roles']['learner']['max_calls'],0)
        self.assertEqual(policy['episode']['scout_calls'],0)
        self.assertEqual(policy['roles']['planner']['max_calls'],3)
        self.assertEqual(policy['roles']['scout']['max_calls'],3)


if __name__=='__main__':unittest.main()
