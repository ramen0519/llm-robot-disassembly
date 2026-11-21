"""Real saved input + scripted model/physics; NO paid request or simulator."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_agent_knowledge_recovery as run
from orchestrator.hierarchy_budget import HierarchyBudget
from orchestrator.lesson_memory import load_bundle
from orchestrator.explicit_action_provider_v5 import restore_request_v5
from worker.path_probe_round import canonical_path_round_response
from tests.test_hierarchical_planner import plan_value
from tests.test_plan_pose_continuity import wire


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='knowledge_recovery_OFFLINE_',dir=run.ROOT/'runs'))
        self.prepared=run.source_case(run.SPEC)
        self.prepared['source_budget']=run.source_budget(self.prepared)
        self.data=load_bundle(run.SOURCE/'knowledge_bundle.json')
        p=run.policy();p['enabled']=False
        self.budget=HierarchyBudget(self.root/'budget.sqlite',p,transport_failure_policy='latch_on_error_v1')
        self.env=patch.dict(os.environ,OPENAI_API_KEY='OFFLINE_FAKE_KEY');self.env.start();self.addCleanup(self.env.stop)
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NETWORK_FORBIDDEN'))
        self.net=net.start();self.addCleanup(net.stop);self.addCleanup(self.net.assert_not_called)
        self.bodies=[];self.physics_calls=[];self.direct=False

    def provider(self,url,headers,body,timeout):
        self.bodies.append(deepcopy(body));packet=json.loads(body['input'][0]['content'])
        if body['text']['format']['name']=='knowledge_tool_requests':
            ctx=packet['context'];operations=[]
            if self.direct:
                value=dict(request_id=packet['request_id'],state_id=packet['state_id'],finish=True,
                    reason='Explicit fixture declines knowledge',operations=[])
            else:
                if not ctx['tool_history']:
                    operations=[dict(tool='search',query='',kind=kind,trigger='all',offset=0,limit=4,document_id=None)
                        for kind in ('skill','lesson','memory')]
                else:
                    ids=[r['document_id'] for operation in ctx['tool_history'][-1]['results']
                        for r in operation['result']['results']][:4]
                    operations=[dict(tool='read',query=None,kind=None,trigger=None,offset=None,limit=None,document_id=i) for i in ids]
                value=dict(request_id=packet['request_id'],state_id=packet['state_id'],finish=False,
                    reason='SCRIPTED offline branch coverage',operations=operations)
        else:
            r=restore_request_v5(packet['observation']);cards=packet['registered_skills']
            if self.direct:
                choice=canonical_path_round_response(r,None,fallback_policy='path_stage_v1')['choice']
                value=wire(plan_value(r,parts=[choice['part_id']],direction=choice['direction_id']))
                s=value['steps'][0];s.update(pose_reference=choice['pose_id'],support_ids=choice['support_ids'],
                    primitive=choice['skill_id'],second_direction=choice['second_direction_id'],evidence_ids=choice['evidence_ids'])
                value['skill_review']=[]
            else:
                self.assertTrue(packet['retrieved_knowledge']['documents'])
                c=cards[0];self.assertIsNotNone(c['currently_proposed_choice'])
                value=dict(request_id=r['request_id'],state_id=r['state_id'],candidate_order=r['candidate_ids'],
                    steps=[dict(part_id=c['currently_proposed_choice']['part_id'],mode='registered_skill',
                        pose_reference=None,primitive=None,direction=None,second_direction=None,
                        support_ids=[],evidence_ids=[],registered_skill_id=c['skill_id'],pose_policy=None)],
                    recovery_skill_ids=[],skill_review=[dict(skill_id=c['skill_id'],disposition='execute_now',
                        reason='SCRIPTED selection, not evidence of live reasoning')])
        return dict(status='completed',model=body['model'],error=None,id='OFFLINE_FIXTURE',
            output=[dict(type='message',role='assistant',status='completed',
                content=[dict(type='output_text',text=json.dumps(value))])])

    def physics(self,packet,folder,source_budget):
        self.physics_calls.append(deepcopy(packet));folder.mkdir(exist_ok=False)
        if len(self.physics_calls)==1:return deepcopy(self.prepared['raw'])
        # Synthetic success ONLY for testing flow; independent binding tests below.
        return dict(status='PASS',path_success=True,stability_success=True,elapsed_seconds=1,
            path_check_calls=1,stability_check_calls=1,chosen=deepcopy(packet['choice']))

    def test_search_read_selected_vm_output_reaches_physics_without_extra_model(self):
        original=run.check_physics
        def checked(p,r,c,raw):return original(p,r,c,raw) if 'scope' in raw else True
        with patch.object(run,'check_physics',side_effect=checked):
            result=run.trial(self.root,self.prepared,self.data,self.budget,physics=self.physics,backend=self.provider)
        self.assertEqual(result['status'],'SKILL_RECOVERY_PASS')
        self.assertEqual(result['physics_attempts'],2);self.assertEqual(len(self.bodies),3)
        self.assertTrue(result['selection']['skill_selected']);self.assertTrue(result['selection']['read_ids'])
        self.assertEqual(self.physics_calls[1]['choice'],result['selection']['response']['choice'])
        self.assertEqual(result['actual_api_calls'],0);self.assertTrue(result['injected_test'])
        self.assertEqual(self.physics_calls[1]['context']['remaining_wall_seconds'],self.prepared['source_budget']['seconds'])
        self.assertNotIn('stability_timeout',self.physics_calls[1]['context'])

    def test_direct_action_is_allowed_and_not_labelled_skill_reuse(self):
        self.direct=True
        with patch.object(run,'check_physics',return_value=True):
            result=run.trial(self.root,self.prepared,self.data,self.budget,physics=self.physics,backend=self.provider)
        self.assertEqual(result['status'],'DIRECT_ACTION_PASS_NO_SKILL')
        self.assertFalse(result['selection']['skill_selected']);self.assertEqual(len(self.bodies),2)

    def test_successful_trigger_stops_before_any_model_or_recovery(self):
        def success(*args,**kwargs):return dict(status='PASS',path_success=True,stability_success=True)
        with patch.object(run,'check_physics',return_value=True):
            result=run.trial(self.root,self.prepared,self.data,self.budget,physics=success,backend=self.provider)
        self.assertEqual(result['status'],'TRIGGER_NOT_REPRODUCED')
        self.assertEqual(self.bodies,[]);self.assertEqual(result['physics_attempts'],1)

    def test_transport_failure_stops_without_recovery_or_retry(self):
        calls=[]
        def timeout(*args):calls.append(1);raise TimeoutError('fixture')
        result=run.trial(self.root,self.prepared,self.data,self.budget,physics=self.physics,backend=timeout)
        self.assertEqual(result['status'],'NO_VALID_LIVE_DECISION')
        self.assertEqual(calls,[1]);self.assertEqual(result['physics_attempts'],1)
        self.assertTrue(self.budget.transport_blocked())

    def test_raw_pass_after_deadline_does_not_count_as_recovery_success(self):
        self.direct=True
        with patch.object(run,'check_physics',side_effect=[True,False]):
            result=run.trial(self.root,self.prepared,self.data,self.budget,physics=self.physics,backend=self.provider)
        self.assertEqual(result['status'],'SELECTED_ACTION_NOT_PASS')
        self.assertFalse(result['recovery']['within_original_allowance'])

    def test_real_saved_binding_rejects_changed_solver_or_support_policy(self):
        p=self.prepared
        self.assertTrue(run.check_physics(p,p['request'],p['choice'],p['raw']))
        changed=deepcopy(p['raw']);changed['source_hashes']['explicit_executor_sha256']='0'*64
        with self.assertRaises(ValueError):run.check_physics(p,p['request'],p['choice'],changed)
        packet=run.packet_for(p,p['request'],p['choice'],'test')
        packet['context']['stability_timeout']=30
        from tools.python_source_budget import validate_binding
        with self.assertRaises(ValueError):validate_binding(packet['context'],p['source_budget'])


if __name__=='__main__':unittest.main()
