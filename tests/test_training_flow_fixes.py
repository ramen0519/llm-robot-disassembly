"""Saved actual evidence and injected providers only. No API or new physics."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from tests import test_agent_knowledge as fixtures
from tests.test_hierarchical_planner import policy, plan_value, request
from tests.test_explicit_action_service import provider
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.knowledge_retrieval import KnowledgeRetrieval, KnowledgeRetrievalService
from orchestrator.knowledge_tools import KnowledgeCatalog
from tools.run_luna_experiment import read, write
from tools import prepare_multitask_roles as roles
from tools.skill_nonapplicability import bind_cases, post_pass_state, audit

ROOT=roles.ROOT
COLLECTION=ROOT/'runs/luna_sol_training_20261109_0216/live/collection.json'
SPLIT=ROOT/'runs/multitask_learning_prep_20261105/prepared_v2/split.json'


class RetrievalFixTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixtures.AgentKnowledgeTests();self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        network=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        network.start();self.addCleanup(network.stop)

    def test_current_failures_only_need_no_search_provider(self):
        catalog=KnowledgeCatalog(read(ROOT/'runs/multitask_learning_prep_20261105/final_proposal/empty_knowledge/skills.json'))
        self.assertTrue(catalog.session(self.fx.req).documents)
        send=Mock(side_effect=AssertionError('NO_REDUNDANT_LOOKUP'))
        view=KnowledgeRetrieval(catalog,self.fx.owner(send)).retrieve(self.fx.req,time.monotonic()+90)
        self.assertEqual(view['audit']['stop_reason'],'no_additional_knowledge')
        self.assertEqual(view['audit']['provider_attempts'],0)
        self.assertEqual(view['audit']['request_evidence_ids'],self.fx.req['evidence_ids'])
        send.assert_not_called()

    def test_finish_with_final_read_executes_once(self):
        def send(url,headers,body,timeout):
            packet=json.loads(body['input'][0]['content'])
            raw=provider(dict(request_id=packet['request_id'],state_id=packet['state_id'],
                finish=True,reason='last batch',operations=[fixtures.read('skill:'+self.fx.sid)]))
            raw['model']=body['model'];return raw
        call=Mock(side_effect=send)
        view=KnowledgeRetrieval(self.fx.catalog,self.fx.owner(call)).retrieve(self.fx.req,time.monotonic()+90)
        self.assertEqual(view['audit']['selected_skill_ids'],[self.fx.sid])
        self.assertEqual(view['audit']['stop_reason'],'model_finished');self.assertEqual(call.call_count,1)

    def test_bad_identity_is_diagnosed_and_no_values_are_echoed(self):
        owner=self.fx.owner();service=KnowledgeRetrievalService.for_owner(owner)
        meta={'model_returned':owner.config['model']}
        value,error=service._parse_response(provider(dict(request_id='wrong-private-value',state_id='s',
            finish=True,reason='',operations=[])),{'request_id':'r','state_id':'s'},meta)
        self.assertIsNone(value);self.assertEqual(error,'invalid_knowledge_tool_response')
        self.assertEqual(meta['knowledge_contract_diagnostics']['field'],'request_id')
        self.assertNotIn('wrong-private-value',json.dumps(meta))

    def test_bad_lookup_still_gets_one_real_plan_slot_without_retrieved_material(self):
        sent=[]
        def send(url,headers,body,timeout):
            sent.append(deepcopy(body));packet=json.loads(body['input'][0]['content'])
            if body['text']['format']['name']=='knowledge_tool_requests':
                value=dict(request_id='wrong',state_id=packet['state_id'],finish=True,reason='',operations=[])
            else:
                from orchestrator.explicit_action_provider_v5 import restore_request_v5
                self.assertEqual(packet['registered_skills'],[])
                self.assertEqual(packet['retrieved_knowledge']['documents'],[])
                value=plan_value(restore_request_v5(packet['observation']))
            raw=provider(value);raw['model']=body['model'];return raw
        limits=policy()['episode'];limits.update(scout_calls=0,planner_calls=1)
        svc=HierarchicalDecisionService(Mock(),self.fx.owner(send),self.fx.skills,limits,
            allow_fixture=True,lesson_library=self.fx.lessons,knowledge_policy='agent_tools_v1')
        result=svc.handle(request())
        self.assertEqual(result['metadata']['hierarchy_role'],'plan_action')
        self.assertEqual(svc.planner_calls,1);self.assertEqual(len(sent),2)
        self.assertEqual(svc.knowledge_queries[-1]['audit']['retrieval_status'],'rejected')
        svc.handle(request());self.assertEqual(len(sent),2)

    def test_transport_failure_does_not_consume_plan_slot_or_fall_through(self):
        send=Mock(side_effect=TimeoutError())
        limits=policy()['episode'];limits.update(scout_calls=0,planner_calls=1)
        svc=HierarchicalDecisionService(Mock(),self.fx.owner(send),self.fx.skills,limits,
            allow_fixture=True,lesson_library=self.fx.lessons,knowledge_policy='agent_tools_v1')
        svc.handle(request())
        self.assertEqual(svc.planner_calls,0);self.assertEqual(send.call_count,1)
        self.assertEqual(svc.knowledge_queries[-1]['audit']['stop_reason'],'api_timeout')

    def test_deadline_before_plan_does_not_consume_slot(self):
        owner=self.fx.owner();limits=policy()['episode'];limits['scout_calls']=0
        svc=HierarchicalDecisionService(Mock(),owner,self.fx.skills,limits,allow_fixture=True,
            lesson_library=self.fx.lessons,knowledge_policy='agent_tools_v1')
        view=self.fx.catalog.session(request()).view();view['audit'].update(retrieval_status='rejected',stop_reason='knowledge_deadline')
        with patch.object(svc.knowledge_agent,'retrieve',return_value=view),patch.object(owner,'plan') as plan:
            svc.handle(request());plan.assert_not_called()
        self.assertEqual(svc.planner_calls,0)


class StateRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='state_roles_OFFLINE_',dir=ROOT/'runs')
        cls.root=Path(cls.temp.name)
        with patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_API')):
            cls.path=roles.prepare(COLLECTION,SPLIT,cls.root/'prepared',role_policy='observed_states_v3')
        cls.manifest=read(cls.path)
        cls.collection=read(COLLECTION);cls.split=read(SPLIT)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        network=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_API'))
        network.start();self.addCleanup(network.stop)

    def test_failed_episodes_now_prepare_without_relabeling_results(self):
        with self.assertRaisesRegex(ValueError,'all_PASS'):roles.select_roles(self.collection,self.split)
        choice=roles.select_state_roles(self.collection,self.split)
        self.assertEqual(choice['family'],'dynamic_stability_failed')
        self.assertEqual(choice['generation_tasks'],['07009','02628'])
        self.assertEqual([c['attempt_index'] for c in choice['healthy_cases']],[4,4])
        for c in choice['healthy_cases']:
            report=read(Path(c['row']['report']))
            self.assertFalse(report['success'])
            self.assertEqual(report['attempts'][3]['status'],'UNKNOWN')
        rows=bind_cases(self.manifest)
        self.assertEqual([len(c['request']['remaining_parts']) for c in rows],[6,8])
        self.assertTrue(any(f['status']=='UNKNOWN' for f in rows[0]['request']['episode_feedback']))
        self.assertEqual(rows[0]['source_event_id'],self.manifest['nonapplicability_cases'][0]['state_provenance']['predecessor_event_id'])

    def test_future_action_outcome_is_not_a_normal_state_selection_criterion(self):
        spec=roles.select_state_roles(self.collection,self.split)['healthy_cases'][0]
        report=read(Path(spec['row']['report']));before=post_pass_state(report,4)
        report['attempts'][3]['status']='ERROR';report['attempts'][3]['raw']={}
        self.assertEqual(post_pass_state(report,4),before)

    def test_late_unaccepted_or_nonpass_predecessor_cannot_create_normal_case(self):
        report=read(Path(self.collection['episodes'][1]['report']))
        for field,value in [('deadline_exceeded',True),('accepted_for_transition',False),('status','UNKNOWN')]:
            with self.subTest(field=field):
                bad=deepcopy(report);bad['attempts'][2][field]=value
                with self.assertRaisesRegex(ValueError,'accepted_predecessor'):post_pass_state(bad,4)

    def test_transition_feedback_tampering_is_rejected(self):
        report=read(Path(self.collection['episodes'][1]['report']))
        req=next(d['request'] for d in report['decisions'] if d['request']['request_id']==report['attempts'][3]['request_id'])
        req['episode_feedback'][-1]['accepted_for_transition']=False
        with self.assertRaises(ValueError):post_pass_state(report,4)

    def test_negative_provenance_duplicate_and_evaluation_tampering_rejected(self):
        for mutation in ('provenance','duplicate','evaluation'):
            value=deepcopy(self.manifest)
            if mutation=='provenance':value['nonapplicability_cases'][0]['state_provenance']['predecessor_raw_sha256']='bad'
            elif mutation=='duplicate':value['nonapplicability_cases'][1]=dict(value['nonapplicability_cases'][0],case_id='duplicate')
            else:value['role_assignment']['nonapplicability_tasks']=['02680']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):bind_cases(value)

    def test_no_observed_normal_states_still_blocks_preparation(self):
        with patch('tools.skill_nonapplicability.post_pass_state',side_effect=ValueError('no actual PASS')):
            with self.assertRaisesRegex(ValueError,'two_observed_post_PASS'):roles.select_state_roles(self.collection,self.split)

    def test_real_vm_abstention_gate_is_not_removed(self):
        rows=bind_cases(self.manifest)
        none=dict(language='python_policy_v1',source='def propose(ctx):\n    return None\n')
        self.assertTrue(audit(none,rows)['accepted'])
        wrong=dict(none,source='def propose(ctx):\n    return ctx["missing"]\n')
        self.assertFalse(audit(wrong,rows)['accepted'])

    def test_new_manifest_reaches_source_bound_learning_preflight(self):
        from tools.run_multitask_learning import learn
        out=self.root/'learn_preflight';out.mkdir();calls=[]
        result=learn(out,lambda *args:calls.append(args) or 1,Mock(),
            learning_inputs=(self.path,self.path.parent/'lesson_packet.json'))
        self.assertEqual(result['status'],'LESSON_REJECTED') # injected stop, not real model rejection
        self.assertEqual([c[0] for c in calls],['lesson_extract'])
        self.assertEqual(len(read(out/'learning_observation_contract.json')['cases']),4)

    def test_batch_new_planning_allowance_remains_inside_shared_caps(self):
        from tools.run_luna_sol_training_batch import policy as batch_policy
        from orchestrator.hierarchy_budget import validate_policy
        p=batch_policy();self.assertEqual(p['episode']['planner_calls'],6)
        self.assertEqual(p['max_calls'],80);self.assertFalse(p['enabled'])
        p['episode']['planner_calls']=7
        with self.assertRaises(ValueError):validate_policy(p)

    def test_v3_cli_preview_uses_actual_revision_and_source_clock(self):
        from tools.run_lesson_discovery import extract
        from tools.run_skill_discovery import preview_config
        from tools import run_python_skill_cycle as cycle
        lesson_dir=self.root/'fixture_lesson'
        cfg=dict(preview_config(),max_output_tokens=4096)
        extract(read(self.path.parent/'lesson_packet.json'),lesson_dir,cfg,mode='fixture')
        out=self.root/'preview_candidate'
        with patch.object(cycle,'run_case',side_effect=AssertionError('NO_PHYSICS')) as physics, \
             patch.object(cycle.SkillRegistry,'propose',side_effect=AssertionError('NO_REGISTRY')) as registry:
            code=cycle.main(['--mode','preview','--run-dir',str(out),'--manifest',str(self.path),
                '--lesson',str(lesson_dir/'lesson.json'),'--observation-contract','source_bound_v1',
                '--python-revision-contract','--python-revision-version','2'])
        self.assertEqual(code,0);physics.assert_not_called();registry.assert_not_called()
        self.assertLess(read(out/'api_preview.json')['body_bytes'],196608)
        self.assertEqual(len(read(out/'observation_bindings.json')),4)


if __name__=='__main__':unittest.main()
