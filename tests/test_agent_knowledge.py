"""Offline tool/ledger/decision integration. No live model or physics claims."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.knowledge_tools import KnowledgeCatalog
from orchestrator.knowledge_retrieval import KnowledgeRetrieval, KnowledgeRetrievalService, decision_evidence
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.hierarchy_budget import HierarchyBudget
from orchestrator.explicit_action_service import ExplicitActionService
from orchestrator.explicit_action_memory import append_action_observation, freeze_memory
from orchestrator.explicit_action_provider_v5 import restore_request_v5, encode_provider_decision_v5
from orchestrator.generated_skill_service import GeneratedSkillService
from orchestrator.skill_discovery import SkillDiscoveryService
from tests import test_knowledge_context as knowledge_fixtures
from tests.test_hierarchical_planner import policy, plan_value, feedback
from tests.test_skill_program import request
from tests.test_explicit_action_service import provider
from worker.explicit_action_contract import canonical_response, validate_response, digest


def search(query="", kind="all", trigger="all", offset=0):
    return dict(tool="search", query=query, kind=kind, trigger=trigger, offset=offset, limit=6, document_id=None)


def read(identity):
    return dict(tool="read", query=None, kind=None, trigger=None, offset=None, limit=None, document_id=identity)


class AgentKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = knowledge_fixtures.KnowledgeContextTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.req, self.skills, self.lessons = self.fixture.req, self.fixture.skills, self.fixture.lessons
        self.catalog = KnowledgeCatalog(self.skills, self.lessons, allow_fixture=True)
        env = patch.dict(os.environ, OPENAI_API_KEY="OFFLINE_NOT_A_KEY")
        env.start(); self.addCleanup(env.stop)
        self.sent = []
        self.sid = self.catalog.context._rows[0]['strategy_id']

    def owner(self, transport=None, cfg=None):
        value = SolPlanService(cfg or policy()['roles']['planner'], self.root/'planner.sqlite', transport or self.transport)
        return value

    def transport(self, url, headers, body, timeout):
        self.sent.append(deepcopy(body))
        packet = json.loads(body['input'][0]['content'])
        if body['text']['format']['name'] == 'knowledge_tool_requests':
            ctx = packet['context']
            if not ctx['tool_history']:
                operations = [search(kind="skill")]
            else:
                identities = [x['document_id'] for item in ctx['tool_history'][-1]['results']
                              for x in item['result'].get('results', [])]
                operations = [read(i) for i in identities[:4]]
            value = dict(request_id=packet['request_id'], state_id=packet['state_id'],
                         finish=False, reason="Offline scripted retrieval", operations=operations)
        else:
            req = restore_request_v5(packet['observation'])
            value = plan_value(req)
        raw = provider(value); raw['model'] = body['model']
        return raw

    def memory(self):
        r = request(); response = canonical_response(r, None)
        result = dict(status='UNKNOWN', path_success=False, stability_success=None,
                      verifier_origin='ASAP_RedMax', raw={'endpoint_observation': {'min_separation': 0.5}},
                      evidence_ref='runs/offline_fixture/result.json')
        append_action_observation(self.root/'obs.jsonl', r, response, result, allow_test_root=True)
        return freeze_memory(self.root/'obs.jsonl', self.root/'memory.json', allow_test_root=True)

    def test_search_and_read_scope_immutable_and_no_execution(self):
        before = deepcopy((self.skills, self.lessons, self.req))
        s = self.catalog.session(self.req)
        found = s.search(query='', kind='skill')
        doc = s.read(found['results'][0]['document_id'])
        self.assertEqual(doc['content']['program'], self.fixture.context.row(self.sid)['program'])
        self.assertTrue(doc['content']['lessons'])
        self.assertFalse(doc['content']['current_attempt_verified'])
        self.assertEqual(before, (self.skills, self.lessons, self.req))
        doc['content']['program']['source'] = 'changed'
        self.assertNotEqual(s.read(doc['document_id'])['content']['program']['source'], 'changed')
        self.assertEqual(s.read('E:/do-not-open')['error'], 'unknown_document_id')

    def test_full_pinned_memory_search_beyond_worker_topk(self):
        cat = KnowledgeCatalog(self.skills, self.lessons, self.memory(), allow_fixture=True)
        s = cat.session(self.req)
        found = s.search(query='', kind='memory', trigger='path_failed')
        frozen = [d for d in s.documents.values() if d['document_id'].startswith('memory:')]
        self.assertEqual(len(frozen), 1)
        doc = s.read(frozen[0]['document_id'])
        self.assertIsNone(doc['content']['accepted_for_transition'])
        self.assertIsNone(doc['content']['deadline_exceeded'])
        self.assertFalse(doc['content']['conditions_equivalent'])
        self.assertNotIn(doc['content']['evidence_id'], self.req['evidence_ids'])
        view = s.view()
        self.assertEqual(view['audit']['request_evidence_ids'], self.req['evidence_ids'])
        self.assertFalse(view['audit']['new_citation_ids_added'])

    def test_search_pagination_filters_and_unknown_no_invented_match(self):
        s = self.catalog.session(self.req)
        self.assertEqual(s.search(query='unfindabletokenzz')['total_matches'], 0)
        first = s.search(query='', kind='memory', limit=1)
        second = s.search(query='', kind='memory', offset=1, limit=1)
        self.assertNotEqual(first['results'][0]['document_id'], second['results'][0]['document_id'])
        with self.assertRaises(ValueError): s.search(query='', limit=500)
        with self.assertRaises(ValueError): s.execute(dict(tool='shell', command='anything'))

    def test_large_catalog_not_truncated_to_initial_preview(self):
        s = self.catalog.session(self.req)
        base = next(d for d in s.documents.values() if d['kind']=='skill')
        for n in range(50):
            item = deepcopy(base); item['document_id'] = 'skill:synthetic-'+str(n).zfill(2)
            item['content']['card'] = {'name': 'needle'+str(n)}
            s.documents[item['document_id']] = item
        self.assertEqual(len(s.index()['preview']['skill']), 8)
        self.assertEqual(s.search(query='needle49', kind='skill')['results'][0]['document_id'], 'skill:synthetic-49')

    def test_same_model_two_rounds_metered_cached_and_cannot_execute_tools(self):
        owner = self.owner()
        agent = KnowledgeRetrieval(self.catalog, owner)
        view = agent.retrieve(self.req, time.monotonic()+90)
        self.assertEqual(view['audit']['provider_attempts'], 2)
        self.assertEqual(view['audit']['live_provider_attempts'], 0)
        self.assertEqual(view['audit']['selected_skill_ids'], [self.sid])
        self.assertEqual(agent.service.ledger_path, owner.ledger_path)
        self.assertEqual(agent.service.config_fingerprint, owner.config_fingerprint)
        cached = agent.retrieve(self.req, time.monotonic()+90)
        self.assertTrue(cached['audit']['retrieval_cache_hit']); self.assertEqual(len(self.sent), 2)
        self.assertTrue(all(b['model'] == owner.config['model'] for b in self.sent))

    def test_call_budget_blocks_final_plan_without_an_extra_http_attempt(self):
        cfg = policy()['roles']['planner']; cfg['max_calls'] = 2
        owner = self.owner(cfg=cfg)
        view = KnowledgeRetrieval(self.catalog, owner).retrieve(self.req, time.monotonic()+90)
        result = owner.plan(self.req, [], {'retrieved_knowledge': decision_evidence(view)})
        self.assertEqual(result['fallback_reason'], 'call_budget_exhausted')
        self.assertEqual(len(self.sent), 2)

    def test_cost_budget_prevents_any_transport(self):
        cfg = policy()['roles']['planner']; cfg['max_usd'] = '0.0000001'
        view = KnowledgeRetrieval(self.catalog, self.owner(cfg=cfg)).retrieve(self.req, time.monotonic()+90)
        self.assertEqual(view['audit']['stop_reason'], 'cost_budget_exhausted')
        self.assertEqual(self.sent, [])

    def test_timeout_no_retry_and_no_planner_fallthrough(self):
        transport = Mock(side_effect=TimeoutError())
        owner = self.owner(transport)
        limits = policy()['episode']; limits.update(scout_calls=0, fallback_calls=2)
        scout = Mock()
        service = HierarchicalDecisionService(scout, owner, self.skills, limits, allow_fixture=True,
            lesson_library=self.lessons, knowledge_policy='agent_tools_v1')
        response = service.handle(self.req)
        validate_response(self.req, response)
        self.assertEqual(service.knowledge_queries[-1]['audit']['stop_reason'], 'api_timeout')
        self.assertEqual(transport.call_count, 1); scout.handle.assert_not_called()
        service.handle(self.req); self.assertEqual(transport.call_count, 1)

    def test_wrong_response_identity_fails_closed(self):
        raw = provider(dict(request_id='wrong', state_id='wrong', finish=True, reason='', operations=[]))
        raw['model'] = policy()['roles']['planner']['model']
        owner = self.owner(Mock(return_value=raw))
        view = KnowledgeRetrieval(self.catalog, owner).retrieve(self.req, time.monotonic()+90)
        self.assertEqual(view['audit']['stop_reason'], 'invalid_knowledge_tool_response')
        self.assertEqual(view['skills'], [])

    def test_duplicate_search_not_executed_twice_and_round_limit(self):
        def duplicate(url, headers, body, timeout):
            packet=json.loads(body['input'][0]['content'])
            raw=provider(dict(request_id=packet['request_id'], state_id=packet['state_id'],
                finish=False, reason='fixture', operations=[search(),search()]))
            raw['model']=body['model']; return raw
        view=KnowledgeRetrieval(self.catalog, self.owner(duplicate)).retrieve(self.req,time.monotonic()+90)
        history=view['retrieval_trace']['tool_history']
        self.assertEqual(history[0]['results'][1]['result']['error'], 'duplicate_tool_operation_not_repeated')
        self.assertEqual(history[1]['results'][0]['result']['error'], 'duplicate_tool_operation_not_repeated')
        self.assertEqual(view['audit']['stop_reason'], 'knowledge_tool_errors')
        self.assertEqual(view['audit']['retrieval_status'], 'rejected')
        self.assertEqual(view['audit']['provider_attempts'], 2)
        self.assertEqual(view['skills'], [])

    def test_hierarchical_real_provider_path_to_plan_and_cached_next_action(self):
        limits=policy()['episode']; limits['scout_calls']=0
        owner=self.owner()
        service=HierarchicalDecisionService(Mock(),owner,self.skills,limits,allow_fixture=True,
            lesson_library=self.lessons,knowledge_policy='agent_tools_v1')
        # Initial state with no failed actions permits a complete conditional plan.
        req=request()
        response=service.handle(req)
        self.assertEqual(response['metadata']['hierarchy_role'],'plan_action', response)
        packet=json.loads(self.sent[-1]['input'][0]['content'])
        self.assertEqual(packet['registered_skills'][0]['skill_id'],self.sid)
        self.assertEqual(packet['retrieved_knowledge']['audit']['selected_skill_ids'],[self.sid])
        self.assertEqual(len(self.sent),3)
        next_request=feedback(req,response)
        second=service.handle(next_request)
        self.assertEqual(second['metadata']['hierarchy_role'],'plan_action')
        self.assertEqual(len(self.sent),3)  # No retrieval or planning call per next part.

    def test_shared_global_budget_counts_search_and_final_plan(self):
        p=policy(); b=HierarchyBudget(self.root/'global.sqlite',p)
        owner=self.owner(b.transport('planner',self.transport))
        KnowledgeRetrieval(self.catalog,owner).retrieve(self.req,time.monotonic()+90)
        self.assertEqual(b.summary()['roles']['planner'],2)
        self.assertEqual(b.summary()['live_api_attempts'],0)

    def test_no_secret_or_general_file_tools_and_read_cap(self):
        session=self.catalog.session(self.req)
        doc=next(iter(session.documents.values()))
        doc['content']['oversized']='x'*100000
        self.assertEqual(session.read(doc['document_id'])['error'],'knowledge_read_byte_limit')
        owner=self.owner(); service=KnowledgeRetrievalService.for_owner(owner)
        packet=dict(kind='agent_tools_v1',request=dict(request_id='lookup',state_id='s',context=session.index()),timeout_seconds=20)
        body=service._body(packet)
        self.assertNotIn('OFFLINE_NOT_A_KEY',json.dumps(body))
        self.assertNotIn('tools',body)  # Bounded JSON protocol, no arbitrary vendor tools.

    def test_explicit_fallback_receives_searched_memory_without_new_citation_ids(self):
        session=KnowledgeCatalog(self.skills,self.lessons,self.memory(),allow_fixture=True).session(self.req)
        identity=next(k for k in session.documents if k.startswith('memory:'))
        session.read(identity); view=session.view()
        actor=ExplicitActionService(policy()['roles']['scout'],self.root/'actor.sqlite',Mock(),
            provider_contract='explicit_action_provider_v5',fallback_policy='path_stage_v1')
        envelope=dict(request=self.req,timeout_seconds=30,retrieved_knowledge=decision_evidence(view))
        actor.service._validate_envelope(envelope)
        body=actor.service._body(envelope)
        content=json.loads(body['input'][0]['content'])
        self.assertEqual(content['retrieved_knowledge']['documents'][0]['document_id'],identity)
        self.assertEqual(content['retrieved_knowledge']['audit']['request_evidence_ids'],self.req['evidence_ids'])
        bad=deepcopy(envelope);bad['retrieved_knowledge']['audit']['request_sha256']='bad'
        with self.assertRaises(ValueError):actor.service._validate_envelope(bad)

    def test_luna_search_read_select_executes_real_local_vm_with_fixture_registry_boundary(self):
        # Admission is injected ONLY in this unit test, as in the established
        # GeneratedSkillService tests. No synthetic admission is written/live.
        from tests.test_generated_skill_service import row, snapshot
        from tests.test_skill_program import append_observation
        from tests.test_skill_discovery import raw_response
        rows=[row()]; cfg=policy()['roles']['scout']; cfg['max_calls']=4
        sent=[]
        def transport(url, headers, body, timeout):
            sent.append(deepcopy(body)); packet=json.loads(body['input'][0]['content'])
            if body['text']['format']['name']=='knowledge_tool_requests':
                value=dict(request_id=packet['request_id'], state_id=packet['state_id'], finish=False,
                    reason='offline fixture', operations=[read('skill:'+rows[0]['strategy_id'])])
            else:
                value=dict(request_id=packet['request_id'],state_id=packet['state_id'],
                    selected_skill_id=rows[0]['strategy_id'],evidence_ids=[],explanation_hypothesis='offline fixture')
            raw=raw_response(value);raw['model']=body['model'];return raw
        actor=ExplicitActionService(cfg,self.root/'luna.sqlite',transport,
            provider_contract='explicit_action_provider_v5',fallback_policy='path_stage_v1')
        selector=SkillDiscoveryService(cfg,actor.service.ledger_path,transport)
        with patch('orchestrator.generated_skill_service.retrieve', return_value=deepcopy(rows)), \
             patch('orchestrator.knowledge_context.retrieve', return_value=deepcopy(rows)):
            service=GeneratedSkillService(snapshot(),actor,selector,knowledge_policy='agent_tools_v1',knowledge_max_rounds=1)
            response=service.handle(append_observation(request()))
        self.assertTrue(response['metadata']['generated_program_executed'],response)
        self.assertFalse(response['metadata']['physics_bypass'])
        self.assertFalse(response['metadata']['llm_live'])
        self.assertEqual(response['metadata']['generated_skill_id'],rows[0]['strategy_id'])
        self.assertEqual(len(sent),2)
        self.assertEqual(json.loads(sent[-1]['input'][0]['content'])['context']['retrieved_knowledge']['audit']['selected_skill_ids'],[rows[0]['strategy_id']])

    def test_empty_or_declined_knowledge_still_allows_normal_plan(self):
        def finish(url, headers, body, timeout):
            if body['text']['format']['name']!='knowledge_tool_requests':
                return self.transport(url,headers,body,timeout)
            self.sent.append(deepcopy(body)); p=json.loads(body['input'][0]['content'])
            raw=provider(dict(request_id=p['request_id'],state_id=p['state_id'],finish=True,reason='No relevant knowledge',operations=[]))
            raw['model']=body['model'];return raw
        owner=self.owner(finish);limits=policy()['episode'];limits['scout_calls']=0
        svc=HierarchicalDecisionService(Mock(),owner,self.skills,limits,allow_fixture=True,
            lesson_library=self.lessons,knowledge_policy='agent_tools_v1')
        response=svc.handle(request())
        self.assertEqual(response['metadata']['hierarchy_role'],'plan_action')
        self.assertEqual(svc.knowledge_queries[-1]['skills'],[])
        self.assertEqual(len(self.sent),2)

    def test_no_fresh_physics_is_inferred_from_search_and_wrong_domain_is_excluded(self):
        memory=self.memory(); record=memory['records'][0]
        record['result']['verifier_origin']='MuJoCo_robot'
        record['record_sha256']=digest({k:v for k,v in record.items() if k!='record_sha256'})
        memory['content_sha256']=digest(memory['records'])
        cat=KnowledgeCatalog(self.skills,self.lessons,memory,allow_fixture=True)
        self.assertEqual(len(cat.excluded_memory),1)
        self.assertFalse(any(k.startswith('memory:') for k in cat._documents))

    def test_closed_deadline_does_not_call_provider(self):
        owner=self.owner()
        view=KnowledgeRetrieval(self.catalog,owner).retrieve(self.req,time.monotonic()-1)
        self.assertEqual(view['audit']['stop_reason'],'knowledge_deadline')
        self.assertEqual(self.sent,[])

    def test_conflicting_current_evidence_not_silently_replaced(self):
        req=deepcopy(self.req)
        req['episode_feedback'].append(dict(req['episode_feedback'][0],status='ERROR'))
        with self.assertRaisesRegex(ValueError,'conflicting_knowledge_evidence_id'):
            self.catalog.session(req)

    def test_cli_fixture_requires_explicit_search_decision_and_counts_it(self):
        from tools.run_hierarchical_episode import InjectedProvider, parser
        self.assertEqual(parser().get_default('knowledge_policy'),'legacy_v1')
        owner=self.owner();provider_service=KnowledgeRetrievalService.for_owner(owner)
        envelope=dict(kind='agent_tools_v1',request=dict(request_id='k',state_id='s',context={}),timeout_seconds=20)
        body=provider_service._body(envelope)
        fixture=InjectedProvider({'planner':[{'knowledge_operations':[],'finish':True}], 'scout':[]})
        raw=fixture.transport('planner')('offline',{},body,20)
        value=json.loads(raw['output'][0]['content'][0]['text'])
        self.assertTrue(value['finish']);self.assertEqual(fixture.counts['planner'],1)

    def test_empty_catalog_skips_extra_model_calls(self):
        catalog=deepcopy(self.catalog);catalog._documents={}
        owner=self.owner()
        view=KnowledgeRetrieval(catalog,owner).retrieve(request(),time.monotonic()+90)
        self.assertEqual(view['audit']['stop_reason'],'no_knowledge_available')
        self.assertEqual(view['audit']['provider_attempts'],0)
        self.assertEqual(self.sent,[])

    def test_hierarchy_records_retrieval_before_planning_and_no_recursive_document_history(self):
        limits=policy()['episode'];limits['scout_calls']=0
        journal=self.root/'journal'
        service=HierarchicalDecisionService(Mock(),self.owner(),self.skills,limits,allow_fixture=True,
            lesson_library=self.lessons,knowledge_policy='agent_tools_v1',journal_dir=journal)
        response=service.handle(request())
        self.assertEqual(len(list(journal.glob('knowledge-*.json'))),1)
        self.assertEqual(response['metadata']['plan_knowledge_retrieval']['selected_skill_ids'],[self.sid])

    def test_generated_retrieval_error_is_not_mislabeled_as_deadline(self):
        from tests.test_generated_skill_service import row,snapshot
        rows=[row()];cfg=policy()['roles']['scout'];transport=Mock(side_effect=TimeoutError())
        actor=ExplicitActionService(cfg,self.root/'skip.sqlite',transport,
            provider_contract='explicit_action_provider_v5',fallback_policy='path_stage_v1')
        selector=SkillDiscoveryService(cfg,actor.service.ledger_path,transport)
        with patch('orchestrator.generated_skill_service.retrieve', return_value=deepcopy(rows)), \
             patch('orchestrator.knowledge_context.retrieve', return_value=deepcopy(rows)):
            svc=GeneratedSkillService(snapshot(),actor,selector,knowledge_policy='agent_tools_v1')
            response=svc.handle(self.req)
        self.assertEqual(transport.call_count,1)
        self.assertEqual(response['fallback_reason'],'knowledge_retrieval_unavailable')
        self.assertFalse(response['metadata']['generated_actor_skipped_deadline'])
        self.assertEqual(response['metadata']['generated_actor_skip_reason'],'knowledge_retrieval_unavailable')

    def test_owner_configuration_cannot_change_after_retrieval_binding(self):
        owner=self.owner();agent=KnowledgeRetrieval(self.catalog,owner)
        owner.config['model']='unapproved_model'
        view=agent.retrieve(self.req,time.monotonic()+90)
        self.assertEqual(view['audit']['stop_reason'],'retrieval_owner_configuration_changed')
        self.assertEqual(self.sent,[])


if __name__ == '__main__':
    unittest.main()
