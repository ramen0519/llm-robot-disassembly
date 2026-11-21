"""Offline provider and tool wiring; no model API or physical simulation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.relational_knowledge import freeze_library, make_card
from orchestrator.relational_runtime import RelationalRuntime, contexts_for_request
from orchestrator.sequence_ranker import SequenceRankService, FixtureSequenceService
from orchestrator.knowledge_tools import KnowledgeCatalog
from orchestrator.knowledge_retrieval import KnowledgeRetrieval, decision_evidence
from tests.test_sequence_ranker import envelope, config, raw
from tests import test_agent_knowledge as agent_fixtures

search, read = agent_fixtures.search, agent_fixtures.read


def library(product='source-product', extra=None):
    cards = []
    context = dict(product_id=product, target_part_id='old-part',
        shape=dict(ratio_sorted=[1., 1., 1.], extents_xyz=[1.,1.,1.],
                   frame='CAD_reference', length_unit=None),
        topology=dict(contact_degree=1, degree_fraction=.5, component_delta=0,
                      remainder_isolated_nodes=0), neighbors=[],
        conditions=dict(verifier_domain='ASAP_RedMax', fixed_base='none', known_fields=['fixed_base']))
    for kind, identity, record in [
        ('experience', 'old-experience', dict(status='PASS', measurement_note='offline fixture only')),
        ('lesson', 'old-lesson', dict(statement='Conditional fixture lesson', evidence_ids=['old-experience'])),
        ('skill', 'old-skill', dict(source='def propose(ctx):\n    return 1', status='CANDIDATE'))]:
        card = make_card(dict(record=record, record_id=identity, kind=kind,
                              source_path='runs/offline_fixture/knowledge.json'),
                         context=deepcopy(context) if kind == 'experience' else {})
        if kind != 'experience':
            card['evidence_ids'] = ['relational:old-experience']
        if extra and kind == 'experience':
            card['content'].update(extra)
        card.update(fixture=False, dataset_role='training')
        cards.append(card)
    return freeze_library(cards)


class RelationalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='relational-runtime-')
        self.addCleanup(self.temp.cleanup)
        p = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network forbidden'))
        self.network = p.start(); self.addCleanup(p.stop)
        self.addCleanup(self.network.assert_not_called)
        p = patch.dict(os.environ, INDEPENDENT_FIXTURE_KEY='OFFLINE_NOT_A_KEY')
        p.start(); self.addCleanup(p.stop)

    def service(self, transport=None, source=None, **kwargs):
        return SequenceRankService(config(), Path(self.temp.name)/'ledger.sqlite',
            transport=transport or Mock(), relational_knowledge=source or library(), **kwargs)

    def test_actual_provider_body_contains_cards_and_separate_citations(self):
        sent = []
        def transport(url, headers, body, timeout):
            sent.append(deepcopy(body))
            packet = json.loads(body['input'][0]['content'])
            ids = packet['knowledge']['relational_context']['card_ids']
            return raw(body, relational_card_ids=ids[:1])
        service = self.service(transport)
        result = service.rank(envelope())
        self.assertIsNone(result['fallback_reason'], result)
        packet = json.loads(sent[0]['input'][0]['content'])
        context = packet['knowledge']['relational_context']
        self.assertTrue(context['cards'])
        by_kind = {row['knowledge']['card']['kind']: row['knowledge']['card']
                   for row in context['cards']}
        self.assertEqual(set(by_kind), {'experience', 'lesson', 'skill'})
        self.assertEqual(by_kind['lesson']['content']['statement'], 'Conditional fixture lesson')
        self.assertEqual(by_kind['skill']['content']['source'], 'def propose(ctx):\n    return 1')
        self.assertIn('relational:old-skill', context['card_ids'])
        self.assertFalse(context['execution_authorized'])
        self.assertEqual(packet['knowledge']['sequence_skills'], [])
        self.assertEqual(packet['knowledge']['evidence_ids'], [])
        self.assertEqual(result['metadata']['relational_card_ids'], context['card_ids'][:1])
        self.assertIn('relational_card_ids', sent[0]['text']['format']['schema']['required'])

    def test_candidate_skill_not_executable_and_source_immutable(self):
        source = library(); before = deepcopy(source)
        service = self.service(source=source)
        prepared = service.prepare(envelope())
        knowledge = prepared['features']['sequence_knowledge']
        value = dict(request_id=prepared['request']['request_id'], state_id=prepared['request']['state_id'],
            decision_kind='skill', candidate_ids=None, selected_skill_id='relational:old-skill',
            evidence_ids=[], relational_card_ids=[])
        with self.assertRaisesRegex(ValueError, 'unknown_sequence_skill'):
            service.decide(value, prepared['request'], {}, knowledge)
        self.assertEqual(before, source)

    def test_current_product_excludes_derived_lesson_and_skill(self):
        runtime = RelationalRuntime(library(product='fixture'))
        self.assertEqual(len(runtime.excluded_ids(['fixture'])), 3)
        packet, audit = runtime.packet(envelope()['request'], cad=envelope()['features']['cad'],
                                       excluded_products=['fixture'])
        self.assertFalse(packet['cards'])
        self.assertEqual(runtime.read('relational:old-skill', excluded_products=['fixture'])['error'],
                         'relational_evaluation_product_excluded')

    def test_read_includes_real_context_and_linked_evidence(self):
        runtime = RelationalRuntime(library())
        view = runtime.read('relational:old-skill')
        self.assertIn('def propose', view['card']['content']['source'])
        self.assertEqual(view['linked_evidence'][0]['context']['shape']['ratio_sorted'], [1.,1.,1.])
        self.assertIsNone(view['current_part_mapping'])

    def test_unknown_or_oversize_read_is_not_success(self):
        runtime = RelationalRuntime(library(extra={'detail':'x'*50000}))
        self.assertEqual(runtime.read('not-a-card')['error'], 'unknown_relational_card')
        view = runtime.read('relational:old-experience')
        self.assertNotIn('error', view)
        self.assertIn('detail', view['card']['projection']['original_content_fields_omitted'])
        cards = deepcopy(library()['cards']); cards[2]['content']['source'] = 'x'*50000
        runtime = RelationalRuntime(freeze_library(cards))
        self.assertEqual(runtime.read('relational:old-skill')['error'], 'relational_card_byte_limit')

    def test_hidden_priors_withheld_only_in_provider_projection(self):
        source = library(extra={'native_order':['old-part']})
        runtime = RelationalRuntime(source)
        view = runtime.read('relational:old-experience')
        self.assertNotIn('native_order', view['card']['content'])
        self.assertIn('native_order', view['card']['projection']['original_content_fields_omitted'])
        self.assertIn('native_order', source['cards'][0]['content'])
        self.service(source=source).prepare(envelope())

    def test_all_current_candidate_contexts_in_audit(self):
        runtime = RelationalRuntime(library())
        offered = envelope()
        found = runtime.search(offered['request'], cad=offered['features']['cad'])
        self.assertEqual({r['target_part_id'] for r in found['audit']['per_candidate_queries']}, {'1','2'})
        self.assertNotIn('excluded_ids', json.dumps(found['audit']))

    def test_model_channel_query_is_forwarded_without_fallback(self):
        runtime = RelationalRuntime(library())
        offered = envelope()
        found = runtime.search(offered['request'], cad=offered['features']['cad'], query='shape:')
        self.assertEqual(found['audit']['requested_channels'], ['shape'])
        self.assertTrue(found['results'])
        for entry in found['audit']['per_candidate_queries']:
            self.assertEqual(entry['audit']['channels'], ['shape'])

    def test_actual_physics_pass_does_not_erase_rejected_transition_or_deadline(self):
        cards = deepcopy(library()['cards'])
        cards[0]['context']['observations'] = dict(outcome='PASS', raw_sim_info='Success',
            accepted_for_transition=False, deadline_exceeded=True,
            stability_success=None, detailed_failure_cause=None)
        runtime = RelationalRuntime(freeze_library(cards))
        view = runtime.read(cards[0]['card_id'])
        self.assertEqual(view['card']['context']['observations'], cards[0]['context']['observations'])
        self.assertFalse(view['current_physics_verified'])

    def test_no_fixed_base_is_known_absence_not_missing_information(self):
        request = deepcopy(envelope()['request']); request['base_part'] = None
        context = contexts_for_request(request, envelope()['features']['cad'])[0]
        self.assertEqual(context['conditions']['fixed_base'], 'none')
        self.assertFalse(context['conditions']['fixed_base_present'])
        self.assertIn('fixed_base', context['conditions']['known_fields'])

    def test_fixture_mode_still_runs_with_zero_model_calls(self):
        cfg = config(); cfg.update(enabled=False, max_calls=0, max_usd='0')
        transport = Mock(side_effect=AssertionError('fixture must not call transport'))
        service = SequenceRankService(cfg, Path(self.temp.name)/'fixture.sqlite',
            transport=transport, relational_knowledge=library())
        result = FixtureSequenceService(service).rank(envelope())
        self.assertIsNone(result['fallback_reason'], result)
        transport.assert_not_called()

    def test_bad_relational_citation_fails_closed(self):
        service = self.service(lambda u,h,b,t: raw(b, relational_card_ids=['invented']))
        self.assertEqual(service.rank(envelope())['fallback_reason'], 'unknown_relational_card_reference')

    def test_unknown_fixture_or_nontraining_and_derived_cards_never_transmitted(self):
        for mutation in ({'fixture':None}, {'fixture':True}, {'dataset_role':'evaluation'}):
            cards = deepcopy(library()['cards'])
            cards[0].update(mutation)
            runtime = RelationalRuntime(freeze_library(cards))
            self.assertEqual(len(runtime.excluded_ids([])), 3)
            self.assertIn('error', runtime.read('relational:old-skill'))

    def test_cli_keeps_legacy_default_and_exposes_opt_in(self):
        from tools.run_sequence_native_experiment import parser
        args = parser().parse_args(['--run-dir','runs/new-only'])
        self.assertIsNone(args.relational_knowledge)


class RelationalAgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = agent_fixtures.AgentKnowledgeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        from tests.test_skill_program import request
        # The original fixture already attempted every direction. Start a fresh
        # decision so its scripted plan tests transport rather than repeat guards.
        self.fixture.req = request()
        p = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network forbidden'))
        self.network = p.start(); self.addCleanup(p.stop); self.addCleanup(self.network.assert_not_called)
        self.catalog = KnowledgeCatalog(self.fixture.skills, self.fixture.lessons,
            allow_fixture=True, relational_knowledge=library())

    def test_model_search_read_then_plan_receives_reference_skill_without_execution(self):
        sent = []
        skill_doc = self.catalog.relational_documents['relational:old-skill']
        def transport(url, headers, body, timeout):
            sent.append(deepcopy(body))
            packet = json.loads(body['input'][0]['content'])
            from tests.test_explicit_action_service import provider
            if body['text']['format']['name'] == 'knowledge_tool_requests':
                history = packet['context']['tool_history']
                operations = [search(kind='skill')] if not history else [read(skill_doc)]
                result = dict(request_id=packet['request_id'], state_id=packet['state_id'],
                              finish=bool(history), reason='offline scripted decision', operations=operations)
                response = provider(result); response['model'] = body['model']; return response
            return self.fixture.transport(url, headers, body, timeout)
        owner = self.fixture.owner(transport)
        view = KnowledgeRetrieval(self.catalog, owner).retrieve(self.fixture.req, time.monotonic()+90)
        self.assertEqual(view['audit']['retrieval_status'], 'ready')
        self.assertEqual(view['skills'], [])
        self.assertTrue(view['observations'][0]['reference_only'])
        self.assertIn('def propose', json.dumps(view['observations']))
        evidence = decision_evidence(view)
        result = owner.plan(self.fixture.req, [], {'retrieved_knowledge': evidence})
        self.assertIsNone(result['fallback_reason'], result)
        self.assertIn('def propose', json.dumps(sent[-1]))
        self.assertIn('reference_only code remains historical evidence', sent[0]['instructions'])
        self.assertNotIn('Search uses literal tokens, not embeddings', sent[0]['instructions'])
        self.assertTrue(evidence['action_citation_ids_unchanged'])
        self.assertEqual(view['audit']['request_evidence_ids'], self.fixture.req['evidence_ids'])

    def test_search_has_conditions_and_shape_not_only_card_id(self):
        session = self.catalog.session(self.fixture.req)
        found = session.search(query='', kind='memory')
        row = next(row for row in found['results'] if row.get('reference_only'))
        self.assertIn('shape', row['summary']['context'])
        self.assertIn('condition_differences', row['retrieval'])


if __name__ == '__main__':
    unittest.main()
