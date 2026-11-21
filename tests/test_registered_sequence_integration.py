"""Injected provider selection, real preserved Python VM; no API or physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).absolute().parent))
from test_independent_ranker import config
from test_sequence_ranker import envelope as base_envelope, raw
from orchestrator.knowledge_import import import_sources
from orchestrator.relational_knowledge import freeze_library
from orchestrator.relational_runtime import RelationalRuntime, size
from orchestrator.relational_retrieval import search_cards
from orchestrator.scored_skill_provenance import PACKET_PATH, REGISTRATIONS
from orchestrator.scored_sequence_bridge import build_bridge, execute_bridge
from orchestrator.scored_sequence_runtime import load_bridges, FROZEN_PROTOCOL, reserve_registered_views
from orchestrator.sequence_ranker import SequenceRankService, MAX_BODY_BYTES
from tools.migrate_relational_knowledge import Migrator
from worker.task_memory import digest

ROOT = Path(__file__).absolute().parents[1]


def envelope():
    value = base_envelope()
    # Explicit synthetic observations; the older fixture deliberately had no contacts.
    value['features']['cad']['relations'].update(
        contact_edges=[['1', 'base'], ['2', 'base']], contact_source='fixture_only')
    return value


class RegisteredSequenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = [PACKET_PATH] + [r['run']+'/registry/skill.json' for r in REGISTRATIONS]
        imported = import_sources(ROOT, [dict(path=p, status='DISCOVERED') for p in paths])
        cls.cards = Migrator(ROOT).convert(imported)
        cls.library = freeze_library(cls.cards, purpose='training_runtime')
        cls.bridges = [build_bridge(ROOT/p) for p in paths[1:]]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='registered_sequence_offline_')
        self.addCleanup(self.temp.cleanup)
        self.serial = 0
        env = patch.dict(os.environ, INDEPENDENT_FIXTURE_KEY='fake-fixture-only')
        env.start(); self.addCleanup(env.stop)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        self.network = net.start(); self.addCleanup(net.stop)
        self.addCleanup(self.network.assert_not_called)

    def service(self, selected=None, **kwargs):
        self.serial += 1
        def respond(url, headers, body, timeout):
            payload = json.loads(body['input'][0]['content'])
            knowledge = payload['knowledge']
            changes = dict(relational_card_ids=[])
            if selected is not None:
                changes.update(decision_kind='skill', candidate_ids=None, selected_skill_id=selected)
                cards = [s.get('relational_card_id') for s in knowledge['sequence_skills']
                         if s['skill_id'] == selected]
                changes['relational_card_ids'] = [c for c in cards if c]
            return raw(body, **changes)
        transport = Mock(side_effect=respond)
        service = SequenceRankService(config(), Path(self.temp.name)/('ledger%s.sqlite'%self.serial),
            relational_knowledge=kwargs.pop('relational_knowledge', self.library),
            scored_skills=kwargs.pop('scored_skills', self.bridges), transport=transport, **kwargs)
        return service, transport

    def test_both_registered_sources_retrieved_offered_and_executed(self):
        for bridge in self.bridges:
            with self.subTest(skill=bridge['skill_id']):
                service, transport = self.service(bridge['skill_id'])
                offered = envelope(); before = deepcopy(offered)
                result = service.rank(offered)
                self.assertIsNone(result['fallback_reason'], result['metadata'].get('skill_execution'))
                self.assertEqual(offered, before)
                body = transport.call_args.args[2]
                payload = json.loads(body['input'][0]['content'])
                knowledge = payload['knowledge']
                cards = knowledge['sequence_skills']
                self.assertEqual({s['skill_id'] for s in cards}, {b['skill_id'] for b in self.bridges})
                chosen = next(s for s in cards if s['skill_id'] == bridge['skill_id'])
                self.assertEqual(chosen['program'], bridge['program'])
                self.assertIn(chosen['relational_card_id'], knowledge['relational_context']['card_ids'])
                self.assertIn(bridge['skill_id'], body['text']['format']['schema']['properties']['selected_skill_id']['enum'])
                self.assertLessEqual(size(body), MAX_BODY_BYTES)
                execution = result['metadata']['skill_execution']
                self.assertTrue(execution['vm_executed'])
                self.assertFalse(execution['physics_executed'])
                self.assertEqual(execution['native_physics_validation'], 'NOT_RUN')
                self.assertTrue(execution['base_score_evaluated'])
                self.assertEqual(execution['feature_parent_count'], 3)
                self.assertNotIn('base', result['candidate_ids'])
                expected = execute_bridge(bridge, offered['features']['cad'], offered['request'])
                self.assertEqual(result['candidate_ids'], expected['candidate_order'])
                transport.assert_called_once()

    def test_direct_choice_does_not_execute_available_skill(self):
        service, transport = self.service()
        with patch('orchestrator.scored_sequence_bridge.execute_bridge', side_effect=AssertionError('not selected')) as vm:
            result = service.rank(envelope())
        self.assertIsNone(result['fallback_reason'])
        self.assertIsNone(result['metadata']['selected_skill_id'])
        self.assertIsNone(result['metadata']['skill_execution'])
        vm.assert_not_called(); transport.assert_called_once()

    def test_unobserved_contacts_exclude_scorers_without_blocking_direct(self):
        service, transport = self.service()
        offered = envelope(); offered['features']['cad']['relations']['contact_edges'] = None
        with patch('orchestrator.scored_sequence_bridge.execute_bridge') as vm:
            result = service.rank(offered)
        self.assertIsNone(result['fallback_reason'])
        payload = json.loads(transport.call_args.args[2]['input'][0]['content'])
        self.assertEqual(payload['knowledge']['sequence_skills'], [])
        excluded = result['metadata']['knowledge_audit']['scored_skill_bridge']['excluded']
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(r['diagnostic'] == 'contact_graph_not_observed' for r in excluded))
        vm.assert_not_called()

    def test_observed_empty_contacts_are_distinct_from_unknown(self):
        service, _ = self.service()
        offered = envelope(); offered['features']['cad']['relations']['contact_edges'] = []
        prepared = service.prepare(offered)
        self.assertEqual(len(prepared['features']['sequence_knowledge']['sequence_skills']), 2)

    def test_reservation_removes_stale_excluded_card_audit(self):
        packet = dict(cards=[], card_ids=[], excluded=[dict(card_id='registered', reason='context_byte_limit')],
                      excluded_count=1)
        row = dict(match={'card_id':'registered'}, knowledge={'card':{'kind':'skill'}})
        value = reserve_registered_views(packet, [row], 2048)
        self.assertEqual(value['card_ids'], ['registered'])
        self.assertEqual(value['excluded'], [])
        self.assertEqual(value['excluded_count'], 0)
        self.assertEqual(value['restored_after_registered_reservation'], ['registered'])

    def test_unknown_skill_rejected_without_vm(self):
        service, _ = self.service('unregistered:invented')
        with patch('orchestrator.scored_sequence_bridge.execute_bridge') as vm:
            result = service.rank(envelope())
        self.assertEqual(result['fallback_reason'], 'unknown_sequence_skill')
        vm.assert_not_called()

    def test_archive_without_explicit_bridge_never_authorizes_execution(self):
        service, _ = self.service(self.bridges[0]['skill_id'], scored_skills=[])
        result = service.rank(envelope())
        self.assertEqual(result['fallback_reason'], 'unknown_sequence_skill')

    def test_old_runtime_missing_registration_cannot_offer_bridge(self):
        lib = freeze_library([c for c in self.cards if c['kind'] != 'skill'], purpose='training_runtime')
        service, transport = self.service(relational_knowledge=lib)
        prepared = service.prepare(envelope())
        self.assertEqual(prepared['features']['sequence_knowledge']['sequence_skills'], [])
        reasons = prepared['features']['knowledge_audit']['scored_skill_bridge']['excluded']
        self.assertEqual(len(reasons), 2)
        transport.assert_not_called()

    def test_same_origin_without_verified_provenance_cannot_offer_bridge(self):
        cards = deepcopy(self.cards)
        for card in cards:
            if card['kind'] == 'skill':
                card.pop('provenance_validation', None)
        lib = freeze_library(cards, purpose='training_runtime')
        service, _ = self.service(relational_knowledge=lib)
        prepared = service.prepare(envelope())
        self.assertEqual(prepared['features']['sequence_knowledge']['sequence_skills'], [])

    def test_all_generation_exposure_excluded_even_if_not_directly_cited(self):
        for product in self.bridges[0]['source_task_ids']:
            with self.subTest(product=product):
                service, _ = self.service(knowledge_limits={'excluded_task_ids':[product]})
                prepared = service.prepare(envelope())
                self.assertEqual(prepared['features']['sequence_knowledge']['sequence_skills'], [])
                # References must not leak a source product either, even when
                # the generated program did not cite that product explicitly.
                ids = {c['card_id'] for c in self.cards if c['kind'] == 'skill'}
                self.assertTrue(ids <= service.relational.excluded_ids([product]))
                for identity in ids:
                    self.assertIn('error', service.relational.read(identity, excluded_products=[product]))
                query = dict(context=next(c['context'] for c in self.cards if c['kind']=='experience'), kind='skill')
                result = search_cards(self.cards, query, exclude_products=[product])
                self.assertFalse(ids.intersection(r['card_id'] for r in result['results']))

    def test_return_only_search_filter_retains_evidence_anchors(self):
        runtime = RelationalRuntime(self.library)
        offered = envelope()
        ids = [c['card_id'] for c in self.cards if c['kind'] == 'skill']
        found = runtime.search(offered['request'], cad=offered['features']['cad'], kind='skill', return_card_ids=ids)
        self.assertEqual({r['card_id'] for r in found['results']}, set(ids))
        self.assertTrue(all(r.get('matched_evidence_card_id') for r in found['results']))
        self.assertEqual(runtime.search(offered['request'], cad=offered['features']['cad'],
            kind='skill', return_card_ids=[])['results'], [])

    def test_frozen_library_tamper_and_duplicate_rejected(self):
        frozen = dict(protocol=FROZEN_PROTOCOL, entries=self.bridges)
        frozen['content_sha256'] = digest(frozen)
        self.assertEqual(len(load_bridges(frozen)), 2)
        bad = deepcopy(frozen); bad['entries'].pop()
        with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
            load_bridges(bad)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_bridges([self.bridges[0]]*2)

    def test_prepared_state_or_code_mutation_cannot_execute(self):
        service, _ = self.service()
        prepared = service.prepare(envelope())
        req = prepared['request']; knowledge = prepared['features']['sequence_knowledge']
        value = dict(request_id=req['request_id'], state_id=req['state_id'], decision_kind='skill',
                     candidate_ids=None, selected_skill_id=self.bridges[0]['skill_id'],
                     evidence_ids=[], relational_card_ids=[])
        altered = deepcopy(knowledge)
        altered['sequence_skills'][0]['program']['source'] += '\n# mutation'
        with self.assertRaisesRegex(ValueError, 'state_binding_mismatch'):
            service.decide(value, req, {}, altered)
        req = deepcopy(req); req['request_id'] += ':stale'
        value['request_id'] = req['request_id']
        with self.assertRaisesRegex(ValueError, 'state_binding_mismatch'):
            service.decide(value, req, {}, knowledge)

    def test_vm_decline_is_not_replaced_with_arbitrary_order(self):
        service, _ = self.service(self.bridges[0]['skill_id'])
        declined = dict(status='DECLINED', vm_executed=True, diagnostics=['observed_abstain'])
        with patch('orchestrator.scored_sequence_bridge.execute_bridge', return_value=declined):
            result = service.rank(envelope())
        self.assertEqual(result['fallback_reason'], 'sequence_skill_declined')
        self.assertTrue(result['metadata']['halt_required'])
        self.assertFalse(result['metadata']['fallback_order_executed'])
        self.assertEqual(result['metadata']['selected_skill_id'], self.bridges[0]['skill_id'])
        self.assertEqual(result['metadata']['skill_execution'], declined)

    def test_explicit_bridge_requires_new_relational_library(self):
        with self.assertRaisesRegex(ValueError, 'require_relational'):
            self.service(relational_knowledge=None)

    def test_cached_request_has_one_selection_and_one_execution(self):
        service, transport = self.service(self.bridges[0]['skill_id'])
        with patch('orchestrator.scored_sequence_bridge.execute_bridge', wraps=execute_bridge) as vm:
            first = service.rank(envelope()); second = service.rank(envelope())
        self.assertIsNone(first['fallback_reason'])
        self.assertEqual(first['candidate_ids'], second['candidate_ids'])
        self.assertTrue(second['metadata']['cached'])
        transport.assert_called_once(); vm.assert_called_once()


if __name__ == '__main__':
    unittest.main()
