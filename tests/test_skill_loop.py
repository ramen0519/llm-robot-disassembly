"""Offline contract checks. Historical artifacts are fixtures, never new physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.skill_library import (SkillMemory, FrozenSkills, execute_skill, base_card, digest, module_hash)
from orchestrator.skill_selector import SkillSelector, PROTOCOL, response_fixture
from orchestrator.skill_pipeline import SkillPipeline
from tests.test_llm_ranker import config as live_config
from tests.test_evidence_memory import make_record

ROOT = Path('C:/dev/disassembly_codex_ready_v2')
HISTORY = ROOT / 'runs/luna_live_smoke_20261017'
OFF = dict(enabled=False, model='gpt-6-luna', max_calls=0, max_usd=0)


class SkillLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='skill_test_', dir=ROOT / 'runs/skill_loop_20261017')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = SkillMemory(self.root / 'memory.sqlite')
        self.addCleanup(self.store.close)
        self.context = json.loads((HISTORY / 'context.json').read_text())
        self.events = [json.loads(line) for line in (HISTORY / 'native_off/events.jsonl').read_text().splitlines()]
        self.decisions = [e for e in self.events if e['kind'] == 'rank_decision']
        self.request = deepcopy(self.decisions[0]['request'])
        self.envelope = dict(schema_version=1, kind='rank_request', request=self.request, features={}, timeout_seconds=45)
        self.serial = 0
        guard = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        self.network = guard.start()
        self.addCleanup(guard.stop)
        self.addCleanup(self.network.assert_not_called)
        env = patch.dict(os.environ, {'RANKER_TEST_KEY': 'fixture-only-not-a-real-key'})
        env.start()
        self.addCleanup(env.stop)

    def seed_fixture(self):
        # Historical decisions stand in for invocation entries in unit tests only.
        for decision in self.decisions:
            request = decision['request']
            self.store.put('invocation', request['request_id'], dict(request=request, skill_id='native_order_v1',
                applied_order=decision['applied_order']))
        collected = self.store.collect_completed(HISTORY / 'native_off', self.context)
        for sid in collected['proposals']:
            self.store.review_and_register(sid)
        return collected

    def freeze(self):
        self.serial += 1
        path = self.root / ('frozen' + str(self.serial))
        self.store.freeze(path)
        return path

    def enriched(self):
        value = deepcopy(self.envelope)
        value['request'].update(decision_protocol=PROTOCOL, offered_skill_ids=['native_order_v1', 'observed_test'],
                                library_sha256='a' * 64)
        return value

    def test_empty_memory_retrieves_executable_native(self):
        frozen = FrozenSkills(self.freeze())
        cards, evidence = frozen.retrieve(self.context, self.request)
        self.assertEqual([c['skill_id'] for c in cards], ['native_order_v1'])
        self.assertEqual(evidence['evidence_ids'], [])
        self.assertEqual(execute_skill(cards[0], self.context, self.request, evidence), self.request['candidate_ids'])

    def test_development_collect_register_reopen_and_reuse(self):
        report = self.seed_fixture()
        self.assertEqual(len(report['proposals']), 3)
        self.assertEqual(report['associated_outcomes'], 3)
        self.assertEqual(len(report['unassociated_evidence_ids']), 1)
        self.assertEqual(len(self.store.evidence.records()), 4)
        repeated = self.store.collect_completed(HISTORY / 'native_off', self.context)
        self.assertEqual(repeated['inserted'], dict(inserted=0, duplicates=4))
        reopened = SkillMemory(self.store.path)
        try:
            self.assertEqual(len(reopened.entries('registered_skill')), 3)
        finally:
            reopened.close()
        cards, evidence = FrozenSkills(self.freeze()).retrieve(self.context, self.request)
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[1]['lifecycle'], 'provisional')
        self.assertEqual(execute_skill(cards[1], self.context, self.request, evidence), self.request['candidate_ids'])

    def test_proposal_is_unavailable_before_review(self):
        self.seed_fixture()
        card = self.store.entries('registered_skill')[0]
        card['skill_id'] = 'not_reviewed'
        card['lifecycle'] = 'proposed'
        self.store.put('proposal', card['skill_id'], card)
        cards, _ = FrozenSkills(self.freeze()).retrieve(self.context, self.request)
        self.assertNotIn('not_reviewed', [c['skill_id'] for c in cards])

    def test_experience_skill_changes_reversed_proposal_without_deleting_candidates(self):
        self.seed_fixture()
        request = deepcopy(self.request)
        request['candidate_ids'].reverse()
        before = deepcopy(request)
        cards, evidence = FrozenSkills(self.freeze()).retrieve(self.context, request)
        card = next(c for c in cards if c['skill_id'] != 'native_order_v1')
        order = execute_skill(card, self.context, request, evidence)
        self.assertNotEqual(order, request['candidate_ids'])
        self.assertEqual(order[0], self.request['candidate_ids'][0])
        self.assertEqual(sorted(order), sorted(request['candidate_ids']))
        self.assertEqual(request, before)

    def test_context_and_remaining_state_change_prevent_reuse(self):
        self.seed_fixture()
        frozen = FrozenSkills(self.freeze())
        for change in ('seed', 'mesh', 'support', 'remaining'):
            context, request = deepcopy(self.context), deepcopy(self.request)
            if change == 'seed': context['seed'] += 1
            if change == 'mesh': context['input_snapshot']['commit'] = 'changed'
            if change == 'support': request['base_part'] = '0'; request['candidate_ids'].remove('0')
            if change == 'remaining': request['assembled_parts'].remove('3'); request['candidate_ids'].remove('3')
            with self.subTest(change=change):
                cards, _ = frozen.retrieve(context, request)
                self.assertEqual([c['skill_id'] for c in cards], ['native_order_v1'])

    def test_unknown_error_recorded_and_never_promoted_or_pruned(self):
        records = [make_record(context=self.context, tag='u', status='UNKNOWN', part='0'),
                   make_record(context=self.context, tag='e', status='ERROR', part='3')]
        self.store.evidence.insert_many(records)
        cards, evidence = FrozenSkills(self.freeze()).retrieve(self.context, self.request)
        self.assertEqual(evidence['positive_candidate_ids'], [])
        self.assertEqual(len(evidence['evidence_ids']), 2)
        self.assertEqual(execute_skill(cards[0], self.context, self.request, evidence), self.request['candidate_ids'])
        self.assertEqual(self.store.entries('registered_skill'), [])

    def test_invalid_output_cannot_remove_candidates(self):
        card = base_card()
        with patch.dict('orchestrator.skill_library.CALLABLES', native_order=lambda r, p: []):
            with self.assertRaises(ValueError): execute_skill(card, self.context, self.request, {})

    def test_library_code_hash_and_snapshot_tampering_rejected(self):
        path = self.freeze()
        value = json.loads((path / 'library.json').read_text())
        value['source_code_sha256'] = '0' * 64
        (path / 'library.json').write_text(json.dumps(value))
        with self.assertRaises(ValueError): FrozenSkills(path)

    def test_store_append_only_and_id_conflict(self):
        self.store.put('test', 'id', {'n': 1})
        self.assertFalse(self.store.put('test', 'id', {'n': 1}))
        with self.assertRaises(ValueError): self.store.put('test', 'id', {'n': 2})
        with self.assertRaises(sqlite3.IntegrityError): self.store.db.execute('DELETE FROM skill_entries')
        self.store.db.rollback()

    def test_selector_rejects_unknown_stale_extra_fields(self):
        envelope = self.enriched()
        request = envelope['request']
        for change, expected in [({'skill_id': 'exec_arbitrary'}, 'unoffered_skill'),
                                 ({'state_id': 'stale'}, 'stale_request_or_state'),
                                 ({'library_sha256': 'b' * 64}, 'stale_skill_library'),
                                 ({'code': 'malicious'}, 'invalid_skill_response')]:
            raw = response_fixture(request, 'observed_test')
            content = raw['output'][0]['content'][0]
            value = json.loads(content['text']); value.update(change); content['text'] = json.dumps(value)
            self.assertEqual(SkillSelector._parse(raw, request), (None, expected))

    def test_selector_budget_cache_and_same_id_conflict(self):
        calls = []
        def transport(url, headers, body, timeout):
            calls.append(body)
            request = json.loads(body['input'][0]['content'])['request']
            return response_fixture(request, 'observed_test')
        service = SkillSelector(live_config(max_calls=1), self.root / 'ledger.sqlite', transport)
        envelope = self.enriched()
        response = service.rank(envelope)
        self.assertEqual(response['metadata']['selected_skill_id'], 'observed_test')
        self.assertEqual(len(calls), 1)
        self.assertTrue(service.rank(envelope)['metadata']['cached'])
        changed = deepcopy(envelope); changed['request']['request_id'] += '2'
        self.assertEqual(service.rank(changed)['fallback_reason'], 'call_budget_exhausted')
        changed = deepcopy(envelope); changed['features']['changed'] = True
        self.assertEqual(service.rank(changed)['fallback_reason'], 'request_id_conflict')
        self.assertEqual(len(calls), 1)

    def test_selector_timeout_falls_back_without_retry(self):
        def transport(*args): raise TimeoutError()
        service = SkillSelector(live_config(max_calls=1), self.root / 'ledger.sqlite', transport)
        self.assertEqual(service.rank(self.enriched())['fallback_reason'], 'api_timeout')
        self.assertTrue(service.rank(self.enriched())['metadata']['cached'])

    def test_pipeline_executes_retrieved_card_and_logs_reuse(self):
        self.seed_fixture()
        frozen = self.freeze()
        envelope = deepcopy(self.envelope); envelope['request']['request_id'] = 'new:r1'; envelope['request']['run_id'] = 'new'
        pipeline = SkillPipeline(OFF, self.root / 'ledger.sqlite', frozen, self.context, self.store, 'scripted_test')
        response = pipeline.rank(envelope)
        self.assertTrue(response['metadata']['skill_id'].startswith('observed_positive_'))
        self.assertEqual(response['metadata']['selected_by'], 'scripted_test')
        self.assertFalse(response['metadata']['attempted'])
        self.assertEqual(len(response['metadata']['skill_evidence_ids']), 1)
        count = len(self.store.entries('invocation'))
        self.assertTrue(pipeline.rank(envelope)['metadata']['invocation_reused'])
        self.assertEqual(len(self.store.entries('invocation')), count)
        envelope['features']['changed'] = True
        self.assertEqual(pipeline.rank(envelope)['fallback_reason'], 'request_id_conflict')

    def test_disabled_llm_uses_native_and_logs_reason(self):
        pipeline = SkillPipeline(OFF, self.root / 'ledger.sqlite', self.freeze(), self.context, self.store)
        response = pipeline.rank(self.envelope)
        self.assertEqual(response['candidate_ids'], self.request['candidate_ids'])
        self.assertEqual(response['metadata']['llm_fallback_reason'], 'llm_disabled')
        self.assertEqual(response['metadata']['selected_by'], 'deterministic_fallback')

    def test_full_live_branch_with_injected_transport_and_global_budget(self):
        self.seed_fixture()
        frozen = self.freeze()
        calls = []
        def transport(url, headers, body, timeout):
            data = json.loads(body['input'][0]['content'])
            request = data['request']
            calls.append(body)
            self.assertIn('registered_skills', data['features'])
            self.assertTrue(data['features']['evidence_memory']['evidence_ids'])
            chosen = next(s for s in request['offered_skill_ids'] if s != 'native_order_v1')
            return response_fixture(request, chosen)
        pipeline = SkillPipeline(live_config(max_calls=1), self.root / 'ledger.sqlite', frozen,
                                 self.context, self.store, 'live', transport)
        envelope = deepcopy(self.envelope)
        envelope['request'].update(request_id='live_fixture:r1', run_id='live_fixture')
        response = pipeline.rank(envelope)
        self.assertEqual(response['metadata']['selected_by'], 'llm')
        self.assertFalse(response['metadata']['llm_live'])  # injected transport, not provider
        self.assertTrue(response['metadata']['skill_id'].startswith('observed_positive_'))
        self.assertEqual(set(response['candidate_ids']), set(envelope['request']['candidate_ids']))
        self.assertTrue(pipeline.rank(envelope)['metadata']['invocation_reused'])
        envelope['request']['request_id'] = 'live_fixture:r2'
        response = pipeline.rank(envelope)
        self.assertEqual(response['metadata']['llm_fallback_reason'], 'call_budget_exhausted')
        self.assertEqual(response['metadata']['skill_id'], 'native_order_v1')
        self.assertEqual(len(calls), 1)

    def test_full_pipeline_invalid_selection_falls_back_without_pruning(self):
        def transport(url, headers, body, timeout):
            request = json.loads(body['input'][0]['content'])['request']
            return response_fixture(request, 'unknown_unsafe_skill')
        pipeline = SkillPipeline(live_config(max_calls=1), self.root / 'ledger.sqlite', self.freeze(),
                                 self.context, self.store, 'live', transport)
        response = pipeline.rank(self.envelope)
        self.assertEqual(response['metadata']['llm_fallback_reason'], 'unoffered_skill')
        self.assertEqual(response['metadata']['selected_by'], 'deterministic_fallback')
        self.assertEqual(response['candidate_ids'], self.request['candidate_ids'])

    def test_proposal_wrong_source_cannot_register(self):
        self.seed_fixture()
        card = self.store.entries('proposal')[0]; card['skill_id'] = 'bad'; card['source_evidence_ids'] = ['missing']
        self.store.put('proposal', 'bad', card)
        with self.assertRaises(ValueError): self.store.review_and_register('bad')


if __name__ == '__main__': unittest.main()
