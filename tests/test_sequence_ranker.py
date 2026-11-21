"""No network: strict order/skill choices, replay, budget and fail-closed checks."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
sys.path.insert(0, str(Path(__file__).absolute().parent))
from test_independent_ranker import envelope as old_envelope, config
from orchestrator.sequence_ranker import SequenceRankService, FixtureSequenceService, PROTOCOL, response_object
from orchestrator.sequence_knowledge import make_fixture_sequence_skill


def envelope(identity='fixture:sequence:1'):
    value = old_envelope(identity)
    value['features']['protocol'] = PROTOCOL
    value['request']['candidate_ids'] = sorted(value['request']['candidate_ids'])
    return value


def raw(body, **changes):
    req = json.loads(body['input'][0]['content'])['request']
    value = dict(request_id=req['request_id'], state_id=req['state_id'], decision_kind='direct',
                 candidate_ids=list(reversed(req['candidate_ids'])), selected_skill_id=None, evidence_ids=[])
    value.update(changes)
    return dict(status='completed', output=[dict(type='message', role='assistant',
        content=[dict(type='output_text', text=json.dumps(value))])],
        usage=dict(input_tokens=100, output_tokens=20, total_tokens=120))


class SequenceRankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sequence_rank_fixture_')
        self.addCleanup(self.temp.cleanup)
        self.n = 0
        env = patch.dict(os.environ, INDEPENDENT_FIXTURE_KEY='fake-fixture-only')
        env.start(); self.addCleanup(env.stop)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        self.net = net.start(); self.addCleanup(net.stop); self.addCleanup(self.net.assert_not_called)

    def service(self, changes=None, cfg=None, **kwargs):
        self.n += 1
        transport = Mock(side_effect=lambda u, h, b, t: raw(b, **(changes or {})))
        service = SequenceRankService(cfg or config(), Path(self.temp.name)/('ledger%s.sqlite' % self.n),
                                      transport=transport, **kwargs)
        return service, transport

    def test_direct_reverse_only_changes_candidate_order(self):
        service, transport = self.service()
        request = envelope(); before = deepcopy(request)
        result = service.rank(request)
        self.assertIsNone(result['fallback_reason'])
        self.assertEqual(result['candidate_ids'], ['2', '1'])
        self.assertFalse(result['metadata']['physical_parameters_selected_by_llm'])
        self.assertEqual(request, before)
        transport.assert_called_once()

    def test_cached_request_does_not_call_twice(self):
        service, transport = self.service()
        one, two = service.rank(envelope()), service.rank(envelope())
        self.assertEqual(one['candidate_ids'], two['candidate_ids'])
        self.assertTrue(two['metadata']['cached']); transport.assert_called_once()

    def test_foreign_duplicate_missing_candidate_rejected(self):
        for order in (['1','x'], ['1','1'], ['1']):
            with self.subTest(order=order):
                service, _ = self.service({'candidate_ids': order})
                r = service.rank(envelope())
                self.assertEqual(r['fallback_reason'], 'invalid_permutation')
                self.assertTrue(r['metadata']['halt_required'])

    def test_extra_physical_output_rejected(self):
        for key in ('pose', 'direction', 'parts_fix', 'physics_success'):
            service, _ = self.service({key:'x'})
            self.assertEqual(service.rank(envelope())['fallback_reason'], 'invalid_sequence_response_fields')

    def test_stale_identity_halts_session_without_second_attempt(self):
        service, transport = self.service({'state_id':'wrong'})
        self.assertEqual(service.rank(envelope())['fallback_reason'], 'stale_request_or_state')
        self.assertEqual(service.rank(envelope('r2'))['fallback_reason'], 'sequence_session_halted')
        transport.assert_called_once()

    def test_unknown_citation_rejected(self):
        service, _ = self.service({'evidence_ids':['fabricated']})
        self.assertEqual(service.rank(envelope())['fallback_reason'], 'unknown_evidence_reference')

    def test_unknown_skill_rejected(self):
        service, _ = self.service({'decision_kind':'skill', 'candidate_ids':None, 'selected_skill_id':'missing'})
        self.assertEqual(service.rank(envelope())['fallback_reason'], 'unknown_sequence_skill')

    def test_model_selects_fixture_program_host_executes(self):
        skill = make_fixture_sequence_skill("def propose(ctx):\n    return {'candidate_order': ctx['candidate_ids'][::-1], 'evidence_ids': []}\n")
        service, transport = self.service({'decision_kind':'skill', 'candidate_ids':None,
            'selected_skill_id':skill['skill_id']}, skills=[skill], allow_fixture_skills=True)
        result = service.rank(envelope())
        self.assertIsNone(result['fallback_reason'], result)
        self.assertEqual(result['candidate_ids'], ['2', '1'])
        self.assertEqual(result['metadata']['skill_execution']['status'], 'PROPOSED')
        transport.assert_called_once()

    def test_fixture_skills_cannot_use_live_transport(self):
        with self.assertRaises(ValueError):
            SequenceRankService(config(), Path(self.temp.name)/'invalid.sqlite', allow_fixture_skills=True)

    def test_call_budget_and_cost_budget_no_resend(self):
        service, transport = self.service(cfg=config(max_calls=1))
        self.assertIsNone(service.rank(envelope())['fallback_reason'])
        self.assertEqual(service.rank(envelope('r2'))['fallback_reason'], 'call_budget_exhausted')
        transport.assert_called_once()
        service, transport = self.service(cfg=config(max_usd='0.00000001'))
        self.assertEqual(service.rank(envelope())['fallback_reason'], 'cost_budget_exhausted')
        transport.assert_not_called()

    def test_connection_failure_latches_without_retry(self):
        service, transport = self.service()
        transport.side_effect = TimeoutError('never copy server text')
        self.assertEqual(service.rank(envelope())['fallback_reason'], 'api_timeout')
        self.assertEqual(service.rank(envelope('r2'))['fallback_reason'], 'sequence_session_halted')
        transport.assert_called_once()

    def test_body_size_rejection_has_zero_reservation(self):
        service, transport = self.service()
        offered = envelope()
        offered['features']['episode_feedback'] = [{'raw': 'x' * 70000}]
        self.assertEqual(service.rank(offered)['fallback_reason'], 'invalid_sequence_input')
        transport.assert_not_called()
        with service._connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM rank_requests').fetchone()[0], 0)

    def test_actual_worker_authority_fields_and_episode_citation(self):
        offered = envelope()
        offered['features'].update(decision_scope='complete_part_permutation_only',
            lower_level_owner='original_ASAP_pose_direction_support_search',
            terminal_policy='unchanged_upstream_mass_or_base_rule',
            episode_feedback=[dict(event_id='attempt:observed', status='UNKNOWN')])
        service, transport = self.service({'evidence_ids':['attempt:observed']})
        result = service.rank(offered)
        self.assertIsNone(result['fallback_reason'], result)
        self.assertEqual(result['metadata']['evidence_ids'], ['attempt:observed'])
        self.assertEqual(result['metadata']['episode_evidence_ids'], ['attempt:observed'])
        self.assertEqual(result['metadata']['past_memory_evidence_ids'], [])
        self.assertEqual(result['metadata']['lesson_evidence_ids'], [])

    def test_mutated_authority_rejected_before_call(self):
        offered = envelope(); offered['features']['lower_level_owner'] = 'LLM'
        service, transport = self.service()
        self.assertEqual(service.rank(offered)['fallback_reason'], 'invalid_sequence_input')
        transport.assert_not_called()

    def test_no_heuristic_leak_and_cad_mismatch(self):
        for mutate in (lambda e: e['features']['cad'].update(native_order=['1','2']),
                       lambda e: e['features']['cad']['parts'].pop('1')):
            service, transport = self.service(); offered = envelope(); mutate(offered)
            self.assertEqual(service.rank(offered)['fallback_reason'], 'invalid_sequence_input')
            transport.assert_not_called()

    def test_disabled_fixture_calls_no_provider(self):
        service, transport = self.service(cfg=config(enabled=False, max_calls=0, max_usd=0))
        result = FixtureSequenceService(service, 'descending').rank(envelope())
        self.assertIsNone(result['fallback_reason']); self.assertTrue(result['metadata']['fixture'])
        transport.assert_not_called()

    def test_duplicate_json_keys_rejected(self):
        payload = {'status':'completed','output':[{'type':'message','role':'assistant',
                   'content':[{'type':'output_text','text':'{"state_id":"a","state_id":"b"}'}]}]}
        with self.assertRaisesRegex(ValueError, 'duplicate_json_key'):
            response_object(payload)


if __name__ == '__main__':
    unittest.main()
