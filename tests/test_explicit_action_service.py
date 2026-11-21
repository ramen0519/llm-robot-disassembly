"""Injected provider fixtures only; external HTTP is forbidden."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_explicit_action_contract import request
from orchestrator.explicit_action_service import ExplicitActionService, response_schema
from worker.explicit_action_contract import canonical_response, validate_response


def config(**changes):
    return {'enabled': True, 'model': 'fixture-model', 'api_key_env': 'EXPLICIT_FIXTURE_KEY',
        'max_calls': 3, 'max_usd': '1', 'input_usd_per_million': '1',
        'output_usd_per_million': '1', **changes}


def value(r):
    r = deepcopy(r); response = canonical_response(r, None)
    response['choice'].pop('part_id')
    return {k: response[k] for k in ('request_id', 'state_id', 'candidate_order', 'choice')}


def provider(output):
    return {'id': 'fixture-response', 'model': 'fixture-model', 'status': 'completed', 'error': None,
        'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
            'content': [{'type': 'output_text', 'text': json.dumps(output)}]}],
        'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}}


class ExplicitActionServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='explicit_service_fixture_')
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name); self.serial = 0
        env = patch.dict(os.environ, {'EXPLICIT_FIXTURE_KEY': 'fixture-not-real-key'})
        env.start(); self.addCleanup(env.stop)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        network = net.start(); self.addCleanup(net.stop); self.addCleanup(network.assert_not_called)

    def service(self, transport=None, cfg=None, mode='llm', ledger=None):
        self.serial += 1
        return ExplicitActionService(cfg or config(), ledger or self.root/('ledger%d.sqlite'%self.serial),
                                     transport, mode=mode)

    def transport(self, change=None):
        def call(url, headers, body, timeout):
            r = json.loads(body['input'][0]['content'])['request']; v = value(r)
            if change: change(v)
            return provider(v)
        return Mock(side_effect=call)

    def test_actual_decision_fields_with_empty_memory(self):
        transport = self.transport(); service = self.service(transport)
        result = service.handle(request())
        self.assertIsNone(result['fallback_reason']); validate_response(request(), result)
        self.assertEqual(result['metadata']['selection_source'], 'injected_provider_fixture')
        self.assertFalse(result['metadata']['llm_live']); self.assertTrue(result['metadata']['attempted'])
        self.assertEqual(transport.call_count, 1)

    def test_provider_may_choose_other_part_pose_support_and_two_leg(self):
        def change(v):
            v['candidate_order'] = ['2', '1']
            v['choice'].update(pose_id='p1', skill_id='turn_force_two_leg_v1',
                direction_id='yn', second_direction_id='zp', support_ids=['1'])
        r = request(); pose = deepcopy(r['pose_catalog'][0]); pose['pose_id'] = 'p1'; pose['matrix'][0][3] = 3
        r['pose_catalog'].append(pose)
        result = self.service(self.transport(change)).handle(r)
        self.assertIsNone(result['fallback_reason']); self.assertEqual(result['choice']['part_id'], '2')
        self.assertEqual(result['choice']['pose_id'], 'p1'); self.assertEqual(result['choice']['support_ids'], ['1'])

    def test_terminal_two_still_calls_provider(self):
        r = request(); r.update(remaining_parts=['1', '2'], candidate_ids=['1', '2'], base_part=None)
        transport = self.transport(); result = self.service(transport).handle(r)
        self.assertEqual(transport.call_count, 1); self.assertIsNone(result['fallback_reason'])

    def test_strict_schema_contains_actual_id_enums(self):
        schema = response_schema(request())
        self.assertEqual(schema['properties']['request_id']['enum'], ['fixture:r1'])
        choice = schema['properties']['choice']['properties']
        self.assertNotIn('part_id', choice)
        self.assertNotIn('part_id', schema['properties']['choice']['required'])
        self.assertEqual(choice['pose_id']['enum'], ['p0'])
        self.assertEqual(choice['evidence_ids']['maxItems'], 0)

    def test_invalid_choice_falls_back_without_retry(self):
        for field, bad in [('pose_id', 'invented'), ('support_ids', ['base']), ('evidence_ids', ['fake'])]:
            transport = self.transport(lambda v: v['choice'].update({field: bad}))
            result = self.service(transport).handle(request())
            self.assertEqual(result['fallback_reason'], 'invalid_action_response')
            self.assertEqual(result['metadata']['selection_source'], 'deterministic_fallback')
            self.assertEqual(transport.call_count, 1); validate_response(request(), result)

    def test_stale_id_and_omitted_candidates_fallback(self):
        for change in (lambda v:v.update(request_id='old'), lambda v:v.update(candidate_order=['1'])):
            result = self.service(self.transport(change)).handle(request())
            self.assertEqual(result['fallback_reason'], 'invalid_action_response')

    def test_duplicate_json_key_rejected(self):
        raw = provider(value(request()))
        raw['output'][0]['content'][0]['text'] = '{"request_id":"a","request_id":"b"}'
        result = self.service(Mock(return_value=raw)).handle(request())
        self.assertEqual(result['metadata']['decision_validation_error'], 'duplicate_json_key')

    def test_timeout_has_no_retry(self):
        transport = Mock(side_effect=TimeoutError('do not log provider secrets'))
        result = self.service(transport).handle(request())
        self.assertEqual(result['fallback_reason'], 'api_timeout'); self.assertEqual(transport.call_count, 1)
        self.assertNotIn('do not log', json.dumps(result))

    def test_disabled_reads_no_key_and_makes_no_api_call(self):
        transport = self.transport(); service = self.service(transport, config(enabled=False))
        with patch('orchestrator.llm_ranker.os.environ.get', side_effect=AssertionError('key_forbidden')):
            result = service.handle(request())
        transport.assert_not_called(); self.assertEqual(result['fallback_reason'], 'llm_disabled')

    def test_deterministic_mode_reads_no_key_or_ledger(self):
        service = self.service(mode='deterministic')
        with patch('orchestrator.llm_ranker.os.environ.get', side_effect=AssertionError('key_forbidden')):
            result = service.handle(request())
        self.assertEqual(result['metadata']['selection_source'], 'deterministic_control')
        self.assertFalse(list(self.root.glob('*.sqlite')))

    def test_same_request_reuses_paid_response_without_recall(self):
        transport = self.transport(); service = self.service(transport)
        first = service.handle(request()); second = service.handle(request())
        self.assertEqual(transport.call_count, 1); self.assertTrue(second['metadata']['cached'])
        self.assertEqual(first['choice'], second['choice'])

    def test_shared_call_budget_across_instances(self):
        transport = self.transport(); ledger = self.root/'shared.sqlite'; cfg = config(max_calls=1)
        self.service(transport, cfg, ledger=ledger).handle(request())
        r = request(); r['request_id'] = 'fixture:r2'
        second = self.service(transport, cfg, ledger=ledger).handle(r)
        self.assertEqual(second['fallback_reason'], 'call_budget_exhausted')
        self.assertEqual(transport.call_count, 1)

    def test_cost_cap_prevents_call(self):
        transport = self.transport(); result = self.service(transport, config(max_usd='0.000001')).handle(request())
        self.assertEqual(result['fallback_reason'], 'cost_budget_exhausted'); transport.assert_not_called()

    def test_feedback_and_memory_citation_exposed_without_physics_claim(self):
        r = request(); r['episode_feedback'] = [{'evidence_id': 'measured:1', 'status': 'UNKNOWN'}]
        r['evidence_ids'] = ['measured:1']
        result = self.service(self.transport(lambda v:v['choice'].update(evidence_ids=['measured:1']))).handle(r)
        self.assertEqual(result['metadata']['cited_evidence_ids'], ['measured:1'])
        self.assertFalse(result['metadata']['physics_bypass'])

    def test_provider_repeat_action_fallback_to_another_action(self):
        r = request(); old = value(r); original = canonical_response(r, None)
        r['attempted_action_ids'] = [original['action_id']]
        result = self.service(Mock(return_value=provider(old))).handle(r)
        self.assertEqual(result['fallback_reason'], 'invalid_action_response')
        self.assertNotEqual(result['action_id'], original['action_id'])
        self.assertEqual(result['choice']['part_id'], original['choice']['part_id'])

    def test_invalid_request_rejected_before_api(self):
        transport = self.transport(); r = request(); r['candidate_ids'] = ['1']
        with self.assertRaises(ValueError): self.service(transport).handle(r)
        transport.assert_not_called()

    def test_model_receives_registered_executable_skill_descriptions_and_world_axes(self):
        transport = self.transport(); self.service(transport).handle(request())
        body = transport.call_args.args[2]
        model_input = json.loads(body['input'][0]['content'])
        skills = {item['skill_id']: item for item in model_input['skill_library']}
        self.assertEqual(set(skills), {'single_force_v1', 'turn_force_two_leg_v1'})
        turn = skills['turn_force_two_leg_v1']
        self.assertEqual(turn['switch_after_completed_forward_calls'], 50)
        self.assertFalse(turn['intermediate_reset'])
        self.assertTrue(turn['second_axis_must_be_perpendicular'])
        self.assertTrue(all(item['current_attempt_requires_new_verification'] for item in skills.values()))
        self.assertEqual(model_input['force_direction_vectors']['zp'], [0, 0, 1])
        self.assertIn('not verified', model_input['physics_scope'])
        self.assertNotIn('native_order', model_input)


if __name__ == '__main__': unittest.main()
