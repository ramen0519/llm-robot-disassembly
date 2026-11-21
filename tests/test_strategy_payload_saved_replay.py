"""Saved actual 15-part request; injected response/SQLite, never paid inference."""
from contextlib import ExitStack, closing
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_planner import StrategyPlanService
from orchestrator.strategy_payload_projection import VERSION, restore_context
from orchestrator.strategy_reconsideration import stability_reconsideration_context
from orchestrator.explicit_action_provider_v5 import restore_request_v5
from orchestrator.strategy_failover import POLICY
from tests.test_hierarchical_planner import policy
from tests.test_explicit_action_service import provider
from worker.explicit_action_contract import canonical, digest


ROOT = Path(__file__).absolute().parent.parent
FIXTURES = ROOT/'runs/fifteen_request_fix_20261110'


class StrategySavedPayloadTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for target in ['urllib.request.OpenerDirector.open', 'socket.socket.connect']:
            blocked = self.stack.enter_context(patch(target, side_effect=AssertionError('network_forbidden')))
            self.addCleanup(blocked.assert_not_called)
        self.temp = tempfile.TemporaryDirectory(prefix='saved_strategy_payload_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack.enter_context(patch.dict(os.environ, STRATEGY_PAYLOAD_FIXTURE_KEY='offline-not-a-real-key'))
        self.envelope = json.loads((FIXTURES/'reconstructed_last_envelope.json').read_text(encoding='utf-8'))

    def service(self, transport=None):
        config = dict(policy()['roles']['planner'], api_key_env='STRATEGY_PAYLOAD_FIXTURE_KEY',
                      max_calls=1, max_usd='2.50')
        return StrategyPlanService(config,self.root/'planner.sqlite',transport=transport,
                                   strategy_failure_policy=POLICY)

    def test_saved_large_body_preserves_original_observations_and_new_pose_facts(self):
        envelope = deepcopy(self.envelope); service = self.service()
        body = service._body(envelope)
        self.assertEqual(envelope, self.envelope)
        self.assertLessEqual(len(canonical(body).encode('utf-8')),262144)
        bundle = json.loads(body['input'][0]['content'])
        self.assertEqual(bundle['projection_version'],VERSION)
        context = restore_context(bundle)
        self.assertEqual(restore_request_v5(context['observation']),envelope['request'])
        self.assertEqual(context['registered_skills'],envelope['skills'])
        self.assertEqual(context['strategy_reconsideration'],stability_reconsideration_context(
            envelope['request'],envelope['escalation']['strategy_orientation_reference']))
        self.assertEqual(context['strategy_reconsideration']['catalog_pose_count'],6)
        self.assertEqual(len(context['strategy_reconsideration']['poses_without_recorded_actions']),5)
        self.assertFalse(context['strategy_reconsideration']['automatic_pose_change'])
        self.assertTrue(bundle['projection_audit']['lossless_roundtrip_verified'])

    def test_saved_large_request_crosses_role_ledger_to_injected_valid_plan_once(self):
        req = self.envelope['request']
        wire = dict(request_id=req['request_id'],state_id=req['state_id'],
            candidate_order=list(req['candidate_ids']),recovery_skill_ids=[],steps=[dict(
                part_id=req['candidate_ids'][0], procedure='axis_search_v1', registered_skill_id=None,
                pose_policy='select_catalog_once_v1',support_policy='replace_once_v1',
                max_attempts=3,evidence_ids=[],stability_failover=None)])
        raw = provider(wire);raw['model']='gpt-6.1-sol'
        transport = Mock(return_value=raw)
        service = self.service(transport)
        original = deepcopy(self.envelope)
        result = service.rank(self.envelope)
        self.assertEqual(result['status'],'planned',result)
        self.assertTrue(result['metadata']['attempted'])
        self.assertFalse(result['metadata']['llm_live'])
        self.assertEqual(result['metadata']['retries'],0)
        self.assertIsNone(result['fallback_reason'])
        self.assertEqual(result['plan']['first_request_sha256'],digest(req))
        self.assertEqual(result['plan']['candidate_order'],req['candidate_ids'])
        sent = transport.call_args.args[2]
        self.assertLessEqual(len(canonical(sent).encode('utf-8')),262144)
        self.assertEqual(restore_request_v5(restore_context(json.loads(sent['input'][0]['content']))['observation']),req)
        self.assertEqual(self.envelope,original)
        transport.assert_called_once()
        with closing(sqlite3.connect(self.root/'planner.sqlite')) as db:
            rows=db.execute('SELECT reserved_call,status,payload_json FROM rank_requests').fetchall()
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0][:2],(1,'completed'))
        self.assertEqual(json.loads(rows[0][2]),original)
        cached = service.rank(self.envelope)
        self.assertTrue(cached['metadata']['cached'])
        self.assertFalse(cached['metadata']['attempted'])
        transport.assert_called_once()

    def test_both_prior_successful_packets_remain_accepted_with_added_context(self):
        controls=json.loads((FIXTURES/'saved_successful_envelopes.json').read_text(encoding='utf-8'))
        self.assertEqual(len(controls),2)
        for envelope in controls:
            with self.subTest(request_id=envelope['request']['request_id']):
                body=self.service()._body(envelope)
                self.assertLessEqual(len(canonical(body).encode('utf-8')),262144)
                context=json.loads(body['input'][0]['content'])
                if context.get('projection_version')==VERSION: context=restore_context(context)
                self.assertEqual(restore_request_v5(context['observation']),envelope['request'])
                self.assertEqual(context['registered_skills'],envelope['skills'])
                self.assertIn('strategy_reconsideration',context)


if __name__ == '__main__':
    unittest.main()
