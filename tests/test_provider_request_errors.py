"""Local input failures must be distinct from SQLite and spend no API budget."""
from copy import deepcopy
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.llm_ranker import RankService
from orchestrator.provider_request_errors import ProviderBodyError
from tests.test_llm_ranker import config, envelope, provider_response, KEY


class ProviderRequestErrorsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='provider_input_')
        self.addCleanup(self.temp.cleanup)
        self.transport = Mock(side_effect=lambda url, headers, body, timeout:
            provider_response(json.loads(body['input'][0]['content'])['request']))
        self.service = RankService(config(max_calls=1), Path(self.temp.name)/'ledger.sqlite', self.transport)
        env = patch.dict(os.environ, {'RANKER_TEST_KEY': KEY})
        env.start(); self.addCleanup(env.stop)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        network_mock = network.start(); self.addCleanup(network.stop)
        self.addCleanup(network_mock.assert_not_called)

    def rows(self):
        with closing(sqlite3.connect(self.service.ledger_path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute('SELECT * FROM rank_requests')]

    def test_size_rejection_is_precise_cached_and_unreserved(self):
        request = envelope()
        original = deepcopy(request)
        error = ProviderBodyError(body_bytes=275293, max_bytes=262144)
        with patch.object(self.service, '_body', side_effect=error) as build:
            first = self.service.rank(request)
            again = self.service.rank(request)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(request, original)
        for reply in (first, again):
            self.assertEqual(reply['fallback_reason'], 'plan_provider_body_too_large')
            self.assertFalse(reply['metadata']['attempted'])
            self.assertFalse(reply['metadata']['llm_live'])
            self.assertIsNone(reply['metadata']['call_id'])
            self.assertIsNone(reply['metadata']['cost_estimate'])
            detail = reply['metadata']['provider_request_diagnostics']
            self.assertEqual(detail['body_utf8_bytes'], 275293)
            self.assertEqual(detail['excess_utf8_bytes'], 13149)
            self.assertFalse(detail['reservation_created'])
            self.assertFalse(detail['transmitted'])
        self.assertTrue(again['metadata']['cached'])
        self.assertEqual([(r['reserved_call'], r['reserved_usd'], r['status']) for r in self.rows()],
                         [(0, '0', 'completed')])
        self.transport.assert_not_called()

    def test_arbitrary_body_exception_text_is_not_recorded(self):
        for index, error in enumerate([ValueError(KEY), TypeError(KEY), OSError(KEY), RecursionError(KEY)]):
            with self.subTest(type=type(error).__name__), patch.object(self.service, '_body', side_effect=error):
                reply = self.service.rank(envelope('body-%s' % index))
            self.assertEqual(reply['fallback_reason'], 'provider_body_error')
            self.assertEqual(reply['metadata']['provider_request_diagnostics']['stage'], 'body_build')
            self.assertNotIn(KEY, json.dumps(reply))
        self.assertNotIn(KEY, json.dumps(self.rows()))
        self.assertEqual(sum(r['reserved_call'] for r in self.rows()), 0)
        self.transport.assert_not_called()

    def test_cost_error_has_separate_stage_and_no_budget_use(self):
        with patch.object(self.service, '_estimate', side_effect=ValueError(KEY)):
            reply = self.service.rank(envelope())
        self.assertEqual(reply['fallback_reason'], 'cost_estimation_error')
        self.assertEqual(reply['metadata']['provider_request_diagnostics']['stage'], 'cost_estimation')
        self.assertNotIn(KEY, json.dumps(self.rows()))
        self.assertEqual(self.rows()[0]['reserved_call'], 0)
        self.transport.assert_not_called()

    def test_actual_database_failure_stays_ledger_error(self):
        with patch.object(self.service, '_connect', side_effect=sqlite3.OperationalError(KEY)):
            reply = self.service.rank(envelope())
        self.assertEqual(reply['fallback_reason'], 'ledger_error')
        self.assertNotIn('provider_request_diagnostics', reply['metadata'])
        self.assertNotIn(KEY, json.dumps(reply))
        self.transport.assert_not_called()

    def test_local_rejection_does_not_consume_single_call_allowance(self):
        with patch.object(self.service, '_body', side_effect=ProviderBodyError(body_bytes=262145, max_bytes=262144)):
            self.service.rank(envelope('too-large'))
        reply = self.service.rank(envelope('valid-next-request'))
        self.assertIsNone(reply['fallback_reason'])
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(sum(r['reserved_call'] for r in self.rows()), 1)
        denied = self.service.rank(envelope('over-cap'))
        self.assertEqual(denied['fallback_reason'], 'call_budget_exhausted')
        self.assertEqual(self.transport.call_count, 1)

    def test_same_id_changed_payload_still_rejected(self):
        with patch.object(self.service, '_body', side_effect=ValueError(KEY)):
            self.service.rank(envelope())
        changed = envelope()
        changed['features']['description'] = 'changed'
        reply = self.service.rank(changed)
        self.assertEqual(reply['fallback_reason'], 'request_id_conflict')
        self.assertEqual(len(self.rows()), 1)
        self.transport.assert_not_called()

    def test_missing_key_never_prepares_or_sends_body(self):
        with patch.dict(os.environ, {'RANKER_TEST_KEY': ''}), patch.object(self.service, '_body') as build:
            reply = self.service.rank(envelope())
        self.assertEqual(reply['fallback_reason'], 'missing_api_key')
        build.assert_not_called()
        self.transport.assert_not_called()

    def test_size_error_only_accepts_safe_code_and_valid_size_evidence(self):
        for kwargs in [dict(reason=KEY, body_bytes=20, max_bytes=10),
                       dict(body_bytes=True, max_bytes=1), dict(body_bytes=9, max_bytes=10),
                       dict(body_bytes=10, max_bytes=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError) as ctx:
                ProviderBodyError(**kwargs)
            self.assertNotIn(KEY, str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
