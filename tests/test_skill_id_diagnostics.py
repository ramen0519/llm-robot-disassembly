"""Offline skill-ID contracts; injected responses are never provider/physics runs."""
from contextlib import closing, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.llm_ranker import MAX_RESPONSE_BYTES
from orchestrator.skill_library import SkillMemory
from orchestrator.skill_pipeline import SkillPipeline
from orchestrator.skill_selector import PROTOCOL, SkillSelector, response_fixture
from tests.test_llm_ranker import config, envelope as rank_envelope


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / 'runs/skill_id_diagnostics_20261017/scratch'
FIELDS = ('request_id', 'state_id', 'library_sha256')
DIAGNOSTIC_KEYS = {
    'schema_version', 'object_schema_valid', 'request_id_matches',
    'state_id_matches', 'library_sha256_matches', 'skill_id_offered',
}
MARKER = 'private-provider-returned-id-do-not-persist'


def envelope(request_id='fixture:r1'):
    result = rank_envelope(request_id)
    result['request'].update(decision_protocol=PROTOCOL,
        offered_skill_ids=['native_order_v1', 'observed_fixture'], library_sha256='a' * 64)
    return result


def changed_response(request, changes=None, remove=None):
    raw = response_fixture(request, 'observed_fixture')
    content = raw['output'][0]['content'][0]
    value = json.loads(content['text'])
    value.update(changes or {})
    if remove is not None:
        del value[remove]
    content['text'] = json.dumps(value)
    return raw


class SkillIdDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix='offline_', dir=SCRATCH)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.serial = 0
        for target in ('urllib.request.OpenerDirector.open', 'socket.create_connection'):
            guard = patch(target, side_effect=AssertionError('network_forbidden_in_offline_test'))
            blocked = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(blocked.assert_not_called)
        fixture_environment = patch.dict(os.environ, {'RANKER_TEST_KEY': 'fixture-only-no-real-key'})
        fixture_environment.start()
        self.addCleanup(fixture_environment.stop)

    def service(self, transport=None, settings=None, ledger=None):
        self.serial += 1
        return SkillSelector(config() if settings is None else settings,
            ledger or self.root / ('ledger_%d.sqlite' % self.serial), transport)

    def diagnostics(self, response):
        diagnostic = response['metadata']['selection_diagnostics']
        self.assertEqual(set(diagnostic), DIAGNOSTIC_KEYS)
        self.assertEqual(diagnostic['schema_version'], 1)
        for key, value in diagnostic.items():
            if key != 'schema_version':
                self.assertTrue(value is None or type(value) is bool, key)
        return diagnostic

    def assert_not_evaluated(self, response):
        self.assertTrue(all(value is None for key, value in self.diagnostics(response).items()
                            if key != 'schema_version'))

    def test_schema_binds_all_three_ids_and_offered_skills_per_request(self):
        service = self.service(settings={})
        first = envelope()
        second = deepcopy(first)
        second['request'].update(request_id='fixture:r2', state_id='state1',
            library_sha256='b' * 64, offered_skill_ids=['native_order_v1', 'another_offered_skill'])
        before = deepcopy(first)
        bodies = [service._body(value) for value in (first, second)]
        for value, body in zip((first, second), bodies):
            with self.subTest(request=value['request']['request_id']):
                fmt = body['text']['format']
                self.assertTrue(fmt['strict'])
                schema = fmt['schema']
                self.assertFalse(schema['additionalProperties'])
                self.assertEqual(set(schema['required']), set(FIELDS) | {'skill_id'})
                self.assertEqual(set(schema['properties']), set(schema['required']))
                for name in FIELDS:
                    self.assertEqual(schema['properties'][name],
                                     {'type': 'string', 'enum': [value['request'][name]]})
                self.assertEqual(schema['properties']['skill_id']['enum'],
                                 value['request']['offered_skill_ids'])
        self.assertEqual(first, before)
        self.assertEqual(bodies[0]['text']['format']['schema']['properties']['request_id']['enum'],
                         [first['request']['request_id']])

    def test_exact_response_is_accepted_and_diagnostics_are_boolean_only(self):
        request = envelope()['request']
        raw = response_fixture(request, 'observed_fixture')
        before = deepcopy(raw)
        selection, reason, diagnostic = SkillSelector._parse_with_diagnostics(raw, request)
        self.assertEqual(selection, ['observed_fixture'])
        self.assertIsNone(reason)
        self.assertEqual(set(diagnostic), DIAGNOSTIC_KEYS)
        self.assertTrue(all(value is True for key, value in diagnostic.items() if key != 'schema_version'))
        self.assertEqual(SkillSelector._parse(raw, request), (selection, reason))
        self.assertEqual(raw, before)

    def test_each_identifier_mismatch_is_distinguished_in_service_and_ledger(self):
        request = envelope()['request']
        cases = [({'request_id': 'old_request'}, {'request_id_matches'}, 'stale_request_or_state'),
                 ({'state_id': 'old_state'}, {'state_id_matches'}, 'stale_request_or_state'),
                 ({'request_id': 'old_request', 'state_id': 'old_state'},
                  {'request_id_matches', 'state_id_matches'}, 'stale_request_or_state'),
                 ({'library_sha256': 'b' * 64}, {'library_sha256_matches'}, 'stale_skill_library'),
                 ({'skill_id': 'unregistered_function'}, {'skill_id_offered'}, 'unoffered_skill')]
        for changes, failed_fields, expected_reason in cases:
            with self.subTest(changed_fields=sorted(changes)):
                transport = Mock(return_value=changed_response(request, changes))
                service = self.service(transport)
                response = service.rank(envelope())
                self.assertEqual(response['fallback_reason'], expected_reason)
                self.assertEqual(response['candidate_ids'], request['candidate_ids'])
                self.assertIsNone(response['metadata']['selected_skill_id'])
                diagnostic = self.diagnostics(response)
                self.assertTrue(diagnostic['object_schema_valid'])
                for key in DIAGNOSTIC_KEYS - {'schema_version', 'object_schema_valid'}:
                    self.assertIs(diagnostic[key], key not in failed_fields)
                with closing(sqlite3.connect(service.ledger_path)) as db:
                    stored, count = db.execute('SELECT response_json, reserved_call FROM rank_requests').fetchone()
                self.assertEqual(json.loads(stored)['metadata']['selection_diagnostics'], diagnostic)
                self.assertEqual(count, 1)
                self.assertEqual(transport.call_count, 1)

    def test_malformed_json_and_duplicate_keys_are_rejected(self):
        request = envelope()['request']
        for text in ('{', '[1, 2]', 'null',
                     '{"skill_id":"native_order_v1","skill_id":"observed_fixture",'
                     '"request_id":"fixture:r1","state_id":"state0","library_sha256":"' + 'a' * 64 + '"}'):
            with self.subTest(shape=text[:20]):
                raw = response_fixture(request, 'observed_fixture')
                raw['output'][0]['content'][0]['text'] = text
                response = self.service(Mock(return_value=raw)).rank(envelope())
                self.assertIsNotNone(response['fallback_reason'])
                self.assertEqual(response['candidate_ids'], request['candidate_ids'])
                self.assertFalse(self.diagnostics(response)['object_schema_valid'])

    def test_missing_extra_and_wrong_type_fields_fail_schema(self):
        request = envelope()['request']
        malformed = [('missing_' + name, changed_response(request, remove=name))
                     for name in FIELDS + ('skill_id',)]
        malformed += [('extra', changed_response(request, {'candidate_ids': []}))]
        malformed += [('type_' + name, changed_response(request, {name: 123}))
                      for name in FIELDS + ('skill_id',)]
        for label, raw in malformed:
            with self.subTest(case=label):
                response = self.service(Mock(return_value=raw)).rank(envelope())
                self.assertIsNotNone(response['fallback_reason'])
                self.assertIsNone(response['metadata']['selected_skill_id'])
                self.assertEqual(response['candidate_ids'], request['candidate_ids'])
                diagnostic = self.diagnostics(response)
                self.assertFalse(diagnostic['object_schema_valid'])
                if label.startswith(('missing_', 'type_')):
                    name = label.split('_', 1)[1]
                    check = 'skill_id_offered' if name == 'skill_id' else name + '_matches'
                    self.assertIsNone(diagnostic[check])

    def test_oversized_response_is_rejected_before_evaluating_ids(self):
        raw = response_fixture(envelope()['request'], 'observed_fixture')
        raw['output'][0]['content'][0]['text'] = ' ' * (MAX_RESPONSE_BYTES + 1)
        response = self.service(Mock(return_value=raw)).rank(envelope())
        self.assertIsNotNone(response['fallback_reason'])
        diagnostic = self.diagnostics(response)
        for name in ('request_id_matches', 'state_id_matches', 'library_sha256_matches', 'skill_id_offered'):
            self.assertIsNone(diagnostic[name])

    def test_lone_surrogate_and_deep_json_fail_closed_without_exception_text(self):
        request = envelope()['request']
        depth = sys.getrecursionlimit() + 10
        for label, text in [('lone_surrogate', '\ud800'),
                            ('deep_json', '[' * depth + '0' + ']' * depth)]:
            with self.subTest(case=label):
                raw = response_fixture(request, 'observed_fixture')
                raw['output'][0]['content'][0]['text'] = text
                selection, reason, diagnostic = SkillSelector._parse_with_diagnostics(raw, request)
                self.assertIsNone(selection)
                self.assertEqual(reason, 'invalid_skill_response')
                self.assertFalse(diagnostic['object_schema_valid'])
                for key in DIAGNOSTIC_KEYS - {'schema_version', 'object_schema_valid'}:
                    self.assertIsNone(diagnostic[key])
                transport = Mock(return_value=raw)
                service = self.service(transport)
                response = service.rank(envelope())
                self.assertEqual(response['fallback_reason'], reason)
                self.assertEqual(response['candidate_ids'], request['candidate_ids'])
                self.assertEqual(self.diagnostics(response), diagnostic)
                self.assertNotIn('error_type', response['metadata'])
                self.assertEqual(transport.call_count, 1)

    def test_multiple_messages_and_tool_calls_are_not_treated_as_a_selection(self):
        request = envelope()['request']
        multiple = response_fixture(request, 'observed_fixture')
        multiple['output'].append(deepcopy(multiple['output'][0]))
        tool_call = response_fixture(request, 'observed_fixture')
        tool_call['output'].append({'type': 'function_call', 'name': 'unregistered_action',
                                    'arguments': MARKER, 'call_id': 'fixture_call'})
        for label, raw in [('multiple_messages', multiple), ('tool_call', tool_call)]:
            with self.subTest(case=label):
                transport = Mock(return_value=raw)
                service = self.service(transport)
                response = service.rank(envelope())
                self.assertIsNotNone(response['fallback_reason'])
                self.assertEqual(response['candidate_ids'], request['candidate_ids'])
                self.assertIsNone(response['metadata']['selected_skill_id'])
                self.assert_not_evaluated(response)
                self.assertNotIn(MARKER, json.dumps(response))
                self.assertNotIn(MARKER.encode(), service.ledger_path.read_bytes())
                self.assertEqual(transport.call_count, 1)

    def test_refusal_incomplete_and_invalid_outer_messages_fall_back(self):
        request = envelope()['request']
        refusal = response_fixture(request, 'observed_fixture')
        refusal['output'][0]['content'] = [{'type': 'refusal', 'refusal': MARKER}]
        incomplete = response_fixture(request, 'observed_fixture')
        incomplete['status'] = 'incomplete'
        wrong_role = response_fixture(request, 'observed_fixture')
        wrong_role['output'][0]['role'] = 'user'
        for label, raw in [('refusal', refusal), ('incomplete', incomplete), ('wrong_role', wrong_role),
                           ('wrong_output_type', {'status': 'completed', 'output': 42}), ('not_object', [])]:
            with self.subTest(case=label):
                transport = Mock(return_value=raw)
                response = self.service(transport).rank(envelope())
                self.assertIsNotNone(response['fallback_reason'])
                self.assertEqual(response['candidate_ids'], request['candidate_ids'])
                self.assertIsNone(response['metadata']['selected_skill_id'])
                self.assert_not_evaluated(response)
                self.assertNotIn(MARKER, json.dumps(response))
                self.assertEqual(transport.call_count, 1)

    def test_cached_rejection_retains_diagnostics_after_restart_without_retry(self):
        transport = Mock(return_value=changed_response(envelope()['request'], {'state_id': 'stale'}))
        service = self.service(transport)
        first = service.rank(envelope())
        restarted = self.service(transport, ledger=service.ledger_path)
        second = restarted.rank(envelope())
        self.assertEqual(second['fallback_reason'], 'stale_request_or_state')
        self.assertEqual(self.diagnostics(first), self.diagnostics(second))
        self.assertTrue(second['metadata']['cached'])
        self.assertFalse(second['metadata']['attempted'])
        self.assertEqual(transport.call_count, 1)

    def test_legacy_cache_without_diagnostics_is_not_reissued_or_reconstructed(self):
        transport = Mock(return_value=changed_response(envelope()['request'], {'state_id': 'old'}))
        service = self.service(transport)
        service.rank(envelope())
        # Create a legacy-format fixture only in this test's disposable new ledger.
        with closing(sqlite3.connect(service.ledger_path)) as db, db:
            raw = db.execute('SELECT response_json FROM rank_requests').fetchone()[0]
            legacy = json.loads(raw)
            del legacy['metadata']['selection_diagnostics']
            legacy_text = json.dumps(legacy)
            db.execute('UPDATE rank_requests SET response_json=?', (legacy_text,))
        reopened = self.service(transport, ledger=service.ledger_path)
        response = reopened.rank(envelope())
        self.assertTrue(response['metadata']['cached'])
        self.assertFalse(response['metadata']['attempted'])
        self.assertNotIn('selection_diagnostics', response['metadata'])
        self.assertEqual(response['fallback_reason'], 'stale_request_or_state')
        self.assertEqual(transport.call_count, 1)
        with closing(sqlite3.connect(service.ledger_path)) as db:
            stored = db.execute('SELECT response_json FROM rank_requests').fetchone()[0]
        self.assertEqual(stored, legacy_text)

    def test_diagnostics_do_not_leak_between_requests_on_the_same_service(self):
        first, second, third, fourth = [envelope('fixture:r%d' % n) for n in range(1, 5)]
        second['request'].update(state_id='another_state', library_sha256='b' * 64)
        bad_shape = response_fixture(fourth['request'], 'observed_fixture')
        bad_shape['output'][0]['content'][0]['text'] = '{'
        transport = Mock(side_effect=[
            changed_response(first['request'], {'request_id': 'stale_request'}),
            response_fixture(second['request'], 'observed_fixture'),
            TimeoutError(MARKER), bad_shape])
        service = self.service(transport)
        stale = service.rank(first)
        stale_diagnostics = deepcopy(self.diagnostics(stale))
        self.assertFalse(stale_diagnostics['request_id_matches'])
        accepted = service.rank(second)
        self.assertIsNone(accepted['fallback_reason'])
        self.assertTrue(all(value is True for key, value in self.diagnostics(accepted).items()
                            if key != 'schema_version'))
        timeout = service.rank(third)
        self.assertEqual(timeout['fallback_reason'], 'api_timeout')
        self.assert_not_evaluated(timeout)
        malformed = service.rank(fourth)
        diagnostic = self.diagnostics(malformed)
        self.assertFalse(diagnostic['object_schema_valid'])
        for key in DIAGNOSTIC_KEYS - {'schema_version', 'object_schema_valid'}:
            self.assertIsNone(diagnostic[key])
        self.assertEqual(self.diagnostics(stale), stale_diagnostics)
        self.assertEqual(self.diagnostics(service.rank(first)), stale_diagnostics)
        self.assertEqual(transport.call_count, 4)

    def test_disabled_cost_budget_timeout_and_call_budget_are_not_evaluated(self):
        for settings, side_effect, expected in [(config(enabled=False), None, 'llm_disabled'),
                (config(max_usd='0.00000001'), None, 'cost_budget_exhausted'),
                (config(max_calls=1), TimeoutError(MARKER), 'api_timeout')]:
            with self.subTest(reason=expected):
                transport = Mock(side_effect=side_effect)
                service = self.service(transport, settings)
                response = service.rank(envelope())
                self.assertEqual(response['fallback_reason'], expected)
                self.assert_not_evaluated(response)
                expected_calls = 1 if side_effect is not None else 0
                self.assertEqual(transport.call_count, expected_calls)
                cached = service.rank(envelope())
                self.assert_not_evaluated(cached)
                self.assertEqual(transport.call_count, expected_calls)
                if side_effect is not None:
                    blocked = service.rank(envelope('fixture:r2'))
                    self.assertEqual(blocked['fallback_reason'], 'call_budget_exhausted')
                    self.assert_not_evaluated(blocked)
                    self.assertEqual(transport.call_count, 1)

    def test_returned_sensitive_identifier_is_not_in_response_ledger_or_console(self):
        for name in FIELDS + ('skill_id',):
            with self.subTest(field=name):
                raw = changed_response(envelope()['request'], {name: MARKER})
                service = self.service(Mock(return_value=raw))
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    response = service.rank(envelope())
                self.assertIsNotNone(response['fallback_reason'])
                self.diagnostics(response)
                self.assertNotIn(MARKER, json.dumps(response))
                self.assertNotIn(MARKER, out.getvalue() + err.getvalue())
                self.assertNotIn(MARKER.encode(), service.ledger_path.read_bytes())

    def test_pipeline_stale_selection_preserves_candidates_state_and_frozen_memory(self):
        context = json.loads((ROOT / 'runs/luna_live_smoke_20261017/context.json').read_text())
        store = SkillMemory(self.root / 'memory.sqlite')
        self.addCleanup(store.close)
        frozen = self.root / 'frozen'
        store.freeze(frozen)
        before_files = {p.name: p.read_bytes() for p in frozen.iterdir() if p.is_file()}
        offered = envelope()
        saved_envelope = deepcopy(offered)
        saved_context = deepcopy(context)
        calls = []

        def transport(url, headers, body, timeout):
            calls.append(body)
            request = json.loads(body['input'][0]['content'])['request']
            raw = response_fixture(request, 'native_order_v1')
            content = raw['output'][0]['content'][0]
            value = json.loads(content['text'])
            value['state_id'] = MARKER
            content['text'] = json.dumps(value)
            return raw

        pipeline = SkillPipeline(config(max_calls=1), self.root / 'pipeline_ledger.sqlite',
                                 frozen, context, store, 'live', transport)
        response = pipeline.rank(offered)
        self.assertEqual(response['candidate_ids'], saved_envelope['request']['candidate_ids'])
        self.assertEqual(offered, saved_envelope)
        self.assertEqual(context, saved_context)
        self.assertEqual(response['metadata']['llm_fallback_reason'], 'stale_request_or_state')
        self.assertEqual(response['metadata']['selected_by'], 'deterministic_fallback')
        self.assertEqual(response['metadata']['skill_id'], 'native_order_v1')
        self.assertFalse(response['metadata']['candidate_pruning'])
        self.assertFalse(response['metadata']['physics_bypass'])
        self.assertFalse(self.diagnostics(response)['state_id_matches'])
        invocations = store.entries('invocation')
        self.assertEqual(len(invocations), 1)
        invocation = invocations[0]
        self.assertEqual(invocation['request'], saved_envelope['request'])
        self.assertEqual(invocation['applied_order'], saved_envelope['request']['candidate_ids'])
        self.assertIsNone(invocation['physical_verdict'])
        self.assertFalse(invocation['state_mutations'])
        self.assertEqual(invocation['internal_verifier_calls'], 0)
        self.assertEqual(store.evidence.records(), [])
        self.assertEqual(store.entries('registered_skill'), [])
        self.assertEqual(store.entries('proposal'), [])
        self.assertEqual({p.name: p.read_bytes() for p in frozen.iterdir() if p.is_file()}, before_files)
        cached = pipeline.rank(offered)
        self.assertEqual(self.diagnostics(response), self.diagnostics(cached))
        self.assertTrue(cached['metadata']['invocation_reused'])
        self.assertEqual(len(store.entries('invocation')), 1)
        self.assertEqual(len(calls), 1)
        self.assertNotIn(MARKER, json.dumps(store.entries('decision_response')))
        self.assertNotIn(MARKER, json.dumps(store.entries('invocation')))


if __name__ == '__main__':
    unittest.main()
