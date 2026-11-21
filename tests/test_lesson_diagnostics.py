"""API-free rejection, contract and CLI tests with explicit injected responses."""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import json
import os
import re
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.test_skill_discovery import synthesis, config, raw_response
from tests.test_lesson_pipeline import claim
from orchestrator.lesson_discovery import LessonDiscoveryService, lesson_schema
from orchestrator.lesson_memory import evidence_view, validate_claim, LessonClaimError, claim_contract
from orchestrator.lesson_diagnostics import rejected_candidate, validate_record, save_rejection_artifacts, MAX_CANDIDATE_BYTES
from tools.run_luna_experiment import read

ROOT = Path(__file__).absolute().parents[1]
SAVED = ROOT / 'runs/skill_learning_resume_20261109_0836/live/lesson_draft'


class LessonDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='lesson_diagnostic_OFFLINE_', dir=ROOT / 'runs')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.packet = synthesis()
        self.packet['context']['lesson_stage'] = 'extract'
        self.service = LessonDiscoveryService(config(), self.root / 'direct.sqlite')
        self.addCleanup(self.service.close)
        self.net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.network = self.net.start()
        self.addCleanup(self.net.stop)
        self.addCleanup(self.network.assert_not_called)
        env = patch.dict(os.environ, DISCOVERY_FIXTURE_KEY='OFFLINE_NOT_A_KEY')
        env.start(); self.addCleanup(env.stop)

    def parse(self, candidate, packet=None):
        packet = packet or self.packet
        value = dict(request_id=packet['request_id'], state_id=packet['state_id'], lesson=candidate)
        metadata = {}
        result, reason = self.service._parse_response(raw_response(value), packet, metadata)
        return result, reason, metadata

    def test_fields_are_precise_and_do_not_echo_input_values(self):
        original = claim(self.packet)
        cases = [
            ({'evidence_ids': ['unknown-private-id']}, 'unknown_evidence_id', 'evidence_ids', 0),
            ({'evidence_ids': original['evidence_ids'] * 2}, 'duplicate_evidence_id', 'evidence_ids', 1),
            ({'counterexample_ids': ['unknown-private-id']}, 'unknown_evidence_id', 'counterexample_ids', 0),
            ({'title': ' \t\n'}, 'blank_text', 'title', None),
            ({'hypothesis': 2}, 'text_type', 'hypothesis', None),
            ({'when_to_apply': 'a' * 1501}, 'text_too_long', 'when_to_apply', None),
            ({'roles': ['\t']}, 'role_type_blank_or_length', 'roles', 0),
            ({'roles': []}, 'roles_type_or_count', 'roles', None),
            ({'evidence_ids': []}, 'empty_evidence', 'evidence_ids', None),
            ({'evidence_ids': 'not-a-list'}, 'citation_list_type', 'evidence_ids', None),
            ({'evidence_ids': [False]}, 'citation_id_type', 'evidence_ids', 0),
            ({'trigger_kind': 'invented'}, 'unknown_trigger_kind', 'trigger_kind', None),
            ({'extra_private_name': 'extra'}, 'unexpected_or_missing_fields', 'lesson', None)]
        for update, code, field, index in cases:
            with self.subTest(code=code, field=field):
                result, reason, metadata = self.parse(dict(original, **update))
                self.assertIsNone(result)
                self.assertEqual(reason, 'invalid_lesson_claim_or_evidence')
                diagnostic = metadata['lesson_validation_diagnostics']
                self.assertEqual((diagnostic['code'], diagnostic['field'], diagnostic['index']), (code, field, index))
                self.assertNotIn('unknown-private-id', json.dumps(diagnostic))
                self.assertNotIn('extra_private_name', json.dumps(diagnostic))
                self.assertFalse(validate_record(metadata['lesson_rejected_candidate'])['accepted'])

    def test_trigger_requires_an_actually_cited_matching_observation(self):
        packet = read(SAVED / 'packet.json'); facts = evidence_view(packet)
        path = next(f for f in facts if f['observed_kind'] == 'path_failed')
        stable = next(f for f in facts if f['observed_kind'] == 'stability_failed')
        value = dict(claim(packet), evidence_ids=[path['evidence_id']],
                     counterexample_ids=[stable['evidence_id']], trigger_kind='stability_failed')
        result, reason, meta = self.parse(value, packet)
        self.assertIsNone(result)
        self.assertEqual(meta['lesson_validation_diagnostics']['code'], 'trigger_not_in_cited_evidence')
        value['evidence_ids'].append(stable['evidence_id'])
        self.assertIsNone(self.parse(value, packet)[1])

    def test_valid_claim_is_not_rewritten_or_tightened_to_one_evidence(self):
        packet = read(SAVED / 'packet.json'); value = claim(packet)
        value['evidence_ids'] = [f['evidence_id'] for f in evidence_view(packet)]
        value['counterexample_ids'] = value['evidence_ids'][:1] # allowed legacy overlap
        before = deepcopy(value)
        result, reason, meta = self.parse(value, packet)
        self.assertIsNone(reason)
        self.assertEqual(result['lesson'], before)
        self.assertNotIn('lesson_rejected_candidate', meta)
        self.assertEqual(value, before)

    def test_legacy_error_category_remains_value_error_compatible(self):
        with self.assertRaisesRegex(ValueError, '^invented_or_duplicate_lesson_evidence$') as caught:
            validate_claim(dict(claim(self.packet), evidence_ids=['bad']), evidence_view(self.packet))
        self.assertIsInstance(caught.exception, LessonClaimError)

    def test_identity_rejection_records_only_known_field_names(self):
        meta = {}
        value = dict(request_id='private-wrong-id', state_id=self.packet['state_id'], lesson=claim(self.packet))
        _, reason = self.service._parse_response(raw_response(value), self.packet, meta)
        self.assertEqual(reason, 'invalid_lesson_response_identity')
        self.assertEqual(meta['lesson_validation_diagnostics']['field'], 'request_id')
        self.assertNotIn('private-wrong-id', json.dumps(meta['lesson_validation_diagnostics']))

    def test_malformed_json_records_decode_code_without_raw_text(self):
        raw = raw_response({})
        raw['output'][0]['content'][0]['text'] = 'PRIVATE_INVALID_JSON {'
        meta = {}
        result, reason = self.service._parse_response(raw, self.packet, meta)
        self.assertIsNone(result)
        self.assertEqual(meta['lesson_validation_diagnostics']['phase'], 'decode')
        self.assertNotIn('PRIVATE_INVALID_JSON', json.dumps(meta))
        self.assertNotIn('lesson_rejected_candidate', meta)
        paths = save_rejection_artifacts(self.root, {'metadata': meta})
        self.assertFalse(paths['candidate_available'])

    def test_candidate_is_bounded_redacted_and_inert(self):
        secret = 'private"test\\credential'
        candidate = {'lesson': {'title': secret, secret: 'Bearer other-token',
                               'hypothesis': 'sk-proj-fakeCredential0123456789'}}
        record = rejected_candidate(candidate, {'code': 'fixture_only'}, secret=secret)
        text = record['candidate_text']
        self.assertNotIn(secret, text)
        self.assertNotIn('sk-proj-fake', text)
        self.assertNotIn('other-token', text)
        self.assertTrue(record['redacted'])
        self.assertFalse(record['executable'])
        self.assertEqual(record['physical_calls'], 0)
        self.assertEqual(validate_record(record), record)
        huge = rejected_candidate({'title': 'a' * 100000}, {'code': 'too_big'})
        self.assertTrue(huge['truncated'])
        self.assertLessEqual(len(huge['candidate_text'].encode('utf-8')), MAX_CANDIDATE_BYTES)

    def test_deep_rejected_value_does_not_break_diagnostic_capture(self):
        value = 'untrusted'
        for _ in range(300):
            value = [value]
        record = rejected_candidate({'lesson': value}, {'code': 'invalid_fields'})
        self.assertTrue(record['nesting_truncated'])
        self.assertTrue(record['truncated'])
        self.assertEqual(validate_record(record), record)

    def test_consolidation_failures_keep_the_executable_field_and_recovery_rule(self):
        packet = deepcopy(self.packet)
        good = claim(packet)
        packet['context'].update(lesson_stage='consolidate', repair_binding=dict(
            contract={k: good[k] for k in ('trigger_kind', 'when_to_apply', 'when_not_to_apply')}))
        result, reason, meta = self.parse(dict(good, when_to_apply='mismatched'), packet)
        self.assertIsNone(result)
        self.assertEqual(meta['lesson_validation_diagnostics']['field'], 'when_to_apply')
        self.assertEqual(meta['lesson_validation_diagnostics']['phase'], 'consolidate')
        result, reason, meta = self.parse(good, packet)
        self.assertIsNone(result)
        self.assertEqual(meta['lesson_validation_diagnostics']['code'], 'final_lesson_requires_validated_recovery_citation')

    def test_artifacts_are_exclusive_and_tampering_is_rejected(self):
        _, _, metadata = self.parse(dict(claim(self.packet), evidence_ids=['bad']))
        response = dict(metadata=metadata)
        paths = save_rejection_artifacts(self.root, response)
        self.assertTrue(paths['candidate_available'])
        self.assertFalse((self.root / 'lesson.json').exists())
        with self.assertRaises(FileExistsError):
            save_rejection_artifacts(self.root, response)
        metadata['lesson_rejected_candidate']['candidate_text'] += 'changed'
        with self.assertRaises(ValueError):
            validate_record(metadata['lesson_rejected_candidate'])

    def test_rejected_transport_response_is_cached_without_retry(self):
        packet = self.packet
        wrong = dict(claim(packet), evidence_ids=['bad'])
        send = Mock(return_value=raw_response(dict(request_id=packet['request_id'], state_id=packet['state_id'], lesson=wrong)))
        service = LessonDiscoveryService(config(), self.root / 'cache.sqlite', send)
        self.addCleanup(service.close)
        response = service.propose(packet)
        cached = service.propose(packet)
        self.assertEqual(response['status'], cached['status'])
        self.assertEqual(response['metadata']['lesson_rejected_candidate'], cached['metadata']['lesson_rejected_candidate'])
        self.assertEqual(send.call_count, 1)
        self.assertEqual(response['metadata']['lesson_validation_diagnostics']['code'], 'unknown_evidence_id')

    def test_cli_saves_rejected_candidate_and_never_makes_lesson(self):
        from tools.run_lesson_discovery import extract
        packet = self.packet
        def send(url, headers, body, timeout):
            value = json.loads(body['input'][0]['content'])
            return raw_response(dict(request_id=value['request_id'], state_id=value['state_id'],
                                     lesson=dict(claim(packet), evidence_ids=['bad'])))
        transport = Mock(side_effect=send); transport.live = False
        out = self.root / 'cli'
        with patch('run_python_skill_cycle.require_live_config'), \
             patch('local_api_key.key_environment', return_value=nullcontext('OFFLINE')), \
             patch('tools.run_lesson_discovery.make_lesson', side_effect=AssertionError('REJECTED_NOT_LESSON')):
            result = extract(packet, out, config(), mode='live', transport=transport)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result['status'], 'LESSON_REJECTED')
        self.assertFalse(read(out / 'response.json')['metadata']['llm_live'])
        self.assertTrue((out / 'rejected_lesson_candidate.txt').is_file())
        self.assertEqual(read(out / 'result.json')['diagnostic']['field'], 'evidence_ids')
        self.assertFalse((out / 'lesson.json').exists())

    def test_shared_contract_and_provider_constraints_match(self):
        packet = read(SAVED / 'packet.json')
        body = self.service.preview(packet)['body']
        projected = json.loads(body['input'][0]['content'])
        contract = projected['context']['lesson_claim_contract']
        self.assertEqual(contract, claim_contract(evidence_view(packet)))
        schema = lesson_schema(packet)['properties']['lesson']['properties']
        self.assertEqual(schema['title']['maxLength'], contract['text_max_characters'])
        self.assertEqual(schema['roles']['maxItems'], contract['roles_max_items'])
        self.assertEqual(set(schema['trigger_kind']['enum']), set(contract['trigger_evidence_ids']))
        self.assertIsNone(re.search(schema['title']['pattern'], ' \n\t'))
        self.assertIn('UNIQUE', body['instructions'])
        for unsupported in ('uniqueItems', 'if', 'then', 'dependentSchemas'):
            self.assertNotIn('"' + unsupported + '":', json.dumps(body['text']['format']['schema']))
        self.assertLess(len(json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')), 196608)

    def test_saved_old_rejection_is_not_reclassified_or_fabricated(self):
        response = read(SAVED / 'response.json')
        self.assertNotIn('lesson_rejected_candidate', response['metadata'])
        self.assertNotIn('lesson_validation_diagnostics', response['metadata'])
        self.assertIsNone(save_rejection_artifacts(self.root, response))


if __name__ == '__main__':
    unittest.main()
