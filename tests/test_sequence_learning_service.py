"""Injected-model fixtures only; no live API, physics, or registration claims."""
from copy import deepcopy
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.sequence_knowledge import build_native_snapshot
from orchestrator.sequence_learning_service import (SequenceLearningService, make_learning_request,
    MAX_BODY_BYTES, MODEL)


PROGRAM = """def propose(ctx):
    order = sorted(ctx['candidate_ids'], key=lambda part: ctx['current_features'][part]['volume'], reverse=True)
    return {'candidate_order': order, 'evidence_ids': []}
"""


def packet():
    from orchestrator.sequence_learning import build_sequence_learning_packet
    attempts = []
    for index, passed in enumerate((False, True), 1):
        attempts.append(dict(event_id='fixture:attempt:'+str(index), part_move='a',
            remaining_parts=['a', 'b', 'c'],
            request=dict(part_move='a', parts_rest=['b', 'c'], parts_removed=[], pose=None,
                         max_grippers=3, timeout=60), returned=True,
            status='PASS' if passed else 'UNKNOWN', accepted_for_transition=passed,
            deadline_exceeded=False, verifier_domain='ASAP_RedMax',
            raw_sim_info=dict(feasible=passed, action=[0, 1, 0] if passed else None,
                              parts_fix=['b'] if passed else None),
            source_hashes={'fixture_sha256': 'd'*64}, selection_scope='ranked_candidate'))
    frozen = build_native_snapshot(dict(task=dict(assembly_id='fixture-training', input_sha256='b'*64,
        solver_settings=dict(seed=0, fixture=True))), attempts, 'fixture/worker_result.json', 'c'*64)
    return build_sequence_learning_packet(frozen)


def claim(p):
    ids = [f['evidence_id'] for f in p['facts']]
    return dict(title='Fixture conditional ordering', hypothesis='Current size may guide ordering.',
        when_to_apply='When current candidate volumes are available.',
        when_not_to_apply='When current volume is missing or uncertain.',
        ranking_guidance='Compute a current feature based order; verify every selected part afresh.',
        evidence_ids=[ids[0]], counterexample_ids=ids[1:2])


def contexts(p):
    return [dict(candidate_ids=['x', 'y'], current_features={'x': {'volume': 1}, 'y': {'volume': 2}},
        memory_references=[], lesson_references=[], evidence_ids=[f['evidence_id'] for f in p['facts']])]


def config(**updates):
    value = dict(enabled=True, model=MODEL, base_url='https://api.openai.com/v1',
        api_key_env='SEQUENCE_LEARNING_FIXTURE_KEY', max_calls=4, max_usd='1',
        input_usd_per_million='0.2', output_usd_per_million='1', max_output_tokens=2048,
        reasoning_effort='low', request_overhead_tokens=2048)
    value.update(updates)
    return value


def provider_response(value, model=MODEL):
    return dict(status='completed', model=model, id='fixture-response',
        usage=dict(input_tokens=50, output_tokens=50, total_tokens=100),
        output=[dict(type='message', role='assistant', status='completed',
                     content=[dict(type='output_text', text=json.dumps(value))])])


class LearningServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sequence_learning_OFFLINE_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, SEQUENCE_LEARNING_FIXTURE_KEY='OFFLINE_TEST_SECRET')
        env.start(); self.addCleanup(env.stop)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.network = net.start(); self.addCleanup(net.stop)
        self.addCleanup(self.network.assert_not_called)
        self.p = packet()

    def lesson(self):
        from orchestrator.sequence_learning import validate_sequence_lesson
        return validate_sequence_lesson(self.p, claim(self.p), dict(origin='injected_transport',
            model=MODEL, request_id='fixture-initial', response_sha256='e'*64))

    def request(self, stage='lesson_initial', identity='task:initial', **updates):
        material = dict(packet=self.p)
        if stage.startswith('python_'):
            material.update(lesson=self.lesson(), runtime_contexts=contexts(self.p))
        material.update(updates)
        return make_learning_request('fixture-run', identity, stage, material)

    def value(self, envelope, source=PROGRAM):
        r = envelope['request']
        field = 'lesson' if r['stage'].startswith('lesson_') else 'candidate'
        output = claim(self.p) if field == 'lesson' else dict(name='Fixture volume order',
            description='Order current parts by supplied volume.', source=source,
            evidence_ids=[self.p['facts'][0]['evidence_id']])
        return dict(request_id=r['request_id'], state_id=r['state_id'], **{field: output})

    def service(self, transport=None, policy=None, name='learner.sqlite'):
        return SequenceLearningService(policy or config(), self.root/name, transport)

    def test_initial_lesson_returns_validated_hypothesis_and_cache_uses_no_second_call(self):
        req = self.request()
        send = Mock(return_value=provider_response(self.value(req)))
        service = self.service(send)
        result = service.learn(req)
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['lesson']['protocol'], 'sequence_lesson_v1')
        self.assertEqual(result['origin'], 'injected_transport')
        self.assertFalse(result['metadata']['physics_verified'])
        self.assertFalse(result['metadata']['registered'])
        cached = service.learn(req)
        self.assertTrue(cached['ok'])
        self.assertFalse(cached['metadata']['attempted'])
        self.assertEqual(send.call_count, 1)

    def test_generated_python_is_candidate_not_registered_or_physically_verified(self):
        req = self.request('python_generate', 'task:python')
        send = Mock(return_value=provider_response(self.value(req)))
        result = self.service(send).learn(req)
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['candidate']['status'], 'CANDIDATE')
        self.assertEqual(result['candidate']['program']['source'], PROGRAM)
        self.assertFalse(result['metadata']['registered'])
        self.assertEqual(send.call_count, 1)

    def test_rejected_source_is_inert_and_preserved_for_explicit_revision(self):
        req = self.request('python_generate', 'task:bad-python')
        bad = "import os\ndef propose(ctx):\n    return None\n"
        result = self.service(Mock(return_value=provider_response(self.value(req, bad)))).learn(req)
        self.assertFalse(result['ok'])
        self.assertFalse(result['halt_required'])
        self.assertEqual(result['raw_candidate_source']['source_text'], bad)
        self.assertFalse(result['raw_candidate_source']['executable'])
        self.assertFalse(result['raw_candidate_source']['registered'])

    def test_validation_feedback_and_previous_source_reach_separate_revision_request(self):
        previous = dict(source="def propose(ctx):\n    return None\n", status='REJECTED')
        feedback = dict(stage='contract', status='REJECTED', reason='always_abstains', actual_physics=False)
        req = self.request('python_revision', 'task:revision', previous_candidate=previous,
                           validation_feedback=feedback)
        send = Mock(return_value=provider_response(self.value(req)))
        result = self.service(send).learn(req)
        self.assertTrue(result['ok'], result)
        body = send.call_args.args[2]
        value = json.loads(body['input'][0]['content'])
        self.assertEqual(value['input']['previous_candidate'], previous)
        self.assertEqual(value['input']['validation_feedback'], feedback)
        self.assertIn('superficial renaming is not a repair', body['instructions'])
        self.assertEqual(send.call_count, 1)

    def test_postvalidation_lesson_reads_fixed_candidate_and_feedback(self):
        previous = dict(program=dict(language='python_policy_v1', source=PROGRAM), status='CANDIDATE')
        feedback = dict(status='PASS', scope='fixture_only_no_actual_physics')
        req = self.request('lesson_postvalidation', 'task:post', previous_candidate=previous,
                           validation_feedback=feedback)
        send = Mock(return_value=provider_response(self.value(req)))
        result = self.service(send).learn(req)
        self.assertTrue(result['ok'], result)
        self.assertIn('Do not modify code', send.call_args.args[2]['instructions'])

    def test_shared_ledger_enforces_one_pool_across_stages_and_task_run_ids(self):
        first = self.request('lesson_initial', 'task1:initial')
        second = self.request('python_generate', 'task2:python')
        send = Mock(side_effect=[provider_response(self.value(first)), provider_response(self.value(second))])
        policy = config(max_calls=1)
        self.assertTrue(self.service(send, policy).learn(first)['ok'])
        result = self.service(send, policy).learn(second)
        self.assertEqual(result['error'], 'call_budget_exhausted')
        self.assertTrue(result['halt_required'])
        self.assertEqual(send.call_count, 1)

    def test_transport_failure_stops_following_paid_requests_without_retry(self):
        send = Mock(side_effect=TimeoutError('fixture timeout'))
        service = self.service(send)
        result = service.learn(self.request())
        self.assertEqual(result['error'], 'api_timeout')
        self.assertTrue(result['halt_required'])
        result = service.learn(self.request(identity='task:second'))
        self.assertEqual(result['error'], 'sequence_learning_session_halted')
        self.assertEqual(send.call_count, 1)
        self.assertEqual(result['metadata']['retries'], 0)

    def test_stale_response_is_contract_error_and_can_be_explicitly_revised(self):
        req = self.request()
        value = self.value(req); value['request_id'] = 'old-request'
        send = Mock(return_value=provider_response(value))
        result = self.service(send).learn(req)
        self.assertFalse(result['ok'])
        self.assertFalse(result['halt_required'])
        self.assertEqual(result['error'], 'stale_request_or_state')
        self.assertEqual(send.call_count, 1)

    def test_duplicate_json_keys_and_incomplete_output_are_never_accepted(self):
        req = self.request()
        for raw in (dict(provider_response(self.value(req)), status='incomplete'),
                    provider_response(self.value(req))):
            if raw['status'] == 'completed':
                raw['output'][0]['content'][0]['text'] = '{"request_id":"a","request_id":"b"}'
            with self.subTest(status=raw['status']):
                result = self.service(Mock(return_value=raw), name=raw['status']+'.sqlite').learn(req)
                self.assertFalse(result['ok'])
                self.assertIsNone(result['lesson'])

    def test_provider_secret_is_redacted_even_in_rejected_python_and_ledger(self):
        req = self.request('python_generate', 'task:secret')
        source = "def propose(ctx):\n    return 'OFFLINE_TEST_SECRET'\n"
        result = self.service(Mock(return_value=provider_response(self.value(req, source)))).learn(req)
        self.assertEqual(result['error'], 'provider_secret_in_response')
        self.assertTrue(result['halt_required'])
        self.assertNotIn('OFFLINE_TEST_SECRET', json.dumps(result))
        self.assertIn('[REDACTED_API_KEY]', result['raw_candidate_source']['source_text'])
        with closing(sqlite3.connect(self.root/'learner.sqlite')) as db:
            stored = db.execute('select response_json from rank_requests').fetchone()[0]
        self.assertNotIn('OFFLINE_TEST_SECRET', stored)

    def test_preview_disabled_needs_no_key_and_consumes_no_reservation(self):
        send = Mock(side_effect=AssertionError('TRANSPORT_FORBIDDEN'))
        service = self.service(send, config(enabled=False, max_calls=0, max_usd=0))
        with patch.dict(os.environ, {}, clear=True):
            preview = service.preview(self.request())
        self.assertEqual(preview['API_calls'], 0)
        self.assertFalse(preview['key_accessed'])
        self.assertLessEqual(preview['body_bytes'], MAX_BODY_BYTES)
        send.assert_not_called()
        with closing(sqlite3.connect(self.root/'learner.sqlite')) as db:
            self.assertEqual(db.execute('select count(*) from rank_requests').fetchone()[0], 0)

    def test_cost_exhaustion_does_not_send(self):
        send = Mock(side_effect=AssertionError('TRANSPORT_FORBIDDEN'))
        result = self.service(send, config(max_usd='0.0000001')).learn(self.request())
        self.assertEqual(result['error'], 'cost_budget_exhausted')
        send.assert_not_called()

    def test_unknown_evidence_and_wrong_provider_model_rejected(self):
        req = self.request()
        for kind in ('unknown_evidence', 'wrong_model'):
            value = self.value(req)
            if kind == 'unknown_evidence':
                value['lesson']['evidence_ids'] = ['invented']
            send = Mock(return_value=provider_response(value, model='not-luna' if kind == 'wrong_model' else MODEL))
            result = self.service(send, name=kind+'.sqlite').learn(req)
            self.assertFalse(result['ok'])

    def test_learning_input_hash_detects_mutation_before_reservation(self):
        req = self.request()
        req['request']['input']['runtime_contexts'] = [{}]
        send = Mock(side_effect=AssertionError('TRANSPORT_FORBIDDEN'))
        result = self.service(send).learn(req)
        self.assertEqual(result['error'], 'invalid_request')
        send.assert_not_called()

    def test_revision_requires_explicit_prior_and_feedback(self):
        with self.assertRaisesRegex(ValueError, 'revision_requires'):
            self.request('python_revision', 'task:missing')

    def test_body_byte_limit_rejects_before_key_or_transport_and_keeps_cap_exact(self):
        req = self.request('python_revision', 'task:large', previous_candidate={'source': PROGRAM},
                           validation_feedback={'detail': 'x'*145000})
        service = self.service(Mock(side_effect=AssertionError('TRANSPORT_FORBIDDEN')))
        with self.assertRaisesRegex(ValueError, 'body_too_large'):
            service.preview(req)

    def test_compact_fact_tables_reconstruct_every_condition_without_loss(self):
        preview = self.service(Mock()).preview(self.request())
        compact = json.loads(preview['body']['input'][0]['content'])['input']['packet']
        reconstructed = []
        for supplied in compact['facts']:
            row = deepcopy(supplied)
            row['source_task'] = compact['source_tasks'][row.pop('source_task_ref')]
            row['provenance'] = dict(compact['provenance_sources'][row.pop('provenance_ref')], **row['provenance'])
            reconstructed.append(row)
        self.assertEqual(reconstructed, self.p['facts'])
        self.assertEqual(compact['source_packet_sha256'], self.p['content_sha256'])

    def test_real_stored_00335_input_four_stage_previews_fit_without_network(self):
        """Read-only sizing replay, NOT training or validation on the held-out task."""
        from orchestrator.sequence_learning import (build_sequence_learning_packet,
            validate_sequence_lesson, make_sequence_candidate)
        from orchestrator.sequence_ranker import SequenceRankService
        from orchestrator.llm_ranker import DEFAULTS
        root = Path(__file__).resolve().parents[1]
        run = root/'runs/sequence_native_00335_live_20261110'
        memory_file = run/'trial/native_observations.json'
        if not memory_file.is_file():
            self.skipTest('Optional real saved-input sizing fixture is unavailable')
        def read(p):
            return json.loads(p.read_text(encoding='utf8'))
        actual = build_sequence_learning_packet(read(memory_file))
        actual_claim = claim(actual)
        provenance = dict(origin='injected_transport', model=MODEL, request_id='sizing-only', response_sha256='e'*64)
        lesson = validate_sequence_lesson(actual, actual_claim, provenance)
        candidate = make_sequence_candidate(lesson, dict(name='Sizing fixture',
            description='Fixture only, not generated or registered.', source=PROGRAM,
            evidence_ids=actual_claim['evidence_ids']), provenance)
        ranker = SequenceRankService(dict(DEFAULTS), self.root/'actual-context.sqlite',
            memory_records=read(run/'frozen/memory.json'), skills=read(run/'frozen/skills.json'),
            lessons=read(run/'frozen/lessons.json'))
        request_file = max((run/'trial/episode/ipc/requests').glob('*.json'), key=lambda p: p.stat().st_size)
        runtime = ranker.prepare(read(request_file))['features']['sequence_knowledge']
        service = self.service(Mock(side_effect=AssertionError('TRANSPORT_FORBIDDEN')))
        for stage in ('lesson_initial', 'python_generate', 'python_revision', 'lesson_postvalidation'):
            material = dict(packet=actual, lesson=None if stage=='lesson_initial' else lesson,
                runtime_contexts=[runtime], previous_candidate=candidate if stage in ('python_revision','lesson_postvalidation') else None,
                validation_feedback={'status':'fixture_only', 'feedback': [{'code':'sizing-only'}]} if stage in ('python_revision','lesson_postvalidation') else None)
            req = make_learning_request('sizing-only', 'sizing:'+stage, stage, material)
            preview = service.preview(req)
            with self.subTest(stage=stage, bytes=preview['body_bytes']):
                self.assertLessEqual(preview['body_bytes'], MAX_BODY_BYTES)
                sent = json.loads(preview['body']['input'][0]['content'])['input']
                expected = {k: deepcopy(runtime[k]) for k in ('protocol','candidate_ids','task','current_features',
                    'memory_references','lesson_references','evidence_ids') if k in runtime}
                self.assertEqual(sent['runtime_contexts'], [expected])
                self.assertEqual(len(sent['packet']['facts']), actual['selected_fact_count'])
                if sent['lesson'] is not None:
                    self.assertNotIn('facts', sent['lesson'])
                    self.assertNotIn('packet', sent['lesson'])


if __name__ == '__main__':
    unittest.main()
