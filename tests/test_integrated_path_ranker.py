"""Offline contracts only: provider transport is injected, physics never runs."""
from contextlib import closing
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_independent_ranker import config
from test_prototype_ranker import make_envelope
from test_path_repair_ranker import envelope as path_envelope, candidate_with_path_feedback, provider
from test_constraint_memory import event as measured_event
from orchestrator.integrated_path_ranker import (IntegratedPathPipeline, IntegratedRankService,
    CANDIDATE_FIELDS, PATH_FIELDS, validate_worker_envelope, validate_outcome, project_feedback)
from worker.constraint_memory import records_from_evidence, FrozenConstraintMemory, save_snapshot


def candidate_history(passed=True):
    request = candidate_with_path_feedback()
    outcome = deepcopy(request['features']['episode_feedback'][0]['raw']['path_extension'])
    outcome['single_direction_failure_recovery'] = False
    outcome['native_six_axis_failure_recovery'] = True
    if passed:
        request['features']['episode_feedback'][0]['raw']['path_extension'] = deepcopy(outcome)
    else:
        request['features']['episode_feedback'] = []
        outcome.update(path_success=False, stability_success=None, parts_fix=None, feasible=False)
    context = request['features']['memory_context']
    event = measured_event(outcome['failed_evidence_id'])
    event['scope'] = deepcopy(outcome['scope'])
    event['scope']['action_schedule'] = [{'step': 0, 'action': [1, 0, 0]}]
    event['trajectory']['relative_path'] = outcome['failed_path']
    event['measurements']['pairs'][0].update(moving_part_id='1', rest_part_id='2')
    event['measurements']['pairs'][1]['moving_part_id'] = '1'
    records = records_from_evidence(context, [event], 'fixture/measured.jsonl', 'a' * 64)
    refs = FrozenConstraintMemory(records).retrieve(context, request['request']['assembled_parts'],
                                                    request['request']['candidate_ids'])['records']
    request['features']['path_feedback'] = [{'outcome': outcome,
        'constraint_references': [dict(r, origin='current_episode_history') for r in refs]}]
    return request


def choices(body):
    payload = json.loads(body['input'][0]['content'])
    request, features = payload['request'], payload['features']
    value = {'request_id': request['request_id'], 'state_id': request['state_id'],
        'candidate_ids': list(reversed(request['candidate_ids'])),
        'used_constraint_ids': [r['record_id'] for r in features['constraint_references']],
        'used_skill_evidence_ids': [c['skill_evidence_id'] for c in features['frozen_skill_evidence']],
        'used_path_feedback_ids': [h['feedback_id'] for h in features['path_feedback']]}
    if features['decision_route'] == 'path':
        value.update(skill_id='turn_force_two_leg_v1', second_axis='+Z')
    else:
        value.update(skill_id='normal_rank', repair_part_id=None)
    return value


class IntegratedPathRankerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='integrated_contract_', dir=ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.directory, self.serial = Path(self.temp.name), 0
        env = patch.dict(os.environ, {'INDEPENDENT_FIXTURE_KEY': 'fixture-not-provider-key'})
        env.start()
        self.addCleanup(env.stop)
        blocked = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network forbidden'))
        network = blocked.start()
        self.addCleanup(blocked.stop)
        self.addCleanup(network.assert_not_called)

    def pipeline(self, transport=None, cfg=None, **kwargs):
        self.serial += 1
        return IntegratedPathPipeline('llm', config() if cfg is None else cfg,
            self.directory / ('ledger%d.sqlite' % self.serial), transport=transport, **kwargs)

    def transport(self, changes=None, wrapper=None):
        def choose(url, headers, body, timeout):
            value = choices(body)
            value.update(changes or {})
            return provider(value, **(wrapper or {}))
        return Mock(side_effect=choose)

    def test_first_candidate_is_provider_choice_without_controlled_initial(self):
        transport = self.transport()
        request = make_envelope()
        saved = deepcopy(request)
        pipeline = self.pipeline(transport)
        response = pipeline.rank(request)
        self.assertIsNone(response['fallback_reason'])
        self.assertEqual(response['candidate_ids'], ['2', '1'])
        self.assertEqual(request, saved)
        self.assertEqual(transport.call_count, 1)
        self.assertTrue(response['metadata']['attempted'])
        self.assertFalse(response['metadata']['llm_live'])
        self.assertFalse(response['metadata']['controlled_candidate_selection'])
        body = transport.call_args.args[2]
        self.assertEqual(set(body['text']['format']['schema']['required']), CANDIDATE_FIELDS)

    def test_successful_multi_action_is_kept_and_deduplicated_in_candidate_input(self):
        request = candidate_history(True)
        transport = self.transport()
        pipeline = self.pipeline(transport)
        result = pipeline.rank(request)
        self.assertIsNone(result['fallback_reason'])
        model = json.loads(transport.call_args.args[2]['input'][0]['content'])['features']
        self.assertEqual(len(model['path_feedback']), 1)
        record = model['path_feedback'][0]
        self.assertEqual(len(record['action_schedule']), 2)
        self.assertEqual(record['result']['path_success'], True)
        self.assertTrue(record['result']['stability_success'])
        self.assertEqual(record['provenance']['trajectory_sha256'], '2' * 64)
        raw = model['episode_feedback'][0]['raw']
        self.assertNotIn('action', raw)
        self.assertEqual(raw['initial_force'], [1, 0, 0])
        self.assertEqual(raw['path_feedback_id'], record['feedback_id'])
        self.assertEqual(result['metadata']['used_path_feedback_ids'], [record['feedback_id']])
        self.assertEqual(result['metadata']['omitted_path_feedback_ids'], [])
        self.assertIn('path_extension', pipeline.decisions[-1]['request']['features']['episode_feedback'][0]['raw'])

    def test_failed_multi_action_is_kept_as_unknown_without_stability_claim(self):
        request = candidate_history(False)
        transport = self.transport()
        result = self.pipeline(transport).rank(request)
        self.assertIsNone(result['fallback_reason'])
        model = json.loads(transport.call_args.args[2]['input'][0]['content'])['features']
        item = model['path_feedback'][0]
        self.assertEqual(item['status'], 'UNKNOWN')
        self.assertIsNone(item['result']['stability_success'])
        self.assertFalse(item['result']['feasible'])
        self.assertIsNone(item['failure_cause'])
        self.assertEqual(model['constraint_references'][0]['source_verdict_scope'], 'measured_endpoint_predicate_only')
        self.assertFalse(result['metadata']['physics_verdict_reused'])

    def test_path_route_accepts_completed_failure_history_and_keeps_current_constraints(self):
        request = candidate_history(False)
        path, _ = path_envelope()
        # Rebind the historical request to the exact path task/state; all IDs
        # and provenance are reconstructed rather than blindly source-mapped.
        history = deepcopy(request['features']['path_feedback'])
        history[0]['constraint_references'] = []
        scope = deepcopy(path['features']['constraint_references'][0]['source_scope'])
        scope['action_schedule'] = history[0]['outcome']['scope']['action_schedule']
        history[0]['outcome']['scope'] = scope
        path['features']['path_feedback'] = history
        transport = self.transport()
        result = self.pipeline(transport).rank(path)
        self.assertIsNone(result['fallback_reason'])
        self.assertEqual(result['metadata']['path_skill']['second_axis'], '+Z')
        model = json.loads(transport.call_args.args[2]['input'][0]['content'])['features']
        self.assertEqual(len(model['path_feedback']), 1)
        self.assertGreaterEqual(len(model['constraint_references']), 1)
        self.assertEqual(set(transport.call_args.args[2]['text']['format']['schema']['required']),
                         PATH_FIELDS | {'used_path_feedback_ids'})

    def test_original_raw_extension_alone_is_not_silently_omitted(self):
        transport = self.transport()
        result = self.pipeline(transport).rank(candidate_with_path_feedback())
        self.assertIsNone(result['fallback_reason'])
        model = json.loads(transport.call_args.args[2]['input'][0]['content'])['features']
        self.assertEqual(len(model['path_feedback']), 1)
        self.assertEqual(model['path_feedback'][0]['trigger'], 'controlled_single_direction_failure_probe')

    def test_conflicting_duplicate_outcome_is_rejected_before_transport(self):
        request = candidate_history(True)
        request['features']['path_feedback'][0]['outcome']['trajectory_sha256'] = '3' * 64
        transport = self.transport()
        response = self.pipeline(transport).rank(request)
        self.assertEqual(response['fallback_reason'], 'invalid_request')
        transport.assert_not_called()

    def test_failed_path_cannot_claim_stability_pass_or_physical_feasibility(self):
        for field, value in [('stability_success', True), ('parts_fix', ['base']), ('feasible', True)]:
            with self.subTest(field=field):
                request = candidate_history(False)
                request['features']['path_feedback'][0]['outcome'][field] = value
                transport = self.transport()
                self.assertEqual(self.pipeline(transport).rank(request)['fallback_reason'], 'invalid_request')
                transport.assert_not_called()

    def test_unknown_outcome_field_and_invented_blocker_are_rejected(self):
        for where, key in [('outcome', 'blocker'), ('controller', 'native_order'), ('geometry', 'collision_cause')]:
            with self.subTest(where=where):
                request = candidate_history(False)
                outcome = request['features']['path_feedback'][0]['outcome']
                target = outcome if where == 'outcome' else outcome[where]
                target[key] = 'unmeasured'
                transport = self.transport()
                self.assertEqual(self.pipeline(transport).rank(request)['fallback_reason'], 'invalid_request')
                transport.assert_not_called()

    def test_outcome_scope_schedule_hash_and_nonfinite_guards(self):
        cases = [lambda o: o['scope'].update(part_move='ghost'),
                 lambda o: o['controller']['actions'][1].update(action=[1, 0, 0]),
                 lambda o: o.update(trajectory_file='../outside.json'),
                 lambda o: o.update(trajectory_sha256='invalid'),
                 lambda o: o['geometry'].update(first_leg_distance=float('nan')),
                 lambda o: o['controller'].update(reset_count=True),
                 lambda o: o['controller'].update(physics_steps=20, switch_count=1)]
        for change in cases:
            request = candidate_history(False)
            change(request['features']['path_feedback'][0]['outcome'])
            transport = self.transport()
            self.assertEqual(self.pipeline(transport).rank(request)['fallback_reason'], 'invalid_request')
            transport.assert_not_called()

    def test_termination_before_switch_is_honest_not_a_completed_turn(self):
        request = candidate_history(False)
        outcome = request['features']['path_feedback'][0]['outcome']
        outcome['controller'].update(physics_steps=20, switch_count=0)
        outcome['geometry']['noncollinear'] = False
        transport = self.transport()
        self.assertIsNone(self.pipeline(transport).rank(request)['fallback_reason'])
        model = json.loads(transport.call_args.args[2]['input'][0]['content'])['features']
        self.assertEqual(model['path_feedback'][0]['controller']['switch_count'], 0)

    def test_history_constraint_must_bind_to_same_task_event_and_path(self):
        changes = [lambda r: r.update(origin='frozen_previous_episode'),
                   lambda r: r['evidence_origin'].update(event_id='invented'),
                   lambda r: r['evidence_origin']['trajectory'].update(relative_path='paths/not-observed.json'),
                   lambda r: r['source_scope'].update(part_move='ghost')]
        for change in changes:
            request = candidate_history(False)
            change(request['features']['path_feedback'][0]['constraint_references'][0])
            transport = self.transport()
            self.assertEqual(self.pipeline(transport).rank(request)['fallback_reason'], 'invalid_request')
            transport.assert_not_called()

    def test_candidate_set_must_be_complete_and_worker_cannot_inject_host_cards(self):
        changes = [lambda e: e['request'].update(candidate_ids=['1']),
                   lambda e: e['request'].update(candidate_ids=['1', '1']),
                   lambda e: e['features'].update(frozen_skill_evidence=[]),
                   lambda e: e['features'].update(integrated_context={}),
                   lambda e: e['features'].update(native_order=['1', '2'])]
        for change in changes:
            request = make_envelope()
            change(request)
            transport = self.transport()
            self.assertEqual(self.pipeline(transport).rank(request)['fallback_reason'], 'invalid_request')
            transport.assert_not_called()

    def test_untrusted_service_direct_call_is_rejected_without_send(self):
        transport = self.transport()
        service = IntegratedRankService(config(), self.directory / 'direct.sqlite', transport)
        self.assertEqual(service.rank(make_envelope())['fallback_reason'], 'invalid_request')
        transport.assert_not_called()

    def test_invalid_response_ids_candidates_skills_and_citations_fallback(self):
        cases = [({'request_id': 'stale'}, 'stale_request_or_state'),
                 ({'state_id': 'stale'}, 'stale_request_or_state'),
                 ({'candidate_ids': ['1', '1']}, 'invalid_permutation'),
                 ({'candidate_ids': ['1']}, 'invalid_permutation'),
                 ({'candidate_ids': ['1', 'ghost']}, 'invalid_permutation'),
                 ({'skill_id': 'exec_generated_code'}, 'unregistered_skill'),
                 ({'repair_part_id': '1'}, 'unexpected_repair_argument'),
                 ({'used_constraint_ids': ['fabricated']}, 'invalid_constraint_id'),
                 ({'used_skill_evidence_ids': ['fabricated']}, 'invalid_skill_evidence_id'),
                 ({'used_path_feedback_ids': ['fabricated']}, 'invalid_path_feedback_id'),
                 ({'code': 'unrequested'}, 'invalid_integrated_response_fields')]
        for changed, expected in cases:
            with self.subTest(changed=changed):
                transport = self.transport(changed)
                result = self.pipeline(transport).rank(make_envelope())
                self.assertEqual(result['fallback_reason'], expected)
                self.assertEqual(result['candidate_ids'], ['1', '2'])
                self.assertNotIn('used_path_feedback_ids', result['metadata'])
                self.assertEqual(transport.call_count, 1)

    def test_invalid_path_choice_is_safe_fallback(self):
        for changed in ({'second_axis': '+X'}, {'skill_id': 'no_repair', 'second_axis': '+Z'}):
            transport = self.transport(changed)
            result = self.pipeline(transport).rank(path_envelope()[0])
            self.assertEqual(result['fallback_reason'], 'invalid_registered_path_skill_or_argument')
            self.assertNotIn('path_skill', result['metadata'])

    def test_refusal_incomplete_and_api_errors_are_not_valid_decisions(self):
        cases = [({'status': 'incomplete'}, 'llm_incomplete'),
                 ({'error': {'message': 'fixture'}}, 'api_error'),
                 ({'output': []}, 'invalid_response'),
                 ({'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'refusal', 'refusal': 'fixture'}]}]}, 'llm_refusal')]
        for wrapper, expected in cases:
            with self.subTest(expected=expected):
                result = self.pipeline(self.transport(wrapper=wrapper)).rank(make_envelope())
                self.assertEqual(result['fallback_reason'], expected)

    def test_duplicate_json_keys_and_malformed_are_rejected(self):
        for text in ('{"request_id":"x","request_id":"y"}', '{malformed'):
            raw = provider({})
            raw['output'][0]['content'][0]['text'] = text
            result = self.pipeline(Mock(return_value=raw)).rank(make_envelope())
            self.assertEqual(result['fallback_reason'], 'invalid_response')

    def test_existing_state_change_retry_skill_is_retained(self):
        transport = self.transport({'skill_id': 'retry_after_state_change_v1', 'repair_part_id': '2',
                                    'candidate_ids': ['1', '2']})
        result = self.pipeline(transport).rank(make_envelope(repair=True))
        self.assertIsNone(result['fallback_reason'])
        self.assertEqual(result['candidate_ids'], ['2', '1'])
        self.assertEqual(result['metadata']['repair_invocation']['physical_result'], 'PENDING')

    def test_one_shared_ledger_for_candidate_path_candidate(self):
        transport = self.transport()
        pipeline = self.pipeline(transport, cfg=config(max_calls=2))
        first = pipeline.rank(make_envelope('fixture:ordinary1'))
        second = pipeline.rank(path_envelope('fixture:path1')[0])
        third = pipeline.rank(make_envelope('fixture:ordinary2'))
        self.assertIsNone(first['fallback_reason'])
        self.assertIsNone(second['fallback_reason'])
        self.assertEqual(third['fallback_reason'], 'call_budget_exhausted')
        self.assertEqual(transport.call_count, 2)
        with closing(sqlite3.connect(self.directory / 'ledger1.sqlite')) as db:
            self.assertEqual(db.execute('SELECT SUM(reserved_call) FROM rank_requests').fetchone()[0], 2)

    def test_cost_cap_prevents_transport_and_no_hidden_retry(self):
        transport = self.transport()
        result = self.pipeline(transport, cfg=config(max_usd='0.000001')).rank(make_envelope())
        self.assertEqual(result['fallback_reason'], 'cost_budget_exhausted')
        transport.assert_not_called()
        timed = Mock(side_effect=TimeoutError('fixture'))
        pipeline = self.pipeline(timed)
        first = pipeline.rank(make_envelope())
        repeat = pipeline.rank(make_envelope())
        self.assertEqual(first['fallback_reason'], 'api_timeout')
        self.assertTrue(repeat['metadata']['cached'])
        self.assertEqual(timed.call_count, 1)

    def test_duplicate_request_uses_ledger_and_changed_feedback_conflicts(self):
        transport = self.transport()
        pipeline = self.pipeline(transport)
        request = candidate_history(False)
        self.assertIsNone(pipeline.rank(request)['fallback_reason'])
        cached = pipeline.rank(deepcopy(request))
        self.assertTrue(cached['metadata']['cached'])
        self.assertFalse(cached['metadata']['attempted'])
        request['features']['path_feedback'][0]['outcome']['geometry']['noncollinear'] = False
        conflict = pipeline.rank(request)
        self.assertEqual(conflict['fallback_reason'], 'request_id_conflict')
        self.assertEqual(transport.call_count, 1)

    def test_fixture_cannot_enable_api_and_does_not_select_for_live(self):
        with self.assertRaises(ValueError):
            IntegratedPathPipeline('fixture', config(), self.directory / 'bad.sqlite')
        transport = self.transport()
        pipeline = IntegratedPathPipeline('fixture', config(enabled=False), self.directory / 'fixture.sqlite', transport=transport)
        result = pipeline.rank(path_envelope()[0])
        self.assertTrue(result['metadata']['fixture'])
        self.assertEqual(result['metadata']['ranking_source'], 'integrated_offline_fixture')
        self.assertEqual(result['metadata']['path_skill']['second_axis'], '+Z')
        transport.assert_not_called()

    def test_frozen_constraints_are_visible_in_candidate_and_path_routes(self):
        _, records = path_envelope(source_name='other-source')
        path = self.directory / 'frozen.json'
        save_snapshot(records, path)
        transport = self.transport()
        pipeline = self.pipeline(transport, memory_path=path)
        for request in (make_envelope(), path_envelope()[0]):
            result = pipeline.rank(request)
            self.assertIsNone(result['fallback_reason'])
            model = json.loads(transport.call_args.args[2]['input'][0]['content'])['features']
            frozen = [r for r in model['constraint_references'] if r['origin'] == 'frozen_previous_episode']
            self.assertTrue(frozen)
            self.assertIsNone(frozen[0]['current_part_id_mapping'])
            self.assertNotIn('geometry_features', frozen[0]['source_scope']['task_context'])

    def test_existing_real_outcomes_are_read_only_compatible_not_new_physics(self):
        source = ROOT / 'runs/constraint_path_repair_20261018/live_source/path_outcomes.json'
        reuse = ROOT / 'runs/constraint_path_repair_20261018/live_reuse/path_outcomes.json'
        self.assertTrue(source.is_file())
        self.assertTrue(reuse.is_file())
        for path in (source, reuse):
            outcomes = json.loads(path.read_text(encoding='utf-8'))['outcomes']
            self.assertEqual(len(outcomes), 1)
            outcome = deepcopy(outcomes[0])
            if not outcome['path_success']:
                outcome.update(parts_fix=None, feasible=False)
            validate_outcome(outcome, outcome['scope']['task_context'])

    def test_measurements_survive_no_repair_without_completed_outcome(self):
        request = candidate_history(False)
        request['features']['constraint_history'] = request['features']['path_feedback'][0]['constraint_references']
        request['features']['path_feedback'] = []
        transport = self.transport()
        result = self.pipeline(transport).rank(request)
        self.assertIsNone(result['fallback_reason'])
        model = json.loads(transport.call_args.args[2]['input'][0]['content'])['features']
        self.assertEqual(model['path_feedback'], [])
        self.assertEqual(len(model['constraint_references']), 1)
        self.assertEqual(model['constraint_references'][0]['origin'], 'current_episode_history')
        self.assertTrue(result['metadata']['used_constraint_ids'])

    def test_current_path_reference_history_is_deduplicated_by_identity(self):
        request, _ = path_envelope()
        request['features']['constraint_history'] = [dict(deepcopy(r), origin='current_episode_history')
            for r in request['features']['constraint_references']]
        transport = self.transport()
        result = self.pipeline(transport).rank(request)
        self.assertIsNone(result['fallback_reason'])
        self.assertEqual(len(result['metadata']['offered_constraint_ids']),
                         len(request['features']['constraint_references']))

    def test_constraint_history_rejects_other_task_and_invented_claim(self):
        request, _ = path_envelope(source_name='other-task')
        foreign = [dict(deepcopy(r), origin='current_episode_history') for r in request['features']['constraint_references']]
        candidate = make_envelope()
        candidate['features']['constraint_history'] = foreign
        transport = self.transport()
        self.assertEqual(self.pipeline(transport).rank(candidate)['fallback_reason'], 'invalid_request')
        transport.assert_not_called()
        request, _ = path_envelope()
        request['features']['constraint_history'] = [dict(deepcopy(r), origin='current_episode_history') for r in request['features']['constraint_references']]
        request['features']['constraint_history'][0]['blocker'] = 'a'
        self.assertEqual(self.pipeline(transport).rank(request)['fallback_reason'], 'invalid_request')
        transport.assert_not_called()

    def test_actual_frozen_success_card_reaches_different_assembly_candidate(self):
        snapshot = ROOT / 'runs/constraint_path_repair_20261018/frozen_path_skills.json'
        old_run = ROOT / 'runs/prototype_completion_20261018/transfer_live'
        context = json.loads((old_run / 'context.json').read_text(encoding='utf-8'))
        cad = json.loads((old_run / 'features.json').read_text(encoding='utf-8'))
        from worker.independent_features import filter_features
        request = make_envelope('fixture:actual-cross-assembly')
        parts = sorted(context['initial_parts'])
        base = context['task']['base_part']
        request['request'].update(assembled_parts=parts, candidate_ids=[p for p in parts if p != base], base_part=base)
        request['features'].update(cad=filter_features(cad, parts), memory_context=context)
        transport = self.transport()
        result = self.pipeline(transport, skill_memory_path=snapshot).rank(request)
        self.assertIsNone(result['fallback_reason'])
        model = json.loads(transport.call_args.args[2]['input'][0]['content'])['features']
        self.assertEqual(len(model['frozen_skill_evidence']), 1)
        card = model['frozen_skill_evidence'][0]
        self.assertEqual(card['applicability'], 'cross_assembly_procedure_reference')
        self.assertEqual(card['source_scope']['task_context']['task']['assembly_id'], 'beam_assembly/original')
        self.assertEqual(card['target_context']['task']['assembly_id'], '07009')
        self.assertIsNone(card['current_part_id_mapping'])
        self.assertIsNone(card['transfer_semantics']['target_selected_schedule'])
        self.assertEqual(card['transfer_semantics']['target_path_verdict'], 'UNKNOWN')
        self.assertTrue(result['metadata']['used_skill_evidence_ids'])
        self.assertFalse(result['metadata']['physics_verdict_reused'])

    def test_actual_frozen_source_card_path_citation_does_not_force_source_axis(self):
        snapshot = ROOT / 'runs/constraint_path_repair_20261018/frozen_path_skills.json'
        transport = self.transport({'second_axis': '+Y'})
        result = self.pipeline(transport, skill_memory_path=snapshot).rank(path_envelope()[0])
        self.assertIsNone(result['fallback_reason'])
        self.assertTrue(result['metadata']['used_skill_evidence_ids'])
        self.assertFalse(result['metadata']['source_schedule_numeric_match'])
        self.assertTrue(result['metadata']['path_skill']['cross_task_schedule_match_is_not_physical_equivalence'])
        self.assertEqual(result['metadata']['path_skill']['second_axis'], '+Y')

    def test_actual_same_task_source_card_is_still_offered_on_path_route(self):
        snapshot = ROOT / 'runs/constraint_path_repair_20261018/frozen_path_skills.json'
        directory = ROOT / 'runs/constraint_path_repair_20261018/live_reuse/ipc/requests'
        requests = [json.loads(path.read_text(encoding='utf-8')) for path in directory.glob('*.json')]
        request = next(r for r in requests if r['features']['protocol'] == 'constraint_path_repair_v1')
        transport = self.transport()
        result = self.pipeline(transport, skill_memory_path=snapshot).rank(request)
        self.assertIsNone(result['fallback_reason'])
        self.assertTrue(result['metadata']['used_skill_evidence_ids'])
        self.assertTrue(result['metadata']['source_schedule_numeric_match'])
        self.assertFalse(result['metadata']['physics_verdict_reused'])

    def test_real_child_state_relabels_duplicate_history_without_losing_evidence(self):
        source = ROOT / 'runs/constraint_path_repair_20261018/live_source'
        requests = [json.loads(path.read_text(encoding='utf-8')) for path in (source / 'ipc/requests').glob('*.json')]
        request = next(r for r in requests if r['request']['assembled_parts'] == ['0', '2', '3', '6'])
        outcome = json.loads((source / 'path_outcomes.json').read_text(encoding='utf-8'))['outcomes'][0]
        from worker.constraint_memory import load_snapshot
        memory = FrozenConstraintMemory(load_snapshot(source / 'constraint_memory.json')['records'])
        ctx = request['features']['memory_context']
        past = memory.retrieve(ctx, outcome['scope']['assembled_parts'],
            sorted(p for p in outcome['scope']['assembled_parts'] if p != outcome['scope']['base_part']))['records']
        current = memory.retrieve(ctx, request['request']['assembled_parts'], request['request']['candidate_ids'])['records']
        request['features']['path_feedback'] = [{'outcome': deepcopy(outcome),
            'constraint_references': [dict(r, origin='current_episode_history') for r in past]}]
        request['features']['constraint_history'] = [dict(r, origin='current_episode_history') for r in current]
        self.assertEqual(past[0]['applicability'], 'same_task_state_reference')
        self.assertEqual(current[0]['applicability'], 'source_conditions_reference')
        transport = self.transport()
        result = self.pipeline(transport).rank(request)
        self.assertIsNone(result['fallback_reason'])
        model = json.loads(transport.call_args.args[2]['input'][0]['content'])['features']
        self.assertEqual(len(model['path_feedback']), 1)
        self.assertEqual(len(model['constraint_references']), len(current))
        self.assertTrue(all(r['applicability'] == 'source_conditions_reference' for r in model['constraint_references']))
        self.assertEqual(request['features']['path_feedback'][0]['constraint_references'][0]['applicability'],
                         'same_task_state_reference')


if __name__ == '__main__':
    unittest.main()
