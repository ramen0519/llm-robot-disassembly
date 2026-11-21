"""Synthetic saved-decision contracts only; API=0 and actual physics=0.

Actual F2/diagnostic artifact validation is recorded separately. These small
fixtures intentionally fabricate metadata for guard tests, never real success.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
import execute_saved_path_choice as component


def fixture():
    context = {'schema_version': 2, 'usage': 'soft_ranking_only',
        'task': {'assembly_id': '02628', 'input_sha256': {'0.obj': 'a' * 64},
                 'source_commit': 'b' * 40, 'base_part': '2',
                 'solver_settings': {'seed': 0, 'budget': 20}, 'goal': {'type': 'native_full_disassembly'}},
        'initial_parts': ['0', '1', '2'], 'geometry_features': {'0': {}, '1': {}, '2': {}}}
    scope = {'task_context': context, 'assembled_parts': ['0', '1', '2'], 'part_move': '0',
        'parts_rest': ['1', '2'], 'parts_removed': [], 'base_part': '2', 'pose': None,
        'support_policy': {'kind': 'synthetic'}, 'action_schedule': [{'step': 0, 'action': [1, 0, 0]}],
        'physics': {'min_sep': .5}, 'solver_fingerprint': 'c' * 64, 'seed': 0, 'budget': 20,
        'rng_before': 'd' * 64, 'initial_q': [0] * 6, 'collector_version': 'synthetic_only',
        'frame': 'world_translation_xyz_euler_xyz'}
    request = {'run_id': 'synthetic-original', 'request_id': 'synthetic-original:path1',
        'state_id': hashlib.sha256(json.dumps(['0', '1', '2']).encode()).hexdigest(),
        'assembled_parts': ['0', '1', '2'], 'candidate_ids': ['0', '1'], 'base_part': '2'}
    features = {'protocol': 'constraint_path_repair_v1', 'memory_context': context, 'moving_part_id': '0',
        'failure_event_id': 'synthetic-failure', 'first_axis': '+X',
        'allowed_second_axes': ['+Y', '-Y', '+Z', '-Z'], 'switch_step': 50}
    original = {'schema_version': 1, 'kind': 'rank_request', 'request': request,
                'features': features, 'timeout_seconds': 90}
    diagnostic = deepcopy(original)
    diagnostic['request'].update(run_id='synthetic-diagnostic', request_id='synthetic-diagnostic:path')
    enriched = deepcopy(diagnostic)
    enriched['features']['integrated_context'] = {'route': 'path',
        'constraint_references': [{'record_id': 'constraint:synthetic'}],
        'frozen_skill_evidence': [{'skill_evidence_id': 'skill-evidence:synthetic'}],
        'path_feedback': []}
    choice = {'skill_id': 'turn_force_two_leg_v1', 'second_axis': '+Y', 'switch_step': 50,
              'used_constraint_ids': ['constraint:synthetic'], 'used_skill_evidence_ids': [],
              'used_path_feedback_ids': []}
    metadata = {'llm_live': True, 'attempted': True, 'decision_route': 'path',
        'ranking_source': 'integrated_llm', 'model_requested': 'gpt-6-luna', 'model_returned': 'gpt-6-luna',
        'path_skill': choice, 'used_constraint_ids': choice['used_constraint_ids'],
        'used_skill_evidence_ids': [], 'used_path_feedback_ids': []}
    response = {'schema_version': 1, 'kind': 'rank_response',
        'request_id': diagnostic['request']['request_id'], 'state_id': request['state_id'],
        'candidate_ids': ['1', '0'], 'fallback_reason': None, 'metadata': metadata}
    lineage = {'route': 'path', 'state_id_preserved': True, 'solver_not_resumed': True,
        'path_response_caused_stored_outcome': False, 'source_request_id': request['request_id'],
        'source_run_id': request['run_id'], 'original_state_id': request['state_id'],
        'diagnostic_request_id': diagnostic['request']['request_id']}
    path = [[0] * 6, [.1, 0, 0, 0, 0, 0]]
    event = {'schema_version': 1, 'kind': 'path_endpoint_measurement', 'event_id': 'synthetic-failure',
        'scope': scope, 'raw_success': False, 'trajectory': {'sha256': 'e' * 64, 'relative_path': 'paths/failure.json', 'states': 2},
        'terminal_q': path[-1], 'measurements': {'min_sep': .5, 'pairs': [
            {'rest_part_id': rest, 'moving_part_id': '0', 'd_rest_move': .1, 'd_move_rest': .2,
             'min_distance': .1, 'separation_unsatisfied': True} for rest in ['1', '2']],
            'contact_history': [], 'geometry_queries': 8, 'measurement_seconds': .001},
        'blocker': None, 'termination_reason': 'unknown'}
    trajectory = {'raw_success': False, 'path': path, 'controller': {
        'skill_id': 'native_single_force_probe', 'actions': scope['action_schedule'], 'reset_count': 1,
        'switch_count': 0, 'intermediate_resets': 0, 'upstream_check_success_calls': 1}}
    return original, diagnostic, response, enriched, lineage, event, trajectory, context


class SavedChoiceContractTests(unittest.TestCase):
    def setUp(self):
        self.values = fixture()

    def choice(self):
        return component.validate_choice(*self.values[:5])

    def failure(self):
        original, _, _, _, _, event, trajectory, context = self.values
        return component.validate_failure(event, trajectory, original, context)

    def test_live_choice_keeps_model_axis_without_forcing_source_Z(self):
        choice = self.choice()
        self.assertEqual(choice['second_axis'], '+Y')
        self.assertEqual(choice['used_skill_evidence_ids'], [])
        self.failure()

    def test_every_allowed_second_axis_accepted(self):
        for axis in ['+Y', '-Y', '+Z', '-Z']:
            self.values[2]['metadata']['path_skill']['second_axis'] = axis
            self.assertEqual(self.choice()['second_axis'], axis)

    def test_no_repair_is_valid_without_physics(self):
        self.values[2]['metadata']['path_skill'].update(skill_id='no_repair', second_axis='none')
        self.assertEqual(self.choice()['skill_id'], 'no_repair')

    def test_fixture_or_unattempted_response_rejected(self):
        for change in ({'llm_live': False}, {'attempted': False}, {'ranking_source': 'integrated_offline_fixture'},
                       {'decision_route': 'candidate'}, {'model_requested': 'other'}, {'model_returned': None}):
            self.values = fixture()
            self.values[2]['metadata'].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'accepted_live_path_response'):
                self.choice()

    def test_stale_or_fallback_response_rejected(self):
        for change in ({'request_id': 'old'}, {'state_id': 'old'}, {'fallback_reason': 'api_error'},
                       {'candidate_ids': ['0']}, {'candidate_ids': ['0', '0']}, {'candidate_ids': ['0', 'unknown']}):
            self.values = fixture()
            self.values[2].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.choice()

    def test_lineage_cannot_claim_old_fixture_result_as_live(self):
        for change in ({'path_response_caused_stored_outcome': True}, {'solver_not_resumed': False},
                       {'source_request_id': 'other'}, {'original_state_id': 'other'}, {'route': 'candidate'}):
            self.values = fixture()
            self.values[4].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'lineage'):
                self.choice()

    def test_diagnostic_conditions_cannot_change(self):
        self.values[1]['features']['switch_step'] = 100
        with self.assertRaisesRegex(ValueError, 'changed_original_failure_conditions'):
            self.choice()

    def test_prepared_context_binds_original_request(self):
        self.values[3]['features']['moving_part_id'] = '1'
        with self.assertRaisesRegex(ValueError, 'prepared_request_not_bound'):
            self.choice()

    def test_invalid_skill_axis_switch_or_norepair_argument_rejected(self):
        for change in ({'skill_id': 'generate_code'}, {'second_axis': '+X'}, {'second_axis': 'none'},
                       {'switch_step': 51}, {'switch_step': True}, {'skill_id': 'no_repair', 'second_axis': '+Z'}):
            self.values = fixture()
            self.values[2]['metadata']['path_skill'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.choice()

    def test_unoffered_or_inconsistent_citations_rejected(self):
        for key in ('used_constraint_ids', 'used_skill_evidence_ids', 'used_path_feedback_ids'):
            self.values = fixture()
            self.values[2]['metadata']['path_skill'][key] = ['not-offered']
            self.values[2]['metadata'][key] = ['not-offered']
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unoffered_or_inconsistent'):
                self.choice()
        self.values = fixture()
        self.values[2]['metadata']['used_constraint_ids'] = []
        with self.assertRaisesRegex(ValueError, 'unoffered_or_inconsistent'):
            self.choice()

    def test_choice_is_copied_not_aliased(self):
        value = self.choice()
        value['second_axis'] = '+Z'
        self.assertEqual(self.values[2]['metadata']['path_skill']['second_axis'], '+Y')

    def test_raw_success_and_event_success_cannot_be_relabelled_failure(self):
        self.values[6]['raw_success'] = True
        with self.assertRaisesRegex(ValueError, 'raw_single_force_trajectory'):
            self.failure()

    def test_trajectory_start_endpoint_count_and_reset_are_bound(self):
        changes = [lambda t: t['path'][0].__setitem__(0, 10),
                   lambda t: t['path'][-1].__setitem__(0, 10),
                   lambda t: t['path'].append([0] * 6),
                   lambda t: t['controller'].update(reset_count=2),
                   lambda t: t['controller'].update(switch_count=1)]
        for change in changes:
            self.values = deepcopy(fixture())
            # Break fixture aliases so edits test mismatched artifacts.
            self.values = tuple(deepcopy(value) for value in self.values)
            change(self.values[6])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.failure()

    def test_nonfinite_or_boolean_trajectory_coordinates_rejected(self):
        for bad in (float('nan'), float('inf'), True, '0'):
            self.values = tuple(deepcopy(value) for value in fixture())
            self.values[6]['path'][0][0] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.failure()

    def test_failure_evidence_requires_actual_predicate(self):
        for pair in self.values[5]['measurements']['pairs']:
            pair.update(d_rest_move=.6, d_move_rest=.7, min_distance=.6, separation_unsatisfied=False)
        with self.assertRaisesRegex(ValueError, 'unsatisfied_endpoint_predicate'):
            self.failure()

    def test_state_id_and_complete_candidates_required(self):
        for change in ({'state_id': 'unknown'}, {'candidate_ids': ['0']}):
            self.values = fixture()
            self.values[0]['request'].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'state_identity_mismatch'):
                self.failure()

    def test_fixed_target_fixture_cannot_transfer_part_ids(self):
        self.values[7]['task']['assembly_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'recorded_02628_part0'):
            self.failure()

    def test_forbidden_paths_rejected_before_IO(self):
        with patch.object(Path, 'lstat', side_effect=AssertionError('no filesystem access')):
            for value in ('E:/not-accessed', '/mnt/e/not-accessed', 'C:/dev/../not-accessed'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    component.local_path(value)

    def test_raw_trajectory_traversal_rejected(self):
        for value in ('../old.json', 'E:/not-accessed', '/tmp/file', 'paths//file'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                component.relative(ROOT / 'runs', value)


class SavedChoiceDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_saved_choice_', dir=ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = SimpleNamespace(source_run=root / 'source', decision_run=root / 'diagnostic',
                                    output_dir=root / 'new', asap_root=root / 'unused', assembly_dir=root / 'unused2')
        original, request, response, _, _, event, probe, _ = fixture()
        self.bundle = {'choice': response['metadata']['path_skill'], 'request': request,
            'response': response, 'source_failed_event': event, 'source_probe': probe, 'scope': event['scope'],
            'source_outcome': {'request_id': original['request']['request_id'],
                               'decision': {'second_axis': '+Z'}, 'path_success': False}, 'input_hashes': {}}

    def test_no_repair_dispatch_never_constructs_physics(self):
        self.bundle['choice'].update(skill_id='no_repair', second_axis='none')
        with patch.object(component, 'load_choice', return_value=self.bundle), \
             patch.object(component, 'execute_physics', side_effect=AssertionError('physics must not run')):
            result = component.run(self.args)
        self.assertEqual(result['status'], 'SKIPPED')
        self.assertEqual(result['path_check_calls'], 0)
        self.assertEqual(result['stability_check_calls'], 0)
        self.assertIsNone(result['stability_success'])
        self.assertFalse(result['full_DFS'])
        self.assertFalse(result['selected_plan_edge'])
        self.assertFalse(result['source_fixture_result_relabelled'])

    def test_selected_live_choice_reaches_physics_seam_unchanged(self):
        with patch.object(component, 'load_choice', return_value=self.bundle), \
             patch.object(component, 'execute_physics', side_effect=RuntimeError('synthetic-stop-before-physics')) as physics:
            result = component.run(self.args)
        self.assertEqual(physics.call_count, 1)
        self.assertEqual(physics.call_args.args[0]['choice']['second_axis'], '+Y')
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['path_check_calls'], 0)
        self.assertEqual(result['used_skill_evidence_ids'], [])
        self.assertFalse(result['source_skill_card_citation_claim'])
        self.assertEqual(result['source_fixture_second_axis'], '+Z')

    def test_input_contract_error_never_reaches_physics(self):
        with patch.object(component, 'load_choice', side_effect=ValueError('synthetic-invalid')), \
             patch.object(component, 'execute_physics', side_effect=AssertionError('must not execute')):
            result = component.run(self.args)
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['API_calls'], 0)
        self.assertEqual(result['path_check_calls'], 0)

    def test_existing_output_directory_not_overwritten(self):
        self.args.output_dir.mkdir()
        marker = self.args.output_dir / 'existing.txt'
        marker.write_text('preserve')
        with self.assertRaises(FileExistsError):
            component.run(self.args)
        self.assertEqual(marker.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
