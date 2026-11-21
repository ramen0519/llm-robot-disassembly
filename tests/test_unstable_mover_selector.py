"""Synthetic selector fixtures only: no API or physics execution."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.test_skill_program import request, choice, append_observation, program, rule
from orchestrator.skill_program import program_hash, program_schema, step_program, validate_program
from orchestrator.skill_validation import check_evidence
from worker.explicit_action_contract import validate_response
from worker.stability_diagnostics import StabilityObservation


def measured_request(unstable='b'):
    value = append_observation(request(), path=True, stability=False)
    diagnostic = StabilityObservation().data
    diagnostic.update(phase='dynamic_failure', check_success_returned=True,
        returned_success=False, initial_disconnected_parts=[], returned_parts=[unstable],
        returned_parts_semantics='upstream_representatives_after_optional_contact_component_grouping',
        dynamic_failure_parts=[dict(part_id=unstable, displacement=2, displacement_threshold=1,
                                   threshold_exceeded=True, lost_initial_contacts=[])])
    value['episode_feedback'][0]['stability_diagnostics'] = diagnostic
    return value


def mover_program():
    return program(rule(when=['last_path_passed', 'last_stability_failed', 'has_unstable_parts'],
        part='observed_unstable', direction='zp', second_direction='none', support='empty', rank=[]))


class UnstableMoverSelectorTests(unittest.TestCase):
    def setUp(self):
        for name in ('subprocess.Popen', 'urllib.request.OpenerDirector.open'):
            guard = patch(name, side_effect=AssertionError('unit_test_cannot_run_physics_or_API'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_schema_exposes_closed_relative_role_without_literal_ids(self):
        schema = program_schema()['properties']['rules']['items']['properties']['select']['properties']
        self.assertIn('observed_unstable', schema['part']['enum'])
        self.assertNotIn('observed_unstable', schema['support']['enum'])
        self.assertIn('contains_observed_unstable', schema['support']['enum'])
        policy = mover_program()
        validate_program(policy)
        policy['rules'][0]['select']['part'] = 'b'
        with self.assertRaisesRegex(ValueError, 'invalid_program_selector'):
            validate_program(policy)

    def test_mover_and_support_mean_different_actions_with_complete_candidates(self):
        req = measured_request()
        before = deepcopy(req)
        moved = step_program(mover_program(), req)
        support_policy = program(rule(when=['last_stability_failed', 'has_unstable_parts'],
            direction='same_as_last', support='contains_observed_unstable', rank=[]))
        supported = step_program(support_policy, req)
        self.assertEqual(moved['response']['choice']['part_id'], 'b')
        self.assertEqual(moved['response']['choice']['support_ids'], [])
        self.assertEqual(supported['response']['choice']['part_id'], 'a')
        self.assertEqual(supported['response']['choice']['support_ids'], ['b'])
        self.assertEqual(set(moved['response']['candidate_order']), set(req['candidate_ids']))
        self.assertFalse(moved['candidate_pruning'])
        self.assertFalse(moved['response']['metadata']['physics_bypass'])
        self.assertEqual(moved['physical_calls'], 0)
        self.assertEqual(req, before)
        validate_response(req, moved['response'])

    def test_missing_feedback_does_not_scan_or_choose_any_candidate(self):
        policy = program(rule(when=['always'], part='observed_unstable', rank=[]))
        result = step_program(policy, request())
        self.assertIsNone(result['response'])
        self.assertEqual(result['scanned'], 0)
        self.assertFalse(result['candidate_pruning'])

    def test_initial_disconnection_is_not_a_dynamic_mover_observation(self):
        req = measured_request()
        diagnostic = req['episode_feedback'][0]['stability_diagnostics']
        diagnostic.update(phase='initial_disconnected', initial_disconnected_parts=['b'],
            dynamic_failure_parts=[], returned_parts_semantics='initial_contact_graph_degree_zero')
        self.assertIsNone(step_program(mover_program(), req)['response'])

    def test_missing_invalid_stale_error_late_and_mock_observations_abstain(self):
        mutations = (
            lambda r: r['episode_feedback'][0].pop('stability_diagnostics'),
            lambda r: r['episode_feedback'][0]['stability_diagnostics'].update(extra_physics_calls=1),
            lambda r: r['episode_feedback'][0].update(remaining_parts=['a', 'base']),
            lambda r: r['episode_feedback'][0]['selected_pose_matrix'][0].__setitem__(3, 2),
            lambda r: (r['episode_feedback'][0].update(status='ERROR'),
                       r['path_stage_observations'][0].update(status='ERROR')),
            lambda r: (r['episode_feedback'][0].update(deadline_exceeded=True),
                       r['path_stage_observations'][0].update(deadline_exceeded=True)),
            lambda r: r['path_stage_observations'][0].update(backend='mock'),
        )
        for mutate in mutations:
            req = measured_request()
            mutate(req)
            with self.subTest(mutation=mutate):
                self.assertIsNone(step_program(mover_program(), req)['response'])

    def test_source_memory_cannot_supply_mover_id_without_local_observation(self):
        req = request()
        req['memory_references'] = [dict(evidence_id='source_memory', reference_only=True,
            current_part_id_mapping=None, hard_pruning=False, physics_reuse_allowed=False,
            choice=choice(), path_success=True, stability_success=False,
            stability_diagnostics=measured_request()['episode_feedback'][0]['stability_diagnostics'])]
        req['evidence_ids'] = ['source_memory']
        self.assertIsNone(step_program(mover_program(), req)['response'])

    def test_foreign_removed_base_or_previously_moved_part_is_not_a_new_mover(self):
        for identity in ('foreign_source_id', 'base', 'a'):
            with self.subTest(identity=identity):
                self.assertIsNone(step_program(mover_program(), measured_request(identity))['response'])

    def test_current_ids_rebind_without_program_edit(self):
        req = request()
        req.update(remaining_parts=['new_base', 'new_mover', 'new_unstable'],
            candidate_ids=['new_mover', 'new_unstable'], base_part='new_base')
        append_observation(req, choice(part_id='new_mover'), path=True, stability=False)
        req['episode_feedback'][0]['stability_diagnostics'] = measured_request('new_unstable')[
            'episode_feedback'][0]['stability_diagnostics']
        response = step_program(mover_program(), req)['response']
        self.assertEqual(response['choice']['part_id'], 'new_unstable')
        self.assertEqual(set(response['candidate_order']), {'new_mover', 'new_unstable'})

    def test_old_registered_path_program_hash_and_validation_replay_are_unchanged(self):
        # Read-only replay of old local artifacts, not a new physical test.
        root = Path(__file__).resolve().parents[1]
        path = root / 'runs/automatic_learning_cycle_20261019/live_01/library_for_next_episode.json'
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        self.assertTrue(snapshot['entries'])
        for entry in snapshot['entries']:
            self.assertEqual(program_hash(entry['candidate']['program']), entry['candidate']['program_hash'])
            for record in entry['validations']:
                self.assertEqual(check_evidence(record), record)


if __name__ == '__main__':
    unittest.main()
