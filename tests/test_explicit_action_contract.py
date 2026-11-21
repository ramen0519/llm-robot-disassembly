"""Offline protocol tests; no simulator, API, or historical artifact writes."""
from copy import deepcopy
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker.explicit_action_contract import (PROTOCOL, validate_request, validate_response,
    validate_matrix, action_id, action_scope, canonical_response, enumerate_actions)


def request():
    return {'schema_version': 1, 'protocol': PROTOCOL, 'run_id': 'fixture',
        'request_id': 'fixture:r1', 'state_id': 's1', 'remaining_parts': ['2', '1', 'base'],
        'candidate_ids': ['2', '1'], 'base_part': 'base', 'max_grippers': 2,
        'pose_catalog': [{'pose_id': 'p0', 'matrix': [[1, 0, 0, 0], [0, 1, 0, 0],
            [0, 0, 1, 0], [0, 0, 0, 1]]}], 'attempted_action_ids': [],
        'task_context': {'assembly_id': 'fixture', 'input_sha256': 'a'*64,
            'solver_settings': {'seed': 0, 'budget': 4, 'force_mag': 100}},
        'evidence_ids': [], 'episode_feedback': [], 'memory_references': [],
        'geometry': {'parts': {'1': {'extents': [1, 2, 3]}, '2': {'extents': [2, 3, 4]}}}}


def attach_attempts(r, choices):
    r['attempted_actions'] = [dict(action_id=action_id(r, choice),
        **{key: deepcopy(value) for key, value in choice.items()
           if key != 'evidence_ids'}) for choice in choices]
    r['attempted_action_ids'] = [row['action_id'] for row in r['attempted_actions']]
    return r


class ExplicitActionContractTests(unittest.TestCase):
    def test_readable_attempts_are_optional_and_do_not_change_action_identity(self):
        r = request()
        choice = canonical_response(r, None)['choice']
        original = action_id(r, choice)
        self.assertNotIn('attempted_actions', validate_request(r))
        attach_attempts(r, [choice])
        self.assertEqual(action_id(validate_request(r), choice), original)
        self.assertNotIn('evidence_ids', r['attempted_actions'][0])
        self.assertEqual(r['candidate_ids'], ['2', '1'])

    def test_readable_attempts_require_complete_ordered_unique_id_match(self):
        r = request()
        attach_attempts(r, list(itertools.islice(enumerate_actions(r), 2)))
        mutations = (
            lambda x: x['attempted_actions'].pop(),
            lambda x: x['attempted_actions'].reverse(),
            lambda x: x['attempted_actions'].append(deepcopy(x['attempted_actions'][0])),
            lambda x: x['attempted_action_ids'].pop(),
        )
        for mutate in mutations:
            changed = deepcopy(r)
            mutate(changed)
            with self.subTest(request=changed):
                with self.assertRaisesRegex(ValueError, 'attempted_actions_must_match_ids'):
                    validate_request(changed)

    def test_readable_attempts_reject_unknown_fields_and_embedded_evidence(self):
        r = request()
        attach_attempts(r, [canonical_response(r, None)['choice']])
        for key, value in [('evidence_ids', []), ('physics_failure_cause', 'collision')]:
            changed = deepcopy(r)
            changed['attempted_actions'][0][key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'attempted_action_fields'):
                    validate_request(changed)

    def test_readable_attempts_reject_wrong_task_remaining_pose_or_action(self):
        r = request()
        attach_attempts(r, [canonical_response(r, None)['choice']])
        mutations = (
            lambda x: x['task_context'].update(assembly_id='another-task'),
            lambda x: x.update(remaining_parts=['1', 'base'], candidate_ids=['1']),
            lambda x: x['pose_catalog'][0]['matrix'][0].__setitem__(3, 1),
            lambda x: x['attempted_actions'][0].update(direction_id='yp'),
        )
        for mutate in mutations:
            changed = deepcopy(r)
            mutate(changed)
            with self.subTest(request=changed):
                with self.assertRaisesRegex(ValueError, 'attempted_action_scope_mismatch'):
                    validate_request(changed)

    def test_readable_attempt_support_order_does_not_change_exact_identity(self):
        r = request()
        r.update(remaining_parts=['1', '2', '3', 'base'], candidate_ids=['1', '2', '3'],
                 max_grippers=3)
        choice = canonical_response(r, None)['choice']
        choice['support_ids'] = ['2', '3']
        attach_attempts(r, [choice])
        r['attempted_actions'][0]['support_ids'].reverse()
        validate_request(r)
        self.assertEqual(action_id(r, choice), r['attempted_actions'][0]['action_id'])

    def test_readable_attempts_leave_different_action_for_same_part_available(self):
        r = request()
        first = canonical_response(r, None)
        attach_attempts(r, [first['choice']])
        after = canonical_response(r, 'offline')
        self.assertEqual(after['choice']['part_id'], first['choice']['part_id'])
        self.assertNotEqual(after['action_id'], first['action_id'])
        self.assertEqual(after['candidate_order'], first['candidate_order'])
        with self.assertRaisesRegex(ValueError, 'action_already_attempted'):
            validate_response(r, first)

    def test_readable_attempts_do_not_inherit_episode_feedback_window_limit(self):
        r = request()
        attach_attempts(r, list(itertools.islice(enumerate_actions(r), 13)))
        self.assertEqual(len(validate_request(r)['attempted_actions']), 13)
        self.assertEqual(r['episode_feedback'], [])

    def test_readable_attempts_preserve_request_size_limit_and_reject_bad_types(self):
        r = request()
        for value in (None, {}, 'text', [None]):
            changed = deepcopy(r)
            changed['attempted_actions'] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_request(changed)
        attach_attempts(r, [canonical_response(r, None)['choice']])
        r['geometry']['padding'] = 'x' * 196608
        with self.assertRaisesRegex(ValueError, 'request_too_large'):
            validate_request(r)

    def test_terminal_two_candidates_use_same_contract(self):
        r = request()
        r.update(remaining_parts=['2', '1'], candidate_ids=['2', '1'], base_part=None)
        response = canonical_response(r, 'llm_disabled')
        self.assertEqual(response['candidate_order'], ['1', '2'])
        self.assertEqual(validate_response(r, response), response)

    def test_fixed_base_is_never_candidate_or_additional_support(self):
        r = request()
        for choice in enumerate_actions(r):
            self.assertNotEqual(choice['part_id'], 'base')
            self.assertNotIn('base', choice['support_ids'])

    def test_candidate_omission_rejected(self):
        r = request(); r['candidate_ids'] = ['1']
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_request(r)

    def test_candidate_duplicates_rejected(self):
        r = request(); r['candidate_ids'] = ['1', '1']
        with self.assertRaises(ValueError): validate_request(r)

    def test_hidden_order_rejected_recursively(self):
        r = request(); r['geometry']['nested'] = {'gnn_score': 1}
        with self.assertRaisesRegex(ValueError, 'hidden_priority'): validate_request(r)

    def test_unknown_request_fields_rejected(self):
        r = request(); r['instructions'] = 'prune all parts'
        with self.assertRaises(ValueError): validate_request(r)

    def test_matrix_rejects_nonfinite_scale_reflection_and_bad_bottom(self):
        for row, col, value in [(0, 0, float('nan')), (1, 1, 2), (0, 0, -1), (3, 0, 1)]:
            with self.subTest(row=row, col=col, value=value):
                matrix = request()['pose_catalog'][0]['matrix']; matrix[row][col] = value
                with self.assertRaises(ValueError): validate_matrix(matrix)

    def test_rigid_rotated_translated_pose_valid(self):
        validate_matrix([[0, -1, 0, 5], [1, 0, 0, 3], [0, 0, 1, -2], [0, 0, 0, 1]])

    def test_duplicate_pose_id_rejected(self):
        r = request(); r['pose_catalog'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate_pose'): validate_request(r)

    def test_all_force_axes_enumerated_without_feasibility_filter(self):
        choices = list(enumerate_actions(request()))
        self.assertEqual({c['direction_id'] for c in choices}, {'xp', 'xn', 'yp', 'yn', 'zp', 'zn'})
        self.assertEqual(len(choices), 120)  # 2 parts * 30 schedules * 2 support subsets.
        self.assertEqual(len({action_id(request(), c) for c in choices}), 120)

    def test_attempt_failure_only_skips_exact_action(self):
        r = request(); first = canonical_response(r, 'failure')
        r['attempted_action_ids'] = [first['action_id']]
        next_choice = canonical_response(r, 'failure')
        self.assertEqual(next_choice['choice']['part_id'], first['choice']['part_id'])
        self.assertNotEqual(next_choice['action_id'], first['action_id'])
        self.assertEqual(set(next_choice['candidate_order']), {'1', '2'})

    def test_exhaustion_does_not_drop_candidate_set(self):
        r = request(); r['attempted_action_ids'] = [action_id(r, c) for c in enumerate_actions(r)]
        response = canonical_response(r, 'disabled')
        self.assertIsNone(response['choice'])
        self.assertEqual(response['candidate_order'], ['1', '2'])
        self.assertTrue(response['metadata']['exhausted'])
        validate_response(r, response)

    def test_false_exhaustion_rejected(self):
        r = request(); response = canonical_response(r, None)
        response.update(choice=None, action_id=None)
        with self.assertRaisesRegex(ValueError, 'premature'): validate_response(r, response)

    def test_stale_response_rejected(self):
        r = request(); response = canonical_response(r, None); response['state_id'] = 'old'
        with self.assertRaisesRegex(ValueError, 'stale'): validate_response(r, response)

    def test_omitted_permutation_rejected(self):
        r = request(); response = canonical_response(r, None); response['candidate_order'] = ['1']
        with self.assertRaisesRegex(ValueError, 'permutation'): validate_response(r, response)

    def test_selected_part_must_be_first(self):
        r = request(); response = canonical_response(r, None); response['candidate_order'].reverse()
        with self.assertRaisesRegex(ValueError, 'first_candidate'): validate_response(r, response)

    def test_unknown_pose_and_skill_rejected(self):
        for field in ('pose_id', 'skill_id', 'direction_id'):
            r = request(); response = canonical_response(r, None); response['choice'][field] = 'made_up'
            with self.assertRaises(ValueError): validate_response(r, response)

    def test_malformed_choice_ids_report_value_error(self):
        for key in ('part_id', 'pose_id', 'skill_id', 'direction_id', 'second_direction_id'):
            r = request(); response = canonical_response(r, None); response['choice'][key] = []
            with self.assertRaises(ValueError): validate_response(r, response)

    def test_two_leg_collinear_and_opposite_rejected(self):
        for direction in ('xp', 'xn'):
            r = request(); response = canonical_response(r, None)
            response['choice'].update(skill_id='turn_force_two_leg_v1', second_direction_id=direction)
            with self.assertRaisesRegex(ValueError, 'noncollinear'): action_id(r, response['choice'])

    def test_single_force_second_direction_rejected(self):
        r = request(); response = canonical_response(r, None); response['choice']['second_direction_id'] = 'zp'
        with self.assertRaisesRegex(ValueError, 'no_second'): action_id(r, response['choice'])

    def test_support_moving_base_unknown_duplicate_over_budget_rejected(self):
        for support in (['1'], ['base'], ['unknown'], ['2', '2']):
            r = request(); response = canonical_response(r, None); response['choice']['support_ids'] = support
            with self.assertRaises(ValueError): action_id(r, response['choice'])
        r = request(); r['max_grippers'] = 1; response = canonical_response(r, None)
        response['choice']['support_ids'] = ['2']
        with self.assertRaises(ValueError): action_id(r, response['choice'])

    def test_unknown_citation_rejected(self):
        r = request(); response = canonical_response(r, None); response['choice']['evidence_ids'] = ['invented']
        with self.assertRaisesRegex(ValueError, 'unknown_evidence'): action_id(r, response['choice'])

    def test_supplied_evidence_must_be_present_not_just_id(self):
        r = request(); r['evidence_ids'] = ['invented']
        with self.assertRaisesRegex(ValueError, 'must_match'): validate_request(r)

    def test_action_identity_binds_physics_not_episode_identity(self):
        r = request(); choice = canonical_response(r, None)['choice']; original = action_id(r, choice)
        changed = deepcopy(r); changed.update(run_id='new', request_id='new:r', state_id='new:s')
        self.assertEqual(action_id(changed, choice), original)
        changed['task_context']['solver_settings']['seed'] = 1
        self.assertNotEqual(action_id(changed, choice), original)

    def test_action_identity_binds_pose_support_and_schedule(self):
        r = request(); choice = canonical_response(r, None)['choice']; original = action_id(r, choice)
        for key, value in [('support_ids', ['2']), ('direction_id', 'yp')]:
            changed = deepcopy(choice); changed[key] = value
            self.assertNotEqual(action_id(r, changed), original)
        r['pose_catalog'][0]['matrix'][0][3] = 2
        self.assertNotEqual(action_id(r, choice), original)

    def test_already_attempted_response_rejected(self):
        r = request(); response = canonical_response(r, None); r['attempted_action_ids'] = [response['action_id']]
        with self.assertRaisesRegex(ValueError, 'already_attempted'): validate_response(r, response)

    def test_inputs_not_mutated(self):
        r = request(); saved = deepcopy(r)
        canonical_response(r, 'offline')
        self.assertEqual(r, saved)


if __name__ == '__main__': unittest.main()
