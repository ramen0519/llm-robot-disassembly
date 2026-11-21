"""Opt-in fallback ordering tests. No API, filesystem artifacts, or simulator."""
from copy import deepcopy
import ast
import hashlib
import itertools
import json
from pathlib import Path
import unittest

from tests.test_explicit_action_contract import request, attach_attempts
from worker.explicit_action_contract import (action_id, canonical_response,
    enumerate_actions, path_signature, validate_response)


POLICY = 'path_stage_v1'


def scoped_feedback(r, choice, path=True, stability=False):
    item = {'evidence_id': 'current:physical:1', 'status': 'UNKNOWN',
        'action_id': action_id(r, choice), 'choice': deepcopy(choice),
        'remaining_parts': deepcopy(r['remaining_parts']),
        'selected_pose_matrix': deepcopy(next(p['matrix'] for p in r['pose_catalog']
                                              if p['pose_id'] == choice['pose_id'])),
        'path_success': path, 'stability_success': stability,
        'accepted_for_transition': False, 'deadline_exceeded': False}
    r['episode_feedback'] = [item]
    r['evidence_ids'] = [item['evidence_id']]
    return item


class ExplicitActionNoveltyTests(unittest.TestCase):
    def test_default_response_and_enumeration_remain_legacy(self):
        r = request()
        first = canonical_response(r, 'offline')
        self.assertEqual(first, canonical_response(r, 'offline', policy='legacy_v1'))
        attach_attempts(r, [first['choice']])
        old = canonical_response(r, 'offline')
        self.assertEqual(old['choice']['support_ids'], ['2'])
        self.assertEqual(path_signature(r, old['choice']), path_signature(r, first['choice']))
        self.assertNotIn('fallback_policy', old['metadata'])
        self.assertEqual(list(enumerate_actions(r)),
                         list(enumerate_actions(r, policy='legacy_v1')))

    def test_failed_path_support_variants_are_deferred_not_pruned(self):
        r = request()
        first = canonical_response(r, None)['choice']
        attach_attempts(r, [first])
        scoped_feedback(r, first, path=False, stability=None)
        response = canonical_response(r, 'provider_error', policy=POLICY)
        self.assertNotEqual(path_signature(r, first), path_signature(r, response['choice']))
        self.assertEqual(response['choice']['part_id'], '2')
        self.assertEqual(response['metadata']['path_stage_selection'], 'untried_path')
        self.assertTrue(response['metadata']['path_novelty_available'])
        deferred = deepcopy(first)
        deferred['support_ids'] = ['2']
        choices = list(enumerate_actions(r, policy=POLICY))
        self.assertIn(deferred, choices)
        validate_response(r, response)

    def test_new_paths_cycle_parts_before_directions(self):
        r = request()
        choices = list(itertools.islice(enumerate_actions(r, policy=POLICY), 4))
        self.assertEqual([(c['part_id'], c['direction_id']) for c in choices],
                         [('1', 'xp'), ('2', 'xp'), ('1', 'xn'), ('2', 'xn')])
        self.assertTrue(all(not c['support_ids'] for c in choices))

    def test_action_identity_set_equal_with_and_without_history(self):
        for count in (0, 1, 13, 60, 119, 120):
            r = request()
            original = list(enumerate_actions(r))
            attach_attempts(r, original[:count])
            old = {action_id(r, c) for c in enumerate_actions(r)}
            new_list = list(enumerate_actions(r, policy=POLICY))
            new = [action_id(r, c) for c in new_list]
            with self.subTest(count=count):
                self.assertEqual(set(new), old)
                self.assertEqual(len(new), len(set(new)))
                self.assertEqual({action_id(r, c) for c in
                    enumerate_actions(r, include_attempted=True, policy=POLICY)},
                    {action_id(r, c) for c in original})

    def test_path_pass_stability_failure_gets_one_support_priority_slot(self):
        r = request()
        first = canonical_response(r, None)['choice']
        attach_attempts(r, [first])
        scoped_feedback(r, first)
        second = canonical_response(r, 'bad_response', policy=POLICY)
        self.assertEqual(second['choice']['support_ids'], ['2'])
        self.assertEqual(path_signature(r, second['choice']), path_signature(r, first))
        self.assertEqual(second['metadata']['path_stage_selection'],
                         'support_retry_after_path_success')
        self.assertFalse(second['metadata']['physics_bypass'])
        self.assertFalse(second['metadata']['llm_live'])
        self.assertTrue(second['metadata']['path_novelty_available'])
        attach_attempts(r, [first, second['choice']])
        scoped_feedback(r, second['choice'])
        third = canonical_response(r, 'bad_response', policy=POLICY)
        self.assertNotEqual(path_signature(r, third['choice']), path_signature(r, first))
        self.assertEqual(third['metadata']['path_stage_selection'], 'untried_path')

    def test_support_retry_keeps_remaining_supports_available_after_priority(self):
        r = request()
        r.update(remaining_parts=['1', '2', '3', 'base'],
                 candidate_ids=['1', '2', '3'], max_grippers=3)
        first = canonical_response(r, None)['choice']
        attach_attempts(r, [first])
        scoped_feedback(r, first)
        second = canonical_response(r, 'bad', policy=POLICY)['choice']
        attach_attempts(r, [first, second])
        scoped_feedback(r, second)
        third = canonical_response(r, 'bad', policy=POLICY)['choice']
        self.assertNotEqual(path_signature(r, third), path_signature(r, first))
        deferred = deepcopy(first)
        deferred['support_ids'] = ['2', '3']
        self.assertIn(deferred, list(enumerate_actions(r, policy=POLICY)))

    def test_feedback_outside_current_scope_cannot_grant_support_priority(self):
        mutations = (
            lambda item: item.update(path_success=None),
            lambda item: item.update(path_success=1),
            lambda item: item.update(stability_success=None),
            lambda item: item.update(stability_success=True),
            lambda item: item.update(deadline_exceeded=True),
            lambda item: item.pop('deadline_exceeded'),
            lambda item: item.update(accepted_for_transition=True),
            lambda item: item.update(status='ERROR'),
            lambda item: item.update(remaining_parts=['1', 'base']),
            lambda item: item['choice'].update(direction_id='yn'),
            lambda item: item['selected_pose_matrix'][0].__setitem__(3, 2),
            lambda item: item['selected_pose_matrix'][0].__setitem__(0, True),
            lambda item: item.update(action_id='action:' + 'a'*64),
        )
        for index, mutate in enumerate(mutations):
            r = request()
            first = canonical_response(r, None)['choice']
            attach_attempts(r, [first])
            mutate(scoped_feedback(r, first))
            with self.subTest(mutation=index):
                response = canonical_response(r, 'offline', policy=POLICY)
                self.assertEqual(response['metadata']['path_stage_selection'], 'untried_path')

    def test_duplicate_or_missing_physical_feedback_is_conservative(self):
        r = request()
        first = canonical_response(r, None)['choice']
        attach_attempts(r, [first])
        item = scoped_feedback(r, first)
        r['episode_feedback'].append(deepcopy(item))
        response = canonical_response(r, 'offline', policy=POLICY)
        self.assertEqual(response['metadata']['path_stage_selection'], 'untried_path')
        r['episode_feedback'], r['evidence_ids'] = [], []
        response = canonical_response(r, 'offline', policy=POLICY)
        self.assertEqual(response['metadata']['path_stage_selection'], 'untried_path')

    def test_old_success_does_not_override_latest_different_path_observation(self):
        r = request()
        choices = list(itertools.islice(enumerate_actions(r, policy=POLICY), 2))
        attach_attempts(r, choices)
        scoped_feedback(r, choices[0])
        response = canonical_response(r, 'offline', policy=POLICY)
        self.assertEqual(response['metadata']['path_stage_selection'], 'untried_path')
        self.assertNotEqual(path_signature(r, response['choice']), path_signature(r, choices[0]))

    def test_all_untried_paths_exhausted_still_returns_deferred_support(self):
        r = request()
        representatives = [choice for choice in enumerate_actions(r)
                           if not choice['support_ids']]
        attach_attempts(r, representatives)
        response = canonical_response(r, 'offline', policy=POLICY)
        self.assertIsNotNone(response['choice'])
        self.assertFalse(response['metadata']['path_novelty_available'])
        self.assertEqual(response['metadata']['path_stage_selection'], 'deferred_support')
        self.assertEqual(set(response['candidate_order']), set(r['candidate_ids']))

    def test_readable_history_not_limited_to_twelve_feedback_rows(self):
        r = request()
        representatives = [choice for choice in enumerate_actions(r)
                           if not choice['support_ids']][:13]
        attach_attempts(r, representatives)
        response = canonical_response(r, 'offline', policy=POLICY)
        self.assertNotIn(path_signature(r, response['choice']),
                         {path_signature(r, c) for c in representatives})

    def test_legacy_hash_only_history_falls_back_explicitly(self):
        r = request()
        r['attempted_action_ids'] = [canonical_response(r, None)['action_id']]
        old = canonical_response(r, 'offline')
        response = canonical_response(r, 'offline', policy=POLICY)
        self.assertEqual(response['choice'], old['choice'])
        self.assertFalse(response['metadata']['path_history_complete'])
        self.assertEqual(response['metadata']['path_stage_selection'],
                         'legacy_missing_readable_attempts')

    def test_signature_excludes_only_support_and_citation_not_physical_scope(self):
        r = request()
        first = canonical_response(r, None)['choice']
        signature = path_signature(r, first)
        other_support = deepcopy(first)
        other_support['support_ids'] = ['2']
        self.assertEqual(signature, path_signature(r, other_support))
        self.assertNotEqual(action_id(r, first), action_id(r, other_support))
        scoped_feedback(r, first)
        cited = deepcopy(first)
        cited['evidence_ids'] = list(r['evidence_ids'])
        self.assertEqual(signature, path_signature(r, cited))
        for mutate in (
            lambda req, choice: choice.update(direction_id='yn'),
            lambda req, choice: choice.update(part_id='2'),
            lambda req, choice: choice.update(skill_id='turn_force_two_leg_v1', second_direction_id='yp'),
            lambda req, choice: req['pose_catalog'][0]['matrix'][0].__setitem__(3, 2),
            lambda req, choice: req['task_context']['solver_settings'].update(seed=1),
            lambda req, choice: req.update(max_grippers=3),
        ):
            changed, chosen = deepcopy(r), deepcopy(first)
            mutate(changed, chosen)
            self.assertNotEqual(signature, path_signature(changed, chosen))

    def test_pose_aliases_preserve_action_identity_set(self):
        r = request()
        r['pose_catalog'].append(dict(pose_id='same_matrix_alias',
                                     matrix=deepcopy(r['pose_catalog'][0]['matrix'])))
        expected = {action_id(r, c) for c in enumerate_actions(r)}
        actual = [action_id(r, c) for c in enumerate_actions(r, policy=POLICY)]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))

    def test_exhaustion_preserves_candidates_and_existing_response_validation(self):
        r = request()
        attach_attempts(r, list(enumerate_actions(r)))
        response = canonical_response(r, 'offline', policy=POLICY)
        self.assertIsNone(response['choice'])
        self.assertEqual(response['metadata']['path_stage_selection'], 'exhausted')
        self.assertEqual(set(response['candidate_order']), set(r['candidate_ids']))
        validate_response(r, response)

    def test_valid_llm_same_failed_path_new_support_is_still_allowed(self):
        r = request()
        first = canonical_response(r, None)['choice']
        attach_attempts(r, [first])
        scoped_feedback(r, first, path=False, stability=None)
        response = canonical_response(r, None)
        response['metadata']['selection_source'] = 'llm'
        validate_response(r, response)
        self.assertEqual(path_signature(r, response['choice']), path_signature(r, first))

    def test_unknown_policy_rejected_and_request_unchanged(self):
        r = request()
        saved = deepcopy(r)
        for policy in ('made_up', None, [], 1):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, 'unknown_fallback_policy'):
                    canonical_response(r, 'offline', policy=policy)
        canonical_response(r, 'offline', policy=POLICY)
        list(enumerate_actions(r, policy=POLICY))
        self.assertEqual(r, saved)

    def test_worker_contract_keeps_python38_grammar(self):
        source = Path(__file__).resolve().parents[1] / 'worker/explicit_action_contract.py'
        ast.parse(source.read_text(encoding='utf-8'), feature_version=8)


class HistoricalNoveltyReadOnlyTests(unittest.TestCase):
    def test_recorded_01256_fallbacks_now_have_new_path_available(self):
        path = (Path(__file__).resolve().parents[1] /
            'runs/assembly_01256_learning_20261018/catalog_learn_01/planner/planner_report.json')
        if not path.is_file():
            self.skipTest('recorded 01256 diagnostic is not installed; no physics fixture is invented')
        before = path.read_bytes()
        report = json.loads(before)
        seen = 0
        replaced_repeated_path = 0
        support_retry = 0
        for row in report['decisions']:
            if row['decision_source'] != 'canonical_fallback':
                continue
            r = row['request']
            new = canonical_response(r, 'offline_historical_replay', policy=POLICY)
            validate_response(r, new)
            self.assertTrue(new['metadata']['path_novelty_available'])
            self.assertIn(new['metadata']['path_stage_selection'],
                          ('untried_path', 'support_retry_after_path_success'))
            support_retry += (new['metadata']['path_stage_selection'] ==
                              'support_retry_after_path_success')
            old_signature = path_signature(r, row['response']['choice'])
            previous_signatures = {path_signature(r, dict(evidence_ids=[], **{
                key: value for key, value in attempt.items() if key != 'action_id'}))
                for attempt in r['attempted_actions']}
            if old_signature in previous_signatures:
                self.assertNotEqual(old_signature, path_signature(r, new['choice']))
                replaced_repeated_path += 1
            self.assertEqual(set(new['candidate_order']), set(r['candidate_ids']))
            seen += 1
        self.assertEqual(seen, 40)
        self.assertEqual(replaced_repeated_path, 39)
        self.assertEqual(support_retry, 6)
        self.assertEqual(hashlib.sha256(before).hexdigest(),
                         hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
