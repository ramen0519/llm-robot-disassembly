"""Synthetic IR contracts and read-only history replay; no API or physics."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.test_skill_program import request, choice, append_observation, program, rule
from tests.test_unstable_mover_selector import measured_request, mover_program as dynamic_program
from orchestrator.skill_program import program_hash, program_schema, step_program, validate_program
from orchestrator.skill_validation import check_evidence
from worker.explicit_action_contract import validate_response
from worker.stability_diagnostics import StabilityObservation


def initial_request(parts=None, selected=None):
    value = append_observation(request(), selected, path=True, stability=False)
    parts = ['b'] if parts is None else parts
    diagnostic = StabilityObservation().data
    diagnostic.update(phase='initial_disconnected', check_success_returned=True,
        returned_success=False, initial_disconnected_parts=parts[:], returned_parts=parts[:],
        returned_parts_semantics='initial_contact_graph_degree_zero')
    value['episode_feedback'][0]['stability_diagnostics'] = diagnostic
    return value


def initial_program(**changes):
    selection = dict(part='observed_initial_disconnected', direction='same_as_last',
                     support='empty', rank=[])
    selection.update(changes)
    return program(rule(when=['last_initial_disconnected', 'has_initial_disconnected_parts'],
                        **selection))


class InitialContactSelectorTests(unittest.TestCase):
    def setUp(self):
        for name in ('subprocess.Popen', 'urllib.request.OpenerDirector.open'):
            guard = patch(name, side_effect=AssertionError('unit_test_cannot_run_physics_or_API'))
            guard.start()
            self.addCleanup(guard.stop)

    def assert_abstains(self, req):
        for policy in (initial_program(), program(rule(when=['always'],
                part='observed_initial_disconnected', rank=[])),
                program(rule(when=['always'], part='any',
                    support='contains_observed_initial_disconnected', rank=[]))):
            result = step_program(policy, req)
            self.assertIsNone(result['response'])
            self.assertEqual(result['scanned'], 0)
            self.assertFalse(result['candidate_pruning'])

    def test_schema_exposes_separate_roles_and_predicates_without_literals(self):
        schema = program_schema()['properties']['rules']['items']['properties']
        self.assertIn('last_initial_disconnected', schema['when']['items']['enum'])
        self.assertIn('has_initial_disconnected_parts', schema['when']['items']['enum'])
        selectors = schema['select']['properties']
        self.assertIn('observed_initial_disconnected', selectors['part']['enum'])
        self.assertIn('contains_observed_initial_disconnected', selectors['support']['enum'])
        self.assertNotIn('contains_observed_initial_disconnected', selectors['part']['enum'])
        validate_program(initial_program())
        with self.assertRaisesRegex(ValueError, 'invalid_program_selector'):
            validate_program(initial_program(part='b'))

    def test_current_initial_observation_selects_mover_without_pruning_or_verdict(self):
        req, policy = initial_request(), initial_program()
        before = deepcopy((req, policy))
        result = step_program(policy, req)
        self.assertTrue(result['triggered'])
        self.assertEqual(result['response']['choice']['part_id'], 'b')
        self.assertEqual(result['response']['choice']['support_ids'], [])
        self.assertEqual(result['response']['choice']['evidence_ids'], req['evidence_ids'])
        self.assertEqual(set(result['response']['candidate_order']), set(req['candidate_ids']))
        self.assertEqual(result['physical_calls'], 0)
        self.assertFalse(result['response']['metadata']['physics_bypass'])
        self.assertFalse(result['physics_verdict_reused'])
        validate_response(req, result['response'])
        self.assertEqual((req, policy), before)

    def test_mover_role_and_support_role_are_different(self):
        req = initial_request()
        moved = step_program(initial_program(), req)['response']['choice']
        held = step_program(initial_program(part='same_as_last',
            support='contains_observed_initial_disconnected'), req)['response']['choice']
        self.assertEqual((moved['part_id'], moved['support_ids']), ('b', []))
        self.assertEqual((held['part_id'], held['support_ids']), ('a', ['b']))

    def test_dynamic_and_initial_failure_roles_do_not_cross_activate(self):
        self.assert_abstains(measured_request())
        self.assertIsNone(step_program(dynamic_program(), initial_request())['response'])
        self.assertEqual(step_program(dynamic_program(), measured_request())[
            'response']['choice']['part_id'], 'b')

    def test_missing_observation_and_source_memory_only_abstain(self):
        self.assert_abstains(request())
        req = request()
        req['memory_references'] = [dict(evidence_id='source', reference_only=True,
            current_part_id_mapping=None, hard_pruning=False, physics_reuse_allowed=False,
            stability_diagnostics=initial_request()['episode_feedback'][0]['stability_diagnostics'])]
        req['evidence_ids'] = ['source']
        self.assert_abstains(req)

    def test_inconsistent_initial_diagnostics_are_rejected_as_a_whole(self):
        mutations = [
            dict(check_success_returned=False), dict(physics_steps=1),
            dict(returned_success=True), dict(returned_parts=['a']),
            dict(initial_disconnected_parts=[]), dict(initial_disconnected_parts=['b', 'b']),
            dict(returned_parts_semantics='not_observed'), dict(extra_physics_calls=1),
            dict(extra_geometry_queries=1), dict(instrumentation_errors=['SyntheticError']),
            dict(dynamic_failure_parts=[dict(part_id='b', displacement=2,
                displacement_threshold=1, threshold_exceeded=True, lost_initial_contacts=[])]),
        ]
        for mutation in mutations:
            req = initial_request()
            req['episode_feedback'][0]['stability_diagnostics'].update(mutation)
            with self.subTest(mutation=mutation):
                self.assert_abstains(req)

    def test_foreign_base_mover_and_prior_support_ids_reject_entire_role(self):
        for parts in (['foreign'], ['base'], ['a'], ['b', 'foreign'], ['b', 'base']):
            with self.subTest(parts=parts):
                self.assert_abstains(initial_request(parts))
        self.assert_abstains(initial_request(['b'], choice(support_ids=['b'])))

    def test_stale_state_pose_action_error_late_mock_and_unaccepted_fields(self):
        mutations = (
            lambda r: r['episode_feedback'][0].update(remaining_parts=['a', 'base']),
            lambda r: r['episode_feedback'][0]['selected_pose_matrix'][0].__setitem__(3, 2),
            lambda r: r['episode_feedback'][0]['choice'].update(part_id='b'),
            lambda r: r['episode_feedback'][0].update(accepted_for_transition=None),
            lambda r: (r['episode_feedback'][0].update(status='ERROR'),
                       r['path_stage_observations'][0].update(status='ERROR')),
            lambda r: (r['episode_feedback'][0].update(deadline_exceeded=True),
                       r['path_stage_observations'][0].update(deadline_exceeded=True)),
            lambda r: r['path_stage_observations'][0].update(backend='mock'),
            lambda r: r['path_stage_observations'][0].update(source_event_id='unbound'),
        )
        for mutate in mutations:
            req = initial_request()
            mutate(req)
            with self.subTest(mutation=mutate):
                self.assert_abstains(req)

    def test_nonfailure_stage_cannot_activate_initial_predicate(self):
        for updates in (dict(stability_success=True), dict(stability_success=None),
                        dict(accepted_for_transition=True)):
            req = initial_request()
            req['episode_feedback'][0].update(updates)
            with self.subTest(updates=updates):
                self.assert_abstains(req)

    def test_raw_return_contradiction_and_newer_pass_do_not_reuse_old_failure(self):
        req = initial_request()
        req['episode_feedback'][0]['raw_stability_parts'] = ['a']
        self.assert_abstains(req)
        req = initial_request()
        append_observation(req, choice(part_id='b'), path=True, stability=True,
                           status='PASS', accepted=True)
        self.assert_abstains(req)

    def test_new_current_ids_rebind_without_source_ids_or_program_change(self):
        req = request()
        req.update(remaining_parts=['new_mover', 'new_rest', 'new_base'],
                   candidate_ids=['new_mover', 'new_rest'], base_part='new_base')
        append_observation(req, choice(part_id='new_mover'), path=True, stability=False)
        req['episode_feedback'][0]['stability_diagnostics'] = initial_request(['new_rest'])[
            'episode_feedback'][0]['stability_diagnostics']
        result = step_program(initial_program(), req)
        self.assertEqual(result['response']['choice']['part_id'], 'new_rest')
        self.assertEqual(set(result['response']['candidate_order']), {'new_mover', 'new_rest'})

    def test_support_contains_means_at_least_one_and_preserves_gripper_budget(self):
        req = request()
        req.update(remaining_parts=['a', 'b', 'c', 'base'], candidate_ids=['a', 'b', 'c'])
        append_observation(req, path=True, stability=False)
        req['episode_feedback'][0]['stability_diagnostics'] = initial_request(['b', 'c'])[
            'episode_feedback'][0]['stability_diagnostics']
        policy = initial_program(part='same_as_last', support='contains_observed_initial_disconnected')
        held = step_program(policy, req)['response']
        self.assertEqual(held['choice']['support_ids'], ['b'])
        self.assertLessEqual(1 + len(held['choice']['support_ids']), req['max_grippers'])
        self.assertEqual(set(held['candidate_order']), {'a', 'b', 'c'})
        req['max_grippers'] = 1
        # Task-scoped action identity includes the gripper budget. Rebuild it.
        req.update(attempted_action_ids=[], attempted_actions=[], path_stage_observations=[],
                   episode_feedback=[], evidence_ids=[])
        append_observation(req, path=True, stability=False)
        req['episode_feedback'][0]['stability_diagnostics'] = initial_request(['b', 'c'])[
            'episode_feedback'][0]['stability_diagnostics']
        self.assertIsNone(step_program(policy, req)['response'])
        # A mover proposal remains possible despite too few supports for every observed body.
        self.assertIsNotNone(step_program(initial_program(), req)['response'])

    def test_registered_path_and_dynamic_library_hash_and_replay_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        snapshot = root / 'runs/stability_skill_learning_20261019/live_02/library_for_reuse.json'
        before = snapshot.read_bytes()
        entries = json.loads(before)['entries']
        self.assertEqual(len(entries), 2)
        for entry in entries:
            candidate = entry['candidate']
            self.assertEqual(program_hash(candidate['program']), candidate['program_hash'])
            for record in entry['validations']:
                self.assertEqual(check_evidence(record), record)
        self.assertEqual(snapshot.read_bytes(), before)

    def test_saved_actual_initial_contact_activates_ir_without_new_physics(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'runs/stability_skill_learning_20261019/live_02/next_episode/host_decisions.json'
        before = path.read_bytes()
        req = json.loads(before)[2]['request']
        result = step_program(initial_program(), req)
        self.assertTrue(result['triggered'])
        self.assertIn(result['response']['choice']['part_id'], ['1', '2', '3'])
        self.assertEqual(set(result['response']['candidate_order']), set(req['candidate_ids']))
        self.assertEqual(result['physical_calls'], 0)
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
