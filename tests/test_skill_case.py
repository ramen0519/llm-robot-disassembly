"""Case boundary tests using copied preserved artifacts; no new physics/API.

Historical files provide realistic field shapes only. Calling a helper with
their copies is NOT a fresh physical experiment or evidence of a new recovery.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.skill_case import physical_context, request_after_trigger, is_reproduced_trigger
from orchestrator.skill_program import step_program
from worker.explicit_action_contract import action_id, validate_request


def fixtures():
    source = ROOT / 'runs/explicit_action_v2_live_20261018/beam/planner'
    report = json.loads((source / 'planner_report.json').read_text(encoding='utf-8'))
    packet = json.loads((source / 'attempt_000001/request.json').read_text(encoding='utf-8'))
    raw = json.loads((source / 'attempt_000001/result.json').read_text(encoding='utf-8'))
    identity = report['attempts'][0]['request_id']
    template = next(d['request'] for d in report['decisions'] if d['request']['request_id'] == identity)
    return deepcopy(template), deepcopy(raw), deepcopy(packet['choice']), deepcopy(packet['context'])


def build(template, raw, choice, **kwargs):
    return request_after_trigger(template, raw, choice, run_id='unit_case_not_physics',
                                 event_id='unit_case_event', **kwargs)


class SkillCaseTests(unittest.TestCase):
    def stability_fixture(self):
        source = ROOT / 'runs/automatic_learning_cycle_20261019/live_01/next_episode/planner'
        report = json.loads((source / 'planner_report.json').read_text(encoding='utf-8'))
        packet = json.loads((source / 'attempt_000002/request.json').read_text(encoding='utf-8'))
        raw = json.loads((source / 'attempt_000002/result.json').read_text(encoding='utf-8'))
        identity = report['attempts'][1]['request_id']
        template = next(d['request'] for d in report['decisions'] if d['request']['request_id'] == identity)
        selected = deepcopy(packet['choice'])
        selected['evidence_ids'] = []
        raw['chosen']['evidence_ids'] = []
        raw['scope']['selected_action']['evidence_ids'] = []
        return template, raw, selected

    def test_stability_observation_and_actual_checker_bound_without_new_physics(self):
        template, raw, selected = self.stability_fixture()
        current = build(template, raw, selected)
        self.assertFalse(current['episode_feedback'][0]['stability_success'])
        self.assertEqual(current['episode_feedback'][0]['stability_diagnostics']['phase'], 'dynamic_failure')
        self.assertTrue(is_reproduced_trigger(raw, 'stability_failed'))
        self.assertTrue(is_reproduced_trigger(raw, 'stability_failure'))
        self.assertFalse(is_reproduced_trigger(raw, 'path_failed'))

    def test_stability_scope_rejects_no_checker_wrong_mode_or_movers(self):
        template, raw, selected = self.stability_fixture()
        for mutation in (
            lambda r: r.update(stability_check_calls=0),
            lambda r: r.update(stability_mode='not_run'),
            lambda r: r['executed'].update(stability_moving_parts=[]),
        ):
            changed = deepcopy(raw)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'stability_checker'):
                build(template, changed, selected)

    def test_trigger_helper_rejects_error_timeout_mock_and_mismatched_stage(self):
        _, raw, _ = self.stability_fixture()
        for mutation in (
            lambda r: r.update(status='ERROR'),
            lambda r: r.update(elapsed_seconds=30.001),
            lambda r: r.update(backend='fixture_not_actual_physics'),
            lambda r: r.update(stability_check_calls=0),
            lambda r: r.update(path_check_calls=0),
            lambda r: r['raw_upstream'].update(stability_success=None),
        ):
            changed = deepcopy(raw)
            mutation(changed)
            with self.subTest(mutation=mutation):
                self.assertFalse(is_reproduced_trigger(changed, 'stability_failed'))
        with self.assertRaisesRegex(ValueError, 'unsupported_trigger_kind'):
            is_reproduced_trigger(raw, 'collision')

    def test_old_path_alias_still_accepts_observed_path_failure(self):
        _, raw, _, _ = fixtures()
        self.assertTrue(is_reproduced_trigger(raw, 'path_failure'))
        self.assertTrue(is_reproduced_trigger(raw, 'path_failed'))
        self.assertFalse(is_reproduced_trigger(raw, 'stability_failed'))

    def test_exact_scope_projection_is_independent_and_clears_past_evidence(self):
        template, raw, choice, _ = fixtures()
        before = deepcopy((template, raw, choice))
        result = build(template, raw, choice)
        self.assertEqual((template, raw, choice), before)
        self.assertEqual(result['evidence_ids'], ['unit_case_event'])
        self.assertEqual(result['memory_references'], [])
        self.assertEqual(len(result['episode_feedback']), 1)
        self.assertEqual(result['path_stage_observations'][0]['source_event_id'], 'unit_case_event')
        self.assertEqual(result['attempted_action_ids'], [action_id(result, choice)])
        validate_request(result)

    def test_raw_backend_choice_or_preservation_mismatch_rejected(self):
        template, raw, choice, _ = fixtures()
        for mutation in (
            lambda r: r.update(backend='mock'),
            lambda r: r['chosen'].update(direction_id='xn'),
            lambda r: r.update(input_files_unchanged=False),
            lambda r: r.update(source_files_unchanged=False),
            lambda r: r.update(task_context_input_and_run_conditions_checked=False),
            lambda r: r.update(protocol='fake'),
        ):
            changed = deepcopy(raw)
            mutation(changed)
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    build(template, changed, choice)

    def test_stale_physics_scope_cannot_be_relabelled_as_current(self):
        template, raw, choice, _ = fixtures()
        for mutation in (
            lambda r: r['scope']['task_context']['task'].update(assembly_id='different_assembly'),
            lambda r: r['scope'].update(base_part='1'),
            lambda r: r['scope'].update(assembled_parts=['0', '6']),
            lambda r: r['scope'].update(parts_removed=['1']),
            lambda r: r['scope']['pose'][0].__setitem__(3, 99),
            lambda r: r['scope']['support_policy'].update(max_grippers=99),
            lambda r: r['executed'].update(support_ids=[]),
            lambda r: r['input_hashes'].update({'0.obj': '0'*64}),
            lambda r: r['scope'].update(seed=99),
            lambda r: r['scope'].update(budget=999),
        ):
            changed = deepcopy(raw)
            mutation(changed)
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    build(template, changed, choice)

    def test_changed_request_task_does_not_reuse_old_raw_result(self):
        template, raw, choice, _ = fixtures()
        template['task_context']['task']['assembly_id'] = 'different_assembly'
        with self.assertRaises(ValueError):
            build(template, raw, choice)

    def test_nonfinite_negative_boolean_elapsed_is_rejected(self):
        template, raw, choice, _ = fixtures()
        for value in (float('nan'), float('inf'), -1, True, None, '1'):
            with self.subTest(value=value):
                changed = deepcopy(raw)
                changed['elapsed_seconds'] = value
                with self.assertRaisesRegex(ValueError, 'invalid_trigger_elapsed_seconds'):
                    build(template, changed, choice)

    def test_invalid_wall_allowance_is_rejected(self):
        template, raw, choice, _ = fixtures()
        for value in (float('nan'), float('inf'), -1, True, 0, 46):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'invalid_case_wall_seconds'):
                    build(template, raw, choice, wall_seconds=value)

    def test_late_failure_is_reported_and_cannot_trigger_recovery(self):
        template, raw, choice, _ = fixtures()
        raw['elapsed_seconds'] = 31
        req = build(template, raw, choice, wall_seconds=30)
        self.assertTrue(req['episode_feedback'][0]['deadline_exceeded'])
        self.assertTrue(req['path_stage_observations'][0]['deadline_exceeded'])
        policy = {'language': 'recovery_rules_v1', 'rules': [{
            'when': ['last_path_failed'], 'select': {'primitive': 'single_force_v1',
                'part': 'same_as_last', 'pose': 'same_as_last', 'direction': 'zp',
                'second_direction': 'none', 'support': 'same_as_last'}, 'rank': []}]}
        self.assertIsNone(step_program(policy, req)['response'])

    def test_stage_results_must_agree_with_raw_and_execution_order(self):
        template, raw, choice, _ = fixtures()
        for mutation in (
            lambda r: r.update(status='PASS'),
            lambda r: r['raw_upstream'].update(path_success=True),
            lambda r: r.update(stability_success=True),
            lambda r: r.update(path_check_calls=0),
        ):
            changed = deepcopy(raw)
            mutation(changed)
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    build(template, changed, choice)

    def test_physical_context_binds_task_and_solver_settings(self):
        template, _, choice, context = fixtures()
        for mutation in (
            lambda c: c['task_context']['task'].update(assembly_id='different_assembly'),
            lambda c: c.update(seed=9),
            lambda c: c.update(budget=1),
            lambda c: c.update(remaining_parts=['0', '6']),
            lambda c: c.update(parts_removed=['1']),
        ):
            changed = deepcopy(context)
            mutation(changed)
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    physical_context(changed, template, choice, event_id='unit_context')

    def test_physical_context_updates_redundant_action_fields_without_mutation(self):
        template, _, choice, context = fixtures()
        choice['direction_id'] = 'zn'
        before = deepcopy((template, choice, context))
        projected = physical_context(context, template, choice, event_id='unit_context')
        self.assertEqual((template, choice, context), before)
        self.assertEqual(projected['event_id'], 'unit_context')
        self.assertEqual(projected['action_id'], action_id(template, choice))
        self.assertEqual(projected['remaining_wall_seconds'], 30)
        self.assertEqual(projected['stability_timeout'], 30)
        self.assertNotIn('time_remaining_seconds', projected)

    def test_physical_context_rejects_incomplete_candidates_or_unknown_choice(self):
        template, _, choice, context = fixtures()
        incomplete = deepcopy(template)
        incomplete['candidate_ids'].pop()
        with self.assertRaises(ValueError):
            physical_context(context, incomplete, choice, event_id='unit_context')
        choice['part_id'] = 'not_a_current_part'
        with self.assertRaises(ValueError):
            physical_context(context, template, choice, event_id='unit_context')


if __name__ == '__main__':
    unittest.main()
