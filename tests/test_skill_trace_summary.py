"""Read-only historical-trace projection tests, not new physical experiments."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.skill_trace_summary import summarize_recovery_pairs


def historical():
    return json.loads((ROOT / 'runs/explicit_action_v2_live_20261018/beam/planner/planner_report.json').read_text(encoding='utf-8'))


class SkillTraceSummaryTests(unittest.TestCase):
    def test_actual_beam_pair_is_factual_and_no_cross_remaining_pair(self):
        report = historical()
        before = deepcopy(report)
        result = summarize_recovery_pairs(report)
        self.assertEqual(report, before)
        self.assertEqual(result['emitted_pair_count'], 1)
        pair = result['pairs'][0]
        self.assertEqual(pair['failure_evidence_id'], report['attempts'][0]['event_id'])
        self.assertEqual(pair['success_evidence_id'], report['attempts'][5]['event_id'])
        self.assertEqual(pair['changed_choice_fields'], ['skill_id', 'direction_id', 'second_direction_id'])
        self.assertTrue(pair['same_state'])
        self.assertTrue(pair['same_part'])
        self.assertTrue(pair['same_pose'])
        self.assertEqual(pair['after_controller']['reported_switch_count'], 1)
        self.assertTrue(pair['reference_only'])
        self.assertIsNone(pair['current_part_id_mapping'])
        self.assertIsNone(pair['failure_cause'])
        self.assertIsNone(pair['prescription'])
        self.assertFalse(pair['generalization_established'])

    def test_mock_late_false_pass_or_unaccepted_success_excluded(self):
        for mutation in (
            lambda a: a['raw'].update(backend='mock'),
            lambda a: a.update(deadline_exceeded=True),
            lambda a: a.update(accepted_for_transition=False),
            lambda a: a['raw'].update(path_success=False),
            lambda a: a['raw'].update(stability_success=False),
            lambda a: a['raw'].update(feasible=False),
        ):
            report = historical()
            mutation(report['attempts'][5])
            with self.subTest(mutation=mutation):
                self.assertEqual(summarize_recovery_pairs(report)['emitted_pair_count'], 0)

    def test_failed_predecessor_must_be_nonlate_actual_stage_failure(self):
        for mutation in (
            lambda a: a.update(deadline_exceeded=True),
            lambda a: a['raw'].update(backend='mock'),
            lambda a: a.update(status='ERROR'),
            lambda a: a['raw'].update(path_success=None),
        ):
            report = historical()
            mutation(report['attempts'][0])
            with self.subTest(mutation=mutation):
                self.assertEqual(summarize_recovery_pairs(report)['emitted_pair_count'], 0)

    def test_different_task_input_or_remaining_never_pairs(self):
        for mutation in (
            lambda a: a['context']['task_context']['task'].update(assembly_id='another_task'),
            lambda a: a['raw']['input_hashes'].update({'0.obj': 'f'*64}),
            lambda a: a.update(remaining_parts=['0', '6']),
        ):
            report = historical()
            mutation(report['attempts'][5])
            with self.subTest(mutation=mutation):
                self.assertEqual(summarize_recovery_pairs(report)['emitted_pair_count'], 0)

    def test_truncation_is_chronological_counted_and_never_silent(self):
        base = historical()
        failure, success = base['attempts'][0], base['attempts'][5]
        attempts = []
        for index in range(20):
            row = deepcopy(failure)
            row['event_id'] = 'synthetic_counter_failure:' + str(index)
            attempts.append(row)
        attempts.append(deepcopy(success))
        result = summarize_recovery_pairs({'attempts': attempts})
        self.assertEqual(result['emitted_pair_count'], 16)
        self.assertEqual(result['total_pair_count'], 20)
        self.assertEqual(result['omitted_pair_count'], 4)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['pairs'][0]['failure_attempt_index'], 0)
        self.assertEqual(result['pairs'][-1]['failure_attempt_index'], 15)

    def test_controller_switch_not_inferred_from_requested_primitive(self):
        report = historical()
        report['attempts'][5]['raw']['controller']['switch_count'] = 0
        report['attempts'][5]['raw']['controller']['actions'] = report['attempts'][5]['raw']['controller']['actions'][:1]
        pair = summarize_recovery_pairs(report)['pairs'][0]
        self.assertEqual(pair['after']['skill_id'], 'turn_force_two_leg_v1')
        self.assertEqual(pair['after_controller']['reported_switch_count'], 0)
        self.assertEqual(len(pair['after_controller']['reported_force_actions']), 1)

    def test_pose_change_is_explicit_and_not_same_pose(self):
        report = historical()
        success = report['attempts'][5]
        changed_pose = deepcopy(success['context']['pose_matrix'])
        changed_pose[0][3] = 2
        success['context']['pose_matrix'] = deepcopy(changed_pose)
        success['context']['pose_catalog'][0]['matrix'] = deepcopy(changed_pose)
        success['raw']['pose'] = deepcopy(changed_pose)
        success['raw']['scope']['pose'] = deepcopy(changed_pose)
        success['raw']['executed']['pose'] = deepcopy(changed_pose)
        pair = summarize_recovery_pairs(report)['pairs'][0]
        self.assertFalse(pair['same_pose'])
        self.assertIn('pose_matrix', pair['changed_choice_fields'])

    def test_rng_and_remaining_deadline_differences_are_reported(self):
        report = historical()
        report['attempts'][5]['raw']['rng_before'] = 'synthetic_different_rng'
        pair = summarize_recovery_pairs(report)['pairs'][0]
        self.assertFalse(pair['same_rng_state'])
        self.assertNotEqual(pair['before_execution']['remaining_wall_seconds'],
                            pair['after_execution']['remaining_wall_seconds'])
        self.assertFalse(pair['causal_effect_established'])

    def test_invalid_bounds_and_duplicate_events(self):
        for limit in (0, True, 17, 1.5):
            with self.assertRaises(ValueError):
                summarize_recovery_pairs(historical(), max_pairs=limit)
        report = historical()
        report['attempts'].insert(1, deepcopy(report['attempts'][0]))
        result = summarize_recovery_pairs(report)
        self.assertEqual(result['emitted_pair_count'], 1)
        self.assertTrue(any(row['reason'] == 'duplicate_event_identity' for row in result['excluded_attempts']))


if __name__ == '__main__':
    unittest.main()
