"""Saved actual evidence replay and negative tests, never fresh physics/API."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import prepare_stability_transition as prep
from tests.test_prepare_stability_learning import saved_inputs

SOURCE = ROOT / 'runs/stability_support_review_20261105/live/episode'


class TransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.physicals, cls.considered = saved_inputs(SOURCE)

    def select(self, report=None, considered=None, physicals=None):
        return prep.select_transition(report or deepcopy(self.report), considered or deepcopy(self.considered),
                                      physicals or deepcopy(self.physicals))

    def test_actual_pair_over_capacity_and_no_causal_claim(self):
        result = self.select()
        self.assertEqual(len(result['considered']), 8)
        pair = result['selected']
        self.assertEqual((pair['failure_attempt_number'], pair['success_attempt_number']), (2, 3))
        self.assertEqual(pair['pair']['changed_choice_fields'], ['direction_id', 'support_ids', 'pose_matrix'])
        self.assertFalse(pair['pair']['causal_effect_established'])
        self.assertIsNone(pair['pair']['current_part_id_mapping'])
        row = result['considered'][1]
        self.assertEqual(row['observed_dynamic_parts'], ['1', '2', '3', '4'])
        self.assertEqual(row['available_support_budget'], 2)
        self.assertTrue(row['observed_count_exceeds_support_budget'])
        self.assertFalse(row['all_alternative_actions_impossible'])

    def test_legacy_all_supports_policy_unchanged(self):
        row = prep.classify_condition(self.report['attempts'][1], self.considered[1], self.physicals[1])
        self.assertFalse(row['eligible'])
        self.assertEqual(row['reason'], 'observed_dynamic_parts_exceed_declared_support_budget')

    def test_integrity_exclusion_never_revived(self):
        checks = deepcopy(self.considered)
        checks[1].update(eligible=False, reason='deadline_exceeded_observation_not_eligible')
        result = self.select(considered=checks)
        self.assertIsNone(result['selected'])
        self.assertEqual(result['pair_decisions'][0]['reason'], checks[1]['reason'])

    def test_held_body_or_incomplete_raw_cannot_be_observed_free(self):
        for mutation in ('held', 'moving', 'returned', 'instrumentation', 'instrumentation_missing'):
            with self.subTest(mutation=mutation):
                a, p = deepcopy(self.report['attempts'][1]), deepcopy(self.physicals[1])
                if mutation == 'held':
                    p['choice']['support_ids'] = ['0', '1']
                elif mutation == 'moving':
                    a['raw']['executed']['stability_moving_parts'] = ['1']
                elif mutation == 'returned':
                    a['raw']['raw_upstream']['parts_fall'] = ['0']
                elif mutation == 'instrumentation':
                    a['raw']['stability_diagnostics']['instrumentation_errors'] = ['failure']
                else:
                    a['raw']['stability_diagnostics'].pop('instrumentation_errors')
                self.assertFalse(prep.classify_transition_failure(a, self.considered[1], p)['eligible'])

    def test_path_failure_is_not_stability_failure(self):
        row = prep.classify_transition_failure(self.report['attempts'][0], self.considered[0], self.physicals[0])
        self.assertFalse(row['eligible'])

    def test_pass_without_real_gravity_check_does_not_qualify(self):
        report = deepcopy(self.report)
        report['attempts'][2]['raw']['stability_check_calls'] = 0
        result = self.select(report)
        self.assertIsNone(result['selected'])
        # The shared raw validator may reject the inconsistent checker count
        # before the more specific transition gate is reached.
        self.assertFalse(prep._checked_gravity_pass(report['attempts'][2]))
        self.assertGreater(result['pair_summary']['excluded_attempt_count'], 0)

    def test_changed_source_scope_or_late_pass_never_paired(self):
        for key, value in (('deadline_exceeded', True), ('accepted_for_transition', False)):
            report = deepcopy(self.report)
            report['attempts'][2][key] = value
            self.assertIsNone(self.select(report)['selected'])

    def test_missing_rows_and_truncated_denominator_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'input_count_mismatch'):
            prep.select_transition(self.report, self.considered[:-1], self.physicals)
        with patch.object(prep, 'summarize_recovery_pairs', return_value={'truncated': True}):
            with self.assertRaisesRegex(ValueError, 'truncated'):
                self.select()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='transition_checks_')
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name).absolute() / 'prepared'

    def test_full_binding_new_manifest_and_rotations_without_new_physics(self):
        with patch('subprocess.Popen', side_effect=AssertionError('no API or physics subprocess')):
            result = prep.prepare_transition(SOURCE, self.output, allow_test_root=True)
        self.assertEqual(result['status'], 'PREPARED')
        self.assertEqual(result['memory_record_count'], 8)
        self.assertEqual(result['selected_observed_pairs'], 1)
        self.assertEqual(result['independent_source_tasks'], 1)
        self.assertFalse(result['target_is_held_out'])
        self.assertEqual((result['new_api_calls'], result['new_physical_calls']), (0, 0))
        manifest = json.loads(Path(result['manifest_path']).read_text(encoding='utf-8'))
        self.assertEqual(manifest['python_failure_scope'], 'remaining_body_stability_failed')
        self.assertNotIn('synthesis_failure_scope', manifest)
        self.assertEqual([c['attempt_index'] for c in manifest['cases']], [2, 2, 2])
        self.assertEqual([c.get('pose_yaw_degrees', 0) for c in manifest['cases']], [0, 90, 180])
        self.assertEqual([c['phase'] for c in manifest['cases']], ['debug', 'admission', 'admission'])
        self.assertTrue(all(c['expected_diagnostic_phase'] == 'dynamic_failure' for c in manifest['cases']))
        for path, sha in result['original_artifact_sha256'].items():
            self.assertEqual(prep.evidence._sha(Path(path)), sha)

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        (self.output / 'keep').write_text('keep')
        with self.assertRaises(FileExistsError):
            prep.prepare_transition(SOURCE, self.output, allow_test_root=True)
        self.assertEqual((self.output / 'keep').read_text(), 'keep')

    def test_no_pair_never_creates_manifest(self):
        with patch.object(prep, 'select_transition', return_value=dict(
                considered=[], pair_summary={}, pair_decisions=[], selected=None)):
            result = prep.prepare_transition(SOURCE, self.output, allow_test_root=True)
        self.assertEqual(result['status'], 'NO_ELIGIBLE_OBSERVED_TRANSITION')
        self.assertIsNone(result['manifest_path'])
        self.assertFalse((self.output / 'manifest.json').exists())

    def test_full_integrity_failure_prevents_manifest(self):
        with patch.object(prep.evidence, 'prepare_episode_learning', side_effect=ValueError('raw_binding_mismatch')):
            with self.assertRaisesRegex(ValueError, 'raw_binding_mismatch'):
                prep.prepare_transition(SOURCE, self.output, allow_test_root=True)
        self.assertFalse(self.output.exists())

    def test_forbidden_path_rejected_as_string_before_read(self):
        for path in ('E:/never_access', '/mnt/e/never_access', 'runs/../outside'):
            with patch.object(prep.evidence, '_load', side_effect=AssertionError('no filesystem read')):
                with self.assertRaises(ValueError):
                    prep.prepare_transition(path, self.output)


class LearningTransportTests(unittest.TestCase):
    """Exercise the real CLI routing, stopping before transport/key access."""
    def test_lesson_routes_the_shared_transport_policy(self):
        import run_lesson_discovery as lesson
        from unittest.mock import Mock
        cfg = {'enabled': True, 'model': 'fixture'}
        policy = {'enabled': True, 'roles': {'learner': cfg}}
        pool = Mock()
        with patch.object(lesson, 'read', side_effect=[cfg, policy, {'context': {}}]), \
             patch.object(lesson, 'checked', side_effect=lambda p, *a: p), \
             patch('orchestrator.hierarchy_budget.HierarchyBudget', return_value=pool) as budget, \
             patch.object(lesson, 'extract', return_value={'status': 'CANDIDATE'}) as extract:
            self.assertEqual(lesson.main(['--run-dir', 'unused', '--mode', 'live', '--allow-live-api',
                '--packet', 'unused_packet', '--config', 'unused_config', '--ledger', 'unused_ledger',
                '--hierarchy-policy', 'unused_policy', '--hierarchy-ledger', 'unused_pool',
                '--transport-failure-policy', 'latch_on_error_v1']), 0)
        self.assertEqual(budget.call_args.kwargs['transport_failure_policy'], 'latch_on_error_v1')
        self.assertEqual(extract.call_args.kwargs['transport'], pool.transport.return_value)

    def test_python_routes_policy_before_any_key_or_transport(self):
        import run_python_skill_cycle as cycle
        from run_luna_experiment import write
        prepared = ROOT / 'runs/stability_transition_learning_20261105'
        # All files are new temporary test artifacts under the explicit workspace.
        with tempfile.TemporaryDirectory(prefix='transition_cli_test_', dir=ROOT / 'runs') as directory:
            folder = Path(directory)
            policy = json.loads((prepared / 'preflight_02/policy_DISABLED.json').read_text())
            policy['enabled'] = True
            write(folder / 'policy.json', policy)
            write(folder / 'config.json', policy['roles']['learner'])
            with patch('orchestrator.hierarchy_budget.HierarchyBudget', side_effect=RuntimeError('STOP_BEFORE_TRANSPORT')) as budget, \
                 patch('tools.local_api_key.key_environment', side_effect=AssertionError('No credential access')):
                with self.assertRaisesRegex(RuntimeError, 'STOP_BEFORE_TRANSPORT'):
                    cycle.main(['--run-dir', str(folder / 'preview'), '--mode', 'live', '--allow-live-api',
                        '--manifest', str(prepared / 'prepared/manifest.json'),
                        '--llm-config', str(folder / 'config.json'), '--ledger', str(folder / 'learner.sqlite'),
                        '--hierarchy-policy', str(folder / 'policy.json'), '--hierarchy-ledger', str(folder / 'pool.sqlite'),
                        '--transport-failure-policy', 'latch_on_error_v1'])
            self.assertEqual(budget.call_args.kwargs['transport_failure_policy'], 'latch_on_error_v1')


if __name__ == '__main__':
    unittest.main()
