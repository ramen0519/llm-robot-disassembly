"""Opt-in selection integration; injected physics, not a CAD success test."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'tools'))
from test_explicit_action_planner import make_planner, result_for, response_for
from worker.explicit_action_planner import _selection_feedback
from run_explicit_action_experiment import parser as host_parser
from worker.run_explicit_actions import parser as worker_parser


class SelectionFixIntegrationTests(unittest.TestCase):
    def test_planner_callback_exception_uses_opt_in_policy(self):
        def unavailable(request):
            raise RuntimeError('injected_offline_failure')
        planner = make_planner(unavailable,
            lambda context, choice, output: result_for(context, choice, False),
            fallback_policy='path_stage_v1')
        planner.plan(budget=3, state_action_budget=3)
        report = planner.report()
        self.assertEqual([x['choice']['part_id'] for x in report['attempts']], ['a', 'b', 'c'])
        self.assertEqual(report['total_n_eval'], 3)
        self.assertEqual(report['fallback_policy'], 'path_stage_v1')
        self.assertFalse(report['success'])
        self.assertFalse(report['hard_pruning'])
        self.assertTrue(all(d['request']['candidate_ids'] == ['a', 'b', 'c'] for d in report['decisions']))

    def test_legacy_exception_order_remains_legacy(self):
        def unavailable(request):
            raise RuntimeError('injected_offline_failure')
        planner = make_planner(unavailable,
            lambda context, choice, output: result_for(context, choice, False))
        planner.plan(budget=3, state_action_budget=3)
        self.assertEqual([x['choice']['part_id'] for x in planner.report()['attempts']], ['a', 'a', 'a'])
        self.assertEqual(planner.report()['fallback_policy'], 'canonical_untried_explicit_action')

    def test_v3_selection_diagnostic_survives_feedback_projection(self):
        value = _selection_feedback({'metadata': {'provider_contract': 'explicit_action_provider_v3',
            'decision_validation_error': None, 'raw_response_recorded': False}}, {'attempted_action_ids': []})
        self.assertEqual(value['provider_contract'], 'explicit_action_provider_v3')
        self.assertIsNone(value['decision_validation_error'])

    def test_invalid_policy_is_rejected_before_search(self):
        with self.assertRaisesRegex(ValueError, 'unknown_fallback_policy'):
            make_planner(fallback_policy='unknown')

    def test_host_cli_keeps_legacy_and_accepts_explicit_opt_in(self):
        base = ['--run-dir', str(ROOT / 'runs/not_executed_fixture')]
        old = host_parser().parse_args(base)
        self.assertEqual((old.provider_contract, old.fallback_policy),
            ('explicit_action_provider_v2', 'legacy_v1'))
        new = host_parser().parse_args(base + ['--provider-contract', 'explicit_action_provider_v3',
                                             '--fallback-policy', 'path_stage_v1'])
        self.assertEqual((new.provider_contract, new.fallback_policy),
            ('explicit_action_provider_v3', 'path_stage_v1'))

    def test_worker_cli_accepts_policy_without_changing_default(self):
        base = ['--asap-root', '/opt/asap-research/ASAP', '--assembly-dir', '/fixture',
                '--output-dir', '/fixture/output', '--run-id', 'fixture']
        self.assertEqual(worker_parser().parse_args(base).fallback_policy, 'legacy_v1')
        self.assertEqual(worker_parser().parse_args(base + ['--fallback-policy', 'path_stage_v1']).fallback_policy,
                         'path_stage_v1')


if __name__ == '__main__':
    unittest.main()
