"""Authorization and first-episode failure gates; no API or physics calls."""
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from tools.run_skill_experiment import parser, validate_selection_mode, first_live_episode_ready


class LiveDevelopmentTests(unittest.TestCase):
    def args(self, **updates):
        values = dict(selector_mode='live', allow_live_api=True,
                      frozen_library=None, live_development=True)
        values.update(updates)
        return SimpleNamespace(**values)

    def test_explicit_live_development_starts_without_old_library(self):
        validate_selection_mode(self.args())
        args = parser().parse_args(['--run-dir', 'runs/new', '--selector-mode', 'live',
                                    '--allow-live-api', '--live-development'])
        validate_selection_mode(args)
        self.assertIsNone(args.frozen_library)

    def test_invalid_modes_rejected(self):
        for updates in ({'live_development': False}, {'allow_live_api': False},
                        {'frozen_library': 'old/library'}, {'selector_mode': 'scripted_test'},
                        {'selector_mode': 'scripted_test', 'allow_live_api': False}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_selection_mode(self.args(**updates))

    def test_existing_frozen_live_and_offline_modes_remain_valid(self):
        validate_selection_mode(self.args(live_development=False, frozen_library='old/library'))
        validate_selection_mode(self.args(live_development=False, selector_mode='scripted_test',
                                         allow_live_api=False))

    def successful(self):
        row = dict(plan_status='PASS', plan_success=True)
        metadata = [dict(attempted=True, llm_live=True, selected_by='llm',
                         skill_id='native_order_v1', llm_fallback_reason=None,
                         skill_error_type=None) for _ in range(3)]
        return row, metadata

    def test_successful_real_first_episode_can_proceed(self):
        self.assertTrue(first_live_episode_ready(*self.successful()))

    def test_failed_or_unknown_plan_cannot_start_second_episode(self):
        for status in ('UNKNOWN', 'ERROR'):
            row, metadata = self.successful()
            row.update(plan_status=status, plan_success=False)
            self.assertFalse(first_live_episode_ready(row, metadata))

    def test_fallback_cached_or_missing_live_response_cannot_proceed(self):
        for field, value in (('attempted', False), ('llm_live', False),
                ('selected_by', 'scripted_test'), ('skill_id', 'observed_positive_old'),
                ('llm_fallback_reason', 'call_budget_exhausted'), ('skill_error_type', 'ValueError')):
            row, metadata = self.successful()
            metadata[1][field] = value
            self.assertFalse(first_live_episode_ready(row, metadata))
        row, metadata = self.successful()
        self.assertFalse(first_live_episode_ready(row, metadata[:2]))


if __name__ == '__main__':
    unittest.main()
