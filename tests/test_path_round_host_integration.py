"""Host opt-in alignment only; no key, API, WSL or simulation."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_explicit_action_experiment as runner

class PathRoundHostIntegrationTests(unittest.TestCase):
    def test_legacy_defaults_unchanged(self):
        args = runner.parser().parse_args(['--run-dir', str(ROOT / 'runs/not_executed')])
        self.assertEqual(args.provider_contract, 'explicit_action_provider_v2')
        self.assertEqual(args.path_selection_policy, 'legacy_v1')

    def test_v4_pair_is_explicit(self):
        args = runner.parser().parse_args(['--run-dir', str(ROOT / 'runs/not_executed'),
            '--provider-contract', 'explicit_action_provider_v4',
            '--path-selection-policy', 'path_probe_round_v1'])
        self.assertEqual(args.provider_contract, 'explicit_action_provider_v4')
        self.assertEqual(args.path_selection_policy, 'path_probe_round_v1')

    def test_v4_without_worker_policy_rejected_before_output_or_key(self):
        args = runner.parser().parse_args(['--run-dir', str(ROOT / 'runs/not_executed'),
            '--provider-contract', 'explicit_action_provider_v4'])
        with self.assertRaisesRegex(ValueError, 'provider_v4_requires_explicit_path_probe_policy'):
            runner.execute(args)

    def test_worker_policy_without_v4_rejected_before_output_or_key(self):
        args = runner.parser().parse_args(['--run-dir', str(ROOT / 'runs/not_executed'),
            '--path-selection-policy', 'path_probe_round_v1'])
        with self.assertRaisesRegex(ValueError, 'provider_v4_requires_explicit_path_probe_policy'):
            runner.execute(args)

if __name__ == '__main__':
    unittest.main()
