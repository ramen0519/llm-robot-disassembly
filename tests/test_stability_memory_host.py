"""Host opt-in checks; no API key, network, WSL or physics."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_explicit_action_experiment as runner


class StabilityMemoryHostTests(unittest.TestCase):
    def test_old_defaults_preserved(self):
        args = runner.parser().parse_args(['--run-dir', str(ROOT / 'runs/not_executed')])
        self.assertEqual(args.provider_contract, 'explicit_action_provider_v2')
        self.assertFalse(args.stability_diagnostics)

    def test_new_options_are_explicit(self):
        args = runner.parser().parse_args(['--run-dir', str(ROOT / 'runs/not_executed'),
            '--provider-contract', 'explicit_action_provider_v5',
            '--path-selection-policy', 'path_probe_round_v1', '--stability-diagnostics'])
        self.assertEqual(args.provider_contract, 'explicit_action_provider_v5')
        self.assertTrue(args.stability_diagnostics)

    def test_v5_without_round_rejected_before_output_or_key(self):
        args = runner.parser().parse_args(['--run-dir', str(ROOT / 'runs/not_executed'),
            '--provider-contract', 'explicit_action_provider_v5'])
        with self.assertRaisesRegex(ValueError, 'requires_explicit_path_probe_policy'):
            runner.execute(args)


if __name__ == '__main__':
    unittest.main()
