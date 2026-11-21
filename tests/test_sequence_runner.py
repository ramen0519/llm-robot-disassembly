"""Offline entrypoint guards: no process, API, key or simulator is started."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import run_sequence_native_experiment as runner


class SequenceRunnerTests(unittest.TestCase):
    def args(self, *extra):
        return runner.parser().parse_args(['--run-dir', str(ROOT/'runs/uncreated_sequence_runner_test'), *extra])

    def test_offline_defaults_never_read_config_or_key(self):
        with patch.object(runner, 'read', side_effect=AssertionError('unexpected read')):
            output, config, pins, knowledge = runner.validate(self.args())
        self.assertFalse(config['enabled']); self.assertEqual(config['max_calls'], 0)
        self.assertEqual(pins, {})

    def test_e_path_rejected_before_filesystem_lookup(self):
        args = self.args(); args.run_dir = Path('E:/do-not-access')
        with patch.object(Path, 'lstat', side_effect=AssertionError('forbidden filesystem lookup')):
            with self.assertRaises(ValueError):
                runner.validate(args)

    def test_outside_or_traversal_assembly_is_rejected(self):
        for path in ('/mnt/e/x', '/mnt/c/dev/../forbidden', '/etc/asap'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner.validate(self.args('--assembly-dir', path))

    def test_fixture_cannot_load_registered_key(self):
        with self.assertRaisesRegex(ValueError, 'offline_mode_must_not_load_credentials'):
            runner.validate(self.args('--mode', 'fixture', '--use-registered-key'))

    def test_live_requires_flag_and_configuration(self):
        for options in (['--mode','llm'], ['--mode','llm','--allow-live-api'], ['--allow-live-api']):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runner.validate(self.args(*options))

    def test_positive_bounds_and_outer_deadline(self):
        for options in (['--path-limit','0'], ['--budget','-1'], ['--response-timeout','121'],
                        ['--outer-timeout','120'], ['--max-gripper','1']):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runner.validate(self.args(*options))

    def test_existing_run_is_not_reused(self):
        args=self.args(); args.run_dir=ROOT/'runs/sequence_native_prepare_20261110'
        with self.assertRaisesRegex(ValueError,'new_run_directory_required'):
            runner.validate(args)


if __name__ == '__main__':
    unittest.main()
