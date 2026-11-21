"""Bounded runner policy checks: no live calls, keys, or actual physics."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_integrated_repair as runner


class IntegratedRunnerTests(unittest.TestCase):
    def args(self, *extra):
        return runner.parser().parse_args(['--run-dir', str(ROOT/'runs/integrated_repair_20261018/not_executed_test_run'), *extra])

    def test_controlled_selection_flags_are_not_available(self):
        for flag in ('--controlled-repair', '--candidate-mode'):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                self.args(flag)

    def test_only_explicit_live_authorization_and_config_enable_API(self):
        with self.assertRaisesRegex(ValueError, 'explicit_API_authorization'):
            runner.validate_arguments(self.args('--mode', 'llm'))
        with self.assertRaisesRegex(ValueError, 'explicit_API_authorization'):
            runner.validate_arguments(self.args('--allow-live-api'))

    def test_official_target_paths_are_accepted_without_new_write(self):
        for assembly in (runner.APPROVED_EXT4_ASSEMBLY,
                         '/opt/asap-research/integrated_repair_20261018/02628'):
            result = runner.validate_arguments(self.args('--assembly-dir', assembly, '--base-part', 'none'))
            self.assertEqual(result[1], assembly)
            self.assertIsNone(result[2])
            self.assertFalse(result[-1])
            self.assertEqual(result[5]['max_calls'], 0)

    def test_excluded_drive_is_rejected_before_filesystem_access(self):
        with patch.object(runner, 'safe') as filesystem:
            for path in ('E:/never', '/mnt/e/never', 'e:\\never', '/mnt/c/../e/never'):
                with self.assertRaises(ValueError):
                    runner.checked(path)
            filesystem.assert_not_called()

    def test_existing_run_is_not_overwritten(self):
        args = self.args()
        args.run_dir = ROOT/'runs/constraint_path_repair_20261018/live_source'
        with self.assertRaises(FileExistsError):
            runner.validate_arguments(args)

    def test_source_archive_covers_new_modules(self):
        paths = runner.source_hashes()
        for name in ('tools/run_integrated_repair.py', 'worker/integrated_path_hook.py',
                     'worker/run_integrated_repair.py', 'orchestrator/transfer_path_skills.py'):
            self.assertIn(name, paths)
        self.assertTrue(all(name.endswith('.py') for name in paths))
        self.assertFalse(any(name.endswith(('.key', '.dpapi', '.env')) for name in paths))


if __name__ == '__main__':
    unittest.main()
