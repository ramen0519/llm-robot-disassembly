"""Offline checks for the credential-session launcher; never contact a provider."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import start_luna_one_call as launcher


class OneCallLauncherTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(
            prefix='launcher_test_', dir=launcher.ROOT / 'runs' / launcher.RUN_NAME)
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.run = self.root / 'runs' / launcher.RUN_NAME
        self.run.mkdir(parents=True)
        self.config = self.run / 'approved_llm_config.json'
        self.config.write_bytes((launcher.ROOT / 'runs' / launcher.RUN_NAME /
                                 'approved_llm_config.json').read_bytes())
        for name in ('task_start.json', 'preserved_before.json'):
            (self.run / name).write_text('{}', encoding='utf-8')
        self.recorder = Mock(return_value={'exit_code': 0})

    def invoke(self, key='offline-test-placeholder'):
        with patch.dict(os.environ, {'OPENAI_API_KEY': key}), contextlib.redirect_stdout(io.StringIO()) as output:
            result = launcher.launch(self.root, self.recorder)
        self.assertNotIn('offline-test-placeholder', output.getvalue())
        return result

    def test_missing_key_does_not_claim_or_execute(self):
        self.assertEqual(self.invoke(''), 2)
        self.recorder.assert_not_called()
        self.assertFalse((self.run / 'integration_command').exists())

    def test_modified_budget_rejected_before_execution(self):
        self.config.write_text(self.config.read_text().replace('"max_calls": 1', '"max_calls": 2'))
        with self.assertRaises(ValueError):
            self.invoke()
        self.recorder.assert_not_called()

    def test_any_previous_start_refuses_retry(self):
        for name in ('integration_command', 'execution_start.json', 'implementation', 'batch_ledger.sqlite'):
            with self.subTest(name=name):
                marker = self.run / name
                marker.touch()
                self.assertEqual(self.invoke(), 3)
                marker.unlink()
        self.recorder.assert_not_called()

    def test_authorized_launch_uses_single_fixed_run(self):
        self.assertEqual(self.invoke(), 0)
        self.recorder.assert_called_once()
        command, cwd, output = self.recorder.call_args.args
        self.assertEqual(cwd, self.root)
        self.assertEqual(output, self.run / 'integration_command')
        self.assertEqual(command[command.index('--run-dir') + 1], str(self.run))
        self.assertEqual(command[command.index('--base-order') + 1], 'native')
        self.assertEqual(command[command.index('--memory') + 1], 'off')
        self.assertIn('--allow-live-api', command)
        self.assertNotIn('offline-test-placeholder', ' '.join(command))

    def test_failed_process_is_not_retried(self):
        self.recorder.return_value = {'exit_code': 1}
        self.assertEqual(self.invoke(), 1)
        self.recorder.assert_called_once()


if __name__ == '__main__':
    unittest.main()
