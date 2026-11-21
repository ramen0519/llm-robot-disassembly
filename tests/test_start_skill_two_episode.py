"""No real key, provider, or physics: validate the two-episode opt-in launcher."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from tools import start_skill_two_episode as launcher


class TwoEpisodeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='twoepisode_launcher_',
                                               dir=launcher.ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'runs').mkdir()
        for name in launcher.REQUIRED_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture source', encoding='utf-8')
        self.output = self.root / 'runs' / launcher.RUN_NAME
        self.recorder = Mock(return_value={'exit_code': 0})
        self.key_mock = Mock()

    def invoke(self, authorize=False, source='windows_dpapi'):
        @contextlib.contextmanager
        def fake_key_environment(root):
            self.assertEqual(root, self.root)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'twoepisode-fixture-secret'}):
                yield source

        self.key_mock.side_effect = fake_key_environment
        with patch.object(launcher, 'key_environment', self.key_mock), \
             patch.object(launcher, 'preserved_artifacts', return_value={}), \
             patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('no_network')) as network, \
             patch('socket.create_connection', side_effect=AssertionError('no_network')) as socket, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = launcher.launch(authorize, self.root, self.recorder)
        self.assertNotIn('twoepisode-fixture-secret', stdout.getvalue())
        for file in self.output.glob('*.json'):
            self.assertNotIn('twoepisode-fixture-secret', file.read_text(encoding='utf-8'))
        network.assert_not_called()
        socket.assert_not_called()
        return result

    def test_check_only_does_not_read_key_or_create_output(self):
        with patch.object(launcher, 'validate_config',
                          side_effect=AssertionError('live validation must not read the key')) as validate:
            self.assertEqual(self.invoke(), 0)
        validate.assert_not_called()
        self.key_mock.assert_not_called()
        self.recorder.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_missing_key_cannot_claim_run_or_execute(self):
        self.assertEqual(self.invoke(True, 'missing'), 2)
        self.key_mock.assert_called_once()
        self.recorder.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_new_shared_six_call_budget_and_no_prior_library(self):
        self.assertEqual(self.invoke(True), 0)
        config = json.loads((self.output / 'approved_llm_config.json').read_text())
        self.assertEqual((config['max_calls'], config['max_usd']), (6, .01))
        self.assertEqual((config['model'], config['base_url']),
                         ('gpt-6-luna', 'https://api.openai.com/v1'))
        self.assertEqual((config['input_usd_per_million'], config['output_usd_per_million']),
                         (.10, .50))
        task = json.loads((self.output / 'task_start.json').read_text())
        self.assertFalse(task['prior_one_call_budget_reused'])
        self.assertFalse(task['prior_three_call_budget_reused'])
        self.assertIsNone(task['source_library'])
        self.assertEqual(task['max_calls_scope'], 'both episodes combined')
        self.assertEqual(task['episodes'], ['development_empty', 'reuse_frozen'])
        command = self.recorder.call_args.args[0]
        self.assertIn('--allow-live-api', command)
        self.assertIn('--live-development', command)
        self.assertIn('--prepared-run', command)
        self.assertNotIn('--frozen-library', command)
        self.assertEqual(command[command.index('--seed') + 1], '0')
        self.assertNotIn('twoepisode-fixture-secret', ' '.join(command))
        self.recorder.assert_called_once()

    def test_fixed_run_refuses_second_invocation_without_key_or_overwrite(self):
        self.assertEqual(self.invoke(True), 0)
        before = {p.name: p.read_bytes() for p in self.output.glob('*.json')}
        self.key_mock.reset_mock()
        self.assertEqual(self.invoke(True), 3)
        self.key_mock.assert_not_called()
        self.recorder.assert_called_once()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.glob('*.json')})

    def test_child_failure_is_not_retried(self):
        self.recorder.return_value = {'exit_code': 1}
        self.assertEqual(self.invoke(True), 1)
        self.assertEqual(self.invoke(True), 3)
        self.recorder.assert_called_once()

    def test_missing_child_exit_is_explicit_failure(self):
        self.recorder.return_value = {'exit_code': None}
        self.assertEqual(self.invoke(True), 127)
        self.recorder.assert_called_once()

    def test_missing_required_input_rejected_before_key(self):
        (self.root / 'tools/run_skill_experiment.py').unlink()
        with self.assertRaises(ValueError):
            self.invoke(True)
        self.key_mock.assert_not_called()
        self.recorder.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
