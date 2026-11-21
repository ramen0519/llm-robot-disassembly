"""No provider or physics calls: validate explicit authorization and fixed caps."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from tools import start_skill_smoke as launcher


class SkillLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='skill_launcher_', dir=launcher.ROOT / 'runs/skill_loop_20261017')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dev = self.root / 'runs' / launcher.DEVELOPMENT
        self.library = self.dev / 'frozen_reuse'
        self.library.mkdir(parents=True)
        (self.dev / 'result.json').write_text(json.dumps({'status': 'PASS'}))
        source = self.root / 'orchestrator/skill_library.py'
        source.parent.mkdir()
        source.write_text('test source')
        (self.library / 'library.json').write_text(json.dumps({'source_code_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}))
        (self.library / 'memory.json').write_text('{}')
        self.output = self.root / 'runs' / launcher.RUN_NAME
        self.recorder = Mock(return_value={'exit_code': 0})

    def invoke(self, authorize=False, key='fixture-secret'):
        with patch.dict(os.environ, {'OPENAI_API_KEY': key}), \
             patch.object(launcher, 'preserved_artifacts', return_value={}), \
             patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('no_network')) as network, \
             contextlib.redirect_stdout(io.StringIO()) as text:
            result = launcher.launch(authorize, self.root, self.recorder)
        self.assertNotIn('fixture-secret', text.getvalue())
        network.assert_not_called()
        return result

    def test_default_is_read_only_no_key_or_execution_required(self):
        self.assertEqual(self.invoke(key=''), 0)
        self.assertFalse(self.output.exists())
        self.recorder.assert_not_called()

    def test_missing_key_does_not_claim(self):
        self.assertEqual(self.invoke(True, ''), 2)
        self.assertFalse(self.output.exists())
        self.recorder.assert_not_called()

    def test_explicit_new_authorization_uses_fixed_caps_and_frozen_library(self):
        self.assertEqual(self.invoke(True), 0)
        config = json.loads((self.output / 'approved_llm_config.json').read_text())
        self.assertEqual((config['max_calls'], config['max_usd']), (3, .01))
        command = self.recorder.call_args.args[0]
        self.assertIn('--allow-live-api', command)
        self.assertEqual(command[command.index('--frozen-library') + 1], str(self.library))
        self.assertNotIn('fixture-secret', ' '.join(command))
        self.assertEqual(self.invoke(True), 3)
        self.recorder.assert_called_once()

    def test_no_retry_on_child_failure(self):
        self.recorder.return_value = {'exit_code': 1}
        self.assertEqual(self.invoke(True), 1)
        self.assertEqual(self.invoke(True), 3)
        self.recorder.assert_called_once()

    def test_changed_skill_code_rejected(self):
        (self.root / 'orchestrator/skill_library.py').write_text('changed')
        with self.assertRaises(ValueError):
            self.invoke(True)
        self.recorder.assert_not_called()

    def test_failed_development_rejected(self):
        (self.dev / 'result.json').write_text(json.dumps({'status': 'ERROR'}))
        with self.assertRaises(ValueError):
            self.invoke(True)
        self.recorder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
