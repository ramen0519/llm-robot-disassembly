"""Real Windows encryption with dummy strings only; no provider requests."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch
import warnings

ROOT = Path('C:/dev/disassembly_codex_ready_v2')
sys.path.insert(0, str(ROOT / 'tools'))
import local_api_key as keys
import start_skill_smoke as launcher
import run_skill_experiment as experiment
from types import SimpleNamespace

DUMMY = 'unit-test-placeholder-not-an-api-key'


class LocalKeyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dpapi_test_', dir=ROOT / 'runs/persistent_key_setup_20261017')
        self.root = Path(self.temp.name)
        self.assertTrue(self.root.is_relative_to(ROOT / 'runs/persistent_key_setup_20261017'))
        self.addCleanup(self.temp.cleanup)
        guard = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        self.network = guard.start()
        self.addCleanup(guard.stop)
        self.addCleanup(self.network.assert_not_called)

    def test_real_dpapi_roundtrip_no_plaintext_on_disk(self):
        keys.save_key(DUMMY, self.root)
        raw = keys.key_path(self.root).read_bytes()
        self.assertNotIn(DUMMY.encode(), raw)
        self.assertEqual(keys.load_key(self.root), DUMMY)
        self.assertEqual(list((self.root / '.secrets').iterdir()), [keys.key_path(self.root)])

    def test_new_process_reads_persisted_key_without_env(self):
        keys.save_key(DUMMY, self.root)
        child_env = dict(os.environ)
        child_env.pop('OPENAI_API_KEY', None)
        command = [sys.executable, '-B', '-c',
            "import sys; from pathlib import Path; sys.path.insert(0,'tools'); from local_api_key import load_key; assert load_key(Path(sys.argv[1])); print('LOADED_WITHOUT_ENV')", str(self.root)]
        child = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stdout.strip(), 'LOADED_WITHOUT_ENV')
        self.assertNotIn(DUMMY, child.stdout + child.stderr)

    def test_existing_encrypted_key_never_overwritten(self):
        keys.save_key(DUMMY, self.root)
        before = keys.key_path(self.root).read_bytes()
        with self.assertRaises(keys.KeyStoreError):
            keys.save_key('another-placeholder-api-key', self.root)
        self.assertEqual(keys.key_path(self.root).read_bytes(), before)

    def existing_fixture(self):
        path = keys.key_path(self.root)
        path.parent.mkdir()
        path.write_bytes(keys.MAGIC + b'fixture-old-ciphertext')
        return path

    def test_replace_publishes_only_verified_ciphertext_in_same_directory(self):
        path = self.existing_fixture()
        before = path.read_bytes()
        encrypted = b'fixture-new-ciphertext'
        replace_file = os.replace

        def publish(source, target):
            self.assertEqual(Path(source).parent, path.parent)
            self.assertNotEqual(Path(source), path)
            self.assertEqual(Path(target), path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(Path(source).read_bytes(), keys.MAGIC + encrypted)
            self.assertEqual(crypt.call_args_list, [call(DUMMY.encode(), False), call(encrypted, True)])
            replace_file(source, target)

        with patch.object(keys, '_crypt', side_effect=[encrypted, DUMMY.encode()]) as crypt, \
             patch.object(os, 'replace', side_effect=publish) as replace:
            keys.save_key(DUMMY, self.root, replace=True)
        replace.assert_called_once()
        self.assertEqual(path.read_bytes(), keys.MAGIC + encrypted)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_replace_failed_crypto_roundtrip_preserves_existing_file(self):
        path = self.existing_fixture()
        before = path.read_bytes()
        with patch.object(keys, '_crypt', side_effect=[b'new-ciphertext', b'different-key']), \
             self.assertRaises(keys.KeyStoreError):
            keys.save_key(DUMMY, self.root, replace=True)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_replace_write_or_publish_failure_preserves_existing_file(self):
        path = self.existing_fixture()
        before = path.read_bytes()
        for operation in ('fsync', 'replace'):
            with self.subTest(operation=operation):
                with patch.object(keys, '_crypt', side_effect=[b'fixture-ciphertext', DUMMY.encode()]), \
                     patch.object(os, operation, side_effect=OSError('fixture write failure')) as fail, \
                     self.assertRaises((OSError, keys.KeyStoreError)):
                    keys.save_key(DUMMY, self.root, replace=True)
                fail.assert_called_once()
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(list(path.parent.iterdir()), [path])

    def test_missing_file_returns_none_and_no_directory_creation(self):
        self.assertIsNone(keys.load_key(self.root))
        self.assertFalse((self.root / '.secrets').exists())

    def test_corrupted_or_plaintext_file_fails_closed(self):
        keys.key_path(self.root).parent.mkdir()
        for value in (DUMMY.encode(), keys.MAGIC + b'broken-ciphertext'):
            keys.key_path(self.root).write_bytes(value)
            with self.assertRaises(keys.KeyStoreError):
                keys.load_key(self.root)

    def test_environment_injected_temporarily_and_restored_on_exception(self):
        keys.save_key(DUMMY, self.root)
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            with self.assertRaises(RuntimeError):
                with keys.key_environment(self.root) as source:
                    self.assertEqual(source, 'windows_dpapi')
                    self.assertEqual(os.environ['OPENAI_API_KEY'], DUMMY)
                    raise RuntimeError('fixture')
            self.assertEqual(os.environ['OPENAI_API_KEY'], '')

    def test_process_key_has_priority_without_disk_read(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': DUMMY}), patch.object(keys, 'load_key') as load:
            with keys.key_environment(self.root) as source:
                self.assertEqual(source, 'process_environment')
            load.assert_not_called()

    def test_invalid_keys_rejected_without_file(self):
        for value in ('', 'short', DUMMY + '\n', '\uac00' * 20):
            with self.assertRaises(keys.KeyStoreError): keys.save_key(value, self.root)
        self.assertFalse(keys.key_path(self.root).exists())

    def test_excluded_path_rejected_before_io(self):
        with patch('pathlib.Path.lstat', side_effect=AssertionError('must_not_touch_path')):
            with self.assertRaises(ValueError): keys.key_path(Path('E:/excluded'))

    def test_registration_cli_no_echo_or_api(self):
        with patch.object(keys, 'ROOT', self.root), patch.object(keys, 'key_path', return_value=self.root/'new.dpapi'), \
             patch.object(keys, 'save_key') as save, patch.dict(os.environ, {'OPENAI_API_KEY': DUMMY}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(keys.main(['--register']), 0)
        self.assertNotIn(DUMMY, output.getvalue())
        save.assert_called_once_with(DUMMY)

    def test_registration_cli_preserves_existing_file_without_prompt(self):
        path = self.existing_fixture()
        with patch.object(keys, 'key_path', return_value=path), patch.object(keys, 'save_key') as save, \
             patch.object(keys.getpass, 'getpass') as prompt, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(keys.main(['--register']), 3)
        save.assert_not_called()
        prompt.assert_not_called()

    def test_replace_cli_uses_hidden_input_even_with_process_key(self):
        path = self.existing_fixture()
        stale = 'fixture-inherited-process-value'
        with patch.object(keys, 'key_path', return_value=path), patch.object(keys, 'save_key') as save, \
             patch.object(os, 'environ', {'OPENAI_API_KEY': stale}), patch.object(sys.stdin, 'isatty', return_value=True), \
             patch.object(keys.getpass, 'getpass', return_value=DUMMY) as prompt, \
             contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(output):
            self.assertEqual(keys.main(['--replace']), 0)
        prompt.assert_called_once()
        save.assert_called_once_with(DUMMY, replace=True)
        for value in (DUMMY, stale):
            self.assertNotIn(value, output.getvalue())

    def test_replace_cli_rejects_noninteractive_input_even_with_process_key(self):
        path = self.existing_fixture()
        with patch.object(keys, 'key_path', return_value=path), patch.object(keys, 'save_key') as save, \
             patch.object(os, 'environ', {'OPENAI_API_KEY': DUMMY}), patch.object(sys.stdin, 'isatty', return_value=False), \
             patch.object(keys.getpass, 'getpass') as prompt, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertNotEqual(keys.main(['--replace']), 0)
        prompt.assert_not_called()
        save.assert_not_called()
        self.assertNotIn(DUMMY, output.getvalue())

    def test_replace_cli_aborts_if_terminal_cannot_disable_echo(self):
        path = self.existing_fixture()

        def unprotected_prompt(prompt):
            warnings.warn('Cannot disable echo', keys.getpass.GetPassWarning)
            return DUMMY

        with patch.object(keys, 'key_path', return_value=path), patch.object(keys, 'save_key') as save, \
             patch.object(os, 'environ', {}), patch.object(sys.stdin, 'isatty', return_value=True), \
             patch.object(keys.getpass, 'getpass', side_effect=unprotected_prompt) as prompt, \
             contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(output):
            self.assertNotEqual(keys.main(['--replace']), 0)
        prompt.assert_called_once()
        save.assert_not_called()
        self.assertNotIn(DUMMY, output.getvalue())

    def test_registered_key_reaches_launcher_child_without_argv_leak(self):
        keys.save_key(DUMMY, self.root)
        recorder = Mock(side_effect=lambda command, cwd, output: self.record_child(command))
        output = self.root / 'runs/one_run'
        output.parent.mkdir()
        library = self.root / 'library'
        library.mkdir()
        for name in ('library.json', 'memory.json'):
            (library / name).write_text('{}')
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}), \
             patch.object(launcher, 'preserved_artifacts', return_value={}), \
             contextlib.redirect_stdout(io.StringIO()) as text:
            with keys.key_environment(self.root) as source:
                self.assertEqual(launcher._launch_ready(self.root, output, library, recorder, source), 0)
            self.assertEqual(os.environ['OPENAI_API_KEY'], '')
        self.assertNotIn(DUMMY, text.getvalue())
        for path in output.rglob('*'):
            if path.is_file(): self.assertNotIn(DUMMY.encode(), path.read_bytes())
        recorder.assert_called_once()

    def record_child(self, command):
        self.assertEqual(os.environ['OPENAI_API_KEY'], DUMMY)
        self.assertNotIn(DUMMY, ' '.join(command))
        return {'exit_code': 0}

    def test_direct_live_entrypoint_loads_registered_key_without_api(self):
        keys.save_key(DUMMY, self.root)
        def execute(args):
            self.assertEqual(os.environ['OPENAI_API_KEY'], DUMMY)
            return 0
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}), patch.object(experiment, 'ROOT', self.root), \
             patch.object(experiment, 'execute', side_effect=execute) as child:
            self.assertEqual(experiment.launch(SimpleNamespace(allow_live_api=True)), 0)
            self.assertEqual(os.environ['OPENAI_API_KEY'], '')
            child.assert_called_once()

    def test_direct_offline_entrypoint_never_reads_stored_key(self):
        with patch.object(keys, 'load_key', side_effect=AssertionError('no_key_needed')), \
             patch.object(experiment, 'execute', return_value=0):
            self.assertEqual(experiment.launch(SimpleNamespace(allow_live_api=False)), 0)


if __name__ == '__main__':
    unittest.main()
