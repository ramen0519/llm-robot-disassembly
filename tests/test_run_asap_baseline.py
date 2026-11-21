import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_asap_baseline', ROOT / 'tools/run_asap_baseline.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class NoOpAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/beam_baseline.wsl.json').read_text())

    def test_exact_native_arguments(self):
        command = adapter.beam_command(self.config, ROOT / 'runs/test_noop')
        self.assertEqual(command[command.index('plan_sequence/run_seq_plan.py') + 1:], [
            '--dir', 'beam_assembly', '--id', 'original', '--planner', 'dfs',
            '--generator', 'heur-out', '--max-gripper', '2', '--base-part', '6',
            '--log-dir', self.config['project_wsl'] + '/runs/test_noop/native', '--early-term'])

    def test_reject_outside_runs_and_extensions(self):
        for path in (ROOT, ROOT / 'runs', ROOT / 'runs/../../outside'):
            with self.assertRaises(ValueError):
                adapter.run_path(path)
        self.config['extensions_enabled'] = True
        with self.assertRaises(ValueError):
            adapter.beam_command(self.config, ROOT / 'runs/test_noop')

    def test_existing_run_is_not_reused(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as directory:
            with self.assertRaises(FileExistsError):
                adapter.execute(self.config, Path(directory))


if __name__ == '__main__':
    unittest.main()
