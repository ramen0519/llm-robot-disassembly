import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("run_recorded", Path(__file__).resolve().parents[1] / "tools/run_recorded.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecordedRunTests(unittest.TestCase):
    def test_failure_preserves_streams_and_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = module.run_recorded([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"], root, root / "run")
            self.assertEqual(result["exit_code"], 7)
            self.assertIn("out", (root / "run/stdout.log").read_text())
            self.assertIn("err", (root / "run/stderr.log").read_text())
            self.assertEqual(json.loads((root / "run/command.json").read_text()), result)

    def test_zero_is_not_plan_success_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = module.run_recorded([sys.executable, "-c", "pass"], root, root / "run")
            self.assertEqual(result["exit_code"], 0)
            self.assertIsNone(result["plan_success"])
            with self.assertRaises(FileExistsError):
                module.run_recorded([sys.executable, "-c", "pass"], root, root / "run")

    def test_launch_error_and_empty_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = module.run_recorded([str(root / "missing_executable")], root, root / "run")
            self.assertIsNone(result["exit_code"])
            self.assertIsNotNone(result["launch_error"])
            with self.assertRaises(ValueError):
                module.run_recorded([], root, root / "empty")


if __name__ == "__main__":
    unittest.main()
