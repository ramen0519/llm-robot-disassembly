"""Assembly CLI/dispatch fixtures, not physical planning or upstream execution."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from run_assembly_hook import (COMMIT, MAX_SEED, build_native_argv, create_parser,
                               invoke_official, main, run_with_mode)


class AssemblyEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.asap = self.root / "asap"
        self.data = self.root / "run/data"
        (self.data / "fixture_assembly").mkdir(parents=True)
        self.native = self.root / "new-native"
        self.trace = self.root / "new-events.jsonl"
        self.argv = ["run_assembly_hook.py", "--asap-root", str(self.asap),
                     "--assembly-dir", str(self.data), "--assembly-id", "fixture_assembly",
                     "--native-log-dir", str(self.native), "--trace-file", str(self.trace),
                     "--run-id", "fixture", "--mode", "native"]

    def build(self, **kwargs):
        return build_native_argv(self.asap, self.data, "fixture_assembly", self.native, **kwargs)

    def test_default_native_argv_fixes_official_settings_without_a_base_part(self):
        argv = self.build()
        self.assertEqual(argv, [str(self.asap / "plan_sequence/run_seq_plan.py"),
            "--dir", str(self.data), "--id", "fixture_assembly", "--planner", "dfs",
            "--generator", "heur-out", "--num-proc", "1", "--seed", "0", "--budget", "100",
            "--max-gripper", "2", "--max-pose", "3", "--pose-reuse", "0", "--timeout", "180",
            "--log-dir", str(self.native), "--early-term", "--disable-save-sdf"])
        self.assertNotIn("--base-part", argv)

    def test_optional_base_and_nondefault_valid_settings_are_passed_exactly(self):
        argv = self.build(seed=7, base_part="10", max_gripper=3, budget=40,
                          max_pose=4, pose_reuse=2, timeout=90)
        for option, value in (("--seed", "7"), ("--base-part", "10"), ("--max-gripper", "3"),
                              ("--budget", "40"), ("--max-pose", "4"), ("--pose-reuse", "2"),
                              ("--timeout", "90")):
            self.assertEqual(argv[argv.index(option) + 1], value)
        self.assertIn("--disable-save-sdf", argv)

    def test_invalid_numeric_settings_fail_before_source_access(self):
        for values in ({"seed": -1}, {"seed": MAX_SEED + 1}, {"seed": True}, {"budget": 0},
                       {"max_gripper": 0}, {"max_pose": 0}, {"pose_reuse": -1},
                       {"pose_reuse": 4}, {"timeout": 0}, {"budget": 1.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.build(**values)

    def test_relative_data_and_path_traversal_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            build_native_argv(self.asap, Path("relative"), "fixture", self.native)
        for assembly in ("", ".", "..", "../outside", "sub/fixture", "sub\\fixture", "--bad", "C:escape"):
            with self.subTest(assembly=assembly), self.assertRaises(ValueError):
                build_native_argv(self.asap, self.data, assembly, self.native)

    def test_native_dispatch_never_installs_hooks(self):
        argv = self.build()
        with patch("run_assembly_hook.invoke_official") as invoke, patch("run_assembly_hook.installed_hooks") as hooks:
            run_with_mode(argv, "native", "noop", None)
        invoke.assert_called_once_with(argv)
        hooks.assert_not_called()

    def test_official_invocation_restores_argv_on_success_and_failure(self):
        argv = self.build()
        original = sys.argv
        def check(*args, **kwargs):
            self.assertIs(sys.argv, argv)
            self.assertEqual(args, (argv[0],))
            self.assertEqual(kwargs, {"run_name": "__main__"})
        with patch("run_assembly_hook.runpy.run_path", side_effect=check):
            invoke_official(argv)
        self.assertIs(sys.argv, original)
        with patch("run_assembly_hook.runpy.run_path", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                invoke_official(argv)
        self.assertIs(sys.argv, original)

    def test_cli_help_explains_native_mode_and_official_timeout(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as result:
            create_parser().parse_args(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("native,trace,rank", output.getvalue())
        self.assertIn("official planner timeout", output.getvalue())

    def test_memory_options_and_nonrank_rule_are_rejected(self):
        for extra in (["--memory-mode", "on"], ["--ranker", "reverse_native"], ["--seed", "-1"]):
            with self.subTest(extra=extra), patch.object(sys, "argv", self.argv + extra):
                with patch("run_assembly_hook.subprocess.check_output") as check, redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        main()
                self.assertEqual(result.exception.code, 2)
                check.assert_not_called()

    def test_native_main_preserves_metadata_and_source_checks_without_hooks(self):
        with patch.object(sys, "argv", self.argv), patch("run_assembly_hook.subprocess.check_output", return_value=COMMIT + "\n") as revision:
            with patch("run_assembly_hook.subprocess.run") as clean, patch("run_assembly_hook.invoke_official") as invoke:
                with patch("run_assembly_hook.installed_hooks") as hooks:
                    self.assertEqual(main(), 0)
        revision.assert_called_once_with(["git", "-C", str(self.asap), "rev-parse", "HEAD"], text=True)
        clean.assert_called_once_with(["git", "-C", str(self.asap), "diff", "--exit-code", "HEAD", "--"], check=True)
        hooks.assert_not_called()
        trace = [json.loads(line) for line in self.trace.read_text().splitlines()]
        self.assertEqual([event["kind"] for event in trace], ["run_start", "run_end"])
        self.assertEqual(trace[0]["native_argv"], invoke.call_args[0][0])
        self.assertFalse(trace[0]["hooks_installed"])
        self.assertFalse(trace[0]["sdf_writes_enabled"])
        self.assertIsNone(trace[0]["base_part"])
        self.assertIsNone(trace[0]["ranker_name"])
        self.assertEqual(trace[-1]["status"], "returned_from_upstream")

    def test_wrong_commit_and_dirty_source_do_not_create_trace(self):
        with patch.object(sys, "argv", self.argv), patch("run_assembly_hook.subprocess.check_output", return_value="wrong"):
            with self.assertRaises(RuntimeError):
                main()
        self.assertFalse(self.trace.exists())
        with patch.object(sys, "argv", self.argv), patch("run_assembly_hook.subprocess.check_output", return_value=COMMIT):
            with patch("run_assembly_hook.subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")):
                with self.assertRaises(subprocess.CalledProcessError):
                    main()
        self.assertFalse(self.trace.exists())

    def test_existing_trace_or_native_output_never_overwritten(self):
        self.trace.write_text("existing evidence", encoding="utf-8")
        with patch.object(sys, "argv", self.argv), self.assertRaises(FileExistsError):
            main()
        self.assertEqual(self.trace.read_text(), "existing evidence")
        other_trace = self.root / "other-events.jsonl"
        argv = [str(other_trace) if token == str(self.trace) else token for token in self.argv]
        self.native.mkdir()
        with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
            main()
        self.assertFalse(other_trace.exists())

    def test_native_error_is_logged_and_rethrown(self):
        with patch.object(sys, "argv", self.argv), patch("run_assembly_hook.subprocess.check_output", return_value=COMMIT):
            with patch("run_assembly_hook.subprocess.run"), patch("run_assembly_hook.invoke_official", side_effect=RuntimeError("fixture")):
                with self.assertRaises(RuntimeError):
                    main()
        trace = [json.loads(line) for line in self.trace.read_text().splitlines()]
        self.assertEqual(trace[-1]["status"], "ERROR")
        self.assertEqual(trace[-1]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
