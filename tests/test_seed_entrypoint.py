"""Seed propagation contracts; these fixtures do not execute physical validation."""
import argparse
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from evidence_memory import beam_context
from memory_ranker import BEAM_SETUP
from run_rank_hook import MAX_SEED, build_run_inputs, main, seed_argument


class SeedEntrypointTests(unittest.TestCase):
    def test_seed_zero_retains_original_native_argv_and_memory_context(self):
        root, log_dir = Path("asap"), Path("new-run")
        snapshot = {"fixture_only": True}
        argv, context = build_run_inputs(root, log_dir, 0, snapshot)
        self.assertEqual(argv, [str(root / "plan_sequence/run_seq_plan.py"),
            "--dir", "beam_assembly", "--id", "original", "--planner", "dfs",
            "--generator", "heur-out", "--max-gripper", "2", "--base-part", "6",
            "--log-dir", str(log_dir), "--early-term"])
        self.assertEqual(context, beam_context(snapshot, BEAM_SETUP, seed=0))
        self.assertEqual(build_run_inputs(root, log_dir)[0], argv)

    def test_nonzero_seed_matches_native_cli_and_memory_context(self):
        snapshot = {"fixture_only": True}
        baseline, _ = build_run_inputs(Path("asap"), Path("new-run"))
        for seed in (1, 7, MAX_SEED):
            with self.subTest(seed=seed):
                argv, context = build_run_inputs(Path("asap"), Path("new-run"), seed, snapshot)
                self.assertEqual(argv, baseline + ["--seed", str(seed)])
                self.assertEqual(context, beam_context(snapshot, BEAM_SETUP, seed=seed))

    def test_trace_without_memory_still_receives_selected_seed(self):
        argv, context = build_run_inputs(Path("asap"), Path("new-run"), 3)
        self.assertEqual(argv[-2:], ["--seed", "3"])
        self.assertIsNone(context)

    def test_seed_parser_accepts_numpy_range_boundaries(self):
        self.assertEqual(seed_argument("0"), 0)
        self.assertEqual(seed_argument(str(MAX_SEED)), MAX_SEED)

    def test_seed_parser_rejects_out_of_range_and_noninteger(self):
        for value in ("-1", str(MAX_SEED + 1), "1.5", "unknown"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    seed_argument(value)

    def test_direct_helper_rejects_invalid_seed_types(self):
        for value in (True, False, "1", 1.0, None, -1, MAX_SEED + 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_run_inputs(Path("asap"), Path("new-run"), value)

    def test_bad_cli_seed_is_rejected_before_upstream_or_output_access(self):
        required = ["run_rank_hook.py", "--asap-root", "asap", "--native-log-dir", "new-run",
                    "--trace-file", "trace.jsonl", "--run-id", "fixture", "--mode", "trace"]
        for value in ("-1", str(MAX_SEED + 1), "1.5"):
            with self.subTest(value=value), patch.object(sys, "argv", required + ["--seed", value]):
                with patch("run_rank_hook.subprocess.check_output") as source_check:
                    with patch("run_rank_hook.EventLog") as event_log, redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            main()
                    self.assertEqual(raised.exception.code, 2)
                    source_check.assert_not_called()
                    event_log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
