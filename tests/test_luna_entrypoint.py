"""Offline Luna CLI and pipeline-mailbox boundaries; no API or physics runs."""
from copy import deepcopy
from decimal import InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))
import run_luna_experiment as entry
from orchestrator.file_service import FileRankServer, atomic_new


def luna_config(live=False, **changes):
    return {"enabled": live, "model": "gpt-6-luna", "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY", "max_calls": 4 if live else 0,
            "max_usd": "1" if live else 0, "input_usd_per_million": "2" if live else None,
            "output_usd_per_million": "4" if live else None, "max_output_tokens": 256,
            "reasoning_effort": "none", "request_overhead_tokens": 2048, **changes}


def rank_request():
    return {"schema_version": 1, "kind": "rank_request", "request": {
        "run_id": "fixture-run", "request_id": "fixture-run:r1", "state_id": "state1",
        "assembled_parts": ["2", "1", "3"], "candidate_ids": ["2", "1"], "base_part": "3"},
        "features": {"only_hint": True}, "timeout_seconds": 4}


class LunaEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="asap_luna_entrypoint_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network_forbidden"))
        self.network = network.start()
        self.addCleanup(network.stop)
        self.addCleanup(self.network.assert_not_called)
        physics = patch("subprocess.Popen", side_effect=AssertionError("physics_and_subprocess_forbidden"))
        self.physics = physics.start()
        self.addCleanup(physics.stop)
        self.addCleanup(self.physics.assert_not_called)

    def test_disabled_zero_caps_need_no_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.assertIsNone(entry.validate_config(luna_config(), False))

    def test_enabled_and_allow_live_flag_must_both_agree(self):
        for enabled, flag in ((True, False), (False, True), (1, True), ("true", True)):
            with self.subTest(enabled=enabled, flag=flag):
                with self.assertRaises(ValueError):
                    entry.validate_config(luna_config(enabled=enabled), flag)

    def test_offline_authorization_caps_must_stay_zero(self):
        for change in ({"max_calls": 1}, {"max_usd": "0.1"}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    entry.validate_config(luna_config(**change), False)

    def test_live_validation_requires_explicit_key_without_exposing_value(self):
        for key in ("", "  "):
            with patch.dict(os.environ, {"OPENAI_API_KEY": key}):
                with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY is missing"):
                    entry.validate_config(luna_config(True), True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-not-real-key"}):
            self.assertIsNone(entry.validate_config(luna_config(True), True))

    def test_live_caps_rates_and_generation_limits_fail_closed(self):
        cases = ({"max_calls": 0}, {"max_calls": True}, {"max_calls": -1},
                 {"max_usd": 0}, {"max_usd": "NaN"}, {"input_usd_per_million": None},
                 {"output_usd_per_million": -1}, {"max_output_tokens": 4097},
                 {"request_overhead_tokens": 0}, {"reasoning_effort": "unlimited"},
                 {"api_key_env": "OTHER_ENV"})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-not-real-key"}):
            for change in cases:
                with self.subTest(change=change):
                    with self.assertRaises((ValueError, InvalidOperation)):
                        entry.validate_config(luna_config(True, **change), True)

    def test_model_endpoint_and_complete_config_are_required(self):
        cases = (luna_config(model="other-model"), luna_config(base_url="https://example.invalid/v1"),
                 luna_config(api_key="inline-secret-forbidden"), {})
        for cfg in cases:
            with self.subTest(keys=sorted(cfg)):
                with self.assertRaises(ValueError):
                    entry.validate_config(cfg, False)

    def test_variants_have_stable_cartesian_order(self):
        self.assertEqual(entry.variants("both", "both"), [
            ("native", "off"), ("native", "on"), ("reverse", "off"), ("reverse", "on")])
        self.assertEqual(entry.variants("native", "both"), [("native", "off"), ("native", "on")])
        self.assertEqual(entry.variants("reverse", "off"), [("reverse", "off")])

    def test_parser_defaults_and_invalid_variant_are_explicit(self):
        args = entry.parser().parse_args(["--run-dir", str(self.root / "new")])
        self.assertFalse(args.allow_live_api)
        self.assertFalse(args.check_only)
        self.assertEqual((args.seed, args.memory, args.base_order), (0, "both", "native"))
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            entry.parser().parse_args(["--run-dir", str(self.root / "new"), "--memory", "learn"])

    def test_e_drive_and_outside_project_paths_are_rejected_before_stat(self):
        with patch("pathlib.Path.lstat", side_effect=AssertionError("filesystem_must_not_be_accessed")):
            for raw in ("E:/do-not-open/config.json", "/mnt/e/do-not-open", "C:/outside-project/config.json"):
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    entry.safe(Path(raw))

    def test_parent_traversal_is_rejected_before_stat(self):
        with patch("pathlib.Path.lstat", side_effect=AssertionError("filesystem_must_not_be_accessed")):
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                entry.safe(PROJECT / "runs" / ".." / "outside")

    def test_link_and_reparse_paths_are_rejected(self):
        for attributes, mode in ((0, stat.S_IFLNK), (1024, stat.S_IFDIR)):
            fake = SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
            with patch("pathlib.Path.lstat", return_value=fake):
                with self.assertRaisesRegex(ValueError, "links and reparse"):
                    entry.safe(PROJECT / "runs" / "unused-fixture")

    def _check_only(self, *, live=False, memory="off", seed="0", nested=False):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(luna_config(live)), encoding="utf-8")
        memory_path = self.root / "runs" / "memory.json"
        if memory != "off":
            memory_path.parent.mkdir(exist_ok=True)
            memory_path.write_text("{}", encoding="utf-8")
        output = self.root / "runs" / "new-run"
        if nested:
            output = output / "nested"
        argv = ["--run-dir", str(output), "--llm-config", str(config_path),
                "--memory-snapshot", str(memory_path), "--memory", memory,
                "--base-order", "both", "--seed", seed, "--check-only"]
        if live:
            argv.append("--allow-live-api")
        before = {path.relative_to(self.root).as_posix(): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        captured = io.StringIO()
        with patch.object(entry, "ROOT", self.root), patch("sys.stdout", captured), \
                patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-not-real-key" if live else ""}), \
                patch("pathlib.Path.mkdir", side_effect=AssertionError("check_only_must_not_create_output")), \
                patch("shutil.copyfile", side_effect=AssertionError("check_only_must_not_copy")), \
                patch("run_recorded.run_recorded", side_effect=AssertionError("no_worker")), \
                patch("run_memory_multiseed.preserved_artifacts", side_effect=AssertionError("no_old_results_scan")), \
                patch("orchestrator.file_service.FileRankServer", side_effect=AssertionError("no_service")):
            result = entry.execute(entry.parser().parse_args(argv))
        self.assertEqual(result, 0)
        self.assertFalse(output.exists())
        after = {path.relative_to(self.root).as_posix(): path.read_bytes()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        message = json.loads(captured.getvalue())
        self.assertEqual((message["api_calls"], message["physics_runs"]), (0, 0))
        self.assertEqual(message["live_enabled"], live)
        self.assertNotIn("fixture-not-real-key", captured.getvalue())
        return message

    def test_check_only_offline_creates_no_output_and_starts_nothing(self):
        message = self._check_only()
        self.assertEqual(message["variants"], [["native", "off"], ["reverse", "off"]])

    def test_check_only_live_authorization_still_sends_nothing(self):
        self._check_only(live=True)

    def test_check_only_memory_on_checks_only_file_presence(self):
        message = self._check_only(memory="both")
        self.assertEqual(len(message["variants"]), 4)

    def test_check_only_rejects_invalid_seed_and_nested_run(self):
        for seed, nested in (("-1", False), (str(2 ** 32), False), ("0", True)):
            with self.subTest(seed=seed, nested=nested), self.assertRaises(ValueError):
                self._check_only(seed=seed, nested=nested)

    def test_pipeline_backend_requires_exclusive_explicit_service(self):
        target = self.root / "ipc-never-created"
        cases = ({"backend": "pipeline"}, {"backend": "pipeline", "rank_service": object()},
                 {"backend": "noop", "rank_service": Mock(rank=Mock())},
                 {"backend": "pipeline", "rank_service": Mock(rank=Mock()), "llm_config": {}},
                 {"backend": "pipeline", "rank_service": Mock(rank=Mock()), "transport": Mock()})
        for kwargs in cases:
            with self.subTest(keys=sorted(kwargs)), self.assertRaises(ValueError):
                FileRankServer(target, "fixture-run", emit=Mock(), allow_test_root=True, **kwargs)
        self.assertFalse(target.exists())

    def test_pipeline_is_injected_and_response_reused_without_reexecution(self):
        offered = rank_request()
        immutable = deepcopy(offered)
        pipeline = Mock()
        pipeline.rank.return_value = {"schema_version": 1, "kind": "rank_response",
            "request_id": "fixture-run:r1", "state_id": "state1", "candidate_ids": ["1", "2"],
            "fallback_reason": None, "metadata": {"attempted": False, "llm_live": False}}
        root = self.root / "ipc"
        emitted = Mock()
        with patch("orchestrator.llm_ranker.RankService", side_effect=AssertionError("use_injected_service")):
            server = FileRankServer(root, "fixture-run", "pipeline", emitted,
                                    allow_test_root=True, rank_service=pipeline)
        filename = hashlib.sha256(b"fixture-run:r1").hexdigest() + ".json"
        atomic_new(root / "requests" / filename, offered, allow_test_root=True)
        server.serve_pending()
        saved = (root / "responses" / filename).read_bytes()
        server.serve_pending()
        pipeline.rank.assert_called_once_with(immutable)
        self.assertEqual(offered, immutable)
        answer = json.loads(saved)
        self.assertEqual(answer["candidate_ids"], ["1", "2"])
        self.assertEqual(answer["metadata"]["backend"], "pipeline")
        self.assertFalse(answer["metadata"]["llm_live"])
        server.close()
        pipeline.close.assert_called_once_with()
        restarted_pipeline = Mock()
        restarted = FileRankServer(root, "fixture-run", "pipeline", emitted,
                                   allow_test_root=True, rank_service=restarted_pipeline)
        restarted.serve_pending()
        restarted.close()
        restarted_pipeline.rank.assert_not_called()
        self.assertEqual((root / "responses" / filename).read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
