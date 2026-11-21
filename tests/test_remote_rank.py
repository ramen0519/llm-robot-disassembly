"""Local file IPC fixtures only; no WSL, actual physics, network, or API calls."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import stat
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
import rank_contract
import rank_hook
from rank_contract import RankRequest, RankResponse
from remote_rank_client import (MAX_BYTES, RemoteRankClient, RemoteRankingError, canonical_bytes,
                                checked_path, installed_remote_ranker, read_json_bounded,
                                request_filename, write_json_new)
from run_remote_rank import create_parser, timeout_argument
from test_rank_hook import FixtureGenerator, FixturePlanner, rng_digest


class RemoteRankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ipc = self.root / "ipc"
        self.audit = self.root / "audit.jsonl"
        self.request = RankRequest("fixture", "fixture:r1", "state", ("0", "1", "2", "6"),
                                   ("2", "0", "1"), "6")
        self.client = RemoteRankClient(self.ipc, self.audit, {"geometry": {"observed": True}},
                                       timeout_seconds=0.08, allow_test_root=True)
        self.addCleanup(self.client.close)

    def response(self, **changes):
        value = dict(schema_version=1, kind="rank_response", request_id=self.request.request_id,
                     state_id=self.request.state_id, candidate_ids=["1", "0", "2"],
                     fallback_reason=None, metadata={"backend": "fixture", "llm_live": False})
        value.update(changes)
        return value

    def publish(self, value):
        path = self.ipc / "responses" / request_filename(self.request.request_id)
        write_json_new(path, value, allow_test_root=True)
        return path

    def decision(self, request=None):
        with installed_remote_ranker(self.client):
            return rank_hook.rank_candidates(request or self.request, "remote")

    def audit_rows(self):
        return [json.loads(line) for line in self.audit.read_text(encoding="utf-8").splitlines()]

    def test_valid_response_reorders_and_request_keeps_observation_and_ids(self):
        self.publish(self.response())
        result = self.decision()
        self.assertEqual(result.candidate_ids, ("1", "0", "2"))
        self.assertIsNone(result.fallback_reason)
        path = self.ipc / "requests" / request_filename(self.request.request_id)
        request, _ = read_json_bounded(path, True)
        self.assertEqual(request["request"]["candidate_ids"], list(self.request.candidate_ids))
        self.assertEqual(request["features"], {"geometry": {"observed": True}})
        self.assertEqual(request["timeout_seconds"], 0.08)
        self.assertEqual(request["kind"], "rank_request")
        self.assertEqual(self.audit_rows()[-1]["response_metadata"]["backend"], "fixture")

    def test_service_fallback_reason_forces_native_order_even_with_valid_permutation(self):
        self.publish(self.response(fallback_reason="api_disabled"))
        result = self.decision()
        self.assertEqual(result.candidate_ids, self.request.candidate_ids)
        self.assertIsNotNone(result.fallback_reason)
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "api_disabled")

    def test_duplicate_ids_missing_ids_and_unknown_candidate_fallback(self):
        for index, candidates in enumerate((["0", "0", "1"], ["0", "1"], ["0", "1", "foreign"])):
            request = RankRequest("fixture", "fixture:bad" + str(index), "state",
                                  self.request.assembled_parts, self.request.candidate_ids, "6")
            path = self.ipc / "responses" / request_filename(request.request_id)
            write_json_new(path, self.response(request_id=request.request_id, candidate_ids=candidates), True)
            with self.subTest(candidates=candidates):
                self.assertEqual(self.decision(request).candidate_ids, request.candidate_ids)
                self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "invalid_permutation")

    def test_stale_request_or_state_is_rejected(self):
        self.publish(self.response(state_id="different"))
        self.assertEqual(self.decision().candidate_ids, self.request.candidate_ids)
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "stale_request_or_state")

    def test_invalid_response_schema_is_rejected(self):
        self.publish(self.response(schema_version=True))
        self.assertEqual(self.decision().candidate_ids, self.request.candidate_ids)
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "invalid_response_schema")

    def test_nonstring_candidates_are_rejected(self):
        self.publish(self.response(candidate_ids=[0, 1, 2]))
        self.assertEqual(self.decision().candidate_ids, self.request.candidate_ids)
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "invalid_candidate_type")

    def test_timeout_falls_back_and_duplicate_does_not_wait_or_reissue(self):
        self.assertEqual(self.decision().candidate_ids, self.request.candidate_ids)
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "response_timeout")
        with patch("remote_rank_client.write_json_new", side_effect=AssertionError("duplicate write")):
            with patch("remote_rank_client.time.sleep", side_effect=AssertionError("duplicate wait")):
                self.assertEqual(self.decision().candidate_ids, self.request.candidate_ids)
        self.assertTrue(self.audit_rows()[-1]["duplicate_cached"])
        self.assertEqual(len(list((self.ipc / "requests").glob("*.json"))), 1)

    def test_duplicate_success_is_cached_and_conflicting_payload_falls_back(self):
        self.publish(self.response())
        first = self.decision()
        with patch("remote_rank_client.read_json_bounded", side_effect=AssertionError("duplicate read")):
            self.assertEqual(self.decision(), first)
        conflict = RankRequest("fixture", self.request.request_id, "changed-state",
                               self.request.assembled_parts, self.request.candidate_ids, "6")
        self.assertEqual(self.decision(conflict).candidate_ids, conflict.candidate_ids)
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "duplicate_request_id_conflict")

    def test_existing_identical_request_is_reused_without_new_publication(self):
        self.publish(self.response())
        self.decision()
        path = self.ipc / "requests" / request_filename(self.request.request_id)
        original = path.read_bytes()
        second = RemoteRankClient(self.ipc, self.root / "second-audit.jsonl", self.client.features,
                                  timeout_seconds=0.08, allow_test_root=True)
        try:
            self.assertEqual(second(self.request).candidate_ids, ("1", "0", "2"))
        finally:
            second.close()
        self.assertEqual(path.read_bytes(), original)
        row = json.loads((self.root / "second-audit.jsonl").read_text())
        self.assertFalse(row["request_created"])

    def test_existing_conflicting_request_is_not_overwritten(self):
        path = self.ipc / "requests" / request_filename(self.request.request_id)
        original = b'{"foreign":"request"}'
        path.write_bytes(original)
        self.assertEqual(self.decision().candidate_ids, self.request.candidate_ids)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "existing_request_id_conflict")

    def test_oversized_or_invalid_json_response_is_bounded_fallback(self):
        path = self.ipc / "responses" / request_filename(self.request.request_id)
        path.write_bytes(b"x" * (MAX_BYTES + 1))
        self.assertEqual(self.decision().candidate_ids, self.request.candidate_ids)
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "ipc_error:ValueError")
        other = RankRequest("fixture", "fixture:bad-json", "state", self.request.assembled_parts,
                            self.request.candidate_ids, "6")
        (self.ipc / "responses" / request_filename(other.request_id)).write_text("{invalid", encoding="utf-8")
        self.assertEqual(self.decision(other).candidate_ids, other.candidate_ids)
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "ipc_error:JSONDecodeError")

    def test_oversized_request_is_not_published(self):
        large = RankRequest("fixture", "fixture:large", "x" * MAX_BYTES,
                            self.request.assembled_parts, self.request.candidate_ids, "6")
        self.assertEqual(self.decision(large).candidate_ids, large.candidate_ids)
        self.assertFalse((self.ipc / "requests" / request_filename(large.request_id)).exists())
        self.assertEqual(self.audit_rows()[-1]["fallback_reason"], "request_too_large")

    def test_atomic_writer_never_overwrites_existing_target(self):
        path = self.root / "atomic.json"
        write_json_new(path, {"original": True}, True)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_json_new(path, {"overwrite": True}, True)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_request_filename_does_not_use_raw_request_id_as_path(self):
        value = "run/request:../id"
        self.assertEqual(request_filename(value), hashlib.sha256(value.encode()).hexdigest() + ".json")
        self.assertNotIn("/", request_filename(value))

    def test_e_drive_traversal_and_foreign_root_rejected_before_filesystem_probe(self):
        for path in ("E:/locked", "/mnt/e/locked", str(self.root / ".." / "escape"), "/unexpected/outside"):
            with self.subTest(path=path), patch.object(Path, "lstat", side_effect=AssertionError("must not inspect")):
                with self.assertRaises(ValueError):
                    checked_path(path)
        with patch.object(Path, "lstat", side_effect=AssertionError("must not inspect")):
            with self.assertRaises(ValueError):
                checked_path("/mnt/./e/locked", run_only=False)

    def test_symlink_or_windows_reparse_is_rejected_before_file_open(self):
        target = self.root / "linked.json"
        original = Path.lstat
        for link_stat in (SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0),
                          SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)):
            def lstat(path):
                return link_stat if path == target else original(path)
            with patch.object(Path, "lstat", lstat), patch("remote_rank_client.os.open", side_effect=AssertionError("must not open")):
                with self.assertRaises(ValueError):
                    read_json_bounded(target, True)

    def test_context_restores_original_hook_and_does_not_change_registry(self):
        original = rank_hook.rank_candidates
        registry = dict(rank_contract.REGISTRY)
        with self.assertRaises(RuntimeError):
            with installed_remote_ranker(self.client):
                self.assertIsNot(rank_hook.rank_candidates, original)
                raise RuntimeError("fixture")
        self.assertIs(rank_hook.rank_candidates, original)
        self.assertEqual(dict(rank_contract.REGISTRY), registry)

    def test_file_client_does_not_consume_python_rng(self):
        self.publish(self.response())
        before = random.getstate()
        self.decision()
        self.assertEqual(random.getstate(), before)

    def test_poll_wait_accepts_later_atomic_response(self):
        failures = []
        def serve():
            try:
                request_path = self.ipc / "requests" / request_filename(self.request.request_id)
                deadline = time.monotonic() + 1
                while not request_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.002)
                self.publish(self.response())
            except Exception as error:
                failures.append(error)
        thread = threading.Thread(target=serve)
        thread.start()
        result = self.decision()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        self.assertEqual(result.candidate_ids, ("1", "0", "2"))

    def test_fallback_installed_hooks_keep_all_candidates_and_same_verifier(self):
        log = rank_hook.EventLog(self.root / "hook.jsonl", "fixture-hook")
        previous_rng = random.getstate()
        try:
            random.seed(9)
            planner = FixturePlanner()
            with installed_remote_ranker(self.client):
                with rank_hook.installed_hooks(FixtureGenerator, FixturePlanner, log, "rank", rng_digest,
                                               ranker_name="remote"):
                    candidates = list(FixtureGenerator().generate_candidate_part(["0", "1", "2", "6"]))
                    self.assertEqual(set(candidates), {"0", "1", "2"})
                    returned = planner._simulate(candidates[0], [part for part in ["0", "1", "2", "6"] if part != candidates[0]], [], None, 2)
                    self.assertIs(returned, planner.raw)
                    self.assertEqual(planner.calls, 1)
        finally:
            log.close()
            random.setstate(previous_rng)
        rows = [json.loads(line) for line in (self.root / "hook.jsonl").read_text().splitlines()]
        decision = next(row for row in rows if row["kind"] == "rank_decision")
        self.assertIsNotNone(decision["fallback_reason"])
        self.assertEqual(decision["applied_order"], decision["request"]["candidate_ids"])

    def test_cli_documents_ipc_features_and_bounded_timeout_without_api_options(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as result:
            create_parser().parse_args(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--ipc-dir", output.getvalue())
        self.assertIn("--features-file", output.getvalue())
        self.assertIn("--response-timeout", output.getvalue())
        self.assertNotIn("--api-key", output.getvalue())
        for value in ("0", "-1", "nan", "inf", "bad"):
            with self.subTest(value=value), self.assertRaises(Exception):
                timeout_argument(value)


if __name__ == "__main__":
    unittest.main()
