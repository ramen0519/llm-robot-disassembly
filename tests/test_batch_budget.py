"""Shared-ledger policy and accounting tests; external HTTP is prohibited."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.batch_budget import BatchPolicyConflict, budget_summary, bind_batch_policy
from orchestrator.llm_ranker import RankService


def settings(**overrides):
    return {"enabled": True, "model": "fixture-model", "api_key_env": "BATCH_TEST_API_KEY",
            "max_calls": 4, "max_usd": "1", "input_usd_per_million": "2",
            "output_usd_per_million": "4", **overrides}


def request(run="run1", index=1):
    return {"schema_version": 1, "kind": "rank_request", "request": {
        "run_id": run, "request_id": run + ":r" + str(index), "state_id": "state1",
        "assembled_parts": ["1", "2", "3"], "candidate_ids": ["1", "2"], "base_part": "3"},
        "features": {"private_fixture_note": "never-copy-payload-into-summary"}, "timeout_seconds": 4}


def response(url, headers, body, timeout):
    offered = json.loads(body["input"][0]["content"])["request"]
    output = {"request_id": offered["request_id"], "state_id": offered["state_id"],
              "candidate_ids": list(reversed(offered["candidate_ids"]))}
    return {"id": "resp_fixture", "model": "fixture-model", "status": "completed", "error": None,
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(output)}]}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}


class BatchBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="asap_batch_budget_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = self.root / "shared.sqlite"
        self.transport = Mock(side_effect=response)
        key = patch.dict(os.environ, {"BATCH_TEST_API_KEY": "fixture-secret-not-a-real-key"})
        key.start()
        self.addCleanup(key.stop)
        network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network_forbidden"))
        self.network = network.start()
        self.addCleanup(network.stop)
        self.addCleanup(self.network.assert_not_called)

    def service(self, config=None, transport=None, ledger=None):
        return RankService(settings() if config is None else config, ledger or self.ledger,
                           self.transport if transport is None else transport)

    def summary(self):
        return budget_summary(self.ledger)

    def test_two_services_and_runs_share_call_cap_atomically(self):
        cfg = settings(max_calls=1)
        services = [self.service(cfg), self.service(cfg)]
        gate = Barrier(2)
        def rank(index):
            gate.wait(timeout=5)
            return services[index].rank(request("run" + str(index)))
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(rank, range(2)))
        self.assertEqual(sum(item["fallback_reason"] is None for item in results), 1)
        self.assertEqual(sum(item["fallback_reason"] == "call_budget_exhausted" for item in results), 1)
        self.assertEqual(self.transport.call_count, 1)
        summary = self.summary()
        self.assertEqual((summary["request_count"], summary["run_count"], summary["reserved_calls"]), (2, 2, 1))
        self.assertEqual(summary["remaining_calls"], 0)

    def test_concurrent_initialization_binds_one_policy(self):
        gate = Barrier(2)
        def initialize(index):
            gate.wait(timeout=5)
            try:
                self.service(settings(max_calls=index + 1))
                return "bound"
            except BatchPolicyConflict:
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(initialize, range(2)))
        self.assertCountEqual(results, ["bound", "conflict"])
        self.assertEqual(self.summary()["request_count"], 0)
        self.transport.assert_not_called()

    def test_runs_share_monetary_cap_and_failed_calls_keep_reservation(self):
        probe = self.service(ledger=self.root / "probe.sqlite")
        estimate = Decimal(probe.rank(request())["metadata"]["cost_estimate"]["usd"])
        cfg = settings(max_usd=str(estimate * Decimal("1.5")))
        timeout = Mock(side_effect=TimeoutError("not logged"))
        first, second = self.service(cfg, timeout), self.service(cfg, timeout)
        self.assertEqual(first.rank(request("run1"))["fallback_reason"], "api_timeout")
        self.assertEqual(second.rank(request("run2"))["fallback_reason"], "cost_budget_exhausted")
        self.assertEqual(timeout.call_count, 1)
        summary = self.summary()
        self.assertEqual(summary["reserved_calls"], 1)
        self.assertGreater(Decimal(summary["estimated_reservations_usd"]), 0)
        self.assertEqual(summary["usage_reported_calls"], 0)

    def test_full_configuration_is_immutable_across_instances_and_restart(self):
        cfg = settings()
        service = self.service(cfg)
        service.rank(request())
        before = hashlib.sha256(self.ledger.read_bytes()).hexdigest()
        changes = ({"max_calls": 5}, {"max_calls": 3}, {"max_usd": "2"},
                   {"model": "changed-model"}, {"input_usd_per_million": "1"},
                   {"output_usd_per_million": "5"}, {"reasoning_effort": "low"},
                   {"max_output_tokens": 512}, {"request_overhead_tokens": 4096},
                   {"enabled": False}, {"api_key_env": "OTHER_KEY_ENV"})
        for changed in changes:
            with self.subTest(changed=changed):
                with self.assertRaises(BatchPolicyConflict):
                    self.service(settings(**changed))
                self.assertEqual(hashlib.sha256(self.ledger.read_bytes()).hexdigest(), before)
        self.assertEqual(self.transport.call_count, 1)

    def test_disabled_policy_cannot_be_enabled_in_place(self):
        cfg = {"enabled": False, "model": "gpt-6-luna", "max_calls": 0, "max_usd": 0}
        first, second = self.service(cfg), self.service(cfg)
        self.assertEqual(first.rank(request("run1"))["fallback_reason"], "llm_disabled")
        self.assertEqual(second.rank(request("run2"))["fallback_reason"], "llm_disabled")
        with self.assertRaises(BatchPolicyConflict):
            self.service(settings(model="gpt-6-luna"))
        self.assertEqual(self.summary()["reserved_calls"], 0)
        self.transport.assert_not_called()

    def test_mutating_service_configuration_does_not_bypass_policy(self):
        service = self.service(settings(max_calls=1))
        service.rank(request())
        service.config["max_calls"] = 100
        result = service.rank(request("run2"))
        self.assertEqual(result["fallback_reason"], "batch_policy_conflict")
        self.assertFalse(result["metadata"]["attempted"])
        self.assertEqual(self.summary()["request_count"], 1)
        self.assertEqual(self.transport.call_count, 1)

    def test_completed_request_cache_and_new_run_reservations_survive_restart(self):
        cfg = settings(max_calls=2)
        original = self.service(cfg).rank(request("run1"))
        restarted = self.service(cfg)
        cached = restarted.rank(request("run1"))
        self.assertEqual(original["candidate_ids"], cached["candidate_ids"])
        self.assertTrue(cached["metadata"]["cached"])
        self.assertFalse(cached["metadata"]["attempted"])
        self.assertIsNone(restarted.rank(request("run2"))["fallback_reason"])
        self.assertEqual(self.service(cfg).rank(request("run3"))["fallback_reason"], "call_budget_exhausted")
        self.assertEqual(self.transport.call_count, 2)
        self.assertEqual(self.summary()["reserved_calls"], 2)

    def test_same_request_id_different_run_payload_cannot_reuse_or_resend(self):
        service = self.service()
        service.rank(request("run1"))
        changed = request("run2")
        changed["request"]["request_id"] = "run1:r1"
        self.assertEqual(self.service().rank(changed)["fallback_reason"], "request_id_conflict")
        self.assertEqual(self.transport.call_count, 1)

    def test_pending_outcome_remains_reserved_after_restart_without_resend(self):
        class ProcessDeath(BaseException):
            pass
        transport = Mock(side_effect=ProcessDeath())
        cfg = settings(max_calls=1)
        with self.assertRaises(ProcessDeath):
            self.service(cfg, transport).rank(request())
        pending = self.summary()
        self.assertEqual((pending["pending_requests"], pending["unknown_outcomes"]), (1, 1))
        restarted = self.service(cfg, transport)
        result = restarted.rank(request())
        self.assertEqual(result["fallback_reason"], "unknown_outcome")
        self.assertEqual(restarted.rank(request("run2"))["fallback_reason"], "call_budget_exhausted")
        summary = self.summary()
        self.assertEqual((summary["pending_requests"], summary["completed_unknown_outcomes"]), (0, 1))
        self.assertEqual(summary["reserved_calls"], 1)
        self.assertEqual(transport.call_count, 1)

    def test_summary_is_read_only_aggregated_and_does_not_expose_payload(self):
        service = self.service()
        service.rank(request("run1"))
        service.rank(request("run2"))
        before = hashlib.sha256(self.ledger.read_bytes()).hexdigest()
        summary = self.summary()
        self.assertEqual(hashlib.sha256(self.ledger.read_bytes()).hexdigest(), before)
        encoded = json.dumps(summary)
        for private in ("never-copy-payload", "fixture-secret", "run1:r1", "Authorization"):
            self.assertNotIn(private, encoded)
        self.assertEqual((summary["input_tokens"], summary["output_tokens"], summary["total_tokens"]), (200, 40, 240))
        self.assertEqual(Decimal(summary["usage_cost_usd"]), Decimal("0.00056"))
        self.assertEqual(Decimal(summary["budget_committed_usd"]), Decimal(summary["estimated_reservations_usd"]))
        self.assertFalse(summary["billing_cap_guaranteed"])

    def test_summary_and_later_budget_use_usage_above_reservation(self):
        def huge_usage(*args):
            raw = response(*args)
            raw["usage"] = {"input_tokens": 1000000, "output_tokens": 20, "total_tokens": 1000020}
            return raw
        service = self.service(transport=Mock(side_effect=huge_usage))
        service.rank(request("run1"))
        summary = self.summary()
        self.assertGreater(Decimal(summary["usage_cost_usd"]), Decimal(summary["estimated_reservations_usd"]))
        self.assertEqual(summary["budget_committed_usd"], summary["usage_cost_usd"])
        self.assertEqual(Decimal(summary["remaining_estimated_usd"]), 0)
        self.assertEqual(self.service().rank(request("run2"))["fallback_reason"], "cost_budget_exhausted")

    def test_missing_summary_ledger_is_not_created(self):
        absent = self.root / "does-not-exist.sqlite"
        with self.assertRaises(sqlite3.OperationalError):
            budget_summary(absent)
        self.assertFalse(absent.exists())

    def test_summary_rejects_non_c_drive_before_any_filesystem_access(self):
        with patch("pathlib.Path.lstat", side_effect=AssertionError("no filesystem access")):
            with self.assertRaisesRegex(ValueError, "ledger_drive_not_allowed"):
                budget_summary(Path("E:/do-not-open.sqlite"))

    def test_summary_rejects_links_before_sqlite_open(self):
        fake = SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        with patch("pathlib.Path.lstat", return_value=fake), patch("sqlite3.connect") as connect:
            with self.assertRaisesRegex(ValueError, "ledger_link_not_allowed"):
                budget_summary(self.ledger)
            connect.assert_not_called()

    def test_binding_requires_transaction_and_cannot_be_updated_deleted_or_replaced(self):
        service = self.service()
        with closing(sqlite3.connect(self.ledger)) as db:
            with self.assertRaisesRegex(ValueError, "requires_transaction"):
                bind_batch_policy(db, service.config_fingerprint, service.config)
            for sql in ("UPDATE batch_policy SET max_calls=100", "DELETE FROM batch_policy",
                        "INSERT OR REPLACE INTO batch_policy SELECT * FROM batch_policy"):
                with self.subTest(sql=sql):
                    with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable_batch_policy"):
                        db.execute(sql)
                    db.rollback()
        self.assertEqual(self.summary()["max_calls"], 4)

    def test_legacy_completed_request_is_adopted_without_rewriting_or_resending(self):
        service = self.service()
        original = service.rank(request())
        with closing(sqlite3.connect(self.ledger)) as db:
            before = db.execute("SELECT * FROM rank_requests").fetchall()
            for name in ("batch_policy_no_update", "batch_policy_no_delete", "batch_policy_no_replace"):
                db.execute("DROP TRIGGER " + name)
            db.execute("DROP TABLE batch_policy")
            db.commit()
        cached = self.service().rank(request())
        with closing(sqlite3.connect(self.ledger)) as db:
            self.assertEqual(db.execute("SELECT * FROM rank_requests").fetchall(), before)
        self.assertEqual(cached["candidate_ids"], original["candidate_ids"])
        self.assertTrue(cached["metadata"]["cached"])
        self.assertEqual(self.transport.call_count, 1)

    def test_legacy_mixed_policy_is_rejected_without_rebinding(self):
        service = self.service()
        service.rank(request("run1"))
        service.rank(request("run2"))
        with closing(sqlite3.connect(self.ledger)) as db:
            for name in ("batch_policy_no_update", "batch_policy_no_delete", "batch_policy_no_replace"):
                db.execute("DROP TRIGGER " + name)
            db.execute("DROP TABLE batch_policy")
            db.execute("UPDATE rank_requests SET config_fingerprint=? WHERE request_id=?", ("0" * 64, "run2:r1"))
            db.commit()
        before = self.ledger.read_bytes()
        with self.assertRaisesRegex(BatchPolicyConflict, "legacy_policy_conflict"):
            self.service()
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertEqual(self.transport.call_count, 2)


if __name__ == "__main__":
    unittest.main()
