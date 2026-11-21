"""Synthetic seed-memory audit tests; these do not execute ASAP physics."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
from compare_seed_memories import (
    DEVELOPMENT_VARIANTS, SEEDS, VARIANTS, aggregate_results, audit_development_events,
    audit_seed_source, audit_timeline, compare, plan_outcome_valid, validation_exit_valid,
)
from compare_memory_runs import POLICY, digest, independent_lookup
from evidence_memory import beam_context, record_from_pair
from memory_ranker import BEAM_SETUP
from test_evidence_memory import event_pair
from test_memory_comparison import memory_fixture


def source_fixture(seed=2):
    context = beam_context({"commit": "fixture", "files": {"mesh": "same"}}, BEAM_SETUP, seed)
    records, observed = [], {}
    for tag, status in (("positive", "PASS"), ("negative", "UNKNOWN")):
        start, result = event_pair(tag=tag, status=status)
        record = record_from_pair(context, start, result, "development/events.jsonl", "a" * 64)
        records.append(record)
        observed[record["record_id"]] = (start, result, "a" * 64)
    records.sort(key=lambda item: item["record_id"])
    snapshot = {"schema_version": 1, "record_count": 2, "records": records,
                "source_record_ids": [record["record_id"] for record in records],
                "content_sha256": digest(records), "policy": POLICY, "allow_hard_pruning": False}
    counts = Counter(record["status"] for record in records)
    manifest = {"seed": seed, "context": context, "record_count": 2, "policy": POLICY,
                "allow_hard_pruning": False,
                "status_counts": {status: counts[status] for status in ("PASS", "UNKNOWN", "ERROR")}}
    return snapshot, manifest, context, observed


class SeedMemoryAuditTests(unittest.TestCase):
    def test_same_seed_pass_and_unknown_records_both_retained(self):
        snapshot, manifest, context, observed = source_fixture()
        checks = audit_seed_source(2, snapshot, manifest, context, observed)
        self.assertTrue(all(checks.values()), checks)
        found = independent_lookup(snapshot["records"], context, context["initial_parts"], "6")
        self.assertEqual(found["status_counts"], {"PASS": 1, "UNKNOWN": 1, "ERROR": 0})
        self.assertEqual(found["positive_candidate_ids"], ["1"])

    def test_seed_specific_source_cannot_be_relabelled_for_another_seed(self):
        snapshot, manifest, context, observed = source_fixture()
        wrong = deepcopy(context)
        wrong["seed"] = 3
        checks = audit_seed_source(3, snapshot, manifest, wrong, observed)
        self.assertFalse(checks["single_expected_seed_context"])
        self.assertFalse(checks["records_equal_all_development_requests_and_raw_results"])
        found = independent_lookup(snapshot["records"], wrong, wrong["initial_parts"], "6")
        self.assertEqual(found["matched_evidence_ids"], [])

    def test_filtering_unknown_records_fails_even_with_recomputed_snapshot_hash(self):
        snapshot, manifest, context, observed = source_fixture()
        snapshot["records"] = [record for record in snapshot["records"] if record["status"] == "PASS"]
        snapshot["record_count"] = 1
        snapshot["source_record_ids"] = [snapshot["records"][0]["record_id"]]
        snapshot["content_sha256"] = digest(snapshot["records"])
        manifest["record_count"] = 1
        manifest["status_counts"] = {"PASS": 1, "UNKNOWN": 0, "ERROR": 0}
        checks = audit_seed_source(2, snapshot, manifest, context, observed)
        self.assertTrue(checks["snapshot_schema_and_policy"])
        self.assertFalse(checks["records_equal_all_development_requests_and_raw_results"])

    def test_normal_failed_plan_and_validator_exit_one_remain_valid(self):
        data = {"stats": {"success": False},
                "parsed": {"status": "UNKNOWN", "plan_success": False, "tree_checked": False}}
        self.assertTrue(plan_outcome_valid(data))
        self.assertTrue(validation_exit_valid(data, 1))
        self.assertFalse(validation_exit_valid(data, 0))
        data["parsed"]["status"] = "ERROR"
        self.assertFalse(validation_exit_valid(data, 1))

    def test_development_noop_is_rank_mode_and_cannot_query_memory(self):
        data, _, _ = memory_fixture("off")
        for event in data["events"]:
            for key in ("memory", "memory_mode", "memory_context_digest", "memory_snapshot_sha256"):
                event.pop(key, None)
        checks = {}
        row = audit_development_events(data, "baseline", checks)
        self.assertEqual(row["variant"], "baseline")
        self.assertTrue(all(checks.values()), checks)
        data["events"][0]["memory_mode"] = "off"
        checks = {}
        audit_development_events(data, "baseline", checks)
        self.assertFalse(checks["baseline:development_memory_absent"])

    def test_all_five_seeds_and_failures_stay_in_aggregate(self):
        rows, pairs = [], []
        for seed in SEEDS:
            for variant in VARIANTS:
                rows.append({"seed": seed, "variant": variant, "base_ranker": "noop" if variant.startswith("heur") else "reverse_native",
                             "memory_mode": variant.rsplit("_", 1)[1], "plan_success": seed != 4,
                             "verifier_calls": 4 if seed != 4 else 400, "native_time_seconds": None})
            pairs.extend({"seed": seed, "base_rule": rule, "verifier_calls_on_minus_off": 0}
                         for rule in ("heur", "reverse"))
        result = aggregate_results(rows, pairs)
        self.assertEqual(len(result), 2)
        for group in result:
            self.assertEqual(group["pair_count"], 5)
            for mode in ("off", "on"):
                value = group["modes"][mode]
                self.assertEqual(value["run_count"], 5)
                self.assertEqual(value["success_count"], 4)
                self.assertEqual(value["failure_count"], 1)
                self.assertEqual(value["verifier_calls"]["sum"], 416)
                self.assertEqual(value["native_time_seconds"]["missing_count"], 5)
                self.assertIsNone(value["native_time_seconds"]["sum"])

    def test_missing_artifacts_keep_fifteen_development_and_twenty_evaluation_rows(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as temporary:
            result = compare(Path(temporary))
        self.assertEqual(result["integrity_status"], "ERROR")
        self.assertEqual(len(result["development_rows"]), 15)
        self.assertEqual(len(result["rows"]), 20)
        self.assertEqual(len(result["pair_deltas"]), 10)
        self.assertTrue(all(row["plan_success"] is None for row in result["rows"]))


class SeedMemoryTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(ROOT))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = datetime(2026, 10, 17, tzinfo=timezone.utc)
        self.protocol = {
            "development_order": [{"seed": seed, "variant": variant} for seed in SEEDS for variant in DEVELOPMENT_VARIANTS],
            "execution_order": [{"seed": seed, "variant": variant} for seed in SEEDS for variant in VARIANTS]}
        self.progress, self.loaded = [], {}
        for stage, names, offset, kinds in (
                ("development", self.protocol["development_order"], 0, ("development_start", "development_end")),
                ("evaluation", self.protocol["execution_order"], 64, ("run_start", "run_end"))):
            if stage == "evaluation":
                self.progress.append({"kind": "memory_freeze", "utc": self.iso(63)})
            for index, entry in enumerate(names):
                begin = offset + 4 * index
                self.progress.extend([
                    dict(kind=kinds[0], utc=self.iso(begin), index=index + 1, total=len(names), **entry),
                    dict(kind=kinds[1], utc=self.iso(begin + 3), index=index + 1, total=len(names),
                         process_exit_code=0, validation_exit_code=1, **entry)])
                key = "{}:s{}_{}".format(stage, entry["seed"], entry["variant"])
                self.loaded[key] = {"command": {"started_at": self.iso(begin + 1), "elapsed_seconds": 1},
                                    "stats": {"success": False},
                                    "parsed": {"status": "UNKNOWN", "plan_success": False}}
        self.write_progress()
        self.write("memories_frozen.json", {"utc": self.iso(62)})
        self.write("source_build/command.json", {"started_at": self.iso(60), "elapsed_seconds": 1, "exit_code": 0, "launch_error": None})
        self.write("evaluation_archive/command.json", {"started_at": self.iso(144), "elapsed_seconds": 1, "exit_code": 0, "launch_error": None})
        self.write("comparison/command.json", {"started_at": self.iso(146)})

    def iso(self, seconds):
        return (self.base + timedelta(seconds=seconds)).isoformat()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_progress(self):
        (self.root / "progress.jsonl").write_text("".join(json.dumps(event) + "\n" for event in self.progress), encoding="utf-8")

    def test_completed_development_freeze_evaluation_archive_order_is_observed(self):
        checks, details = audit_timeline(self.root, self.protocol, self.loaded)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(details["memory_freeze"], self.iso(62))

    def test_freezing_before_source_build_completion_is_rejected(self):
        self.write("memories_frozen.json", {"utc": self.iso(59)})
        checks, _ = audit_timeline(self.root, self.protocol, self.loaded)
        self.assertFalse(checks["source_build_completed_between_development_and_freeze"])

    def test_early_evaluation_archive_is_rejected(self):
        self.write("evaluation_archive/command.json", {"started_at": self.iso(142), "elapsed_seconds": 1, "exit_code": 0, "launch_error": None})
        checks, _ = audit_timeline(self.root, self.protocol, self.loaded)
        self.assertFalse(checks["evaluation_archive_started_after_all_evaluation_runs"])

    def test_evaluation_started_before_all_development_is_rejected(self):
        first_evaluation = self.progress.pop(31)
        self.progress.insert(0, first_evaluation)
        self.write_progress()
        with self.assertRaisesRegex(ValueError, "stage order"):
            audit_timeline(self.root, self.protocol, self.loaded)


if __name__ == "__main__":
    unittest.main()
