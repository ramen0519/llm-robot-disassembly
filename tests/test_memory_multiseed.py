"""Synthetic multi-seed audit contracts; these tests do not execute physics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from compare_memory_multiseed import (
    COMMIT, SEEDS, VARIANTS, _evaluation_records, aggregate_results,
    archive_evaluation, archive_records_match, compare, expected_cli,
    identity, numeric_summary, seed_identity_valid, suffix,
)
from compare_memory_runs import audit_memory_events, digest, independent_lookup
from evidence_memory import beam_context
from memory_ranker import BEAM_SETUP
from test_build_evidence import pair
from test_memory_comparison import SNAPSHOT_SHA, memory_fixture


def start_event(seed, explicit=True):
    argv = ["/asap/plan_sequence/run_seq_plan.py"] + expected_cli()
    if explicit:
        argv += ["--seed", str(seed)]
    return {"seed": seed, "native_argv": argv}


class MultiSeedAuditTests(unittest.TestCase):
    def test_native_seed_and_envelope_must_both_identify_actual_seed(self):
        self.assertTrue(seed_identity_valid(start_event(0, False), 0))
        self.assertTrue(seed_identity_valid(start_event(0), 0))
        self.assertTrue(seed_identity_valid(start_event(4), 4))
        self.assertFalse(seed_identity_valid(start_event(1, False), 1))
        self.assertFalse(seed_identity_valid(start_event(0), 1))
        altered = start_event(1)
        altered["seed"] = 0
        self.assertFalse(seed_identity_valid(altered, 1))
        for extra in (["--seed", "1"], ["--budget", "1"], ["--seed"]):
            altered = start_event(1)
            altered["native_argv"] += extra
            self.assertFalse(seed_identity_valid(altered, 1))

    def test_seed_zero_source_cannot_hit_other_live_seeds(self):
        _, records, context = memory_fixture()
        self.assertTrue(independent_lookup(records, context, context["initial_parts"], "6")["matched_evidence_ids"])
        for seed in SEEDS[1:]:
            live = deepcopy(context)
            live["seed"] = seed
            found = independent_lookup(records, live, live["initial_parts"], "6")
            self.assertEqual(found["matched_evidence_ids"], [])
            self.assertEqual(found["positive_candidate_ids"], [])

    def test_claimed_seed_zero_hits_are_rejected_using_live_context(self):
        data, records, context = memory_fixture()
        live = deepcopy(context)
        live["seed"] = 2
        checks = {}
        audit_memory_events(data, "heur_on", checks, records, live, SNAPSHOT_SHA)
        self.assertFalse(checks["heur_on:memory_ordering_recomputed"])
        self.assertFalse(checks["heur_on:run_memory_identity"])

    def test_unknown_metrics_remain_missing_and_not_zero(self):
        summary = numeric_summary([4, None, 8, True])
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["missing_count"], 2)
        self.assertEqual(summary["sum"], 12)
        self.assertEqual(summary["mean"], 6)
        self.assertIsNone(numeric_summary([None])["sum"])
        self.assertIsNone(numeric_summary([])["mean"])

    def test_aggregation_separates_seed_zero_and_retains_failed_run_denominator(self):
        rows, pairs = [], []
        for seed in SEEDS:
            for variant in VARIANTS:
                rows.append({"seed": seed, "variant": variant,
                    "base_ranker": "noop" if variant.startswith("heur") else "reverse_native",
                    "memory_mode": variant.rsplit("_", 1)[1], "plan_success": seed != 4,
                    "verifier_calls": None if seed == 4 else 4,
                    "native_time_seconds": None if seed == 4 else 1.0})
            for rule in ("heur", "reverse"):
                pairs.append({"seed": seed, "base_rule": rule,
                              "verifier_calls_on_minus_off": -3 if seed == 0 else 0})
        aggregated = aggregate_results(rows, pairs)
        self.assertEqual(len(aggregated), 4)
        match = next(item for item in aggregated if item["base_rule"] == "heur" and item["group"] == "source_seed_revisit")
        mismatch = next(item for item in aggregated if item["base_rule"] == "heur" and item["group"] == "seed_mismatch")
        self.assertEqual(match["modes"]["off"]["run_count"], 1)
        self.assertEqual(match["paired_verifier_delta_on_minus_off"]["sum"], -3)
        self.assertEqual(mismatch["modes"]["off"]["run_count"], 4)
        self.assertEqual(mismatch["modes"]["off"]["success_count"], 3)
        self.assertEqual(mismatch["modes"]["off"]["failure_count"], 1)
        self.assertEqual(mismatch["modes"]["off"]["verifier_calls"]["missing_count"], 1)
        self.assertEqual(mismatch["paired_verifier_delta_on_minus_off"]["sum"], 0)


class MultiSeedArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        snapshot = {"commit": COMMIT, "tracked_source_clean": True, "files": {"fixture": "same"}}
        for seed in SEEDS:
            context = beam_context(snapshot, BEAM_SETUP, seed)
            for variant in VARIANTS:
                name = identity(seed, variant)
                folder = self.root / ("s" + str(seed)) / variant
                native = folder / suffix(seed)
                native.mkdir(parents=True)
                self.write(native / "setup.json", BEAM_SETUP)
                for stage in ("before", "after"):
                    directory = self.root / ("inputs_" + stage + "_" + name)
                    directory.mkdir()
                    self.write(directory / "stdout.log", snapshot)
                start = start_event(seed)
                start.update(kind="run_start", event_id=name + ":begin", run_id=name,
                             source_commit=COMMIT, memory_context_digest=digest(context))
                attempt = pair(name)
                for event in attempt:
                    event["run_id"] = name
                trace = [start] + attempt + [{"kind": "run_end", "event_id": name + ":end",
                                             "run_id": name, "status": "returned_from_upstream"}]
                self.write_trace(folder / "events.jsonl", trace)
        self.write(self.root / "memory_snapshot.json", {"source_record_ids": ["source:original"]})

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_trace(self, path, trace):
        path.write_text("".join(json.dumps(event) + "\n" for event in trace), encoding="utf-8")

    def test_archive_preserves_each_actual_seed_and_is_disjoint_and_append_only(self):
        original = (self.root / "memory_snapshot.json").read_bytes()
        manifest = archive_evaluation(self.root)
        self.assertEqual(manifest["record_count"], 20)
        self.assertEqual(manifest["records_by_seed"], {str(seed): 4 for seed in SEEDS})
        self.assertEqual(manifest["repeated_import"], {"inserted": 0, "duplicates": 20})
        self.assertEqual((self.root / "memory_snapshot.json").read_bytes(), original)
        with self.assertRaises(FileExistsError):
            archive_evaluation(self.root)
        records, _, _, observed, _ = _evaluation_records(self.root)
        self.assertTrue(archive_records_match(records, observed))
        records[4]["context"]["seed"] = 0
        self.assertFalse(archive_records_match(records, observed))

    def test_bad_native_seed_stops_archive_before_database_creation(self):
        path = self.root / "s2/heur_on/events.jsonl"
        trace = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        trace[0]["native_argv"][-1] = "0"
        self.write_trace(path, trace)
        with self.assertRaisesRegex(ValueError, "seed/CLI/context"):
            archive_evaluation(self.root)
        self.assertFalse((self.root / "evaluation_evidence").exists())

    def test_source_id_overlap_cannot_be_archived_as_new_evaluation(self):
        self.write(self.root / "memory_snapshot.json", {"source_record_ids": ["s0_heur_off:result"]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            archive_evaluation(self.root)
        self.assertFalse((self.root / "evaluation_evidence").exists())

    def test_missing_native_artifacts_report_errors_without_dropping_trials(self):
        result = compare(self.root)
        self.assertEqual(result["integrity_status"], "ERROR")
        self.assertFalse(result["all_plans_success"])
        self.assertEqual(len(result["rows"]), 20)
        self.assertEqual(len(result["pair_deltas"]), 10)
        self.assertTrue(all(row["plan_success"] is None for row in result["rows"]))


if __name__ == "__main__":
    unittest.main()
