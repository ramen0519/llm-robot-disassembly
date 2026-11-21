"""Ordering/failure contracts; fixtures are not physical validation results."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from evidence_memory import EvidenceStore, beam_context, export_snapshot, record_from_pair
from memory_ranker import BEAM_SETUP, MemoryRanker
from rank_contract import RankRequest
from rank_hook import EventLog, installed_hooks
from test_rank_hook import FixtureGenerator, FixturePlanner, rng_digest


class MemoryRankerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.context = beam_context({"test_only": True}, BEAM_SETUP, seed=0)
        self.request = RankRequest("test", "test:r1", "state", ("0", "1", "2", "3", "6"),
                                   ("2", "3", "0", "1"), "6")
        records = []
        for index, (part, status) in enumerate((("2", "UNKNOWN"), ("0", "PASS"))):
            call = "v" + str(index)
            start = dict(kind="verifier_start", run_id="test-source", event_id=call + "start", call_id=call,
                         request=dict(part_move=part, parts_rest=[p for p in self.request.assembled_parts if p != part],
                                      parts_removed=[], pose=None, max_grippers=2, timeout=None,
                                      optimizer="L-BFGS-B", debug=1, render=False))
            result = dict(kind="verifier_result", run_id="test-source", event_id=call + "result", call_id=call,
                          status=status, raw=dict(feasible=status == "PASS", part_move=part, base_part="6",
                          pose=None, action=[0, 0, 1] if status == "PASS" else None, parts_fix=[] if status == "PASS" else None,
                          grasp=None), blocker=None, path=None)
            records.append(record_from_pair(self.context, start, result, "fixture.jsonl", "a" * 64))
        store = EvidenceStore(self.root / "source.sqlite")
        try:
            store.insert_many(records)
            export_snapshot(store, self.root / "snapshot.json")
        finally:
            store.close()

    def tearDown(self):
        self.directory.cleanup()

    def ranker(self, mode="on"):
        return MemoryRanker(self.root / "snapshot.json", self.context, mode)

    def test_positive_promotion_keeps_unknown_and_unobserved_in_stable_order(self):
        applied, metadata = self.ranker().apply(self.request, self.request.candidate_ids)
        self.assertEqual(applied, ("0", "2", "3", "1"))
        self.assertEqual(set(applied), set(self.request.candidate_ids))
        self.assertEqual(metadata["positive_candidate_ids"], ["0"])
        self.assertEqual(metadata["status_counts"]["UNKNOWN"], 1)
        self.assertTrue(metadata["changed"])

    def test_off_does_not_query_or_change_base_order(self):
        ranker = self.ranker("off")
        ranker.memory = SimpleNamespace(lookup=lambda *args, **kwargs: self.fail("off must not query"))
        applied, metadata = ranker.apply(self.request, self.request.candidate_ids)
        self.assertEqual(applied, self.request.candidate_ids)
        self.assertEqual(metadata["matched_evidence_ids"], [])

    def test_lookup_error_falls_back_to_validated_base_order(self):
        ranker = self.ranker()
        def fail(*args, **kwargs):
            raise ValueError("bad memory")
        ranker.memory = SimpleNamespace(lookup=fail)
        applied, metadata = ranker.apply(self.request, self.request.candidate_ids)
        self.assertEqual(applied, self.request.candidate_ids)
        self.assertIn("ValueError", metadata["fallback_reason"])

    def test_invalid_snapshot_is_logged_and_does_not_break_baseline_order(self):
        bad = self.root / "bad.json"
        bad.write_text('{"schema_version":0}', encoding="utf-8")
        ranker = MemoryRanker(bad, self.context, "on")
        applied, metadata = ranker.apply(self.request, self.request.candidate_ids)
        self.assertEqual(applied, self.request.candidate_ids)
        self.assertIn("ValueError", metadata["fallback_reason"])

    def test_context_mismatch_does_not_promote(self):
        ranker = self.ranker()
        ranker.context = dict(self.context, seed=1)
        applied, metadata = ranker.apply(self.request, self.request.candidate_ids)
        self.assertEqual(applied, self.request.candidate_ids)
        self.assertEqual(metadata["positive_candidate_ids"], [])

    def test_hook_still_calls_same_verifier_and_retains_all_candidates(self):
        log = EventLog(self.root / "events.jsonl", "hook-test")
        planner = FixturePlanner()
        try:
            with installed_hooks(FixtureGenerator, FixturePlanner, log, "rank", rng_digest,
                                 memory=self.ranker()):
                candidates = list(FixtureGenerator().generate_candidate_part(list(self.request.assembled_parts)))
                self.assertEqual(candidates[0], "0")
                self.assertEqual(set(candidates), {"0", "1", "2", "3"})
                raw = planner._simulate(candidates[0], [p for p in self.request.assembled_parts if p != candidates[0]],
                                        [], None, 2)
                self.assertIs(raw, planner.raw)
                self.assertEqual(planner.calls, 1)
        finally:
            log.close()
        entries = [json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()]
        rank = next(e for e in entries if e["kind"] == "rank_decision")
        self.assertEqual(rank["rng_before"], rank["rng_after"])
        self.assertEqual(next(e for e in entries if e["kind"] == "verifier_result")["status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
