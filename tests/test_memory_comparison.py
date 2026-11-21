"""Synthetic memory-log audit tests, not actual CAD or physics validation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

from compare_memory_runs import (
    POLICY, VARIANTS, audit_evaluation_archive, audit_frozen_hashes,
    audit_memory_events, digest, independent_lookup,
)
from test_rule_comparison import successful_ascending_fixture


SNAPSHOT_SHA = "synthetic-snapshot-sha256"


def memory_fixture(mode="on"):
    data = successful_ascending_fixture()
    context = {
        "version": 1, "scope": "official_beam_original", "usage": "soft_ranking_only",
        "input_snapshot": {"commit": "synthetic-source", "files": {"mesh": "mesh-sha"}},
        "setup": {"optimizer": "L-BFGS-B", "max_grippers": 2}, "seed": 0,
        "base_part": "6", "initial_parts": ["0", "1", "2", "3", "6"],
    }
    starts = {event["call_id"]: event for event in data["events"] if event["kind"] == "verifier_start"}
    records = []
    for result in [event for event in data["events"] if event["kind"] == "verifier_result"]:
        records.append({"record_id": "source:" + result["call_id"], "context": deepcopy(context),
                        "request": deepcopy(starts[result["call_id"]]["request"]),
                        "status": "PASS", "raw_result": deepcopy(result["raw"])})
    unknown = deepcopy(records[0])
    unknown.update(record_id="source:unknown", status="UNKNOWN")
    unknown["request"].update(part_move="1", parts_rest=["0", "2", "3", "6"])
    unknown["raw_result"].update(feasible=False, part_move="1", action=None, parts_fix=None)
    records.append(unknown)
    records.sort(key=lambda record: record["record_id"])
    start = data["events"][0]
    start.update(ranker_name="noop", memory_mode=mode, memory_snapshot_sha256=SNAPSHOT_SHA,
                 memory_context_digest=digest(context))
    generations = {event["generation_id"]: event for event in data["events"] if event["kind"] == "generation"}
    ranks = [event for event in data["events"] if event["kind"] == "rank_decision"]
    for index, event in enumerate(ranks):
        generation = generations[event["generation_id"]]
        if mode == "off":
            # Preserve this fixture's consumed path while testing no lookup.
            generation["native_order"] = list(generation["applied_order"])
        base = list(generation["native_order"])
        event["request"]["candidate_ids"] = list(base)
        positive = [str(index)] if mode == "on" else []
        applied = [part for part in base if part in positive] + [part for part in base if part not in positive]
        matched = ["source:v{}".format(index + 1)] if mode == "on" else []
        if mode == "on" and index == 0:
            matched.insert(0, "source:unknown")
        counts = {"PASS": 1, "UNKNOWN": int(index == 0), "ERROR": 0} if mode == "on" else {}
        event.update(ranker_name="noop", base_order=base, applied_order=list(applied), memory={
            "mode": mode, "snapshot_sha256": SNAPSHOT_SHA, "context_digest": digest(context),
            "matched_evidence_ids": matched, "positive_candidate_ids": positive,
            "status_counts": counts, "changed": applied != base, "fallback_reason": None,
        })
        generation["applied_order"] = list(applied)
    return data, records, context


class MemoryComparisonTests(unittest.TestCase):
    def audit(self, data, records, context, mode="on"):
        checks = {}
        row = audit_memory_events(data, "heur_" + mode, checks, records, context, SNAPSHOT_SHA)
        return row, checks

    def assert_pass(self, checks):
        self.assertTrue(checks)
        self.assertEqual([key for key, value in checks.items() if not value], [])

    def test_positive_only_promotion_is_recomputed_with_all_candidates_kept(self):
        data, records, context = memory_fixture()
        row, checks = self.audit(data, records, context)
        self.assert_pass(checks)
        self.assertEqual(row["memory_requests"], 3)
        self.assertEqual(row["memory_evidence_hit_requests"], 3)
        self.assertEqual(row["memory_positive_hit_requests"], 3)
        self.assertEqual(row["memory_evidence_references"], 4)
        self.assertEqual(row["memory_changed_requests"], 3)
        self.assertEqual(row["memory_promoted_candidates"], 3)
        self.assertEqual(row["candidate_orders"][0]["applied"], ["0", "1", "3", "2"])
        self.assertEqual(row["verifier_calls"], 4)

    def test_off_uses_no_evidence_even_when_matching_records_exist(self):
        data, records, context = memory_fixture("off")
        row, checks = self.audit(data, records, context, "off")
        self.assert_pass(checks)
        self.assertEqual(row["memory_requests"], 0)
        self.assertEqual(row["memory_evidence_hit_requests"], 0)
        self.assertEqual(row["memory_promoted_candidates"], 0)
        rank = next(event for event in data["events"] if event["kind"] == "rank_decision")
        rank["memory"]["matched_evidence_ids"] = ["source:v1"]
        _, checks = self.audit(data, records, context, "off")
        self.assertFalse(checks["heur_off:memory_ordering_recomputed"])

    def test_unknown_evidence_is_counted_without_positive_score(self):
        _, records, context = memory_fixture()
        unknown_only = [record for record in records if record["status"] == "UNKNOWN"]
        result = independent_lookup(unknown_only, context, context["initial_parts"], "6")
        self.assertEqual(result["matched_evidence_ids"], ["source:unknown"])
        self.assertEqual(result["positive_candidate_ids"], [])
        self.assertEqual(result["status_counts"], {"PASS": 0, "UNKNOWN": 1, "ERROR": 0})

    def test_unknown_candidate_cannot_be_claimed_as_positive(self):
        data, records, context = memory_fixture()
        rank = next(event for event in data["events"] if event["kind"] == "rank_decision")
        rank["memory"]["positive_candidate_ids"].append("1")
        _, checks = self.audit(data, records, context)
        self.assertFalse(checks["heur_on:memory_ordering_recomputed"])

    def test_context_or_request_scope_changes_cause_no_match(self):
        _, records, context = memory_fixture()
        for key, value in (("seed", 1), ("input_snapshot", {"files": {"mesh": "changed"}}),
                           ("setup", {"optimizer": "different"})):
            with self.subTest(context_key=key):
                query = deepcopy(context)
                query[key] = value
                result = independent_lookup(records, query, context["initial_parts"], "6")
                self.assertEqual(result["matched_evidence_ids"], [])
        source = next(record for record in records if record["record_id"] == "source:v1")
        for key, value in (("pose", [[1, 0], [0, 1]]), ("max_grippers", 3),
                           ("parts_removed", ["2"]), ("timeout", 1.0), ("optimizer", "different")):
            with self.subTest(request_key=key):
                changed = deepcopy(source)
                changed["request"][key] = value
                result = independent_lookup([changed], context, context["initial_parts"], "6")
                self.assertEqual(result["matched_evidence_ids"], [])
        self.assertEqual(independent_lookup(records, context, ["0", "1", "6"], "6")["matched_evidence_ids"], [])
        self.assertEqual(independent_lookup(records, context, context["initial_parts"], "0")["matched_evidence_ids"], [])

    def test_unbacked_evidence_id_and_changed_snapshot_identity_are_detected(self):
        data, records, context = memory_fixture()
        rank = next(event for event in data["events"] if event["kind"] == "rank_decision")
        rank["memory"]["matched_evidence_ids"].append("current-evaluation:e9")
        rank["memory"]["snapshot_sha256"] = "different-snapshot"
        _, checks = self.audit(data, records, context)
        self.assertFalse(checks["heur_on:matched_ids_are_frozen_source_ids"])
        self.assertFalse(checks["heur_on:memory_ordering_recomputed"])

    def test_memory_pass_does_not_substitute_for_missing_physics_result(self):
        data, records, context = memory_fixture()
        data["events"] = [event for event in data["events"]
                          if not (event["kind"] == "verifier_result" and event["call_id"] == "v1")]
        _, checks = self.audit(data, records, context)
        self.assertFalse(checks["heur_on:starts_results_same_order"])
        self.assertFalse(checks["heur_on:raw_verifier_order_matches_stored_edges"])

    def test_frozen_database_and_snapshot_hash_changes_are_detected(self):
        before = {"snapshot": SNAPSHOT_SHA, "source_db": "synthetic-database-sha256"}
        hashes = {"before": deepcopy(before)}
        hashes.update({"after_" + name: deepcopy(before) for name in VARIANTS})
        hashes["after_evaluation_archive"] = deepcopy(before)
        checks = {}
        audit_frozen_hashes(hashes, SNAPSHOT_SHA, before["source_db"], checks)
        self.assert_pass(checks)
        hashes["after_heur_on"]["source_db"] = "database-was-modified"
        audit_frozen_hashes(hashes, SNAPSHOT_SHA, before["source_db"], checks)
        self.assertFalse(checks["heur_on:memory_db_and_snapshot_unchanged"])
        audit_frozen_hashes(hashes, "snapshot-was-modified", before["source_db"], checks)
        self.assertFalse(checks["frozen_memory_hashes_match_files"])
        hashes["after_evaluation_archive"]["snapshot"] = "source-contaminated-by-evaluation"
        audit_frozen_hashes(hashes, SNAPSHOT_SHA, before["source_db"], checks)
        self.assertFalse(checks["source_memory_unchanged_after_evaluation_archive"])

    def test_evaluation_archive_detects_missing_results_raw_change_and_source_overlap(self):
        data, _, context = memory_fixture()
        starts = {event["call_id"]: event for event in data["events"] if event["kind"] == "verifier_start"}
        observed, archived = {}, []
        for result in [event for event in data["events"] if event["kind"] == "verifier_result"]:
            start = starts[result["call_id"]]
            observed[result["event_id"]] = (start, result, "evaluation-events-sha")
            archived.append({"record_id": result["event_id"], "context": deepcopy(context),
                "request": deepcopy(start["request"]), "raw_result": deepcopy(result["raw"]),
                "status": result["status"], "error_type": None, "source": {
                    "run_id": result["run_id"], "call_id": result["call_id"],
                    "start_event_id": start["event_id"], "result_event_id": result["event_id"],
                    "sha256": "evaluation-events-sha"}})
        archived.sort(key=lambda record: record["record_id"])
        snapshot = {"schema_version": 1, "records": archived, "record_count": len(archived),
            "source_record_ids": [record["record_id"] for record in archived],
            "content_sha256": digest(archived), "policy": POLICY, "allow_hard_pruning": False}
        checks = {}
        audit_evaluation_archive(snapshot, context, {"source:v1"}, observed, checks)
        self.assert_pass(checks)
        altered = deepcopy(snapshot)
        altered["records"][0]["raw_result"]["action"] = [1, 0, 0]
        audit_evaluation_archive(altered, context, {"source:v1"}, observed, checks)
        self.assertFalse(checks["evaluation_archive_preserves_raw_requests_results"])
        self.assertFalse(checks["evaluation_archive_snapshot_valid"])
        missing = deepcopy(snapshot)
        missing["records"].pop()
        audit_evaluation_archive(missing, context, {"source:v1"}, observed, checks)
        self.assertFalse(checks["evaluation_archive_contains_exactly_evaluation_results"])
        audit_evaluation_archive(snapshot, context, {archived[0]["record_id"]}, observed, checks)
        self.assertFalse(checks["evaluation_archive_disjoint_from_frozen_source"])


if __name__ == "__main__":
    unittest.main()
