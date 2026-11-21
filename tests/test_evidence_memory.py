"""Synthetic memory contracts only; no test here executes ASAP physics."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from evidence_memory import (
    EvidenceStore, FrozenMemory, beam_context, canonical, digest,
    export_snapshot, record_from_pair,
)


def fixture_context():
    return beam_context(
        {"commit": "pinned", "asset_hash": "mesh-a", "sdf_hash": "sdf-a",
         "config_hash": "config-a", "binary_hash": "binary-a"},
        {"optimizer": "L-BFGS-B", "max_grippers": 2, "support_config": "base6"})


def event_pair(tag="1", status="PASS", part="1", remaining=None, pose=None):
    assembled = ["0", "1", "2", "3", "6"] if remaining is None else remaining
    request = {
        "part_move": part, "parts_rest": [p for p in assembled if p != part],
        "parts_removed": [p for p in ["0", "1", "2", "3", "6"] if p not in assembled],
        "pose": pose, "max_grippers": 2, "timeout": None,
        "optimizer": "L-BFGS-B", "debug": 1, "render": False,
    }
    start = {"kind": "verifier_start", "run_id": "source", "event_id": "s" + tag,
             "call_id": "v" + tag, "request": request}
    raw = {"feasible": status == "PASS", "action": [0, 0, 1] if status == "PASS" else None,
           "parts_fix": [] if status == "PASS" else None, "base_part": "6",
           "part_move": part, "pose": pose, "grasp": None}
    result = {"kind": "verifier_result", "run_id": "source", "event_id": "r" + tag,
              "call_id": "v" + tag, "status": status,
              "raw": None if status == "ERROR" else raw,
              "blocker": None, "path": None}
    if status == "ERROR":
        result["error_type"] = "RuntimeError"
    return start, result


def make_record(context=None, **kwargs):
    start, result = event_pair(**kwargs)
    return record_from_pair(fixture_context() if context is None else context,
                            start, result, "source/events.jsonl", "a" * 64)


class EvidenceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = EvidenceStore(self.root / "evidence.sqlite")
        self.addCleanup(self.store.close)
        self.context = fixture_context()

    def freeze(self, records):
        self.store.insert_many(records)
        path = self.root / "frozen.json"
        export_snapshot(self.store, path)
        return FrozenMemory(path), path

    def lookup(self, memory, context=None, parts=("0", "1", "2", "3", "6"), **kwargs):
        return memory.lookup(self.context if context is None else context,
                             parts, kwargs.pop("base_part", "6"), **kwargs)

    def test_canonical_is_stable_unicode_and_rejects_nonfinite(self):
        self.assertEqual(canonical({"z": "근거", "a": 1}), '{"a":1,"z":"근거"}')
        self.assertEqual(digest({"b": 2, "a": 1}), digest({"a": 1, "b": 2}))
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical({"value": value})

    def test_pair_preserves_raw_and_false_is_only_unknown(self):
        start, result = event_pair(status="UNKNOWN")
        record = record_from_pair(self.context, start, result, "events.jsonl", "a" * 64)
        self.assertEqual(record["status"], "UNKNOWN")
        self.assertEqual(record["raw_result"], result["raw"])
        self.assertEqual(record["record_id"], result["event_id"])
        self.assertIsNone(record["blocker"])
        self.assertIsNone(record["path"])
        start["request"]["parts_rest"].clear()
        result["raw"]["feasible"] = True
        self.context["input_snapshot"]["asset_hash"] = "changed"
        self.assertFalse(record["raw_result"]["feasible"])
        self.assertEqual(len(record["request"]["parts_rest"]), 4)
        self.assertEqual(record["context"]["input_snapshot"]["asset_hash"], "mesh-a")

    def test_pair_mismatch_and_status_raw_disagreement_fail(self):
        for field, value in (("run_id", "another"), ("call_id", "another"),
                             ("status", "PASS")):
            start, result = event_pair(status="UNKNOWN")
            result[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                record_from_pair(self.context, start, result, "events.jsonl", "a" * 64)
        start, result = event_pair(status="ERROR")
        record = record_from_pair(self.context, start, result, "events.jsonl", "a" * 64)
        self.assertEqual(record["status"], "ERROR")
        self.assertIsNone(record["raw_result"])

    def test_idempotence_conflicting_id_rolls_back_entire_batch(self):
        first = make_record()
        self.assertEqual(self.store.insert_many([first]), {"inserted": 1, "duplicates": 0})
        self.assertEqual(self.store.insert_many([deepcopy(first)]), {"inserted": 0, "duplicates": 1})
        conflict = deepcopy(first)
        conflict["source"]["ref"] = "different-source.jsonl"
        with self.assertRaises(ValueError):
            self.store.insert_many([make_record(tag="2"), conflict])
        self.assertEqual(self.store.records(), [first])

    def test_sqlite_update_and_delete_are_blocked(self):
        self.store.insert_many([make_record()])
        connection = sqlite3.connect(str(self.root / "evidence.sqlite"))
        try:
            for sql in ("UPDATE records SET hash = 'changed'", "DELETE FROM records"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(sql)
                connection.rollback()
        finally:
            connection.close()
        self.assertEqual(len(self.store.records()), 1)

    def test_frozen_snapshot_excludes_later_evaluation_records_and_is_immutable(self):
        memory, path = self.freeze([make_record()])
        original_bytes = path.read_bytes()
        before = self.lookup(memory)
        self.store.insert_many([make_record(tag="evaluation", part="2")])
        self.assertEqual(self.lookup(memory), before)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(before["positive_candidate_ids"], ["1"])
        with self.assertRaises(FileExistsError):
            export_snapshot(self.store, path)
        with self.assertRaises(FrozenInstanceError):
            memory.content_sha256 = "changed"
        before["evidence_ids"].clear()
        self.assertEqual(self.lookup(memory)["evidence_ids"], ["r1"])

    def test_context_mesh_sdf_support_settings_and_seed_mismatches_miss(self):
        memory, _ = self.freeze([make_record()])
        contexts = []
        for key in ("asset_hash", "sdf_hash", "config_hash", "binary_hash", "commit"):
            altered = deepcopy(self.context)
            altered["input_snapshot"][key] = "different"
            contexts.append(altered)
        altered = deepcopy(self.context)
        altered["setup"]["support_config"] = "different"
        contexts.append(altered)
        altered = deepcopy(self.context)
        altered["seed"] = 1
        contexts.append(altered)
        for context in contexts:
            with self.subTest(context=context):
                self.assertEqual(self.lookup(memory, context=context)["evidence_ids"], [])

    def test_remaining_pose_base_gripper_timeout_optimizer_mismatches_miss(self):
        timed_start, timed_result = event_pair(tag="timed")
        timed_start["request"]["timeout"] = 10
        timed = record_from_pair(self.context, timed_start, timed_result, "events.jsonl", "a" * 64)
        other_start, other_result = event_pair(tag="optimizer")
        other_start["request"]["optimizer"] = "other"
        other = record_from_pair(self.context, other_start, other_result, "events.jsonl", "a" * 64)
        memory, _ = self.freeze([make_record(), timed, other])
        self.assertEqual(self.lookup(memory)["evidence_ids"], ["r1"])
        cases = ({"parts": ("1", "2", "3", "6")}, {"pose": [[1, 0], [0, 1]]},
                 {"base_part": "3"}, {"max_grippers": 3})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.lookup(memory, **kwargs)["evidence_ids"], [])

    def test_repeated_positive_is_boolean_and_unknown_error_do_not_promote(self):
        records = [make_record(tag="2"), make_record(tag="1"),
                   make_record(tag="3", part="2", status="UNKNOWN"),
                   make_record(tag="4", part="3", status="ERROR")]
        memory, _ = self.freeze(records)
        result = self.lookup(memory)
        self.assertEqual(result["evidence_ids"], ["r1", "r2", "r3", "r4"])
        self.assertEqual(result["positive_candidate_ids"], ["1"])
        self.assertEqual(result["status_counts"], {"PASS": 2, "UNKNOWN": 1, "ERROR": 1})

    def test_only_unknown_and_error_produce_no_positive_candidates(self):
        memory, _ = self.freeze([make_record(status="UNKNOWN"),
                                 make_record(tag="2", status="ERROR")])
        result = self.lookup(memory)
        self.assertEqual(result["positive_candidate_ids"], [])
        self.assertEqual(len(result["evidence_ids"]), 2)

    def test_snapshot_schema_hash_and_rehashed_invalid_record_are_errors(self):
        _, path = self.freeze([make_record()])
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        variants = []
        for key, value in (("schema_version", 2), ("schema_version", True),
                           ("content_sha256", "bad"),
                           ("allow_hard_pruning", True), ("record_count", 2),
                           ("source_record_ids", ["foreign"]), ("policy", "hard-prune")):
            variant = deepcopy(snapshot)
            variant[key] = value
            variants.append(variant)
        variant = deepcopy(snapshot)
        variant["records"][0]["status"] = "UNKNOWN"
        variant["content_sha256"] = digest(variant["records"])
        variants.append(variant)
        for index, variant in enumerate(variants):
            tampered = self.root / ("tampered-{}.json".format(index))
            tampered.write_text(canonical(variant), encoding="utf-8")
            with self.subTest(index=index), self.assertRaises(ValueError):
                FrozenMemory(tampered)


if __name__ == "__main__":
    unittest.main()
