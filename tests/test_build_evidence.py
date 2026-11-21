"""Importer contracts plus preservation checks against available real M4 events.

Fixture tests do not execute physics. The optional real-event check only imports
the existing observations into a temporary database, never modifies source logs.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
from build_evidence import build
from compare_rank_hook import SUFFIX, events
from evidence_memory import EvidenceStore
from memory_ranker import BEAM_SETUP


def pair(tag, status="PASS"):
    run = "fixture/source"
    request = dict(part_move="1", parts_rest=["0", "2", "3", "6"],
                   parts_removed=[], pose=None, max_grippers=2, timeout=None,
                   optimizer="L-BFGS-B", debug=1, render=False)
    start = dict(kind="verifier_start", run_id=run, event_id=tag + ":start",
                 call_id=tag, request=request)
    raw = dict(feasible=status == "PASS", part_move="1", base_part="6", pose=None,
               action=[0, 0, 1] if status == "PASS" else None,
               parts_fix=[] if status == "PASS" else None, grasp=None)
    result = dict(kind="verifier_result", run_id=run, event_id=tag + ":result",
                  call_id=tag, status=status, raw=None if status == "ERROR" else raw,
                  blocker=None, path=None)
    if status == "ERROR":
        result["error_type"] = "RuntimeError"
    return [start, result]


class BuildEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def variant(self, name, source_events, snapshot=None, setup=None):
        snapshot = {"commit": "fixture", "files": {"mesh": "same"}} if snapshot is None else snapshot
        setup = deepcopy(BEAM_SETUP) if setup is None else setup
        for stage in ("before", "after"):
            directory = self.source / ("inputs_" + stage + "_" + name)
            directory.mkdir()
            (directory / "stdout.log").write_text(json.dumps(snapshot), encoding="utf-8")
        folder = self.source / name
        (folder / SUFFIX).mkdir(parents=True)
        (folder / SUFFIX / "setup.json").write_text(json.dumps(setup), encoding="utf-8")
        envelope = dict(kind="run_start", run_id="fixture/source", event_id="fixture:run_start",
                        source_commit=snapshot["commit"], native_argv=[
                            "/asap/plan_sequence/run_seq_plan.py", "--dir", "beam_assembly",
                            "--id", "original", "--planner", "dfs", "--generator", "heur-out",
                            "--max-gripper", "2", "--base-part", "6", "--log-dir", "/run/native",
                            "--early-term"])
        wrapped = [envelope] + source_events + [dict(
            kind="run_end", run_id="fixture/source", event_id="fixture:run_end",
            status="returned_from_upstream")]
        (folder / "events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in wrapped), encoding="utf-8")

    def import_records(self, variants, label="output", source=None):
        directory = self.root / label
        directory.mkdir()
        database = directory / "evidence.sqlite"
        snapshot = directory / "snapshot.json"
        manifest = directory / "manifest.json"
        result = build(self.source if source is None else source, variants,
                       database, snapshot, manifest)
        store = EvidenceStore(database)
        try:
            records = store.records()
        finally:
            store.close()
        return result, records, directory

    def test_all_paired_pass_unknown_error_preserved_with_idempotent_import(self):
        source_events = pair("positive") + pair("negative", "UNKNOWN") + pair("error", "ERROR")
        self.variant("one", source_events)
        original = (self.source / "one/events.jsonl").read_bytes()
        result, records, directory = self.import_records(["one"])
        expected = {event["event_id"]: event for event in source_events
                    if event["kind"] == "verifier_result"}
        self.assertEqual({record["record_id"] for record in records}, set(expected))
        self.assertEqual(result["status_counts"], {"PASS": 1, "UNKNOWN": 1, "ERROR": 1})
        self.assertEqual(result["first_import"], {"inserted": 3, "duplicates": 0})
        self.assertEqual(result["repeated_import"], {"inserted": 0, "duplicates": 3})
        for record in records:
            event = expected[record["record_id"]]
            self.assertEqual(record["status"], event["status"])
            self.assertEqual(record["raw_result"], event["raw"])
            self.assertEqual(record["source"]["sha256"], hashlib.sha256(original).hexdigest())
            self.assertIsNone(record["blocker"])
            self.assertIsNone(record["path"])
        self.assertEqual((self.source / "one/events.jsonl").read_bytes(), original)
        snapshot = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["record_count"], 3)
        self.assertFalse(snapshot["allow_hard_pruning"])

    def test_unmatched_duplicate_and_cross_call_events_rejected_before_database_write(self):
        valid = pair("valid")
        mismatched = deepcopy(valid)
        mismatched[1]["call_id"] = "different"
        cases = {
            "start_only": valid[:1], "result_only": valid[1:],
            "nested_start": valid[:1] + pair("nested")[:1],
            "duplicate_event": valid + valid, "cross_call": mismatched,
        }
        for name, source_events in cases.items():
            self.variant(name, source_events)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.import_records([name], label=name)
            self.assertFalse((self.root / name / "evidence.sqlite").exists())

    def test_mixed_input_or_setup_context_rejected_before_database_write(self):
        self.variant("base", pair("base"))
        self.variant("mesh", pair("mesh"), snapshot={"commit": "fixture", "files": {"mesh": "different"}})
        settings = deepcopy(BEAM_SETUP)
        settings["max_grippers"] = 3
        self.variant("settings", pair("settings"), setup=settings)
        for name in ("mesh", "settings"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.import_records(["base", name], label=name)
            self.assertFalse((self.root / name / "evidence.sqlite").exists())

    def test_mixed_run_seed_commit_or_duplicate_envelope_rejected(self):
        for name in ("mixed_run", "seed", "commit", "duplicate_start"):
            self.variant(name, pair(name))
            path = self.source / name / "events.jsonl"
            trace = events(path)
            if name == "mixed_run":
                trace[1]["run_id"] = trace[2]["run_id"] = "foreign/source"
            elif name == "seed":
                trace[0]["native_argv"] += ["--seed", "1"]
            elif name == "commit":
                trace[0]["source_commit"] = "foreign-commit"
            else:
                duplicate = deepcopy(trace[0])
                duplicate["event_id"] = "second_start"
                trace.insert(1, duplicate)
            path.write_text("".join(json.dumps(event) + "\n" for event in trace), encoding="utf-8")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.import_records([name], label=name)
            self.assertFalse((self.root / name / "evidence.sqlite").exists())

    def test_changed_source_inputs_rejected(self):
        self.variant("one", pair("one"))
        (self.source / "inputs_after_one/stdout.log").write_text('{"commit":"changed"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source input/cache mismatch"):
            self.import_records(["one"])
        self.assertFalse((self.root / "output/evidence.sqlite").exists())

    def test_existing_output_artifact_is_never_overwritten(self):
        self.variant("one", pair("one"))
        names = ("evidence.sqlite", "snapshot.json", "manifest.json")
        for index, name in enumerate(names):
            directory = self.root / ("existing_" + str(index))
            directory.mkdir()
            existing = directory / name
            existing.write_bytes(b"preserve previous evidence")
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                build(self.source, ["one"], *(directory / filename for filename in names))
            self.assertEqual(existing.read_bytes(), b"preserve previous evidence")
            self.assertEqual(list(directory.iterdir()), [existing])

    @unittest.skipUnless((ROOT / "runs/m4_rules_20261017").is_dir(), "real M4 artifacts unavailable")
    def test_real_m4_all_pass_unknown_pairs_import_without_source_changes(self):
        source = ROOT / "runs/m4_rules_20261017"
        variants = ["baseline", "id_ascending", "reverse_native"]
        expected, source_files = {}, []
        for variant in variants:
            path = source / variant / "events.jsonl"
            source_files.extend([path, source / variant / SUFFIX / "setup.json"])
            source_files.extend(source / ("inputs_" + stage + "_" + variant) / "stdout.log"
                                for stage in ("before", "after"))
            for event in events(path, "verifier_result"):
                self.assertNotIn(event["event_id"], expected)
                expected[event["event_id"]] = event
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
        result, records, _ = self.import_records(variants, source=source)
        self.assertEqual({record["record_id"] for record in records}, set(expected))
        self.assertEqual(result["record_count"], len(expected))
        self.assertEqual(result["status_counts"], dict(Counter(event["status"] for event in expected.values())))
        self.assertGreater(result["status_counts"].get("UNKNOWN", 0), 0)
        self.assertGreater(result["status_counts"].get("PASS", 0), 0)
        for record in records:
            self.assertEqual(record["status"], expected[record["record_id"]]["status"])
            self.assertEqual(record["raw_result"], expected[record["record_id"]]["raw"])
        self.assertEqual(before, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in source_files})


if __name__ == "__main__":
    unittest.main()
