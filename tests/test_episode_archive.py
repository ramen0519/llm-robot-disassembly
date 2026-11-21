"""Archive fixtures only: no actual result writes, API, worker, or physics calls."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.episode_archive import archive_episodes, guarded
from evidence_memory import FrozenMemory, beam_context, canonical
from memory_ranker import BEAM_SETUP


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class EpisodeArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.context = beam_context({"commit": "fixture-commit", "tracked_source_clean": True,
                                     "files": {"fixture.obj": "a" * 64}}, BEAM_SETUP, seed=0)

    def episode(self, name="run1", statuses=("PASS", "UNKNOWN")):
        folder = self.root / name
        run_id = self.root.name + "/" + name
        argv = ["/opt/asap-research/ASAP/plan_sequence/run_seq_plan.py", "--dir", "beam_assembly",
                "--id", "original", "--planner", "dfs", "--generator", "heur-out",
                "--max-gripper", "2", "--base-part", "6", "--log-dir", str(folder / "native"), "--early-term"]
        trace = [{"kind": "run_start", "seed": 0, "source_commit": "fixture-commit", "native_argv": argv}]
        for number, status in enumerate(statuses, 1):
            request = {"part_move": "1", "parts_rest": ["0", "2", "3", "6"], "parts_removed": [],
                       "pose": None, "max_grippers": 2, "timeout": None,
                       "optimizer": "L-BFGS-B", "debug": 0, "render": False}
            trace.append({"kind": "verifier_start", "call_id": "v" + str(number), "request": request})
            raw = {"feasible": status == "PASS", "part_move": "1", "base_part": "6", "pose": None,
                   "parts_fix": [] if status == "PASS" else None,
                   "action": [0, 0, 1] if status == "PASS" else None}
            trace.append({"kind": "verifier_result", "call_id": "v" + str(number), "status": status,
                          "raw": None if status == "ERROR" else raw,
                          "error_type": "RuntimeError" if status == "ERROR" else None,
                          "blocker": None, "path": None})
        failed = statuses[-1] == "ERROR" if statuses else False
        trace.append({"kind": "run_end", "status": "ERROR" if failed else "returned_from_upstream",
                      "error_type": "RuntimeError" if failed else None})
        for index, event in enumerate(trace, 1):
            event.update(run_id=run_id, event_id=run_id + ":e" + str(index))
        command = {"exit_code": 1 if failed else 0, "launch_error": None,
                   "elapsed_seconds": 0.1, "started_at": "2026-10-17T00:00:00+00:00"}
        write_json(folder / "execute/command.json", command)
        write_json(folder / "native/dfs-heur-out/s0/original/setup.json", deepcopy(BEAM_SETUP))
        self.save_trace(folder, trace)
        return folder, trace, command

    def save_trace(self, folder, trace):
        (folder / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in trace), encoding="utf-8")

    def reject(self, folder, context=None):
        output = self.root / "archive"
        with self.assertRaises((ValueError, FileNotFoundError)):
            archive_episodes([folder], self.context if context is None else context, output)
        self.assertFalse(output.exists())

    def test_preserves_all_statuses_raw_provenance_and_repeated_import(self):
        first, trace, _ = self.episode()
        second, errors, _ = self.episode("run2", ("ERROR",))
        sources_before = {path: path.read_bytes() for root in (first, second) for path in root.rglob("*") if path.is_file()}
        manifest = archive_episodes([first, second], self.context, self.root / "archive")
        self.assertEqual(manifest["episode_count"], 2)
        self.assertEqual(manifest["record_count"], 3)
        self.assertEqual(manifest["status_counts"], {"PASS": 1, "UNKNOWN": 1, "ERROR": 1})
        self.assertEqual(manifest["first_import"], {"inserted": 3, "duplicates": 0})
        self.assertEqual(manifest["repeated_import"], {"inserted": 0, "duplicates": 3})
        self.assertEqual(manifest["usage"], "evaluation_archive_only_not_read_during_evaluation")
        self.assertFalse(manifest["allow_hard_pruning"])
        snapshot = self.root / "archive/memory_snapshot.json"
        data = json.loads(snapshot.read_text())
        records = {record["record_id"]: record for record in data["records"]}
        for folder, events in ((first, trace), (second, errors)):
            expected_sha = hashlib.sha256((folder / "events.jsonl").read_bytes()).hexdigest()
            for event in events:
                if event["kind"] == "verifier_result":
                    record = records[event["event_id"]]
                    self.assertEqual(record["raw_result"], event["raw"])
                    self.assertEqual(record["source"]["ref"], str(folder / "events.jsonl"))
                    self.assertEqual(record["source"]["sha256"], expected_sha)
                    self.assertEqual(record["context"], self.context)
        self.assertEqual(len(FrozenMemory(snapshot)._records), 3)
        self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), manifest["snapshot_sha256"])
        self.assertEqual(json.loads((self.root / "archive/memory_manifest.json").read_text()), manifest)
        self.assertTrue(all(path.read_bytes() == content for path, content in sources_before.items()))
        with closing(sqlite3.connect(self.root / "archive/evaluation_memory.sqlite")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM records").fetchone()[0], 3)
            for sql in ("DELETE FROM records", "UPDATE records SET hash='changed'"):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)
                db.rollback()

    def test_existing_output_never_reopened_or_overwritten(self):
        folder, _, _ = self.episode(statuses=("UNKNOWN",))
        output = self.root / "archive"
        archive_episodes([folder], self.context, output)
        before = {path: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(FileExistsError), patch("orchestrator.episode_archive.EvidenceStore") as store:
            archive_episodes([folder], self.context, output)
        store.assert_not_called()
        self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))

    def test_seed_native_cli_and_setup_binding(self):
        for change in ("event_seed", "native_seed", "setup", "commit", "context_digest", "native_output"):
            with self.subTest(change=change):
                folder, trace, _ = self.episode(change)
                if change == "event_seed":
                    trace[0]["seed"] = 1
                elif change == "native_seed":
                    trace[0]["native_argv"].extend(["--seed", "1"])
                elif change == "setup":
                    altered = dict(BEAM_SETUP, max_grippers=3)
                    write_json(folder / "native/dfs-heur-out/s0/original/setup.json", altered)
                elif change == "commit":
                    trace[0]["source_commit"] = "another"
                elif change == "context_digest":
                    trace[0]["memory_context_digest"] = "wrong"
                else:
                    argv = trace[0]["native_argv"]
                    argv[argv.index("--log-dir") + 1] = str(self.root / "another/native")
                self.save_trace(folder, trace)
                self.reject(folder)

    def test_nonzero_seed_and_wsl_path_are_bound_without_accessing_wsl(self):
        folder, trace, _ = self.episode()
        context = deepcopy(self.context)
        context["seed"] = trace[0]["seed"] = 2
        argv = trace[0]["native_argv"]
        argv.extend(["--seed", "2"])
        argv[argv.index("--log-dir") + 1] = "/mnt/c/" + (folder / "native").as_posix()[3:]
        write_json(folder / "native/dfs-heur-out/s2/original/setup.json", BEAM_SETUP)
        self.save_trace(folder, trace)
        result = archive_episodes([folder], context, self.root / "archive")
        self.assertEqual(result["sources"][0]["seed"], 2)

    def test_rejects_incomplete_pairs_duplicate_ids_and_cross_run_events(self):
        for change in ("pending", "orphan", "duplicate_event", "duplicate_call", "cross_run"):
            with self.subTest(change=change):
                folder, trace, _ = self.episode(change)
                if change == "pending":
                    del trace[-2]
                elif change == "orphan":
                    del trace[1]
                elif change == "duplicate_event":
                    trace[2]["event_id"] = trace[1]["event_id"]
                elif change == "duplicate_call":
                    trace[3]["call_id"] = trace[1]["call_id"]
                else:
                    trace[2]["run_id"] = "foreign-run"
                self.save_trace(folder, trace)
                self.reject(folder)

    def test_preflight_all_runs_prevents_partial_archive_on_duplicate_global_identity(self):
        first, trace, _ = self.episode()
        second, other, _ = self.episode("run2")
        other[1]["event_id"] = trace[1]["event_id"]
        self.save_trace(second, other)
        with self.assertRaises(ValueError):
            archive_episodes([first, second], self.context, self.root / "archive")
        self.assertFalse((self.root / "archive").exists())
        for event in other:
            event["run_id"] = trace[0]["run_id"]
        self.save_trace(second, other)
        with self.assertRaises(ValueError):
            archive_episodes([first, second], self.context, self.root / "archive")
        self.assertFalse((self.root / "archive").exists())

    def test_process_exit_and_terminal_envelope_must_agree(self):
        for change in ("pending_exit", "nonzero", "launch_error", "missing_end", "extra_end", "error_exit_zero"):
            with self.subTest(change=change):
                folder, trace, command = self.episode(change)
                if change == "pending_exit":
                    command["exit_code"] = None
                elif change == "nonzero":
                    command["exit_code"] = 1
                elif change == "launch_error":
                    command["launch_error"] = "fixture launch failure"
                elif change == "missing_end":
                    trace.pop()
                elif change == "extra_end":
                    trace.insert(-1, deepcopy(trace[-1]))
                else:
                    trace[-1].update(status="ERROR", error_type="RuntimeError")
                write_json(folder / "execute/command.json", command)
                self.save_trace(folder, trace)
                self.reject(folder)

    def test_status_raw_disagreement_is_not_imported(self):
        folder, trace, _ = self.episode(statuses=("UNKNOWN",))
        trace[2]["status"] = "PASS"
        self.save_trace(folder, trace)
        self.reject(folder)

    def test_duplicate_json_keys_rejected_before_output_creation(self):
        folder, _, _ = self.episode()
        (folder / "execute/command.json").write_text('{"exit_code":1,"exit_code":0}', encoding="utf-8")
        self.reject(folder)

    def test_path_guard_rejects_excluded_paths_before_lstat(self):
        for path in ("E:/unused", "/mnt/e/unused", "C:/dev/disassembly_codex_ready_v2/../outside",
                     "C:/dev/disassembly_codex_ready_v2/sub/../../outside", "C:/Windows/Temp/archive"):
            with self.subTest(path=path), patch.object(Path, "lstat") as lstat:
                with self.assertRaises(ValueError):
                    guarded(Path(path))
                lstat.assert_not_called()

    def test_path_guard_rejects_symlink_and_reparse_without_following(self):
        for info in (SimpleNamespace(st_mode=stat.S_IFLNK),
                     SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)):
            with self.subTest(info=info), patch.object(Path, "lstat", return_value=info):
                with self.assertRaises(ValueError):
                    guarded(self.root / "archive")

    def test_archive_cannot_be_inside_input_and_duplicate_runs_are_rejected(self):
        folder, _, _ = self.episode()
        with self.assertRaises(ValueError):
            archive_episodes([folder], self.context, folder / "archive")
        self.assertFalse((folder / "archive").exists())
        with self.assertRaises(ValueError):
            archive_episodes([folder, folder], self.context, self.root / "archive")
        self.assertFalse((self.root / "archive").exists())


if __name__ == "__main__":
    unittest.main()
