"""Importer fixtures verify contracts only; no physics or old run files are read."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from build_seed_memories import COMMIT, RANKERS, SEEDS, VARIANTS, build, collect_development, file_hash
from evidence_memory import EvidenceStore, FrozenMemory
from memory_ranker import BEAM_SETUP
from run_rank_hook import build_run_inputs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_events(path, value):
    path.write_text("".join(json.dumps(event) + "\n" for event in value), encoding="utf-8")


def fixture_run(root, seed, variant):
    key = "s" + str(seed)
    folder = root / "development" / key / variant
    native = folder / "native" / "dfs-heur-out" / key / "original"
    source = root / "fixture_asap"
    run_id = "runs/" + root.name + "/development/" + key + "/" + variant
    snapshot = {"commit": COMMIT, "tracked_source_clean": True,
                "files": {"fixture_mesh_and_binary": "a" * 64}}
    for stage in ("before", "after"):
        write_json(root / "development" / ("inputs_" + stage + "_" + key + "_" + variant) / "stdout.log", snapshot)
    write_json(native / "setup.json", BEAM_SETUP)
    stats = dict(success=True, stop_msg="solution found", sequence=["1", "0", "3", "2"],
                 n_eval=3, total_n_eval=3)
    write_json(native / "stats.json", stats)
    write_json(folder / "parsed_result.json", dict(schema_version=1, raw_stats=stats,
        status="PASS", plan_success=True, tree_checked=True, trusted_local_tree=True,
        physics_replayed=False, terminal_parts=["6"], log_dir=str(native), issues=[]))
    command = ["fixture-python", str(root / "implementation" / "run_rank_hook.py"),
               "--asap-root", str(source), "--native-log-dir", str(folder / "native"),
               "--trace-file", str(folder / "events.jsonl"), "--run-id", run_id,
               "--mode", "rank", "--ranker", RANKERS[variant], "--seed", str(seed)]
    write_json(folder / "execute" / "command.json", dict(command=command, exit_code=0, launch_error=None))
    for stream in ("stdout.log", "stderr.log"):
        (folder / "execute" / stream).write_text("fixture only\n", encoding="utf-8")
    argv, _ = build_run_inputs(source, folder / "native", seed)
    trace = [dict(kind="run_start", run_id=run_id, event_id=run_id + ":start", seed=seed,
                  native_argv=argv, source_commit=COMMIT, mode="rank", ranker_name=RANKERS[variant],
                  live_llm_calls=0)]
    for index, status in enumerate(("PASS", "UNKNOWN", "ERROR")):
        call_id = "v" + str(index)
        trace.append(dict(kind="verifier_start", run_id=run_id, event_id=run_id + ":" + call_id + ":start",
            call_id=call_id, request=dict(part_move="1", parts_rest=["0", "2", "3", "6"],
            parts_removed=[], pose=None, max_grippers=2, timeout=None, optimizer="L-BFGS-B", debug=1, render=False)))
        raw = dict(feasible=status == "PASS", part_move="1", base_part="6", pose=None,
                   action=[0, 0, 1] if status == "PASS" else None,
                   parts_fix=[] if status == "PASS" else None, grasp=None)
        event = dict(kind="verifier_result", run_id=run_id, event_id=run_id + ":" + call_id + ":result",
                     call_id=call_id, status=status, raw=None if status == "ERROR" else raw,
                     blocker=None, path=None)
        if status == "ERROR":
            event["error_type"] = "RuntimeError"
        trace.append(event)
    trace.append(dict(kind="run_end", run_id=run_id, event_id=run_id + ":end", status="returned_from_upstream"))
    write_events(folder / "events.jsonl", trace)


class SeedMemoriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "fixture_experiment"
        for seed in SEEDS:
            for variant in VARIANTS:
                fixture_run(self.root, seed, variant)
        self.folder = self.root / "development/s4/reverse_native"
        self.trace_path = self.folder / "events.jsonl"

    def trace(self):
        return [json.loads(line) for line in self.trace_path.read_text(encoding="utf-8").splitlines()]

    def rejected(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            build(self.root)
        self.assertFalse((self.root / "memories").exists())

    def test_all_seeds_all_raw_statuses_provenance_and_duplicate_import_preserved(self):
        source_hashes = {str(path): file_hash(path) for path in (self.root / "development").rglob("*") if path.is_file()}
        collected = collect_development(self.root)
        index = build(self.root)
        self.assertEqual(index["seeds"], list(SEEDS))
        self.assertEqual(index["total_development_verifier_calls"], 45)
        self.assertFalse(index["allow_hard_pruning"])
        for seed in SEEDS:
            key = "s" + str(seed)
            directory = self.root / "memories" / key
            manifest = json.loads((directory / "memory_manifest.json").read_text())
            snapshot = json.loads((directory / "memory_snapshot.json").read_text())
            self.assertEqual(manifest["context"]["seed"], seed)
            self.assertEqual(manifest["status_counts"], {"PASS": 3, "UNKNOWN": 3, "ERROR": 3})
            self.assertEqual(manifest["first_import"], {"inserted": 9, "duplicates": 0})
            self.assertEqual(manifest["repeated_import"], {"inserted": 0, "duplicates": 9})
            self.assertEqual({record["record_id"] for record in snapshot["records"]}, set(collected["observed_records"][key]))
            self.assertEqual(index["per_seed"][key], {
                "source_db": file_hash(directory / "source_memory.sqlite"),
                "snapshot": file_hash(directory / "memory_snapshot.json"),
                "manifest": file_hash(directory / "memory_manifest.json")})
            self.assertEqual(manifest["sources"], collected["sources"][key])
            store = EvidenceStore(directory / "source_memory.sqlite")
            try:
                self.assertEqual(store.records(), snapshot["records"])
            finally:
                store.close()
            FrozenMemory(directory / "memory_snapshot.json")
            for record in snapshot["records"]:
                start, result, sha = collected["observed_records"][key][record["record_id"]]
                self.assertEqual(record["raw_result"], result["raw"])
                self.assertEqual(record["request"], start["request"])
                self.assertEqual(record["source"]["sha256"], sha)
                self.assertEqual(record["source"]["ref"], next(source["events"] for source in manifest["sources"]
                    if source["run_id"] == record["source"]["run_id"]))
        self.assertEqual(source_hashes, {str(path): file_hash(path) for path in (self.root / "development").rglob("*") if path.is_file()})

    def test_source_seed_cannot_be_relabelled(self):
        trace = self.trace()
        trace[0]["seed"] = 0
        write_events(self.trace_path, trace)
        self.rejected()

    def test_native_seed_must_match_recorded_context_seed(self):
        trace = self.trace()
        trace[0]["native_argv"][-1] = "3"
        write_events(self.trace_path, trace)
        self.rejected()

    def test_wrapper_seed_must_match_native_seed(self):
        path = self.folder / "execute/command.json"
        command = json.loads(path.read_text())
        command["command"][-1] = "3"
        write_json(path, command)
        self.rejected()

    def test_incomplete_final_run_blocks_all_seed_outputs(self):
        write_events(self.trace_path, self.trace()[:-1])
        self.rejected()

    def test_unpaired_attempt_rejected_before_any_freeze(self):
        trace = self.trace()
        del trace[-2]
        write_events(self.trace_path, trace)
        self.rejected()

    def test_duplicate_event_across_development_runs_rejected(self):
        trace = self.trace()
        other = self.root / "development/s0/baseline/events.jsonl"
        trace[2]["event_id"] = json.loads(other.read_text().splitlines()[2])["event_id"]
        write_events(self.trace_path, trace)
        self.rejected()

    def test_mixed_input_identity_rejected(self):
        for stage in ("before", "after"):
            path = self.root / "development" / ("inputs_" + stage + "_s4_reverse_native/stdout.log")
            value = json.loads(path.read_text())
            value["files"]["fixture_mesh_and_binary"] = "b" * 64
            write_json(path, value)
        self.rejected()

    def test_memory_enabled_development_trace_rejected(self):
        trace = self.trace()
        trace[0]["memory_mode"] = "on"
        write_events(self.trace_path, trace)
        self.rejected()

    def test_memory_off_wrapper_also_rejected_as_development_source(self):
        path = self.folder / "execute/command.json"
        value = json.loads(path.read_text())
        value["command"] += ["--memory-mode", "off", "--memory-snapshot", "evaluation.json"]
        write_json(path, value)
        self.rejected()

    def test_evaluation_trace_command_injection_rejected(self):
        path = self.folder / "execute/command.json"
        value = json.loads(path.read_text())
        position = value["command"].index("--trace-file")
        value["command"][position + 1] = str(self.root / "s4/reverse_on/events.jsonl")
        write_json(path, value)
        self.rejected()

    def test_evaluation_run_identity_injection_rejected(self):
        trace = self.trace()
        for event in trace:
            event["run_id"] = "runs/fixture_experiment/s4/reverse_on"
        write_events(self.trace_path, trace)
        path = self.folder / "execute/command.json"
        value = json.loads(path.read_text())
        position = value["command"].index("--run-id")
        value["command"][position + 1] = trace[0]["run_id"]
        write_json(path, value)
        self.rejected()

    def test_parsed_result_provenance_mismatch_rejected(self):
        path = self.folder / "parsed_result.json"
        parsed = json.loads(path.read_text())
        parsed["raw_stats"]["n_eval"] = 999
        write_json(path, parsed)
        self.rejected()

    def test_dirty_source_or_changed_settings_rejected(self):
        path = self.root / "development/inputs_before_s4_reverse_native/stdout.log"
        value = json.loads(path.read_text())
        value["tracked_source_clean"] = False
        write_json(path, value)
        self.rejected()

    def test_failed_execution_is_not_hidden_by_successful_plan_json(self):
        path = self.folder / "execute/command.json"
        command = json.loads(path.read_text())
        command["exit_code"] = 1
        write_json(path, command)
        self.rejected()

    def test_normal_no_plan_keeps_all_observations(self):
        native = self.folder / "native/dfs-heur-out/s4/original"
        stats = dict(success=False, stop_msg="budget reached", sequence=None, n_eval=3, total_n_eval=3)
        write_json(native / "stats.json", stats)
        path = self.folder / "parsed_result.json"
        parsed = json.loads(path.read_text())
        parsed.update(raw_stats=stats, status="UNKNOWN", plan_success=False, tree_checked=False,
                      terminal_parts=None, issues=["no complete plan"])
        write_json(path, parsed)
        index = build(self.root)
        self.assertEqual(index["record_counts"]["s4"], 9)

    def test_existing_freeze_is_not_overwritten(self):
        build(self.root)
        before = {str(path): file_hash(path) for path in (self.root / "memories").rglob("*") if path.is_file()}
        with self.assertRaises(FileExistsError):
            build(self.root)
        self.assertEqual(before, {str(path): file_hash(path) for path in (self.root / "memories").rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
