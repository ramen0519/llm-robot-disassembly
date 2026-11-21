"""Freeze all predeclared development observations separately by seed (Python 3.8).

This importer never runs physics, reads a pickle, filters failed attempts, or
learns from evaluation traces. Every development run must validate before writes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

from compare_rank_hook import events, read
from evidence_memory import EvidenceStore, POLICY, STATUSES, beam_context, export_snapshot, record_from_pair
from memory_ranker import BEAM_SETUP
from run_rank_hook import COMMIT, build_run_inputs


SEEDS = tuple(range(5))
VARIANTS = ("baseline", "id_ascending", "reverse_native")
RANKERS = {"baseline": "noop", "id_ascending": "id_ascending", "reverse_native": "reverse_native"}
NORMAL_STOPS = ("solution found", "budget reached", "tree fully explored", "timeout")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path_matches(value: str, expected: Path) -> bool:
    # Executed by the worker on WSL; fixtures use local absolute paths too.
    return isinstance(value, str) and Path(value).resolve() == expected.resolve()


def _wrapper_arguments(command: dict) -> dict:
    argv = command.get("command", [])
    _require(isinstance(argv, list) and all(isinstance(token, str) for token in argv),
             "invalid recorded wrapper command")
    positions = [index for index, token in enumerate(argv)
                 if token.replace("\\", "/").endswith("/run_rank_hook.py")]
    _require(len(positions) == 1, "expected one recorded rank hook entrypoint")
    values = argv[positions[0] + 1:]
    _require(len(values) % 2 == 0, "invalid wrapper options")
    options = dict(zip(values[::2], values[1::2]))
    required = {"--asap-root", "--native-log-dir", "--trace-file", "--run-id", "--mode", "--ranker"}
    _require(len(options) * 2 == len(values) and required.issubset(options)
             and set(options).issubset(required | {"--seed"}),
             "unexpected, duplicated, or memory-enabled development wrapper options")
    return options


def _parsed_contract(parsed: dict, stats: dict, native: Path) -> None:
    _require(parsed.get("schema_version") == 1 and parsed.get("raw_stats") == stats
             and parsed.get("trusted_local_tree") is True
             and parsed.get("physics_replayed") is False
             and _path_matches(parsed.get("log_dir"), native), "invalid parsed development provenance")
    _require(type(stats.get("success")) is bool and stats.get("stop_msg") in NORMAL_STOPS,
             "development planner did not complete normally")
    if stats["success"]:
        _require(parsed.get("status") == "PASS" and parsed.get("plan_success") is True
                 and parsed.get("tree_checked") is True and parsed.get("terminal_parts") == ["6"]
                 and parsed.get("issues") == [], "successful development plan lacks checked artifacts")
        sequence = stats.get("sequence")
        _require(isinstance(sequence, list) and len(sequence) == 4
                 and set(sequence) == {"0", "1", "2", "3"}, "invalid development plan sequence")
    else:
        _require(parsed.get("status") == "UNKNOWN" and parsed.get("plan_success") is False
                 and parsed.get("tree_checked") is False and parsed.get("terminal_parts") is None,
                 "unsuccessful development plan must remain UNKNOWN")


def collect_development(root: Path) -> dict:
    """Read all 15 fixed development runs, preserving every paired raw attempt.

    Returned mappings use keys s0..s4. observed_records retains the independent
    (start event, result event, event-file SHA256) inputs for audit consumers.
    """
    root = root.resolve()
    result = dict(seeds=list(SEEDS), contexts={}, records={}, sources={}, observed_records={},
                  total_development_verifier_calls=0)
    all_event_ids, all_run_ids, input_identity = set(), set(), None
    for seed in SEEDS:
        key = "s" + str(seed)
        records, sources, observed = [], [], {}
        for variant in VARIANTS:
            folder = root / "development" / key / variant
            _require(folder.resolve() == folder, "development source must not redirect outside its declared path")
            native = folder / "native" / "dfs-heur-out" / key / "original"
            before_path = root / "development" / ("inputs_before_" + key + "_" + variant) / "stdout.log"
            after_path = root / "development" / ("inputs_after_" + key + "_" + variant) / "stdout.log"
            before, after = read(before_path), read(after_path)
            _require(before == after, "development input/cache changed during run")
            _require(before.get("commit") == COMMIT and before.get("tracked_source_clean") is True
                     and isinstance(before.get("files"), dict) and bool(before["files"]),
                     "unsupported or dirty development source")
            if input_identity is None:
                input_identity = before
            _require(before == input_identity, "mixed development input/cache/source identities")
            setup = read(native / "setup.json")
            _require(setup == BEAM_SETUP, "development setup differs from fixed beam budget/settings")
            context = beam_context(before, setup, seed=seed)
            result["contexts"][key] = context
            command = read(folder / "execute" / "command.json")
            _require(type(command.get("exit_code")) is int and command["exit_code"] == 0
                     and command.get("launch_error") is None, "development execution failed")
            for stream in ("stdout.log", "stderr.log"):
                _require((folder / "execute" / stream).is_file(), "missing raw development command stream")
            options = _wrapper_arguments(command)
            _require(options["--mode"] == "rank" and options["--ranker"] == RANKERS[variant]
                     and options.get("--seed", "0") == str(seed), "development ranker or wrapper seed mismatch")
            _require(_path_matches(options["--native-log-dir"], folder / "native")
                     and _path_matches(options["--trace-file"], folder / "events.jsonl"),
                     "development command redirects to a foreign/evaluation source")
            trace_path = folder / "events.jsonl"
            trace = events(trace_path)
            _require(bool(trace) and trace[0].get("kind") == "run_start"
                     and trace[-1].get("kind") == "run_end"
                     and sum(event.get("kind") == "run_start" for event in trace) == 1
                     and sum(event.get("kind") == "run_end" for event in trace) == 1
                     and trace[-1].get("status") == "returned_from_upstream", "incomplete development run envelope")
            first = trace[0]
            run_id = first.get("run_id")
            suffix = root.name + "/development/" + key + "/" + variant
            _require(isinstance(run_id, str) and (run_id == suffix or run_id.endswith("/" + suffix))
                     and run_id == options["--run-id"] and run_id not in all_run_ids,
                     "foreign, evaluation, or duplicate development run identity")
            all_run_ids.add(run_id)
            argv, _ = build_run_inputs(Path(options["--asap-root"]), folder / "native", seed)
            _require(first.get("native_argv") == argv and type(first.get("seed")) is int
                     and first["seed"] == seed and first.get("source_commit") == COMMIT
                     and first.get("mode") == "rank" and first.get("ranker_name") == RANKERS[variant]
                     and first.get("live_llm_calls") == 0, "development native seed/source/ranker mismatch")
            sha256 = file_hash(trace_path)
            pending, call_ids = None, set()
            count = 0
            for event in trace:
                event_id = event.get("event_id")
                _require(event.get("run_id") == run_id and isinstance(event_id, str)
                         and bool(event_id) and event_id not in all_event_ids,
                         "mixed run or duplicate development event ID")
                all_event_ids.add(event_id)
                _require(not any(name == "memory" or name.startswith("memory_") for name in event),
                         "development source must not use memory")
                if event.get("kind") == "rank_decision":
                    _require(event.get("ranker_name") == RANKERS[variant], "development rank decision mismatch")
                if event.get("kind") == "verifier_start":
                    call_id = event.get("call_id")
                    _require(pending is None and isinstance(call_id, str) and bool(call_id)
                             and call_id not in call_ids, "unpaired or duplicate development verifier start")
                    call_ids.add(call_id)
                    pending = event
                elif event.get("kind") == "verifier_result":
                    _require(pending is not None, "development result without start")
                    record = record_from_pair(context, pending, event, str(trace_path), sha256)
                    records.append(record)
                    observed[event_id] = (pending, event, sha256)
                    pending = None
                    count += 1
            _require(pending is None and count > 0, "incomplete or empty development verifier observations")
            parsed, stats = read(folder / "parsed_result.json"), read(native / "stats.json")
            _parsed_contract(parsed, stats, native)
            artifacts = [folder / "execute" / name for name in ("command.json", "stdout.log", "stderr.log")]
            artifacts += [folder / "parsed_result.json", native / "setup.json", native / "stats.json",
                          before_path, after_path]
            sources.append(dict(seed=seed, variant=variant, events=str(trace_path), sha256=sha256,
                                records=count, run_id=run_id,
                                artifact_sha256={str(path): file_hash(path) for path in artifacts}))
        result["records"][key], result["sources"][key], result["observed_records"][key] = records, sources, observed
        result["total_development_verifier_calls"] += len(records)
    return result


def _write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build(root: Path) -> dict:
    root = root.resolve()
    output = root / "memories"
    if output.exists():
        raise FileExistsError(str(output))
    collected = collect_development(root)  # Validate every run before creating any output.
    output.mkdir(exist_ok=False)
    index = dict(schema_version=1, seeds=list(SEEDS), per_seed={}, record_counts={}, status_counts={},
                 policy=POLICY, allow_hard_pruning=False,
                 total_development_verifier_calls=collected["total_development_verifier_calls"])
    for seed in SEEDS:
        started, key = perf_counter(), "s" + str(seed)
        directory = output / key
        directory.mkdir(exist_ok=False)
        database, snapshot, manifest = (directory / name for name in
            ("source_memory.sqlite", "memory_snapshot.json", "memory_manifest.json"))
        with database.open("xb"):
            pass
        records = collected["records"][key]
        store = EvidenceStore(database)
        try:
            first = store.insert_many(records)
            repeated = store.insert_many(records)
            exported = export_snapshot(store, snapshot)
        finally:
            store.close()
        counts = Counter(record["status"] for record in records)
        status_counts = {status: counts[status] for status in STATUSES}
        result = dict(schema_version=1, seed=seed, context=collected["contexts"][key],
                      sources=collected["sources"][key], policy=POLICY, allow_hard_pruning=False,
                      record_count=len(records), status_counts=status_counts,
                      source_database_sha256=file_hash(database), snapshot_sha256=file_hash(snapshot),
                      snapshot_content_sha256=exported["content_sha256"], first_import=first,
                      repeated_import=repeated, preparation_seconds=perf_counter() - started,
                      preparation_verifier_calls=0, development_verifier_calls=len(records),
                      scope="same beam and seed development observations; soft ranking only")
        _write_new(manifest, result)
        index["per_seed"][key] = dict(snapshot=file_hash(snapshot), source_db=file_hash(database),
                                      manifest=file_hash(manifest))
        index["record_counts"][key], index["status_counts"][key] = len(records), status_counts
    index["created_utc"] = datetime.now(timezone.utc).isoformat()
    _write_new(output / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_dir), indent=2, sort_keys=True))
