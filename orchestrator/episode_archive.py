"""Archive completed local verifier observations without changing ranking memory.

This module never launches a worker, invokes a model, or reads a tree pickle.
Call it only after every evaluation process has terminated. The caller checks
the live input snapshot and frozen ranking sources; this module binds each
trace's solver seed/settings and paired raw observations to that context.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import stat
import sys


PROJECT = Path("C:/dev/disassembly_codex_ready_v2")


def guarded(path: Path) -> Path:
    """Reject excluded lexical paths before any filesystem lookup."""
    text = str(path).replace("\\", "/")
    if ".." in text.split("/") or text.lower().startswith(("e:", "/mnt/e")):
        raise ValueError("excluded archive path")
    path = Path(path).absolute()
    if not path.is_relative_to(PROJECT):
        raise ValueError("archive paths must stay in the C: project")
    for item in reversed((path,) + tuple(path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("archive links and reparse points are excluded")
    return path


# This relative location also selects the copied worker modules when running
# from a frozen implementation directory. Imports below have no physics side effects.
WORKER = guarded(Path(__file__).absolute().parent.parent / "worker")
if str(WORKER) not in sys.path:
    sys.path.insert(0, str(WORKER))
from evidence_memory import (  # noqa: E402
    EvidenceStore, POLICY, STATUSES, beam_context, canonical, digest,
    export_snapshot, record_from_pair,
)
from memory_ranker import BEAM_SETUP  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(text: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite JSON value")
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


def _read(path: Path, hashes: dict, json_lines: bool = False):
    path = guarded(path)
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    if json_lines:
        return [_json(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return _json(raw.decode("utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(guarded(path).read_bytes()).hexdigest()


def _same_native_path(value: str, expected: Path) -> bool:
    # Compare recorded argv as data; never dereference a WSL path on Windows.
    normalized = str(value).replace("\\", "/")
    host = expected.as_posix()
    wsl = "/mnt/c/" + host[3:] if host.lower().startswith("c:/") else host
    return normalized in (host, wsl)


def _solver_binding(start: dict, context: dict, run: Path) -> None:
    seed = context["seed"]
    _require(type(start.get("seed")) is int and start["seed"] == seed,
             "trace seed does not match archive context")
    _require(start.get("source_commit") == context["input_snapshot"].get("commit"),
             "trace commit does not match archive context")
    argv = start.get("native_argv")
    _require(type(argv) is list and len(argv) > 1 and all(type(arg) is str for arg in argv),
             "missing native solver argv")
    options, index = {}, 1
    while index < len(argv):
        key = argv[index]
        _require(key not in options, "duplicate native solver option")
        if key == "--early-term":
            options[key] = True
            index += 1
        else:
            _require(key.startswith("--") and index + 1 < len(argv), "invalid native solver argv")
            options[key] = argv[index + 1]
            index += 2
    expected = {"--dir": "beam_assembly", "--id": "original", "--planner": "dfs",
                "--generator": "heur-out", "--max-gripper": "2", "--base-part": "6",
                "--early-term": True}
    _require(set(options) in (set(expected) | {"--log-dir"}, set(expected) | {"--log-dir", "--seed"})
             and all(options[key] == value for key, value in expected.items())
             and options.get("--seed", "0") == str(seed)
             and _same_native_path(options.get("--log-dir"), run / "native"),
             "native solver settings/path do not match the archived beam run")
    if "memory_context_digest" in start:
        _require(start["memory_context_digest"] == digest(context), "trace memory context mismatch")


def _collect_run(run: Path, context: dict, event_ids: set, run_ids: set):
    hashes = {}
    trace_path = run / "events.jsonl"
    trace = _read(trace_path, hashes, json_lines=True)
    command = _read(run / "execute/command.json", hashes)
    setup = _read(run / ("native/dfs-heur-out/s" + str(context["seed"]) + "/original/setup.json"), hashes)
    _require(canonical(setup) == canonical(context["setup"]), "saved solver setup differs from context")
    _require(type(command) is dict and type(command.get("exit_code")) is int
             and command.get("launch_error") is None, "worker process has not completed successfully launching")
    _require(type(command.get("elapsed_seconds")) in (int, float)
             and math.isfinite(command["elapsed_seconds"]) and command["elapsed_seconds"] >= 0
             and type(command.get("started_at")) is str and bool(command["started_at"]),
             "missing process timing/completion metadata")
    _require(bool(trace) and all(type(event) is dict for event in trace)
             and trace[0].get("kind") == "run_start" and trace[-1].get("kind") == "run_end"
             and sum(event.get("kind") == "run_start" for event in trace) == 1
             and sum(event.get("kind") == "run_end" for event in trace) == 1,
             "incomplete trace envelope")
    start, end = trace[0], trace[-1]
    run_id = start.get("run_id")
    _require(type(run_id) is str and bool(run_id) and run_id not in run_ids, "duplicate or invalid run ID")
    _solver_binding(start, context, run)
    returned = end.get("status") == "returned_from_upstream" and command["exit_code"] == 0
    failed = (end.get("status") == "ERROR" and command["exit_code"] != 0
              and type(end.get("error_type")) is str and bool(end["error_type"]))
    _require(returned or failed, "trace termination and process exit disagree")
    run_ids.add(run_id)
    pending, call_ids, records = None, set(), []
    trace_sha = hashes[str(guarded(trace_path))]
    for event in trace:
        event_id = event.get("event_id")
        _require(event.get("run_id") == run_id and type(event_id) is str
                 and bool(event_id) and event_id not in event_ids, "duplicate or cross-run event identity")
        event_ids.add(event_id)
        if event.get("kind") == "verifier_start":
            call_id = event.get("call_id")
            _require(pending is None and type(call_id) is str and bool(call_id)
                     and call_id not in call_ids, "duplicate or unpaired verifier start")
            call_ids.add(call_id)
            pending = event
        elif event.get("kind") == "verifier_result":
            _require(pending is not None, "verifier result without start")
            records.append(record_from_pair(context, pending, event, str(guarded(trace_path)), trace_sha))
            pending = None
    _require(pending is None and bool(records), "incomplete or empty verifier observations")
    return records, {
        "run_id": run_id, "run_root": str(run), "events": str(guarded(trace_path)),
        "sha256": trace_sha, "artifact_sha256": hashes,
        "seed": context["seed"], "process_exit_code": command["exit_code"],
        "run_end_status": end["status"], "record_count": len(records),
        "status_counts": {status: sum(record["status"] == status for record in records) for status in STATUSES},
    }


def archive_episodes(run_roots: list[Path], context: dict, output_dir: Path) -> dict:
    """Validate all completed episodes, then create one separate append-only archive.

    A completed bounded UNKNOWN is valid evidence. An ERROR episode requires a
    matching nonzero process exit, complete verifier pairs and saved setup. A
    crash without those artifacts is rejected rather than relabelled as evidence.
    Existing output directories are never opened as a mutable evidence store.
    """
    _require(type(run_roots) is list and bool(run_roots), "at least one run is required")
    runs = [guarded(run) for run in run_roots]
    output = guarded(output_dir)
    _require(len(set(runs)) == len(runs), "duplicate run directory")
    _require(not any(output.is_relative_to(run) or run.is_relative_to(output) for run in runs),
             "archive directory must be separate from episode inputs")
    if output.exists():
        raise FileExistsError("archive output already exists")
    _require(output.parent.is_dir() and all(run.is_dir() for run in runs), "missing archive parent or episode")
    _require(type(context) is dict and type(context.get("seed")) is int
             and 0 <= context["seed"] <= 2 ** 32 - 1, "invalid context seed")
    context = _json(canonical(context))
    _require(canonical(context) == canonical(beam_context(context["input_snapshot"], context["setup"], context["seed"]))
             and canonical(context["setup"]) == canonical(BEAM_SETUP), "unsupported beam archive context")
    records, sources, event_ids, run_ids = [], [], set(), set()
    for run in runs:
        collected, source = _collect_run(run, context, event_ids, run_ids)
        records.extend(collected)
        sources.append(source)
    _require(all(_sha(Path(path)) == expected for source in sources
                 for path, expected in source["artifact_sha256"].items()), "episode input changed during collection")
    guarded(output).mkdir(exist_ok=False)
    database, snapshot, manifest_path = (output / name for name in
        ("evaluation_memory.sqlite", "memory_snapshot.json", "memory_manifest.json"))
    # Reserve a new file before SQLite can create or open a store.
    with guarded(database).open("xb"):
        pass
    store = EvidenceStore(database)
    try:
        inserted = store.insert_many(records)
        repeated = store.insert_many(records)
        _require(inserted == {"inserted": len(records), "duplicates": 0}
                 and repeated == {"inserted": 0, "duplicates": len(records)}, "evidence idempotency check failed")
        exported = export_snapshot(store, guarded(snapshot))
    finally:
        store.close()
    _require(all(_sha(Path(path)) == expected for source in sources
                 for path, expected in source["artifact_sha256"].items()), "episode input changed while archiving")
    counts = Counter(record["status"] for record in records)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "context": context, "context_digest": digest(context), "sources": sources,
        "episode_count": len(runs), "record_count": len(records),
        "status_counts": {status: counts[status] for status in STATUSES},
        "source_record_ids": exported["source_record_ids"],
        "first_import": inserted, "repeated_import": repeated,
        "source_database_sha256": _sha(database), "snapshot_sha256": _sha(snapshot),
        "snapshot_content_sha256": exported["content_sha256"],
        "policy": POLICY, "allow_hard_pruning": False,
        "usage": "evaluation_archive_only_not_read_during_evaluation",
        "preparation_verifier_calls": 0,
    }
    with guarded(manifest_path).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest
