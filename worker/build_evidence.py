"""Import paired real verifier events into an append-only evidence store (3.8)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

from compare_rank_hook import SUFFIX, events, read
from evidence_memory import EvidenceStore, beam_context, export_snapshot, record_from_pair
from memory_ranker import BEAM_SETUP


def build(source: Path, variants: list, database: Path, snapshot: Path, manifest: Path) -> dict:
    started = perf_counter()
    for path in (database, snapshot, manifest):
        if path.exists():
            raise FileExistsError(str(path))
    records, sources, context = [], [], None
    for variant in variants:
        folder = source / variant
        before = read(source / ("inputs_before_" + variant) / "stdout.log")
        if before != read(source / ("inputs_after_" + variant) / "stdout.log"):
            raise ValueError("source input/cache mismatch: " + variant)
        current = beam_context(before, read(folder / SUFFIX / "setup.json"), seed=0)
        if context is not None and context != current:
            raise ValueError("mixed source contexts")
        context = current
        path = folder / "events.jsonl"
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        source_events = events(path)
        run_starts = [event for event in source_events if event["kind"] == "run_start"]
        if len(run_starts) != 1:
            raise ValueError("source must identify one official run")
        run_start = run_starts[0]
        argv = list(run_start.get("native_argv", []))[1:]
        if "--log-dir" in argv and argv.index("--log-dir") + 1 < len(argv):
            argv[argv.index("--log-dir") + 1] = "<run-output>"
        expected_argv = ["--dir", "beam_assembly", "--id", "original", "--planner", "dfs",
                         "--generator", "heur-out", "--max-gripper", "2", "--base-part", "6",
                         "--log-dir", "<run-output>", "--early-term"]
        if (run_start.get("source_commit") != before.get("commit")
                or argv != expected_argv or current["setup"] != BEAM_SETUP
                or any(event["run_id"] != run_start["run_id"] for event in source_events)):
            raise ValueError("source events do not match the fixed official beam/seed0 context")
        pending, seen = None, set()
        imported = 0
        for event in source_events:
            if event["event_id"] in seen:
                raise ValueError("duplicate source event id")
            seen.add(event["event_id"])
            if event["kind"] == "verifier_start":
                if pending is not None:
                    raise ValueError("unpaired verifier start")
                pending = event
            elif event["kind"] == "verifier_result":
                if pending is None:
                    raise ValueError("result without start")
                records.append(record_from_pair(current, pending, event, str(path), source_hash))
                pending = None
                imported += 1
        if pending is not None:
            raise ValueError("incomplete verifier attempt")
        sources.append(dict(variant=variant, events=str(path), sha256=source_hash,
                            records=imported))
    if not records:
        raise ValueError("no verifier evidence")
    store = EvidenceStore(database)
    try:
        inserted = store.insert_many(records)
        repeated = store.insert_many(records)
        exported = export_snapshot(store, snapshot)
    finally:
        store.close()
    result = dict(schema_version=1, context=context, sources=sources,
                  policy="positive_only_stable_promotion", allow_hard_pruning=False,
                  source_run=str(source), record_count=len(records),
                  status_counts=dict(Counter(record["status"] for record in records)),
                  first_import=inserted, repeated_import=repeated,
                  snapshot_content_sha256=exported["content_sha256"],
                  source_database_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
                  snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                  preparation_seconds=perf_counter() - started,
                  preparation_verifier_calls=0, prior_source_verifier_calls=len(records),
                  scope="known beam revisit; soft hints only; no verdict cache")
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_run, args.variants, args.db, args.snapshot, args.manifest),
                     indent=2, sort_keys=True))
