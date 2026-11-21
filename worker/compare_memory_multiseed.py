"""Audit paired real beam runs across fixed seeds, retaining seed-scoped memory.

Python 3.8. Only explicitly trusted local tree files may be unpickled. Archive
creation reads completed evaluation logs and never supplies them to the ranker.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import statistics
from time import perf_counter

from compare_memory_runs import (
    POLICY, VARIANTS, PAIRS, _observed, _records_match, _snapshot_valid,
    _source_records, audit_memory_events, digest, file_hash,
)
from compare_rank_hook import events, read, stats_data, tree_data
from compare_rule_rankers import NORMAL_STOPS
from evidence_memory import EvidenceStore, beam_context, export_snapshot, record_from_pair
from memory_ranker import BEAM_SETUP


SEEDS = (0, 1, 2, 3, 4)
SOURCE_RUN = "runs/m4_memory_20261017"
COMMIT = "3727e716b57e0b508c4e9ecd9fefa6953e8f72da"
COPY_FILES = {"snapshot": "memory_snapshot.json", "source_db": "source_memory.sqlite",
              "manifest": "memory_manifest.json"}


def suffix(seed):
    return "native/dfs-heur-out/s{}/original".format(seed)


def identity(seed, variant):
    return "s{}_{}".format(seed, variant)


def expected_cli():
    return ["--dir", "beam_assembly", "--id", "original", "--planner", "dfs",
            "--generator", "heur-out", "--max-gripper", "2", "--base-part", "6",
            "--log-dir", "<run-output>", "--early-term"]


def normalized_cli(start):
    """Normalize output path and an optional explicit seed without hiding flags."""
    argv = start.get("native_argv")
    if not isinstance(argv, list) or not argv or not all(type(arg) is str for arg in argv):
        raise ValueError("invalid native argv")
    if not argv[0].replace("\\", "/").endswith("/plan_sequence/run_seq_plan.py"):
        raise ValueError("unexpected native entrypoint")
    argv = argv[1:]
    if argv.count("--log-dir") != 1 or argv.count("--seed") > 1:
        raise ValueError("duplicate or absent native option")
    log_index = argv.index("--log-dir")
    if log_index + 1 >= len(argv):
        raise ValueError("missing log directory")
    argv[log_index + 1] = "<run-output>"
    seed = 0
    if "--seed" in argv:
        index = argv.index("--seed")
        if index + 1 >= len(argv):
            raise ValueError("missing native seed")
        seed = int(argv[index + 1])
        del argv[index:index + 2]
    return argv, seed


def seed_identity_valid(start, seed):
    try:
        argv, actual = normalized_cli(start)
        return (type(seed) is int and type(start.get("seed")) is int
                and start["seed"] == seed == actual and argv == expected_cli())
    except (TypeError, ValueError, IndexError):
        return False


def load_run(root, seed, variant):
    folder = root / ("s" + str(seed)) / variant
    native = folder / suffix(seed)
    for filename in ("stdout.log", "stderr.log"):
        if not (folder / "execute" / filename).is_file():
            raise FileNotFoundError(str(folder / "execute" / filename))
    trace = events(folder / "events.jsonl")
    return {"command": read(folder / "execute/command.json"),
            "setup": read(native / "setup.json"), "stats": read(native / "stats.json"),
            "tree": tree_data(native), "parsed": read(folder / "parsed_result.json"),
            "events": trace, "native_command": normalized_cli(trace[0])}


def _database_records(path):
    """Read SQLite without invoking the writer/schema constructor."""
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        result = []
        for record_id, payload, stored_hash in connection.execute(
                "SELECT record_id, payload, hash FROM records ORDER BY record_id"):
            record = json.loads(payload)
            if record.get("record_id") != record_id or digest(record) != stored_hash:
                raise ValueError("database record identity/hash mismatch")
            result.append(record)
        return result
    finally:
        connection.close()


def _evaluation_records(root):
    """Collect paired traces with the live seed, never relabel source evidence."""
    records, contexts, sources, observed = [], {}, [], {}
    all_event_ids, run_ids = set(), set()
    for seed in SEEDS:
        for variant in VARIANTS:
            name = identity(seed, variant)
            folder = root / ("s" + str(seed)) / variant
            before = read(root / ("inputs_before_" + name) / "stdout.log")
            after = read(root / ("inputs_after_" + name) / "stdout.log")
            setup = read(folder / suffix(seed) / "setup.json")
            if before != after or setup != BEAM_SETUP:
                raise ValueError("evaluation input/cache/setup mismatch: " + name)
            context = beam_context(before, setup, seed)
            if str(seed) in contexts and contexts[str(seed)] != context:
                raise ValueError("different contexts within seed")
            contexts[str(seed)] = context
            path = folder / "events.jsonl"
            trace = events(path)
            if (not trace or trace[0].get("kind") != "run_start"
                    or trace[-1].get("kind") != "run_end"
                    or trace[-1].get("status") != "returned_from_upstream"
                    or sum(event.get("kind") == "run_start" for event in trace) != 1
                    or sum(event.get("kind") == "run_end" for event in trace) != 1):
                raise ValueError("evaluation run incomplete: " + name)
            start = trace[0]
            if (not seed_identity_valid(start, seed)
                    or start.get("source_commit") != before.get("commit")
                    or start.get("source_commit") != COMMIT
                    or start.get("memory_context_digest") != digest(context)):
                raise ValueError("evaluation seed/CLI/context mismatch: " + name)
            run_id = start.get("run_id")
            if not run_id or run_id in run_ids:
                raise ValueError("duplicate or absent evaluation run ID")
            run_ids.add(run_id)
            source_sha = file_hash(path)
            pending, count = None, 0
            for event in trace:
                event_id = event.get("event_id")
                if not event_id or event_id in all_event_ids or event.get("run_id") != run_id:
                    raise ValueError("duplicate event ID or cross-run event")
                all_event_ids.add(event_id)
                if event["kind"] == "verifier_start":
                    if pending is not None:
                        raise ValueError("unpaired evaluation start")
                    pending = event
                elif event["kind"] == "verifier_result":
                    if pending is None:
                        raise ValueError("evaluation result without start")
                    records.append(record_from_pair(context, pending, event, str(path), source_sha))
                    observed[event_id] = (pending, event, source_sha, context, str(path))
                    pending = None
                    count += 1
            if pending is not None:
                raise ValueError("incomplete verifier attempt")
            sources.append({"seed": seed, "variant": variant, "events": str(path),
                            "sha256": source_sha, "records": count})
    return records, contexts, sources, observed, all_event_ids


def archive_evaluation(root):
    started = perf_counter()
    directory = root / "evaluation_evidence"
    if directory.exists():
        raise FileExistsError(str(directory))
    records, contexts, sources, observed, event_ids = _evaluation_records(root)
    source_ids = set(read(root / "memory_snapshot.json")["source_record_ids"])
    if not records or event_ids.intersection(source_ids):
        raise ValueError("empty evaluation or source/evaluation ID overlap")
    directory.mkdir()
    database, snapshot = directory / "evaluation_memory.sqlite", directory / "memory_snapshot.json"
    store = EvidenceStore(database)
    try:
        inserted = store.insert_many(records)
        repeated = store.insert_many(records)
        exported = export_snapshot(store, snapshot)
    finally:
        store.close()
    manifest = {
        "schema_version": 1, "seeds": list(SEEDS), "variants": list(VARIANTS),
        "contexts_by_seed": contexts, "sources": sources, "record_count": len(records),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "records_by_seed": {str(seed): sum(record["context"]["seed"] == seed for record in records)
                            for seed in SEEDS},
        "first_import": inserted, "repeated_import": repeated,
        "snapshot_sha256": file_hash(snapshot), "snapshot_content_sha256": exported["content_sha256"],
        "source_database_sha256": file_hash(database), "preparation_seconds": perf_counter() - started,
        "preparation_verifier_calls": 0, "policy": POLICY, "allow_hard_pruning": False,
        "usage": "evaluation_archive_only_not_read_during_evaluation",
    }
    _write_json(directory / "memory_manifest.json", manifest)
    return manifest


def archive_records_match(records, observed):
    """Compare archive to raw events and each run's actual seed independently."""
    if len(records) != len(observed) or {record["record_id"] for record in records} != set(observed):
        return False
    for record in records:
        start, result, sha, context, ref = observed[record["record_id"]]
        provenance = record["source"]
        if not (record["request"] == start["request"] and record["raw_result"] == result["raw"]
                and record["status"] == result["status"]
                and record.get("error_type") == result.get("error_type")
                and record["context"] == context and record.get("blocker") is None
                and record.get("path") is None and provenance["run_id"] == result["run_id"]
                and provenance["call_id"] == result["call_id"]
                and provenance["start_event_id"] == start["event_id"]
                and provenance["result_event_id"] == result["event_id"]
                and provenance["sha256"] == sha and provenance["ref"] == ref):
            return False
    return True


def numeric_summary(values):
    available = [value for value in values if type(value) in (int, float)]
    return {"count": len(available), "missing_count": len(values) - len(available),
            "sum": sum(available) if available else None,
            "mean": statistics.mean(available) if available else None,
            "median": statistics.median(available) if available else None,
            "min": min(available) if available else None,
            "max": max(available) if available else None}


def aggregate_results(rows, pairs):
    result = []
    for rule, base_ranker in (("heur", "noop"), ("reverse", "reverse_native")):
        for group, seeds in (("source_seed_revisit", (0,)), ("seed_mismatch", SEEDS[1:])):
            selected_pairs = [pair for pair in pairs if pair["base_rule"] == rule and pair["seed"] in seeds]
            item = {"base_rule": rule, "group": group, "seeds": list(seeds),
                    "planned_pair_count": len(seeds), "pair_count": len(selected_pairs), "modes": {}}
            for mode in ("off", "on"):
                selected = [row for row in rows if row.get("base_ranker") == base_ranker
                            and row.get("memory_mode") == mode and row.get("seed") in seeds]
                item["modes"][mode] = {
                    "planned_run_count": len(seeds), "run_count": len(selected),
                    "success_count": sum(row.get("plan_success") is True for row in selected),
                    "failure_count": sum(row.get("plan_success") is False for row in selected),
                    "unknown_outcome_count": sum(row.get("plan_success") is None for row in selected),
                    "verifier_calls": numeric_summary([row.get("verifier_calls") for row in selected]),
                    "native_time_seconds": numeric_summary([row.get("native_time_seconds") for row in selected]),
                    "memory_evidence_hit_requests": numeric_summary([
                        row.get("memory_evidence_hit_requests") for row in selected]),
                    "memory_changed_requests": numeric_summary([row.get("memory_changed_requests") for row in selected]),
                }
            item["paired_verifier_delta_on_minus_off"] = numeric_summary([
                pair.get("verifier_calls_on_minus_off") for pair in selected_pairs])
            item["paired_native_time_delta_on_minus_off"] = numeric_summary([
                pair.get("native_time_seconds_on_minus_off") for pair in selected_pairs])
            result.append(item)
    return result


def compare(root):
    checks, errors, rows, loaded, snapshots, contexts = {}, [], [], {}, {}, {}
    records, source_context, frozen_sha, protocol = [], None, None, None

    def error(scope, exception):
        checks[scope + ":readable_and_complete"] = False
        errors.append({"scope": scope, "error_type": type(exception).__name__, "message": str(exception)})

    try:
        protocol = read(root / "protocol.json")
        expected_runs = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
        order = [(entry["seed"], entry["variant"]) for entry in protocol["execution_order"]]
        checks["predeclared_seeds_variants_pairs"] = (
            protocol.get("seeds") == list(SEEDS) and protocol.get("variants") == list(VARIANTS)
            and protocol.get("pairs") == [list(pair) for pair in PAIRS]
            and protocol.get("source_run") == SOURCE_RUN and protocol.get("policy") == POLICY
            and protocol.get("allow_hard_pruning") is False and bool(protocol.get("failure_usage")))
        checks["execution_order_covers_twenty_runs_once"] = len(order) == 20 and set(order) == expected_runs
        before, after = read(root / "preserved_before.json"), read(root / "preserved_after.json")
        checks["all_prior_run_artifact_bytes_preserved"] = bool(before) and before == after
        hashes = read(root / "implementation_sha256.json")
        checks["executed_implementation_snapshot_unchanged"] = bool(hashes) and all(
            Path(name).name == name and file_hash(root / "implementation" / name) == value
            for name, value in hashes.items())
        checks["working_implementation_unchanged_during_runs"] = (
            read(root / "implementation_final_sha256.json") == hashes)
    except Exception as exception:
        error("protocol_or_preservation", exception)
    try:
        prior_memory = root.parent / "m4_memory_20261017"
        source_copy = read(root / "source_copy_manifest.json")
        copy_hashes = {key: file_hash(root / name) for key, name in COPY_FILES.items()}
        checks["frozen_source_copied_byte_identical"] = (
            source_copy.get("source_run") == SOURCE_RUN and source_copy.get("hashes") == copy_hashes
            and all(copy_hashes[key] == file_hash(prior_memory / name) for key, name in COPY_FILES.items()))
        frozen = read(root / "memory_snapshot.json")
        manifest = read(root / "memory_manifest.json")
        records, source_context = frozen["records"], manifest["context"]
        frozen_sha = copy_hashes["snapshot"]
        checks["frozen_snapshot_schema_and_policy"] = _snapshot_valid(frozen)
        checks["source_is_fifteen_seed_zero_records"] = (
            len(records) == 15 and all(record["context"] == source_context for record in records)
            and source_context == beam_context(source_context["input_snapshot"], BEAM_SETUP, 0)
            and Counter(record["status"] for record in records) == {"PASS": 12, "UNKNOWN": 3})
        checks["source_manifest_hashes_match"] = (manifest.get("snapshot_sha256") == frozen_sha
            and manifest.get("source_database_sha256") == copy_hashes["source_db"]
            and manifest.get("record_count") == len(records))
        checks["source_database_equals_frozen_snapshot"] = _database_records(root / "source_memory.sqlite") == records
        checks["source_records_equal_original_m4_observations"] = _records_match(
            records, source_context, _source_records(root.parent / "m4_rules_20261017"))
        hashes = read(root / "memory_hashes.json")
        expected = copy_hashes
        checks["frozen_memory_hashes_before_match_files"] = hashes.get("before") == expected
        for seed in SEEDS:
            for variant in VARIANTS:
                name = identity(seed, variant)
                checks[name + ":frozen_memory_unchanged"] = hashes.get("after_" + name) == expected
        checks["frozen_memory_unchanged_after_evaluation_archive"] = hashes.get("after_evaluation_archive") == expected
    except Exception as exception:
        error("frozen_source", exception)

    event_ids, run_ids, request_ids = [], [], []
    for seed in SEEDS:
        for variant in VARIANTS:
            name = identity(seed, variant)
            try:
                before = read(root / ("inputs_before_" + name) / "stdout.log")
                after = read(root / ("inputs_after_" + name) / "stdout.log")
                snapshots[name] = before
                checks[name + ":input_cache_binary_source_unchanged"] = before == after
                data = load_run(root, seed, variant)
                loaded[name] = data
                start, parsed, stats = data["events"][0], data["parsed"], data["stats"]
                context = beam_context(before, data["setup"], seed)
                contexts[name] = context
                checks[name + ":live_seed_cli_match"] = seed_identity_valid(start, seed)
                checks[name + ":supported_setup_and_source"] = (
                    data["setup"] == BEAM_SETUP and start.get("source_commit") == before.get("commit") == COMMIT
                    and before.get("tracked_source_clean") is True and source_context is not None
                    and before == source_context["input_snapshot"])
                checks[name + ":process_exit_zero"] = (data["command"].get("exit_code") == 0
                                                        and data["command"].get("launch_error") is None)
                checks[name + ":normal_planner_termination"] = stats.get("stop_msg") in NORMAL_STOPS
                checks[name + ":parsed_stats_match_native"] = parsed.get("raw_stats") == stats
                checks[name + ":plan_outcome_valid"] = (
                    (stats.get("success") is True and parsed.get("status") == "PASS"
                     and parsed.get("plan_success") is True and parsed.get("tree_checked") is True)
                    or (stats.get("success") is False and parsed.get("status") == "UNKNOWN"
                        and parsed.get("plan_success") is False))
                if not records or frozen_sha is None:
                    raise ValueError("frozen memory unavailable")
                local = {}
                row = audit_memory_events(data, variant, local, records, context, frozen_sha)
                checks.update({"s{}:".format(seed) + key: value for key, value in local.items()})
                row.update(seed=seed, run_key=name,
                           source_seed_match=seed == source_context["seed"],
                           observed_memory_no_hit=row["memory_evidence_hit_requests"] == 0,
                           observed_memory_no_order_change=row["memory_changed_requests"] == 0)
                checks[name + ":unsupported_metrics_are_null"] = row.get("skipped_edges") is None
                rows.append(row)
                event_ids.extend(event["event_id"] for event in data["events"])
                run_ids.append(start["run_id"])
                request_ids.extend(event["request"]["request_id"] for event in data["events"]
                                   if event["kind"] == "rank_decision")
            except Exception as exception:
                error(name, exception)
                rows.append({"variant": variant, "seed": seed, "run_key": name,
                             "base_ranker": "noop" if variant.startswith("heur_") else "reverse_native",
                             "memory_mode": variant.rsplit("_", 1)[1],
                             "plan_status": "ERROR", "plan_success": None})

    checks["all_twenty_run_ids_unique"] = len(run_ids) == 20 and len(set(run_ids)) == 20
    checks["all_evaluation_event_ids_unique"] = bool(event_ids) and len(event_ids) == len(set(event_ids))
    checks["all_rank_request_ids_unique"] = bool(request_ids) and len(request_ids) == len(set(request_ids))
    checks["source_and_evaluation_event_ids_disjoint"] = not set(event_ids).intersection(
        record["record_id"] for record in records)
    checks["identical_initial_input_cache_binary_source_all_runs"] = (len(snapshots) == 20
        and all(value == snapshots.get("s0_heur_off") for value in snapshots.values()))
    evaluation_count = None
    try:
        _, archive_contexts, sources, observed_records, archive_event_ids = _evaluation_records(root)
        directory = root / "evaluation_evidence"
        archive = read(directory / "memory_snapshot.json")
        manifest = read(directory / "memory_manifest.json")
        evaluation_count = len(archive["records"])
        checks["evaluation_archive_snapshot_valid"] = _snapshot_valid(archive)
        checks["evaluation_archive_raw_requests_results_seeds_match"] = archive_records_match(
            archive["records"], observed_records)
        checks["evaluation_archive_disjoint_from_source"] = not set(archive["source_record_ids"]).intersection(
            record["record_id"] for record in records)
        checks["evaluation_database_equals_archive_snapshot"] = _database_records(
            directory / "evaluation_memory.sqlite") == archive["records"]
        checks["evaluation_archive_manifest_matches"] = (
            manifest.get("seeds") == list(SEEDS) and manifest.get("variants") == list(VARIANTS)
            and manifest.get("contexts_by_seed") == archive_contexts and manifest.get("sources") == sources
            and manifest.get("record_count") == evaluation_count
            and manifest.get("snapshot_sha256") == file_hash(directory / "memory_snapshot.json")
            and manifest.get("source_database_sha256") == file_hash(directory / "evaluation_memory.sqlite")
            and manifest.get("preparation_verifier_calls") == 0
            and manifest.get("usage") == "evaluation_archive_only_not_read_during_evaluation")
    except Exception as exception:
        error("evaluation_archive", exception)

    for variant, prior in (("heur_off", "baseline"), ("reverse_off", "reverse_native")):
        name = identity(0, variant)
        try:
            data = loaded[name]
            previous = root.parent / "m4_rules_20261017" / prior
            checks[name + ":prior_m4_setup_equal"] = data["setup"] == read(previous / suffix(0) / "setup.json")
            checks[name + ":prior_m4_stats_except_time_equal"] = (
                {key: value for key, value in data["stats"].items() if key != "time"}
                == stats_data(previous / suffix(0)))
            checks[name + ":prior_m4_ordered_tree_equal"] = data["tree"] == tree_data(previous / suffix(0))
            checks[name + ":prior_m4_physics_observations_equal"] = (
                _observed(root / "s0" / variant / "events.jsonl") == _observed(previous / "events.jsonl"))
            previous_memory = root.parent / "m4_memory_20261017" / variant
            checks[name + ":prior_memory_off_stats_except_time_equal"] = (
                {key: value for key, value in data["stats"].items() if key != "time"}
                == stats_data(previous_memory / suffix(0)))
            checks[name + ":prior_memory_off_ordered_tree_equal"] = (
                data["tree"] == tree_data(previous_memory / suffix(0)))
            if variant == "heur_off":
                native = root.parent / "m3_wsl_20261016/direct" / suffix(0)
                checks[name + ":original_baseline_stats_except_time_equal"] = (
                    {key: value for key, value in data["stats"].items() if key != "time"} == stats_data(native))
                checks[name + ":original_baseline_ordered_tree_equal"] = data["tree"] == tree_data(native)
        except Exception as exception:
            error(name + ":prior_equivalence", exception)

    row_map = {row["run_key"]: row for row in rows}
    pairs = []
    for seed in SEEDS:
        for off, on in PAIRS:
            left_key, right_key = identity(seed, off), identity(seed, on)
            left, right = row_map[left_key], row_map[right_key]
            delta = {"seed": seed, "base_rule": off.split("_", 1)[0], "off": left_key, "on": right_key,
                     "source_seed_match": seed == 0,
                     "plan_success_off": left.get("plan_success"), "plan_success_on": right.get("plan_success"),
                     "sequence_off": left.get("sequence"), "sequence_on": right.get("sequence"),
                     "sequence_changed": left.get("sequence") != right.get("sequence")
                         if left.get("sequence") is not None and right.get("sequence") is not None else None,
                     "ordered_tree_changed": None, "physics_observations_equal": None,
                     "no_memory_hit_observed_on": right.get("observed_memory_no_hit"),
                     "no_memory_order_change_observed_on": right.get("observed_memory_no_order_change")}
            if left_key in loaded and right_key in loaded:
                a, b = loaded[left_key], loaded[right_key]
                checks[left_key + "_vs_" + right_key + ":same_settings_seed_verifier"] = (
                    a["setup"] == b["setup"] and a["native_command"] == b["native_command"]
                    and snapshots.get(left_key) == snapshots.get(right_key)
                    and contexts[left_key] == contexts[right_key])
                delta["ordered_tree_changed"] = a["tree"] != b["tree"]
                delta["physics_observations_equal"] = (
                    _observed(root / ("s" + str(seed)) / off / "events.jsonl")
                    == _observed(root / ("s" + str(seed)) / on / "events.jsonl"))
            else:
                checks[left_key + "_vs_" + right_key + ":both_runs_present"] = False
            for metric in ("verifier_calls", "verifier_PASS", "verifier_UNKNOWN", "verifier_ERROR",
                           "native_n_eval", "native_total_n_eval", "native_time_seconds"):
                delta[metric + "_on_minus_off"] = (right[metric] - left[metric]
                    if type(left.get(metric)) in (int, float) and type(right.get(metric)) in (int, float) else None)
            pairs.append(delta)
    for row in rows:
        prefixes = (row["run_key"] + ":", "s{}:{}:".format(row["seed"], row["variant"]))
        relevant = [value for key, value in checks.items() if key.startswith(prefixes)]
        row["integrity_status"] = "PASS" if relevant and all(relevant) else "ERROR"
    status = "PASS" if checks and all(checks.values()) else "ERROR"
    return {"schema_version": 1, "status": status, "integrity_status": status,
            "all_plans_success": len(rows) == 20 and all(row.get("plan_success") is True for row in rows),
            "checks": checks, "errors": errors, "protocol": protocol, "rows": rows,
            "pair_deltas": pairs, "aggregates": aggregate_results(rows, pairs),
            "source_record_count": len(records), "evaluation_archive_record_count": evaluation_count,
            "source_status_counts": dict(Counter(record["status"] for record in records)),
            "source_seed": 0, "evaluated_seeds": list(SEEDS),
            "scope": "one known beam; seed0 revisit plus seed1-4 compatibility mismatch, not new-assembly generalization",
            "policy": POLICY, "hard_pruning": False, "exact_verdict_cache": False, "physics_bypass": False,
            "live_llm_calls": 0, "internal_geometry_operations": None, "performance_claim": None,
            "notes": {"seed_scope": "seed equality remains required; no relabeling or weakened lookup",
                      "counts": "coarse _simulate calls include leaf; native n_eval excludes leaf",
                      "outcomes": "pair plan/tree/count equality is an observation, not an integrity requirement",
                      "timing": "descriptive only; no significance or general speedup claim",
                      "rng": "ranking preserves RNG; changed action order may change later solver RNG use",
                      "archive": "evaluation evidence saved after all 20 runs, never used by their rankers"}}


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trusted-local-tree", action="store_true")
    parser.add_argument("--archive-only", action="store_true")
    args = parser.parse_args()
    if args.archive_only:
        result = archive_evaluation(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if not args.trusted_local_tree:
        parser.error("only your trusted local tree.pkl may be loaded: explicit trust required")
    for filename in ("comparison.json", "rows.csv"):
        if (args.run_dir / filename).exists():
            raise FileExistsError(str(args.run_dir / filename))
    result = compare(args.run_dir)
    _write_json(args.run_dir / "comparison.json", result)
    fields = ["seed", "variant", "run_key", "integrity_status", "base_ranker", "memory_mode",
              "source_seed_match", "plan_status", "plan_success", "sequence", "native_n_eval",
              "native_total_n_eval", "verifier_calls", "verifier_PASS", "verifier_UNKNOWN", "verifier_ERROR",
              "leaf_verifier_calls", "rank_requests", "rank_fallbacks", "memory_requests",
              "memory_evidence_hit_requests", "memory_positive_hit_requests", "memory_promoted_candidates",
              "memory_changed_requests", "memory_fallbacks", "candidate_yields", "skipped_edges",
              "native_time_seconds", "process_elapsed_seconds", "local_rank_seconds"]
    with (args.run_dir / "rows.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({field: json.dumps(row.get(field)) if isinstance(row.get(field), list)
                             else row.get(field) for field in fields})
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["integrity_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
