"""Audit separate development/frozen-memory/evaluation stages (Python 3.8).

A complete unsuccessful plan remains in the denominator. Only trusted local
trees may be unpickled. Evaluation records never become ranking inputs.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
from time import perf_counter

from build_seed_memories import collect_development
from compare_memory_multiseed import (
    COMMIT, SEEDS, VARIANTS, _database_records, _evaluation_records, archive_records_match,
    identity, load_run, numeric_summary, seed_identity_valid, suffix,
)
from compare_memory_runs import (
    PAIRS, POLICY, _observed, _records_match, _snapshot_valid,
    audit_memory_events, file_hash,
)
from compare_rank_hook import read, stats_data, tree_data
from compare_rule_rankers import NORMAL_STOPS, _audit_events
from evidence_memory import EvidenceStore, beam_context, export_snapshot
from memory_ranker import BEAM_SETUP

DEVELOPMENT_VARIANTS = ("baseline", "id_ascending", "reverse_native")
MEMORY_FILES = {"snapshot": "memory_snapshot.json", "source_db": "source_memory.sqlite",
                "manifest": "memory_manifest.json"}


def memory_hashes(root):
    return {"s" + str(seed): {key: file_hash(root / "memories" / ("s" + str(seed)) / filename)
                             for key, filename in MEMORY_FILES.items()} for seed in SEEDS}


def frozen_source_ids(root):
    ids = []
    for seed in SEEDS:
        snapshot = read(root / "memories" / ("s" + str(seed)) / "memory_snapshot.json")
        if not _snapshot_valid(snapshot):
            raise ValueError("invalid per-seed frozen memory")
        if any(record["context"]["seed"] != seed for record in snapshot["records"]):
            raise ValueError("frozen source has another seed")
        ids.extend(snapshot["source_record_ids"])
    if len(ids) != len(set(ids)):
        raise ValueError("source record IDs overlap across seeds")
    return set(ids)


def archive_evaluation(root):
    started = perf_counter()
    directory = root / "evaluation_evidence"
    if directory.exists():
        raise FileExistsError(str(directory))
    records, contexts, sources, observed, event_ids = _evaluation_records(root)
    if not records or event_ids.intersection(frozen_source_ids(root)):
        raise ValueError("evaluation empty or overlaps development evidence IDs")
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
    write_json(directory / "memory_manifest.json", manifest)
    return manifest


def audit_seed_source(seed, snapshot, manifest, context, observed):
    records = snapshot.get("records", [])
    counts = Counter(record["status"] for record in records)
    return {
        "snapshot_schema_and_policy": _snapshot_valid(snapshot),
        "single_expected_seed_context": bool(records) and context.get("seed") == seed
            and all(record.get("context") == context for record in records),
        "manifest_context_count_and_statuses": manifest.get("seed") == seed
            and manifest.get("context") == context and manifest.get("record_count") == len(records)
            and manifest.get("status_counts") == {status: counts[status] for status in ("PASS", "UNKNOWN", "ERROR")}
            and manifest.get("policy") == POLICY and manifest.get("allow_hard_pruning") is False,
        "records_equal_all_development_requests_and_raw_results": _records_match(records, context, observed),
    }


def plan_outcome_valid(data):
    parsed, stats = data["parsed"], data["stats"]
    return ((stats.get("success") is True and parsed.get("status") == "PASS"
             and parsed.get("plan_success") is True and parsed.get("tree_checked") is True)
            or (stats.get("success") is False and parsed.get("status") == "UNKNOWN"
                and parsed.get("plan_success") is False))


def validation_exit_valid(data, exit_code):
    return plan_outcome_valid(data) and exit_code == (0 if data["stats"]["success"] else 1)


def audit_outcome(data, seed):
    start, parsed, stats = data["events"][0], data["parsed"], data["stats"]
    return {
        "seed_and_native_cli_match": seed_identity_valid(start, seed),
        "official_setup_and_source": data["setup"] == BEAM_SETUP and start.get("source_commit") == COMMIT,
        "process_exit_zero": data["command"].get("exit_code") == 0 and data["command"].get("launch_error") is None,
        "normal_planner_termination": stats.get("stop_msg") in NORMAL_STOPS,
        "parsed_stats_match_native": parsed.get("raw_stats") == stats,
        "plan_outcome_valid": plan_outcome_valid(data),
    }


def audit_development_events(data, variant, checks):
    if variant == "baseline":
        row = _audit_events(data, "dev_noop", checks, expected_ranker="noop",
                            ordering_function=lambda event, generation: list(generation["native_order"]))
    else:
        row = _audit_events(data, variant, checks)
    row["variant"] = variant
    checks[variant + ":development_memory_absent"] = all(
        "memory_mode" not in event and "memory_context_digest" not in event
        and "memory_snapshot_sha256" not in event and "memory" not in event
        for event in data["events"])
    return row


def aggregate_results(rows, pairs):
    result = []
    for rule, ranker in (("heur", "noop"), ("reverse", "reverse_native")):
        selected_pairs = [pair for pair in pairs if pair["base_rule"] == rule]
        item = {"base_rule": rule, "seeds": list(SEEDS), "planned_pair_count": 5,
                "pair_count": len(selected_pairs), "modes": {}}
        for mode in ("off", "on"):
            selected = [row for row in rows if row.get("base_ranker") == ranker
                        and row.get("memory_mode") == mode]
            item["modes"][mode] = {
                "planned_run_count": 5, "run_count": len(selected),
                "success_count": sum(row.get("plan_success") is True for row in selected),
                "failure_count": sum(row.get("plan_success") is False for row in selected),
                "unknown_outcome_count": sum(row.get("plan_success") is None for row in selected),
                **{metric: numeric_summary([row.get(metric) for row in selected]) for metric in (
                    "verifier_calls", "native_time_seconds", "memory_evidence_hit_requests", "memory_changed_requests")},
            }
        for metric in ("verifier_calls", "native_time_seconds"):
            item["paired_" + metric + "_on_minus_off"] = numeric_summary([
                pair.get(metric + "_on_minus_off") for pair in selected_pairs])
        result.append(item)
    return result


def _stamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed


def _command_window(record):
    start = _stamp(record["started_at"])
    duration = record["elapsed_seconds"]
    if type(duration) not in (int, float) or duration < 0:
        raise ValueError("invalid command duration")
    return start, start + timedelta(seconds=duration)


def audit_timeline(root, protocol, loaded):
    checks, details = {}, {}
    with (root / "progress.jsonl").open(encoding="utf-8") as stream:
        progress = [json.loads(line) for line in stream if line.strip()]
    times = [_stamp(event["utc"]) for event in progress]
    checks["progress_time_monotonic"] = all(a <= b for a, b in zip(times, times[1:]))
    expected = []
    for entry in protocol["development_order"]:
        expected.extend((kind, entry["seed"], entry["variant"])
                        for kind in ("development_start", "development_end"))
    expected.append(("memory_freeze", None, None))
    for entry in protocol["execution_order"]:
        expected.extend((kind, entry["seed"], entry["variant"]) for kind in ("run_start", "run_end"))
    relevant = [event for event in progress if event["kind"] != "comparison_end"]
    observed = [(event["kind"], event.get("seed"), event.get("variant")) for event in relevant]
    checks["development_freeze_evaluation_order_matches_protocol"] = observed == expected
    if observed != expected:
        raise ValueError("recorded stage order differs from protocol")
    last_development, first_evaluation, last_evaluation = None, None, None
    for stage, names, offset in (("development", protocol["development_order"], 0),
                                 ("evaluation", protocol["execution_order"], 31)):
        for index, entry in enumerate(names):
            begin, end = relevant[offset + 2 * index:offset + 2 * index + 2]
            seed, variant = entry["seed"], entry["variant"]
            key = stage + ":" + identity(seed, variant)
            start_utc, end_utc = _stamp(begin["utc"]), _stamp(end["utc"])
            data = loaded[key]
            actual_start, actual_end = _command_window(data["command"])
            checks[key + ":command_within_progress_window"] = start_utc <= actual_start <= actual_end <= end_utc
            checks[key + ":indices_and_exit_codes"] = (
                begin.get("index") == end.get("index") == index + 1
                and begin.get("total") == end.get("total") == len(names)
                and end.get("process_exit_code") == 0
                and validation_exit_valid(data, end.get("validation_exit_code")))
            if stage == "development":
                last_development = end_utc
            else:
                if first_evaluation is None:
                    first_evaluation = start_utc
                last_evaluation = end_utc
    frozen = read(root / "memories_frozen.json")
    freeze_time, freeze_event_time = _stamp(frozen["utc"]), _stamp(relevant[30]["utc"])
    source_build = read(root / "source_build/command.json")
    source_start, source_end = _command_window(source_build)
    checks["source_build_completed_between_development_and_freeze"] = (
        source_build.get("exit_code") == 0 and source_build.get("launch_error") is None
        and last_development <= source_start <= source_end <= freeze_time <= freeze_event_time <= first_evaluation)
    archive = read(root / "evaluation_archive/command.json")
    archive_start, archive_end = _command_window(archive)
    checks["evaluation_archive_started_after_all_evaluation_runs"] = (
        archive.get("exit_code") == 0 and archive.get("launch_error") is None and last_evaluation <= archive_start)
    # This comparator's recorder is still unfinished while this function runs.
    comparator = read(root / "comparison/command.json")
    checks["comparison_started_after_archive_completed"] = archive_end <= _stamp(comparator["started_at"])
    details.update(last_development_end=last_development.isoformat(), source_build_start=source_start.isoformat(),
                   source_build_end=source_end.isoformat(), memory_freeze=freeze_time.isoformat(),
                   first_evaluation_start=first_evaluation.isoformat(), last_evaluation_end=last_evaluation.isoformat(),
                   archive_start=archive_start.isoformat())
    return checks, details


def compare(root):
    checks, errors, rows, development_rows, loaded, snapshots = {}, [], [], [], {}, {}
    sources, records_by_seed, contexts_by_seed, hashes, protocol = None, {}, {}, None, None
    timeline, evaluation_count, source_build_seconds = None, None, None
    import_preparation_seconds = []

    def error(scope, exception):
        checks[scope + ":readable_and_complete"] = False
        errors.append({"scope": scope, "error_type": type(exception).__name__, "message": str(exception)})

    def add(prefix, local):
        checks.update({prefix + ":" + key.split(":", 1)[-1]: bool(value) for key, value in local.items()})

    try:
        protocol = read(root / "protocol.json")
        checks["predeclared_seeds_variants_pairs"] = (
            protocol.get("seeds") == list(SEEDS) and protocol.get("variants") == list(VARIANTS)
            and protocol.get("pairs") == [list(pair) for pair in PAIRS]
            and protocol.get("development_variants") == list(DEVELOPMENT_VARIANTS)
            and protocol.get("policy") == POLICY and protocol.get("allow_hard_pruning") is False)
        for label, variants in (("development", DEVELOPMENT_VARIANTS), ("execution", VARIANTS)):
            order = [(item["seed"], item["variant"]) for item in protocol[label + "_order"]]
            checks[label + "_order_complete_once"] = (len(order) == len(SEEDS) * len(variants)
                and set(order) == {(seed, variant) for seed in SEEDS for variant in variants})
        before, after = read(root / "preserved_before.json"), read(root / "preserved_after.json")
        checks["all_prior_result_bytes_preserved"] = bool(before) and before == after
        implementation = read(root / "implementation_sha256.json")
        checks["frozen_implementation_bytes_preserved"] = bool(implementation) and all(
            Path(name).name == name and file_hash(root / "implementation" / name) == value
            for name, value in implementation.items())
        checks["working_implementation_unchanged"] = implementation == read(root / "implementation_final_sha256.json")
    except Exception as exception:
        error("protocol_and_preservation", exception)

    try:
        sources = collect_development(root)
        source_build_seconds = read(root / "source_build/command.json").get("elapsed_seconds")
        hashes = memory_hashes(root)
        index, frozen = read(root / "memories/index.json"), read(root / "memories_frozen.json")
        checks["index_policy_and_hashes_match_files"] = (
            index.get("seeds") == list(SEEDS) and index.get("policy") == POLICY
            and index.get("allow_hard_pruning") is False and index.get("per_seed") == hashes)
        checks["frozen_manifest_matches_current_memories_and_index"] = (
            frozen.get("hashes") == hashes and frozen.get("index_sha256") == file_hash(root / "memories/index.json"))
        checkpoints = read(root / "memory_hashes.json")
        keys = ["before", "after_evaluation_archive"] + ["after_" + identity(seed, variant)
                for seed in SEEDS for variant in VARIANTS]
        for key in keys:
            checks["memory_checkpoint:" + key] = checkpoints.get(key) == hashes
        for seed in SEEDS:
            key, directory = "s" + str(seed), root / "memories" / ("s" + str(seed))
            snapshot, manifest = read(directory / "memory_snapshot.json"), read(directory / "memory_manifest.json")
            import_preparation_seconds.append(manifest.get("preparation_seconds"))
            records_by_seed[key], contexts_by_seed[key] = snapshot["records"], sources["contexts"][key]
            add("source:" + key, audit_seed_source(seed, snapshot, manifest, contexts_by_seed[key],
                                                    sources["observed_records"][key]))
            checks["source:" + key + ":database_equals_snapshot"] = _database_records(directory / "source_memory.sqlite") == snapshot["records"]
            checks["source:" + key + ":manifest_hashes_and_source_files_match"] = (
                manifest.get("sources") == sources["sources"][key]
                and manifest.get("snapshot_sha256") == hashes[key]["snapshot"]
                and manifest.get("source_database_sha256") == hashes[key]["source_db"]
                and manifest.get("development_verifier_calls") == len(snapshot["records"])
                and index.get("record_counts", {}).get(key) == len(snapshot["records"])
                and index.get("status_counts", {}).get(key) == manifest.get("status_counts"))
        source_ids = [record["record_id"] for records in records_by_seed.values() for record in records]
        checks["source_record_ids_globally_unique"] = bool(source_ids) and len(source_ids) == len(set(source_ids))
        checks["source_cost_count_matches_all_development_verifier_results"] = (
            index.get("total_development_verifier_calls") == sources["total_development_verifier_calls"] == len(source_ids))
    except Exception as exception:
        error("source_memories", exception)

    event_ids, request_ids, run_ids = [], [], []
    for stage, variants, container in (("development", DEVELOPMENT_VARIANTS, development_rows),
                                       ("evaluation", VARIANTS, rows)):
        stage_root = root / "development" if stage == "development" else root
        for seed in SEEDS:
            for variant in variants:
                name = stage + ":" + identity(seed, variant)
                row = {"variant": variant, "seed": seed, "run_key": name,
                       "plan_status": "ERROR", "plan_success": None}
                if stage == "evaluation":
                    row.update(base_ranker="noop" if variant.startswith("heur_") else "reverse_native",
                               memory_mode=variant.rsplit("_", 1)[1])
                try:
                    before = read(stage_root / ("inputs_before_" + identity(seed, variant)) / "stdout.log")
                    after = read(stage_root / ("inputs_after_" + identity(seed, variant)) / "stdout.log")
                    snapshots[name] = before
                    checks[name + ":input_cache_binary_source_unchanged"] = before == after
                    data = load_run(stage_root, seed, variant)
                    add(name, audit_outcome(data, seed))
                    context = beam_context(before, data["setup"], seed)
                    checks[name + ":context_matches_own_seed_source"] = context == contexts_by_seed.get("s" + str(seed))
                    checks[name + ":tracked_official_source_clean"] = before.get("tracked_source_clean") is True and before.get("commit") == COMMIT
                    validation = read(stage_root / ("s" + str(seed)) / variant / "validation/command.json")
                    checks[name + ":validation_exit_agrees_with_bounded_outcome"] = (
                        validation.get("launch_error") is None and validation_exit_valid(data, validation.get("exit_code")))
                    local = {}
                    if stage == "development":
                        measured = audit_development_events(data, variant, local)
                    else:
                        key = "s" + str(seed)
                        measured = audit_memory_events(data, variant, local, records_by_seed[key], context, hashes[key]["snapshot"])
                        argv = data["command"]["command"]
                        value = argv[argv.index("--memory-snapshot") + 1]
                        checks[name + ":loaded_only_own_seed_frozen_snapshot"] = (
                            argv.count("--memory-snapshot") == 1
                            and value.replace("\\", "/") == (root / "memories" / key / "memory_snapshot.json").as_posix())
                    add(name, local)
                    measured.update(seed=seed, run_key=name)
                    row = measured
                    loaded[name] = data
                    event_ids.extend(event["event_id"] for event in data["events"])
                    request_ids.extend(event["request"]["request_id"] for event in data["events"] if event["kind"] == "rank_decision")
                    run_ids.append(data["events"][0]["run_id"])
                except Exception as exception:
                    error(name, exception)
                container.append(row)
    checks["all_35_run_ids_unique"] = len(run_ids) == 35 and len(set(run_ids)) == 35
    checks["all_stage_event_ids_unique"] = bool(event_ids) and len(event_ids) == len(set(event_ids))
    checks["all_stage_rank_request_ids_unique"] = bool(request_ids) and len(request_ids) == len(set(request_ids))
    checks["same_initial_input_cache_binary_source_for_all_35_runs"] = (len(snapshots) == 35
        and all(value == snapshots.get("development:s0_baseline") for value in snapshots.values()))

    try:
        _, contexts, archive_sources, observed, evaluation_ids = _evaluation_records(root)
        directory = root / "evaluation_evidence"
        snapshot, manifest = read(directory / "memory_snapshot.json"), read(directory / "memory_manifest.json")
        evaluation_count = len(snapshot["records"])
        source_ids = frozen_source_ids(root)
        checks["evaluation_archive_schema_valid"] = _snapshot_valid(snapshot)
        checks["evaluation_raw_events_and_actual_seeds_preserved"] = archive_records_match(snapshot["records"], observed)
        checks["evaluation_and_all_development_evidence_disjoint"] = not evaluation_ids.intersection(source_ids)
        checks["evaluation_database_matches_snapshot"] = _database_records(directory / "evaluation_memory.sqlite") == snapshot["records"]
        checks["evaluation_archive_manifest_matches"] = (
            manifest.get("contexts_by_seed") == contexts and manifest.get("sources") == archive_sources
            and manifest.get("seeds") == list(SEEDS) and manifest.get("variants") == list(VARIANTS)
            and manifest.get("record_count") == evaluation_count
            and manifest.get("snapshot_sha256") == file_hash(directory / "memory_snapshot.json")
            and manifest.get("source_database_sha256") == file_hash(directory / "evaluation_memory.sqlite")
            and manifest.get("preparation_verifier_calls") == 0
            and manifest.get("usage") == "evaluation_archive_only_not_read_during_evaluation")
    except Exception as exception:
        error("evaluation_archive", exception)

    pairs = []
    row_map = {(row["seed"], row["variant"]): row for row in rows}
    for seed in SEEDS:
        for off, on in PAIRS:
            a_name, b_name = "evaluation:" + identity(seed, off), "evaluation:" + identity(seed, on)
            left, right = row_map[seed, off], row_map[seed, on]
            delta = {"seed": seed, "base_rule": off.split("_", 1)[0], "off": a_name, "on": b_name,
                     "plan_success_off": left.get("plan_success"), "plan_success_on": right.get("plan_success"),
                     "sequence_off": left.get("sequence"), "sequence_on": right.get("sequence"),
                     "sequence_changed": left.get("sequence") != right.get("sequence")
                        if left.get("sequence") is not None and right.get("sequence") is not None else None,
                     "ordered_tree_changed": None, "physics_observations_equal": None}
            if a_name in loaded and b_name in loaded:
                a, b = loaded[a_name], loaded[b_name]
                checks[a_name + "_vs_" + b_name + ":same_off_on_conditions"] = (
                    a["setup"] == b["setup"] and a["native_command"] == b["native_command"]
                    and snapshots[a_name] == snapshots[b_name])
                delta["ordered_tree_changed"] = a["tree"] != b["tree"]
                delta["physics_observations_equal"] = _observed(root / ("s" + str(seed)) / off / "events.jsonl") == _observed(root / ("s" + str(seed)) / on / "events.jsonl")
            else:
                checks[a_name + "_vs_" + b_name + ":both_runs_present"] = False
            for metric in ("verifier_calls", "verifier_PASS", "verifier_UNKNOWN", "verifier_ERROR", "native_n_eval", "native_total_n_eval", "native_time_seconds"):
                delta[metric + "_on_minus_off"] = right[metric] - left[metric] if type(left.get(metric)) in (int, float) and type(right.get(metric)) in (int, float) else None
            pairs.append(delta)
            try:
                data = loaded[a_name]
                prior = root.parent / "memory_multiseed_20261017" / ("s" + str(seed)) / off
                checks[a_name + ":previous_multiseed_off_setup_equal"] = data["setup"] == read(prior / suffix(seed) / "setup.json")
                checks[a_name + ":previous_multiseed_off_stats_except_time_equal"] = (
                    {key: value for key, value in data["stats"].items() if key != "time"} == stats_data(prior / suffix(seed)))
                checks[a_name + ":previous_multiseed_off_ordered_tree_equal"] = data["tree"] == tree_data(prior / suffix(seed))
                checks[a_name + ":previous_multiseed_off_physics_observations_equal"] = _observed(root / ("s" + str(seed)) / off / "events.jsonl") == _observed(prior / "events.jsonl")
            except Exception as exception:
                error(a_name + ":previous_off", exception)
    try:
        local, timeline = audit_timeline(root, protocol, loaded)
        checks.update({"timeline:" + key: value for key, value in local.items()})
    except Exception as exception:
        error("timeline", exception)

    for row in development_rows + rows:
        local = [value for key, value in checks.items() if key.startswith(row["run_key"] + ":")]
        row["integrity_status"] = "PASS" if local and all(local) else "ERROR"
    status = "PASS" if checks and all(checks.values()) else "ERROR"
    return {
        "schema_version": 1, "status": status, "integrity_status": status,
        "all_plans_success": len(rows) == 20 and all(row.get("plan_success") is True for row in rows),
        "all_development_plans_success": len(development_rows) == 15 and all(row.get("plan_success") is True for row in development_rows),
        "checks": checks, "errors": errors, "protocol": protocol, "timeline": timeline,
        "development_rows": development_rows, "rows": rows, "pair_deltas": pairs,
        "aggregates": aggregate_results(rows, pairs),
        "costs": {
            "development_runs": 15, "online_evaluation_runs": 20,
            **{stage + "_" + metric: numeric_summary([row.get(metric) for row in container])
               for stage, container in (("development", development_rows), ("online", rows))
               for metric in ("verifier_calls", "native_time_seconds", "process_elapsed_seconds", "local_rank_seconds")},
            "source_import_verifier_calls": 0, "evaluation_archive_verifier_calls": 0,
            "source_build_process_seconds": source_build_seconds,
            "source_import_preparation_seconds": numeric_summary(import_preparation_seconds),
            "source_record_count": sum(len(records) for records in records_by_seed.values()),
            "evaluation_archive_record_count": evaluation_count},
        "policy": POLICY, "hard_pruning": False, "exact_verdict_cache": False, "physics_bypass": False,
        "live_llm_calls": 0, "internal_geometry_operations": None, "performance_claim": None,
        "scope": "same known beam and seed-specific development revisit; not unseen-seed or unseen-assembly generalization",
        "notes": {"source": "all 15 fresh development runs archived, including UNKNOWN; frozen before 20 evaluation runs",
                  "memory": "same seed/context positive-only stable promotion; failures never remove candidates",
                  "outcomes": "normal unsuccessful plans remain in denominators; plan/tree/count changes are observations",
                  "timing": "descriptive only; preparation cost separate from online cost",
                  "verifier": "coarse _simulate including leaf; native n_eval excludes leaf; internal operations unsupported"},
    }


def write_json(path, value):
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
    for name in ("comparison.json", "rows.csv", "development_rows.csv"):
        if (args.run_dir / name).exists():
            raise FileExistsError(str(args.run_dir / name))
    result = compare(args.run_dir)
    write_json(args.run_dir / "comparison.json", result)
    fields = ["seed", "variant", "run_key", "integrity_status", "base_ranker", "memory_mode",
              "plan_status", "plan_success", "sequence", "native_n_eval", "native_total_n_eval",
              "verifier_calls", "verifier_PASS", "verifier_UNKNOWN", "verifier_ERROR", "leaf_verifier_calls",
              "rank_requests", "rank_fallbacks", "memory_requests", "memory_evidence_hit_requests",
              "memory_positive_hit_requests", "memory_promoted_candidates", "memory_changed_requests",
              "memory_fallbacks", "candidate_yields", "skipped_edges", "native_time_seconds", "local_rank_seconds"]
    for name, rows in (("rows.csv", result["rows"]), ("development_rows.csv", result["development_rows"])):
        with (args.run_dir / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key) for key in fields})
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["integrity_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
