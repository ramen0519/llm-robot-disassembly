"""Audit frozen positive-only memory comparisons; never infer a physics verdict.

Python 3.8. The source is a previously observed beam, not a held-out task.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from compare_rank_hook import SUFFIX, events, read, stats_data, tree_data
from compare_rule_rankers import NORMAL_STOPS, _audit_events, _load


VARIANTS = ("heur_off", "heur_on", "reverse_off", "reverse_on")
PAIRS = (("heur_off", "heur_on"), ("reverse_off", "reverse_on"))
POLICY = "positive_only_stable_promotion"


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_lookup(records, context, assembled, base_part):
    """Recompute from immutable records, independently of FrozenMemory.lookup."""
    matched = []
    if base_part == context["base_part"]:
        expected_removed = sorted(set(context["initial_parts"]) - set(assembled))
        for record in records:
            request = record["request"]
            if (digest(record["context"]) == digest(context)
                    and sorted([request["part_move"]] + request["parts_rest"]) == sorted(assembled)
                    and sorted(request["parts_removed"]) == expected_removed
                    and request["pose"] is None and request["max_grippers"] == 2
                    and request["timeout"] is None
                    and request["optimizer"] == context["setup"]["optimizer"]):
                matched.append(record)
    counts = Counter(record["status"] for record in matched)
    return {
        "matched_evidence_ids": sorted(record["record_id"] for record in matched),
        "positive_candidate_ids": sorted({record["request"]["part_move"] for record in matched
                                           if record["status"] == "PASS"}),
        "status_counts": {status: counts[status] for status in ("PASS", "UNKNOWN", "ERROR")},
    }


def audit_memory_events(data, name, checks, records, context, snapshot_sha256):
    expected_ranker = "noop" if name.startswith("heur_") else "reverse_native"
    memory_mode = name.rsplit("_", 1)[1]
    context_sha = digest(context)
    ranks = [event for event in data["events"] if event["kind"] == "rank_decision"]
    checks[name + ":memory_ordering_recomputed"] = True

    def ordering(event, generation):
        native = generation["native_order"]
        base = list(native) if expected_ranker == "noop" else list(reversed(native))
        expected = (independent_lookup(records, context, generation["assembled_parts"],
                                       event["request"]["base_part"])
                    if memory_mode == "on" else {
                        "matched_evidence_ids": [], "positive_candidate_ids": [], "status_counts": {}})
        positive = set(expected["positive_candidate_ids"])
        applied = [part for part in base if part in positive] + [part for part in base if part not in positive]
        memory = event.get("memory", {})
        valid = (event.get("base_order") == base and memory.get("mode") == memory_mode
                 and memory.get("snapshot_sha256") == snapshot_sha256
                 and memory.get("context_digest") == context_sha
                 and memory.get("fallback_reason") is None
                 and memory.get("changed") is (applied != base)
                 and all(memory.get(key) == value for key, value in expected.items()))
        checks[name + ":memory_ordering_recomputed"] &= valid
        return applied

    row = _audit_events(data, name, checks, expected_ranker=expected_ranker, ordering_function=ordering)
    start = data["events"][0]
    checks[name + ":run_memory_identity"] = (
        start.get("memory_mode") == memory_mode and start.get("memory_snapshot_sha256") == snapshot_sha256
        and start.get("memory_context_digest") == context_sha)
    source_ids = {record["record_id"] for record in records}
    checks[name + ":matched_ids_are_frozen_source_ids"] = all(
        set(event.get("memory", {}).get("matched_evidence_ids", [])).issubset(source_ids) for event in ranks)
    memory_events = [event.get("memory", {}) for event in ranks]
    promoted = 0
    for event in ranks:
        base, applied = event.get("base_order", []), event["applied_order"]
        positive = set(event.get("memory", {}).get("positive_candidate_ids", []))
        promoted += sum(part in base and part in applied and base.index(part) > applied.index(part)
                        for part in positive)
    row.update({
        "base_ranker": expected_ranker, "memory_mode": memory_mode,
        "memory_requests": len(ranks) if memory_mode == "on" else 0,
        "memory_evidence_hit_requests": sum(bool(item.get("matched_evidence_ids")) for item in memory_events),
        "memory_positive_hit_requests": sum(bool(item.get("positive_candidate_ids")) for item in memory_events),
        "memory_evidence_references": sum(len(item.get("matched_evidence_ids", [])) for item in memory_events),
        "memory_positive_candidate_occurrences": sum(len(item.get("positive_candidate_ids", [])) for item in memory_events),
        "memory_promoted_candidates": promoted,
        "memory_changed_requests": sum(item.get("changed") is True for item in memory_events),
        "memory_fallbacks": sum(item.get("fallback_reason") is not None for item in memory_events),
    })
    return row


def audit_frozen_hashes(hashes, snapshot_sha256, database_sha256, checks):
    before = hashes.get("before")
    checks["frozen_memory_hashes_match_files"] = before == {
        "snapshot": snapshot_sha256, "source_db": database_sha256}
    for name in VARIANTS:
        checks[name + ":memory_db_and_snapshot_unchanged"] = (
            isinstance(before, dict) and hashes.get("after_" + name) == before)
    checks["source_memory_unchanged_after_evaluation_archive"] = (
        isinstance(before, dict) and hashes.get("after_evaluation_archive") == before)


def _observed(path):
    kinds = {"generation", "candidate_yield", "verifier_start", "verifier_result"}
    return [{key: value for key, value in event.items() if key not in ("run_id", "event_id")}
            for event in events(path) if event["kind"] in kinds]


def _source_records(source_root, variants=("baseline", "id_ascending", "reverse_native")):
    result = {}
    for variant in variants:
        path = source_root / variant / "events.jsonl"
        source_events = events(path)
        starts = {(event["run_id"], event["call_id"]): event for event in source_events
                  if event["kind"] == "verifier_start"}
        for event in source_events:
            if event["kind"] == "verifier_result":
                start = starts[(event["run_id"], event["call_id"])]
                if event["event_id"] in result:
                    raise ValueError("duplicate source result event ID")
                result[event["event_id"]] = (start, event, file_hash(path))
    return result


def _snapshot_valid(snapshot):
    records = snapshot["records"]
    ids = [record["record_id"] for record in records]
    return (snapshot.get("schema_version") == 1 and snapshot.get("record_count") == len(records)
            and len(ids) == len(set(ids)) and ids == sorted(ids)
            and snapshot.get("source_record_ids") == ids and snapshot.get("content_sha256") == digest(records)
            and snapshot.get("policy") == POLICY and snapshot.get("allow_hard_pruning") is False)


def _records_match(records, context, observed_records):
    if {record["record_id"] for record in records} != set(observed_records):
        return False
    for record in records:
        start, result, sha256 = observed_records[record["record_id"]]
        provenance = record["source"]
        if not (record["request"] == start["request"] and record["status"] == result["status"]
                and record["raw_result"] == result["raw"]
                and record.get("error_type") == result.get("error_type")
                and digest(record["context"]) == digest(context)
                and provenance["run_id"] == result["run_id"] and provenance["call_id"] == result["call_id"]
                and provenance["start_event_id"] == start["event_id"]
                and provenance["result_event_id"] == result["event_id"] and provenance["sha256"] == sha256):
            return False
    return True


def audit_evaluation_archive(snapshot, context, source_ids, observed_records, checks):
    checks["evaluation_archive_snapshot_valid"] = _snapshot_valid(snapshot)
    records = snapshot["records"]
    ids = {record["record_id"] for record in records}
    checks["evaluation_archive_contains_exactly_evaluation_results"] = ids == set(observed_records)
    checks["evaluation_archive_disjoint_from_frozen_source"] = not ids.intersection(source_ids)
    checks["evaluation_archive_preserves_raw_requests_results"] = _records_match(records, context, observed_records)


def compare(root):
    checks, errors, loaded, snapshots, rows = {}, [], {}, {}, []
    records, context, snapshot_sha256, protocol = None, None, None, None

    def error(scope, exception):
        checks[scope + ":readable_and_complete"] = False
        errors.append({"scope": scope, "error_type": type(exception).__name__, "message": str(exception)})

    try:
        protocol = read(root / "protocol.json")
        checks["predeclared_variants_and_pairs"] = (protocol.get("variants") == list(VARIANTS)
            and protocol.get("pairs") == [list(pair) for pair in PAIRS]
            and protocol.get("source_run") == "runs/m4_rules_20261017" and protocol.get("policy") == POLICY)
        before, after = read(root / "preserved_before.json"), read(root / "preserved_after.json")
        checks["m1_through_m4_artifact_bytes_preserved"] = bool(before) and before == after
    except Exception as exception:
        error("protocol_or_preservation", exception)
    try:
        snapshot = read(root / "memory_snapshot.json")
        records = snapshot["records"]
        manifest = read(root / "memory_manifest.json")
        context = manifest["context"]
        ids = [record["record_id"] for record in records]
        checks["frozen_snapshot_schema_and_policy"] = _snapshot_valid(snapshot)
        checks["fixed_supported_context"] = (context.get("seed") == 0 and context.get("base_part") == "6"
            and context.get("initial_parts") == ["0", "1", "2", "3", "6"]
            and context.get("scope") == "official_beam_original" and context.get("usage") == "soft_ranking_only")
        snapshot_sha256 = file_hash(root / "memory_snapshot.json")
        audit_frozen_hashes(read(root / "memory_hashes.json"), snapshot_sha256,
                           file_hash(root / "source_memory.sqlite"), checks)
        source = _source_records(root.parent / "m4_rules_20261017")
        checks["frozen_records_equal_prior_m4_evidence"] = _records_match(records, context, source)
        checks["source_manifest_hashes_match_files"] = (
            manifest.get("snapshot_sha256") == snapshot_sha256
            and manifest.get("source_database_sha256") == file_hash(root / "source_memory.sqlite")
            and manifest.get("record_count") == len(records))
    except Exception as exception:
        error("frozen_memory", exception)
    for name in VARIANTS:
        try:
            before = read(root / ("inputs_before_" + name) / "stdout.log")
            after = read(root / ("inputs_after_" + name) / "stdout.log")
            snapshots[name] = before
            checks[name + ":input_cache_binary_source_unchanged"] = before == after
        except Exception as exception:
            error(name + ":snapshot", exception)
        try:
            data = _load(root, name)
            loaded[name] = data
            parsed, stats = data["parsed"], data["stats"]
            checks[name + ":process_exit_zero"] = (data["command"].get("exit_code") == 0
                                                     and data["command"].get("launch_error") is None)
            checks[name + ":normal_planner_termination"] = stats.get("stop_msg") in NORMAL_STOPS
            checks[name + ":parsed_stats_match_native"] = parsed.get("raw_stats") == stats
            checks[name + ":plan_outcome_valid"] = (
                (stats.get("success") is True and parsed.get("status") == "PASS"
                 and parsed.get("plan_success") is True and parsed.get("tree_checked") is True)
                or (stats.get("success") is False and parsed.get("status") == "UNKNOWN"
                    and parsed.get("plan_success") is False))
            if records is None or context is None or snapshot_sha256 is None:
                raise ValueError("frozen memory metadata unavailable")
            checks[name + ":live_context_matches_frozen_context"] = (
                context["input_snapshot"] == snapshots.get(name) and context["setup"] == data["setup"]
                and context["input_snapshot"].get("commit") == data["events"][0].get("source_commit"))
            eval_ids = {event["event_id"] for event in data["events"]}
            checks[name + ":evaluation_events_not_in_source_memory"] = not eval_ids.intersection(
                record["record_id"] for record in records)
            rows.append(audit_memory_events(data, name, checks, records, context, snapshot_sha256))
        except Exception as exception:
            error(name, exception)
            rows.append({"variant": name, "plan_status": "ERROR", "plan_success": None})
    checks["identical_initial_input_cache_binary_source"] = (
        len(snapshots) == len(VARIANTS) and all(value == snapshots["heur_off"] for value in snapshots.values()))
    evaluation_record_count = None
    try:
        archive = root / "evaluation_evidence"
        archive_snapshot = read(archive / "memory_snapshot.json")
        archive_manifest = read(archive / "memory_manifest.json")
        audit_evaluation_archive(archive_snapshot, context, {record["record_id"] for record in records},
                                 _source_records(root, VARIANTS), checks)
        evaluation_record_count = len(archive_snapshot["records"])
        checks["evaluation_archive_manifest_matches_files"] = (
            archive_manifest.get("context") == context
            and archive_manifest.get("record_count") == evaluation_record_count
            and archive_manifest.get("snapshot_sha256") == file_hash(archive / "memory_snapshot.json")
            and archive_manifest.get("source_database_sha256") == file_hash(archive / "evaluation_memory.sqlite"))
    except Exception as exception:
        error("evaluation_archive", exception)
    for current, prior in (("heur_off", "baseline"), ("reverse_off", "reverse_native")):
        try:
            data = loaded[current]
            old_root = root.parent / "m4_rules_20261017" / prior
            checks[current + ":prior_setup_equal"] = data["setup"] == read(old_root / SUFFIX / "setup.json")
            checks[current + ":prior_stats_except_time_equal"] = (
                {key: value for key, value in data["stats"].items() if key != "time"} == stats_data(old_root / SUFFIX))
            checks[current + ":prior_ordered_tree_equal"] = data["tree"] == tree_data(old_root / SUFFIX)
            checks[current + ":prior_physics_observations_equal"] = (
                _observed(root / current / "events.jsonl") == _observed(old_root / "events.jsonl"))
        except Exception as exception:
            error(current + ":prior_equivalence", exception)
    row_map = {row["variant"]: row for row in rows}
    pair_deltas = []
    for off, on in PAIRS:
        if off not in loaded or on not in loaded:
            checks[off + "_vs_" + on + ":both_runs_present"] = False
            continue
        checks[off + "_vs_" + on + ":same_effective_setup_and_cli"] = (
            loaded[off]["setup"] == loaded[on]["setup"]
            and loaded[off]["native_command"] == loaded[on]["native_command"]
            and loaded[off]["events"][0].get("source_commit") == loaded[on]["events"][0].get("source_commit"))
        left, right = row_map[off], row_map[on]
        delta = {"off": off, "on": on, "sequence_off": left.get("sequence"),
                 "sequence_on": right.get("sequence"), "plan_success_off": left.get("plan_success"),
                 "plan_success_on": right.get("plan_success"),
                 "sequence_changed": left.get("sequence") != right.get("sequence"),
                 "ordered_tree_changed": loaded[off]["tree"] != loaded[on]["tree"]}
        for metric in ("verifier_calls", "verifier_PASS", "verifier_UNKNOWN", "verifier_ERROR",
                       "native_n_eval", "native_total_n_eval", "native_time_seconds"):
            delta[metric + "_on_minus_off"] = (right[metric] - left[metric]
                if isinstance(left.get(metric), (int, float)) and isinstance(right.get(metric), (int, float)) else None)
        pair_deltas.append(delta)
    for row in rows:
        relevant = [value for key, value in checks.items() if key.startswith(row["variant"] + ":")]
        row["integrity_status"] = "PASS" if relevant and all(relevant) else "ERROR"
    integrity = "PASS" if checks and all(checks.values()) else "ERROR"
    return {
        "schema_version": 1, "status": integrity, "integrity_status": integrity,
        "all_plans_success": len(rows) == len(VARIANTS) and all(row.get("plan_success") is True for row in rows),
        "checks": checks, "errors": errors, "protocol": protocol, "rows": rows, "pair_deltas": pair_deltas,
        "source_record_count": len(records) if records is not None else None,
        "evaluation_archive_record_count": evaluation_record_count,
        "source_status_counts": dict(Counter(record["status"] for record in records)) if records is not None else None,
        "scope": "same-beam revisit development experiment; fixed prior M4 evidence; no held-out generalization claim",
        "policy": POLICY, "hard_pruning": False, "exact_verdict_cache": False, "physics_bypass": False,
        "counting_notes": {
            "memory_requests": "lookup requests; off performs no lookup",
            "memory_evidence_hit_requests": "matched PASS, UNKNOWN or ERROR evidence; not a successful plan count",
            "memory_promoted_candidates": "PASS-backed candidates moved earlier than their base ordering",
            "native_total_n_eval": "ordinary planner calls; upstream excludes final leaf checks",
            "verifier_calls": "observed coarse _simulate calls including leaf; not internal geometry operations",
            "skipped_edges": "not instrumented; null",
        },
        "comparison_notes": {
            "pairs": "each on condition is compared only with its corresponding base-order off condition",
            "rng": "local rank/memory operation must preserve RNG; subsequent physics trajectories may differ",
            "outcomes": "plan, tree and verifier count changes are results, never integrity requirements",
            "budget": "native budget only; no integrated memory/API/physics budget controller",
        },
        "live_llm_calls": 0, "internal_geometry_operations": None, "performance_claim": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trusted-local-tree", action="store_true")
    args = parser.parse_args()
    if not args.trusted_local_tree:
        parser.error("only your own trusted local tree.pkl may be loaded: explicit trust required")
    csv_path = args.run_dir / "rows.csv"
    if csv_path.exists():
        raise FileExistsError(str(csv_path))
    with (args.run_dir / "comparison.json").open("x", encoding="utf-8") as stream:
        result = compare(args.run_dir)
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    fields = ["variant", "integrity_status", "base_ranker", "memory_mode", "plan_status", "plan_success",
              "sequence", "native_n_eval", "native_total_n_eval", "verifier_calls", "verifier_PASS",
              "verifier_UNKNOWN", "verifier_ERROR", "leaf_verifier_calls", "rank_requests", "rank_fallbacks",
              "memory_requests", "memory_evidence_hit_requests", "memory_positive_hit_requests",
              "memory_promoted_candidates", "memory_changed_requests", "memory_fallbacks",
              "candidate_yields", "skipped_edges", "native_time_seconds", "local_rank_seconds"]
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({field: json.dumps(row.get(field)) if isinstance(row.get(field), list)
                             else row.get(field) for field in fields})
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if integrity_pass(result) else 1


def integrity_pass(result):
    return result["integrity_status"] == "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
