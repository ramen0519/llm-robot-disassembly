"""Audit fixed-rule experiments without requiring their plans to match baseline.

Python 3.8. Tree pickle files must come from the user's trusted local ASAP runs.
An integrity PASS does not mean every variant found a complete plan.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from compare_rank_hook import (
    SUFFIX, events, native_command, observed, read, stats_data, tree_data,
)


VARIANTS = ("baseline", "id_ascending", "reverse_native")
NORMAL_STOPS = ("solution found", "budget reached", "tree fully explored", "timeout")


def _unique(values):
    return all(isinstance(value, str) and value for value in values) and len(values) == len(set(values))


def _permutation(native, applied):
    return (isinstance(native, list) and isinstance(applied, list)
            and all(type(part) is str for part in native + applied)
            and len(native) == len(set(native))
            and len(applied) == len(native) and set(applied) == set(native))


def _load(root, name):
    folder = root / name
    native = folder / SUFFIX
    for filename in ("stdout.log", "stderr.log"):
        if not (folder / "execute" / filename).is_file():
            raise FileNotFoundError(str(folder / "execute" / filename))
    return {
        "command": read(folder / "execute/command.json"),
        "setup": read(native / "setup.json"), "stats": read(native / "stats.json"),
        "tree": tree_data(native), "parsed": read(folder / "parsed_result.json"),
        "events": events(folder / "events.jsonl"),
        "native_command": native_command(folder / "events.jsonl"),
    }


def _audit_events(data, name, checks, expected_ranker=None, ordering_function=None):
    def check(label, value):
        checks[name + ":" + label] = bool(value)

    all_events = data["events"]
    by_kind = {}
    for event in all_events:
        by_kind.setdefault(event["kind"], []).append(event)
    generations = by_kind.get("generation", [])
    ranks = by_kind.get("rank_decision", [])
    yields = by_kind.get("candidate_yield", [])
    starts = by_kind.get("verifier_start", [])
    results = by_kind.get("verifier_result", [])
    run_starts, run_ends = by_kind.get("run_start", []), by_kind.get("run_end", [])
    check("one_complete_run", len(run_starts) == len(run_ends) == 1
          and all_events[0]["kind"] == "run_start" and all_events[-1]["kind"] == "run_end"
          and run_ends[0].get("status") == "returned_from_upstream")
    first = run_starts[0]
    check("expected_mode_and_ranker", first.get("mode") == ("trace" if name == "baseline" else "rank")
          and first.get("ranker_name") == (None if name == "baseline" else (expected_ranker or name)))
    check("local_only", first.get("live_llm_calls") == 0
          and first.get("internal_geometry_operations") is None)
    check("event_ids_unique", _unique([event["event_id"] for event in all_events]))
    check("single_run_id", all(event["run_id"] == first["run_id"] for event in all_events))
    check("generation_ids_unique", _unique([event["generation_id"] for event in generations]))
    check("request_ids_unique", _unique([event["request"]["request_id"] for event in ranks]))
    check("start_call_ids_unique", _unique([event["call_id"] for event in starts]))
    check("result_call_ids_unique", _unique([event["call_id"] for event in results]))
    check("starts_results_same_order", [event["call_id"] for event in starts]
          == [event["call_id"] for event in results])
    verifier_events = [(event["kind"], event["call_id"]) for event in all_events
                       if event["kind"] in ("verifier_start", "verifier_result")]
    check("serial_start_result_pairs", verifier_events == [item for start in starts
          for item in (("verifier_start", start["call_id"]), ("verifier_result", start["call_id"]))])
    check("no_verifier_errors", all(event.get("status") in ("PASS", "UNKNOWN") for event in results))
    check("raw_status_preserved", all(isinstance(event.get("raw"), dict)
          and event["status"] == ("PASS" if event["raw"].get("feasible") is True else "UNKNOWN")
          and event.get("blocker") is None and event.get("path") is None for event in results))
    check("candidate_sets_preserved", all(_permutation(event["native_order"], event["applied_order"])
          and set(event["native_order"]).issubset(event["assembled_parts"])
          and "6" not in event["native_order"] for event in generations))
    generation_map = {event["generation_id"]: event for event in generations}
    yielded = {key: [] for key in generation_map}
    current_generation, current_candidate = None, None
    links_valid = True
    for event in all_events:
        if event["kind"] == "generation":
            current_generation, current_candidate = event["generation_id"], None
        elif event["kind"] == "candidate_yield":
            key = event["generation_id"]
            if key not in generation_map or key != current_generation:
                links_valid = False
                continue
            yielded[key].append(event["part"])
            current_candidate = event["part"]
        elif event["kind"] == "verifier_start":
            key, request = event["generation_id"], event["request"]
            if key is None:
                # In the pinned source only the final 2-to-1 leaf uses no generator.
                links_valid = links_valid and len(request["parts_rest"]) == 1
            else:
                generation = generation_map.get(key)
                links_valid = links_valid and generation is not None and key == current_generation
                if generation is not None:
                    links_valid = (links_valid and current_candidate == request["part_move"]
                                   and set(generation["assembled_parts"])
                                   == set([request["part_move"]] + request["parts_rest"]))
    check("yield_and_verifier_links", links_valid)
    check("consumed_prefixes", all(parts == generation_map[key]["applied_order"][:len(parts)]
          for key, parts in yielded.items()))
    check("one_rank_per_generation", (not ranks if name == "baseline" else
          [event["generation_id"] for event in ranks] == [event["generation_id"] for event in generations]))
    check("no_rank_fallback", all(event["fallback_reason"] is None for event in ranks))
    rank_contract_ok = True
    for event in ranks:
        generation = generation_map.get(event["generation_id"])
        request = event["request"]
        if generation is None:
            rank_contract_ok = False
            continue
        native = generation["native_order"]
        expected = (ordering_function(event, generation) if ordering_function else
                    sorted(native) if name == "id_ascending" else list(reversed(native)))
        rank_contract_ok = (rank_contract_ok and event.get("ranker_name") == (expected_ranker or name)
                            and request["run_id"] == first["run_id"]
                            and request["state_id"] == generation["state_id"]
                            and request["assembled_parts"] == generation["assembled_parts"]
                            and request["base_part"] == "6" and request["candidate_ids"] == native
                            and event["applied_order"] == generation["applied_order"] == expected)
    check("expected_rule_and_request_response", rank_contract_ok)
    check("ranker_does_not_consume_rng", all(isinstance(event.get("rng_before"), str)
          and bool(event["rng_before"]) and event["rng_before"] == event.get("rng_after") for event in ranks))
    if name == "baseline":
        check("baseline_native_order", all(event["native_order"] == event["applied_order"] for event in generations))

    # Reconstruct only edge records written by the pinned upstream. Ordinary
    # attempts overwrite an existing edge. Failed leaf attempts are not stored.
    expected_edges = {}
    results_by_id = {event["call_id"]: event for event in results}
    nonleaf_count = 0
    raw_request_match = True
    for start in starts:
        result = results_by_id.get(start["call_id"])
        if result is None or not isinstance(result.get("raw"), dict):
            raw_request_match = False
            continue
        request, raw = start["request"], result["raw"]
        raw_request_match = (raw_request_match and raw.get("part_move") == request["part_move"]
                             and raw.get("pose") == request["pose"] and raw.get("base_part") == "6")
        parent = tuple(sorted([request["part_move"]] + request["parts_rest"]))
        child = tuple(request["parts_rest"])
        leaf = start["generation_id"] is None
        if not leaf:
            nonleaf_count += 1
        if not leaf or raw.get("feasible") is True:
            expected_edges[(parent, child)] = {"sim_info": raw, "n_eval": None if leaf else nonleaf_count}
    actual_edges = {(tuple(parent), tuple(child)): attributes
                    for parent, child, attributes in data["tree"]["edges"]}
    nodes = {tuple(node): attributes for node, attributes in data["tree"]["nodes"]}
    edges_match = set(expected_edges) == set(actual_edges)
    for key, expected in expected_edges.items():
        actual = actual_edges.get(key, {})
        expected_eval = expected["n_eval"] if expected["n_eval"] is not None else nodes.get(key[0], {}).get("n_eval")
        edges_match = (edges_match and actual.get("sim_info") == expected["sim_info"]
                       and actual.get("n_eval") == expected_eval)
    check("raw_request_fields_match", raw_request_match)
    check("raw_verifier_order_matches_stored_edges", edges_match)
    check("native_total_n_eval_matches_nonleaf_calls", data["stats"].get("total_n_eval") == nonleaf_count)
    successful_leaves = [attributes for node, attributes in nodes.items()
                         if len(node) == 1 and attributes.get("n_gripper") is not None]
    check("tree_and_stats_success_agree", data["stats"].get("success") is bool(successful_leaves))
    expected_n_eval = min(item["n_eval"] for item in successful_leaves) if successful_leaves else None
    check("native_n_eval_matches_success_leaf", data["stats"].get("n_eval") == expected_n_eval)
    sequence = data["stats"].get("sequence")
    path_ok = sequence is None
    if data["stats"].get("success") is True:
        parent = ("0", "1", "2", "3", "6")
        path_ok = (isinstance(sequence, list) and len(sequence) == 4
                   and set(sequence) == {"0", "1", "2", "3"})
        if path_ok:
            for part in sequence:
                child = tuple(p for p in parent if p != part)
                raw = actual_edges.get((parent, child), {}).get("sim_info", {})
                path_ok = path_ok and raw.get("feasible") is True and raw.get("part_move") == part
                parent = child
            path_ok = path_ok and parent == ("6",)
    check("reported_plan_is_connected_feasible_path", path_ok)
    counts = Counter(event.get("status") for event in results)
    first_generation = generations[0] if generations else {}
    return {
        "variant": name, "plan_status": data["parsed"].get("status"),
        "plan_success": data["stats"].get("success"), "stop_msg": data["stats"].get("stop_msg"),
        "sequence": sequence, "native_n_eval": data["stats"].get("n_eval"),
        "native_total_n_eval": data["stats"].get("total_n_eval"),
        "verifier_calls": len(starts), "verifier_PASS": counts["PASS"],
        "verifier_UNKNOWN": counts["UNKNOWN"], "verifier_ERROR": counts["ERROR"],
        "leaf_verifier_calls": sum(event["generation_id"] is None for event in starts),
        "rank_requests": len(ranks), "rank_fallbacks": sum(event["fallback_reason"] is not None for event in ranks),
        "candidate_yields": len(yields), "skipped_edges": None,
        "generations": len(generations), "changed_order_generations": sum(
            event["native_order"] != event["applied_order"] for event in generations),
        "first_native_order": first_generation.get("native_order"),
        "first_applied_order": first_generation.get("applied_order"),
        "native_time_seconds": data["stats"].get("time"),
        "process_elapsed_seconds": data["command"].get("elapsed_seconds"),
        "local_rank_seconds": sum(event.get("rank_seconds", 0.0) for event in ranks),
        "candidate_orders": [{"generation_id": event["generation_id"],
            "assembled_parts": event["assembled_parts"], "native": event["native_order"],
            "applied": event["applied_order"], "consumed": yielded[event["generation_id"]]}
            for event in generations],
    }


def compare(root):
    checks, errors, loaded, rows = {}, [], {}, []

    def error(scope, exception):
        checks[scope + ":readable_and_complete"] = False
        errors.append({"scope": scope, "error_type": type(exception).__name__, "message": str(exception)})

    protocol = None
    try:
        protocol = read(root / "protocol.json")
        checks["predeclared_variant_order"] = protocol.get("variants") == list(VARIANTS)
    except Exception as exception:
        error("protocol", exception)
    try:
        before, after = read(root / "preserved_before.json"), read(root / "preserved_after.json")
        checks["m1_m2_m3_artifact_bytes_preserved"] = bool(before) and before == after
    except Exception as exception:
        error("preserved_artifacts", exception)
    snapshots = {}
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
            checks[name + ":process_exit_zero"] = (data["command"].get("exit_code") == 0
                                                     and data["command"].get("launch_error") is None)
            parsed, stats = data["parsed"], data["stats"]
            checks[name + ":normal_planner_termination"] = stats.get("stop_msg") in NORMAL_STOPS
            checks[name + ":parsed_stats_match_native"] = parsed.get("raw_stats") == stats
            checks[name + ":plan_outcome_valid"] = (
                (stats.get("success") is True and parsed.get("status") == "PASS"
                 and parsed.get("plan_success") is True and parsed.get("tree_checked") is True)
                or (stats.get("success") is False and parsed.get("status") == "UNKNOWN"
                    and parsed.get("plan_success") is False))
            rows.append(_audit_events(data, name, checks))
        except Exception as exception:
            error(name, exception)
            rows.append({"variant": name, "plan_status": "ERROR", "plan_success": None})
    if "baseline" in snapshots:
        checks["identical_initial_input_cache_binary_source"] = (
            len(snapshots) == len(VARIANTS)
            and all(value == snapshots["baseline"] for value in snapshots.values()))
    if "baseline" in loaded:
        baseline = loaded["baseline"]
        baseline_start = baseline["events"][0]
        for name, data in loaded.items():
            checks[name + ":same_official_cli"] = data["native_command"] == baseline["native_command"]
            checks[name + ":same_effective_setup"] = data["setup"] == baseline["setup"]
            checks[name + ":same_source_commit"] = data["events"][0]["source_commit"] == baseline_start["source_commit"]
            if name in snapshots:
                checks[name + ":snapshot_matches_source_commit"] = (
                    snapshots[name].get("commit") == data["events"][0]["source_commit"]
                    and snapshots[name].get("tracked_source_clean") is True)
        for previous in ("direct", "trace"):
            try:
                old = root.parent / "m3_wsl_20261016" / previous / SUFFIX
                checks["baseline:m3_" + previous + "_setup_equal"] = baseline["setup"] == read(old / "setup.json")
                checks["baseline:m3_" + previous + "_stats_except_time_equal"] = (
                    {key: value for key, value in baseline["stats"].items() if key != "time"} == stats_data(old))
                checks["baseline:m3_" + previous + "_ordered_tree_equal"] = baseline["tree"] == tree_data(old)
            except Exception as exception:
                error("m3_" + previous, exception)
        try:
            old_trace = root.parent / "m3_wsl_20261016/trace/events.jsonl"
            checks["baseline:m3_trace_observations_equal"] = observed(root / "baseline/events.jsonl") == observed(old_trace)
            checks["baseline:m3_trace_native_cli_equal"] = baseline["native_command"] == native_command(old_trace)
            checks["baseline:m3_source_commit_equal"] = (
                baseline_start["source_commit"] == events(old_trace, "run_start")[0]["source_commit"])
        except Exception as exception:
            error("m3_trace_events", exception)
        baseline_calls = sum(event["kind"] == "verifier_start" for event in baseline["events"])
        for row in rows:
            if row["variant"] in loaded and "verifier_calls" in row:
                data = loaded[row["variant"]]
                row["sequence_changed_from_baseline"] = data["stats"].get("sequence") != baseline["stats"].get("sequence")
                row["ordered_tree_changed_from_baseline"] = data["tree"] != baseline["tree"]
                row["delta_verifier_calls"] = row["verifier_calls"] - baseline_calls
                row["delta_native_time_seconds"] = row["native_time_seconds"] - baseline["stats"]["time"]
    for row in rows:
        local_checks = [value for key, value in checks.items() if key.startswith(row["variant"] + ":")]
        row["integrity_status"] = "PASS" if local_checks and all(local_checks) else "ERROR"
    integrity = "PASS" if checks and all(checks.values()) else "ERROR"
    return {
        "schema_version": 1, "status": integrity, "integrity_status": integrity,
        "all_plans_success": len(rows) == len(VARIANTS) and all(row.get("plan_success") is True for row in rows),
        "checks": checks, "errors": errors, "protocol": protocol, "rows": rows,
        "scope": "one official beam assembly, seed 0, fixed local rules, warm cache",
        "counting_notes": {
            "native_total_n_eval": "ordinary planner calls; final leaf verification is excluded upstream",
            "verifier_calls": "observed coarse _simulate calls including leaf; not internal geometry operations",
            "skipped_edges": "not instrumented; null, never inferred from unmatched candidate yields",
            "state_id": "remaining-part membership only, not a physical-state or cache identity",
            "tree_evidence": "latest ordinary edge result and successful leaf result; failed leaf calls stay in events",
        },
        "comparison_notes": {
            "baseline_equivalence": "M3 stats exclude only top-level time; tree preserves full values and ordering",
            "rules": "different plans, trees, verifier counts and downstream RNG are allowed",
            "rng": "only the local rank operation must preserve RNG; complete trajectories need not match",
            "budget": "native setup budget, not an implemented global rank/API/physics budget controller",
        },
        "internal_geometry_operations": None, "live_llm_calls": 0,
        "memory_enabled": False, "performance_claim": None,
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
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    fields = ["variant", "integrity_status", "plan_status", "plan_success", "stop_msg", "sequence",
              "native_n_eval", "native_total_n_eval", "verifier_calls", "verifier_PASS", "verifier_UNKNOWN",
              "verifier_ERROR", "leaf_verifier_calls", "rank_requests", "rank_fallbacks", "candidate_yields",
              "skipped_edges", "changed_order_generations", "first_native_order", "first_applied_order",
              "sequence_changed_from_baseline", "ordered_tree_changed_from_baseline", "delta_verifier_calls",
              "native_time_seconds", "delta_native_time_seconds", "local_rank_seconds"]
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({field: json.dumps(row.get(field), ensure_ascii=False)
                             if isinstance(row.get(field), (list, dict)) else row.get(field) for field in fields})
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["integrity_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
