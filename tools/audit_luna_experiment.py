"""Read-only completion audit for the disabled-Luna/frozen-memory experiment.

No API, worker, physics, mutable SQLite connection, or pickle decoder is used.
Only --output is created, exclusively. Execute after all evaluation runs finish.
"""
import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from audit_langgraph_completion import (
    Audit, COMMIT, PHASES, PROJECT, artifact_hashes, audit_checkpoint,
    canonical, digest, json_object, safe, walk_files,
)


VARIANTS = ("native_off", "native_on", "reverse_off", "reverse_on")
SOURCE_MEMORY = PROJECT / "runs/memory_per_seed_20261017/memories/s0"
POLICY = "positive_only_stable_promotion"
STATUSES = ("PASS", "UNKNOWN", "ERROR")
SOURCE_FILES = (
    *("worker/" + name for name in (
        "rank_contract.py", "rank_hook.py", "run_rank_hook.py", "compare_rank_hook.py",
        "validate_baseline.py", "remote_rank_client.py", "run_remote_rank.py", "describe_beam.py",
        "export_graph_artifacts.py", "evidence_memory.py", "memory_ranker.py")),
    *("orchestrator/" + name for name in (
        "__init__.py", "file_service.py", "llm_ranker.py", "batch_budget.py",
        "ranking_pipeline.py", "episode_archive.py", "langgraph_workflow.py")),
    *("tools/" + name for name in (
        "run_luna_experiment.py", "run_recorded.py", "run_memory_multiseed.py",
        "run_asap_baseline.py", "run_langgraph_integration.py")),
)


def value_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def stamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed


def stats_without_time(parsed):
    return {key: value for key, value in parsed["raw_stats"].items() if key != "time"}


def valid_snapshot(data):
    records = data["records"]
    ids = [record["record_id"] for record in records]
    return (data.get("schema_version") == 1 and data.get("policy") == POLICY
            and data.get("allow_hard_pruning") is False and ids == sorted(set(ids))
            and data.get("source_record_ids") == ids and data.get("record_count") == len(records)
            and data.get("content_sha256") == value_sha(records))


def expected_memory(records, context, request, mode):
    """Independently recompute the bounded lookup; do not call MemoryRanker."""
    if mode == "off":
        return {"matched_evidence_ids": [], "positive_candidate_ids": [], "status_counts": {}}
    selected = []
    assembled = set(request["assembled_parts"])
    if request["base_part"] == context["base_part"] and context["base_part"] in assembled:
        for record in records:
            source_request = record["request"]
            if (canonical(record["context"]) == canonical(context)
                    and set(source_request["parts_rest"]) | {source_request["part_move"]} == assembled
                    and set(source_request["parts_removed"]) == set(context["initial_parts"]) - assembled
                    and source_request["pose"] is None and source_request["max_grippers"] == 2
                    and source_request["optimizer"] == context["setup"]["optimizer"]
                    and source_request["timeout"] is None):
                selected.append(record)
    return {"matched_evidence_ids": [record["record_id"] for record in selected],
            "positive_candidate_ids": sorted({record["request"]["part_move"]
                                               for record in selected if record["status"] == "PASS"}),
            "status_counts": {status: sum(record["status"] == status for record in selected) for status in STATUSES}}


def evidence_database(audit, path, snapshot, label):
    # Historical evidence lives outside the current run. The original database
    # helper records schema names relative to its root, so use project scope
    # for that read while retaining the same read-only validation boundary.
    reader = audit if path.is_relative_to(audit.root) else Audit(PROJECT)
    with closing(reader.database(path, {"records": ("record_id", "payload", "hash")})) as db:
        rows = [dict(row) for row in db.execute("SELECT record_id,payload,hash FROM records ORDER BY record_id")]
        triggers = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    if reader is not audit:
        audit.inputs.update(reader.inputs)
        audit.schemas.update({"project/" + name: schema for name, schema in reader.schemas.items()})
    audit.check(label + "/append_only_triggers", {"records_no_update", "records_no_delete"}.issubset(triggers))
    parsed = [json_object(row["payload"]) for row in rows]
    audit.check(label + "/database_equals_snapshot", parsed == snapshot["records"] and all(
        row["record_id"] == record["record_id"] and row["hash"] == value_sha(record)
        for row, record in zip(rows, parsed)))


def audit_frozen(audit, context):
    frozen = audit.read(audit.root / "frozen_memory.json")
    freeze = audit.read(audit.root / "memory_freeze.json")
    original = audit.read(SOURCE_MEMORY / "memory_snapshot.json")
    manifest = audit.read(SOURCE_MEMORY / "memory_manifest.json")
    index = audit.read(SOURCE_MEMORY.parent / "index.json")
    snapshot_sha = audit.sha(audit.root / "frozen_memory.json")
    audit.check("memory/exact_development_copy", frozen == original and valid_snapshot(frozen)
                and snapshot_sha == freeze["sha256"] == manifest["snapshot_sha256"]
                == audit.sha(SOURCE_MEMORY / "memory_snapshot.json") == index["per_seed"]["s0"]["snapshot"])
    audit.check("memory/development_context_and_policy", manifest["context"] == context
                and all(record["context"] == context for record in frozen["records"])
                and frozen["record_count"] == freeze["count"] == manifest["record_count"] == 15
                and freeze["source_record_ids"] == frozen["source_record_ids"]
                and freeze["policy"] == manifest["policy"] == POLICY
                and freeze["allow_hard_pruning"] is False and freeze["current_evaluation_used"] is False
                and freeze["source"] == str(SOURCE_MEMORY / "memory_snapshot.json"))
    audit.check("memory/source_manifest_and_database_hashes", audit.sha(SOURCE_MEMORY / "memory_manifest.json")
                == index["per_seed"]["s0"]["manifest"] and audit.sha(SOURCE_MEMORY / "source_memory.sqlite")
                == index["per_seed"]["s0"]["source_db"] == manifest["source_database_sha256"])
    audit.check("memory/development_only_provenance", len(manifest["sources"]) == 3
                and {source["variant"] for source in manifest["sources"]} == {"baseline", "id_ascending", "reverse_native"}
                and all("/development/s0/" in record["source"]["run_id"] for record in frozen["records"])
                and Counter(record["status"] for record in frozen["records"]) == {"PASS": 12, "UNKNOWN": 3})
    evidence_database(audit, SOURCE_MEMORY / "source_memory.sqlite", frozen, "memory/source")
    return frozen["records"], snapshot_sha, freeze


def audit_variant(audit, variant, row, context, records, snapshot_sha, features, environment, config, freeze):
    run = audit.root / variant
    run_id = audit.root.name + "/" + variant
    mode = variant.rsplit("_", 1)[1]
    state = audit.read(run / "graph_state.json")
    parsed = audit.read(run / "parsed_result.json")
    reuse = audit.read(run / "reuse_check.json")
    execution = audit.command(run / "execute", variant + "/execute")
    validation = audit.command(run / "validation", variant + "/validation")
    audit.check(variant + "/plan_and_graph_success", parsed["status"] == state["status"] == row["plan_status"] == "PASS"
                and parsed["plan_success"] is True and parsed["tree_checked"] is True and not parsed["issues"]
                and parsed["terminal_parts"] == ["6"] and state["phase"] == "END" and state["error_type"] is None)
    audit.check(variant + "/graph_config_binding", state["run_id"] == run_id and state["config"] == {
        "model": "gpt-6-luna", "config_sha256": audit.sha(audit.root / "llm_config.json"), "seed": 0,
        "memory": mode, "base_order": variant.split("_")[0], "memory_sha256": snapshot_sha,
        "feature_sha256": audit.sha(audit.root / "features.json"), "ledger": str(audit.root / "batch_ledger.sqlite")})
    audit.check(variant + "/process_metrics", state["metrics"]["process_exit"] == execution["exit_code"] == 0
                and state["metrics"]["validation_exit"] == validation["exit_code"] == 0
                and state["metrics"]["process_seconds"] == execution["elapsed_seconds"]
                and state["metrics"]["plan_success"] is True)
    audit.check(variant + "/memory_frozen_before_execution", stamp(freeze["utc"]) < stamp(execution["started_at"]))
    audit.check(variant + "/callbacks_once_and_artifacts_unchanged", reuse["callbacks"] == {phase: 1 for phase in PHASES}
                and reuse["reused_state_equal"] is True and reuse["artifact_hashes"] == artifact_hashes(run))
    audit_checkpoint(audit, run, state)
    orchestration = audit.read(run / "orchestrator_events.jsonl", lines=True)
    graph_events = [item["graph_event"] for item in orchestration if item["kind"] == "graph_event"]
    expected = [(kind, phase) for phase in PHASES for kind in ("phase_before", "phase_after")] + [("checkpoint_reuse", "END")]
    audit.check(variant + "/phase_order_and_reuse", [(event["kind"], event["phase"]) for event in graph_events] == expected
                and all(event["run_id"] == event["thread_id"] == run_id and event["error_type"] is None for event in graph_events)
                and len({item["event_id"] for item in orchestration}) == len(orchestration)
                and all(item["kind"] in ("graph_event", "service_response") for item in orchestration))
    trace = audit.read(run / "events.jsonl", lines=True)
    expected_native_argv = ["/opt/asap-research/ASAP/plan_sequence/run_seq_plan.py", "--dir", "beam_assembly",
        "--id", "original", "--planner", "dfs", "--generator", "heur-out", "--max-gripper", "2",
        "--base-part", "6", "--log-dir", "/mnt/c/dev/disassembly_codex_ready_v2/"
        + (run / "native").relative_to(PROJECT).as_posix(), "--early-term"]
    audit.check(variant + "/trace_complete_and_seed_bound", trace[0]["kind"] == "run_start" and trace[-1]["kind"] == "run_end"
                and trace[-1]["status"] == "returned_from_upstream" and trace[0]["seed"] == context["seed"] == 0
                and trace[0]["native_argv"] == expected_native_argv
                and trace[0]["source_commit"] == context["input_snapshot"]["commit"] == COMMIT
                and all(item["run_id"] == run_id for item in trace)
                and sum(item["kind"] == "run_start" for item in trace) == sum(item["kind"] == "run_end" for item in trace) == 1
                and len({item["event_id"] for item in trace}) == len(trace))
    starts = [item for item in trace if item["kind"] == "verifier_start"]
    results = [item for item in trace if item["kind"] == "verifier_result"]
    generations = [item for item in trace if item["kind"] == "generation"]
    decisions = [item for item in trace if item["kind"] == "rank_decision"]
    audit.check(variant + "/verifier_pairing_and_rng", bool(starts) and len(starts) == len(results)
                and len({item["call_id"] for item in starts}) == len(starts)
                and [item["call_id"] for item in starts] == [item["call_id"] for item in results]
                and all(item["rng_before"] == item["rng_after"] for item in decisions))
    pending, paired = None, []
    for item in trace:
        if item["kind"] == "verifier_start":
            if pending is not None:
                raise ValueError("overlapping verifier attempts")
            pending = item
        elif item["kind"] == "verifier_result":
            if pending is None or pending["call_id"] != item["call_id"]:
                raise ValueError("unpaired verifier result")
            paired.append((pending, item))
            pending = None
    audit.check(variant + "/no_pending_verifier", pending is None)
    requests = [audit.read(path) for path in sorted(safe(run / "ipc/requests").glob("*.json"))]
    responses = [audit.read(path) for path in sorted(safe(run / "ipc/responses").glob("*.json"))]
    clients = audit.read(run / "remote_audit.jsonl", lines=True)
    services = [item for item in orchestration if item["kind"] == "service_response"]
    groups = [{item["request"]["request_id"]: item for item in requests},
              {item["request_id"]: item for item in responses}, {item["request_id"]: item for item in clients},
              {item["request_id"]: item for item in services},
              {item["request"]["request_id"]: item for item in decisions}]
    ids, count = set(groups[0]), len(generations)
    audit.check(variant + "/actual_ipc_counts_and_ids", count > 0
                and len(requests) == len(responses) == len(clients) == len(services) == len(decisions) == count
                and len(ids) == count and all(set(group) == ids for group in groups))
    generation_map = {item["generation_id"]: item for item in generations}
    audit.check(variant + "/one_decision_per_generation", len(generation_map) == count
                and len({item["generation_id"] for item in decisions}) == count
                and {item["generation_id"] for item in decisions} == set(generation_map))
    enriched = {}
    memory_hits, memory_changes = 0, 0
    source_ids = {record["record_id"] for record in records}
    for index, request_id in enumerate(sorted(ids), 1):
        envelope, response, client, served, decision = [group[request_id] for group in groups]
        request, metadata = envelope["request"], response["metadata"]
        memory = metadata["memory"]
        found = expected_memory(records, context, request, mode)
        base = list(reversed(request["candidate_ids"])) if variant.startswith("reverse") else request["candidate_ids"]
        positives = set(found["positive_candidate_ids"])
        order = [part for part in base if part in positives] + [part for part in base if part not in positives]
        name = variant + "/r" + str(index)
        filename = hashlib.sha256(request_id.encode("utf-8")).hexdigest() + ".json"
        audit.check(name + "/identity_and_permutation", request == decision["request"] and request["run_id"] == run_id
                    and request["state_id"] == response["state_id"] == client["state_id"] == served["state_id"]
                    and envelope["features"] == features and sorted(order) == sorted(request["candidate_ids"])
                    and order == response["candidate_ids"] == decision["applied_order"] == served["candidate_ids"]
                    == generation_map[decision["generation_id"]]["applied_order"]
                    and generation_map[decision["generation_id"]]["native_order"] == request["candidate_ids"])
        audit.check(name + "/independent_memory_lookup", all(memory[key] == value for key, value in found.items())
                    and memory["mode"] == mode and memory["context_digest"] == value_sha(context)
                    and memory["snapshot_sha256"] == (snapshot_sha if mode == "on" else None)
                    and memory["evidence_loaded"] is (mode == "on") and memory["changed"] is (order != base)
                    and memory["fallback_reason"] is None and memory["allow_hard_pruning"] is False
                    and memory["policy"] == POLICY and memory["usage"] == "soft_ranking_only"
                    and set(memory["matched_evidence_ids"]).issubset(source_ids)
                    and memory["offered_positive_candidate_ids"] == [part for part in request["candidate_ids"] if part in positives])
        audit.check(name + "/disabled_api_and_fallback_layer", metadata["llm_live"] is False
                    and metadata["attempted"] is False and metadata["model_requested"] == "gpt-6-luna"
                    and metadata["model_returned"] is None and metadata["response_id"] is None
                    and metadata["usage"] is None and metadata["call_id"] is None and metadata["retries"] == 0
                    and metadata["llm_fallback_reason"] == "llm_disabled" and metadata["ranking_source"] == "deterministic_fallback"
                    and metadata["backend"] == served["backend"] == "pipeline" and served["llm_live"] is False
                    and metadata["base_ranker"] == ("reverse_native" if variant.startswith("reverse") else "noop")
                    and response["fallback_reason"] is decision["fallback_reason"] is client["fallback_reason"] is served["fallback_reason"] is None)
        saved = deepcopy(envelope)
        saved["features"].update(memory=memory, deterministic_fallback_order=order)
        saved["timeout_seconds"] = min(30.0, float(envelope["timeout_seconds"]) - 5.0
                                       if float(envelope["timeout_seconds"]) > 5 else float(envelope["timeout_seconds"]) * .5)
        enriched[request_id] = saved
        audit.check(name + "/raw_message_hashes", client["filename"] == filename
                    and client["response_metadata"] == metadata and client["request_created"] is True
                    and client["duplicate_cached"] is False and client["outcome"] == "accepted"
                    and client["request_sha256"] == metadata["input_envelope_sha256"] == value_sha(envelope)
                    and metadata["envelope_sha256"] == hashlib.sha256(canonical(envelope, ensure_ascii=True)).hexdigest()
                    and metadata["llm_enriched_envelope_sha256"] == value_sha(saved)
                    and client["response_sha256"] == audit.sha(run / "ipc/responses" / filename)
                    and metadata["worker_timeout_seconds"] == envelope["timeout_seconds"]
                    and metadata["llm_timeout_seconds"] == saved["timeout_seconds"])
        memory_hits += bool(found["matched_evidence_ids"])
        memory_changes += order != base
    native = run / "native/dfs-heur-out/s0/original"
    audit.check(variant + "/parsed_native_artifacts", audit.read(native / "stats.json") == parsed["raw_stats"]
                and audit.read(native / "setup.json") == parsed["setup"] == context["setup"])
    audit.check(variant + "/row_matches_raw", row == audit.read(run / "row.json") and row["verifier_calls"] == len(results)
                and row["unknown_calls"] == sum(item["status"] == "UNKNOWN" for item in results)
                and row["rank_requests"] == len(decisions) and row["memory_hits"] == memory_hits
                and row["sequence"] == parsed["raw_stats"]["sequence"] and row["native_seconds"] == parsed["raw_stats"]["time"]
                and row["tree_sha256"] == parsed["tree_fingerprint_sha256"])
    audit.check(variant + "/input_binary_cache_preserved", audit.read(audit.root / ("environment_after_" + variant) / "stdout.log") == environment)
    audit.command(audit.root / ("environment_after_" + variant), variant + "/input_snapshot")
    return {"parsed": parsed, "pairs": paired, "enriched": enriched, "responses": groups[1],
            "trace_sha": audit.sha(run / "events.jsonl"), "execution": execution, "validation": validation,
            "memory_changes": memory_changes, "events": trace}


def audit_ledger(audit, loaded, config, result):
    required = {"rank_requests": ("request_id", "payload_json", "payload_hash", "config_fingerprint", "status",
                                   "reserved_call", "reserved_usd", "call_id", "response_json", "usage_json", "actual_usd"),
                "batch_policy": ("singleton", "schema_version", "config_fingerprint", "max_calls", "max_usd", "created_utc")}
    with closing(audit.database(audit.root / "batch_ledger.sqlite", required)) as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM rank_requests ORDER BY request_id")]
        policies = [dict(row) for row in db.execute("SELECT * FROM batch_policy")]
    expected = {key: value for data in loaded.values() for key, value in data["enriched"].items()}
    responses = {key: value for data in loaded.values() for key, value in data["responses"].items()}
    audit.check("ledger/single_immutable_zero_budget_policy", len(policies) == 1 and policies[0]["singleton"] == 1
                and policies[0]["schema_version"] == 1 and policies[0]["config_fingerprint"] == value_sha(config)
                and policies[0]["max_calls"] == 0 and Decimal(policies[0]["max_usd"]) == 0)
    audit.check("ledger/all_requests_once", len(rows) == len(expected) and {row["request_id"] for row in rows} == set(expected))
    for index, row in enumerate(rows, 1):
        saved, payload = json_object(row["response_json"]), json_object(row["payload_json"])
        audit.check("ledger/r" + str(index), row["status"] == "completed" and row["reserved_call"] == 0
                    and Decimal(row["reserved_usd"]) == 0 and row["call_id"] is None
                    and row["usage_json"] is None and row["actual_usd"] is None
                    and row["config_fingerprint"] == value_sha(config)
                    and payload == expected[row["request_id"]] and row["payload_hash"] == value_sha(payload)
                    and saved["fallback_reason"] == "llm_disabled" and saved["request_id"] == row["request_id"]
                    and saved["candidate_ids"] == payload["request"]["candidate_ids"]
                    and all(responses[row["request_id"]]["metadata"].get(key) == value for key, value in saved["metadata"].items()))
    summary = result["budget"]
    audit.check("ledger/reported_zero_cost_counts", summary["request_count"] == summary["completed_requests"] == len(rows)
                and summary["run_count"] == 4 and summary["policy_fingerprint"] == value_sha(config)
                and all(summary[key] == 0 for key in ("reserved_calls", "pending_requests", "unknown_outcomes",
                    "usage_reported_calls", "input_tokens", "output_tokens", "total_tokens", "remaining_calls"))
                and all(Decimal(summary[key]) == 0 for key in ("max_usd", "estimated_reservations_usd", "usage_cost_usd",
                    "budget_committed_usd", "remaining_estimated_usd")))
    cumulative = 0
    for index, variant in enumerate(VARIANTS, 1):
        cumulative += len(loaded[variant]["enriched"])
        reported = audit.read(audit.root / variant / "batch_budget_after.json")
        audit.check(variant + "/shared_budget_cumulative", reported["request_count"] == reported["completed_requests"] == cumulative
                    and reported["run_count"] == index and reported["reserved_calls"] == 0
                    and Decimal(reported["budget_committed_usd"]) == 0 and reported["policy_fingerprint"] == value_sha(config))


def audit_archive(audit, loaded, context, source_records):
    folder = audit.root / "evaluation_evidence"
    manifest = audit.read(folder / "memory_manifest.json")
    snapshot = audit.read(folder / "memory_snapshot.json")
    expected = []
    for variant, data in loaded.items():
        for start, end in data["pairs"]:
            expected.append({"schema_version": 1, "record_id": end["event_id"],
                "source": {"run_id": start["run_id"], "call_id": start["call_id"],
                           "start_event_id": start["event_id"], "result_event_id": end["event_id"],
                           "ref": str(audit.root / variant / "events.jsonl"), "sha256": data["trace_sha"]},
                "context": context, "request": start["request"], "status": end["status"],
                "raw_result": end["raw"], "error_type": end.get("error_type"),
                "scope": "bounded_search", "usage": "soft_ranking_only", "blocker": None, "path": None})
    expected.sort(key=lambda record: record["record_id"])
    audit.check("archive/exact_raw_observations_and_disjoint_source", valid_snapshot(snapshot)
                and snapshot["records"] == expected
                and set(snapshot["source_record_ids"]).isdisjoint(record["record_id"] for record in source_records))
    audit.check("archive/manifest_and_idempotency", manifest == audit.read(audit.root / "archive_result.json")
                and manifest["episode_count"] == 4 and manifest["context"] == context
                and manifest["context_digest"] == value_sha(context) and manifest["record_count"] == len(expected)
                and manifest["status_counts"] == {status: sum(record["status"] == status for record in expected) for status in STATUSES}
                and manifest["source_record_ids"] == snapshot["source_record_ids"]
                and manifest["first_import"] == {"inserted": len(expected), "duplicates": 0}
                and manifest["repeated_import"] == {"inserted": 0, "duplicates": len(expected)}
                and manifest["allow_hard_pruning"] is False and manifest["policy"] == POLICY
                and manifest["usage"] == "evaluation_archive_only_not_read_during_evaluation"
                and manifest["snapshot_sha256"] == audit.sha(folder / "memory_snapshot.json")
                and manifest["source_database_sha256"] == audit.sha(folder / "evaluation_memory.sqlite"))
    latest_end = max(stamp(data["validation"]["started_at"]) + timedelta(seconds=data["validation"]["elapsed_seconds"])
                     for data in loaded.values())
    audit.check("archive/created_after_all_evaluations", stamp(manifest["created_utc"]) >= latest_end)
    audit.check("archive/sources_and_hashes", len(manifest["sources"]) == 4 and all(
        source["run_root"] == str(audit.root / variant) and source["run_id"] == audit.root.name + "/" + variant
        and source["record_count"] == len(loaded[variant]["pairs"])
        and source["sha256"] == loaded[variant]["trace_sha"]
        and all(audit.sha(safe(path)) == expected_sha for path, expected_sha in source["artifact_sha256"].items())
        for source, variant in zip(manifest["sources"], VARIANTS)))
    evidence_database(audit, folder / "evaluation_memory.sqlite", snapshot, "archive")


def audit_preservation(audit, result):
    before = audit.read(audit.root / "preserved_before.json")
    after = audit.read(audit.root / "preserved_after.json")
    current = {path.relative_to(PROJECT).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
               for path in walk_files(PROJECT / "runs", exclude=audit.root)}
    audit.check("preservation/old_8642_files_unchanged", len(before) == result["preserved_file_count"] == 8642
                and before == after == current)
    source_before = audit.read(audit.root / "source_before.json")
    source_after = audit.read(audit.root / "source_after.json")
    audit.check("preservation/exact_source_inventory", set(source_before) == set(SOURCE_FILES))
    current_sources = {name: audit.sha(PROJECT / name) for name in SOURCE_FILES}
    frozen_sources = {name: audit.sha(audit.root / "implementation" / name) for name in SOURCE_FILES}
    audit.check("preservation/all_source_copies_unchanged", source_before == source_after == current_sources == frozen_sources)
    loaded = audit.read(audit.root / "loaded_runtime_sources.json")
    audit.check("preservation/runtime_loaded_frozen_modules", bool(loaded) and all(
        safe(path).is_relative_to(audit.root / "implementation") and audit.sha(path)
        == source_before[safe(path).relative_to(audit.root / "implementation").as_posix()]
        for path in loaded.values()))


def historical_comparisons(audit, loaded):
    observations = {}
    old = PROJECT / "runs/llm_langgraph_20261017"
    baseline = audit.read(old / "historical_baseline.json")
    for variant in ("native_off", "native_on"):
        current = loaded[variant]["parsed"]
        audit.check("historical/" + variant + "_matches_M1", baseline["status"] == current["status"] == "PASS"
                    and baseline["tree_fingerprint_sha256"] == current["tree_fingerprint_sha256"]
                    and baseline["setup"] == current["setup"]
                    and stats_without_time(baseline) == stats_without_time(current))
    for variant, previous in (("native_off", "native_trace"), ("native_on", "native_trace"), ("reverse_off", "remote_reverse")):
        reference = audit.read(old / previous / "parsed_result.json")
        current = loaded[variant]["parsed"]
        audit.check("historical/" + variant + "_same_tree_and_stats", reference["status"] == current["status"] == "PASS"
                    and reference["setup"] == current["setup"] and reference["tree_fingerprint_sha256"] == current["tree_fingerprint_sha256"]
                    and stats_without_time(reference) == stats_without_time(current))
    prior = PROJECT / "runs/memory_per_seed_20261017/s0/reverse_on"
    reference = audit.read(prior / "parsed_result.json")
    current = loaded["reverse_on"]["parsed"]
    audit.check("historical/reverse_on_same_sequence_and_stats", reference["status"] == current["status"] == "PASS"
                and reference["raw_stats"]["sequence"] == current["raw_stats"]["sequence"]
                and stats_without_time(reference) == stats_without_time(current))
    same_bytes = audit.sha(prior / "native/dfs-heur-out/s0/original/tree.pkl") == audit.sha(
        audit.root / "reverse_on/native/dfs-heur-out/s0/original/tree.pkl")
    observations["reverse_on_previous_memory_tree"] = {"raw_pickle_bytes_equal": same_bytes,
        "full_tree_equality": "PASS" if same_bytes else "UNVERIFIED",
        "method": "raw tree bytes; no pickle decoding in this audit"}
    optional = safe(audit.root / "historical_memory_reverse_on.json")
    if optional.is_file():
        exported = audit.read(optional)
        audit.check("historical/reverse_on_optional_export_identity", exported["status"] == "PASS"
                    and exported["raw_stats"] == reference["raw_stats"]
                    and str(exported["log_dir"]).replace("\\", "/") in (
                        str(prior / "native/dfs-heur-out/s0/original").replace("\\", "/"),
                        "/mnt/c/dev/disassembly_codex_ready_v2/" + (prior / "native/dfs-heur-out/s0/original").relative_to(PROJECT).as_posix()))
        audit.check("historical/reverse_on_optional_full_tree", exported["tree_fingerprint_sha256"] == current["tree_fingerprint_sha256"])
        observations["reverse_on_previous_memory_tree"]["full_tree_equality"] = "PASS" if exported["tree_fingerprint_sha256"] == current["tree_fingerprint_sha256"] else "DIFFERENT"
        observations["reverse_on_previous_memory_tree"]["method"] = "trusted worker export full-tree fingerprint; original pickle bytes differ; auditor does not decode pickle"
    return observations


def run_audit(root):
    audit = Audit(root)
    result = audit.read(root / "result.json")
    execution = audit.read(root / "execution_start.json")
    config = audit.read(root / "llm_config.json")
    context = audit.read(root / "context.json")
    environment = audit.read(root / "environment_before/stdout.log")
    features = audit.read(root / "features.json")
    audit.check("result/completed_four_runs", result["status"] == "PASS" and result["errors"] == []
                and [row["variant"] for row in result["rows"]] == list(VARIANTS)
                and bool(result["checks"]) and all(value is True for value in result["checks"].values()))
    audit.check("result/disabled_Luna_authorization", result["model"] == config["model"] == execution["model"] == "gpt-6-luna"
                and result["live_api_authorized"] is execution["live_api_authorized"] is config["enabled"] is False
                and config["max_calls"] == config["max_usd"] == 0 and result["live_model_quality"] == "NOT_RUN"
                and execution["candidate_pruning"] is execution["physics_bypass"] is False
                and execution["jobs"] == [[variant.split("_")[0], variant.split("_")[1]] for variant in VARIANTS])
    audit.check("context/input_settings_and_seed", context["input_snapshot"] == environment and context["seed"] == execution["seed"] == 0
                and context["base_part"] == "6" and context["initial_parts"] == ["0", "1", "2", "3", "6"]
                and context["usage"] == "soft_ranking_only" and environment["commit"] == COMMIT
                and environment["tracked_source_clean"] is True and context["setup"]["budget"] == 400
                and context["setup"]["max_grippers"] == 2 and context["setup"]["max_poses"] == 3)
    audit.check("features/observations_only_and_hash_binding", features["source_commit"] == COMMIT
                and features["assembly"] == "beam_assembly/original" and features["base_part"] == "6"
                and features["verifier_calls"] == 0 and features["prior_plan_or_verdicts_used"] is False
                and set(features["parts"]) == {"0", "1", "2", "3", "6"}
                and all(environment["files"].get("/opt/asap-research/ASAP/assets/beam_assembly/original/" + name) == value
                        for name, value in features["source_sha256"].items()))
    for name in ("environment_before", "feature_preparation"):
        audit.command(root / name, name)
    records, snapshot_sha, freeze = audit_frozen(audit, context)
    loaded = {}
    row_map = {row["variant"]: row for row in result["rows"]}
    for variant in VARIANTS:
        def inspect(variant=variant):
            loaded[variant] = audit_variant(audit, variant, row_map[variant], context, records, snapshot_sha, features, environment, config, freeze)
        audit.section(variant, inspect)
    all_event_ids = [event["event_id"] for data in loaded.values() for event in data["events"]]
    all_request_ids = [key for data in loaded.values() for key in data["enriched"]]
    audit.check("identity/global_event_and_request_uniqueness", len(all_event_ids) == len(set(all_event_ids))
                and len(all_request_ids) == len(set(all_request_ids)))
    audit.section("ledger", lambda: audit_ledger(audit, loaded, config, result))
    audit.section("archive", lambda: audit_archive(audit, loaded, context, records))
    observations = {}
    audit.section("historical", lambda: observations.update(historical_comparisons(audit, loaded)))
    audit.section("preservation", lambda: audit_preservation(audit, result))
    audit.check("audit/read_inputs_unchanged", all(digest(PROJECT / path) == sha for path, sha in audit.inputs.items()))
    failures = [name for name, passed in audit.checks.items() if not passed]
    return {"schema_version": 1, "status": "PASS" if audit.checks and not failures else "ERROR",
            "created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(root),
            "scope": "post-execution artifact audit; no API/physics/pickle decoding",
            "check_count": len(audit.checks), "checks": audit.checks, "failed_checks": failures,
            "errors": audit.errors, "observations": observations, "rows": result["rows"],
            "memory_changed_requests": {name: data["memory_changes"] for name, data in loaded.items()},
            "sqlite_schemas": audit.schemas, "audited_input_sha256": audit.inputs,
            "live_model_quality": "NOT_RUN", "audit_script_sha256": digest(Path(__file__)),
            "audit_helper_sha256": digest(PROJECT / "tools/audit_langgraph_completion.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = safe(args.run_dir)
    if root.parent != PROJECT / "runs":
        parser.error("run-dir must be a direct child of project runs")
    output = safe(args.output or root / "completion_audit.json")
    if output.parent != root:
        parser.error("output must be a new direct child of this run")
    with output.open("x", encoding="utf-8") as stream:
        try:
            result = run_audit(root)
        except Exception as error:
            result = {"status": "ERROR", "check_count": 0,
                      "errors": [{"section": "setup", "error_type": type(error).__name__}],
                      "scope": "post-execution read-only artifact audit"}
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "output": str(output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
