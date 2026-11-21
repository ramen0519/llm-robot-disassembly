"""Read-only prototype evidence audit: no API, WSL, physics or SQLite writes.

Write only a new audit directory. Fixtures, actual API responses, full-plan
success and cited-reference execution are separate; citation is not causation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import stat

ROOT = Path(__file__).absolute().parents[1]
LABELS = ("repair_fixture", "transfer_fixture", "repair_live", "transfer_live")
EXT4_LABEL = "transfer_fixture_ext4"
COMMIT = "3727e716b57e0b508c4e9ecd9fefa6953e8f72da"


def safe(value):
    raw = str(value).replace("\\", "/")
    lower = raw.lower()
    if ("\x00" in raw or lower.startswith("e:") or lower == "/mnt/e"
            or lower.startswith("/mnt/e/") or raw.startswith("//") or ".." in raw.split("/")):
        raise ValueError("excluded_path")
    prefix = "/mnt/c/dev/disassembly_codex_ready_v2/"
    if raw.startswith(prefix):
        raw = str(ROOT / raw[len(prefix):])
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    try:
        path.relative_to(ROOT)
    except ValueError:
        raise ValueError("audit_reads_are_project_scoped")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("symlink_or_reparse_path")
    return path


def sha(path):
    result = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read(path, default=None):
    path = safe(path)
    if not path.is_file():
        return default
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("audit_JSON_size_limit")
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    path = safe(path)
    if not path.is_file():
        return []
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("audit_JSONL_size_limit")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def full_permutation(request, order):
    if not isinstance(request, dict):
        return False
    assembled, candidates, base = (request.get("assembled_parts"),
        request.get("candidate_ids"), request.get("base_part"))
    if (type(assembled) is not list or type(candidates) is not list or type(order) is not list
            or any(type(v) is not str for v in assembled + candidates + order)):
        return False
    expected = set(assembled) - ({base} if base is not None else set())
    return (len(set(assembled)) == len(assembled) and (base is None or base in assembled)
        and candidates == sorted(expected) and len(order) == len(set(order)) == len(expected)
        and set(order) == expected)


def evidence_pairs(events):
    starts, results, checks = {}, {}, {}
    event_ids = [e.get("event_id") for e in events]
    checks["event_ids_unique"] = len(event_ids) == len(set(event_ids))
    for event in events:
        kind = event.get("kind")
        if kind not in ("verifier_start", "verifier_result"):
            continue
        key = (event.get("run_id"), event.get("call_id"))
        target = starts if kind == "verifier_start" else results
        checks["unique_{}_{}".format(kind, key[1])] = key not in target
        if kind == "verifier_result":
            checks["prior_start_" + str(key[1])] = key in starts
        target[key] = event
    checks["all_verifier_calls_completed"] = set(starts) == set(results)
    for key, result in results.items():
        request = starts.get(key, {}).get("request", {})
        raw, status = result.get("raw"), result.get("status")
        valid = status in ("PASS", "UNKNOWN", "ERROR") and result.get("blocker") is None and result.get("path") is None
        if status == "ERROR":
            valid = valid and raw is None and type(result.get("error_type")) is str
        else:
            valid = (valid and type(raw) is dict and type(raw.get("feasible")) is bool
                and raw.get("feasible") == (status == "PASS") and raw.get("part_move") == request.get("part_move"))
            if status == "PASS":
                valid = valid and raw.get("action") is not None and raw.get("parts_fix") is not None
        checks["raw_status_binding_" + str(key[1])] = bool(valid)
    return starts, results, checks


def verify_memory(snapshot, trace_path, context, events):
    records = snapshot.get("records", [])
    checks = {"snapshot_contract": (snapshot.get("schema_version") == 2
        and snapshot.get("record_count") == len(records) and snapshot.get("allow_hard_pruning") is False
        and snapshot.get("content_sha256") == digest(records)
        and snapshot.get("source_record_ids") == [r.get("record_id") for r in records])}
    starts, results, _ = evidence_pairs(events)
    checks["every_verifier_result_stored_once"] = (len(records) == len(results)
        and {r.get("record_id") for r in records} == {e.get("event_id") for e in results.values()})
    trace_hash = sha(trace_path)
    for record in records:
        source = record.get("source", {})
        key = (source.get("run_id"), source.get("call_id"))
        start, result = starts.get(key, {}), results.get(key, {})
        prefix = str(record.get("record_id"))
        try:
            source_matches = safe(source.get("ref", "")) == safe(trace_path)
        except ValueError:
            source_matches = False
        checks[prefix + "/provenance"] = (source_matches and source.get("sha256") == trace_hash
            and source.get("start_event_id") == start.get("event_id")
            and source.get("result_event_id") == result.get("event_id") == record.get("record_id")
            and record.get("context") == context and record.get("request") == start.get("request")
            and record.get("raw_result") == result.get("raw") and record.get("status") == result.get("status"))
        checks[prefix + "/no_invented_constraints"] = (record.get("blocker") is None
            and record.get("path") is None and record.get("constraints") == []
            and record.get("allow_hard_pruning") is False)
    return checks


def source_archive_checks(directory):
    before, after = read(directory / "source_before.json", {}), read(directory / "source_after.json", {})
    checks = {"source_before_equals_after": bool(before) and before == after}
    for name, expected in before.items():
        if Path(name).is_absolute() or ".." in name.replace("\\", "/").split("/"):
            raise ValueError("unsafe_source_snapshot_name")
        path = safe(directory / "implementation" / name)
        checks["snapshot/" + name] = path.is_file() and sha(path) == expected
    return checks


def repair_chains(decisions, events, reported):
    by_event = {e.get("event_id"): e for e in events}
    index = {e.get("event_id"): i for i, e in enumerate(events)}
    starts, results, _ = evidence_pairs(events)
    ranks = {e["request"]["request_id"]: e for e in events if e.get("kind") == "rank_decision"}
    reports = {x.get("invocation_id"): x for x in reported} if type(reported) is list else {}
    chains = []
    for decision in decisions:
        metadata = decision.get("response", {}).get("metadata", {})
        invocation = metadata.get("repair_invocation")
        if not isinstance(invocation, dict):
            continue
        failure = by_event.get(invocation.get("source_failure_event_id"), {})
        rank = ranks.get(invocation.get("request_id"), {})
        changes = [by_event.get(key, {}) for key in invocation.get("state_change_event_ids", [])]
        failed = starts.get((failure.get("run_id"), failure.get("call_id")), {}).get("request", {})
        state = set(failed.get("parts_rest", [])) | {failed.get("part_move")}
        previous_index = index.get(failure.get("event_id"), -1)
        valid = (invocation.get("skill_id") == "retry_after_state_change_v1"
            and failure.get("status") == "UNKNOWN" and failure.get("raw", {}).get("feasible") is False
            and failed.get("part_move") == invocation.get("failed_part_id")
            and rank.get("fallback_reason") is None and rank.get("applied_order") == invocation.get("applied_order")
            and bool(changes))
        for changed in changes:
            attempt = starts.get((changed.get("run_id"), changed.get("call_id")), {}).get("request", {})
            source_state = set(attempt.get("parts_rest", [])) | {attempt.get("part_move")}
            order = index.get(changed.get("event_id"), -1)
            valid = (valid and changed.get("status") == "PASS" and source_state == state
                and attempt.get("part_move") != invocation.get("failed_part_id")
                and previous_index < order < index.get(rank.get("event_id"), -1))
            state, previous_index = set(attempt.get("parts_rest", [])), order
        valid = valid and state == set(invocation.get("assembled_parts", []))
        retries = []
        for key, start in starts.items():
            attempt = start.get("request", {})
            if (start.get("generation_id") == rank.get("generation_id")
                and attempt.get("part_move") == invocation.get("failed_part_id")
                and (set(attempt.get("parts_rest", [])) | {attempt.get("part_move")}) == state
                and index.get(start.get("event_id"), -1) > index.get(rank.get("event_id"), -1) and key in results):
                retries.append(results[key])
        passed = valid and any(e.get("status") == "PASS" for e in retries)
        report = reports.get(invocation.get("invocation_id"), {})
        matches = (bool(report) and report.get("source_failure_event_id") == failure.get("event_id")
            and report.get("state_change_event_ids") == invocation.get("state_change_event_ids")
            and ((not retries and report.get("status") == "NOT_RUN")
                or any(r.get("event_id") == report.get("retry_event_id") and report.get("raw_result") == r.get("raw")
                    and report.get("status") == r.get("status") for r in retries))
            and report.get("recovery_chain_observed") == bool(passed))
        chains.append({"invocation_id": invocation.get("invocation_id"), "request_id": invocation.get("request_id"),
            "source_failure_event_id": failure.get("event_id"), "change_event_ids": invocation.get("state_change_event_ids"),
            "retry_event_ids": [r.get("event_id") for r in retries], "raw_chain_valid": bool(valid),
            "retry_pass_observed": bool(passed), "reported_outcome_matches_raw": bool(matches),
            "actual_llm_selected": metadata.get("llm_live") is True and metadata.get("attempted") is True,
            "decision_source": metadata.get("ranking_source"), "lifecycle": "provisional",
            "causal_benefit_demonstrated": False, "generalization_validated": False})
    return chains


def transfer_chains(decisions, events, source_snapshot):
    records = {r.get("record_id"): r for r in source_snapshot.get("records", [])}
    starts, results, _ = evidence_pairs(events)
    ranks = {e["request"]["request_id"]: e for e in events if e.get("kind") == "rank_decision"}
    checks, chains = {}, []
    for decision in decisions:
        request, response = decision.get("request", {}), decision.get("response", {})
        metadata, transfer = response.get("metadata", {}), decision.get("transfer", {})
        rid = request.get("request_id")
        hints = {h.get("hint_id"): h for h in transfer.get("tentative_correspondences", [])}
        used = metadata.get("used_hint_ids", [])
        checks[str(rid) + "/cites_offered_only"] = set(used).issubset(hints)
        checks[str(rid) + "/frozen_record_digest"] = transfer.get("source_records_sha256") == digest(source_snapshot.get("records", []))
        checks[str(rid) + "/no_pruning_or_physics_reuse"] = (transfer.get("allow_hard_pruning") is False
            and transfer.get("physics_reuse_allowed") is False and transfer.get("constraints") == [])
        for hint_id, hint in hints.items():
            record = records.get(hint.get("source_evidence_id"), {})
            provenance, scope = hint.get("source_provenance", {}), hint.get("task_scope", {})
            key = str(rid) + "/" + str(hint_id)
            checks[key + "/source_record"] = (bool(record) and provenance.get("log") == record.get("source")
                and provenance.get("context_sha256") == digest(record.get("context"))
                and provenance.get("input_sha256") == record.get("context", {}).get("task", {}).get("input_sha256")
                and hint.get("source_part_id") == record.get("request", {}).get("part_move"))
            checks[key + "/reference_only"] = (hint.get("mapping_status") == "tentative_geometry_correspondence"
                and hint.get("current_physical_facts") is None and hint.get("current_attempt_verified") is False
                and hint.get("current_candidate_id") in request.get("candidate_ids", [])
                and hint.get("allow_hard_pruning") is False and hint.get("constraints") == [])
            if hint_id not in used:
                continue
            rank = ranks.get(rid, {})
            tested = []
            for pair, start in starts.items():
                attempt = start.get("request", {})
                if (start.get("generation_id") == rank.get("generation_id")
                    and attempt.get("part_move") == hint.get("current_candidate_id")
                    and sorted(attempt.get("parts_rest", []) + [attempt.get("part_move")]) == request.get("assembled_parts")
                    and pair in results):
                    tested.append(results[pair])
            chains.append({"request_id": rid, "hint_id": hint_id, "source_evidence_id": hint.get("source_evidence_id"),
                "current_candidate_id": hint.get("current_candidate_id"),
                "different_mesh_inputs": scope.get("mesh_input_bytes_differ") is True,
                "different_geometry_descriptors": scope.get("geometry_descriptor_multiset_differs") is True,
                "source_descriptor_present": isinstance(hint.get("source_geometry_descriptor"), dict),
                "current_descriptor_present": isinstance(hint.get("current_geometry_descriptor"), dict),
                "cited": True, "worker_order_applied": rank.get("applied_order") == response.get("candidate_ids"),
                "actual_verifier_event_ids": [r.get("event_id") for r in tested],
                "actual_verifier_statuses": [r.get("status") for r in tested],
                "cited_candidate_tested": bool(tested), "cited_candidate_passed": any(r.get("status") == "PASS" for r in tested),
                "actual_llm_cited": metadata.get("llm_live") is True and metadata.get("attempted") is True,
                "decision_source": metadata.get("ranking_source"),
                "interpretation": "model-cited reference; candidate testing is reported separately; not causal proof",
                "causal_benefit_demonstrated": False})
    return checks, chains


def ledger_summary(path):
    """Immutable read of a quiescent ledger, with all sidecars unchanged."""
    path = safe(path)
    if not path.is_file():
        return {"status": "NOT_RUN", "path": str(path)}
    before, sidecars = sha(path), {}
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = safe(Path(str(path) + suffix))
        sidecars[suffix] = sha(sidecar) if sidecar.is_file() else None
        if suffix in ("-wal", "-journal") and sidecar.is_file() and sidecar.stat().st_size:
            return {"status": "UNVERIFIED_ACTIVE_SQLITE_SIDECAR", "path": str(path)}
    db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT reserved_call,reserved_usd,actual_usd,status,response_json FROM rank_requests").fetchall()
        policy = db.execute("SELECT max_calls,max_usd FROM batch_policy WHERE singleton=1").fetchone()
    finally:
        db.close()
    reserved = sum((Decimal(r["reserved_usd"]) for r in rows), Decimal(0))
    cost = sum((Decimal(r["actual_usd"] or "0") for r in rows), Decimal(0))
    committed = sum((max(Decimal(r["reserved_usd"]), Decimal(r["actual_usd"] or "0")) for r in rows), Decimal(0))
    attempts = valid = 0
    for row in rows:
        response = json.loads(row["response_json"]) if row["response_json"] else {}
        metadata = response.get("metadata", {})
        live = metadata.get("llm_live") is True and metadata.get("attempted") is True
        attempts += live
        valid += live and response.get("fallback_reason") is None and metadata.get("ranking_source") == "prototype_llm"
    after_sidecars = {s: sha(Path(str(path) + s)) if safe(Path(str(path) + s)).is_file() else None for s in sidecars}
    calls = sum(r["reserved_call"] for r in rows)
    return {"status": "PASS", "path": str(path), "read_mode": "ro_immutable_quiescent",
        "file_sha256": before, "file_and_sidecars_unchanged": before == sha(path) and sidecars == after_sidecars,
        "requests": len(rows), "reserved_calls": calls, "actual_live_attempts": attempts, "actual_live_valid_responses": valid,
        "pending_requests": sum(r["status"] == "pending" for r in rows), "reserved_usd": str(reserved),
        "usage_estimated_usd": str(cost), "committed_usd": str(committed),
        "max_calls": policy["max_calls"], "max_usd": policy["max_usd"],
        "within_caps": calls <= policy["max_calls"] and committed <= Decimal(policy["max_usd"])}


def audit_case(directory, label):
    directory = safe(directory)
    if not directory.is_dir():
        return {"status": "NOT_RUN", "label": label}, None
    result, config = read(directory / "result.json"), read(directory / "config.json", {})
    if result is None:
        return {"status": "INCOMPLETE", "label": label, "reason": "result.json absent"}, config.get("ledger")
    events, decisions = jsonl(directory / "events.jsonl"), jsonl(directory / "prototype_decisions.jsonl")
    parsed = read(directory / "parsed_result.json", {})
    starts, results, checks = evidence_pairs(events)
    checks.update(source_archive_checks(directory))
    begin = [e for e in events if e.get("kind") == "run_start"]
    end = [e for e in events if e.get("kind") == "run_end"]
    checks["pinned_actual_upstream_run"] = (len(begin) == 1 and begin[0].get("source_commit") == COMMIT
        and begin[0].get("tracked_source_clean") is True and begin[0].get("state_owner") == "upstream_DFS"
        and config.get("physics_backend") == result.get("verifier_backend") == "actual_ASAP_RedMax")
    checks["heuristic_tripwire_zero"] = len(end) == 1 and end[0].get("heuristic_tripwire_calls") == result.get("heuristic_tripwire_calls") == 0
    checks["stats_plan_success_consistent"] = parsed.get("plan_success") == result.get("plan_success")
    checks["native_artifacts_validated"] = parsed.get("validation_ok") is True and parsed.get("tree_checked") is True
    checks["recorded_verifier_counts"] = result.get("verifier_calls") == len(starts) and result.get("verifier_results") == len(results)
    ranks = {e["request"]["request_id"]: e for e in events if e.get("kind") == "rank_decision"}
    responses = {}
    response_dir = safe(directory / "ipc/responses")
    if response_dir.is_dir():
        for path in sorted(response_dir.glob("*.json")):
            response = read(path)
            responses[response.get("request_id")] = response
    for rid, rank in ranks.items():
        checks[rid + "/full_permutation"] = full_permutation(rank.get("request"), rank.get("applied_order"))
        response = responses.get(rid, {})
        checks[rid + "/IPC_applied"] = (response.get("candidate_ids") == rank.get("applied_order")
            and response.get("state_id") == rank.get("request", {}).get("state_id"))
    for decision in decisions:
        request, response = decision.get("request", {}), decision.get("response", {})
        rid, metadata = request.get("request_id"), response.get("metadata", {})
        checks[str(rid) + "/decision_permutation"] = full_permutation(request, response.get("candidate_ids"))
        ipc = responses.get(rid, {})
        checks[str(rid) + "/host_decision_IPC"] = (ipc.get("candidate_ids") == response.get("candidate_ids")
            and all(ipc.get("metadata", {}).get(k) == metadata.get(k) for k in
                ("ranking_source", "selected_skill_id", "used_hint_ids", "repair_invocation", "llm_live", "attempted")))
    memory = read(directory / "episode_memory.json", {})
    if events:
        checks.update({"new_memory/" + k: v for k, v in verify_memory(
            memory, directory / "events.jsonl", read(directory / "context.json"), events).items()})
    source_path = config.get("arguments", {}).get("memory")
    source_snapshot = read(safe(source_path), {}) if source_path else {"records": []}
    checks["frozen_source_memory_file"] = (sha(source_path) == config.get("frozen_memory_sha256")) if source_path else config.get("frozen_memory_sha256") is None
    if label.startswith("transfer"):
        previous = directory.parent / ("repair_live" if label.endswith("live") else "repair_fixture") / "episode_memory.json"
        fresh = read(previous, {})
        checks["source_is_previous_fresh_episode"] = (bool(fresh) and bool(source_path)
            and source_snapshot.get("records") == fresh.get("records")
            and source_snapshot.get("content_sha256") == fresh.get("content_sha256"))
        checks["frozen_snapshot_content_hash"] = source_snapshot.get("content_sha256") == digest(source_snapshot.get("records", []))
    transfer_checks, citations = transfer_chains(decisions, events, source_snapshot)
    checks.update({"transfer/" + k: v for k, v in transfer_checks.items()})
    recovery = repair_chains(decisions, events, read(directory / "repair_outcomes.json", []))
    for chain in recovery:
        checks[chain["invocation_id"] + "/repair_provenance"] = chain["raw_chain_valid"] and chain["reported_outcome_matches_raw"]
    expected_repair_hash = read(directory / "source_before.json", {}).get("orchestrator/repair_library.py")
    for decision in decisions:
        invocation = decision.get("response", {}).get("metadata", {}).get("repair_invocation")
        if isinstance(invocation, dict):
            checks[str(invocation.get("invocation_id")) + "/registered_code_version"] = (
                expected_repair_hash is not None and invocation.get("code_sha256") == expected_repair_hash)
    attempts = sum(r.get("metadata", {}).get("llm_live") is True and r.get("metadata", {}).get("attempted") is True for r in responses.values())
    valid = sum(r.get("metadata", {}).get("llm_live") is True and r.get("metadata", {}).get("attempted") is True
        and r.get("metadata", {}).get("ranking_source") == "prototype_llm" and r.get("fallback_reason") is None for r in responses.values())
    checks["recorded_API_counts"] = result.get("live_api_attempts") == attempts and result.get("live_valid_responses") == valid
    if label in ("repair_fixture", "transfer_fixture", EXT4_LABEL):
        checks["fixture_is_not_real_LLM"] = result.get("model_decision_fixture") is True and attempts == 0
    return {"label": label, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "failed_checks": [k for k, v in checks.items() if not v], "original_run_status": result.get("status"),
        "full_plan_success": result.get("plan_success"), "sequence": result.get("sequence"),
        "actual_physics_verifier_calls": len(starts), "verifier_statuses": dict(Counter(r.get("status") for r in results.values())),
        "decision_sources": dict(Counter(d.get("response", {}).get("metadata", {}).get("ranking_source") for d in decisions)),
        "actual_live_API_attempts": attempts, "actual_live_valid_responses": valid,
        "repair_chains": recovery, "transfer_citation_chains": citations,
        "actual_llm_recovery_passes": sum(c["actual_llm_selected"] and c["retry_pass_observed"] for c in recovery),
        "actual_llm_cross_geometry_cited_tested": sum(c["actual_llm_cited"] and c["different_mesh_inputs"]
            and c["different_geometry_descriptors"] and c["cited_candidate_tested"] for c in citations),
        "actual_llm_cross_geometry_cited_passed": sum(c["actual_llm_cited"] and c["different_mesh_inputs"]
            and c["different_geometry_descriptors"] and c["cited_candidate_passed"] for c in citations),
        "performance_comparison": "NOT_RUN", "causal_memory_or_repair_benefit": "UNVERIFIED"}, config.get("ledger")


def audit_preservation(run):
    baseline = read(run / "preservation_before.json", {})
    changed, missing = [], []
    for name, expected in baseline.items():
        path = safe(ROOT / name)
        if not path.is_file():
            missing.append(name)
        elif sha(path) != expected:
            changed.append(name)
    return {"status": "PASS" if baseline and not changed and not missing else "FAIL",
        "expected_file_count": len(baseline), "changed": changed, "missing": missing,
        "new_files_are_not_previous_file_modifications": True}


def retry_gate_checks(run, manifest_path):
    """Verify a predeclared I/O retry without deleting the failed first case.

    Input hashes and solver settings must match. Operational timeout changes,
    if any, are reported separately; no performance improvement is inferred.
    """
    manifest_path = safe(manifest_path)
    manifest = read(manifest_path, {})
    first, retry = run / "transfer_fixture", run / EXT4_LABEL
    first_config, retry_config = read(first / "config.json", {}), read(retry / "config.json", {})
    first_features, retry_features = read(first / "features.json", {}), read(retry / "features.json", {})
    first_task, retry_task = first_features.get("task", {}), retry_features.get("task", {})
    original, command = read(first / "result.json", {}), read(first / "execute/command.json", {})
    retry_command = read(retry / "execute/command.json", {})
    try:
        declared = datetime.fromisoformat(manifest["created_utc"].replace("Z", "+00:00"))
        launched = datetime.fromisoformat(retry_command["started_at"].replace("Z", "+00:00"))
        before_execution = declared <= launched
    except (ValueError, TypeError, KeyError):
        before_execution = False
    checks = {
        "explicit_source_and_target": manifest.get("source_label") == "transfer_fixture"
            and manifest.get("target_label") == EXT4_LABEL,
        "explicit_reason": type(manifest.get("reason")) is str and bool(manifest["reason"].strip()),
        "declared_before_execution": manifest.get("declared_before_execution") is True and before_execution,
        "original_exit_124_preserved": command.get("exit_code") == original.get("process_exit") == 124,
        "original_verifier_zero_preserved": original.get("verifier_calls") == original.get("verifier_results") == 0,
        "original_has_no_success_claim": original.get("plan_success") is not True,
        "original_source_snapshot_preserved": all(source_archive_checks(first).values()),
        "same_nonempty_mesh_and_JSON_hashes": bool(first_task.get("input_sha256"))
            and first_task.get("input_sha256") == retry_task.get("input_sha256"),
        "same_solver_settings": bool(first_task.get("solver_settings"))
            and first_task.get("solver_settings") == retry_task.get("solver_settings"),
        "same_base_and_goal": first_task.get("base_part") == retry_task.get("base_part")
            and first_task.get("goal") == retry_task.get("goal"),
        "same_frozen_memory": first_config.get("frozen_memory_sha256") is not None
            and first_config.get("frozen_memory_sha256") == retry_config.get("frozen_memory_sha256"),
    }
    args1, args2 = first_config.get("arguments", {}), retry_config.get("arguments", {})
    physics_keys = ("mode", "seed", "budget", "max_gripper", "max_pose", "timeout", "base_part", "controlled_repair")
    checks["same_declared_physics_arguments"] = all(args1.get(k) == args2.get(k) for k in physics_keys)
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path),
        "manifest": manifest, "source_label": "transfer_fixture", "functional_gate_label": EXT4_LABEL,
        "original_failure_remains_in_cases": True, "selection_is_not_based_on_retry_success": before_execution,
        "operational_differences": {k: {"original": args1.get(k), "retry": args2.get(k)}
            for k in sorted(set(args1) | set(args2)) if args1.get(k) != args2.get(k)},
        "scope": "bounded functional I/O retry, not a performance comparison"}


def audit(run_dir, output, transfer_fixture_label="transfer_fixture", retry_manifest=None):
    run, output = safe(run_dir), safe(output)
    if not run.is_dir() or output.exists():
        raise ValueError("run must exist and audit output must be new")
    try:
        output.relative_to(run)
    except ValueError:
        raise ValueError("audit output must be inside the selected run")
    if transfer_fixture_label not in ("transfer_fixture", EXT4_LABEL):
        raise ValueError("unsupported_functional_transfer_label")
    if (transfer_fixture_label == EXT4_LABEL) != (retry_manifest is not None):
        raise ValueError("explicit_ext4_gate_requires_retry_manifest")
    cases, ledger_paths = {}, set()
    labels = list(LABELS)
    if safe(run / EXT4_LABEL).is_dir() or transfer_fixture_label == EXT4_LABEL:
        labels.append(EXT4_LABEL)
    for label in labels:
        try:
            case, ledger = audit_case(run / label, label)
        except (ValueError, TypeError, KeyError, OSError) as error:
            case, ledger = {"status": "ERROR", "label": label, "error_type": type(error).__name__, "error": str(error)}, None
        cases[label] = case
        if ledger:
            ledger_paths.add(str(safe(ledger)))
    ledgers = []
    for path in sorted(ledger_paths):
        try:
            ledgers.append(ledger_summary(path))
        except (ValueError, TypeError, KeyError, OSError, sqlite3.Error) as error:
            ledgers.append({"status": "ERROR", "path": path, "error_type": type(error).__name__})
    preservation = audit_preservation(run)
    retry_gate = retry_gate_checks(run, retry_manifest) if retry_manifest is not None else None
    excluded_historical = {"transfer_fixture"} if transfer_fixture_label == EXT4_LABEL else set()
    checked_cases = [c for label, c in cases.items() if c["status"] != "NOT_RUN" and label not in excluded_historical]
    integrity = (preservation["status"] == "PASS" and all(c["status"] == "PASS" for c in checked_cases)
        and all(l["status"] == "PASS" and l["file_and_sidecars_unchanged"] and l["within_caps"] for l in ledgers)
        and (retry_gate is None or retry_gate["status"] == "PASS"))
    live_pending = any(cases[k]["status"] == "NOT_RUN" for k in ("repair_live", "transfer_live"))
    demonstrated = (cases["repair_live"].get("actual_llm_recovery_passes", 0) > 0
        and cases["transfer_live"].get("actual_llm_cross_geometry_cited_tested", 0) > 0)
    summary = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity else "FAIL", "cases": cases, "ledgers": ledgers,
        "preservation": preservation, "live_execution_pending": live_pending,
        "functional_transfer_fixture_label": transfer_fixture_label,
        "retry_gate": retry_gate,
        "historical_failed_attempts": [cases[k] for k in sorted(excluded_historical)],
        "live_prototype_behavior_demonstrated": integrity and not live_pending and demonstrated,
        "actual_live_API_attempts": sum(l.get("actual_live_attempts", 0) for l in ledgers),
        "usage_estimated_usd": str(sum((Decimal(l.get("usage_estimated_usd", "0")) for l in ledgers), Decimal(0))),
        "limitations": ["Fixture decisions are not actual LLM responses.",
            "Full-plan success is distinct from cited-reference or repair execution.",
            "Citation followed by checking a candidate does not prove causal benefit.",
            "One failure/change/retry PASS remains a provisional scoped recovery.",
            "No performance comparison or generalization effect is claimed.",
            "Actual physics attribution uses pinned source/events/artifact validation; no physics replay."],
        "audit_execution": {"API_calls": 0, "physics_calls": 0, "WSL_calls": 0,
            "SQLite_write": False, "existing_file_write": False, "E_drive_access": False}}
    output.mkdir()
    with (output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transfer-fixture-label", choices=("transfer_fixture", EXT4_LABEL), default="transfer_fixture")
    parser.add_argument("--retry-manifest", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir, args.output, args.transfer_fixture_label, args.retry_manifest)
    print(json.dumps({"status": report["status"], "cases": {k: v["status"] for k, v in report["cases"].items()},
        "preserved_file_count": report["preservation"]["expected_file_count"],
        "live_execution_pending": report["live_execution_pending"],
        "live_prototype_behavior_demonstrated": report["live_prototype_behavior_demonstrated"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
