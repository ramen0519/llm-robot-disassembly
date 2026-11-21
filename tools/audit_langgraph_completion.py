"""Read-only completion audit for the four no-live-API integration runs.

Run only after the integration process has exited. No graph, worker, pickle,
physics, transport, or mutable SQLite connection is invoked. The only new output
is an exclusively created completion_audit.json. Requires ormsgpack from the
project's orchestrator environment to decode primitive checkpoint data safely.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat


PROJECT = Path(__file__).absolute().parent.parent
VARIANTS = ("native_trace", "remote_noop", "remote_disabled", "remote_reverse")
PHASES = ("validate", "prepare", "execute", "collect", "summarize")
STATE_KEYS = ("run_id", "status", "config", "artifacts", "metrics", "error_type", "phase")
COMMIT = "3727e716b57e0b508c4e9ecd9fefa6953e8f72da"
AMENDED_SOURCE = "tools/run_langgraph_integration.py"
ORIGINAL_FAILED_CHECKS = {
    "remote_reverse/remote_roundtrip_used", "remote_reverse/expected_order_change",
}
SOURCES = (
    "worker/rank_contract.py", "worker/rank_hook.py", "worker/run_rank_hook.py",
    "worker/compare_rank_hook.py", "worker/validate_baseline.py", "worker/remote_rank_client.py",
    "worker/run_remote_rank.py", "worker/describe_beam.py", "worker/export_graph_artifacts.py",
    "orchestrator/__init__.py", "orchestrator/file_service.py", "orchestrator/llm_ranker.py",
    "orchestrator/langgraph_workflow.py", "tools/run_langgraph_integration.py",
    "tools/run_recorded.py", "tools/run_memory_multiseed.py", "tools/run_asap_baseline.py",
)


def safe(path):
    raw = str(path).replace("\\", "/")
    if ".." in raw.split("/"):
        raise ValueError("parent traversal is excluded")
    if str(PROJECT).replace("\\", "/").lower() != "c:/dev/disassembly_codex_ready_v2":
        raise ValueError("audit must run from the C: project")
    path = Path(path).absolute()
    if not path.is_relative_to(PROJECT):
        raise ValueError("path is outside the C: project")
    for item in reversed((path,) + tuple(path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("links and reparse points are excluded")
    return path


def digest(path):
    value = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value, ensure_ascii=False):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=ensure_ascii, allow_nan=False).encode("utf-8")


def json_object(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite JSON value")
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


class Audit:
    def __init__(self, root):
        self.root = root
        self.checks, self.inputs, self.errors, self.schemas = {}, {}, [], {}

    def check(self, name, value):
        if name in self.checks:
            raise ValueError("duplicate audit check")
        self.checks[name] = bool(value)

    def read(self, path, lines=False):
        path = safe(path)
        data = path.read_bytes()
        self.inputs[str(path.relative_to(PROJECT)).replace("\\", "/")] = hashlib.sha256(data).hexdigest()
        if lines:
            return [json_object(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        return json_object(data.decode("utf-8"))

    def sha(self, path):
        path = safe(path)
        result = digest(path)
        self.inputs[path.relative_to(PROJECT).as_posix()] = result
        return result

    def section(self, name, function):
        try:
            function()
            self.check(name + "/completed", True)
        except Exception as error:
            self.check(name + "/completed", False)
            self.errors.append({"section": name, "error_type": type(error).__name__})

    def command(self, folder, name):
        data = self.read(folder / "command.json")
        self.check(name + "/exit_zero", data.get("exit_code") == 0 and data.get("launch_error") is None)
        self.check(name + "/elapsed_recorded", type(data.get("elapsed_seconds")) in (int, float)
                   and data["elapsed_seconds"] >= 0 and bool(data.get("started_at")))
        return data

    def database(self, path, required):
        path = safe(path)
        self.sha(path)
        # immutable=1 prevents reader-created WAL/SHM files. Refuse a live WAL
        # rather than silently ignoring its uncheckpointed data.
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = safe(Path(str(path) + suffix))
            if sidecar.exists():
                self.sha(sidecar)
                if suffix in ("-wal", "-journal") and sidecar.stat().st_size:
                    raise ValueError("database has an active journal; close its owner first")
        db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA trusted_schema=OFF")
            objects = {row["name"]: row["type"] for row in db.execute(
                "SELECT name,type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
            schema = {}
            for table, columns in required.items():
                if objects.get(table) != "table":
                    raise ValueError("expected static SQLite table is missing")
                found = [row["name"] for row in db.execute('PRAGMA table_info("' + table + '")')]
                if not set(columns).issubset(found):
                    raise ValueError("SQLite table schema mismatch")
                schema[table] = found
            self.schemas[path.relative_to(self.root).as_posix()] = schema
            return db
        except BaseException:
            db.close()
            raise


def unpack_checkpoint(kind, blob):
    if len(blob) > 16 * 1024 * 1024:
        raise ValueError("oversized checkpoint")
    if kind == "json":
        return json_object(blob.decode("utf-8"))
    if kind != "msgpack":
        raise ValueError("only primitive JSON/msgpack checkpoints are accepted")
    import ormsgpack
    def reject_extension(code, payload):
        raise ValueError("checkpoint object extensions are not accepted")
    return ormsgpack.unpackb(blob, ext_hook=reject_extension)


def audit_checkpoint(audit, run, state):
    variant, run_id = run.name, state["run_id"]
    required = {
        "workflow_invocations": ("thread_id", "run_id", "config_sha256", "state"),
        "checkpoints": ("thread_id", "checkpoint_ns", "checkpoint_id", "type", "checkpoint", "metadata"),
        "writes": ("thread_id", "checkpoint_ns", "checkpoint_id", "channel", "type", "value"),
    }
    with closing(audit.database(run / "workflow.sqlite", required)) as db:
        claims = [dict(row) for row in db.execute("SELECT * FROM workflow_invocations")]
        expected = {"thread_id": run_id, "run_id": run_id, "state": "COMPLETED",
                    "config_sha256": hashlib.sha256(canonical(state["config"])).hexdigest()}
        audit.check(variant + "/sqlite_completed_claim", claims == [expected])
        identities = [tuple(row) for row in db.execute("SELECT DISTINCT thread_id,checkpoint_ns FROM checkpoints")]
        audit.check(variant + "/sqlite_thread_identity", identities == [(run_id, "")])
        latest = db.execute("SELECT checkpoint_id,type,checkpoint,metadata FROM checkpoints "
                            "WHERE thread_id=? AND checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1", (run_id,)).fetchone()
        if latest is None:
            raise ValueError("no actual checkpoint")
        checkpoint = unpack_checkpoint(latest["type"], latest["checkpoint"])
        metadata = json_object(latest["metadata"].decode("utf-8"))
        values = checkpoint["channel_values"]
        audit.check(variant + "/sqlite_final_state_matches_json",
                    {key: values.get(key) for key in STATE_KEYS} == state and all(key in values for key in STATE_KEYS))
        audit.check(variant + "/sqlite_end_marker", values.get("phase") == "END" and values.get("status") == "PASS"
                    and checkpoint.get("id") == latest["checkpoint_id"] and metadata.get("source") == "loop")
        pending = db.execute("SELECT COUNT(*) FROM writes WHERE thread_id=? AND checkpoint_ns='' AND checkpoint_id=?",
                             (run_id, latest["checkpoint_id"])).fetchone()[0]
        audit.check(variant + "/sqlite_no_pending_terminal_writes", pending == 0)


def artifact_hashes(run):
    output = {}
    for name in ("execute", "validation", "native", "ipc/requests", "ipc/responses"):
        folder = safe(run / name)
        if folder.exists():
            for path in walk_files(folder):
                output[path.relative_to(run).as_posix()] = digest(path)
    for name in ("events.jsonl", "remote_audit.jsonl", "parsed_result.json"):
        path = safe(run / name)
        if path.is_file():
            output[name] = digest(path)
    return output


def walk_files(folder, exclude=None):
    pending = [safe(folder)]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                if path == exclude:
                    continue
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise ValueError("refusing result link/reparse point")
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    yield path
                else:
                    raise ValueError("unsupported result file type")


def audit_preservation(audit, amendment=None):
    before, after = (audit.read(audit.root / name) for name in ("preserved_before.json", "preserved_after.json"))
    current = {path.relative_to(PROJECT).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
               for path in walk_files(PROJECT / "runs", exclude=audit.root)}
    audit.check("preservation/old_run_bytes_and_inventory", before == after == current)
    source_before, source_after = (audit.read(audit.root / name) for name in ("source_before.json", "source_after.json"))
    audit.check("preservation/expected_frozen_source_inventory", set(source_before) == set(SOURCES))
    current_sources = {name: audit.sha(PROJECT / name) for name in SOURCES}
    frozen_sources = {name: audit.sha(audit.root / "implementation" / name) for name in SOURCES}
    audit.check("preservation/original_frozen_sources", source_before == source_after == frozen_sources)
    if amendment is None:
        audit.check("preservation/current_sources", current_sources == source_before)
    else:
        changes = amendment.get("root_changes", {})
        audit.check("preservation/one_declared_postrun_source_amendment", set(changes) == {AMENDED_SOURCE}
                    and changes.get(AMENDED_SOURCE) == {
                        "old_sha256": source_before[AMENDED_SOURCE],
                        "new_sha256": current_sources[AMENDED_SOURCE],
                    } and source_before[AMENDED_SOURCE] != current_sources[AMENDED_SOURCE])
        audit.check("preservation/other_current_sources_unchanged", all(
            current_sources[name] == source_before[name] for name in SOURCES if name != AMENDED_SOURCE))


def audit_remote(audit, run, decisions, generations, orchestration, features):
    variant = run.name
    backend = {"remote_noop": "noop", "remote_disabled": "llm_disabled", "remote_reverse": "reverse_native"}[variant]
    reason = "llm_disabled" if variant == "remote_disabled" else None
    request_paths = sorted(safe(run / "ipc/requests").glob("*.json"))
    response_paths = sorted(safe(run / "ipc/responses").glob("*.json"))
    requests = [audit.read(path) for path in request_paths]
    responses = [audit.read(path) for path in response_paths]
    client = audit.read(run / "remote_audit.jsonl", lines=True)
    service = [item for item in orchestration if item["kind"] == "service_response"]
    expected_count = len(generations)
    audit.check(variant + "/one_roundtrip_per_actual_generation", expected_count > 0
                and len(decisions) == len(requests) == len(responses) == len(client) == len(service) == expected_count)
    indexes = [{item["request"]["request_id"]: item for item in requests},
               {item["request_id"]: item for item in responses},
               {item["request_id"]: item for item in client},
               {item["request"]["request_id"]: item for item in decisions},
               {item["request_id"]: item for item in service}]
    ids = set(indexes[0])
    audit.check(variant + "/unique_matching_request_ids", len(ids) == expected_count and all(set(index) == ids for index in indexes))
    generation_map = {item["generation_id"]: item for item in generations}
    response_path_map = {path.name: path for path in response_paths}
    for number, request_id in enumerate(sorted(ids), 1):
        envelope, response, observed, decision, served = [index[request_id] for index in indexes]
        request = envelope["request"]
        expected_order = list(reversed(request["candidate_ids"])) if variant == "remote_reverse" else request["candidate_ids"]
        name = variant + "/request" + str(number)
        filename = hashlib.sha256(request_id.encode("utf-8")).hexdigest() + ".json"
        audit.check(name + "/identity_feature_binding", request == decision["request"] and envelope["features"] == features
                    and request["run_id"] == audit.root.name + "/" + variant
                    and response["state_id"] == observed["state_id"] == served["state_id"] == request["state_id"]
                    and filename in {path.name for path in request_paths} and observed["filename"] == filename)
        audit.check(name + "/response_and_decision_order", response["candidate_ids"] == decision["applied_order"] == expected_order
                    and generation_map[decision["generation_id"]]["applied_order"] == expected_order
                    and sorted(expected_order) == sorted(request["candidate_ids"]))
        audit.check(name + "/expected_fallback", response["fallback_reason"] == observed["fallback_reason"] == served["fallback_reason"] == reason
                    and decision["fallback_reason"] == ("ranker_exception:RemoteRankingError" if reason else None))
        audit.check(name + "/backend_and_no_live_api", response["metadata"].get("backend") == backend == served["backend"]
                    and response["metadata"].get("llm_live") is False and response["metadata"].get("attempted") is False
                    and served.get("llm_live") is False and observed["response_metadata"] == response["metadata"])
        audit.check(name + "/raw_message_hashes", observed["request_sha256"] == hashlib.sha256(canonical(envelope)).hexdigest()
                    and response["metadata"]["envelope_sha256"] == hashlib.sha256(canonical(envelope, ensure_ascii=True)).hexdigest()
                    and observed["response_sha256"] == audit.sha(response_path_map[filename]))
        audit.check(name + "/no_client_retry", observed.get("request_created") is True and observed.get("duplicate_cached") is False)
    if variant == "remote_disabled":
        required = {"rank_requests": ("request_id", "payload_json", "payload_hash", "config_fingerprint", "status",
                                      "reserved_call", "reserved_usd", "call_id", "response_json", "usage_json", "actual_usd")}
        with closing(audit.database(run / "ipc/llm_ledger.sqlite", required)) as db:
            rows = [dict(row) for row in db.execute("SELECT * FROM rank_requests")]
        audit.check(variant + "/disabled_ledger_three_requests", len(rows) == 3 and {row["request_id"] for row in rows} == ids)
        audit.check(variant + "/disabled_ledger_zero_reservations", all(row["status"] == "completed"
                    and row["reserved_call"] == 0 and Decimal(row["reserved_usd"]) == 0
                    and row["call_id"] is None and row["usage_json"] is None and row["actual_usd"] is None for row in rows))
        for number, row in enumerate(rows, 1):
            saved = json_object(row["response_json"])
            audit.check(variant + "/ledger_payload" + str(number), json_object(row["payload_json"]) == indexes[0][row["request_id"]]
                        and hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest() == row["payload_hash"]
                        and saved["fallback_reason"] == "llm_disabled" and saved["metadata"]["attempted"] is False
                        and saved["metadata"]["llm_live"] is False)


def audit_variant(audit, variant, row, features, environment, reference):
    run = audit.root / variant
    state = audit.read(run / "graph_state.json")
    reused = audit.read(run / "reuse_check.json")
    parsed = audit.read(run / "parsed_result.json")
    run_id = audit.root.name + "/" + variant
    audit.check(variant + "/graph_end_and_plan_pass", state.get("phase") == row.get("graph_phase") == "END"
                and state.get("status") == row.get("graph_status") == parsed.get("status") == "PASS"
                and state.get("error_type") is None and row.get("graph_error_type") is None
                and parsed.get("plan_success") is True and row.get("plan_success") is True)
    audit.check(variant + "/graph_identity_and_features", state["run_id"] == run_id
                and state["config"] == {"variant": variant, "feature_sha256": audit.sha(audit.root / "features.json"),
                                        "seed": 0, "live_api_authorized": False, "memory_enabled": False})
    audit.check(variant + "/five_callbacks_once", reused["callback_counts"] == row["callbacks"] == {name: 1 for name in PHASES})
    reuse_keys = {variant + "/" + name for name in ("completed_state_reused", "no_callbacks_reexecuted", "no_worker_or_rank_artifacts_rewritten")}
    audit.check(variant + "/completed_reuse_checks", set(reused["checks"]) == reuse_keys and all(value is True for value in reused["checks"].values()))
    audit.check(variant + "/worker_artifacts_still_unchanged", reused["artifacts_sha256"] == artifact_hashes(run))
    execution = audit.command(run / "execute", variant + "/execute")
    validation = audit.command(run / "validation", variant + "/validation")
    audit.check(variant + "/recorded_process_metrics", state["metrics"]["process_exit"] == execution["exit_code"] == 0
                and state["metrics"]["validation_exit"] == validation["exit_code"] == 0
                and state["metrics"]["process_seconds"] == execution["elapsed_seconds"]
                and state["metrics"]["live_llm_calls"] == 0 and state["metrics"]["plan_success"] is True)
    audit_checkpoint(audit, run, state)
    orchestration = audit.read(run / "orchestrator_events.jsonl", lines=True)
    graph_events = [event["graph_event"] for event in orchestration if event["kind"] == "graph_event"]
    expected_graph = [(kind, phase) for phase in PHASES for kind in ("phase_before", "phase_after")] + [("checkpoint_reuse", "END")]
    audit.check(variant + "/phase_and_reuse_event_order", [(event["kind"], event["phase"]) for event in graph_events] == expected_graph
                and all(event["run_id"] == event["thread_id"] == run_id and event["error_type"] is None for event in graph_events))
    audit.check(variant + "/orchestration_ids_unique", len({event["log_event_id"] for event in orchestration}) == len(orchestration)
                and len({event["event_id"] for event in graph_events}) == len(graph_events)
                and all(event["kind"] in ("graph_event", "service_response") for event in orchestration))
    events = audit.read(run / "events.jsonl", lines=True)
    generations = [event for event in events if event["kind"] == "generation"]
    decisions = [event for event in events if event["kind"] == "rank_decision"]
    starts = [event for event in events if event["kind"] == "verifier_start"]
    verifier = [event for event in events if event["kind"] == "verifier_result"]
    audit.check(variant + "/actual_trace_complete", events[0]["kind"] == "run_start" and events[-1]["kind"] == "run_end"
                and events[-1]["status"] == "returned_from_upstream" and bool(starts)
                and len(starts) == len(verifier) and [event["call_id"] for event in starts] == [event["call_id"] for event in verifier]
                and len({event["event_id"] for event in events}) == len(events))
    audit.check(variant + "/candidate_and_rng_contract", bool(generations)
                and (variant == "remote_reverse" or len(generations) == 3)
                and all(sorted(event["native_order"]) == sorted(event["applied_order"]) for event in generations)
                and all(event["rng_before"] == event["rng_after"] for event in decisions))
    changed = sum(event["native_order"] != event["applied_order"] for event in generations)
    audit.check(variant + "/rows_match_raw_counts", row["verifier_calls"] == len(verifier)
                and row["unknown_calls"] == sum(event["status"] == "UNKNOWN" for event in verifier)
                and row["rank_requests"] == len(decisions) and row["changed_orders"] == changed
                and row["rank_fallbacks"] == [event["fallback_reason"] for event in decisions]
                and row["tree_sha256"] == parsed["tree_fingerprint_sha256"]
                and row["native_seconds"] == parsed["raw_stats"]["time"] and row["sequence"] == parsed["raw_stats"]["sequence"])
    audit.check(variant + "/expected_order_change", (
        changed > 0 and all(event["applied_order"] == list(reversed(event["native_order"])) for event in generations)
        and changed == sum(event["native_order"] != list(reversed(event["native_order"])) for event in generations)
    ) if variant == "remote_reverse" else changed == 0)
    native = run / "native/dfs-heur-out/s0/original"
    audit.check(variant + "/parsed_matches_native_json", audit.read(native / "stats.json") == parsed["raw_stats"]
                and audit.read(native / "setup.json") == parsed["setup"] == reference["setup"])
    audit.check(variant + "/input_source_cache_binary_unchanged", audit.read(audit.root / ("environment_after_" + variant) / "stdout.log") == environment)
    audit.command(audit.root / ("environment_after_" + variant), variant + "/environment_snapshot")
    if variant == "native_trace":
        audit.check(variant + "/no_remote_decisions", decisions == [])
    else:
        audit_remote(audit, run, decisions, generations, orchestration, features)
    if variant in ("remote_noop", "remote_disabled"):
        audit.check(variant + "/baseline_tree_and_stats_equal", parsed["tree_fingerprint_sha256"] == reference["tree_fingerprint_sha256"]
                    and {key: value for key, value in parsed["raw_stats"].items() if key != "time"}
                    == {key: value for key, value in reference["raw_stats"].items() if key != "time"})


def run_audit(root):
    audit = Audit(root)
    original = audit.read(root / "comparison.json")
    comparison, selected_comparison, amendment = original, "comparison.json", None
    if safe(root / "comparison_v2.json").is_file():
        selected_comparison = "comparison_v2.json"
        comparison = audit.read(root / selected_comparison)
        amendment = audit.read(root / "postrun_amendment.json")
        original_failed = {key for key, value in original.get("checks", {}).items() if value is not True}
        audit.check("amendment/original_reporting_failure_preserved", original.get("status") == "ERROR"
                    and original_failed == ORIGINAL_FAILED_CHECKS
                    and set(amendment.get("original_failed_checks", [])) == ORIGINAL_FAILED_CHECKS
                    and amendment.get("original_comparison_sha256") == audit.sha(root / "comparison.json"))
        audit.check("amendment/no_physics_rerun_declared", amendment.get("physics_rerun") is False)
        audit.check("amendment/observed_rows_preserved", comparison.get("rows") == original.get("rows"))
        audit.check("amendment/other_original_checks_preserved", all(
            comparison.get("checks", {}).get(key) is value
            for key, value in original.get("checks", {}).items() if key not in ORIGINAL_FAILED_CHECKS))
        audit.check("amendment/corrected_checks_pass", all(
            comparison.get("checks", {}).get(key) is True for key in ORIGINAL_FAILED_CHECKS))
    execution = audit.read(root / "execution_start.json")
    features = audit.read(root / "features.json")
    environment = audit.read(root / "environment_before/stdout.log")
    reference = audit.read(root / "native_trace/parsed_result.json")
    historical = audit.read(root / "historical_baseline.json")
    rows = comparison.get("rows", [])
    audit.check("comparison/pass_four_rows", comparison.get("status") == "PASS" and len(rows) == 4
                and [row.get("variant") for row in rows] == list(VARIANTS))
    audit.check("comparison/all_existing_checks_true", bool(comparison.get("checks"))
                and all(value is True for value in comparison["checks"].values()))
    baseline_keys = ("original_m1_same_setup", "original_m1_same_full_tree", "original_m1_same_stats_except_time")
    audit.check("comparison/historical_baseline_checks_present", all(comparison["checks"].get(key) is True for key in baseline_keys))
    audit.check("comparison/no_live_api_authorized_or_reported", comparison.get("live_llm_calls") == 0
                and comparison.get("llm_quality_evaluation") == "NOT_RUN" and execution.get("live_api_authorized") is False
                and execution.get("live_llm_calls") == 0 and execution.get("variants") == list(VARIANTS))
    audit.check("comparison/official_beam_settings", execution.get("seed") == 0 and execution.get("base_part") == "6"
                and execution.get("budget") == 400 and execution.get("max_gripper") == 2 and execution.get("max_pose") == 3
                and execution.get("timeout") is None and execution.get("memory_enabled") is False)
    audit.check("historical/baseline_export_still_equal", historical["status"] == "PASS" and historical["plan_success"] is True
                and historical["setup"] == reference["setup"] and historical["tree_fingerprint_sha256"] == reference["tree_fingerprint_sha256"]
                and {key: value for key, value in historical["raw_stats"].items() if key != "time"}
                == {key: value for key, value in reference["raw_stats"].items() if key != "time"})
    audit.check("features/observation_scope", features["source_commit"] == environment["commit"] == COMMIT
                and environment["tracked_source_clean"] is True and features["assembly"] == "beam_assembly/original"
                and features["base_part"] == "6" and set(features["parts"]) == {"0", "1", "2", "3", "6"}
                and features["verifier_calls"] == 0 and features["prior_plan_or_verdicts_used"] is False)
    audit.check("features/source_hashes_match_input_snapshot", bool(features["source_sha256"]) and all(
        environment["files"].get("/opt/asap-research/ASAP/assets/beam_assembly/original/" + name) == value
        for name, value in features["source_sha256"].items()))
    for label in ("environment_before", "feature_preparation", "historical_baseline_validation"):
        audit.section(label, lambda label=label: audit.command(root / label, label))
    row_map = {row["variant"]: row for row in rows}
    for variant in VARIANTS:
        audit.section(variant, lambda variant=variant: audit_variant(audit, variant, row_map[variant], features, environment, reference))
    audit.section("preservation", lambda: audit_preservation(audit, amendment))
    audit.check("audit/read_inputs_unchanged", all(digest(PROJECT / name) == value for name, value in audit.inputs.items()))
    failed = [name for name, value in audit.checks.items() if not value]
    return {"schema_version": 1, "status": "PASS" if audit.checks and not failed else "ERROR",
            "created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(root),
            "selected_comparison": selected_comparison, "postrun_amendment_present": amendment is not None,
            "scope": "post-execution artifact audit; no graph/worker/physics/API execution",
            "check_count": len(audit.checks), "checks": audit.checks, "failed_checks": failed,
            "errors": audit.errors, "sqlite_schemas": audit.schemas,
            "audited_input_sha256": audit.inputs, "live_llm_quality_evaluation": "NOT_RUN",
            "audit_script_sha256": digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = safe(args.run_dir)
    if root.parent != PROJECT / "runs":
        parser.error("run-dir must be a direct child of project runs")
    output = safe(root / "completion_audit.json")
    with output.open("x", encoding="utf-8") as stream:
        try:
            result = run_audit(root)
        except Exception as error:
            result = {"status": "ERROR", "errors": [{"section": "audit_setup", "error_type": type(error).__name__}],
                      "scope": "post-execution artifact audit", "check_count": 0,
                      "audit_script_sha256": digest(Path(__file__))}
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "output": str(output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
