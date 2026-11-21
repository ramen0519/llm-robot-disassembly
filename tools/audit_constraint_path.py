"""Read-only saved-evidence audit; no API, physics, pickle or SQLite writes.

Audit PASS means consistent evidence, including honestly recorded failures.
Each run's repair, whole-plan, replay and reuse outcomes remain separate.
Only one new, exclusive JSON output is written.
"""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_prototype_completion import safe, sha, digest, full_permutation, evidence_pairs
from worker.constraint_memory import _normal_records, POLICY

LABELS = ("fixture", "live_source", "live_reuse", "live_reuse_with_skill")
SKILL = "turn_force_two_leg_v1"


class Audit:
    def __init__(self):
        self.checks, self.inputs = [], {}

    def read(self, path, default=None, lines=False):
        path = safe(path)
        if not path.is_file():
            return default
        with path.open("rb") as stream:
            raw = stream.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError("audit_input_size_limit")
        self.inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)

    def check(self, name, passed, evidence=None):
        self.checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        return bool(passed)

    def protect(self, name, operation):
        try:
            return operation()
        except (OSError, ValueError, KeyError, TypeError, IndexError, sqlite3.DatabaseError) as error:
            self.check(name + "/artifact_error", False, {"error_type": type(error).__name__})
            return {"audit_status": "ERROR", "error_type": type(error).__name__}


def sidecar(folder, reference):
    if type(reference) is not str or not reference:
        raise ValueError("missing_sidecar_reference")
    text = reference.replace("\\", "/")
    if (PureWindowsPath(reference).drive or text.startswith("/") or ":" in text
            or any(part in ("", ".", "..") for part in text.split("/"))):
        raise ValueError("unsafe_sidecar_reference")
    return safe(folder / text)


def snapshot(audit, path):
    value = audit.read(path)
    if type(value) is not dict or value.get("policy") != POLICY:
        raise ValueError("invalid_constraint_snapshot")
    records = value.get("records")
    if (type(records) is not list or records != _normal_records(records)
            or value.get("record_count") != len(records)
            or value.get("content_sha256") != digest(records)
            or value.get("hard_pruning") is not False or value.get("reuse_verdict") is not False):
        raise ValueError("constraint_snapshot_integrity_mismatch")
    return value


def archives(audit, folder, label):
    before, after = audit.read(folder / "source_before.json", {}), audit.read(folder / "source_after.json", {})
    audit.check(label + "/runtime_sources_unchanged", bool(before) and before == after)
    failures = []
    for name, expected in before.items():
        path = sidecar(folder / "implementation", name)
        if not path.is_file() or sha(path) != expected:
            failures.append(name)
    audit.check(label + "/source_archive_hashes", bool(before) and not failures,
                {"files": len(before), "mismatches": failures})
    return {"files": len(before), "before_equals_after": bool(before) and before == after,
            "archive_mismatches": failures, "historical_versions_not_required_to_equal_current_source": True}


def constraints(audit, folder, label, context):
    frozen = snapshot(audit, folder / "constraint_memory.json")
    measurements = []
    for record in frozen["records"]:
        prefix = label + "/constraint/" + record["record_id"]
        origin = safe(record["source"]["ref"])
        event = audit.read(origin)
        audit.check(prefix + "/raw_source", event == record["evidence"]
                    and sha(origin) == record["source"]["sha256"]
                    and digest(event) == record["evidence_sha256"])
        trajectory_file = sidecar(folder, event["trajectory"]["relative_path"])
        trajectory = audit.read(trajectory_file)
        path = trajectory.get("path", [])
        audit.check(prefix + "/saved_endpoint", bool(path)
                    and sha(trajectory_file) == event["trajectory"]["sha256"]
                    and len(path) == event["trajectory"]["states"]
                    and path[-1] == event["terminal_q"] and trajectory.get("raw_success") is False
                    and event["scope"]["task_context"] == context)
        predicate = record["predicate"]
        pair = next(p for p in event["measurements"]["pairs"] if p["rest_part_id"] == predicate["rest_part_id"])
        audit.check(prefix + "/bounded_measured_predicate", pair["min_distance"] < event["measurements"]["min_sep"]
                    and pair["separation_unsatisfied"] is True and record["verdict"] == "REJECTED"
                    and record["verdict_scope"] == "measured_endpoint_predicate_only"
                    and record["whole_part_verdict"] == "UNKNOWN" and record["blocker"] is None
                    and record["all_paths_impossible"] is False
                    and record["hard_pruning"] is False and record["reuse_verdict"] is False)
        measurements.append({"record_id": record["record_id"], "event_id": event["event_id"],
            "moving_part": predicate["moving_part_id"], "measured_rest_part": predicate["rest_part_id"],
            "directional_distances": [pair["d_rest_move"], pair["d_move_rest"]],
            "minimum_distance": predicate["measured_min_distance"], "required_min_sep": predicate["required_min_sep"],
            "action_schedule": event["scope"]["action_schedule"], "scope_sha256": record["scope_sha256"],
            "geometry_queries": event["measurements"]["geometry_queries"],
            "measurement_seconds": event["measurements"]["measurement_seconds"],
            "whole_part_verdict": "UNKNOWN", "blocker": None,
            "units": "ASAP_model_units_not_calibrated_metres"})
    return frozen, measurements


def replays(audit, folder, label, outcomes, parsed):
    files = sorted(safe(folder).glob("replay*.json"), key=lambda p:
                   [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)])
    history, by_id = [], {o["request_id"]: o for o in outcomes}
    for file in files:
        value = audit.read(file)
        tree = safe(parsed["log_dir"]) / "tree.pkl"
        common = (value.get("actual_physics") is True and value.get("api_calls") == 0
                  and value.get("source_outcomes_sha256") == sha(folder / "path_outcomes.json")
                  and value.get("parsed_result_sha256") == sha(folder / "parsed_result.json")
                  and value.get("saved_outcomes_unchanged") is True
                  and value.get("saved_selected_plan_unchanged") is True
                  and value.get("input_files_unchanged") is True and value.get("sdf_cache_unchanged") is True
                  and value.get("source_and_binary_unchanged") is True
                  and tree.is_file() and value.get("tree_file_sha256") == sha(tree))
        audit.check(label + "/" + file.name + "/input_binding", common)
        details = []
        for item in value.get("replays", []):
            outcome = by_id.get(item.get("request_id"), {})
            binding, comparison = item.get("plan_binding", {}), item.get("comparison", {})
            bound = (bool(outcome) and binding.get("selected_plan_edge") is True
                     and item.get("selected_plan_edge") is True
                     and binding.get("extension_sha256") == digest(outcome)
                     and binding.get("parent_call_id") == outcome.get("parent_call_id")
                     and binding.get("parent") == outcome.get("scope", {}).get("assembled_parts")
                     and binding.get("child") == sorted(outcome.get("scope", {}).get("parts_rest", [])))
            passed = (common and bound and item.get("status") == "PASS"
                      and item.get("path_success") is True and item.get("stability_success") is True
                      and item.get("support_set_equal") is True and item.get("path_check_calls", 0) >= 1
                      and item.get("stability_check_calls", 0) >= 1
                      and item.get("source_binary_controller_hashes_equal") is True
                      and item.get("source_sdf_cache_hashes_equal") is True
                      and comparison.get("continuous_single_simulation") is True
                      and comparison.get("path_allclose") is True and comparison.get("path_hash_equal") is True
                      and comparison.get("switch_count") == 1 and comparison.get("noncollinear") is True)
            if item.get("status") == "PASS":
                audit.check(label + "/" + file.name + "/reported_replay_PASS", passed)
            details.append({"request_id": item.get("request_id"), "reported_status": item.get("status"),
                "verified_pass": bool(passed), "plan_binding": binding, "binding_matches_outcome": bound,
                "path_calls": item.get("path_check_calls"), "stability_calls": item.get("stability_check_calls"),
                "internal_stability_evaluations": item.get("internal_stability_evaluations"),
                "maximum_path_difference": comparison.get("maximum_absolute_path_difference"),
                "error_type": item.get("error_type"), "error": item.get("error")})
        passed = value.get("status") == "PASS" and bool(details) and all(d["verified_pass"] for d in details)
        if value.get("status") == "PASS":
            audit.check(label + "/" + file.name + "/aggregate_replay_PASS", passed)
        history.append({"file": str(file), "reported_status": value.get("status"), "verified_pass": passed,
                        "cache_policy": value.get("cache_policy"), "replays": details})
    return {"history": history, "latest_status": history[-1]["reported_status"] if history else "NOT_RUN",
            "latest_verified_pass": history[-1]["verified_pass"] if history else False,
            "historical_failures_preserved": sum(h["reported_status"] != "PASS" for h in history)}


def offered_skill_cards(audit, envelope, frozen_skill, prefix):
    features = envelope["features"]
    cards = features.get("frozen_skill_evidence", [])
    available = {r["skill_evidence_id"]: r for r in (frozen_skill or {}).get("records", [])}
    for card in cards:
        identity = card["skill_evidence_id"]
        record = available.get(identity, {})
        # A card adds applicability labels but preserves every projected source field.
        shared = {key: value for key, value in card.items() if key in record}
        audit.check(prefix + "/frozen_skill_card/" + identity, bool(record)
                    and all(record[key] == value for key, value in shared.items())
                    and card.get("force_new_physics") is True
                    and card.get("current_attempt_verified") is False
                    and card.get("hard_pruning") is False and card.get("reuse_verdict") is False)
    return {card["skill_evidence_id"]: card for card in cards}


def audit_case(audit, folder, label, source_ids):
    result = audit.read(folder / "result.json")
    if result is None:
        return {"audit_status": "NOT_RUN", "functional_status": "NOT_RUN",
                "existing_directory": folder.exists()}
    mark = len(audit.checks)
    config = audit.read(folder / "config.json")
    parsed = audit.read(folder / "parsed_result.json", {})
    context = audit.read(folder / "context.json")
    events = audit.read(folder / "events.jsonl", [], lines=True)
    decisions = audit.read(folder / "prototype_decisions.jsonl", [], lines=True)
    raw_outcomes = audit.read(folder / "path_outcomes.json", {})
    outcomes = raw_outcomes.get("outcomes", [])
    source_archive = archives(audit, folder, label)
    frozen, measurements = constraints(audit, folder, label, context)
    current_ids = {r["record_id"] for r in frozen["records"]}
    arguments = config["arguments"]
    prior = None
    if arguments.get("memory"):
        path = safe(arguments["memory"])
        prior = snapshot(audit, path)
        audit.check(label + "/frozen_constraint_file_unchanged",
                    sha(path) == config.get("frozen_memory_sha256") and result.get("memory_frozen") is True)
    skill_snapshot = None
    if arguments.get("skill_memory"):
        from orchestrator.path_skill_memory import _validate_snapshot
        path = safe(arguments["skill_memory"])
        skill_snapshot = audit.read(path)
        # Validators hash source files only; they never invoke physics or deserialize a tree.
        _validate_snapshot(skill_snapshot, check_sources=True)
        audit.check(label + "/frozen_skill_file_unchanged",
                    sha(path) == config.get("frozen_skill_memory_sha256")
                    and result.get("skill_memory_frozen") is True)
    starts, results, evidence_checks = evidence_pairs(events)
    for name, passed in evidence_checks.items():
        audit.check(label + "/events/" + name, passed)
    statuses = Counter(e.get("status") for e in results.values())
    audit.check(label + "/verifier_counts", result.get("verifier_calls") == len(starts)
                and result.get("verifier_results") == len(results)
                and all(result.get("verifier_statuses", {}).get(key, 0) == statuses.get(key, 0)
                        for key in ("PASS", "UNKNOWN", "ERROR")))
    actual_backend = (result.get("verifier_backend") == "actual_ASAP_RedMax"
                      and config.get("physics_backend") == "actual_ASAP_RedMax")
    audit.check(label + "/actual_physics_backend", actual_backend)
    whole_plan = (parsed.get("validation_ok") is True and parsed.get("plan_success") is True
                  and parsed.get("status") == "PASS" and parsed.get("tree_checked") is True
                  and parsed.get("raw_stats", {}).get("success") is True
                  and parsed.get("raw_stats", {}).get("sequence") == result.get("sequence"))
    if result.get("plan_success") is True:
        audit.check(label + "/reported_whole_plan_PASS", whole_plan)
    audit.check(label + "/no_heuristic_calls", result.get("heuristic_tripwire_calls") == 0)
    audit.check(label + "/no_generic_action_only_memory_export",
                result.get("generic_episode_memory_export") == "NOT_USED_action_field_requires_schedule_sidecar")
    by_request, decision_details = {}, []
    path_fallbacks = candidate_live = path_live = live_attempts = valid_live = 0
    forbidden = {"native_order", "heuristic_score", "heuristic_rank", "fallback_order", "recommended_order"}

    def forbidden_keys(value):
        if isinstance(value, dict):
            return sorted((set(value) & forbidden) | {k for v in value.values() for k in forbidden_keys(v)})
        if isinstance(value, list):
            return sorted({k for v in value for k in forbidden_keys(v)})
        return []

    for item in decisions:
        envelope, response = item["request"], item["response"]
        request, features, metadata = envelope["request"], envelope["features"], response.get("metadata", {})
        identity = request["request_id"]
        audit.check(label + "/unique_decision/" + identity, identity not in by_request)
        by_request[identity] = item
        audit.check(label + "/complete_permutation/" + identity,
                    full_permutation(request, response.get("candidate_ids"))
                    and response.get("request_id") == identity
                    and response.get("state_id") == request.get("state_id"))
        audit.check(label + "/no_heuristic_input/" + identity, not forbidden_keys(features))
        live = metadata.get("llm_live") is True and metadata.get("attempted") is True
        valid = live and response.get("fallback_reason") is None
        live_attempts += int(live)
        valid_live += int(valid)
        path_request = features.get("protocol") == "constraint_path_repair_v1"
        path_live += int(live and path_request)
        candidate_live += int(live and not path_request)
        if live:
            audit.check(label + "/live_identity/" + identity, metadata.get("model_requested") == "gpt-6-luna"
                        and metadata.get("model_returned") == "gpt-6-luna"
                        and metadata.get("retries") == 0 and bool(metadata.get("response_id")))
        detail = {"request_id": identity, "route": "path" if path_request else "candidate",
                  "live_attempt": live, "valid_live": valid, "ranking_source": metadata.get("ranking_source"),
                  "fallback_reason": response.get("fallback_reason"),
                  "model_returned": metadata.get("model_returned"), "usage": metadata.get("usage"),
                  "configured_rate_cost_usd": metadata.get("actual_cost_from_usage_usd"),
                  "applied_candidate_order": response.get("candidate_ids") if not path_request else None}
        if path_request:
            path_fallbacks += int(response.get("fallback_reason") is not None)
            refs = {ref["record_id"]: ref for ref in features.get("constraint_references", [])}
            skill = metadata.get("path_skill") or {}
            used = set(skill.get("used_constraint_ids", []))
            cards = offered_skill_cards(audit, envelope, skill_snapshot, label + "/" + identity)
            used_cards = set(skill.get("used_skill_evidence_ids", []))
            linked = set(skill.get("linked_source_constraint_ids", []))
            audit.check(label + "/offered_constraint_ids/" + identity, used.issubset(refs))
            audit.check(label + "/offered_skill_ids/" + identity, used_cards.issubset(cards))
            expected_linked = {cid for card_id in used_cards for cid in cards[card_id]["failed_constraint_ids"]}
            audit.check(label + "/automatic_skill_links/" + identity, linked == expected_linked)
            for rid, reference in refs.items():
                expected = next((r for r in frozen["records"] if r["record_id"] == rid), None)
                expected_origin = "current_episode_measurement"
                if expected is None:
                    expected = next((r for r in (prior or {}).get("records", []) if r["record_id"] == rid), None)
                    expected_origin = "frozen_previous_episode"
                audit.check(label + "/reference_provenance/" + rid, expected is not None
                            and reference.get("source_scope") == expected["scope"]
                            and reference.get("evidence_origin", {}).get("source") == expected["source"]
                            and reference.get("origin") == expected_origin
                            and reference.get("reuse_verdict") is False
                            and reference.get("hard_pruning") is False)
            old_cards = sorted(cid for cid in used_cards
                               if set(cards[cid]["failed_constraint_ids"]) & source_ids)
            detail.update(selected_skill=skill, offered_constraint_ids=sorted(refs),
                direct_current_constraint_ids=sorted(used & current_ids),
                direct_previous_source_constraint_ids=sorted(used & source_ids) if label != "live_source" else [],
                cited_previous_source_skill_evidence_ids=old_cards,
                automatic_linked_previous_constraint_ids=sorted(linked & source_ids),
                direct_constraint_citation_is_distinct_from_automatic_link=True,
                reference_origins={rid: ref.get("origin") for rid, ref in refs.items()},
                offered_skill_evidence_ids=sorted(cards),
                matched_recorded_schedule=metadata.get("matching_schedule"))
        decision_details.append(detail)
    audit.check(label + "/recorded_API_counts", result.get("live_api_attempts") == live_attempts
                and result.get("live_valid_responses") == valid_live
                and result.get("path_decision_fallbacks") == path_fallbacks)
    rank_events = [e for e in events if e.get("kind") == "rank_decision"]
    rank_fallbacks = sum(e.get("fallback_reason") is not None for e in rank_events)
    audit.check(label + "/rank_fallback_count", result.get("rank_fallbacks") == rank_fallbacks)
    for event in rank_events:
        request = event["request"]
        response = by_request.get(request["request_id"], {}).get("response", {})
        # A worker rejection may replace the service's response; retain its explicit fallback.
        expected = response.get("candidate_ids") if event.get("fallback_reason") is None else request["candidate_ids"]
        audit.check(label + "/actual_rank_application/" + request["request_id"],
                    full_permutation(request, event["applied_order"]) and event["applied_order"] == expected)
    chains = []
    for outcome in outcomes:
        identity = outcome["request_id"]
        item = by_request.get(identity, {})
        response = item.get("response", {})
        metadata = response.get("metadata", {})
        decision = outcome.get("decision", {})
        controller = outcome.get("controller", {})
        failed = audit.read(sidecar(folder, outcome["failed_path"]))
        repaired_path = sidecar(folder, outcome["trajectory_file"])
        repaired = audit.read(repaired_path)
        constraints_for_failure = [r for r in frozen["records"]
                                   if r["evidence_event_id"] == outcome.get("failed_evidence_id")]
        raw_parent = results.get((result["run_id"], outcome["parent_call_id"]), {}).get("raw", {})
        extension = raw_parent.get("path_extension")
        path_pass = outcome.get("path_success") is True
        stability_pass = outcome.get("stability_success") is True
        selected = (decision.get("skill_id") == SKILL and metadata.get("path_skill") == decision
                    and response.get("fallback_reason") is None)
        continuous = (controller.get("switch_count") == 1 and controller.get("reset_count") == 1
                      and controller.get("intermediate_resets") == 0
                      and controller.get("dfs_state_mutations") == 0)
        axes = {"+X": [1, 0, 0], "-X": [-1, 0, 0], "+Y": [0, 1, 0],
                "-Y": [0, -1, 0], "+Z": [0, 0, 1], "-Z": [0, 0, -1]}
        actions = controller.get("actions", [])
        schedule_bound = (len(actions) == 2 and actions == outcome.get("scope", {}).get("action_schedule")
                          and actions[0].get("step") == 0
                          and actions[1] == {"step": decision.get("switch_step"),
                                            "action": axes.get(decision.get("second_axis"))})
        binding = extension == outcome if path_pass else (
            not isinstance(extension, dict) or extension.get("request_id") != identity)
        feasible_consistent = (outcome.get("feasible") == (path_pass and stability_pass)
                               if path_pass else outcome.get("feasible") in (None, False))
        audit.check(label + "/path_chain/" + identity, selected and bool(constraints_for_failure)
                    and failed.get("raw_success") is False and continuous
                    and schedule_bound and repaired.get("raw_success") == path_pass
                    and sha(repaired_path) == outcome.get("trajectory_sha256")
                    and repaired.get("controller") == controller
                    and repaired.get("geometry") == outcome.get("geometry") and binding
                    and feasible_consistent)
        if path_pass:
            audit.check(label + "/path_stability_binding/" + identity,
                        type(outcome.get("stability_success")) is bool
                        and raw_parent.get("feasible") == outcome.get("feasible")
                        and outcome.get("geometry", {}).get("noncollinear") is True)
        else:
            audit.check(label + "/failed_path_has_no_stability_verdict/" + identity,
                        outcome.get("stability_success") is None and outcome.get("feasible") in (None, False))
        details = next((d for d in decision_details if d["request_id"] == identity), {})
        chains.append({"request_id": identity, "failed_evidence_id": outcome.get("failed_evidence_id"),
            "constraint_record_ids": [r["record_id"] for r in constraints_for_failure],
            "registered_choice_verified": selected, "continuous_two_leg_controller": continuous,
            "actions": controller.get("actions"), "noncollinear": outcome.get("geometry", {}).get("noncollinear"),
            "angle_degrees": outcome.get("geometry", {}).get("angle_degrees"),
            "path_success": path_pass, "stability_success": outcome.get("stability_success"),
            "repair_success": path_pass and stability_pass, "parent_verifier_feasible": raw_parent.get("feasible"),
            "raw_parent_extension_equal": extension == outcome, "raw_parent_binding_valid": binding,
            "single_direction_failure_recovery": outcome.get("single_direction_failure_recovery"),
            "native_six_axis_failure_recovery": outcome.get("native_six_axis_failure_recovery"),
            "direct_previous_source_constraint_ids": details.get("direct_previous_source_constraint_ids", []),
            "cited_previous_source_skill_evidence_ids": details.get("cited_previous_source_skill_evidence_ids", []),
            "automatic_linked_previous_constraint_ids": details.get("automatic_linked_previous_constraint_ids", [])})
    replay = replays(audit, folder, label, outcomes, parsed)
    successful_repairs = sum(chain["repair_success"] for chain in chains)
    if successful_repairs:
        functional = "REPAIR_AND_REPLAY_PASS" if replay["latest_verified_pass"] else "REPAIR_PASS_REPLAY_NOT_VERIFIED"
    elif any(chain["path_success"] for chain in chains):
        functional = "REPAIR_PATH_PASS_STABILITY_FAILED"
    elif chains:
        functional = "REPAIR_FAILED_NATIVE_PLAN_FALLBACK" if whole_plan else "REPAIR_FAILED_PLAN_NOT_SUCCESSFUL"
    else:
        functional = "NO_REPAIR_EXECUTED"
    return {"audit_status": "PASS" if all(x["passed"] for x in audit.checks[mark:]) else "FAIL",
        "functional_status": functional, "reported_status": result.get("status"), "whole_plan_success": whole_plan,
        "sequence": result.get("sequence"), "actual_physics": actual_backend,
        "model_decision_fixture": result.get("model_decision_fixture"),
        "controlled_initial_probe": result.get("controlled_initial_probe"),
        "candidate_mode": arguments.get("candidate_mode", "controlled"),
        "candidate_ranking": result.get("candidate_ranking"),
        "verifier_calls": len(starts), "verifier_statuses": dict(statuses),
        "terminal_verifier_calls": result.get("terminal_verifier_calls"),
        "upstream_total_n_eval": result.get("upstream_total_n_eval"),
        "path_check_success_function_calls": result.get("path_check_success_calls"),
        "path_check_metric_is_invocations_not_success_count": True,
        "live_attempts": live_attempts, "live_valid_responses": valid_live,
        "live_candidate_calls": candidate_live, "live_path_calls": path_live,
        "rank_fallbacks": rank_fallbacks, "path_fallbacks": path_fallbacks,
        "heuristic_calls": result.get("heuristic_tripwire_calls"), "source_archive": source_archive,
        "constraint_count": len(frozen["records"]), "constraint_measurements": measurements,
        "decisions": decision_details, "repair_chains": chains, "replay": replay,
        "successful_repairs": successful_repairs, "performance_comparison": "NOT_RUN",
        "generalization": "UNVERIFIED", "causal_memory_benefit": "UNVERIFIED"}


def ledger(audit, path, prefix, expected_calls=None, expected_cap=None):
    path = safe(path)
    if not path.is_file():
        return {"status": "NOT_RUN"}
    sidecars = [safe(str(path) + suffix) for suffix in ("-wal", "-shm", "-journal")]
    # immutable prevents SQLite from creating SHM or recovering a journal.
    active = [p.name for p in sidecars if not p.name.endswith("-shm") and p.is_file() and p.stat().st_size]
    if active:
        return {"status": "UNVERIFIED_ACTIVE_SQLITE_SIDECAR", "active_sidecars": active}
    before = {str(p): sha(p) if p.is_file() else None for p in [path] + sidecars}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        rows = [dict(row) for row in db.execute("SELECT * FROM rank_requests ORDER BY created_utc,request_id")]
        policy_row = db.execute("SELECT * FROM batch_policy WHERE singleton=1").fetchone()
        policy = dict(policy_row) if policy_row else {}
    after = {str(p): sha(p) if p.is_file() else None for p in [path] + sidecars}
    audit.check(prefix + "/read_only_files_unchanged", before == after)
    audit.inputs[str(path)] = before[str(path)]
    calls, pending, reported_usage = 0, 0, 0
    reserved, actual, committed = Decimal(0), Decimal(0), Decimal(0)
    token_totals = Counter()
    details = []
    for row in rows:
        response = json.loads(row["response_json"]) if row["response_json"] else {}
        metadata = response.get("metadata", {})
        usage = json.loads(row["usage_json"]) if row["usage_json"] else None
        reserve, cost = Decimal(row["reserved_usd"] or "0"), Decimal(row["actual_usd"] or "0")
        if not reserve.is_finite() or not cost.is_finite() or min(reserve, cost) < 0:
            raise ValueError("invalid_ledger_cost")
        calls += int(row["reserved_call"])
        reserved += reserve
        actual += cost
        committed += max(reserve, cost)
        pending += int(row["status"] != "completed")
        audit.check(prefix + "/request_policy/" + row["request_id"],
                    row["config_fingerprint"] == policy.get("config_fingerprint")
                    and row["status"] in ("pending", "completed") and row["reserved_call"] in (0, 1))
        if usage is not None:
            keys = ("input_tokens", "output_tokens", "total_tokens")
            valid = all(type(usage.get(key)) is int and usage[key] >= 0 for key in keys)
            audit.check(prefix + "/token_sum/" + row["request_id"],
                        valid and usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"])
            if valid:
                token_totals.update({key: usage[key] for key in keys})
            reported_usage += 1
        if metadata.get("attempted") is True:
            audit.check(prefix + "/no_retries/" + row["request_id"], metadata.get("retries") == 0)
        details.append({"request_id": row["request_id"], "status": row["status"],
            "reserved_call": row["reserved_call"], "reserved_usd": str(reserve), "usage_cost_usd": str(cost),
            "model_requested": metadata.get("model_requested"), "model_returned": metadata.get("model_returned"),
            "ranking_source": metadata.get("ranking_source"), "usage": usage,
            "fallback_reason": response.get("fallback_reason"), "call_id": row["call_id"]})
    cap = Decimal(policy.get("max_usd") or "0")
    audit.check(prefix + "/caps", bool(policy) and calls <= policy.get("max_calls", -1) and committed <= cap)
    if expected_calls is not None:
        audit.check(prefix + "/approved_batch_caps", policy.get("max_calls") == expected_calls
                    and cap == Decimal(str(expected_cap)))
    return {"status": "COMPLETE" if not pending else "PENDING", "policy": policy,
        "request_count": len(rows), "reserved_calls": calls, "pending_requests": pending,
        "estimated_reservations_usd": str(reserved), "usage_cost_usd": str(actual),
        "budget_committed_usd": str(committed), "usage_reported_calls": reported_usage,
        "token_totals": dict(token_totals), "requests": details, "read_only_files_unchanged": before == after,
        "billing_cap_guaranteed": False, "costs_are_configured_rate_estimates": True}


def preservation(audit, batch):
    before = audit.read(batch / "preservation_before.json")
    if type(before) is not dict or not before:
        raise ValueError("missing_preservation_snapshot")
    changed, missing, groups = [], [], Counter()
    for name, expected in before.items():
        path = sidecar(ROOT, name)
        group = "/".join(name.replace("\\", "/").split("/")[:2])
        groups[group] += 1
        if not path.is_file():
            missing.append(name)
        elif sha(path) != expected:
            changed.append(name)
    audit.check("old_files_preserved", not missing and not changed,
                {"files": len(before), "missing": missing, "changed": changed})
    environment = audit.read(batch / "environment_preservation.json")
    if environment is not None:
        previous = audit.read(ROOT / "runs/prototype_completion_20261018/environment_final/stdout.log")
        current = audit.read(batch / "environment_final/stdout.log")
        unchanged = (type(previous) is dict and previous == current and bool(previous.get("files"))
                     and current.get("tracked_source_clean") is True)
        audit.check("WSL_environment_preserved", unchanged and environment.get("status") == "PASS"
                    and environment.get("checked_file_count") == len(current["files"])
                    and environment.get("differences") == [] and environment.get("extra_files") == [])
    return {"status": "PASS" if not missing and not changed else "FAIL",
        "checked_file_count": len(before), "missing": missing, "changed": changed,
        "groups": dict(sorted(groups.items())),
        "environment_preservation": environment if environment is not None else {"status": "NOT_RUN"},
        "path_policy": "project C-drive only; excluded drive rejected lexically before access"}


def public_api_review(audit, batch):
    rejection = audit.read(batch / "api_autoreview_rejection.json")
    byte_comparison = audit.read(batch / "api_payload_review_v2" / "review.json")
    git_proof = audit.read(batch / "api_payload_review_v2" / "review_git_objects.json")
    if git_proof:
        audit.check("public_git_blob_proof", git_proof.get("status") == "PUBLIC_COMMITTED_INPUT_VERIFIED"
                    and bool(git_proof.get("public_git_blob_matches"))
                    and all(git_proof["public_git_blob_matches"].values()))
    return {"initial_automatic_review_rejection": rejection,
        "initial_working_copy_byte_comparison": byte_comparison,
        "subsequent_public_git_blob_proof": git_proof,
        "final_tool_approval_receipt": "NOT_STORED_IN_RUN_FILES",
        "interpretation": "Initial rejection and checkout line-ending mismatch remain recorded. "
                          "Public committed-input proof and later actual API receipts are separate evidence."}


def run(batch):
    audit = Audit()
    batch = safe(batch)
    source = audit.read(batch / "live_source" / "constraint_memory.json", {})
    source_ids = {r["record_id"] for r in source.get("records", [])}
    cases = {name: audit.protect(name, lambda name=name: audit_case(audit, batch / name, name, source_ids))
             for name in LABELS}
    authorization = audit.read(batch / "authorization.json")
    current = audit.protect("live_ledger", lambda: ledger(audit, batch / "live_batch.sqlite",
                                                       "live_ledger", expected_calls=6, expected_cap="0.03"))
    prior = audit.protect("prior_ledger", lambda: ledger(audit,
        ROOT / "runs/prototype_completion_20261018/live_batch.sqlite", "prior_ledger"))
    if current.get("status") in ("COMPLETE", "PENDING") and prior.get("status") in ("COMPLETE", "PENDING"):
        total = Decimal(current["budget_committed_usd"]) + Decimal(prior["budget_committed_usd"])
        audit.check("prior_reservation_matches_authorization",
                    Decimal(prior["estimated_reservations_usd"]) == Decimal(str(authorization["prior_reserved_usd"])))
        audit.check("cumulative_user_approval_cap", total <= Decimal("0.10")
                    and authorization.get("new_max_calls") == 6
                    and Decimal(str(authorization["new_max_estimated_usd"])) == Decimal("0.03"))
        cumulative = {"prior_committed_usd": prior["budget_committed_usd"],
            "current_committed_usd": current["budget_committed_usd"], "total_committed_usd": str(total),
            "user_approved_usd": "0.10", "within_user_approval": total <= Decimal("0.10"),
            "prior_plus_new_reservation_ceiling_usd": str(Decimal(prior["budget_committed_usd"]) + Decimal("0.03"))}
        live_count = sum(case.get("live_attempts", 0) for case in cases.values())
        audit.check("case_receipts_match_ledger_attempts",
                    live_count == current["reserved_calls"],
                    {"case_attempts": live_count, "ledger_reservations": current["reserved_calls"]})
    else:
        cumulative = {"status": "UNVERIFIED_LEDGER_NOT_QUIESCENT"}
    preserved = audit.protect("preservation", lambda: preservation(audit, batch))
    reviewed = audit.protect("api_review", lambda: public_api_review(audit, batch))
    input_changes = [path for path, expected in audit.inputs.items() if not safe(path).is_file() or sha(path) != expected]
    audit.check("audit_read_artifacts_unchanged", not input_changes, input_changes)
    failed = [check for check in audit.checks if not check["passed"]]
    incomplete = [name for name, value in cases.items() if value.get("audit_status") == "NOT_RUN"]
    if preserved.get("environment_preservation", {}).get("status") == "NOT_RUN":
        incomplete.append("environment_preservation")
    status = "FAIL" if failed else ("PARTIAL" if incomplete or current.get("status") != "COMPLETE" else "PASS")
    return {"schema_version": 1, "kind": "constraint_path_saved_evidence_audit", "audit_status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(), "batch_dir": str(batch),
        "meaning_of_PASS": "Saved evidence is internally consistent; individual failed experiments remain failures.",
        "audit_api_calls": 0, "audit_physics_calls": 0, "sqlite_writes": 0, "pickle_loading": False,
        "audited_labels": list(LABELS), "missing_cases": incomplete, "cases": cases,
        "authorization": authorization, "live_ledger": current, "prior_ledger": prior,
        "cumulative_budget": cumulative, "preservation": preserved, "api_review_history": reviewed,
        "checks": audit.checks, "failed_checks": failed, "read_artifact_sha256": audit.inputs,
        "limitations": [
            "Controlled single-direction failure probe is not proof that native six-axis search failed.",
            "Constraint REJECTED applies only to measured endpoint separation, never every path of a part.",
            "Whole-plan PASS after native fallback does not mean the requested repair succeeded.",
            "Direct old-constraint citation, validated-skill citation and automatic links are distinct.",
            "Skill-card introduction after a failed development run is not a controlled memory-effect comparison.",
            "Source and reuse currently use the same official Beam; generalization and performance are unverified.",
            "Replay evidence is hash-bound to local artifacts; this audit itself does not execute physics."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = safe(args.output)
    if output.exists():
        raise FileExistsError("audit_output_must_be_new")
    if not output.parent.is_dir():
        raise ValueError("audit_output_parent_must_exist")
    result = run(safe(args.batch_dir))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"audit_status": result["audit_status"], "output": str(output),
                      "checks": len(result["checks"]), "failed_checks": len(result["failed_checks"]),
                      "api_calls": 0, "physics_calls": 0}, ensure_ascii=False))
    return 0 if result["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
