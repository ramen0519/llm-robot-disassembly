"""Read-only audit of independent ranking, feedback and cross-assembly repair.

Individual experiment failures are retained. Audit PASS means the recorded
facts agree, not that repair, the full plan or generalization succeeded.
No API, physics, pickle loading or database writes; one exclusive JSON output.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_constraint_path import (
    Audit, safe, sha, digest, sidecar, archives, constraints, replays, ledger,
    snapshot, full_permutation, evidence_pairs)

LABELS = ("transfer_fixture", "natural_fixture", "transfer_live", "natural_live")
PRIOR_BATCHES = ("prototype_completion_20261018", "constraint_path_repair_20261018")
AXES = {"+X": [1, 0, 0], "-X": [-1, 0, 0], "+Y": [0, 1, 0],
        "-Y": [0, -1, 0], "+Z": [0, 0, 1], "-Z": [0, 0, -1]}
NATIVE_AXES = [AXES[k] for k in ("+Z", "-Z", "+X", "-X", "+Y", "-Y")]


def text_file(audit, path):
    path = safe(path)
    if not path.is_file():
        return None
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("audit_text_size_limit")
    value = path.read_bytes()
    import hashlib
    audit.inputs[str(path)] = hashlib.sha256(value).hexdigest()
    return value.decode("utf-8")


def native_groups(audit, folder):
    text = text_file(audit, folder / "execute/stdout.log") or ""
    pattern = re.compile(r"\[check_assemblable\] success: (True|False), parts_fix: (\[[^\]]*\]), "
                         r"part_move: ([^,\s]+), action: (\[[^\]]*\]), path_len: (\d+)")
    rows = []
    for match in pattern.finditer(text):
        success, rest, move, vector, length = match.groups()
        rows.append({"success": success == "True", "parts_rest": ast.literal_eval(rest),
                     "part_move": move, "action": [float(x) for x in vector.strip("[]").split()],
                     "path_states": int(length)})
    return [rows[n:n + 6] for n in range(0, len(rows), 6)]


def no_priors(value):
    forbidden = {"native_order", "heuristic_score", "heuristic_rank", "recommended_order", "fallback_order"}
    if isinstance(value, dict):
        return not (set(value) & forbidden) and all(no_priors(v) for v in value.values())
    if isinstance(value, list):
        return all(no_priors(v) for v in value)
    return True


def expected_projection(reference, origin):
    """Independent reproduction of the recorded v1 projection, not inference."""
    projected = deepcopy(reference)
    scope = projected.get("source_scope", {})
    context = scope.get("task_context", {})
    if context:
        scope["task_context"] = {key: context[key]
                                 for key in ("schema_version", "task", "initial_parts")}
    projected["origin"] = origin
    projected["projection"] = "geometry_arrays_omitted_full_scope_preserved_in_frozen_file"
    return projected


def card_check(audit, card, source, context, name):
    identity = card.get("skill_evidence_id")
    record = next((r for r in (source or {}).get("records", [])
                   if r["skill_evidence_id"] == identity), None)
    if record is None:
        audit.check(name + "/source_card_registered", False)
        return False
    preserved = all(card.get(key) == record[key] for key in (
        "skill_id", "controller_sha256", "source_run_id", "source_request_id",
        "failed_constraint_ids", "source_scope", "source_scope_sha256", "validated_schedule",
        "source_trajectory_sha256", "validation", "lifecycle", "generalized",
        "force_new_physics", "hard_pruning", "reuse_verdict"))
    cross = card.get("applicability") == "cross_assembly_procedure_reference"
    if cross:
        transfer, target = card.get("transfer_semantics", {}), card.get("target_context", {})
        mapping_keys = ("current_part_id_mapping", "coordinate_transform", "axis_mapping",
                        "unit_scale_mapping", "target_selected_schedule", "target_axis_recommendation")
        valid = (preserved and card.get("current_part_id_mapping") is None
                 and all(transfer.get(key) is None for key in mapping_keys)
                 and card.get("source_validation_applies_to_target") is False
                 and card["source_scope"]["task_context"]["task"]["assembly_id"] != context["task"]["assembly_id"]
                 and target.get("task") == context["task"]
                 and target.get("task_context_sha256") == digest(context)
                 and transfer.get("past_pass_reused") is False
                 and transfer.get("source_failed_constraint_ids_are_target_constraints") is False
                 and transfer.get("trajectory_copy_allowed") is False
                 and transfer.get("target_path_verdict") == "UNKNOWN"
                 and transfer.get("target_stability_verdict") == "UNKNOWN")
    else:
        valid = preserved
    audit.check(name + "/card_provenance_and_no_mapping", valid
                and card.get("reference_only") is True and card.get("current_attempt_verified") is False)
    return cross


def preserved_files(audit, batch):
    before = audit.read(batch / "preservation_before.json")
    if type(before) is not dict or not before:
        raise ValueError("missing_preservation_snapshot")
    missing, changed = [], []
    for name, expected in before.items():
        path = sidecar(ROOT, name)
        if not path.is_file():
            missing.append(name)
        elif sha(path) != expected:
            changed.append(name)
    audit.check("old_project_files_preserved", not missing and not changed,
                {"checked_file_count": len(before), "missing": missing, "changed": changed})
    reference = audit.read(batch / "environment_reference.json")
    final = audit.read(batch / "environment_final.json")
    post_component = audit.read(batch / "environment_final_after_saved_choice.json")
    old_beam = audit.read(ROOT / "runs/constraint_path_repair_20261018/environment_final/stdout.log")
    environment = {"status": "NOT_RUN"}
    if reference is not None:
        audit.check("beam_reference_matches_previous_batch", reference.get("files") == old_beam.get("files")
                    and reference.get("commit") == old_beam.get("commit")
                    and reference.get("tracked_source_clean") is True)
    if reference is not None and final is not None:
        beam_ok = (reference.get("files") == final.get("files")
                   and reference.get("commit") == final.get("commit")
                   and final.get("tracked_source_clean") is True)
        task_ok = reference.get("existing_07009_files") == final.get("existing_07009_files")
        audit.check("final_beam_environment_preserved", beam_ok)
        audit.check("existing_07009_preserved_since_reference", task_ok)
        environment = {"status": "PASS" if beam_ok and task_ok else "FAIL",
            "beam_file_count": len(reference.get("files", {})),
            "existing_07009_file_count": len(reference.get("existing_07009_files", {})),
            "beam_equal_previous_batch": reference.get("files") == old_beam.get("files"),
            "existing_07009_reference_timing": "reference captured after F1 started; preservation claimed only since that capture",
            "reference_file": str(batch / "environment_reference.json"),
            "final_file": str(batch / "environment_final.json")}
        if post_component is not None:
            post_ok = (post_component.get("files") == final.get("files")
                       and post_component.get("existing_07009_files") == final.get("existing_07009_files")
                       and post_component.get("commit") == final.get("commit")
                       and post_component.get("tracked_source_clean") is True)
            audit.check("environment_preserved_after_saved_choice", post_ok)
            environment["after_saved_choice"] = "PASS" if post_ok else "FAIL"
            environment["after_saved_choice_file"] = str(batch / "environment_final_after_saved_choice.json")
    return {"status": "PASS" if not missing and not changed else "FAIL",
            "checked_file_count": len(before), "missing": missing, "changed": changed,
            "environment": environment, "excluded_drive_access": False}


def audit_case(audit, folder, label):
    result = audit.read(folder / "result.json")
    if result is None:
        return {"audit_status": "NOT_RUN", "functional_status": "NOT_RUN"}
    mark = len(audit.checks)
    config = audit.read(folder / "config.json", {})
    arguments = config.get("arguments", {})
    parsed = audit.read(folder / "parsed_result.json", {})
    context = audit.read(folder / "context.json")
    source_archive = archives(audit, folder, label)
    events = audit.read(folder / "events.jsonl", [], lines=True)
    path_events = audit.read(folder / "path_events.jsonl", [], lines=True)
    decisions = audit.read(folder / "prototype_decisions.jsonl", [], lines=True)
    previews = audit.read(folder / "request_body_previews.jsonl", [], lines=True)
    raw_outcomes = audit.read(folder / "path_outcomes.json")
    completed = audit.read(folder / "integrated_feedback.json")
    starts, results, event_checks = evidence_pairs(events)
    interrupted = result.get("process_exit") not in (0, None)
    for key, ok in event_checks.items():
        if key == "all_verifier_calls_completed" and interrupted:
            continue
        audit.check(label + "/events/" + key, ok)
    audit.check(label + "/counts", result.get("verifier_calls") == len(starts)
                and result.get("verifier_results") == len(results))
    audit.check(label + "/terminal_and_status_counts",
                result.get("terminal_verifier_calls") == sum(e.get("generation_id") is None for e in starts.values())
                and all(result.get("verifier_statuses", {}).get(status, 0) == count for status, count
                        in Counter(e["status"] for e in results.values()).items()))
    audit.check(label + "/no_controlled_probe", config.get("controlled_initial_probe") is False
                and result.get("controlled_initial_probe") is False
                and result.get("controlled_probe_responses") == 0
                and arguments.get("controlled_repair", False) is False)
    if result.get("heuristic_tripwire_calls") is not None:
        audit.check(label + "/no_heuristic_calls", result["heuristic_tripwire_calls"] == 0)
    actual = result.get("verifier_backend") == config.get("physics_backend") == "actual_ASAP_RedMax"
    audit.check(label + "/actual_backend", actual)
    plan = (parsed.get("status") == "PASS" and parsed.get("validation_ok") is True
            and parsed.get("plan_success") is True and parsed.get("tree_checked") is True
            and parsed.get("raw_stats", {}).get("success") is True
            and parsed.get("raw_stats", {}).get("sequence") == result.get("sequence"))
    if result.get("plan_success") is True:
        audit.check(label + "/reported_plan_success", plan)
    missing = [name for name, value in (("context.json", context), ("path_outcomes.json", raw_outcomes),
                                       ("integrated_feedback.json", completed)) if value is None]
    if missing or not safe(folder / "constraint_memory.json").is_file():
        return {"audit_status": "INCOMPLETE", "functional_status": "RUN_INCOMPLETE",
                "reported_status": result.get("status"), "process_exit": result.get("process_exit"),
                "missing_artifacts": missing, "whole_plan_success": plan,
                "verifier_starts": len(starts), "verifier_results": len(results),
                "source_archive": source_archive}
    frozen, measurements = constraints(audit, folder, label, context)
    records = {r["record_id"]: r for r in frozen["records"]}
    old_records, skill_source = {}, None
    if arguments.get("memory"):
        memory_path = safe(arguments["memory"])
        old_records = {r["record_id"]: r for r in snapshot(audit, memory_path)["records"]}
        audit.check(label + "/frozen_memory", sha(memory_path) == config.get("frozen_memory_sha256")
                    and result.get("memory_frozen") is True)
    if arguments.get("skill_memory"):
        from orchestrator.path_skill_memory import _validate_snapshot
        skill_path = safe(arguments["skill_memory"])
        skill_source = audit.read(skill_path)
        _validate_snapshot(skill_source, check_sources=True)
        audit.check(label + "/frozen_skill_source", sha(skill_path) == config.get("frozen_skill_memory_sha256")
                    and result.get("skill_memory_frozen") is True)
    outcomes = raw_outcomes.get("outcomes", [])
    by_outcome = {o["request_id"]: o for o in outcomes}
    audit.check(label + "/published_complete_feedback", completed.get("controlled_choices") is False
                and completed.get("physics_result_reuse") is False
                and {item["outcome"]["request_id"]: item["outcome"]
                     for item in completed.get("completed_path_feedback", [])} == by_outcome)
    groups = native_groups(audit, folder)
    native_by_call = {}
    for index, (key, start) in enumerate(starts.items()):
        if index >= len(groups):
            break
        group, request = groups[index], start["request"]
        valid = (len(group) == 6 and [v["action"] for v in group] == NATIVE_AXES
                 and all(v["part_move"] == request["part_move"]
                         and v["parts_rest"] == request["parts_rest"] for v in group))
        audit.check(label + "/native_six_axis_binding/" + key[1], valid)
        native_by_call[key[1]] = group
    triggers = [e for e in path_events if e.get("kind") == "repair_trigger"]
    for event in triggers:
        group = native_by_call.get(event["call_id"], [])
        audit.check(label + "/natural_trigger/" + event["call_id"],
                    event.get("trigger") == "native_six_axis_search_returned_none"
                    and len(group) == 6 and all(v["success"] is False for v in group))
    audit.check(label + "/one_natural_trigger_cap", len(triggers) <= 1)
    measurement_count = sum(e.get("kind") == "measurement_saved" for e in path_events)
    if not interrupted:
        audit.check(label + "/path_function_call_accounting",
                    result.get("path_check_success_calls") == raw_outcomes.get("path_check_success_calls")
                    == sum(len(g) for g in groups) + measurement_count
                    and len(groups) == len(starts))
    published_ids, measured_ids, feedback_offers = set(), set(), {}
    for event in path_events:
        if event.get("kind") == "measurement_saved":
            measured_ids.update(event.get("record_ids", []))
        elif event.get("kind") == "integrated_feedback_published":
            published_ids.add(event["request_id"])
        elif event.get("kind") == "integrated_feedback_offered":
            identity = event["request_id"]
            feedback_offers[identity] = event
            audit.check(label + "/feedback_available_before_request/" + identity,
                        set(event.get("path_request_ids", [])) == published_ids
                        and set(event.get("constraint_ids", [])).issubset(measured_ids))
    decisions_by_id, details = {}, []
    live_attempts = valid_live = 0
    for item in decisions:
        worker = item["request"]
        enriched = item.get("enriched_request", worker)
        request, features = enriched["request"], enriched["features"]
        identity = request["request_id"]
        response, info = item["response"], item["response"].get("metadata", {})
        ic = features.get("integrated_context", {})
        route = ic.get("route", "path" if worker["features"].get("protocol") == "constraint_path_repair_v1" else "candidate")
        offered_log = feedback_offers.get(identity)
        if offered_log is not None:
            audit.check(label + "/worker_feedback_receipt/" + identity,
                        sorted(item["outcome"]["request_id"] for item in worker["features"].get("path_feedback", []))
                            == sorted(offered_log.get("path_request_ids", []))
                        and sorted(ref["record_id"] for ref in worker["features"].get("constraint_history", []))
                            == sorted(offered_log.get("constraint_ids", [])))
        audit.check(label + "/unique_decision/" + identity, identity not in decisions_by_id)
        decisions_by_id[identity] = item
        audit.check(label + "/permutation/" + identity, full_permutation(request, response.get("candidate_ids"))
                    and response.get("request_id") == identity and response.get("state_id") == request["state_id"])
        audit.check(label + "/no_priors/" + identity, no_priors(features))
        audit.check(label + "/no_controlled_decision/" + identity,
                    info.get("controlled_candidate_selection") is False
                    and info.get("controlled_path_selection") is False)
        live = info.get("llm_live") is True and info.get("attempted") is True
        valid = live and response.get("fallback_reason") is None
        live_attempts += int(live)
        valid_live += int(valid)
        if live:
            returned_ok = (info.get("model_returned") == "gpt-6-luna" if valid
                           else info.get("model_returned") in (None, "gpt-6-luna"))
            audit.check(label + "/live_identity/" + identity, info.get("model_requested") == "gpt-6-luna"
                        and returned_ok and info.get("retries") == 0)
        cards = {c["skill_evidence_id"]: c for c in ic.get("frozen_skill_evidence", [])}
        cross_ids = {cid for cid, card in cards.items()
                     if card_check(audit, card, skill_source, context, label + "/" + identity + "/" + cid)}
        refs = {r["record_id"]: r for r in ic.get("constraint_references", [])}
        for rid, reference in refs.items():
            record = records.get(rid, old_records.get(rid))
            audit.check(label + "/reference/" + identity + "/" + rid, record is not None
                        and reference.get("source_scope") == record["scope"]
                        and reference.get("evidence_origin", {}).get("source") == record["source"]
                        and reference.get("current_part_id_mapping") is None
                        and reference.get("hard_pruning") is False and reference.get("reuse_verdict") is False)
        feedback = {f["feedback_id"]: f for f in ic.get("path_feedback", [])}
        if response.get("fallback_reason") is None:
            audit.check(label + "/no_path_feedback_omission/" + identity,
                        info.get("omitted_path_feedback_ids") == []
                        and {item["outcome"]["request_id"] for item in worker["features"].get("path_feedback", [])}
                            .issubset(feedback)
                        and {ref["record_id"] for ref in worker["features"].get("constraint_history", [])}
                            .issubset(refs))
        for fid, projected in feedback.items():
            original = by_outcome.get(fid)
            audit.check(label + "/projected_path_feedback/" + identity + "/" + fid, original is not None
                        and projected.get("action_schedule") == original["scope"]["action_schedule"]
                        and projected.get("controller") == original["controller"]
                        and projected.get("geometry") == original["geometry"]
                        and projected.get("result") == {k: original[k] for k in
                            ("path_success", "stability_success", "parts_fix", "feasible")}
                        and projected.get("failure_cause") is None
                        and projected.get("hard_pruning") is False and projected.get("reuse_verdict") is False)
        used_cards = set(info.get("used_skill_evidence_ids", []))
        used_constraints = set(info.get("used_constraint_ids", []))
        used_feedback = set(info.get("used_path_feedback_ids", []))
        audit.check(label + "/citation_subset/" + identity, used_cards.issubset(cards)
                    and used_constraints.issubset(refs) and used_feedback.issubset(feedback))
        details.append({"request_id": identity, "route": route, "live_attempt": live, "valid_live": valid,
            "fallback_reason": response.get("fallback_reason"), "ranking_source": info.get("ranking_source"),
            "offered_cross_assembly_skill_ids": sorted(cross_ids),
            "cited_cross_assembly_skill_ids": sorted(used_cards & cross_ids),
            "offered_constraint_ids": sorted(refs), "used_constraint_ids": sorted(used_constraints),
            "direct_previous_constraint_ids": sorted(used_constraints & set(old_records)),
            "offered_path_feedback_ids": sorted(feedback), "used_path_feedback_ids": sorted(used_feedback),
            "path_skill": info.get("path_skill"), "candidate_order": response.get("candidate_ids") if route == "candidate" else None,
            "selected_skill_id": info.get("selected_skill_id"), "repair_invocation": info.get("repair_invocation")})
    for preview in previews:
        identity, body = preview["request_id"], preview.get("body", {})
        original = decisions_by_id.get(identity, {}).get("enriched_request")
        if original is None:
            audit.check(label + "/preview_has_decision/" + identity, False)
            continue
        model = json.loads(body["input"][0]["content"])
        ic, features = original["features"]["integrated_context"], model["features"]
        audit.check(label + "/actual_model_projection/" + identity, model.get("request") == original["request"]
                    and features.get("decision_route") == ic["route"]
                    and features.get("path_feedback") == ic["path_feedback"]
                    and features.get("episode_feedback") == ic["episode_feedback"]
                    and features.get("constraint_references", []) == [
                        expected_projection(x, x["origin"]) for x in ic["constraint_references"]]
                    and features.get("frozen_skill_evidence", []) == [
                        expected_projection(x, "frozen_skill_validation") for x in ic["frozen_skill_evidence"]]
                    and no_priors(model) and features.get("hard_pruning") is False
                    and features.get("physics_reuse_allowed") is False)
    rank_events = [e for e in events if e.get("kind") == "rank_decision"]
    rank_fallbacks = sum(e.get("fallback_reason") is not None for e in rank_events)
    path_fallbacks = sum(d["fallback_reason"] is not None and d["route"] == "path" for d in details)
    audit.check(label + "/API_response_counts", result.get("live_api_attempts") == live_attempts
                and result.get("live_valid_responses") == valid_live)
    audit.check(label + "/fallback_counts", result.get("rank_fallbacks") == rank_fallbacks
                and result.get("path_decision_fallbacks") == path_fallbacks)
    route_counts = {route: {
        "responses": sum(d["route"] == route for d in details),
        "live_attempts": sum(d["route"] == route and d["live_attempt"] for d in details),
        "valid_live_responses": sum(d["route"] == route and d["valid_live"] for d in details),
        "fallbacks": sum(d["route"] == route and d["fallback_reason"] is not None for d in details),
    } for route in ("candidate", "path")}
    audit.check(label + "/decision_route_counts", result.get("decision_routes") == route_counts)
    budget_before = audit.read(folder / "budget_before.json", {})
    budget_after = audit.read(folder / "budget_after.json", {})
    delta = budget_after.get("reserved_calls", 0) - budget_before.get("reserved_calls", 0)
    audit.check(label + "/run_budget_receipts", delta == live_attempts == result.get("run_reserved_calls_delta")
                and result.get("batch_reserved_calls_after") == budget_after.get("reserved_calls")
                and result.get("batch_usage_cost_usd_after") == budget_after.get("usage_cost_usd")
                and result.get("batch_budget_committed_usd_after") == budget_after.get("budget_committed_usd")
                and result.get("batch_unknown_outcomes_after") == budget_after.get("unknown_outcomes"))
    for event in rank_events:
        request = event["request"]
        response = decisions_by_id.get(request["request_id"], {}).get("response", {})
        expected = response.get("candidate_ids") if event.get("fallback_reason") is None else request["candidate_ids"]
        audit.check(label + "/actual_rank_application/" + request["request_id"],
                    full_permutation(request, event.get("applied_order")) and event["applied_order"] == expected)
    chains = []
    for outcome in outcomes:
        identity = outcome["request_id"]
        detail = next((d for d in details if d["request_id"] == identity), {})
        controller, scope = outcome["controller"], outcome["scope"]
        selected = detail.get("path_skill") == outcome["decision"] and detail.get("fallback_reason") is None
        failed_path = audit.read(sidecar(folder, outcome["failed_path"]))
        repaired_file = sidecar(folder, outcome["trajectory_file"])
        repaired = audit.read(repaired_file)
        failure_records = [r for r in frozen["records"]
                           if r["evidence_event_id"] == outcome["failed_evidence_id"]]
        parent = results.get((result["run_id"], outcome["parent_call_id"]), {}).get("raw", {})
        native = native_by_call.get(outcome["parent_call_id"], [])
        switch = controller.get("switch_count")
        shape = (controller.get("reset_count") == 1 and controller.get("intermediate_resets") == 0
                 and controller.get("dfs_state_mutations") == 0 and switch in (0, 1)
                 and controller.get("actions") == scope["action_schedule"])
        path_pass, stable = outcome.get("path_success"), outcome.get("stability_success")
        expected_feasible = path_pass is True and stable is True
        parent_binding = parent.get("path_extension") == outcome if path_pass else (
            parent.get("action") is None and parent.get("feasible") is False)
        audit.check(label + "/actual_natural_repair_chain/" + identity,
                    selected and shape and bool(failure_records)
                    and len(native) == 6 and all(x["success"] is False for x in native)
                    and outcome.get("native_six_axis_failure_recovery") is True
                    and outcome.get("single_direction_failure_recovery") is False
                    and failed_path.get("raw_success") is False
                    and sha(repaired_file) == outcome["trajectory_sha256"]
                    and repaired.get("controller") == controller and repaired.get("geometry") == outcome["geometry"]
                    and repaired.get("raw_success") == path_pass and parent_binding
                    and outcome.get("feasible") == expected_feasible
                    and (type(stable) is bool if path_pass else stable is None))
        executed_turn = switch == 1 and outcome["geometry"].get("noncollinear") is True
        linked = outcome["decision"].get("linked_source_constraint_ids", [])
        cards_by_id = {r["skill_evidence_id"]: r for r in (skill_source or {}).get("records", [])}
        cited = outcome["decision"].get("used_skill_evidence_ids", [])
        expected_linked = sorted({rid for cid in cited for rid in cards_by_id.get(cid, {}).get("failed_constraint_ids", [])})
        audit.check(label + "/automatic_links_not_direct_citation/" + identity, sorted(linked) == expected_linked)
        chains.append({"request_id": identity, "native_six_axis_failure": True,
            "constraint_record_ids": [r["record_id"] for r in failure_records],
            "selected_registered_skill": selected, "switch_count": switch, "actual_noncollinear_turn": executed_turn,
            "path_success": path_pass, "stability_success": stable, "feasible": outcome.get("feasible"),
            "repair_success": expected_feasible and executed_turn, "raw_parent_binding": parent_binding,
            "cited_cross_assembly_skill_ids": detail.get("cited_cross_assembly_skill_ids", []),
            "automatic_linked_source_constraint_ids": linked, "automatic_link_is_not_direct_LLM_citation": True,
            "current_part_mapping": None})
    replay = replays(audit, folder, label, outcomes, parsed)
    candidate_feedback = [d for d in details if d["route"] == "candidate" and d["offered_path_feedback_ids"]]
    path_feedback = [d for d in details if d["route"] == "path" and d["offered_path_feedback_ids"]]
    candidate_constraints = [d for d in details if d["route"] == "candidate"
                             and set(d["offered_constraint_ids"]) & set(records)]
    success_count = sum(c["repair_success"] for c in chains)
    cross_cited = [d for d in details if d["cited_cross_assembly_skill_ids"]]
    cross_executed = [c for c in chains if c["cited_cross_assembly_skill_ids"]]
    previous_citations = [d for d in details if d["direct_previous_constraint_ids"]]
    previous_live_citations = [d for d in previous_citations if d["valid_live"]]
    if success_count:
        functional = "NATURAL_REPAIR_AND_REPLAY_PASS" if replay["latest_verified_pass"] else "NATURAL_REPAIR_PASS_REPLAY_NOT_VERIFIED"
    elif chains:
        functional = "REPAIR_ATTEMPT_FAILED_OR_NO_ACTUAL_TURN"
    elif triggers:
        functional = "NATURAL_TRIGGER_NO_REPAIR_EXECUTION"
    else:
        functional = "NATURAL_REPAIR_NOT_TRIGGERED"
    cross_status = ("CITED_EXECUTED_AND_PHYSICS_PASS" if any(c["repair_success"] for c in cross_executed)
                    else "CITED_EXECUTED_PHYSICS_NOT_SUCCESSFUL" if cross_executed
                    else "CITED_NOT_EXECUTED" if cross_cited
                    else "EXPOSED_NOT_CITED" if any(d["offered_cross_assembly_skill_ids"] for d in details)
                    else "NOT_EXPOSED")
    return {"audit_status": "PASS" if all(c["passed"] for c in audit.checks[mark:]) else "FAIL",
        "functional_status": functional, "reported_status": result.get("status"), "whole_plan_success": plan,
        "process_exit": result.get("process_exit"), "sequence": result.get("sequence"),
        "actual_physics": actual, "model_decision_fixture": result.get("model_decision_fixture"),
        "controlled_initial_probe": False, "candidate_ranking": result.get("candidate_ranking"),
        "verifier_calls": len(starts), "verifier_results": len(results),
        "verifier_statuses": dict(Counter(e["status"] for e in results.values())),
        "terminal_verifier_calls": result.get("terminal_verifier_calls"),
        "upstream_total_n_eval": result.get("upstream_total_n_eval"),
        "path_check_function_calls": result.get("path_check_success_calls"),
        "path_check_metric_is_invocations_not_success_count": True,
        "natural_trigger_count": len(triggers), "constraint_count": len(records),
        "constraint_measurements": measurements, "source_archive": source_archive,
        "live_attempts": live_attempts, "live_valid_responses": valid_live,
        "live_candidate_calls": sum(d["live_attempt"] and d["route"] == "candidate" for d in details),
        "live_path_calls": sum(d["live_attempt"] and d["route"] == "path" for d in details),
        "rank_fallbacks": rank_fallbacks, "path_fallbacks": path_fallbacks,
        "decisions": details, "repair_chains": chains, "replay": replay,
        "decision_routes": route_counts, "budget_before": budget_before, "budget_after": budget_after,
        "successful_actual_turn_repairs": success_count, "cross_assembly_reuse_status": cross_status,
        "frozen_previous_episode_direct_citation_requests": len(previous_citations),
        "frozen_previous_episode_live_direct_citation_requests": len(previous_live_citations),
        "frozen_previous_episode_live_direct_citation_ids": sorted({
            rid for d in previous_live_citations for rid in d["direct_previous_constraint_ids"]}),
        "previous_citation_is_soft_reference_not_target_mapping_or_causal_effect": True,
        "completed_feedback_count": len(completed.get("completed_path_feedback", [])),
        "completed_feedback_delivery_status": (
            "SUPPLIED" if candidate_feedback or path_feedback else
            "STORED_NOT_SUPPLIED" if completed.get("completed_path_feedback") else "NONE_CREATED"),
        "candidate_requests_with_completed_path_feedback": len(candidate_feedback),
        "path_requests_with_completed_path_feedback": len(path_feedback),
        "candidate_requests_with_current_constraints": len(candidate_constraints),
        "completed_feedback_on_both_routes_live_observed": bool(
            any(d["live_attempt"] for d in candidate_feedback) and any(d["live_attempt"] for d in path_feedback)),
        "performance_comparison": "NOT_RUN", "generalization_effect": "UNVERIFIED"}


def audit_saved_diagnostic(audit, folder):
    """Audit separate, unexecuted decisions; never add these to DFS success."""
    result = audit.read(folder / "result.json")
    if result is None:
        return {"audit_status": "NOT_RUN", "live_attempts": 0,
                "full_DFS": False, "new_physics_calls": 0}
    label, mark = "saved_feedback_live", len(audit.checks)
    config = audit.read(folder / "config.json", {})
    lineage = audit.read(folder / "lineage.json", {})
    before = audit.read(folder / "source_before.json", {})
    after = audit.read(folder / "source_after.json", {})
    changed = [name for name, expected in before.items()
               if not safe(name).is_file() or sha(safe(name)) != expected]
    audit.check(label + "/source_preserved", bool(before) and before == after
                and not changed and result.get("source_preserved") is True, changed)
    audit.check(label + "/diagnostic_only", all(value.get("diagnostic_only") is True
                and value.get("full_DFS") is False and value.get("solver_not_resumed") is True
                for value in (config, result, lineage))
                and all(result.get(key) == 0 for key in (
                    "selected_path_executions", "candidate_orders_consumed_by_DFS", "new_physics_calls"))
                and result.get("requests_independent") is True
                and result.get("path_response_caused_stored_outcome") is False
                and result.get("source_model_decision_fixture") is True
                and result.get("source_physics_backend") == "actual_ASAP_RedMax")
    policy = config.get("api_config", {})
    audit.check(label + "/shared_policy", config.get("maximum_new_requests") == 2
                and config.get("automatic_retries") == 0 and policy.get("max_calls") == 12
                and Decimal(str(policy.get("max_usd", -1))) == Decimal("0.05")
                and policy.get("model") == "gpt-6-luna"
                and policy.get("base_url") == "https://api.openai.com/v1")
    requests = audit.read(folder / "requests.jsonl", [], lines=True)
    prepared = audit.read(folder / "prepared_requests.jsonl", [], lines=True)
    decisions = audit.read(folder / "decisions.jsonl", [], lines=True)
    responses = audit.read(folder / "responses.jsonl", [], lines=True)
    previews = audit.read(folder / "request_body_previews.jsonl", [], lines=True)
    request_map = {item["request"]["request_id"]: item for item in requests}
    prepared_map = {item["request"]["request_id"]: item for item in prepared}
    decision_map = {item["request"]["request"]["request_id"]: item for item in decisions}
    response_map = {item["request_id"]: item for item in responses}
    audit.check(label + "/independent_two_requests", len(requests) == len(request_map) == 2
                and len(prepared_map) == 2 and set(request_map) == set(prepared_map)
                and len(lineage.get("requests", [])) == 2)
    source = safe(config["arguments"]["source_run"])
    source_result = audit.read(source / "result.json", {})
    source_completed = audit.read(source / "integrated_feedback.json", {}).get("completed_path_feedback", [])
    source_events = audit.read(source / "events.jsonl", [], lines=True)
    starts, source_results, _ = evidence_pairs(source_events)
    episode = [{"event_id": event["event_id"], "call_id": event["call_id"],
                "status": event["status"], "request": starts[key]["request"],
                "raw": event.get("raw"), "observed_stage": event.get("observed_stage"),
                "blocker": None, "path": None} for key, event in source_results.items()]
    source_memory = snapshot(audit, source / "constraint_memory.json")
    source_records = {r["record_id"]: r for r in source_memory["records"]}
    audit.check(label + "/historical_source_physics", source_result.get("verifier_backend") == "actual_ASAP_RedMax"
                and source_result.get("model_decision_fixture") is True
                and source_result.get("controlled_initial_probe") is False
                and bool(episode) and all(e["status"] != "PASS" for e in episode)
                and len(source_completed) == 1 and source_completed[0]["outcome"]["path_success"] is False)
    for origin in lineage.get("requests", []):
        identity, route = origin["diagnostic_request_id"], origin["route"]
        original = audit.read(safe(origin["source_request_file"]))
        current = request_map[identity]
        expected_request = dict(original["request"], run_id=config["run_id"], request_id=identity)
        expected_features = deepcopy(original["features"])
        if route == "candidate":
            expected_features["episode_feedback"] = episode
            expected_features["path_feedback"] = source_completed
            # Retrieval selection is bounded; provenance of each selected
            # reference is checked independently against the frozen records.
            expected_features["constraint_history"] = current["features"].get("constraint_history", [])
            history = current["features"].get("constraint_history", [])
            audit.check(label + "/historical_constraints", 0 < len(history) <= 4 and all(
                ref["record_id"] in source_records
                and ref.get("source_scope") == source_records[ref["record_id"]]["scope"]
                and ref.get("evidence_origin", {}).get("source") == source_records[ref["record_id"]]["source"]
                and ref.get("origin") == "current_episode_history"
                and ref.get("hard_pruning") is False and ref.get("reuse_verdict") is False
                for ref in history))
        audit.check(label + "/lineage/" + route, current.get("request") == expected_request
                    and current.get("features") == expected_features
                    and origin.get("source_request_id") == original["request"]["request_id"]
                    and origin.get("source_run_id") == original["request"]["run_id"]
                    and origin.get("original_state_id") == original["request"]["state_id"]
                    and origin.get("state_id_preserved") is True
                    and origin.get("historical_event_ids_renamed") is False
                    and origin.get("diagnostic_candidate_request_generated") == (route == "candidate")
                    and origin.get("path_response_caused_stored_outcome") is False)
    old_records, skill_source = {}, None
    arguments = config.get("arguments", {})
    if arguments.get("memory"):
        old_records = {r["record_id"]: r for r in snapshot(audit, safe(arguments["memory"]))["records"]}
    if arguments.get("skill_memory"):
        skill_source = audit.read(safe(arguments["skill_memory"]))
        from orchestrator.path_skill_memory import _validate_snapshot
        _validate_snapshot(skill_source, check_sources=True)
    preview_map = {p["request_id"]: p for p in previews}
    details, live_attempts, live_valid = [], 0, 0
    for identity, envelope in prepared_map.items():
        request, features = envelope["request"], envelope["features"]
        context, ic = features["memory_context"], features["integrated_context"]
        route = ic["route"]
        original = request_map[identity]
        audit.check(label + "/prepared_request/" + route, request == original["request"] and no_priors(features))
        cards = {c["skill_evidence_id"]: c for c in ic["frozen_skill_evidence"]}
        cross = {cid for cid, card in cards.items()
                 if card_check(audit, card, skill_source, context, label + "/" + route + "/" + cid)}
        refs = {r["record_id"]: r for r in ic["constraint_references"]}
        records = dict(old_records, **source_records)
        for rid, ref in refs.items():
            record = records.get(rid)
            audit.check(label + "/reference/" + route + "/" + rid, record is not None
                        and ref.get("source_scope") == record["scope"]
                        and ref.get("evidence_origin", {}).get("source") == record["source"]
                        and ref.get("current_part_id_mapping") is None
                        and ref.get("hard_pruning") is False and ref.get("reuse_verdict") is False)
        feedback = {f["feedback_id"]: f for f in ic["path_feedback"]}
        for fid, projected in feedback.items():
            actual = next((f["outcome"] for f in source_completed if f["outcome"]["request_id"] == fid), {})
            audit.check(label + "/completed_feedback/" + route, bool(actual)
                        and projected.get("action_schedule") == actual["scope"]["action_schedule"]
                        and projected.get("controller") == actual["controller"]
                        and projected.get("geometry") == actual["geometry"]
                        and projected.get("result") == {key: actual[key] for key in
                            ("path_success", "stability_success", "parts_fix", "feasible")}
                        and projected.get("failure_cause") is None
                        and projected.get("hard_pruning") is False and projected.get("reuse_verdict") is False)
        if route == "candidate":
            audit.check(label + "/completed_candidate_feedback_offered",
                        set(feedback) == {f["outcome"]["request_id"] for f in source_completed})
        preview = preview_map.get(identity)
        response = response_map.get(identity)
        if preview:
            model = json.loads(preview["body"]["input"][0]["content"])
            mf = model["features"]
            audit.check(label + "/actual_model_projection/" + route, model["request"] == request
                        and mf["decision_route"] == route and mf["path_feedback"] == ic["path_feedback"]
                        and mf["episode_feedback"] == ic["episode_feedback"]
                        and mf["constraint_references"] == [
                            expected_projection(r, r["origin"]) for r in ic["constraint_references"]]
                        and mf["frozen_skill_evidence"] == [
                            expected_projection(c, "frozen_skill_validation") for c in ic["frozen_skill_evidence"]]
                        and no_priors(model) and mf.get("physics_reuse_allowed") is False)
        info = (response or {}).get("metadata", {})
        attempted = info.get("llm_live") is True and info.get("attempted") is True
        valid = attempted and response.get("fallback_reason") is None
        live_attempts += int(attempted)
        live_valid += int(valid)
        if response:
            decision = decision_map.get(identity, {})
            audit.check(label + "/decision_receipt/" + route, decision.get("request") == original
                        and decision.get("enriched_request") == envelope and decision.get("response") == response
                        and full_permutation(request, response.get("candidate_ids"))
                        and response.get("state_id") == request["state_id"])
        if attempted:
            audit.check(label + "/live_receipt/" + route, preview is not None
                        and info.get("model_requested") == "gpt-6-luna" and info.get("retries") == 0
                        and (info.get("model_returned") == "gpt-6-luna" if valid
                             else info.get("model_returned") in (None, "gpt-6-luna")))
        used_c = set(info.get("used_constraint_ids", []))
        used_s = set(info.get("used_skill_evidence_ids", []))
        used_f = set(info.get("used_path_feedback_ids", []))
        audit.check(label + "/citation_subset/" + route, used_c.issubset(refs)
                    and used_s.issubset(cards) and used_f.issubset(feedback))
        details.append({"route": route, "request_id": identity, "live_attempt": attempted,
            "valid_live": valid, "fallback_reason": (response or {}).get("fallback_reason"),
            "offered_cross_assembly_skill_ids": sorted(cross),
            "cited_cross_assembly_skill_ids": sorted(used_s & cross),
            "offered_constraint_ids": sorted(refs), "used_constraint_ids": sorted(used_c),
            "direct_previous_constraint_ids": sorted(used_c & set(old_records)),
            "offered_completed_feedback_ids": sorted(feedback), "cited_completed_feedback_ids": sorted(used_f),
            "path_choice": info.get("path_skill"), "candidate_order": (response or {}).get("candidate_ids"),
            "selected_path_executed": False, "candidate_order_consumed_by_DFS": False})
    budget_before, budget_after = audit.read(folder / "budget_before.json", {}), audit.read(folder / "budget_after.json", {})
    delta = budget_after.get("reserved_calls", 0) - budget_before.get("reserved_calls", 0)
    audit.check(label + "/call_accounting", delta == live_attempts == result.get("new_api_attempts")
                == result.get("budget_delta_calls") and live_valid == result.get("live_valid_responses")
                and len(responses) == len(response_map) == result.get("response_count") and live_attempts <= 2)
    if result.get("status") == "PASS":
        audit.check(label + "/reported_live_PASS", live_attempts == live_valid == 2 and result.get("error_type") is None)
    return {"audit_status": "PASS" if all(c["passed"] for c in audit.checks[mark:]) else "FAIL",
        "reported_status": result.get("status"), "error_type": result.get("error_type"),
        "diagnostic_only": True, "full_DFS": False, "new_physics_calls": 0,
        "source_model_decision_fixture": True, "source_physics_backend": "actual_ASAP_RedMax",
        "live_attempts": live_attempts, "live_valid_responses": live_valid, "decisions": details,
        "source_preserved": not changed and before == after, "lineage": lineage,
        "budget_before": budget_before, "budget_after": budget_after,
        "selected_path_executions": 0, "candidate_orders_consumed_by_DFS": 0,
        "fresh_repair_success": "NOT_RUN", "actual_DFS_feedback_consumption": "NOT_RUN"}


def audit_saved_physics(audit, batch):
    folder, label = batch / "saved_choice_physics", "saved_choice_physics"
    result = audit.read(folder / "result.json")
    if result is None:
        return {"audit_status": "NOT_RUN", "reported_status": "NOT_RUN"}
    mark = len(audit.checks)
    worker_manifest = audit.read(folder / "execution_worker_manifest.json", {})
    archived_worker = safe(folder / "execution_worker.py")
    audit.check(label + "/execution_worker_archived_after_run",
                archived_worker.is_file()
                and sha(archived_worker) == worker_manifest.get("sha256")
                and worker_manifest.get("matches_pre_run_frozen_hash") is True
                and worker_manifest.get("archived_after_run") is True,
                {"source": worker_manifest.get("source"), "sha256": worker_manifest.get("sha256"),
                 "capture_timing": "archived_after_run; pre-run hash attestation retained"})
    before = audit.read(folder / "input_hashes_before.json", {})
    after = audit.read(folder / "input_hashes_after.json", {})
    changed = [name for name, expected in before.items()
               if not safe(name).is_file() or sha(safe(name)) != expected]
    audit.check(label + "/source_and_decision_unchanged", bool(before) and before == after
                and not changed and result.get("saved_source_and_decision_files_unchanged") is True, changed)
    diagnostic = batch / "saved_feedback_live"
    response = audit.read(diagnostic / "response_1.json", {})
    request = audit.read(diagnostic / "requests.jsonl", [], lines=True)[0]
    decision = response.get("metadata", {}).get("path_skill", {})
    source = batch / "natural_fixture"
    source_outcome = audit.read(source / "path_outcomes.json")["outcomes"][0]
    source_records = snapshot(audit, source / "constraint_memory.json")["records"]
    source_record = next((r for r in source_records
                          if r["evidence_event_id"] == source_outcome["failed_evidence_id"]), None)
    if source_record is None:
        raise ValueError("component_requires_original_measured_failure_record")
    source_event = audit.read(safe(source_record["source"]["ref"]))
    source_scope = source_event["scope"]
    audit.check(label + "/live_choice_binding", decision == result.get("decision")
                and response.get("request_id") == request["request"]["request_id"] == result.get("new_decision_request_id")
                and response.get("fallback_reason") is None
                and response.get("metadata", {}).get("llm_live") is True
                and response.get("metadata", {}).get("attempted") is True
                and result.get("source_decision_request_id") == source_outcome["request_id"]
                and result.get("source_failed_evidence_id") == source_event["event_id"]
                and result.get("source_fixture_second_axis") == source_outcome["decision"]["second_axis"]
                and result.get("current_second_axis") == decision.get("second_axis")
                and result.get("source_fixture_path_success") == source_outcome["path_success"])
    audit.check(label + "/component_not_DFS_or_source_relabelling",
                result.get("API_calls") == 0 and result.get("full_DFS") is False
                and result.get("selected_plan_edge") is False and result.get("solver_not_resumed") is True
                and result.get("recorded_failure_state_component_test") is True
                and result.get("source_fixture_result_relabelled") is False
                and result.get("candidate_diagnostic_response_consumed") is False
                and result.get("source_failed_probe_not_reexecuted") is True
                and result.get("hard_pruning") is False and result.get("physics_result_reuse") is False
                and result.get("source_skill_card_citation_claim") == bool(decision.get("used_skill_evidence_ids"))
                and result.get("used_skill_evidence_ids") == decision.get("used_skill_evidence_ids"))
    if not result.get("actual_physics"):
        return {"audit_status": "PASS" if all(c["passed"] for c in audit.checks[mark:]) else "FAIL",
                "reported_status": result.get("status"), "error_type": result.get("error_type"),
                "new_physics_calls": 0, "full_DFS": False, "selected_plan_edge": False}
    path_file, event_file = folder / "trajectory.json", folder / "evidence.json"
    trajectory, event = audit.read(path_file), audit.read(event_file)
    scope, controller, geometry = result["new_scope"], trajectory["controller"], trajectory["geometry"]
    from worker.constraint_memory import validate_event
    validate_event(event, scope["task_context"])
    changed_scope_keys = {"action_schedule", "rng_before", "collector_version", "initial_q"}
    same_conditions = ({k: v for k, v in scope.items() if k not in changed_scope_keys}
                       == {k: v for k, v in source_scope.items() if k not in changed_scope_keys})
    same_start = (len(trajectory["path"][0]) == len(source_scope["initial_q"]) == 6
                  and all(abs(a - b) <= 1e-9 + 1e-7 * abs(b)
                          for a, b in zip(trajectory["path"][0], source_scope["initial_q"])))
    audit.check(label + "/physical_conditions_and_start", same_conditions and same_start
                and event["scope"] == scope and result.get("rng_restored_from_source") is False
                and all(result.get(key) is True for key in (
                    "task_context_equal", "source_binary_controller_hashes_equal", "source_sdf_cache_hashes_equal",
                    "collision_thresholds_equal", "path_state_start_matches_source",
                    "input_files_unchanged", "sdf_cache_unchanged", "source_and_binary_unchanged")))
    expected_schedule = [{"step": 0, "action": AXES["+X"]},
                         {"step": decision["switch_step"], "action": AXES[decision["second_axis"]]}]
    audit.check(label + "/trajectory_and_endpoint_binding",
                sha(path_file) == result.get("trajectory_sha256") == event["trajectory"]["sha256"]
                and sha(event_file) == result.get("evidence_sha256")
                and event["trajectory"]["relative_path"] == "trajectory.json"
                and event["event_id"] == request["request"]["request_id"] + ":new-component-endpoint"
                and len(trajectory["path"]) == event["trajectory"]["states"] == geometry["path_state_count"]
                and trajectory["path"][-1] == event["terminal_q"] == geometry["end_q"]
                and trajectory["path"][0] == scope["initial_q"] == geometry["start_q"]
                and trajectory["raw_success"] == event["raw_success"] == result.get("path_success")
                and controller == result.get("controller") and geometry == result.get("geometry")
                and controller["actions"] == scope["action_schedule"] == expected_schedule
                and controller["skill_id"] == decision["skill_id"]
                and controller["reset_count"] == 1 and controller["intermediate_resets"] == 0
                and controller["dfs_state_mutations"] == 0 and controller["upstream_check_success_calls"] == 1
                and result.get("path_check_calls") == 1)
    switch = controller.get("switch_q")
    noncollinear = False
    if switch:
        first = [b - a for a, b in zip(geometry["start_q"][:3], switch[:3])]
        second = [b - a for a, b in zip(switch[:3], geometry["end_q"][:3])]
        cross = [first[1] * second[2] - first[2] * second[1],
                 first[2] * second[0] - first[0] * second[2],
                 first[0] * second[1] - first[1] * second[0]]
        norm = lambda v: math.sqrt(sum(x * x for x in v))
        noncollinear = (min(norm(first), norm(second)) > 1e-9
                        and norm(cross) / (norm(first) * norm(second)) > 1e-6)
    audit.check(label + "/actual_turn_geometry", geometry["noncollinear"] == noncollinear
                and (not noncollinear or switch in trajectory["path"])
                and controller["switch_count"] in (0, 1))
    path_pass, stability = result.get("path_success"), result.get("stability_success")
    feasible = path_pass is True and stability is True
    audit.check(label + "/fresh_path_then_stability_verdict", result.get("feasible") == feasible
                and result.get("status") == ("PASS" if feasible else "UNKNOWN")
                and result.get("completed_noncollinear_recovery") == (
                    feasible and controller["switch_count"] == 1 and noncollinear)
                and (result.get("stability_check_calls") == 1 and type(stability) is bool
                     and stability == (result.get("parts_fix") is not None) if path_pass else
                     result.get("stability_check_calls") == 0 and stability is None and result.get("parts_fix") is None))
    frozen, measurements = constraints(audit, folder, label, scope["task_context"])
    expected_count = sum(p["separation_unsatisfied"] for p in event["measurements"]["pairs"]) if not path_pass else 0
    audit.check(label + "/new_conditional_memory_only", len(frozen["records"])
                == result.get("stored_conditional_constraints") == expected_count
                and all(r["evidence_event_id"] == event["event_id"] for r in frozen["records"])
                and result.get("endpoint_geometry_queries") == event["measurements"]["geometry_queries"])
    cache_io = result.get("path_cache_io", []) + result.get("stability_cache_io", [])
    audit.check(label + "/read_only_SDF_io", bool(cache_io) and all(
        item.get("only_save_flag_changed") is True and item.get("xml_save_sdf") is False
        and item.get("xml_load_sdf") is True for item in cache_io))
    return {"audit_status": "PASS" if all(c["passed"] for c in audit.checks[mark:]) else "FAIL",
        "reported_status": result.get("status"), "actual_physics": True, "API_calls": 0,
        "path_check_calls": result.get("path_check_calls"), "stability_check_calls": result.get("stability_check_calls"),
        "path_success": path_pass, "stability_success": stability, "feasible": feasible,
        "full_DFS": False, "selected_plan_edge": False, "solver_not_resumed": True,
        "source_fixture_second_axis": source_outcome["decision"]["second_axis"],
        "new_selected_second_axis": decision["second_axis"], "decision": decision,
        "source_decision_request_id": result["source_decision_request_id"],
        "new_decision_request_id": result["new_decision_request_id"],
        "actual_noncollinear_turn": controller["switch_count"] == 1 and noncollinear,
        "new_constraint_count": len(frozen["records"]), "constraint_measurements": measurements,
        "source_card_cited": bool(decision["used_skill_evidence_ids"]),
        "source_files_preserved": not changed and before == after,
        "conditions_match_except_new_action_and_rng": same_conditions and same_start,
        "rng_restored_from_source": False, "performance_or_generalization_effect": "UNVERIFIED"}


def run(batch):
    batch, audit = safe(batch), Audit()
    cases = {label: audit.protect(label, lambda label=label: audit_case(audit, batch / label, label))
             for label in LABELS}
    diagnostic = audit.protect("saved_feedback_live", lambda: audit_saved_diagnostic(audit, batch / "saved_feedback_live"))
    component = audit.protect("saved_choice_physics", lambda: audit_saved_physics(audit, batch))
    authorization = audit.read(batch / "authorization.json", {})
    current = audit.protect("current_ledger", lambda: ledger(audit, batch / "live_batch.sqlite",
        "current_ledger", expected_calls=12, expected_cap="0.05"))
    previous = {name: audit.protect(name + "_ledger", lambda name=name: ledger(audit,
        ROOT / "runs" / name / "live_batch.sqlite", name + "_ledger")) for name in PRIOR_BATCHES}
    if current.get("status") in ("COMPLETE", "PENDING") and all(
            v.get("status") in ("COMPLETE", "PENDING") for v in previous.values()):
        prior_reserve = sum((Decimal(v["estimated_reservations_usd"]) for v in previous.values()), Decimal(0))
        prior_commit = sum((Decimal(v["budget_committed_usd"]) for v in previous.values()), Decimal(0))
        total = prior_commit + Decimal(current["budget_committed_usd"])
        audit.check("prior_budget_provenance", prior_reserve == Decimal(str(authorization["prior_reserved_usd"])))
        audit.check("cumulative_approval", total <= Decimal("0.10")
                    and authorization.get("new_max_calls") == 12
                    and Decimal(str(authorization.get("new_max_estimated_usd"))) == Decimal("0.05"))
        audit.check("live_case_and_diagnostic_receipts_equal_ledger",
                    sum(c.get("live_attempts", 0) for c in cases.values()) + diagnostic.get("live_attempts", 0)
                    == current["reserved_calls"])
        budget = {"prior_reserved_usd": str(prior_reserve), "current_committed_usd": current["budget_committed_usd"],
                  "total_committed_usd": str(total), "user_approved_usd": "0.10",
                  "configured_rate_cost_estimates_only": True, "billing_cap_guaranteed": False}
    else:
        budget = {"status": "UNVERIFIED_INCOMPLETE_LEDGER"}
    preserved = audit.protect("preservation", lambda: preserved_files(audit, batch))
    staging = audit.read(batch / "staging.json")
    if staging:
        audit.check("new_input_staging_hashes", staging.get("source_hashes") == staging.get("target_hashes")
                    and staging.get("source_unchanged") is True and staging.get("copied_sdf_files") == 0)
    selection = audit.read(batch / "selection_manifest.json", {})
    public = audit.read(batch / "public_input_provenance.json", {})
    audit.check("preregistered_task_set", {t.get("id") for t in selection.get("tasks", [])} == {"07009", "02628"}
                and all(t.get("held_out") is False for t in selection.get("tasks", []))
                and selection.get("controlled_probe") is False and selection.get("performance_comparison") is False)
    if selection.get("source_skill_file"):
        audit.check("preregistered_source_skill_frozen",
                    sha(safe(selection["source_skill_file"])) == selection.get("source_skill_sha256"))
    audit.check("public_input_provenance_attestation", public.get("status") == "PASS"
                and set(public.get("assemblies", {})) == {"07009", "02628"}
                and all(v is True for files in public.get("assemblies", {}).values() for v in files.values()))
    changed = [path for path, expected in audit.inputs.items()
               if not safe(path).is_file() or sha(path) != expected]
    audit.check("audit_inputs_unchanged", not changed, changed)
    failed = [c for c in audit.checks if not c["passed"]]
    incomplete = [key for key, value in cases.items() if value.get("audit_status") in ("NOT_RUN", "INCOMPLETE")]
    if preserved.get("environment", {}).get("status") == "NOT_RUN":
        incomplete.append("environment_final")
    if component.get("actual_physics") and not preserved.get("environment", {}).get("after_saved_choice"):
        incomplete.append("environment_final_after_saved_choice")
    status = "FAIL" if failed else "PARTIAL" if incomplete or current.get("status") != "COMPLETE" else "PASS"
    return {"schema_version": 1, "kind": "integrated_repair_saved_evidence_audit", "audit_status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(), "batch_dir": str(batch),
        "meaning_of_PASS": "Evidence consistency, not success of every function or proof of improved performance.",
        "audit_api_calls": 0, "audit_physics_calls": 0, "sqlite_writes": 0, "pickle_loading": False,
        "cases": cases, "authorization": authorization, "current_ledger": current, "previous_ledgers": previous,
        "saved_evidence_diagnostic": diagnostic,
        "saved_choice_physics_component": component,
        "cumulative_budget": budget, "preservation": preserved, "staging": staging,
        "preregistration": selection, "public_input_provenance": public,
        "missing_or_incomplete_cases": incomplete, "checks": audit.checks, "failed_checks": failed,
        "read_artifact_sha256": audit.inputs,
        "limitations": ["Fixture choices are model doubles; physical verification remains actual ASAP.",
            "A prior failed development task is not held-out evaluation data.",
            "Source IDs and axes are not mapped to the target, even if their text matches.",
            "Exposure, citation, procedure execution, new path/stability success and replay are distinct.",
            "One repair per run cannot by itself demonstrate completed feedback on a later path request.",
            "Saved-evidence diagnostic API decisions are neither resumed DFS nor new physical repair executions.",
            "Untriggered or failed repairs are retained; whole-plan success is a separate outcome.",
            "Last-two-part upstream selection does not use the LLM. Performance and causal benefit are unverified."]}


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
    value = run(safe(args.batch_dir))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"audit_status": value["audit_status"], "checks": len(value["checks"]),
        "failed_checks": len(value["failed_checks"]), "output": str(output), "api_calls": 0, "physics_calls": 0}))
    return 0 if value["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
