"""Stage-separated counts from saved decisions and actual worker artifacts.

No provider or verifier is called. Missing logs remain unsupported, not zero.
Interpreter execution means a proposal; physical attribution additionally needs
the exact host/worker/attempt chain and an ASAP_RedMax result.
"""
from collections import Counter
from copy import deepcopy
import math

from orchestrator.generated_execution_audit import execution_bindings
from orchestrator.knowledge_index import observed_signature
from orchestrator.skill_program import _scoped_last, _predicate
from orchestrator.skill_registry import retrieve
from worker.explicit_action_contract import validate_request, validate_response, canonical

COUNTS = ("checked", "triggered", "applicable", "offered_to_llm_selector", "selected",
          "actual_llm_selected", "deterministic_selected", "program_proposals",
          "worker_bound_attempts", "physical_attempts", "physical_passes", "transition_passes")


def _unique(rows, key):
    found, duplicates = {}, 0
    for row in rows:
        identity = key(row)
        if identity is None:
            raise ValueError("missing_usage_identity")
        if identity in found:
            if canonical(found[identity]) != canonical(row):
                raise ValueError("conflicting_usage_identity")
            duplicates += 1
        else:
            found[identity] = row
    return list(found.values()), duplicates


def _live_selection_bound(decision, selected, selections):
    request, meta = decision["request"], decision["response"]["metadata"]
    identity = selected.get("selected_skill_id")
    matches = []
    for item in selections:
        packet, response = item.get("packet", {}), item.get("response", {})
        cards = [c for c in packet.get("skills", []) if c.get("skill_id") == identity
                 and c.get("program_hash") == identity]
        provider = response.get("metadata") or {}
        if (packet.get("current_request") == request and len(cards) == 1
                and packet.get("run_id") == request["run_id"]
                and packet.get("state_id") == response.get("state_id") == request["state_id"]
                and packet.get("operation") == response.get("operation") == "select"
                and type(packet.get("request_id")) is str
                and packet["request_id"] == response.get("request_id")
                and response == selected and response.get("origin") == "actual_llm"
                and (provider.get("llm_live") is True or provider.get("original_llm_live") is True)
                and type(meta.get("generated_library_sha256")) is str
                and packet.get("context", {}).get("library_sha256") == meta["generated_library_sha256"]):
            matches.append(item)
    return len(matches) == 1


def _physical_dispatch(raw):
    """Backend identity alone does not establish that a checker was called."""
    if raw.get("backend") != "ASAP_RedMax":
        return False
    counts = [raw.get(key) for key in ("path_check_calls", "stability_check_calls")]
    if any(type(value) is int and value > 0 for value in counts):
        return True
    if all(type(value) is int and value == 0 for value in counts):
        return False
    return None


def summarize_usage(decisions: list, report=None, selections=None, library=None,
                    *, allow_fixture=False) -> dict:
    """Count exact-request stages; never attribute another skill's execution."""
    if type(decisions) is not list or len(decisions) > 10000:
        raise ValueError("invalid_usage_decisions")
    decisions, duplicate_decisions = _unique(decisions, lambda d: (
        d.get("request", {}).get("run_id"), d.get("request", {}).get("request_id")))
    run_ids = {d["request"]["run_id"] for d in decisions}
    if len(run_ids) > 1:
        raise ValueError("usage_report_requires_one_episode")
    programs = {r["program_hash"]: r["program"] for r in
                retrieve(library, allow_fixture=allow_fixture)} if library is not None else {}
    if selections is not None:
        selections, duplicate_selections = _unique(selections, lambda s: s.get("packet", {}).get("request_id"))
    else:
        duplicate_selections = None
    rows, request_rows, missing = {}, [], Counter()
    def bucket(identity):
        if type(identity) is not str or not identity:
            raise ValueError("invalid_usage_skill_identity")
        if identity not in rows:
            rows[identity] = {"program_hash": identity, **{key: 0 for key in COUNTS},
                              "not_applicable_reasons": {}, "no_trigger_details": {}}
        return rows[identity]
    for identity in programs:
        bucket(identity)
    hosts = {}
    for decision in decisions:
        request = validate_request(decision["request"])
        response = validate_response(request, decision["response"])
        hosts[request["request_id"]] = decision
        metadata = response["metadata"]
        audit = metadata.get("generated_skill_applicability")
        if audit is None:
            missing["applicability_requests"] += 1
            continue
        if type(audit) is not list:
            raise ValueError("invalid_applicability_audit")
        if library is not None and metadata.get("generated_library_sha256") != library["content_sha256"]:
            raise ValueError("usage_library_mismatch")
        seen = set()
        for item in audit:
            identity = item["program_hash"]
            if item.get("strategy_id") != identity or identity in seen or (library is not None and identity not in programs):
                raise ValueError("ambiguous_applicability_program")
            seen.add(identity)
            row = bucket(identity)
            row["checked"] += 1
            if type(item.get("triggered")) is not bool:
                raise ValueError("unknown_triggered_value")
            row["triggered"] += int(item["triggered"])
            reason = item.get("not_applicable_reason")
            row["applicable"] += int(item["triggered"] and reason is None)
            if reason is not None:
                counts = row["not_applicable_reasons"]
                counts[reason] = counts.get(reason, 0) + 1
            detail = None
            if reason == "no_trigger" and identity in programs and programs[identity]["language"] != "python_policy_v1":
                last = _scoped_last(request)
                predicates = [{p: _predicate(p, request, last) for p in rule["when"]}
                              for rule in programs[identity]["rules"]]
                detail = ("no_current_attempt_or_observation" if last is None and (
                    not request.get("attempted_actions") or not request.get("path_stage_observations"))
                    else "current_feedback_binding_rejected" if last is None
                    else "predicate_false" if not any(all(values.values()) for values in predicates)
                    else "current_interpreter_disagrees_with_saved_no_trigger")
                counts = row["no_trigger_details"]
                counts[detail] = counts.get(detail, 0) + 1
            request_rows.append(dict(run_id=request["run_id"], request_id=request["request_id"],
                program_hash=identity, saved_triggered=item["triggered"], saved_reason=reason,
                current_interpreter_detail=detail))
        selected = metadata.get("generated_selector_response")
        if type(selected) is dict and selected.get("status") == "selected":
            identity = selected.get("selected_skill_id")
            if identity not in seen:
                raise ValueError("selected_skill_not_checked")
            if not any(a["program_hash"] == identity and a["triggered"] and
                       a.get("not_applicable_reason") is None for a in audit):
                raise ValueError("selected_skill_not_applicable")
            row = bucket(identity)
            row["selected"] += 1
            row["deterministic_selected"] += int(selected.get("origin") == "deterministic_control")
            if selections is not None and _live_selection_bound(decision, selected, selections):
                row["actual_llm_selected"] += 1
        if metadata.get("generated_program_executed") is True:
            identity = metadata.get("generated_program_hash")
            if identity != metadata.get("generated_skill_id") or identity not in seen:
                raise ValueError("proposal_program_mismatch")
            if (type(selected) is not dict or selected.get("status") != "selected"
                    or selected.get("selected_skill_id") != identity):
                raise ValueError("proposal_does_not_match_selected_skill")
            if not any(a["program_hash"] == identity and a["triggered"] and
                       a.get("not_applicable_reason") is None for a in audit):
                raise ValueError("proposal_program_not_applicable")
            bucket(identity)["program_proposals"] += 1
    if selections is not None:
        for selection in selections:
            packet = selection["packet"]
            request = packet.get("current_request") or {}
            host = hosts.get(request.get("request_id"))
            if host is None or host["request"] != request:
                missing["unbound_selector_packets"] += 1
                continue
            identities = [r["skill_id"] for r in packet.get("skills", [])]
            if len(set(identities)) != len(identities):
                raise ValueError("duplicate_selector_card")
            for identity in identities:
                bucket(identity)["offered_to_llm_selector"] += 1
    attempt_rows, stages, duplicate_attempts, excluded = [], Counter(), None, []
    unknown_physical_skills = set()
    if report is not None:
        if type(report) is not dict or type(report.get("attempts")) is not list or type(report.get("decisions")) is not list:
            raise ValueError("incomplete_worker_usage_report")
        attempts, duplicate_attempts = _unique(report["attempts"], lambda a: a.get("event_id"))
        workers, _ = _unique(report["decisions"], lambda d: d.get("request", {}).get("request_id"))
        bound, discarded = execution_bindings(decisions, {"attempts": attempts, "decisions": workers})
        bound_events = {a["event_id"] for a in bound}
        excluded = [a["event_id"] for a in discarded]
        for attempt in attempts:
            raw = attempt.get("raw") or {}
            actual_backend = raw.get("backend") == "ASAP_RedMax"
            dispatch = _physical_dispatch(raw)
            actual = dispatch is True
            signature = observed_signature(dict(status=attempt.get("status"),
                path_success=raw.get("path_success"), stability_success=raw.get("stability_success"),
                raw=raw, verifier_origin="actual_ASAP_RedMax" if actual_backend else "unsupported"), attempt.get("choice") or {})
            stage_key = str(signature["observed_failure_stage"]) + ":" + str(signature["stability_phase"])
            if actual:
                stages[stage_key] += 1
            identity, origin = None, None
            matched = (attempt["event_id"] in bound_events and
                       attempt.get("state_id") == hosts[attempt["request_id"]]["request"]["state_id"])
            if attempt["event_id"] in bound_events and not matched:
                excluded.append(attempt["event_id"])
            if matched:
                host = hosts[attempt["request_id"]]
                meta = host["response"]["metadata"]
                identity, origin = meta["generated_program_hash"], meta.get("generated_skill_selection_origin")
                row = bucket(identity)
                row["worker_bound_attempts"] += 1
                if dispatch is None:
                    missing["physical_dispatch_attempts"] += 1
                    unknown_physical_skills.add(identity)
                row["physical_attempts"] += int(actual)
                passed = (actual and attempt.get("status") == raw.get("status") == "PASS"
                          and raw.get("path_success") is True and raw.get("stability_success") is True
                          and raw.get("feasible") is True and attempt.get("deadline_exceeded") is False)
                row["physical_passes"] += int(passed)
                row["transition_passes"] += int(passed and attempt.get("accepted_for_transition") is True)
            attempt_rows.append(dict(event_id=attempt["event_id"], request_id=attempt.get("request_id"),
                action_id=attempt.get("action_id"), generated_skill_id=identity, program_hash=identity,
                decision_origin=origin, binding_status="exact_host_worker_attempt" if matched else "no_generated_binding",
                actual_backend_recorded=actual_backend, physical_dispatch_recorded=dispatch,
                observed_signature=signature))
    # An absent evidence file is not proof that its event count was zero.
    for row in rows.values():
        if report is None:
            for key in ("worker_bound_attempts", "physical_attempts", "physical_passes", "transition_passes"):
                row[key] = None
        if selections is None:
            row["actual_llm_selected"] = row["offered_to_llm_selector"] = None
        if row["program_hash"] in unknown_physical_skills:
            for key in ("physical_attempts", "physical_passes", "transition_passes"):
                row[key] = None
    return dict(schema_version=1, kind="generated_skill_usage_v1", decisions=len(decisions),
        per_skill={k: rows[k] for k in sorted(rows)}, request_skill_rows=request_rows,
        attempt_rows=attempt_rows, actual_attempt_stage_counts=dict(stages),
        missing_evidence=dict(missing), duplicate_decisions=duplicate_decisions,
        duplicate_attempts=duplicate_attempts, duplicate_selections=duplicate_selections,
        discarded_generated_attempt_event_ids=excluded,
        coverage=dict(worker_report=report is not None, selector_log=selections is not None,
                      library=library is not None),
        trigger_detail_basis="current_pure_predicates; saved trigger result remains authoritative",
        source_library_fixture=library.get("fixture") if library is not None else None,
        physical_evidence_basis="saved_artifacts; not a new physical experiment",
        performance_effect="NOT_RUN", new_api_calls=0, new_physical_calls=0)


def summarize_declared_case(request, response, selections, raw_recovery, case_report,
                            library=None, *, allow_fixture=False):
    """Declared-state recovery has no DFS worker decision/transition event.

    Keep that distinction explicit instead of manufacturing a worker report.
    A matching saved physical result may establish a recovery execution only.
    """
    from orchestrator.skill_case import validate_trigger_scope
    if case_report.get("case_type") != "declared_state_replay_then_free_recovery":
        raise ValueError("unsupported_declared_case_type")
    request = validate_request(request)
    response = validate_response(request, response)
    result = summarize_usage([dict(request=request, response=response)], None, selections,
                             library, allow_fixture=allow_fixture)
    result["execution_mode"] = "declared_state_replay_then_free_recovery"
    result["coverage"]["declared_recovery_result"] = raw_recovery is not None
    if raw_recovery is None:
        return result
    validate_trigger_scope(request, raw_recovery, response["choice"])
    metadata = response["metadata"]
    identity = metadata.get("generated_program_hash")
    if metadata.get("generated_program_executed") is True:
        if identity != metadata.get("generated_skill_id"):
            raise ValueError("declared_case_program_mismatch")
        row = result["per_skill"][identity]
        elapsed, budget = raw_recovery.get("elapsed_seconds"), case_report.get("case_wall_seconds")
        timely = (type(elapsed) in (int, float) and type(budget) in (int, float)
                  and math.isfinite(elapsed) and math.isfinite(budget) and 0 <= elapsed <= budget)
        passed = (raw_recovery.get("status") == "PASS" and raw_recovery.get("path_success") is True
                  and raw_recovery.get("stability_success") is True and raw_recovery.get("feasible") is True
                  and timely)
        row["physical_attempts"] = 1
        row["physical_passes"] = int(passed)
        row["declared_recovery_passes"] = int(passed)
        # No DFS acceptance was performed in this execution mode.
        row["worker_bound_attempts"] = row["transition_passes"] = None
        result["declared_recovery_binding"] = dict(request_id=request["request_id"],
            action_id=response["action_id"], generated_skill_id=identity, program_hash=identity,
            binding_status="saved_request_response_actual_scope_checked", timely=timely,
            observed_signature=observed_signature(dict(status=raw_recovery["status"],
                path_success=raw_recovery["path_success"], stability_success=raw_recovery["stability_success"],
                raw=raw_recovery, verifier_origin="ASAP_RedMax"), response["choice"]))
    return result
