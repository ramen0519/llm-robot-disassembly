"""Audit generic assembly event contracts without executing or judging physics.

Integrity status is separate from execution status: a faithfully logged verifier
ERROR is counted, not reinterpreted as a collision or hidden as malformed data.
"""
from collections import Counter
import hashlib
import json
import math


KINDS = {"run_start", "run_end", "generation", "rank_decision", "candidate_yield",
         "verifier_start", "verifier_result"}
RANKERS = {"noop", "id_ascending", "reverse_native"}


def _ids(values):
    return (isinstance(values, (list, tuple)) and all(type(value) is str and value for value in values)
            and len(values) == len(set(values)))


def _unique(values):
    return all(type(value) is str and value for value in values) and len(values) == len(set(values))


def _state(parts):
    return hashlib.sha256(json.dumps(sorted(parts)).encode("utf-8")).hexdigest()


def _order(native, ranker):
    if ranker == "id_ascending":
        return sorted(native)
    if ranker == "reverse_native":
        return list(reversed(native))
    return list(native)


def _action(action):
    return (isinstance(action, list) and len(action) == 3
            and all(type(value) in (int, float) and math.isfinite(value) for value in action)
            and any(value != 0 for value in action))


def _audit_events(events, mode, ranker, part_ids, base_part=None):
    """Return integrity checks and raw execution/call metrics; never load a tree.

    Native mode has no feasibility instrumentation, so its verifier count is
    unknown. Multiple pose attempts and the final unranked two-part leaf remain
    distinct calls. There is deliberately no comparison to the upstream budget.
    """
    checks = {}
    metrics = dict(verifier_calls=None if mode == "native" else 0,
                   verifier_status_counts={status: 0 for status in ("PASS", "UNKNOWN", "ERROR")},
                   generation_count=0, rank_requests=0, candidate_yields=0,
                   changed_generations=0, execution_status="UNKNOWN",
                   run_completed_normally=False, verifier_errors_reported=0,
                   internal_geometry_operations=None)

    def check(label, value):
        checks[label] = bool(value)

    def finish():
        return dict(status="PASS" if all(checks.values()) else "ERROR", checks=checks, metrics=metrics)

    check("valid_audit_arguments", mode in ("native", "trace", "rank")
          and ranker in RANKERS | {None} and (mode != "rank" or ranker in RANKERS)
          and _ids(part_ids) and len(part_ids) >= 2 and (base_part is None or base_part in part_ids))
    check("event_objects", isinstance(events, list) and bool(events)
          and all(isinstance(event, dict) for event in events))
    if not all(checks.values()):
        return finish()
    starts = [event for event in events if event.get("kind") == "run_start"]
    ends = [event for event in events if event.get("kind") == "run_end"]
    check("one_bounded_run", len(starts) == len(ends) == 1
          and events[0].get("kind") == "run_start" and events[-1].get("kind") == "run_end")
    if not checks["one_bounded_run"]:
        return finish()
    first, end = starts[0], ends[0]
    run_id = first.get("run_id")
    check("single_run_id", type(run_id) is str and bool(run_id)
          and all(event.get("run_id") == run_id for event in events))
    check("event_ids_unique", _unique([event.get("event_id") for event in events]))
    check("known_event_kinds", all(event.get("kind") in KINDS for event in events))
    check("mode_and_ranker", first.get("mode") == mode
          and first.get("ranker_name") == (ranker if mode == "rank" else None))
    check("run_base_identity", "base_part" in first and first["base_part"] == base_part)
    check("local_without_memory", type(first.get("live_llm_calls")) is int
          and first["live_llm_calls"] == 0
          and all(not any(key == "memory" or key.startswith("memory_") for key in event) for event in events))
    check("terminal_status_is_explicit", end.get("status") in ("returned_from_upstream", "ERROR")
          and (end.get("status") != "ERROR" or (type(end.get("error_type")) is str and bool(end["error_type"]))))
    metrics["execution_status"] = ("RETURNED" if end.get("status") == "returned_from_upstream"
                                   else "ERROR" if end.get("status") == "ERROR" else "UNKNOWN")
    metrics["run_completed_normally"] = end.get("status") == "returned_from_upstream"
    if mode == "native":
        check("native_has_no_hook_events", len(events) == 2)
        return finish()

    by_kind = {kind: [event for event in events if event.get("kind") == kind] for kind in KINDS}
    generations, decisions = by_kind["generation"], by_kind["rank_decision"]
    yields, verifier_starts, results = (by_kind[name] for name in ("candidate_yield", "verifier_start", "verifier_result"))
    metrics.update(generation_count=len(generations), rank_requests=len(decisions), candidate_yields=len(yields),
                   verifier_calls=len(verifier_starts),
                   changed_generations=sum(event.get("native_order") != event.get("applied_order") for event in generations))
    counts = Counter(event.get("status") for event in results if isinstance(event.get("status"), str))
    metrics["verifier_status_counts"] = {status: counts[status] for status in ("PASS", "UNKNOWN", "ERROR")}
    metrics["verifier_errors_reported"] = counts["ERROR"]
    check("generation_ids_unique", _unique([event.get("generation_id") for event in generations]))
    generation_map = {event.get("generation_id"): event for event in generations
                      if isinstance(event.get("generation_id"), str)}
    check("request_ids_unique", _unique([event.get("request", {}).get("request_id")
          if isinstance(event.get("request"), dict) else None for event in decisions]))
    check("verifier_call_ids_unique", _unique([event.get("call_id") for event in verifier_starts])
          and _unique([event.get("call_id") for event in results]))
    check("candidate_sets_preserved", all(
        _ids(event.get("assembled_parts")) and set(event["assembled_parts"]).issubset(part_ids)
        and (base_part is None or base_part in event["assembled_parts"])
        and _ids(event.get("native_order")) and _ids(event.get("applied_order"))
        and set(event["native_order"]) == set(event["assembled_parts"]) - ({base_part} if base_part is not None else set())
        and set(event["applied_order"]) == set(event["native_order"])
        and event.get("state_id") == _state(event["assembled_parts"])
        for event in generations))
    check("expected_candidate_order", all(
        isinstance(event.get("native_order"), list)
        and event.get("applied_order") == _order(event["native_order"], ranker if mode == "rank" else "noop")
        for event in generations))
    decision_generations = [event.get("generation_id") for event in decisions]
    check("one_rank_per_generation", decision_generations == [event.get("generation_id") for event in generations]
          if mode == "rank" else not decisions)
    decision_contract = True
    for event in decisions:
        generation = generation_map.get(event.get("generation_id"))
        request = event.get("request")
        if not isinstance(request, dict) or generation is None:
            decision_contract = False
            continue
        decision_contract &= (event.get("ranker_name") == ranker and event.get("fallback_reason") is None
                              and request.get("run_id") == run_id and request.get("base_part") == base_part
                              and request.get("state_id") == generation.get("state_id")
                              and request.get("assembled_parts") == generation.get("assembled_parts")
                              and request.get("candidate_ids") == generation.get("native_order")
                              and event.get("applied_order") == generation.get("applied_order"))
    check("rank_request_response_links", decision_contract)
    check("rank_does_not_consume_rng", all(type(event.get("rng_before")) is str and bool(event["rng_before"])
          and event["rng_before"] == event.get("rng_after") for event in decisions))

    pending, current_generation, current_candidate = None, None, None
    yielded = {key: [] for key in generation_map}
    paired, request_contract, raw_contract, link_contract, decision_position = True, True, True, True, True
    waiting_decision = None
    for event in events:
        kind = event.get("kind")
        if kind == "rank_decision":
            decision_position &= waiting_decision is None
            waiting_decision = event.get("generation_id")
        elif kind == "generation":
            current_generation, current_candidate = event.get("generation_id"), None
            if mode == "rank":
                decision_position &= waiting_decision == current_generation
                waiting_decision = None
        elif kind == "candidate_yield":
            key, candidate = event.get("generation_id"), event.get("part")
            if not isinstance(key, str) or key not in generation_map or key != current_generation:
                link_contract = False
                continue
            yielded[key].append(candidate)
            current_candidate = candidate
        elif kind == "verifier_start":
            paired &= pending is None
            pending = event
            request, key = event.get("request"), event.get("generation_id")
            valid = (isinstance(request, dict) and type(request.get("part_move")) is str
                     and request["part_move"] in part_ids and request["part_move"] != base_part
                     and _ids(request.get("parts_rest")) and bool(request["parts_rest"])
                     and _ids(request.get("parts_removed")) and request["part_move"] not in request["parts_rest"]
                     and set(request["parts_rest"]).issubset(part_ids)
                     and set(request["parts_removed"]) == set(part_ids) - set(request["parts_rest"]) - {request["part_move"]}
                     and (base_part is None or base_part in request["parts_rest"])
                     and "pose" in request and type(request.get("max_grippers")) is int and request["max_grippers"] >= 1)
            request_contract &= valid
            if valid:
                if key is None:
                    link_contract &= len(request["parts_rest"]) == 1
                else:
                    generation = generation_map.get(key) if isinstance(key, str) else None
                    link_contract &= (generation is not None and key == current_generation
                                      and current_candidate == request["part_move"]
                                      and set(generation.get("assembled_parts", [])) == set(request["parts_rest"]) | {request["part_move"]})
        elif kind == "verifier_result":
            if pending is None or event.get("call_id") != pending.get("call_id"):
                paired = False
                pending = None
                continue
            request = pending.get("request")
            status, raw = event.get("status"), event.get("raw")
            valid = (status in ("PASS", "UNKNOWN", "ERROR") and "blocker" in event and event["blocker"] is None
                     and "path" in event and event["path"] is None and isinstance(request, dict))
            if status == "ERROR":
                valid &= raw is None and type(event.get("error_type")) is str and bool(event["error_type"])
            else:
                valid &= (isinstance(raw, dict) and type(raw.get("feasible")) is bool
                          and raw["feasible"] == (status == "PASS") and isinstance(request, dict))
                if isinstance(raw, dict) and isinstance(request, dict):
                    valid &= (raw.get("part_move") == request.get("part_move")
                              and "pose" in raw and raw["pose"] == request.get("pose")
                              and "base_part" in raw and raw["base_part"] == base_part)
                    if status == "PASS":
                        valid &= (_action(raw.get("action")) and _ids(raw.get("parts_fix"))
                                  and set(raw["parts_fix"]).issubset(request.get("parts_rest", []))
                                  and len(raw["parts_fix"]) + 1 <= request.get("max_grippers", 0))
            raw_contract &= valid
            pending = None
    check("serial_verifier_pairs", paired and pending is None and len(verifier_starts) == len(results))
    check("verifier_request_states", request_contract)
    check("raw_verifier_results_preserved", raw_contract)
    check("candidate_yield_and_verifier_links", link_contract)
    check("rank_decision_precedes_generation", decision_position and waiting_decision is None)
    check("consumed_candidate_prefixes", all(isinstance(generation_map[key].get("applied_order"), list)
          and parts == generation_map[key]["applied_order"][:len(parts)] for key, parts in yielded.items()))
    return finish()


def audit_events(events, mode, ranker, part_ids, base_part=None):
    """Audit event invariants; malformed nested fields produce an integrity error."""
    try:
        return _audit_events(events, mode, ranker, part_ids, base_part)
    except (TypeError, ValueError, KeyError, OverflowError) as error:
        # Do not let a malformed artifact be confused with a successful audit or
        # with an ERROR result that the physical worker actually reported.
        return dict(status="ERROR", checks={"event_fields_readable": False},
                    metrics=dict(verifier_calls=None, execution_status="UNKNOWN",
                                 run_completed_normally=False, internal_geometry_operations=None),
                    error_type=type(error).__name__, error=str(error))
