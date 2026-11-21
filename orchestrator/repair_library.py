"""A reviewed, bounded retry procedure; no search state or physics ownership.

``retry_after_state_change_v1`` promotes a previously UNKNOWN candidate only
after another part's real PASS observations explain the current smaller state.
This is an order-only recovery attempt, not a blocker diagnosis or a new path
planner. Only ASAP can execute the candidate and declare its physical result.
One observed failure/change/success chain remains a provisional scoped case.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


SKILL_ID = "retry_after_state_change_v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def module_hash():
    # The only file access is this module itself; no caller-supplied path exists.
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _ids(value, label):
    if (type(value) not in (list, tuple) or any(type(p) is not str or not p for p in value)
            or len(value) != len(set(value))):
        raise ValueError("invalid_" + label)
    return set(value)


def _request(request):
    if type(request) is not dict:
        raise ValueError("invalid_rank_request")
    for key in ("run_id", "request_id", "state_id"):
        if type(request.get(key)) is not str or not request[key]:
            raise ValueError("missing_" + key)
    assembled = _ids(request.get("assembled_parts"), "assembled_parts")
    candidates = _ids(request.get("candidate_ids"), "candidate_ids")
    base = request.get("base_part")
    if base is not None and (type(base) is not str or base not in assembled):
        raise ValueError("invalid_base_part")
    if candidates != assembled - ({base} if base is not None else set()):
        raise ValueError("repair_requires_complete_candidate_set")
    return assembled, candidates


def _observation(value, request):
    """Return a defensively checked worker observation, or no usable evidence.

    These are trusted-worker event references, never LLM-written explanations.
    The caller preserves raw logs and binds their hashes in its run manifest.
    """
    if type(value) is not dict:
        return None
    event_id, call_id = value.get("event_id"), value.get("call_id")
    prefix = request["run_id"] + ":e"
    if (type(event_id) is not str or not event_id.startswith(prefix)
            or not event_id[len(prefix):].isdigit()
            or type(call_id) is not str or not call_id):
        return None
    attempted, raw = value.get("request"), value.get("raw")
    if type(attempted) is not dict or type(raw) is not dict:
        return None
    part = attempted.get("part_move")
    try:
        rest = _ids(attempted.get("parts_rest"), "parts_rest")
        removed = _ids(attempted.get("parts_removed"), "parts_removed")
    except ValueError:
        return None
    if (type(part) is not str or not part or part in rest or part in removed
            or rest.intersection(removed) or raw.get("part_move") != part
            or raw.get("base_part") != request.get("base_part")):
        return None
    status = value.get("status")
    if not ((status == "UNKNOWN" and raw.get("feasible") is False)
            or (status == "PASS" and raw.get("feasible") is True)):
        return None
    if status == "PASS" and (raw.get("action") is None or raw.get("parts_fix") is None):
        return None
    if request.get("base_part") is not None and request["base_part"] not in rest:
        return None
    return {"event_id": event_id, "event_number": int(event_id[len(prefix):]),
            "call_id": call_id, "status": status, "part": part,
            "assembled": rest | {part}, "rest": rest, "removed": removed,
            "observed_stage": value.get("observed_stage"),
            "request_sha256": _digest(attempted), "raw_sha256": _digest(raw)}


def _eligible(request, episode_feedback):
    current, candidates = _request(request)
    if type(episode_feedback) is not list:
        raise ValueError("feedback_must_be_current_episode_list")
    observations = [_observation(value, request) for value in episode_feedback]
    observations = [value for value in observations if value is not None]
    numbers = [value["event_number"] for value in observations]
    if numbers != sorted(set(numbers)):
        raise ValueError("feedback_event_order_or_identity_invalid")
    result = {}
    for index, failed in enumerate(observations):
        part = failed["part"]
        if (failed["status"] != "UNKNOWN" or part not in candidates
                or not current < failed["assembled"]):
            continue
        later = observations[index + 1:]
        # The upstream DFS owns visited edges. This guard avoids advertising a
        # repair that would request an already observed candidate in this state.
        if any(v["part"] == part and v["assembled"] == current for v in later):
            continue
        state, removed, transitions = set(failed["assembled"]), set(failed["removed"]), []
        for observed in later:
            if (observed["status"] == "PASS" and observed["part"] != part
                    and observed["assembled"] == state and observed["removed"] == removed
                    and current.issubset(observed["rest"])):
                state = set(observed["rest"])
                removed.add(observed["part"])
                transitions.append(observed)
                if state == current:
                    break
        if state != current or not transitions:
            continue
        result[part] = {
            "part_id": part, "failure_event_id": failed["event_id"],
            "failure_call_id": failed["call_id"], "failure_status": "UNKNOWN",
            "failure_observed_stage": failed["observed_stage"],
            "failure_request_sha256": failed["request_sha256"],
            "failure_raw_sha256": failed["raw_sha256"],
            "source_remaining_parts": sorted(failed["assembled"]),
            "current_remaining_parts": sorted(current),
            "state_change_event_ids": [v["event_id"] for v in transitions],
            "state_change_call_ids": [v["call_id"] for v in transitions],
            "removed_part_ids": [v["part"] for v in transitions],
            "blocker": None, "current_retry_verified": False,
        }
    return [result[part] for part in sorted(result)]


def list_applicable(request, episode_feedback):
    """Offer one registered local template iff live observations justify retry.

    No source-task numeric part ID is mapped into this task. A new task must
    supply its own failed attempt and actual state-changing PASS evidence.
    """
    eligible = _eligible(request, episode_feedback)
    if not eligible:
        return []
    return [{
        "skill_id": SKILL_ID, "version": 1, "code_sha256": module_hash(),
        "description": "Retry a previously UNKNOWN candidate after verified removals changed its remaining assembly.",
        "lifecycle": "provisional", "validation_scope": "reviewed_order_only_contract; physical outcomes are logged separately",
        "preconditions": "Current episode UNKNOWN, followed by PASS removals reaching current state; candidate not retried here.",
        "eligible_part_ids": [v["part_id"] for v in eligible], "candidates": eligible,
        "input_contract": "Current complete candidate permutation and one eligible failed_part_id",
        "output_contract": "Same full permutation with the selected retry candidate first",
        "state_mutations": False, "internal_verifier_calls": 0, "allow_hard_pruning": False,
        "blocker_inference": False, "generalization_validated": False,
        "origin": "reviewed_local_procedure_not_generated_code",
    }]


def execute(skill_id, request, proposed_order, args=None, episode_feedback=None):
    """Execute the reviewed permutation procedure; never simulate or alter DFS."""
    if skill_id != SKILL_ID:
        raise ValueError("unregistered_repair_skill")
    _, candidates = _request(request)
    order_set = _ids(proposed_order, "proposed_order")
    if order_set != candidates:
        raise ValueError("repair_requires_full_proposed_permutation")
    if type(args) is not dict or set(args) != {"failed_part_id"}:
        raise ValueError("repair_requires_exact_failed_part_argument")
    failed_part = args["failed_part_id"]
    if type(failed_part) is not str:
        raise ValueError("invalid_failed_part_id")
    applicable = {value["part_id"]: value for value in _eligible(request, episode_feedback)}
    if failed_part not in applicable:
        raise ValueError("repair_not_applicable")
    evidence = applicable[failed_part]
    order = [failed_part] + [p for p in proposed_order if p != failed_part]
    invocation = {
        "skill_id": SKILL_ID, "skill_version": 1, "code_sha256": module_hash(),
        "run_id": request["run_id"], "request_id": request["request_id"],
        "state_id": request["state_id"], "assembled_parts": sorted(request["assembled_parts"]),
        "failed_part_id": failed_part, "source_failure_event_id": evidence["failure_event_id"],
        "source_failure_call_id": evidence["failure_call_id"],
        "state_change_event_ids": evidence["state_change_event_ids"],
        "state_change_call_ids": evidence["state_change_call_ids"],
        "source_evidence": deepcopy(evidence),
        "proposed_order": list(proposed_order), "applied_order": order,
        "full_permutation_preserved": True, "state_mutations": False,
        "internal_verifier_calls": 0, "physical_result": "PENDING",
        "lifecycle": "provisional", "generalization_validated": False,
        "causal_benefit_demonstrated": False,
    }
    invocation["invocation_id"] = "repair:" + _digest(invocation)
    return {"candidate_ids": order, "invocation": invocation}


def feedback_from_events(events):
    """Join the existing worker's raw verifier events without editing old logs."""
    starts, feedback = {}, []
    for event in events:
        if event.get("kind") == "verifier_start":
            key = (event.get("run_id"), event.get("call_id"))
            if key in starts:
                raise ValueError("duplicate_verifier_start")
            starts[key] = event
        elif event.get("kind") == "verifier_result":
            start = starts.get((event.get("run_id"), event.get("call_id")))
            if start is None:
                raise ValueError("verifier_result_without_start")
            feedback.append({"event_id": event["event_id"], "call_id": event["call_id"],
                             "status": event["status"], "request": deepcopy(start["request"]),
                             "raw": deepcopy(event.get("raw")),
                             "observed_stage": event.get("observed_stage"),
                             "blocker": None, "path": None})
    return feedback


def collect_outcomes(invocations, events):
    """Bind invocations to worker-applied ranking and subsequent real results.

    PASS records an observed failure/change/retry success chain. It does not
    establish the intervention's causal benefit over native DFS/backtracking.
    """
    decisions = {event["request"]["request_id"]: event for event in events
                 if event.get("kind") == "rank_decision"}
    starts = {(event.get("run_id"), event.get("call_id")): event for event in events
              if event.get("kind") == "verifier_start"}
    results = {(event.get("run_id"), event.get("call_id")): event for event in events
               if event.get("kind") == "verifier_result"}
    feedback = feedback_from_events(events)
    outcomes = []
    for invocation in invocations:
        decision = decisions.get(invocation["request_id"])
        outcome = {"invocation_id": invocation["invocation_id"], "skill_id": SKILL_ID,
                   "request_id": invocation["request_id"], "status": "NOT_RUN",
                   "source_failure_event_id": invocation["source_failure_event_id"],
                   "state_change_event_ids": invocation["state_change_event_ids"],
                   "retry_event_id": None, "retry_call_id": None,
                   "recovery_chain_observed": False, "lifecycle": "provisional",
                   "generalization_validated": False, "causal_benefit_demonstrated": False}
        if decision is None:
            outcomes.append(outcome)
            continue
        request = decision["request"]
        if (request["run_id"] != invocation["run_id"] or request["state_id"] != invocation["state_id"]
                or decision["applied_order"] != invocation["applied_order"]
                or decision.get("fallback_reason") is not None):
            raise ValueError("worker_did_not_apply_repair")
        # Re-establish the provenance from the raw events, before this ranking.
        decision_no = int(decision["event_id"].rsplit(":e", 1)[1])
        preceding = [value for value in feedback
                     if value["event_id"].startswith(request["run_id"] + ":e")
                     and int(value["event_id"].rsplit(":e", 1)[1]) < decision_no]
        eligible = {value["part_id"]: value for value in _eligible(request, preceding)}
        evidence = eligible.get(invocation["failed_part_id"])
        if evidence is None or evidence != invocation["source_evidence"]:
            raise ValueError("repair_source_evidence_mismatch")
        for key, start in starts.items():
            attempt = start["request"]
            if (key[0] == invocation["run_id"] and start.get("generation_id") == decision["generation_id"]
                    and attempt["part_move"] == invocation["failed_part_id"]
                    and sorted([attempt["part_move"]] + attempt["parts_rest"]) == invocation["assembled_parts"]):
                result = results.get(key)
                if result is not None:
                    observed = _observation({"event_id": result["event_id"], "call_id": key[1],
                                             "status": result["status"], "request": attempt,
                                             "raw": result.get("raw")}, request)
                    if result["status"] in ("PASS", "UNKNOWN") and observed is None:
                        raise ValueError("invalid_retry_result")
                    outcome.update(status=result["status"], retry_event_id=result["event_id"],
                                   retry_call_id=key[1], raw_result=deepcopy(result.get("raw")),
                                   recovery_chain_observed=result["status"] == "PASS")
                break
        outcomes.append(outcome)
    return outcomes
