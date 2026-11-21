"""Read-only diagnosis of offline skill revisions on bound debug observations.

These helpers neither call physics nor modify candidate/action availability.
A repeated development proposal can be rejected as an uninformative revision;
that is not a claim that its part/action is impossible. A changed proposal is
still unvalidated. Local provenance checks are not execution attestation.
"""
from copy import deepcopy

from orchestrator.skill_program import (
    _metric, _predicate, _scoped_last, program_hash, step_program, validate_program,
)
from worker.explicit_action_contract import (
    DIRECTIONS, SKILLS, action_id, canonical, validate_choice, validate_matrix,
    validate_request, validate_response,
)
from worker.path_probe_round import partition_path_round, semantic_path_key


MAX_FEEDBACK_CASES = 32
SEMANTICS = (
    "Each invocation returns ONE action from the first rule with matching "
    "predicates and a legal proposal. Later rules are not subsequent physical "
    "steps and are not executed after that action fails. The caller supplies "
    "fresh feedback on a later invocation; there is no hidden rule cursor. "
    "has_unstable_parts requires actual dynamic stability-failure diagnostics; "
    "stability is checked only after path_success=true. It cannot be true "
    "together with last_path_failed in the current interpreter."
)


def _choice(value):
    """Physical choice identity, without citation ordering or request IDs."""
    if type(value) is not dict:
        raise ValueError("invalid_observed_choice")
    keys = {"part_id", "pose_id", "skill_id", "direction_id",
            "second_direction_id", "support_ids", "evidence_ids"}
    if set(value) != keys or type(value["support_ids"]) is not list:
        raise ValueError("invalid_observed_choice")
    result = {key: deepcopy(item) for key, item in value.items() if key != "evidence_ids"}
    result["support_ids"] = sorted(result["support_ids"])
    return result


def _observation(raw, expected, label):
    if type(raw) is not dict or raw.get("backend") != "ASAP_RedMax":
        raise ValueError(label + "_requires_actual_backend")
    if raw.get("deadline_exceeded") is not False:
        raise ValueError(label + "_requires_explicit_nonlate_observation")
    if _choice(raw.get("chosen")) != _choice(expected):
        raise ValueError(label + "_chosen_action_mismatch")
    path, stability, status = (raw.get(k) for k in ("path_success", "stability_success", "status"))
    if (type(path) not in (bool, type(None))
            or type(stability) not in (bool, type(None))
            or status not in ("PASS", "UNKNOWN", "REJECTED", "ERROR")
            or stability is not None and path is not True
            or status == "PASS" and (path is not True or stability is not True)):
        raise ValueError(label + "_inconsistent_stage_result")
    failure = status in ("UNKNOWN", "REJECTED") and (
        path is False or path is True and stability is False)
    return {"status": status, "path_success": path, "stability_success": stability,
            "known_attempt_failure": failure, "deadline_exceeded": False,
            "backend": "ASAP_RedMax", "failure_cause": None,
            "all_paths_impossible": False, "chosen": _choice(expected)}


def _feedback_items(feedback):
    if type(feedback) is not list or not 1 <= len(feedback) <= MAX_FEEDBACK_CASES:
        raise ValueError("invalid_debug_feedback_count")
    return feedback


def _replay_previous(program, request, trace):
    previous = validate_program(program)
    replay = step_program(previous, request)
    expected = trace.get("response")
    if (replay.get("program_sha256") != trace.get("program_sha256")
            or replay["rule_index"] != trace.get("rule_index")
            or replay["triggered"] != trace.get("triggered")
            or (replay["response"] is None) != (expected is None)
            or replay["response"] and _choice(replay["response"]["choice"]) != _choice(expected["choice"])):
        raise ValueError("previous_program_trace_mismatch")
    return previous


def _bound_case(item):
    if type(item) is not dict:
        raise ValueError("invalid_debug_feedback_item")
    for key in ("phase", "validation_phase"):
        if key in item and item[key] != "debug":
            raise ValueError("non_debug_feedback_forbidden")
    if type(item.get("case_id")) is not str or not 0 < len(item["case_id"]) <= 512:
        raise ValueError("invalid_debug_case_id")
    request = validate_request(item.get("current_observation_request"))
    last = _scoped_last(request)
    if last is None:
        raise ValueError("debug_request_has_no_bound_current_feedback")
    trigger = _observation(item.get("trigger"), last["choice"], "trigger")
    for key in ("status", "path_success", "stability_success"):
        if trigger[key] != last["feedback"].get(key):
            raise ValueError("trigger_current_feedback_mismatch")
    if not trigger["known_attempt_failure"]:
        raise ValueError("debug_trigger_has_no_observed_failure")
    trace = item.get("interpreter_trace")
    if (type(trace) is not dict or trace.get("physical_calls") != 0
            or trace.get("candidate_pruning") is not False
            or trace.get("physics_verdict_reused") is not False
            or trace.get("scoped_feedback_evidence_id") != last["feedback"]["evidence_id"]
            or type(trace.get("triggered")) is not bool):
        raise ValueError("invalid_bound_interpreter_trace")
    response = trace.get("response")
    recovery = None
    if response is not None:
        response = validate_response(request, response)
        if (type(trace.get("rule_index")) is not int or trace["rule_index"] < 0
                or response["metadata"].get("generated_rule_index") != trace["rule_index"]
                or response["metadata"].get("program_sha256") != trace.get("program_sha256")):
            raise ValueError("interpreter_trace_identity_mismatch")
        recovery = _observation(item.get("recovery"), response["choice"], "recovery")
    elif item.get("recovery") is not None:
        raise ValueError("recovery_without_interpreter_action")
    elif trace["triggered"]:
        if (type(trace.get("rule_index")) is not int or trace["rule_index"] < 0
                or trace.get("not_applicable_reason") not in (
                    "no_matching_untried_action", "scan_limit")):
            raise ValueError("invalid_triggered_no_action_trace")
    elif trace.get("rule_index") is not None or trace.get("not_applicable_reason") != "no_trigger":
        raise ValueError("invalid_no_trigger_trace")
    if "observed_program" in item:
        _replay_previous(item["observed_program"], request, trace)
    return request, trace, trigger, recovery


def summarize_feedback(feedback, previous_program=None):
    """Compact observed debug outcomes and interpreter control-flow facts.

    Optional individual-rule probes only explain reachability in the recorded
    request. Their actions are hypothetical, never physical observations or a
    prescription. Admission/transfer results are rejected, never summarized.
    """
    result = {"schema_version": 1, "kind": "debug_revision_diagnosis",
              "valid": True, "cases": [], "errors": [], "semantics": SEMANTICS,
              "physical_calls": 0, "candidate_pruning": False,
              "failure_cause": None, "performance_effect": "NOT_RUN"}
    try:
        items = _feedback_items(feedback)
        previous = validate_program(previous_program) if previous_program is not None else None
    except (ValueError, TypeError, KeyError) as error:
        result.update(valid=False, errors=[str(error)])
        return result
    for index, item in enumerate(items):
        try:
            request, trace, trigger, recovery = _bound_case(item)
            row = {"case_id": item["case_id"], "state_id": request["state_id"],
                   "selected_rule_index": trace["rule_index"],
                   "previous_response_kind": "action" if recovery else "no_action",
                   "no_action_reason": trace.get("not_applicable_reason"),
                   "trigger": trigger, "recovery": recovery,
                   "later_available_but_not_reached": [], "rule_predicate_diagnostics": [],
                   "failure_cause": None}
            case_program = item.get("observed_program", previous)
            if case_program is not None:
                case_program = _replay_previous(case_program, request, trace)
                last = _scoped_last(request)
                for rule_index, rule in enumerate(case_program["rules"]):
                    names = set(rule["when"])
                    impossible = []
                    if {"has_unstable_parts", "last_path_failed"}.issubset(names):
                        impossible.append("unstable_diagnostics_require_path_passed")
                    for pair in (("last_path_failed", "last_path_passed"),
                                 ("last_stability_failed", "last_stability_passed"),
                                 ("last_single_force", "last_turn_force")):
                        if set(pair).issubset(names):
                            impossible.append("mutually_exclusive:" + "+".join(pair))
                    row["rule_predicate_diagnostics"].append({
                        "rule_index": rule_index,
                        "unsatisfied_predicates": [name for name in rule["when"]
                                                   if not _predicate(name, request, last)],
                        "interpreter_inconsistent_conjunctions": impossible,
                        "physical_failure_cause": None})
                if recovery:
                    for later in range(trace["rule_index"] + 1, len(case_program["rules"])):
                        one = {"language": case_program["language"], "rules": [case_program["rules"][later]]}
                        probe = step_program(one, request)
                        if probe["response"] is not None:
                            row["later_available_but_not_reached"].append({
                                "rule_index": later, "physical_execution": False,
                                "reason": "earlier_rule_already_returned_one_action"})
            result["cases"].append(row)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            result["valid"] = False
            result["errors"].append({"feedback_index": index, "error": str(error)})
    return result


def audit_program_novelty(program, prior_program_hashes):
    """Reject an identical normalized development procedure before physics.

    The caller supplies bound prior candidate hashes, not failed part IDs.
    This is a proposal-identity gate only: it never excludes runtime actions,
    invalidates a prior PASS, reuses a verdict, or proves behavioral novelty.
    Card names/descriptions are outside the executable program and its hash.
    A revision may preserve successful debug behavior while changing another
    condition. The separate ``audit_revision`` still checks failed behavior.
    """
    result = {"schema_version": 1, "kind": "offline_program_novelty_audit",
              "accepted": True, "accepted_for_fresh_validation": True,
              "reasons": [], "cases": [], "errors": [],
              "program_hash": None, "prior_program_count": 0,
              "exact_program_duplicate": False, "matched_prior_program_hash": None,
              "novelty_scope": "normalized_program_identity_only",
              "behavioral_novelty_proven": False, "physical_calls": 0,
              "candidate_pruning": False, "physics_verdict_reused": False,
              "physics_validated": False, "registered": False}
    try:
        if (type(prior_program_hashes) is not list
                or any(type(value) is not str or len(value) != 64
                       or any(c not in "0123456789abcdef" for c in value)
                       for value in prior_program_hashes)):
            raise ValueError("invalid_prior_program_hashes")
        if len(set(prior_program_hashes)) != len(prior_program_hashes):
            raise ValueError("duplicate_prior_program_hashes")
        identity = program_hash(program)
    except (ValueError, TypeError, KeyError) as error:
        result.update(accepted=False, accepted_for_fresh_validation=False,
                      reasons=["invalid_program_novelty_input"], errors=[str(error)])
        return result
    duplicate = identity in prior_program_hashes
    result.update(program_hash=identity, prior_program_count=len(prior_program_hashes),
                  exact_program_duplicate=duplicate,
                  matched_prior_program_hash=identity if duplicate else None,
                  accepted=not duplicate, accepted_for_fresh_validation=not duplicate,
                  reasons=["unchanged_program_no_new_procedure"] if duplicate else [])
    return result


def audit_revision(program, feedback):
    """Reject unchanged failed debug behavior before a fresh physical test.

    Changed actions receive only permission to continue validation, never PASS
    or registration. No-action revisions on an actual failed debug trigger are
    rejected before physics; unrelated predicates cannot masquerade as repair.
    This offline development gate does not alter runtime abstention semantics.
    """
    result = {"schema_version": 1, "kind": "offline_revision_behavior_audit",
              "accepted": True, "accepted_for_fresh_validation": True,
              "reasons": [], "cases": [], "errors": [],
              "physical_calls": 0, "candidate_pruning": False,
              "physics_verdict_reused": False, "physics_validated": False,
              "registered": False}
    try:
        candidate = validate_program(program)
        items = _feedback_items(feedback)
    except (ValueError, TypeError, KeyError) as error:
        result.update(accepted=False, accepted_for_fresh_validation=False,
                      reasons=["invalid_revision_input"], errors=[str(error)])
        return result
    for index, item in enumerate(items):
        try:
            request, previous, trigger, recovery = _bound_case(item)
            current = step_program(candidate, request)
            proposed = current["response"]
            reason = None
            extra_reasons = []
            if proposed is None:
                reason = "no_recovery_proposal"
                if recovery is None:
                    extra_reasons.append("unchanged_no_action")
            elif recovery is not None and recovery["known_attempt_failure"]:
                if _choice(proposed["choice"]) == recovery["chosen"]:
                    reason = "unchanged_failed_action"
            row = {"case_id": item["case_id"], "state_id": request["state_id"],
                   "previous_rule_index": previous["rule_index"],
                   "proposed_rule_index": current["rule_index"],
                   "previous_choice": recovery["chosen"] if recovery else None,
                   "proposed_choice": _choice(proposed["choice"]) if proposed else None,
                   "rejection_reason": reason, "additional_rejection_reasons": extra_reasons,
                   "physics_validated": False,
                   "candidate_pruning": False}
            result["cases"].append(row)
            if reason:
                result["accepted"] = False
                if reason not in result["reasons"]:
                    result["reasons"].append(reason)
                for extra in extra_reasons:
                    if extra not in result["reasons"]:
                        result["reasons"].append(extra)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            result["accepted"] = False
            if "invalid_debug_feedback" not in result["reasons"]:
                result["reasons"].append("invalid_debug_feedback")
            result["errors"].append({"feedback_index": index, "error": str(error)})
    result["accepted_for_fresh_validation"] = result["accepted"]
    return result


MOTION_SELECTORS = {"skill_id": "primitive", "direction_id": "direction",
                    "second_direction_id": "second_direction"}


def _source_motion(pair):
    if (type(pair) is not dict or pair.get("reference_only") is not True
            or pair.get("physics_reuse_allowed") is not False
            or pair.get("current_part_id_mapping") is not None
            or pair.get("same_state") is not True or pair.get("same_part") is not True
            or type(pair.get("success_evidence_id")) is not str):
        raise ValueError("invalid_source_observation_pair")
    outcome = pair.get("after_outcome")
    if type(outcome) is not dict or any(outcome.get(k) is not True for k in (
            "path_success", "stability_success", "accepted_for_transition")):
        raise ValueError("source_pair_missing_observed_accepted_pass")
    after = pair.get("after")
    if type(after) is not dict:
        raise ValueError("source_pair_missing_motion")
    validate_matrix(after.get("pose_matrix"))
    skill, direction, second = (after.get(k) for k in MOTION_SELECTORS)
    if skill not in SKILLS or direction not in DIRECTIONS:
        raise ValueError("source_pair_invalid_motion")
    if (skill == "single_force_v1" and second is not None
            or skill == "turn_force_two_leg_v1" and (
                second not in DIRECTIONS or sum(a*b for a, b in zip(
                    DIRECTIONS[direction], DIRECTIONS[second])) != 0)):
        raise ValueError("source_pair_invalid_secondary_motion")
    return {key: deepcopy(after[key]) for key in MOTION_SELECTORS}, after["pose_matrix"]


def _resolution_evidence(field, source_value, choice, request, rule, last):
    """Check a local equal-rank alternative, without recommending its execution."""
    if rule["select"][MOTION_SELECTORS[field]] != "any":
        return None
    alternative = deepcopy(choice)
    alternative[field] = source_value
    alternative["evidence_ids"] = []
    chosen = deepcopy(choice)
    chosen["evidence_ids"] = []
    try:
        validate_choice(request, alternative)
    except (ValueError, TypeError, KeyError):
        return None
    if action_id(request, alternative) in request["attempted_action_ids"]:
        return None
    deferred = partition_path_round(request)["deferred_path_keys"]
    if semantic_path_key(request, alternative) in deferred:
        return None
    values = [_metric(term["metric"], chosen, request, last) for term in rule["rank"]]
    other = [_metric(term["metric"], alternative, request, last) for term in rule["rank"]]
    if values != other or any(value is None for value in values):
        return None
    return {"field": field, "source_field_value_alternative_legal_and_untried": True,
            "equal_listed_rank_values": True,
            "resolution": "canonical_choice_ascending" if rule["rank"] else "first_matching_enumerated_action",
            "chosen_canonical_precedes_source_field_alternative":
                canonical(chosen) < canonical(alternative) if rule["rank"] else None,
            "physical_execution": False, "recommendation_generated": False}


def summarize_motion_alignment(program, feedback, pair_summary):
    """Compare emitted debug motion with source observations at equal pose matrices.

    This is a factual reference comparison, not a similarity score, admission
    gate or axis prescription. Numeric pose equality does not establish matching
    geometry, coordinate semantics, part correspondence or cross-task success.
    Only source pairs and bound DEBUG observations are accepted as inputs.
    """
    result = {"schema_version": 1, "kind": "source_motion_reference_alignment",
              "valid": True, "cases": [], "errors": [], "reference_only": True,
              "recommendation_generated": False, "admission_gate": False,
              "score": None, "physics_validated": False, "transfer_inferred": False,
              "source_part_ids_are_current_bindings": False,
              "physics_verdict_reused": False, "physical_calls": 0,
              "failure_cause": None, "pose_comparison": "numeric_matrix_equality_only"}
    try:
        policy = validate_program(program)
        items = _feedback_items(feedback)
        if (type(pair_summary) is not dict
                or pair_summary.get("kind") != "observed_same_state_failure_later_pass_pairs"
                or type(pair_summary.get("pairs")) is not list
                or len(pair_summary["pairs"]) > 16):
            raise ValueError("invalid_or_unbounded_source_pair_summary")
        pairs = [(pair, *_source_motion(pair)) for pair in pair_summary["pairs"]]
    except (ValueError, TypeError, KeyError) as error:
        result.update(valid=False, errors=[str(error)])
        return result
    for index, item in enumerate(items):
        try:
            request, _, _, _ = _bound_case(item)
            step = step_program(policy, request)
            response = step["response"]
            row = {"case_id": item["case_id"], "selected_rule_index": step["rule_index"],
                   "proposed_motion": None, "comparisons": [],
                   "unmatched_pose_pair_count": 0, "no_action_reason": step.get("not_applicable_reason")}
            if response is not None:
                choice = response["choice"]
                motion = {key: choice[key] for key in MOTION_SELECTORS}
                row["proposed_motion"] = motion
                pose = next(p["matrix"] for p in request["pose_catalog"] if p["pose_id"] == choice["pose_id"])
                rule = policy["rules"][step["rule_index"]]
                last = _scoped_last(request)
                for pair, observed, source_pose in pairs:
                    if pose != source_pose:
                        row["unmatched_pose_pair_count"] += 1
                        continue
                    changed = [field for field in MOTION_SELECTORS if motion[field] != observed[field]]
                    resolution = [_resolution_evidence(field, observed[field], choice, request, rule, last)
                                  for field in changed]
                    row["comparisons"].append({
                        "source_success_evidence_id": pair["success_evidence_id"],
                        "observed_source_motion": observed, "changed_motion_fields": changed,
                        "selected_rule_motion_selectors": {field: rule["select"][selector]
                                                           for field, selector in MOTION_SELECTORS.items()},
                        "unconstrained_changed_motion_fields": [field for field in changed
                            if rule["select"][MOTION_SELECTORS[field]] == "any"],
                        "deterministic_resolution_evidence": [value for value in resolution if value],
                        "reference_only": True, "source_part_id_mapping": None,
                        "physical_failure_cause": None, "transfer_inferred": False})
            result["cases"].append(row)
        except (ValueError, TypeError, KeyError, IndexError, StopIteration) as error:
            result["valid"] = False
            result["errors"].append({"feedback_index": index, "error": str(error)})
    return result
