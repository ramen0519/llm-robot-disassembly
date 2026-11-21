"""Deterministic program scope and offline checks of structured behavior claims.

This module has no API, file I/O or physics authority. It checks what the closed
IR interpreter proposes on supplied, bound debug requests. Matching a recorded
PASS action is not a fresh PASS and does not establish a cause or generality.
Free-form model prose remains unverified; only the generated scope description
and the individually checked structured predictions have this contract.
"""
from copy import deepcopy

from orchestrator.skill_program import (
    MAX_RULES, MAX_SCAN_ACTIONS, SELECTORS, program_hash, step_program,
    validate_program,
)
from orchestrator.skill_refinement import MAX_FEEDBACK_CASES, _bound_case, _choice
from worker.explicit_action_contract import (
    DIRECTIONS, SKILLS, canonical, digest, validate_choice,
)


RELATIONS_TO_PASS = (
    "preserve_observed_pass", "change_observed_pass", "no_observed_pass",
)
CHOICE_FIELDS = (
    "part_id", "pose_id", "skill_id", "direction_id", "second_direction_id",
    "support_ids",
)
PREDICATE_DESCRIPTIONS = {
    "always": "No observation predicate is required.",
    "has_current_feedback": "A current-state, actual-backend, nonlate action observation is bound.",
    "last_path_passed": "The bound observation has path_success=true.",
    "last_path_failed": "The bound observation has path_success=false; the cause is not inferred.",
    "last_stability_failed": "The bound observation has stability_success=false.",
    "last_stability_passed": "The bound observation has stability_success=true.",
    "last_not_accepted": "The bound observation has accepted_for_transition=false.",
    "has_unstable_parts": "The bound dynamic-failure diagnostic has current remaining nonmover/nonbase parts.",
    "last_single_force": "The bound previous action uses single_force_v1.",
    "last_turn_force": "The bound previous action uses turn_force_two_leg_v1.",
    "has_multiple_candidates": "There is more than one current candidate.",
    "supports_available": "max_grippers > 1 and there are more than two remaining parts; this does not prove a feasible support.",
    "last_initial_disconnected": "The strictly bound initial-contact diagnostic reports a nonempty free-rest degree-zero set before any physics steps, with no dynamic-failure diagnostic.",
    "has_initial_disconnected_parts": "The same strictly bound initial-contact degree-zero set is nonempty; this is not observed motion or a proved cause.",
}
METRIC_DESCRIPTIONS = {
    "support_count": "number of selected support parts",
    "changed_support_count": "symmetric difference size from the bound previous support set",
    "part_volume": "current mover's finite nonnegative geometry volume",
    "part_extent_z": "current mover's finite nonnegative geometry extents[2]",
    "direction_up": "literal WORLD Z component of the first force direction; no clearance proof",
    "turning": "1 for turn_force_two_leg_v1, otherwise 0",
}


def _selector_description(field, value):
    if value == "any":
        return "No additional restriction within the current legal proposal domain."
    if field == "primitive":
        return "Use the registered primitive " + value + "."
    if field == "part" and value.startswith("observed_"):
        role = ("dynamic-failure" if value == "observed_unstable" else "initial-contact degree-zero")
        return "Select a mover from the current bound " + role + " set; do not copy source-task IDs."
    if field == "support" and value.startswith("contains_observed_"):
        role = ("dynamic-failure" if value == "contains_observed_unstable" else "initial-contact degree-zero")
        return "The support set contains at least one current bound " + role + " part, not necessarily all; it is distinct from selecting the mover."
    if field == "support" and value in ("empty", "nonempty"):
        return "Select " + ("zero" if value == "empty" else "one or more legal") + " support parts."
    if value == "none":
        return "The second direction is null."
    if field in ("direction", "second_direction") and value in DIRECTIONS:
        return "Use the literal WORLD axis " + value + "."
    target = ("pose matrix" if field == "pose" else "support set" if field == "support" else
              "first direction" if field == "direction" else "second direction" if field == "second_direction" else "mover ID")
    if value in ("same_as_last", "different_from_last"):
        return "Require " + target + (" equal to" if value == "same_as_last" else " different from") + " the bound previous action's corresponding value. Missing feedback abstains."
    return "Require this direction's dot product with the bound previous corresponding direction to be " + ("-1 (opposite)." if value == "opposite_last" else "0 (perpendicular).")


def program_description(program):
    """Render the executable scope from validated IR, never from model prose."""
    normalized = validate_program(program)
    execution = [
        "Stateless invocation: rules are checked in listed order and predicates within a rule are ANDed.",
        "Return ONE action from the first matching rule that has an admissible proposal. A matching rule with no proposal falls through; a scan-limit result abstains immediately.",
        "Later rules are not physical follow-up steps. There is no hidden rule cursor or implicit retry after an action fails.",
        "Actual attempted action IDs and deferred path-round variants are skipped. The planner's complete candidate set is preserved.",
        "With rank terms, compare them lexicographically in listed priority; asc minimizes, desc maximizes, and canonical(choice) breaks remaining ties. Missing metric values skip that proposal.",
        "With rank=[], choose the FIRST admissible action in canonical_filtered enumeration, not a geometry-ranked optimum or a globally minimal canonical(choice).",
        "Enumeration order: canonical candidate IDs; pose ID order; registered primitive order " + canonical(list(SKILLS)) + "; WORLD direction order " + canonical(list(DIRECTIONS)) + "; perpendicular second directions in that order; support size then lexicographic combinations (same_as_last uses its one set).",
        "At most " + str(MAX_SCAN_ACTIONS) + " proposal actions are scanned across rules. An incomplete ranked scan abstains.",
        "No simulator, physical PASS, candidate pruning, verdict reuse or registration follows from this description. Fresh ASAP verification remains required.",
    ]
    rules = []
    rendered = list(execution)
    for index, rule in enumerate(normalized["rules"]):
        selectors = [{"field": field, "value": rule["select"][field],
                      "meaning": _selector_description(field, rule["select"][field])}
                     for field in SELECTORS]
        predicates = [{"name": value, "meaning": PREDICATE_DESCRIPTIONS[value]}
                      for value in rule["when"]]
        ranking = [{"priority": priority, **deepcopy(term),
                    "meaning": METRIC_DESCRIPTIONS[term["metric"]]}
                   for priority, term in enumerate(rule["rank"])]
        rank_text = ("; ".join(term["metric"] + " " + term["order"] for term in rule["rank"])
                     if ranking else "empty rank: FIRST canonical_filtered admissible action")
        line = ("Rule " + str(index) + ": WHEN " + " AND ".join(rule["when"])
                + "; SELECT " + ", ".join(field + "=" + rule["select"][field] for field in SELECTORS)
                + "; RANK " + rank_text + ".")
        rules.append({"rule_index": index, "predicates": predicates,
                      "selectors": selectors, "ranking": ranking, "rendered_text": line})
        rendered.append(line)
    return {"schema_version": 1, "kind": "deterministic_program_description_v1",
            "program_sha256": program_hash(normalized), "execution_semantics": execution,
            "rules": rules, "rendered_text": "\n".join(rendered),
            "raw_model_prose_status": "unverified", "physics_validated": False,
            "physical_calls": 0, "candidate_pruning": False, "physics_verdict_reused": False}


def _semantic_context(request):
    """Remove provenance labels, retaining state, task, geometry and action guards.

    Different replay request/evidence labels do not change executable context.
    All remaining fields must match exactly before sharing a historical PASS.
    This is conditional reference matching, never a physical-result cache.
    """
    value = deepcopy(request)
    for field in ("run_id", "request_id", "evidence_ids"):
        value.pop(field, None)
    for item in value.get("path_stage_observations", []):
        item.pop("source_event_id", None)
    for item in value.get("episode_feedback", []):
        item.pop("evidence_id", None)
        if type(item.get("choice")) is dict:
            item["choice"].pop("evidence_ids", None)
        # A replay's evidence-file identity is provenance, not an endpoint
        # value. All measured endpoint values and other fields remain bound.
        if type(item.get("measured_endpoint")) is dict:
            item["measured_endpoint"].pop("evidence_sha256", None)
    return digest(value)


def _examples(feedback):
    if type(feedback) is not list or len(feedback) > MAX_FEEDBACK_CASES:
        raise ValueError("invalid_behavior_feedback_count")
    examples, by_request, passes = [], {}, {}
    for item in feedback:
        request, trace, trigger, recovery = _bound_case(item)
        identifier = request["request_id"]
        context = _semantic_context(request)
        if identifier in by_request:
            row = by_request[identifier]
            if canonical(row["request"]) != canonical(request):
                raise ValueError("conflicting_behavior_request_id")
            if item["case_id"] not in row["source_case_ids"]:
                row["source_case_ids"].append(item["case_id"])
        else:
            row = {"request_id": identifier, "state_id": request["state_id"],
                   "context_sha256": context, "source_case_ids": [item["case_id"]],
                   "request": request}
            by_request[identifier] = row
            examples.append(row)
        if recovery is not None and recovery["status"] == "PASS":
            passes.setdefault(context, {})[canonical(recovery["chosen"])] = recovery["chosen"]
    for row in examples:
        row["observed_pass_choices"] = [deepcopy(choice) for _, choice in sorted(
            passes.get(row["context_sha256"], {}).items())]
    return examples


def behavior_examples(feedback):
    """Unique debug prediction IDs and same-context historical PASS references.

    Full requests remain in the caller's bound input packet. Repeating a request
    ID with changed request data is an error; repeated observations of the exact
    request are combined. No answers for a newly generated program are supplied.
    """
    return [{key: deepcopy(value) for key, value in row.items() if key != "request"}
            for row in _examples(feedback)]


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def behavior_claim_schema(feedback):
    """Request-specific array schema; runtime audit enforces unique full coverage."""
    examples = _examples(feedback)
    alternatives = []
    for row in examples:
        request = row["request"]
        choice = _object({
            "part_id": {"type": "string", "enum": request["candidate_ids"]},
            "pose_id": {"type": "string", "enum": [p["pose_id"] for p in request["pose_catalog"]]},
            "skill_id": {"type": "string", "enum": list(SKILLS)},
            "direction_id": {"type": "string", "enum": list(DIRECTIONS)},
            "second_direction_id": {"type": ["string", "null"], "enum": list(DIRECTIONS) + [None]},
            "support_ids": {"type": "array", "items": {"type": "string", "enum": request["remaining_parts"]},
                            "maxItems": request["max_grippers"] - 1},
        })
        alternatives.append(_object({
            "request_id": {"type": "string", "enum": [row["request_id"]]},
            "predicted_rule_index": {"type": ["integer", "null"], "enum": list(range(MAX_RULES)) + [None]},
            "predicted_choice": {"anyOf": [choice, {"type": "null"}]},
            "relation_to_observed_pass": {"type": "string", "enum": list(RELATIONS_TO_PASS)},
        }))
    return {"type": "array", "minItems": len(examples), "maxItems": len(examples),
            "items": {"anyOf": alternatives} if alternatives else _object({})}


def _validate_claim(claim, request):
    fields = {"request_id", "predicted_rule_index", "predicted_choice", "relation_to_observed_pass"}
    if type(claim) is not dict or set(claim) != fields:
        raise ValueError("invalid_behavior_claim_fields")
    if type(claim["request_id"]) is not str or claim["request_id"] != request["request_id"]:
        raise ValueError("invalid_behavior_claim_request")
    index = claim["predicted_rule_index"]
    if index is not None and (type(index) is not int or not 0 <= index < MAX_RULES):
        raise ValueError("invalid_behavior_claim_rule_index")
    if (type(claim["relation_to_observed_pass"]) is not str
            or claim["relation_to_observed_pass"] not in RELATIONS_TO_PASS):
        raise ValueError("invalid_behavior_claim_pass_relation")
    value = claim["predicted_choice"]
    if value is not None:
        if type(value) is not dict or set(value) != set(CHOICE_FIELDS):
            raise ValueError("invalid_behavior_claim_choice_fields")
        checked = validate_choice(request, {**deepcopy(value), "evidence_ids": []})
        value = _choice(checked)
        if index is None:
            raise ValueError("behavior_choice_without_rule_index")
    return {**deepcopy(claim), "predicted_choice": value}


def audit_behavior_claims(program, full_debug_feedback, claims):
    """Compare every declared prediction against actual offline IR execution.

    A correctly declared change to an old PASS is allowed here and explicitly
    reported. Other development gates may reject unchanged failed behavior.
    Even a fully correct declaration remains physically unvalidated.
    """
    result = {"schema_version": 1, "kind": "offline_behavior_claim_audit_v1",
              "accepted": True, "accepted_for_fresh_validation": True,
              "reasons": [], "errors": [], "cases": [], "example_count": 0,
              "physical_calls": 0, "candidate_pruning": False,
              "physics_verdict_reused": False, "physics_validated": False,
              "registered": False, "failure_cause": None,
              "raw_model_prose_status": "unverified",
              "observed_pass_relation_semantics": "historical_action_identity_only_not_new_physics_pass"}
    try:
        candidate = validate_program(program)
        examples = _examples(full_debug_feedback)
        result["example_count"] = len(examples)
        if type(claims) is not list or len(claims) != len(examples):
            raise ValueError("behavior_claim_coverage_mismatch")
        lookup = {}
        requests = {row["request_id"]: row["request"] for row in examples}
        for claim in claims:
            if type(claim) is not dict or type(claim.get("request_id")) is not str:
                raise ValueError("invalid_behavior_claim_fields")
            identifier = claim["request_id"]
            if identifier not in requests:
                raise ValueError("unknown_behavior_claim_request_id")
            if identifier in lookup:
                raise ValueError("duplicate_behavior_claim_request_id")
            lookup[identifier] = _validate_claim(claim, requests[identifier])
        if set(lookup) != set(requests):
            raise ValueError("behavior_claim_coverage_mismatch")
    except (ValueError, TypeError, KeyError, IndexError) as error:
        result.update(accepted=False, accepted_for_fresh_validation=False,
                      reasons=["invalid_behavior_claim_input"], errors=[str(error)])
        return result
    result["program_sha256"] = program_hash(candidate)
    for example in examples:
        request, claim = example["request"], lookup[example["request_id"]]
        actual = step_program(candidate, request)
        choice = _choice(actual["response"]["choice"]) if actual["response"] else None
        passes = example["observed_pass_choices"]
        relation = ("no_observed_pass" if not passes else
                    "preserve_observed_pass" if choice in passes else "change_observed_pass")
        mismatches = []
        for name, expected, predicted in (
            ("rule_index", actual["rule_index"], claim["predicted_rule_index"]),
            ("choice", choice, claim["predicted_choice"]),
            ("observed_pass_relation", relation, claim["relation_to_observed_pass"]),
        ):
            if expected != predicted:
                mismatches.append(name)
        differences = [{"observed_pass_choice": deepcopy(previous),
                        "different_fields": [field for field in CHOICE_FIELDS
                                             if choice is None or choice[field] != previous[field]]}
                       for previous in passes]
        result["cases"].append({
            "request_id": example["request_id"], "state_id": example["state_id"],
            "context_sha256": example["context_sha256"],
            "source_case_ids": deepcopy(example["source_case_ids"]),
            "expected_rule_index": actual["rule_index"], "expected_choice": choice,
            "expected_relation_to_observed_pass": relation, "declared": claim,
            "matches": not mismatches, "mismatched_fields": mismatches,
            "previous_pass_differences": differences,
            "not_applicable_reason": actual["not_applicable_reason"],
            "failure_cause": None, "physics_validated": False,
        })
        if mismatches:
            result["accepted"] = False
            if "behavior_claim_mismatch" not in result["reasons"]:
                result["reasons"].append("behavior_claim_mismatch")
    result["accepted_for_fresh_validation"] = result["accepted"]
    return result
