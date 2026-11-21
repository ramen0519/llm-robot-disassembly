"""Bounded, stateless conditional recovery programs over current legal actions.

This interpreter executes data, not Python. It has no simulator, I/O, persistent
state, task-ID literals, or authority to accept a removal. Its output still goes
through both existing action guards and the actual ASAP verifier. A unit fixture
using this language is not evidence that a model discovered a useful skill.
"""
from copy import deepcopy
from itertools import combinations
import math

from worker.explicit_action_contract import (
    DIRECTIONS, FALLBACK_POLICIES, PROTOCOL, SKILLS, action_id, canonical, digest,
    validate_choice, validate_request, validate_response,
)
from worker.path_probe_round import (
    PATH_SELECTION_POLICIES, partition_path_round, semantic_path_key,
    validate_path_round_response,
)
from worker.stability_diagnostics import project_stability_diagnostics


LANGUAGE = "recovery_rules_v1"
MAX_RULES = 8
MAX_PREDICATES = 8
MAX_RANK_TERMS = 4
MAX_PROGRAM_BYTES = 16384
MAX_SCAN_ACTIONS = 50000
PREDICATES = (
    "always", "has_current_feedback", "last_path_passed", "last_path_failed",
    "last_stability_failed", "last_stability_passed", "last_not_accepted",
    "has_unstable_parts", "last_single_force", "last_turn_force",
    "has_multiple_candidates", "supports_available",
    "last_initial_disconnected", "has_initial_disconnected_parts",
)
RELATIONS = ("any", "same_as_last", "different_from_last")
DIRECTION_SELECTORS = RELATIONS + ("opposite_last", "perpendicular_last") + tuple(DIRECTIONS)
SELECTORS = {
    "primitive": ("any",) + SKILLS,
    "part": RELATIONS + ("observed_unstable", "observed_initial_disconnected"),
    "pose": RELATIONS,
    "direction": DIRECTION_SELECTORS,
    "second_direction": DIRECTION_SELECTORS + ("none",),
    "support": ("any", "empty", "nonempty", "same_as_last", "different_from_last",
                "contains_observed_unstable", "contains_observed_initial_disconnected"),
}
METRICS = ("support_count", "changed_support_count", "part_volume",
           "part_extent_z", "direction_up", "turning")


def _object(properties):
    return dict(type="object", properties=properties, required=list(properties),
                additionalProperties=False)


def program_schema():
    """Return a strict Responses-compatible JSON Schema, without its wrapper."""
    return _object({
        "language": {"type": "string", "enum": [LANGUAGE]},
        "rules": {"type": "array", "minItems": 1, "maxItems": MAX_RULES,
            "items": _object({
                "when": {"type": "array", "minItems": 1, "maxItems": MAX_PREDICATES,
                         "items": {"type": "string", "enum": list(PREDICATES)}},
                "select": _object({key: {"type": "string", "enum": list(values)}
                                   for key, values in SELECTORS.items()}),
                "rank": {"type": "array", "maxItems": MAX_RANK_TERMS,
                    "items": _object({"metric": {"type": "string", "enum": list(METRICS)},
                                      "order": {"type": "string", "enum": ["asc", "desc"]}})},
            })},
    })


def validate_program(program):
    """Validate closed vocabulary/size and return normalized independent data.

    There are no string-valued expressions or literal IDs. Rule order and rank
    priority are semantic; conjunction order is not. Unknown fields, including
    loops, imported code, target IDs and claimed physical verdicts, are errors.
    """
    if type(program) is dict and program.get("language") == "python_policy_v1":
        from orchestrator.python_skill_program import validate_program as validate_python
        return validate_python(program)
    if type(program) is not dict or set(program) != {"language", "rules"}:
        raise ValueError("invalid_program_fields")
    if program["language"] != LANGUAGE or type(program["language"]) is not str:
        raise ValueError("unsupported_program_language")
    rules = program["rules"]
    if type(rules) is not list or not 1 <= len(rules) <= MAX_RULES:
        raise ValueError("invalid_program_rule_count")
    normalized = []
    for rule in rules:
        if type(rule) is not dict or set(rule) != {"when", "select", "rank"}:
            raise ValueError("invalid_program_rule_fields")
        when = rule["when"]
        if (type(when) is not list or not 1 <= len(when) <= MAX_PREDICATES
                or any(type(value) is not str or value not in PREDICATES for value in when)
                or len(when) != len(set(when))):
            raise ValueError("invalid_program_predicate")
        if "always" in when and len(when) != 1:
            raise ValueError("redundant_always_predicate")
        select = rule["select"]
        if (type(select) is not dict or set(select) != set(SELECTORS)
                or any(type(select[key]) is not str or select[key] not in values
                       for key, values in SELECTORS.items())):
            raise ValueError("invalid_program_selector")
        rank = rule["rank"]
        if type(rank) is not list or len(rank) > MAX_RANK_TERMS:
            raise ValueError("invalid_program_rank_count")
        metrics = []
        for term in rank:
            if (type(term) is not dict or set(term) != {"metric", "order"}
                    or type(term["metric"]) is not str or term["metric"] not in METRICS
                    or type(term["order"]) is not str or term["order"] not in ("asc", "desc")):
                raise ValueError("invalid_program_rank_term")
            metrics.append(term["metric"])
        if len(metrics) != len(set(metrics)):
            raise ValueError("duplicate_program_rank_metric")
        normalized.append(dict(when=sorted(when), select=deepcopy(select), rank=deepcopy(rank)))
    result = dict(language=LANGUAGE, rules=normalized)
    if len(canonical(result).encode("utf-8")) > MAX_PROGRAM_BYTES:
        raise ValueError("program_too_large")
    return result


def program_hash(program):
    return digest(validate_program(program))


def _scoped_last(request):
    """Bind only the current state's latest actual nonlate attempted action.

    A source Memory ID is never a local binding. A recent feedback item from a
    different DFS state/pose/configuration is not a current observation either.
    This validates provenance links, not cryptographic trust in caller data.
    """
    attempts = request.get("attempted_actions", [])
    observations = request.get("path_stage_observations", [])
    if not attempts or not observations:
        return None
    attempt, observation = attempts[-1], observations[-1]
    identifier = attempt["action_id"]
    matches = [item for item in request["episode_feedback"] if item.get("action_id") == identifier]
    if len(matches) != 1:
        return None
    item = matches[0]
    if (observation["backend"] != "ASAP_RedMax" or observation["deadline_exceeded"]
            or observation["status"] not in ("PASS", "UNKNOWN", "REJECTED")
            or item.get("deadline_exceeded") is not False
            or item.get("status") != observation["status"]
            or item.get("path_success") is not observation["path_success"]
            or item.get("evidence_id") != observation["source_event_id"]
            or type(item.get("remaining_parts")) is not list
            or any(type(part) is not str for part in item["remaining_parts"])
            or sorted(item["remaining_parts"]) != request["remaining_parts"]):
        return None
    choice = deepcopy(item.get("choice"))
    if type(choice) is not dict:
        return None
    choice["evidence_ids"] = []
    try:
        validate_choice(request, choice)
        if action_id(request, choice) != identifier:
            return None
    except (ValueError, TypeError):
        return None
    pose = next(p for p in request["pose_catalog"] if p["pose_id"] == choice["pose_id"])
    if item.get("selected_pose_matrix") != pose["matrix"]:
        return None
    for key in ("path_success", "stability_success", "accepted_for_transition"):
        if item.get(key) is not None and type(item[key]) is not bool:
            return None
    if (item.get("stability_success") is not None and item.get("path_success") is not True
            or item.get("status") == "PASS" and (
                item.get("path_success") is not True or item.get("stability_success") is not True)):
        return None
    if item.get("accepted_for_transition") is True and (
            item.get("status") != "PASS" or item.get("path_success") is not True
            or item.get("stability_success") is not True):
        return None
    unstable = set()
    try:
        diagnostic = project_stability_diagnostics(item.get("stability_diagnostics"))
    except (ValueError, TypeError, KeyError):
        diagnostic = None
    if (diagnostic is not None and diagnostic["phase"] == "dynamic_failure"
            and item.get("status") in ("UNKNOWN", "REJECTED")
            and item.get("path_success") is True and item.get("stability_success") is False
            and item.get("accepted_for_transition") is False
            and diagnostic["check_success_returned"] is True
            and diagnostic["returned_success"] is False
            and ("raw_stability_parts" not in item
                 or item["raw_stability_parts"] == diagnostic["returned_parts"])
            and not (item.get("stability_diagnostics") or {}).get("instrumentation_errors")):
        observed = {row["part_id"] for row in diagnostic["dynamic_failure_parts"]}
        free_rest = (set(request["remaining_parts"]) - {choice["part_id"], request["base_part"]}
                     - set(choice["support_ids"]))
        # Do not salvage an inconsistent diagnosis by silently deleting IDs.
        # Held bodies are not free bodies in this exact checker invocation.
        if observed and observed.issubset(free_rest):
            unstable = observed
    # Initial degree-zero contact is a separate upstream branch, not observed
    # movement or a root cause. Reject the whole new role if any ID could not
    # have belonged to this exact checker's free remaining bodies.
    initial_disconnected = set()
    if (diagnostic is not None and diagnostic["phase"] == "initial_disconnected"
            and item.get("status") in ("UNKNOWN", "REJECTED")
            and item.get("path_success") is True and item.get("stability_success") is False
            and item.get("accepted_for_transition") is False
            and diagnostic["check_success_returned"] is True
            and diagnostic["returned_success"] is False
            and diagnostic["physics_steps"] == 0
            and diagnostic["returned_parts_semantics"] == "initial_contact_graph_degree_zero"
            and not diagnostic["dynamic_failure_parts"]
            and ("raw_stability_parts" not in item
                 or item["raw_stability_parts"] == diagnostic["returned_parts"])
            and not (item.get("stability_diagnostics") or {}).get("instrumentation_errors")):
        observed = set(diagnostic["initial_disconnected_parts"])
        free_rest = (set(request["remaining_parts"]) - {choice["part_id"], request["base_part"]}
                     - set(choice["support_ids"]))
        if observed and observed.issubset(free_rest):
            initial_disconnected = observed
    return dict(choice=choice, feedback=item, unstable=unstable,
                initial_disconnected=initial_disconnected, pose_matrix=pose["matrix"])


def _predicate(name, request, last):
    if name == "always":
        return True
    if name == "has_multiple_candidates":
        return len(request["candidate_ids"]) > 1
    if name == "supports_available":
        return request["max_grippers"] > 1 and len(request["remaining_parts"]) > 2
    if last is None:
        return False
    feedback, choice = last["feedback"], last["choice"]
    return {
        "has_current_feedback": True,
        "last_path_passed": feedback.get("path_success") is True,
        "last_path_failed": feedback.get("path_success") is False,
        "last_stability_failed": feedback.get("stability_success") is False,
        "last_stability_passed": feedback.get("stability_success") is True,
        "last_not_accepted": feedback.get("accepted_for_transition") is False,
        "has_unstable_parts": bool(last["unstable"]),
        "last_initial_disconnected": bool(last["initial_disconnected"]),
        "has_initial_disconnected_parts": bool(last["initial_disconnected"]),
        "last_single_force": choice["skill_id"] == "single_force_v1",
        "last_turn_force": choice["skill_id"] == "turn_force_two_leg_v1",
    }[name]


def _relation(selector, value, previous, available):
    if selector == "any":
        return True
    if not available:
        return False
    return (value == previous) if selector == "same_as_last" else (value != previous)


def _direction(selector, value, previous, available):
    if selector == "none":
        return value is None
    if selector in DIRECTIONS:
        return value == selector
    if selector in RELATIONS:
        return _relation(selector, value, previous, available)
    if not available or value not in DIRECTIONS or previous not in DIRECTIONS:
        return False
    dot = sum(a * b for a, b in zip(DIRECTIONS[value], DIRECTIONS[previous]))
    return dot == (-1 if selector == "opposite_last" else 0)


def _choices(request, select, last):
    """Finite lazy legal Cartesian product, with early relational restriction.

    No failed part is removed from the planner. These are only this program's
    matching proposals; absence of a match means not_applicable to the caller.
    """
    previous = last["choice"] if last is not None else {}
    part_role = {"observed_unstable": "unstable",
                 "observed_initial_disconnected": "initial_disconnected"}.get(select["part"])
    support_role = {"contains_observed_unstable": "unstable",
                    "contains_observed_initial_disconnected": "initial_disconnected"}.get(select["support"])
    if part_role is not None and (last is None or not last[part_role]):
        return
    if support_role is not None and (last is None or not last[support_role]):
        return
    if select["support"] in ("same_as_last", "different_from_last") and last is None:
        return
    # This chooses a mover from a NEW current-state measured role.
    # It is distinct from holding an unstable part as a support. It restricts
    # this program's proposal only; response candidate_order remains complete.
    parts = ([part for part in request['candidate_ids'] if part in last[part_role]]
             if part_role is not None else
             [part for part in request["candidate_ids"] if _relation(
                 select["part"], part, previous.get("part_id"), last is not None)])
    poses = [pose for pose in sorted(request["pose_catalog"], key=lambda p: p["pose_id"])
             if _relation(select["pose"], pose["matrix"],
                          last["pose_matrix"] if last else None, last is not None)]
    for part in parts:
        possible_supports = sorted(set(request["remaining_parts"]) - {part, request["base_part"]})
        if (support_role is not None
                and not set(possible_supports).intersection(last[support_role])):
            continue
        max_supports = min(len(possible_supports), request["max_grippers"] - 1)
        for pose in poses:
            for primitive in SKILLS:
                if select["primitive"] not in ("any", primitive):
                    continue
                for direction in DIRECTIONS:
                    if not _direction(select["direction"], direction,
                                      previous.get("direction_id"), last is not None):
                        continue
                    seconds = [None] if primitive == "single_force_v1" else [
                        second for second in DIRECTIONS if sum(a*b for a, b in zip(
                            DIRECTIONS[direction], DIRECTIONS[second])) == 0]
                    for second in seconds:
                        if not _direction(select["second_direction"], second,
                                          previous.get("second_direction_id"), last is not None):
                            continue
                        if select["support"] == "same_as_last":
                            supports_iter = [tuple(sorted(previous["support_ids"]))] if last else []
                        else:
                            sizes = ([0] if select["support"] == "empty" else
                                     range(1, max_supports+1) if select["support"] == "nonempty" else
                                     range(max_supports+1))
                            supports_iter = (support for size in sizes
                                             for support in combinations(possible_supports, size))
                        for supports in supports_iter:
                            support_set = set(supports)
                            if len(supports) > max_supports or not support_set.issubset(possible_supports):
                                continue
                            if select["support"] == "different_from_last" and (
                                    last is None or support_set == set(previous["support_ids"])):
                                continue
                            # "contains" means at least one observed body,
                            # not all of them and never a stability guarantee.
                            if support_role is not None and not support_set.intersection(last[support_role]):
                                continue
                            yield dict(part_id=part, pose_id=pose["pose_id"], skill_id=primitive,
                                       direction_id=direction, second_direction_id=second,
                                       support_ids=list(supports), evidence_ids=[])


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _metric(name, choice, request, last):
    if name == "support_count":
        return len(choice["support_ids"])
    if name == "changed_support_count":
        return None if last is None else len(set(choice["support_ids"]) ^ set(last["choice"]["support_ids"]))
    if name == "direction_up":
        # Literal world-axis force component, not a clearance or gravity proof.
        return DIRECTIONS[choice["direction_id"]][2]
    if name == "turning":
        return int(choice["skill_id"] == "turn_force_two_leg_v1")
    parts = request["geometry"].get("parts")
    data = parts.get(choice["part_id"]) if type(parts) is dict else None
    if type(data) is not dict:
        return None
    if name == "part_volume":
        value = _number(data.get("volume"))
        return value if value is not None and value >= 0 else None
    extents = data.get("extents")
    if type(extents) is not list or len(extents) != 3:
        return None
    value = _number(extents[2])
    return value if value is not None and value >= 0 else None


def step_program(program, request, *, fallback_policy="path_stage_v1",
                 path_policy="path_probe_round_v1", max_actions=MAX_SCAN_ACTIONS):
    """Propose one guarded action or explicitly abstain, without running physics.

    Rules are tried in order; their predicates are conjunctions. Rank terms are
    lexicographic, with current canonical action identity as final tie-break.
    Missing observations never become zero/false facts. An incomplete ranked
    scan abstains; its prefix is not represented as an exhaustive best choice.

    ``fallback_policy`` is checked for caller configuration consistency only;
    this function never executes fallback. The service owns that decision.
    ``scanned`` counts candidate actions examined, including duplicates/deferred
    actions. Interpreter invocations and rejected proposals are not physics.
    """
    if type(program) is dict and program.get("language") == "python_policy_v1":
        if fallback_policy != "path_stage_v1" or path_policy != "path_probe_round_v1":
            raise ValueError("python_policy_requires_current_path_guards")
        from orchestrator.python_skill_program import step_program as step_python
        return step_python(program, request, max_actions=max_actions)
    program = validate_program(program)
    request = validate_request(request)
    if type(fallback_policy) is not str or fallback_policy not in FALLBACK_POLICIES:
        raise ValueError("unknown_fallback_policy")
    if type(path_policy) is not str or path_policy not in PATH_SELECTION_POLICIES:
        raise ValueError("unknown_path_policy")
    if type(max_actions) is not int or not 1 <= max_actions <= MAX_SCAN_ACTIONS:
        raise ValueError("invalid_program_scan_limit")
    partition = partition_path_round(request) if path_policy == "path_probe_round_v1" else None
    deferred = set(partition["deferred_path_keys"]) if partition else set()
    attempted = set(request["attempted_action_ids"])
    last = _scoped_last(request)
    sha = digest(program)
    result = dict(response=None, triggered=False, rule_index=None, scanned=0,
                  program_sha256=sha, not_applicable_reason="no_trigger",
                  decision_origin="generated_program", physical_calls=0,
                  candidate_pruning=False, physics_verdict_reused=False,
                  scoped_feedback_evidence_id=last["feedback"]["evidence_id"] if last else None)
    for index, rule in enumerate(program["rules"]):
        if not all(_predicate(name, request, last) for name in rule["when"]):
            continue
        result.update(triggered=True, rule_index=index, not_applicable_reason="no_matching_untried_action")
        best, best_rank, seen = None, None, set()
        for choice in _choices(request, rule["select"], last):
            if result["scanned"] >= max_actions:
                result["not_applicable_reason"] = "scan_limit"
                return result
            result["scanned"] += 1
            identifier = action_id(request, choice)
            if identifier in attempted or identifier in seen:
                continue
            seen.add(identifier)
            if deferred and semantic_path_key(request, choice) in deferred:
                continue
            values = [_metric(term["metric"], choice, request, last) for term in rule["rank"]]
            if any(value is None for value in values):
                continue
            rank = tuple(value if term["order"] == "asc" else -value
                         for value, term in zip(values, rule["rank"])) + (canonical(choice),)
            if best_rank is None or rank < best_rank:
                best, best_rank = choice, rank
            if not rule["rank"]:
                # Explicit first matching action; no optimum is asserted.
                break
        if best is None:
            continue
        if last is not None:
            best["evidence_ids"] = [last["feedback"]["evidence_id"]]
        response = dict(schema_version=1, protocol=PROTOCOL,
            request_id=request["request_id"], state_id=request["state_id"],
            candidate_order=[best["part_id"]] + [part for part in request["candidate_ids"]
                                               if part != best["part_id"]],
            choice=best, action_id=action_id(request, best), fallback_reason=None,
            metadata=dict(selection_source="generated_program", decision_origin="generated_program",
                program_sha256=sha, program_language=LANGUAGE, generated_rule_index=index,
                llm_live=False, attempted=False, exhausted=False, candidate_pruning=False,
                heuristic_order_used=False, physics_bypass=False, physics_verdict_reused=False,
                path_selection_policy=path_policy, fallback_policy=fallback_policy,
                enumeration_policy="canonical_filtered", generated_scan_count=result["scanned"]))
        response = validate_response(request, response)
        if partition:
            response = validate_path_round_response(request, response)
            response["metadata"]["path_probe_round"] = deepcopy(partition["round_audit"])
        result.update(response=response, not_applicable_reason=None)
        return result
    return result
