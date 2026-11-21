"""LLM chooses actions; code derives conditional remaining sets, never DFS state."""
from copy import deepcopy

from orchestrator.conditional_plan import PLAN_KEYS, STEP_KEYS, schema, validate_plan
from worker.explicit_action_contract import digest, validate_request

CONTRACT = "conditional_plan_v2"


def schema_v2(request, library_ids, max_steps=64):
    result = schema(request, library_ids, max_steps)
    step = result["properties"]["steps"]["items"]
    del step["properties"]["expected_remaining"]
    step["required"].remove("expected_remaining")
    return result


def compile_plan(value, request, library_ids, max_steps=64, diagnostics=None):
    """Reject invalid choices; add only mechanically derived state guards.

    This is not a repair of a wrong v1 answer. A v2 answer must omit remaining
    sets entirely. Future sets are conditional on prior actual verifier PASS.
    The original execution/feedback guards still own advancement and backtracking.
    """
    request = validate_request(request)
    info = diagnostics if diagnostics is not None else {}
    info.update(contract=CONTRACT, wire_value_sha256=digest(value),
                raw_response_recorded=False, phase="identity", step_index=None,
                expected_remaining_source="current_state_minus_preceding_proposed_movers")
    if (type(value) is not dict or set(value) != PLAN_KEYS
            or value["request_id"] != request["request_id"] or value["state_id"] != request["state_id"]):
        raise ValueError("stale_or_invalid_plan_identity")
    steps = value["steps"]
    info["phase"] = "length"
    if type(steps) is not list or not 1 <= len(steps) <= min(max_steps, len(request["remaining_parts"])-1):
        raise ValueError("invalid_plan_length")
    remaining = set(request["remaining_parts"])
    compiled = deepcopy(value)
    compiled["steps"] = []
    bound = None
    for index, step in enumerate(steps):
        info.update(phase="step", step_index=index, expected_remaining=sorted(remaining))
        if type(step) is not dict or set(step) != STEP_KEYS - {"expected_remaining"}:
            raise ValueError("invalid_v2_step_fields")
        # Diagnostic strings are drawn from known IDs, never arbitrary model text.
        part = step["part_id"]
        info["part_id"] = part if type(part) is str and part in request["remaining_parts"] else None
        support = step["support_ids"]
        info["known_support_ids"] = ([x for x in support if type(x) is str and x in request["remaining_parts"]]
                                     if type(support) is list else None)
        if type(part) is not str or part not in remaining - {request["base_part"]}:
            raise ValueError("invalid_or_repeated_planned_mover")
        compiled["steps"].append(dict(deepcopy(step), expected_remaining=sorted(remaining)))
        # Prefix validation identifies the rejected step without relaxing any
        # existing candidate, support, pose, evidence, or path-round contract.
        bound = validate_plan(compiled, request, library_ids, max_steps)
        remaining.remove(part)
    info.update(phase="validated", steps_compiled=len(steps))
    return bound
