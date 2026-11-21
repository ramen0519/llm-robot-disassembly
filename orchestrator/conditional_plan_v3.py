"""Per-step pose adjustment is selected explicitly by the model."""
from copy import deepcopy

from orchestrator.conditional_plan_v2 import compile_plan as compile_v2, schema_v2
from orchestrator.conditional_plan import STEP_KEYS, PLAN_KEYS
from orchestrator.plan_pose_binding import POLICIES
from worker.explicit_action_contract import digest

CONTRACT = "conditional_plan_v3"


def schema_v3(request, library_ids, max_steps=64):
    result = schema_v2(request, library_ids, max_steps)
    step = result["properties"]["steps"]["items"]
    step["properties"]["pose_policy"] = {"anyOf": [
        {"type": "string", "enum": list(POLICIES)}, {"type": "null"}]}
    step["required"].append("pose_policy")
    return result


def compile_plan(value, request, library_ids, max_steps=64, diagnostics=None):
    info = diagnostics if diagnostics is not None else {}
    info.update(contract=CONTRACT, phase="pose_policy", step_index=None)
    if type(value) is not dict or set(value) != PLAN_KEYS or type(value.get("steps")) is not list:
        raise ValueError("invalid_v3_plan_fields")
    clean = deepcopy(value)
    policies = []
    for index, step in enumerate(clean["steps"]):
        info["step_index"] = index
        if type(step) is not dict or set(step) != (STEP_KEYS - {"expected_remaining"}) | {"pose_policy"}:
            raise ValueError("invalid_v3_step_fields")
        policy = step.pop("pose_policy")
        if ((step["mode"] == "action" and policy not in POLICIES)
                or (step["mode"] == "registered_skill" and policy is not None)):
            raise ValueError("invalid_step_pose_policy")
        policies.append(policy)
    v2_info = {}
    try:
        bound = compile_v2(clean, request, library_ids, max_steps, v2_info)
    finally:
        info.update(base_compiler=v2_info)
    bound.update(plan_contract=CONTRACT, pose_policy_by_step=policies,
        plan_id="plan:" + digest(dict(contract=CONTRACT, value=value)),
        pose_rule="model_selected_per_step_exact_or_bounded_current_catalog_world_axes")
    info.update(phase="validated", steps_compiled=len(policies), wire_value_sha256=digest(value))
    return bound
