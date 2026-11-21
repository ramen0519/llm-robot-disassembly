"""Disposable, metered Python proposals for a reproducible pre-physics guard."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from orchestrator.python_skill_program import validate_program, _response, MAX_INPUT_BYTES
from orchestrator.python_skill_vm import VM, bounded
from orchestrator.pose_guard_recovery import validate_guard
from worker.explicit_action_contract import DIRECTIONS, SKILLS, digest, validate_request


def evaluate(program, request, guard, plan, index):
    program = validate_program(program)
    request = validate_request(request)
    if guard is not None:
        validate_guard(guard, plan, index, request)
    context = {k: deepcopy(request[k]) for k in ("candidate_ids", "remaining_parts",
        "base_part", "max_grippers", "pose_catalog", "geometry", "memory_references")}
    context.update(last=None, plan_guard=deepcopy(guard), directions=deepcopy(DIRECTIONS),
                   primitive_ids=list(SKILLS))
    bounded(context)
    count = [0]

    def eligible(choice):
        count[0] += 1
        if count[0] > 512:
            raise ValueError("python_eligibility_budget")
        try:
            _response(request, deepcopy(choice), digest(program))
            return True
        except (ValueError, TypeError, KeyError):
            return False

    vm = VM(program["source"], eligible)
    choice = vm.call("propose", [context])
    return dict(choice=choice, steps=vm.steps, eligibility_checks=count[0])


def step_guard_program(program, request, guard, plan, index, timeout_seconds=4):
    program = validate_program(program)
    request = validate_request(request)
    if guard is None:
        return dict(response=None, triggered=False, not_applicable_reason="no_plan_guard",
                    physical_calls=0, api_calls=0)
    validate_guard(guard, plan, index, request)
    payload = json.dumps(dict(program=program, request=request, guard=guard, plan=plan,
        index=index), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("python_guard_input_byte_limit")
    if not 0 < timeout_seconds <= 10:
        raise ValueError("invalid_python_guard_timeout")
    runner = Path(__file__).absolute().with_name("pose_guard_skill_child.py")
    environment = {k: os.environ[k] for k in ("SystemRoot", "WINDIR") if k in os.environ}
    result = dict(response=None, triggered=True, not_applicable_reason="python_abstained",
        guard_evidence_id=guard["evidence_id"], physical_calls=0, api_calls=0,
        program_sha256=digest(program))
    try:
        child = subprocess.run([sys.executable, "-I", "-S", "-B", str(runner)],
            input=payload, capture_output=True, cwd=str(runner.parent), env=environment,
            timeout=timeout_seconds, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if child.returncode != 0 or len(child.stdout) > 65536:
            raise ValueError("python_guard_child_failed")
        value = json.loads(child.stdout)
        if "error" in value:
            raise ValueError("python_guard_execution_rejected")
        result.update(vm_steps=value["steps"], eligibility_checks=value["eligibility_checks"])
        if value["choice"] is not None:
            if value["choice"].get("evidence_ids") != []:
                raise ValueError("guard_is_not_physical_evidence")
            response = _response(request, value["choice"], digest(program))
            response["metadata"].update(guard_evidence_id=guard["evidence_id"],
                guard_is_physical_failure=False, generated_program_executed=True,
                program_runtime="pose_guard_python_vm_v1")
            result.update(response=response, not_applicable_reason=None)
    except (subprocess.TimeoutExpired, ValueError, TypeError, KeyError, OSError):
        result["not_applicable_reason"] = "python_guard_execution_rejected"
    return result
