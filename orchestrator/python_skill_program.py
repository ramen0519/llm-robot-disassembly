"""Metered Python recovery policies proposing actions to the existing DFS host.

This is interpreted Python syntax, not arbitrary native Python execution.
The child has no LLM key/environment inheritance. It never imports user code.
The evaluator's data-only capability boundary is primary; a subprocess alone
is not claimed to be an OS security sandbox.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from orchestrator.python_skill_vm import LANGUAGE, VM, parse, bounded, SYNTAX_DESCRIPTION
from worker.explicit_action_contract import (
    PROTOCOL, SKILLS, DIRECTIONS, action_id, digest, validate_request, validate_response,
)
from worker.path_probe_round import partition_path_round, validate_path_round_response

MAX_INPUT_BYTES = 512000
RUNTIME_VERSION = "python_data_vm_v2"
API_DESCRIPTION = """Python subset python_policy_v1:
Define propose(ctx) and optional helper functions. Return one choice dict or None.
Allowed: local variables, arithmetic (+,-,*,/,//,numeric %), for, if, break,
continue, return, comparisons, list/dict literals, indexing, slices, .get(),
.append(), len/abs/min/max/sum/round/sorted/range/int/float/bool/list/dict.
No imports, classes, while, eval/exec, file/network, attributes, global state,
exceptions, recursion beyond 16 calls, or external libraries.
The module contains ONLY function definitions, without annotations/default args.
Assignments and for-loop targets must be a SINGLE local name (no tuple unpacking,
no dict/list element assignment). Construct new dict literals instead of modifying
keys. No sets, f-strings, **,
enumerate/zip, .items/.keys/.values/.copy. Iterate a dict directly for its keys.
At most 50000 VM steps; bounded source/values; disposable host-metered process.
eligible(choice) checks current action/duplicate/path-round contracts ONLY;
it does NOT run physics, certify a path, or provide a physical verdict.
ctx keys: candidate_ids (complete), remaining_parts, base_part, max_grippers,
pose_catalog (actual selectable pose matrices), geometry, memory_references
(source-task evidence only, no automatic ID mapping), last (bound current
feedback or None), directions (world force axes), primitive_ids.
last contains choice, feedback, unstable IDs, initial_disconnected IDs,
pose_matrix. Do not infer a blocker/collision when feedback lacks it.
last["feedback"] is a dict with evidence_id (string), path_success
(True/False/None), stability_success (True/False/None), status and
accepted_for_transition. There is no ctx["evidence_ids"]: cite
last["feedback"]["evidence_id"] for the current failure.
last["choice"] has the same exact Choice keys listed below.
ctx["directions"] is a dict from xp/xn/yp/yn/zp/zn to 3-number world vectors.
pose_catalog is a list of {"pose_id": string, "matrix": 4x4 nested number list}.
last["pose_matrix"] is the corresponding CAD-to-world rigid transform.
ctx["geometry"] is a dict, ctx["memory_references"] is a list.
For recovery admission, abstain if the applicable current failure is absent.
Choice keys exactly: part_id, pose_id, skill_id, direction_id,
second_direction_id, support_ids, evidence_ids. IDs must come from CURRENT ctx.
Only registered single_force_v1 and turn_force_two_leg_v1 are executable.
Cite supplied evidence; never copy source-task part IDs into a new target.
Host calls propose anew after each physical feedback/state change. It owns
DFS/backtracking, time/attempt budgets, acceptance and every physical check.
No generated code can remove parts, set PASS, or update a persistent library.
""" + SYNTAX_DESCRIPTION


def validate_program(program):
    if (type(program) is not dict or set(program) != {"language", "source"}
            or program["language"] != LANGUAGE):
        raise ValueError("invalid_python_program_fields")
    parse(program["source"])
    return deepcopy(program)


def _response(request, choice, program_hash):
    response = dict(schema_version=1, protocol=PROTOCOL,
        request_id=request["request_id"], state_id=request["state_id"],
        candidate_order=[choice["part_id"]] + [p for p in request["candidate_ids"] if p != choice["part_id"]],
        choice=choice, action_id=action_id(request, choice), fallback_reason=None,
        metadata=dict(selection_source="generated_program", decision_origin="generated_program",
            program_sha256=program_hash, program_language=LANGUAGE,
            python_runtime=RUNTIME_VERSION, generated_rule_index=None,
            llm_live=False, attempted=False, exhausted=False, candidate_pruning=False,
            heuristic_order_used=False, physics_bypass=False, physics_verdict_reused=False,
            path_selection_policy="path_probe_round_v1", fallback_policy="path_stage_v1",
            enumeration_policy="canonical_filtered"))
    return validate_path_round_response(request, validate_response(request, response))


def build_context(request, extension=None):
    """One data projection for admission, evidence replay and online execution."""
    from orchestrator.skill_program import _scoped_last
    request = validate_request(request)
    last = _scoped_last(request)
    if last is not None:
        last["unstable"] = sorted(last["unstable"])
        last["initial_disconnected"] = sorted(last["initial_disconnected"])
    context = {key: deepcopy(request[key]) for key in (
        "candidate_ids", "remaining_parts", "base_part", "max_grippers",
        "pose_catalog", "geometry", "memory_references")}
    context.update(last=deepcopy(last), directions=deepcopy(DIRECTIONS), primitive_ids=list(SKILLS))
    if extension is not None:
        from orchestrator.lesson_runtime import runtime_context
        context.update(runtime_context(request, last, extension))
    bounded(context)
    return context, last


def evaluate(program, request, max_actions=512, max_steps=50000, extension=None, diagnose=False,
             observe_abstention=False):
    """Trusted child entry. No physics; duplicate/path guards use original input."""
    program = validate_program(program)
    request = validate_request(request)
    context, last = build_context(request, extension)
    identity = digest(program)
    calls = [0]

    def eligible(choice):
        calls[0] += 1
        if calls[0] > max_actions:
            raise ValueError("python_eligibility_budget")
        try:
            _response(request, deepcopy(choice), identity)
            return True
        except (ValueError, TypeError, KeyError):
            return False

    from orchestrator.python_input_contract import ReadAudit
    audit = ReadAudit(context) if diagnose else None
    vm_class = VM
    if observe_abstention:
        from orchestrator.python_abstention import ObservedVM
        vm_class = ObservedVM
    vm = vm_class(program["source"], eligible, max_steps=max_steps, read_audit=audit)
    # Generated code may mutate its local context. Bind the pre-execution input.
    original_context = deepcopy(context) if observe_abstention else None
    error_type = None
    try:
        choice = vm.call("propose", [context])
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, RecursionError) as error:
        if not diagnose:
            raise
        choice, error_type = None, type(error).__name__
    result = dict(choice=choice, steps=vm.steps, eligibility_checks=calls[0],
                scoped_feedback_evidence_id=last["feedback"]["evidence_id"] if last else None)
    if audit is not None:
        result.update(input_audit=audit.report(), diagnostic_runtime_error=error_type)
    if observe_abstention and choice is None and error_type is None:
        from orchestrator.skill_runtime_feedback import trace_from_vm
        result['runtime_abstention'] = trace_from_vm(vm, program, request, original_context)
    return result


def step_program(program, request, *, max_actions=50000, timeout_seconds=4, extension=None, diagnose=False,
                 observe_abstention=False):
    if type(diagnose) is not bool:
        raise ValueError('invalid_diagnostic_flag')
    if type(observe_abstention) is not bool:
        raise ValueError('invalid_abstention_observation_flag')
    program = validate_program(program)
    request = validate_request(request)
    if type(max_actions) is not int or not 1 <= max_actions <= 50000:
        raise ValueError("invalid_program_scan_limit")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 10:
        raise ValueError("invalid_python_runtime_timeout")
    partition = partition_path_round(request)
    result = dict(response=None, triggered=False, rule_index=None, scanned=0,
        program_sha256=digest(program), not_applicable_reason="python_abstained",
        decision_origin="generated_program", physical_calls=0, api_calls=0,
        candidate_pruning=False, physics_verdict_reused=False,
        scoped_feedback_evidence_id=None, runtime=RUNTIME_VERSION, vm_steps=0)
    packet = dict(program=program, request=request, max_actions=min(512, max_actions))
    if diagnose:
        packet['diagnose'] = True
    if observe_abstention:
        packet['observe_abstention'] = True
    if extension is not None:
        # Validate a guard against the actual plan/request, in both parent and child.
        from orchestrator.lesson_runtime import runtime_context
        runtime_context(request, None, extension)
        packet["extension"] = deepcopy(extension)
    payload = json.dumps(packet, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > MAX_INPUT_BYTES:
        result["not_applicable_reason"] = "python_input_byte_limit"
        return result
    runner = Path(__file__).absolute().with_name("python_skill_child.py")
    # -I/-S avoid startup hooks, user site and PYTHONPATH. Only Windows OS loader
    # variables are inherited; no provider keys or user profiles are sent.
    environment = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    try:
        child = subprocess.run([sys.executable, "-I", "-S", "-B", str(runner)],
            input=payload, capture_output=True, timeout=timeout_seconds,
            cwd=str(runner.parent), env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if child.returncode != 0 or len(child.stdout) > 65536:
            raise ValueError("python_child_failed")
        value = json.loads(child.stdout)
        if "error" in value:
            raise ValueError(value["error"])
        if observe_abstention and 'runtime_abstention' in value:
            trace = value['runtime_abstention']
            context, _ = build_context(request, extension)
            if (trace.get('program_sha256') != digest(program)
                    or trace.get('request_sha256') != digest(request)
                    or trace.get('context_sha256') != digest(context)
                    or trace.get('code_observation_only') is not True
                    or trace.get('physical_failure_observed') is not False
                    or value['choice'] is not None):
                raise ValueError('python_abstention_binding_mismatch')
            result['runtime_abstention'] = trace
        if diagnose:
            result.update(input_audit=value['input_audit'], diagnostic_choice=value['choice'],
                          diagnostic_runtime_error=value['diagnostic_runtime_error'])
            if value['diagnostic_runtime_error']:
                raise ValueError('python_diagnostic_runtime_error')
        result.update(scanned=value["eligibility_checks"], vm_steps=value["steps"],
                      scoped_feedback_evidence_id=value["scoped_feedback_evidence_id"])
        result["triggered"] = value["scoped_feedback_evidence_id"] is not None
        if extension is not None and extension["guard"] is not None:
            result.update(triggered=True, guard_evidence_id=extension["guard"]["evidence_id"])
        if value["choice"] is None:
            result["not_applicable_reason"] = "python_abstained" if result["triggered"] else "no_trigger"
            return result
        # A child cannot supply trusted metadata, request IDs, candidates or PASS.
        response = _response(request, value["choice"], digest(program))
        response["metadata"].update(path_probe_round=deepcopy(partition["round_audit"]),
            generated_scan_count=result["scanned"], python_vm_steps=result["vm_steps"])
        result.update(response=response, triggered=True, not_applicable_reason=None)
    except subprocess.TimeoutExpired:
        result["not_applicable_reason"] = "python_wall_timeout"
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as error:
        result["not_applicable_reason"] = "python_execution_rejected"
        # No arbitrary source/runtime messages that might contain sensitive data.
        result["error_type"] = type(error).__name__
        text = str(error)
        result["error_code"] = text if text.startswith("python_") and len(text) < 120 else "invalid_python_action_or_runtime"
    return result
