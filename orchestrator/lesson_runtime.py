"""Bound feedback projection for the existing disposable Python interpreter."""
from copy import deepcopy
from orchestrator.pose_guard_recovery import validate_guard
from worker.task_memory import digest

CONTRACT = "lesson_runtime_v2"


def extension_for(guard=None, plan=None, index=0):
    """Executable code receives observations, not mutable natural-language cards.

    Lessons are planner/generator guidance. Keeping this reserved list empty
    avoids changing a validated program branch when its explanation is updated.
    """
    return dict(guard=deepcopy(guard), plan=deepcopy(plan), index=index, lessons=[])


def bind_execution(program, request, extension=None):
    from orchestrator.python_skill_program import build_context
    extension = extension_for() if extension is None else deepcopy(extension)
    context, _ = build_context(request, extension)
    return dict(version=CONTRACT, program_sha256=digest(program), request_sha256=digest(request),
                extension=extension, context_sha256=digest(context))


def checked_execution(program, request, selected, case):
    """Old artifacts remain old. New artifacts cannot silently replay as old."""
    version, binding = case.get("runtime_contract"), selected.get("program_runtime")
    if version is None and binding is None:
        return None
    if version != CONTRACT or type(binding) is not dict or binding.get("version") != CONTRACT:
        raise ValueError("program_runtime_contract_mismatch")
    # This physical admission runner does not create nonphysical plan guards.
    if binding.get("extension") != extension_for():
        raise ValueError("physical_admission_runtime_extension_mismatch")
    if binding != bind_execution(program, request, binding["extension"]):
        raise ValueError("program_runtime_input_binding_mismatch")
    return deepcopy(binding["extension"])


def execute_bound(program, request, extension):
    if extension is None:
        from orchestrator.skill_program import step_program
        return step_program(program, request)
    from orchestrator.python_skill_program import step_program
    return step_program(program, request, extension=extension)


def runtime_context(request, last, extension):
    if type(extension) is not dict or set(extension) != {"guard", "plan", "index", "lessons"}:
        raise ValueError("python_invalid_lesson_extension")
    guard = extension["guard"]
    if guard is not None:
        validate_guard(guard, extension["plan"], extension["index"], request)
        failure = dict(kind="plan_pose_guard_v1", evidence_id=guard["evidence_id"],
                       physical_attempt_executed=False, path_success=None, stability_success=None)
    elif last is not None:
        f = last["feedback"]
        failure = dict(kind=("stability_failed" if f.get("path_success") is True
                            and f.get("stability_success") is False else
                            "path_failed" if f.get("path_success") is False else "unknown"),
                       evidence_id=f["evidence_id"], physical_attempt_executed=True,
                       path_success=f.get("path_success"), stability_success=f.get("stability_success"))
    else:
        failure = None
    history = [{k: deepcopy(row.get(k)) for k in ("evidence_id", "choice", "status",
                "path_success", "stability_success", "deadline_exceeded", "selected_pose_matrix")}
               for row in request["episode_feedback"]
               if set(row.get("remaining_parts", [])) == set(request["remaining_parts"])][-16:]
    lessons = extension["lessons"]
    if type(lessons) is not list or len(lessons) > 8:
        raise ValueError("python_invalid_lesson_references")
    return dict(plan_guard=deepcopy(guard), failure=failure, skill_history=history,
                lesson_references=deepcopy(lessons))
