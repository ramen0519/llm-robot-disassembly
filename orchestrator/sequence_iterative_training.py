"""Wave-by-wave training coordinator with a mandatory two-round review gate.

This module performs no I/O, model call or simulation itself. Backends supply
bounded operations and immutable evidence. COMPLETED means a scheduled training
round was handled; it does not mean the assembly or generated skill succeeded.
The existing two-episode coordinator remains unchanged.
"""
from copy import deepcopy


LEARNING_STATUSES = frozenset({"REGISTERED", "DUPLICATE_PROGRAM", "NO_OBSERVATIONS",
    "LESSON_REJECTED", "CANDIDATE_REJECTED", "VALIDATION_FAILED", "POSTLESSON_REJECTED"})
NOVELTY_FIELDS = frozenset({"new_record_count", "new_condition_count", "repeated_condition_count",
                          "new_skill_count", "duplicate_programs", "behavior_equivalence"})


class _Stop(Exception):
    def __init__(self, reason):
        self.reason = reason


def _identity(value):
    return type(value) is str and bool(value.strip()) and len(value) <= 256


def _episode_ref(value):
    if (type(value) is not dict or set(value) != {"path", "sha256"}
            or type(value["path"]) is not str or not value["path"]
            or type(value["sha256"]) is not str or len(value["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in value["sha256"])):
        raise ValueError("round_summary_requires_episode_artifact_ref")
    # The checkpoint writer verifies bytes and permitted paths. A pure
    # coordinator neither opens this pointer nor certifies it as actual physics.
    return deepcopy(value)


def _validate(tasks, start_round, stop_round, max_rounds, held_out_ids, prior_rounds, review_approved):
    if type(tasks) not in (list, tuple) or len(tasks) != 5:
        raise ValueError("exactly_five_iterative_training_tasks_required")
    if any(type(task) is not dict or not _identity(task.get("assembly_id")) for task in tasks):
        raise ValueError("invalid_iterative_training_task")
    ids = [task["assembly_id"] for task in tasks]
    if any(identity in (".", "..") or any(char in identity for char in "/\\:") for identity in ids):
        raise ValueError("iterative_task_ids_must_be_path_free_names")
    if len(set(ids)) != 5 or set(ids).intersection(held_out_ids):
        raise ValueError("invalid_or_leaking_iterative_training_split")
    if (any(type(value) is not int for value in (start_round, stop_round, max_rounds))
            or not 1 <= start_round <= stop_round <= max_rounds <= 5):
        raise ValueError("invalid_iterative_round_limits")
    if type(review_approved) is not bool:
        raise ValueError("invalid_iterative_review_approval")
    if start_round >= 3 and not review_approved:
        raise ValueError("explicit_review_approval_required_before_round_three")
    if type(prior_rounds) not in (list, tuple):
        raise ValueError("invalid_prior_training_rounds")
    seen = set()
    for row in prior_rounds:
        if (type(row) is not dict or row.get("status") != "COMPLETED"
                or row.get("assembly_id") not in ids or type(row.get("round_index")) is not int
                or row.get("learning_status") not in LEARNING_STATUSES
                or row.get("episode_status") not in ("PASS", "UNKNOWN")):
            raise ValueError("invalid_prior_completed_training_round")
        key = row["assembly_id"], row["round_index"]
        if key in seen:
            raise ValueError("duplicate_prior_training_round")
        seen.add(key)
        _episode_ref(row.get("episode_ref"))
    expected = {(identity, index) for identity in ids for index in range(1, start_round)}
    if seen != expected:
        raise ValueError("complete_prior_waves_required_for_resume")
    if [(row["assembly_id"], row["round_index"]) for row in prior_rounds] != [
            (identity, index) for index in range(1, start_round) for identity in ids]:
        raise ValueError("prior_training_rounds_must_preserve_wave_order")
    return ids


def _halt(backend, result=None):
    reason = getattr(backend, "halted", None)
    if reason:
        raise _Stop(str(reason))
    if type(result) is dict and result.get("halt_required"):
        raise _Stop(result.get("error") or "backend_requested_halt")


def _duplicate(value):
    if type(value) is bool:
        return value
    if type(value) is dict and type(value.get("duplicate")) is bool:
        return value["duplicate"]
    raise ValueError("duplicate_candidate_requires_explicit_boolean")


def _learn_round(task, packet, backend, row):
    """One lesson, at most two candidate versions, and at most one post-lesson."""
    def record(stage, result, **fields):
        if type(result) is not dict:
            raise ValueError("training_stage_result_must_be_object")
        row["stages"].append(dict(stage=stage, result=deepcopy(result), **fields))
        _halt(backend, result)
        return result

    initial = record("initial_lesson", backend.lesson(task, packet))
    if not initial.get("ok"):
        return "LESSON_REJECTED"
    lesson = initial.get("lesson")
    if type(lesson) is not dict:
        raise ValueError("successful_learning_requires_lesson")
    previous, feedback = None, None
    learning_status = "CANDIDATE_REJECTED"
    try:
        for revision in range(2):
            _halt(backend)
            result = record("candidate", backend.candidate(task, packet, lesson, previous, feedback, revision),
                            revision=revision)
            if not result.get("ok"):
                previous = result.get("candidate") or {"rejected_result": deepcopy(result)}
                feedback = result
                learning_status = "CANDIDATE_REJECTED"
                continue
            candidate = result.get("candidate")
            if type(candidate) is not dict:
                raise ValueError("successful_learning_requires_candidate")
            match = backend.duplicate_candidate(candidate)
            duplicate = _duplicate(match)
            row["stages"].append(dict(stage="duplicate_check", duplicate=duplicate, result=deepcopy(match)))
            _halt(backend)
            if duplicate:
                row["duplicate_program_sha256"] = candidate.get("program_sha256")
                row["duplicate_skill_id"] = match.get("skill_id") if type(match) is dict else None
                backend.event("duplicate_candidate", task=task["assembly_id"], round_index=row["round_index"],
                    candidate_sha256=candidate.get("content_sha256"), program_sha256=candidate.get("program_sha256"),
                    physics_validation_skipped=True, postvalidation_API_skipped=True)
                return "DUPLICATE_PROGRAM"
            validation = record("validation", backend.validate_candidate(task, candidate, lesson), revision=revision)
            if not validation.get("ok"):
                previous, feedback = candidate, validation
                learning_status = "VALIDATION_FAILED"
                continue
            post = record("postvalidation_lesson", backend.lesson(task, packet, previous_lesson=lesson,
                          validation=validation, candidate=candidate))
            if not post.get("ok"):
                return "POSTLESSON_REJECTED"
            card = backend.register(candidate, post["lesson"], validation)
            if type(card) is not dict or not _identity(card.get("skill_id")):
                raise ValueError("registration_requires_skill_card")
            row["registered_skill"] = card["skill_id"]
            row["stages"].append(dict(stage="registration", skill_id=card["skill_id"],
                                      performance_effect="UNVERIFIED"))
            _halt(backend)
            return "REGISTERED"
        return learning_status
    finally:
        # Preserve a grounded initial lesson even if a later candidate fails.
        # Do this after candidate handling: validation backends may already bind
        # this lesson separately and should not receive a duplicate initial row.
        backend.retain_lesson(lesson)
        row["initial_lesson_retained"] = True


def _summary(backend, task, round_number, row):
    summary = backend.round_summary(task, round_number)
    if type(summary) is not dict:
        raise ValueError("round_summary_must_be_object")
    row["episode_ref"] = _episode_ref(summary.get("episode_ref"))
    novelty = deepcopy(summary.get("novelty", {}))
    if type(novelty) is not dict:
        raise ValueError("invalid_training_novelty_summary")
    for key, value in summary.items():
        if key in NOVELTY_FIELDS:
            novelty[key] = deepcopy(value)
        elif key not in ("episode_ref", "novelty"):
            if key in row and row[key] != value:
                raise ValueError("round_summary_cannot_override_coordinator_outcome")
            row[key] = deepcopy(value)
    for key, value in novelty.items():
        if key in NOVELTY_FIELDS - {"behavior_equivalence"} and (type(value) is not int or value < 0):
            raise ValueError("training_novelty_counts_must_be_nonnegative")
    row["novelty"] = novelty


def run_iterative_training(tasks, backend, *, start_round=1, stop_round=2, max_rounds=5,
                           held_out_ids=(), prior_rounds=(), review_approved=False):
    """Run waves across all five products, learning after EVERY free episode.

    Backend hooks: begin_round(task, number), freeze(name), episode(task,
    number-1, frozen), collect, packet, lesson, candidate, duplicate_candidate,
    validate_candidate, register, retain_lesson, round_summary(task, number),
    event, and ``halted``. round_summary must supply a hashed episode_ref; the
    checkpoint module checks its bytes. No novelty-based automatic stop is used.

    An invocation starting in rounds 1/2 ALWAYS stops for review after round 2,
    even if stop_round=5 or review_approved=True was accidentally supplied.
    Resuming at 3+ requires explicit approval AND every prior task-round record.
    """
    ids = _validate(tasks, start_round, stop_round, max_rounds, held_out_ids, prior_rounds, review_approved)
    effective_stop = min(stop_round, 2) if start_round <= 2 else stop_round
    completed = deepcopy(list(prior_rounds))
    current_rows, waves, frozen, stop_reason, last_round = [], [], None, None, start_round-1
    for round_number in range(start_round, effective_stop+1):
        wave = dict(round_index=round_number, task_order=list(ids), completed_tasks=0, status="RUNNING")
        waves.append(wave)
        for task in tasks:
            row = dict(assembly_id=task["assembly_id"], round_index=round_number, status="INCOMPLETE",
                episode_status="NOT_RUN", learning_status=None, reuse_status="NOT_RUN", stages=[],
                registered_skill=None, initial_lesson_retained=False, actual_effect="UNVERIFIED")
            current_rows.append(row)
            try:
                _halt(backend)
                backend.begin_round(task, round_number)
                _halt(backend)
                backend.event("task_round_started", task=task["assembly_id"], round_index=round_number)
                frozen = backend.freeze("wave%02d_%s_before_episode" % (round_number, task["assembly_id"]))
                _halt(backend)
                episode = backend.episode(task, round_number-1, frozen)
                if type(episode) is not dict:
                    raise ValueError("episode_result_must_be_object")
                row["stages"].append(dict(stage="free_episode", result=deepcopy(episode)))
                row["episode_status"] = episode.get("status", "NOT_RUN")
                row["reuse_status"] = episode.get("reuse_status", "UNVERIFIED")
                _halt(backend, episode)
                if row["episode_status"] not in ("PASS", "UNKNOWN"):
                    raise _Stop("episode_not_safely_completed")
                backend.collect(episode)
                _halt(backend)
                packet = backend.packet(task, episode)
                _halt(backend)
                if packet is None:
                    row["learning_status"] = "NO_OBSERVATIONS"
                    row["stages"].append(dict(stage="learning", status="NO_OBSERVATIONS"))
                else:
                    row["learning_status"] = _learn_round(task, packet, backend, row)
                _halt(backend)
                _summary(backend, task, round_number, row)
                _halt(backend)
                row["status"] = "COMPLETED"
                backend.event("task_round_finished", summary=deepcopy(row))
                completed.append(deepcopy(row))
                wave["completed_tasks"] += 1
            except _Stop as error:
                stop_reason = error.reason
            except Exception as error:
                # Backends preserve raw logs. Do not copy API/server/secret text
                # into coordinator errors or continue paid work after uncertainty.
                stop_reason = "stage_exception:" + type(error).__name__
                row["stage_error_type"] = type(error).__name__
            if stop_reason:
                row["status"] = "INCOMPLETE"
                row["stop_reason"] = stop_reason
                wave["status"] = "INCOMPLETE"
                break
        if stop_reason:
            break
        wave["status"] = "COMPLETED"
        last_round = round_number
        try:
            frozen = backend.freeze("wave%02d_completed" % round_number)
        except Exception as error:
            stop_reason = "round_freeze_exception:" + type(error).__name__
            break
    status = ("HALTED" if stop_reason else "REVIEW_REQUIRED" if start_round <= 2 and last_round == 2 else
              "COMPLETED" if last_round == max_rounds else "ROUND_LIMIT_REACHED")
    return dict(protocol="sequence_iterative_training_v1", status=status, stop_reason=stop_reason,
        task_order=ids, start_round=start_round, requested_stop_round=stop_round,
        effective_stop_round=effective_stop, max_rounds=max_rounds, last_completed_round=last_round,
        review_approved=review_approved, review_required=status == "REVIEW_REQUIRED",
        completed_rounds=completed, current_rounds=current_rows, waves=waves, frozen=frozen,
        new_completed_round_count=len(completed)-len(prior_rounds),
        next_round=(last_round+1 if not stop_reason and last_round < max_rounds else None),
        generalization="NOT_TESTED", performance_effect="UNVERIFIED", automatic_retries=0)
