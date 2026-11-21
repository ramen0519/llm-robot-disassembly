"""Read-only knowledge for sequence_native_v1; no model, files, or physics calls.

Historical physical observations are references, never current verdicts or hard
constraints. Legacy physical-action programs are deliberately incompatible.
New sequence programs run in the existing metered data-only AST interpreter and
can propose only a complete permutation. A valid permutation is not a verified
disassembly. No actual model-generated sequence card has been created here.
"""
from copy import deepcopy
import json

from orchestrator.explicit_action_memory import FrozenActionMemory
from orchestrator.lesson_memory import checked, validate_claim
from orchestrator.python_skill_vm import LANGUAGE, VM, bounded, parse, PolicyError
from worker.explicit_action_contract import PROTOCOL as MEMORY_PROTOCOL
from worker.task_memory import digest, _input_hash


PROTOCOL = "sequence_knowledge_v1"
SKILL_PROTOCOL = "sequence_skill_v1"
NATIVE_MEMORY_PROTOCOL = "sequence_native_observation_v1"
PROFILE = "sequence_native_v1"
PART_FIELDS = {"bounds", "extents", "bounds_center", "center_mass", "volume",
               "watertight", "validity"}
CARD_FIELDS = {"schema_version", "protocol", "skill_id", "name", "description",
               "program", "program_sha256", "source_task_ids", "validation",
               "fixture", "content_sha256"}
VALIDATION_FIELDS = {"status", "profile", "program_sha256", "run_refs",
                     "source_hashes", "evidence_ids", "scope", "performance_effect"}
LIMITS = dict(max_records=8, max_skills=4, max_lessons=3, max_context_bytes=49152,
              memory_enabled=True, skills_enabled=True, allow_fixture_skills=False,
              exclude_current_task=True, excluded_task_ids=["00335"])


def _text(value, label, maximum=256):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError("invalid_" + label)
    return value


def _hash(value):
    if (type(value) is not str or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("invalid_sequence_source_hash")
    return value


def _input_identity(value):
    # Schema-2 CAD records preserve each OBJ/config hash; earlier fixtures use a
    # single aggregate digest. Neither representation opens the referenced files.
    if not _input_hash(value):
        raise ValueError("invalid_sequence_input_hash")
    return value


def _ids(values, label):
    if type(values) is not list or len(values) > 256:
        raise ValueError("invalid_" + label)
    for value in values:
        _text(value, label)
    if len(set(values)) != len(values):
        raise ValueError("duplicate_" + label)
    return list(values)


def _size(value):
    return len(json.dumps(value, allow_nan=False, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8"))


def _task(value):
    """Keep flat legacy and nested CAD identities distinct from part identity."""
    if type(value) is not dict:
        return {}
    result = value.get("task", value)
    return deepcopy(result) if type(result) is dict else {}


def _source_task_id(value):
    task = _task(value)
    identity = task.get("assembly_id")
    return identity if type(identity) is str and identity else None


def _records(value):
    if isinstance(value, FrozenActionMemory):
        return deepcopy(value.snapshot["records"]), value.snapshot["content_sha256"]
    if type(value) is dict and value.get("protocol") == NATIVE_MEMORY_PROTOCOL:
        frozen = validate_native_snapshot(value)
        return frozen["records"], frozen["content_sha256"]
    if isinstance(value, (list, tuple)) and value and all(type(v) is dict and "records" in v for v in value):
        collected, hashes, seen = [], [], set()
        for frozen in value:
            rows, identity = _records(frozen)
            hashes.append(identity)
            for row in rows:
                if row["evidence_id"] in seen:
                    raise ValueError("duplicate_combined_sequence_evidence")
                seen.add(row["evidence_id"])
                collected.append(row)
        return collected, digest(hashes)
    if type(value) is dict:
        checked_memory = FrozenActionMemory(value).snapshot
    else:
        if not isinstance(value, (list, tuple)):
            raise ValueError("sequence_memory_requires_records_or_frozen_snapshot")
        rows = list(value)
        checked_memory = FrozenActionMemory(dict(schema_version=1, protocol=MEMORY_PROTOCOL,
            records=rows, record_count=len(rows), content_sha256=digest(rows),
            hard_pruning=False, physics_reuse_allowed=False, frozen_utc=None)).snapshot
    return checked_memory["records"], checked_memory["content_sha256"]


def _record_context(record):
    return (record["source_task_context"] if record["protocol"] == NATIVE_MEMORY_PROTOCOL
            else record["scope"]["task_context"])


def build_native_snapshot(context, attempts, source_ref, source_hash):
    """Freeze final worker attempts after transitions updated acceptance fields.

    ``context`` is worker task_context(CAD), and ``attempts`` is the FINAL
    worker_result['attempts'], not simulate_end events emitted before acceptance.
    Callers hash the saved worker result themselves. This function never reads
    source_ref and never infers an explicit single-direction action contract.
    """
    _text(source_ref, "native_source_ref", 2048)
    _hash(source_hash)
    if type(context) is not dict or not _source_task_id(context):
        raise ValueError("native_memory_requires_task_identity")
    task = _task(context)
    _input_identity(task.get("input_sha256"))
    if type(attempts) is not list or len(attempts) > 4096:
        raise ValueError("invalid_native_attempts")
    records = []
    for attempt in attempts:
        _validate_native_attempt(attempt)
        row = dict(schema_version=1, protocol=NATIVE_MEMORY_PROTOCOL,
            source_task_context=deepcopy(context), source_ref=source_ref, source_sha256=source_hash,
            source_observation_id=attempt["event_id"], attempt=deepcopy(attempt))
        row["evidence_id"] = "sequence-native-evidence:" + digest(dict(
            source_sha256=source_hash, observation_id=attempt["event_id"]))
        row["record_sha256"] = digest(row)
        records.append(row)
    frozen = dict(schema_version=1, protocol=NATIVE_MEMORY_PROTOCOL, records=records,
        record_count=len(records), hard_pruning=False, physics_reuse_allowed=False,
        native_source="final_worker_result_attempts", content_sha256=digest(records))
    return validate_native_snapshot(frozen)


def _validate_native_attempt(attempt):
    required = {"event_id", "part_move", "remaining_parts", "request", "returned", "status",
                "accepted_for_transition", "deadline_exceeded", "verifier_domain"}
    if type(attempt) is not dict or not required.issubset(attempt):
        raise ValueError("native_attempt_requires_final_observation")
    _text(attempt["event_id"], "native_observation_id")
    parts = _ids(attempt["remaining_parts"], "native_remaining_parts")
    if attempt["part_move"] not in parts or type(attempt["request"]) is not dict:
        raise ValueError("native_attempt_part_scope_mismatch")
    req = attempt["request"]
    if (req.get("part_move") != attempt["part_move"] or type(req.get("parts_rest")) is not list
            or set(req["parts_rest"]) != set(parts) - {attempt["part_move"]}):
        raise ValueError("native_attempt_request_scope_mismatch")
    if (type(attempt["returned"]) is not bool or attempt["status"] not in ("PASS", "UNKNOWN", "ERROR")
            or attempt["verifier_domain"] != "ASAP_RedMax"):
        raise ValueError("invalid_native_attempt_status")
    for key in ("accepted_for_transition", "deadline_exceeded"):
        if attempt[key] is not None and type(attempt[key]) is not bool:
            raise ValueError("invalid_native_execution_observation")
    if attempt["accepted_for_transition"] is True and (
            attempt["status"] != "PASS" or attempt["deadline_exceeded"] is not False):
        raise ValueError("native_accepted_requires_timely_pass")
    raw = attempt.get("raw_sim_info")
    if attempt["returned"]:
        if type(raw) is not dict or type(raw.get("feasible")) is not bool:
            raise ValueError("native_return_requires_raw_feasible")
        if attempt["status"] == "PASS" and not raw["feasible"]:
            raise ValueError("native_pass_requires_raw_feasible")
        if attempt["status"] == "UNKNOWN" and raw["feasible"]:
            raise ValueError("native_raw_pass_status_mismatch")
    elif attempt["status"] == "PASS":
        raise ValueError("native_pass_requires_returned_physics")
    # Reject NaN/objects without dereferencing source paths or loading modules.
    _size(attempt)


def validate_native_snapshot(value):
    required = {"schema_version", "protocol", "records", "record_count", "hard_pruning",
                "physics_reuse_allowed", "native_source", "content_sha256"}
    if (type(value) is not dict or set(value) != required or value["schema_version"] != 1
            or value["protocol"] != NATIVE_MEMORY_PROTOCOL or type(value["records"]) is not list
            or type(value["record_count"]) is not int or value["record_count"] != len(value["records"])
            or value["hard_pruning"] is not False or value["physics_reuse_allowed"] is not False
            or value["native_source"] != "final_worker_result_attempts"
            or value["content_sha256"] != digest(value["records"])):
        raise ValueError("invalid_native_memory_snapshot")
    seen = set()
    for row in value["records"]:
        fields = {"schema_version", "protocol", "source_task_context", "source_ref", "source_sha256",
                  "source_observation_id", "attempt", "evidence_id", "record_sha256"}
        if (type(row) is not dict or set(row) != fields or row["schema_version"] != 1
                or row["protocol"] != NATIVE_MEMORY_PROTOCOL
                or not _source_task_id(row["source_task_context"])
                or row["record_sha256"] != digest({k: v for k, v in row.items() if k != "record_sha256"})):
            raise ValueError("invalid_native_memory_record")
        _input_identity(_task(row["source_task_context"]).get("input_sha256"))
        _text(row["source_ref"], "native_source_ref", 2048)
        _hash(row["source_sha256"])
        _validate_native_attempt(row["attempt"])
        expected = "sequence-native-evidence:" + digest(dict(source_sha256=row["source_sha256"],
            observation_id=row["source_observation_id"]))
        if row["source_observation_id"] != row["attempt"]["event_id"] or row["evidence_id"] != expected:
            raise ValueError("native_memory_identity_mismatch")
        if row["evidence_id"] in seen:
            raise ValueError("duplicate_native_memory_evidence")
        seen.add(row["evidence_id"])
    if _size(value) > 32 * 1024 * 1024:
        raise ValueError("native_memory_size_limit")
    return deepcopy(value)


def _native_observation(record):
    attempt = record["attempt"]
    raw, req = attempt.get("raw_sim_info") or {}, attempt["request"]
    path = stable = None
    if attempt["returned"]:
        if raw["feasible"]:
            path, stable = True, True
        elif "action" in raw:
            path = raw["action"] is not None
            stable = False if path else None
    return dict(evidence_id=record["evidence_id"], record_sha256=record["record_sha256"],
        source_protocol=NATIVE_MEMORY_PROTOCOL, source_task_context=deepcopy(record["source_task_context"]),
        source_observation_id=record["source_observation_id"], source_ref=record["source_ref"],
        source_sha256=record["source_sha256"], source_remaining_parts=deepcopy(attempt["remaining_parts"]),
        source_choice=dict(part_id=attempt["part_move"], native_action=deepcopy(raw.get("action")),
            native_support_ids=deepcopy(raw.get("parts_fix")),
            selection_scope=attempt.get("selection_scope")),
        source_pose_matrix=deepcopy(req.get("pose")),
        source_settings={key: deepcopy(req.get(key)) for key in ("max_grippers", "timeout")},
        source_hashes=deepcopy(attempt.get("source_hashes")),
        status=attempt["status"], path_success=path, stability_success=stable,
        observed_stage=("both_passed" if path is True and stable is True else
            "path_search_failed" if path is False else "stability_failed" if stable is False else "unknown"),
        accepted_for_transition=attempt["accepted_for_transition"],
        deadline_exceeded=attempt["deadline_exceeded"], verifier_origin=attempt["verifier_domain"],
        source_result_sha256=digest(attempt), source_raw_sha256=digest(raw),
        reference_only=True, conditions_equivalent=False, current_part_id_mapping=None,
        current_attempt_verified=False, hard_pruning=False, physics_reuse_allowed=False,
        blocker=None, failure_cause=None)


def _observation(record):
    if record["protocol"] == NATIVE_MEMORY_PROTOCOL:
        return _native_observation(record)
    scope, result = record["scope"], record["result"]
    execution = result.get("execution") or {}
    path, stable = result["path_success"], result["stability_success"]
    return dict(evidence_id=record["evidence_id"], record_sha256=record["record_sha256"],
        source_task_context=deepcopy(scope["task_context"]),
        source_run_id=record["source_run_id"], source_state_id=record["source_state_id"],
        source_remaining_parts=deepcopy(scope["remaining_parts"]),
        source_choice=deepcopy(record["choice"]), source_pose_matrix=deepcopy(scope["pose_matrix"]),
        source_base_part=scope["base_part"], source_max_grippers=scope["max_grippers"],
        status=result["status"], path_success=path, stability_success=stable,
        observed_stage=("path_search_failed" if path is False else
            "stability_failed" if path is True and stable is False else
            "both_passed" if path is True and stable is True else "unknown"),
        verifier_origin=result["verifier_origin"], source_evidence_ref=result["evidence_ref"],
        accepted_for_transition=execution.get("accepted_for_transition"),
        deadline_exceeded=execution.get("deadline_exceeded"),
        parent_failure_evidence_id=record["parent_failure_evidence_id"],
        source_result_sha256=digest(result), source_raw_sha256=digest(result["raw"]),
        reference_only=True, conditions_equivalent=False, current_part_id_mapping=None,
        current_attempt_verified=False, hard_pruning=False, physics_reuse_allowed=False,
        blocker=None, failure_cause=None)


def _balanced(rows):
    """Deterministically interleave source assemblies and observed stages."""
    buckets = {}
    for row in sorted(rows, key=lambda item: item["evidence_id"]):
        key = (_source_task_id(row["source_task_context"]), row["observed_stage"])
        buckets.setdefault(key, []).append(row)
    result = []
    while buckets:
        for key in sorted(buckets):
            result.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
    return result


def validate_sequence_skill(skill, *, allow_fixture=False):
    """Check an immutable card; provenance file contents are not opened here.

    Registration is an upstream responsibility. This loader requires its exact
    program hash, run references, and source hashes. It does not certify that the
    program is effective or infer physical success from registration metadata.
    """
    if type(skill) is not dict or set(skill) != CARD_FIELDS:
        raise ValueError("incompatible_or_invalid_sequence_skill")
    if skill["schema_version"] != 1 or skill["protocol"] != SKILL_PROTOCOL:
        raise ValueError("incompatible_sequence_skill_protocol")
    if type(skill["fixture"]) is not bool or (skill["fixture"] and not allow_fixture):
        raise ValueError("fixture_sequence_skill_not_allowed")
    if skill["content_sha256"] != digest({k: v for k, v in skill.items() if k != "content_sha256"}):
        raise ValueError("sequence_skill_hash_mismatch")
    for field, maximum in (("skill_id", 256), ("name", 256), ("description", 1500)):
        _text(skill[field], field, maximum)
    program = skill["program"]
    if type(program) is not dict or set(program) != {"language", "source"} or program["language"] != LANGUAGE:
        raise ValueError("invalid_sequence_python_program")
    parse(program["source"])
    if skill["program_sha256"] != digest(program):
        raise ValueError("sequence_program_hash_mismatch")
    tasks = _ids(skill["source_task_ids"], "sequence_source_task_ids")
    if not tasks:
        raise ValueError("sequence_source_task_provenance_required")
    validation = skill["validation"]
    if (type(validation) is not dict or set(validation) != VALIDATION_FIELDS
            or validation["status"] != "REGISTERED" or validation["profile"] != PROFILE
            or validation["program_sha256"] != skill["program_sha256"]
            or validation["performance_effect"] != "UNVERIFIED"
            or validation["scope"] != ("offline_contract_fixture" if skill["fixture"] else
                                       "sequence_contract_and_native_physics")):
        raise ValueError("sequence_registration_required")
    if not _ids(validation["run_refs"], "sequence_validation_runs"):
        raise ValueError("sequence_validation_runs_required")
    _ids(validation["evidence_ids"], "sequence_validation_evidence")
    if (type(validation["source_hashes"]) is not list or not validation["source_hashes"]
            or len(validation["source_hashes"]) > 64):
        raise ValueError("sequence_validation_hashes_required")
    for value in validation["source_hashes"]:
        _hash(value)
    bounded(skill)
    return deepcopy(skill)


def make_fixture_sequence_skill(source, skill_id="fixture:reverse"):
    """Create explicit offline fixture data, never an actual registered skill."""
    program = dict(language=LANGUAGE, source=source)
    value = dict(schema_version=1, protocol=SKILL_PROTOCOL, skill_id=skill_id,
        name="Offline sequence contract fixture", description="No model or physical validation.",
        program=program, program_sha256=digest(program), source_task_ids=["fixture-training"], fixture=True,
        validation=dict(status="REGISTERED", profile=PROFILE, program_sha256=digest(program),
            run_refs=["fixture:no_physics"], source_hashes=[digest(program)], evidence_ids=[],
            scope="offline_contract_fixture", performance_effect="UNVERIFIED"))
    value["content_sha256"] = digest(value)
    return value


def _skill_rows(skills):
    if isinstance(skills, (list, tuple)):
        return list(skills)
    if type(skills) is dict and skills.get("protocol") == "frozen_sequence_skills_v1":
        from orchestrator.sequence_learning import validate_sequence_skill_library
        return validate_sequence_skill_library(skills)["entries"]
    if type(skills) is dict and skills.get("kind") == "frozen_generated_skills_v1":
        # Do not invoke legacy retrieve(): it may follow validation file refs and
        # all of these programs have a physical-action contract, not sequence.
        if skills.get("content_sha256") != digest({k: v for k, v in skills.items() if k != "content_sha256"}):
            raise ValueError("legacy_skill_snapshot_hash_mismatch")
        return [dict(protocol="legacy_physical_action", skill_id=str(index),
                     source_sha256=digest(entry)) for index, entry in enumerate(skills["entries"])]
    raise ValueError("invalid_sequence_skill_input")


def _lesson_rows(lessons, *, allow_fixture=False):
    if not lessons:
        return []
    if type(lessons) is dict and lessons.get("protocol") == "frozen_sequence_lessons_v1":
        from orchestrator.sequence_learning import validate_sequence_lesson_library
        library = validate_sequence_lesson_library(lessons)
        if library["fixture"] and not allow_fixture:
            raise ValueError("fixture_sequence_lesson_not_allowed")
        rows = []
        for entry in library["entries"]:
            cited = set(entry["claim"]["evidence_ids"] + entry["claim"]["counterexample_ids"])
            selected = [fact for fact in entry["facts"] if fact["evidence_id"] in cited]
            rows.append(dict(lesson_id=entry["content_sha256"], claim=deepcopy(entry["claim"]),
                # Retain conservative full provenance for holdout exclusion.
                source_task_ids=sorted({fact["source_task"]["assembly_id"] for fact in entry["facts"]}),
                observed_facts=deepcopy(selected), source_packet_sha256=entry["source_packet_sha256"],
                fact_projection="claimed_and_counterexample_evidence_only_v1",
                source_fact_count=len(entry["facts"]), provided_fact_count=len(selected),
                omitted_fact_count=len(entry["facts"])-len(selected),
                original_facts_sha256=digest(entry["facts"]),
                claims_are_hypotheses=True, target_verified=False, current_part_id_mapping=None,
                hard_pruning=False, physics_reuse_allowed=False,
                provenance_check="hash_checked_source_files_not_reopened"))
        return rows
    library = checked(lessons, "frozen_lessons_v1")
    if library.get("hard_pruning") is not False or library.get("physics_reuse_allowed") is not False:
        raise ValueError("unsafe_sequence_lesson_library")
    rows = []
    for entry in library["entries"]:
        lesson = checked(entry["lesson"], "lesson_candidate_v1")
        validate_claim(lesson["claim"], lesson["observed_facts"])
        if lesson.get("claims_are_hypotheses") is not True:
            raise ValueError("unsafe_sequence_lesson_claim")
        # Existing lesson facts encode source_task either as an identity object
        # or as an explicit assembly name (e.g. "07009"). A name scopes a claim;
        # it does not supply missing mesh/solver identity or map source part IDs.
        tasks = []
        for fact in lesson["observed_facts"]:
            source = fact.get("source_task")
            tasks.append(source if type(source) is str and source.strip() else _source_task_id(source))
        rows.append(dict(lesson_id=lesson["content_sha256"], claim=deepcopy(lesson["claim"]),
            source_task_ids=tasks, observed_facts=deepcopy(lesson["observed_facts"]),
            source_packet_sha256=lesson["source_packet_sha256"],
            claims_are_hypotheses=True, target_verified=False, current_part_id_mapping=None,
            hard_pruning=False, physics_reuse_allowed=False,
            provenance_check="hash_checked_source_files_not_reopened"))
    return rows


def build_sequence_knowledge(envelope, memory_records=(), skills=(), limits=None, *, lessons=()):
    """Project frozen data into bounded ordering context; never modify inputs."""
    if type(envelope) is not dict:
        raise ValueError("invalid_sequence_envelope")
    options = deepcopy(LIMITS)
    if limits is not None:
        if type(limits) is not dict or not set(limits).issubset(LIMITS):
            raise ValueError("invalid_sequence_knowledge_limits")
        options.update(deepcopy(limits))
    for key in ("memory_enabled", "skills_enabled", "allow_fixture_skills", "exclude_current_task"):
        if type(options[key]) is not bool:
            raise ValueError("invalid_sequence_knowledge_flag")
    for key, maximum in (("max_records", 32), ("max_skills", 8), ("max_lessons", 8)):
        if type(options[key]) is not int or not 0 <= options[key] <= maximum:
            raise ValueError("invalid_sequence_knowledge_count")
    if type(options["max_context_bytes"]) is not int or not 1024 <= options["max_context_bytes"] <= 131072:
        raise ValueError("invalid_sequence_context_byte_limit")
    excluded = set(_ids(options["excluded_task_ids"], "excluded_task_ids"))
    request = envelope.get("request", envelope)
    features = envelope.get("features", {})
    if type(request) is not dict or type(features) is not dict:
        raise ValueError("invalid_sequence_envelope_fields")
    candidates = _ids(request.get("candidate_ids"), "candidate_ids")
    cad = features.get("cad", envelope.get("cad", {}))
    if type(cad) is not dict:
        raise ValueError("invalid_sequence_cad")
    task = _task(cad.get("task", request.get("task_context", {})))
    target = _source_task_id(task)
    if options["exclude_current_task"] and target:
        excluded.add(target)
    parts = cad.get("parts", {})
    if type(parts) is not dict:
        raise ValueError("invalid_sequence_cad_parts")
    current_features = {}
    for identity in candidates:
        part = parts.get(identity, {})
        if type(part) is not dict:
            raise ValueError("invalid_sequence_cad_part")
        current_features[identity] = {k: deepcopy(v) for k, v in part.items() if k in PART_FIELDS}
    context = dict(protocol=PROTOCOL, candidate_ids=candidates, task=task,
        current_features=current_features, memory_references=[], lesson_references=[], sequence_skills=[],
        evidence_ids=[], hard_pruning=False, physics_reuse_allowed=False,
        physics_owner="original_ASAP", order_is_not_physics_success=True,
        source_part_ids_are_not_current_ids=True)
    audit = dict(profile=PROFILE, excluded_task_ids=sorted(excluded), excluded_memory=[],
        excluded_skills=[], excluded_lessons=[], retrieval="source_and_observed_stage_round_robin_v1",
        memory_enabled=options["memory_enabled"], memory_content_sha256=None,
        no_source_files_opened=True, source_files_modified=False, model_calls=0, physics_calls=0,
        hard_pruning=False, physics_reuse_allowed=False, performance_effect="UNVERIFIED",
        actual_sequence_skill_status="NOT_YET_GENERATED")
    if options["memory_enabled"]:
        records, audit["memory_content_sha256"] = _records(memory_records)
        views = []
        for record in records:
            identity = _source_task_id(_record_context(record))
            reason = "source_task_unknown" if identity is None else "evaluation_task_excluded" if identity in excluded else None
            if reason:
                audit["excluded_memory"].append(dict(evidence_id=record["evidence_id"], reason=reason))
            else:
                views.append(_observation(record))
        for row in _balanced(views):
            if len(context["memory_references"]) >= options["max_records"]:
                audit["excluded_memory"].append(dict(evidence_id=row["evidence_id"], reason="record_limit"))
                continue
            context["memory_references"].append(row)
        for row in _lesson_rows(lessons, allow_fixture=options["allow_fixture_skills"]):
            reason = ("source_task_unknown" if not row["source_task_ids"] or None in row["source_task_ids"] else
                "evaluation_task_excluded" if excluded.intersection(row["source_task_ids"]) else
                "lesson_limit" if len(context["lesson_references"]) >= options["max_lessons"] else None)
            if reason:
                audit["excluded_lessons"].append(dict(lesson_id=row["lesson_id"], reason=reason))
            else:
                context["lesson_references"].append(row)
    if options["skills_enabled"]:
        for row in _skill_rows(skills):
            identity = row.get("skill_id", row.get("strategy_id", "unknown")) if type(row) is dict else "unknown"
            try:
                card = validate_sequence_skill(row, allow_fixture=options["allow_fixture_skills"])
                if excluded.intersection(card["source_task_ids"]):
                    raise ValueError("evaluation_task_excluded")
                if len(context["sequence_skills"]) >= options["max_skills"]:
                    raise ValueError("skill_limit")
                if any(old["skill_id"] == card["skill_id"] for old in context["sequence_skills"]):
                    raise ValueError("duplicate_sequence_skill_id")
                context["sequence_skills"].append(card)
            except (ValueError, TypeError, KeyError) as error:
                audit["excluded_skills"].append(dict(skill_id=identity, status="INCOMPATIBLE_OR_EXCLUDED",
                    reason=str(error), do_not_execute=True))
    def update_citations():
        context["evidence_ids"] = [row["evidence_id"] for row in context["memory_references"]]
        context["evidence_ids"].extend("lesson:" + row["lesson_id"] for row in context["lesson_references"])

    # Citation IDs are part of the serialized input budget, not a later add-on.
    # Rebuild them after each whole-object removal to avoid dangling references.
    update_citations()
    # Trim whole knowledge objects, never their condition metadata or program.
    while _size(context) > options["max_context_bytes"]:
        changed = False
        # Observations carry repeated source geometry and can dominate the
        # budget. Keep executable cards and their lesson explanations available
        # by reducing reference count first; never truncate any source conditions.
        for field, collection, identity in (("memory_references", "excluded_memory", "evidence_id"),
                ("lesson_references", "excluded_lessons", "lesson_id"),
                ("sequence_skills", "excluded_skills", "skill_id")):
            if context[field]:
                row = context[field].pop()
                audit[collection].append({identity: row[identity], "reason": "context_byte_limit"})
                update_citations()
                changed = True
                break
        if not changed:
            raise ValueError("sequence_current_input_exceeds_context_limit")
    bounded(context)
    audit.update(selected_memory_ids=[row["evidence_id"] for row in context["memory_references"]],
        selected_skill_ids=[row["skill_id"] for row in context["sequence_skills"]],
        context_sha256=digest(context), context_bytes=_size(context),
        missing_acceptance_ids=[row["evidence_id"] for row in context["memory_references"]
                                if row["accepted_for_transition"] is None or row["deadline_exceeded"] is None])
    if any(not row["fixture"] for row in context["sequence_skills"]):
        audit["actual_sequence_skill_status"] = "PROVIDED_REGISTERED_CARD_EFFECT_UNVERIFIED"
    return dict(context=context, audit=audit)


def _proposal(value, candidates, evidence):
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"candidate_order", "evidence_ids"}:
        raise ValueError("sequence_skill_output_fields")
    order = _ids(value["candidate_order"], "skill_candidate_order")
    if len(order) != len(candidates) or set(order) != set(candidates):
        raise ValueError("sequence_skill_requires_complete_current_permutation")
    citations = _ids(value["evidence_ids"], "skill_evidence_ids")
    if not set(citations).issubset(evidence):
        raise ValueError("sequence_skill_unknown_evidence")
    return dict(candidate_order=order, evidence_ids=citations)


def execute_sequence_skill(skill, candidate_ids, context, *, allow_fixture=False):
    try:
        card = validate_sequence_skill(skill, allow_fixture=allow_fixture)
    except (ValueError, TypeError, KeyError) as error:
        return _rejected_execution(error)
    result = execute_sequence_program(card["program"], candidate_ids, context)
    result["skill_id"] = card["skill_id"]
    return result


def _rejected_execution(error=None):
    return dict(status="REJECTED", candidate_order=None, evidence_ids=[], diagnostics=(
        [getattr(error, "diagnostic", {"code": str(error)})] if error is not None else []),
        physics_executed=False, current_attempt_verified=False, hard_pruning=False, vm_steps=0)


def execute_sequence_program(program, candidate_ids, context):
    """Run only pure ordering code, with current-ID equivariance and metering.

    A second execution renames current IDs while keeping historical IDs fixed.
    This catches foreign/hardcoded IDs even when they happen to match current
    labels. It is a contract probe, not physical validation or a general proof.
    """
    result = _rejected_execution()
    try:
        if type(program) is not dict or set(program) != {"language", "source"} or program["language"] != LANGUAGE:
            raise ValueError("invalid_sequence_python_program")
        candidates = _ids(candidate_ids, "candidate_ids")
        if (type(context) is not dict or context.get("protocol") != PROTOCOL
                or context.get("candidate_ids") != candidates
                or context.get("hard_pruning") is not False
                or context.get("physics_reuse_allowed") is not False):
            raise ValueError("sequence_skill_context_mismatch")
        allowed = set(_ids(context.get("evidence_ids", []), "context_evidence_ids"))
        # Programs do not see other executable source or their registration data.
        inputs = {key: deepcopy(context[key]) for key in ("protocol", "candidate_ids", "task",
            "current_features", "memory_references", "lesson_references", "evidence_ids") if key in context}
        bounded(inputs)

        def run(data, current):
            original = deepcopy(data)
            def eligible(value):
                try:
                    return _proposal(value, current, allowed) is not None
                except (ValueError, TypeError):
                    return False
            vm = VM(program["source"], eligible, max_steps=25000)
            value = vm.call("propose", [data])
            result["vm_steps"] += vm.steps
            if data != original:
                raise ValueError("sequence_skill_mutated_read_only_context")
            return _proposal(value, current, allowed)

        proposal = run(inputs, candidates)
        rename = {identity: "__current_sequence_part_%04d__" % (len(candidates) - index)
                  for index, identity in enumerate(candidates)}
        renamed = deepcopy(inputs)
        renamed["candidate_ids"] = [rename[identity] for identity in candidates]
        renamed["current_features"] = {rename[key]: deepcopy(value)
            for key, value in inputs.get("current_features", {}).items()}
        probe = run(renamed, renamed["candidate_ids"])
        expected = None if proposal is None else dict(
            candidate_order=[rename[identity] for identity in proposal["candidate_order"]],
            evidence_ids=proposal["evidence_ids"])
        if probe != expected:
            raise ValueError("sequence_skill_depends_on_part_labels_or_foreign_ids")
        result["status"] = "DECLINED" if proposal is None else "PROPOSED"
        if proposal is not None:
            result.update(proposal)
        result["program_sha256"] = digest(program)
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError, RecursionError) as error:
        result["diagnostics"].append(getattr(error, "diagnostic", {"code": str(error)}))
    return result
