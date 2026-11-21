"""Evidence -> hypotheses -> restricted order program -> scoped registration.

No provider, simulator or network call is made here. Registration reopens hashed
validation artifacts and replays the exact order program on recorded contexts;
it does not infer usefulness from a bare PASS. Native physics proves only its
recorded part-level plan. It never proves robot execution or a causal benefit.
"""
from copy import deepcopy
import hashlib
import json

from orchestrator.python_skill_vm import LANGUAGE, parse
from orchestrator.sequence_knowledge import (PROFILE, SKILL_PROTOCOL,
    validate_native_snapshot, execute_sequence_program, validate_sequence_skill,
    _native_observation, _task, _hash, _ids, _text, _size, _input_identity)
from worker.task_memory import digest, _safe_path


PACKET_PROTOCOL = "sequence_learning_packet_v1"
LESSON_PROTOCOL = "sequence_lesson_v1"
CANDIDATE_PROTOCOL = "sequence_candidate_v1"
PROOF_PROTOCOL = "sequence_validation_proof_v1"
TRACE_PROTOCOL = "sequence_candidate_trace_v1"
CLAIM_FIELDS = {"title", "hypothesis", "when_to_apply", "when_not_to_apply",
                "ranking_guidance", "evidence_ids", "counterexample_ids"}
CANDIDATE_RESPONSE_FIELDS = {"name", "description", "source", "evidence_ids"}
MAX_FACTS = 24
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024


def _signed(value):
    result = deepcopy(value)
    result["content_sha256"] = digest(result)
    return result


def _checked(value, protocol):
    if (type(value) is not dict or value.get("schema_version") != 1
            or value.get("protocol") != protocol
            or value.get("content_sha256") != digest({k: v for k, v in value.items() if k != "content_sha256"})):
        raise ValueError("sequence_learning_hash_or_protocol")
    return deepcopy(value)


def _provenance(value):
    if type(value) is not dict or value.get("origin") not in (
            "actual_llm", "injected_transport", "manual_authored"):
        raise ValueError("sequence_learning_provenance_required")
    if value["origin"] == "actual_llm":
        _text(value.get("model"), "generation_model")
        _text(value.get("request_id"), "generation_request_id")
        _hash(value.get("response_sha256"))
    _size(value)
    return deepcopy(value)


def build_sequence_learning_packet(native_snapshot, max_records=MAX_FACTS):
    """Select bounded native facts without promoting diagnoses or source IDs."""
    if type(max_records) is not int or not 1 <= max_records <= MAX_FACTS:
        raise ValueError("sequence_learning_fact_limit")
    snapshots = native_snapshot if isinstance(native_snapshot, (list, tuple)) else [native_snapshot]
    rows, sources, seen = [], [], set()
    for snapshot in snapshots:
        checked = validate_native_snapshot(snapshot)
        sources.append(checked["content_sha256"])
        for record in checked["records"]:
            if record["evidence_id"] in seen:
                raise ValueError("duplicate_sequence_learning_fact")
            seen.add(record["evidence_id"])
            observed = _native_observation(record)
            attempt = record["attempt"]
            geometry = record["source_task_context"].get("geometry_features")
            if geometry is not None and type(geometry) is not dict:
                raise ValueError("invalid_sequence_source_geometry")
            part_features = (geometry or {}).get(attempt["part_move"])
            if part_features is not None and type(part_features) is not dict:
                raise ValueError("invalid_sequence_source_part_features")
            rows.append(dict(evidence_id=record["evidence_id"],
                source_task=_task(record["source_task_context"]),
                source_part_features=deepcopy(part_features),
                source_part_features_status="AVAILABLE" if part_features is not None else "NOT_AVAILABLE",
                source_state=dict(remaining_parts=deepcopy(attempt["remaining_parts"]),
                    part_move=attempt["part_move"], request=deepcopy(attempt["request"])),
                observed={key: deepcopy(observed[key]) for key in ("status", "path_success",
                    "stability_success", "observed_stage", "accepted_for_transition", "deadline_exceeded")},
                native_result=deepcopy(attempt.get("raw_sim_info")),
                provenance=dict(source_ref=record["source_ref"], source_sha256=record["source_sha256"],
                    record_sha256=record["record_sha256"], source_observation_id=record["source_observation_id"],
                    verifier_domain=attempt["verifier_domain"], source_hashes=deepcopy(attempt.get("source_hashes"))),
                blocker=None, failure_cause=None, current_part_id_mapping=None))
    if not rows:
        raise ValueError("sequence_learning_requires_native_observations")
    # Interleave measured success and failure per assembly, independent of a
    # candidate's promised effectiveness. Unselected facts remain in the source.
    buckets = {}
    for row in sorted(rows, key=lambda r: r["evidence_id"]):
        key = (row["source_task"]["assembly_id"], row["observed"]["observed_stage"])
        buckets.setdefault(key, []).append(row)
    selected = []
    while buckets and len(selected) < max_records:
        for key in sorted(buckets):
            selected.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
            if len(selected) == max_records:
                break
    excluded = [row["evidence_id"] for row in rows if row not in selected]
    packet = _signed(dict(schema_version=1, protocol=PACKET_PROTOCOL, facts=selected,
        source_snapshot_sha256=sources, source_fact_count=len(rows), selected_fact_count=len(selected),
        excluded_evidence_ids=excluded[:64], excluded_evidence_count=len(excluded),
        excluded_evidence_sha256=digest(excluded), excluded_ids_are_prefix=len(excluded) > 64,
        claims_are_hypotheses=True, hard_pruning=False, physics_reuse_allowed=False,
        target_success="UNVERIFIED", output_scope="complete_current_candidate_permutation_only"))
    return validate_sequence_learning_packet(packet)


def validate_sequence_learning_packet(packet):
    packet = _checked(packet, PACKET_PROTOCOL)
    facts = packet.get("facts")
    if (type(facts) is not list or not 1 <= len(facts) <= MAX_FACTS
            or packet.get("selected_fact_count") != len(facts)
            or type(packet.get("source_fact_count")) is not int
            or packet["source_fact_count"] < len(facts)
            or packet.get("claims_are_hypotheses") is not True
            or packet.get("hard_pruning") is not False or packet.get("physics_reuse_allowed") is not False
            or packet.get("target_success") != "UNVERIFIED"
            or packet.get("output_scope") != "complete_current_candidate_permutation_only"):
        raise ValueError("invalid_sequence_learning_packet")
    _ids([fact["evidence_id"] for fact in facts], "learning_evidence_ids")
    _ids(packet["excluded_evidence_ids"], "excluded_learning_evidence")
    excluded_count = packet.get("excluded_evidence_count")
    if (type(excluded_count) is not int or excluded_count < 0
            or excluded_count + len(facts) != packet["source_fact_count"]
            or len(packet["excluded_evidence_ids"]) != min(64, excluded_count)
            or packet.get("excluded_ids_are_prefix") is not (excluded_count > 64)):
        raise ValueError("sequence_learning_fact_count_mismatch")
    _hash(packet.get("excluded_evidence_sha256"))
    if excluded_count <= 64 and packet["excluded_evidence_sha256"] != digest(packet["excluded_evidence_ids"]):
        raise ValueError("sequence_learning_exclusion_hash_mismatch")
    for source in packet["source_snapshot_sha256"]:
        _hash(source)
    for fact in facts:
        task, obs, prov = fact["source_task"], fact["observed"], fact["provenance"]
        _text(task.get("assembly_id"), "learning_source_task")
        _input_identity(task.get("input_sha256"))
        _hash(prov["source_sha256"]); _hash(prov["record_sha256"])
        if (prov.get("verifier_domain") != "ASAP_RedMax" or obs.get("status") not in ("PASS", "UNKNOWN", "ERROR")
                or fact.get("blocker") is not None or fact.get("failure_cause") is not None
                or fact.get("current_part_id_mapping") is not None):
            raise ValueError("unsafe_sequence_learning_fact")
        if obs.get("accepted_for_transition") is True and (
                obs.get("status") != "PASS" or obs.get("deadline_exceeded") is not False):
            raise ValueError("sequence_learning_accepted_requires_timely_pass")
        # Historical packets without this optional pair remain readable. New
        # packets copy only the chosen SOURCE part's recorded CAD observations;
        # these are neither current part correspondences nor physics verdicts.
        if "source_part_features" in fact or "source_part_features_status" in fact:
            features, status = fact.get("source_part_features"), fact.get("source_part_features_status")
            if not ((features is None and status == "NOT_AVAILABLE")
                    or (type(features) is dict and status == "AVAILABLE")):
                raise ValueError("sequence_source_geometry_availability_mismatch")
    _size(packet)
    return packet


def _claim(packet, claim):
    if type(claim) is not dict or set(claim) != CLAIM_FIELDS:
        raise ValueError("invalid_sequence_lesson_claim_fields")
    for field in CLAIM_FIELDS - {"evidence_ids", "counterexample_ids"}:
        _text(claim[field], "lesson_" + field, 256 if field == "title" else 1500)
    allowed = {fact["evidence_id"] for fact in packet["facts"]}
    for field in ("evidence_ids", "counterexample_ids"):
        ids = _ids(claim[field], "lesson_" + field)
        if len(ids) > MAX_FACTS or not set(ids).issubset(allowed):
            raise ValueError("sequence_lesson_unobserved_evidence")
    if not claim["evidence_ids"]:
        raise ValueError("sequence_lesson_requires_evidence")
    return deepcopy(claim)


def validate_sequence_lesson(packet, claim, provenance):
    packet = validate_sequence_learning_packet(packet)
    claim, provenance = _claim(packet, claim), _provenance(provenance)
    return _signed(dict(schema_version=1, protocol=LESSON_PROTOCOL, claim=claim,
        facts=deepcopy(packet["facts"]), source_packet_sha256=packet["content_sha256"],
        packet=packet, provenance=provenance, claims_are_hypotheses=True,
        hard_pruning=False, physics_reuse_allowed=False, usefulness="UNVERIFIED"))


def check_sequence_lesson(lesson):
    lesson = _checked(lesson, LESSON_PROTOCOL)
    expected = validate_sequence_lesson(lesson["packet"], lesson["claim"], lesson["provenance"])
    if expected != lesson:
        raise ValueError("sequence_lesson_packet_binding_mismatch")
    return lesson


def make_sequence_candidate(lesson, response, provenance):
    """Preserve rejected code as data; never silently normalize model output."""
    lesson = check_sequence_lesson(lesson)
    provenance = _provenance(provenance)
    _size(response)
    value = dict(schema_version=1, protocol=CANDIDATE_PROTOCOL, status="CANDIDATE",
        raw_candidate_response=deepcopy(response), lesson_sha256=lesson["content_sha256"],
        lesson=lesson, source_task_ids=sorted({f["source_task"]["assembly_id"] for f in lesson["facts"]}),
        source_evidence_ids=[], program=None, program_sha256=None, provenance=provenance,
        rejection=None, registered=False, performance_effect="UNVERIFIED")
    try:
        if type(response) is not dict or set(response) != CANDIDATE_RESPONSE_FIELDS:
            raise ValueError("invalid_sequence_candidate_response_fields")
        _text(response["name"], "candidate_name")
        _text(response["description"], "candidate_description", 1500)
        ids = _ids(response["evidence_ids"], "candidate_evidence_ids")
        if not ids or len(ids) > MAX_FACTS or not set(ids).issubset({f["evidence_id"] for f in lesson["facts"]}):
            raise ValueError("sequence_candidate_requires_observed_evidence")
        value["source_evidence_ids"] = ids
        value["program"] = dict(language=LANGUAGE, source=response["source"])
        value["program_sha256"] = digest(value["program"])
        parse(response["source"])
    except (ValueError, TypeError, KeyError) as error:
        value["status"] = "REJECTED"
        value["rejection"] = getattr(error, "diagnostic", dict(code=str(error)))
    return _signed(value)


def validate_sequence_candidate(candidate, *, require_executable=True):
    candidate = _checked(candidate, CANDIDATE_PROTOCOL)
    expected = make_sequence_candidate(candidate["lesson"], candidate["raw_candidate_response"], candidate["provenance"])
    if expected != candidate or candidate.get("registered") is not False:
        raise ValueError("sequence_candidate_binding_mismatch")
    if require_executable and candidate["status"] != "CANDIDATE":
        raise ValueError("rejected_sequence_candidate_cannot_execute")
    return candidate


def execute_sequence_candidate(candidate, candidate_ids, context):
    """Validation-only execution without fabricating a registered skill card."""
    candidate = validate_sequence_candidate(candidate)
    result = execute_sequence_program(candidate["program"], candidate_ids, context)
    result.update(validation_role="sequence_candidate", registered=False,
        candidate_sha256=candidate["content_sha256"], context_sha256=digest(context))
    return result


def write_sequence_artifact(path, value, *, allow_test_root=False):
    """Exclusive creation, including raw rejected model code; no overwrite."""
    target = _safe_path(path, allow_test_root)
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    if len(content) > MAX_ARTIFACT_BYTES:
        raise ValueError("sequence_artifact_size_limit")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(content)
    return dict(path=str(target), sha256=hashlib.sha256(content).hexdigest())


def _read_artifact(ref, *, allow_test_root=False):
    if type(ref) is not dict or set(ref) != {"path", "sha256"}:
        raise ValueError("sequence_validation_requires_artifact_ref")
    _hash(ref["sha256"])
    path = _safe_path(ref["path"], allow_test_root)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("sequence_validation_artifact_size")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("sequence_validation_artifact_hash_mismatch")
    return json.loads(data.decode("utf-8"))


def _worker_success(worker, *, allow_fixture):
    if (type(worker) is not dict or worker.get("profile") != PROFILE
            or worker.get("status") != "PASS" or worker.get("success") is not True
            or worker.get("error") is not None or worker.get("decision_halt_reason") is not None
            or worker.get("lower_level_policy") != "original_ASAP"
            or worker.get("hard_pruning") is not False or worker.get("complete_permutation") is not True
            or worker.get("heuristic_tripwire_calls") != 0
            or worker.get("source_methods_restored") is not True
            or worker.get("sources_unchanged") is not True or worker.get("input_unchanged") is not True
            or worker.get("full_robot_execution") is not False):
        raise ValueError("sequence_validation_requires_complete_native_run")
    actual = worker.get("actual_physics")
    if type(actual) is not bool or (not actual and not allow_fixture):
        raise ValueError("sequence_validation_actual_physics_required")
    if not actual and worker.get("fixture") is not True:
        raise ValueError("sequence_validation_fixture_label_required")
    binding = worker.get("physics_binding")
    if type(binding) is not dict or "redmax_binary_sha256" not in binding or "physics_source_sha256" not in binding:
        raise ValueError("sequence_validation_native_physics_binding_required")
    for value in binding.values():
        _hash(value)
    methods = worker.get("source_methods_sha256")
    if type(methods) is not dict or not {"simulate", "update_tree", "path_check", "stability_check"}.issubset(methods):
        raise ValueError("sequence_validation_native_methods_required")
    for value in methods.values():
        _hash(value)
    connected = worker.get("connected", {})
    chain = connected.get("complete_chain")
    if (connected.get("success") is not True or type(chain) is not list or not chain
            or worker.get("selected_plan") != chain):
        raise ValueError("sequence_validation_requires_connected_complete_chain")
    initial = chain[0]["parent"]
    if len(chain) != len(initial) - 1:
        raise ValueError("sequence_validation_chain_incomplete")
    parent = list(initial)
    for edge in chain:
        child, mover = edge["child"], edge["part_move"]
        if (edge["parent"] != parent or set(parent) - set(child) != {mover}
                or len(parent) != len(child) + 1 or not set(child).issubset(parent)
                or edge.get("sim_info", {}).get("feasible") is not True):
            raise ValueError("sequence_validation_chain_disconnected")
        matching = [a for a in worker.get("attempts", []) if a.get("remaining_parts") == sorted(parent)
            and a.get("part_move") == mover and a.get("accepted_for_transition") is True
            and a.get("deadline_exceeded") is False and a.get("status") == "PASS"
            and a.get("verifier_domain") == "ASAP_RedMax"
            and a.get("raw_sim_info") == edge["sim_info"]]
        if not matching:
            raise ValueError("sequence_validation_chain_missing_accepted_physics")
        parent = list(child)
    if len(parent) != 1:
        raise ValueError("sequence_validation_chain_incomplete")
    return actual


def validate_sequence_registration(candidate, proof, *, allow_fixture=False, min_cases=2,
                                   allow_test_root=False):
    """Admit a version only after bound, complete native validation runs.

    Hash checks detect changed local artifacts, not hostile fabrication of an
    entire simulator and its logs. Real admission uses trusted worker-produced
    files. A fixture or manually-authored generation remains a fixture even when
    it was evaluated with real physics. Registration is not a benefit claim.
    """
    candidate = validate_sequence_candidate(candidate)
    if type(min_cases) is not int or min_cases < 2:
        raise ValueError("sequence_registration_requires_two_conditions")
    if (type(proof) is not dict or proof.get("protocol") != PROOF_PROTOCOL
            or proof.get("candidate_sha256") != candidate["content_sha256"]
            or proof.get("program_sha256") != candidate["program_sha256"]
            or type(proof.get("cases")) is not list or not min_cases <= len(proof["cases"]) <= 16):
        raise ValueError("invalid_sequence_registration_proof")
    fixture = candidate["provenance"]["origin"] != "actual_llm"
    if fixture and not allow_fixture:
        raise ValueError("fixture_sequence_generation_not_allowed")
    cases, conditions, case_ids, run_hashes, refs, hashes, evidence = [], set(), set(), set(), [], [], []
    source_tasks = set(candidate["source_task_ids"])
    for case in proof["cases"]:
        if type(case) is not dict or set(case) != {"case_id", "worker_result", "decision_trace"}:
            raise ValueError("invalid_sequence_validation_case")
        case_id = _text(case["case_id"], "validation_case_id")
        if case_id in case_ids or case["worker_result"]["sha256"] in run_hashes:
            raise ValueError("duplicate_sequence_validation_case_or_run")
        case_ids.add(case_id); run_hashes.add(case["worker_result"]["sha256"])
        worker = _read_artifact(case["worker_result"], allow_test_root=allow_test_root)
        actual = _worker_success(worker, allow_fixture=allow_fixture)
        fixture = fixture or not actual
        trace = _read_artifact(case["decision_trace"], allow_test_root=allow_test_root)
        if (trace.get("protocol") != TRACE_PROTOCOL or trace.get("candidate_sha256") != candidate["content_sha256"]
                or trace.get("program_sha256") != candidate["program_sha256"]
                or type(trace.get("calls")) is not list or not trace["calls"]):
            raise ValueError("sequence_candidate_trace_required")
        decisions = worker.get("rank_decisions")
        if type(decisions) is not list or len(decisions) != len(trace["calls"]):
            raise ValueError("sequence_candidate_trace_must_cover_all_decisions")
        task = None
        seen_requests = set()
        for call, decision in zip(trace["calls"], decisions):
            if type(call) is not dict or set(call) != {"request", "context", "response_ref"}:
                raise ValueError("invalid_sequence_candidate_call")
            request, context = call["request"], call["context"]
            if request != decision.get("request") or len(request["assembled_parts"]) <= 2:
                raise ValueError("sequence_candidate_nonterminal_request_binding")
            assembled = _ids(request["assembled_parts"], "validation_assembled_parts")
            expected_candidates = sorted(part for part in assembled if part != request.get("base_part"))
            if request.get("candidate_ids") != expected_candidates:
                raise ValueError("sequence_validation_requires_all_native_candidates")
            if request["request_id"] in seen_requests:
                raise ValueError("duplicate_sequence_candidate_request")
            seen_requests.add(request["request_id"])
            outcome = execute_sequence_candidate(candidate, request["candidate_ids"], context)
            if outcome["status"] != "PROPOSED":
                raise ValueError("sequence_candidate_must_propose_during_validation")
            response = _read_artifact(call["response_ref"], allow_test_root=allow_test_root)
            metadata = response.get("metadata", {})
            if (response.get("request_id") != request["request_id"] or response.get("state_id") != request["state_id"]
                    or response.get("fallback_reason") is not None
                    or response.get("candidate_ids") != outcome["candidate_order"]
                    or decision.get("applied_order") != outcome["candidate_order"]
                    or decision.get("full_candidate_set") is not True
                    or decision.get("heuristic_or_gnn_invoked") is not False
                    or decision.get("rng_before") != decision.get("rng_after")
                    or metadata.get("validation_role") != "sequence_candidate"
                    or metadata.get("candidate_sha256") != candidate["content_sha256"]
                    or metadata.get("program_sha256") != candidate["program_sha256"]
                    or metadata.get("context_sha256") != digest(context)):
                raise ValueError("sequence_candidate_execution_binding_mismatch")
            current = context.get("task")
            if type(current) is not dict or not current.get("assembly_id"):
                raise ValueError("sequence_validation_task_required")
            _input_identity(current.get("input_sha256"))
            if task is not None and current != task:
                raise ValueError("sequence_validation_task_changed_mid_run")
            task = deepcopy(current)
        config = worker.get("configuration", {})
        if (type(config.get("seed")) is not int or config.get("automatic_retries") != 0
                or config.get("candidate_pruning") is not False or config.get("physics_verdict_reuse") is not False):
            raise ValueError("sequence_validation_settings_required")
        if task.get("solver_settings", {}).get("seed") != config["seed"]:
            raise ValueError("sequence_validation_context_seed_mismatch")
        # Run labels/output paths do not make an independent condition. Only
        # physical task identity, native settings and physics implementation do.
        condition_config = {key: deepcopy(config.get(key)) for key in ("seed", "base_part", "budget",
            "path_limit", "stability_limit", "timeout", "max_gripper", "max_pose", "pose_reuse",
            "num_proc", "early_term")}
        condition = digest(dict(task=dict(assembly_id=task["assembly_id"],
            input_sha256=task["input_sha256"], base_part=task.get("base_part")),
            config=condition_config, physics_binding=worker["physics_binding"]))
        if condition in conditions:
            raise ValueError("sequence_registration_requires_distinct_conditions")
        conditions.add(condition); source_tasks.add(task["assembly_id"])
        for key in ("worker_result", "decision_trace"):
            refs.append(case[key]["path"]); hashes.append(case[key]["sha256"])
        case_evidence = [a["event_id"] for a in worker["attempts"] if a.get("accepted_for_transition") is True]
        evidence.append("validation-case:" + condition)
        cases.append(dict(case_id=case_id, task=task, configuration=config, actual_physics=actual,
            validation_calls=len(trace["calls"]), complete_native_plan=True, condition_sha256=condition,
            accepted_evidence_ids=case_evidence))
    # The compact card points to the immutable proof and its observed scope.
    # Readers must not interpret REGISTERED as better performance or generality.
    response = candidate["raw_candidate_response"]
    card = dict(schema_version=1, protocol=SKILL_PROTOCOL,
        skill_id="sequence:" + candidate["program_sha256"], name=response["name"], description=response["description"],
        program=deepcopy(candidate["program"]), program_sha256=candidate["program_sha256"],
        source_task_ids=sorted(source_tasks), fixture=fixture,
        validation=dict(status="REGISTERED", profile=PROFILE, program_sha256=candidate["program_sha256"],
            run_refs=list(dict.fromkeys(refs)), source_hashes=list(dict.fromkeys(hashes)),
            evidence_ids=list(dict.fromkeys(evidence)),
            scope="offline_contract_fixture" if fixture else "sequence_contract_and_native_physics",
            performance_effect="UNVERIFIED"))
    card = _signed(card)
    validate_sequence_skill(card, allow_fixture=allow_fixture)
    return dict(card=card, admission=_signed(dict(schema_version=1, protocol="sequence_admission_v1",
        candidate_sha256=candidate["content_sha256"], program_sha256=candidate["program_sha256"],
        proof_sha256=digest(proof), cases=cases, fixture=fixture, status="REGISTERED_SCOPED",
        generalization="UNVERIFIED", causal_benefit="NOT_TESTED", robot_execution="NOT_TESTED")))


def freeze_sequence_lessons(lessons):
    entries = [check_sequence_lesson(value) for value in lessons]
    if len({row["content_sha256"] for row in entries}) != len(entries):
        raise ValueError("duplicate_sequence_lesson")
    return _signed(dict(schema_version=1, protocol="frozen_sequence_lessons_v1", entries=entries,
        fixture=any(row["provenance"]["origin"] != "actual_llm" for row in entries),
        hard_pruning=False, physics_reuse_allowed=False))


def validate_sequence_lesson_library(library):
    library = _checked(library, "frozen_sequence_lessons_v1")
    if freeze_sequence_lessons(library["entries"]) != library:
        raise ValueError("sequence_lesson_library_binding")
    return library


def freeze_sequence_skills(cards):
    entries = [validate_sequence_skill(card, allow_fixture=True) for card in cards]
    if len({row["skill_id"] for row in entries}) != len(entries):
        raise ValueError("duplicate_sequence_skill")
    return _signed(dict(schema_version=1, protocol="frozen_sequence_skills_v1", entries=entries,
        fixture=any(row["fixture"] for row in entries), hard_pruning=False, physics_reuse_allowed=False))


def validate_sequence_skill_library(library):
    library = _checked(library, "frozen_sequence_skills_v1")
    if freeze_sequence_skills(library["entries"]) != library:
        raise ValueError("sequence_skill_library_binding")
    return library
