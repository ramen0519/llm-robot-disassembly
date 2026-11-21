"""Evidence-grounded lessons. Natural-language claims remain hypotheses.

No model or simulator is called here. A lesson never certifies a target action.
Legacy memory and registry bytes remain authoritative and are not rewritten.
"""
from copy import deepcopy
from pathlib import Path
import json

from worker.task_memory import _safe_path, digest
from orchestrator.skill_registry import retrieve

KINDS = ("path_failed", "stability_failed", "plan_pose_guard_v1", "unknown")
FIELDS = {"title", "hypothesis", "when_to_apply", "when_not_to_apply",
          "evidence_ids", "counterexample_ids", "roles", "trigger_kind"}
TEXT_FIELDS = ("title", "hypothesis", "when_to_apply", "when_not_to_apply")
MAX_TEXT = 1500
MAX_ROLE_TEXT = 400
MAX_ROLES = 8
NONBLANK_PATTERN = r"\S"


class LessonClaimError(ValueError):
    """Keep legacy error strings while exposing value-free field diagnostics."""
    def __init__(self, category, code, field, index=None):
        super().__init__(category)
        self.diagnostic = dict(schema_version=1, phase='claim', category=category,
                               code=code, field=field, index=index, raw_values_recorded=False)


def claim_contract(facts):
    """Public generation rules drawn from the same constants as validation."""
    return dict(version='lesson_claim_contract_v1', required_fields=sorted(FIELDS),
        text_fields=list(TEXT_FIELDS), text_max_characters=MAX_TEXT, nonblank_text=True,
        roles_min_items=1, roles_max_items=MAX_ROLES, role_max_characters=MAX_ROLE_TEXT,
        citations=dict(allowed_ids=[f['evidence_id'] for f in facts],
            unique_within_each_list=True, evidence_min_items=1, counterexample_min_items=0,
            evidence_and_counterexample_overlap_allowed=True),
        trigger_evidence_ids={kind: [f['evidence_id'] for f in facts if f['observed_kind'] == kind]
                              for kind in KINDS if any(f['observed_kind'] == kind for f in facts)},
        trigger_rule='At least one evidence_ids entry must belong to trigger_evidence_ids[trigger_kind]. '
                     'An ID only in counterexample_ids does not satisfy this rule.',
        authoritative_check='local validate_claim; schema alone does not enforce all semantic rules',
        automatic_normalization=False)


def sign(value):
    result = deepcopy(value)
    result["content_sha256"] = digest(result)
    return result


def checked(value, kind):
    if (type(value) is not dict or value.get("kind") != kind
            or value.get("content_sha256") != digest({k: v for k, v in value.items()
                                                       if k != "content_sha256"})):
        raise ValueError("lesson_hash_or_kind_mismatch")
    return deepcopy(value)


def write_new(path, value, *, allow_test_root=False):
    path = _safe_path(path, allow_test_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def evidence_view(packet):
    """Allowlisted observations, not an LLM-written diagnosis or successful answer."""
    rows = []
    for item in packet["evidence"]:
        # All verbatim observations keep their source hash; no prose is promoted
        # to fact. Missing stage/cause fields stay null.
        observed = item.get("result") if type(item.get("result")) is dict else item
        row = {k: deepcopy(item.get(k)) for k in (
            "evidence_id", "source_task", "status", "path_success", "stability_success")}
        for key in ("status", "path_success", "stability_success"):
            row[key] = deepcopy(observed.get(key))
        row["source_sha256"] = digest(item)
        if 'execution' in observed:
            from orchestrator.execution_observation import project_execution
            row.update(project_execution(observed))
        guard = item.get("kind") == "plan_pose_guard_v1"
        row["observed_kind"] = ("plan_pose_guard_v1" if guard else
            "stability_failed" if row["path_success"] is True and row["stability_success"] is False else
            "path_failed" if row["path_success"] is False else "unknown")
        row.update(physical_attempt_executed=False if guard else None,
                   reason=item.get("reason") if guard else None,
                   blocker=None, failure_cause=None, source_part_ids_are_target_ids=False)
        rows.append(row)
    if (not rows or any(type(r["evidence_id"]) is not str for r in rows)
            or len({r["evidence_id"] for r in rows}) != len(rows)):
        raise ValueError("unique_lesson_evidence_required")
    return rows


def validate_claim(claim, facts):
    if type(claim) is not dict or set(claim) != FIELDS:
        raise LessonClaimError('invalid_lesson_fields', 'unexpected_or_missing_fields', 'lesson')
    for key in TEXT_FIELDS:
        if type(claim[key]) is not str:
            raise LessonClaimError('invalid_lesson_text', 'text_type', key)
        if not claim[key].strip():
            raise LessonClaimError('invalid_lesson_text', 'blank_text', key)
        if len(claim[key]) > MAX_TEXT:
            raise LessonClaimError('invalid_lesson_text', 'text_too_long', key)
    allowed = {r["evidence_id"] for r in facts}
    for key in ("evidence_ids", "counterexample_ids"):
        ids = claim[key]
        if type(ids) is not list:
            raise LessonClaimError('invented_or_duplicate_lesson_evidence', 'citation_list_type', key)
        seen = set()
        for index, identity in enumerate(ids):
            if type(identity) is not str:
                raise LessonClaimError('invented_or_duplicate_lesson_evidence', 'citation_id_type', key, index)
            if identity in seen:
                raise LessonClaimError('invented_or_duplicate_lesson_evidence', 'duplicate_evidence_id', key, index)
            if identity not in allowed:
                raise LessonClaimError('invented_or_duplicate_lesson_evidence', 'unknown_evidence_id', key, index)
            seen.add(identity)
    if not claim["evidence_ids"]:
        raise LessonClaimError('lesson_requires_evidence', 'empty_evidence', 'evidence_ids')
    cited = [r for r in facts if r["evidence_id"] in claim["evidence_ids"]]
    if claim['trigger_kind'] not in KINDS:
        raise LessonClaimError('lesson_trigger_not_observed', 'unknown_trigger_kind', 'trigger_kind')
    if not any(r["observed_kind"] == claim["trigger_kind"] for r in cited):
        raise LessonClaimError('lesson_trigger_not_observed', 'trigger_not_in_cited_evidence', 'trigger_kind')
    roles = claim["roles"]
    if type(roles) is not list or not 1 <= len(roles) <= MAX_ROLES:
        raise LessonClaimError('invalid_lesson_role_hypotheses', 'roles_type_or_count', 'roles')
    for index, role in enumerate(roles):
        if type(role) is not str or not role.strip() or len(role) > MAX_ROLE_TEXT:
            raise LessonClaimError('invalid_lesson_role_hypotheses', 'role_type_blank_or_length', 'roles', index)
    return deepcopy(claim)


def make_lesson(claim, packet, provenance, previous=None):
    facts = evidence_view(packet)
    claim = validate_claim(claim, facts)
    if provenance.get("origin") not in ("actual_llm", "injected_transport", "manual_authored"):
        raise ValueError("explicit_lesson_origin_required")
    previous_hash = checked(previous, "lesson_candidate_v1")["content_sha256"] if previous else None
    return sign(dict(schema_version=1, kind="lesson_candidate_v1", claim=claim,
        observed_facts=facts, source_packet_sha256=digest(packet), provenance=deepcopy(provenance),
        previous_version=previous_hash, status="candidate", claims_are_hypotheses=True,
        hard_pruning=False, physics_reuse_allowed=False, generalization="UNVERIFIED",
        human_solution_constraints=deepcopy(packet.get("context", {}).get("human_solution_constraints", []))))


def validate_lesson(value):
    value = checked(value, "lesson_candidate_v1")
    validate_claim(value["claim"], value["observed_facts"])
    if (value.get("status") != "candidate" or value.get("claims_are_hypotheses") is not True
            or value.get("hard_pruning") is not False or value.get("physics_reuse_allowed") is not False):
        raise ValueError("unsafe_lesson_claim")
    origin = value.get("provenance", {}).get("origin")
    if origin not in ("actual_llm", "injected_transport", "manual_authored"):
        raise ValueError("explicit_lesson_origin_required")
    if origin == "actual_llm":
        import hashlib
        provenance = value["provenance"]
        objects = {}
        for name in ("response", "packet"):
            data = _safe_path(provenance[name + "_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != provenance[name + "_sha256"]:
                raise ValueError("lesson_source_file_changed")
            objects[name] = json.loads(data)
        response, packet = objects["response"], objects["packet"]
        if (response.get("origin") != "actual_llm" or response.get("status") != "proposed"
                or response.get("lesson") != value["claim"]
                or response.get("metadata", {}).get("llm_live") is not True
                or digest(packet) != value["source_packet_sha256"]
                or evidence_view(packet) != value["observed_facts"]):
            raise ValueError("lesson_actual_provenance_mismatch")
    if 'repair_binding' in value:
        from orchestrator.lesson_consolidation import validate_final_lesson
        validate_final_lesson(value)
    return value


def freeze_lessons(lessons, skills, *, allow_fixture=False):
    """Link only registry-admitted versions. Physical evidence never proves prose."""
    rows = retrieve(skills, allow_fixture=allow_fixture)
    entries = []
    for lesson in lessons:
        lesson = validate_lesson(lesson)
        from orchestrator.lesson_runtime import CONTRACT
        binding = lesson.get('repair_binding')
        matches = [r for r in rows if
            (r['card'].get('runtime_contract') != CONTRACT and
             r["card"].get("lesson_id") == lesson["content_sha256"]) or
            (binding and r['card'].get('runtime_contract') == CONTRACT and
             r['card'].get('lesson_id') == binding['draft_id'] and
             r['program_hash'] == binding['program_hash'])]
        if not binding and any(r['card'].get('runtime_contract') == CONTRACT and
                              r['card'].get('lesson_id') == lesson['content_sha256'] for r in rows):
            raise ValueError('new_runtime_requires_post_validation_lesson')
        if not allow_fixture and lesson["provenance"]["origin"] != "actual_llm":
            raise ValueError("fixture_lesson_cannot_enter_live_library")
        if not matches:
            continue
        entries.append(dict(lesson=lesson, program_hashes=[r["program_hash"] for r in matches],
            validation_scope=[dict(program_hash=r["program_hash"], cases=r["validation_scope"])
                              for r in matches],
            status="code_validated_for_scope", lesson_transfer_test="NOT_RUN",
            causal_claim_verified=False, usefulness="NOT_RUN"))
    return sign(dict(schema_version=1, kind="frozen_lessons_v1", entries=entries,
        skills_content_sha256=skills["content_sha256"], fixture=skills["fixture"],
        hard_pruning=False, physics_reuse_allowed=False))


def validate_library(library, skills, *, allow_fixture=False):
    library = checked(library, "frozen_lessons_v1")
    if (library.get("skills_content_sha256") != skills["content_sha256"]
            or library.get("fixture") != skills["fixture"]
            or library.get("hard_pruning") is not False or library.get("physics_reuse_allowed") is not False):
        raise ValueError("lesson_library_binding_mismatch")
    expected = freeze_lessons([e["lesson"] for e in library["entries"]], skills,
                              allow_fixture=allow_fixture)
    if library != expected:
        raise ValueError("lesson_admission_or_scope_changed")
    return library


def lesson_cards(library, trigger=None, limit=3):
    """Bounded hypotheses only; no code or target-specific success answer."""
    library = checked(library, "frozen_lessons_v1")
    if type(limit) is not int or not 0 <= limit <= 8:
        raise ValueError("invalid_lesson_limit")
    entries = sorted(library["entries"], key=lambda e: (
        e["lesson"]["claim"]["trigger_kind"] != trigger, e["lesson"]["content_sha256"]))
    return [dict(lesson_id=e["lesson"]["content_sha256"], **deepcopy(e["lesson"]["claim"]),
        observed_facts=deepcopy(e["lesson"]["observed_facts"]),
        program_hashes=list(e["program_hashes"]), claims_are_hypotheses=True,
        validated_repair_observations=deepcopy(e['lesson'].get('repair_binding', {}).get('repair_observations', [])),
        target_verified=False, validation_scope=deepcopy(e["validation_scope"])) for e in entries[:limit]]


def published_card(row, library):
    """Replace a new code version's draft card with its finalized explanation."""
    from orchestrator.lesson_runtime import CONTRACT
    if row['card'].get('runtime_contract') != CONTRACT:
        return deepcopy(row['card'])
    finalized = [e['lesson'] for e in (library or {}).get('entries', [])
                 if e['lesson'].get('repair_binding', {}).get('program_hash') == row['program_hash']]
    if len(finalized) != 1:
        raise ValueError('runtime_skill_requires_finalized_lesson')
    lesson = finalized[0]
    return dict(name=lesson['claim']['title'], applicability=lesson['claim']['when_to_apply'],
        hypothesis=lesson['claim']['hypothesis'], lesson_id=lesson['content_sha256'],
        claims_are_hypotheses=True, runtime_contract=CONTRACT,
        validated_repair_observations=deepcopy(lesson['repair_binding']['repair_observations']))


def with_prior_knowledge(packet, skills_path, lessons_path=None, *, allow_fixture=False):
    """Bounded prior training knowledge for NEW generation; no admission traces."""
    from orchestrator.skill_validation import read_artifact
    skills_file = read_artifact(skills_path, allow_fixture)
    skills = skills_file['value']
    lessons_file = read_artifact(lessons_path, allow_fixture) if lessons_path else None
    library = (validate_library(lessons_file['value'], skills, allow_fixture=allow_fixture)
               if lessons_file else None)
    rows = retrieve(skills, allow_fixture=allow_fixture)
    kinds = {f['observed_kind'] for f in evidence_view(packet)}
    finalized = {e['lesson']['repair_binding']['program_hash']:e['lesson']
                 for e in (library or {}).get('entries', []) if e['lesson'].get('repair_binding')}
    rows.sort(key=lambda r:(finalized.get(r['program_hash'], {}).get('claim', {}).get('trigger_kind') not in kinds,
                            r['program_hash']))
    knowledge=[]
    for row in rows[:3]:
        card=published_card(row,library)
        # Legacy descriptions can contain long API manuals. The exact code is
        # separately supplied, and prose remains an unverified claim.
        brief={k:deepcopy(card[k]) for k in ('name','applicability','hypothesis','lesson_id') if k in card}
        knowledge.append(dict(program_hash=row['program_hash'],program=deepcopy(row['program']),card=brief,
            lesson=deepcopy(finalized.get(row['program_hash'], {}).get('claim')),
            target_verified=False, source_ids_are_target_ids=False))
    value=deepcopy(packet)
    value['context']['prior_knowledge']=dict(skills_file_sha256=skills_file['sha256'],
        lessons_file_sha256=lessons_file['sha256'] if lessons_file else None,
        available_programs=len(rows), supplied_programs=len(knowledge), entries=knowledge,
        admission_results_exposed=False, retrieval='matching_failure_kind_then_program_hash_max3')
    return value


def freeze_bundle(memory_path, skills_path, lessons_path, output_path, *, allow_fixture=False, allow_test_root=False):
    """Pin all three existing formats; never migrate files or change ACTIVE."""
    import hashlib
    from worker.task_memory import _lexical_path
    from orchestrator.explicit_action_memory import FrozenActionMemory
    for path in (memory_path, skills_path, lessons_path, output_path):
        _lexical_path(path)
    files, values = {}, {}
    for kind, path in (("memory", memory_path), ("skills", skills_path), ("lessons", lessons_path)):
        path = _safe_path(path, allow_test_root)
        data = path.read_bytes()
        values[kind] = json.loads(data)
        files[kind] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
    FrozenActionMemory(values["memory"])
    validate_library(values["lessons"], values["skills"], allow_fixture=allow_fixture)
    bundle = sign(dict(schema_version=1, kind="lesson_knowledge_bundle_v1", files=files,
        fixture=values["skills"]["fixture"], hard_pruning=False, physics_reuse_allowed=False,
        active_changed=False, performance_effect="NOT_RUN"))
    write_new(output_path, bundle, allow_test_root=allow_test_root)
    return bundle


def load_bundle(path, *, allow_fixture=False, allow_test_root=False):
    import hashlib
    from worker.task_memory import _lexical_path
    from orchestrator.explicit_action_memory import FrozenActionMemory
    bundle = checked(json.loads(_safe_path(path, allow_test_root).read_text(encoding="utf-8")),
                     "lesson_knowledge_bundle_v1")
    if set(bundle["files"]) != {"memory", "skills", "lessons"}:
        raise ValueError("lesson_bundle_files_mismatch")
    for spec in bundle["files"].values():
        _lexical_path(spec["path"])
    values = {}
    for kind, spec in bundle["files"].items():
        data = _safe_path(spec["path"], allow_test_root).read_bytes()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError("lesson_bundle_file_changed")
        values[kind] = json.loads(data)
    FrozenActionMemory(values["memory"])
    validate_library(values["lessons"], values["skills"], allow_fixture=allow_fixture)
    if (bundle.get("fixture") != values["skills"]["fixture"] or bundle.get("hard_pruning") is not False
            or bundle.get("physics_reuse_allowed") is not False):
        raise ValueError("lesson_bundle_policy_changed")
    return dict(bundle=bundle, **values)
