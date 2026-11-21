"""Read-only joins over the existing frozen registry, lessons and request evidence.

No new store, model call, program execution or physics. The worker remains the
owner of memory retrieval and response-citable evidence IDs. Historical lesson
facts are provenance, not additional citations or target-success certificates.
"""
from copy import deepcopy
import json

from orchestrator.lesson_memory import published_card, validate_library
from orchestrator.skill_registry import retrieve
from worker.explicit_action_contract import digest, validate_request

POLICIES = ("legacy_v1", "unified_v1", "agent_tools_v1")


def observed_trigger(request):
    for row in reversed(request["episode_feedback"]):
        # The latest observation supersedes older failures; do not infer cause.
        if row.get("path_success") is False:
            return "path_failed"
        if row.get("path_success") is True and row.get("stability_success") is False:
            return "stability_failed"
        return "unknown"
    return "unknown"


def skill_knowledge(bundle):
    """One linked explanation beside an existing provider's code/proposal fields."""
    return deepcopy({k: bundle[k] for k in (
        "lessons", "evidence_links", "lesson_status", "knowledge_id",
        "claims_are_hypotheses", "requires_fresh_physics", "current_part_id_mapping",
        "current_attempt_verified", "physics_reuse_allowed", "hard_pruning", "execution_roles")})


class KnowledgeContext:
    def __init__(self, snapshot, lessons=None, *, allow_fixture=False):
        self._rows = deepcopy(retrieve(snapshot, allow_fixture=allow_fixture))
        self._lessons = (deepcopy(validate_library(lessons, snapshot, allow_fixture=allow_fixture))
                         if lessons is not None else None)
        self.sources = dict(skills_sha256=snapshot["content_sha256"],
            lessons_sha256=lessons["content_sha256"] if lessons is not None else None)
        self._bundles = {}
        for row in self._rows:
            identity = row["strategy_id"]
            if identity in self._bundles:
                raise ValueError("duplicate_knowledge_skill_id")
            linked = []
            for entry in (self._lessons or {}).get("entries", []):
                if row["program_hash"] not in entry["program_hashes"]:
                    continue
                lesson = entry["lesson"]
                linked.append(dict(lesson_id=lesson["content_sha256"],
                    claim=deepcopy(lesson["claim"]), observed_facts=deepcopy(lesson["observed_facts"]),
                    repair_observations=deepcopy(lesson.get("repair_binding", {}).get("repair_observations", [])),
                    causal_claim_verified=False, target_verified=False))
            linked.sort(key=lambda item: item["lesson_id"])
            bundle = dict(skill_id=identity, program_hash=row["program_hash"],
                program=deepcopy(row["program"]), card=published_card(row, self._lessons),
                lifecycle=row["lifecycle"], fixture=row["fixture"], validation_scope=deepcopy(row["validation_scope"]),
                development_scope=deepcopy(row.get("development_scope", [])),
                execution_roles=deepcopy(row["execution_roles"]),
                lessons=linked, lesson_status="linked" if linked else "NO_LINKED_LESSON",
                claims_are_hypotheses=True, requires_fresh_physics=True,
                current_part_id_mapping=None, current_attempt_verified=False,
                physics_reuse_allowed=False, hard_pruning=False)
            bundle["knowledge_id"] = digest(dict(sources=self.sources, bundle=bundle))
            self._bundles[identity] = bundle

    def row(self, skill_id):
        """Return pinned executable data. This does not execute or validate physics."""
        for row in self._rows:
            if row["strategy_id"] == skill_id:
                return deepcopy(row)
        raise ValueError("unknown_knowledge_skill_id")

    def query(self, request, *, trigger=None, max_skills=16, max_skill_bytes=98304):
        request = validate_request(request)
        if type(max_skills) is not int or not 1 <= max_skills <= 32:
            raise ValueError("knowledge_skill_limit_must_be_1_to_32")
        if type(max_skill_bytes) is not int or not 1024 <= max_skill_bytes <= 196608:
            raise ValueError("invalid_knowledge_byte_limit")
        if trigger is None:
            trigger = observed_trigger(request)
        if trigger not in ("path_failed", "stability_failed", "plan_pose_guard_v1", "unknown"):
            raise ValueError("invalid_knowledge_trigger")
        observations = {}
        for origin in ("episode_feedback", "memory_references"):
            for record in request[origin]:
                identity = record["evidence_id"]
                if identity not in request["evidence_ids"]:
                    raise ValueError("knowledge_observation_not_in_request")
                if identity in observations:
                    if observations[identity]["record"] != record:
                        raise ValueError("conflicting_knowledge_evidence_id")
                    observations[identity]["origins"].append(origin)
                    continue
                observations[identity] = dict(evidence_id=identity, origins=[origin], record=deepcopy(record),
                    accepted_for_transition=record.get("accepted_for_transition"),
                    deadline_exceeded=record.get("deadline_exceeded"),
                    missing_acceptance_metadata=[k for k in ("accepted_for_transition", "deadline_exceeded")
                                                 if record.get(k) is None])
        ranked = sorted(self._bundles.values(), key=lambda b: (
            not any(e["claim"]["trigger_kind"] == trigger and trigger != "unknown" for e in b["lessons"]),
            b["skill_id"]))
        selected, excluded, used_bytes, linked_observations = [], [], 0, set()
        for original in ranked:
            bundle = deepcopy(original)
            links = []
            for lesson in bundle["lessons"]:
                for fact in lesson["observed_facts"]:
                    identity = fact["evidence_id"]
                    # ID presence alone does not establish equal task/conditions.
                    links.append(dict(evidence_id=identity, lesson_id=lesson["lesson_id"],
                        historical_fact_sha256=fact.get("source_sha256"),
                        present_in_request=identity in observations,
                        citable_in_current_response=identity in request["evidence_ids"],
                        conditions_equivalent=False, source_part_ids_are_target_ids=False))
            bundle["evidence_links"] = links
            size = len(json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
            reason = ("skill_count_limit" if len(selected) >= max_skills else
                      "skill_context_byte_limit" if used_bytes + size > max_skill_bytes else None)
            if reason:
                excluded.append(dict(skill_id=bundle["skill_id"], reason=reason, bytes=size))
                continue
            selected.append(bundle)
            used_bytes += size
            linked_observations.update(link["evidence_id"] for link in links if link["present_in_request"])
        audit = dict(policy="unified_v1", sources=deepcopy(self.sources), request_sha256=digest(request),
            observation_scope="worker_selected_references_and_episode_feedback_only",
            retrieval="observed_trigger_match_then_program_hash", trigger=trigger,
            available_skills=len(ranked), selected_skill_ids=[b["skill_id"] for b in selected],
            excluded_skills=excluded, max_skills=max_skills, max_skill_bytes=max_skill_bytes,
            selected_skill_bytes=used_bytes, request_evidence_ids=list(request["evidence_ids"]),
            unlinked_observation_ids=sorted(set(observations) - linked_observations),
            observations_with_missing_acceptance=[k for k, v in observations.items() if v["missing_acceptance_metadata"]],
            new_citation_ids_added=False, source_files_modified=False,
            performance_effect="UNVERIFIED")
        return dict(schema_version=1, kind="knowledge_context_v1", skills=selected,
            observations=list(observations.values()), audit=audit)
