"""Frozen skill selection/interpreter adapter for the existing action mailbox.

The adapter has no search state and never calls a simulator. Generated programs
can only propose already-supported primitive actions. Selection may abstain;
the original actor remains available and all actual actions still pass the
existing action/path-round guards before the worker verifies physics.
"""
from copy import deepcopy
import time

from orchestrator.skill_discovery import SkillDiscoveryService
from orchestrator.skill_program import program_hash, step_program
from orchestrator.skill_registry import retrieve
from worker.explicit_action_contract import digest, validate_request, validate_response
from worker.path_probe_round import (canonical_path_round_response, partition_path_round,
                                     validate_path_round_response)


class GeneratedSkillService:
    def __init__(self, snapshot, fallback_service, discovery_service=None, *, mode="llm",
                 allow_fixture=False, timeout_seconds=None, max_scan_actions=50000, lesson_library=None,
                 knowledge_policy="legacy_v1", knowledge_max_skills=16,
                 knowledge_memory=None, knowledge_max_rounds=2, relational_knowledge=None):
        from orchestrator.knowledge_context import KnowledgeContext, POLICIES as KNOWLEDGE_POLICIES
        if knowledge_policy not in KNOWLEDGE_POLICIES:
            raise ValueError("unknown_knowledge_policy")
        if relational_knowledge is not None and knowledge_policy != 'agent_tools_v1':
            raise ValueError('relational_knowledge_requires_agent_tools')
        if type(knowledge_max_skills) is not int or not 1 <= knowledge_max_skills <= 32:
            raise ValueError("knowledge_skill_limit_must_be_1_to_32")
        self.knowledge_policy, self.knowledge_max_skills = knowledge_policy, knowledge_max_skills
        self.knowledge = (KnowledgeContext(snapshot, lesson_library, allow_fixture=allow_fixture)
                          if knowledge_policy == "unified_v1" else None)
        self.knowledge_queries = []
        self.knowledge_agent = None
        self._active_knowledge_view = None
        if mode not in ("llm", "deterministic"):
            raise ValueError("invalid_generated_skill_selection_mode")
        if not callable(getattr(fallback_service, "handle", None)):
            raise ValueError("fallback_action_service_required")
        if mode == "llm" and not isinstance(discovery_service, SkillDiscoveryService):
            raise ValueError("budgeted_skill_discovery_service_required")
        if knowledge_policy == "agent_tools_v1":
            if mode != "llm":
                raise ValueError("agent_knowledge_requires_model_mode")
            from orchestrator.knowledge_tools import KnowledgeCatalog
            from orchestrator.knowledge_retrieval import KnowledgeRetrieval
            catalog = KnowledgeCatalog(snapshot, lesson_library, knowledge_memory, allow_fixture=allow_fixture,
                                       relational_knowledge=relational_knowledge)
            self.knowledge = catalog.context
            self.knowledge_agent = KnowledgeRetrieval(catalog, discovery_service,
                max_rounds=knowledge_max_rounds, max_skills=knowledge_max_skills)
        fallback_live = getattr(fallback_service, "mode", None) == "llm"
        if allow_fixture and (mode != "deterministic" or fallback_live):
            raise ValueError("fixture_library_requires_wholly_offline_selection")
        if mode == "llm" and fallback_live:
            fallback_provider = getattr(fallback_service, "service", None)
            if (fallback_provider is None
                    or fallback_provider.ledger_path != discovery_service.ledger_path
                    or fallback_provider.config_fingerprint != discovery_service.config_fingerprint):
                raise ValueError("selector_and_actor_must_share_immutable_budget")
        if type(max_scan_actions) is not int or not 1 <= max_scan_actions <= 50000:
            raise ValueError("invalid_program_scan_limit")
        self._snapshot = deepcopy(snapshot)
        rows = retrieve(self._snapshot, allow_fixture=allow_fixture)
        from orchestrator.lesson_memory import validate_library, published_card
        self.lesson_library = (validate_library(lesson_library, snapshot, allow_fixture=allow_fixture)
                               if lesson_library is not None else None)
        for row in rows:
            self._validate_row(row, allow_fixture)
            published_card(row, self.lesson_library)
        self.snapshot_sha256 = snapshot["content_sha256"]
        self.fallback_service, self.discovery_service = fallback_service, discovery_service
        self.mode, self.allow_fixture = mode, allow_fixture
        self.max_scan_actions = max_scan_actions
        self.timeout_seconds = (getattr(fallback_service, "timeout_seconds", 60)
                                if timeout_seconds is None else timeout_seconds)
        if type(self.timeout_seconds) not in (int, float) or not 0 < self.timeout_seconds <= 120:
            raise ValueError("invalid_timeout")
        self.decisions, self.selector_decisions = [], []

    @staticmethod
    def _validate_row(row, allow_fixture):
        from orchestrator.lesson_runtime import CONTRACT
        permitted = {"registered_scoped", "transfer_observed"}
        if allow_fixture:
            permitted.add("fixture_registered_scoped")
        if (type(row) is not dict or row.get("lifecycle") not in permitted
                or type(row.get("fixture")) is not bool
                or (row["fixture"] and not allow_fixture)
                or row.get("strategy_id") != row.get("program_hash")
                or program_hash(row.get("program")) != row.get("program_hash")
                or row.get("physics_reuse_allowed") is not False
                or row.get("hard_pruning") is not False):
            raise ValueError("invalid_registered_generated_skill")
        if row.get('card', {}).get('runtime_contract') not in (None, CONTRACT):
            raise ValueError('unsupported_registered_runtime_contract')

    def _rows(self, request):
        if self._snapshot.get("content_sha256") != self.snapshot_sha256:
            raise ValueError("frozen_snapshot_identity_changed")
        rows = retrieve(self._snapshot, request["task_context"], allow_fixture=self.allow_fixture)
        for row in rows:
            self._validate_row(row, self.allow_fixture)
        return sorted(rows, key=lambda row: row["strategy_id"])

    def selection_packet(self, request, applicable, *, timeout_seconds=None):
        """Prepare the exact selector input; no key, API, or simulator access."""
        from orchestrator.explicit_action_provider_v5 import current_failure_brief, memory_alignment
        evidence = {}
        for record in request["episode_feedback"] + request["memory_references"]:
            if type(record) is dict and record.get("evidence_id") in request["evidence_ids"]:
                evidence[record["evidence_id"]] = deepcopy(record)
        cards = []
        view = (self._active_knowledge_view if self.knowledge_agent is not None else
                self.knowledge.query(request, max_skills=self.knowledge_max_skills)
                if self.knowledge is not None else None)
        if self.knowledge_agent is not None and (view is None or
                view['audit'].get('request_sha256') != digest(validate_request(request))):
            raise ValueError('agent_knowledge_view_request_mismatch')
        bundles = {b["skill_id"]: b for b in view["skills"]} if view else {}
        for row, step in applicable:
            from orchestrator.lesson_memory import published_card
            cards.append({"skill_id": row["strategy_id"], "lifecycle": row["lifecycle"],
                "program_hash": row["program_hash"], "card": published_card(row, self.lesson_library),
                "program": deepcopy(row["program"]),
                "validation_scope": deepcopy(row["validation_scope"]),
                "development_scope": deepcopy(row.get("development_scope", [])),
                "target_task_verified": False,
                "proposed_action": deepcopy(step["response"]["choice"]),
                "triggered_rule_index": step["rule_index"]})
            if view is not None:
                from orchestrator.knowledge_context import skill_knowledge
                if row["strategy_id"] not in bundles:
                    raise ValueError("selector_skill_not_in_knowledge_view")
                cards[-1]["knowledge"] = skill_knowledge(bundles[row["strategy_id"]])
        packet = {"run_id": request["run_id"], "request_id": "skill-select:" + digest({
                    "parent_request_id": request["request_id"], "run_id": request["run_id"]}),
                "state_id": request["state_id"], "operation": "select",
                "current_request": deepcopy(request), "skills": cards,
                "evidence": list(evidence.values()),
                "context": {"library_sha256": self.snapshot_sha256,
                            "current_failure": current_failure_brief(request),
                            "memory_alignment": [
                                dict(evidence_id=ref["evidence_id"], **memory_alignment(request, ref))
                                for ref in request["memory_references"]],
                            "stability_stage_semantics": (
                                "Fresh remaining-body check uses mover, pose and supports; "
                                "changing only force/path does not change that configuration. "
                                "Initial disconnection is not measured falling. "
                                "Geometry alignment permits a hypothesis, not outcome reuse. "
                                "Selecting a skill or citing memory is optional."),
                            "source_ids_are_not_target_ids": True,
                            "physical_verification_required": True},
                "timeout_seconds": (self.timeout_seconds / 2 if timeout_seconds is None
                                    else timeout_seconds)}

        if view is not None:
            packet["context"]["knowledge_context"] = deepcopy(view["audit"])
            if self.knowledge_agent is not None:
                from orchestrator.knowledge_retrieval import decision_evidence
                packet["context"]["retrieved_knowledge"] = decision_evidence(view)
        return packet

    def _fallback(self, request, reason, audit, selection=None, *, deadline=None):
        from orchestrator.knowledge_retrieval import RECOVERABLE_RETRIEVAL_ERRORS
        remaining = (self.timeout_seconds if deadline is None
                     else max(0.0, deadline - time.monotonic()))
        # Do not launch another billable request after the mailbox deadline.
        # Leave a small serialization margin inside that same total deadline.
        lookup_failed = (self._active_knowledge_view is not None and
                         self._active_knowledge_view['audit'].get('retrieval_status') == 'rejected' and
                         self._active_knowledge_view['audit'].get('stop_reason') not in RECOVERABLE_RETRIEVAL_ERRORS)
        if remaining <= 0.05 or lookup_failed:
            response = canonical_path_round_response(request, reason if lookup_failed else "generated_service_deadline",
                fallback_policy=getattr(self.fallback_service, "fallback_policy", "path_stage_v1"))
            response["metadata"].update(selection_source="deterministic_fallback",
                                        llm_live=False, attempted=False)
            actor_skipped = True
        else:
            original_timeout = self.fallback_service.timeout_seconds
            provider = getattr(self.fallback_service, "service", None)
            original_transport = getattr(provider, "transport", None)
            if callable(original_transport):
                # Keep the ledger's request timeout/fingerprint stable. Only
                # the actual transport receives the remaining wall-clock cap;
                # replaying this same request can therefore use its old result.
                def within_deadline(url, headers, body, timeout):
                    budget = (remaining if deadline is None else deadline - time.monotonic()) - 0.05
                    if budget <= 0:
                        raise TimeoutError("generated_service_deadline")
                    return original_transport(url, headers, body, min(timeout, budget))
                provider.transport = within_deadline
            else:
                self.fallback_service.timeout_seconds = min(original_timeout, remaining - 0.05)
            try:
                if self.knowledge_agent is not None:
                    from orchestrator.knowledge_retrieval import decision_evidence
                    response = self.fallback_service.handle(deepcopy(request),
                        retrieved_knowledge=decision_evidence(self._active_knowledge_view))
                else:
                    response = self.fallback_service.handle(deepcopy(request))
            finally:
                self.fallback_service.timeout_seconds = original_timeout
                if callable(original_transport):
                    provider.transport = original_transport
            actor_skipped = False
        response = validate_path_round_response(request, validate_response(request, response))
        response["metadata"].update(generated_program_executed=False,
            generated_skill_id=None, generated_program_hash=None,
            generated_library_sha256=self.snapshot_sha256,
            generated_skill_fallback_reason=reason,
            generated_skill_applicability=deepcopy(audit),
            generated_selector_response=deepcopy(selection),
            generated_library_fixture=self._snapshot["fixture"],
            generated_actor_skipped_deadline=actor_skipped and remaining <= .05,
            generated_actor_skipped=actor_skipped,
            generated_actor_skip_reason=('knowledge_retrieval_unavailable' if lookup_failed else
                                          'deadline' if actor_skipped else None),
            generated_service_timeout_seconds=self.timeout_seconds)
        if self.knowledge is not None and self.knowledge_queries:
            response["metadata"]["knowledge_context"] = deepcopy(self.knowledge_queries[-1]["audit"])
        self.decisions.append({"request": deepcopy(request), "response": deepcopy(response)})
        return response

    def handle(self, request):
        from orchestrator.skill_runtime_feedback import FeedbackBudget, INTERPRETATION
        from orchestrator.knowledge_retrieval import RECOVERABLE_RETRIEVAL_ERRORS, discard_rejected_knowledge
        deadline = time.monotonic() + self.timeout_seconds
        request = validate_request(request)
        partition_path_round(request)
        applicable, audit = [], []
        rows = self._rows(request)
        if self.knowledge is not None:
            view = (self.knowledge_agent.retrieve(request, deadline) if self.knowledge_agent is not None else
                    self.knowledge.query(request, max_skills=self.knowledge_max_skills))
            status = view['audit'].get('retrieval_status')
            if status in ('partial', 'rejected'):
                if status == 'partial':
                    # Keep only successfully read documents; expose incompleteness
                    # to both the selector and ordinary actor without relabeling.
                    view['audit']['decision_context'] = 'partial_knowledge_after_tool_errors'
                elif view['audit'].get('stop_reason') in RECOVERABLE_RETRIEVAL_ERRORS:
                    view = discard_rejected_knowledge(view)
            self._active_knowledge_view = view
            self.knowledge_queries.append(deepcopy(view))
            if (status == 'rejected' and
                    view['audit'].get('stop_reason') not in RECOVERABLE_RETRIEVAL_ERRORS):
                return self._fallback(request, 'knowledge_retrieval_unavailable', audit, deadline=deadline)
            rows = [self.knowledge.row(b["skill_id"]) for b in view["skills"]]
        diagnostics = FeedbackBudget(count=len(rows)) if self.knowledge_agent is not None else None
        for row in rows:
            if row["program"]["language"] == "python_policy_v1":
                remaining = deadline - time.monotonic() - 0.05
                if remaining <= 0:
                    return self._fallback(request, "python_applicability_deadline", audit, deadline=deadline)
                from orchestrator.python_skill_program import step_program as step_python
                from orchestrator.lesson_runtime import CONTRACT, extension_for
                step = step_python(row["program"], request, max_actions=self.max_scan_actions,
                                   timeout_seconds=min(4, remaining),
                                   extension=extension_for() if row['card'].get('runtime_contract') == CONTRACT else None,
                                   observe_abstention=self.knowledge_policy == 'agent_tools_v1')
            else:
                step = step_program(row["program"], request, max_actions=self.max_scan_actions)
            audit.append({"strategy_id": row["strategy_id"], "program_hash": row["program_hash"],
                          "triggered": step["triggered"], "scanned": step["scanned"],
                          "not_applicable_reason": step["not_applicable_reason"],
                          "rule_index": step["rule_index"]})
            if self.knowledge_agent is not None:
                details = diagnostics.take(step)
                if details is not None:
                    audit[-1]['runtime_feedback'] = details
            if step["response"] is not None:
                applicable.append((row, step))
        if self.knowledge_agent is not None and self._active_knowledge_view is not None:
            # Goes to the selector AND the ordinary actor after abstention.
            # Keep diagnostics outside historical documents and citation IDs.
            self._active_knowledge_view['audit']['skill_runtime_feedback'] = deepcopy(audit)
            self._active_knowledge_view['audit']['runtime_feedback_interpretation'] = INTERPRETATION
            self.knowledge_queries[-1]['audit'] = deepcopy(self._active_knowledge_view['audit'])
        if not applicable:
            return self._fallback(request, "no_applicable_registered_skill", audit, deadline=deadline)
        if self.mode == "deterministic":
            selected_id = applicable[0][0]["strategy_id"]
            selection = {"status": "selected", "selected_skill_id": selected_id,
                         "origin": "deterministic_control", "fallback_reason": None,
                         "metadata": {"llm_live": False, "attempted": False, "retries": 0}}
        else:
            if deadline - time.monotonic() <= (5 if self.knowledge_agent else self.timeout_seconds / 2 + 0.05):
                return self._fallback(request, "insufficient_selector_time", audit, deadline=deadline)
            packet = self.selection_packet(request, applicable,
                timeout_seconds=max(.01, (deadline-time.monotonic())/2) if self.knowledge_agent else None)
            selection = self.discovery_service.select(packet)
            self.selector_decisions.append({"packet": deepcopy(packet), "response": deepcopy(selection)})
            if selection.get("status") not in ("selected", "abstained"):
                return self._fallback(request, "skill_selector_rejected", audit, selection, deadline=deadline)
            if selection["status"] == "abstained":
                return self._fallback(request, "skill_selector_abstained", audit, selection, deadline=deadline)
            selected_id = selection.get("selected_skill_id")
        matches = [(row, step) for row, step in applicable if row["strategy_id"] == selected_id]
        if len(matches) != 1:
            return self._fallback(request, "skill_selector_unknown_id", audit, selection, deadline=deadline)
        row, step = matches[0]
        # Revalidate the chosen immutable program and already-computed response.
        if program_hash(row["program"]) != row["program_hash"]:
            return self._fallback(request, "generated_program_hash_mismatch", audit, selection, deadline=deadline)
        if deadline - time.monotonic() <= 0.05:
            return self._fallback(request, "generated_selection_after_deadline", audit, selection, deadline=deadline)
        response = validate_path_round_response(request, validate_response(request, step["response"]))
        metadata = selection["metadata"]
        actual_selected = (selection["origin"] == "actual_llm"
                           and (metadata.get("llm_live") or metadata.get("original_llm_live")))
        source = ("live_llm" if actual_selected else "deterministic_control"
                  if self.mode == "deterministic" else "injected_provider_fixture")
        response["metadata"].update(selection_source=source,
            selected_part_source="generated_program", decision_origin="generated_program",
            generated_program_executed=True, generated_skill_id=row["strategy_id"],
            generated_program_hash=row["program_hash"], generated_rule_index=step["rule_index"],
            generated_library_sha256=self.snapshot_sha256,
            generated_skill_selection_origin=selection["origin"],
            generated_skill_applicability=deepcopy(audit),
            generated_selector_response=deepcopy(selection),
            generated_library_fixture=self._snapshot["fixture"],
            llm_live=bool(metadata.get("llm_live")), attempted=bool(metadata.get("attempted")),
            generated_skill_fallback_reason=None, physics_bypass=False,
            physics_verdict_reused=False, candidate_pruning=False,
            generated_service_timeout_seconds=self.timeout_seconds)
        if self.knowledge is not None:
            response["metadata"]["knowledge_context"] = deepcopy(self.knowledge_queries[-1]["audit"])
        self.decisions.append({"request": deepcopy(request), "response": deepcopy(response)})
        return response

    rank = handle

    def close(self):
        closer = getattr(self.fallback_service, "close", None)
        if callable(closer):
            closer()
        closer = getattr(self.discovery_service, "close", None)
        if callable(closer):
            closer()
