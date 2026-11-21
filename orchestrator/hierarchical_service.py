"""Luna scout -> Sol conditional plans -> guarded code/registered skill execution.

Only stores a policy cursor and expected observations. The existing ASAP adapter
owns the search, verifies every proposal and decides every state transition.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

from orchestrator.conditional_plan import action_for_step, response_for, task_scope, verify_geometry
from orchestrator.generated_skill_service import GeneratedSkillService
from orchestrator.skill_program import step_program, program_hash
from orchestrator.skill_registry import retrieve
from worker.explicit_action_contract import action_id, digest, validate_request, validate_response
from worker.path_probe_round import canonical_path_round_response, validate_path_round_response
from orchestrator.recovery_support_guard import POLICIES, check_recovery_supports
from orchestrator.lesson_runtime import CONTRACT


class HierarchicalDecisionService:
    def __init__(self, scout, planner, snapshot, limits, *, timeout_seconds=90,
                 allow_fixture=False, journal_dir=None, recovery_support_policy="legacy_v1",
                 planning_feedback_policy="legacy_v1", lesson_library=None,
                 lesson_runtime=False, max_skill_attempts=1, transport_guard=None,
                 knowledge_policy="legacy_v1", knowledge_max_skills=16,
                 knowledge_memory=None, knowledge_max_rounds=2, knowledge_runtime_feedback=False,
                 relational_knowledge=None):
        from orchestrator.knowledge_context import KnowledgeContext, POLICIES as KNOWLEDGE_POLICIES
        if knowledge_policy not in KNOWLEDGE_POLICIES:
            raise ValueError("unknown_knowledge_policy")
        if relational_knowledge is not None and knowledge_policy != 'agent_tools_v1':
            raise ValueError('relational_knowledge_requires_agent_tools')
        if type(knowledge_max_skills) is not int or not 1 <= knowledge_max_skills <= 32:
            raise ValueError("knowledge_skill_limit_must_be_1_to_32")
        if (type(knowledge_runtime_feedback) is not bool
                or knowledge_runtime_feedback and knowledge_policy == 'legacy_v1'):
            raise ValueError('runtime_feedback_requires_explicit_nonlegacy_knowledge')
        self.observe_skill_abstention = knowledge_policy == 'agent_tools_v1' or knowledge_runtime_feedback
        self.knowledge_policy, self.knowledge_max_skills = knowledge_policy, knowledge_max_skills
        self.knowledge = (KnowledgeContext(snapshot, lesson_library, allow_fixture=allow_fixture)
                          if knowledge_policy == "unified_v1" else None)
        self.knowledge_queries = []
        self.knowledge_agent = None
        if knowledge_policy == "agent_tools_v1":
            from orchestrator.knowledge_tools import KnowledgeCatalog
            from orchestrator.knowledge_retrieval import KnowledgeRetrieval
            catalog = KnowledgeCatalog(snapshot, lesson_library, knowledge_memory, allow_fixture=allow_fixture,
                                       relational_knowledge=relational_knowledge)
            self.knowledge = catalog.context
            self.knowledge_agent = KnowledgeRetrieval(catalog, planner,
                max_rounds=knowledge_max_rounds, max_skills=knowledge_max_skills)
        self.transport_guard = transport_guard
        from orchestrator.path_failure_feedback import POLICIES as FEEDBACK_POLICIES
        if planning_feedback_policy not in FEEDBACK_POLICIES:
            raise ValueError("unknown_planning_feedback_policy")
        self.planning_feedback_policy = planning_feedback_policy
        if recovery_support_policy not in POLICIES:
            raise ValueError("unknown_recovery_support_policy")
        self.recovery_support_policy = recovery_support_policy
        self.last_failure_context = None
        # Only proposals actually evaluated under a model-authorized recovery
        # may cross into the next explicit plan, and only for this request.
        self.recovery_handoffs = {}
        self.planning_card_capacity = None
        self.scout, self.planner = scout, planner
        self.snapshot = deepcopy(snapshot)
        self.library_hash = digest(snapshot)
        self.rows = retrieve(self.snapshot, allow_fixture=allow_fixture)
        self.lesson_library = None
        if lesson_library is not None:
            from orchestrator.lesson_memory import validate_library
            self.lesson_library = validate_library(lesson_library, snapshot, allow_fixture=allow_fixture)
        if type(lesson_runtime) is not bool or type(max_skill_attempts) is not int or not 1 <= max_skill_attempts <= 4:
            raise ValueError("invalid_lesson_runtime_budget")
        if not lesson_runtime and max_skill_attempts != 1:
            raise ValueError("skill_retry_requires_lesson_runtime")
        self.lesson_runtime, self.max_skill_attempts = lesson_runtime, max_skill_attempts
        self.active_guard = None
        self.planning_guard_context = None
        for row in self.rows:
            GeneratedSkillService._validate_row(row, allow_fixture)
            if row['card'].get('runtime_contract') == CONTRACT and not self.lesson_runtime:
                raise ValueError('new_runtime_requires_explicit_lesson_mode')
            from orchestrator.lesson_memory import published_card
            published_card(row, self.lesson_library)
        self.limits = deepcopy(limits)
        self.timeout_seconds = timeout_seconds
        self.plan = None
        self.plan_result = None
        self.index = 0
        self.pending = None
        self.failed = False
        self.recovery_used = 0
        self.scout_calls = self.planner_calls = self.fallback_calls = 0
        self.bound_scope = None
        self.bound_geometry = None
        self.cache, self.decisions, self.events, self.planner_decisions = {}, [], [], []
        self.journal = Path(journal_dir) if journal_dir else None
        if self.journal:
            if self.journal.exists():
                raise ValueError("partial_hierarchy_resume_not_supported")
            self.journal.mkdir(parents=True, exist_ok=False)

    def _event(self, kind, **fields):
        event = dict(kind=kind, sequence=len(self.events) + 1, **deepcopy(fields))
        self.events.append(event)
        if self.journal:
            with (self.journal / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def _invalidate(self, reason):
        self._event("plan_invalidated", reason=reason,
                    plan_id=self.plan["plan_id"] if self.plan else None, step=self.index)
        self.plan = None
        self.failed = False
        self.recovery_used = 0
        self.active_guard = None
        self.planning_guard_context = None
        return reason

    def _reconcile(self, request):
        if self.pending is None:
            return None
        pending, self.pending = self.pending, None
        matches = [r for r in request["episode_feedback"] if r.get("action_id") == pending["action_id"]]
        if len(matches) != 1:
            self.last_failure_context = None
            return self._invalidate("missing_or_ambiguous_verifier_feedback")
        observed = matches[0]
        success = (observed.get("accepted_for_transition") is True
            and observed.get("status") == "PASS" and observed.get("deadline_exceeded") is False
            and observed.get("path_success") is True and observed.get("stability_success") is True)
        expected = set(pending["remaining"]) - {pending["part_id"]} if success else set(pending["remaining"])
        self._event("physics_feedback", action_id=pending["action_id"], evidence_id=observed["evidence_id"],
            role=pending["role"], status=observed["status"], accepted=success,
            generated_skill_id=pending.get("generated_skill_id"))
        if self.recovery_support_policy != "legacy_v1":
            self.last_failure_context = (dict(choice=deepcopy(pending["choice"]),
                remaining_parts=list(pending["remaining"]), evidence_id=observed["evidence_id"],
                path_success=observed.get("path_success"), stability_success=observed.get("stability_success"))
                if not success else None)
        if expected != set(request["remaining_parts"]):
            self.last_failure_context = None
            return self._invalidate("backtrack_or_unexpected_state")
        if pending["role"] in ("scout", "cheap_fallback", "deterministic_fallback"):
            return None if success else "observed_scout_failure"
        if self.plan is None or pending.get("plan_id") != self.plan["plan_id"]:
            return self._invalidate("plan_feedback_identity_changed")
        if success:
            self.active_guard = None
            step = self.plan["steps"][self.index]
            if pending["part_id"] != step["part_id"]:
                return self._invalidate("recovery_changed_expected_removal")
            self.index += 1
            self.failed, self.recovery_used = False, 0
            if self.index == len(self.plan["steps"]):
                return self._invalidate("plan_horizon_completed")
            return None
        self.active_guard = None  # A new physical observation supersedes the pre-physics guard.
        if pending["role"] == "recovery" and not (
                self.lesson_runtime and self.recovery_used < self.max_skill_attempts):
            return self._invalidate("registered_recovery_failed")
        self.failed = True
        return "observed_plan_failure"

    def _execution_plan(self):
        """Plan context for the interpreter; legacy profiles keep their contract."""
        return self.plan

    def _planning_context(self, request):
        """Profile-specific host facts; legacy profiles add no new contract."""
        return {}

    def _planning_preflight(self, request):
        """Optional no-API rejection before scout, retrieval or planning."""
        return None

    def _planning_rejection_feedback(self, request, result, event_start):
        """Opt-in semantic correction. Legacy profiles never repeat a plan call."""
        return None

    def _record_planning_rejection(self, request, result, event_start, ordinal):
        feedback = self._planning_rejection_feedback(request, result, event_start)
        if feedback is None:
            return None
        feedback = dict(deepcopy(feedback), correction_ordinal=ordinal,
                        completed_plan_call=self.planner_calls)
        self._event('strategy_plan_rejection_feedback', feedback=deepcopy(feedback))
        return feedback

    def _skill_step(self, row, request, deadline):
        if program_hash(row["program"]) != row["program_hash"]:
            raise ValueError("registered_program_hash_changed")
        if row["program"]["language"] == "python_policy_v1":
            from orchestrator.python_skill_program import step_program as step_python
            left = deadline - time.monotonic() - .1
            if left <= 0:
                return None
            extension = None
            execution_plan = self._execution_plan()
            from orchestrator.lesson_runtime import CONTRACT, extension_for
            if row["card"].get("runtime_contract") == CONTRACT:
                extension = extension_for(self.active_guard, execution_plan, self.index)
                if self.active_guard is None and self.planning_guard_context is not None:
                    extension = extension_for(**self.planning_guard_context)
            elif self.lesson_runtime:
                extension = dict(guard=self.active_guard, plan=execution_plan, index=self.index,
                    lessons=self._lessons())
                if self.active_guard is None and self.planning_guard_context is not None:
                    extension.update(deepcopy(self.planning_guard_context))
            result = step_python(row["program"], request, max_actions=50000,
                                 timeout_seconds=min(4, left), extension=extension,
                                 observe_abstention=self.observe_skill_abstention)
        else:
            result = step_program(row["program"], request, max_actions=50000)
        return result

    def _lessons(self):
        if self.lesson_library is None:
            return []
        from orchestrator.lesson_memory import lesson_cards
        return lesson_cards(self.lesson_library,
            "plan_pose_guard_v1" if self.active_guard or self.planning_guard_context else None)

    def _current_recovery_handoffs(self, request):
        identity = digest(request)
        rows = {row['strategy_id']: row for row in self.rows}
        return [deepcopy(value) for key, value in self.recovery_handoffs.items()
                if value['request_sha256'] == identity
                and value['state_id'] == request['state_id']
                and value['task_scope'] == task_scope(request)
                and value['library_sha256'] == self.library_hash
                and key in rows and value['program_hash'] == rows[key]['program_hash']
                and value['source_evidence_id'] in request['evidence_ids']
                and value['proposal_action_id'] == action_id(request, value['proposed_choice'])]

    def _cards(self, request, deadline, view=None):
        from orchestrator.skill_runtime_feedback import FeedbackBudget, INTERPRETATION
        cards = []
        self.planning_card_capacity = None
        bundles = {}
        rows = self.rows
        handoffs = {h['skill_id']: h for h in self._current_recovery_handoffs(request)}
        if self.knowledge is not None:
            if self.knowledge_agent is not None and view is None:
                raise ValueError("agent_selected_knowledge_view_required")
            if self.knowledge_agent is not None and view['audit'].get('request_sha256') != digest(validate_request(request)):
                raise ValueError('agent_knowledge_view_request_mismatch')
            if view is None:
                view = self.knowledge.query(request, max_skills=self.knowledge_max_skills,
                    trigger="plan_pose_guard_v1" if self.active_guard or self.planning_guard_context else None)
            self.knowledge_queries.append(deepcopy(view))
            self._event("knowledge_supplied", request_id=request["request_id"], **view["audit"])
            bundles = {b["skill_id"]: b for b in view["skills"]}
            rows = [self.knowledge.row(identity) for identity in bundles]
            # This is current execution feedback from the previously accepted
            # plan, not a search fallback or a read of an unseen library entry.
            supplied = {row['strategy_id'] for row in rows}
            rows += [row for row in self.rows if row['strategy_id'] in handoffs
                     and row['strategy_id'] not in supplied]
            limit = self.knowledge_max_skills
            if len(rows) > limit:
                # Reserve capacity for the current, actually executed refusal.
                # Other read documents remain in the audit; no physical action
                # or candidate part is pruned and no skill selection is forced.
                reserved = [row for row in rows if row['strategy_id'] in handoffs][:limit]
                other = [row for row in rows if row['strategy_id'] not in handoffs]
                retained = {row['strategy_id'] for row in reserved + other[:limit-len(reserved)]}
                excluded = [row['strategy_id'] for row in rows if row['strategy_id'] not in retained]
                rows = [row for row in rows if row['strategy_id'] in retained]
                self.planning_card_capacity = dict(maximum_cards=limit,
                    supplied_skill_ids=[row['strategy_id'] for row in rows],
                    omitted_from_this_plan_skill_ids=excluded,
                    reason='reserve_current_recovery_handoff_within_card_limit',
                    hard_pruning=False, selection_forced=False)
                self._event('planning_skill_card_capacity', **self.planning_card_capacity)
        # Retrieval policy changes how cards are found, not the quality of
        # execution evidence shown on those cards. Local bounded retrieval must
        # retain the same bounded abstention trace without an extra model call.
        diagnostics = FeedbackBudget(count=len(rows)) if self.observe_skill_abstention else None
        for row in rows:
            step = self._skill_step(row, request, deadline)
            cards.append(dict(skill_id=row["strategy_id"], program_hash=row["program_hash"],
                lifecycle=row["lifecycle"], card=row["card"], program=row["program"],
                validation_scope=row["validation_scope"], target_success_guaranteed=False,
                currently_proposed_choice=(step["response"]["choice"] if step and step["response"] else None),
                applicability=(step.get("not_applicable_reason") if step else "deadline")))
            choice = cards[-1]['currently_proposed_choice']
            if choice is not None:
                from orchestrator.skill_execution_roles import assess_execution_roles
                cards[-1]['execution_roles'] = assess_execution_roles(request, choice)
            cards[-1]['validated_execution_roles'] = deepcopy(row.get('execution_roles',
                dict(status='NOT_RECORDED', automatic_execution_verified=False)))
            if row['strategy_id'] in handoffs:
                cards[-1]['recovery_handoff'] = deepcopy(handoffs[row['strategy_id']])
                cards[-1]['handoff_proposal_matches_current_vm'] = (
                    choice == handoffs[row['strategy_id']]['proposed_choice'])
                if not cards[-1]['handoff_proposal_matches_current_vm']:
                    cards[-1]['currently_proposed_choice'] = None
                    cards[-1]['applicability'] = 'recovery_handoff_proposal_changed'
            if diagnostics is not None:
                details = diagnostics.take(step)
                if details is not None:
                    cards[-1]['runtime_feedback'] = details
                    cards[-1]['runtime_feedback_interpretation'] = INTERPRETATION
            if self.lesson_runtime and row["card"].get("lesson_id"):
                # Execute the pinned code locally. Planning sees the lesson and
                # concrete proposal, not another copy of the source/API manual.
                cards[-1]["card"] = {k: deepcopy(row["card"][k]) for k in
                    ("name", "applicability", "hypothesis", "lesson_id") if k in row["card"]}
                cards[-1]["program"] = dict(language=row["program"]["language"],
                    source_omitted_for_planning=True, program_hash=row["program_hash"])
                if row['card'].get('runtime_contract') == CONTRACT:
                    from orchestrator.lesson_memory import published_card
                    cards[-1]['card'] = published_card(row, self.lesson_library)
            if self.knowledge is not None and row['strategy_id'] in bundles:
                from orchestrator.knowledge_context import skill_knowledge
                bundle = bundles[row["strategy_id"]]
                cards[-1]["card"] = deepcopy(bundle["card"])
                cards[-1]["knowledge"] = skill_knowledge(bundle)
            elif self.knowledge is not None and row['strategy_id'] in handoffs:
                cards[-1]['knowledge_source'] = 'prior_model_authorized_recovery_this_request'
            if self.planning_feedback_policy in ("path_evidence_v1", "repair_evidence_v2"):
                from orchestrator.path_failure_feedback import proposal_history
                cards[-1]["proposed_action_observations"] = proposal_history(
                    request, cards[-1]["currently_proposed_choice"])
                cards[-1]["execution_scope"] = (
                    "This code proposes one bounded action from its coded pattern. "
                    "It does not search all paths or certify target success. "
                    "Prior outcomes are scoped evidence; no proposal is disabled by memory.")
        return cards

    def _program_response(self, identity, request, deadline, role, expected_part=None):
        row = next((r for r in self.rows if r["strategy_id"] == identity), None)
        if row is None:
            return None
        result = self._skill_step(row, request, deadline)
        self._event("skill_checked", skill_id=identity, plan_id=self.plan["plan_id"],
            triggered=bool(result and result["triggered"]),
            not_applicable_reason=result.get("not_applicable_reason") if result else "deadline")
        if self.observe_skill_abstention and (not result or not result['response']):
            from orchestrator.skill_runtime_feedback import FeedbackBudget
            self._event('skill_runtime_feedback', skill_id=identity,
                runtime_feedback=FeedbackBudget().take(result))
        if not result or not result["response"]:
            return None
        choice = result["response"]["choice"]
        if expected_part is not None and choice["part_id"] != expected_part:
            return None
        if role == "recovery" and self.recovery_support_policy == "preserve_existing_v1":
            guard = check_recovery_supports(self.last_failure_context, choice, request)
            self._event("recovery_support_guard", skill_id=identity, **guard)
            if not guard["allowed"]:
                if (guard.get('reason') == 'recovery_support_release_requires_new_plan'
                        and guard.get('source_evidence_id')
                        and identity in self.plan['recovery_skill_ids']):
                    handoff = dict(version='recovery_replan_handoff_v1',
                        request_id=request['request_id'], request_sha256=digest(request),
                        state_id=request['state_id'], skill_id=identity,
                        task_scope=task_scope(request), proposal_action_id=action_id(request, choice),
                        program_hash=row['program_hash'], library_sha256=self.library_hash,
                        source_plan_id=self.plan['plan_id'], source_plan_step=self.index,
                        source_evidence_id=guard['source_evidence_id'],
                        proposed_choice=deepcopy(choice), support_guard=deepcopy(guard),
                        physical_attempt_executed=False, physical_verdict=None,
                        requires_new_explicit_plan=True, selection_forced=False,
                        fresh_physics_required=True)
                    self.recovery_handoffs[identity] = handoff
                    self._event('recovery_proposal_deferred_to_plan', **handoff)
                return None
        if role == 'plan_skill':
            handoff = next((h for h in self._current_recovery_handoffs(request)
                            if h['skill_id'] == identity), None)
            if handoff is not None and choice != handoff['proposed_choice']:
                self._event('recovery_handoff_proposal_changed', skill_id=identity,
                    physical_attempt_executed=False, requires_new_explicit_plan=True)
                return None
        response = response_for(request, choice, self.plan["candidate_order"], {
            "plan_id": self.plan["plan_id"], "plan_step": self.index,
            "generated_program_executed": True, "generated_skill_id": identity,
            "generated_program_hash": row["program_hash"],
            "plan_execution_without_api": True, "skill_authorized_by_plan": True})
        from orchestrator.skill_execution_roles import assess_execution_roles
        response['metadata']['skill_execution_roles'] = assess_execution_roles(request, choice)
        response['metadata']['skill_execution_role'] = (
            'automatic_recovery' if role == 'recovery' else 'explicit_plan_skill')
        if role == 'plan_skill' and handoff is not None:
            response['metadata']['recovery_replan_handoff'] = deepcopy(handoff)
        return self._finish(request, response, role)

    def _execute(self, request, deadline):
        if self.plan is None:
            return None
        if task_scope(request) != self.plan["task_scope"]:
            self._invalidate("plan_task_scope_changed")
            return None
        if self.failed:
            if self.recovery_used < (self.max_skill_attempts if self.lesson_runtime else self.limits["recovery_per_step"]):
                for identity in self.plan["recovery_skill_ids"]:
                    response = self._program_response(identity, request, deadline, "recovery")
                    if response:
                        self.recovery_used += 1
                        return response
            self._invalidate("no_usable_authorized_recovery")
            return None
        step = self.plan["steps"][self.index]
        if set(step["expected_remaining"]) != set(request["remaining_parts"]):
            self._invalidate("plan_remaining_guard_failed")
            return None
        if step["mode"] == "registered_skill":
            response = self._program_response(step["registered_skill_id"], request, deadline,
                                               "plan_skill", step["part_id"])
            if response:
                return response
            self._invalidate("planned_registered_skill_not_applicable")
            return None
        try:
            response = action_for_step(self.plan, self.index, request)
        except (ValueError, KeyError, TypeError) as error:
            reason = str(error)
            guard_context = None
            if self.lesson_runtime and reason == "planned_orientation_unavailable":
                from orchestrator.pose_guard_recovery import make_guard
                self.active_guard = make_guard(self.plan, self.index, request)
                self._event("plan_guard_observed", guard=deepcopy(self.active_guard))
                guard_context = dict(guard=deepcopy(self.active_guard), plan=deepcopy(self.plan), index=self.index)
                # Only skills authorized by the model's accepted plan may execute.
                for identity in self.plan["recovery_skill_ids"]:
                    response = self._program_response(identity, request, deadline, "recovery",
                                                      expected_part=step["part_id"])
                    if response:
                        self.recovery_used += 1
                        return response
            self._invalidate(reason if reason.isidentifier() and len(reason) < 100 else "invalid_plan_action")
            # The next model may freely select a guard skill even when the old
            # plan did not authorize one. Bind to the old plan for this request
            # only; never reinterpret it as a failure of the replacement plan.
            self.planning_guard_context = guard_context
            return None
        if "plan_pose_binding" in response["metadata"]:
            self._event("plan_pose_resolved", request_id=request["request_id"],
                plan_id=self.plan["plan_id"], step=self.index,
                **response["metadata"]["plan_pose_binding"])
        return self._finish(request, response, "plan_action")

    def _record_plan_skill_review(self, request, result, role):
        """Log declared model reasons without inferring a missing explanation."""
        policy = getattr(self, 'skill_review_policy', None)
        if policy is None:
            return
        review = result.get('metadata', {}).get('skill_selection_review')
        common = dict(request_id=request['request_id'], role=role, policy=policy,
            plan_id=(result.get('plan') or {}).get('plan_id'),
            provider_status=result.get('status'), selection_forced=False,
            physical_success_implied=False)
        if review is not None:
            self._event('skill_selection_review_recorded', **common, review=deepcopy(review),
                        model_self_report=True)
        else:
            error = result.get('metadata', {}).get('skill_selection_review_error')
            self._event('skill_selection_review_not_recorded', **common,
                reason=('legacy_not_requested' if policy == 'legacy_v1' else
                        'invalid_or_missing_review' if error else 'plan_review_unavailable'),
                review_error=deepcopy(error), model_self_report=bool(error))

    def _planning_prior_events(self):
        # Audit-only additions must not evict physical/plan feedback or change
        # legacy prompt event numbering. The immutable journal keeps its full
        # sequence; this is the existing planning view with audits excluded.
        audit_kinds = {'skill_selection_review_recorded', 'skill_selection_review_not_recorded',
                       'skill_selection_review_context'}
        visible = [event for event in self.events if event['kind'] not in audit_kinds]
        return [dict(deepcopy(event), sequence=index + 1) for index, event in enumerate(visible)][-12:]

    def _finish(self, request, response, role):
        response = deepcopy(response)
        metadata = response["metadata"]
        if role.startswith("plan_") or role == "recovery":
            origin = (self.plan_result or {}).get("metadata", {})
            metadata.update(plan_model=origin.get("model_returned"),
                plan_provider_call_id=origin.get("call_id"),
                plan_actual_llm=bool(origin.get("llm_live") or origin.get("original_llm_live")),
                plan_provider_attempt_on_this_request=self.plan["request_id"] == request["request_id"])
            if 'knowledge_retrieval' in origin:
                metadata['plan_knowledge_retrieval'] = deepcopy(origin['knowledge_retrieval'])
        metadata.update(hierarchy_role=role, hierarchy_version="luna_sol_v1",
                        hierarchy_library_sha256=self.library_hash)
        if self.planning_feedback_policy == 'repair_evidence_v2' and response['choice'] is not None:
            from orchestrator.planning_repair_review import stability_choice_comparison
            comparison = stability_choice_comparison(request, response['choice'])
            metadata['stability_repair_comparison'] = comparison
            if comparison is not None:
                self._event('stability_repair_choice', role=role, **comparison)
        response = validate_path_round_response(request, validate_response(request, response))
        if 'skill_selection_review_context' in metadata:
            self._event('skill_selection_review_context', request_id=request['request_id'],
                        role=role, **metadata['skill_selection_review_context'])
        self.planning_guard_context = None
        if response["choice"] is not None:
            self.pending = dict(action_id=response["action_id"], part_id=response["choice"]["part_id"],
                remaining=request["remaining_parts"], role=role,
                plan_id=metadata.get("plan_id"), generated_skill_id=metadata.get("generated_skill_id"))
            if self.recovery_support_policy != "legacy_v1":
                self.pending["choice"] = deepcopy(response["choice"])
        self.decisions.append(dict(request=deepcopy(request), response=deepcopy(response)))
        self.cache[request["request_id"]] = (digest(request), deepcopy(response))
        if self.journal:
            record = dict(request_sha256=digest(request), response=response,
                planner_calls=self.planner_calls, scout_calls=self.scout_calls, index=self.index,
                resume="fail_closed_new_episode_required")
            with (self.journal / (digest(request["request_id"]) + ".json")).open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
        return response

    def _scout(self, request, deadline, role):
        if role == "scout":
            self.scout_calls += 1
        else:
            self.fallback_calls += 1
        provider = getattr(self.scout, "service", None)
        original = self.scout.timeout_seconds
        self.scout.timeout_seconds = max(.01, min(original, deadline - time.monotonic() - .1))
        try:
            response = self.scout.handle(deepcopy(request))
        finally:
            self.scout.timeout_seconds = original
        if response["fallback_reason"] is None:
            return self._finish(request, response, role)
        details = response.get('metadata', {}).get('transport_diagnostics')
        self._event("scout_rejected", reason=response["fallback_reason"], role=role,
                    **({'transport_diagnostics': details} if details else {}))
        return None

    def _blocked_transport_response(self, request):
        blocked = self.transport_guard() if self.transport_guard is not None else None
        if not blocked:
            return None
        self._event('paid_api_blocked', **blocked)
        response = canonical_path_round_response(request, 'transport_circuit_open', fallback_policy='path_stage_v1')
        response['metadata']['transport_guard'] = deepcopy(blocked)
        return self._finish(request, response, 'deterministic_fallback')

    def handle(self, request):
        request = validate_request(request)
        if request["request_id"] in self.cache:
            identity, response = self.cache[request["request_id"]]
            if identity != digest(request):
                raise ValueError("hierarchy_request_id_conflict")
            return deepcopy(response)
        current_scope = task_scope(request)
        if self.bound_scope is None:
            self.bound_scope = current_scope
            self.bound_geometry = deepcopy(request["geometry"].get("parts", {}))
        elif current_scope != self.bound_scope:
            raise ValueError("hierarchy_run_or_physics_scope_changed")
        verify_geometry(self.bound_geometry, request)
        if digest(self.snapshot) != self.library_hash:
            raise ValueError("hierarchy_library_changed")
        deadline = time.monotonic() + self.timeout_seconds
        self.recovery_handoffs = {}
        reason = self._reconcile(request)
        execution_event_start = len(self.events)
        execution_result = self.plan_result
        response = self._execute(request, deadline)
        if response:
            return response
        correction_ordinal = 1
        rejection_feedback = self._record_planning_rejection(
            request, execution_result, execution_event_start, correction_ordinal)
        if rejection_feedback is not None:
            correction_ordinal += 1
        response = self._planning_preflight(request)
        if response:
            return response
        blocked = self._blocked_transport_response(request)
        if blocked:
            return blocked
        if getattr(self.planner, "provider_contract", None) in ("conditional_plan_v3", "conditional_plan_v4", "strategy_plan_v1"):
            invalidations = [e for e in self.events[execution_event_start:] if e["kind"] == "plan_invalidated"]
            if invalidations:
                reason = invalidations[-1]["reason"]
        if reason is None and self.planner_calls == 0 and self.scout_calls < self.limits["scout_calls"]:
            scout_event_start = len(self.events)
            scout_decision_start = len(self.planner_decisions)
            response = self._scout(request, deadline, "scout")
            if response:
                return response
            reason = "scout_contract_or_api_failure"
            scout_result = (self.planner_decisions[-1]['result']
                if len(self.planner_decisions) > scout_decision_start else None)
            scout_feedback = self._record_planning_rejection(
                request, scout_result, scout_event_start, correction_ordinal)
            if scout_feedback is not None:
                rejection_feedback = scout_feedback
                correction_ordinal += 1
                reason = scout_feedback['rejection_reason']
            blocked = self._blocked_transport_response(request)
            if blocked:
                return blocked
        # A support-only semantic correction shares this request's deadline,
        # role allowance and immutable provider ledgers. It is never an HTTP
        # retry, never changes the worker state, and is opt-in via the hook.
        view = None
        while self.planner_calls < self.limits["planner_calls"] and deadline - time.monotonic() > 5:
            blocked = self._blocked_transport_response(request)
            if blocked:
                return blocked
            next_plan_call = self.planner_calls + 1
            escalation = dict(reason=reason or "scout_allowance_or_plan_horizon",
                episode_planner_call=next_plan_call,
                prior_events=[{k:v for k,v in e.items() if k != "prior_events" and
                    not (self.knowledge_agent is not None and k in ('retrieved_knowledge', 'knowledge_context'))}
                    for e in self._planning_prior_events()],
                first_planning=next_plan_call == 1)
            if rejection_feedback is not None:
                escalation['plan_rejection_feedback'] = deepcopy(rejection_feedback)
            planning_context = self._planning_context(request)
            escalation.update(planning_context)
            if self.lesson_library is not None and self.knowledge is None:
                escalation["lesson_references"] = self._lessons()
                self._event("lessons_supplied", lesson_ids=[r["lesson_id"] for r in self._lessons()],
                            claims_are_hypotheses=True)
            if self.lesson_runtime:
                escalation["lesson_runtime"] = dict(max_skill_attempts=self.max_skill_attempts,
                    guard=self.planning_guard_context["guard"] if self.planning_guard_context else None,
                    host_owns_physics_and_state=True)
            if self.planning_feedback_policy in ("path_evidence_v1", "repair_evidence_v2"):
                escalation["planning_allowance"] = dict(
                    current_call=next_plan_call, maximum_calls=self.limits["planner_calls"],
                    calls_after_this=max(0, self.limits["planner_calls"]-next_plan_call),
                    maximum_steps=self.limits["max_plan_steps"],
                    current_remaining_count=len(request["remaining_parts"]),
                    full_disassembly_goal_remaining_count=1,
                    automatic_retries=0, calls_beyond_limit_authorized=False)
            if self.recovery_support_policy != "legacy_v1":
                escalation["recovery_support_policy"] = self.recovery_support_policy
            handoffs = self._current_recovery_handoffs(request)
            if handoffs:
                escalation['recovery_replan_handoffs'] = handoffs
                escalation['recovery_handoff_instruction'] = (
                    'These pinned proposals were produced by skills already authorized by your prior plan, '
                    'but automatic support release was refused before physics. They are current execution '
                    'feedback, not new physical successes or a preferred order. You may freely select the '
                    'skill in a fresh registered_skill plan step, choose another action, or defer. '
                    'A fresh plan selection and fresh physics are required; no automatic release is authorized.')
            # No new observation was produced by a pre-physics rejection.
            # Reuse the same retrieved view, avoiding extra paid lookups and
            # duplicate immutable journal names for this worker request.
            if self.knowledge_agent is not None and view is None:
                view = self.knowledge_agent.retrieve(request, deadline, planning_context=dict(
                    guard_context=deepcopy(self.planning_guard_context),
                    recovery_support_policy=self.recovery_support_policy,
                    plan_horizon=self.limits['max_plan_steps'],
                    **planning_context,
                    **({'recovery_replan_handoffs': handoffs} if handoffs else {})))
                if self.journal:
                    with (self.journal / ('knowledge-' + digest(request['request_id']) + '.json')).open('x', encoding='utf-8') as stream:
                        json.dump(view, stream, ensure_ascii=False, indent=2)
                if view["audit"]["retrieval_status"] not in ("ready", "partial"):
                    self._event("knowledge_retrieval_rejected", **view["audit"])
                    # Contract errors do not consume a plan slot. They may
                    # degrade to current observations, but transport/budget
                    # failures must never initiate another paid request.
                    blocked = self._blocked_transport_response(request)
                    from orchestrator.knowledge_retrieval import RECOVERABLE_RETRIEVAL_ERRORS
                    recoverable = view['audit']['stop_reason'] in RECOVERABLE_RETRIEVAL_ERRORS
                    if blocked or not recoverable:
                        self.knowledge_queries.append(deepcopy(view))
                        if blocked:
                            return blocked
                        response = canonical_path_round_response(request, "knowledge_retrieval_unavailable",
                                                                   fallback_policy="path_stage_v1")
                        response["metadata"]["knowledge_context"] = deepcopy(view["audit"])
                        return self._finish(request, response, "deterministic_fallback")
                    from orchestrator.knowledge_retrieval import discard_rejected_knowledge
                    view = discard_rejected_knowledge(view)
                    self._event('knowledge_degraded_to_current_observation',
                                reason=view['audit']['stop_reason'], plan_slot_consumed=False)
                elif view['audit']['retrieval_status'] == 'partial':
                    self._event('knowledge_partial_results_supplied', **view['audit'])
            if self.knowledge_agent is not None:
                from orchestrator.knowledge_retrieval import decision_evidence
                escalation["retrieved_knowledge"] = decision_evidence(view)
            cards = self._cards(request, deadline, view=view)
            if self.planning_card_capacity is not None:
                escalation['skill_card_capacity'] = deepcopy(self.planning_card_capacity)
            if self.knowledge is not None:
                escalation["knowledge_context"] = deepcopy(self.knowledge_queries[-1]["audit"])
            if ((self.knowledge_agent is not None or
                    getattr(self.planner, 'provider_contract', None) == 'strategy_plan_v1')
                    and deadline - time.monotonic() <= .5):
                stop_reason = ('knowledge_planning_deadline' if self.knowledge_agent is not None
                               else 'strategy_planning_deadline')
                response = canonical_path_round_response(request, stop_reason,
                                                          fallback_policy='path_stage_v1')
                return self._finish(request, response, 'deterministic_fallback')
            self.planner_calls += 1
            self._event("sol_requested", **escalation)
            plan_event_start = len(self.events)
            result = self.planner.plan(request, cards, escalation,
                max_steps=self.limits["max_plan_steps"],
                timeout_seconds=max(.01, min(120, deadline - time.monotonic() - .1)))
            if self.knowledge_agent is not None:
                result.setdefault('metadata', {})['knowledge_retrieval'] = deepcopy(view['audit'])
            self.planner_decisions.append(dict(request_id=request["request_id"], result=deepcopy(result)))
            self._record_plan_skill_review(request, result, 'planner')
            if result["status"] == "planned":
                self.plan, self.plan_result = result["plan"], result
                self.index, self.failed, self.recovery_used = 0, False, 0
                self._event("sol_plan_accepted", plan_id=self.plan["plan_id"], steps=len(self.plan["steps"]),
                            recovery_skill_ids=self.plan["recovery_skill_ids"])
                if self.plan.get('plan_contract') == 'conditional_plan_v4':
                    self._event('skill_review_recorded', plan_id=self.plan['plan_id'],
                        reviews=self.plan['skill_review'], model_self_report=True,
                        physical_success_implied=False)
                response = self._execute(request, deadline)
                if response:
                    return response
            else:
                details = {}
                if getattr(self.planner, "provider_contract", None) in ("conditional_plan_v2", "conditional_plan_v3", "conditional_plan_v4", "strategy_plan_v1"):
                    details = dict(contract_error=result["metadata"].get("plan_contract_error"),
                        contract_diagnostics=result["metadata"].get("plan_contract_diagnostics"))
                self._event("sol_rejected", reason=result["fallback_reason"], **details)
            blocked = self._blocked_transport_response(request)
            if blocked:
                return blocked
            rejection_feedback = self._record_planning_rejection(
                request, result, plan_event_start, correction_ordinal)
            if rejection_feedback is None:
                break
            correction_ordinal += 1
            reason = rejection_feedback['rejection_reason']
            # The next iteration is bounded by the unchanged role budget and
            # deadline. A non-support response (including any transport,
            # timeout, contract, configuration or budget error) stops here.
        blocked = self._blocked_transport_response(request)
        if blocked:
            return blocked
        if self.fallback_calls < self.limits["fallback_calls"] and deadline - time.monotonic() > 5:
            response = self._scout(request, deadline, "cheap_fallback")
            if response:
                return response
        response = canonical_path_round_response(request, "hierarchy_plan_unavailable",
                                                   fallback_policy="path_stage_v1")
        return self._finish(request, response, "deterministic_fallback")

    rank = handle

    def close(self):
        self.scout.close()
