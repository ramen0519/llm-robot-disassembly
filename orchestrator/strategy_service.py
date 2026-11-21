"""Opt-in strategic LLM decisions with bounded, locally computed proposals.

The worker remains the only owner of physics, DFS and acceptance. This service
never labels local search exhaustion as physical impossibility.
"""
from copy import deepcopy
import time

from orchestrator.conditional_plan import response_for, task_scope, verify_geometry
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.strategy_plan import CONTRACT, wire_step
from orchestrator.strategy_candidates import (build_anchor, propose, resolve_reference,
                                              pose_policy_context, support_policy_context)
from orchestrator.strategy_evidence import project_plan_evidence
from worker.explicit_action_contract import action_id, digest, validate_choice
from worker.decision_halt import halt_response
from orchestrator import strategy_failover


class StrategyDecisionService(HierarchicalDecisionService):
    def __init__(self, scout, planner, snapshot, limits, *, strategy_failure_policy='legacy_v1',
                 skill_review_policy='legacy_v1', **kwargs):
        from orchestrator.strategy_skill_review import LEGACY, validate_policy
        self.skill_review_policy = validate_policy(skill_review_policy)
        if self.skill_review_policy != LEGACY and any(
                getattr(provider, 'skill_review_policy', None) != self.skill_review_policy
                for provider in (scout, planner)):
            raise ValueError('skill_review_policy_provider_mismatch')
        if any(getattr(provider, 'provider_contract', None) != CONTRACT for provider in (scout, planner)):
            raise ValueError('strategy_providers_required_for_both_roles')
        if kwargs.get('recovery_support_policy') != 'preserve_existing_v1':
            raise ValueError('strategy_requires_preserve_existing_guard')
        self.strategy_failure_policy = strategy_failover.validate_policy(strategy_failure_policy)
        if (self.strategy_failure_policy != strategy_failover.LEGACY and any(
                getattr(provider, 'strategy_failure_policy', None) != self.strategy_failure_policy
                for provider in (scout, planner))):
            raise ValueError('strategy_failure_policy_provider_mismatch')
        super().__init__(scout, planner, snapshot, limits, **kwargs)
        self.strategy_cursor = None
        self.orientation_reference = None
        self.restore_revisions = {}
        self.restore_event_numbers = {}
        self.accepted_entry_revisions = {}
        self.failover_branch = None
        self.stability_failure_ledger = {}

    def handle(self, request):
        from worker.explicit_action_contract import validate_request
        request = validate_request(request)
        if self.strategy_failure_policy != strategy_failover.LEGACY:
            strategy_failover.observe(request, self.stability_failure_ledger)
        key = (request['run_id'], request['state_id'])
        carry = request.get('accepted_pose_carry')
        new_entry = False
        if carry is not None:
            from worker.accepted_pose_carry import validate, entry_reference
            carry = validate(request)
            entry = (carry['activation_revision'], digest(carry['active']), carry['entry_attempt_count'])
            previous_entry = self.accepted_entry_revisions.get(key)
            if previous_entry is not None and (entry[0] < previous_entry[0]
                    or entry[0] == previous_entry[0] and entry != previous_entry):
                raise ValueError('stale_or_conflicting_accepted_pose_entry')
            new_entry = (entry_reference(request) is not None
                         and (previous_entry is None or entry[0] > previous_entry[0]))
        elif key in self.accepted_entry_revisions:
            raise ValueError('missing_accepted_pose_entry_context')
        restored = request.get('virtual_search_restore')
        previous = self.restore_revisions.get(key)
        if restored is not None:
            revision = restored['node_visit_revision']
            identity = digest({k:v for k,v in restored.items()
                               if k not in ('request_id','request_binding_sha256')})
            if previous is not None and (revision < previous[0] or
                    revision == previous[0] and identity != previous[1]):
                raise ValueError('stale_or_conflicting_virtual_restore')
            source_event = restored['source']['event_id']
            prefix = request['run_id'] + ':attempt:'
            suffix = source_event[len(prefix):] if source_event.startswith(prefix) else ''
            restore_number = int(suffix) if suffix.isdigit() else None
            if carry is not None and (restore_number is None or restore_number <=
                    int(carry['active']['source']['event_id'].rsplit(':',1)[1])):
                raise ValueError('stale_virtual_restore_before_accepted_entry')
            self.restore_revisions[key] = (revision, identity)
            self.restore_event_numbers[key] = restore_number
        elif previous is not None:
            entry_number = int(carry['active']['source']['event_id'].rsplit(':',1)[1]) if carry else None
            restored_number = self.restore_event_numbers.get(key)
            if not (new_entry and restored_number is not None and entry_number > restored_number):
                raise ValueError('missing_virtual_restore_context')
            self.restore_revisions.pop(key)
            self.restore_event_numbers.pop(key)
            self._event('virtual_restore_superseded_by_accepted_entry', state_id=request['state_id'],
                activation_revision=carry['activation_revision'], source_event=carry['active']['source']['event_id'],
                state_history_reset=False, physical_success_implied=False)
        if carry is not None:
            self.accepted_entry_revisions[key] = entry
        return super().handle(request)

    rank = handle

    def _invalidate(self, reason):
        self.strategy_cursor = None
        self.failover_branch = None
        return super()._invalidate(reason)

    def _reconcile_failover(self, request, pending):
        """An alternative removal is a real worker transition, never step+1."""
        self.pending = None
        matches = [row for row in request['episode_feedback'] if row.get('action_id') == pending['action_id']]
        if len(matches) != 1:
            return self._invalidate('missing_or_ambiguous_verifier_feedback')
        observed = matches[0]
        binding = pending.get('strategy_failover_binding') or {}
        observed_choice = deepcopy(observed.get('choice'))
        if isinstance(observed_choice, dict):
            observed_choice['evidence_ids'] = []
        if (observed_choice != binding.get('choice')
                or observed.get('remaining_parts') != pending['remaining']
                or observed.get('selected_pose_matrix') != binding.get('pose_matrix')):
            return self._invalidate('strategy_failover_feedback_binding_changed')
        success = (observed.get('accepted_for_transition') is True and observed.get('status') == 'PASS'
            and observed.get('deadline_exceeded') is False and observed.get('path_success') is True
            and observed.get('stability_success') is True)
        expected = set(pending['remaining']) - {pending['part_id']} if success else set(pending['remaining'])
        self._event('physics_feedback', action_id=pending['action_id'], evidence_id=observed['evidence_id'],
            role=pending['role'], status=observed['status'], accepted=success,
            generated_skill_id=pending.get('generated_skill_id'), strategy_failover=True)
        self.last_failure_context = (dict(choice=deepcopy(pending['choice']),
            remaining_parts=list(pending['remaining']), evidence_id=observed['evidence_id'],
            path_success=observed.get('path_success'), stability_success=observed.get('stability_success'))
            if not success else None)
        if expected != set(request['remaining_parts']):
            self.last_failure_context = None
            return self._invalidate('backtrack_or_unexpected_state')
        if (self.plan is None or self.failover_branch is None
                or pending.get('plan_id') != self.plan['plan_id']
                or pending['strategy_failover'] != self.failover_branch['branch_id']):
            return self._invalidate('strategy_failover_feedback_identity_changed')
        if success:
            self._event('strategy_failover_removed', branch_id=self.failover_branch['branch_id'],
                removed_part=pending['part_id'], remaining_parts=request['remaining_parts'],
                original_primary_part=self.plan['steps'][self.index]['part_id'],
                original_plan_continuation_authorized=False, requires_new_plan=True)
            return self._invalidate('strategy_failover_part_removed_replan')
        self.active_guard = None
        self.failed = True
        return 'observed_plan_failure'

    def _reconcile(self, request):
        pending = deepcopy(self.pending)
        old_cursor = self.strategy_cursor
        reason = (self._reconcile_failover(request, pending) if pending and pending.get('strategy_failover')
                  else super()._reconcile(request))
        if pending:
            matches = [row for row in request['episode_feedback'] if row.get('action_id') == pending['action_id']]
            if len(matches) == 1:
                observed = matches[0]
                if observed.get('selected_pose_matrix') is not None:
                    # These are known commanded simulation conditions, not a
                    # successful result or measured robot hold state. Preserve
                    # their support guard even if the verdict is uncertain.
                    self.orientation_reference = deepcopy(observed)
                    self.orientation_reference['choice'] = deepcopy(pending['choice'])
                if self.strategy_cursor is old_cursor and old_cursor is not None:
                    old_cursor['verified_attempts'] += 1
                    self._event('strategy_attempt_observed', plan_id=old_cursor['plan_id'],
                        step=old_cursor['step'], verified_attempts=old_cursor['verified_attempts'],
                        action_id=pending['action_id'], accepted=observed.get('accepted_for_transition') is True)
            if reason in ('backtrack_or_unexpected_state', 'missing_or_ambiguous_verifier_feedback'):
                self.orientation_reference = None
        return reason

    def _reference(self, request):
        return resolve_reference(request, self.orientation_reference)

    def _planning_context(self, request):
        ref = self._reference(request)
        # No duplicate raw verifier diagnostics in the planning packet. These
        # fields preserve the same host reference used by runtime guards.
        ref = ({key: deepcopy(ref[key]) for key in ('scope', 'node_visit_revision',
            'remaining_parts', 'action_id', 'evidence_id', 'choice',
            'selected_pose_matrix', 'reference_error') if key in ref} if ref else None)
        result = dict(strategy_orientation_reference=ref,
                      strategy_pose_preflight=pose_policy_context(request, ref))
        if self.strategy_failure_policy != strategy_failover.LEGACY:
            result['strategy_stability_history'] = strategy_failover.summary(request, self.stability_failure_ledger)
        return result

    def _active_step(self):
        primary = self.plan['steps'][self.index]
        if self.failover_branch is None:
            return primary
        branch = self.failover_branch
        if branch['plan_id'] != self.plan['plan_id'] or branch['step'] != self.index:
            raise ValueError('strategy_failover_branch_identity_changed')
        return primary['stability_failover']['alternatives'][branch['alternative_index']]

    def _maybe_failover(self, request):
        """Switch only to a separately model-authorized same-state alternative."""
        if self.strategy_failure_policy == strategy_failover.LEGACY:
            return
        primary = self.plan['steps'][self.index]
        permission = primary.get('stability_failover')
        if permission is None:
            return
        step = self._active_step()
        trigger = strategy_failover.trigger(request, self.stability_failure_ledger,
            step['part_id'], permission['after_failures'])
        if trigger is None:
            return
        alternative_index = self.failover_branch['alternative_index'] + 1 if self.failover_branch else 0
        if alternative_index >= len(permission['alternatives']):
            self._event('strategy_failover_exhausted', trigger=trigger, remaining_parts=request['remaining_parts'],
                proves_impossible=False, hard_pruning=False)
            self._invalidate('strategy_failover_options_exhausted')
            return
        alternative = permission['alternatives'][alternative_index]
        if (alternative['part_id'] not in request['candidate_ids']
                or set(alternative['expected_remaining']) != set(request['remaining_parts'])):
            self._invalidate('strategy_failover_remaining_guard_failed')
            return
        branch = dict(plan_id=self.plan['plan_id'], step=self.index, alternative_index=alternative_index,
            branch_id=strategy_failover.branch_identity(self.plan['plan_id'], self.index, alternative_index, trigger),
            trigger=trigger, target_part_id=alternative['part_id'],
            expected_remaining=list(request['remaining_parts']))
        self.failover_branch = branch
        self.strategy_cursor = None
        authorization = wire_step(primary, self.strategy_failure_policy)['stability_failover']
        self._event('strategy_failover_selected', **branch, selection_authority='explicit_model_alternative',
            primary_part_id=primary['part_id'], model_authorization=authorization,
            model_authorization_sha256=digest(authorization),
            failed_part_removed=False, current_state_mutated=False, hard_pruning=False,
            accepted_alternative_requires_replan=True)

    def _planning_preflight(self, request):
        availability = self._planning_context(request)['strategy_pose_preflight']
        if availability['planning_ready']:
            return None
        self._event('strategy_planning_preflight_rejected', pose_preflight=availability,
                    api_calls=0, physics_calls=0)
        response = halt_response(request, 'strategy_reference_preflight_unavailable')
        response['metadata'].update(decision_scope=CONTRACT, pose_preflight=availability,
                                    fallback_action_executed=False)
        return self._finish(request, response, 'decision_halt')

    def _planning_rejection_feedback(self, request, result, event_start):
        """Only a proved current support conflict permits a semantic new plan.

        No timeout, API error, generic parse error or unknown state is repaired
        by resending. The worker request and all observed physics stay intact.
        """
        if type(result) is not dict:
            return None
        metadata = result.get('metadata') or {}
        if (metadata.get('transport_diagnostics') or metadata.get('transport_guard')
                or result.get('fallback_reason') in (
                    'api_error', 'api_timeout', 'unknown_outcome', 'transport_circuit_open',
                    'call_budget_exhausted', 'cost_budget_exhausted', 'ledger_error')):
            return None
        diagnostics = metadata.get('plan_contract_diagnostics') or {}
        contract_error = metadata.get('plan_contract_error')
        if (result.get('status') == 'rejected'
                and result.get('fallback_reason') == 'invalid_strategy_plan'
                and contract_error == 'strategy_first_step_support_conflict'
                and result.get('request_id') == request['request_id']):
            facts = diagnostics.get('support_preflight')
            phase, reason = 'compile_support_preflight', contract_error
        elif result.get('status') == 'planned':
            failures = [event for event in self.events[event_start:]
                        if event.get('kind') == 'strategy_scope_unavailable'
                        and event.get('reason') == 'preserved_support_conflicts_with_selected_part']
            if not failures:
                return None
            facts = (failures[-1].get('diagnostics') or {}).get('support_preflight')
            phase, reason = 'runtime_support_guard', 'preserved_support_conflicts_with_selected_part'
        else:
            return None
        if type(facts) is not dict:
            return None
        try:
            checked = support_policy_context(request, self._reference(request), step=dict(
                part_id=facts['selected_part_id'], support_policy=facts['selected_support_policy'],
                procedure=facts['selected_procedure']))
        except (ValueError, TypeError, KeyError):
            return None
        if (checked != facts or not checked['planning_ready']
                or checked['support_policy_available']
                or not checked['conflicting_support_ids']):
            return None
        return dict(kind='strategy_support_replan_v1',
            worker_request_id=request['request_id'], state_id=request['state_id'],
            request_sha256=digest(request), rejection_reason=reason, rejected_phase=phase,
            rejected_plan_id=(result.get('plan') or {}).get('plan_id'),
            rejected_wire_sha256=diagnostics.get('wire_value_sha256'),
            support_preflight=deepcopy(checked))

    @staticmethod
    def _same_orientation(left, right):
        return all(abs(left[i][j] - right[i][j]) <= 1e-6 for i in range(3) for j in range(3))

    def _execution_plan(self):
        # Historical source snapshots belong to host plan provenance, not to
        # the skill's current observations. Do not duplicate them inside the
        # bounded interpreter packet or let them consume its input allowance.
        if self.plan is None:
            return None
        return {key: deepcopy(value) for key, value in self.plan.items() if key != 'evidence_binding'}

    def _skill_proposal(self, request, step, cursor, deadline):
        # The bounded replacement search belongs to the trusted candidate
        # generator, not arbitrary registered code. Keep this runtime guard
        # even though the model schema/compiler reject the combination too.
        if step['support_policy'] not in ('preserve_add_v1', 'replace_once_v1'):
            return None, 'strategy_registered_support_policy_not_supported'
        identity = step['registered_skill_id']
        row = next((row for row in self.rows if row['strategy_id'] == identity), None)
        if row is None or self.plan['registered_program_hashes'].get(identity) != row['program_hash']:
            return None, 'strategy_registered_program_pin_mismatch'
        result = self._skill_step(row, request, deadline)
        self._event('skill_checked', skill_id=identity, plan_id=self.plan['plan_id'],
                    triggered=bool(result and result.get('triggered')),
                    not_applicable_reason=result.get('not_applicable_reason') if result else 'deadline')
        if not result or not result.get('response'):
            from orchestrator.skill_runtime_feedback import FeedbackBudget
            self._event('skill_runtime_feedback', skill_id=identity,
                        runtime_feedback=FeedbackBudget().take(result))
            return None, 'strategy_registered_skill_abstained'
        choice = deepcopy(result['response']['choice'])
        if choice['part_id'] != step['part_id']:
            return None, 'strategy_registered_mover_mismatch'
        pose = next(p['matrix'] for p in request['pose_catalog'] if p['pose_id'] == choice['pose_id'])
        if cursor['skill_pose'] is None:
            reference = self._reference(request)
            if reference and reference.get('reference_error'):
                return None, reference['reference_error']
            if step['pose_policy'] == 'keep_orientation_v1':
                if not reference or not reference.get('selected_pose_matrix'):
                    return None, 'orientation_reference_missing'
                if not self._same_orientation(reference['selected_pose_matrix'], pose):
                    return None, 'strategy_skill_pose_change_requires_new_plan'
                if request.get('accepted_pose_carry') is not None:
                    from worker.accepted_pose_carry import resolve_pose
                    if choice['pose_id'] != resolve_pose(request, reference):
                        return None, 'strategy_skill_pose_change_requires_new_plan'
            if step['support_policy'] == 'preserve_add_v1':
                prior = (set(reference['choice']['support_ids']) if reference and
                    reference.get('scope') != 'orientation_only_after_transition' else set())
                if not prior.issubset(choice['support_ids']):
                    return None, 'strategy_skill_support_release_requires_new_plan'
        else:
            if not self._same_orientation(cursor['skill_pose'], pose):
                return None, 'strategy_skill_pose_change_requires_new_plan'
            if request.get('accepted_pose_carry') is not None and cursor['skill_pose'] != pose:
                return None, 'strategy_skill_pose_change_requires_new_plan'
            if not set(cursor['skill_supports']).issubset(choice['support_ids']):
                return None, 'strategy_skill_support_release_requires_new_plan'
        cursor['skill_pose'] = deepcopy(pose)
        cursor['skill_supports'] = list(choice['support_ids'])
        return choice, None

    def _execute(self, request, deadline):
        if self.plan is None:
            return None
        if self.plan.get('plan_contract') != CONTRACT:
            self._invalidate('strategy_contract_required')
            return None
        if self.plan.get('strategy_failure_policy', 'legacy_v1') != self.strategy_failure_policy:
            self._invalidate('strategy_failure_policy_plan_mismatch')
            return None
        if task_scope(request) != self.plan['task_scope']:
            self._invalidate('plan_task_scope_changed')
            return None
        verify_geometry(self.plan['source_geometry_parts'], request)
        primary = self.plan['steps'][self.index]
        if set(primary['expected_remaining']) != set(request['remaining_parts']):
            self._invalidate('plan_remaining_guard_failed')
            return None
        # Validate the immutable whole plan pin BEFORE interpreting a nested
        # model permission or choosing another mover from it.
        if self.strategy_failure_policy != strategy_failover.LEGACY:
            try:
                project_plan_evidence(self.plan, request, primary['evidence_ids'])
            except ValueError as error:
                self._event('strategy_evidence_rejected', reason=str(error), physical_attempt_executed=False)
                self._invalidate('strategy_evidence_binding_invalid')
                return None
        self._maybe_failover(request)
        if self.plan is None:
            return None
        step = self._active_step()
        try:
            evidence = project_plan_evidence(self.plan, request, step['evidence_ids'])
        except ValueError as error:
            self._event('strategy_evidence_rejected', reason=str(error),
                        physical_attempt_executed=False)
            self._invalidate('strategy_evidence_binding_invalid')
            return None
        # Evidence provenance belongs to the plan. A changing retrieval window
        # must not change the hash of an already anchored search procedure or
        # impersonate historical observations as current worker evidence.
        candidate_step = dict(deepcopy(step), evidence_ids=[])
        key = (self.plan['plan_id'], self.index, request['state_id'])
        if request.get('accepted_pose_carry') is not None:
            key += (request['accepted_pose_carry']['activation_revision'],)
        if self.strategy_failure_policy != strategy_failover.LEGACY:
            key += (self.failover_branch['branch_id'] if self.failover_branch else None,)
        if self.strategy_cursor is None or self.strategy_cursor['key'] != key:
            self.strategy_cursor = dict(key=key, plan_id=self.plan['plan_id'], step=self.index,
                verified_attempts=0, anchor=None, skill_pose=None, skill_supports=[])
        cursor = self.strategy_cursor
        if cursor['verified_attempts'] >= step['max_attempts']:
            self._invalidate('strategy_attempt_budget_exhausted')
            return None
        if time.monotonic() >= deadline - .1:
            self._invalidate('strategy_proposal_deadline')
            return None
        skill = step['procedure'] == 'registered_skill'
        if skill:
            if cursor['skill_pose'] is None and step['pose_policy'] == 'keep_orientation_v1':
                availability = pose_policy_context(request, self._reference(request))
                if not availability['keep_orientation_available']:
                    reason = ('orientation_reference_missing'
                        if availability['keep_orientation_unavailable_reason'] == 'orientation_reference_missing'
                        else 'orientation_unavailable_or_ambiguous')
                    self._event('strategy_scope_unavailable', reason=reason,
                                pose_preflight=availability, physical_attempt_executed=False)
                    self._invalidate(reason)
                    return None
            choice, reason = self._skill_proposal(request, step, cursor, deadline)
            diagnostics = dict(generator='registered_python', program_id=step['registered_skill_id'])
        else:
            if cursor['anchor'] is None:
                bound = build_anchor(request, candidate_step, reference=self._reference(request))
                if bound['anchor'] is None:
                    self._event('strategy_scope_unavailable', **bound)
                    self._invalidate(bound['reason'])
                    return None
                cursor['anchor'] = bound['anchor']
                self._event('strategy_anchor_bound', plan_id=self.plan['plan_id'], step=self.index, **bound)
            proposed = propose(request, candidate_step, cursor['anchor'], reference=self._reference(request))
            choice, reason, diagnostics = proposed['choice'], proposed['reason'], proposed['diagnostics']
        if choice is None:
            self._event('strategy_local_exhaustion', reason=reason, diagnostics=diagnostics,
                        proves_impossible=False, physical_verdict=None)
            self._invalidate(reason or 'strategy_no_proposal')
            return None
        choice = deepcopy(choice)
        choice['evidence_ids'] = sorted(set(choice['evidence_ids']) | set(evidence['evidence_ids']))
        meta = dict(plan_id=self.plan['plan_id'], plan_step=self.index,
            decision_scope=CONTRACT, selected_procedure=step['procedure'],
            plan_evidence=evidence['provenance'],
            local_search_diagnostics=diagnostics, strategy_attempt_number=cursor['verified_attempts'] + 1,
            strategy_attempt_limit=step['max_attempts'],
            plan_execution_without_api=self.plan['request_id'] != request['request_id'],
            generated_program_executed=skill,
            generated_skill_id=step['registered_skill_id'] if skill else None,
            generated_program_hash=self.plan['registered_program_hashes'].get(step['registered_skill_id']) if skill else None,
            skill_authorized_by_plan=skill, lower_level_choice_source='registered_python' if skill else 'bounded_strategy_code',
            strategic_change_authority=(
                'model_authorized_bounded_support_search'
                if diagnostics.get('support_replacement_authorized') else
                'first_proposal_of_model_selected_step'
                if cursor['verified_attempts'] == 0 else 'hold_pose_preserve_supports'))
        if self.failover_branch is not None:
            meta['strategy_failover'] = deepcopy(self.failover_branch)
        try:
            response = response_for(request, choice, self.plan['candidate_order'], meta)
        except (ValueError, KeyError, TypeError) as error:
            self._event('strategy_proposal_rejected', reason=str(error), physical_attempt_executed=False)
            self._invalidate('strategy_proposal_contract_rejected')
            return None
        self._event('strategy_proposal_issued', plan_id=self.plan['plan_id'], step=self.index,
                    action_id=action_id(request, choice), metadata=meta)
        return self._finish(request, response, 'plan_skill' if skill else 'plan_strategy')

    def _scout(self, request, deadline, role):
        if role == 'scout': self.scout_calls += 1
        else: self.fallback_calls += 1
        # The inexpensive model uses the SAME strategic output contract. It
        # does not regain direct control of directions/poses/support IDs.
        result = self.scout.plan(request, [], dict(reason=role,
            decision_scope=CONTRACT, recovery_support_policy=self.recovery_support_policy,
            **self._planning_context(request)),
            max_steps=self.limits['max_plan_steps'],
            timeout_seconds=max(.01, min(120, deadline-time.monotonic()-.1)))
        self.planner_decisions.append(dict(request_id=request['request_id'], role=role, result=deepcopy(result)))
        self._record_plan_skill_review(request, result, role)
        if result['status'] != 'planned':
            self._event('strategy_scout_rejected', reason=result['fallback_reason'], role=role)
            return None
        self.plan, self.plan_result = result['plan'], result
        self.index, self.failed, self.recovery_used = 0, False, 0
        self.strategy_cursor = None
        self.failover_branch = None
        self._event('strategy_scout_plan_accepted', plan_id=self.plan['plan_id'], role=role)
        return self._execute(request, deadline)

    def _finish(self, request, response, role):
        if role == 'deterministic_fallback':
            # No silent canonical part choice in this opt-in research profile.
            reason = response.get('fallback_reason') or 'strategy_plan_unavailable'
            prior = deepcopy(response.get('metadata', {}))
            response = halt_response(request, reason)
            response['metadata'].update(decision_scope=CONTRACT,
                suppressed_fallback_metadata=prior, fallback_action_executed=False)
            role = 'decision_halt'
        # A continued local search is not another LLM selection opportunity.
        # Preserve the originating model review separately from its execution.
        origin = (self.plan_result or {}).get('metadata', {})
        plan_action = role.startswith('plan_') or role == 'recovery'
        source = self.plan if plan_action and self.plan else None
        review = origin.get('skill_selection_review') if source else None
        review_error = origin.get('skill_selection_review_error') if source else None
        new_choice = bool(source and source['request_id'] == request['request_id'])
        status = ('no_new_model_choice' if source and not new_choice else
                  'legacy_not_requested' if self.skill_review_policy == 'legacy_v1' else
                  'new_model_review' if review else
                  'invalid_model_review' if review_error else 'review_unavailable')
        meta = response.setdefault('metadata', {})
        meta['skill_selection_review_context'] = dict(policy=self.skill_review_policy, status=status,
            source_request_id=source['request_id'] if source else None,
            source_plan_id=source['plan_id'] if source else None, new_model_choice=new_choice,
            physical_success_implied=False, selection_forced=False)
        if review is not None:
            meta['plan_skill_selection_review'] = deepcopy(review)
            if new_choice:
                meta['skill_selection_review'] = deepcopy(review)
        if review_error is not None:
            meta['plan_skill_selection_review_error'] = deepcopy(review_error)
            if new_choice:
                meta['skill_selection_review_error'] = deepcopy(review_error)
        result = super()._finish(request, response, role)
        branch = result.get('metadata', {}).get('strategy_failover')
        if branch is not None and self.pending is not None:
            self.pending['strategy_failover'] = branch['branch_id']
            choice = deepcopy(response['choice']); choice['evidence_ids'] = []
            self.pending['strategy_failover_binding'] = dict(choice=choice,
                pose_matrix=deepcopy(next(p['matrix'] for p in request['pose_catalog']
                                         if p['pose_id']==choice['pose_id'])))
        return result

    def close(self):
        # RankService owns no external process; close if a supplied adapter does.
        close = getattr(self.scout, 'close', None)
        if close is not None: close()
