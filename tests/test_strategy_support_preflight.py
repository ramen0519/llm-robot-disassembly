"""Offline support permission checks and saved-input replay; no real API/physics."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_candidates import build_anchor, resolve_reference, support_policy_context
from orchestrator.strategy_failover import POLICY
from orchestrator.strategy_plan import CONTRACT, compile_plan, wire_step
from orchestrator.strategy_planner import StrategyPlanService
from tests.test_explicit_action_service import provider
from tests.test_hierarchical_planner import policy
from tests.test_skill_program import choice
from tests.test_strategy_candidates import state, observe
from tests.test_strategy_plan import card
from tests.test_strategy_service import wire
from worker.explicit_action_contract import digest, validate_request

ROOT = Path(__file__).absolute().parents[1]
SAVED = ROOT/'runs/repeat_00161_prepare_20261110/trial_3/episode'


def supported_request():
    req = state()
    observe(req, choice(support_ids=['b', 'c']), stability=True)
    return validate_request(req)


def candidate(req, part='b', support='preserve_add_v1', procedure='axis_search_v1', sid=None):
    return wire(req, parts=[part], pose='keep_orientation_v1', supports=support,
                procedure=procedure, sid=sid)


def rejection_feedback(req, step, ordinal=1, completed=1):
    return dict(kind='strategy_support_replan_v1', worker_request_id=req['request_id'],
        state_id=req['state_id'], request_sha256=digest(req),
        rejection_reason='strategy_first_step_support_conflict',
        rejected_phase='compile_support_preflight', rejected_plan_id=None,
        rejected_wire_sha256=digest(step),
        support_preflight=support_policy_context(req, step=step),
        completed_plan_call=completed, correction_ordinal=ordinal)


class SupportPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='support_preflight_')
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        blocked = network.start(); self.addCleanup(network.stop)
        self.addCleanup(blocked.assert_not_called)
        env = patch.dict('os.environ', OPENAI_API_KEY='fixture-only-not-a-real-key')
        env.start(); self.addCleanup(env.stop)

    def service(self, transport=None, **changes):
        config = dict(policy()['roles']['planner'], **changes)
        return StrategyPlanService(config, self.folder/'ledger.sqlite',
            transport if transport is not None else Mock(), role='planner')

    def envelope(self, req, feedback=None):
        escalation = dict(strategy_orientation_reference=resolve_reference(req))
        if feedback is not None:
            escalation['plan_rejection_feedback'] = feedback
        return dict(request=req, skills=[], escalation=escalation, max_steps=8,
            timeout_seconds=30, plan_provider_contract=CONTRACT,
            planning_feedback_policy='repair_evidence_v2')

    def test_permission_map_preserves_every_candidate_and_request(self):
        req = supported_request(); original = deepcopy(req)
        context = support_policy_context(req)
        self.assertTrue(context['planning_ready'])
        self.assertEqual(context['current_support_ids'], ['b', 'c'])
        self.assertEqual(context['reference_source'], 'latest_bound_command')
        self.assertEqual(set(context['allowed_first_step_support_policies_by_part']), set(req['candidate_ids']))
        self.assertEqual(context['allowed_first_step_support_policies_by_part']['b'], ['replace_once_v1'])
        self.assertEqual(context['allowed_first_step_support_policies_by_part']['a'],
            ['preserve_add_v1', 'replace_once_v1', 'bounded_support_search_v1'])
        self.assertFalse(context['automatic_support_release'])
        self.assertFalse(context['physics_success_implied'])
        self.assertEqual(req, original)

    def test_both_preserving_first_step_policies_reject_before_physics(self):
        req = supported_request()
        for permission in ('preserve_add_v1', 'bounded_support_search_v1'):
            chosen = candidate(req, support=permission)
            original = deepcopy(chosen)
            diagnostics = {}
            with self.subTest(policy=permission), self.assertRaisesRegex(ValueError, 'strategy_first_step_support_conflict'):
                compile_plan(chosen, req, [], diagnostics=diagnostics)
            self.assertEqual(diagnostics['phase'], 'support_preflight')
            self.assertEqual(diagnostics['support_preflight']['selected_part_id'], 'b')
            self.assertEqual(diagnostics['support_preflight']['conflicting_support_ids'], ['b'])
            self.assertEqual(diagnostics['support_preflight']['allowed_support_policies'], ['replace_once_v1'])
            self.assertEqual(diagnostics['support_preflight']['physics_calls'], 0)
            self.assertEqual(chosen, original)

    def test_explicit_replace_or_different_mover_keeps_current_guard(self):
        req = supported_request(); original = deepcopy(req)
        for part, permission in (('b', 'replace_once_v1'), ('d', 'preserve_add_v1'),
                                 ('d', 'bounded_support_search_v1')):
            chosen = candidate(req, part, permission)
            plan = compile_plan(chosen, req, [])
            step = dict(plan['steps'][0]); step['evidence_ids'] = []
            anchor = build_anchor(req, step, resolve_reference(req))
            self.assertIsNotNone(anchor['anchor'], anchor)
            self.assertNotIn(part, anchor['anchor']['support_ids'])
            if permission != 'replace_once_v1':
                self.assertEqual(anchor['anchor']['support_ids'], ['b', 'c'])
        self.assertEqual(req, original)

    def test_future_primary_supports_are_not_invented_or_pruned(self):
        req = supported_request()
        chosen = wire(req, parts=['a', 'b'], pose='keep_orientation_v1')
        plan = compile_plan(chosen, req, [])
        self.assertEqual(plan['steps'][1]['part_id'], 'b')
        self.assertEqual(plan['steps'][1]['support_policy'], 'preserve_add_v1')
        self.assertEqual(plan['candidate_order'], chosen['candidate_order'])

    def test_failure_alternative_supports_require_future_runtime_check(self):
        req = supported_request(); chosen = candidate(req, 'a')
        alternative = candidate(req, 'b')['steps'][0]
        chosen['steps'][0]['stability_failover'] = dict(after_failures=2, alternatives=[alternative])
        plan = compile_plan(chosen, req, [], strategy_failure_policy=POLICY)
        self.assertEqual(plan['steps'][0]['stability_failover']['alternatives'][0]['part_id'], 'b')
        self.assertEqual(plan['candidate_order'], chosen['candidate_order'])

    def test_registered_skill_cannot_bypass_support_guard_or_run_during_compile(self):
        req = supported_request(); selected = card()
        chosen = candidate(req, procedure='registered_skill', sid=selected['skill_id'])
        with self.assertRaisesRegex(ValueError, 'strategy_first_step_support_conflict'):
            compile_plan(chosen, req, [selected])
        chosen['steps'][0]['support_policy'] = 'replace_once_v1'
        plan = compile_plan(chosen, req, [selected])
        self.assertEqual(plan['steps'][0]['registered_skill_id'], selected['skill_id'])
        self.assertFalse(plan['physics_success_implied'])

    def test_missing_binding_is_unknown_not_empty_supports(self):
        req = supported_request(); req['episode_feedback'] = []; req['evidence_ids'] = []
        context = support_policy_context(req)
        self.assertFalse(context['planning_ready'])
        self.assertIsNone(context['current_support_ids'])
        self.assertTrue(all(not v for v in context['allowed_first_step_support_policies_by_part'].values()))
        with self.assertRaisesRegex(ValueError, 'strategy_reference_preflight_unavailable'):
            compile_plan(candidate(req, support='replace_once_v1'), req, [])

    def test_stale_reference_is_not_reinterpreted_as_release_authority(self):
        req = supported_request(); reference = resolve_reference(req)
        reference['choice']['support_ids'] = ['b']
        context = support_policy_context(req, reference)
        self.assertFalse(context['planning_ready'])
        self.assertEqual(context['preflight_error'], 'support_reference_not_latest_command')
        with self.assertRaisesRegex(ValueError, 'strategy_reference_preflight_unavailable'):
            compile_plan(candidate(req, support='replace_once_v1'), req, [], orientation_reference=reference)

    def test_runtime_refusal_reports_same_host_diagnostics(self):
        req = supported_request(); step = candidate(req)['steps'][0]
        bound = build_anchor(req, step, resolve_reference(req))
        self.assertIsNone(bound['anchor'])
        self.assertEqual(bound['reason'], 'preserved_support_conflicts_with_selected_part')
        self.assertEqual(bound['diagnostics']['support_preflight'], support_policy_context(req, step=step))

    def test_native_body_keeps_current_support_facts_and_rejection_outside_compacted_events(self):
        req = supported_request(); step = candidate(req)['steps'][0]
        feedback = rejection_feedback(req, step)
        envelope = self.envelope(req, feedback)
        original = deepcopy(envelope)
        body = self.service()._body(envelope)
        context = json.loads(body['input'][0]['content'])
        self.assertEqual(context['strategy_support_preflight']['current_support_ids'], ['b', 'c'])
        self.assertEqual(context['plan_rejection_feedback'], feedback)
        self.assertNotIn('plan_rejection_feedback', context['escalation'])
        self.assertIn('does NOT authorize releasing', body['instructions'])
        self.assertEqual(envelope, original)

    def test_corrected_plan_has_new_internal_key_same_worker_identity_and_cached_resend(self):
        req = supported_request()
        bad = candidate(req)
        good = candidate(req, support='replace_once_v1')
        outputs = [provider(bad), provider(good)]
        for item in outputs: item['model'] = 'gpt-6.1-sol'
        transport = Mock(side_effect=outputs)
        service = self.service(transport)
        first = service.plan(req, [], self.envelope(req)['escalation'])
        self.assertEqual(first['metadata']['plan_contract_error'], 'strategy_first_step_support_conflict')
        feedback = rejection_feedback(req, bad['steps'][0])
        escalation = self.envelope(req, feedback)['escalation']
        second = service.plan(req, [], escalation)
        self.assertEqual(second['status'], 'planned')
        self.assertEqual(second['request_id'], req['request_id'])
        self.assertEqual(second['plan']['request_id'], req['request_id'])
        self.assertTrue(second['metadata']['support_replan']['same_budget_ledger'])
        self.assertTrue(service.plan(req, [], escalation)['metadata']['cached'])
        self.assertEqual(transport.call_count, 2)
        db = sqlite3.connect(self.folder/'ledger.sqlite')
        try:
            ids = [row[0] for row in db.execute('SELECT request_id FROM rank_requests')]
        finally:
            db.close()
        self.assertIn(req['request_id'], ids)
        self.assertIn(second['metadata']['support_replan']['ledger_request_id'], ids)
        self.assertEqual(len(ids), 2)

    def test_correction_still_uses_original_call_budget(self):
        req = supported_request(); bad = candidate(req)
        raw = provider(bad); raw['model'] = 'gpt-6.1-sol'
        transport = Mock(return_value=raw)
        service = self.service(transport, max_calls=1)
        service.plan(req, [], self.envelope(req)['escalation'])
        followup = self.envelope(req, rejection_feedback(req, bad['steps'][0]))['escalation']
        result = service.plan(req, [], followup)
        self.assertEqual(result['fallback_reason'], 'call_budget_exhausted')
        self.assertFalse(result['metadata']['attempted'])
        self.assertEqual(transport.call_count, 1)

    def test_forged_or_stale_correction_rejected_without_call(self):
        req = supported_request(); base = rejection_feedback(req, candidate(req)['steps'][0])
        changes = [lambda f:f.update(worker_request_id='stale'), lambda f:f.update(state_id='stale'),
            lambda f:f.update(request_sha256='0'*64), lambda f:f.update(correction_ordinal=0),
            lambda f:f.update(completed_plan_call=True), lambda f:f.update(rejection_reason='api_timeout'),
            lambda f:f['support_preflight'].update(current_support_ids=[]),
            lambda f:f['support_preflight'].update(physics_success_implied=True)]
        service = self.service()
        for change in changes:
            feedback = deepcopy(base); change(feedback)
            with self.subTest(feedback=feedback):
                result = service.plan(req, [], self.envelope(req, feedback)['escalation'])
                self.assertEqual(result['fallback_reason'], 'invalid_request')
                self.assertFalse(result['metadata']['attempted'])
        service.transport.assert_not_called()

    def test_same_correction_identity_with_changed_payload_is_not_a_new_attempt(self):
        req = supported_request(); good = candidate(req, support='replace_once_v1')
        raw = provider(good); raw['model'] = 'gpt-6.1-sol'
        transport = Mock(return_value=raw)
        service = self.service(transport)
        escalation = self.envelope(req, rejection_feedback(req, candidate(req)['steps'][0]))['escalation']
        self.assertEqual(service.plan(req, [], escalation)['status'], 'planned')
        changed = dict(escalation, unrelated_change=True)
        result = service.plan(req, [], changed)
        self.assertEqual(result['fallback_reason'], 'request_id_conflict')
        self.assertEqual(transport.call_count, 1)

    def test_saved_trial3_rejected_before_anchor_with_actionable_facts(self):
        decisions_path, plans_path = SAVED/'hierarchy_decisions.json', SAVED/'hierarchy_plans.json'
        before = [path.read_bytes() for path in (decisions_path, plans_path)]
        req = json.loads(before[0])[-1]['request']
        stored = json.loads(before[1])[-1]['result']['plan']
        chosen = dict(request_id=req['request_id'], state_id=req['state_id'],
            candidate_order=stored['candidate_order'], recovery_skill_ids=[],
            steps=[wire_step(step, POLICY) for step in stored['steps']])
        diagnostics = {}
        with self.assertRaisesRegex(ValueError, 'strategy_first_step_support_conflict'):
            compile_plan(chosen, req, [], diagnostics=diagnostics,
                         orientation_reference=resolve_reference(req), strategy_failure_policy=POLICY)
        facts = diagnostics['support_preflight']
        self.assertEqual(facts['current_support_ids'], ['1', '8'])
        self.assertEqual(facts['selected_part_id'], '8')
        self.assertEqual(facts['conflicting_support_ids'], ['8'])
        self.assertEqual(facts['allowed_support_policies'], ['replace_once_v1'])
        repaired = deepcopy(chosen); repaired['steps'][0]['support_policy'] = 'replace_once_v1'
        compiled = compile_plan(repaired, req, [], orientation_reference=resolve_reference(req),
                                strategy_failure_policy=POLICY)
        self.assertEqual(compiled['steps'][0]['part_id'], '8')
        self.assertEqual([path.read_bytes() for path in (decisions_path, plans_path)], before)


if __name__ == '__main__':
    unittest.main()
