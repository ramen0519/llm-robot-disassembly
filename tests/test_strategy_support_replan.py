"""Offline orchestration/ledger integration, injected models and no physics."""
from copy import deepcopy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from urllib.error import HTTPError
from unittest.mock import Mock, patch

from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.hierarchy_budget import HierarchyBudget
from orchestrator.strategy_plan import CONTRACT, compile_plan
from orchestrator.strategy_planner import StrategyPlanService
from orchestrator.strategy_service import StrategyDecisionService
from tests.test_explicit_action_service import provider
from tests.test_hierarchical_planner import policy
from tests.test_skill_program import choice
from tests.test_strategy_candidates import state, observe
from tests.test_strategy_service import wire
from worker.decision_halt import is_halt_response
from worker.explicit_action_contract import digest, validate_request

ROOT = Path(__file__).absolute().parents[1]


def supported_request():
    req = state()
    observe(req, choice(support_ids=['b', 'c']), stability=True)
    return validate_request(req)


class StrategySupportReplanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/'runs', prefix='support_replan_fixture_')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.req = supported_request()
        self.bad = wire(self.req, parts=['b'], pose='keep_orientation_v1')
        self.good = wire(self.req, parts=['b'], pose='keep_orientation_v1', supports='replace_once_v1')
        environment = patch.dict('os.environ', OPENAI_API_KEY='injected-fixture-not-a-real-key')
        environment.start(); self.addCleanup(environment.stop)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.network = network.start(); self.addCleanup(network.stop)
        self.addCleanup(self.network.assert_not_called)
        self.bodies = []

    def service(self, answers, *, planner_calls=3, role_calls=4, role_usd='1',
                scout_answer=None, journal=False, failure_policy='legacy_v1'):
        p = policy()
        p['roles']['planner'].update(max_calls=role_calls, max_usd=role_usd)
        p['max_calls'] = sum(cfg['max_calls'] for cfg in p['roles'].values())
        self.shared = HierarchyBudget(self.folder/'global.sqlite', p,
                                      transport_failure_policy='latch_on_error_v1')
        queue = iter(answers)
        def transport(url, headers, body, timeout):
            self.bodies.append(deepcopy(body))
            answer = next(queue)
            if callable(answer): answer = answer()
            if isinstance(answer, Exception): raise answer
            result = provider(answer); result['model'] = p['roles']['planner']['model']
            return result
        planner = StrategyPlanService(p['roles']['planner'], self.folder/'planner.sqlite',
            self.shared.transport('planner', backend=transport), role='planner',
            strategy_failure_policy=failure_policy)
        def scout_transport(url, headers, body, timeout):
            if scout_answer is None: raise AssertionError('UNEXPECTED_SCOUT')
            result = provider(scout_answer); result['model'] = p['roles']['scout']['model']
            return result
        scout = StrategyPlanService(p['roles']['scout'], self.folder/'scout.sqlite',
            self.shared.transport('scout', backend=scout_transport), role='scout',
            strategy_failure_policy=failure_policy)
        limits = dict(p['episode'], scout_calls=int(scout_answer is not None),
                      planner_calls=planner_calls, fallback_calls=0)
        snapshot = dict(kind='frozen_generated_skills_v1', entries=[], fixture=False,
                        usable_for_actual_planning=True, physics_reuse_allowed=False, hard_pruning=False)
        snapshot['content_sha256'] = digest(snapshot)
        service = StrategyDecisionService(scout, planner, snapshot, limits, timeout_seconds=120,
            journal_dir=self.folder/'journal' if journal else None,
            recovery_support_policy='preserve_existing_v1', planning_feedback_policy='repair_evidence_v2',
            transport_guard=self.shared.transport_blocked, strategy_failure_policy=failure_policy)
        return service

    def packet(self, index):
        return json.loads(self.bodies[index]['input'][0]['content'])

    def test_conflict_replans_with_same_worker_state_and_separate_ledger_identity(self):
        service = self.service([self.bad, self.good]); before = digest(self.req)
        answer = service.handle(self.req)
        self.assertFalse(is_halt_response(answer), answer)
        self.assertEqual(answer['choice']['part_id'], 'b')
        self.assertNotIn('b', answer['choice']['support_ids'])
        self.assertEqual(digest(self.req), before)
        self.assertEqual(service.planner_calls, 2)
        feedback = self.packet(1)['plan_rejection_feedback']
        self.assertEqual(feedback['worker_request_id'], self.req['request_id'])
        self.assertEqual(feedback['state_id'], self.req['state_id'])
        self.assertEqual(feedback['request_sha256'], before)
        self.assertEqual(feedback['completed_plan_call'], 1)
        self.assertEqual(feedback['correction_ordinal'], 1)
        self.assertEqual(feedback['support_preflight']['conflicting_support_ids'], ['b'])
        self.assertEqual(feedback['support_preflight']['physics_calls'], 0)
        self.assertEqual(service.handle(deepcopy(self.req)), answer)
        self.assertEqual(len(self.bodies), 2)
        with closing(sqlite3.connect(self.folder/'planner.sqlite')) as db:
            rows = db.execute('SELECT request_id,payload_json,reserved_call FROM rank_requests').fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row[0] for row in rows}), 2)
        self.assertEqual(sum(row[2] for row in rows), 2)
        self.assertEqual({json.loads(row[1])['request']['request_id'] for row in rows}, {self.req['request_id']})
        self.assertEqual(self.shared.summary()['injected_provider_attempts'], 2)
        self.assertEqual(self.shared.summary()['live_api_attempts'], 0)
        self.assertEqual(self.shared.summary()['automatic_http_retries'], 0)

    def test_identical_invalid_plans_consume_finite_role_slots_and_halt(self):
        service = self.service([self.bad]*3)
        before = digest(self.req); result = service.handle(self.req)
        self.assertTrue(is_halt_response(result))
        self.assertEqual(service.planner_calls, 3)
        self.assertEqual(len(self.bodies), 3)
        self.assertEqual([self.packet(i)['plan_rejection_feedback']['correction_ordinal'] for i in (1,2)], [1,2])
        self.assertEqual(self.shared.summary()['injected_provider_attempts'], 3)
        self.assertEqual(digest(self.req), before)
        self.assertIsNone(service.pending)
        self.assertFalse(any(event['kind']=='strategy_proposal_issued' for event in service.events))

    def test_role_call_exhaustion_does_not_send_an_extra_request(self):
        service = self.service([self.bad, self.good], role_calls=1)
        self.assertTrue(is_halt_response(service.handle(self.req)))
        self.assertEqual(len(self.bodies), 1)
        self.assertEqual(service.planner_decisions[-1]['result']['fallback_reason'], 'call_budget_exhausted')
        self.assertEqual(self.shared.summary()['injected_provider_attempts'], 1)

    def test_cost_exhaustion_never_enters_transport(self):
        service = self.service([self.good], role_usd='.0000001')
        self.assertTrue(is_halt_response(service.handle(self.req)))
        self.assertEqual(len(self.bodies), 0)
        self.assertEqual(service.planner_decisions[-1]['result']['fallback_reason'], 'cost_budget_exhausted')
        self.assertEqual(self.shared.summary()['transport_attempts'], 0)

    def test_timeout_after_semantic_rejection_latches_all_remaining_requests(self):
        service = self.service([self.bad, TimeoutError('injected timeout'), self.good])
        result = service.handle(self.req)
        self.assertTrue(is_halt_response(result))
        self.assertEqual(len(self.bodies), 2)
        self.assertEqual(self.shared.summary()['injected_provider_attempts'], 2)
        self.assertTrue(self.shared.transport_blocked()['blocked'])
        second = deepcopy(self.req); second['request_id'] += ':another'
        self.assertTrue(is_halt_response(service.handle(second)))
        self.assertEqual(len(self.bodies), 2)

    def test_http_429_after_rejection_stops_without_http_retry(self):
        error = HTTPError('https://api.openai.com/v1/responses', 429, 'injected limit', {}, None)
        service = self.service([self.bad, error, self.good])
        self.assertTrue(is_halt_response(service.handle(self.req)))
        self.assertEqual(len(self.bodies), 2)
        self.assertTrue(self.shared.transport_blocked()['blocked'])
        self.assertEqual(self.shared.summary()['automatic_http_retries'], 0)

    def test_one_deadline_is_not_renewed_for_semantic_correction(self):
        clock = [0.0]
        def expired_answer():
            clock[0] = 121.0
            return self.bad
        service = self.service([expired_answer, self.good])
        with patch('orchestrator.hierarchical_service.time.monotonic', side_effect=lambda: clock[0]):
            self.assertTrue(is_halt_response(service.handle(self.req)))
        self.assertEqual(len(self.bodies), 1)

    def test_slow_local_cards_do_not_reserve_a_late_request(self):
        clock = [0.0]
        service = self.service([self.good])
        def slow_cards(*args, **kwargs):
            clock[0] = 121.0
            return []
        with patch.object(service, '_cards', side_effect=slow_cards), patch(
                'orchestrator.hierarchical_service.time.monotonic', side_effect=lambda: clock[0]):
            result = service.handle(self.req)
        self.assertTrue(is_halt_response(result))
        self.assertEqual(result['fallback_reason'], 'strategy_planning_deadline')
        self.assertEqual(len(self.bodies), 0)
        self.assertEqual(service.planner_calls, 0)

    def test_generic_contract_error_does_not_enable_semantic_loop(self):
        bad = deepcopy(self.bad); bad['steps'][0]['part_id'] = 'absent'
        service = self.service([bad, self.good])
        self.assertTrue(is_halt_response(service.handle(self.req)))
        self.assertEqual(len(self.bodies), 1)
        self.assertFalse(any(event['kind']=='strategy_plan_rejection_feedback' for event in service.events))

    def test_scout_conflict_is_preserved_in_first_sol_feedback(self):
        service = self.service([self.good], scout_answer=self.bad)
        self.assertFalse(is_halt_response(service.handle(self.req)))
        feedback = self.packet(0)['plan_rejection_feedback']
        self.assertEqual(feedback['completed_plan_call'], 0)
        self.assertEqual(feedback['support_preflight']['selected_part_id'], 'b')
        self.assertEqual(feedback['support_preflight']['conflicting_support_ids'], ['b'])
        self.assertEqual(self.shared.summary()['roles']['scout'], 1)
        self.assertEqual(self.shared.summary()['roles']['planner'], 1)

    def test_knowledge_retrieval_and_immutable_journal_written_once_per_request(self):
        service = self.service([self.bad, self.good], journal=True)
        service.knowledge_agent = Mock()
        view = dict(audit=dict(retrieval_status='ready', source='INJECTED'))
        service.knowledge_agent.retrieve.return_value = view
        with patch.object(service, '_cards', return_value=[]), patch(
                'orchestrator.knowledge_retrieval.decision_evidence', return_value=dict(test='same-view')):
            result = service.handle(self.req)
            self.assertFalse(is_halt_response(result))
            self.assertEqual(service.handle(deepcopy(self.req)), result)
        self.assertEqual(service.knowledge_agent.retrieve.call_count, 1)
        self.assertEqual(len(list((self.folder/'journal').glob('knowledge-*.json'))), 1)
        self.assertEqual(len(self.bodies), 2)
        self.assertEqual(self.packet(0)['retrieved_knowledge'], self.packet(1)['retrieved_knowledge'])

    def test_runtime_future_step_guard_produces_replan_feedback_without_releasing_supports(self):
        service = self.service([self.good])
        original = wire(self.req, parts=['a','b'], pose='keep_orientation_v1')
        plan = compile_plan(original, self.req, [])
        # The future step is conditioned on a worker transition. This isolated
        # hook replay uses the recorded current reference; no physics is run.
        result = dict(status='planned', request_id=self.req['request_id'], plan=plan, metadata={})
        from orchestrator.strategy_candidates import build_anchor, resolve_reference
        anchor = build_anchor(self.req, plan['steps'][1], resolve_reference(self.req))
        self.assertIsNone(anchor['anchor'])
        service.events.append(dict(kind='strategy_scope_unavailable', reason=anchor['reason'],
                                   diagnostics=anchor['diagnostics']))
        feedback = service._record_planning_rejection(self.req, result, 0, 1)
        self.assertEqual(feedback['rejected_phase'], 'runtime_support_guard')
        self.assertEqual(feedback['rejected_plan_id'], plan['plan_id'])
        self.assertEqual(feedback['support_preflight']['current_support_ids'], ['b','c'])
        self.assertEqual(len(self.bodies), 0)

    def test_conditional_failure_alternative_guard_replans_through_real_provider(self):
        from orchestrator.strategy_failover import POLICY
        req = state()
        for direction in ('xp', 'yp'):
            observe(req, choice(part_id='d', direction_id=direction, support_ids=['b','c']), stability=True)
        req = validate_request(req)
        first = wire(req, parts=['d'], pose='keep_orientation_v1')
        first['steps'][0]['stability_failover'] = dict(after_failures=2,
            alternatives=wire(req, parts=['b'], pose='keep_orientation_v1')['steps'])
        corrected = wire(req, parts=['b'], pose='keep_orientation_v1', supports='replace_once_v1')
        corrected['steps'][0]['stability_failover'] = None
        service = self.service([first, corrected], failure_policy=POLICY)
        before = digest(req)
        result = service.handle(req)
        self.assertFalse(is_halt_response(result), result)
        self.assertEqual(result['choice']['part_id'], 'b')
        self.assertNotIn('b', result['choice']['support_ids'])
        self.assertEqual(len(self.bodies), 2)
        feedback = self.packet(1)['plan_rejection_feedback']
        self.assertEqual(feedback['rejected_phase'], 'runtime_support_guard')
        self.assertEqual(feedback['rejection_reason'], 'preserved_support_conflicts_with_selected_part')
        self.assertEqual(feedback['support_preflight']['selected_part_id'], 'b')
        self.assertEqual(feedback['support_preflight']['current_support_ids'], ['b','c'])
        self.assertEqual(digest(req), before)
        self.assertEqual(self.shared.summary()['injected_provider_attempts'], 2)
        self.assertFalse(any(event['kind']=='physics_feedback' for event in service.events))

    def test_tampered_support_facts_are_not_sent_as_correction(self):
        service = self.service([self.bad])
        service.handle(self.req)
        result = deepcopy(service.planner_decisions[0]['result'])
        result['metadata']['plan_contract_diagnostics']['support_preflight']['current_support_ids'] = []
        self.assertIsNone(service._planning_rejection_feedback(self.req, result, len(service.events)))

    def test_legacy_service_hook_does_not_enable_replanning(self):
        hook = HierarchicalDecisionService._planning_rejection_feedback
        self.assertIsNone(hook(None, self.req, dict(status='rejected', fallback_reason='invalid_strategy_plan'), 0))


if __name__ == '__main__':
    unittest.main()
