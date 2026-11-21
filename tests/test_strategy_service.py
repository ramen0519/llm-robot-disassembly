"""Injected strategic model and physics; real candidate code and upstream DFS methods."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_plan import CONTRACT, compile_plan
from orchestrator.strategy_service import StrategyDecisionService
from tests.test_hierarchical_planner import request, feedback, policy
from tests.test_explicit_action_planner import make_planner, result_for


def wire(req, *, attempts=6, parts=None, procedure='axis_search_v1', sid=None,
         pose='select_catalog_once_v1', supports='preserve_add_v1'):
    parts = parts if parts is not None else sorted(req['candidate_ids'])[:len(req['remaining_parts'])-1]
    return dict(request_id=req['request_id'], state_id=req['state_id'],
        candidate_order=list(req['candidate_ids']), recovery_skill_ids=[],
        steps=[dict(part_id=p, procedure=procedure, registered_skill_id=sid,
            pose_policy=pose, support_policy=supports, max_attempts=attempts,
            evidence_ids=[]) for p in parts])


class StrategyServiceTests(unittest.TestCase):
    def setUp(self):
        self.net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.blocked_network = self.net.start()
        self.addCleanup(self.net.stop)
        self.addCleanup(self.blocked_network.assert_not_called)

    def service(self, *, attempts=6, scout_calls=0, planner_calls=1, rows=None, reply=None):
        planner = Mock(provider_contract=CONTRACT)
        def plan(req, cards, escalation, **kwargs):
            value = reply(req, cards, escalation) if reply else wire(req, attempts=attempts)
            return dict(status='planned', plan=compile_plan(value, req, cards),
                metadata=dict(llm_live=False, model_returned='INJECTED', call_id='INJECTED'))
        planner.plan.side_effect = plan
        scout = Mock(provider_contract=CONTRACT)
        scout.plan.side_effect = plan
        limits = dict(policy()['episode'], scout_calls=scout_calls, planner_calls=planner_calls)
        with patch('orchestrator.hierarchical_service.retrieve', return_value=rows or []):
            service = StrategyDecisionService(scout, planner, {}, limits,
                recovery_support_policy='preserve_existing_v1', planning_feedback_policy='repair_evidence_v2')
        return service, planner, scout

    def worker(self, service, verifier=None, **kwargs):
        return make_planner(decision=service.handle, verifier=verifier,
            path_selection_policy='path_probe_round_v1', fallback_policy='path_stage_v1',
            decision_failure_policy='halt_v1', **kwargs)

    def test_full_worker_two_removals_and_final_pair_one_model_plan(self):
        service, planner, _ = self.service()
        worker = self.worker(service)
        worker.plan(budget=4)
        report = worker.report()
        self.assertTrue(report['success'], report)
        self.assertEqual(report['total_n_eval'], 2)
        self.assertEqual(report['final_pair_calls'], 1)
        self.assertEqual(planner.plan.call_count, 1)
        self.assertEqual([a['choice']['part_id'] for a in report['attempts']], ['a', 'b'])
        self.assertTrue(service.decisions[1]['response']['metadata']['plan_execution_without_api'])

    def test_path_failure_changes_direction_without_second_model_call(self):
        first = []
        def verify(context, choice, params):
            first.append(deepcopy(choice))
            result = result_for(context, choice, feasible=len(first) > 1)
            result['backend'] = 'ASAP_RedMax'  # injected label for protocol coverage only
            return result
        service, planner, _ = self.service()
        worker = self.worker(service, verify)
        worker.plan(budget=5)
        self.assertTrue(worker.report()['success'])
        self.assertEqual(planner.plan.call_count, 1)
        self.assertEqual(first[0]['part_id'], first[1]['part_id'])
        self.assertEqual(first[0]['pose_id'], first[1]['pose_id'])
        self.assertNotEqual(first[0]['direction_id'], first[1]['direction_id'])

    def test_scout_uses_strategy_and_multiple_actions_not_direct_action_contract(self):
        service, planner, scout = self.service(scout_calls=1)
        worker = self.worker(service)
        worker.plan(budget=4)
        self.assertTrue(worker.report()['success'])
        self.assertEqual(scout.plan.call_count, 1)
        scout.handle.assert_not_called()
        planner.plan.assert_not_called()

    def test_stability_failure_adds_reported_support_without_new_model_call(self):
        def verify(context, choice, params):
            feasible = choice['part_id'] != 'a' or 'b' in choice['support_ids']
            result = result_for(context, choice, feasible)
            result.update(backend='ASAP_RedMax', path_success=True, stability_success=feasible,
                          raw_upstream=dict(parts_fall=['b'] if not feasible else []))
            return result
        service, planner, _ = self.service()
        worker = self.worker(service, verify)
        worker.plan(budget=5)
        report = worker.report()
        self.assertTrue(report['success'], report)
        self.assertEqual(planner.plan.call_count, 1)
        choices = [row['choice'] for row in report['attempts']]
        self.assertEqual(choices[0]['support_ids'], [])
        self.assertEqual(choices[1]['support_ids'], ['b'])
        self.assertEqual(choices[0]['direction_id'], choices[1]['direction_id'])
        self.assertEqual(choices[0]['pose_id'], choices[1]['pose_id'])

    def test_strategy_limit_halts_unknown_without_hidden_physics_or_fallback(self):
        service, planner, _ = self.service(attempts=1)
        worker = self.worker(service, lambda c, a, p: dict(result_for(c, a, False), backend='ASAP_RedMax'))
        worker.plan(budget=8)
        report = worker.report()
        self.assertFalse(report['success'])
        self.assertEqual(report['total_n_eval'], 1)
        self.assertTrue(report['stop_msg'].startswith('decision halted:'), report)
        self.assertEqual(planner.plan.call_count, 1)
        self.assertTrue(service.decisions[-1]['response']['metadata']['decision_halt_v1'])
        self.assertFalse(any(e['kind'] == 'strategy_proposal_issued' and
            e['metadata']['strategy_attempt_number'] > 1 for e in service.events))

    def test_current_orientation_required_and_no_implicit_initial_default(self):
        service, planner, _ = self.service(reply=lambda r, c, e: wire(r, pose='keep_orientation_v1'))
        worker = self.worker(service)
        worker.plan(budget=4)
        self.assertEqual(worker.report()['total_n_eval'], 0)
        self.assertFalse(worker.report()['success'])
        self.assertTrue(any(e.get('reason') == 'orientation_reference_missing' for e in service.events))

    def test_duplicate_request_and_changed_scope_fail_closed(self):
        service, planner, _ = self.service()
        req = request(); original = deepcopy(req)
        response = service.handle(req)
        self.assertEqual(service.handle(deepcopy(req)), response)
        self.assertEqual(req, original)
        self.assertEqual(planner.plan.call_count, 1)
        req['state_id'] += ':changed'
        with self.assertRaisesRegex(ValueError, 'request_id_conflict'):
            service.handle(req)

    def test_late_physical_pass_does_not_advance_plan(self):
        service, planner, _ = self.service()
        req = request(); response = service.handle(req)
        new = feedback(req, response, False)
        new['episode_feedback'][-1].update(status='PASS', path_success=True,
            stability_success=True, accepted_for_transition=False, deadline_exceeded=True)
        new['path_stage_observations'][-1].update(status='PASS', path_success=True, deadline_exceeded=True)
        service.handle(new)
        self.assertEqual(service.index, 0)

    def test_missing_feedback_invalidates_not_assumed_success(self):
        service, planner, _ = self.service()
        req = request(); service.handle(req)
        req['request_id'] += ':next'
        response = service.handle(req)
        self.assertIsNone(response['choice'])
        self.assertTrue(any(e.get('reason') == 'missing_or_ambiguous_verifier_feedback' for e in service.events))

    def test_rejected_plan_does_not_become_canonical_part_action(self):
        service, planner, _ = self.service()
        planner.plan.side_effect = None
        planner.plan.return_value = dict(status='rejected', plan=None,
            fallback_reason='injected_contract_failure', metadata={})
        worker = self.worker(service)
        worker.plan(budget=4)
        self.assertEqual(worker.report()['total_n_eval'], 0)


if __name__ == '__main__': unittest.main()
