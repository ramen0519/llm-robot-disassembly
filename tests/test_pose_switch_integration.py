"""Combined pose carry and authorized part switch, with injected outcomes only.

StrategyDecisionService and ExplicitActionDFS are production code. The graph,
model replies and verifier are fixtures. ``ASAP_RedMax`` in fixture provenance
exercises the contract, not a new physical observation. Network and subprocess
entry points are blocked; these tests claim no CAD or robot execution success.
"""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_failover import POLICY as SWITCH_POLICY
from orchestrator.strategy_plan import CONTRACT, compile_plan
from orchestrator.strategy_service import StrategyDecisionService
from tests.test_accepted_pose_carry import bound_fixture_result, fixture_worker
from tests.test_explicit_action_planner import IDENTITY
from tests.test_strategy_failover import authorized
from tests.test_strategy_service import wire
from worker.accepted_pose_carry import FIELD, entry_reference, validate


class PoseSwitchIntegrationTests(unittest.TestCase):
    def setUp(self):
        for target in ('urllib.request.OpenerDirector.open',
                       'socket.create_connection', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('NO_API_OR_NEW_PHYSICS'))
            blocked = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(blocked.assert_not_called)

    def run_chain(self, *, planner_calls=2, tamper_child=False):
        requests, physics, planned = [], [], []
        source_matrix = deepcopy(IDENTITY)
        source_matrix[0][3], source_matrix[2][3] = 7.25, 0.75

        def reply(req, cards, escalation, **kwargs):
            planned.append(deepcopy(req))
            if len(req['remaining_parts']) == 4:
                value = authorized(req, primary='a', alternative='b')
            else:
                value = wire(req, parts=['a', 'c'], pose='keep_orientation_v1',
                             supports='preserve_add_v1')
                for step in value['steps']:
                    step['stability_failover'] = None
            return dict(status='planned',
                plan=compile_plan(value, req, cards, strategy_failure_policy=SWITCH_POLICY),
                metadata=dict(llm_live=False, model_returned='INJECTED_FIXTURE'))

        planner = Mock(provider_contract=CONTRACT, strategy_failure_policy=SWITCH_POLICY)
        scout = Mock(provider_contract=CONTRACT, strategy_failure_policy=SWITCH_POLICY)
        planner.plan.side_effect = reply
        scout.plan.side_effect = AssertionError('UNEXPECTED_SCOUT')
        with patch('orchestrator.hierarchical_service.retrieve', return_value=[]):
            service = StrategyDecisionService(scout, planner, {}, dict(scout_calls=0,
                planner_calls=planner_calls, fallback_calls=0, max_plan_steps=8, recovery_per_step=1),
                strategy_failure_policy=SWITCH_POLICY,
                recovery_support_policy='preserve_existing_v1',
                planning_feedback_policy='repair_evidence_v2')

        def decide(req):
            requests.append(deepcopy(req))
            if tamper_child and FIELD in req:
                req = deepcopy(req)
                carried_id = req[FIELD]['active']['pose_id']
                next(p for p in req['pose_catalog'] if p['pose_id'] == carried_id)['matrix'][0][3] += 1
            return service.handle(req)

        def verify(context, choice, params):
            physics.append((deepcopy(context), deepcopy(choice)))
            success = not (len(context['remaining_parts']) == 4 and choice['part_id'] == 'a')
            outcome = bound_fixture_result(context, choice, success)
            outcome.update(path_success=True, stability_success=success,
                           raw_upstream=dict(parts_fall=[] if success else ['c']))
            return outcome

        def poses(parts):
            # Same rotation but a deliberately different geometric translation
            # after removal: rotation-only matching would lose the original pose.
            matrix = deepcopy(source_matrix)
            if len(parts) < 4:
                matrix[0][3] += 100 * (4 - len(parts))
            return [dict(pose_id='geometric_' + str(len(parts)), matrix=matrix)]

        worker = fixture_worker(decide, verify, poses)
        worker.plan(budget=12)
        return dict(worker=worker, report=worker.report(), service=service,
                    planner=planner, scout=scout, requests=requests, physics=physics,
                    planned=planned, source_matrix=source_matrix)

    def test_two_stability_failures_switch_then_replan_keep_exact_pose_and_continue(self):
        run = self.run_chain()
        report, service = run['report'], run['service']
        self.assertTrue(report['success'], report['stop_msg'])
        self.assertEqual([a['choice']['part_id'] for a in report['attempts']], ['a', 'a', 'b', 'a', 'c'])
        self.assertEqual(report['total_n_eval'], 5)  # injected calls, not RedMax measurements
        self.assertEqual(run['planner'].plan.call_count, 2)
        run['scout'].plan.assert_not_called()
        self.assertEqual(run['requests'][2]['remaining_parts'], ['a', 'b', 'c', 'd'])
        alternative = service.decisions[2]['response']['metadata']['strategy_failover']
        self.assertEqual(alternative['target_part_id'], 'b')
        self.assertEqual(run['planned'][1]['remaining_parts'], ['a', 'c', 'd'])
        self.assertTrue(any(e.get('reason') == 'strategy_failover_part_removed_replan'
                            for e in service.events))
        child = run['requests'][3]
        self.assertIsNotNone(entry_reference(child))
        self.assertFalse(validate(child)['physics_success_implied'])
        self.assertTrue(validate(child)['requires_fresh_verification'])
        child_pose = next(p for p in child['pose_catalog'] if p['pose_id'] == child[FIELD]['active']['pose_id'])
        self.assertEqual(child_pose['matrix'], run['source_matrix'])
        self.assertNotEqual(child['pose_catalog'][0]['matrix'], child_pose['matrix'])
        self.assertTrue(all(c['pose_matrix'] == run['source_matrix'] for c, _ in run['physics']))
        self.assertEqual([a['status'] for a in report['attempts']], ['UNKNOWN', 'UNKNOWN', 'PASS', 'PASS', 'PASS'])
        self.assertTrue(service.decisions[-1]['response']['metadata']['plan_execution_without_api'])

    def test_alternative_removal_requires_new_plan_and_cannot_spend_exhausted_model_budget(self):
        run = self.run_chain(planner_calls=1)
        self.assertFalse(run['report']['success'])
        self.assertEqual([a['choice']['part_id'] for a in run['report']['attempts']], ['a', 'a', 'b'])
        self.assertEqual(len(run['physics']), 3)
        self.assertEqual(run['planner'].plan.call_count, 1)
        self.assertIsNotNone(entry_reference(run['requests'][-1]))
        removed = [e for e in run['service'].events if e['kind'] == 'strategy_failover_removed']
        self.assertEqual(len(removed), 1)
        self.assertFalse(removed[0]['original_plan_continuation_authorized'])
        self.assertTrue(removed[0]['requires_new_plan'])

    def test_tampered_child_translation_stops_before_next_model_or_verifier_call(self):
        run = self.run_chain(tamper_child=True)
        self.assertFalse(run['report']['success'])
        self.assertEqual(len(run['physics']), 3)
        self.assertEqual(run['planner'].plan.call_count, 1)
        self.assertEqual(run['requests'][-1]['remaining_parts'], ['a', 'c', 'd'])
        self.assertIsNotNone(validate(run['requests'][-1]))
        self.assertTrue(run['report']['stop_msg'].startswith('decision halted:'), run['report']['stop_msg'])


if __name__ == '__main__':
    unittest.main()
