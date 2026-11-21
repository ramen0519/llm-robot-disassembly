"""DFS/strategy restoration boundaries with injected models and physics only.

The planner adapter and strategic service are real code. The graph and verifier
are fixtures, and pinned upstream DFS methods are loaded without importing ASAP
or RedMax. The ASAP_RedMax label below exercises the provenance contract only;
these tests do not establish CAD feasibility, robot restoration or live success.
"""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_candidates import build_anchor
from orchestrator.strategy_plan import CONTRACT, compile_plan
from orchestrator.strategy_service import StrategyDecisionService
from tests.test_explicit_action_planner import make_planner, response_for, result_for
from tests.test_strategy_service import wire
from worker.explicit_action_contract import action_id, canonical_response, validate_request
from worker.virtual_search_restore import FIELD, SCOPE, bind, restored_reference


def fixture_service(reply):
    """No provider/network implementation is constructed or called."""
    planner = Mock(provider_contract=CONTRACT)
    scout = Mock(provider_contract=CONTRACT)

    def plan(request, cards, escalation, **kwargs):
        value = reply(request, cards, escalation)
        return dict(status='planned', plan=compile_plan(value, request, cards),
                    metadata=dict(llm_live=False, model_returned='INJECTED',
                                  call_id='INJECTED'))

    planner.plan.side_effect = plan
    scout.plan.side_effect = AssertionError('UNEXPECTED_SCOUT')
    limits = dict(scout_calls=0, planner_calls=6, fallback_calls=0,
                  max_plan_steps=8, recovery_per_step=1)
    with patch('orchestrator.hierarchical_service.retrieve', return_value=[]):
        service = StrategyDecisionService(scout, planner, {}, limits,
            recovery_support_policy='preserve_existing_v1',
            planning_feedback_policy='repair_evidence_v2')
    return service, planner, scout


def fixture_worker(decision, verifier, **kwargs):
    return make_planner(decision=decision, verifier=verifier,
        path_selection_policy='path_probe_round_v1',
        fallback_policy='path_stage_v1', decision_failure_policy='halt_v1', **kwargs)


def fixture_result(context, choice, feasible):
    return dict(result_for(context, choice, feasible), backend='ASAP_RedMax')


class VirtualSearchRestoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        for target in ('urllib.request.OpenerDirector.open', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('NO_NETWORK_OR_SUBPROCESS'))
            mock = guard.start()
            self.addCleanup(mock.assert_not_called)
            self.addCleanup(guard.stop)

    def run_expired_feedback_restore(self):
        """Fail thirteen child proposals, then finish via another root mover."""
        requests = []

        def reply(request, cards, escalation):
            if request['remaining_parts'] == ['b', 'c', 'd']:
                return wire(request, parts=['b'], attempts=8,
                            procedure='two_leg_search_v1')
            if FIELD in request:
                return wire(request, parts=['b', 'a', 'c'], attempts=8,
                            procedure='two_leg_search_v1', pose='keep_orientation_v1')
            return wire(request, parts=['a'], attempts=8,
                        procedure='two_leg_search_v1')

        service, provider, scout = fixture_service(reply)

        def decide(request):
            requests.append(deepcopy(request))
            return service.handle(request)

        def verify(context, choice, params):
            return fixture_result(context, choice,
                                  context['remaining_parts'] != ['b', 'c', 'd'])

        worker = fixture_worker(decide, verify, parts=['a', 'b', 'c', 'd'])
        worker.plan(budget=24, state_action_budget=13, timeout=None)
        return worker, service, provider, scout, requests

    def test_thirteen_child_failures_restore_parent_without_rewriting_pass(self):
        worker, service, provider, scout, requests = self.run_expired_feedback_restore()
        report = worker.report()
        self.assertTrue(report['success'], (report['stop_msg'], report['planner_error']))
        self.assertEqual(report['total_n_eval'], 17)
        self.assertEqual([a['choice']['part_id'] for a in report['selected_plan']],
                         ['b', 'a', 'c'])
        self.assertEqual(sum(a['remaining_parts'] == ['b', 'c', 'd'] and
                             a['status'] == 'UNKNOWN' for a in report['attempts']), 13)
        old_pass = report['attempts'][0]
        restored = next(r for r in requests if FIELD in r)
        self.assertEqual(len(restored['episode_feedback']), 12)
        self.assertNotIn(old_pass['event_id'], restored['evidence_ids'])
        self.assertFalse(any(row['evidence_id'] == old_pass['event_id']
                             for row in restored['episode_feedback']))
        source = restored[FIELD]['source']
        self.assertEqual(source['event_id'], old_pass['event_id'])
        self.assertEqual(source['action_id'], old_pass['action_id'])
        self.assertEqual(source['status'], 'PASS')
        self.assertTrue(source['accepted_for_transition'])
        self.assertTrue(source['path_success'])
        self.assertTrue(source['stability_success'])
        self.assertFalse(source['deadline_exceeded'])
        self.assertEqual(source['choice']['evidence_ids'], [])
        self.assertEqual(old_pass['status'], 'PASS')
        self.assertTrue(old_pass['accepted_for_transition'])
        self.assertNotIn(old_pass['event_id'],
                         [a['event_id'] for a in report['selected_plan']])
        self.assertEqual(restored_reference(restored)['scope'], SCOPE)
        self.assertEqual(provider.plan.call_count, 4)
        scout.plan.assert_not_called()
        self.assertTrue(any(d['response']['metadata'].get('plan_execution_without_api')
                            for d in service.decisions))
        events = [e for e in worker.events if e['kind'] == 'virtual_search_restored']
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]['robot_state_changed'])
        self.assertFalse(events[0]['physical_success_implied'])
        # A restore is context, not permission to relabel or cite old evidence.
        altered = deepcopy(restored)
        altered[FIELD]['source']['status'] = 'UNKNOWN'
        with self.assertRaises(ValueError):
            validate_request(altered)
        with self.assertRaises(ValueError):
            response_for(restored, part='b', evidence_ids=[old_pass['event_id']])

    def test_multilevel_backtracking_revisits_nodes_with_monotonic_revisions(self):
        requests = []

        def decide(request):
            requests.append(deepcopy(request))
            parts = request['remaining_parts']
            count = len(request['attempted_action_ids'])
            if len(parts) == 5:
                return response_for(request, part='a' if count == 0 else 'b')
            if parts == ['b', 'c', 'd', 'e']:
                return response_for(request, part='b', direction='xp' if count == 0 else 'yp')
            if parts == ['a', 'c', 'd', 'e']:
                return response_for(request, part='a' if count == 0 else 'c')
            return canonical_response(request, 'fixture_untried', policy='path_stage_v1')

        def verify(context, choice, params):
            return fixture_result(context, choice, len(context['remaining_parts']) >= 4)

        worker = fixture_worker(decide, verify, parts=['a', 'b', 'c', 'd', 'e'])
        worker.plan(budget=20, state_action_budget=2, timeout=None)
        report = worker.report()
        self.assertEqual(report['status'], 'UNKNOWN', report['planner_error'])
        self.assertEqual(report['total_n_eval'], 10)
        self.assertFalse(report['success'])
        self.assertEqual(report['selected_plan'], [])
        restore_events = [e for e in worker.events if e['kind'] == 'virtual_search_restored']
        for parts in [('a', 'b', 'c', 'd', 'e'), ('b', 'c', 'd', 'e'), ('a', 'c', 'd', 'e')]:
            state = worker.states[parts]
            events = [e for e in restore_events if e['state_id'] == state['state_id']]
            self.assertEqual([e['node_visit_revision'] for e in events], [1, 2])
            self.assertEqual(state['node_visit_revision'], 2)
            self.assertEqual(state['n_attempts'], 2)
            self.assertEqual(len(set(state['attempted_action_ids'])), 2)
            for event in events:
                source = next(a for a in worker.attempts if a['event_id'] == event['source_event_id'])
                child = worker.states[tuple(source['context']['parts_rest'])]
                self.assertEqual(event['returned_from_state_id'], child['state_id'])
                self.assertEqual(source['state_id'], state['state_id'])
                self.assertFalse(event['robot_state_changed'])
                self.assertFalse(event['physical_success_implied'])
        revisited_child = worker.states[('c', 'd', 'e')]
        self.assertEqual(revisited_child['n_attempts'], 2)
        self.assertEqual(sum(a['remaining_parts'] == ['c', 'd', 'e'] for a in worker.attempts), 2)
        self.assertTrue(any(FIELD in r and len(r['remaining_parts']) == 5 for r in requests))
        self.assertTrue(any(FIELD in r and len(r['remaining_parts']) == 4 for r in requests))

    @staticmethod
    def append_uncertain_command(restored, *, late):
        request = deepcopy(restored)
        request['request_id'] += ':fixture-late' if late else ':fixture-error'
        choice = dict(part_id='b', pose_id='p0', skill_id='single_force_v1',
            direction_id='xp', second_direction_id=None, support_ids=['c', 'd'], evidence_ids=[])
        identity = action_id(request, choice)
        event = 'fixture:new-late' if late else 'fixture:new-error'
        status, path, stability = ('PASS', True, True) if late else ('ERROR', None, None)
        request['attempted_action_ids'].append(identity)
        request['attempted_actions'].append(dict(action_id=identity,
            **{key:value for key,value in choice.items() if key != 'evidence_ids'}))
        request['path_stage_observations'].append(dict(action_id=identity,
            path_success=path, status=status, deadline_exceeded=late,
            backend='ASAP_RedMax', source_event_id=event))
        row = dict(evidence_id=event, action_id=identity, status=status,
            remaining_parts=list(request['remaining_parts']), choice=choice,
            path_success=path, stability_success=stability, accepted_for_transition=False,
            deadline_exceeded=late, selected_pose_matrix=deepcopy(request['pose_catalog'][0]['matrix']))
        request['episode_feedback'] = (request['episode_feedback'] + [row])[-12:]
        request['evidence_ids'] = sorted(row['evidence_id'] for row in request['episode_feedback'])
        request[FIELD] = bind(request, request[FIELD])
        return validate_request(request)

    def test_error_and_late_pass_preserve_new_supports_and_cannot_reuse_old_restore(self):
        _, _, _, _, requests = self.run_expired_feedback_restore()
        restored = next(r for r in requests if FIELD in r)
        old_reference = restored_reference(restored)
        self.assertEqual(old_reference['choice']['support_ids'], [])
        for late in (False, True):
            with self.subTest(late=late):
                request = self.append_uncertain_command(restored, late=late)
                self.assertIsNone(restored_reference(request))
                service, provider, _ = fixture_service(
                    lambda r, c, e: wire(r, parts=['b'], pose='keep_orientation_v1'))
                current = service._reference(request)
                self.assertEqual(current['choice']['support_ids'], ['c', 'd'])
                self.assertFalse(current['accepted_for_transition'])
                before = deepcopy(request)
                response = service.handle(request)
                self.assertIsNotNone(response['choice'], response)
                self.assertEqual(response['choice']['support_ids'], ['c', 'd'])
                self.assertNotEqual(response['action_id'], request['attempted_action_ids'][-1])
                self.assertEqual(request, before)
                self.assertEqual(provider.plan.call_count, 1)
                step = wire(request, parts=['b'], pose='keep_orientation_v1')['steps'][0]
                rejected = build_anchor(request, step, reference=old_reference)
                self.assertIsNone(rejected['anchor'])
                self.assertEqual(rejected['reason'], 'support_reference_not_latest_command')

    def test_missing_new_command_feedback_cannot_fall_back_to_old_restore(self):
        _, _, _, _, requests = self.run_expired_feedback_restore()
        request = self.append_uncertain_command(next(r for r in requests if FIELD in r), late=False)
        request['episode_feedback'].pop()
        request['evidence_ids'] = sorted(row['evidence_id'] for row in request['episode_feedback'])
        request['request_id'] += ':missing-current-feedback'
        request[FIELD] = bind(request, request[FIELD])
        validate_request(request)
        self.assertIsNone(restored_reference(request))
        service, _, _ = fixture_service(
            lambda r, c, e: wire(r, parts=['b'], pose='keep_orientation_v1'))
        self.assertEqual(service._reference(request),
                         dict(reference_error='current_support_reference_unavailable'))
        response = service.handle(request)
        self.assertIsNone(response['choice'])
        self.assertIn('decision_halt_v1', response['metadata'])

    def test_legacy_pass_without_subverdicts_clears_previous_restore_evidence(self):
        requests = []

        def decide(request):
            requests.append(deepcopy(request))
            if len(request['remaining_parts']) == 3:
                return response_for(request, part='a' if not request['attempted_action_ids'] else 'b')
            return canonical_response(request, 'fixture_untried', policy='path_stage_v1')

        def verify(context, choice, params):
            result = fixture_result(context, choice, len(context['remaining_parts']) == 3)
            if result['feasible'] and choice['part_id'] == 'b':
                # Legacy custom verifier contract, not actual RedMax output.
                result.pop('path_success')
                result.pop('stability_success')
            return result

        worker = fixture_worker(decide, verify, parts=['a', 'b', 'c'])
        worker.plan(budget=10, state_action_budget=2, timeout=None)
        report = worker.report()
        self.assertEqual(report['status'], 'UNKNOWN', report['planner_error'])
        self.assertEqual(report['total_n_eval'], 6)
        restored = next(r for r in requests if FIELD in r)
        self.assertEqual(restored[FIELD]['source']['choice']['part_id'], 'a')
        root = worker.states[('a', 'b', 'c')]
        self.assertEqual(root['node_visit_revision'], 2)
        self.assertNotIn('virtual_restore_snapshot', root)
        unavailable = [e for e in worker.events if e['kind'] == 'virtual_search_restore_unavailable']
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0]['reason'], 'complete_subverdicts_required')
        self.assertFalse(unavailable[0]['physical_success_implied'])
        source = next(a for a in worker.attempts if a['event_id'] == unavailable[0]['source_event_id'])
        self.assertEqual(source['status'], 'PASS')
        self.assertNotIn('path_success', source['raw'])
        self.assertNotIn('stability_success', source['raw'])


if __name__ == '__main__':
    unittest.main()
