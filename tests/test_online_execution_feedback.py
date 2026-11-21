"""Online request plumbing; injected physics and saved observations, never API."""
from copy import deepcopy
import unittest
from tests.test_explicit_action_planner import make_planner, response_for, result_for
from orchestrator.execution_observation import from_attempt
from orchestrator.python_skill_program import build_context
from orchestrator.lesson_runtime import extension_for


class OnlineExecutionFeedbackTests(unittest.TestCase):
    def test_actual_clock_and_remaining_allowance_reach_next_request(self):
        now = [10.0]
        requests = []
        def choose(request):
            requests.append(deepcopy(request))
            return response_for(request, 'a', 'xp' if len(requests) == 1 else 'yp')
        def verifier(context, choice, path):
            now[0] += 3
            value = result_for(context, choice, False)
            # Injected RedMax-shaped record for binding coverage, not physics.
            value.update(elapsed_seconds=2.5, path_success=True, stability_success=False,
                         backend='ASAP_RedMax')
            return value
        planner = make_planner(choose, verifier, clock=lambda: now[0],
                               path_selection_policy='path_probe_round_v1')
        planner.plan(budget=2, timeout=100)
        observed = requests[1]['episode_feedback'][-1]
        self.assertEqual(observed['execution'], from_attempt(planner.attempts[0]))
        self.assertEqual(observed['execution']['attempt_elapsed_seconds'], 3)
        self.assertEqual(observed['execution']['physics_elapsed_seconds'], 2.5)
        self.assertEqual(observed['execution']['remaining_wall_seconds'], 100)
        ctx, last = build_context(requests[1], extension_for())
        self.assertEqual(last['feedback']['execution'], observed['execution'])
        self.assertEqual(ctx['failure']['kind'], 'stability_failed')
        self.assertFalse(observed['execution']['accepted_for_transition'])

    def test_unspecified_allowance_stays_unknown(self):
        requests = []
        def choose(request):
            requests.append(deepcopy(request))
            return response_for(request, 'a', 'xp' if len(requests) == 1 else 'yp')
        planner = make_planner(choose, lambda c,a,p: result_for(c,a,False))
        planner.plan(budget=2, timeout=None)
        value = requests[1]['episode_feedback'][-1]['execution']
        self.assertIsNone(value['remaining_wall_seconds'])
        self.assertIsNone(value['allowance_source'])
        self.assertIsNone(value['physics_elapsed_seconds'])

    def test_raw_late_pass_is_not_accepted(self):
        now = [0.0]
        def verifier(context, choice, path):
            now[0] = 11
            return result_for(context, choice, True)
        planner = make_planner(verifier=verifier, clock=lambda: now[0])
        planner.plan(budget=2, timeout=10)
        attempt = planner.attempts[0]
        value = from_attempt(attempt)
        self.assertEqual(attempt['status'], 'PASS')
        self.assertTrue(value['deadline_exceeded'])
        self.assertFalse(value['accepted_for_transition'])
        self.assertFalse(planner.report()['success'])


if __name__ == '__main__':
    unittest.main()
