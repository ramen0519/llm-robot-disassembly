"""Injected ASAP return values only. No physics or asset loading."""
from copy import deepcopy
import unittest

from worker.proposal_stability import assess_step


POSE = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]


class ProposalStabilityTests(unittest.TestCase):
    def setUp(self):
        self.step = dict(child=['b', 'a', 'c'], pose=deepcopy(POSE), part_move='x')
        self.now = 0.0
        self.calls = []

    def assess(self, replies, support_cap=2, wall_time_s=5.0, duration=1.0):
        def check(asset_folder, assembly_dir, fixed, free, *, pose, timeout):
            self.calls.append((fixed, free, timeout, pose))
            self.now += duration
            item = replies[len(self.calls) - 1]
            if isinstance(item, Exception):
                raise item
            return item

        return assess_step(self.step, asset_folder='assets', assembly_dir='product',
                           support_cap=support_cap, wall_time_s=wall_time_s,
                           check_stable=check, clock=lambda: self.now)

    def test_uses_original_pose_and_all_reported_free_parts_in_order(self):
        result = self.assess([(False, ['c', 'b']), (True, None)])
        self.assertEqual(self.calls, [
            ([], ['b', 'a', 'c'], 5.0, POSE), (['c', 'b'], ['a'], 4.0, POSE)])
        self.assertEqual(result['held_parts'], ['c', 'b'])
        self.assertFalse(result['all_remaining_fixed'])
        self.assertEqual(result['no_support_status'], 'REJECTED')
        self.assertEqual(result['up_to_cap_status'], 'PASS')
        self.assertEqual(result['trials'][0]['parts_fall'], ['c', 'b'])
        self.assertEqual(result['elapsed_s'], 2.0)

    def test_no_support_pass_returns_empty_witness(self):
        self.step['pose'] = None
        result = self.assess([(True, None)], support_cap=0)
        self.assertEqual(self.calls[0][3], None)
        self.assertEqual(result['no_support_status'], 'PASS')
        self.assertEqual(result['up_to_cap_status'], 'PASS')
        self.assertEqual(result['held_parts'], [])

    def test_greedy_extension_over_cap_stops_before_next_call(self):
        result = self.assess([(False, ['a']), (False, ['b', 'c'])])
        self.assertEqual(result['up_to_cap_status'], 'REJECTED')
        self.assertEqual(result['reason'], 'greedy_support_cap_exceeded')
        self.assertIsNone(result['held_parts'])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result['trials'][1]['parts_fall'], ['b', 'c'])
        self.assertEqual(result['observation_coverage']['support_search'],
                         'native_serial_greedy')

    def test_all_fixed_singleton_needs_actual_supported_return(self):
        self.step['child'] = ['a']
        result = self.assess([(False, ['a']), (True, None)])
        self.assertEqual(result['no_support_status'], 'REJECTED')
        self.assertEqual(result['no_support_reason'],
                         'native_singleton_parts_reported_initial_connectivity_unconfirmed')
        self.assertEqual(self.calls[1][:2], (['a'], []))
        self.assertTrue(result['trials'][1]['all_remaining_fixed'])
        self.assertTrue(result['all_remaining_fixed'])
        self.assertFalse(result['observation_coverage']['ground_contact_observed'])

    def test_native_timeout_does_not_reject(self):
        result = self.assess([(False, None)])
        self.assertEqual(result['no_support_status'], 'UNKNOWN')
        self.assertEqual(result['reason'], 'native_timeout')
        self.assertIs(result['trials'][0]['success'], False)
        self.assertIsNone(result['held_parts'])

    def test_late_supported_pass_preserves_raw_without_witness(self):
        result = self.assess([(False, ['a']), (True, None)], duration=2.6)
        self.assertEqual(result['no_support_status'], 'REJECTED')
        self.assertEqual(result['up_to_cap_status'], 'UNKNOWN')
        self.assertEqual(result['reason'], 'wall_time_exhausted')
        self.assertIs(result['trials'][1]['success'], True)
        self.assertTrue(result['trials'][1]['deadline_exhausted'])
        self.assertIsNone(result['held_parts'])

    def test_empty_budget_does_not_call_backend(self):
        result = self.assess([], wall_time_s=0)
        self.assertEqual(result['no_support_status'], 'UNKNOWN')
        self.assertEqual(result['up_to_cap_status'], 'UNKNOWN')
        self.assertEqual(result['trials'], [])

    def test_native_exception_preserves_first_trial_result(self):
        result = self.assess([(False, ['a']), RuntimeError('sim init')])
        self.assertEqual(result['no_support_status'], 'REJECTED')
        self.assertEqual(result['up_to_cap_status'], 'ERROR')
        self.assertEqual(result['trials'][1]['error']['type'], 'RuntimeError')
        self.assertIsNone(result['held_parts'])

    def test_invalid_native_extensions_do_not_launch_another_trial(self):
        for parts in ([], ['a', 'a'], ['x'], [None], 'a'):
            with self.subTest(parts=parts):
                self.calls.clear()
                result = self.assess([(False, parts)])
                self.assertEqual(result['up_to_cap_status'], 'ERROR')
                self.assertEqual(result['trials'][0]['parts_fall'], parts)
                self.assertEqual(len(self.calls), 1)

    def test_cannot_fix_an_already_fixed_part(self):
        result = self.assess([(False, ['a']), (False, ['a'])])
        self.assertEqual(result['up_to_cap_status'], 'ERROR')
        self.assertEqual(result['reason'], 'invalid_native_parts_fall')

    def test_invalid_child_pose_or_missing_pose_never_reaches_physics(self):
        for change in (dict(child=[]), dict(child=['x']), dict(child=['a', 'a']),
                       dict(pose=[[0, 0, 0, 0]] * 4)):
            with self.subTest(change=change):
                saved = self.step.copy()
                self.step.update(change)
                self.assertEqual(self.assess([])['up_to_cap_status'], 'ERROR')
                self.step = saved
        del self.step['pose']
        self.assertEqual(self.assess([])['up_to_cap_status'], 'ERROR')
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main()
