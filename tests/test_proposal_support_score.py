"""Offline fixtures. Native stability replies are injected; no physics is run."""
import unittest

from worker.path_only_sequence_proposal import build_proposal
from worker.proposal_stability import assess_step
from worker.proposal_support_score import score_plan


PARTS = ['a', 'b', 'c', 'd', 'e']
POSE = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def observation(step, outcome, cap):
    if outcome == 'NOT_RUN':
        return dict(no_support_status='NOT_RUN', up_to_cap_status='NOT_RUN',
                    reason='audit_deadline', held_parts=None, trials=[],
                    all_remaining_fixed=None)

    def check_stable(asset, assembly, fixed, free, **kwargs):
        if outcome == 'ERROR':
            raise RuntimeError('simulated native error')
        if outcome == 'UNKNOWN':
            return False, None
        if outcome == 'REJECTED':
            return False, list(free)  # More than the cap at the chosen step.
        if len(fixed) < outcome:
            return False, [free[0]]
        return True, None

    return assess_step(step, asset_folder='fixture', assembly_dir='fixture',
                       support_cap=min(cap, len(step['child'])), check_stable=check_stable,
                       wall_time_s=10.0, clock=lambda: 0.0)


def proposal(outcomes, cap=2):
    edges = []
    for index, outcome in enumerate(outcomes):
        child = PARTS[index + 1:]
        step = dict(attempt_id=index + 1, part_move=PARTS[index], parent=PARTS[index:],
                    child=child, action=[1, 0, 0], pose=POSE, path_fixed_parts=child,
                    path_pass=True, accepted_for_transition=True, deadline_exceeded=False,
                    terminal=len(child) == 1)
        step['stability'] = observation(step, outcome, cap)
        edges.append(step)
    result = build_proposal(PARTS, edges)
    result.update(initial_parts=list(PARTS), execution_status='OK',
                  input_unchanged=True, asap_sources_unchanged=True)
    return result


class ProposalSupportScoreTests(unittest.TestCase):
    def assert_unscorable(self, result, cap=2):
        try:
            summary = score_plan(result, support_cap=cap)
        except ValueError:
            return
        self.assertIs(summary['eligible'], False)
        self.assertIsNone(summary['score'])

    def test_complete_chain_uses_gravity_witness_count_including_last_remainder(self):
        summary = score_plan(proposal([0, 1, 1, 1]), support_cap=2)
        self.assertIs(summary['eligible'], True)
        self.assertEqual(summary['step_counts'], [0, 1, 1, 1])
        self.assertEqual(summary['score'], [1, 3])
        self.assertEqual(summary['step_count'], 4)
        self.assertEqual(summary['stable_prefix'], 4)
        self.assertEqual(summary['passed_steps'], 4)
        self.assertIsNone(summary['first_blocking_step'])

    def test_plans_compare_peak_first_and_sum_breaks_peak_ties(self):
        def score(counts):
            result = score_plan(proposal(counts), support_cap=2)
            self.assertIs(result['eligible'], True)
            return result['score']

        a = score([0, 1, 1, 1])
        b = score([0, 0, 2, 1])
        self.assertEqual(a, [1, 3])
        self.assertEqual(b, [2, 3])
        self.assertLess(a, b)
        self.assertLess(a, score([0, 0, 2, 0]))  # Lower peak despite higher sum.
        self.assertLess(score([0, 0, 1, 1]), a)

    def test_later_passes_count_in_total_but_stop_prefix_at_first_nonpass(self):
        for outcome in ('REJECTED', 'UNKNOWN', 'ERROR', 'NOT_RUN'):
            with self.subTest(outcome=outcome):
                result = proposal([0, outcome, 2, 1])
                self.assertEqual(result['steps'][1]['stability']['up_to_cap_status'], outcome)
                summary = score_plan(result, support_cap=2)
                self.assertIs(summary['eligible'], False)
                self.assertIsNone(summary['score'])
                self.assertEqual(summary['step_counts'], [0, None, 2, 1])
                self.assertEqual(summary['step_count'], 4)
                self.assertEqual(summary['stable_prefix'], 1)
                self.assertEqual(summary['passed_steps'], 3)
                self.assertEqual(summary['first_blocking_step'], 2)

    def test_partial_path_is_unscored_even_if_every_audited_step_passed(self):
        result = proposal([0, 1])
        self.assertFalse(result['complete'])
        summary = score_plan(result, support_cap=2)
        self.assertIs(summary['eligible'], False)
        self.assertIsNone(summary['score'])
        self.assertEqual(summary['stable_prefix'], 2)
        self.assertEqual(summary['passed_steps'], 2)
        self.assertEqual(summary['step_counts'], [0, 1])
        self.assertEqual(summary['step_count'], 2)

    def test_missing_audit_cannot_be_inferred_from_the_verified_path(self):
        result = proposal([0, 1, 1, 1])
        del result['steps'][0]['stability']
        summary = score_plan(result, support_cap=2)
        self.assertIs(summary['eligible'], False)
        self.assertIsNone(summary['score'])
        self.assertEqual(summary['step_counts'], [None, 1, 1, 1])
        self.assertEqual(summary['stable_prefix'], 0)
        self.assertEqual(summary['first_blocking_step'], 1)
        self.assertEqual(summary['passed_steps'], 3)

    def test_missing_run_integrity_or_worker_error_prevents_ranking(self):
        for field in ('input_unchanged', 'asap_sources_unchanged'):
            with self.subTest(field=field):
                result = proposal([0, 1, 1, 1])
                del result[field]
                self.assert_unscorable(result)
        result = proposal([0, 1, 1, 1])
        result['execution_status'] = 'ERROR'
        self.assert_unscorable(result)

    def test_over_cap_and_invalid_held_parts_cannot_qualify(self):
        self.assert_unscorable(proposal([0, 0, 2, 1]), cap=1)
        for held in (None, [], ['a'], ['foreign'], ['b', 'b'], ['c']):
            with self.subTest(held=held):
                result = proposal([1, 0, 0, 0])
                result['steps'][0]['stability']['held_parts'] = held
                self.assert_unscorable(result)

    def test_claimed_pass_requires_matching_timely_native_success(self):
        changes = [dict(success=False), dict(parts_fall=['c']), dict(status='UNKNOWN'),
                   dict(deadline_exhausted=True), dict(parts_fix=[]), dict(parts_move=[]),
                   dict(all_remaining_fixed=True)]
        for change in changes:
            with self.subTest(change=change):
                result = proposal([1, 0, 0, 0])
                result['steps'][0]['stability']['trials'][-1].update(change)
                self.assert_unscorable(result)
        result = proposal([1, 0, 0, 0])
        result['steps'][0]['stability']['trials'] = []
        self.assert_unscorable(result)
        result = proposal([1, 0, 0, 0])
        del result['steps'][0]['stability']['trials'][-1]['parts_fall']
        self.assert_unscorable(result)

    def test_path_needs_explicit_accepted_and_timely_transition(self):
        for field, value in [('path_pass', False), ('accepted_for_transition', False),
                             ('deadline_exceeded', True)]:
            with self.subTest(field=field):
                result = proposal([0, 1, 1, 1])
                result['steps'][0][field] = value
                self.assert_unscorable(result)
        result = proposal([0, 1, 1, 1])
        del result['steps'][0]['accepted_for_transition']
        self.assert_unscorable(result)


if __name__ == '__main__':
    unittest.main()
