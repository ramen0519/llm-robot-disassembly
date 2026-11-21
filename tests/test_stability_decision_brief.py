"""Synthetic contract checks only; no API, saved PASS replay, or physics."""
from copy import deepcopy
import unittest

from orchestrator.explicit_action_provider_v5 import (
    current_failure_brief, decision_brief_v5, memory_alignment, WIRE_INSTRUCTIONS_V5)
from tests.test_explicit_provider_v5 import with_evidence
from tests.test_skill_program import request, append_observation
from worker.stability_diagnostics import StabilityObservation


class StabilityDecisionBriefTests(unittest.TestCase):
    def test_stage_distinction_and_raw_parts_do_not_invent_diagnostics(self):
        req = append_observation(request(), path=True, stability=False)
        req['episode_feedback'][-1]['raw_stability_parts'] = ['b']
        before = deepcopy(req)
        brief = current_failure_brief(req)
        self.assertEqual(brief['stage'], 'remaining_body_stability_not_successful')
        self.assertEqual(brief['measured_dynamic_failure_parts'], [])
        self.assertEqual(brief['observed_initial_disconnected_parts'], [])
        self.assertFalse(brief['hard_pruning'])
        self.assertEqual(req, before)
        self.assertEqual(current_failure_brief(append_observation(request()))['stage'],
                         'path_not_successful')

    def test_initial_disconnection_is_not_dynamic_falling(self):
        req = append_observation(request(), path=True, stability=False)
        observer = StabilityObservation()
        observer.checker_count = 1
        observer.data['initial_disconnected_parts'] = ['b']
        observer.returned(False, ['b'])
        req['episode_feedback'][-1]['stability_diagnostics'] = observer.data
        brief = current_failure_brief(req)
        self.assertEqual(brief['observed_initial_disconnected_parts'], ['b'])
        self.assertEqual(brief['measured_dynamic_failure_parts'], [])

    def test_stale_late_or_unbound_feedback_cannot_drive_current_repair(self):
        for mutation in ('late', 'state', 'action', 'pose'):
            req = append_observation(request(), path=True, stability=False)
            feedback = req['episode_feedback'][-1]
            if mutation == 'late':
                feedback['deadline_exceeded'] = True
            elif mutation == 'state':
                feedback['remaining_parts'] = ['a', 'base']
            elif mutation == 'action':
                feedback['action_id'] = 'action:' + '0'*64
            else:
                feedback['selected_pose_matrix'] = None
            self.assertIsNone(current_failure_brief(req), mutation)
        self.assertIsNone(current_failure_brief(request()))

    def test_memory_match_is_geometry_only_without_upgrading_reference(self):
        req = with_evidence()
        ref = req['memory_references'][0]
        ref.update(reference_only=True, same_conditions=False)
        before = deepcopy(ref)
        comparison = memory_alignment(req, ref)
        self.assertTrue(comparison['all_remaining_obj_bytes_equal'])
        self.assertTrue(comparison['source_mover_is_current_candidate'])
        self.assertFalse(comparison['physics_reuse_allowed'])
        self.assertEqual(ref, before)
        ref['source_task_context']['task']['input_sha256']['0.obj'] = 'a'*64
        self.assertFalse(memory_alignment(req, ref)['all_remaining_obj_bytes_equal'])
        self.assertFalse(memory_alignment(req, ref)['source_mover_is_current_candidate'])

    def test_partial_state_or_unknown_geometry_never_becomes_same_state_evidence(self):
        req = with_evidence()
        ref = req['memory_references'][0]
        ref['source_remaining_parts'] = ref['source_remaining_parts'][:-1]
        self.assertFalse(memory_alignment(req, ref)['remaining_part_ids_equal'])
        self.assertFalse(memory_alignment(req, ref)['all_remaining_obj_bytes_equal'])
        del ref['source_task_context']
        self.assertFalse(memory_alignment(req, ref)['source_supports_present'])

    def test_success_is_not_failure_and_instructions_preserve_choice_freedom(self):
        req = append_observation(request(), path=True, stability=True, status='PASS', accepted=True)
        self.assertIsNone(current_failure_brief(req))
        self.assertTrue(decision_brief_v5(req)['empty_citations_allowed'])
        self.assertIn('Changing only force direction or path primitive', WIRE_INSTRUCTIONS_V5)
        self.assertIn('not pruning', WIRE_INSTRUCTIONS_V5)
        self.assertIn('No citation is required', WIRE_INSTRUCTIONS_V5)


if __name__ == '__main__':
    unittest.main()
