"""Offline facts/permission tests plus immutable 00365 saved-input replay."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from orchestrator.strategy_candidates import build_anchor, resolve_reference, support_policy_context
from orchestrator.strategy_reconsideration import stability_reconsideration_context
from tests.test_skill_program import append_observation, choice
from tests.test_strategy_candidates import state, strategy
from worker.explicit_action_contract import validate_request

ROOT = Path(__file__).absolute().parents[1]
SAVED = ROOT / 'runs/fifteen_part_prepare_20261110/episode'
TURNED = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def two_poses():
    req = state()
    req['pose_catalog'].append(dict(pose_id='turned', matrix=deepcopy(TURNED)))
    return req


def failed(req=None, **changes):
    req = two_poses() if req is None else req
    append_observation(req, choice(**changes), path=True, stability=False)
    return req


class StrategyReconsiderationTests(unittest.TestCase):
    def setUp(self):
        blocker = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        mocked = blocker.start()
        self.addCleanup(blocker.stop)
        self.addCleanup(mocked.assert_not_called)

    def test_initial_summary_is_not_failure_or_pose_permission(self):
        req = two_poses(); original = deepcopy(req)
        info = stability_reconsideration_context(req)
        self.assertEqual(info['bound_failure_count'], 0)
        self.assertEqual(info['catalog_pose_count'], 2)
        self.assertEqual(info['recorded_pose_count'], 0)
        self.assertFalse(info['reconsider_mover_pose_or_support'])
        self.assertIsNone(info['current_reference_pose_id'])
        self.assertFalse(info['current_reference_legal_to_keep'])
        self.assertTrue(info['preview_is_not_action_permission'])
        self.assertEqual(req, original)

    def test_bound_stability_facts_are_pose_scoped_and_citable(self):
        req = failed(); original = deepcopy(req)
        info = stability_reconsideration_context(req)
        row = info['bound_failures_by_pose'][0]
        self.assertEqual(row['bound_stability_failures'], 1)
        self.assertEqual(row['bound_path_failures'], 0)
        self.assertEqual(row['tested_mover_ids'], ['a'])
        self.assertEqual(row['distinct_failed_stability_configurations'], 1)
        self.assertEqual(row['stability_failure_evidence_ids'], req['evidence_ids'])
        self.assertEqual(info['poses_without_recorded_actions'], ['turned'])
        self.assertTrue(info['reconsider_mover_pose_or_support'])
        self.assertFalse(info['selection_forced'])
        self.assertFalse(info['candidate_pruning'])
        self.assertEqual(req, original)

    def test_path_failure_does_not_become_stability_failure(self):
        req = two_poses(); append_observation(req)
        info = stability_reconsideration_context(req)
        self.assertFalse(info['reconsider_mover_pose_or_support'])
        self.assertEqual(info['bound_failures_by_pose'][0]['bound_path_failures'], 1)
        self.assertEqual(info['bound_failures_by_pose'][0]['bound_stability_failures'], 0)

    def test_late_error_unbound_or_different_state_do_not_supply_stability_facts(self):
        for kind in ['late', 'error', 'backend', 'state', 'pose', 'support', 'missing']:
            req = failed()
            row = req['episode_feedback'][-1]
            stage = req['path_stage_observations'][-1]
            if kind == 'late':
                row['deadline_exceeded'] = stage['deadline_exceeded'] = True
            elif kind == 'error':
                row['status'] = stage['status'] = 'ERROR'
            elif kind == 'backend':
                stage['backend'] = 'mock'
            elif kind == 'state':
                row['remaining_parts'] = ['a', 'b']
            elif kind == 'pose':
                row['selected_pose_matrix'] = deepcopy(TURNED)
            elif kind == 'support':
                row['choice']['support_ids'] = ['b']
            else:
                req['episode_feedback'] = []; req['evidence_ids'] = []
            with self.subTest(kind=kind):
                info = stability_reconsideration_context(req)
                self.assertEqual(info['bound_failure_count'], 0)
                self.assertEqual(info['recorded_actions_without_bound_failure_count'], 1)
                self.assertFalse(info['reconsider_mover_pose_or_support'])

    def test_keep_orientation_remains_explicit_legal_choice_after_failure(self):
        req = failed(); ref = resolve_reference(req)
        info = stability_reconsideration_context(req, ref)
        step = strategy(pose_policy='keep_orientation_v1')
        anchor = build_anchor(req, step, ref)['anchor']
        self.assertTrue(info['current_reference_legal_to_keep'])
        self.assertEqual(anchor['pose_id'], info['current_reference_pose_id'])
        self.assertFalse(info['automatic_pose_change'])

    def test_select_once_preview_matches_actual_anchor_for_every_mover(self):
        req = failed(); ref = resolve_reference(req)
        info = stability_reconsideration_context(req, ref)
        for part in req['candidate_ids']:
            step = strategy(part_id=part, support_policy='replace_once_v1')
            anchor = build_anchor(req, step, ref)['anchor']
            with self.subTest(part=part):
                self.assertEqual(anchor['pose_id'], info['select_catalog_once_preview'][part]['pose_id'])
        self.assertEqual(info['select_catalog_once_preview']['a']['pose_id'], 'turned')
        self.assertFalse(info['select_catalog_once_preview']['a']['same_as_current_reference'])
        self.assertTrue(info['select_catalog_once_preview']['b']['same_as_current_reference'])

    def test_registered_skill_is_not_promised_fixed_procedure_preview(self):
        info = stability_reconsideration_context(failed())
        self.assertEqual(info['preview_procedures'], ['axis_search_v1', 'two_leg_search_v1'])
        self.assertTrue(info['registered_skill_pose_is_program_selected'])
        self.assertFalse(info['future_state_preview_valid'])
        self.assertEqual(info['physics_calls'], 0)

    def test_pose_preview_does_not_relax_current_support_conflict(self):
        req = failed(support_ids=['b', 'c'])
        info = stability_reconsideration_context(req)
        step = strategy(part_id='b', support_policy='bounded_support_search_v1')
        allowed = support_policy_context(req, step=step)
        self.assertTrue(info['support_guards_unchanged'])
        self.assertFalse(allowed['support_policy_available'])
        self.assertEqual(build_anchor(req, step, resolve_reference(req))['reason'],
                         'preserved_support_conflicts_with_selected_part')

    def test_retained_window_is_not_reported_as_complete_history(self):
        req = failed()
        append_observation(req, choice(direction_id='yp'), path=True, stability=False)
        req['episode_feedback'] = req['episode_feedback'][-1:]
        req['evidence_ids'] = [req['episode_feedback'][-1]['evidence_id']]
        info = stability_reconsideration_context(req)
        self.assertEqual(info['recorded_action_count'], 2)
        self.assertEqual(info['bound_failure_count'], 1)
        self.assertFalse(info['coverage_is_complete_episode_history'])
        self.assertEqual(info['recorded_actions_without_bound_failure_count'], 1)

    def test_optional_readable_history_absence_is_unknown_not_untried(self):
        for kind in ['both', 'stage_only']:
            req = failed()
            req.pop('path_stage_observations')
            if kind == 'both':
                req.pop('attempted_actions')
            with self.subTest(kind=kind):
                info = stability_reconsideration_context(req)
                self.assertFalse(info['available'])
                self.assertIsNone(info['bound_failure_count'])
                self.assertIsNone(info['recorded_pose_count'])
                self.assertIsNone(info['poses_without_recorded_actions'])
                self.assertIsNone(info['select_catalog_once_preview'])
                self.assertFalse(info['automatic_pose_change'])

    def test_saved_00365_exposes_ten_failures_one_pose_and_five_untried(self):
        path = SAVED / 'hierarchy_decisions.json'
        original = path.read_bytes()
        req = json.loads(original)[-1]['request']
        info = stability_reconsideration_context(req)
        self.assertEqual(info['recorded_action_count'], 13)
        self.assertEqual(info['bound_failure_count'], 12)
        self.assertEqual(info['recorded_pose_count'], 1)
        self.assertEqual(info['catalog_pose_count'], 6)
        self.assertEqual(len(info['poses_without_recorded_actions']), 5)
        rows = info['bound_failures_by_pose']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['bound_stability_failures'], 10)
        self.assertEqual(rows[0]['bound_path_failures'], 2)
        self.assertEqual(rows[0]['tested_mover_ids'], ['1', '2', '3', '9'])
        self.assertLess(len(json.dumps(info, ensure_ascii=False).encode('utf-8')), 6000)
        ref = resolve_reference(req)
        for part in req['candidate_ids']:
            anchor = build_anchor(req, strategy(part_id=part, support_policy='replace_once_v1'), ref)['anchor']
            self.assertEqual(anchor['pose_id'], info['select_catalog_once_preview'][part]['pose_id'])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(original).hexdigest())

    def test_saved_plans_explicitly_requested_keep_in_both_sol_first_steps(self):
        plans = json.loads((SAVED / 'hierarchy_plans.json').read_text(encoding='utf-8'))
        self.assertEqual(plans[0]['result']['plan']['steps'][0]['pose_policy'], 'select_catalog_once_v1')
        for row in plans[1:3]:
            self.assertEqual(row['result']['plan']['steps'][0]['pose_policy'], 'keep_orientation_v1')

    def test_summary_is_deterministic_and_does_not_rank_parts(self):
        req = failed()
        first = stability_reconsideration_context(req)
        self.assertEqual(first, stability_reconsideration_context(deepcopy(req)))
        self.assertEqual(list(first['select_catalog_once_preview']), sorted(req['candidate_ids']))
        self.assertNotIn('candidate_order', first)
        self.assertNotIn('recommended_pose', first)


if __name__ == '__main__':
    unittest.main()
