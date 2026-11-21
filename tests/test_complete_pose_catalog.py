"""API-free catalog and budget regression tests; no simulated success claims."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from worker.pose_catalog import select_pose_catalog
from worker.explicit_action_contract import validate_request
from orchestrator.explicit_action_provider_v5 import restore_request_v5, decision_brief_v5
from orchestrator.explicit_action_service import build_provider_body
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchy_budget import HierarchyBudget, HierarchyBudgetError, validate_policy
from tests.test_hierarchical_planner import policy, request, feedback, plan_value
from orchestrator.conditional_plan import validate_plan, action_for_step


class CompletePoseTests(unittest.TestCase):
    def catalog(self):
        result = {}
        for index in range(6):
            # Different rigid rotations, so future orientation matches are unique.
            import math
            a = index * math.pi / 3
            c, s = round(math.cos(a), 10), round(math.sin(a), 10)
            result[str(index)] = dict(pose_id="pose_"+str(index),
                matrix=[[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
        return result

    def test_legacy_default_keeps_exact_prefix_and_complete_keeps_all(self):
        catalog = self.catalog()
        self.assertEqual(select_pose_catalog(catalog, 3),
                         [catalog[str(i)] for i in range(3)])
        before = deepcopy(catalog)
        selected = select_pose_catalog(catalog, 64, 'complete_v1')
        self.assertEqual(selected, list(catalog.values()))
        selected[0]['matrix'][0][0] = 9
        self.assertEqual(catalog, before)

    def test_complete_and_evaluation_reject_overflow_never_silently_cut(self):
        with self.assertRaisesRegex(ValueError, 'complete_pose_catalog_overflow'):
            select_pose_catalog(self.catalog(), 3, 'complete_v1')
        for mode in ('complete_v1', 'legacy_truncate_v1'):
            with self.assertRaisesRegex(ValueError, 'evaluation_pose_catalog_overflow'):
                select_pose_catalog(self.catalog(), 3, mode, evaluation=True)
        with self.assertRaisesRegex(ValueError, 'empty'):
            select_pose_catalog({}, 64, 'complete_v1')

    def test_six_poses_reach_both_bodies_without_recommendations(self):
        r = request()
        r['pose_catalog'] = select_pose_catalog(self.catalog(), 64, 'complete_v1')
        r = validate_request(r)
        cfg = policy()
        with tempfile.TemporaryDirectory() as folder:
            sol = SolPlanService(cfg['roles']['planner'], Path(folder)/'sol.sqlite', Mock())
            body = sol._body(dict(request=r, skills=[], escalation={}, max_steps=8))
        packet = json.loads(body['input'][0]['content'])
        self.assertEqual(restore_request_v5(packet['observation'])['pose_catalog'], r['pose_catalog'])
        luna = build_provider_body(cfg['roles']['scout'], r, 'explicit_action_provider_v5')
        packet_luna = json.loads(luna['input'][0]['content'])
        self.assertEqual(restore_request_v5({k:packet_luna[k] for k in
            ('projection_version','request','shared_source_contexts')})['pose_catalog'], r['pose_catalog'])
        self.assertEqual(len(decision_brief_v5(r)['pose_gravity']), 6)
        for forbidden in ('native_order', 'heuristic_score', 'recommended_pose', 'winning_pose'):
            self.assertNotIn(forbidden, json.dumps([packet, packet_luna]))

    def test_plan_can_use_previously_omitted_pose_and_current_ground_translation(self):
        r = request()
        r['pose_catalog'] = select_pose_catalog(self.catalog(), 64, 'complete_v1')
        value = plan_value(r)
        for step in value['steps']:
            step['pose_reference'] = 'pose_4'
        plan = validate_plan(value, r, [])
        first = action_for_step(plan, 0, r)
        self.assertEqual(first['choice']['pose_id'], 'pose_4')
        # Injected feedback tests execution guards only, not actual physics.
        next_request = feedback(r, first)
        next_request['pose_catalog'][4]['matrix'][2][3] = 0.7
        next_request['pose_catalog'][4]['pose_id'] = 'new_pose_4'
        self.assertEqual(action_for_step(plan, 1, next_request)['choice']['pose_id'], 'new_pose_4')

    def test_zero_learner_allocation_cannot_send_even_with_a_backend(self):
        cfg = policy()
        cfg['roles']['learner'].update(max_calls=0, max_usd='0')
        cfg['max_calls'] = 8
        cfg['max_usd'] = '2'
        validate_policy(cfg)
        backend = Mock()
        body = dict(model=cfg['roles']['learner']['model'], store=False,
                    reasoning={'effort':'none'}, max_output_tokens=16)
        with tempfile.TemporaryDirectory() as folder:
            budget = HierarchyBudget(Path(folder)/'budget.sqlite', cfg)
            with self.assertRaisesRegex(HierarchyBudgetError, 'call_or_cost_limit'):
                budget.transport('learner', backend)(
                    'https://api.openai.com/v1/responses', {}, body, 1)
            self.assertEqual(budget.summary()['transport_attempts'], 0)
        backend.assert_not_called()
        for role in ('scout', 'planner'):
            bad = deepcopy(cfg)
            bad['roles'][role].update(max_calls=0, max_usd='0')
            with self.assertRaises(ValueError):
                validate_policy(bad)
        bad = deepcopy(cfg)
        bad['roles']['learner']['max_usd'] = '0.01'
        with self.assertRaises(ValueError):
            validate_policy(bad)
