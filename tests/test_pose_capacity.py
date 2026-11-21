"""Pose growth and safe bounds; fixtures here are NOT physical success evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from worker.pose_catalog import PoseCatalogError, select_pose_catalog
from worker.explicit_action_contract import MAX_POSE_CATALOG, validate_request
from worker.explicit_request_budget import bounded_memory_request, capacity_probe, MAX_REQUEST_BYTES
from orchestrator.explicit_action_service import build_provider_body
from orchestrator.explicit_action_provider_v5 import restore_request_v5
from orchestrator.hierarchical_planner import SolPlanService
from tests.test_hierarchical_planner import policy, request
from tests.test_explicit_action_planner import make_planner, response_for


def catalog(count):
    # No probability scores or geometry-based pruning; all entries are distinct.
    result = {}
    for i in range(count):
        angle = i * 2 * math.pi / 128
        c, s = round(math.cos(angle), 10), round(math.sin(angle), 10)
        result['%03d' % i] = dict(pose_id='p'+str(i),
            matrix=[[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
    return result


class PoseCapacityTests(unittest.TestCase):
    def test_growth_44_to_77_keeps_all_and_allows_next_state_selection(self):
        seen = []
        def poses(parts):
            return select_pose_catalog(catalog(44 if len(parts) == 3 else 77), 128, 'complete_v1')
        def decide(r):
            seen.append(deepcopy(r))
            return response_for(r, pose='p0' if len(seen) == 1 else 'p76')
        planner = make_planner(decision=decide, pose_provider=poses)
        planner.plan(budget=2, timeout=None)
        self.assertTrue(planner.report()['success'])  # fixture-only control flow
        self.assertEqual([len(r['pose_catalog']) for r in seen], [44,77])
        self.assertEqual(len(planner.attempts), 2)
        self.assertEqual(planner.attempts[1]['choice']['pose_id'], 'p76')
        self.assertIsNone(planner.report()['planner_error'])
        self.assertEqual(seen[1]['candidate_ids'], ['b','c'])

    def test_old_64_cap_is_not_silently_relaxed_and_error_has_failed_child(self):
        decisions = Mock(side_effect=lambda r: response_for(r))
        planner = make_planner(decision=decisions, pose_provider=lambda parts:
            select_pose_catalog(catalog(44 if len(parts)==3 else 77),64,'complete_v1'))
        planner.plan(budget=2, timeout=None)
        report = planner.report()
        self.assertEqual(report['status'], 'ERROR')
        self.assertEqual(decisions.call_count, 1)
        self.assertEqual(report['total_n_eval'], 1)
        self.assertTrue(report['attempts'][0]['accepted_for_transition'])
        error = report['planner_error']
        self.assertEqual(error['phase'], 'pose_catalog')
        self.assertEqual(error['remaining_parts'], ['b','c'])
        self.assertEqual(error['generated_count'], 77)
        self.assertEqual(error['configured_limit'], 64)
        self.assertEqual(error['contract_limit'], MAX_POSE_CATALOG)
        self.assertEqual(error['code'], 'complete_pose_catalog_overflow_no_truncation')
        self.assertEqual([e for e in planner.events if e['kind']=='planner_error'][0]['code'], error['code'])

    def test_129_candidates_stop_before_decision_or_physics_without_cutting(self):
        decision, physics = Mock(), Mock()
        planner = make_planner(decision=decision, verifier=physics, pose_provider=lambda parts:
            select_pose_catalog(catalog(129),128,'complete_v1'))
        planner.plan(budget=2, timeout=None)
        decision.assert_not_called(); physics.assert_not_called()
        report = planner.report()
        self.assertEqual(report['total_n_eval'], 0)
        self.assertEqual(report['status'], 'ERROR')
        self.assertEqual(report['planner_error']['generated_count'], 129)
        self.assertEqual(report['planner_error']['contract_limit'], 128)
        self.assertFalse(report['hard_pruning'])

    def test_complete_configuration_cannot_exceed_the_protocol_limit(self):
        with self.assertRaisesRegex(PoseCatalogError,'configured_pose_limit_exceeds_contract'):
            select_pose_catalog(catalog(77),129,'complete_v1')
        with self.assertRaisesRegex(PoseCatalogError,'configured_pose_limit_exceeds_contract'):
            select_pose_catalog(catalog(77),129,evaluation=True)

    def test_full_128_contract_boundary_and_legacy_prefix_remain_valid(self):
        original = catalog(128)
        r = request(); r['pose_catalog'] = select_pose_catalog(original,128,'complete_v1')
        self.assertEqual(len(validate_request(r)['pose_catalog']),128)
        self.assertEqual(select_pose_catalog(original,3),[original[k] for k in sorted(original)[:3]])
        with self.assertRaisesRegex(PoseCatalogError,'evaluation_pose_catalog_overflow'):
            select_pose_catalog(original,64,evaluation=True)

    def test_77_poses_reach_luna_and_sol_without_hidden_selection(self):
        r = request(); r['pose_catalog'] = select_pose_catalog(catalog(77),128,'complete_v1')
        r, audit = bounded_memory_request(r, [])
        self.assertEqual(audit['pose_count'],77)
        cfg = policy()
        luna = build_provider_body(cfg['roles']['scout'],r,'explicit_action_provider_v5')
        lp = json.loads(luna['input'][0]['content'])
        self.assertEqual(restore_request_v5({k:lp[k] for k in
            ('projection_version','request','shared_source_contexts')})['pose_catalog'],r['pose_catalog'])
        with tempfile.TemporaryDirectory() as tmp:
            sol = SolPlanService(cfg['roles']['planner'],Path(tmp)/'sol.sqlite',Mock(),
                provider_contract='conditional_plan_v4',planning_feedback_policy='repair_evidence_v2')
            body = sol._body(dict(request=r,skills=[],escalation={},max_steps=8,
                plan_provider_contract='conditional_plan_v4',
                planning_feedback_policy='repair_evidence_v2'))
        sp = json.loads(body['input'][0]['content'])
        self.assertEqual(restore_request_v5(sp['observation'])['pose_catalog'],r['pose_catalog'])
        for word in ('native_order','heuristic_score','recommended_pose'):
            self.assertNotIn(word,json.dumps([lp,sp]))

    def test_77_pose_core_keeps_history_and_capacity_bound(self):
        r = request(); r['pose_catalog'] = select_pose_catalog(catalog(77),128,'complete_v1')
        stress, audit = capacity_probe(r, attempted_count=60,episode_count=12,memory_count=4)
        self.assertEqual(stress['pose_catalog'],r['pose_catalog'])
        self.assertEqual(stress['candidate_ids'],r['candidate_ids'])
        self.assertEqual(len(stress['attempted_actions']),60)
        self.assertLessEqual(audit['request_bytes'],MAX_REQUEST_BYTES)

    def test_oversize_core_records_bytes_and_never_calls_decider(self):
        decision, physics = Mock(), Mock()
        def enrich(r):
            r['geometry']['padding'] = 'x'*MAX_REQUEST_BYTES
            return bounded_memory_request(r,[])[0]
        planner = make_planner(decision=decision,verifier=physics,request_enricher=enrich)
        planner.plan(budget=2,timeout=None)
        decision.assert_not_called(); physics.assert_not_called()
        error = planner.report()['planner_error']
        self.assertEqual(error['phase'],'request_construction')
        self.assertEqual(error['code'],'request_core_too_large')
        self.assertGreater(error['core_bytes'],error['limit_bytes'])

    def test_unrecognized_error_message_and_arbitrary_audit_are_not_logged(self):
        def fail(parts):
            error = ValueError('untrusted sensitive payload')
            error.audit = dict(secret='untrusted sensitive payload',generated_count=999)
            raise error
        planner=make_planner(pose_provider=fail)
        planner.plan(budget=1,timeout=None)
        self.assertNotIn('untrusted sensitive payload',json.dumps(planner.report()))
        self.assertEqual(planner.report()['planner_error']['code'],'unclassified_planner_error')
        self.assertNotIn('generated_count',planner.report()['planner_error'])
