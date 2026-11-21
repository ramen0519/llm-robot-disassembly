"""Actual saved 10-part requests + frozen skills, no API/physics/new success."""
from copy import deepcopy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.hierarchy_budget import estimate_request
from orchestrator.research_profile import EPISODE_SETTINGS
from orchestrator.strategy_plan import CONTRACT
from orchestrator.strategy_planner import StrategyPlanService
from orchestrator.strategy_service import StrategyDecisionService
from worker.explicit_action_contract import canonical, validate_request

ROOT = Path(__file__).absolute().parents[1]
RUN = ROOT/'runs/capacity_10part_prepare_20261109'
LIBRARY = ROOT/'runs/support_learning_live_20261109_1155/revalidation_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class ResearchSavedPayloadTests(unittest.TestCase):
    def test_initial_transition_and_backtrack_inputs_with_real_frozen_cards_fit(self):
        decisions = read(RUN/'episode/hierarchy_decisions.json')
        requests = [decisions[0]['request'], decisions[3]['request'],
                    read(ROOT/'runs/dfs_restore_fix_20261109/final_replay/restored_request.json')]
        limits = dict(scout_calls=0, planner_calls=1, fallback_calls=0,
                      max_plan_steps=8, recovery_per_step=1)
        policy = read(RUN/'disabled_policy.json')
        for ordinal, request in enumerate(requests):
            with self.subTest(saved_state=ordinal), patch('urllib.request.OpenerDirector.open',
                    side_effect=AssertionError('NO_NETWORK')) as network:
                original = deepcopy(request)
                validate_request(request)
                planner = Mock(provider_contract=CONTRACT)
                scout = Mock(provider_contract=CONTRACT)
                service = StrategyDecisionService(scout, planner, read(LIBRARY/'skills_frozen.json'), limits,
                    recovery_support_policy='preserve_existing_v1', planning_feedback_policy='repair_evidence_v2',
                    lesson_runtime=True, lesson_library=read(LIBRARY/'lessons_frozen.json'),
                    knowledge_policy=EPISODE_SETTINGS['knowledge_policy'],
                    knowledge_runtime_feedback=EPISODE_SETTINGS['knowledge_runtime_feedback'],
                    knowledge_max_skills=EPISODE_SETTINGS['knowledge_max_skills'])
                self.assertIsNone(service.knowledge_agent)
                cards = service._cards(request, time.monotonic()+60)
                self.assertTrue(cards, 'Saved registered skill disappeared from local retrieval')
                body_service = object.__new__(StrategyPlanService)
                body_service.config = policy['roles']['planner']
                body_service.provider_contract = CONTRACT
                body_service.role = 'planner'
                body_service.planning_feedback_policy = 'repair_evidence_v2'
                envelope = dict(plan_provider_contract=CONTRACT,
                    planning_feedback_policy='repair_evidence_v2', request=request, skills=cards,
                    escalation=dict(strategy_orientation_reference=service._reference(request)), max_steps=8)
                body = body_service._body(envelope)
                estimated = estimate_request(policy, 'planner', body)
                self.assertLessEqual(estimated['body_utf8_bytes'], 262144)
                self.assertLessEqual(len(canonical(request).encode('utf-8')), 196608)
                packet = json.loads(body['input'][0]['content'])
                self.assertIn('strategy_pose_preflight', packet)
                self.assertNotIn('native_order', packet)
                self.assertEqual(request, original)
                planner.plan.assert_not_called(); scout.plan.assert_not_called()
                network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
