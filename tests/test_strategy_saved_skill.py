"""Saved actual inputs and frozen Python, simulated provider responses, NO physics/API."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_plan import CONTRACT
from orchestrator.strategy_planner import StrategyPlanService
from orchestrator.strategy_service import StrategyDecisionService
from tests.test_explicit_action_service import provider
from tests.test_agent_knowledge import read as read_operation
from tests.test_strategy_service import wire

ROOT = Path(__file__).absolute().parents[1]
LIVE = ROOT/'runs/support_free_reuse_20261109_1250/live/episode'
TRAIN = ROOT/'runs/support_learning_live_20261109_1155/revalidation_v2'


def load(path): return json.loads(path.read_text(encoding='utf-8'))


class StrategySavedSkillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='strategy_saved_OFFLINE_')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.env = patch.dict(os.environ, OPENAI_API_KEY='OFFLINE_NOT_A_REAL_KEY')
        self.env.start(); self.addCleanup(self.env.stop)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.network = net.start(); self.addCleanup(net.stop)
        self.addCleanup(self.network.assert_not_called)
        self.requests = load(LIVE/'hierarchy_decisions.json')
        self.snapshot = load(TRAIN/'skills_frozen.json')
        self.lessons = load(TRAIN/'lessons_frozen.json')
        self.sid = self.snapshot['entries'][0]['candidate']['program_hash']

    def run_case(self, support_policy):
        packets = []
        request = deepcopy(self.requests[2]['request'])
        def transport(url, headers, body, timeout):
            packet = json.loads(body['input'][0]['content']); packets.append(packet)
            if body['text']['format']['name'] == 'knowledge_tool_requests':
                operation = read_operation('skill:'+self.sid); operation['kind'] = 'skill'
                value = dict(request_id=packet['request_id'], state_id=packet['state_id'],
                    operations=[operation], finish=True, reason='Injected offline skill read')
            else:
                value = wire(request, parts=['1'], procedure='registered_skill', sid=self.sid,
                    supports=support_policy)
            result = provider(value); result['model'] = body['model']
            return result
        cfg = load(LIVE/'config.json')['extension']['hierarchy_policy']['roles']['planner']
        cfg.update(max_calls=2, max_usd='2')
        planner = StrategyPlanService(cfg, self.home/'planner.sqlite', transport)
        scout = Mock(provider_contract=CONTRACT)
        service = StrategyDecisionService(scout, planner, self.snapshot,
            dict(scout_calls=0, planner_calls=1, fallback_calls=0, max_plan_steps=8, recovery_per_step=1),
            recovery_support_policy='preserve_existing_v1', planning_feedback_policy='repair_evidence_v2',
            lesson_library=self.lessons, lesson_runtime=True, knowledge_policy='agent_tools_v1',
            knowledge_max_rounds=1)
        response = service.handle(request)
        self.assertEqual(len(packets), 2)
        self.assertEqual(service.planner_calls, 1)
        self.assertEqual(service.knowledge_queries[0]['audit']['retrieval_status'], 'ready')
        self.assertEqual(service.handle(deepcopy(request)), response)
        self.assertEqual(len(packets), 2)
        return response, service

    def test_model_read_and_selected_real_frozen_python_proposes_support_replacement(self):
        response, service = self.run_case('replace_once_v1')
        self.assertIsNotNone(response['choice'], response)
        self.assertEqual(response['choice']['part_id'], '1')
        self.assertEqual(set(response['choice']['support_ids']), {'2', '3'})
        self.assertEqual(response['metadata']['generated_skill_id'], self.sid)
        self.assertTrue(response['metadata']['generated_program_executed'])
        self.assertFalse(response['metadata']['plan_actual_llm'])
    def test_preserve_policy_does_not_release_supports_in_saved_initial_failure(self):
        response, service = self.run_case('preserve_add_v1')
        self.assertIsNone(response['choice'])
        self.assertTrue(any(e.get('reason') == 'strategy_skill_support_release_requires_new_plan'
                            for e in service.events))


if __name__ == '__main__': unittest.main()
