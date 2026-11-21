"""Saved actual requests/program + injected plan replies; no API or new physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_plan import CONTRACT
from orchestrator.strategy_planner import StrategyPlanService
from orchestrator.strategy_service import StrategyDecisionService
from orchestrator.skill_runtime_feedback import TOTAL_FEEDBACK_BYTES, size
from tests.test_explicit_action_service import provider
from tests.test_strategy_service import wire

ROOT = Path(__file__).absolute().parents[1]
EPISODE = ROOT/'runs/support_free_reuse_20261109_1250/live/episode'
TRAIN = ROOT/'runs/support_learning_live_20261109_1155/revalidation_v2'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


class ResearchKnowledgeFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='research_knowledge_OFFLINE_')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.requests = load(EPISODE/'hierarchy_decisions.json')
        self.snapshot = load(TRAIN/'skills_frozen.json')
        self.lessons = load(TRAIN/'lessons_frozen.json')
        self.sid = self.snapshot['entries'][0]['candidate']['program_hash']
        env = patch.dict(os.environ, OPENAI_API_KEY='OFFLINE_NOT_A_REAL_KEY')
        env.start(); self.addCleanup(env.stop)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.net = net.start(); self.addCleanup(net.stop)
        self.addCleanup(self.net.assert_not_called)

    def service(self, transport=None, *, policy='unified_v1', name='planner', feedback=None):
        cfg = load(EPISODE/'config.json')['extension']['hierarchy_policy']['roles']['planner']
        cfg.update(max_calls=1, max_usd='2')
        planner = StrategyPlanService(cfg, self.home/(name+'.sqlite'), transport or Mock())
        return StrategyDecisionService(Mock(provider_contract=CONTRACT), planner, self.snapshot,
            dict(scout_calls=0, planner_calls=1, fallback_calls=0, max_plan_steps=8, recovery_per_step=1),
            recovery_support_policy='preserve_existing_v1', planning_feedback_policy='repair_evidence_v2',
            lesson_library=self.lessons, lesson_runtime=True, knowledge_policy=policy,
            knowledge_runtime_feedback=(policy == 'unified_v1' if feedback is None else feedback))

    def test_saved_initial_abstention_retains_observed_guard_without_lookup(self):
        request = deepcopy(self.requests[0]['request'])
        original = deepcopy(request)
        transport = Mock()
        service = self.service(transport)
        cards = service._cards(request, time.monotonic()+15)
        self.assertIsNone(service.knowledge_agent)
        self.assertEqual(len(cards), 1)
        feedback = cards[0]['runtime_feedback']
        self.assertTrue(feedback['code_observation_only'])
        self.assertFalse(feedback['physical_failure_observed'])
        self.assertEqual(feedback['trace']['kind'], 'runtime_skill_abstention_v1')
        self.assertTrue(feedback['trace']['return_site'])
        self.assertEqual(feedback['trace']['api_calls'], 0)
        self.assertEqual(feedback['trace']['physical_calls'], 0)
        self.assertLessEqual(size(feedback), TOTAL_FEEDBACK_BYTES)
        self.assertEqual(service.knowledge_queries[0]['audit']['policy'], 'unified_v1')
        self.assertEqual(request, original)
        transport.assert_not_called()

    def test_local_and_agent_retrieval_preserve_same_code_observation(self):
        request = deepcopy(self.requests[0]['request'])
        local = self.service(name='local')
        agent = self.service(policy='agent_tools_v1', name='agent')
        # Explicitly injected read view: no retrieval HTTP and no LLM claim.
        view = agent.knowledge.query(request)
        a = local._cards(request, time.monotonic()+15)[0]
        b = agent._cards(request, time.monotonic()+15, view=view)[0]
        self.assertEqual(a['runtime_feedback'], b['runtime_feedback'])
        self.assertEqual(a['runtime_feedback_interpretation'], b['runtime_feedback_interpretation'])
        self.assertEqual(a['currently_proposed_choice'], b['currently_proposed_choice'])

    def run_injected_plan(self, select_skill):
        request = deepcopy(self.requests[2]['request'])
        packets = []
        def transport(url, headers, body, timeout):
            packets.append(deepcopy(body))
            self.assertNotEqual(body['text']['format']['name'], 'knowledge_tool_requests')
            value = wire(request, parts=['1'],
                procedure='registered_skill' if select_skill else 'axis_search_v1',
                sid=self.sid if select_skill else None, supports='replace_once_v1')
            reply = provider(value); reply['model'] = body['model']
            return reply
        service = self.service(transport)
        response = service.handle(request)
        self.assertEqual(len(packets), 1)
        packet = json.loads(packets[0]['input'][0]['content'])
        self.assertEqual(len(packet['registered_skills']), 1)
        self.assertEqual(packet['registered_skills'][0]['skill_id'], self.sid)
        self.assertIsNotNone(packet['registered_skills'][0]['currently_proposed_choice'])
        self.assertIsNone(service.knowledge_agent)
        self.assertEqual(service.handle(deepcopy(request)), response)
        self.assertEqual(len(packets), 1)
        return response

    def test_injected_llm_can_select_frozen_program_with_one_plan_call(self):
        response = self.run_injected_plan(True)
        self.assertIsNotNone(response['choice'], response)
        self.assertEqual(response['choice']['part_id'], '1')
        self.assertEqual(set(response['choice']['support_ids']), {'2', '3'})
        self.assertEqual(response['metadata']['generated_skill_id'], self.sid)
        self.assertTrue(response['metadata']['generated_program_executed'])
        self.assertFalse(response['metadata']['plan_actual_llm'])

    def test_injected_llm_can_decline_applicable_program_without_forced_selection(self):
        response = self.run_injected_plan(False)
        self.assertIsNotNone(response['choice'], response)
        self.assertFalse(response['metadata'].get('generated_program_executed', False))
        self.assertIsNone(response['metadata'].get('generated_skill_id'))

    def test_legacy_keeps_old_diagnostic_contract(self):
        service = self.service(policy='legacy_v1')
        cards = service._cards(deepcopy(self.requests[0]['request']), time.monotonic()+15)
        self.assertNotIn('runtime_feedback', cards[0])
        self.assertEqual(service.knowledge_queries, [])

    def test_existing_unified_profile_keeps_old_contract_without_opt_in(self):
        service = self.service(feedback=False)
        cards = service._cards(deepcopy(self.requests[0]['request']), time.monotonic()+15)
        self.assertNotIn('runtime_feedback', cards[0])
        self.assertFalse(service.observe_skill_abstention)


if __name__ == '__main__':
    unittest.main()
