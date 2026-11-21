"""Saved real observations + real pinned VM, injected models, no new physics/API."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from tests.test_explicit_action_service import provider
from tests.test_agent_knowledge import read as read_operation
from worker.explicit_action_contract import action_id, digest

ROOT = Path(__file__).absolute().parents[1]
LIVE = ROOT / 'runs/support_free_reuse_20261109_1250/live'
LEARNING = ROOT / 'runs/support_learning_live_20261109_1155/revalidation_v2'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


class RecoveryReplanHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decisions = load(LIVE / 'episode/hierarchy_decisions.json')
        cls.snapshot = load(LEARNING / 'skills_frozen.json')
        cls.lessons = load(LEARNING / 'lessons_frozen.json')
        cls.live_config = load(LIVE / 'episode/config.json')
        cls.sid = cls.snapshot['entries'][0]['candidate']['program_hash']

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='handoff_OFFLINE_')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        env = patch.dict(os.environ, OPENAI_API_KEY='OFFLINE_NOT_A_REAL_KEY')
        env.start(); self.addCleanup(env.stop)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.network = network.start(); self.addCleanup(network.stop)
        self.addCleanup(self.network.assert_not_called)
        self.packets = []

    def req(self, index):
        return deepcopy(self.decisions[index]['request'])

    def service(self, *, select=True, failed_read=True, query_timeout=False, partial_first=False):
        first_id = self.req(1)['request_id']

        def send(url, headers, body, timeout):
            packet = json.loads(body['input'][0]['content'])
            self.packets.append(deepcopy(packet))
            if body['text']['format']['name'] == 'knowledge_tool_requests':
                # Retrieval uses derived request IDs; bind using the state packet.
                first = not any('registered_skills' in p for p in self.packets)
                if query_timeout and not first:
                    raise TimeoutError('injected_no_network')
                op = read_operation('skill:' + self.sid if first or not failed_read else 'skill:missing')
                op['kind'] = 'skill'
                value = dict(request_id=packet['request_id'], state_id=packet['state_id'],
                    finish=True, reason='Injected read request', operations=[op] +
                    ([read_operation('lesson:missing')] if partial_first and first else []))
            else:
                from orchestrator.explicit_action_provider_v5 import restore_request_v5
                request = restore_request_v5(packet['observation'])
                first = request['request_id'] == first_id
                choice = deepcopy(self.decisions[1 if first else 2]['response']['choice'])
                if not first and select:
                    step = dict(part_id='1', mode='registered_skill', pose_reference=None,
                        primitive=None, direction=None, second_direction=None, support_ids=[],
                        evidence_ids=[], registered_skill_id=self.sid, pose_policy=None)
                else:
                    step = dict(part_id=choice['part_id'], mode='action', pose_reference=choice['pose_id'],
                        primitive=choice['skill_id'], direction=choice['direction_id'],
                        second_direction=choice['second_direction_id'], support_ids=choice['support_ids'],
                        evidence_ids=choice['evidence_ids'], registered_skill_id=None,
                        pose_policy='exact_orientation_v1')
                value = dict(request_id=request['request_id'], state_id=request['state_id'],
                    candidate_order=request['candidate_ids'], steps=[step],
                    recovery_skill_ids=[self.sid] if first else [],
                    skill_review=[dict(skill_id=c['skill_id'], disposition=(
                        'unavailable' if c['currently_proposed_choice'] is None else
                        'execute_now' if not first and select else 'defer'),
                        reason='Injected test decision, not real model judgement')
                        for c in packet['registered_skills']])
            raw = provider(value); raw['model'] = body['model']
            return raw

        cfg = deepcopy(self.live_config['extension']['hierarchy_policy']['roles']['planner'])
        cfg.update(max_calls=8, max_usd='10')
        owner = SolPlanService(cfg, self.path / 'planner.sqlite', send,
            provider_contract='conditional_plan_v4', planning_feedback_policy='repair_evidence_v2')
        limits = dict(scout_calls=0, planner_calls=2, fallback_calls=0, max_plan_steps=8, recovery_per_step=1)
        return HierarchicalDecisionService(Mock(), owner, self.snapshot, limits,
            recovery_support_policy='preserve_existing_v1', planning_feedback_policy='repair_evidence_v2',
            lesson_library=self.lessons, lesson_runtime=True, max_skill_attempts=3,
            knowledge_policy='agent_tools_v1', knowledge_max_rounds=1)

    def start_failure(self, service):
        request = self.req(1)
        first = service.handle(request)
        self.assertEqual(first['metadata']['hierarchy_role'], 'plan_action', first)
        self.assertEqual(first['action_id'], self.decisions[1]['response']['action_id'])
        return service.handle(self.req(2))

    def test_actual_vm_guard_to_fresh_plan_skill_even_when_new_read_fails(self):
        service = self.service()
        before = deepcopy((self.snapshot, self.lessons, self.decisions))
        response = self.start_failure(service)
        self.assertEqual(response['metadata']['hierarchy_role'], 'plan_skill', response)
        self.assertEqual(response['choice']['support_ids'], ['2', '3'])
        self.assertEqual(response['metadata']['skill_execution_role'], 'explicit_plan_skill')
        handoff = response['metadata']['recovery_replan_handoff']
        self.assertFalse(handoff['physical_attempt_executed'])
        self.assertIsNone(handoff['physical_verdict'])
        self.assertEqual(handoff['proposal_action_id'], action_id(self.req(2), response['choice']))
        self.assertEqual(handoff['request_sha256'], digest(self.req(2)))
        self.assertFalse(handoff['support_guard']['allowed'])
        self.assertEqual(handoff['support_guard']['released_support_ids'], ['0', '5'])
        self.assertFalse(any(d['response']['metadata']['hierarchy_role'] == 'recovery' for d in service.decisions))
        self.assertEqual(service.knowledge_queries[-1]['audit']['retrieval_status'], 'rejected')
        self.assertEqual(service.knowledge_queries[-1]['audit']['selected_skill_ids'], [])
        plan_packet = [p for p in self.packets if 'registered_skills' in p][-1]
        card = plan_packet['registered_skills'][0]
        self.assertEqual(card['knowledge_source'], 'prior_model_authorized_recovery_this_request')
        self.assertNotIn('knowledge', card)
        self.assertTrue(card['handoff_proposal_matches_current_vm'])
        self.assertEqual(plan_packet['escalation']['recovery_replan_handoffs'], [handoff])
        self.assertEqual(service.planner_calls, 2)
        count = len(self.packets)
        self.assertEqual(service.handle(self.req(2)), response)
        self.assertEqual(len(self.packets), count)
        self.assertEqual((self.snapshot, self.lessons, self.decisions), before)

    def test_model_can_decline_handoff_and_choose_other_supports(self):
        service = self.service(select=False)
        response = self.start_failure(service)
        self.assertEqual(response['metadata']['hierarchy_role'], 'plan_action')
        self.assertEqual(response['choice']['support_ids'], ['3', '5'])
        self.assertFalse(response['metadata']['generated_program_executed'])
        self.assertEqual(len(service._current_recovery_handoffs(self.req(2))), 1)

    def test_transport_failure_does_not_bypass_paid_request_latch(self):
        service = self.service(query_timeout=True)
        response = self.start_failure(service)
        self.assertEqual(response['metadata']['hierarchy_role'], 'deterministic_fallback')
        self.assertEqual(service.planner_calls, 1)
        self.assertEqual(len([p for p in self.packets if 'registered_skills' in p]), 1)
        self.assertEqual(service.knowledge_queries[-1]['audit']['stop_reason'], 'api_timeout')

    def test_handoff_cannot_cross_request_state_library_or_failure_identity(self):
        service = self.service()
        self.start_failure(service)
        original = deepcopy(service.recovery_handoffs)
        self.assertEqual(len(service._current_recovery_handoffs(self.req(2))), 1)
        for key, value in [('request_sha256', 'x'), ('state_id', 'x'), ('program_hash', 'x'),
                           ('library_sha256', 'x'), ('source_evidence_id', 'x'), ('proposal_action_id', 'x')]:
            service.recovery_handoffs = deepcopy(original)
            service.recovery_handoffs[self.sid][key] = value
            self.assertEqual(service._current_recovery_handoffs(self.req(2)), [], key)
        service.recovery_handoffs = original
        self.assertEqual(service._current_recovery_handoffs(self.req(3)), [])
        service.handle(self.req(3))
        self.assertEqual(service.recovery_handoffs, {})

    def test_changed_vm_output_after_fresh_selection_is_not_executed(self):
        service = self.service()
        self.start_failure(service)
        handoff = service._current_recovery_handoffs(self.req(2))[0]
        changed = deepcopy(handoff['proposed_choice']); changed['support_ids'] = ['2', '4']
        result = dict(triggered=True, response=dict(choice=changed))
        with patch.object(service, '_skill_step', return_value=result):
            response = service._program_response(self.sid, self.req(2), time.monotonic()+20, 'plan_skill', '1')
        self.assertIsNone(response)
        self.assertEqual(service.events[-1]['kind'], 'recovery_handoff_proposal_changed')
        self.assertFalse(service.events[-1]['physical_attempt_executed'])

    def test_unavailable_reference_does_not_create_a_replan_proposal(self):
        service = self.service()
        service.handle(self.req(1))
        service.last_failure_context = None
        result = service._program_response(self.sid, self.req(2), time.monotonic()+20, 'recovery')
        self.assertIsNone(result)
        self.assertEqual(service.recovery_handoffs, {})
        self.assertEqual(service.events[-1]['reason'], 'recovery_support_context_unavailable')

    def test_partial_read_keeps_valid_skill_and_reports_error_to_real_plan_codec(self):
        service = self.service(partial_first=True)
        response = service.handle(self.req(1))
        self.assertEqual(response['metadata']['hierarchy_role'], 'plan_action')
        audit = service.knowledge_queries[-1]['audit']
        self.assertEqual(audit['retrieval_status'], 'partial')
        self.assertEqual(audit['selected_skill_ids'], [self.sid])
        self.assertTrue(audit['knowledge_incomplete'])
        packet = [p for p in self.packets if 'registered_skills' in p][0]
        self.assertEqual(packet['retrieved_knowledge']['audit']['retrieval_status'], 'partial')

    def test_32_read_cards_plus_handoff_stays_bounded_and_reports_omissions(self):
        service = self.service()
        rows = [dict(strategy_id='skill-'+str(i), program_hash='hash-'+str(i),
            lifecycle='registered_scoped', card={'name': 'fixture'},
            program={'language':'python_policy_v1'}, validation_scope=[]) for i in range(33)]
        service.rows = rows
        view = dict(audit=dict(request_sha256=digest(self.req(2))),
            skills=[dict(skill_id=r['strategy_id'], card=r['card']) for r in rows[:32]])
        fake_handoff = dict(skill_id=rows[-1]['strategy_id'], proposed_choice=None)
        service.knowledge = Mock()
        service.knowledge.row.side_effect = lambda identity: next(r for r in rows if r['strategy_id']==identity)
        service.knowledge_max_skills = 32
        with patch.object(service, '_current_recovery_handoffs', return_value=[fake_handoff]), \
             patch.object(service, '_skill_step', return_value={'response':None, 'triggered':False}), \
             patch('orchestrator.knowledge_context.skill_knowledge', return_value={'fixture':True}):
            cards = service._cards(self.req(2), time.monotonic()+20, view=view)
        self.assertEqual(len(cards), 32)
        self.assertIn('skill-32', [c['skill_id'] for c in cards])
        self.assertEqual(service.planning_card_capacity['omitted_from_this_plan_skill_ids'], ['skill-31'])
        self.assertFalse(service.planning_card_capacity['hard_pruning'])

    def test_changed_preview_is_unavailable_before_model_can_select_it(self):
        service = self.service()
        self.start_failure(service)
        old = service._current_recovery_handoffs(self.req(2))[0]['proposed_choice']
        changed = deepcopy(old); changed['support_ids'] = ['2', '4']
        result = dict(triggered=True, response=dict(choice=changed))
        view = deepcopy(service.knowledge_queries[-1])
        with patch.object(service, '_skill_step', return_value=result):
            cards = service._cards(self.req(2), time.monotonic()+20, view=view)
        self.assertIsNone(cards[0]['currently_proposed_choice'])
        self.assertFalse(cards[0]['handoff_proposal_matches_current_vm'])
        self.assertEqual(cards[0]['applicability'], 'recovery_handoff_proposal_changed')


if __name__ == '__main__':
    unittest.main()
