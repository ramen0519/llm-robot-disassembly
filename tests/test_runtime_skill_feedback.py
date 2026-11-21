"""Local VM + injected providers only; no live LLM or physical verification."""
from copy import deepcopy
import json
import subprocess
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.python_skill_program import step_program, evaluate, build_context
from orchestrator.skill_runtime_feedback import FeedbackBudget, size, MAX_FEEDBACK_BYTES, TOTAL_FEEDBACK_BYTES
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.explicit_action_service import ExplicitActionService
from orchestrator.skill_discovery import SkillDiscoveryService
from orchestrator.generated_skill_service import GeneratedSkillService
from orchestrator.explicit_action_provider_v5 import restore_request_v5, encode_provider_decision_v5
from orchestrator.skill_program import program_hash
from tests.test_skill_program import request, append_observation
from tests.test_hierarchical_planner import policy
from tests import test_agent_knowledge as agent_fixtures
from tests.test_agent_knowledge import read
from tests.test_generated_skill_service import row, snapshot
from tests.test_explicit_action_service import provider
from tools.python_skill_fixture import PROGRAM
from worker.explicit_action_contract import digest, validate_request, validate_response
from worker.path_probe_round import canonical_path_round_response


def program(source):
    return dict(language='python_policy_v1', source=source)


class RuntimeFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.req = append_observation(request())
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('HTTP_FORBIDDEN'))
        self.net = net.start(); self.addCleanup(net.stop); self.addCleanup(self.net.assert_not_called)

    def run_pair(self, source):
        p = program(source); before = deepcopy((p, self.req))
        plain = step_program(p, self.req)
        original = subprocess.run
        with patch('subprocess.run', wraps=original) as child:
            traced = step_program(p, self.req, observe_abstention=True)
        self.assertEqual(child.call_count, 1)  # No second diagnostic execution.
        self.assertEqual((p, self.req), before)
        self.assertEqual({k:v for k,v in traced.items() if k != 'runtime_abstention'}, plain)
        return traced

    def test_actual_guard_operands_and_exact_input_binding(self):
        got = self.run_pair('def propose(ctx):\n    required = 4\n    capacity = ctx["max_grippers"] - 1\n'
            '    if required > capacity:\n        return None\n    return None\n')
        trace = got['runtime_abstention']; compare = trace['return_site']['conditions'][0]['comparisons'][0]
        self.assertEqual([o['value'] for o in compare['operands']], [4, 1])
        self.assertTrue(compare['result']); self.assertFalse(trace['physical_failure_observed'])
        self.assertEqual(trace['request_sha256'], digest(validate_request(self.req)))
        self.assertEqual(trace['context_sha256'], digest(build_context(self.req)[0]))

    def test_missing_measurement_stays_none_not_invented_collision(self):
        got = self.run_pair('def propose(ctx):\n    endpoint = ctx["last"]["feedback"].get("measured_endpoint")\n'
            '    if endpoint is None:\n        return None\n    return None\n')
        operands = got['runtime_abstention']['return_site']['conditions'][0]['comparisons'][0]['operands']
        self.assertIsNone(operands[0]['value'])
        self.assertNotIn('blocker', got['runtime_abstention'])
        self.assertEqual(got['physical_calls'], 0)

    def test_short_circuit_does_not_read_or_invent_operand(self):
        got = self.run_pair('def propose(ctx):\n    if 0 > 1 > ctx["MUST_NOT_READ"]:\n        return {}\n    return None\n')
        comparison = got['runtime_abstention']['return_site']['recent_conditions'][0]['comparisons'][0]
        self.assertFalse(comparison['operands'][-1]['evaluated'])
        self.assertNotIn('value', comparison['operands'][-1])

    def test_observation_does_not_repeat_side_effecting_condition(self):
        got = self.run_pair('def mark(xs):\n    xs.append(1)\n    return len(xs)\n'
            'def propose(ctx):\n    xs = []\n    if mark(xs) > 0 and len(xs) == 1:\n        return None\n    return {}\n')
        comparisons = got['runtime_abstention']['return_site']['conditions'][0]['comparisons']
        self.assertEqual([c['result'] for c in comparisons], [True, True])
        self.assertEqual(comparisons[-1]['operands'][0]['value'], 1)

    def test_error_and_timeout_are_not_explained_as_normal_abstention(self):
        error = self.run_pair('def propose(ctx):\n    x = 1 / 0\n')
        self.assertEqual(error['not_applicable_reason'], 'python_execution_rejected')
        self.assertNotIn('runtime_abstention', error)
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('fixture', 1)) as child:
            result = step_program(program('def propose(ctx):\n    return None\n'), self.req, observe_abstention=True)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(result['not_applicable_reason'], 'python_wall_timeout')
        self.assertNotIn('runtime_abstention', result)

    def test_context_mutation_does_not_change_provenance_hash(self):
        got = self.run_pair('def propose(ctx):\n    parts = ctx["candidate_ids"]\n    parts.append("local_only")\n'
            '    if len(parts) == 3:\n        return None\n    return None\n')
        trace = got['runtime_abstention']
        self.assertEqual(trace['context_sha256'], digest(build_context(self.req)[0]))
        self.assertEqual(trace['return_site']['conditions'][0]['comparisons'][0]['operands'][0]['value'], 3)

    def test_proposals_remain_identical_without_abstention_trace(self):
        plain = step_program(PROGRAM, self.req)
        traced = step_program(PROGRAM, self.req, observe_abstention=True)
        self.assertIsNotNone(traced['response'])
        self.assertEqual(plain, traced)

    def test_unconditional_and_implicit_returns_do_not_invent_conditions(self):
        for source, implicit in [('def propose(ctx):\n    return None\n', False),
                                  ('def propose(ctx):\n    x = 1\n', True)]:
            trace = self.run_pair(source)['runtime_abstention']
            self.assertEqual(trace['implicit_return'], implicit)
            if trace['return_site']:
                self.assertEqual(trace['return_site']['conditions'], [])

    def test_large_trace_and_many_skills_have_explicit_bounded_omissions(self):
        expr = ' and '.join(['xs == xs']*4)
        got = self.run_pair('def propose(ctx):\n    xs = list(range(1000))\n    if '+expr+':\n        return None\n')
        self.assertLessEqual(size(got['runtime_abstention']), MAX_FEEDBACK_BYTES)
        self.assertTrue(got['runtime_abstention']['truncated'])
        budget = FeedbackBudget(count=32)
        rows = [budget.take(got) for _ in range(32)]
        self.assertLessEqual(sum(map(size, rows)), TOTAL_FEEDBACK_BYTES)
        self.assertTrue(any(x.get('trace_unavailable') == 'decision_diagnostic_byte_limit' for x in rows))

    def test_changed_child_binding_is_rejected(self):
        p = program('def propose(ctx):\n    return None\n')
        value = evaluate(p, self.req, observe_abstention=True)
        value['runtime_abstention']['request_sha256'] = 'wrong'
        with patch('subprocess.run', return_value=Mock(returncode=0, stdout=json.dumps(value).encode())):
            got = step_program(p, self.req, observe_abstention=True)
        self.assertEqual(got['not_applicable_reason'], 'python_execution_rejected')
        self.assertNotIn('runtime_abstention', got)


class RuntimeProviderTests(unittest.TestCase):
    def setUp(self):
        self.helper = agent_fixtures.AgentKnowledgeTests()
        self.helper.setUp(); self.addCleanup(self.helper.doCleanups)

    def test_sol_final_provider_sees_actual_abstention_and_may_choose_direct_action(self):
        h = self.helper; limits = policy()['episode']; limits['scout_calls'] = 0
        service = HierarchicalDecisionService(Mock(), h.owner(), h.skills, limits,
            allow_fixture=True, lesson_library=h.lessons, knowledge_policy='agent_tools_v1')
        req = request(); before = deepcopy(req)
        response = service.handle(req)
        packet = json.loads(h.sent[-1]['input'][0]['content'])
        details = packet['registered_skills'][0]['runtime_feedback']
        self.assertEqual(details['reason'], 'no_trigger')
        self.assertIsNone(details['trace']['return_site']['conditions'][0]['comparisons'][0]['operands'][0]['value'])
        self.assertEqual(response['metadata']['hierarchy_role'], 'plan_action')
        self.assertEqual(req, before); self.assertEqual(len(h.sent), 3)
        validate_response(req, response)

    def test_legacy_and_unified_paths_keep_old_card_contract(self):
        h = self.helper
        for name in ('legacy_v1', 'unified_v1'):
            service = HierarchicalDecisionService(Mock(), Mock(), h.skills, policy()['episode'],
                allow_fixture=True, lesson_library=h.lessons, knowledge_policy=name)
            cards = service._cards(request(), time.monotonic()+30)
            self.assertNotIn('runtime_feedback', cards[0])

    def test_luna_ordinary_actor_receives_reasons_when_all_read_skills_abstain(self):
        h = self.helper; cfg = policy()['roles']['scout']; sent = []
        r = row(); r['program'] = deepcopy(PROGRAM)
        r['strategy_id'] = r['program_hash'] = program_hash(PROGRAM)
        def transport(url, headers, body, timeout):
            sent.append(deepcopy(body)); packet = json.loads(body['input'][0]['content'])
            if body['text']['format']['name'] == 'knowledge_tool_requests':
                value = dict(request_id=packet['request_id'], state_id=packet['state_id'], finish=False,
                    reason='OFFLINE scripted read', operations=[read('skill:'+r['strategy_id'])])
            else:
                req = restore_request_v5(packet['observation'])
                value = encode_provider_decision_v5(req, canonical_path_round_response(req, None))
            result = provider(value); result['model'] = body['model']; return result
        actor = ExplicitActionService(cfg, h.root/'luna-feedback.sqlite', transport,
            provider_contract='explicit_action_provider_v5', fallback_policy='path_stage_v1')
        selector = SkillDiscoveryService(cfg, actor.service.ledger_path, transport)
        with patch('orchestrator.generated_skill_service.retrieve', return_value=[deepcopy(r)]), \
             patch('orchestrator.knowledge_context.retrieve', return_value=[deepcopy(r)]):
            service = GeneratedSkillService(snapshot(), actor, selector,
                knowledge_policy='agent_tools_v1', knowledge_max_rounds=1)
            req = request(); response = service.handle(req)
        self.assertEqual(len(sent), 2)  # Existing lookup + ordinary action; no diagnostic API.
        self.assertFalse(response['metadata']['generated_program_executed'])
        packet = json.loads(sent[-1]['input'][0]['content'])
        audit = packet['retrieved_knowledge']['audit']
        details = audit['skill_runtime_feedback'][0]['runtime_feedback']
        self.assertEqual(details['trace']['request_sha256'], digest(validate_request(req)))
        self.assertEqual(audit['request_evidence_ids'], req['evidence_ids'])
        self.assertEqual(service.knowledge_queries[-1]['audit'], audit)
        self.assertFalse(details['physical_failure_observed'])
        validate_response(req, response)
