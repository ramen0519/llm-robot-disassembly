"""Host pose legality: synthetic boundary tests and immutable saved-input replay.

No network, model quality, new physics, WSL or full-product success is asserted.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_candidates import pose_policy_context, resolve_reference
from orchestrator.strategy_plan import CONTRACT, STEP_KEYS, compile_plan, schema
from orchestrator.strategy_planner import StrategyPlanService
from tests.test_explicit_action_service import provider
from tests.test_hierarchical_planner import request, feedback, policy
from tests.test_strategy_plan import value
from tests import test_strategy_service as service_fixtures
from tests.test_strategy_service import wire
from worker.decision_halt import is_halt_response
from worker.explicit_action_contract import canonical_response, validate_request

ROOT = Path(__file__).absolute().parents[1]
SAVED = ROOT/'runs/capacity_10part_prepare_20261109/episode/ipc/requests'
FIRST = SAVED/'86cf6bd8633572700ddc83f2d850ea4f4303c99f480ca81461ff147a25a8f025.json'
TRANSITION = SAVED/'2c3bf46c1b7bd41d74b5e75ed7561e2e54b70ceff47a9b7251429f5601c89c38.json'
RESTORED = ROOT/'runs/dfs_restore_fix_20261109/final_replay/restored_request.json'
TURNED = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def load_saved(path):
    packet = json.loads(path.read_text(encoding='utf-8'))
    return packet['features']['explicit_action'] if 'features' in packet else packet


class PosePreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pose_preflight_')
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        blocked = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        network = blocked.start(); self.addCleanup(blocked.stop)
        self.addCleanup(network.assert_not_called)
        env = patch.dict('os.environ', OPENAI_API_KEY='fixture-only-not-a-real-key')
        env.start(); self.addCleanup(env.stop)

    def provider(self, transport=None):
        return StrategyPlanService(policy()['roles']['planner'], self.folder/'ledger.sqlite',
                                   transport if transport is not None else Mock(), role='planner')

    def envelope(self, req, reference=None, max_steps=8):
        return dict(request=req, skills=[], escalation=dict(strategy_orientation_reference=reference),
                    max_steps=max_steps, timeout_seconds=30, plan_provider_contract=CONTRACT,
                    planning_feedback_policy='repair_evidence_v2')

    def test_initial_body_exposes_legal_permission_without_inventing_pose(self):
        req = request(); before = deepcopy(req)
        context = json.loads(self.provider()._body(self.envelope(req))['input'][0]['content'])
        available = context['strategy_pose_preflight']
        self.assertEqual(available['allowed_first_step_pose_policies'], ['select_catalog_once_v1'])
        self.assertEqual(available['keep_orientation_unavailable_reason'], 'orientation_reference_missing')
        self.assertFalse(available['automatic_pose_fallback'])
        self.assertFalse(available['physics_success_implied'])
        self.assertEqual(req, before)

    def test_single_step_schema_excludes_known_unavailable_keep(self):
        definition = schema(request(), [], 1, orientation_reference=None)
        enum = definition['properties']['steps']['items']['properties']['pose_policy']['enum']
        self.assertEqual(enum, ['select_catalog_once_v1'])

    def test_first_step_rejected_but_later_keep_stays_conditional(self):
        req = request(); candidate = value(req)
        plan = compile_plan(candidate, req, [], orientation_reference=None)
        self.assertEqual(plan['steps'][1]['pose_policy'], 'keep_orientation_v1')
        candidate['steps'][0]['pose_policy'] = 'keep_orientation_v1'
        diagnostics = {}
        with self.assertRaisesRegex(ValueError, 'first_step_pose_policy_unavailable'):
            compile_plan(candidate, req, [], diagnostics=diagnostics, orientation_reference=None)
        self.assertEqual(diagnostics['step_index'], 0)
        self.assertFalse(diagnostics['pose_preflight']['keep_orientation_available'])

    def test_provider_refuses_illegal_return_without_rewriting_or_retry(self):
        req = request(); candidate = value(req)
        candidate['steps'][0]['pose_policy'] = 'keep_orientation_v1'
        raw = provider(candidate); raw['model'] = 'gpt-6.1-sol'
        transport = Mock(return_value=raw)
        svc = self.provider(transport)
        result = svc.plan(req, [], {})
        self.assertEqual(result['metadata']['plan_contract_error'], 'strategy_first_step_pose_policy_unavailable')
        self.assertIsNone(result['plan'])
        self.assertEqual(svc.plan(req, [], {})['status'], 'rejected')
        self.assertEqual(transport.call_count, 1)

    def test_known_current_command_allows_keep_and_preserves_inputs(self):
        req = request()
        req = feedback(req, canonical_response(req, None), False)
        original = deepcopy(req)
        available = pose_policy_context(req)
        self.assertTrue(available['keep_orientation_available'])
        candidate = value(req); candidate['steps'][0]['pose_policy'] = 'keep_orientation_v1'
        compiled = compile_plan(candidate, req, [], orientation_reference=resolve_reference(req))
        self.assertEqual(compiled['steps'][0]['pose_policy'], 'keep_orientation_v1')
        self.assertEqual(req, original)

    def test_missing_current_command_halts_before_scout_or_paid_search(self):
        req = request(); req = feedback(req, canonical_response(req, None), False)
        req['episode_feedback'] = []; req['evidence_ids'] = []
        req = validate_request(req)
        svc, planner, scout = service_fixtures.StrategyServiceTests().service(scout_calls=1)
        knowledge = Mock(); svc.knowledge_agent = knowledge
        result = svc.handle(req)
        self.assertTrue(is_halt_response(result))
        self.assertEqual(result['fallback_reason'], 'strategy_reference_preflight_unavailable')
        planner.plan.assert_not_called(); scout.plan.assert_not_called()
        knowledge.retrieve.assert_not_called()

    def test_unbound_support_reference_rejected_before_provider_reservation(self):
        req = request(); req = feedback(req, canonical_response(req, None), False)
        req['episode_feedback'] = []; req['evidence_ids'] = []
        transport = Mock()
        result = self.provider(transport).plan(req, [], {})
        self.assertEqual(result['fallback_reason'], 'strategy_reference_preflight_unavailable')
        self.assertFalse(result['metadata']['attempted'])
        transport.assert_not_called()

    def test_transition_missing_rotation_requires_new_explicit_plan(self):
        calls = []
        def reply(req, cards, escalation):
            calls.append(deepcopy(escalation))
            result = wire(req)
            if len(calls) == 1:
                result['steps'][1]['pose_policy'] = 'keep_orientation_v1'
            return result
        svc, planner, _ = service_fixtures.StrategyServiceTests().service(planner_calls=2, reply=reply)
        req = request(); first = svc.handle(req)
        next_req = feedback(req, first, True)
        next_req['pose_catalog'][0]['matrix'] = deepcopy(TURNED)
        original = deepcopy(next_req)
        second = svc.handle(next_req)
        self.assertEqual(planner.plan.call_count, 2)
        self.assertFalse(is_halt_response(second))
        self.assertEqual(calls[-1]['strategy_pose_preflight']['keep_orientation_unavailable_reason'],
                         'orientation_not_in_current_catalog')
        self.assertEqual(calls[-1]['strategy_pose_preflight']['allowed_first_step_pose_policies'],
                         ['select_catalog_once_v1'])
        self.assertTrue(any(e.get('reason') == 'orientation_unavailable_or_ambiguous' for e in svc.events))
        self.assertEqual(next_req, original)
        self.assertFalse(second['metadata']['plan_execution_without_api'])

    def test_transition_missing_rotation_without_new_plan_halts(self):
        def reply(req, cards, escalation):
            result = wire(req); result['steps'][1]['pose_policy'] = 'keep_orientation_v1'
            return result
        svc, planner, _ = service_fixtures.StrategyServiceTests().service(planner_calls=1, reply=reply)
        req = request(); first = svc.handle(req)
        next_req = feedback(req, first, True)
        next_req['pose_catalog'][0]['matrix'] = deepcopy(TURNED)
        result = svc.handle(next_req)
        self.assertTrue(is_halt_response(result))
        self.assertEqual(planner.plan.call_count, 1)
        self.assertEqual(len([e for e in svc.events if e['kind'] == 'strategy_proposal_issued']), 1)

    def test_duplicate_rotation_is_not_unique_keep_anchor(self):
        req = request(); req = feedback(req, canonical_response(req, None), True)
        ref = req['episode_feedback'][-1]
        req['pose_catalog'].append(deepcopy(req['pose_catalog'][0]))
        req['pose_catalog'][-1]['pose_id'] = 'alias'
        req['pose_catalog'][-1]['matrix'][2][3] = 3
        available = pose_policy_context(req, ref)
        self.assertEqual(available['matching_pose_count'], 2)
        self.assertEqual(available['keep_orientation_unavailable_reason'], 'orientation_match_ambiguous')

    def test_registered_skill_missing_anchor_rejected_before_vm(self):
        from tests.test_strategy_plan import card
        selected = card(); req = request()
        svc, _, _ = service_fixtures.StrategyServiceTests().service()
        candidate = wire(req, procedure='registered_skill', sid=selected['skill_id'], pose='keep_orientation_v1')
        svc.plan = compile_plan(candidate, req, [selected])
        svc._skill_step = Mock(side_effect=AssertionError('NO_VM_WITHOUT_POSE_AUTHORITY'))
        import time
        self.assertIsNone(svc._execute(req, time.monotonic() + 30))
        svc._skill_step.assert_not_called()


class SavedPoseReplayTests(unittest.TestCase):
    def test_actual_saved_scout_keep_is_rejected_by_current_compiler(self):
        req = load_saved(FIRST)
        plans = json.loads((SAVED.parents[1]/'hierarchy_plans.json').read_text(encoding='utf-8'))
        original = plans[0]['result']['plan']
        candidate = dict(request_id=req['request_id'], state_id=req['state_id'],
            candidate_order=original['candidate_order'], recovery_skill_ids=[],
            steps=[{k: deepcopy(s[k]) for k in STEP_KEYS} for s in original['steps']])
        self.assertEqual(candidate['steps'][0]['pose_policy'], 'keep_orientation_v1')
        with self.assertRaisesRegex(ValueError, 'first_step_pose_policy_unavailable'):
            compile_plan(candidate, req, [], orientation_reference=None)

    def test_actual_first_four_requests_continue_then_replan_missing_pose(self):
        plans = json.loads((SAVED.parents[1]/'hierarchy_plans.json').read_text(encoding='utf-8'))
        report = json.loads((SAVED.parents[1]/'worker_result.json').read_text(encoding='utf-8'))
        requests = [load_saved(path) for path in SAVED.glob('*.json')]
        requests = sorted(requests, key=lambda r:int(r['request_id'].rsplit(':', 1)[1]))[:4]
        called = []
        def reply(req, cards, escalation):
            called.append(deepcopy(escalation))
            stored = next(row['result']['plan'] for row in plans
                          if row['request_id'] == req['request_id'] and row.get('role') != 'scout')
            return dict(request_id=req['request_id'], state_id=req['state_id'],
                candidate_order=stored['candidate_order'], recovery_skill_ids=[],
                steps=[{k: deepcopy(s[k]) for k in STEP_KEYS} for s in stored['steps']])
        svc, planner, _ = service_fixtures.StrategyServiceTests().service(planner_calls=2, reply=reply)
        with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK')) as network:
            answers = [svc.handle(req) for req in requests]
            network.assert_not_called()
        self.assertEqual(planner.plan.call_count, 2)
        self.assertEqual([row['action_id'] for row in answers],
                         [row['action_id'] for row in report['attempts'][:4]])
        self.assertFalse(any(is_halt_response(answer) for answer in answers))
        self.assertEqual(called[-1]['strategy_pose_preflight']['keep_orientation_unavailable_reason'],
                         'orientation_not_in_current_catalog')
        self.assertEqual(called[-1]['strategy_pose_preflight']['matching_pose_count'], 0)

    def test_actual_00161_initial_input_disallows_keep_without_modification(self):
        original = FIRST.read_bytes(); req = load_saved(FIRST)
        available = pose_policy_context(req)
        self.assertEqual(len(req['remaining_parts']), 10)
        self.assertEqual(available['keep_orientation_unavailable_reason'], 'orientation_reference_missing')
        self.assertEqual(FIRST.read_bytes(), original)

    def test_actual_00161_after_part9_rotation_missing_is_not_ambiguity(self):
        original = TRANSITION.read_bytes(); req = load_saved(TRANSITION)
        ref = next(row for row in req['episode_feedback'] if row['accepted_for_transition'])
        self.assertEqual(ref['choice']['part_id'], '9')
        available = pose_policy_context(req, ref)
        self.assertEqual(len(req['remaining_parts']), 9)
        self.assertEqual(len(req['pose_catalog']), 3)
        self.assertEqual(available['matching_pose_count'], 0)
        self.assertEqual(available['keep_orientation_unavailable_reason'], 'orientation_not_in_current_catalog')
        candidate = wire(req, pose='keep_orientation_v1')
        with self.assertRaisesRegex(ValueError, 'first_step_pose_policy_unavailable'):
            compile_plan(candidate, req, [], orientation_reference=ref)
        candidate['steps'][0]['pose_policy'] = 'select_catalog_once_v1'
        compile_plan(candidate, req, [], orientation_reference=ref)
        self.assertEqual(TRANSITION.read_bytes(), original)

    def test_actual_restored_request_keeps_worker_bound_reference(self):
        original = RESTORED.read_bytes(); req = load_saved(RESTORED)
        available = pose_policy_context(req)
        self.assertTrue(available['planning_ready'])
        self.assertTrue(available['keep_orientation_available'])
        candidate = wire(req, pose='keep_orientation_v1')
        compile_plan(candidate, req, [], orientation_reference=resolve_reference(req))
        self.assertEqual(RESTORED.read_bytes(), original)

    def test_actual_restore_binding_mutation_is_rejected_before_model(self):
        req = load_saved(RESTORED)
        req['virtual_search_restore']['node_visit_revision'] = 0
        with self.assertRaisesRegex(ValueError, 'invalid_virtual_search_restore'):
            pose_policy_context(req)


if __name__ == '__main__':
    unittest.main()
