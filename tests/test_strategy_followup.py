"""Saved real observations and injected responses; no new API or physics."""
from copy import deepcopy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_plan import CONTRACT, STEP_KEYS, compile_plan
from orchestrator.strategy_service import StrategyDecisionService
from tests.test_strategy_service import wire
from tests.test_hierarchical_planner import request, feedback
from tests.test_explicit_action_planner import make_planner, result_for
from worker.explicit_action_contract import digest, validate_request, validate_response

ROOT = Path(__file__).absolute().parents[1]
LIVE = ROOT/'runs/strategy_live_20261109_1631/episode'


def load(name): return json.loads((LIVE/name).read_text(encoding='utf-8'))


class StrategyFollowupTests(unittest.TestCase):
    def setUp(self):
        block = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.network = block.start()
        self.addCleanup(block.stop)
        self.addCleanup(self.network.assert_not_called)

    def service(self, value):
        scout = Mock(provider_contract=CONTRACT)
        planner = Mock(provider_contract=CONTRACT)
        def plan(req, cards, escalation, **kwargs):
            response = value(req) if callable(value) else deepcopy(value)
            return dict(status='planned', plan=compile_plan(response, req, cards),
                metadata=dict(llm_live=False, model_returned='INJECTED', call_id='INJECTED'))
        scout.plan.side_effect = plan
        with patch('orchestrator.hierarchical_service.retrieve', return_value=[]):
            service = StrategyDecisionService(scout, planner, {},
                dict(scout_calls=1, planner_calls=0, fallback_calls=0, max_plan_steps=8, recovery_per_step=1),
                recovery_support_policy='preserve_existing_v1', planning_feedback_policy='repair_evidence_v2')
        self.addCleanup(planner.plan.assert_not_called)
        return service, scout

    def cited_state(self):
        req = request()
        req['memory_references'] = [dict(evidence_id='frozen:old', current_part_id_mapping=None,
            hard_pruning=False, physics_reuse_allowed=False, reference_only=True,
            same_task=False, same_conditions=False, status='UNKNOWN', record_sha256='a'*64)]
        req['evidence_ids'] = ['frozen:old'] + [r['evidence_id'] for r in req['episode_feedback']]
        validate_request(req)
        value = wire(req)
        for step in value['steps']: step['evidence_ids'] = ['frozen:old']
        return req, value

    def test_saved_real_first_plan_survives_window_change_after_removal(self):
        decisions = load('hierarchy_decisions.json')
        old_plan = load('hierarchy_plans.json')[0]['result']['plan']
        value = {key: deepcopy(old_plan[key]) for key in
                 ('request_id','state_id','candidate_order','recovery_skill_ids')}
        value['steps'] = [{key: deepcopy(s[key]) for key in STEP_KEYS} for s in old_plan['steps']]
        service, scout = self.service(value)
        for item in decisions[:3]:
            req = deepcopy(item['request']); before = digest(req)
            response = service.handle(req)
            self.assertEqual(response['action_id'], item['response']['action_id'])
            self.assertEqual(digest(req), before)
        req = deepcopy(decisions[3]['request']); before = digest(req)
        response = service.handle(req)
        self.assertEqual(response['choice']['part_id'], '5')
        missing = set(value['steps'][1]['evidence_ids'])-set(req['evidence_ids'])
        self.assertTrue(missing)
        self.assertEqual(set(response['metadata']['plan_evidence']['pinned_only_evidence_ids']), missing)
        self.assertFalse(set(response['choice']['evidence_ids']) & missing)
        self.assertTrue(response['metadata']['plan_execution_without_api'])
        self.assertFalse(response['metadata']['plan_actual_llm'])
        validate_response(req, response)
        self.assertEqual(scout.plan.call_count, 1)
        self.assertEqual(digest(req), before)
        self.assertFalse(any(e.get('reason')=='strategy_citations_expired' for e in service.events))

    def test_same_step_window_change_does_not_change_anchor(self):
        req, value = self.cited_state()
        service, scout = self.service(value)
        first = service.handle(req)
        anchor = deepcopy(service.strategy_cursor['anchor'])
        nxt = feedback(req, first, False)
        nxt['memory_references'] = []
        nxt['evidence_ids'].remove('frozen:old')
        second = service.handle(validate_request(nxt))
        self.assertIsNotNone(second['choice'], second)
        self.assertEqual(service.strategy_cursor['anchor'], anchor)
        self.assertNotEqual(first['action_id'], second['action_id'])
        self.assertNotIn('frozen:old', second['choice']['evidence_ids'])
        self.assertEqual(scout.plan.call_count, 1)

    def test_same_id_changed_observation_halts_without_action(self):
        req, value = self.cited_state()
        service, scout = self.service(value)
        first = service.handle(req)
        nxt = feedback(req, first, False)
        nxt['memory_references'][0]['status'] = 'PASS'
        second = service.handle(nxt)
        self.assertIsNone(second['choice'])
        self.assertTrue(second['metadata']['decision_halt_v1'])
        self.assertTrue(any(e['kind']=='strategy_evidence_rejected' for e in service.events))
        self.assertEqual(scout.plan.call_count, 1)

    def test_same_id_contextual_match_change_is_not_source_mutation(self):
        req, value = self.cited_state()
        service, scout = self.service(value)
        first = service.handle(req)
        nxt = feedback(req, first, False)
        nxt['memory_references'][0].update(same_task=True, same_conditions=True, reference_only=False)
        second = service.handle(nxt)
        self.assertIsNotNone(second['choice'])
        self.assertEqual(scout.plan.call_count, 1)

    def test_missing_binding_in_saved_old_plan_fails_closed(self):
        req, value = self.cited_state()
        service, scout = self.service(value)
        first = service.handle(req)
        service.plan.pop('evidence_binding')
        second = service.handle(feedback(req, first, False))
        self.assertIsNone(second['choice'])
        self.assertTrue(second['metadata']['decision_halt_v1'])

    def test_registered_python_cannot_inherit_new_search_permission(self):
        req, value = self.cited_state()
        service, _ = self.service(value)
        with patch.object(service, '_skill_step', side_effect=AssertionError('VM_MUST_NOT_RUN')):
            choice, reason = service._skill_proposal(req,
                dict(support_policy='bounded_support_search_v1'), {}, time.monotonic()+1)
        self.assertIsNone(choice)
        self.assertEqual(reason, 'strategy_registered_support_policy_not_supported')

    def test_pinned_snapshots_stay_out_of_actual_vm_packet(self):
        req, value = self.cited_state()
        service, _ = self.service(value)
        service.handle(req)
        before = digest(service.plan)
        # Exercise the actual inherited packet path, intercepting only the
        # disposable VM call; no generated code or real API is executed here.
        from orchestrator.skill_program import program_hash
        from orchestrator.lesson_runtime import CONTRACT as RUNTIME
        frozen = json.loads((ROOT/'runs/support_learning_live_20261109_1155/revalidation_v2/skills_frozen.json').read_text(encoding='utf-8'))
        candidate = frozen['entries'][0]['candidate']
        row = dict(program=candidate['program'], program_hash=program_hash(candidate['program']),
                   card=dict(runtime_contract=RUNTIME))
        with patch('orchestrator.python_skill_program.step_program', return_value={}) as vm:
            service._skill_step(row, req, time.monotonic()+10)
        packet = vm.call_args.kwargs['extension']
        self.assertNotIn('evidence_binding', packet['plan'])
        self.assertEqual(packet['plan']['plan_id'], service.plan['plan_id'])
        self.assertEqual(before, digest(service.plan))
        self.assertEqual(packet['lessons'], [])

    def test_worker_support_replacement_and_transition_one_injected_plan(self):
        seen = []
        def verify(context, choice, params):
            seen.append(deepcopy(choice))
            # Controlled physics double: first [] fails, first support b fails,
            # and changing to c succeeds. This tests orchestration, not RedMax.
            failing = choice['part_id']=='a' and choice['support_ids'] != ['c']
            result = result_for(context, choice, feasible=not failing)
            result.update(backend='ASAP_RedMax', path_success=True, stability_success=not failing,
                raw_upstream=dict(parts_fall=['b'] if not choice['support_ids'] else ['c']))
            return result
        service, scout = self.service(lambda r: wire(r, supports='bounded_support_search_v1'))
        worker = make_planner(decision=service.handle, verifier=verify, max_grippers=2,
            path_selection_policy='path_probe_round_v1', fallback_policy='path_stage_v1',
            decision_failure_policy='halt_v1')
        worker.plan(budget=6)
        report = worker.report()
        self.assertTrue(report['success'], report)
        choices = [a for a in seen if a['part_id']=='a']
        self.assertEqual([a['support_ids'] for a in choices], [[], ['b'], ['c']])
        self.assertEqual(len({a['pose_id'] for a in choices}), 1)
        self.assertEqual(scout.plan.call_count, 1)
        self.assertTrue(any(e.get('metadata', {}).get('strategic_change_authority') ==
                            'model_authorized_bounded_support_search' for e in service.events))

    def test_saved_registered_python_same_context_and_action_with_compact_plan(self):
        from orchestrator.python_skill_program import build_context, step_program
        from orchestrator.lesson_runtime import extension_for
        req = deepcopy(load('hierarchy_decisions.json')[6]['request'])
        original = load('hierarchy_plans.json')[2]['result']['plan']
        value = {k: deepcopy(original[k]) for k in
                 ('request_id', 'state_id', 'candidate_order', 'recovery_skill_ids')}
        value['steps'] = [{k: deepcopy(s[k]) for k in STEP_KEYS} for s in original['steps']]
        frozen = json.loads((ROOT/'runs/support_learning_live_20261109_1155/revalidation_v2/skills_frozen.json').read_text(encoding='utf-8'))
        candidate = frozen['entries'][0]['candidate']
        identity, program = candidate['program_hash'], candidate['program']
        plan = compile_plan(value, req, [dict(skill_id=identity, program_hash=identity, program=program)])
        service, _ = self.service(value)
        service.plan = deepcopy(plan)
        before = digest(service.plan)
        full = extension_for(plan=plan)
        compact = extension_for(plan=service._execution_plan())
        self.assertEqual(build_context(req, full), build_context(req, compact))
        self.assertLess(len(json.dumps(compact)), len(json.dumps(full)))
        a = step_program(program, req, max_actions=50000, extension=full)
        b = step_program(program, req, max_actions=50000, extension=compact)
        self.assertIsNotNone(a['response'], a)
        self.assertEqual(a['response']['choice'], b['response']['choice'])
        self.assertEqual(set(b['response']['choice']['support_ids']), {'2','3'})
        self.assertEqual(digest(service.plan), before)


if __name__ == '__main__': unittest.main()
