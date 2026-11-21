"""Explicit alternatives with injected outcomes and saved observations; API/physics 0."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_plan import CONTRACT, STEP_KEYS, compile_plan, schema, wire_step
from orchestrator.strategy_service import StrategyDecisionService
from orchestrator.strategy_evidence import validate_plan_evidence
from orchestrator.strategy_failover import POLICY, observe, trigger, summary
from tests.test_hierarchical_planner import request
from tests.test_strategy_service import wire
from tests.test_explicit_action_planner import make_planner, result_for
from tests.test_accepted_pose_carry import carry
from worker.accepted_pose_carry import FIELD, bind
from worker.explicit_action_contract import action_id, digest, validate_request

ROOT = Path(__file__).absolute().parents[1]
SAVED = ROOT / 'runs/part_research_00161_prepare_20261109/episode/hierarchy_decisions.json'


def synthetic_reentry():
    """Saved geometry with an explicitly injected newer parent PASS proof.

    This is a reentry contract fixture, not a new physics/model observation.
    """
    requests = json.loads(SAVED.read_text(encoding='utf-8'))
    req = deepcopy(requests[9]['request'])
    worker = json.loads((SAVED.parent/'worker_result.json').read_text(encoding='utf-8'))
    edge = deepcopy(worker['attempts'][3])
    pose = deepcopy(edge['context']['pose_catalog'][0])
    edge['event_id'] = req['run_id'] + ':attempt:50'
    edge['request_id'] = req['run_id'] + ':request:50'
    edge['context']['request_id'] = edge['request_id']
    edge['choice']['pose_id'] = pose['pose_id']
    identity = action_id(requests[3]['request'],edge['choice'])
    edge['action_id'] = identity; edge['context']['action_id'] = identity
    edge['context']['pose_matrix'] = deepcopy(pose['matrix'])
    edge['raw']['pose'] = deepcopy(pose['matrix'])
    edge['raw']['chosen'] = deepcopy(edge['choice'])
    edge['raw']['scope']['pose'] = deepcopy(pose['matrix'])
    edge['raw']['scope']['selected_action'] = deepcopy(edge['choice'])
    edge['raw']['executed']['pose'] = deepcopy(pose['matrix'])
    edge['raw']['executed']['pose_id'] = pose['pose_id']
    projected,state,_,pose_id = carry(req,edge)
    return projected,state,pose_id


def inject_new_failure(req,state,pose_id):
    current=deepcopy(req)
    choice=dict(part_id='0',pose_id=pose_id,skill_id='single_force_v1',direction_id='xp',
        second_direction_id=None,support_ids=[],evidence_ids=[])
    identity=action_id(current,choice); event=current['run_id']+':attempt:51'
    assert identity not in current['attempted_action_ids']
    current['attempted_action_ids'].append(identity)
    current['attempted_actions'].append(dict(action_id=identity,
        **{k:v for k,v in choice.items() if k!='evidence_ids'}))
    current['path_stage_observations'].append(dict(action_id=identity,path_success=True,
        status='UNKNOWN',deadline_exceeded=False,backend='ASAP_RedMax',source_event_id=event))
    current['episode_feedback'].append(dict(evidence_id=event,action_id=identity,choice=choice,
        selected_pose_matrix=next(p['matrix'] for p in current['pose_catalog'] if p['pose_id']==pose_id),
        remaining_parts=current['remaining_parts'],status='UNKNOWN',accepted_for_transition=False,
        deadline_exceeded=False,path_success=True,stability_success=False))
    current['evidence_ids'].append(event); current[FIELD]=bind(current,state)
    return validate_request(current)


def authorized(req, *, threshold=2, primary=None, alternative=None):
    value = wire(req, attempts=4, supports='bounded_support_search_v1')
    for step in value['steps']:
        step['stability_failover'] = None
    if primary is not None:
        value['steps'] = [next(s for s in value['steps'] if s['part_id']==primary)]
    if len(req['remaining_parts']) > 2:
        first = value['steps'][0]
        other = alternative or next(p for p in req['candidate_ids'] if p != first['part_id'])
        alt = {k:deepcopy(first[k]) for k in STEP_KEYS}
        alt.update(part_id=other, support_policy='replace_once_v1', pose_policy='keep_orientation_v1')
        first['stability_failover'] = dict(after_failures=threshold, alternatives=[alt])
    return value


class StrategyFailoverTests(unittest.TestCase):
    def setUp(self):
        for target in ('urllib.request.OpenerDirector.open', 'socket.create_connection'):
            guard = patch(target, side_effect=AssertionError('NO_NETWORK'))
            blocked = guard.start(); self.addCleanup(guard.stop); self.addCleanup(blocked.assert_not_called)

    def service(self, *, planner_calls=1, threshold=2, null=False):
        def plan(req, cards, escalation, **kwargs):
            value = authorized(req, threshold=threshold)
            if null:
                for step in value['steps']: step['stability_failover'] = None
            return dict(status='planned', plan=compile_plan(value, req, cards, strategy_failure_policy=POLICY),
                metadata=dict(llm_live=False, model_returned='INJECTED'))
        scout = Mock(provider_contract=CONTRACT, strategy_failure_policy=POLICY)
        planner = Mock(provider_contract=CONTRACT, strategy_failure_policy=POLICY)
        scout.plan.side_effect = planner.plan.side_effect = plan
        with patch('orchestrator.hierarchical_service.retrieve', return_value=[]):
            service = StrategyDecisionService(scout, planner, {}, dict(scout_calls=0,
                planner_calls=planner_calls, fallback_calls=0, max_plan_steps=8, recovery_per_step=1),
                strategy_failure_policy=POLICY, recovery_support_policy='preserve_existing_v1',
                planning_feedback_policy='repair_evidence_v2')
        return service, planner

    def worker(self, svc, *, a_succeeds_in_child=False):
        def verifier(context, choice, params):
            success = choice['part_id'] != 'a' or (a_succeeds_in_child and len(context['parts_rest']) == 1)
            result = result_for(context, choice, success)
            result.update(backend='ASAP_RedMax', path_success=True, stability_success=success,
                raw_upstream=dict(parts_fall=['c'] if not success else []))
            return result
        return make_planner(decision=svc.handle, verifier=verifier,
            path_selection_policy='path_probe_round_v1', fallback_policy='path_stage_v1',
            decision_failure_policy='halt_v1')

    def test_legacy_schema_and_wire_are_unchanged(self):
        req = request(); value = wire(req)
        self.assertNotIn('stability_failover', schema(req, [])['properties']['steps']['items']['properties'])
        compile_plan(value, req, [])
        with self.assertRaisesRegex(ValueError, 'step_fields'):
            compile_plan(authorized(req), req, [])
        with self.assertRaisesRegex(ValueError, 'step_fields'):
            compile_plan(value, req, [], strategy_failure_policy=POLICY)

    def test_permission_is_pinned_and_tamper_is_rejected(self):
        req=request(); value=authorized(req)
        plan=compile_plan(value,req,[],strategy_failure_policy=POLICY)
        self.assertEqual(wire_step(plan['steps'][0],POLICY),value['steps'][0])
        validate_plan_evidence(plan,req)
        plan['steps'][0]['stability_failover']['after_failures']=3
        with self.assertRaisesRegex(ValueError,'plan_identity'):
            validate_plan_evidence(plan,req)

    def test_invalid_threshold_duplicate_mover_nested_permission_and_unknown_evidence_rejected(self):
        req=request()
        def field(value): return value['steps'][0]['stability_failover']
        mutations=[lambda v:field(v).update(after_failures=True),
            lambda v:field(v).update(after_failures=1),
            lambda v:v['steps'][0].update(max_attempts=1),
            lambda v:field(v)['alternatives'][0].update(part_id=v['steps'][0]['part_id']),
            lambda v:field(v)['alternatives'].append(deepcopy(field(v)['alternatives'][0])),
            lambda v:field(v)['alternatives'][0].update(stability_failover=None),
            lambda v:field(v)['alternatives'][0].update(evidence_ids=['invented'])]
        for mutate in mutations:
            value=authorized(req); mutate(value)
            with self.subTest(value=value),self.assertRaises(ValueError):
                compile_plan(value,req,[],strategy_failure_policy=POLICY)

    def test_provider_policy_must_match_service(self):
        with self.assertRaisesRegex(ValueError,'provider_mismatch'):
            StrategyDecisionService(Mock(provider_contract=CONTRACT),Mock(provider_contract=CONTRACT),{}, {},
                strategy_failure_policy=POLICY,recovery_support_policy='preserve_existing_v1')

    def test_two_bound_failures_switch_in_same_state_without_model_call(self):
        svc, planner=self.service(); worker=self.worker(svc)
        worker.plan(budget=8); report=worker.report()
        self.assertEqual([a['choice']['part_id'] for a in report['attempts']],['a','a','b'])
        self.assertEqual(planner.plan.call_count,1)
        self.assertFalse(report['success'])
        self.assertEqual(svc.decisions[2]['request']['remaining_parts'],['a','b','c'])
        self.assertEqual(svc.decisions[-1]['request']['remaining_parts'],['a','c'])
        self.assertEqual(svc.decisions[2]['response']['metadata']['strategy_failover']['target_part_id'],'b')
        selection=next(e for e in svc.events if e['kind']=='strategy_failover_selected')
        self.assertEqual(selection['model_authorization_sha256'],digest(selection['model_authorization']))
        self.assertEqual(selection['model_authorization']['alternatives'][0]['part_id'],'b')
        self.assertTrue(any(e.get('reason')=='strategy_failover_part_removed_replan' for e in svc.events))
        self.assertFalse(any(e.get('reason')=='recovery_changed_expected_removal' for e in svc.events))

    def test_deferred_part_can_be_selected_again_in_a_new_plan(self):
        svc, planner=self.service(planner_calls=2); worker=self.worker(svc,a_succeeds_in_child=True)
        worker.plan(budget=8); report=worker.report()
        self.assertTrue(report['success'],report['stop_msg'])
        self.assertEqual([a['choice']['part_id'] for a in report['attempts']],['a','a','b','a'])
        self.assertEqual(planner.plan.call_count,2)

    def test_null_permission_does_not_switch(self):
        svc, planner=self.service(null=True); worker=self.worker(svc)
        worker.plan(budget=8)
        self.assertTrue(all(a['choice']['part_id']=='a' for a in worker.report()['attempts']))
        self.assertFalse(any(e['kind']=='strategy_failover_selected' for e in svc.events))

    def test_higher_threshold_does_not_switch_at_two_failures(self):
        svc, _=self.service(threshold=3); worker=self.worker(svc)
        worker.plan(budget=2)
        self.assertEqual([a['choice']['part_id'] for a in worker.report()['attempts']],['a','a'])
        self.assertFalse(any(e['kind']=='strategy_failover_selected' for e in svc.events))

    def test_saved_failures_persist_across_replans_without_memory_substitution(self):
        requests=[r['request'] for r in json.loads(SAVED.read_text(encoding='utf-8'))]
        ledger={}
        for req in requests[5:10]: observe(req,ledger)
        req=requests[9]
        result=trigger(req,ledger,'0',3)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['observed_failures'],3)
        count=result['observed_failures']
        observe(req,ledger)
        self.assertEqual(trigger(req,ledger,'0',3)['observed_failures'],count)
        self.assertEqual(req,requests[9])
        self.assertFalse(summary(req,ledger)['candidate_pruning'])

    def test_saved_invalid_latest_fact_cannot_authorize_switch(self):
        base=json.loads(SAVED.read_text(encoding='utf-8'))[9]['request']
        def last(req): return next(f for f in req['episode_feedback'] if f['action_id']==req['attempted_action_ids'][-1])
        mutations=[lambda r:last(r).update(deadline_exceeded=True),
            lambda r:last(r).update(status='ERROR'),
            lambda r:last(r).update(accepted_for_transition=True),
            lambda r:last(r).update(remaining_parts=['not-this-state']),
            lambda r:r['path_stage_observations'][-1].update(backend='MOCK'),
            lambda r:last(r)['choice'].update(part_id='9'),
            lambda r:last(r)['selected_pose_matrix'][0].__setitem__(3,999)]
        for mutate in mutations:
            req=deepcopy(base); ledger={}; observe(req,ledger); mutate(req)
            with self.subTest(mutate=mutate):
                self.assertIsNone(trigger(req,ledger,'0',2))

    def test_only_error_path_or_memory_failure_is_not_counted(self):
        base=json.loads(SAVED.read_text(encoding='utf-8'))[1]['request']
        ledger={}; current,latest=observe(base,ledger)
        self.assertEqual(current,{})  # first actual observation was path failure
        self.assertIsNone(latest)
        self.assertIsNone(trigger(base,ledger,'3',2))

    def test_reentry_preserves_counts_but_requires_a_new_current_visit_failure(self):
        req,state,pose_id=synthetic_reentry(); ledger={}
        records,latest=observe(req,ledger)
        self.assertGreaterEqual(sum(row['part_id']=='0' for row in records.values()),3)
        self.assertIsNone(latest)
        self.assertIsNone(trigger(req,ledger,'0',3))
        current=inject_new_failure(req,state,pose_id)
        evidence=trigger(current,ledger,'0',3)
        self.assertEqual(evidence['latest_evidence_id'],req['run_id']+':attempt:51')
        self.assertGreaterEqual(evidence['observed_failures'],4)

    def test_new_entry_supersedes_only_older_restore_and_keeps_missing_context_guard(self):
        req,_,_=synthetic_reentry(); svc,_=self.service()
        key=(req['run_id'],req['state_id'])
        svc.restore_revisions[key]=(1,'previous_validated_restore_fixture')
        svc.restore_event_numbers[key]=40
        with patch('orchestrator.hierarchical_service.HierarchicalDecisionService.handle',return_value='accepted'):
            self.assertEqual(svc.handle(req),'accepted')
            self.assertNotIn(key,svc.restore_revisions)
            self.assertTrue(any(e['kind']=='virtual_restore_superseded_by_accepted_entry' for e in svc.events))
            missing=deepcopy(req);missing.pop(FIELD)
            with self.assertRaisesRegex(ValueError,'missing_accepted_pose_entry_context'):
                svc.handle(missing)
        self.assertEqual(svc.scout_calls,0)
        self.assertEqual(svc.planner_calls,0)

    def test_current_visit_attempt_or_nonolder_restore_cannot_fake_new_entry(self):
        req,state,pose_id=synthetic_reentry()
        cases=[(req,50),(req,60),(inject_new_failure(req,state,pose_id),40)]
        for current,previous_number in cases:
            svc,_=self.service();key=(current['run_id'],current['state_id'])
            svc.restore_revisions[key]=(1,'previous_validated_restore_fixture')
            svc.restore_event_numbers[key]=previous_number
            with self.subTest(previous_number=previous_number),self.assertRaisesRegex(ValueError,'missing_virtual_restore_context'):
                svc.handle(current)
            self.assertNotIn(key,svc.accepted_entry_revisions)

    def test_entry_revision_cannot_regress(self):
        req,state,_=synthetic_reentry(); svc,_=self.service()
        newer=deepcopy(req);state['pose_carry_activation_revision']=2;newer[FIELD]=bind(newer,state)
        with patch('orchestrator.hierarchical_service.HierarchicalDecisionService.handle',return_value='accepted'):
            svc.handle(newer)
            with self.assertRaisesRegex(ValueError,'stale_or_conflicting_accepted_pose_entry'):
                svc.handle(req)


if __name__=='__main__': unittest.main()
