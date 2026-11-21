"""Saved observations and injected plans only; no new API or physics success."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.planning_repair_review import (
    stability_repair, repair_review, compact_escalation, stability_choice_comparison)
from orchestrator.conditional_plan_v4 import CONTRACT, compile_plan, schema_v4
from orchestrator.conditional_plan import action_for_step
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.explicit_action_provider_v5 import restore_request_v5
from tests.test_explicit_action_service import provider
from tests.test_hierarchical_planner import policy, request, plan_value, feedback
from tests.test_plan_pose_continuity import wire, tilted

ROOT = Path(__file__).absolute().parents[1]
SAVED = ROOT/'runs/frozen_skill_transfer_02628_20261022'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def envelopes():
    with sqlite3.connect((SAVED/'live/planner.sqlite').as_uri()+'?mode=ro', uri=True) as c:
        return [(json.loads(a), json.loads(b)) for a,b in c.execute(
            'SELECT payload_json,response_json FROM rank_requests ORDER BY created_utc')]


def reviewed(plan, cards):
    value = wire(plan)
    value['skill_review'] = [dict(skill_id=c['skill_id'],
        disposition='defer' if c.get('currently_proposed_choice') else 'unavailable',
        reason='Fixture justification; no claim about actual model reasoning.') for c in cards]
    return value


class RepairReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        env = patch.dict(os.environ, OPENAI_API_KEY='OFFLINE_NOT_A_REAL_KEY')
        env.start(); self.addCleanup(env.stop)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        self.net = net.start(); self.addCleanup(net.stop); self.addCleanup(self.net.assert_not_called)
        self.e, self.result = envelopes()[0]
        self.request, self.cards = self.e['request'], self.e['skills']
        self.value = reviewed(self.result['plan'], self.cards)

    def service(self, name='new', transport=None, **kwargs):
        return SolPlanService(policy()['roles']['planner'], self.path/(name+'.sqlite'), transport or Mock(),
            provider_contract=CONTRACT, planning_feedback_policy='repair_evidence_v2', **kwargs)

    def test_first_actual_path_failure_has_immediate_proposal_not_future_execution(self):
        old = deepcopy((self.request,self.cards))
        review = repair_review(self.request,self.cards,'preserve_existing_v1')
        self.assertEqual(review['current_failure']['kind'],'path_failed')
        row=review['skills'][0]
        self.assertTrue(row['current_proposal_available'])
        self.assertEqual(row['currently_proposed_choice']['direction_id'],'xn')
        self.assertTrue(row['automatic_recovery_support_guard']['allowed'])
        self.assertFalse(review['selection_forced'])
        self.assertEqual((self.request,self.cards),old)

    def test_actual_stability_support_capacity_and_separate_history(self):
        d=read(SAVED/'live/episode/hierarchy_decisions.json')
        a=stability_repair(d[2]['request']);b=stability_repair(d[3]['request'])
        self.assertEqual(a['current']['observed_unheld_parts'],['1','2','3','4'])
        self.assertFalse(a['current']['all_observed_fit_as_supports'])
        self.assertEqual(b['current']['observed_unheld_parts'],['2','4'])
        self.assertTrue(b['current']['all_observed_fit_as_supports'])
        self.assertTrue(b['current']['preserve_existing_union_exceeds_capacity'])
        self.assertEqual(b['current']['preserve_existing_union'],['1','2','3','4'])
        self.assertEqual(len(b['same_state_observations']),2)
        self.assertFalse(b['hard_pruning'])

    def test_missing_diagnostics_and_raw_parts_do_not_become_support_targets(self):
        r=deepcopy(envelopes()[1][0]['request'])
        r['episode_feedback'][-1].pop('stability_diagnostics')
        r['episode_feedback'][-1]['raw_stability_parts']=['4']
        facts=stability_repair(r)['current']
        self.assertFalse(facts['diagnostic_available'])
        self.assertEqual(facts['observed_unheld_parts'],[])
        self.assertIsNone(facts['all_observed_fit_as_supports'])

    def test_stale_timeout_or_bad_provenance_cannot_supply_current_repair(self):
        for field, val in [('deadline_exceeded',True),('selected_pose_matrix',None),
                           ('remaining_parts',['0']),('action_id','action:'+'0'*64)]:
            r=deepcopy(envelopes()[1][0]['request']);r['episode_feedback'][-1][field]=val
            self.assertIsNone(stability_repair(r)['current'],field)
        r=deepcopy(envelopes()[1][0]['request']);r['episode_feedback']=[]
        r['evidence_ids']=[x['evidence_id'] for x in r['memory_references']]
        self.assertIsNone(stability_repair(r)['current'])

    def test_dynamic_diagnostics_cannot_assign_held_or_foreign_body(self):
        r=deepcopy(envelopes()[1][0]['request'])
        r['episode_feedback'][-1]['stability_diagnostics']['dynamic_failure_parts'][0]['part_id']='0'
        facts=stability_repair(r)['current']
        self.assertEqual(facts['observed_unheld_parts'],[])
        self.assertIsNone(facts['support_target_recommendation'])

    def test_initial_disconnected_is_not_dynamic_movement(self):
        from tests.test_skill_program import request as r0, append_observation
        from worker.stability_diagnostics import StabilityObservation
        r=append_observation(r0(),path=True,stability=False)
        obs=StabilityObservation();obs.checker_count=1
        obs.data['initial_disconnected_parts']=['b'];obs.returned(False,['b'])
        r['episode_feedback'][-1]['stability_diagnostics']=obs.data
        f=stability_repair(r)['current']
        self.assertEqual(f['initial_disconnected_parts'],['b'])
        self.assertEqual(f['dynamic_failure_parts'],[])

    def test_defer_is_valid_without_forcing_skill_or_altering_action(self):
        before=deepcopy(self.value)
        p=compile_plan(self.value,self.request,self.cards)
        actual=action_for_step(p,0,self.request)
        self.assertEqual(actual['choice']['part_id'],'6')
        self.assertEqual(self.value,before)
        self.assertEqual(p['skill_review'][0]['disposition'],'defer')
        self.assertFalse(p['physics_success_implied'])

    def select_now(self):
        v=deepcopy(self.value);c=self.cards[0]['currently_proposed_choice']
        v['steps']=[dict(part_id=c['part_id'],mode='registered_skill',pose_reference=None,
            primitive=None,direction=None,second_direction=None,support_ids=[],evidence_ids=[],
            registered_skill_id=self.cards[0]['skill_id'],pose_policy=None)]
        v['skill_review'][0]['disposition']='execute_now'
        return v

    def test_selected_now_requires_matching_mover_and_review(self):
        v=self.select_now();compile_plan(v,self.request,self.cards)
        v['steps'][0]['part_id']='6'
        with self.assertRaisesRegex(ValueError,'proposal_mover_mismatch'):compile_plan(v,self.request,self.cards)
        v=self.select_now();v['skill_review'][0]['disposition']='defer'
        with self.assertRaisesRegex(ValueError,'review_action_mismatch'):compile_plan(v,self.request,self.cards)

    def test_unavailable_cannot_be_selected_now_but_future_authorization_allowed(self):
        e,result=envelopes()[1];v=reviewed(result['plan'],e['skills'])
        self.assertTrue(v['recovery_skill_ids'])
        compile_plan(v,e['request'],e['skills'])
        v['skill_review'][0]['disposition']='execute_now'
        with self.assertRaisesRegex(ValueError,'review_action_mismatch'):compile_plan(v,e['request'],e['skills'])

    def test_missing_duplicate_unknown_or_blank_reviews_rejected(self):
        modifications=[lambda v:v.update(skill_review=[]),
            lambda v:v['skill_review'].append(deepcopy(v['skill_review'][0])),
            lambda v:v['skill_review'][0].update(skill_id='invented'),
            lambda v:v['skill_review'][0].update(reason=' '),
            lambda v:v['skill_review'][0].update(disposition='unavailable')]
        for mutation in modifications:
            v=deepcopy(self.value);mutation(v)
            with self.assertRaises(ValueError):compile_plan(v,self.request,self.cards)

    def test_empty_library_needs_empty_review(self):
        r=request();v=reviewed(plan_value(r),[])
        p=compile_plan(v,r,[]);self.assertEqual(p['skill_review'],[])
        self.assertEqual(schema_v4(r,[])['properties']['skill_review']['maxItems'],0)

    def test_v4_retains_bounded_pose_and_remaining_guards(self):
        r=request();v=reviewed(plan_value(r),[]);p=compile_plan(v,r,[])
        altered=deepcopy(r);altered['pose_catalog']=[tilted(3)]
        out=action_for_step(p,0,altered)
        self.assertEqual(out['metadata']['plan_pose_binding']['match_kind'],'bounded_rebind')
        with self.assertRaisesRegex(ValueError,'remaining_guard'):action_for_step(p,1,r)

    def test_prompt_resolves_immediate_vs_future_and_ambiguous_action_only(self):
        e=deepcopy(self.e);e.update(plan_provider_contract=CONTRACT,planning_feedback_policy='repair_evidence_v2')
        body=self.service()._body(e);packet=json.loads(body['input'][0]['content'])
        self.assertNotIn('Return action steps only',body['instructions'])
        self.assertIn('either action or registered_skill mode',body['instructions'])
        self.assertIn('does NOT execute it now',body['instructions'])
        self.assertEqual(restore_request_v5(packet['observation']),self.request)
        self.assertIn('stability_repair',packet)
        self.assertEqual(len(packet['repair_review']['skills']),1)
        self.assertEqual(set(body['text']['format']['schema']['required']),set(body['text']['format']['schema']['properties']))

    def test_compaction_keeps_guard_facts_without_nested_lesson_or_review_copy(self):
        e=envelopes()[1][0]['escalation'];before=deepcopy(e);out=compact_escalation(e)
        self.assertEqual(e,before)
        self.assertEqual(out['lesson_references'],e['lesson_references'])
        self.assertTrue(all('lesson_references' not in x for x in out['prior_events']))
        self.assertLess(len(json.dumps(out)),len(json.dumps(e)))

    def test_force_only_repeat_is_reported_not_banned_support_change_is_distinct(self):
        r=envelopes()[1][0]['request'];old=deepcopy(r['episode_feedback'][-1]['choice'])
        old['direction_id']='xn'
        self.assertTrue(stability_choice_comparison(r,old)['same_failed_stability_configuration'])
        old['support_ids']=['1','3']
        result=stability_choice_comparison(r,old)
        self.assertEqual(result['changed_configuration_fields'],['supports'])
        self.assertFalse(result['hard_pruning']);self.assertIsNone(result['next_physics_result'])

    def test_provider_invalid_review_rejected_cached_without_retry(self):
        v=deepcopy(self.value);v['skill_review'][0]['disposition']='execute_now'
        raw=provider(v);raw['model']='gpt-6.1-sol';send=Mock(return_value=raw)
        svc=self.service(transport=send)
        a=svc.plan(self.request,self.cards,self.e['escalation'],8,30)
        self.assertEqual(a['status'],'rejected')
        self.assertEqual(a['metadata']['plan_contract_error'],'skill_review_action_mismatch')
        cached=svc.plan(self.request,self.cards,self.e['escalation'],8,30)
        self.assertEqual(cached['status'],'rejected')
        self.assertEqual(cached['plan'],a['plan'])
        self.assertEqual(cached['metadata']['plan_contract_error'],a['metadata']['plan_contract_error'])
        self.assertEqual(send.call_count,1)

    def test_provider_success_caches_and_no_contract_cross_reuse(self):
        raw=provider(self.value);raw['model']='gpt-6.1-sol';send=Mock(return_value=raw)
        svc=self.service(transport=send)
        a=svc.plan(self.request,self.cards,self.e['escalation'],8,30)
        self.assertEqual(a['status'],'planned',a)
        self.assertFalse(a['metadata']['llm_live'])
        svc.plan(self.request,self.cards,self.e['escalation'],8,30)
        self.assertEqual(send.call_count,1)
        old=SolPlanService(policy()['roles']['planner'],self.path/'new.sqlite',send,
            provider_contract='conditional_plan_v3',planning_feedback_policy='path_evidence_v1')
        self.assertEqual(old.plan(self.request,self.cards,self.e['escalation'],8,30)['fallback_reason'],'request_id_conflict')

    def test_contract_policy_mismatch_fails_before_api(self):
        for contract,feedback_policy in [(CONTRACT,'path_evidence_v1'),('conditional_plan_v3','repair_evidence_v2')]:
            with self.assertRaisesRegex(ValueError,'v4_requires'):
                SolPlanService(policy()['roles']['planner'],self.path/'bad.sqlite',Mock(),
                    provider_contract=contract,planning_feedback_policy=feedback_policy)

    def test_current_skill_selection_reaches_code_and_preserves_input_without_physics(self):
        p=compile_plan(self.select_now(),self.request,self.cards)
        planner=Mock();planner.provider_contract=CONTRACT
        planner.plan.return_value=dict(status='planned',plan=p,metadata={'llm_live':False})
        limits=policy()['episode'];limits['scout_calls']=0
        svc=HierarchicalDecisionService(Mock(),planner,read(SAVED/'frozen/skills.json'),limits,
            lesson_library=read(SAVED/'frozen/lessons.json'),lesson_runtime=True,
            recovery_support_policy='preserve_existing_v1',planning_feedback_policy='repair_evidence_v2')
        original=deepcopy(self.request)
        result=svc.handle(self.request)
        self.assertEqual(result['choice'],self.cards[0]['currently_proposed_choice'])
        self.assertEqual(result['metadata']['hierarchy_role'],'plan_skill')
        self.assertTrue(result['metadata']['generated_program_executed'])
        self.assertEqual(planner.plan.call_count,1)
        self.assertTrue(any(e['kind']=='skill_review_recorded' for e in svc.events))
        self.assertEqual(self.request,original)
        self.assertFalse(any(e['kind']=='physics_feedback' for e in svc.events))

    def test_stability_replan_changes_supports_keeps_model_choice_and_records_configuration(self):
        e,r=envelopes()[1];v=reviewed(r['plan'],e['skills']);p=compile_plan(v,e['request'],e['skills'])
        planner=Mock();planner.provider_contract=CONTRACT
        planner.plan.return_value=dict(status='planned',plan=p,metadata={'llm_live':False})
        limits=policy()['episode'];limits['scout_calls']=0
        svc=HierarchicalDecisionService(Mock(),planner,read(SAVED/'frozen/skills.json'),limits,
            lesson_library=read(SAVED/'frozen/lessons.json'),lesson_runtime=True,
            recovery_support_policy='preserve_existing_v1',planning_feedback_policy='repair_evidence_v2')
        out=svc.handle(e['request'])
        self.assertEqual(out['choice']['support_ids'],['1','3'])
        self.assertEqual(out['metadata']['stability_repair_comparison']['changed_configuration_fields'],['supports'])
        self.assertFalse(out['metadata']['stability_repair_comparison']['hard_pruning'])


if __name__=='__main__':unittest.main()
