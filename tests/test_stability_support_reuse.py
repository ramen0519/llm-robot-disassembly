"""Offline historical replays and injected plans, never live model/physics tests."""
from copy import deepcopy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import Mock

from orchestrator.skill_registry import retrieve
from orchestrator.python_skill_program import evaluate, build_context
from orchestrator.planning_repair_review import stability_repair, repair_review, support_change_assessment
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.conditional_plan_v4 import compile_plan
from tests.test_hierarchical_planner import policy

ROOT = Path(__file__).absolute().parents[1]
OLD = ROOT/'runs/python_stability_threepart_20261020/generation_02'
NEW = ROOT/'runs/stability_support_review_20261105'

def read(p): return json.loads(p.read_text(encoding='utf8'))

class StabilitySupportReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decisions = read(ROOT/'runs/pose_capacity_fix_20261105/live/episode/hierarchy_decisions.json')
        cls.old_snapshot = read(OLD/'skills_frozen.json')
        cls.row = retrieve(cls.old_snapshot)[0]
        cls.program = cls.row['program']
        cls.snapshot = read(NEW/'frozen/skills.json')

    def r(self,n=5): return deepcopy(self.decisions[n-1]['request'])

    def test_existing_skill_proposes_from_measured_current_ids(self):
        for n,supports in [(5,['3','4']),(6,['3','4']),(7,['3','5'])]:
            r=self.r(n); before=deepcopy(r)
            choice=evaluate(self.program,r)['choice']
            self.assertEqual(choice['support_ids'],supports)
            self.assertEqual(choice['part_id'],r['episode_feedback'][-1]['choice']['part_id'])
            self.assertEqual(choice['evidence_ids'],[r['episode_feedback'][-1]['evidence_id']])
            self.assertEqual(r,before)

    def test_four_measured_bodies_two_supports_abstains_not_impossible(self):
        r=self.r(3)
        self.assertEqual(build_context(r)[0]['last']['unstable'],['1','2','3','4'])
        self.assertIsNone(evaluate(self.program,r)['choice'])
        f=stability_repair(r)
        self.assertFalse(f['current']['all_observed_fit_as_supports'])
        self.assertFalse(f['hard_pruning'])

    def test_initial_path_failure_and_previous_pass_cannot_trigger_stability_skill(self):
        for n in (1,2,4,8): self.assertIsNone(evaluate(self.program,self.r(n))['choice'])

    def test_failed_or_inconsistent_instrumentation_cannot_supply_dynamic_roles(self):
        for mutate in (
            lambda f:f['stability_diagnostics'].update(check_success_returned=False),
            lambda f:f['stability_diagnostics'].update(instrumentation_errors=['observer_error']),
            lambda f:f.update(raw_stability_parts=['wrong']),
            lambda f:f.update(accepted_for_transition=None),
        ):
            r=self.r();mutate(r['episode_feedback'][-1])
            self.assertEqual(build_context(r)[0]['last']['unstable'],[])
            self.assertIsNone(evaluate(self.program,r)['choice'])
            self.assertEqual(stability_repair(r)['current']['observed_unheld_parts'],[])

    def test_held_mover_base_or_foreign_attribution_rejects_whole_dynamic_list(self):
        for part in ('1','0','6','foreign'):
            r=self.r();f=r['episode_feedback'][-1]
            f['stability_diagnostics']['dynamic_failure_parts'][0]['part_id']=part
            # Returned representative remains 4, so this exercises role binding.
            self.assertEqual(build_context(r)[0]['last']['unstable'],[])
            self.assertIsNone(evaluate(self.program,r)['choice'])

    def test_stale_or_late_observation_is_not_current(self):
        for field,value in [('deadline_exceeded',True),('selected_pose_matrix',None),('remaining_parts',['0'])]:
            r=self.r();r['episode_feedback'][-1][field]=value
            self.assertIsNone(build_context(r)[1])
            self.assertIsNone(evaluate(self.program,r)['choice'])

    def test_raw_representative_is_distinct_from_full_measured_list(self):
        r=self.r();f=r['episode_feedback'][-1]
        self.assertEqual(f['raw_stability_parts'],['4'])
        self.assertEqual(build_context(r)[0]['last']['unstable'],['3','4'])

    def test_support_replacement_explains_release_without_success_claim(self):
        r=self.r();c=evaluate(self.program,r)['choice'];before=deepcopy((r,c))
        a=support_change_assessment(r,c)
        self.assertEqual(a['released_support_ids'],['1','2'])
        self.assertEqual(a['added_support_ids'],['3','4'])
        self.assertEqual(a['observed_bodies_remaining_unheld'],[])
        self.assertTrue(a['fresh_path_and_stability_required'])
        self.assertEqual(a['newly_released_support_stability'],'UNKNOWN')
        self.assertFalse(a['physical_success_known'])
        self.assertEqual((r,c),before)

    def test_same_supports_report_unaddressed_observations_but_do_not_ban(self):
        r=self.r();c=deepcopy(r['episode_feedback'][-1]['choice'])
        a=support_change_assessment(r,c)
        self.assertEqual(a['observed_bodies_remaining_unheld'],['3','4'])
        self.assertFalse(a['hard_pruning'])
        c['part_id']='3'
        a=support_change_assessment(r,c)
        self.assertTrue(a['prior_observations_reference_only'])

    def test_existing_registered_entries_and_admission_scopes_preserved(self):
        rows=retrieve(self.snapshot)
        self.assertEqual(len(rows),2)
        self.assertIn(self.old_snapshot['entries'][0],self.snapshot['entries'])
        r=next(r for r in rows if r['program_hash']==self.row['program_hash'])
        self.assertFalse(r['current_attempt_verified'])
        self.assertTrue(r['requires_fresh_physics'])
        self.assertEqual(r['validation_scope'],self.row['validation_scope'])
        self.assertTrue(all(v['source_task']=='07009' for v in r['validation_scope']))

    def service(self):
        return HierarchicalDecisionService(Mock(),Mock(),self.snapshot,policy()['episode'],
            recovery_support_policy='preserve_existing_v1',planning_feedback_policy='repair_evidence_v2',
            lesson_library=read(NEW/'frozen/lessons.json'),lesson_runtime=True,max_skill_attempts=3)

    def test_explicit_plan_can_select_existing_skill_automatic_release_is_blocked(self):
        r=self.r();svc=self.service();cards=svc._cards(r,time.monotonic()+20)
        identity=self.row['program_hash'];row=next(c for c in cards if c['skill_id']==identity)
        review=repair_review(r,cards,'preserve_existing_v1')
        rr=next(c for c in review['skills'] if c['skill_id']==identity)
        self.assertTrue(rr['current_proposal_available'])
        self.assertFalse(rr['automatic_recovery_support_guard']['allowed'])
        self.assertTrue(rr['automatic_recovery_support_guard']['requires_new_explicit_plan'])
        c=row['currently_proposed_choice']
        wire=dict(request_id=r['request_id'],state_id=r['state_id'],candidate_order=r['candidate_ids'],
            steps=[dict(part_id=c['part_id'],mode='registered_skill',pose_reference=None,primitive=None,
                direction=None,second_direction=None,support_ids=[],evidence_ids=[],registered_skill_id=identity,pose_policy=None)],
            recovery_skill_ids=[identity],skill_review=[dict(skill_id=x['skill_id'],
                disposition='execute_now' if x['skill_id']==identity else 'unavailable',reason='Injected offline test choice.') for x in cards])
        svc.plan=compile_plan(wire,r,cards)
        svc.last_failure_context=dict(choice=r['episode_feedback'][-1]['choice'],
            remaining_parts=r['remaining_parts'],evidence_id=r['episode_feedback'][-1]['evidence_id'])
        self.assertIsNone(svc._program_response(identity,r,time.monotonic()+20,'recovery'))
        out=svc._execute(r,time.monotonic()+20)
        self.assertEqual(out['choice']['support_ids'],['3','4'])
        self.assertEqual(out['metadata']['hierarchy_role'],'plan_skill')
        self.assertTrue(out['metadata']['generated_program_executed'])
        svc.scout.handle.assert_not_called();svc.planner.plan.assert_not_called()

    def test_old_seven_registered_case_outputs_still_match(self):
        for v in self.old_snapshot['entries'][0]['validations']:
            case=OLD/'cases'/v['case']['case_id']
            r=read(case/'program_request.json');out=evaluate(self.program,r)
            selected=read(case/'program_step.json')['response']['choice']
            self.assertEqual(out['choice'],selected,v['case']['case_id'])

if __name__=='__main__': unittest.main()
