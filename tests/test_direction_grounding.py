"""Synthetic code-sensitivity fixtures only; no model, CAD solver or physical PASS."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch
from tests.test_python_revision import packet
from tests.test_skill_discovery import config
from orchestrator.lesson_discovery import LessonDiscoveryService
from orchestrator.lesson_runtime import extension_for
from orchestrator.python_skill_program import step_program
from orchestrator.python_skill_revision import audit_revision,FLAG
from orchestrator.python_direction_grounding import check_direction_use,source_diagnostics
from orchestrator.python_direction_probe import inspect
from worker.task_memory import digest

ROOT=Path(__file__).absolute().parents[1]
OLD=ROOT/'runs/execution_revision_live_20261021/live_cycle/candidate_02'

SELECTOR='''def select_directions(ctx, part_id, pose_id):
    parts = ctx['geometry']['parts']
    origin = parts[part_id]['bounds_center']
    pose = None
    for candidate in ctx['pose_catalog']:
        if candidate['pose_id'] == pose_id:
            pose = candidate['matrix']
    if pose is None:
        return None
    best = None
    best_score = -1000000
    for direction in ctx['directions']:
        vector = ctx['directions'][direction]
        score = 0.0
        for other in ctx['remaining_parts']:
            if other == part_id or other not in parts:
                continue
            center = parts[other]['bounds_center']
            relative = [center[i] - origin[i] for i in range(3)]
            world = [sum([pose[i][j] * relative[j] for j in range(3)]) for i in range(3)]
            score -= max(0.0, sum([world[i] * vector[i] for i in range(3)]))
        if direction == ctx['last']['choice']['direction_id']:
            score -= 100.0
        if score > best_score or (score == best_score and (best is None or direction < best)):
            best_score = score
            best = direction
    return [best]
'''
PROPOSE='''def propose(ctx):
    part = ctx['candidate_ids'][-1]
    pose = ctx['last']['choice']['pose_id']
    dirs = select_directions(ctx, part, pose)
    if dirs is None:
        return None
    primitive = 'single_force_v1'
    second = None
    if len(dirs) == 2:
        primitive = 'turn_force_two_leg_v1'
        second = dirs[1]
    return {'part_id':part, 'pose_id':pose, 'skill_id':primitive,
        'direction_id':dirs[0], 'second_direction_id':second,
        'support_ids':[], 'evidence_ids':[ctx['failure']['evidence_id']]}
'''

def program(selector=SELECTOR,propose=PROPOSE):
    return dict(language='python_policy_v1',source=propose+'\n'+selector)

def grounded_packet():
    p=packet();p['context'][FLAG]=3
    req=p['validation_feedback'][0]['current_observation_request']
    for name,center in [('a',[3,0,0]),('b',[0,0,0]),('base',[0,0,-2])]:
        req['geometry']['parts'][name]=dict(bounds_center=center,center_mass=list(center),
            bounds=[[v-0.5 for v in center],[v+0.5 for v in center]],extents=[1,1,1],
            validity=dict(bounds_finite=True,winding_consistent=True))
    return p

class DirectionGroundingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='direction_grounding_UNIT_',dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.p=grounded_packet();self.req=self.p['validation_feedback'][0]['current_observation_request']
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.net=net.start();self.addCleanup(net.stop);self.addCleanup(self.net.assert_not_called)
    def check(self,source):
        result=step_program(source,self.req,extension=extension_for(),diagnose=True)
        self.assertIsNotNone(result['response'],result)
        return check_direction_use(source,self.req,result['response']['choice'])

    def test_geometry_and_failure_affect_actual_directions_without_mutating_input(self):
        before=deepcopy(self.p)
        result=audit_revision(program(),self.p,None)
        self.assertTrue(result['accepted'],result)
        audit=result['checks'][0]['direction_grounding']
        self.assertTrue(audit['geometry_direction_sensitive']);self.assertTrue(audit['failure_direction_sensitive'])
        self.assertTrue(audit['actual_helper_bound_to_choice']);self.assertTrue(audit['map_order_invariant'])
        self.assertTrue(all(r['actual_choice_follows'] for r in audit['helper_return_interventions']))
        self.assertEqual((result['api_calls'],result['physical_calls']),(0,0))
        self.assertFalse(result['new_physical_success']);self.assertEqual(self.p,before)

    def test_saved_real_candidate_is_not_relabelled_as_grounded(self):
        p=json.loads((OLD/'generation_packet.json').read_text());p['context'][FLAG]=3
        source=json.loads((OLD/'provider_proposal.json').read_text())['program']
        result=audit_revision(source,p,None)
        self.assertFalse(result['accepted'])
        self.assertIn('direction_helper_contract_missing',result['checks'][0]['direction_grounding']['errors'])
        self.assertIn(dict(function='propose',name='best_extent'),source_diagnostics(source)['unused_assigned_locals'])

    def test_read_but_ignore_geometry_and_failure_is_rejected(self):
        selector="def select_directions(ctx, part_id, pose_id):\n    dead = ctx['geometry']['parts'][part_id]['extents']\n    failure = ctx['last']['choice']['direction_id']\n    return ['yn']\n"
        value=self.check(program(selector))
        self.assertFalse(value['accepted'])
        self.assertIn('geometry_direction_dependence_not_demonstrated',value['errors'])
        self.assertIn('failure_direction_dependence_not_demonstrated',value['errors'])

    def test_geometry_used_only_for_mover_does_not_pass_direction_gate(self):
        propose=PROPOSE.replace("part = ctx['candidate_ids'][-1]", "part = max(ctx['candidate_ids'], key=lambda p: sum(ctx['geometry']['parts'][p]['extents']))")
        value=self.check(program("def select_directions(ctx, part_id, pose_id):\n    return ['yn']\n",propose))
        self.assertFalse(value['geometry_direction_sensitive'])

    def test_geometry_only_without_failure_details_is_unverified(self):
        value=self.check(program(SELECTOR.replace("        if direction == ctx['last']['choice']['direction_id']:\n            score -= 100.0\n",'')))
        self.assertTrue(value['geometry_direction_sensitive']);self.assertFalse(value['failure_direction_sensitive'])

    def test_failure_only_without_geometry_is_unverified(self):
        selector="def select_directions(ctx, part_id, pose_id):\n    if ctx['last']['choice']['direction_id'] == 'xp':\n        return ['yn']\n    return ['xp']\n"
        value=self.check(program(selector))
        self.assertFalse(value['geometry_direction_sensitive']);self.assertTrue(value['failure_direction_sensitive'])

    def test_only_abstention_changes_do_not_count_as_direction_dependence(self):
        selector="def select_directions(ctx, part_id, pose_id):\n    if ctx['geometry']['parts']['a']['bounds_center'][0] != 3:\n        return None\n    return ['yn']\n"
        value=self.check(program(selector))
        self.assertFalse(value['geometry_direction_sensitive']);self.assertFalse(value['accepted'])

    def test_helper_called_but_result_ignored_is_rejected(self):
        baseline=self.check(program())['original_direction_sequence'][0]
        source=program(propose=PROPOSE.replace("'direction_id':dirs[0]","'direction_id':"+repr(baseline)))
        value=self.check(source)
        self.assertIn('direction_helper_result_not_driving_actual_choice',value['errors'])

    def test_helper_not_called_cannot_pass(self):
        source=program(propose=PROPOSE.replace('dirs = select_directions(ctx, part, pose)',"dirs = ['yn']"))
        self.assertIn('direction_helper_not_bound_to_actual_choice',self.check(source)['errors'])

    def test_dict_enumeration_order_not_a_directional_measurement(self):
        selector="def select_directions(ctx, part_id, pose_id):\n    for name in ctx['directions']:\n        if name != ctx['last']['choice']['direction_id']:\n            return [name]\n    return None\n"
        value=self.check(program(selector))
        self.assertFalse(value['map_order_invariant'])

    def test_both_directions_of_two_leg_action_are_bound(self):
        selector=SELECTOR.replace('    return [best]',"    return [best, 'zp' if best != 'zp' else 'xn']")
        value=self.check(program(selector))
        self.assertTrue(value['accepted'],value)
        self.assertEqual(len(value['original_direction_sequence']),2)

    def test_nonidentity_pose_and_no_fixed_base_still_supported(self):
        from tests.test_skill_program import request,append_observation
        geometry=deepcopy(self.req['geometry'])
        self.req=request()
        self.req['base_part']=None
        self.req['candidate_ids']=list(self.req['remaining_parts'])
        self.req['geometry']=geometry
        rotation=[[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]
        self.req['pose_catalog'][0]['matrix']=rotation
        append_observation(self.req)
        value=self.check(program())
        self.assertTrue(value['accepted'],value)

    def test_missing_geometry_is_not_invented_to_pass(self):
        for r in self.req['geometry']['parts'].values():r.pop('bounds_center')
        constant=program("def select_directions(ctx, part_id, pose_id):\n    return ['yn']\n")
        value=self.check(constant)
        self.assertFalse(value['geometry_direction_sensitive'])

    def test_helper_rejects_eligibility_queries_and_input_mutation(self):
        for addition in ("    q = eligible({})\n", "    ctx['remaining_parts'].append('bad')\n"):
            source=program(SELECTOR.replace('    parts =',addition+'    parts =',1))
            expected=step_program(program(),self.req,extension=extension_for())['response']['choice']
            value=check_direction_use(source,self.req,expected)
            self.assertFalse(value['accepted'])

    def test_child_has_no_key_environment_and_timeout_is_not_pass(self):
        expected=step_program(program(),self.req,extension=extension_for())['response']['choice']
        with patch.dict(os.environ,OPENAI_API_KEY='UNIT_DO_NOT_FORWARD'),patch('subprocess.run',side_effect=subprocess.TimeoutExpired('unit',8)) as run:
            value=check_direction_use(program(),self.req,expected)
        self.assertIn('python_direction_diagnostic_timeout',value['errors'])
        self.assertNotIn('OPENAI_API_KEY',run.call_args.kwargs['env'])
        self.assertEqual(run.call_args.args[0][1:4],['-I','-S','-B'])

    def test_repeated_audit_is_deterministic(self):
        self.assertEqual(self.check(program()),self.check(program()))

    def test_generation_prompt_has_executable_contract_not_fake_verdicts(self):
        s=LessonDiscoveryService(config(),self.root/'preview.sqlite');self.addCleanup(s.close)
        self.p['previous_program']=json.loads((OLD/'provider_proposal.json').read_text())['program']
        body=s.preview(self.p)['body'];sent=json.loads(body['input'][0]['content'])
        self.assertNotIn('previous_program',sent)
        self.assertNotIn('behavior_checks',body['text']['format']['schema']['properties'])
        self.assertIn('select_directions',body['instructions'])
        self.assertIn('at most 32 calls',body['instructions'])
        self.assertEqual(sent['context']['direction_use_contract']['maximum_helper_calls_per_evaluation'],32)
        self.assertEqual(sent['context'][FLAG],3)
        self.assertTrue(sent['context']['direction_use_contract']['counterfactuals_are_not_evidence'])
        self.assertEqual(sent['evidence'],self.p['evidence'])

    def test_direction_rejection_diagnostics_reach_next_revision(self):
        self.p['context']['development_preflight_feedback']=dict(checks=[dict(case_id=self.p['validation_feedback'][0]['case_id'],errors=[],
            direction_grounding=dict(accepted=False,errors=['geometry_direction_dependence_not_demonstrated'],
                geometry_direction_sensitive=False,scope='synthetic_NOT_physics'))])
        s=LessonDiscoveryService(config(),self.root/'next.sqlite');self.addCleanup(s.close)
        sent=json.loads(s.preview(self.p)['body']['input'][0]['content'])
        self.assertFalse(sent['context']['revision_cases'][0]['previous_direction_check']['geometry_direction_sensitive'])

    def test_v2_saved_body_stays_exact(self):
        p=json.loads((OLD/'generation_packet.json').read_text())
        cfg=json.loads((OLD/'config.json').read_text())['api']
        s=LessonDiscoveryService(cfg,self.root/'v2.sqlite');self.addCleanup(s.close)
        self.assertEqual(s.preview(p)['body'],json.loads((OLD/'api_preview.json').read_text())['body'])

if __name__=='__main__':unittest.main()
