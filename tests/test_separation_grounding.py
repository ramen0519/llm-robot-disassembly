"""Hand-authored synthetic policies, saved-output replay. No API or new physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.test_direction_grounding import grounded_packet, SELECTOR, PROPOSE, program
from tests.test_skill_discovery import config
from orchestrator.python_skill_program import build_context, step_program
from orchestrator.lesson_runtime import extension_for
from orchestrator.python_skill_revision import audit_revision, FLAG
from orchestrator.lesson_discovery import LessonDiscoveryService
from orchestrator.python_separation_probe import unmet, semantic_cases, SeparationVM
from orchestrator.python_separation_grounding import check_separation_use

ROOT = Path(__file__).absolute().parents[1]
OLD = ROOT/'runs/direction_grounding_live_20261021/live_cycle/candidate_01'

HELPER = '''def unseparated_parts(ctx, part_id, pose_id):
    last = ctx.get('last')
    if last is None or last['choice']['part_id'] != part_id or last['choice']['pose_id'] != pose_id:
        return None
    measured = last['feedback'].get('measured_endpoint')
    if measured is None or measured.get('scope') != 'selected_trajectory_terminal_state_only':
        return None
    threshold = measured.get('min_sep')
    pairs = measured.get('pairs')
    if threshold is None or threshold < 0 or pairs is None:
        return None
    expected = sorted([p for p in ctx['remaining_parts'] if p != part_id])
    seen = []
    result = []
    for pair in pairs:
        other = pair.get('rest_part_id')
        distance = pair.get('min_distance')
        if pair.get('moving_part_id') != part_id or other not in expected or other in seen or distance is None:
            return None
        flag = pair.get('separation_unsatisfied')
        if flag is None or flag != (distance < threshold):
            return None
        seen.append(other)
        if distance < threshold:
            result.append(other)
    if sorted(seen) != expected:
        return None
    return sorted(result)
'''


def policy(helper=HELPER, use=True):
    selector = SELECTOR.replace("    parts = ctx['geometry']['parts']",
        "    unmet = unseparated_parts(ctx, part_id, pose_id)\n    if not unmet:\n        return None\n    parts = ctx['geometry']['parts']")
    if use:
        selector = selector.replace("for other in ctx['remaining_parts']:", 'for other in unmet:')
    # Signed projection gives a deterministic test fixture, not a host-provided
    # successful physics policy. It is never registered or sent as a solution.
    selector = selector.replace('max(0.0, sum([world[i] * vector[i] for i in range(3)]))',
                                'sum([world[i] * vector[i] for i in range(3)])')
    propose = PROPOSE.replace("part = ctx['candidate_ids'][-1]", "part = ctx['last']['choice']['part_id']")
    return program(helper+'\n'+selector, propose)


def packet():
    p = grounded_packet()
    p['context'][FLAG] = 4
    req = p['validation_feedback'][0]['current_observation_request']
    for name, center in [('a',[0,0,0]), ('b',[3,0,0]), ('base',[0,3,0])]:
        req['geometry']['parts'][name].update(bounds_center=center, center_mass=center,
            bounds=[[v-.5 for v in center],[v+.5 for v in center]])
    req['episode_feedback'][-1]['measured_endpoint'] = dict(
        scope='selected_trajectory_terminal_state_only', min_sep=0.5,
        pairs=[dict(moving_part_id='a', rest_part_id=x, min_distance=d,
                    separation_unsatisfied=d<.5) for x,d in [('b',5.0),('base',.25)]],
        blocker=None, all_paths_impossible=False, termination_reason='unknown', evidence_sha256='f'*64)
    return p


class SeparationGroundingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='separation_UNIT_',dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.p = packet()
        self.req = self.p['validation_feedback'][0]['current_observation_request']
        self.ctx,_ = build_context(self.req, extension_for())
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.net=net.start(); self.addCleanup(net.stop); self.addCleanup(self.net.assert_not_called)

    def check(self, p):
        result = step_program(p,self.req,extension=extension_for(),diagnose=True)
        self.assertIsNotNone(result['response'],result)
        return check_separation_use(p,self.req,result['response']['choice'])

    def test_end_to_end_preflight_binds_predicate_to_real_direction(self):
        before = deepcopy(self.p)
        result = audit_revision(policy(),self.p,None)
        self.assertTrue(result['accepted'],result)
        audit=result['checks'][0]['separation_grounding']
        self.assertTrue(audit['actual_helper_bound_to_direction'])
        self.assertTrue(audit['endpoint_pair_direction_sensitive'])
        self.assertTrue(audit['helper_result_direction_sensitive'])
        self.assertLessEqual(audit['probe_count'],32)
        self.assertEqual((result['api_calls'],result['physical_calls']),(0,0))
        self.assertFalse(result['new_physical_success']); self.assertEqual(self.p,before)

    def test_numeric_scope_semantics_positive_distance_boundary_and_threshold(self):
        cases=list(semantic_cases(self.ctx,'a','pose_current'))
        found={}
        for label,ctx,part,pose in cases:
            vm=SeparationVM(policy()['source'],ctx)
            actual=vm.call('unseparated_parts',[ctx,part,pose])
            self.assertEqual(actual,unmet(ctx,part,pose),label)
            found[label]=actual
        self.assertEqual(found['one_small_positive_among_far'],['b'])
        self.assertEqual(found['same_distances_lower_threshold'],[])
        self.assertEqual(found['at_boundary'],[])
        self.assertEqual(found['below_boundary'],['b','base'])
        for name in ('missing_endpoint','missing_pair','duplicate_pair','wrong_mover',
                     'wrong_scope','different_selected_mover','different_selected_pose','inconsistent_flag'):
            self.assertIsNone(found[name],name)

    def test_threshold_sign_and_sum_shortcuts_fail(self):
        for bad in (
            HELPER.replace('if distance < threshold:\n            result.append(other)',
                           'if distance < 0:\n            result.append(other)'),
            HELPER.replace('if distance < threshold:\n            result.append(other)',
                           "if sum([p['min_distance'] for p in pairs]) < threshold:\n            result.append(other)"),
            HELPER.replace('if distance < threshold:\n            result.append(other)',
                           'if distance <= threshold:\n            result.append(other)')):
            # Call diagnostic directly: bad policies may abstain on the real input.
            choice=step_program(policy(),self.req,extension=extension_for())['response']['choice']
            result=check_separation_use(policy(bad),self.req,choice)
            self.assertIn('per_pair_separation_semantics_incorrect',result['errors'])

    def test_correct_but_dead_helper_cannot_pass(self):
        result=self.check(policy(use=False))
        self.assertIn('endpoint_pair_direction_dependence_not_demonstrated',result['errors'])
        self.assertIn('helper_result_direction_dependence_not_demonstrated',result['errors'])

    def test_correct_uncalled_helper_cannot_pass(self):
        p=policy()
        p['source']=p['source'].replace('unmet = unseparated_parts(ctx, part_id, pose_id)',"unmet = [p for p in ctx['remaining_parts'] if p != part_id]")
        self.assertIn('separation_helper_not_bound_to_actual_direction',self.check(p)['errors'])

    def test_helper_call_on_wrong_context_cannot_pass(self):
        p=policy()
        # Copy containing an extra diagnostic key is not the exact runtime ctx.
        p['source']=p['source'].replace('unmet = unseparated_parts(ctx, part_id, pose_id)',
            "altered = dict(last=ctx['last'], remaining_parts=ctx['remaining_parts'])\n    unmet = unseparated_parts(altered, part_id, pose_id)")
        self.assertIn('separation_helper_not_bound_to_actual_direction',self.check(p)['errors'])

    def test_abstention_only_not_counted_as_directional_use(self):
        p=policy(use=False)
        p['source']=p['source'].replace('    return [best]',"    if len(unmet) != 1 or unmet[0] != 'base':\n        return None\n    return [best]")
        result=self.check(p)
        self.assertFalse(result['endpoint_pair_direction_sensitive'])

    def test_missing_or_wrong_scope_evidence_not_fabricated(self):
        choice=step_program(policy(),self.req,extension=extension_for())['response']['choice']
        for key,value in [('measured_endpoint',None),('remaining_parts',['a','b'])]:
            req=deepcopy(self.req); req['episode_feedback'][-1][key]=value
            result=check_separation_use(policy(),req,choice)
            self.assertIn('scoped_unmet_separation_not_available',result['errors'])

    def test_no_fixed_base_and_rotated_geometry(self):
        from tests.test_skill_program import request,append_observation,choice
        req=request(); req['base_part']=None; req['candidate_ids']=list(req['remaining_parts'])
        req['geometry']=deepcopy(self.req['geometry'])
        req['pose_catalog'][0]['matrix']=[[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]
        append_observation(req,choice(direction_id='yp'))
        req['episode_feedback'][-1]['measured_endpoint']=deepcopy(self.req['episode_feedback'][-1]['measured_endpoint'])
        self.req=req
        value=self.check(policy()); self.assertTrue(value['accepted'],value)
        actual=step_program(policy(),req,extension=extension_for())['response']['choice']
        self.assertEqual(actual['direction_id'],'xp')

    def test_recursion_and_eligibility_queries_fail_closed(self):
        choice=step_program(policy(),self.req,extension=extension_for())['response']['choice']
        for addition in ('    eligible({})\n','    unseparated_parts(ctx, part_id, pose_id)\n'):
            p=policy(HELPER.replace('    last =',addition+'    last =',1))
            result=check_separation_use(p,self.req,choice)
            self.assertFalse(result['accepted'])
            self.assertIn('python_separation_helper_must_be_pure',result['errors'])

    def test_child_timeout_does_not_receive_secret_environment(self):
        choice=step_program(policy(),self.req,extension=extension_for())['response']['choice']
        with patch.dict(os.environ,OPENAI_API_KEY='UNIT_ONLY'),patch('subprocess.run',side_effect=subprocess.TimeoutExpired('unit',8)) as child:
            result=check_separation_use(policy(),self.req,choice)
        self.assertIn('python_separation_diagnostic_timeout',result['errors'])
        self.assertNotIn('OPENAI_API_KEY',child.call_args.kwargs['env'])
        self.assertEqual(child.call_args.args[0][1:4],['-I','-S','-B'])

    def test_real_v3_candidate_rejected_without_changing_saved_result(self):
        p=json.loads((OLD/'generation_packet.json').read_text()); p['context'][FLAG]=4
        generated=json.loads((OLD/'provider_proposal.json').read_text())['program']
        result=audit_revision(generated,p,None)
        self.assertFalse(result['accepted'])
        self.assertIn('separation_helper_contract_missing',result['checks'][0]['separation_grounding']['errors'])

    def test_saved_v3_body_and_preflight_replay_are_exact(self):
        p=json.loads((OLD/'generation_packet.json').read_text())
        cfg=json.loads((OLD/'config.json').read_text())['api']
        service=LessonDiscoveryService(cfg,self.root/'old.sqlite'); self.addCleanup(service.close)
        self.assertEqual(service.preview(p)['body'],json.loads((OLD/'api_preview.json').read_text())['body'])
        generated=json.loads((OLD/'provider_proposal.json').read_text())['program']
        self.assertEqual(audit_revision(generated,p,None),json.loads((OLD/'development_preflight.json').read_text()))

    def test_prompt_is_scoped_no_runtime_field_addition_or_fake_success(self):
        service=LessonDiscoveryService(config(),self.root/'new.sqlite'); self.addCleanup(service.close)
        body=service.preview(self.p)['body']; sent=json.loads(body['input'][0]['content'])
        self.assertNotIn('behavior_checks',body['text']['format']['schema']['properties'])
        self.assertIn('unseparated_parts',body['instructions'])
        self.assertIn('INITIAL assembled CAD',body['instructions'])
        ctx=sent['context']; self.assertEqual(ctx[FLAG],4)
        self.assertEqual(ctx['revision_cases'][0]['original_endpoint_interpretation']['unmet_part_ids'],['base'])
        # Existing generator intentionally excludes raw memory references from
        # examples; the declared evidence/lesson fields carry training context.
        self.assertEqual(ctx['runtime_contract']['examples'][0]['runtime_input_example'],
                         {k:v for k,v in self.ctx.items() if k != 'memory_references'})
        self.assertFalse(ctx['admission_inputs_supplied']); self.assertEqual(sent['evidence'],self.p['evidence'])

    def test_failed_semantics_feed_the_next_candidate(self):
        self.p['context']['development_preflight_feedback']=dict(checks=[dict(case_id=self.p['validation_feedback'][0]['case_id'],errors=[],
            separation_grounding=dict(accepted=False,errors=['per_pair_separation_semantics_incorrect']))])
        service=LessonDiscoveryService(config(),self.root/'next.sqlite'); self.addCleanup(service.close)
        sent=json.loads(service.preview(self.p)['body']['input'][0]['content'])
        self.assertEqual(sent['context']['revision_cases'][0]['previous_separation_check']['errors'],['per_pair_separation_semantics_incorrect'])


if __name__ == '__main__':
    unittest.main()
