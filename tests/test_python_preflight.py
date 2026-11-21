"""Synthetic/offline contracts; fake backend labels are NOT real physics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_skill_program import request, append_observation, choice
from tests.test_skill_discovery import synthesis, config
from tests.test_lesson_pipeline import claim
from tests.test_skill_validation import save
from orchestrator.lesson_memory import make_lesson
from orchestrator.lesson_runtime import extension_for
from orchestrator.python_skill_program import step_program, build_context
from orchestrator.python_skill_preflight import audit_program, generation_contract
from orchestrator.lesson_discovery import LessonDiscoveryService
from worker.explicit_action_contract import digest
import tools.run_python_skill_cycle as cycle

ROOT = Path(__file__).absolute().parents[1]


def policy(prefix='', direction='xn', supports='[]', pose='ctx["last"]["choice"]["pose_id"]', citations='[ctx["last"]["feedback"]["evidence_id"]]'):
    return dict(language='python_policy_v1', source='def propose(ctx):\n' + prefix +
        '    return {"part_id": ctx["last"]["choice"]["part_id"], "pose_id": '+pose+',\n'
        '        "skill_id": "single_force_v1", "direction_id": "'+direction+'",\n'
        '        "second_direction_id": None, "support_ids": '+supports+', "evidence_ids": '+citations+'}\n')


def row(req=None, outcome='UNKNOWN'):
    req = req or append_observation(request())
    return dict(case_id='synthetic_debug', phase='debug', outcome=outcome,
        current_observation_request=req, validation_record_sha256='synthetic_not_evidence',
        recovery=dict(backend='ASAP_RedMax', chosen=choice(direction_id='xn'),
            path_success=False, stability_success=None))


class PythonPreflightTests(unittest.TestCase):
    def setUp(self):
        self.req=append_observation(request())
        net=patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.net=net.start(); self.addCleanup(net.stop); self.addCleanup(self.net.assert_not_called)

    def diag(self, program):
        return step_program(program,self.req,extension=extension_for(),diagnose=True)

    def test_missing_get_is_reported_even_with_silent_default(self):
        p=policy('    shape = ctx.get("geometry")\n    supports = shape.get("support_ids", [])\n')
        got=self.diag(p)
        self.assertIsNotNone(got['response']) # legal action is not correct input use
        self.assertFalse(got['input_audit']['accepted'])
        self.assertEqual(got['input_audit']['findings'][0]['path'],'ctx.geometry')
        self.assertEqual(got['input_audit']['findings'][0]['key'],'support_ids')
        self.assertFalse(audit_program(p,[row(self.req)])['accepted'])

    def test_subscript_exception_preserves_diagnostic_and_no_response(self):
        got=self.diag(policy('    invalid = ctx["geometry"]["support_ids"]\n'))
        self.assertIsNone(got['response'])
        self.assertEqual(got['diagnostic_runtime_error'],'KeyError')
        self.assertFalse(got['input_audit']['accepted'])

    def test_alias_copy_and_helper_cannot_hide_typo(self):
        p=policy('    data = dict(ctx["geometry"])\n    bad = helper(data)\n')
        p['source']+='\ndef helper(data):\n    return data.get("wrong_field")\n'
        self.assertFalse(self.diag(p)['input_audit']['accepted'])

    def test_optional_dynamic_and_local_get_are_not_false_errors(self):
        p=policy('    missing = ctx["geometry"]["parts"].get("not_here")\n'
                 '    unknown = ctx["geometry"]["parts"]["a"].get("center_mass")\n'
                 '    optional = ctx["last"]["feedback"].get("stability_diagnostics")\n'
                 '    local = {"support_ids": []}\n    own_default = local.get("other")\n')
        got=self.diag(p)
        self.assertTrue(got['input_audit']['accepted'])
        self.assertEqual(len(got['input_audit']['findings']),3)

    def test_nested_part_typo_and_top_level_typo(self):
        for prefix in ('    x = ctx.get("evidence_ids")\n',
                       '    x = ctx["geometry"]["parts"]["a"].get("centre")\n',
                       '    x = ctx["last"]["feedback"].get("made_up_blocker")\n'):
            self.assertFalse(self.diag(policy(prefix))['input_audit']['accepted'])

    def test_same_action_renamed_code_and_changed_citations_rejected(self):
        for p in (policy(), policy('    renamed = 123\n',citations='[]')):
            result=audit_program(p,[row(self.req)])
            self.assertFalse(result['accepted'])
            self.assertIn('unchanged_failed_debug_action', result['checks'][0]['errors'])

    def test_pose_alias_is_same_physical_action(self):
        self.req['pose_catalog'].append(dict(pose_id='alias',matrix=deepcopy(self.req['pose_catalog'][0]['matrix'])))
        got=audit_program(policy(pose='"alias"'),[row(self.req)])
        self.assertTrue(got['checks'][0]['repeated_failed_action'])

    def test_support_order_is_same_but_changed_support_is_distinct(self):
        self.req=request()
        self.req['remaining_parts'].append('c'); self.req['candidate_ids'].append('c');self.req['max_grippers']=3
        append_observation(self.req)
        r=row(self.req);r['recovery']['chosen']['support_ids']=['b','c']
        got=audit_program(policy(supports='["c", "b"]'),[r])
        self.assertFalse(got['accepted'])
        got=audit_program(policy(supports='["b"]'),[r])
        self.assertTrue(got['accepted'])
        self.assertFalse(got['new_physical_success'])

    def test_new_direction_is_only_preflight_pass_not_physics(self):
        got=audit_program(policy(direction='yp'),[row(self.req)])
        self.assertTrue(got['accepted'])
        self.assertEqual(got['physical_calls'],0)
        self.assertFalse(got['new_physical_success'])
        self.assertFalse(got['candidate_pruning'])

    def test_unknown_without_measured_failure_pass_and_late_not_repeat_errors(self):
        for outcome in ('PASS','LATE','ERROR'):
            self.assertTrue(audit_program(policy(),[row(self.req,outcome)])['accepted'])
        r=row(self.req);r['recovery']['path_success']=None
        self.assertTrue(audit_program(policy(),[r])['accepted'])

    def test_abstain_is_not_success_and_error_is_safe(self):
        for source in ('def propose(ctx):\n    return None\n',
                       'def propose(ctx):\n    return 1 / 0\n'):
            got=audit_program(dict(language='python_policy_v1',source=source),[row(self.req)])
            self.assertFalse(got['accepted'])
            self.assertIn('no_valid_debug_repair',got['checks'][0]['errors'])

    def test_old_runtime_results_and_context_unchanged(self):
        p=policy();before=deepcopy(self.req)
        ordinary=step_program(p,self.req,extension=extension_for())
        traced=self.diag(p)
        self.assertEqual(ordinary['response'],traced['response'])
        self.assertNotIn('input_audit',ordinary)
        self.assertEqual(self.req,before)

    def test_prompt_has_actual_nested_contract_and_diagnostics_no_admission(self):
        packet=synthesis();lesson=make_lesson(claim(packet),packet,dict(origin='manual_authored'))
        packet['context'].update(lesson_stage='program',lesson=lesson)
        packet.update(operation='refine',previous_program=policy())
        packet['validation_feedback']=[row(self.req)]
        packet['context']['development_preflight_feedback']=audit_program(policy(),[row(self.req)])
        contract=generation_contract(packet)
        ctx=contract['examples'][0]['runtime_input_example']
        self.assertEqual(ctx['geometry'],build_context(self.req,extension_for())[0]['geometry'])
        with tempfile.TemporaryDirectory() as tmp:
            service=LessonDiscoveryService(config(enabled=False,max_calls=0,max_usd=0),Path(tmp)/'preview.sqlite')
            try: body=service.preview(packet)['body']
            finally: service.close()
        sent=body['input'][0]['content']
        self.assertIn('unknown',sent)
        self.assertIn('unchanged_failed_debug_action',sent)
        self.assertIn('geometry.support_ids',sent)
        bad=deepcopy(packet);bad['validation_feedback'][0]['phase']='admission'
        with self.assertRaisesRegex(ValueError,'debug'):
            generation_contract(bad)

    def test_cycle_rejects_before_physics_and_preserves_refinement_feedback(self):
        with tempfile.TemporaryDirectory(prefix='preflight_FIXTURE_',dir=ROOT/'runs') as tmp:
            root=Path(tmp);out=root/'candidate'
            packet=synthesis();lesson=make_lesson(claim(packet),packet,dict(origin='manual_authored'))
            save(root/'lesson.json',lesson)
            manifest=dict(cases=[dict(case_id=k,phase=p,source_task='fixture',trigger_kind='path_failed')
                for k,p in [('synthetic_debug','debug'),('hidden_1','admission'),('hidden_2','admission')]])
            save(root/'manifest.json',manifest)
            def prepared(spec):
                return dict(spec=spec,request=self.req,raw=dict(input_hashes={}),expected_source_hashes={})
            def attach(p,*args):
                p['context']['python_debug_requests']=[row(self.req)];return p
            bad=policy('    x = ctx["geometry"].get("support_ids")\n')
            with patch.object(cycle,'source_case',side_effect=prepared), \
                 patch.object(cycle,'python_generation_packet',return_value=packet), \
                 patch.object(cycle,'attach_debug_inputs',side_effect=attach), \
                 patch.object(cycle,'PROGRAM',bad), \
                 patch.object(cycle,'run_case',side_effect=AssertionError('NO_PHYSICS')) as physics:
                code=cycle.main(['--run-dir',str(out),'--manifest',str(root/'manifest.json'),
                    '--lesson',str(root/'lesson.json'),'--mode','fixture'])
            self.assertEqual(code,1);physics.assert_not_called()
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'],'PREFLIGHT_REJECTED')
            self.assertFalse(result['registered'])
            self.assertTrue((out/'candidate.py').is_file())
            refined=cycle.refinement_packet(synthesis(),out,manifest)
            self.assertIn('development_preflight_feedback',refined['context'])
            self.assertNotIn('hidden_1',json.dumps(refined))
            audit=json.loads((out/'development_preflight.json').read_text());audit['accepted']=True
            save(out/'development_preflight.json',audit)
            with self.assertRaisesRegex(ValueError,'binding_mismatch'):
                cycle.refinement_packet(synthesis(),out,manifest)


if __name__=='__main__':
    unittest.main()
