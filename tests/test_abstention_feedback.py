"""Bounded VM/input/feedback tests. Saved observations are not new physics."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from orchestrator import python_abstention as observer
from orchestrator.python_skill_program import evaluate, build_context
from orchestrator.lesson_runtime import extension_for
from orchestrator.python_skill_preflight import preflight_rows
from orchestrator.python_skill_revision import audit_revision
from orchestrator.lesson_discovery import LessonDiscoveryService
from tools.run_python_skill_cycle import refinement_packet, python_generation_packet
from tools.python_revision_parent import historical_packet
from tools.run_luna_experiment import read, sha
from worker.task_memory import digest

ROOT = Path(__file__).absolute().parents[1]
OLD = ROOT / 'runs/stability_transition_learning_20261105'


def policy(source):
    return dict(language='python_policy_v1', source=source)


def declaration(folder):
    r = read(folder / 'result.json')
    return dict(schema_version=1, run_dir=str(folder), result_sha256=sha(folder / 'result.json'),
        manifest_sha256=sha(folder / 'precommitted_cases.json'), candidate_id=r['candidate_id'],
        debug_record_sha256=[], preflight_sha256=r['preflight_sha256'], authorization_inherited=False)


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = read(OLD / 'live/candidate_02/generation_packet.json')
        cls.request = preflight_rows(cls.packet)[0]['current_observation_request']

    def observed(self, source):
        p = policy(source)
        old = evaluate(p, self.request, extension=extension_for(), diagnose=True)
        new = observer.evaluate_observed(p, self.request, extension_for())
        self.assertEqual(new['vm_steps'], old['steps'])
        self.assertEqual(new['runtime_error'], old['diagnostic_runtime_error'])
        self.assertEqual(new['abstained'], old['choice'] is None and old['diagnostic_runtime_error'] is None)
        self.assertEqual(new['input_audit'], old['input_audit'])
        return new

    def test_both_actual_candidates_have_exact_4_vs_2_condition(self):
        for name, line in (('candidate_01',44), ('candidate_02',42)):
            folder = OLD / 'live' / name
            p = read(folder / 'provider_proposal.json')['program']
            packet = read(folder / 'generation_packet.json')
            request = preflight_rows(packet)[0]['current_observation_request']
            before = deepcopy((p, request))
            got = observer.trace_abstention(p, request)
            self.assertEqual((p,request), before)
            self.assertTrue(got['abstained'])
            self.assertEqual(got['return_site']['line'], line)
            guard = got['return_site']['conditions'][-1]
            self.assertIs(guard['condition_truth'], True)
            compare = next(c for c in guard['comparisons'] if ' > ' in c['expression'])
            self.assertEqual([o['value'] for o in compare['operands']], [4,2])
            self.assertTrue(compare['result'])
            self.assertFalse(got['physical_failure_observed'])
            self.assertEqual((got['api_calls'],got['physical_calls']), (0,0))

    def test_side_effecting_condition_is_never_evaluated_twice(self):
        got = self.observed('def mark(xs):\n    xs.append(1)\n    return len(xs)\n'
            'def propose(ctx):\n    xs = []\n    if mark(xs) > 0 and len(xs) == 1:\n        return None\n    return {}\n')
        comparisons = got['return_site']['conditions'][0]['comparisons']
        self.assertEqual([c['result'] for c in comparisons], [True,True])
        self.assertEqual(comparisons[-1]['operands'][0]['value'],1)

    def test_short_circuit_operand_is_not_invented_or_read(self):
        got = self.observed('def propose(ctx):\n    if 0 > 1 > ctx["MUST_NOT_READ"]:\n        return {}\n    return None\n')
        self.assertEqual(got['input_audit']['findings'], [])
        comparison = got['return_site']['recent_conditions'][0]['comparisons'][0]
        self.assertFalse(comparison['operands'][-1]['evaluated'])
        self.assertNotIn('value',comparison['operands'][-1])

    def test_nested_guard_chain_and_else_branch(self):
        got = self.observed('def propose(ctx):\n    if ctx["max_grippers"] > 1:\n        if ctx["max_grippers"] == 0:\n            return {}\n        else:\n            return None\n')
        self.assertEqual([c['branch'] for c in got['return_site']['conditions']],['then','else'])
        self.assertEqual([c['condition_truth'] for c in got['return_site']['conditions']],[True,False])

    def test_implicit_and_unconditional_returns_not_invented_conditions(self):
        implicit = self.observed('def propose(ctx):\n    value = 1\n')
        self.assertTrue(implicit['implicit_return'])
        self.assertIsNone(implicit['return_site'])
        explicit = self.observed('def propose(ctx):\n    return None\n')
        self.assertFalse(explicit['implicit_return'])
        self.assertEqual(explicit['return_site']['conditions'], [])

    def test_large_operand_is_bounded_and_truncation_explicit(self):
        got = self.observed('def propose(ctx):\n    xs = list(range(2000))\n    if xs == xs:\n        return None\n')
        operand = got['return_site']['conditions'][0]['comparisons'][0]['operands'][0]
        self.assertEqual(operand['length'],2000)
        self.assertEqual(len(operand['value']),8)
        self.assertTrue(operand['truncated'])
        self.assertLess(len(json.dumps(got)),32768)

    def test_errors_and_step_exhaustion_are_not_normal_abstention(self):
        for source in ('def propose(ctx):\n    value = 1 / 0\n',
                'def propose(ctx):\n    for i in range(4000):\n        for j in range(4000):\n            value = i + j\n'):
            got = self.observed(source)
            self.assertFalse(got['abstained'])
            self.assertIsNotNone(got['runtime_error'])
            self.assertFalse(got['implicit_return'])

    def test_invalid_source_rejected_before_child_and_timeout_not_fake_success(self):
        with patch('subprocess.run',side_effect=AssertionError('no child')):
            with self.assertRaises(ValueError):
                observer.trace_abstention(policy('import os\ndef propose(ctx):\n    return None\n'),self.request)
        with patch('subprocess.run',side_effect=subprocess.TimeoutExpired('test',5)):
            with self.assertRaises(subprocess.TimeoutExpired):
                observer.trace_abstention(policy('def propose(ctx):\n    return None\n'),self.request)

    def test_child_environment_excludes_keys_and_source_is_data(self):
        original = subprocess.run
        with patch.dict('os.environ',OPENAI_API_KEY='TEST_ONLY_DO_NOT_FORWARD'), \
                patch('subprocess.run', wraps=original) as call:
            got = observer.trace_abstention(policy('def propose(ctx):\n    return None\n'),self.request)
        self.assertTrue(got['abstained'])
        self.assertNotIn('OPENAI_API_KEY',call.call_args.kwargs['env'])
        self.assertEqual(call.call_args.args[0][1:4],['-I','-S','-B'])


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = OLD / 'live/candidate_02'
        cls.manifest = read(cls.folder/'precommitted_cases.json')
        base = python_generation_packet(cls.manifest,'offline-abstention-test')
        cls.refined = refinement_packet(base,cls.folder,cls.manifest)
        cls.enriched = observer.attach_feedback(cls.refined)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='abstention_TEST_',dir=ROOT/'runs')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def body(self, packet):
        packet = deepcopy(packet)
        packet['context'].update(lesson_stage='program',python_revision_contract_version=2,
            lesson=read(OLD/'live/lesson_initial/lesson.json'),python_output_tokens=4096)
        cfg = read(OLD/'live/learner_config.json')
        service = LessonDiscoveryService(cfg,self.root/'preview.sqlite',lambda *a: self.fail('No API'))
        self.addCleanup(service.close)
        return service.preview(packet)['body']

    def test_verified_abstention_values_reach_actual_provider_body(self):
        body = self.body(self.enriched)
        sent = json.loads(body['input'][0]['content'])
        case = sent['context']['revision_cases'][0]
        self.assertEqual(case['revision_stage'],'preflight_abstention_without_physical_attempt')
        self.assertFalse(case['measured_failed_previous_repair'])
        self.assertIsNone(case['previous_repair'])
        diag = case['previous_preflight']['abstention_diagnostic']
        self.assertEqual(diag['return_site']['line'],42)
        self.assertIn('4',json.dumps(diag))
        self.assertIn('NOT a failed physical repair',body['instructions'])
        self.assertFalse(sent['context']['admission_inputs_supplied'])
        self.assertNotIn('previous_program',sent)

    def test_historical_saved_audits_and_bodies_are_unchanged_without_opt_in(self):
        for name in ('candidate_01','candidate_02'):
            folder = OLD/'live'/name
            packet = read(folder/'generation_packet.json')
            program = read(folder/'provider_proposal.json')['program']
            self.assertEqual(audit_revision(program,packet,None),read(folder/'development_preflight.json'))
            service = LessonDiscoveryService(read(OLD/'live/learner_config.json'),self.root/(name+'.sqlite'))
            try:self.assertEqual(service.preview(packet)['body'],read(folder/'api_preview.json')['body'])
            finally:service.close()

    def test_old_rejection_is_read_without_inheriting_paid_authority(self):
        before = sha(OLD/'live/learner.sqlite')
        base = python_generation_packet(self.manifest,'offline-parent-test')
        value = historical_packet(base,declaration(self.folder),self.manifest)
        self.assertEqual(value['context']['historical_debug_parent']['kind'],'preflight_abstention_not_physical_failure')
        self.assertFalse(value['context']['historical_debug_parent']['authorization_inherited'])
        self.assertEqual(before,sha(OLD/'live/learner.sqlite'))

    def test_tampered_historical_preflight_hash_or_authority_rejected(self):
        for field,value in (('preflight_sha256','0'*64),('authorization_inherited',True),
                            ('debug_record_sha256',['fabricated'])):
            d=declaration(self.folder);d[field]=value
            with self.assertRaises(ValueError):historical_packet(deepcopy(self.refined),d,self.manifest)

    def test_changed_program_request_or_diagnostic_rejected(self):
        for mutate in ('program','request','diagnostic','phase','case'):
            p=deepcopy(self.enriched)
            if mutate=='program':p['previous_program']['source']+='\n'
            elif mutate=='request':p['validation_feedback'][0]['current_observation_request']['max_grippers']=4
            elif mutate=='diagnostic':p['context'][observer.FLAG]['rows'][0]['diagnostic']['return_site']['line']=1
            elif mutate=='phase':p['context'][observer.FLAG]['rows'][0]['phase']='admission'
            else:p['context'][observer.FLAG]['rows'][0]['case_id']='not_a_debug_case'
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.body(p)

    def test_stale_preflight_binding_rejected_before_diagnostic_child(self):
        p=deepcopy(self.refined)
        p['context']['development_preflight_feedback']['checks'][0]['request_sha256']='0'*64
        with patch.object(observer,'trace_abstention',side_effect=AssertionError('No trace')):
            with self.assertRaisesRegex(ValueError,'binding_mismatch'):observer.attach_feedback(p)

    def test_no_abstention_and_missing_parent_fail_closed(self):
        p=deepcopy(self.refined);p.pop('previous_program')
        with self.assertRaisesRegex(ValueError,'previous_program'):observer.attach_feedback(p)
        p=deepcopy(self.refined)
        p['context']['development_preflight_feedback']['checks'][0]['runtime_reason']='python_execution_rejected'
        with self.assertRaisesRegex(ValueError,'no_bound'):observer.attach_feedback(p)


if __name__=='__main__':
    unittest.main()
