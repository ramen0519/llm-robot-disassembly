"""Offline routing and applicability tests; fixtures are NOT model/physics success."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_hierarchical_planner import policy
from tests.test_python_preflight import policy as program, row
from tests.test_skill_program import request, append_observation
from tests.test_skill_validation import save
from tests.test_skill_discovery import synthesis, config
from tests.test_lesson_pipeline import claim
from orchestrator.lesson_memory import make_lesson
from orchestrator.lesson_discovery import LessonDiscoveryService
from orchestrator.python_skill_preflight import preflight_rows, audit_program, generation_contract
from tools import run_hierarchical_cycle as runner
from tools import run_python_skill_cycle as python_cycle

ROOT = runner.ROOT


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='continuation_UNIT_', dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.ledger = self.base/'learner.sqlite'
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.network = net.start(); self.addCleanup(net.stop)
        self.addCleanup(self.network.assert_not_called)

    def reject(self, folder, reason='invalid_generated_python', live=True, ledger=None):
        save(folder/'config.json', dict(mode='live', ledger=str(ledger or self.ledger)))
        save(folder/'provider_proposal.json', dict(status='rejected', fallback_reason=reason,
            metadata=dict(llm_live=live, python_validation_error='python_keyword_arguments_unsupported')))
        save(folder/'result.json', dict(status='GENERATION_REJECTED', physical_cases_not_run=True))

    def test_only_completed_contract_rejection_routes_to_generation_feedback(self):
        self.reject(self.base/'c1')
        got=runner.candidate_followup_args(self.base/'c1',dict(status='GENERATION_REJECTED'),
                                         live=True,ledger=self.ledger)
        self.assertEqual(got,['--generation-feedback-from',self.base/'c1'])
        feedback=python_cycle.contract_rejection_feedback(self.base/'c1',self.ledger)
        self.assertEqual(feedback['python_validation_error'],'python_keyword_arguments_unsupported')
        self.assertNotIn('program',feedback)

    def test_network_budget_timeout_and_unbound_failure_do_not_retry(self):
        for i,reason in enumerate(('http_429','api_error','budget_exhausted','timeout','invalid_response')):
            folder=self.base/str(i)
            self.reject(folder,reason,live=reason!='invalid_response')
            self.assertEqual(runner.candidate_followup_args(folder,dict(status='GENERATION_REJECTED'),
                live=True,ledger=self.ledger),[])
        self.reject(self.base/'wrong_ledger',ledger=self.base/'other.sqlite')
        self.assertEqual(runner.candidate_followup_args(self.base/'wrong_ledger',
            dict(status='GENERATION_REJECTED'),live=True,ledger=self.ledger),[])

    def test_missing_artifacts_and_offline_do_not_spawn_candidates(self):
        for live in (True,False):
            self.assertEqual(runner.candidate_followup_args(self.base/'absent',
                dict(status='GENERATION_REJECTED'),live=live,ledger=self.ledger),[])

    def test_preflight_and_physics_feedback_use_refine_not_generation_retry(self):
        for status in ('PREFLIGHT_REJECTED','NOT_REGISTERED'):
            self.assertEqual(runner.candidate_followup_args(self.base,dict(status=status),
                live=True,ledger=self.ledger),['--refine-from',self.base])

    def run_learning_stage(self, *, max_proposals=2, reason='invalid_generated_python', changed=False):
        output=self.base/'cycle'
        plan=json.loads((ROOT/'configs/automatic_learning_cycle_v1.json').read_text(encoding='utf-8'))
        plan.update(initial_library=None,initial_memory=None,max_proposals=max_proposals)
        save(self.base/'policy.json',policy(enabled=True))
        calls=[]
        class LearnOnly:
            def __init__(self,folder,identity,stages):self.stages=stages
            def run(self):
                result=self.stages['learn']({'prepare':dict(status='PREPARED',manifest='UNIT_ONLY')})
                save(output/'result.json',result)
                return result
        def subprocess_fixture(argv,cwd,logs):
            calls.append(argv)
            folder=Path(argv[argv.index('--run-dir')+1])
            self.reject(folder,reason,ledger=output/'learner.sqlite')
            if changed:
                save(folder/'result.json',dict(status='GENERATION_REJECTED',source_unchanged=False))
            return dict(exit_code=1)
        with patch.object(runner,'LearningCycleWorkflow',LearnOnly), \
             patch.object(runner,'run_recorded',side_effect=subprocess_fixture):
            result=runner.execute_cycle(output,plan,self.base/'policy.json',live=True)
        return result,calls

    def test_actual_learning_loop_dispatches_second_candidate_with_shared_budget(self):
        result,calls=self.run_learning_stage()
        self.assertEqual(len(calls),2)
        self.assertNotIn('--generation-feedback-from',calls[0])
        self.assertIn('--generation-feedback-from',calls[1])
        self.assertNotIn('--refine-from',calls[1])
        for flag in ('--ledger','--hierarchy-ledger','--hierarchy-policy','--manifest'):
            self.assertEqual(calls[0][calls[0].index(flag)+1],calls[1][calls[1].index(flag)+1])
        self.assertEqual(result['status'],'NO_NEW_SKILL')
        self.assertEqual(result['learning_error'],'candidate_limit_exhausted:GENERATION_REJECTED')
        self.assertEqual(result['new_ids'],[])

    def test_candidate_limit_does_not_expand(self):
        result,calls=self.run_learning_stage(max_proposals=1)
        self.assertEqual(len(calls),1)
        self.assertEqual(len(result['candidates']),1)

    def test_api_error_stops_actual_loop(self):
        result,calls=self.run_learning_stage(reason='http_429')
        self.assertEqual(len(calls),1)
        self.assertEqual(result['learning_error'],'learning_stopped:GENERATION_REJECTED')

    def test_changed_source_stops_actual_loop(self):
        with self.assertRaisesRegex(ValueError,'learning_source_changed'):
            self.run_learning_stage(changed=True)


class ApplicabilityTests(unittest.TestCase):
    def setUp(self):
        self.req=request()
        self.req['base_part']=None
        self.req['candidate_ids']=list(self.req['remaining_parts'])
        append_observation(self.req)
        self.packet=synthesis()
        self.packet['context']['python_debug_requests']=[row()]
        target=row(self.req);target.update(case_id='no_fixed_base',
            purpose='development_applicability_NOT_evaluation',recovery=None)
        self.packet['context']['python_applicability_requests']=[target]
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.network=net.start();self.addCleanup(net.stop);self.addCleanup(self.network.assert_not_called)

    def test_base_required_program_is_not_mistaken_for_applicable_skill(self):
        p=program('    if ctx.get("base_part") is None:\n        return None\n',direction='yp')
        out=audit_program(p,preflight_rows(self.packet))
        self.assertTrue(out['checks'][0]['accepted'])
        self.assertFalse(out['checks'][1]['accepted'])
        self.assertIn('no_valid_debug_repair',out['checks'][1]['errors'])
        self.assertFalse(out['accepted']);self.assertEqual(out['physical_calls'],0)

    def test_base_independent_fixture_emits_action_without_faking_physics(self):
        before=deepcopy(self.req)
        out=audit_program(program(direction='yp'),preflight_rows(self.packet))
        self.assertTrue(out['accepted'])
        self.assertFalse(out['new_physical_success'])
        self.assertEqual(out['physical_calls'],0)
        self.assertEqual(self.req,before)

    def test_observation_cause_is_not_runtime_failure_field(self):
        p=program('    x = ctx["failure"].get("failure_cause")\n',direction='yp')
        out=audit_program(p,preflight_rows(self.packet))
        self.assertFalse(out['accepted'])
        self.assertTrue(all('unknown_input_field' in c['errors'] for c in out['checks']))

    def test_prompt_uses_explicit_null_base_and_no_target_outcomes(self):
        lesson=make_lesson(claim(self.packet),self.packet,dict(origin='manual_authored'))
        self.packet['context'].update(lesson_stage='program',lesson=lesson)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            service=LessonDiscoveryService(config(enabled=False,max_calls=0,max_usd=0),Path(tmp)/'preview.sqlite')
            try: sent=service.preview(self.packet)['body']
            finally:service.close()
        projected=json.loads(sent['input'][0]['content'])
        self.assertNotIn('python_applicability_requests',projected['context'])
        contract=projected['context']['python_runtime_contract']
        self.assertIsNone(contract['examples'][1]['runtime_input_example']['base_part'])
        self.assertFalse(contract['admission_inputs_supplied'])
        self.assertIn('failure_cause',contract['description'])

    def test_admission_duplicate_and_oversized_probe_sets_rejected(self):
        for alteration in ('admission','duplicate','oversized'):
            p=deepcopy(self.packet)
            probe=p['context']['python_applicability_requests'][0]
            if alteration=='admission':probe['phase']='admission'
            elif alteration=='duplicate':probe['case_id']='synthetic_debug'
            else:p['context']['python_applicability_requests']*=5
            with self.assertRaises(ValueError):preflight_rows(p)

    def test_required_abstention_is_not_physics_failure_or_candidate_pruning(self):
        p=dict(language='python_policy_v1',source='def propose(ctx):\n    return None\n')
        got=audit_program(p,preflight_rows(self.packet))
        self.assertFalse(got['accepted']);self.assertFalse(got['candidate_pruning'])
        self.assertEqual(got['api_calls'],0)


class SavedApplicabilityBindingTests(unittest.TestCase):
    """The new declaration boundary must fail before model or physics access."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='applicability_UNIT_',dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        save(self.base/'saved.json',{'synthetic':True})
        self.spec=dict(case_id='no_base',phase='debug',source_task='synthetic',trigger_kind='path_failed',
            report=(self.base/'saved.json').relative_to(ROOT).as_posix(),attempt_index=1,
            report_sha256=python_cycle.sha(self.base/'saved.json'),
            purpose='development_applicability_NOT_evaluation')
        self.manifest=dict(cases=[dict(phase='admission',report='hidden',attempt_index=1)],
                           python_applicability_cases=[self.spec])
        self.packet=synthesis();self.packet['context']['python_debug_requests']=[row()]

    def test_modified_report_rejected_before_loading_case(self):
        self.spec['report_sha256']='0'*64
        with patch.object(python_cycle,'source_case',side_effect=AssertionError('NO_CASE')) as case:
            with self.assertRaisesRegex(ValueError,'hash_mismatch'):
                python_cycle.attach_applicability_inputs(self.packet,self.manifest,'test')
        case.assert_not_called()

    def test_admission_disguised_as_development_and_overrides_rejected(self):
        for change in ('admission','override','same_admission'):
            m=deepcopy(self.manifest)
            if change=='admission':m['python_applicability_cases'][0]['phase']='admission'
            elif change=='override':m['python_applicability_cases'][0]['trigger_override']={}
            else:m['cases'][0]['report']=self.spec['report']
            with self.assertRaisesRegex(ValueError,'not_admission'):
                python_cycle.attach_applicability_inputs(self.packet,m,'test')

    def test_pass_timeout_and_unknown_without_measured_failure_rejected(self):
        for raw in (dict(status='PASS',path_success=True,stability_success=True),
                    dict(status='UNKNOWN',path_success=False,deadline_exceeded=True),
                    dict(status='UNKNOWN',path_success=None,stability_success=None)):
            with patch.object(python_cycle,'source_case',return_value=dict(raw=raw)):
                with self.assertRaisesRegex(ValueError,'observed_failure'):
                    python_cycle.attach_applicability_inputs(self.packet,self.manifest,'test')

    def test_saved_input_binding_does_not_make_new_physical_evidence(self):
        req=request();req['base_part']=None;req['candidate_ids']=list(req['remaining_parts'])
        append_observation(req)
        raw=dict(status='UNKNOWN',path_success=False,stability_success=None)
        with patch.object(python_cycle,'source_case',return_value=dict(raw=raw,request=req,choice={})), \
             patch('orchestrator.skill_case.request_after_trigger',return_value=req):
            result=python_cycle.attach_applicability_inputs(self.packet,self.manifest,'test')
        extra=result['context']['python_applicability_requests'][0]
        self.assertEqual(extra['source_report_sha256'],self.spec['report_sha256'])
        self.assertIsNone(extra['current_observation_request']['base_part'])
        self.assertIsNone(extra['recovery'])
        self.assertNotIn('outcome',extra)


if __name__=='__main__':unittest.main()
