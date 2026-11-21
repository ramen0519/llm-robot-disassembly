"""Synthetic transport/VM regression tests; no live API, WSL or new physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from tests.test_python_revision import packet, predictions
from tests.test_python_preflight import policy
from tests.test_skill_discovery import config, decision, raw_response
from orchestrator.lesson_discovery import LessonDiscoveryService
from orchestrator.python_skill_revision import audit_revision, FLAG
from orchestrator.python_execution_revision import execution_view
from orchestrator.python_skill_evidence import generation_view
from tools.remaining_split_budget import remaining_proposal, validate_remaining_binding, claim_remaining_binding
from tools import run_python_skill_cycle as runner
from tools import run_split_revision_cycle as split
from worker.task_memory import digest

ROOT=Path(__file__).absolute().parents[1]
OLD=ROOT/'runs/split_revision_live_20261021/live_cycle'


class ExecutionRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='execution_revision_UNIT_',dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        self.p=packet();self.p['context'][FLAG]=2
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.net=net.start();self.addCleanup(net.stop);self.addCleanup(self.net.assert_not_called)

    def service(self,transport=None):
        s=LessonDiscoveryService(config(),self.root/(str(len(list(self.root.iterdir())))+'.sqlite'),transport)
        self.addCleanup(s.close);return s

    def test_v2_omits_old_source_and_prediction_contract_preserves_all_evidence(self):
        before=deepcopy(self.p)
        body=self.service().preview(self.p)['body'];sent=json.loads(body['input'][0]['content'])
        self.assertNotIn('previous_program',sent)
        self.assertNotIn('behavior_checks',body['text']['format']['schema']['properties'])
        self.assertEqual(sent['evidence'],self.p['evidence'])
        self.assertEqual(sent['context']['previous_program_sha256'],digest(self.p['previous_program']))
        self.assertFalse(sent['context']['previous_source_supplied'])
        self.assertEqual(sent['context']['revision_cases'][0]['revision_requirement'],'change_path_conditions_or_abstain')
        self.assertEqual(self.p,before)

    def test_same_failed_action_still_rejected_without_prediction_fields(self):
        result=audit_revision(policy(),self.p,None)
        self.assertFalse(result['accepted'])
        self.assertIn('unchanged_failed_debug_action',result['checks'][0]['errors'])
        self.assertIn('unchanged_failed_path_conditions',result['checks'][0]['errors'])
        self.assertFalse(result['model_predictions_required'])

    def test_support_only_change_cannot_repair_same_failed_path(self):
        req=self.p['validation_feedback'][0]['current_observation_request']
        mover=req['episode_feedback'][-1]['chosen']['part_id'] if 'chosen' in req['episode_feedback'][-1] else '0'
        support=next(p for p in req['remaining_parts'] if p not in (mover,req['base_part']))
        result=audit_revision(policy(supports=repr([support])),self.p,None)
        self.assertFalse(result['accepted']);self.assertTrue(result['checks'][0]['same_failed_path_conditions'])
        self.assertIn('support_ids',result['checks'][0]['changed_physical_fields'])

    def test_stability_failure_allows_support_only_change(self):
        row=self.p['validation_feedback'][0];row['recovery'].update(path_success=True,stability_success=False)
        req=row['current_observation_request'];mover=row['recovery']['chosen']['part_id']
        support=next(p for p in req['remaining_parts'] if p not in (mover,req['base_part']))
        result=audit_revision(policy(supports=repr([support])),self.p,None)
        self.assertTrue(result['accepted'],result)
        self.assertIsNone(result['checks'][0]['same_failed_path_conditions'])

    def test_actual_path_change_only_passes_preflight_not_physics(self):
        result=audit_revision(policy(direction='yp'),self.p,None)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['checks'][0]['changed_physical_fields'],['direction_id'])
        self.assertFalse(result['new_physical_success']);self.assertEqual(result['physical_calls'],0)

    def test_pose_alias_and_evidence_only_changes_do_not_bypass_path_gate(self):
        req=self.p['validation_feedback'][0]['current_observation_request']
        req['pose_catalog'].append(dict(pose_id='alias',matrix=deepcopy(req['pose_catalog'][0]['matrix'])))
        result=audit_revision(policy(pose='"alias"',citations='[]'),self.p,None)
        self.assertFalse(result['accepted']);self.assertTrue(result['checks'][0]['same_failed_path_conditions'])

    def test_timeout_error_and_pass_are_not_failed_path_evidence(self):
        for outcome in ('PASS','ERROR','LATE'):
            p=deepcopy(self.p);p['validation_feedback'][0]['outcome']=outcome
            self.assertTrue(audit_revision(policy(),p,None)['accepted'])
        p=deepcopy(self.p);p['validation_feedback'][0]['recovery']['deadline_exceeded']=True
        self.assertTrue(audit_revision(policy(),p,None)['accepted'])

    def test_abstain_unknown_input_and_runtime_error_do_not_pass(self):
        for source in ('def propose(ctx):\n    return None\n','def propose(ctx):\n    return 1/0\n'):
            self.assertFalse(audit_revision(dict(language='python_policy_v1',source=source),self.p,None)['accepted'])
        self.assertFalse(audit_revision(policy('    q=ctx["nonexistent"]\n',direction='yp'),self.p,None)['accepted'])

    def test_admission_feedback_rejected_before_transport(self):
        self.p['validation_feedback'][0]['phase']='admission'
        with self.assertRaises(ValueError):self.service().preview(self.p)

    def test_saved_actual_failed_candidate_remains_rejected(self):
        p=json.loads((OLD/'candidate_01/generation_packet.json').read_text());p['context'][FLAG]=2
        candidate=json.loads((OLD/'candidate_01/provider_proposal.json').read_text())['program']
        result=audit_revision(candidate,p,None)
        self.assertFalse(result['accepted']);self.assertTrue(result['checks'][0]['repeated_failed_action'])
        self.assertTrue(result['checks'][0]['same_failed_path_conditions'])

    def test_v1_body_reproduces_saved_request_exactly(self):
        p=json.loads((OLD/'candidate_01/generation_packet.json').read_text())
        cfg=json.loads((OLD/'learner_config.json').read_text())
        s=LessonDiscoveryService(cfg,self.root/'saved.sqlite');self.addCleanup(s.close)
        expected=json.loads((OLD/'candidate_01/api_preview.json').read_text())['body']
        self.assertEqual(s.preview(p)['body'],expected)

    def test_v2_parser_accepts_only_code_contract_not_second_predictions(self):
        val=decision(self.p);val['program']=policy(direction='yp')
        transport=Mock(return_value=raw_response(val))
        with patch.dict(os.environ,DISCOVERY_FIXTURE_KEY='UNIT_NOT_KEY'):
            response=self.service(transport).propose(self.p)
        self.assertEqual(response['status'],'proposed');self.assertNotIn('behavior_checks',response)
        extra=dict(val,behavior_checks=[])
        with patch.dict(os.environ,DISCOVERY_FIXTURE_KEY='UNIT_NOT_KEY'):
            invalid=self.service(Mock(return_value=raw_response(extra))).propose(self.p)
        self.assertEqual(invalid['fallback_reason'],'invalid_response_fields')

    def test_explicit_90_seconds_reaches_transport_without_changing_body(self):
        p=deepcopy(self.p);p['timeout_seconds']=90
        val=decision(p);val['program']=policy(direction='yp')
        transport=Mock(return_value=raw_response(val));s=self.service(transport)
        self.assertEqual(s.preview(p)['body'],s.preview(self.p)['body'])
        with patch.dict(os.environ,DISCOVERY_FIXTURE_KEY='UNIT_NOT_KEY'):s.propose(p)
        self.assertEqual(transport.call_args.args[3],90.0);transport.assert_called_once()
        self.assertEqual(s._envelope(self.p)['timeout_seconds'],30)

    def test_timeout_is_cached_and_never_automatically_resent(self):
        p=deepcopy(self.p);p['timeout_seconds']=90
        transport=Mock(side_effect=TimeoutError());s=self.service(transport)
        with patch.dict(os.environ,DISCOVERY_FIXTURE_KEY='UNIT_NOT_KEY'):
            first=s.propose(p);second=s.propose(p)
        self.assertEqual(first['fallback_reason'],'api_timeout')
        self.assertEqual(second['fallback_reason'],'api_timeout');transport.assert_called_once()

    def test_invalid_timeout_does_not_send(self):
        for timeout in (0,121,float('nan'),True):
            p=deepcopy(self.p);p['timeout_seconds']=timeout;transport=Mock()
            s=self.service(transport)
            with patch.dict(os.environ,DISCOVERY_FIXTURE_KEY='UNIT_NOT_KEY'):s.propose(p)
            transport.assert_not_called()

    def test_residual_keeps_unknown_cost_and_call_reserved_without_old_write(self):
        value=remaining_proposal(OLD/'hierarchy.sqlite')
        self.assertEqual(value['prior_calls'],2);self.assertEqual(value['pending_or_unknown'],1)
        self.assertEqual(value['policy']['max_calls'],6)
        self.assertEqual(value['policy']['max_usd'],'1.3336990')
        self.assertEqual(value['policy']['roles']['learner']['max_calls'],3)
        self.assertFalse(value['policy']['enabled']);self.assertTrue(value['parent_unchanged'])

    def test_cli_requires_revision_opt_in_and_bounded_timeout(self):
        with self.assertRaises(ValueError):runner.main(['--run-dir',str(self.root/'bad'),'--python-revision-version','2'])
        with self.assertRaises(SystemExit):runner.main(['--run-dir',str(self.root/'bad'),'--generation-timeout-seconds','999'])

    def remaining(self):
        r=remaining_proposal(OLD/'hierarchy.sqlite')
        binding={k:r[k] for k in ('parent_ledger','parent_ledger_sha256','parent_policy_sha256','parent_rows_sha256')}
        return binding,r['policy']

    def test_residual_policy_cannot_refund_unknown_or_raise_calls(self):
        binding,p=self.remaining()
        validate_remaining_binding(binding,p)
        for key,value in [('max_calls',7),('max_usd','1.35')]:
            bad=deepcopy(p);bad[key]=value
            with self.assertRaisesRegex(ValueError,'increase_or_reallocate'):
                validate_remaining_binding(binding,bad)

    def test_changed_parent_hash_is_rejected(self):
        binding,p=self.remaining();binding['parent_rows_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'parent_changed'):validate_remaining_binding(binding,p)

    def test_only_one_explicit_continuation_can_claim_remaining_budget(self):
        binding,p=self.remaining();p['enabled']=True
        with patch('tools.remaining_split_budget.CLAIM_ROOT',self.root/'claims'):
            claim=claim_remaining_binding(binding,p,self.root/'one')
            self.assertTrue(claim['unknown_reservations_retained'])
            with self.assertRaises(FileExistsError):claim_remaining_binding(binding,p,self.root/'two')

    def test_disabled_preparation_cannot_claim_budget(self):
        binding,p=self.remaining()
        with patch('tools.remaining_split_budget.CLAIM_ROOT',self.root/'claims'):
            with self.assertRaisesRegex(ValueError,'disabled'):claim_remaining_binding(binding,p,self.root/'one')
        self.assertFalse((self.root/'claims').exists())

    def test_generation_options_must_be_paired_and_bounded(self):
        p=json.loads((OLD.parent/'protocol.json').read_text())
        for extras in ({'generation_contract_version':2},
                       {'generation_timeout_seconds':90},
                       {'generation_contract_version':2,'generation_timeout_seconds':999},
                       {'generation_contract_version':True,'generation_timeout_seconds':90}):
            with self.assertRaises(ValueError):split.validate_protocol(dict(p,**extras),{})

    def test_candidate_loop_passes_explicit_version_and_timeout_once(self):
        output=self.root/'controller';output.mkdir()
        protocol=dict(physical_revisions=1,contract_corrections=1,
                      generation_contract_version=2,generation_timeout_seconds=90)
        paths={k:self.root/k for k in ('parent','manifest','draft_lesson','prior_library')}
        calls=[]
        def invoke(label,tool,args):
            calls.append(args)
            return 2  # Unknown subprocess result must stop, never reissue.
        result=split.run_candidates(output,protocol,paths,dict(learner='UNIT'),invoke)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][calls[0].index('--python-revision-version')+1],2)
        self.assertEqual(calls[0][calls[0].index('--generation-timeout-seconds')+1],90)
        self.assertIsNone(result['admitted_folder'])


if __name__=='__main__':unittest.main()
