"""Synthetic bookkeeping only; backend labels exercise contracts, not physics."""
from copy import deepcopy
import unittest

from orchestrator.skill_usage import summarize_usage
from orchestrator.skill_program import program_hash, step_program
from tests.test_skill_program import request, append_observation, program
from worker.explicit_action_contract import validate_request, canonical_response


def bundle():
    req=validate_request(append_observation(request()))
    policy=program(); ident=program_hash(policy); step=step_program(policy,req)
    response=step['response']
    selected=dict(operation='select',request_id='select:r1',state_id=req['state_id'],
        selected_skill_id=ident,status='selected',origin='actual_llm',metadata={'llm_live':True})
    audit=dict(strategy_id=ident,program_hash=ident,triggered=True,scanned=step['scanned'],
               not_applicable_reason=None,rule_index=step['rule_index'])
    response['metadata'].update(generated_program_executed=True,generated_skill_id=ident,
        generated_program_hash=ident,generated_library_sha256='synthetic_library',
        generated_skill_selection_origin='actual_llm',generated_skill_applicability=[audit],
        generated_selector_response=selected)
    host={'request':req,'response':response}
    selection={'packet':dict(operation='select',request_id='select:r1',state_id=req['state_id'],
        run_id=req['run_id'],current_request=deepcopy(req),skills=[{'skill_id':ident,'program_hash':ident}],
        context={'library_sha256':'synthetic_library'}),'response':deepcopy(selected)}
    attempt=dict(event_id='fixture:event',request_id=req['request_id'],state_id=req['state_id'],
        action_id=response['action_id'],choice=deepcopy(response['choice']),status='PASS',
        deadline_exceeded=False,accepted_for_transition=True,
        raw=dict(backend='ASAP_RedMax',fixture_only=True,chosen=deepcopy(response['choice']),
                 path_check_calls=1,stability_check_calls=1,
                 status='PASS',feasible=True,path_success=True,stability_success=True))
    report={'decisions':[deepcopy(host)],'attempts':[attempt]}
    return ident,host,selection,report


class UsageTests(unittest.TestCase):
    def summary(self,host,selection,report):
        return summarize_usage([host],report,[selection])

    def test_stage_counts_and_exact_physical_binding(self):
        ident,host,selection,report=bundle();s=self.summary(host,selection,report)
        row=s['per_skill'][ident]
        for key in ['checked','triggered','applicable','offered_to_llm_selector','selected','actual_llm_selected',
                    'program_proposals','worker_bound_attempts','physical_attempts','physical_passes','transition_passes']:
            self.assertEqual(row[key],1,key)
        self.assertEqual(s['attempt_rows'][0]['generated_skill_id'],ident)

    def test_missing_report_is_unknown_not_zero(self):
        ident,host,selection,_=bundle();row=summarize_usage([host],None,[selection])['per_skill'][ident]
        self.assertEqual(row['selected'],1);self.assertIsNone(row['physical_attempts'])

    def test_missing_selector_log_is_unsupported(self):
        ident,host,_,report=bundle();row=summarize_usage([host],report)['per_skill'][ident]
        self.assertEqual(row['selected'],1);self.assertIsNone(row['actual_llm_selected'])

    def test_selection_without_emitted_proposal_is_still_selection(self):
        ident,host,selection,report=bundle()
        host['response']['metadata']['generated_program_executed']=False
        row=self.summary(host,selection,report)['per_skill'][ident]
        self.assertEqual(row['selected'],1);self.assertEqual(row['actual_llm_selected'],1)
        self.assertEqual(row['program_proposals'],0);self.assertEqual(row['physical_attempts'],0)

    def test_worker_timeout_replacement_not_generated_execution(self):
        ident,host,selection,report=bundle()
        report['decisions'][0]['response']['metadata']['generated_program_executed']=False
        row=self.summary(host,selection,report)['per_skill'][ident]
        self.assertEqual(row['program_proposals'],1);self.assertEqual(row['physical_attempts'],0)

    def test_wrong_raw_choice_not_credited(self):
        ident,host,selection,report=bundle();report['attempts'][0]['raw']['chosen']['part_id']='base'
        self.assertEqual(self.summary(host,selection,report)['per_skill'][ident]['physical_attempts'],0)

    def test_wrong_state_not_credited(self):
        ident,host,selection,report=bundle();report['attempts'][0]['state_id']='other'
        self.assertEqual(self.summary(host,selection,report)['per_skill'][ident]['physical_attempts'],0)

    def test_non_asap_result_not_counted_as_actual_physics(self):
        ident,host,selection,report=bundle();report['attempts'][0]['raw']['backend']='mock'
        row=self.summary(host,selection,report)['per_skill'][ident]
        self.assertEqual(row['worker_bound_attempts'],1);self.assertEqual(row['physical_attempts'],0)

    def test_backend_label_without_dispatch_not_physical_attempt(self):
        ident,host,selection,report=bundle()
        report['attempts'][0]['raw'].update(path_check_calls=0,stability_check_calls=0)
        row=self.summary(host,selection,report)['per_skill'][ident]
        self.assertEqual(row['worker_bound_attempts'],1)
        self.assertEqual(row['physical_attempts'],0);self.assertEqual(row['physical_passes'],0)

    def test_missing_dispatch_counters_are_unknown(self):
        ident,host,selection,report=bundle()
        del report['attempts'][0]['raw']['path_check_calls']
        del report['attempts'][0]['raw']['stability_check_calls']
        result=self.summary(host,selection,report);row=result['per_skill'][ident]
        self.assertEqual(row['worker_bound_attempts'],1)
        self.assertIsNone(row['physical_attempts']);self.assertIsNone(row['physical_passes'])
        self.assertEqual(result['missing_evidence']['physical_dispatch_attempts'],1)

    def test_proposal_cannot_use_a_different_selected_skill(self):
        ident,host,selection,report=bundle()
        other='other_program'
        audit=deepcopy(host['response']['metadata']['generated_skill_applicability'][0])
        audit.update(strategy_id=other,program_hash=other)
        host['response']['metadata']['generated_skill_applicability'].append(audit)
        host['response']['metadata']['generated_selector_response']['selected_skill_id']=other
        with self.assertRaisesRegex(ValueError,'proposal_does_not_match_selected_skill'):
            self.summary(host,selection,report)

    def test_raw_failure_cannot_borrow_pass_label(self):
        ident,host,selection,report=bundle();report['attempts'][0]['raw']['stability_success']=False
        row=self.summary(host,selection,report)['per_skill'][ident]
        self.assertEqual(row['physical_attempts'],1);self.assertEqual(row['physical_passes'],0)

    def test_deadline_pass_not_counted(self):
        ident,host,selection,report=bundle();report['attempts'][0]['deadline_exceeded']=True
        self.assertEqual(self.summary(host,selection,report)['per_skill'][ident]['physical_passes'],0)

    def test_unaccepted_pass_not_transition(self):
        ident,host,selection,report=bundle();report['attempts'][0]['accepted_for_transition']=False
        row=self.summary(host,selection,report)['per_skill'][ident]
        self.assertEqual(row['physical_passes'],1);self.assertEqual(row['transition_passes'],0)

    def test_duplicate_events_and_decisions_count_once(self):
        ident,host,selection,report=bundle()
        report['attempts']*=2
        result=summarize_usage([host,deepcopy(host)],report,[selection,deepcopy(selection)])
        self.assertEqual(result['per_skill'][ident]['physical_attempts'],1)
        self.assertEqual(result['per_skill'][ident]['actual_llm_selected'],1)
        self.assertEqual(result['duplicate_decisions'],1);self.assertEqual(result['duplicate_attempts'],1)

    def test_conflicting_duplicate_rejected(self):
        _,host,selection,report=bundle();other=deepcopy(report['attempts'][0]);other['status']='UNKNOWN'
        report['attempts'].append(other)
        with self.assertRaisesRegex(ValueError,'conflicting'):self.summary(host,selection,report)

    def test_different_selector_request_cannot_lend_llm_origin(self):
        ident,host,selection,report=bundle();selection['packet']['current_request']['request_id']='elsewhere'
        self.assertEqual(self.summary(host,selection,report)['per_skill'][ident]['actual_llm_selected'],0)

    def test_no_trigger_has_no_selection(self):
        ident,host,selection,report=bundle()
        req=validate_request(request());response=canonical_response(req,None)
        response['metadata'].update(generated_program_executed=False,generated_skill_applicability=[
            dict(strategy_id=ident,program_hash=ident,triggered=False,scanned=0,not_applicable_reason='no_trigger',rule_index=None)])
        row=summarize_usage([{'request':req,'response':response}],{'decisions':[],'attempts':[]},[])['per_skill'][ident]
        self.assertEqual(row['checked'],1);self.assertEqual(row['triggered'],0)
        self.assertEqual(row['not_applicable_reasons'],{'no_trigger':1})

    def test_triggered_but_exhausted_not_applicable(self):
        ident,host,selection,report=bundle();meta=host['response']['metadata']
        meta['generated_skill_applicability'][0]['not_applicable_reason']='no_matching_untried_action'
        meta['generated_selector_response']=None;meta['generated_program_executed']=False
        row=summarize_usage([host],None,[])['per_skill'][ident]
        self.assertEqual(row['triggered'],1);self.assertEqual(row['applicable'],0)

    def test_legacy_missing_audit_reported(self):
        req=validate_request(request());response=canonical_response(req,None)
        summary=summarize_usage([{'request':req,'response':response}])
        self.assertEqual(summary['missing_evidence']['applicability_requests'],1)


if __name__=='__main__':unittest.main()
