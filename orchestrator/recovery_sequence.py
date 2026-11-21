"""Opt-in bounded support recovery; legacy one-step evidence stays unchanged."""
from copy import deepcopy
import math
from pathlib import Path

from worker.task_memory import digest


def validate_protocol(value):
    if (type(value) is not dict or set(value) != {'version','max_recovery_attempts','max_total_seconds','action_scope'}
            or value['version'] not in ('bounded_support_recovery_v1','bounded_support_recovery_v2') or value['action_scope'] != 'support_only'
            or type(value['max_recovery_attempts']) is not int or not 1 <= value['max_recovery_attempts'] <= 3
            or type(value['max_total_seconds']) not in (int,float)
            or not math.isfinite(value['max_total_seconds']) or not 0 < value['max_total_seconds'] <= 3600):
        raise ValueError('invalid_bounded_recovery_protocol')
    return deepcopy(value)


def validation_execution(raw, seconds, command):
    """Measured worker/host duration under an unchanged source allowance."""
    from orchestrator.execution_observation import validate_execution
    if (command.get('exit_code') != 0 or command.get('launch_error') is not None
            or type(command.get('elapsed_seconds')) not in (int,float)
            or command['elapsed_seconds'] < raw['elapsed_seconds']):
        raise ValueError('validation_execution_command_not_bound')
    return validate_execution(dict(version='planner_execution_v1',source='validation_attempt',
        accepted_for_transition=False,deadline_exceeded=raw['elapsed_seconds']>seconds,
        attempt_elapsed_seconds=command['elapsed_seconds'],physics_elapsed_seconds=raw['elapsed_seconds'],
        remaining_wall_seconds=seconds,allowance_source='context.remaining_wall_seconds'),raw['status'])


def append_feedback(template, raw, choice, *, run_id, event_id, seconds, clear=False, execution=None):
    """Append a scope-validated real observation without losing earlier choices."""
    from orchestrator.skill_case import validate_trigger_scope
    from worker.explicit_action_contract import validate_request, action_id
    validate_trigger_scope(template, raw, choice)
    request=deepcopy(template)
    if clear:
        request.update(attempted_action_ids=[],attempted_actions=[],path_stage_observations=[],
                       episode_feedback=[],memory_references=[],evidence_ids=[])
    identity=action_id(request,choice)
    if identity in request['attempted_action_ids'] or event_id in request['evidence_ids']:
        raise ValueError('recovery_sequence_duplicate_action_or_event')
    clean=deepcopy(choice);clean['evidence_ids']=[]
    late=raw['elapsed_seconds'] > seconds
    pose=next(p['matrix'] for p in request['pose_catalog'] if p['pose_id']==choice['pose_id'])
    feedback=dict(evidence_id=event_id,status=raw['status'],action_id=identity,
        remaining_parts=deepcopy(request['remaining_parts']),choice=clean,
        decision_source='generated_program_validation',selected_pose_matrix=deepcopy(pose),
        path_success=raw['path_success'],stability_success=raw['stability_success'],
        measured_endpoint=deepcopy(raw.get('endpoint_observation')),
        raw_stability_parts=deepcopy(raw.get('raw_upstream',{}).get('parts_fall')),
        accepted_for_transition=False,deadline_exceeded=late)
    if execution is not None:
        from orchestrator.execution_observation import validate_execution
        execution=validate_execution(execution,raw['status'])
        if (execution['source']!='validation_attempt' or execution['accepted_for_transition'] is not False
                or execution['physics_elapsed_seconds']!=raw['elapsed_seconds']
                or execution['remaining_wall_seconds']!=seconds or execution['deadline_exceeded']!=late):
            raise ValueError('validation_execution_feedback_mismatch')
        feedback['execution']=execution
    if raw.get('stability_diagnostics') is not None:
        from worker.stability_diagnostics import project_stability_diagnostics
        feedback['stability_diagnostics']=project_stability_diagnostics(raw['stability_diagnostics'])
    request['attempted_action_ids'].append(identity)
    request['attempted_actions'].append(dict(action_id=identity,**{k:v for k,v in clean.items() if k!='evidence_ids'}))
    request['path_stage_observations'].append(dict(action_id=identity,path_success=raw['path_success'],
        status=raw['status'],deadline_exceeded=late,backend=raw['backend'],source_event_id=event_id))
    request['episode_feedback'].append(feedback);request['evidence_ids'].append(event_id)
    request.update(run_id=run_id,request_id=run_id+':recovery')
    return validate_request(request)


def same_support_scope(before, after):
    return all(before.get(k)==after.get(k) for k in
        ('part_id','pose_id','skill_id','direction_id','second_direction_id'))


def record_sequence(case, artifacts, program_hash, allow_test_root, fixture, program,
                    *, classification_version=4):
    from orchestrator.skill_validation import read_artifact, record_validation
    protocol=validate_protocol(case.get('recovery_protocol'))
    if program is None:raise ValueError('sequence_requires_executable_program')
    file=read_artifact(artifacts['recovery_sequence_path'],allow_test_root);trace=file['value']
    if (trace.get('protocol')!=protocol or trace.get('program_hash')!=program_hash
            or trace.get('case_id')!=case['case_id'] or trace.get('schema_version')!=1
            or not 1 <= len(trace.get('steps',[])) <= protocol['max_recovery_attempts']):
        raise ValueError('recovery_sequence_scope_or_count_mismatch')
    seconds=case.get('case_wall_seconds',30)
    records=[];seen=[];previous_request=None;first_choice=None;observed_seconds=0;execution_refs=[]
    for index, step in enumerate(trace['steps']):
        if 'recovery_sequence_path' in step:raise ValueError('nested_recovery_sequence')
        rec=record_validation(case,step,program_hash,allow_test_root,fixture,program,
                              _classification_version=classification_version)
        selected=read_artifact(step['selection_path'],allow_test_root)['value']
        req=read_artifact(selected['program_request_path'],allow_test_root)['value']
        prior=read_artifact(Path(step['trigger_attempt_dir'])/'result.json',allow_test_root)['value']
        prior_choice=prior['chosen']
        execution=None
        if protocol['version']=='bounded_support_recovery_v2':
            command=read_artifact(Path(step['trigger_attempt_dir']).parent/'execute/command.json',allow_test_root)
            execution=validation_execution(prior,seconds,command['value'])
            execution_refs.append({'path':command['path'],'sha256':command['sha256']})
        if index == 0:
            first_choice=prior_choice;observed_seconds+=prior['elapsed_seconds']
            expected=append_feedback(req,prior,prior_choice,run_id=req['run_id'],
                event_id=req['episode_feedback'][-1]['evidence_id'],seconds=seconds,clear=True,execution=execution)
            if expected!=req:raise ValueError('sequence_initial_feedback_not_bound')
        else:
            if (records[-1]['outcome'] not in ('UNKNOWN','REJECTED')
                    or step['trigger_attempt_dir'] != trace['steps'][index-1]['attempt_dir']):
                raise ValueError('sequence_must_continue_actual_prior_failure')
            expected=append_feedback(previous_request,prior,prior_choice,
                run_id=req['run_id'],event_id=req['episode_feedback'][-1]['evidence_id'],seconds=seconds,execution=execution)
            if expected!=req:raise ValueError('sequence_feedback_history_not_bound')
        if step['attempt_dir'] is not None:
            raw=read_artifact(Path(step['attempt_dir'])/'result.json',allow_test_root)['value']
            from worker.explicit_action_contract import action_id
            identity=action_id(req,raw['chosen'])
            if identity in seen or not same_support_scope(first_choice,raw['chosen']):
                raise ValueError('sequence_repeated_or_out_of_scope_action')
            seen.append(identity);observed_seconds+=raw['elapsed_seconds']
        records.append(rec);previous_request=req
    elapsed=trace.get('elapsed_seconds')
    if (type(elapsed) not in (float,int) or not math.isfinite(elapsed) or elapsed<0
            or observed_seconds>elapsed+0.001):
        raise ValueError('sequence_elapsed_accounting_mismatch')
    if trace.get('stop_reason') not in ('PASS','ABSTAIN','ATTEMPT_LIMIT','LATE',
            'INSUFFICIENT_TIME_FOR_UNCHANGED_CASE_BUDGET','FAILURE_STAGE_CHANGED_OR_ERROR'):
        raise ValueError('invalid_recovery_sequence_stop_reason')
    if (records[-1]['outcome']=='PASS') != (trace['stop_reason']=='PASS'):
        raise ValueError('recovery_sequence_stop_outcome_mismatch')
    final=deepcopy(records[-1])
    expected_artifacts=dict(trace['steps'][-1],recovery_sequence_path=file['path'])
    if artifacts!=expected_artifacts:raise ValueError('sequence_final_artifacts_mismatch')
    final.update(artifacts=deepcopy(artifacts),condition_hash=records[0]['condition_hash'],
        natural_trigger=records[0]['natural_trigger'],
        recovery_sequence=dict(protocol=protocol,steps=records,elapsed_seconds=elapsed,
            first_attempt_outcome=records[0]['outcome'],physical_recovery_attempts=len(seen),
            stop_reason=trace['stop_reason'],api_calls=0),
        path_calls=sum(r['path_calls'] for r in records),
        stability_calls=sum(r['stability_calls'] for r in records),
        trigger_path_calls=records[0].get('trigger_path_calls',0),
        trigger_stability_calls=records[0].get('trigger_stability_calls',0),
        elapsed_seconds=elapsed)
    if elapsed>protocol['max_total_seconds']:final['outcome']='LATE'
    if classification_version >= 4:
        from orchestrator.skill_execution_roles import sequence_execution_roles
        final['execution_roles']=sequence_execution_roles(records)
    final['source_refs']=[{'path':file['path'],'sha256':file['sha256']}]
    unique={r['path']:r for record in records for r in record['source_refs']}
    final['source_refs']+=list(unique.values())
    final['source_refs']+=execution_refs
    final['record_sha256']=digest({k:v for k,v in final.items() if k!='record_sha256'})
    return final


def debug_steps(record):
    """Expose every DEBUG transition to refinement/consolidation, never admission."""
    if record['case']['phase']!='debug':raise ValueError('debug_sequence_only')
    return record.get('recovery_sequence',{}).get('steps',[record])
