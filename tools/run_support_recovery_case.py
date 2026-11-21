"""One explicitly budgeted chain of generated support proposals; no model calls."""
from copy import deepcopy
import math
import time

from orchestrator.recovery_sequence import validate_protocol, append_feedback, same_support_scope, validation_execution
from orchestrator.skill_case import physical_context
from orchestrator.skill_validation import _natural_trigger
from orchestrator.lesson_runtime import CONTRACT, bind_execution, execute_bound
from orchestrator.skill_program import program_hash
from tools.run_luna_experiment import read,write,sha
from tools.python_source_budget import source_budget
from worker.task_memory import digest


def run_case(prepared,case,program,output,run_id,*,action_executor=None,clock=time.monotonic):
    from tools.run_skill_discovery import actual_action
    execute=action_executor or actual_action
    protocol=validate_protocol(case['recovery_protocol']);binding=source_budget(prepared)
    seconds=case['case_wall_seconds']
    if (case.get('runtime_contract')!=CONTRACT or case.get('source_budget')!=binding
            or seconds!=binding['seconds'] or case['trigger_kind']!='stability_failed'):
        raise ValueError('support_chain_requires_bound_stability_lesson_case')
    folder=output/'cases'/case['case_id'];folder.mkdir(parents=True,exist_ok=False)
    started=clock();reserve=math.ceil(seconds)+35
    if protocol['max_total_seconds']<2*reserve:raise ValueError('chain_budget_cannot_cover_trigger_and_recovery')
    template=prepared['request'];original=prepared['choice'];event=run_id+':'+case['case_id']+':trigger'
    context=physical_context(prepared['context'],template,original,event_id=event,
        seconds=seconds,wall_limit=seconds,preserve_stability_policy=True)
    raw=execute(dict(context=context,choice=original),folder/'trigger',source_budget=binding)
    if not _natural_trigger(raw,'stability_failed',raw['elapsed_seconds']>seconds,
            expected_phase=case.get('expected_diagnostic_phase')):
        raise ValueError('support_chain_natural_trigger_not_reproduced')
    def metadata(raw,physical_folder):
        return (validation_execution(raw,seconds,read(physical_folder/'execute/command.json'))
                if protocol['version']=='bounded_support_recovery_v2' else None)
    request=append_feedback(template,raw,original,run_id=event,event_id=event,seconds=seconds,clear=True,
                            execution=metadata(raw,folder/'trigger'))
    trigger_dir=folder/'trigger/physics';steps=[];stop='ATTEMPT_LIMIT'
    for index in range(protocol['max_recovery_attempts']):
        if protocol['max_total_seconds']-(clock()-started)<reserve:
            stop='INSUFFICIENT_TIME_FOR_UNCHANGED_CASE_BUDGET';break
        here=folder/('step_%02d'%(index+1));here.mkdir()
        write(here/'program_request.json',request)
        runtime=bind_execution(program,request);proposal=execute_bound(program,request,runtime['extension'])
        write(here/'program_step.json',proposal)
        selected=dict(case_id=case['case_id'],source_task=case['source_task'],program_hash=program_hash(program),
            program_status='NOT_APPLICABLE',program_request_path=str(here/'program_request.json'),
            program_request_sha256=sha(here/'program_request.json'),
            trigger_result_sha256=sha(trigger_dir/'result.json'),program_runtime=runtime)
        artifacts=dict(attempt_dir=None,selection_path=str(here/'selection.json'),trigger_attempt_dir=str(trigger_dir))
        response=proposal.get('response')
        if response is None:
            write(here/'selection.json',selected);steps.append(artifacts);stop='ABSTAIN';break
        choice=response['choice']
        if not same_support_scope(original,choice):raise ValueError('proposal_outside_support_recovery_scope')
        context=physical_context(prepared['context'],request,choice,event_id=event+':recovery:'+str(index+1),
            seconds=seconds,wall_limit=seconds,preserve_stability_policy=True)
        raw=execute(dict(context=context,choice=choice),here/'recovery',source_budget=binding)
        artifacts['attempt_dir']=str(here/'recovery/physics')
        selected.update(program_status='ACTION',action_sha256=digest(choice),
            worker_request_sha256=sha(here/'recovery/physics/request.json'),
            worker_result_sha256=sha(here/'recovery/physics/result.json'))
        write(here/'selection.json',selected);steps.append(artifacts)
        if raw['elapsed_seconds']>seconds:stop='LATE';break
        if raw['status']=='PASS':stop='PASS';break
        if not _natural_trigger(raw,'stability_failed',raw['elapsed_seconds']>seconds,
                expected_phase=case.get('expected_diagnostic_phase')):
            stop='FAILURE_STAGE_CHANGED_OR_ERROR';break
        trigger_dir=here/'recovery/physics'
        request=append_feedback(request,raw,choice,run_id=event+':step:'+str(index+2),
            event_id=event+':feedback:'+str(index+1),seconds=seconds,execution=metadata(raw,here/'recovery'))
    if not steps:raise ValueError('no_budget_for_first_recovery')
    write(folder/'recovery_sequence.json',dict(schema_version=1,protocol=protocol,
        case_id=case['case_id'],program_hash=program_hash(program),steps=steps,
        elapsed_seconds=clock()-started,stop_reason=stop,api_calls=0))
    return dict(steps[-1],recovery_sequence_path=str(folder/'recovery_sequence.json'))
