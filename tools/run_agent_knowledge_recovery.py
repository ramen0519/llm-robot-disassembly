"""One declared failure replay -> autonomous Sol lookup -> ONE real action.

prepare is offline. run requires explicit authorization of at most 3 requests,
$1 configured-rate reservation, 2 physical attempts and 20 minutes. This is a
component experiment, not a fresh full-disassembly run or a performance test.
"""
from copy import deepcopy
from datetime import datetime,timezone
from decimal import Decimal
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from tools.run_integrated_repair import checked
from tools.run_skill_discovery import source_case,actual_action
from tools.python_source_budget import source_budget
from tools.local_api_key import key_environment
from tools.run_agent_knowledge_smoke import SOURCE,read,sha,source_hashes as base_hashes
from orchestrator.lesson_memory import load_bundle,write_new
from orchestrator.hierarchy_budget import HierarchyBudget,validate_policy
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.explicit_action_service import ExplicitActionService
from orchestrator.knowledge_tools import KnowledgeCatalog,encoded_size
from orchestrator.skill_case import physical_context,request_after_trigger,validate_trigger_scope
from worker.explicit_action_contract import validate_response,digest

SPEC=dict(case_id='02628_first_path_failure',source_task='02628',phase='transfer',
    report='runs/frozen_skill_transfer_02628_20261022/live/episode/planner/planner_report.json',
    attempt_index=1,trigger_kind='path_failed')
MAX_SECONDS=1200


def policy():
    p=read(SOURCE/'policy.json');p.update(enabled=True,max_calls=3,max_usd='1.00')
    for role in ('scout','planner','learner'):
        p['roles'][role].update(enabled=True,max_calls=3 if role=='planner' else 0,
            max_usd='1.00' if role=='planner' else '0')
    p['episode'].update(scout_calls=0,planner_calls=1,fallback_calls=0,max_plan_steps=1,recovery_per_step=1)
    return validate_policy(p)


def source_hashes():
    hashes=base_hashes()
    for rel in [SPEC['report'],*['tools/'+name for name in (
        'run_agent_knowledge_recovery.py','run_skill_discovery.py','python_source_budget.py',
        'python_recorded_clock.py','run_recorded.py')]]:
        hashes[rel]=sha(ROOT/rel)
    first=ROOT/Path(SPEC['report']).parent/'attempt_000001'
    for name in ('request.json','result.json','trajectory.json','endpoint_evidence.json'):
        hashes[(first/name).relative_to(ROOT).as_posix()]=sha(first/name)
    return hashes


def packet_for(prepared,request,choice,event):
    seconds=prepared['source_budget']['seconds']
    context=physical_context(prepared['context'],request,choice,event_id=event,
        seconds=seconds,wall_limit=seconds,preserve_stability_policy=True)
    from tools.python_source_budget import validate_binding
    validate_binding(context,prepared['source_budget'])
    return dict(context=context,choice=deepcopy(choice))


def check_physics(prepared,request,choice,raw):
    validate_trigger_scope(request,raw,choice)
    if raw['source_hashes']!=prepared['expected_source_hashes']:
        raise ValueError('physical_implementation_changed')
    elapsed=raw.get('elapsed_seconds')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
        raise ValueError('invalid_physics_elapsed')
    return elapsed<=prepared['source_budget']['seconds']


def make_request(prepared,raw,run_id):
    seconds=prepared['source_budget']['seconds']
    return request_after_trigger(prepared['request'],raw,prepared['choice'],run_id=run_id,
        event_id=run_id+':trigger',wall_seconds=seconds,wall_limit=seconds)


def model_decision(folder,budget,request,data,*,backend=None):
    folder.mkdir(exist_ok=False);base=budget.transport('planner',backend);count=0
    def transport(url,headers,body,timeout):
        nonlocal count
        count+=1;write_new(folder/f'provider_body_{count:02}.json',body)
        print('MODEL_ATTEMPT='+str(count)+' '+body['text']['format']['name'],flush=True)
        return base(url,headers,body,timeout)
    transport.live=base.live;transport.circuit_status=base.circuit_status
    cfg=budget.policy['roles']['planner']
    planner=SolPlanService(cfg,folder/'requests.sqlite',transport,
        provider_contract='conditional_plan_v4',planning_feedback_policy='repair_evidence_v2')
    planner._live_transport=base.live
    scout=ExplicitActionService(cfg,folder/'unused_scout.sqlite',mode='deterministic')
    service=HierarchicalDecisionService(scout,planner,data['skills'],budget.policy['episode'],
        lesson_runtime=True,lesson_library=data['lessons'],knowledge_policy='agent_tools_v1',
        knowledge_memory=data['memory'],knowledge_max_rounds=2,timeout_seconds=120,
        recovery_support_policy='preserve_existing_v1',planning_feedback_policy='repair_evidence_v2',
        transport_guard=budget.transport_blocked,journal_dir=folder/'journal')
    try:
        response=service.handle(deepcopy(request));write_new(folder/'response.json',response)
        views=service.knowledge_queries;plans=service.planner_decisions
        view=views[-1] if views else None
        result=dict(response=response,read_ids=(view or {}).get('audit',{}).get('read_document_ids',[]),
            retrieval_status=(view or {}).get('audit',{}).get('retrieval_status'),
            search_count=sum(x['operation']['tool']=='search' for h in
                (view or {}).get('retrieval_trace',{}).get('tool_history',[]) for x in h['results']),
            skill_selected=response['metadata'].get('generated_program_executed') is True,
            selected_skill_id=response['metadata'].get('generated_skill_id'),
            plan_status=plans[-1]['result']['status'] if plans else None,
            model_live=bool(plans and plans[-1]['result']['metadata'].get('llm_live')),
            automatic_retries=0)
        write_new(folder/'summary.json',result);return result
    finally:
        write_new(folder/'knowledge_contexts.json',service.knowledge_queries)
        write_new(folder/'plans.json',service.planner_decisions)
        write_new(folder/'decisions.json',service.decisions);service.close()


def prepare(folder):
    folder=checked(folder,True)
    if folder.exists():raise ValueError('new_preparation_required')
    p=source_case(SPEC);p['source_budget']=source_budget(p)
    if not check_physics(p,p['request'],p['choice'],p['raw']):raise ValueError('original_trigger_late')
    if p['raw']['path_success'] is not False:raise ValueError('source_not_path_failure')
    data=load_bundle(SOURCE/'knowledge_bundle.json');KnowledgeCatalog(data['skills'],data['lessons'],data['memory'])
    folder.mkdir(parents=True,exist_ok=False)
    write_new(folder/'prepared.json',p);write_new(folder/'policy.json',policy())
    for kind in ('skills','lessons','memory'):write_new(folder/(kind+'.json'),data[kind])
    request=make_request(p,p['raw'],'historical-preview:'+digest(str(folder))[:16])
    write_new(folder/'historical_preview_request.json',request)
    packet_for(p,request,p['choice'],'preview-binding-only')
    hashes=source_hashes()
    for rel in hashes:
        target=folder/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(checked(ROOT/rel).read_bytes())
    protocol=dict(kind='agent_knowledge_failure_recovery_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        approval_status_at_preparation='PENDING_NEW_3_CALL_1_USD_2_PHYSICS_SCOPE',
        max_calls=3,max_usd='1.00',model='gpt-6.1-sol',endpoint='https://api.openai.com/v1/responses',
        auto_http_retries=0,max_physics_attempts=2,max_seconds=MAX_SECONDS,
        original_seconds_per_physics_attempt=p['source_budget']['seconds'],
        source_hashes=hashes,selection_rule='First chronologically recorded path failure in the fixed 02628 episode; not selected by recovery outcome.',
        fresh_trigger_required=True,forced_skill_choice=False,model_queries_max=2,model_plans_max=1,
        full_disassembly=False,skill_learning=False,performance_test=False,
        data_scope='Official 02628 state, fresh failure measurements and frozen Beam experience, lesson and Python skill.',
        prepared_files={name:sha(folder/name) for name in (
            'prepared.json','policy.json','skills.json','lessons.json','memory.json','historical_preview_request.json')})
    write_new(folder/'protocol.json',protocol)
    print(json.dumps(dict(status='PREPARED_NOT_AUTHORIZED',api_calls=0,physics_calls=0,
        max_calls=3,max_usd='1.00',max_physics_attempts=2,max_seconds=MAX_SECONDS)))


def trial(folder,prepared,data,budget,*,physics=actual_action,backend=None):
    start=time.monotonic();run_id='knowledge-recovery:'+digest(str(folder))[:16]
    result=dict(status='RUNNING',physics_attempts=0,full_disassembly=False,performance_effect='UNVERIFIED',
        stored_failure_relabelled_as_fresh=False,automatic_http_retries=0)
    def attempt(request,choice,label):
        packet=packet_for(prepared,request,choice,run_id+':'+label)
        if time.monotonic()-start+prepared['source_budget']['seconds']+20>MAX_SECONDS:
            raise TimeoutError('not_enough_trial_time_for_original_physics_budget')
        result['physics_attempts']+=1
        print('PHYSICS_ATTEMPT='+label,flush=True)
        raw=physics(packet,folder/label,source_budget=prepared['source_budget'])
        accepted=check_physics(prepared,request,choice,raw)
        write_new(folder/(label+'_binding.json'),dict(model_request_id=request['request_id'],
            model_request_sha256=digest(request),choice_sha256=digest(choice),raw_sha256=digest(raw),
            raw_path_success=raw['path_success'],raw_stability_success=raw['stability_success'],
            within_original_allowance=accepted,accepted_for_DFS_transition=False))
        return raw,accepted
    try:
        trigger,on_time=attempt(prepared['request'],prepared['choice'],'trigger')
        result['trigger']=dict(status=trigger['status'],path_success=trigger['path_success'],
            stability_success=trigger['stability_success'],within_original_allowance=on_time)
        if not on_time or trigger['status'] not in ('UNKNOWN','REJECTED') or trigger['path_success'] is not False:
            result['status']='TRIGGER_NOT_REPRODUCED';return result
        request=make_request(prepared,trigger,run_id);write_new(folder/'fresh_request.json',request)
        if time.monotonic()-start+120+prepared['source_budget']['seconds']+20>MAX_SECONDS:
            raise TimeoutError('not_enough_trial_time_for_model_and_physics')
        # Credential loading happens only after an actual matching failure.
        if backend is None:
            with key_environment() as origin:
                if origin=='missing':raise ValueError('stored_key_unavailable')
                selected=model_decision(folder/'model',budget,request,data)
        else:selected=model_decision(folder/'model',budget,request,data,backend=backend)
        result['selection']=selected
        response=validate_response(request,selected['response'])
        if (response.get('fallback_reason') is not None or selected['plan_status']!='planned'
                or selected['retrieval_status']!='ready' or (backend is None and not selected['model_live'])):
            result['status']='NO_VALID_LIVE_DECISION';return result
        recovery,on_time=attempt(request,response['choice'],'selected_action')
        passed=on_time and recovery['status']=='PASS' and recovery['path_success'] is True and recovery['stability_success'] is True
        result['recovery']=dict(status=recovery['status'],path_success=recovery['path_success'],
            stability_success=recovery['stability_success'],within_original_allowance=on_time,
            path_calls=recovery['path_check_calls'],stability_calls=recovery['stability_check_calls'])
        result['status']=('SKILL_RECOVERY_PASS' if selected['skill_selected'] else 'DIRECT_ACTION_PASS_NO_SKILL') if passed else 'SELECTED_ACTION_NOT_PASS'
        return result
    except Exception as error:
        result.update(status='ERROR',error_type=type(error).__name__);return result
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        result['budget']=budget.summary()
        result['actual_api_calls']=result['budget']['live_api_attempts']
        result['injected_test']=backend is not None
        write_new(folder/'result.json',result)


def execute(preparation,output):
    preparation=checked(preparation,True);output=checked(output,True)
    if output.exists() or output==preparation or preparation in output.parents:
        raise ValueError('new_separate_output_required')
    protocol=read(preparation/'protocol.json')
    if protocol['max_calls']!=3 or protocol['max_usd']!='1.00' or protocol['max_physics_attempts']!=2:
        raise ValueError('changed_trial_limits')
    if source_hashes()!=protocol['source_hashes']:raise ValueError('source_changed_since_preparation')
    if any(sha(preparation/name)!=value for name,value in protocol['prepared_files'].items()):
        raise ValueError('prepared_data_changed')
    # This preparation can be executed once; no resume or automatic retry.
    write_new(preparation/'execution_claim.json',dict(output=str(output),started_utc=datetime.now(timezone.utc).isoformat()))
    output.mkdir(parents=True,exist_ok=False)
    budget=HierarchyBudget(output/'budget.sqlite',read(preparation/'policy.json'),transport_failure_policy='latch_on_error_v1')
    result=trial(output,read(preparation/'prepared.json'),
        {k:read(preparation/(k+'.json')) for k in ('skills','lessons','memory')},budget)
    unchanged=source_hashes()==protocol['source_hashes']
    write_new(output/'preservation.json',dict(original_files_unchanged=unchanged))
    print(json.dumps(dict(status=result['status'],actual_api_calls=result['actual_api_calls'],
        physics_attempts=result['physics_attempts'],original_files_unchanged=unchanged),ensure_ascii=False))
    return 0 if unchanged and result['status'] in ('SKILL_RECOVERY_PASS','DIRECT_ACTION_PASS_NO_SKILL') else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('prepare','run'))
    p.add_argument('--preparation',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--allow-live-api-and-physics',action='store_true');a=p.parse_args()
    if a.operation=='prepare':
        if a.allow_live_api_and_physics or a.output:raise ValueError('prepare_is_offline')
        prepare(a.preparation);return 0
    if not a.allow_live_api_and_physics or a.output is None:raise ValueError('explicit_trial_authorization_required')
    return execute(a.preparation,a.output)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print('RECOVERY_RUNNER_ERROR='+type(error).__name__);raise SystemExit(1)
