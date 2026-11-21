"""Bounded live search/read/decision on a SAVED 07009 state, never physics.

prepare is offline. run requires an explicit flag, a fresh immutable preparation,
and the user's already declared 6-call/$1 scope. No re-launch or HTTP retry.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_integrated_repair import checked
from local_api_key import key_environment
from orchestrator.lesson_memory import load_bundle, write_new
from orchestrator.hierarchy_budget import HierarchyBudget, validate_policy
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.explicit_action_service import ExplicitActionService
from orchestrator.generated_skill_service import GeneratedSkillService
from orchestrator.skill_discovery import SkillDiscoveryService
from orchestrator.knowledge_tools import KnowledgeCatalog
from worker.explicit_action_contract import digest, validate_request

SOURCE=ROOT/'runs/separation_final_live_20261022/live_cycle'
REQUEST=ROOT/'runs/knowledge_context_20261108_131509/saved_request_00.json'


def sha(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def read(path):
    path=checked(path)
    if path.stat().st_size>32*1024*1024: raise ValueError('smoke_input_too_large')
    return json.loads(path.read_text(encoding='utf-8'))


def source_hashes():
    paths=[REQUEST,SOURCE/'knowledge_bundle.json']
    bundle=read(SOURCE/'knowledge_bundle.json')
    paths += [checked(spec['path']) for spec in bundle['files'].values()]
    for folder in ('orchestrator','worker'):
        paths += list((ROOT/folder).glob('*.py'))
    paths += [Path(__file__).absolute(),ROOT/'tools/local_api_key.py',ROOT/'tools/run_integrated_repair.py',ROOT/'tools/run_luna_experiment.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def make_policy():
    p=read(SOURCE/'policy.json')
    p.update(enabled=True,max_calls=6,max_usd='1.00')
    for role,calls,cost in [('scout',3,'.15'),('planner',3,'.85'),('learner',0,'0')]:
        p['roles'][role].update(enabled=True,max_calls=calls,max_usd=cost)
    p['episode'].update(scout_calls=0,planner_calls=1,fallback_calls=0,max_plan_steps=16,recovery_per_step=1)
    return validate_policy(p)


def prepare(folder):
    folder=checked(folder,True)
    if folder.exists():raise ValueError('new_smoke_directory_required')
    request=validate_request(read(REQUEST));data=load_bundle(SOURCE/'knowledge_bundle.json')
    catalog=KnowledgeCatalog(data['skills'],data['lessons'],data['memory'])
    sources=source_hashes()
    folder.mkdir(parents=True,exist_ok=False)
    write_new(folder/'policy.json',make_policy())
    write_new(folder/'request.json',request)
    for kind in ('memory','skills','lessons'):write_new(folder/(kind+'.json'),data[kind])
    for rel in sources:
        path=folder/'source_snapshot'/rel;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(checked(ROOT/rel).read_bytes())
    protocol=dict(kind='agent_knowledge_saved_state_smoke_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        user_instruction='이어서해줘',authorization_basis='Continuation immediately after the explicit 6-call/$1.00 proposal; no scope expansion.',
        max_calls=6,max_usd='1.00',roles={'scout':'gpt-6-luna','planner':'gpt-6.1-sol'},
        endpoint='https://api.openai.com/v1/responses',automatic_http_retries=0,physics_calls=0,
        source_request=str(REQUEST),source_bundle=str(SOURCE/'knowledge_bundle.json'),
        source_hashes=sources,request_sha256=digest(request),pinned_knowledge=catalog.sources,
        saved_state_not_fresh_episode=True,forced_search_or_selection=False,
        role_timeout_seconds=120,total_model_call_maximum=6,new_skill_generation=False,
        data_scope='Official Beam/07009 numerical geometry, prior observations, registered Python skill and finalized lesson only.')
    protocol['prepared_files']={name:sha(folder/name) for name in ('policy.json','request.json','memory.json','skills.json','lessons.json')}
    write_new(folder/'protocol.json',protocol)
    print(json.dumps(dict(status='PREPARED',run_dir=str(folder),api_calls=0,physics_calls=0,
                         pinned_memory_records=data['memory']['record_count'],catalog_counts=catalog.session(request).index()['counts'])))
    return protocol


def summarize_role(service,response):
    views=service.knowledge_queries
    view=views[-1] if views else None
    history=(view or {}).get('retrieval_trace',{}).get('tool_history',[])
    operations=[r for h in history for r in h['results']]
    if isinstance(service,HierarchicalDecisionService):
        final=service.planner_decisions[-1]['result'] if service.planner_decisions else None
        final_valid=bool(final and final.get('status')=='planned')
        selection='registered_skill' if response['metadata'].get('generated_skill_id') else 'direct_action'
    else:
        final=service.selector_decisions[-1]['response'] if service.selector_decisions else None
        final_valid=response.get('fallback_reason') is None and response['metadata'].get('selection_source')=='live_llm'
        selection='registered_skill' if response['metadata'].get('generated_program_executed') else 'direct_action'
    return dict(retrieval_status=(view or {}).get('audit',{}).get('retrieval_status'),
        retrieval_stop=(view or {}).get('audit',{}).get('stop_reason'),
        read_document_ids=(view or {}).get('audit',{}).get('read_document_ids',[]),
        search_operations=sum(r['operation'].get('tool')=='search' for r in operations),
        read_operations=sum(r['operation'].get('tool')=='read' for r in operations),
        tool_errors=[r['result']['error'] for r in operations if 'error' in r['result']],
        final_decision_valid=final_valid,decision_type=selection,
        selected_skill_id=response['metadata'].get('generated_skill_id'),
        action_proposed=response.get('choice') is not None and response.get('fallback_reason') is None,
        fallback_reason=response.get('fallback_reason'),
        plan_status=final.get('status') if final else None,
        physics_attempts=0,physical_success='NOT_RUN',
        reading_is_not_proven_causal_use=True)


def run_roles(folder,budget,request,data,backends=None):
    results={}
    for role in ('scout','planner'):
        if budget.transport_blocked():
            results[role]=dict(status='NOT_RUN_TRANSPORT_CIRCUIT_OPEN');break
        print('START_ROLE='+role,flush=True)
        sub=folder/role;sub.mkdir(exist_ok=False)
        base=budget.transport(role,(backends or {}).get(role))
        count=0
        def transport(url,headers,body,timeout):
            nonlocal count
            count+=1
            write_new(sub/f'provider_body_{count:02}.json',body)
            write_new(sub/f'attempt_{count:02}.json',dict(model=body['model'],endpoint=url,
                timeout_seconds=timeout,format=body['text']['format']['name'],retries=0))
            print(f'API_ATTEMPT role={role} ordinal={count} format={body["text"]["format"]["name"]}',flush=True)
            return base(url,headers,body,timeout)
        transport.live=base.live;transport.circuit_status=base.circuit_status
        cfg=budget.policy['roles'][role]
        if role=='scout':
            actor=ExplicitActionService(cfg,sub/'requests.sqlite',transport,timeout_seconds=120,
                provider_contract='explicit_action_provider_v5',fallback_policy='path_stage_v1')
            actor.service._live_transport=transport.live
            selector=SkillDiscoveryService(cfg,sub/'requests.sqlite',transport);selector._live_transport=transport.live
            service=GeneratedSkillService(data['skills'],actor,selector,lesson_library=data['lessons'],
                knowledge_policy='agent_tools_v1',knowledge_memory=data['memory'],knowledge_max_rounds=2,timeout_seconds=120)
        else:
            planner=SolPlanService(cfg,sub/'requests.sqlite',transport,provider_contract='conditional_plan_v4',
                planning_feedback_policy='repair_evidence_v2');planner._live_transport=transport.live
            # Scout quota is zero in this planning-only saved-state phase.
            scout=ExplicitActionService(cfg,sub/'unused_scout.sqlite',mode='deterministic')
            service=HierarchicalDecisionService(scout,planner,data['skills'],budget.policy['episode'],
                timeout_seconds=120,journal_dir=sub/'journal',lesson_library=data['lessons'],lesson_runtime=True,
                recovery_support_policy='preserve_existing_v1',planning_feedback_policy='repair_evidence_v2',
                knowledge_policy='agent_tools_v1',knowledge_memory=data['memory'],knowledge_max_rounds=2,
                transport_guard=budget.transport_blocked)
        started=time.monotonic()
        try:
            response=service.handle(deepcopy(request))
            summary=summarize_role(service,response)
            summary.update(elapsed_seconds=round(time.monotonic()-started,3),status='COMPLETED')
            write_new(sub/'response.json',response)
        except Exception as error:
            summary=dict(status='ERROR',error_type=type(error).__name__,physics_attempts=0)
        finally:
            write_new(sub/'knowledge_contexts.json',service.knowledge_queries)
            write_new(sub/'decisions.json',service.decisions)
            if role=='planner':write_new(sub/'plans.json',service.planner_decisions)
            else:write_new(sub/'selections.json',service.selector_decisions)
            service.close()
        results[role]=summary;write_new(sub/'summary.json',summary)
        print(json.dumps(dict(role=role,**summary),ensure_ascii=False),flush=True)
        if summary['status']=='ERROR':break
    return results


def execute(folder):
    folder=checked(folder,True);protocol=read(folder/'protocol.json')
    if protocol['max_calls']!=6 or protocol['max_usd']!='1.00' or protocol['physics_calls']!=0:
        raise ValueError('smoke_scope_changed')
    for name,expected in protocol['prepared_files'].items():
        if sha(folder/name)!=expected:raise ValueError('prepared_input_changed')
    if source_hashes()!=protocol['source_hashes']:raise ValueError('source_changed_after_preparation')
    # One-shot claim before credentials/network; interrupted trials are never resumed.
    write_new(folder/'execution_claim.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    budget=HierarchyBudget(folder/'global_budget.sqlite',read(folder/'policy.json'),transport_failure_policy='latch_on_error_v1')
    result=dict(status='ERROR',physics_calls=0,model_quality='UNVERIFIED',results={})
    try:
        with key_environment() as origin:
            if origin=='missing':raise ValueError('stored_key_unavailable')
            # Only credential source label, never the key itself.
            write_new(folder/'credential_source.json',dict(source=origin,value_recorded=False))
            result['results']=run_roles(folder,budget,read(folder/'request.json'),
                {kind:read(folder/(kind+'.json')) for kind in ('memory','skills','lessons')})
        values=list(result['results'].values())
        result['status']='PASS' if len(values)==2 and all(v.get('final_decision_valid') and
            v.get('retrieval_status')=='ready' for v in values) else 'PARTIAL_OR_FAILED'
    except Exception as error:
        result['error_type']=type(error).__name__
    finally:
        result['budget']=budget.summary()
        result['sources_unchanged']=source_hashes()==protocol['source_hashes']
        result['actual_api_calls']=result['budget']['live_api_attempts']
        result['full_disassembly']='NOT_RUN';result['performance_effect']='UNVERIFIED'
        write_new(folder/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='budget'},ensure_ascii=False),flush=True)
    return 0 if result['status']=='PASS' and result['sources_unchanged'] else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('prepare','run'))
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--allow-live-api',action='store_true')
    args=p.parse_args(argv)
    if args.operation=='prepare':
        if args.allow_live_api:raise ValueError('prepare_never_calls_api')
        prepare(args.run_dir);return 0
    if not args.allow_live_api:raise ValueError('explicit_live_flag_required')
    return execute(args.run_dir)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print('SMOKE_ERROR_TYPE='+type(error).__name__,flush=True);raise SystemExit(1)
