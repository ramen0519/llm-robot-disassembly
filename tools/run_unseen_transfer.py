"""Fixed twelve-run pilot, one immutable paid ledger; no automatic reruns."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'worker')]
from prepare_transfer_pilot import RUN,safe,sha,write,preserve
from run_recorded import run_recorded
from run_integrated_repair import WSL,PYTHON,ASAP
from run_luna_experiment import linux


def read(path):return json.loads(safe(path).read_text(encoding='utf-8'))


def event(kind,**values):
    with (RUN/'batch_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),kind=kind,**values),ensure_ascii=False)+'\n')
    print(json.dumps(dict(kind=kind,**values),ensure_ascii=False),flush=True)


def setup():
    prepared=read(RUN/'prepared_inputs.json');selection=read(RUN/'selection.json')
    settings=dict(budget=40,state_action_budget=40,max_gripper=3,max_pose=128,timeout=900,wall_timeout=960,response_timeout=60)
    snapshot=ROOT/'runs/knowledge_store/revisions/rev_0001'
    for name in ('memory.json','skills.json'):
        dest=RUN/'knowledge'/name;dest.parent.mkdir(exist_ok=True)
        with dest.open('xb') as f:f.write((snapshot/name).read_bytes())
    config=dict(enabled=True,model='gpt-6-luna',base_url='https://api.openai.com/v1',api_key_env='OPENAI_API_KEY',
                max_calls=360,max_usd=5,input_usd_per_million=.1,output_usd_per_million=.5,
                max_output_tokens=1536,reasoning_effort='none',request_overhead_tokens=2048)
    write(RUN/'luna_config.json',config)
    order=[];rng=random.Random(20261019)
    for item in selection['selected']:
        conditions=list('ABCD');rng.shuffle(conditions)
        order.extend(dict(assembly=item['id'],condition=c) for c in conditions)
    manifest=dict(schema='unseen_transfer_v1',purpose='evaluation',settings=settings,automatic_retries=0,
        assembly_dirs=prepared['assembly_dirs'],path_limit=240,stability_limit=120,
        memory_sha256=sha(RUN/'knowledge/memory.json'),skills_sha256=sha(RUN/'knowledge/skills.json'),
        order=order,seed=0,api_calls=360,api_usd=5,body_bytes_limit=100000,
        conditions={'A':'original_ASAP_dfs_heur_out','B':'same_information_rule_memory_skill',
                    'C':'Luna_no_long_term_knowledge','D':'Luna_frozen_memory_skill'},
        budget_amendment='Before any evaluation: native probes six directions per composite, so 40 path calls prevents >=10-part completion. Shared path240/stability120 limits, composite/explicit40, same900s wall. API360/$5 unchanged.',
        same_seed_not_identical_rng_stream=True,no_skill_generation=True)
    write(RUN/'protocol.json',manifest)
    event('protocol_ready',order=order,new_API_calls=0)


def common_args(assembly,out,manifest,mode):
    from run_explicit_action_experiment import parser
    spec=read(manifest);settings=spec['settings']
    args=parser().parse_args(['--run-dir',str(out),'--mode',mode,'--assembly-dir',assembly,
        '--base-part',spec.get('development_base_part') or 'none',
        '--provider-contract','explicit_action_provider_v5','--fallback-policy','path_stage_v1',
        '--path-selection-policy','path_probe_round_v1','--stability-diagnostics',
        '--memory-policy','transfer_v1','--memory-max-records','8','--evaluation-manifest',str(manifest)])
    for k,v in settings.items():setattr(args,k,v)
    return args


def check_pin(spec):
    if sha(RUN/'knowledge/memory.json')!=spec['memory_sha256'] or sha(RUN/'knowledge/skills.json')!=spec['skills_sha256']:
        raise ValueError('frozen_knowledge_changed')


def explicit(assembly,condition,out,manifest):
    from run_explicit_action_experiment import execute
    from orchestrator.explicit_action_service import ExplicitActionService
    from orchestrator.generated_skill_service import GeneratedSkillService
    from orchestrator.transfer_policy import TransferRuleActor,TransferRuleSkills
    from orchestrator.transfer_services import TransferSkillSelector,bound_actor
    from orchestrator.skill_usage import summarize_usage
    spec=read(manifest);check_pin(spec)
    args=common_args(assembly,out,manifest,'deterministic' if condition=='B' else 'llm')
    if condition in 'BD':args.memory=RUN/'knowledge/memory.json'
    if condition in 'CD':
        args.llm_config=RUN/'luna_config.json';args.ledger=RUN/'api_ledger.sqlite';args.allow_live_api=True
    library=read(RUN/'knowledge/skills.json');services=[]
    def factory(config,ledger,parameters):
        if condition=='B':service=TransferRuleSkills(library,TransferRuleActor(parameters.response_timeout),mode='deterministic')
        else:
            actor=bound_actor(ExplicitActionService(config,ledger,mode='llm',timeout_seconds=parameters.response_timeout,
                provider_contract='explicit_action_provider_v5',fallback_policy='path_stage_v1'))
            service=(actor if condition=='C' else GeneratedSkillService(library,actor,TransferSkillSelector(config,ledger),mode='llm'))
        services.append(service);return service
    code=execute(args,service_factory=factory,extension_metadata=dict(kind='unseen_transfer_v1',condition=condition,
        frozen_memory_sha256=spec['memory_sha256'] if condition in 'BD' else None,
        frozen_skills_sha256=spec['skills_sha256'] if condition in 'BD' else None,
        protocol_sha256=sha(manifest),evaluation_first_episode=True))
    check_pin(spec)
    if services:
        service=services[0];decisions=service.decisions;selections=getattr(service,'selector_decisions',[])
        write(out/'generated_skill_selections.json',selections)
        report=read(out/'worker_result.json') if (out/'worker_result.json').exists() else None
        write(out/'skill_usage_summary.json',summarize_usage(decisions,report,selections,library if condition in 'BD' else None))
    return code


def native_command(assembly,out,manifest,unmetered=False):
    argv=WSL+['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(read(manifest)['settings']['wall_timeout'])+'s',
        PYTHON,'-B','-u',linux(ROOT/'worker/run_transfer_native.py'),'--asap-root',ASAP,
        '--assembly-dir',assembly,'--output-dir',linux(out),'--evaluation-manifest',linux(manifest)]
    if unmetered:argv.append('--unmetered-regression')
    return argv


def preflight():
    from orchestrator.explicit_action_service import build_provider_body
    from orchestrator.transfer_services import TransferSkillSelector,compact_body
    from orchestrator.generated_skill_service import GeneratedSkillService
    from orchestrator.transfer_policy import TransferRuleActor
    from orchestrator.skill_registry import retrieve
    from worker.explicit_action_contract import canonical_response
    from worker.explicit_request_budget import capacity_probe
    spec=read(RUN/'protocol.json');config=read(RUN/'luna_config.json');library=read(RUN/'knowledge/skills.json')
    selector=TransferSkillSelector(config,RUN/'preview_only_empty_ledger.sqlite')
    service=GeneratedSkillService(library,TransferRuleActor(),mode='deterministic')
    results=[]
    for assembly in spec['assembly_dirs']:
        ident=assembly.split('/')[-1];out=RUN/'preflight'/ident
        argv=WSL+[PYTHON,'-B','-u',linux(ROOT/'worker/run_explicit_actions.py'),'--asap-root',ASAP,
            '--assembly-dir',assembly,'--output-dir',linux(out),'--run-id','preview:'+ident,
            '--evaluation-manifest',linux(RUN/'protocol.json'),'--request-preflight-only',
            '--frozen-memory',linux(RUN/'knowledge/memory.json'),'--memory-policy','transfer_v1','--memory-max-records','8',
            '--path-selection-policy','path_probe_round_v1','--fallback-policy','path_stage_v1']
        for key in ('budget','state_action_budget','max_gripper','max_pose','timeout','response_timeout'):
            argv+=['--'+key.replace('_','-'),str(spec['settings'][key])]
        # Reuse a successful geometry-only preview; do not overwrite failed host logs.
        if not (out/'preflight_initial_request.json').exists():
            result=run_recorded(argv,ROOT,RUN/'commands'/('preflight_'+ident))
            if result['exit_code']!=0:raise RuntimeError('preflight_failed:'+ident)
        initial=read(out/'preflight_initial_request.json')
        stress,_=capacity_probe(initial,attempted_count=40,episode_count=12,memory_count=0)
        # Stress histories are synthetic, while source Memory remains actual frozen evidence.
        stress['memory_references']=deepcopy(initial['memory_references'])
        stress['evidence_ids']=sorted(set(stress['evidence_ids'])|{r['evidence_id'] for r in stress['memory_references']})
        for label,request in (('initial',initial),('synthetic_history',stress)):
            for condition in ('C','D'):
                r=deepcopy(request)
                if condition=='C':
                    removed={v['evidence_id'] for v in r['memory_references']};r['memory_references']=[]
                    r['evidence_ids']=[v for v in r['evidence_ids'] if v not in removed]
                body=compact_body(build_provider_body(config,r,'explicit_action_provider_v5'))
                row=dict(assembly=ident,kind=label,condition=condition,actor_bytes=len(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()))
                if condition=='D':
                    response=canonical_response(r,None,policy='path_stage_v1')
                    applicable=[(s,dict(response=response,rule_index=0)) for s in retrieve(library)]
                    packet=service.selection_packet(r,applicable)
                    preview=selector.preview(packet,'select');row['selector_bytes']=preview['body_bytes']
                results.append(row)
        event('input_preview_ok',assembly=ident,body_bytes=results[-1])
    write(RUN/'payload_preview.json',dict(results=results,api_calls=0,synthetic_history_not_observed=True,
        all_under_100000=True,real_CAD=True,new_physics_calls=0))


def freeze():
    source={}
    for folder in ('worker','orchestrator','tools','tests'):
        for p in sorted((ROOT/folder).glob('*.py')):
            name=p.relative_to(ROOT).as_posix();source[name]=sha(p)
            target=RUN/'frozen_sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(p.read_bytes())
    write(RUN/'evaluation_source_pin.json',source)
    write(RUN/'evaluation_artifact_pin.json',{name:sha(RUN/name) for name in ('protocol.json','selection.json','luna_config.json','knowledge/memory.json','knowledge/skills.json','payload_preview.json')})


def verify_pin():
    for name,h in read(RUN/'evaluation_source_pin.json').items():
        if sha(ROOT/name)!=h:raise ValueError('evaluation_source_changed:'+name)
    for name,h in read(RUN/'evaluation_artifact_pin.json').items():
        if sha(RUN/name)!=h:raise ValueError('evaluation_artifact_changed:'+name)


def batch():
    verify_pin();spec=read(RUN/'protocol.json')
    for index,item in enumerate(spec['order'],1):
        verify_pin();ident,condition=item['assembly'],item['condition']
        out=RUN/'evaluations'/ident/condition
        if out.exists():raise FileExistsError('no_automatic_rerun:'+str(out))
        assembly=next(k for k in spec['assembly_dirs'] if k.endswith('/'+ident))
        event('evaluation_start',index=index,total=12,assembly=ident,condition=condition)
        if condition=='A':argv=native_command(assembly,out,RUN/'protocol.json')
        else:argv=[sys.executable,'-B','-u',str(Path(__file__)),'one','--assembly',ident,'--condition',condition]
        result=run_recorded(argv,ROOT,RUN/'commands'/('evaluate_'+ident+'_'+condition))
        event('evaluation_end',index=index,assembly=ident,condition=condition,exit_code=result['exit_code'],seconds=result['elapsed_seconds'])
        if result['exit_code']!=0:raise RuntimeError('evaluation_infrastructure_failure:'+ident+':'+condition)
    verify_pin();event('batch_complete',runs=12)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('setup','preflight','freeze','batch','one','dev_rule'))
    p.add_argument('--assembly');p.add_argument('--condition',choices=list('BCD'));args=p.parse_args()
    if args.action=='setup':setup()
    elif args.action=='preflight':preflight()
    elif args.action=='freeze':freeze()
    elif args.action=='batch':batch()
    elif args.action=='dev_rule':
        return explicit('/opt/asap-research/unseen_transfer_pilot_20261019/development_beam','B',RUN/'development/rule',RUN/'development_protocol.json')
    else:
        verify_pin();spec=read(RUN/'protocol.json');assembly=next(k for k in spec['assembly_dirs'] if k.endswith('/'+args.assembly))
        return explicit(assembly,args.condition,RUN/'evaluations'/args.assembly/args.condition,RUN/'protocol.json')
    return 0


if __name__=='__main__':raise SystemExit(main())
