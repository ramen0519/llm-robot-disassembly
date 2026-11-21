"""Approved 34-call multi-product collection, scoped learning, frozen evaluation.

Each phase has an exclusive claim: no automatic resume or replay after an
unknown outcome. The existing hierarchy ledger owns the common paid budget.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import sys,time
ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read,write,sha
from tools.run_abstention_revision_cycle import recorded_command
from tools.run_automatic_learning_cycle import runtime_hashes
from tools.multitask_skill_evidence import validate_roles,training_memory_sources
from orchestrator.hierarchy_budget import HierarchyBudget,validate_policy

PREP=ROOT/'runs/multitask_learning_prep_20261105'
PROPOSAL=PREP/'final_proposal'
FAMILIES=('dynamic_stability_failed','path_failed','initial_disconnected')


def now():return datetime.now(timezone.utc).isoformat()


def setup(output):
    output=checked(output,True)
    if output.exists():raise FileExistsError('new_multitask_run_required')
    policy=read(PROPOSAL/'policy_DISABLED.json');validate_policy(policy)
    if policy['enabled'] or policy['max_calls']!=34 or policy['max_usd']!='12.00':
        raise ValueError('approved_34_call_12_usd_proposal_required')
    protocol=read(PROPOSAL/'protocol_DISABLED.json')
    split=read(PREP/'prepared_v2/split.json')
    if sha(PREP/'prepared_v2/split.json')!=protocol['split_sha256']:raise ValueError('split_changed')
    if (protocol['training_ids']!=['07009','02628','01289'] or protocol['evaluation_ids']!=['02680','02656']
            or protocol['automatic_http_retries']!=0 or protocol['max_total_seconds']!=9000):
        raise ValueError('precommitted_scope_changed')
    schedule=read(PROPOSAL/'schedule_DISABLED.json')['episodes']
    if len(schedule)!=10:raise ValueError('six_collection_four_evaluation_required')
    refs=[PROPOSAL/'policy_DISABLED.json',PROPOSAL/'protocol_DISABLED.json',PROPOSAL/'schedule_DISABLED.json',
          PREP/'prepared_v2/split.json',PREP/'prepared_v2/prepared_inputs.json']
    for row in schedule:
        path=checked(row['manifest'],True)
        if sha(path)!=row['manifest_sha256']:raise ValueError('episode_manifest_changed')
        refs.append(path)
    refs += list((PROPOSAL/'empty_knowledge').glob('*.json'))
    output.mkdir(parents=True,exist_ok=False)
    policy['enabled']=True
    write(output/'policy.json',policy);write(output/'protocol.json',protocol)
    write(output/'approval.json',dict(approved_api_max=34,approved_program_usd='12.00',
        user_authorized_new_trial=True,automatic_http_retries=0,old_residual_reused=False,
        source='Current explicit user approval: 새 API 최대 34회/$12 승인',created_utc=now()))
    write(output/'input_pins.json',{str(p):sha(p) for p in refs})
    write(output/'runtime_before.json',runtime_hashes())
    for role,cfg in policy['roles'].items():write(output/(role+'_config.json'),cfg)
    return policy


def verify_pins(output):
    if runtime_hashes()!=read(output/'runtime_before.json'):raise ValueError('runtime_changed_during_trial')
    if any(sha(checked(p,True))!=h for p,h in read(output/'input_pins.json').items()):
        raise ValueError('precommitted_input_changed')


def episode_args(row,output,frozen=False,*,budget_output=None):
    args=deepcopy(row['argv'][6:])
    def put(flag,value):args[args.index(flag)+1]=str(value)
    run=output/'episodes'/row['id']
    knowledge=output/'frozen_training' if frozen else PROPOSAL/'empty_knowledge'
    for flag,value in {'--run-dir':run,'--hierarchy-policy':output/'policy.json',
        '--hierarchy-ledger':(budget_output or output)/'hierarchy.sqlite','--llm-config':output/'scout_config.json',
        '--ledger':output/(row['id']+'_scout.sqlite'),'--planner-ledger':output/(row['id']+'_planner.sqlite'),
        '--skill-library':knowledge/'skills.json','--lesson-library':knowledge/'lessons.json',
        '--memory':knowledge/'memory.json'}.items():put(flag,value)
    args.append('--allow-live-api')
    return args


def episode_summary(row,output,code):
    folder=output/'episodes'/row['id']
    audit=read(folder/'audit.json') if (folder/'audit.json').is_file() else {}
    hierarchy=read(folder/'hierarchy_audit.json') if (folder/'hierarchy_audit.json').is_file() else {}
    worker=read(folder/'worker_result.json') if (folder/'worker_result.json').is_file() else {}
    attempts=worker.get('attempts',[])
    memory=folder/'action_memory_frozen.json'
    return dict(id=row['id'],task_id=row['task'],purpose=row['purpose'],seed=row['seed'],condition=row['condition'],
        exit_code=code,status=worker.get('status',audit.get('status','NO_RESULT')),plan_success=audit.get('plan_success'),
        validated=audit.get('validation_ok',False),physical_attempts=len(attempts),
        accepted_removals=sum(a.get('accepted_for_transition') is True for a in attempts),
        path_calls=sum(a['raw'].get('path_check_calls',0) for a in attempts),
        stability_calls=sum(a['raw'].get('stability_check_calls',0) for a in attempts),
        generated_skill_attempts=hierarchy.get('generated_program_attempts',0),
        generated_skill_passes=hierarchy.get('generated_program_passes',0),
        memory_visible_requests=hierarchy.get('memory_visible_requests',0),
        memory_cited_requests=hierarchy.get('memory_cited_requests',0),
        hierarchy_role_attempts=hierarchy.get('role_attempts',{}),
        report=str(folder/'planner/planner_report.json'),
        memory=dict(task_id=row['task'],purpose=row['purpose'],path=str(memory),sha256=sha(memory)) if memory.is_file() else None)


def collect(output,invoke,pool):
    schedule=read(PROPOSAL/'schedule_DISABLED.json')['episodes'][:6];rows=[]
    for row in schedule:
        code=invoke(row['id'],'run_hierarchical_episode.py',episode_args(row,output),660)
        value=episode_summary(row,output,code);rows.append(value)
        write(output/('collection_'+row['id']+'.json'),value)
        print('COLLECTED '+row['task']+' seed='+str(row['seed'])+' status='+value['status']+
              ' removals='+str(value['accepted_removals']),flush=True)
        if pool.summary().get('transport_guard'):break
    result=dict(status='COLLECTED' if len(rows)==6 and not pool.summary().get('transport_guard') else 'STOPPED',episodes=rows)
    write(output/'collection.json',result)
    if result['status']=='COLLECTED':
        sources=[r['memory'] for r in rows if r['memory'] and read(Path(r['memory']['path']))['records']]
        if sources:
            from orchestrator.explicit_action_memory import freeze_cumulative_memory
            paths=training_memory_sources(read(PREP/'prepared_v2/split.json'),sources)
            freeze_cumulative_memory(paths,[],output/'training_memory.json')
    return result


def observed_family(attempt):
    raw=attempt.get('raw',{})
    if (raw.get('backend')!='ASAP_RedMax' or attempt.get('deadline_exceeded') is not False
            or attempt.get('status') not in ('UNKNOWN','REJECTED')):return None
    if raw.get('path_check_calls')==1 and raw.get('path_success') is False:return 'path_failed'
    if (raw.get('path_success') is True and raw.get('stability_success') is False
            and raw.get('stability_check_calls')==1):
        return {'dynamic_failure':'dynamic_stability_failed','initial_disconnected':'initial_disconnected'}.get(
            (raw.get('stability_diagnostics') or {}).get('phase'))
    return None


def choose_sources(collection):
    """Fixed family priority and seed/attempt order, never select on recovery success."""
    tasks=['07009','02628','01289'];found={f:{} for f in FAMILIES}
    for row in collection['episodes']:
        if row['purpose']!='training' or row['task_id'] not in tasks:raise ValueError('evaluation_in_training_collection')
        path=checked(row['report'],True)
        if not path.is_file() or not row.get('memory'):continue
        report=read(path)
        for index,attempt in enumerate(report.get('attempts',[]),1):
            family=observed_family(attempt)
            if family and row['task_id'] not in found[family]:
                found[family][row['task_id']]=dict(row=row,attempt_index=index)
    family=next((f for f in FAMILIES if set(found[f])==set(tasks)),None)
    return dict(selected_family=family,selected=found[family] if family else {},
        coverage={f:sorted(v) for f,v in found.items()},source_selection='first matching family, seed-major and attempt order',
        shared_cause_established=False)


def learning_manifest(output,selection):
    from tools.run_skill_discovery import source_case
    folder=output/'learning_inputs';folder.mkdir(exist_ok=False)
    family=selection['selected_family'];refs=[];cases=[]
    for index,task in enumerate(('07009','02628','01289')):
        selected=selection['selected'][task];row=selected['row'];memory=Path(row['memory']['path'])
        spec=dict(case_id='source_'+task,phase='debug' if index<2 else 'admission',source_task=task,
            report=Path(row['report']).relative_to(ROOT).as_posix(),attempt_index=selected['attempt_index'],
            trigger_kind='path_failed' if family=='path_failed' else 'stability_failed')
        if family!='path_failed':spec['expected_diagnostic_phase']='dynamic_failure' if family=='dynamic_stability_failed' else 'initial_disconnected'
        cases.append(spec)
        if index<2:
            child=dict(experience_task=task,experience_report=spec['report'],experience_memory=str(memory),cases=[spec],
                frozen_episode_memory=dict(path=str(memory),file_sha256=sha(memory),content_sha256=read(memory)['content_sha256']))
            if family!='path_failed':child['python_failure_scope']='remaining_body_stability_failed'
            path=folder/('source_'+task+'.json');write(path,child)
            refs.append(dict(task_id=task,manifest_path=str(path),manifest_sha256=sha(path)))
    cases.append(dict(cases[-1],case_id='source_01289_yaw90',pose_yaw_degrees=90))
    for spec in cases:source_case(spec)  # actual report/pose binding, no physics
    split=PREP/'prepared_v2/split.json'
    manifest=dict(multi_task_schema='development_sources_v1',split=dict(path=str(split),sha256=sha(split)),
        experience_sources=refs,cases=cases,require_shared_failure_family=True)
    path=folder/'manifest.json';write(path,manifest)
    from tools.run_python_skill_cycle import python_generation_packet
    packet=python_generation_packet(manifest,'multi-task-learning:'+output.name)
    write(folder/'packet.json',packet)
    return path,folder/'packet.json'


def learn(output,invoke,pool,*,learning_inputs=None,replay_from=None,budget_output=None,source_case_clock=False,
          observation_contract=None):
    if source_case_clock and (learning_inputs is None or replay_from is None):
        raise ValueError('source_clock_requires_bound_v2_replay')
    if learning_inputs is None:
        collected=read(output/'collection.json')
        if collected['status']!='COLLECTED':return dict(status='NOT_RUN',reason='collection_not_completed')
        choice=choose_sources(collected);write(output/'source_selection.json',choice)
        if not choice['selected_family']:return dict(status='NO_ELIGIBLE_SHARED_FAILURE',registration=False,evaluation='NOT_RUN')
        manifest,packet=learning_manifest(output,choice)
        debug_ids={'source_07009','source_02628'}
    else:
        manifest,packet=learning_inputs
        from tools.skill_nonapplicability import bind_cases
        specification=read(manifest)
        if specification.get('multi_task_schema') not in ('development_roles_v2','development_states_v3'):
            raise ValueError('explicit_training_roles_required')
        bind_cases(specification)
        debug_ids={c['case_id'] for c in specification['cases'] if c['phase']=='debug'}
    # Fresh generation and every subsequent revision share one source contract.
    # Historical replay keeps its explicitly selected old clock unless opted in.
    if observation_contract is None:
        observation_contract = 'legacy_v1' if replay_from else 'source_bound_v1'
    if observation_contract not in ('legacy_v1', 'source_bound_v1'):
        raise ValueError('unknown_learning_observation_contract')
    source_observation = observation_contract == 'source_bound_v1'
    if source_observation:
        from tools.run_skill_discovery import source_case
        from tools.python_source_budget import observation_binding
        bindings = {s['case_id']: observation_binding(source_case(s)) for s in read(manifest)['cases']}
        write(output/'learning_observation_contract.json', dict(version=observation_contract,
            cases=bindings, original_artifacts_unchanged=True,
            stage='preflight_before_paid_calls', binding_does_not_authorize_api=True,
            new_api_calls_by_binding=0, new_physics_calls_by_binding=0))
    budget_output=budget_output or output
    # Offline injected callers may omit this file; live children always require it.
    learner_config = read(output/'learner_config.json') if (output/'learner_config.json').is_file() else {}
    astra = learner_config.get('model') == 'gpt-6-astra'
    generation_tokens = learner_config['max_output_tokens'] if astra else 4096
    generation_timeout = 600 if astra else 120
    lesson_timeout = 660 if astra else 180
    candidate_timeout = 3700 if (source_case_clock or source_observation) else 1200
    recovery_protocol = read(manifest).get('recovery_protocol')
    if recovery_protocol is not None:
        from orchestrator.recovery_sequence import validate_protocol
        recovery_protocol = validate_protocol(recovery_protocol)
        if not source_observation:
            raise ValueError('bounded_recovery_requires_source_bound_learning')
        # A chain has an initial trigger plus up to three recovery actions.
        # Do not retain the old one-recovery subprocess allowance silently.
        import math
        candidate_timeout = math.ceil(len(read(manifest)['cases']) *
            recovery_protocol['max_total_seconds']) + generation_timeout + 120
    shared=['--mode','live','--allow-live-api','--ledger',budget_output/'learner.sqlite',
        '--hierarchy-policy',output/'policy.json','--hierarchy-ledger',budget_output/'hierarchy.sqlite',
        '--transport-failure-policy','latch_on_error_v1']
    draft=replay_from.parent/'lesson_draft' if replay_from else output/'lesson_draft'
    code=0 if replay_from else invoke('lesson_extract','run_lesson_discovery.py',shared+[
        '--config',output/'learner_config.json','--packet',packet,'--run-dir',draft],lesson_timeout)
    lesson=read(draft/'result.json') if (draft/'result.json').is_file() else {}
    if code!=0 or lesson.get('status')!='CANDIDATE':return dict(status='LESSON_REJECTED',evaluation='NOT_RUN')
    candidates=[];admitted=None;extra=[]
    for n in (1,2):
        candidate=output/('candidate_%02d'%n)
        clock_args=['--recorded-trigger-clock'] if replay_from else []
        if source_case_clock:clock_args+=['--source-case-clock']
        if source_observation:clock_args+=['--observation-contract','source_bound_v1']
        if replay_from and n==1:clock_args+=['--replay-generated-from',replay_from]
        args=shared+['--run-dir',candidate,'--manifest',manifest,'--llm-config',output/'learner_config.json',
            '--lesson',draft/'lesson.json','--prior-library',PROPOSAL/'empty_knowledge/skills.json',
            '--prior-lessons',PROPOSAL/'empty_knowledge/lessons.json','--python-revision-contract',
            '--python-revision-version','2','--generation-timeout-seconds',str(generation_timeout),
            '--generation-output-tokens',str(generation_tokens),*clock_args,*extra]
        code=invoke(candidate.name,'run_python_skill_cycle.py',args,candidate_timeout)
        value=read(candidate/'result.json') if (candidate/'result.json').is_file() else {}
        candidates.append(dict(path=str(candidate),exit_code=code,result=value))
        if (code==0 and value.get('status')=='REGISTERED_SCOPED' and value.get('admission',{}).get('admitted')
                and value.get('actual_llm_generation') and value.get('source_unchanged')):
            admitted=candidate;break
        if n==2 or pool.summary().get('transport_guard'):break
        if value.get('status')=='GENERATION_REJECTED':extra=['--generation-feedback-from',candidate]
        elif value.get('status') in ('PREFLIGHT_REJECTED','NOT_REGISTERED'):
            # Do not spend a revision based solely on hidden admission outcomes.
            if value['status']=='NOT_REGISTERED' and not any(c.get('case_id') in debug_ids
                and c.get('outcome')!='PASS' for c in value.get('cases',[])):break
            if (learning_inputs is not None and value['status']=='NOT_REGISTERED'
                    and not any(c.get('case_id') in debug_ids and c.get('outcome') in
                        ('UNKNOWN','REJECTED','NOT_APPLICABLE') for c in value.get('cases',[]))):
                # A missing/unreproduced physical trigger or harness error is
                # not evidence that the proposed repair algorithm failed.
                break
            extra=['--refine-from',candidate]
            if value['status']=='PREFLIGHT_REJECTED':
                if learning_inputs is None:
                    extra.append('--abstention-feedback') # preserve legacy behavior
                else:
                    saved=read(candidate/'development_preflight.json')
                    if any(c.get('runtime_reason')=='python_abstained' and c.get('runtime_error') is None
                           for c in saved.get('checks',[])):
                        extra.append('--abstention-feedback')
        else:break
    report=dict(status='NOT_REGISTERED',candidates=candidates,evaluation='NOT_RUN')
    if admitted is None:return report
    final=output/'lesson_finalized'
    code=invoke('lesson_finalize','run_lesson_discovery.py',shared+['--config',output/'learner_config.json',
        '--consolidate-library',admitted/'skills_frozen.json','--draft',draft/'lesson.json','--run-dir',final],lesson_timeout)
    value=read(final/'result.json') if (final/'result.json').is_file() else {}
    if code!=0 or value.get('status')!='CANDIDATE':return dict(report,status='ADMITTED_LESSON_NOT_PUBLISHED')
    from orchestrator.skill_library_merge import freeze_skill_union
    from orchestrator.lesson_memory import freeze_lessons,freeze_bundle
    frozen=output/'frozen_training';frozen.mkdir(exist_ok=False)
    skills=freeze_skill_union([admitted/'skills_frozen.json'],frozen/'skills.json')
    write(frozen/'lessons.json',freeze_lessons([read(final/'lesson.json')],skills))
    with (frozen/'memory.json').open('xb') as f:f.write((output/'training_memory.json').read_bytes())
    freeze_bundle(frozen/'memory.json',frozen/'skills.json',frozen/'lessons.json',frozen/'bundle.json')
    write(output/'frozen_pins.json',{str(p):sha(p) for p in frozen.glob('*.json')})
    return dict(report,status='FROZEN',new_skill_ids=[e['candidate']['candidate_id'] for e in skills['entries']],
        registered_on_training_only=True,evaluation='READY_NOT_RUN')


def evaluate(output,invoke,pool,*,budget_output=None):
    if read(output/'learn_result.json').get('status')!='FROZEN':return dict(status='NOT_RUN',reason='no_published_skill')
    rows=[]
    pins=read(output/'frozen_pins.json')
    for row in read(PROPOSAL/'schedule_DISABLED.json')['episodes'][6:]:
        if any(sha(checked(p,True))!=h for p,h in pins.items()):raise ValueError('frozen_training_changed')
        code=invoke(row['id'],'run_hierarchical_episode.py',episode_args(row,output,
            frozen=row['condition']=='frozen_training_memory_and_skills',budget_output=budget_output),660)
        value=episode_summary(row,output,code);rows.append(value)
        write(output/('evaluation_'+row['id']+'.json'),value)
        if pool.summary().get('transport_guard'):break
    unchanged=all(sha(checked(p,True))==h for p,h in pins.items())
    return dict(status='COMPLETED' if len(rows)==4 and unchanged else 'STOPPED',episodes=rows,
        frozen_bundle_unchanged=unchanged,evaluation_written_to_training=False,performance_superiority='UNVERIFIED')


def execute(output,phase,allow_live):
    output=checked(output,True)
    if not allow_live:raise ValueError('explicit_34_call_trial_permission_required')
    if phase=='collect':setup(output)
    verify_pins(output)
    policy=validate_policy(read(output/'policy.json'))
    if not policy['enabled'] or policy['max_calls']!=34 or policy['max_usd']!='12.00':raise ValueError('budget_changed')
    if phase=='collect':write(output/'clock.json',dict(started_unix=time.time(),max_seconds=9000))
    clock=read(output/'clock.json');remaining=clock['max_seconds']-(time.time()-clock['started_unix'])
    if remaining<=0:raise TimeoutError('global_150_minute_limit')
    write(output/(phase+'_claim.json'),dict(utc=now(),automatic_resume=False))
    deadline=time.monotonic()+remaining
    pool=HierarchyBudget(output/'hierarchy.sqlite',policy,transport_failure_policy='latch_on_error_v1')
    def invoke(name,tool,args,limit):
        verify_pins(output)
        if pool.summary().get('transport_guard'):raise RuntimeError('transport_error_no_further_calls')
        print('STAGE '+name,flush=True)
        from tools.local_api_key import key_environment
        with key_environment(ROOT):return recorded_command(output,deadline,name,tool,args,limit)
    try:result={'collect':collect,'learn':learn,'evaluate':evaluate}[phase](output,invoke,pool)
    except Exception as error:result=dict(status='STOPPED',error_type=type(error).__name__,reason=str(error)[:400],automatic_retry=False)
    result.update(phase=phase,api_budget=pool.summary(),finished_utc=now(),source_unchanged=runtime_hashes()==read(output/'runtime_before.json'))
    write(output/(phase+'_result.json'),result)
    print('PHASE_RESULT '+result['status']+' calls='+str(result['api_budget']['live_api_attempts']),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--phase',choices=('collect','learn','evaluate'),required=True);p.add_argument('--allow-live-api',action='store_true')
    a=p.parse_args();execute(a.run_dir,a.phase,a.allow_live_api)
