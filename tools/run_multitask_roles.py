"""Explicit continuation: bounded old budget, v2 training roles, frozen evaluation.

Never resume the failed transport request or clear its latch. A unique child
allocation keeps the parent's unknown cost reserved. Prepare is offline only.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys,time
from datetime import datetime,timezone
ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read,write,sha
from tools.run_multitask_learning import (PROPOSAL,PREP,now,learn,evaluate,verify_pins)
from tools.run_abstention_revision_cycle import recorded_command
from tools.run_automatic_learning_cycle import runtime_hashes
from tools.remaining_split_budget import validate_remaining_binding,claim_remaining_binding
from tools.prepare_multitask_roles import residual_proposal
from tools.multitask_skill_evidence import training_memory_sources
from tools.run_python_skill_cycle import python_generation_packet
from tools.skill_nonapplicability import bind_cases
from orchestrator.hierarchy_budget import HierarchyBudget,validate_policy
from worker.task_memory import digest

READY=ROOT/'runs/multitask_roles_20261105/prepared'
PARENT=ROOT/'runs/multitask_learning_live_20261105'
BINDING_KEYS=('parent_ledger','parent_ledger_sha256','parent_policy_sha256','parent_rows_sha256')


def prepare(output):
    output=checked(output,True)
    if output.exists():raise FileExistsError('new_preparation_required')
    manifest=read(READY/'manifest.json')
    bind_cases(manifest)
    packet=python_generation_packet(manifest,'multitask-roles:'+output.parent.name)
    partial=read(PARENT/'partial_training_corpus/manifest.json')
    split=read(PREP/'prepared_v2/split.json')
    paths=training_memory_sources(split,partial['sources'])
    if len(paths)!=5 or partial['records']!=32:raise ValueError('declared_partial_training_corpus_changed')
    proposal=residual_proposal(PARENT/'live/hierarchy.sqlite')
    binding={k:proposal['parent'][k] for k in BINDING_KEYS}
    validate_remaining_binding(binding,proposal['policy'],restriction=True)
    refs=[READY/'manifest.json',PREP/'prepared_v2/split.json',PROPOSAL/'schedule_DISABLED.json',
          PARENT/'partial_training_corpus/manifest.json',PARENT/'explicit_transfer_consent.json']
    for ref in manifest['experience_sources']+manifest['nonapplicability_cases']:
        path=checked(ref['manifest_path'],True);refs.append(path)
        child=read(path);report=checked(child['experience_report'],True)
        refs.append(report)
        for a in report.parent.glob('attempt_*'):
            refs.extend([a/'request.json',a/'result.json'])
    refs.extend(paths)
    refs.extend((PROPOSAL/'empty_knowledge').glob('*.json'))
    schedule=read(PROPOSAL/'schedule_DISABLED.json')['episodes'][6:]
    for row in schedule:
        path=checked(row['manifest'],True)
        if sha(path)!=row['manifest_sha256']:raise ValueError('evaluation_manifest_changed')
        refs.append(path)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'policy_DISABLED.json',proposal['policy'])
    write(output/'lesson_packet.json',packet)
    from orchestrator.explicit_action_memory import freeze_cumulative_memory
    frozen=freeze_cumulative_memory(paths,[],output/'training_memory.json')
    if len(frozen['records'])!=32:raise ValueError('incomplete_partial_memory_binding')
    refs.extend([output/'lesson_packet.json',output/'training_memory.json',output/'policy_DISABLED.json'])
    protocol=dict(version='multitask_roles_continuation_v1',budget_parent=binding,
        max_new_calls=16,max_program_usd='4.80',max_seconds=5400,max_physics_attempts=80,
        manifest=str(READY/'manifest.json'),packet=str(output/'lesson_packet.json'),
        memory=str(output/'training_memory.json'),partial_collection=True,
        training_observations=32,collection_status_not_rewritten=True,
        original_transport_latch_reset=False,automatic_http_retries=0,
        evaluation_ids=['02680','02656'],evaluation_episodes=4,
        runtime=runtime_hashes(),input_pins={str(p):sha(p) for p in sorted(set(refs))})
    write(output/'protocol.json',protocol)
    write(output/'budget_audit.json',proposal)
    print('PREPARED roles v2; 32 training observations; max16/$4.80; API0/physics0',flush=True)
    return protocol


def validate(prepared):
    prepared=checked(prepared,True)
    protocol=read(prepared/'protocol.json')
    policy=validate_policy(read(prepared/'policy_DISABLED.json'))
    if (protocol['version']!='multitask_roles_continuation_v1' or policy['enabled']
            or policy['max_calls']!=16 or policy['max_usd']!='4.80'
            or protocol['max_new_calls']!=16 or protocol['max_program_usd']!='4.80'
            or protocol['max_seconds']!=5400 or protocol['max_physics_attempts']!=80
            or protocol['automatic_http_retries']!=0 or protocol['original_transport_latch_reset'] is not False
            or protocol['evaluation_ids']!=['02680','02656'] or protocol['evaluation_episodes']!=4):
        raise ValueError('approved_continuation_bounds_changed')
    validate_remaining_binding(protocol['budget_parent'],policy,restriction=True)
    if runtime_hashes()!=protocol['runtime']:raise ValueError('runtime_changed_since_preparation')
    if any(sha(checked(p,True))!=h for p,h in protocol['input_pins'].items()):raise ValueError('pinned_input_changed')
    for key in ('manifest','packet','memory'):
        if protocol[key] not in protocol['input_pins']:raise ValueError('unbound_input')
    manifest=read(checked(protocol['manifest'],True))
    bind_cases(manifest)
    stored=read(checked(protocol['packet'],True))
    if stored!=python_generation_packet(manifest,stored['run_id']):raise ValueError('generation_packet_binding_changed')
    return protocol,policy


def stages(output,protocol,invoke,pool):
    learned=learn(output,invoke,pool,learning_inputs=(Path(protocol['manifest']),Path(protocol['packet'])))
    write(output/'learn_result.json',learned)
    print('LEARNING '+learned['status'],flush=True)
    if learned['status']!='FROZEN':
        evaluated=dict(status='NOT_RUN',reason='no_published_skill')
    else:
        evaluated=evaluate(output,invoke,pool)
    write(output/'evaluate_result.json',evaluated)
    return dict(status=evaluated['status'] if learned['status']=='FROZEN' else learned['status'],
        learning=learned,evaluation=evaluated,performance_superiority='UNVERIFIED')


def execute(prepared,output,allow_live):
    if not allow_live:raise ValueError('explicit_continuation_permission_required')
    output=checked(output,True)
    if output.exists():raise FileExistsError('new_live_directory_required')
    protocol,policy=validate(prepared)
    policy['enabled']=True
    # Claim the remaining parent snapshot once. Never delete/release this on
    # failure, and never mutate or refund the failed parent's request.
    claim_remaining_binding(protocol['budget_parent'],policy,output,restriction=True)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'approval.json',dict(user_message='좋아 이어가자',
        authorized_original_max_calls=34,authorized_original_program_usd='12.00',
        child_max_calls=16,child_max_program_usd='4.80',parent=protocol['budget_parent'],
        external_transfer_scope='same five official assemblies; existing explicit consent',
        new_request_types=True,old_failed_request_resent=False,automatic_http_retries=0,utc=now()))
    write(output/'policy.json',policy);write(output/'protocol.json',protocol)
    write(output/'runtime_before.json',runtime_hashes())
    pins=dict(protocol['input_pins'])
    pins[str(checked(prepared,True)/'protocol.json')]=sha(checked(prepared,True)/'protocol.json')
    write(output/'input_pins.json',pins)
    for role,cfg in policy['roles'].items():write(output/(role+'_config.json'),cfg)
    with (output/'training_memory.json').open('xb') as f:f.write(Path(protocol['memory']).read_bytes())
    write(output/'clock.json',dict(started_utc=now(),max_seconds=5400))
    deadline=time.monotonic()+5400
    pool=HierarchyBudget(output/'hierarchy.sqlite',policy,transport_failure_policy='latch_on_error_v1')
    def invoke(name,tool,args,limit):
        verify_pins(output)
        validate_remaining_binding(protocol['budget_parent'],policy,restriction=True)
        if pool.summary().get('transport_guard'):raise RuntimeError('transport_error_no_further_paid_calls')
        print('STAGE '+name,flush=True)
        from tools.local_api_key import key_environment
        with key_environment(ROOT):return recorded_command(output,deadline,name,tool,args,limit)
    try:
        result=stages(output,protocol,invoke,pool)
    except Exception as error:
        result=dict(status='STOPPED',error_type=type(error).__name__,reason=str(error)[:400],automatic_retry=False)
    result.update(api_budget=pool.summary(),finished_utc=now(),
        source_unchanged=runtime_hashes()==read(output/'runtime_before.json'),
        original_budget_unchanged=sha(Path(protocol['budget_parent']['parent_ledger']))==protocol['budget_parent']['parent_ledger_sha256'])
    write(output/'result.json',result)
    print('RESULT '+result['status']+' calls='+str(result['api_budget']['live_api_attempts']),flush=True)
    return result


def continuation_physics_attempts(output):
    # Count started requests, including attempts that have no result after a crash.
    return (len(list(output.glob('candidate_*/cases/*/*/input.json')))
            +len(list(output.glob('episodes/*/planner/attempt_*/request.json'))))


def continuation_evaluation_allowed(output):
    # Four prior clock-probe attempts are already charged to the same 80 cap.
    # Reserve ALL four equal-budget evaluation episodes before starting any.
    return 4+continuation_physics_attempts(output)+4*16<=80


def continue_clock(parent,output,allow_live,*,source_case_clock=False):
    """One explicit host-correction continuation on the SAME 16-call ledger."""
    if not allow_live:raise ValueError('explicit_continuation_permission_required')
    parent,output=checked(parent,True),checked(output,True)
    if output.exists():raise FileExistsError('new_clock_run_required')
    policy=validate_policy(read(parent/'policy.json'));protocol=read(parent/'protocol.json')
    if policy['max_calls']!=16 or policy['max_usd']!='4.80':raise ValueError('same_16_call_budget_required')
    validate_remaining_binding(protocol['budget_parent'],policy,restriction=True)
    if any(sha(checked(p,True))!=h for p,h in read(parent/'input_pins.json').items()):
        raise ValueError('historical_inputs_changed')
    candidate=parent/'candidate_01'
    from tools.python_recorded_clock import replay_candidate
    replay_candidate(candidate,read(Path(protocol['manifest'])),parent/'learner.sqlite',policy['roles']['learner']['model'])
    pool=HierarchyBudget(parent/'hierarchy.sqlite',policy,transport_failure_policy='latch_on_error_v1')
    initial=pool.summary()
    if initial['live_api_attempts']!=2 or initial.get('transport_guard'):
        raise ValueError('two_completed_calls_no_transport_guard_required')
    start=datetime.fromisoformat(read(parent/'clock.json')['started_utc'])
    remaining=5400-(datetime.now(timezone.utc)-start).total_seconds()
    if remaining<=0 and not source_case_clock:raise TimeoutError('original_90_minute_deadline')
    old_remaining=remaining
    if source_case_clock:remaining=5400 # Explicit new user continuation; original clock untouched.
    claim='source_budget_continuation_claim.json' if source_case_clock else 'clock_continuation_claim.json'
    write(parent/claim,dict(output=str(output),utc=now(),
        same_shared_ledger=True,old_failed_http_request_replayed=False,new_generation_call_on_replay=False))
    output.mkdir(parents=True,exist_ok=False)
    write(output/'policy.json',policy);write(output/'runtime_before.json',runtime_hashes())
    write(output/'input_pins.json',read(parent/'input_pins.json'))
    write(output/'protocol.json',dict(protocol,clock_correction=True,source_clock='recorded original allowance',
        fresh_case_wall_seconds='bound_original_request' if source_case_clock else 45,
        previous_case_wall_seconds=45 if source_case_clock else 30,physical_thresholds_changed=False,
        shared_budget=str(parent/'hierarchy.sqlite'),remaining_global_seconds=remaining,
        user_continued_new_bounded_window=source_case_clock,old_clock_remaining_seconds=old_remaining,
        previous_physics_attempts=4 if source_case_clock else 0,initial_api_budget=initial,
        original_generation_reused_unchanged=True))
    if source_case_clock:
        write(output/'clock.json',dict(started_utc=now(),max_seconds=5400,old_clock_preserved=True))
        for name in ('hierarchy.sqlite','learner.sqlite'):
            with (output/('before_'+name)).open('xb') as f:f.write((parent/name).read_bytes())
    for role,cfg in policy['roles'].items():write(output/(role+'_config.json'),cfg)
    with (output/'training_memory.json').open('xb') as f:f.write((parent/'training_memory.json').read_bytes())
    deadline=time.monotonic()+remaining
    def invoke(name,tool,args,limit):
        verify_pins(output)
        validate_remaining_binding(protocol['budget_parent'],policy,restriction=True)
        if pool.summary().get('transport_guard'):raise RuntimeError('transport_error_no_further_paid_calls')
        if source_case_clock:
            reserve=8 if tool=='run_python_skill_cycle.py' else (16 if tool=='run_hierarchical_episode.py' else 0)
            if 4+continuation_physics_attempts(output)+reserve>80:
                raise RuntimeError('cumulative_80_physics_attempt_limit')
        print('STAGE '+name,flush=True)
        if '--replay-generated-from' in args:
            # This stage validates the actual saved code; no key or provider needed.
            return recorded_command(output,deadline,name,tool,args,limit)
        from tools.local_api_key import key_environment
        with key_environment(ROOT):return recorded_command(output,deadline,name,tool,args,limit)
    try:
        learned=learn(output,invoke,pool,learning_inputs=(Path(protocol['manifest']),Path(protocol['packet'])),
                      replay_from=candidate,budget_output=parent,source_case_clock=source_case_clock)
        write(output/'learn_result.json',learned)
        if learned['status']!='FROZEN':evaluated=dict(status='NOT_RUN',reason='no_published_skill')
        elif source_case_clock and not continuation_evaluation_allowed(output):
            evaluated=dict(status='NOT_RUN',reason='insufficient_physics_budget_for_four_equal_episodes')
        elif source_case_clock and deadline-time.monotonic()<4*660:
            evaluated=dict(status='NOT_RUN',reason='insufficient_time_for_four_equal_episodes')
        else:evaluated=evaluate(output,invoke,pool,budget_output=parent)
        write(output/'evaluate_result.json',evaluated)
        result=dict(status=evaluated['status'] if learned['status']=='FROZEN' else learned['status'],
                    learning=learned,evaluation=evaluated)
    except Exception as error:
        result=dict(status='STOPPED',error_type=type(error).__name__,reason=str(error)[:400],automatic_retry=False)
    result.update(api_budget=pool.summary(),initial_api_budget=initial,finished_utc=now(),
        new_physics_attempts=continuation_physics_attempts(output),
        previous_physics_attempts=4 if source_case_clock else 0,
        source_unchanged=runtime_hashes()==read(output/'runtime_before.json'))
    write(output/'result.json',result)
    print('RESULT '+result['status']+' calls='+str(result['api_budget']['live_api_attempts']),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--continue-clock-from',type=Path)
    parser.add_argument('--continue-source-clock-from',type=Path)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--allow-live-api',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        if args.allow_live_api:raise ValueError('prepare_is_always_offline')
        prepare(args.run_dir)
    elif args.continue_source_clock_from:continue_clock(args.continue_source_clock_from,args.run_dir,args.allow_live_api,source_case_clock=True)
    elif args.continue_clock_from:continue_clock(args.continue_clock_from,args.run_dir,args.allow_live_api)
    else:execute(args.prepared,args.run_dir,args.allow_live_api)
