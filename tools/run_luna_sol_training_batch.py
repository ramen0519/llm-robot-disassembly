"""Six fresh training episodes, then source-bound learning; no held-out runs.

Separate 80-call/$10 authorization. Reuses the existing worker, VM, registry,
role selection and immutable budget gates. Every phase has an exclusive claim.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read, write, sha
from tools.run_multitask_learning import (PREP, PROPOSAL, now, episode_args,
    episode_summary, learn, verify_pins)
from tools.run_automatic_learning_cycle import runtime_hashes
from tools.run_abstention_revision_cycle import recorded_command
from tools.multitask_skill_evidence import training_memory_sources
from orchestrator.hierarchy_budget import HierarchyBudget, validate_policy

MAX_SECONDS = 7200
MAX_PHYSICS = 112  # six * 16 episode actions + two * four * (trigger + repair)
TRAIN = ['07009', '02628', '01289']


def policy():
    value = read(PROPOSAL/'policy_DISABLED.json')
    value.update(enabled=False, max_calls=80, max_usd='10.00')
    for role, count, cap in [('scout',12,'.30'),('planner',48,'9.20'),('learner',20,'.50')]:
        value['roles'][role].update(max_calls=count, max_usd=cap)
    value['episode'].update(scout_calls=1, planner_calls=6, fallback_calls=0)
    return validate_policy(value)


def prepare(output):
    output = checked(output, True)
    if output.exists(): raise FileExistsError('new_training_preparation_required')
    rows = read(PROPOSAL/'schedule_DISABLED.json')['episodes'][:6]
    if [(r['task'],r['seed']) for r in rows] != [(t,s) for s in (0,1) for t in TRAIN]:
        raise ValueError('six_predeclared_training_episodes_required')
    refs = [PROPOSAL/'schedule_DISABLED.json', PREP/'prepared_v2/split.json']
    refs += list((PROPOSAL/'empty_knowledge').glob('*.json'))
    for row in rows:
        path = checked(row['manifest'], True)
        if sha(path) != row['manifest_sha256'] or row['purpose'] != 'training':
            raise ValueError('training_input_changed')
        refs.append(path)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'policy_DISABLED.json', policy())
    write(output/'schedule.json', dict(episodes=rows))
    protocol = dict(version='luna_sol_training_80_v2', training_ids=TRAIN,
        evaluation_ids=['02680','02656'], evaluation='NOT_RUN_reserved_for_later',
        max_api_calls=80, max_program_usd='10.00', max_seconds=MAX_SECONDS,
        max_physics_attempts=MAX_PHYSICS, automatic_http_retries=0,
        knowledge_policy='agent_tools_v1', observation_contract='source_bound_v1',
        collection_knowledge='empty_frozen_knowledge; pooled_learning_after_six_episodes',
        role_selection='observed_states_v3: two actual post-PASS no-trigger states; two products with shared measured failure',
        candidate_limit=2, runtime=runtime_hashes(),
        input_pins={str(p):sha(p) for p in refs})
    write(output/'protocol.json', protocol)
    print('PREPARED six training episodes; max80/$10; API0/physics0', flush=True)
    return protocol


def validate(prepared):
    prepared = checked(prepared, True)
    protocol = read(prepared/'protocol.json')
    if (protocol['version'] != 'luna_sol_training_80_v2' or protocol['training_ids'] != TRAIN
            or protocol['max_api_calls'] != 80 or protocol['max_program_usd'] != '10.00'
            or protocol['max_seconds'] != MAX_SECONDS or protocol['max_physics_attempts'] != MAX_PHYSICS
            or protocol['automatic_http_retries'] != 0
            or protocol['observation_contract'] != 'source_bound_v1'
            or protocol['knowledge_policy'] != 'agent_tools_v1'
            or read(prepared/'policy_DISABLED.json') != policy()):
        raise ValueError('approved_training_scope_changed')
    if runtime_hashes() != protocol['runtime']: raise ValueError('runtime_changed_before_training')
    for path, expected in protocol['input_pins'].items():
        if sha(checked(path, True)) != expected: raise ValueError('training_input_changed')
    rows = read(prepared/'schedule.json')['episodes']
    if rows != read(PROPOSAL/'schedule_DISABLED.json')['episodes'][:6]:
        raise ValueError('training_schedule_changed')
    return protocol


def physics_attempts(output):
    return (len(list(output.glob('episodes/*/planner/attempt_*/request.json')))
            + len(list(output.glob('candidate_*/cases/*/*/input.json'))))


def collect(output, invoke, pool, rows):
    result = dict(status='COLLECTING', episodes=[])
    for row in rows:
        args = episode_args(row, output) + ['--knowledge-policy','agent_tools_v1',
            '--knowledge-max-rounds','2','--knowledge-max-skills','16']
        code = invoke(row['id'], 'run_hierarchical_episode.py', args, 660)
        value = episode_summary(row, output, code)
        result['episodes'].append(value)
        write(output/('collection_'+row['id']+'.json'), value)
        print('COLLECTED task='+row['task']+' seed='+str(row['seed'])+' status='+value['status']+
              ' physical='+str(value['physical_attempts'])+' accepted='+str(value['accepted_removals']),flush=True)
        if code != 0 or not value['validated'] or not value['memory'] or pool.summary().get('transport_guard'):
            result.update(status='STOPPED', reason='episode_error_or_transport_guard')
            break
    else:
        result['status']='COLLECTED'
    sources = [r['memory'] for r in result['episodes'] if r['memory'] and read(r['memory']['path'])['records']]
    if sources:
        from orchestrator.explicit_action_memory import freeze_cumulative_memory
        paths = training_memory_sources(read(PREP/'prepared_v2/split.json'), sources)
        frozen = freeze_cumulative_memory(paths, [], output/'training_memory.json')
        result['memory_records']=len(frozen['records'])
    write(output/'collection.json', result)
    return result


def stages(output, prepared, invoke, pool):
    collected = collect(output, invoke, pool, read(prepared/'schedule.json')['episodes'])
    result = dict(collection=collected, learning=dict(status='NOT_RUN'),
        evaluation='NOT_RUN', generalization='UNVERIFIED')
    if collected['status'] != 'COLLECTED': return dict(result,status='COLLECTION_STOPPED')
    from tools.prepare_multitask_roles import prepare as prepare_roles
    try:
        manifest = prepare_roles(output/'collection.json', PREP/'prepared_v2/split.json', output/'learning_inputs',
                                 role_policy='observed_states_v3')
    except ValueError as error:
        # Do not choose easier replacement products or fabricate failure triggers.
        result['learning']=dict(status='NOT_RUN',reason=str(error)[:240])
        return dict(result,status='NO_ELIGIBLE_TRAINING_ROLES')
    # The role preparer's old diagnostic projection is not the generation input.
    # learn() binds actual source clocks before the first paid lesson request.
    learned = learn(output, invoke, pool,
        learning_inputs=(manifest, manifest.parent/'lesson_packet.json'),
        observation_contract='source_bound_v1')
    write(output/'learn_result.json', learned)
    print('LEARNING '+learned['status'],flush=True)
    return dict(result,status=learned['status'],learning=learned)


def execute(prepared, output, allow_live):
    if not allow_live: raise ValueError('explicit_current_training_authorization_required')
    protocol = validate(prepared)
    prepared, output = checked(prepared, True), checked(output, True)
    if output.exists(): raise FileExistsError('new_training_execution_required_no_resume')
    write(prepared/'execution_claim.json',dict(output=str(output),utc=now(),automatic_resume=False))
    output.mkdir(parents=True,exist_ok=False)
    active=policy();active['enabled']=True
    write(output/'policy.json',active)
    write(output/'protocol.json',protocol)
    write(output/'approval.json',dict(user_message='오케이 일단 누나랑 솔 조합으로 바로 시작해보자.',
        interpretation='accept prior three products x two episodes, maximum80 calls/$10 proposal',
        max_calls=80,max_program_usd='10.00',automatic_http_retries=0,
        endpoint='https://api.openai.com/v1/responses',old_budget_reused=False,
        transfer_scope='same previously authorized official training three products and derived observations/skills',
        scope='six collection episodes then at most two skill candidates; no held-out evaluation'))
    write(output/'runtime_before.json',runtime_hashes())
    pins=dict(protocol['input_pins'])
    for name in ('protocol.json','schedule.json','policy_DISABLED.json'):pins[str(prepared/name)]=sha(prepared/name)
    write(output/'input_pins.json',pins)
    for role,cfg in active['roles'].items():write(output/(role+'_config.json'),cfg)
    write(output/'clock.json',dict(started_utc=now(),max_seconds=MAX_SECONDS))
    deadline=time.monotonic()+MAX_SECONDS
    pool=HierarchyBudget(output/'hierarchy.sqlite',active,transport_failure_policy='latch_on_error_v1')
    def invoke(name,tool,args,limit):
        verify_pins(output)
        if pool.summary().get('transport_guard'):raise RuntimeError('transport_guard_no_further_paid_calls')
        reserve={'run_hierarchical_episode.py':16,'run_python_skill_cycle.py':8,'run_lesson_discovery.py':0}[tool]
        if physics_attempts(output)+reserve>MAX_PHYSICS:raise RuntimeError('physics_cap')
        if deadline-time.monotonic()<limit:raise TimeoutError('insufficient_remaining_stage_time')
        print('STAGE '+name,flush=True)
        from tools.local_api_key import key_environment
        with key_environment(ROOT) as origin:
            if origin=='missing':raise ValueError('stored_key_unavailable')
            return recorded_command(output,deadline,name,tool,args,limit)
    try:
        result=stages(output,prepared,invoke,pool)
    except Exception as error:
        result=dict(status='STOPPED',error_type=type(error).__name__,reason=str(error)[:400],automatic_retry=False)
    result.update(api_budget=pool.summary(),finished_utc=now(),physical_attempts=physics_attempts(output),
        source_unchanged=runtime_hashes()==read(output/'runtime_before.json'),
        evaluation='NOT_RUN',performance_superiority='UNVERIFIED')
    write(output/'result.json',result)
    print('RESULT '+result['status']+' calls='+str(result['api_budget']['live_api_attempts']),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','execute'))
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--allow-live-api',action='store_true')
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.prepared)
    else:
        if args.output is None:parser.error('--output required')
        execute(args.prepared,args.output,args.allow_live_api)
