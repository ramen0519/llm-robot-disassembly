"""One authorized abstention revision, verified lesson and free 02628 reuse.

No generated-code patching, retry, implicit extra candidate or inherited budget.
Old split-cycle modes, policies, inputs and outputs remain unchanged.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read, write, sha
from tools.run_automatic_learning_cycle import runtime_hashes, task_argv
from tools.remaining_split_budget import validate_remaining_binding, claim_remaining_binding
from tools.python_revision_parent import historical_packet
from tools.run_python_skill_cycle import python_generation_packet
from orchestrator.python_abstention import attach_feedback
from orchestrator.hierarchy_budget import HierarchyBudget, validate_policy

TARGET = dict(assembly_dir='/opt/asap-research/integrated_repair_20261018/02628',
    base_part='none', seed=0, budget=16, state_action_budget=6, max_gripper=3, max_pose=128,
    pose_catalog_policy='complete_v1', timeout=480, wall_timeout=600, response_timeout=120)
FILES = {'parent','manifest','draft_lesson','memory','prior_library','prior_lessons'}


def validate(protocol, policy):
    required = {'schema_version','budget_parent','budget_reallocation','files','source_hashes',
                'target','max_physics_calls','total_timeout_seconds','http_retries'}
    if (set(protocol) != required or protocol['schema_version'] != 1
            or protocol['target'] != TARGET or protocol['max_physics_calls'] != 22
            or protocol['total_timeout_seconds'] != 2400 or protocol['http_retries'] != 0
            or protocol['budget_reallocation'].get('kind') != 'abstention_candidate_5_v1'
            or set(protocol['files']) != FILES):
        raise ValueError('explicit_bounded_abstention_protocol_required')
    validate_remaining_binding(protocol['budget_parent'], policy,
                               reallocation=protocol['budget_reallocation'])
    paths = {}
    for key, ref in protocol['files'].items():
        path = checked(ref['path'], True)
        if sha(path) != ref['sha256']:
            raise ValueError('changed_bound_input:' + key)
        paths[key] = path
    if runtime_hashes() != protocol['source_hashes']:
        raise ValueError('runtime_changed')
    manifest = read(paths['manifest'])
    if (len(manifest['cases']) != 3 or manifest.get('experience_task') != '02628'
            or [c['phase'] for c in manifest['cases']] != ['debug','admission','admission']):
        raise ValueError('predeclared_three_conditions_required')
    parent = read(paths['parent'])
    if 'preflight_sha256' not in parent:
        raise ValueError('abstention_parent_required')
    # Exact old code/input/audit is replayed before any new paid request.
    packet = historical_packet(python_generation_packet(manifest, 'abstention-residual-preflight'), parent, manifest)
    attach_feedback(packet)
    return paths


def recorded_command(output, deadline, name, tool, args, limit):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('total_40_minute_limit')
    log = output / (name + '_command'); log.mkdir(exist_ok=False)
    argv = [sys.executable, '-X', 'utf8', '-B', '-u', str(ROOT/'tools'/tool), *map(str,args)]
    record = dict(command=argv, cwd=str(ROOT), started_utc=datetime.now(timezone.utc).isoformat(),
                  timeout_seconds=min(limit, remaining), timeout=False)
    write(log/'command.json',record)
    started = time.monotonic()
    with (log/'stdout.log').open('xb') as out, (log/'stderr.log').open('xb') as err:
        child = subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            code = child.wait(timeout=record['timeout_seconds'])
        except subprocess.TimeoutExpired:
            record['timeout'] = True
            stop = subprocess.run([r'C:\Windows\System32\taskkill.exe','/PID',str(child.pid),'/T','/F'],
                capture_output=True,timeout=20,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            record['tree_stop_exit_code'] = stop.returncode
            code = child.wait(timeout=20)
    record.update(exit_code=code,elapsed_seconds=time.monotonic()-started)
    write(log/'completion.json',record)
    if record['timeout']:
        raise TimeoutError(name+'_timeout_no_retry')
    return code


def stages(output, protocol, paths, role_paths, invoke):
    """Injectable process boundary for API-free stop/ordering tests."""
    from orchestrator.skill_library_merge import freeze_skill_union
    from orchestrator.lesson_memory import freeze_lessons, validate_library, freeze_bundle
    from orchestrator.explicit_action_memory import freeze_cumulative_memory
    report = dict(status='NO_NEW_SKILL',new_skill_ids=[],target='NOT_RUN',
        new_skill_attempts=0,new_skill_passes=0,target_plan_success=None,
        performance_comparison='NOT_RUN',generalization='UNVERIFIED',http_retries=0,
        forced_failure=False,forced_skill_selection=False)
    shared = ['--mode','live','--allow-live-api','--ledger',output/'learner.sqlite',
        '--hierarchy-policy',output/'policy.json','--hierarchy-ledger',output/'hierarchy.sqlite',
        '--transport-failure-policy','latch_on_error_v1']
    candidate = output/'candidate_01'
    args = shared + ['--run-dir',candidate,'--llm-config',role_paths['learner'],
        '--manifest',paths['manifest'],'--lesson',paths['draft_lesson'],
        '--prior-library',paths['prior_library'],'--prior-lessons',paths['prior_lessons'],
        '--historical-debug-parent',paths['parent'],'--python-revision-contract',
        '--python-revision-version','2','--abstention-feedback',
        '--generation-timeout-seconds','120','--generation-output-tokens','4096']
    code = invoke('candidate_01','run_python_skill_cycle.py',args,1200)
    value = read(candidate/'result.json') if (candidate/'result.json').is_file() else {}
    report['candidate_result'] = value
    if (code != 0 or value.get('status') != 'REGISTERED_SCOPED'
            or value.get('admission',{}).get('admitted') is not True
            or value.get('actual_llm_generation') is not True
            or value.get('source_unchanged') is not True):
        report['reason'] = 'candidate_not_registered_no_further_calls'
        return report
    report.update(status='TECHNICAL_ADMISSION_ONLY',new_skill_ids=[value['candidate_id']])
    finalized = output/'lesson_finalized'
    args = shared + ['--run-dir',finalized,'--config',role_paths['learner'],
        '--consolidate-library',candidate/'skills_frozen.json','--draft',paths['draft_lesson']]
    code = invoke('lesson_finalize','run_lesson_discovery.py',args,180)
    final = read(finalized/'result.json') if (finalized/'result.json').is_file() else {}
    if code != 0 or final.get('status') != 'CANDIDATE':
        report['reason'] = 'lesson_not_finalized_no_target_calls'
        return report
    old_skills = read(paths['prior_library'])
    old_lessons = validate_library(read(paths['prior_lessons']), old_skills)
    library, lessons = output/'skills_frozen.json', output/'lessons_frozen.json'
    frozen = freeze_skill_union([paths['prior_library'],candidate/'skills_frozen.json'],library)
    lesson_snapshot = freeze_lessons([e['lesson'] for e in old_lessons['entries']] +
                                    [read(checked(final['lesson'],True))], frozen)
    write(lessons,lesson_snapshot)
    identity = value['candidate_id']
    report['status'] = 'PUBLISHED'
    args = ['--run-dir',output/'next_episode','--mode','llm','--allow-live-api',
        '--hierarchy-policy',output/'policy.json','--hierarchy-ledger',output/'hierarchy.sqlite',
        '--ledger',output/'scout.sqlite','--planner-ledger',output/'planner.sqlite',
        '--llm-config',role_paths['scout'],'--transport-failure-policy','latch_on_error_v1',
        '--skill-library',library,'--lesson-library',lessons,'--lesson-runtime',
        '--max-skill-attempts','3','--memory',paths['memory'],'--memory-policy','transfer_v1',
        '--memory-max-records','4','--provider-contract','explicit_action_provider_v5',
        '--fallback-policy','path_stage_v1','--path-selection-policy','path_probe_round_v1',
        '--stability-diagnostics','--plan-provider-contract','conditional_plan_v4',
        '--recovery-support-policy','preserve_existing_v1','--planning-feedback-policy','repair_evidence_v2',
        *task_argv(protocol['target']),'--pose-catalog-policy',protocol['target']['pose_catalog_policy']]
    code = invoke('next_episode','run_hierarchical_episode.py',args,660)
    folder = output/'next_episode'
    if code != 0:
        report.update(target='PROCESS_FAILED')
        return report
    audit, hierarchy = read(folder/'audit.json'), read(folder/'hierarchy_audit.json')
    if (not audit['validation_ok'] or not audit['actual_physics']
            or not hierarchy['library_unchanged'] or not hierarchy['lesson_library_unchanged']
            or not all(read(folder/'host_preservation.json').values())):
        raise ValueError('target_evidence_invalid')
    bindings = hierarchy['physics_bindings']
    report.update(status='COMPLETED',target=audit['status'],target_plan_success=audit['plan_success'],
        new_skill_attempts=sum(b['skill_id']==identity for b in bindings),
        new_skill_passes=sum(b['skill_id']==identity and b['accepted'] for b in bindings),
        source_memory_visible_requests=hierarchy['memory_visible_requests'],
        memory_cited_requests=hierarchy['memory_cited_requests'])
    memory = output/'memory_after_cycle.json'
    freeze_cumulative_memory([paths['memory'],folder/'action_memory_frozen.json'],[],memory)
    freeze_bundle(memory,library,lessons,output/'knowledge_bundle.json')
    return report


def execute(output, protocol, policy_path, *, allow_live=False):
    policy = validate_policy(read(checked(policy_path)))
    if not allow_live or not policy['enabled']:
        raise ValueError('explicit_enabled_residual_permission_required')
    paths = validate(protocol,policy)
    output = checked(output,True); output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',protocol); write(output/'policy.json',policy)
    write(output/'budget_parent_claim.json',claim_remaining_binding(protocol['budget_parent'],policy,output,
        reallocation=protocol['budget_reallocation']))
    role_paths = {}
    for role,config in policy['roles'].items():
        role_paths[role] = output/(role+'_config.json');write(role_paths[role],config)
    pool = HierarchyBudget(output/'hierarchy.sqlite',policy,transport_failure_policy='latch_on_error_v1')
    deadline = time.monotonic()+protocol['total_timeout_seconds']
    def invoke(name,tool,args,limit):
        print('STAGE: '+name,flush=True)
        if tool == 'run_hierarchical_episode.py':
            from tools.local_api_key import key_environment
            with key_environment(ROOT):
                return recorded_command(output,deadline,name,tool,args,limit)
        return recorded_command(output,deadline,name,tool,args,limit)
    try:
        result = stages(output,protocol,paths,role_paths,invoke)
    except (ValueError,OSError,RuntimeError,TimeoutError,KeyError) as error:
        result = dict(status='STOPPED',reason=str(error)[:300],target='UNCONFIRMED_READ_STAGE_LOGS',http_retries=0)
    result.update(api_budget=pool.summary(),source_unchanged=runtime_hashes()==protocol['source_hashes'],
        pinned_inputs_unchanged=all(sha(paths[k])==protocol['files'][k]['sha256'] for k in paths))
    write(output/'result.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--hierarchy-policy',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--allow-live-api',action='store_true')
    a=p.parse_args()
    result=execute(a.run_dir,read(checked(a.protocol)),a.hierarchy_policy,allow_live=a.allow_live_api)
    print(result['status'],flush=True)
    raise SystemExit(0 if result['status']=='COMPLETED' else 1)
