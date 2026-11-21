"""Continue one failed skill with separate correction slots, then free reuse.

No source episode rerun, no HTTP retry, no target calls unless publication
succeeds. Requires a new explicitly enabled policy and --allow-live-api.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read, write, sha
from tools.run_recorded import run_recorded
from tools.run_automatic_learning_cycle import runtime_hashes, validate_plan, task_argv
from tools.python_revision_parent import historical_packet, classify_candidate
from tools.run_python_skill_cycle import python_generation_packet
from orchestrator.split_revision_budget import SplitRevisionBudget
from orchestrator.hierarchy_budget import validate_policy, HierarchyBudget


def candidate_slots(protocol):
    mode=protocol.get('mode')
    if mode not in (None,'single_candidate_4_v1','single_candidate_3_v1'):
        raise ValueError('unsupported_split_mode')
    return SplitRevisionBudget(protocol['physical_revisions'],protocol['contract_corrections'],
                               single_candidate=mode in ('single_candidate_4_v1','single_candidate_3_v1'))


def remaining_seconds(protocol):
    if protocol.get('mode') != 'single_candidate_3_v1':
        return None
    deadline = datetime.fromisoformat(protocol['deadline_utc'])
    if deadline.utcoffset() is None:
        raise ValueError('explicit_utc_deadline_required')
    left = (deadline-datetime.now(timezone.utc)).total_seconds()
    if not 0 < left <= 7200:
        raise ValueError('three_call_deadline_expired_or_unbounded')
    return left


def validate_protocol(protocol, policy):
    required = {'schema_version', 'physical_revisions', 'contract_corrections', 'files',
                'source_hashes', 'target', 'max_physics_calls', 'http_retries'}
    optional = {'generation_contract_version', 'generation_timeout_seconds'}
    if (not required <= set(protocol) or set(protocol) - required - optional - {'budget_parent','mode','budget_reallocation','deadline_utc'}
            or (set(protocol) & optional) not in (set(), optional)
            or protocol['schema_version'] != 1 or protocol['http_retries'] != 0):
        raise ValueError('invalid_split_protocol')
    if optional <= set(protocol) and (type(protocol['generation_contract_version']) is not int
            or protocol['generation_contract_version'] not in (1, 2, 3, 4)
            or type(protocol['generation_timeout_seconds']) is not int
            or protocol['generation_timeout_seconds'] not in (30, 60, 90, 120)):
        raise ValueError('invalid_explicit_generation_options')
    three = protocol.get('mode') == 'single_candidate_3_v1'
    single=protocol.get('mode') in ('single_candidate_4_v1','single_candidate_3_v1')
    if (('mode' in protocol or 'budget_reallocation' in protocol) and
            (not single or 'budget_reallocation' not in protocol or 'budget_parent' not in protocol)):
        raise ValueError('single_candidate_requires_explicit_reallocation')
    if single and protocol.get('generation_contract_version') not in (3, 4):
        raise ValueError('single_candidate_requires_direction_contract_v3')
    if three:
        if (protocol.get('generation_contract_version') != 4 or 'deadline_utc' not in protocol
                or protocol['budget_reallocation'].get('kind') != 'single_candidate_3_v1'):
            raise ValueError('three_call_v4_and_deadline_required')
        remaining_seconds(protocol)
        auth = read(checked(protocol['budget_reallocation']['authorization_path'], True))
        if auth.get('deadline_utc') != protocol['deadline_utc']:
            raise ValueError('deadline_must_match_explicit_authorization')
    elif 'deadline_utc' in protocol:
        raise ValueError('deadline_requires_three_call_mode')
    if 'budget_parent' in protocol:
        from tools.remaining_split_budget import validate_remaining_binding
        validate_remaining_binding(protocol['budget_parent'], policy,
                                   reallocation=protocol.get('budget_reallocation'))
    slots = candidate_slots(protocol)
    expected_files = {'parent', 'manifest', 'draft_lesson', 'memory', 'prior_library'}
    if set(protocol['files']) != expected_files:
        raise ValueError('explicit_bound_inputs_required')
    paths = {}
    for name, ref in protocol['files'].items():
        path = checked(Path(ref['path']))
        if sha(path) != ref['sha256']:
            raise ValueError('protocol_input_hash_mismatch:' + name)
        paths[name] = path
    if runtime_hashes() != protocol['source_hashes']:
        raise ValueError('protocol_runtime_changed')
    template = dict(schema_version=1, purpose='scoped split revision', max_proposals=1,
        admission_manifest=str(paths['manifest']), initial_library=None, initial_memory=None,
        source=protocol['target'], target=protocol['target'])
    validate_plan(template)
    target = protocol['target']
    if not target['assembly_dir'].endswith('/07009') or target['base_part'] != 'none':
        raise ValueError('explicit_07009_without_base_required')
    manifest = read(paths['manifest'])
    if protocol['max_physics_calls'] != 2 * len(manifest['cases']) * slots.physical_revisions + target['budget']:
        raise ValueError('physics_bound_mismatch')
    expected = dict(learner=slots.physical_revisions + slots.contract_corrections + 1,
                    scout=0 if three else 1, planner=1 if single else 2)
    if (policy['max_calls'] != sum(expected.values())
            or any(policy['roles'][k]['max_calls'] != v for k, v in expected.items())
            or policy['episode']['scout_calls'] != expected['scout'] or policy['episode']['planner_calls'] != expected['planner']
            or policy['episode']['fallback_calls'] != 0):
        raise ValueError('separate_allocations_do_not_match_shared_budget')
    # Reads and validates only historical DEBUG evidence before the first API.
    historical_packet(python_generation_packet(manifest, 'split-preflight'), read(paths['parent']), manifest)
    return slots, paths


def run_candidates(output, protocol, paths, role_paths, invoke):
    slots = candidate_slots(protocol)
    base = ['--historical-debug-parent', paths['parent']]
    correction = []
    candidates, admitted = [], None
    while (allocation := slots.allocate()) is not None:
        folder = output / ('candidate_%02d' % (len(candidates) + 1))
        # Append-only allocation written BEFORE invoking a subprocess. No resume
        # mode: an interrupted unknown request cannot be automatically reissued.
        write(output / ('allocation_%02d.json' % (len(candidates) + 1)),
              dict(allocation=allocation, budget=slots.summary()))
        args = ['--run-dir', folder, '--manifest', paths['manifest'], '--mode', 'live',
            '--allow-live-api', '--llm-config', role_paths['learner'], '--ledger', output/'learner.sqlite',
            '--hierarchy-policy', output/'policy.json', '--hierarchy-ledger', output/'hierarchy.sqlite',
            '--lesson', paths['draft_lesson'], '--prior-library', paths['prior_library'],
            '--python-revision-contract', '--generation-output-tokens', '4096', *base, *correction]
        if 'generation_contract_version' in protocol:
            args += ['--python-revision-version', protocol['generation_contract_version'],
                     '--generation-timeout-seconds', protocol['generation_timeout_seconds']]
        code = invoke(folder.name, 'run_python_skill_cycle.py', args)
        if not (folder/'result.json').is_file():
            slots.observe('stop')
            candidates.append(dict(folder=str(folder), allocation=allocation, exit_code=code,
                                   outcome='stop', reason='missing_result_no_retry'))
            break
        result = read(folder/'result.json')
        outcome = classify_candidate(folder, result, output/'learner.sqlite')
        if code not in (0, 1):
            outcome = 'stop'
        candidates.append(dict(folder=str(folder), allocation=allocation, exit_code=code,
                               outcome=outcome, result=result))
        slots.observe(outcome)
        if outcome == 'registered':
            admitted = folder
            break
        if result.get('status') == 'GENERATION_REJECTED':
            correction = ['--generation-feedback-from', folder]
            # Keep latest measured physics feedback as well as format feedback.
        else:
            base, correction = ['--refine-from', folder], []
    value = dict(candidates=candidates, allocation=slots.summary(),
                 admitted_folder=str(admitted) if admitted else None)
    write(output/'learning_result.json', value)
    return value


def execute(output, protocol, policy_path, *, allow_live=False):
    policy = validate_policy(read(checked(policy_path)))
    if not allow_live or not policy['enabled']:
        raise ValueError('new_enabled_policy_and_explicit_live_permission_required')
    slots, paths = validate_protocol(protocol, policy)
    output = checked(output, True)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'protocol.json', protocol)
    write(output/'policy.json', policy)
    if 'budget_parent' in protocol:
        from tools.remaining_split_budget import claim_remaining_binding
        write(output/'budget_parent_claim.json', claim_remaining_binding(protocol['budget_parent'], policy, output,
              reallocation=protocol.get('budget_reallocation')))
    role_paths = {}
    for role, config in policy['roles'].items():
        role_paths[role] = output/(role+'_config.json')
        write(role_paths[role], config)
    pool = HierarchyBudget(output/'hierarchy.sqlite', policy)

    def invoke(name, tool, args):
        remaining_seconds(protocol)
        print('STAGE: ' + name, flush=True)
        record = run_recorded([sys.executable, '-B', '-u', str(ROOT/'tools'/tool),
                               *map(str, args)], ROOT, output/(name+'_command'))
        return record['exit_code']

    learning = run_candidates(output, protocol, paths, role_paths, invoke)
    report = dict(status='NO_NEW_SKILL', new_skill_ids=[], target='NOT_RUN',
        new_skill_attempts=0, new_skill_passes=0, target_plan_success=None,
        memory_effect='NOT_RUN', performance_comparison='NOT_RUN', http_retries=0)
    admitted = learning['admitted_folder']
    if admitted:
        args = ['--consolidate-library', Path(admitted)/'skills_frozen.json',
            '--draft', paths['draft_lesson'], '--run-dir', output/'lesson_finalized', '--mode', 'live',
            '--allow-live-api', '--config', role_paths['learner'], '--ledger', output/'learner.sqlite',
            '--hierarchy-policy', output/'policy.json', '--hierarchy-ledger', output/'hierarchy.sqlite']
        code = invoke('lesson_finalize', 'run_lesson_discovery.py', args)
        final_path = output/'lesson_finalized/result.json'
        final = read(final_path) if final_path.is_file() else {}
        report['status'] = 'TECHNICAL_ADMISSION_ONLY'
        if code == 0 and final.get('status') == 'CANDIDATE':
            from orchestrator.skill_library_merge import freeze_skill_union
            from orchestrator.lesson_memory import freeze_lessons, freeze_bundle
            from orchestrator.explicit_action_memory import freeze_cumulative_memory
            library, lessons = output/'skills_frozen.json', output/'lessons_frozen.json'
            frozen = freeze_skill_union([Path(admitted)/'skills_frozen.json'], library)
            lesson_snapshot = freeze_lessons([read(Path(final['lesson']))], frozen)
            if len(frozen['entries']) != 1 or len(lesson_snapshot['entries']) != 1:
                raise ValueError('one_new_skill_and_bound_final_lesson_required')
            write(lessons, lesson_snapshot)
            identity = read(Path(admitted)/'result.json')['candidate_id']
            report.update(status='PUBLISHED', new_skill_ids=[identity])
            args = ['--run-dir', output/'next_episode', '--mode', 'llm',
                '--hierarchy-policy', output/'policy.json', '--hierarchy-ledger', output/'hierarchy.sqlite',
                '--planner-ledger', output/'planner.sqlite', '--ledger', output/'scout.sqlite',
                '--skill-library', library, '--lesson-library', lessons, '--lesson-runtime',
                '--max-skill-attempts', '3', '--memory', paths['memory'], '--memory-policy', 'transfer_v1',
                '--memory-max-records', '4', '--provider-contract', 'explicit_action_provider_v5',
                '--fallback-policy', 'path_stage_v1', '--path-selection-policy', 'path_probe_round_v1',
                '--stability-diagnostics', '--allow-live-api', '--llm-config', role_paths['scout'],
                *task_argv(protocol['target'])]
            code = invoke('next_episode', 'run_hierarchical_episode.py', args)
            folder = output/'next_episode'
            if code != 0:
                report.update(target='PROCESS_FAILED')
            else:
                audit, hierarchy = read(folder/'audit.json'), read(folder/'hierarchy_audit.json')
                if (not audit['validation_ok'] or not audit['actual_physics']
                        or not hierarchy['library_unchanged'] or not hierarchy['lesson_library_unchanged']
                        or not all(read(folder/'host_preservation.json').values())):
                    raise ValueError('target_evidence_invalid')
                bindings = hierarchy['physics_bindings']
                report.update(target=audit['status'], target_plan_success=audit['plan_success'],
                    new_skill_attempts=sum(b['skill_id'] == identity for b in bindings),
                    new_skill_passes=sum(b['skill_id'] == identity and b['accepted'] for b in bindings),
                    source_memory_visible_requests=hierarchy['memory_visible_requests'],
                    memory_cited_requests=hierarchy['memory_cited_requests'])
                memory = output/'memory_after_cycle.json'
                freeze_cumulative_memory([paths['memory'], folder/'action_memory_frozen.json'], [], memory)
                freeze_bundle(memory, library, lessons, output/'knowledge_bundle.json')
    report.update(api_budget=pool.summary(), learning=learning,
        source_unchanged=runtime_hashes() == protocol['source_hashes'])
    write(output/'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--hierarchy-policy', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--allow-live-api', action='store_true')
    args = parser.parse_args()
    result = execute(args.run_dir, read(checked(args.protocol)), args.hierarchy_policy,
                     allow_live=args.allow_live_api)
    print(result['status'], flush=True)
