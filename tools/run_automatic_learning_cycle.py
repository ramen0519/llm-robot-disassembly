"""Whole episode -> actual observations -> skill development -> next whole episode.

The learning boundary is between solver runs. Generated programs never take
ownership of DFS or issue physics verdicts. Preview performs no API or physics.
Offline fixture mode is explicitly NOT an LLM generation/selection experiment.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from run_integrated_repair import checked, _excluded, ASAP
from run_luna_experiment import read, write, sha, CONFIG_KEYS
from run_recorded import run_recorded
from orchestrator.batch_budget import budget_summary
from orchestrator.explicit_action_memory import freeze_cumulative_memory, load_snapshot
from orchestrator.skill_registry import retrieve
from worker.task_memory import digest

TASK_FIELDS = {'assembly_dir', 'base_part', 'seed', 'budget', 'state_action_budget',
               'max_gripper', 'max_pose', 'timeout', 'wall_timeout', 'response_timeout'}
ASSEMBLIES = {ASAP + '/assets/beam_assembly/original',
              '/opt/asap-research/prototype_completion_20261018/07009',
              '/opt/asap-research/integrated_repair_20261018/02628',
              '/opt/asap-research/assembly_01256_learning_20261018/01256'}


def validate_plan(plan, *, research_profile=None):
    expected = {'schema_version', 'purpose', 'max_proposals', 'admission_manifest',
                'initial_memory', 'initial_library', 'source', 'target'}
    if research_profile:
        from orchestrator.research_profile import validate_profile
        validate_profile(research_profile)
    if (type(plan) is not dict or set(plan) != expected or type(plan['schema_version']) is not int
            or plan['schema_version'] != (2 if research_profile else 1)
            or type(plan['purpose']) is not str or not plan['purpose'].strip()):
        raise ValueError('invalid_cycle_plan')
    if type(plan['max_proposals']) is not int or not 1 <= plan['max_proposals'] <= 3:
        raise ValueError('bounded_proposal_count_required')
    for label in ('source', 'target'):
        task = plan[label]
        if research_profile:
            from orchestrator.research_profile import validate_task, EPISODE_SETTINGS
            validate_task(task)
            if task.get('evaluation_manifest'):
                from types import SimpleNamespace
                from worker.transfer_protocol import validate_manifest
                validate_manifest(checked(task['evaluation_manifest'], True), task['assembly_dir'],
                                  SimpleNamespace(**task, **EPISODE_SETTINGS))
            elif task['assembly_dir'] not in ASSEMBLIES:
                raise ValueError('research_task_requires_predeclared_input_manifest')
            continue
        if type(task) is not dict or set(task) != TASK_FIELDS:
            raise ValueError('invalid_cycle_task')
        if str(_excluded(task['assembly_dir'])) not in ASSEMBLIES:
            raise ValueError('only_previously_authorized_assemblies')
        if type(task['base_part']) is not str or not task['base_part'] or any(c in task['base_part'] for c in '/\\:'):
            raise ValueError('invalid_base_id')
        for key in TASK_FIELDS - {'assembly_dir', 'base_part'}:
            if type(task[key]) is not int or task[key] < (0 if key == 'seed' else 1):
                raise ValueError('invalid_task_integer')
        if (task['seed'] >= 2**32 or task['budget'] > 16 or task['state_action_budget'] > 16
                or task['max_gripper'] > 3 or task['max_pose'] > 3
                or task['timeout'] > 240 or task['wall_timeout'] > 300
                or task['response_timeout'] > 120):
            raise ValueError('small_cycle_bounds_exceeded')
    template = read(checked(plan['admission_manifest']))
    if len([c for c in template.get('cases', []) if c.get('phase') == 'admission']) != 2:
        raise ValueError('cycle_requires_exactly_two_predeclared_admissions')
    for key in ('initial_memory', 'initial_library'):
        if plan[key] is not None:
            checked(plan[key], True)
    return deepcopy(plan)


def limits(plan):
    source_factor = 2 if plan['initial_library'] else 1
    return {'max_api_attempts': source_factor * plan['source']['budget']
                + plan['max_proposals'] + 2 * plan['target']['budget'],
            'max_episode_physical_actions': plan['source']['budget'] + plan['target']['budget'],
            'max_validation_physical_actions': 6 * plan['max_proposals'],
            'automatic_http_retries': 0, 'max_synthesis_candidates': plan['max_proposals'],
            'partial_solver_resume': False, 'performance_comparison': 'NOT_RUN'}


def task_argv(task):
    keys = sorted(TASK_FIELDS | ({'evaluation_manifest'} if task.get('evaluation_manifest') else set()))
    return [arg for key in keys for arg in ('--' + key.replace('_', '-'), str(task[key]))]


def runtime_hashes():
    paths = []
    for name in ('orchestrator', 'worker', 'tools'):
        paths.extend(sorted((ROOT / name).glob('*.py')))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def live_inputs(config_path, ledger):
    config_path, ledger = checked(config_path), checked(ledger, True)
    config = read(config_path)
    if (set(config) != CONFIG_KEYS or config['enabled'] is not True or config['model'] != 'gpt-6-luna'
            or config['base_url'] != 'https://api.openai.com/v1' or config['api_key_env'] != 'OPENAI_API_KEY'):
        raise ValueError('pinned_complete_approved_config_required')
    if ledger.exists():
        prior = budget_summary(ledger)
        if prior['pending_requests'] or prior['unknown_outcomes']:
            raise ValueError('unresolved_api_requires_manual_review')
    return config_path, ledger, config


def execute_cycle(output, plan, *, mode, config_path=None, ledger=None, source_fixture=None):
    from orchestrator.learning_cycle_workflow import LearningCycleWorkflow
    from orchestrator.skill_library_merge import freeze_skill_union
    from prepare_episode_learning import prepare_episode_learning
    plan = validate_plan(plan)
    if mode not in ('live', 'offline-fixture'):
        raise ValueError('execute_requires_live_or_offline_fixture_mode')
    if (mode == 'live' and (not config_path or not ledger or source_fixture)
            or mode != 'live' and (config_path or ledger)):
        raise ValueError('invalid_execution_credentials_or_fixture')
    output = checked(output, True)
    output.mkdir(parents=True, exist_ok=True)
    live = mode == 'live'
    config = {'enabled': False, 'max_calls': 0, 'max_usd': 0}
    if live:
        config_path, ledger, config = live_inputs(config_path, ledger)
    refs = {'admission_manifest': sha(checked(plan['admission_manifest']))}
    for key in ('initial_memory', 'initial_library'):
        if plan[key]:
            refs[key] = sha(checked(plan[key], True))
    if source_fixture:
        refs['source_fixture'] = sha(checked(source_fixture, True))
    identity = {'schema_version': 1, 'plan': plan, 'mode': mode, 'config': config,
                'ledger': str(ledger) if ledger else None, 'input_hashes': refs,
                'source_hashes': runtime_hashes(), 'fixture_only': not live,
                'source_fixture': str(source_fixture) if source_fixture else None}
    if not (output / 'cycle_config.json').exists():
        write(output / 'cycle_config.json', identity)
    elif read(output / 'cycle_config.json') != identity:
        raise ValueError('cycle_identity_changed')

    def command(name, tool, args):
        argv = [sys.executable, '-B', '-u', str(ROOT / 'tools' / tool)] + list(map(str, args))
        record = run_recorded(argv, ROOT, output / (name + '_command'))
        return record['exit_code']

    def api_args():
        return ['--llm-config', str(config_path), '--ledger', str(ledger), '--allow-live-api'] if live else []

    def episode(name, task, memory, library, fixture=None):
        folder = output / name
        rows = retrieve(read(library), allow_fixture=not live) if library else []
        tool = 'run_generated_skills.py' if rows else 'run_explicit_action_experiment.py'
        args = ['--run-dir', folder, '--mode', 'llm' if live else 'fixture' if fixture else 'deterministic']
        args += task_argv(task) + ['--provider-contract', 'explicit_action_provider_v5',
                 '--fallback-policy', 'path_stage_v1', '--path-selection-policy', 'path_probe_round_v1',
                 '--stability-diagnostics', '--memory-policy', 'state_diverse_v1', '--memory-max-records', '8']
        if fixture:
            if rows:
                raise ValueError('source_fixture_and_library_cannot_mix')
            args += ['--fixture', fixture]
        if memory:
            args += ['--memory', memory]
        if rows:
            args += ['--skill-library', library]
            if not live:
                args += ['--allow-fixture-library']
        args += api_args()
        code = command(name, tool, args)
        if code != 0 or not (folder / 'audit.json').exists():
            raise RuntimeError('episode_failed_see_recorded_logs')
        audit = read(folder / 'audit.json')
        preservation = read(folder / 'host_preservation.json')
        if (not audit.get('validation_ok') or not audit.get('actual_physics')
                or not all(preservation.get(k) is True for k in ('source_unchanged', 'memory_unchanged'))):
            raise ValueError('episode_evidence_or_preservation_failed')
        memory_path = folder / 'action_memory_frozen.json'
        snapshot = load_snapshot(memory_path)
        generated_path = folder / 'generated_skills_audit.json'
        generated = read(generated_path) if generated_path.exists() else None
        if generated is not None and generated.get('library_unchanged') is not True:
            raise ValueError('episode_library_changed')
        return {'status': audit['status'], 'episode_dir': str(folder), 'plan_success': audit['plan_success'],
                'actual_physics': True, 'fixture_choice_source': bool(fixture),
                'memory_records': snapshot['record_count'], 'metrics': audit,
                'generated_skill_audit': generated,
                'artifacts': {'report': str(folder / 'worker_result.json'), 'audit': str(folder / 'audit.json'),
                    'memory': str(memory_path), 'preservation': str(folder / 'host_preservation.json'),
                    **({'generated_audit': str(generated_path)} if generated else {})}}

    def source(_):
        return episode('source_episode', plan['source'], plan['initial_memory'], plan['initial_library'], source_fixture)

    def prepare(state):
        folder = output / 'learning_inputs'
        prepared = prepare_episode_learning(state['source']['episode_dir'],
                                            checked(plan['admission_manifest']), folder)
        source_memory = state['source']['artifacts']['memory']
        merged_memory = output / 'memory_for_next_episode.json'
        snapshots = ([checked(plan['initial_memory'], True)] if plan['initial_memory'] else []) + [source_memory]
        frozen = freeze_cumulative_memory(snapshots, [], merged_memory)
        original_artifacts = {}
        for index, (path, expected_hash) in enumerate(prepared['original_artifact_sha256'].items()):
            path = checked(path)
            if sha(path) != expected_hash:
                raise ValueError('prepared_source_artifact_changed')
            original_artifacts['original_%d' % index] = str(path)
        return {**prepared, 'cumulative_memory_records': frozen['record_count'],
                'memory': str(merged_memory), 'artifacts': {
                    'memory': str(merged_memory),
                    'preparation': str(folder / 'preparation.json'), **original_artifacts,
                    **({'manifest': prepared['manifest_path']} if prepared.get('manifest_path') else {})}}

    def learn(state):
        prior_library = [checked(plan['initial_library'], True)] if plan['initial_library'] else []
        candidates, admitted, prior = [], [], None
        if state['prepare']['status'] == 'PREPARED':
            for index in range(plan['max_proposals'] if live else 1):
                name = 'candidate_%02d' % (index + 1)
                folder = output / name
                args = ['--run-dir', folder, '--manifest', state['prepare']['manifest_path'],
                        '--mode', 'live' if live else 'fixture'] + api_args()
                if prior is not None:
                    args += ['--refine-from', prior]
                code = command(name, 'run_skill_discovery.py', args)
                result_path = folder / 'result.json'
                if not result_path.exists():
                    raise RuntimeError('discovery_process_failed_without_result')
                result = read(result_path)
                if read(folder / 'source_preservation.json').get('unchanged') is not True:
                    raise ValueError('discovery_source_changed')
                candidates.append({'run_dir': str(folder), 'exit_code': code, 'status': result['status'],
                                   'candidate_id': result.get('candidate_id')})
                if code == 0 and result.get('admission', {}).get('admitted') is True:
                    admitted.append(folder / 'skills_frozen.json')
                    break
                # No same-payload HTTP retries. An invalid/API response stops
                # synthesis; only a valid proposal with debug observations can
                # receive a new bounded refinement request.
                if not live or result['status'] == 'GENERATION_REJECTED' or code != 0:
                    break
                proposal = read(folder / 'provider_proposal.json')
                if proposal.get('origin') != 'actual_llm' or proposal.get('status') != 'proposed':
                    break
                prior = folder
        library_path = output / 'library_for_next_episode.json'
        conflict = None
        try:
            snapshot = freeze_skill_union(prior_library + admitted, library_path, allow_fixture=not live)
        except ValueError as exc:
            if not str(exc).startswith('conflicting_frozen_skill_program:'):
                raise
            # Preserve the already frozen entry, including its scoped evidence.
            # The new independently registered card is retained in its run but
            # is not silently substituted into this next episode's library.
            conflict = str(exc)
            snapshot = freeze_skill_union(prior_library, library_path, allow_fixture=not live)
        new_ids = [c['candidate_id'] for c in candidates if c['status'] == 'REGISTERED_SCOPED'] if admitted and not conflict else []
        artifact_paths = {'library': str(library_path)}
        for index, item in enumerate(candidates):
            artifact_paths['candidate_%d' % index] = str(Path(item['run_dir']) / 'result.json')
        return {'status': 'REGISTERED' if admitted else 'NO_NEW_SKILL', 'candidates': candidates,
                'new_registered_count': len(admitted), 'available_skills': len(snapshot['entries']),
                'activated_new_skill_ids': new_ids, 'library_merge_conflict': conflict,
                'library': str(library_path), 'artifacts': artifact_paths}

    def target(state):
        return episode('next_episode', plan['target'], state['prepare']['memory'], state['learn']['library'])

    def finish(state):
        final_memory = output / 'memory_after_next_episode.json'
        frozen = freeze_cumulative_memory([state['prepare']['memory'], state['target']['artifacts']['memory']], [], final_memory)
        generated = state['target']['generated_skill_audit'] or {}
        used = generated.get('physical_attempts_of_generated_choices', 0)
        live_selected = generated.get('actual_llm_skill_selections', 0)
        new_ids = set(state['learn']['activated_new_skill_ids'])
        selected_new = bool(new_ids & set(generated.get('selected_skill_ids', [])))
        executed_new_ids = sorted(new_ids & set(generated.get('physically_executed_skill_ids', [])))
        live_new_ids = sorted(new_ids & set(generated.get('actual_llm_selected_physically_executed_skill_ids', [])))
        live_pass_ids = sorted(new_ids & set(generated.get('actual_llm_selected_physical_pass_skill_ids', [])))
        complete = bool(live and state['learn']['new_registered_count'] and live_new_ids)
        summary = {'status': 'LIVE_LEARNING_CYCLE_OBSERVED' if complete else 'WORKFLOW_COMPLETED_WITH_LIMITATIONS',
            'mode': mode, 'actual_physics': True, 'new_skill_registered': bool(state['learn']['new_registered_count']),
            'next_episode_received_memory': True, 'next_episode_memory_records': state['prepare']['cumulative_memory_records'],
            'next_episode_generated_physical_attempts': used,
            'next_episode_generated_physical_passes': generated.get('generated_choice_physical_passes', 0),
            'actual_llm_skill_selections': live_selected, 'new_skill_selected': selected_new,
            'new_skill_physically_executed_ids': executed_new_ids,
            'actual_llm_new_skill_physical_ids': live_new_ids,
            'actual_llm_new_skill_pass_ids': live_pass_ids,
            'new_skill_reuse_physical_success': bool(live and live_pass_ids),
            'library_merge_conflict': state['learn']['library_merge_conflict'],
            'source_plan_success': state['source']['plan_success'], 'target_plan_success': state['target']['plan_success'],
            'final_memory_records': frozen['record_count'], 'fixture_only': not live,
            'performance_comparison': 'NOT_RUN', 'unseen_generalization': 'NOT_RUN',
            'automatic_http_retries': 0, 'full_dfs_executed': True, 'generation_inside_solver': False,
            'memory_new_snapshot': str(final_memory), 'skills_new_snapshot': state['learn']['library'],
            'limits': limits(plan), 'budget': budget_summary(ledger) if live and ledger.exists() else {'reserved_calls': 0},
            'artifacts': {'memory': str(final_memory), 'skills': state['learn']['library']}}
        write(output / 'cycle_result.json', summary)
        summary['artifacts']['result'] = str(output / 'cycle_result.json')
        return summary

    workflow = LearningCycleWorkflow(output, identity, dict(source=source, prepare=prepare, learn=learn,
                                                         target=target, finish=finish))
    result = workflow.run()
    return result['stage_results']['finish']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=ROOT / 'configs/automatic_learning_cycle_v1.json')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--mode', choices=('preview', 'offline-fixture', 'live'), default='preview')
    p.add_argument('--llm-config', type=Path)
    p.add_argument('--ledger', type=Path)
    p.add_argument('--allow-live-api', action='store_true')
    p.add_argument('--source-fixture', type=Path)
    args = p.parse_args(argv)
    plan = validate_plan(read(checked(args.plan)))
    output = checked(args.run_dir, True)
    live = args.mode == 'live'
    if live != args.allow_live_api or (live and (not args.llm_config or not args.ledger)):
        raise ValueError('live_requires_approved_config_and_shared_ledger')
    if (args.source_fixture and args.mode != 'offline-fixture') or (not live and (args.llm_config or args.ledger)):
        raise ValueError('offline_must_not_use_live_budget_or_live_fixture')
    if args.mode == 'preview':
        output.mkdir(parents=True, exist_ok=False)
        value = {'status': 'PREVIEW', 'plan': plan, 'limits': limits(plan), 'api_calls': 0,
                 'physics_calls': 0, 'key_accessed': False,
                 'cost': 'Exact request previews required; global spend cap must be separately approved.'}
        write(output / 'preview.json', value)
        print(json.dumps(value), flush=True)
        return 0
    config_path, ledger = None, None
    if live:
        config_path, ledger, _config = live_inputs(args.llm_config, args.ledger)
    source_fixture = checked(args.source_fixture, True) if args.source_fixture else None
    result = execute_cycle(output, plan, mode=args.mode, config_path=config_path,
                           ledger=ledger, source_fixture=source_fixture)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
