"""Bounded dynamic-stability development, fresh reuse, then one short DFS run.

The existing 07009 episode is development data, never held-out evaluation.
At most three new proposals, one freely selected recovery (two API attempts),
and four whole-episode actions (eight API attempts) use the original ledger.
Preview reads inputs but performs no API, key access or physics. Every live
invocation requires a new output directory; interrupted work is not relaunched.
"""
from copy import deepcopy
from decimal import Decimal
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from run_integrated_repair import checked, _excluded
from run_luna_experiment import read, write, sha
from run_recorded import run_recorded
from run_automatic_learning_cycle import live_inputs, runtime_hashes, task_argv
from prepare_stability_learning import prepare_stability_learning
from orchestrator.batch_budget import budget_summary
from orchestrator.explicit_action_memory import freeze_cumulative_memory, load_snapshot
from orchestrator.learning_cycle_workflow import LearningCycleWorkflow
from orchestrator.skill_library_merge import freeze_skill_union
from orchestrator.skill_registry import retrieve

TASK = dict(assembly_dir='/opt/asap-research/prototype_completion_20261018/07009',
    base_part='none', seed=0, budget=4, state_action_budget=4, max_gripper=3,
    max_pose=3, timeout=180, wall_timeout=240, response_timeout=60)
LIMITS = dict(max_api_attempts=13, max_synthesis_proposals=3,
    max_validation_physical_actions=18, max_controlled_physical_actions=2,
    max_episode_physical_actions=4, automatic_http_retries=0,
    performance_comparison='NOT_RUN', unseen_generalization='NOT_RUN')


def cycle_limits(max_proposals=3, episode_budget=4):
    if (type(max_proposals) is not int or not 1 <= max_proposals <= 3
            or type(episode_budget) is not int or not 1 <= episode_budget <= 4):
        raise ValueError('bounded_proposals_and_episode_actions_required')
    return {**LIMITS, 'max_api_attempts': max_proposals + 2 + 2 * episode_budget,
            'max_synthesis_proposals': max_proposals,
            'max_validation_physical_actions': 6 * max_proposals,
            'max_episode_physical_actions': episode_budget}


def check_budget(value, *, required_calls=0, before=None, max_attempts=13):
    """Never widen or replace the user's original 39/$0.40 policy."""
    if (value.get('max_calls') != 39 or Decimal(str(value.get('max_usd'))) != Decimal('0.40')
            or value.get('pending_requests') or value.get('unknown_outcomes')):
        raise ValueError('original_resolved_39_call_040_ledger_required')
    if (type(value.get('remaining_calls')) is not int or value['remaining_calls'] < required_calls
            or Decimal(str(value.get('remaining_estimated_usd'))) <= 0):
        raise ValueError('insufficient_remaining_authorized_budget')
    if before is not None and not 0 <= value['reserved_calls'] - before['reserved_calls'] <= max_attempts:
        raise ValueError('cycle_api_attempt_bound_exceeded')
    return value


def controlled_spec(manifest):
    """Declare a fresh replay of development condition, not a held-out task."""
    cases = manifest.get('cases', [])
    debug = [case for case in cases if case.get('phase') == 'debug']
    admissions = [case for case in cases if case.get('phase') == 'admission']
    if (manifest.get('synthesis_failure_scope') != 'dynamic_stability_failed'
            or len(debug) != 1 or len(admissions) != 2 or len(cases) != 3
            or any(case.get('trigger_kind') != 'stability_failed' for case in cases)):
        raise ValueError('three_predeclared_stability_conditions_required')
    return {**deepcopy(debug[0]), 'case_id': '07009_stability_development_reuse', 'phase': 'transfer'}


def refinement_inputs(prior, ledger, config):
    """Bind completed live debug feedback and every file it references.

    Reuse the synthesis runner's exact rehydration gate. Hashes include original
    proposal/manifest/result and debug artifacts, but no admission observation
    is supplied as model feedback. The mutable shared ledger is never hashed.
    """
    from run_skill_discovery import build_refinement_feedback
    prior = checked(prior, True)
    names = ('provider_proposal.json', 'result.json', 'precommitted_cases.json',
             'config.json', 'generation_packet.json', 'source_preservation.json')
    source_paths = {checked(prior / name, True) for name in names}
    proposal, result = read(prior / names[0]), read(prior / names[1])
    manifest, previous_config = read(prior / names[2]), read(prior / names[3])
    controlled_spec(manifest)
    if (proposal.get('origin') != 'actual_llm' or proposal.get('status') != 'proposed'
            or read(prior / 'source_preservation.json').get('unchanged') is not True
            or previous_config.get('mode') != 'live' or previous_config.get('api') != config
            or checked(previous_config.get('ledger'), True) != ledger):
        raise ValueError('refinement_requires_completed_prior_with_original_authorization')
    feedback = build_refinement_feedback(prior, result, manifest)
    for item in feedback:
        source = checked(item['source_run'], True)
        for name in ('precommitted_cases.json', 'provider_proposal.json'):
            source_paths.add(checked(source / name, True))
        debug = checked(source / 'cases' / item['case_id'], True)
        pending = [debug]
        while pending:
            directory = checked(pending.pop(), True)
            for path in directory.iterdir():
                path = checked(path, True)
                if path.is_dir():
                    pending.append(path)
                elif path.is_file():
                    source_paths.add(path)
    return {str(path): sha(path) for path in sorted(source_paths)}, manifest


def physical_episode_args(folder, memory, library, task=None):
    return ['--run-dir', str(folder), '--mode', 'llm'] + task_argv(TASK if task is None else task) + [
        '--provider-contract', 'explicit_action_provider_v5', '--fallback-policy', 'path_stage_v1',
        '--path-selection-policy', 'path_probe_round_v1', '--stability-diagnostics',
        '--memory-policy', 'state_diverse_v1', '--memory-max-records', '8',
        '--memory', str(memory), '--skill-library', str(library)]


def execute_cycle(output, *, episode_dir, initial_memory, initial_library, config_path, ledger,
                  max_proposals=3, episode_budget=4, refine_from=None):
    limits = cycle_limits(max_proposals, episode_budget)
    task = {**TASK, 'budget': episode_budget, 'state_action_budget': episode_budget}
    paths = [output, episode_dir, initial_memory, initial_library, config_path, ledger]
    if refine_from is not None:
        paths.append(refine_from)
    for path in paths:
        _excluded(path)  # Reject prohibited path strings before any filesystem access.
    output, episode_dir = checked(output, True), checked(episode_dir, True)
    initial_memory, initial_library = checked(initial_memory, True), checked(initial_library, True)
    config_path, ledger, config = live_inputs(config_path, ledger)
    if (config.get('max_calls') != 39 or Decimal(str(config.get('max_usd'))) != Decimal('0.40')
            or not ledger.is_file()):
        raise ValueError('existing_original_authorization_required')
    before_budget = check_budget(budget_summary(ledger), required_calls=limits['max_api_attempts'])
    load_snapshot(initial_memory)
    retrieve(read(initial_library), allow_fixture=False)
    source_before = runtime_hashes()
    input_paths = [initial_memory, initial_library, config_path]
    input_before = {str(path): sha(path) for path in input_paths}
    prior_manifest = None
    if refine_from is not None:
        refine_from = checked(refine_from, True)
        prior_hashes, prior_manifest = refinement_inputs(refine_from, ledger, config)
        input_before.update(prior_hashes)
    identity = dict(schema_version=1, episode_dir=str(episode_dir),
        initial_memory=str(initial_memory), initial_library=str(initial_library),
        config=config, ledger=str(ledger), input_sha256=input_before,
        source_sha256=source_before, task=task, limits=limits,
        refine_from=str(refine_from) if refine_from is not None else None,
        development_only=True, target_is_held_out=False)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'cycle_config.json', identity)
    write(output / 'budget_before.json', before_budget)

    def api_args():
        return ['--llm-config', str(config_path), '--ledger', str(ledger), '--allow-live-api']

    def command(name, tool, args, max_calls):
        check_budget(budget_summary(ledger), required_calls=max_calls, before=before_budget,
                     max_attempts=limits['max_api_attempts'])
        if runtime_hashes() != source_before:
            raise ValueError('runtime_changed_between_stages')
        if any(sha(checked(path)) != value for path, value in input_before.items()):
            raise ValueError('bound_input_or_prior_feedback_changed')
        start = budget_summary(ledger)
        record = run_recorded([sys.executable, '-B', '-u', str(ROOT / 'tools' / tool)]
                              + list(map(str, args)), ROOT, output / (name + '_command'))
        if any(sha(checked(path)) != value for path, value in input_before.items()):
            raise ValueError('bound_input_or_prior_feedback_changed')
        end = check_budget(budget_summary(ledger), before=before_budget,
                           max_attempts=limits['max_api_attempts'])
        if not 0 <= end['reserved_calls'] - start['reserved_calls'] <= max_calls:
            raise ValueError('stage_api_attempt_bound_exceeded')
        return record['exit_code']

    def prepare(_):
        folder = output / 'learning_inputs'
        result = prepare_stability_learning(episode_dir, folder)
        memory = output / 'memory_for_reuse.json'
        frozen = freeze_cumulative_memory([initial_memory, result['memory_path']], [], memory)
        artifacts = dict(preparation=str(folder / 'preparation.json'), memory=str(memory))
        if result['status'] == 'PREPARED':
            manifest = read(result['manifest_path'])
            if prior_manifest is not None and prior_manifest['cases'] != manifest['cases']:
                raise ValueError('refinement_must_preserve_declared_stability_conditions')
            spec = output / 'controlled_case.json'
            write(spec, controlled_spec(manifest))
            artifacts.update(manifest=result['manifest_path'], controlled_case=str(spec))
        for index, (path, expected) in enumerate(result['original_artifact_sha256'].items()):
            if sha(checked(path)) != expected:
                raise ValueError('prepared_episode_artifact_changed')
            artifacts['source_%d' % index] = str(checked(path))
        return {**result, 'memory': str(memory), 'memory_records': frozen['record_count'],
                'artifacts': artifacts}

    def learn(state):
        candidates, libraries, prior = [], [], refine_from
        if state['prepare']['status'] == 'PREPARED':
            for index in range(max_proposals):
                name = 'candidate_%02d' % (index + 1)
                folder = output / name
                args = ['--run-dir', folder, '--manifest', state['prepare']['manifest_path'],
                        '--mode', 'live'] + api_args()
                if prior is not None:
                    args += ['--refine-from', prior]
                code = command(name, 'run_skill_discovery.py', args, 1)
                if not (folder / 'result.json').is_file():
                    raise RuntimeError('discovery_failed_without_result')
                result = read(folder / 'result.json')
                if read(folder / 'source_preservation.json').get('unchanged') is not True:
                    raise ValueError('discovery_source_changed')
                candidates.append(dict(run_dir=str(folder), status=result['status'], exit_code=code,
                                       candidate_id=result.get('candidate_id')))
                if code == 0 and result.get('admission', {}).get('admitted') is True:
                    libraries.append(folder / 'skills_frozen.json')
                    break
                if code != 0 or result['status'] == 'GENERATION_REJECTED':
                    break
                proposal = read(folder / 'provider_proposal.json')
                if proposal.get('origin') != 'actual_llm' or proposal.get('status') != 'proposed':
                    break
                # A new candidate can use debug feedback. Never retry a failed
                # HTTP request or expose fresh admission outcomes to synthesis.
                prior = folder
        library = output / 'library_for_reuse.json'
        merged = freeze_skill_union([initial_library] + libraries, library)
        new_ids = sorted({row['strategy_id'] for path in libraries
                          for row in retrieve(read(path), allow_fixture=False)})
        return dict(status='REGISTERED' if libraries else 'NO_NEW_SKILL', candidates=candidates,
            new_skill_ids=new_ids, library=str(library), library_count=len(merged['entries']),
            artifacts={'library': str(library), **{'candidate_%d' % i: str(Path(item['run_dir']) / 'result.json')
                                                  for i, item in enumerate(candidates)}})

    def controlled(state):
        if not state['learn']['new_skill_ids']:
            return dict(status='NOT_RUN_NO_REGISTERED_STABILITY_SKILL', artifacts={})
        folder = output / 'controlled_reuse'
        args = ['--run-dir', folder, '--mode', 'llm', '--skill-library', state['learn']['library'],
                '--case-spec', output / 'controlled_case.json'] + api_args()
        code = command('controlled_reuse', 'run_generated_skill_case.py', args, 2)
        if code != 0 or not (folder / 'case_report.json').is_file():
            raise RuntimeError('controlled_reuse_failed_see_recorded_logs')
        report = read(folder / 'case_report.json')
        preservation = read(folder / 'preservation.json')
        if (not report.get('workflow_completed') or
                not all(preservation.get(k) is True for k in ('library_unchanged', 'source_unchanged', 'case_spec_unchanged'))):
            raise ValueError('controlled_reuse_evidence_or_preservation_failed')
        artifacts = dict(report=str(folder / 'case_report.json'), preservation=str(folder / 'preservation.json'))
        if report.get('memory_export', {}).get('status') == 'FROZEN':
            load_snapshot(folder / 'memory_frozen.json')
            artifacts['memory'] = str(folder / 'memory_frozen.json')
        selected_new = (report.get('generated_skill_id') in state['learn']['new_skill_ids']
                        and report.get('llm_selection_used') is True
                        and report.get('generated_selector_status') == 'selected')
        return dict(status=report['status'], report=report, new_skill_selected=selected_new,
            new_skill_reuse_pass=bool(selected_new and report.get('generated_reuse_pass')),
            development_condition_replay=True, target_is_held_out=False, artifacts=artifacts)

    def full_episode(state):
        if not state['learn']['new_skill_ids']:
            return dict(status='NOT_RUN_NO_REGISTERED_STABILITY_SKILL', artifacts={})
        folder = output / 'next_episode'
        args = physical_episode_args(folder, state['prepare']['memory'], state['learn']['library'], task) + api_args()
        code = command('next_episode', 'run_generated_skills.py', args, 2 * episode_budget)
        if code != 0 or not (folder / 'audit.json').is_file():
            raise RuntimeError('whole_episode_failed_see_recorded_logs')
        audit = read(folder / 'audit.json')
        preservation = read(folder / 'host_preservation.json')
        generated = read(folder / 'generated_skills_audit.json')
        if (not audit.get('validation_ok') or not audit.get('actual_physics')
                or not all(preservation.get(k) is True for k in ('source_unchanged', 'memory_unchanged'))
                or generated.get('library_unchanged') is not True):
            raise ValueError('whole_episode_evidence_or_preservation_failed')
        load_snapshot(folder / 'action_memory_frozen.json')
        new_ids = set(state['learn']['new_skill_ids'])
        return dict(status=audit['status'], plan_success=audit['plan_success'], metrics=audit,
            new_live_executed_ids=sorted(new_ids & set(generated.get('actual_llm_selected_physically_executed_skill_ids', []))),
            new_live_pass_ids=sorted(new_ids & set(generated.get('actual_llm_selected_physical_pass_skill_ids', []))),
            generated_skill_audit=generated, artifacts=dict(audit=str(folder / 'audit.json'),
                memory=str(folder / 'action_memory_frozen.json'), generated=str(folder / 'generated_skills_audit.json'),
                preservation=str(folder / 'host_preservation.json')))

    def finish(state):
        if any(sha(checked(path)) != value for path, value in input_before.items()):
            raise ValueError('bound_input_or_prior_feedback_changed')
        memories = [state['prepare']['memory']]
        for name in ('controlled', 'full_episode'):
            if 'memory' in state[name]['artifacts']:
                memories.append(state[name]['artifacts']['memory'])
        memory = output / 'memory_after_reuse.json'
        frozen = freeze_cumulative_memory(memories, [], memory)
        controlled_pass = state['controlled'].get('new_skill_reuse_pass', False)
        full_pass = bool(state['full_episode'].get('new_live_pass_ids'))
        summary = dict(status='STABILITY_SKILL_REUSE_OBSERVED' if controlled_pass or full_pass
            else 'WORKFLOW_COMPLETED_WITH_LIMITATIONS', registered_skill_ids=state['learn']['new_skill_ids'],
            controlled_reuse_pass=controlled_pass, whole_episode_reuse_pass=full_pass,
            whole_episode_plan_success=state['full_episode'].get('plan_success'),
            whole_episode_status=state['full_episode']['status'],
            full_dfs_executed=bool(state['learn']['new_skill_ids']),
            memory_records=frozen['record_count'], memory_snapshot=str(memory),
            skill_snapshot=state['learn']['library'], limits=limits,
            development_only=True, target_is_held_out=False, cross_task_generalization='NOT_RUN',
            performance_comparison='NOT_RUN', automatic_http_retries=0,
            budget=budget_summary(ledger), artifacts=dict(memory=str(memory), library=state['learn']['library']))
        write(output / 'cycle_result.json', summary)
        return {**summary, 'artifacts': {**summary['artifacts'], 'result': str(output / 'cycle_result.json')}}

    try:
        workflow = LearningCycleWorkflow(output, identity, dict(prepare=prepare, learn=learn,
            controlled=controlled, full_episode=full_episode, finish=finish))
        result = workflow.run()
        return result['stage_results']['finish']
    except Exception as error:
        write(output / 'cycle_error.json', dict(status='ERROR', error_type=type(error).__name__,
            automatic_retry=False, partial_resume=False, details_in_stage_and_command_logs=True))
        raise
    finally:
        write(output / 'budget_after.json', budget_summary(ledger))
        write(output / 'preservation.json', dict(source_unchanged=runtime_hashes() == source_before,
            inputs_unchanged=all(sha(checked(path)) == value for path, value in input_before.items()),
            input_sha256_before=input_before, authorized_mutable_ledger=str(ledger)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('run-dir', 'episode-dir', 'initial-memory', 'initial-library'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--mode', choices=('preview', 'live'), default='preview')
    parser.add_argument('--llm-config', type=Path)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--allow-live-api', action='store_true')
    parser.add_argument('--max-proposals', type=int, default=3)
    parser.add_argument('--episode-budget', type=int, default=4)
    parser.add_argument('--refine-from', type=Path)
    args = parser.parse_args(argv)
    limits = cycle_limits(args.max_proposals, args.episode_budget)
    task = {**TASK, 'budget': args.episode_budget, 'state_action_budget': args.episode_budget}
    live = args.mode == 'live'
    if live != args.allow_live_api or (live and (not args.llm_config or not args.ledger)):
        raise ValueError('live_requires_approved_config_and_original_ledger')
    for path in (args.run_dir, args.episode_dir, args.initial_memory, args.initial_library,
                 args.llm_config, args.ledger, args.refine_from):
        if path is not None:
            _excluded(path)
    if not live:
        if args.llm_config or args.ledger:
            raise ValueError('preview_does_not_use_live_authorization')
        output = checked(args.run_dir, True)
        output.mkdir(parents=True, exist_ok=False)
        result = dict(status='PREVIEW', task=task, limits=limits, api_calls=0, physics_calls=0,
            refine_from=str(args.refine_from) if args.refine_from is not None else None,
            refinement_evidence_checked=False, key_accessed=False, development_only=True, target_is_held_out=False)
        write(output / 'preview.json', result)
    else:
        result = execute_cycle(args.run_dir, episode_dir=args.episode_dir,
            initial_memory=args.initial_memory, initial_library=args.initial_library,
            config_path=args.llm_config, ledger=args.ledger, max_proposals=args.max_proposals,
            episode_budget=args.episode_budget, refine_from=args.refine_from)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
