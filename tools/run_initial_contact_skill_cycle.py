"""One initial-disconnection proposal, registration, then two free DFS actions.

This is a bounded development smoke test on 07009, not held-out evaluation.
The original shared 39-call/$0.40 ledger permits at most five new API attempts:
one synthesis and at most four actor/selector attempts across two actions.
There is no controlled replay reuse stage or automatic retry. An explicitly
supplied completed prior candidate may supply bound debug feedback for this
one new proposal. API exhaustion can still produce deterministic physics
fallbacks; those are never counted as actual-LLM generated-skill reuse.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from run_integrated_repair import checked, _excluded
from run_luna_experiment import read, write, sha
from run_recorded import run_recorded
from run_automatic_learning_cycle import live_inputs, runtime_hashes
from run_stability_skill_cycle import check_budget, physical_episode_args, TASK as STABILITY_TASK
from prepare_stability_learning import prepare_stability_learning
from orchestrator.batch_budget import budget_summary
from orchestrator.explicit_action_memory import freeze_cumulative_memory, load_snapshot
from orchestrator.learning_cycle_workflow import LearningCycleWorkflow
from orchestrator.skill_library_merge import freeze_skill_union
from orchestrator.skill_registry import retrieve

TASK = {**STABILITY_TASK, 'budget': 2, 'state_action_budget': 2}
LIMITS = dict(max_api_attempts=5, max_synthesis_proposals=1,
    max_validation_physical_actions=6, max_free_dfs_physical_actions=2,
    max_free_dfs_api_attempts=4, controlled_reuse_actions=0,
    automatic_http_retries=0, automatic_refinements=0)


def validate_manifest(manifest):
    cases = manifest.get('cases', [])
    if (manifest.get('synthesis_failure_scope') != 'initial_disconnected' or len(cases) != 3
            or sum(c.get('phase') == 'debug' for c in cases) != 1
            or sum(c.get('phase') == 'admission' for c in cases) != 2
            or any(c.get('trigger_kind') != 'stability_failed'
                   or c.get('expected_diagnostic_phase') != 'initial_disconnected' for c in cases)):
        raise ValueError('three_initial_disconnected_conditions_required')


def refinement_inputs(prior, ledger, config):
    """Rehydrate actual debug observations and bind their immutable files."""
    from run_skill_discovery import build_refinement_feedback
    prior = checked(prior, True)
    names = ('provider_proposal.json', 'result.json', 'precommitted_cases.json',
             'config.json', 'generation_packet.json', 'source_preservation.json', 'source_before.json')
    paths = {checked(prior / name, True) for name in names}
    proposal, result = read(prior / names[0]), read(prior / names[1])
    manifest, previous_config = read(prior / names[2]), read(prior / names[3])
    validate_manifest(manifest)
    if (proposal.get('origin') != 'actual_llm' or proposal.get('status') != 'proposed'
            or result.get('generation_origin') != 'live_llm'
            or read(prior / 'source_preservation.json').get('unchanged') is not True
            or previous_config.get('mode') != 'live' or previous_config.get('api') != config
            or checked(previous_config.get('ledger'), True) != ledger):
        raise ValueError('refinement_requires_completed_prior_with_original_authorization')
    for name, expected in read(prior / 'source_before.json').items():
        archived = checked(prior / 'implementation' / name, True)
        if sha(archived) != expected:
            raise ValueError('prior_archived_source_changed')
        paths.add(archived)
    feedback = build_refinement_feedback(prior, result, manifest)
    for item in feedback:
        source = checked(item['source_run'], True)
        for name in ('precommitted_cases.json', 'provider_proposal.json'):
            paths.add(checked(source / name, True))
        pending = [checked(source / 'cases' / item['case_id'], True)]
        while pending:
            directory = checked(pending.pop(), True)
            for path in directory.iterdir():
                path = checked(path, True)
                if path.is_dir():
                    pending.append(path)
                elif path.is_file():
                    paths.add(path)
    return {str(path): sha(path) for path in sorted(paths)}, manifest


def execute_cycle(output, *, episode_dir, initial_memory, initial_library, config_path, ledger,
                  refine_from=None):
    paths = [output, episode_dir, initial_memory, initial_library, config_path, ledger]
    if refine_from is not None:
        paths.append(refine_from)
    for path in paths:
        _excluded(path)
    output, episode_dir = checked(output, True), checked(episode_dir, True)
    initial_memory, initial_library = checked(initial_memory, True), checked(initial_library, True)
    config_path, ledger, config = live_inputs(config_path, ledger)
    if (config.get('max_calls') != 39 or Decimal(str(config.get('max_usd'))) != Decimal('.40')
            or not ledger.is_file()):
        raise ValueError('existing_original_authorization_required')
    before_budget = check_budget(budget_summary(ledger), required_calls=2, max_attempts=5)
    max_attempts = min(5, before_budget['remaining_calls'])
    limits = {**LIMITS, 'max_api_attempts': max_attempts,
              'max_free_dfs_api_attempts': min(4, before_budget['remaining_calls'] - 1),
              'api_exhaustion_may_execute_deterministic_fallback': True}
    load_snapshot(initial_memory)
    retrieve(read(initial_library), allow_fixture=False)
    source_before = runtime_hashes()
    input_before = {str(path): sha(path) for path in (initial_memory, initial_library, config_path)}
    prior_manifest = None
    if refine_from is not None:
        refine_from = checked(refine_from, True)
        hashes, prior_manifest = refinement_inputs(refine_from, ledger, config)
        input_before.update(hashes)
    identity = dict(schema_version=1, episode_dir=str(episode_dir),
        initial_memory=str(initial_memory), initial_library=str(initial_library),
        config=config, ledger=str(ledger), source_sha256=source_before,
        input_sha256=input_before, task=TASK, limits=limits,
        refine_from=str(refine_from) if refine_from is not None else None,
        development_only=True, target_is_held_out=False)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'cycle_config.json', identity)
    write(output / 'budget_before.json', before_budget)

    def verify_inputs():
        if runtime_hashes() != source_before:
            raise ValueError('runtime_changed_between_stages')
        if any(sha(checked(path)) != value for path, value in input_before.items()):
            raise ValueError('bound_input_changed')

    def api_args():
        return ['--llm-config', str(config_path), '--ledger', str(ledger), '--allow-live-api']

    def command(name, tool, args, stage_calls):
        start = check_budget(budget_summary(ledger), required_calls=stage_calls,
                             before=before_budget, max_attempts=max_attempts)
        verify_inputs()
        record = run_recorded([sys.executable, '-B', '-u', str(ROOT / 'tools' / tool)]
                              + list(map(str, args)), ROOT, output / (name + '_command'))
        verify_inputs()
        end = check_budget(budget_summary(ledger), before=before_budget, max_attempts=max_attempts)
        if not 0 <= end['reserved_calls'] - start['reserved_calls'] <= stage_calls:
            raise ValueError('stage_api_attempt_bound_exceeded')
        return record['exit_code']

    def prepare(_):
        folder = output / 'learning_inputs'
        prepared = prepare_stability_learning(episode_dir, folder, failure_scope='initial_disconnected')
        memory = output / 'memory_for_reuse.json'
        frozen = freeze_cumulative_memory([initial_memory, prepared['memory_path']], [], memory)
        artifacts = dict(preparation=str(folder / 'preparation.json'), memory=str(memory))
        if prepared['status'] == 'PREPARED':
            manifest = read(prepared['manifest_path'])
            validate_manifest(manifest)
            if prior_manifest is not None and prior_manifest['cases'] != manifest['cases']:
                raise ValueError('refinement_must_preserve_declared_initial_contact_conditions')
            artifacts['manifest'] = prepared['manifest_path']
        for index, (path, expected) in enumerate(prepared['original_artifact_sha256'].items()):
            path = checked(path)
            if sha(path) != expected:
                raise ValueError('source_episode_artifact_changed')
            artifacts['source_%d' % index] = str(path)
        return {**prepared, 'memory': str(memory), 'memory_records': frozen['record_count'],
                'artifacts': artifacts}

    def learn(state):
        libraries, candidate, artifacts = [], None, {}
        if state['prepare']['status'] == 'PREPARED':
            folder = output / 'candidate_01'
            args = ['--run-dir', folder, '--manifest', state['prepare']['manifest_path'],
                    '--mode', 'live'] + api_args()
            if refine_from is not None:
                args += ['--refine-from', refine_from]
            code = command('candidate_01', 'run_skill_discovery.py', args, 1)
            if not (folder / 'result.json').is_file():
                raise RuntimeError('discovery_failed_without_result')
            result = read(folder / 'result.json')
            if read(folder / 'source_preservation.json').get('unchanged') is not True:
                raise ValueError('discovery_source_changed')
            candidate = dict(run_dir=str(folder), status=result['status'], exit_code=code,
                             candidate_id=result.get('candidate_id'))
            artifacts['candidate'] = str(folder / 'result.json')
            if (code == 0 and result.get('admission', {}).get('admitted') is True
                    and result.get('generation_origin') == 'live_llm'):
                libraries.append(folder / 'skills_frozen.json')
        library = output / 'library_for_reuse.json'
        snapshot = freeze_skill_union([initial_library] + libraries, library)
        new_ids = sorted({row['strategy_id'] for path in libraries
                          for row in retrieve(read(path), allow_fixture=False)})
        artifacts['library'] = str(library)
        return dict(status='REGISTERED' if new_ids else 'NO_NEW_SKILL', candidate=candidate,
            new_skill_ids=new_ids, library=str(library), library_count=len(snapshot['entries']), artifacts=artifacts)

    def free_dfs(state):
        if not state['learn']['new_skill_ids']:
            return dict(status='NOT_RUN_NO_REGISTERED_INITIAL_CONTACT_SKILL', artifacts={})
        folder = output / 'free_dfs'
        args = physical_episode_args(folder, state['prepare']['memory'], state['learn']['library'], TASK) + api_args()
        remaining = check_budget(budget_summary(ledger), required_calls=1,
                                 before=before_budget, max_attempts=max_attempts)['remaining_calls']
        code = command('free_dfs', 'run_generated_skills.py', args, min(4, remaining))
        if code != 0 or not (folder / 'audit.json').is_file():
            raise RuntimeError('free_dfs_failed_see_recorded_logs')
        audit = read(folder / 'audit.json')
        preservation = read(folder / 'host_preservation.json')
        generated = read(folder / 'generated_skills_audit.json')
        if (not audit.get('validation_ok') or not audit.get('actual_physics')
                or type(audit.get('verifier_calls')) is not int or not 0 <= audit['verifier_calls'] <= 2
                or not all(preservation.get(k) is True for k in ('source_unchanged', 'memory_unchanged'))
                or generated.get('library_unchanged') is not True):
            raise ValueError('free_dfs_evidence_or_preservation_failed')
        load_snapshot(folder / 'action_memory_frozen.json')
        new_ids = set(state['learn']['new_skill_ids'])
        return dict(status=audit['status'], plan_success=audit['plan_success'], metrics=audit,
            new_live_executed_ids=sorted(new_ids & set(generated.get('actual_llm_selected_physically_executed_skill_ids', []))),
            new_live_pass_ids=sorted(new_ids & set(generated.get('actual_llm_selected_physical_pass_skill_ids', []))),
            generated_skill_audit=generated, fixed_selection_injected=False,
            artifacts=dict(audit=str(folder / 'audit.json'), memory=str(folder / 'action_memory_frozen.json'),
                generated=str(folder / 'generated_skills_audit.json'), preservation=str(folder / 'host_preservation.json')))

    def finish(state):
        verify_inputs()
        memories = [state['prepare']['memory']]
        if 'memory' in state['free_dfs']['artifacts']:
            memories.append(state['free_dfs']['artifacts']['memory'])
        memory = output / 'memory_after_reuse.json'
        frozen = freeze_cumulative_memory(memories, [], memory)
        executed = state['free_dfs'].get('new_live_executed_ids', [])
        passed = state['free_dfs'].get('new_live_pass_ids', [])
        status = ('INITIAL_CONTACT_SKILL_REUSE_PASS' if passed else 'INITIAL_CONTACT_SKILL_EXECUTED'
                  if executed else 'NOT_OBSERVED_WITHIN_BUDGET' if state['learn']['new_skill_ids'] else 'NOT_REGISTERED')
        result = dict(status=status, registered_skill_ids=state['learn']['new_skill_ids'],
            new_live_executed_ids=executed, new_live_pass_ids=passed,
            free_dfs_reuse_pass=bool(passed), free_dfs_status=state['free_dfs']['status'],
            free_dfs_plan_success=state['free_dfs'].get('plan_success'),
            free_dfs_fallback_count=state['free_dfs'].get('metrics', {}).get('fallbacks'),
            free_dfs_actual_llm_choices=state['free_dfs'].get('metrics', {}).get('llm_choices'),
            free_dfs_executed=bool(state['learn']['new_skill_ids']), controlled_reuse_executed=False,
            fixed_selection_injected=False, memory_records=frozen['record_count'],
            memory_snapshot=str(memory), skill_snapshot=state['learn']['library'], limits=limits,
            development_only=True, target_is_held_out=False, cross_task_generalization='NOT_RUN',
            performance_comparison='NOT_RUN', automatic_http_retries=0,
            automatic_refinements=0, budget=budget_summary(ledger),
            explicit_prior_refinement=refine_from is not None,
            api_exhaustion_may_execute_deterministic_fallback=True,
            artifacts=dict(memory=str(memory), library=state['learn']['library']))
        write(output / 'cycle_result.json', result)
        return {**result, 'artifacts': {**result['artifacts'], 'result': str(output / 'cycle_result.json')}}

    try:
        workflow = LearningCycleWorkflow(output, identity,
            dict(prepare=prepare, learn=learn, free_dfs=free_dfs, finish=finish))
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
    parser.add_argument('--refine-from', type=Path)
    args = parser.parse_args(argv)
    live = args.mode == 'live'
    if live != args.allow_live_api or (live and (not args.llm_config or not args.ledger)):
        raise ValueError('live_requires_original_approved_config_and_ledger')
    for path in (args.run_dir, args.episode_dir, args.initial_memory, args.initial_library,
                 args.llm_config, args.ledger, args.refine_from):
        if path is not None:
            _excluded(path)
    if not live:
        if args.llm_config or args.ledger:
            raise ValueError('preview_does_not_use_live_authorization')
        output = checked(args.run_dir, True)
        output.mkdir(parents=True, exist_ok=False)
        result = dict(status='PREVIEW', task=TASK, limits=LIMITS, api_calls=0, physics_calls=0,
            key_accessed=False, development_only=True, target_is_held_out=False,
            refine_from=str(args.refine_from) if args.refine_from is not None else None,
            refinement_evidence_checked=False, api_exhaustion_may_execute_deterministic_fallback=True,
            controlled_reuse_executed=False, fixed_selection_injected=False)
        write(output / 'preview.json', result)
    else:
        result = execute_cycle(args.run_dir, episode_dir=args.episode_dir,
            initial_memory=args.initial_memory, initial_library=args.initial_library,
            config_path=args.llm_config, ledger=args.ledger, refine_from=args.refine_from)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
