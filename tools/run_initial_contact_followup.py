"""Separate approved budgets for two refinements and one free 07009 episode.

Only generation uses the existing 39-call/$0.40 ledger. The fresh 6-call/$0.10
ledger is used only after registration, by three unconstrained DFS actions.
No controlled recovery replay, automatic HTTP retry, or performance comparison.
"""
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
from run_automatic_learning_cycle import live_inputs, runtime_hashes
from run_initial_contact_skill_cycle import validate_manifest, refinement_inputs
from run_stability_skill_cycle import physical_episode_args, TASK as STABILITY_TASK
from run_skill_discovery import synthesis_packet
from orchestrator.batch_budget import budget_summary
from orchestrator.explicit_action_memory import freeze_cumulative_memory, load_snapshot
from orchestrator.learning_cycle_workflow import LearningCycleWorkflow
from orchestrator.skill_library_merge import freeze_skill_union
from orchestrator.skill_registry import retrieve

TASK = {**STABILITY_TASK, 'budget': 3, 'state_action_budget': 3}
LIMITS = dict(generation_max_new_api_attempts=2, reuse_max_new_api_attempts=6,
    max_validation_physical_actions=12, max_free_dfs_physical_actions=3,
    max_proposals=2, automatic_http_retries=0, controlled_reuse_actions=0)
POLICIES = {'generation': (39, Decimal('.40')), 'reuse': (6, Decimal('.10'))}


def policy_budget(ledger, kind, *, allow_missing=False, required_calls=0, before=None):
    """Read-only policy check; a missing reuse ledger stays absent."""
    calls, dollars = POLICIES[kind]
    ledger = checked(ledger, True)
    if ledger.exists():
        value = budget_summary(ledger)
    elif allow_missing:
        value = dict(status='NOT_STARTED', max_calls=calls, max_usd=str(dollars),
            reserved_calls=0, request_count=0, pending_requests=0, unknown_outcomes=0,
            remaining_calls=calls, remaining_estimated_usd=str(dollars), budget_committed_usd='0',
            usage_cost_usd='0')
    else:
        raise ValueError('existing_generation_ledger_required')
    if (value.get('max_calls') != calls or Decimal(str(value.get('max_usd'))) != dollars
            or value.get('pending_requests') or value.get('unknown_outcomes')):
        raise ValueError('wrong_or_unresolved_stage_policy')
    if (type(value.get('remaining_calls')) is not int or value['remaining_calls'] < required_calls
            or (required_calls and Decimal(str(value.get('remaining_estimated_usd'))) <= 0)):
        raise ValueError('insufficient_stage_budget')
    if before is not None:
        cap = 2 if kind == 'generation' else 6
        if not 0 <= value['reserved_calls'] - before['reserved_calls'] <= cap:
            raise ValueError('stage_authorization_exceeded')
    return value


def validate_policies(generation, reuse):
    if (generation.get('max_calls') != 39 or Decimal(str(generation.get('max_usd'))) != Decimal('.4')
            or reuse != {**generation, 'max_calls': 6, 'max_usd': .10}):
        raise ValueError('reuse_policy_must_only_change_authorized_caps')


def execute_followup(output, *, manifest_path, prior_candidate, initial_memory, initial_library,
                     generation_config, generation_ledger, reuse_config, reuse_ledger):
    values = (output, manifest_path, prior_candidate, initial_memory, initial_library,
              generation_config, generation_ledger, reuse_config, reuse_ledger)
    for value in values:
        _excluded(value)
    output, manifest_path = checked(output, True), checked(manifest_path, True)
    prior_candidate = checked(prior_candidate, True)
    initial_memory, initial_library = checked(initial_memory, True), checked(initial_library, True)
    generation_config, generation_ledger, old_config = live_inputs(generation_config, generation_ledger)
    reuse_config, reuse_ledger, new_config = live_inputs(reuse_config, reuse_ledger)
    if generation_ledger == reuse_ledger or generation_config == reuse_config:
        raise ValueError('generation_and_reuse_authorizations_must_be_separate')
    validate_policies(old_config, new_config)
    before = {'generation': policy_budget(generation_ledger, 'generation', required_calls=2),
              'reuse': policy_budget(reuse_ledger, 'reuse', allow_missing=True, required_calls=6)}
    if before['reuse']['reserved_calls'] or before['reuse'].get('request_count'):
        raise ValueError('fresh_unused_reuse_ledger_required')
    manifest = read(manifest_path)
    validate_manifest(manifest)
    input_hashes, prior_manifest = refinement_inputs(prior_candidate, generation_ledger, old_config)
    if manifest['cases'] != prior_manifest['cases']:
        raise ValueError('predeclared_initial_contact_cases_must_remain_identical')
    source_paths = [manifest_path, initial_memory, initial_library, generation_config, reuse_config]
    for field in ('experience_report', 'experience_memory'):
        source_paths.append(checked(manifest[field], True))
    if manifest.get('episode_learning_audit'):
        audit_path = checked(manifest['episode_learning_audit'], True)
        source_paths.append(audit_path)
        for path, expected in read(audit_path)['original_artifact_sha256'].items():
            path = checked(path)
            if sha(path) != expected:
                raise ValueError('original_episode_artifact_changed')
            input_hashes[str(path)] = expected
    input_hashes.update({str(path): sha(path) for path in source_paths})
    load_snapshot(initial_memory)
    retrieve(read(initial_library), allow_fixture=False)
    sources = runtime_hashes()
    identity = dict(schema_version=1, manifest=str(manifest_path), prior_candidate=str(prior_candidate),
        initial_memory=str(initial_memory), initial_library=str(initial_library),
        generation_config=old_config, generation_ledger=str(generation_ledger),
        reuse_config=new_config, reuse_ledger=str(reuse_ledger), input_sha256=input_hashes,
        source_sha256=sources, task=TASK, limits=LIMITS, development_only=True,
        target_is_held_out=False, controlled_reuse=False)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'followup_config.json', identity)
    write(output / 'budgets_before.json', before)
    ledgers = {'generation': generation_ledger, 'reuse': reuse_ledger}
    configs = {'generation': generation_config, 'reuse': reuse_config}

    def verify_inputs():
        if runtime_hashes() != sources:
            raise ValueError('runtime_changed_between_stages')
        if any(sha(checked(path)) != value for path, value in input_hashes.items()):
            raise ValueError('bound_input_or_original_evidence_changed')

    def api_args(kind):
        return ['--llm-config', str(configs[kind]), '--ledger', str(ledgers[kind]), '--allow-live-api']

    def command(name, tool, args, kind, cap):
        verify_inputs()
        start = policy_budget(ledgers[kind], kind, allow_missing=kind == 'reuse',
                              required_calls=cap, before=before[kind])
        other = ledgers['reuse' if kind == 'generation' else 'generation']
        other_hash = sha(other) if other.exists() else None
        record = run_recorded([sys.executable, '-B', '-u', str(ROOT / 'tools' / tool)]
                              + list(map(str, args)), ROOT, output / (name + '_command'))
        verify_inputs()
        if (sha(other) if other.exists() else None) != other_hash:
            raise ValueError('wrong_stage_ledger_was_modified')
        end = policy_budget(ledgers[kind], kind, allow_missing=kind == 'reuse', before=before[kind])
        if not 0 <= end['reserved_calls'] - start['reserved_calls'] <= cap:
            raise ValueError('command_api_attempt_bound_exceeded')
        return record['exit_code']

    def prepare(_):
        packet = synthesis_packet(manifest, 'initial-contact-followup-preparation')
        packet_path = output / 'prepared_generation_packet.json'
        write(packet_path, packet)
        memory = output / 'memory_for_reuse.json'
        snapshot = freeze_cumulative_memory([initial_memory, checked(manifest['experience_memory'], True)], [], memory)
        write(output / 'precommitted_cases.json', manifest)
        return dict(status='PREPARED', memory=str(memory), memory_records=snapshot['record_count'],
            source_evidence_records=len(packet['evidence']), new_api_calls=0, new_physical_calls=0,
            artifacts=dict(memory=str(memory), manifest=str(output / 'precommitted_cases.json'),
                           packet=str(packet_path)))

    def learn(state):
        candidates, admitted, prior = [], [], prior_candidate
        for index in range(2):
            name = 'candidate_%02d' % (index + 1)
            folder = output / name
            code = command(name, 'run_skill_discovery.py', ['--run-dir', folder,
                '--manifest', output / 'precommitted_cases.json', '--mode', 'live',
                '--refine-from', prior] + api_args('generation'), 'generation', 1)
            if not (folder / 'result.json').is_file():
                raise RuntimeError('discovery_failed_without_result')
            result = read(folder / 'result.json')
            if read(folder / 'source_preservation.json').get('unchanged') is not True:
                raise ValueError('discovery_source_changed')
            candidates.append(dict(run_dir=str(folder), status=result['status'], exit_code=code,
                                   candidate_id=result.get('candidate_id')))
            if (code == 0 and result.get('admission', {}).get('admitted') is True
                    and result.get('generation_origin') == 'live_llm'):
                admitted.append(folder / 'skills_frozen.json')
                break
            if code != 0 or result['status'] == 'GENERATION_REJECTED':
                break
            proposal = read(folder / 'provider_proposal.json')
            if proposal.get('origin') != 'actual_llm' or proposal.get('status') != 'proposed':
                break
            # A new candidate receives only predeclared debug feedback. This
            # is explicit bounded refinement, not replay of an HTTP request.
            prior = folder
        library = output / 'library_for_reuse.json'
        merged = freeze_skill_union([initial_library] + admitted, library)
        ids = sorted({row['strategy_id'] for path in admitted for row in retrieve(read(path), allow_fixture=False)})
        return dict(status='REGISTERED' if ids else 'NOT_REGISTERED', new_skill_ids=ids,
            candidates=candidates, library=str(library), library_count=len(merged['entries']),
            artifacts={'library': str(library), **{'candidate_%d' % i: str(Path(c['run_dir']) / 'result.json')
                                                 for i, c in enumerate(candidates)}})

    def free_dfs(state):
        if not state['learn']['new_skill_ids']:
            return dict(status='NOT_RUN_NO_REGISTERED_SKILL', artifacts={})
        folder = output / 'free_dfs'
        args = physical_episode_args(folder, state['prepare']['memory'], state['learn']['library'], TASK)
        code = command('free_dfs', 'run_generated_skills.py', args + api_args('reuse'), 'reuse', 6)
        if code != 0 or not (folder / 'audit.json').is_file():
            raise RuntimeError('free_dfs_failed_see_command_logs')
        audit, preservation = read(folder / 'audit.json'), read(folder / 'host_preservation.json')
        generated = read(folder / 'generated_skills_audit.json')
        if (not audit.get('validation_ok') or not audit.get('actual_physics')
                or type(audit.get('verifier_calls')) is not int or not 0 <= audit['verifier_calls'] <= 3
                or not all(preservation.get(k) is True for k in ('source_unchanged', 'memory_unchanged'))
                or generated.get('library_unchanged') is not True):
            raise ValueError('free_dfs_evidence_or_preservation_failed')
        load_snapshot(folder / 'action_memory_frozen.json')
        ids = set(state['learn']['new_skill_ids'])
        return dict(status=audit['status'], plan_success=audit['plan_success'], metrics=audit,
            new_live_executed_ids=sorted(ids & set(generated.get('actual_llm_selected_physically_executed_skill_ids', []))),
            new_live_pass_ids=sorted(ids & set(generated.get('actual_llm_selected_physical_pass_skill_ids', []))),
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
        executed, passed = (state['free_dfs'].get('new_live_executed_ids', []),
                            state['free_dfs'].get('new_live_pass_ids', []))
        status = ('INITIAL_CONTACT_SKILL_REUSE_PASS' if passed else 'INITIAL_CONTACT_SKILL_EXECUTED'
                  if executed else 'NOT_OBSERVED_WITHIN_BUDGET' if state['learn']['new_skill_ids'] else 'NOT_REGISTERED')
        budgets = {kind: policy_budget(path, kind, allow_missing=kind == 'reuse', before=before[kind])
                   for kind, path in ledgers.items()}
        result = dict(status=status, registered_skill_ids=state['learn']['new_skill_ids'],
            new_live_executed_ids=executed, new_live_pass_ids=passed, free_dfs_reuse_pass=bool(passed),
            free_dfs_status=state['free_dfs']['status'], free_dfs_plan_success=state['free_dfs'].get('plan_success'),
            free_dfs_fallbacks=state['free_dfs'].get('metrics', {}).get('fallbacks'),
            free_dfs_executed=bool(state['learn']['new_skill_ids']), controlled_reuse_executed=False,
            fixed_selection_injected=False, memory_records=frozen['record_count'],
            memory_snapshot=str(memory), skill_snapshot=state['learn']['library'], limits=LIMITS,
            budgets=budgets, generation_new_api_calls=budgets['generation']['reserved_calls'] - before['generation']['reserved_calls'],
            reuse_new_api_calls=budgets['reuse']['reserved_calls'] - before['reuse']['reserved_calls'],
            policies_kept_separate=True, development_only=True, target_is_held_out=False,
            cross_task_generalization='NOT_RUN', performance_comparison='NOT_RUN', automatic_http_retries=0,
            artifacts=dict(memory=str(memory), library=state['learn']['library']))
        write(output / 'followup_result.json', result)
        return {**result, 'artifacts': {**result['artifacts'], 'result': str(output / 'followup_result.json')}}

    try:
        result = LearningCycleWorkflow(output, identity,
            dict(prepare=prepare, learn=learn, free_dfs=free_dfs, finish=finish)).run()
        return result['stage_results']['finish']
    except Exception as error:
        write(output / 'followup_error.json', dict(status='ERROR', error_type=type(error).__name__,
            automatic_retry=False, partial_resume=False, details_in_stage_and_command_logs=True))
        raise
    finally:
        write(output / 'budgets_after.json', {kind: policy_budget(path, kind, allow_missing=kind == 'reuse')
                                             for kind, path in ledgers.items()})
        write(output / 'preservation.json', dict(source_unchanged=runtime_hashes() == sources,
            inputs_unchanged=all(sha(checked(path)) == value for path, value in input_hashes.items()),
            authorized_mutable_ledgers={kind: str(path) for kind, path in ledgers.items()},
            input_sha256_before=input_hashes))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'manifest', 'prior-candidate', 'initial-memory', 'initial-library'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('generation-config', 'generation-ledger', 'reuse-config', 'reuse-ledger'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--mode', choices=('preview', 'live'), default='preview')
    parser.add_argument('--allow-live-api', action='store_true')
    args = parser.parse_args(argv)
    live = args.mode == 'live'
    authorizations = (args.generation_config, args.generation_ledger, args.reuse_config, args.reuse_ledger)
    if live != args.allow_live_api or (live and not all(authorizations)) or (not live and any(authorizations)):
        raise ValueError('live_requires_both_explicit_authorizations_preview_requires_neither')
    for path in (args.run_dir, args.manifest, args.prior_candidate, args.initial_memory, args.initial_library, *authorizations):
        if path is not None:
            _excluded(path)
    if live:
        result = execute_followup(args.run_dir, manifest_path=args.manifest, prior_candidate=args.prior_candidate,
            initial_memory=args.initial_memory, initial_library=args.initial_library,
            generation_config=args.generation_config, generation_ledger=args.generation_ledger,
            reuse_config=args.reuse_config, reuse_ledger=args.reuse_ledger)
    else:
        output = checked(args.run_dir, True)
        output.mkdir(parents=True, exist_ok=False)
        result = dict(status='PREVIEW', task=TASK, limits=LIMITS, api_calls=0, physics_calls=0,
            key_accessed=False, authorizations_checked=False, policies_kept_separate=True,
            controlled_reuse=False, fixed_selection_injected=False)
        write(output / 'preview.json', result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
