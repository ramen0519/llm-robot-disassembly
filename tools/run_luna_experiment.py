"""Run official beam with Luna configuration, one batch budget and frozen memory.

Default configuration disables external API calls. Live execution additionally
requires --allow-live-api and explicit positive limits/rates in the config.
This entrypoint never substitutes simulated physics for the ASAP worker.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import uuid

os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

ROOT = Path('C:/dev/disassembly_codex_ready_v2')
ASAP = '/opt/asap-research/ASAP'
DEFAULT_MEMORY = ROOT / 'runs/memory_per_seed_20261017/memories/s0/memory_snapshot.json'
WSL = [r'C:\Windows\System32\wsl.exe', '-d', 'ASAP-Ubuntu', '-u', 'root',
       '--cd', ASAP, '--', '/opt/asap-research/envs/asap/bin/python', '-B']
WORKER = ('rank_contract.py', 'rank_hook.py', 'run_rank_hook.py', 'compare_rank_hook.py',
          'validate_baseline.py', 'remote_rank_client.py', 'run_remote_rank.py',
          'describe_beam.py', 'export_graph_artifacts.py', 'evidence_memory.py', 'memory_ranker.py')
ORCHESTRATOR = ('__init__.py', 'file_service.py', 'llm_ranker.py', 'batch_budget.py',
                'ranking_pipeline.py', 'episode_archive.py', 'langgraph_workflow.py')
TOOLS = ('run_luna_experiment.py', 'run_recorded.py', 'run_memory_multiseed.py',
         'run_asap_baseline.py', 'run_langgraph_integration.py')
CONFIG_KEYS = {'enabled', 'model', 'base_url', 'api_key_env', 'max_calls', 'max_usd',
               'input_usd_per_million', 'output_usd_per_million', 'max_output_tokens',
               'reasoning_effort', 'request_overhead_tokens'}


def safe(value, runs_only=False):
    raw = str(value).replace('\\', '/')
    if '..' in raw.split('/'):
        raise ValueError('parent traversal is excluded')
    path = Path(value).absolute()
    boundary = ROOT / 'runs' if runs_only else ROOT
    if not path.is_relative_to(boundary):
        raise ValueError('path must be inside the C: project')
    for item in reversed((path,) + tuple(path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('links and reparse points are excluded')
    return path


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8'))


def write(path, value):
    with safe(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def linux(path):
    return '/mnt/c/dev/disassembly_codex_ready_v2/' + safe(path).relative_to(ROOT).as_posix()


def validate_config(config, allow_live):
    # Do not import runtime modules before choosing their frozen source directory.
    if type(config) is not dict or set(config) != CONFIG_KEYS:
        raise ValueError('use the complete supported config without inline secrets')
    if config['model'] != 'gpt-6-luna' or config['base_url'] != 'https://api.openai.com/v1':
        raise ValueError('this entrypoint is pinned to OpenAI gpt-6-luna')
    if type(config['enabled']) is not bool or bool(allow_live) != config['enabled']:
        raise ValueError('enabled config and explicit --allow-live-api must agree')
    if config['reasoning_effort'] not in ('none', 'low', 'medium', 'high'):
        raise ValueError('unsupported Luna reasoning setting for this entrypoint')
    if type(config['max_output_tokens']) is not int or not 16 <= config['max_output_tokens'] <= 4096:
        raise ValueError('invalid output token limit')
    if type(config['request_overhead_tokens']) is not int or not 2048 <= config['request_overhead_tokens'] <= 65536:
        raise ValueError('invalid input token overhead')
    if config['api_key_env'] != 'OPENAI_API_KEY':
        raise ValueError('use the documented host-only OPENAI_API_KEY environment variable')
    if not allow_live:
        if type(config['max_calls']) is not int or config['max_calls'] != 0 or config['max_usd'] != 0:
            raise ValueError('offline configuration must retain zero authorization caps')
        return
    if type(config['max_calls']) is not int or not 0 < config['max_calls'] <= 1000000:
        raise ValueError('set an explicit positive total call limit')
    for key in ('max_usd', 'input_usd_per_million', 'output_usd_per_million'):
        value = Decimal(str(config[key]))
        if not value.is_finite() or value <= 0:
            raise ValueError('set positive finite budget and token rates')
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        raise ValueError('OPENAI_API_KEY is missing; no request has been sent')


def variants(base_order, memory):
    orders = ('native', 'reverse') if base_order == 'both' else (base_order,)
    modes = ('off', 'on') if memory == 'both' else (memory,)
    return [(order, mode) for order in orders for mode in modes]


def validation_status(parsed, exit_code):
    """The artifact validator exits 1 for bounded UNKNOWN as well as ERROR."""
    status = parsed.get('status') if type(parsed) is dict else None
    expected_exit = 0 if status == 'PASS' else 1
    return status if status in ('PASS', 'UNKNOWN', 'ERROR') and exit_code == expected_exit else 'ERROR'


def roundtrip_valid(decisions, responses):
    try:
        by_id = {item['request_id']: item for item in responses}
        return (bool(decisions) and len(decisions) == len(responses) == len(by_id) and all(
            item['request']['request_id'] in by_id
            and item['applied_order'] == by_id[item['request']['request_id']]['candidate_ids']
            and sorted(item['applied_order']) == sorted(item['request']['candidate_ids'])
            for item in decisions))
    except (KeyError, TypeError):
        return False


def run_outcome(run, variant, state):
    """Retain a visible row even when a worker never produces final artifacts."""
    issues = []
    def optional(path):
        try:
            return read(path)
        except Exception as error:
            issues.append(dict(artifact=path.relative_to(run).as_posix(), error_type=type(error).__name__))
            return None
    parsed = optional(run / 'parsed_result.json')
    command = optional(run / 'execute/command.json')
    parsed = parsed if type(parsed) is dict else {}
    command = command if type(command) is dict else {}
    events = []
    try:
        for number, line in enumerate(safe(run / 'events.jsonl').read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                if type(event) is not dict:
                    raise ValueError('event must be an object')
                events.append(event)
            except (ValueError, TypeError) as error:
                issues.append(dict(artifact='events.jsonl', line=number, error_type=type(error).__name__))
    except Exception as error:
        issues.append(dict(artifact='events.jsonl', error_type=type(error).__name__))
    responses = []
    response_dir = safe(run / 'ipc/responses')
    if response_dir.is_dir():
        for path in sorted(response_dir.glob('*.json')):
            response = optional(path)
            if type(response) is dict:
                responses.append(response)
    decisions = [item for item in events if item.get('kind') == 'rank_decision']
    verifier = [item for item in events if item.get('kind') == 'verifier_result']
    metadata = [item['metadata'] for item in responses if type(item.get('metadata')) is dict]
    stats = parsed.get('raw_stats')
    stats = stats if type(stats) is dict else {}
    observed = bool(events)
    row = dict(variant=variant, graph_status=state.get('status'), graph_phase=state.get('phase'),
        plan_status=parsed.get('status'), plan_success=parsed.get('plan_success'),
        stop_msg=stats.get('stop_msg'), sequence=stats.get('sequence'),
        process_exit=command.get('exit_code'), process_seconds=command.get('elapsed_seconds'),
        run_end_status=events[-1].get('status') if events and events[-1].get('kind') == 'run_end' else None,
        partial=bool(issues) or not parsed or not events or events[-1].get('kind') != 'run_end',
        artifact_issues=issues,
        verifier_calls=len(verifier) if observed else None,
        unknown_calls=sum(item.get('status') == 'UNKNOWN' for item in verifier) if observed else None,
        error_calls=sum(item.get('status') == 'ERROR' for item in verifier) if observed else None,
        rank_requests=len(decisions) if observed else None,
        memory_hits=sum(type(item.get('memory')) is dict and bool(item['memory'].get('matched_evidence_ids')) for item in metadata),
        llm_attempted_calls=sum(item.get('attempted') is True for item in metadata),
        llm_live_calls=sum(item.get('llm_live') is True for item in metadata),
        llm_fallback_reasons=[item.get('llm_fallback_reason') for item in metadata],
        native_seconds=stats.get('time'), tree_sha256=parsed.get('tree_fingerprint_sha256'))
    return row, parsed, decisions, responses


def archive_completed_runs(run_roots, context, output, collector, archiver):
    """Select by complete raw trace contracts, never by planning success.

    The existing archive collector validates records without writing. Failed
    validation gets private identity sets so it cannot contaminate later runs.
    Every exclusion remains explicit; valid PASS/UNKNOWN/ERROR episodes survive.
    """
    accepted, excluded = [], []
    event_ids, run_ids = set(), set()
    for run in run_roots:
        trial_events, trial_runs = set(event_ids), set(run_ids)
        try:
            records, source = collector(run, context, trial_events, trial_runs)
            accepted.append(dict(run_root=str(run), record_count=len(records),
                                 run_end_status=source['run_end_status'],
                                 status_counts=source['status_counts']))
            event_ids, run_ids = trial_events, trial_runs
        except Exception as error:
            # Collector ValueErrors are fixed local contract diagnostics. Never
            # serialize provider exceptions, source contents, or arbitrary reprs.
            reason = str(error) if type(error) is ValueError else type(error).__name__
            excluded.append(dict(run_root=str(run), error_type=type(error).__name__, reason=reason[:240]))
    selection = dict(accepted=accepted, excluded=excluded,
                     selection_basis='complete raw trace contract, independent of plan success')
    write(output / 'archive_selection.json', selection)
    if accepted:
        manifest = archiver([Path(item['run_root']) for item in accepted], context, output / 'evaluation_evidence')
        write(output / 'archive_result.json', manifest)
    return selection


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--run-dir', required=True, type=Path)
    value.add_argument('--llm-config', type=Path, default=ROOT / 'configs/llm.gpt6_luna.json')
    value.add_argument('--memory-snapshot', type=Path, default=DEFAULT_MEMORY)
    value.add_argument('--memory', choices=('off', 'on', 'both'), default='both')
    value.add_argument('--base-order', choices=('native', 'reverse', 'both'), default='native')
    value.add_argument('--seed', type=int, default=0)
    value.add_argument('--allow-live-api', action='store_true')
    value.add_argument('--prepared-run', action='store_true', help='Use this task\'s exclusive preparation folder; never resume an executed run')
    value.add_argument('--check-only', action='store_true', help='Check configuration/paths without API, physics, or output creation')
    return value


def execute(args):
    sys.path.insert(0, str(ROOT))
    from run_recorded import run_recorded
    from run_memory_multiseed import preserved_artifacts
    from run_langgraph_integration import artifacts
    output = safe(args.run_dir, True)
    if output.parent != ROOT / 'runs' or not 0 <= args.seed < 2 ** 32:
        raise ValueError('use a new direct-child run folder and valid seed')
    config_path = safe(args.llm_config)
    config = read(config_path)
    validate_config(config, args.allow_live_api)
    source_memory = safe(args.memory_snapshot, True)
    if args.memory != 'off' and not source_memory.is_file():
        raise ValueError('frozen memory snapshot is missing')
    if args.check_only:
        print(json.dumps({'status': 'PASS', 'model': config['model'], 'live_enabled': config['enabled'],
                          'variants': variants(args.base_order, args.memory), 'api_calls': 0, 'physics_runs': 0}))
        return 0
    if args.prepared_run:
        if not (output / 'task_start.json').is_file() or (output / 'execution_start.json').exists():
            raise ValueError('not an unused prepared task folder')
        before = read(output / 'preserved_before.json')
        if before != preserved_artifacts(output):
            raise ValueError('historical results changed since preparation')
    else:
        output.mkdir(exist_ok=False)
        before = preserved_artifacts(output)
        write(output / 'preserved_before.json', before)
    frozen = output / 'implementation'
    frozen.mkdir()
    sources = {**{'worker/' + name: ROOT / 'worker' / name for name in WORKER},
               **{'orchestrator/' + name: ROOT / 'orchestrator' / name for name in ORCHESTRATOR},
               **{'tools/' + name: ROOT / 'tools' / name for name in TOOLS}}
    for name, path in sources.items():
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe(path), target)
    source_hashes = {name: sha(path) for name, path in sources.items()}
    write(output / 'source_before.json', source_hashes)
    # Pin modules to this run's preserved implementation.
    import orchestrator
    orchestrator.__path__.insert(0, str(frozen / 'orchestrator'))
    sys.path.insert(0, str(frozen / 'worker'))
    from orchestrator.batch_budget import budget_summary
    from orchestrator.file_service import FileRankServer
    from orchestrator.ranking_pipeline import RankingPipeline
    from orchestrator.episode_archive import archive_episodes, _collect_run
    from orchestrator.langgraph_workflow import Workflow
    from evidence_memory import FrozenMemory, beam_context
    from memory_ranker import BEAM_SETUP
    loaded_sources = {name: str(module.__file__) for name, module in sys.modules.items()
                      if name in ('evidence_memory', 'memory_ranker', 'rank_contract')
                      or name.startswith('orchestrator.')}
    if any(not Path(path).is_relative_to(frozen) for path in loaded_sources.values()):
        raise RuntimeError('runtime modules were imported before source freeze')
    write(output / 'loaded_runtime_sources.json', loaded_sources)
    worker = frozen / 'worker'
    jobs = variants(args.base_order, args.memory)
    started = time.monotonic()
    write(output / 'llm_config.json', config)
    write(output / 'execution_start.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, model=config['model'], live_api_authorized=args.allow_live_api,
        seed=args.seed, jobs=jobs, shared_ledger='batch_ledger.sqlite',
        memory_source=str(source_memory), memory_scope='frozen development; known beam revisit',
        candidate_pruning=False, physics_bypass=False))
    rows, checks, run_roots = [], {}, []
    errors = []
    ledger = output / 'batch_ledger.sqlite'
    memory = None
    memory_source_hash = None
    context = None

    def record(argv, name):
        return run_recorded(WSL + argv, ROOT, output / name)

    def snapshot(name):
        command = record([linux(worker / 'compare_rank_hook.py'), '--snapshot', ASAP], name)
        if command['exit_code'] != 0:
            raise RuntimeError('ASAP environment snapshot failed')
        return read(output / name / 'stdout.log')

    try:
        environment = snapshot('environment_before')
        context = beam_context(environment, BEAM_SETUP, seed=args.seed)
        write(output / 'context.json', context)
        if args.memory != 'off':
            memory_source_hash = sha(source_memory)
            memory = output / 'frozen_memory.json'
            shutil.copyfile(source_memory, memory)
            memory_data = read(memory)
            FrozenMemory(memory)
            if not memory_data['records'] or any(item['context'] != context for item in memory_data['records']):
                raise ValueError('memory input/solver/seed/setup context does not match this experiment')
            write(output / 'memory_freeze.json', dict(utc=datetime.now(timezone.utc).isoformat(),
                source=str(source_memory), sha256=memory_source_hash, count=memory_data['record_count'],
                source_record_ids=memory_data['source_record_ids'], policy=memory_data['policy'],
                allow_hard_pruning=False, current_evaluation_used=False))
        features = record([linux(worker / 'describe_beam.py'), '--asap-root', ASAP,
                           '--output', linux(output / 'features.json')], 'feature_preparation')
        if features['exit_code'] != 0:
            raise RuntimeError('geometry feature preparation failed')
        feature_hash = sha(output / 'features.json')
        for base_order, mode in jobs:
            variant = base_order + '_' + mode
            run = output / variant
            run.mkdir()
            run_roots.append(run)
            run_id = output.name + '/' + variant
            counts = dict.fromkeys(('validate', 'prepare', 'execute', 'collect', 'summarize'), 0)
            def emit(kind, **fields):
                event = dict(kind=kind, event_id=str(uuid.uuid4()), utc=datetime.now(timezone.utc).isoformat(), **fields)
                with (run / 'orchestrator_events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event, allow_nan=False) + '\n')
            print(json.dumps({'phase': 'run_start', 'variant': variant}), flush=True)
            def validate(state):
                counts['validate'] += 1
                if sha(output / 'features.json') != feature_hash or (memory is not None and sha(memory) != memory_source_hash):
                    raise ValueError('frozen inputs changed')
                return {'status': 'RUNNING'}
            def prepare(state):
                counts['prepare'] += 1
                return {'artifacts': {'run_dir': str(run)}}
            def launch(state):
                counts['execute'] += 1
                pipeline = RankingPipeline(config, ledger, memory, context, mode,
                                           'noop' if base_order == 'native' else 'reverse_native')
                server = FileRankServer(run / 'ipc', run_id, 'pipeline', emit, rank_service=pipeline)
                argv = ['-u', linux(worker / 'run_remote_rank.py'), '--asap-root', ASAP,
                        '--native-log-dir', linux(run / 'native'), '--trace-file', linux(run / 'events.jsonl'),
                        '--run-id', run_id, '--seed', str(args.seed), '--ipc-dir', linux(run / 'ipc'),
                        '--remote-trace-file', linux(run / 'remote_audit.jsonl'),
                        '--features-file', linux(output / 'features.json'), '--response-timeout', '45']
                service_error = None
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        process = pool.submit(record, argv, variant + '/execute')
                        while not process.done():
                            if service_error is None:
                                try:
                                    server.serve_pending()
                                except Exception as error:
                                    service_error = type(error).__name__
                                    emit('service_error', error_type=service_error)
                            time.sleep(.05)
                        command = process.result()
                finally:
                    server.close()
                if command['exit_code'] != 0 or service_error:
                    raise RuntimeError('worker or ranking service failed')
                return {'metrics': {'process_exit': command['exit_code'], 'process_seconds': command['elapsed_seconds']}}
            def collect(state):
                counts['collect'] += 1
                command = record([linux(worker / 'export_graph_artifacts.py'), '--log-dir',
                    linux(run / ('native/dfs-heur-out/s' + str(args.seed) + '/original')),
                    '--output', linux(run / 'parsed_result.json'), '--trusted-local-tree'], variant + '/validation')
                parsed = read(run / 'parsed_result.json')
                return {'status': validation_status(parsed, command['exit_code']),
                        'metrics': {'validation_exit': command['exit_code'], 'plan_success': parsed['plan_success']}}
            def summarize(state):
                counts['summarize'] += 1
                return {'status': state['status']}
            callbacks = dict(validate=validate, prepare=prepare, execute=launch, collect=collect, summarize=summarize)
            initial = dict(run_id=run_id, config=dict(model=config['model'], config_sha256=sha(output / 'llm_config.json'),
                seed=args.seed, memory=mode, base_order=base_order, memory_sha256=memory_source_hash,
                feature_sha256=feature_hash, ledger=str(ledger)))
            sink = lambda event: emit('graph_event', graph_event=event)
            with Workflow(run / 'workflow.sqlite', callbacks, sink) as workflow:
                state = workflow.invoke(initial, run_id)
            artifact_before, count_before = artifacts(run), dict(counts)
            ledger_before = sha(ledger) if ledger.exists() else None
            with Workflow(run / 'workflow.sqlite', callbacks, sink) as workflow:
                reused = workflow.invoke(initial, run_id)
            checks[variant + '/completed_reuse'] = state == reused and counts == count_before
            checks[variant + '/no_duplicate_side_effects'] = artifacts(run) == artifact_before and (sha(ledger) if ledger.exists() else None) == ledger_before
            write(run / 'graph_state.json', state)
            write(run / 'reuse_check.json', dict(callbacks=counts, artifact_hashes=artifact_before,
                                                ledger_sha256=ledger_before, reused_state_equal=state == reused))
            checks[variant + '/environment_preserved'] = snapshot('environment_after_' + variant) == environment
            checks[variant + '/graph_end_completed'] = state['phase'] == 'END' and state['status'] in ('PASS', 'UNKNOWN', 'ERROR')
            if state['status'] == 'ERROR':
                errors.append(dict(variant=variant, error_type=state['error_type'], status=state['status']))
            row, parsed, decisions, responses = run_outcome(run, variant, state)
            rows.append(row)
            write(run / 'row.json', row)
            checks[variant + '/outcome_artifacts_readable'] = not row['artifact_issues']
            checks[variant + '/roundtrip_ids_and_applied_permutations'] = roundtrip_valid(decisions, responses)
            checks[variant + '/rng_preserved'] = all(item.get('rng_before') is not None and item.get('rng_before') == item.get('rng_after') for item in decisions)
            checks[variant + '/same_setup'] = parsed.get('setup') == BEAM_SETUP
            metadata = [item['metadata'] for item in responses if type(item.get('metadata')) is dict]
            checks[variant + '/selected_model'] = bool(metadata) and all(item.get('model_requested') == 'gpt-6-luna' for item in metadata)
            if not args.allow_live_api:
                checks[variant + '/api_disabled'] = bool(metadata) and all(item.get('llm_live') is False and item.get('attempted') is False and item.get('llm_fallback_reason') == 'llm_disabled' for item in metadata)
            if ledger.exists():
                write(run / 'batch_budget_after.json', budget_summary(ledger))
            print(json.dumps({'phase': 'run_end', **row}), flush=True)
        checks['all_planned_runs_completed'] = len(rows) == len(jobs) and all(row['graph_phase'] == 'END' for row in rows)
    except Exception as error:
        errors.append(dict(error_type=type(error).__name__))
        checks['execution_without_exception'] = False
    finally:
        # All worker futures have left their context managers before reaching
        # here. Archive usable episodes even after an exception or partial batch.
        if context is not None and run_roots:
            try:
                selection = archive_completed_runs(run_roots, context, output, _collect_run, archive_episodes)
                checks['all_attempted_traces_archivable'] = not selection['excluded']
                if selection['excluded']:
                    errors.append(dict(error_type='IncompleteOrInvalidEpisode', excluded_count=len(selection['excluded'])))
                if memory is not None and selection['accepted']:
                    archived = read(output / 'evaluation_evidence/memory_snapshot.json')
                    checks['evaluation_not_in_frozen_development'] = not (set(archived['source_record_ids']) & set(read(memory)['source_record_ids']))
            except Exception as error:
                checks['evaluation_archive_completed'] = False
                errors.append(dict(stage='archive', error_type=type(error).__name__))
        # A failure before normal post-processing must still expose the attempted
        # variant and every untouched raw log in the final result.
        reported = {row['variant'] for row in rows}
        for run in run_roots:
            if run.name not in reported:
                row, _, _, _ = run_outcome(run, run.name, {'status': 'ERROR', 'phase': None})
                rows.append(row)
                if not (run / 'row.json').exists():
                    write(run / 'row.json', row)
        checks.setdefault('all_planned_runs_completed', False)
        after = preserved_artifacts(output)
        write(output / 'preserved_after.json', after)
        checks['historical_results_preserved'] = before == after
        final_sources = {name: sha(path) for name, path in sources.items()}
        write(output / 'source_after.json', final_sources)
        checks['sources_frozen_during_execution'] = source_hashes == final_sources
        checks['frozen_source_copies_match'] = all(sha(frozen / name) == value for name, value in source_hashes.items())
        if memory is not None:
            checks['frozen_memory_preserved'] = sha(memory) == sha(source_memory) == memory_source_hash
    budget = budget_summary(ledger) if ledger.exists() else None
    if budget is not None:
        checks['shared_ledger_covers_all_rank_requests'] = budget['request_count'] == sum(row['rank_requests'] or 0 for row in rows)
        checks['shared_ledger_covers_all_runs'] = budget['run_count'] == len(rows)
        if not args.allow_live_api:
            checks['no_api_reservations_or_cost'] = budget['reserved_calls'] == 0 and Decimal(budget['budget_committed_usd']) == 0
    result = dict(status='PASS' if checks and all(checks.values()) and not errors else 'ERROR',
        status_scope='execution_and_integrity_not_plan_success',
        plan_status_counts={status: sum(row['plan_status'] == (None if status == 'NOT_RECORDED' else status) for row in rows) for status in ('PASS', 'UNKNOWN', 'ERROR', 'NOT_RECORDED')},
        all_plans_successful=len(rows) == len(jobs) and all(row['plan_status'] == 'PASS' for row in rows),
        not_run_variants=[base + '_' + mode for base, mode in jobs if base + '_' + mode not in {row['variant'] for row in rows}],
        checks=checks, errors=errors, rows=rows, budget=budget, model=config['model'],
        live_api_authorized=args.allow_live_api, live_model_quality='UNVERIFIED' if args.allow_live_api else 'NOT_RUN',
        preserved_file_count=len(before), elapsed_seconds=time.monotonic() - started)
    write(output / 'result.json', result)
    print(json.dumps({'completed': True, 'status': result['status'], 'runs': len(rows)}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(execute(parser().parse_args()))
    except Exception as error:
        print(json.dumps({'status': 'ERROR', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(1)
