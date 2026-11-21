"""Execute the registered skill/memory loop with real official ASAP physics.

Development mode runs two episodes and explicitly reviews template parameters
between them. Live evaluation requires a frozen library; explicit live development
starts empty and freezes only the first new episode before the second episode.
Scripted selections test plumbing only; no LLM result is claimed for them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

from run_luna_experiment import (ROOT, ASAP, WSL, WORKER, ORCHESTRATOR, TOOLS,
    safe, sha, read, write, linux, validate_config, validation_status,
    roundtrip_valid, run_outcome)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--run-dir', required=True, type=Path)
    value.add_argument('--llm-config', type=Path, default=ROOT / 'configs/llm.gpt6_luna.json')
    value.add_argument('--selector-mode', choices=('disabled', 'scripted_test', 'live'), default='scripted_test')
    value.add_argument('--frozen-library', type=Path)
    value.add_argument('--live-development', action='store_true',
                       help='Explicit two-episode live run; start empty, review and freeze between episodes.')
    value.add_argument('--allow-live-api', action='store_true')
    value.add_argument('--prepared-run', action='store_true')
    value.add_argument('--seed', type=int, default=0)
    return value


def validate_selection_mode(args):
    live = args.selector_mode == 'live'
    development = getattr(args, 'live_development', False)
    if live != bool(args.allow_live_api):
        raise ValueError('live selection requires explicit enabled config and authorization')
    if development and (not live or args.frozen_library is not None):
        raise ValueError('live development requires live selection and no preexisting library')
    if live and args.frozen_library is None and not development:
        raise ValueError('live evaluation requires a frozen library or explicit live development')


def first_live_episode_ready(row, metadata):
    """Do not spend a second episode's API budget after a failed first episode."""
    return (row['plan_status'] == 'PASS' and row['plan_success'] is True
            and len(metadata) == 3 and all(
                m.get('attempted') is True and m.get('llm_live') is True
                and m.get('selected_by') == 'llm'
                and m.get('skill_id') == 'native_order_v1'
                and m.get('llm_fallback_reason') is None
                and m.get('skill_error_type') is None for m in metadata))


def execute(args):
    sys.path.insert(0, str(ROOT))
    from run_recorded import run_recorded
    from run_memory_multiseed import preserved_artifacts
    from run_langgraph_integration import artifacts
    output = safe(args.run_dir, True)
    config = read(args.llm_config)
    validate_config(config, args.allow_live_api)
    validate_selection_mode(args)
    if output.parent != ROOT / 'runs' or not 0 <= args.seed < 2 ** 32:
        raise ValueError('new direct-child run folder and valid seed required')
    source_library = safe(args.frozen_library, True) if args.frozen_library else None
    if source_library and not all(safe(source_library / n).is_file() for n in ('library.json', 'memory.json')):
        raise ValueError('frozen library files missing')
    if args.prepared_run:
        if not (output / 'task_start.json').is_file() or (output / 'execution_start.json').exists():
            raise ValueError('prepared folder is absent or already executed')
        before = read(output / 'preserved_before.json')
        if before != preserved_artifacts(output):
            raise ValueError('prior results changed since task preparation')
    else:
        output.mkdir(exist_ok=False)
        before = preserved_artifacts(output)
        write(output / 'preserved_before.json', before)
    frozen = output / 'implementation'
    frozen.mkdir()
    names = {**{'worker/' + n: ROOT / 'worker' / n for n in WORKER},
             **{'orchestrator/' + n: ROOT / 'orchestrator' / n for n in
                ORCHESTRATOR + ('skill_library.py', 'skill_selector.py', 'skill_pipeline.py')},
             **{'tools/' + n: ROOT / 'tools' / n for n in TOOLS + ('run_skill_experiment.py', 'local_api_key.py')}}
    source_hashes = {n: sha(p) for n, p in names.items()}
    for name, path in names.items():
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe(path), target)
    write(output / 'source_before.json', source_hashes)
    import orchestrator
    orchestrator.__path__.insert(0, str(frozen / 'orchestrator'))
    sys.path.insert(0, str(frozen / 'worker'))
    from orchestrator.skill_library import SkillMemory, FrozenSkills
    from orchestrator.skill_pipeline import SkillPipeline
    from orchestrator.file_service import FileRankServer
    from orchestrator.langgraph_workflow import Workflow
    from orchestrator.batch_budget import budget_summary
    from evidence_memory import beam_context
    from memory_ranker import BEAM_SETUP
    loaded = {n: str(m.__file__) for n, m in sys.modules.items()
              if n in ('evidence_memory', 'memory_ranker', 'rank_contract') or n.startswith('orchestrator.')}
    if any(not Path(p).is_relative_to(frozen) for p in loaded.values()):
        raise RuntimeError('runtime source freeze failed')
    write(output / 'loaded_runtime_sources.json', loaded)
    write(output / 'llm_config.json', config)
    jobs = ['frozen_evaluation'] if source_library else ['development_empty', 'reuse_frozen']
    write(output / 'execution_start.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, jobs=jobs, selector_mode=args.selector_mode, live_authorized=args.allow_live_api,
        live_development=getattr(args, 'live_development', False),
        scope='known beam revisit, not held-out generalization', candidate_pruning=False,
        search_state_owner='unchanged ASAP', physics='actual unchanged ASAP verifier'))
    worker = frozen / 'worker'
    ledger = output / 'batch_ledger.sqlite'
    db = output / 'skill_memory.sqlite'
    checks, rows, errors = {}, [], []
    started = time.monotonic()
    memory_store = None

    def record(argv, name):
        return run_recorded(WSL + argv, ROOT, output / name)

    def snapshot(name):
        command = record([linux(worker / 'compare_rank_hook.py'), '--snapshot', ASAP], name)
        if command['exit_code'] != 0:
            raise RuntimeError('ASAP snapshot failed')
        return read(output / name / 'stdout.log')

    try:
        environment = snapshot('environment_before')
        context = beam_context(environment, BEAM_SETUP, seed=args.seed)
        write(output / 'context.json', context)
        command = record([linux(worker / 'describe_beam.py'), '--asap-root', ASAP,
                          '--output', linux(output / 'features.json')], 'feature_preparation')
        if command['exit_code'] != 0:
            raise RuntimeError('geometry feature preparation failed')
        feature_hash = sha(output / 'features.json')
        memory_store = SkillMemory(db)
        initial_library = output / 'frozen_initial'
        source_library_hashes = None
        if source_library:
            initial_library.mkdir()
            source_library_hashes = {n: sha(source_library / n) for n in ('library.json', 'memory.json')}
            for n in source_library_hashes:
                shutil.copyfile(source_library / n, initial_library / n)
            FrozenSkills(initial_library)
            records = read(initial_library / 'memory.json')['records']
            if not records or any(r['context'] != context for r in records):
                raise ValueError('development context differs from evaluation')
            write(output / 'input_library.json', dict(source=str(source_library), hashes=source_library_hashes))
        else:
            write(output / 'initial_freeze.json', memory_store.freeze(initial_library))
        for index, variant in enumerate(jobs):
            library = initial_library if index == 0 else output / 'frozen_reuse'
            input_hashes = {n: sha(library / n) for n in ('library.json', 'memory.json')}
            run = output / variant
            run.mkdir()
            run_id = output.name + '/' + variant
            counts = dict.fromkeys(('validate', 'prepare', 'execute', 'collect', 'summarize'), 0)
            def emit(kind, **fields):
                with (run / 'orchestrator_events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(kind=kind, event_id=str(uuid.uuid4()),
                        utc=datetime.now(timezone.utc).isoformat(), **fields), allow_nan=False) + '\n')
            def validate(state):
                counts['validate'] += 1
                if sha(output / 'features.json') != feature_hash or any(sha(library / n) != h for n, h in input_hashes.items()):
                    raise ValueError('frozen inputs changed')
                return {'status': 'RUNNING'}
            def prepare(state):
                counts['prepare'] += 1
                return {'artifacts': {'run_dir': str(run), 'library': str(library)}}
            def launch(state):
                counts['execute'] += 1
                pipeline = SkillPipeline(config, ledger, library, context, memory_store, args.selector_mode)
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
                return {'metrics': {'process_exit': command['exit_code'], 'seconds': command['elapsed_seconds']}}
            def collect(state):
                counts['collect'] += 1
                command = record([linux(worker / 'export_graph_artifacts.py'), '--log-dir',
                    linux(run / ('native/dfs-heur-out/s' + str(args.seed) + '/original')),
                    '--output', linux(run / 'parsed_result.json'), '--trusted-local-tree'], variant + '/validation')
                parsed = read(run / 'parsed_result.json')
                return {'status': validation_status(parsed, command['exit_code']),
                        'metrics': {'plan_success': parsed['plan_success']}}
            def summarize(state):
                counts['summarize'] += 1
                # Archive complete traces regardless of a PASS vs bounded UNKNOWN plan.
                report = memory_store.collect_completed(run, context)
                write(run / 'memory_collection.json', report)
                return {'status': state['status'], 'metrics': {'associated_outcomes': report['associated_outcomes']}}
            callbacks = dict(validate=validate, prepare=prepare, execute=launch, collect=collect, summarize=summarize)
            initial = dict(run_id=run_id, config=dict(seed=args.seed, selector_mode=args.selector_mode,
                config_sha256=sha(output / 'llm_config.json'), input_hashes=input_hashes, feature_sha256=feature_hash))
            sink = lambda event: emit('graph_event', graph_event=event)
            print(json.dumps(dict(phase='run_start', variant=variant, selection=args.selector_mode)), flush=True)
            with Workflow(run / 'workflow.sqlite', callbacks, sink) as graph:
                state = graph.invoke(initial, run_id)
            prior = (artifacts(run), sha(db), sha(ledger), dict(counts))
            with Workflow(run / 'workflow.sqlite', callbacks, sink) as graph:
                reused = graph.invoke(initial, run_id)
            checks[variant + '/completed_graph_reuse_without_side_effects'] = state == reused and prior == (artifacts(run), sha(db), sha(ledger), counts)
            write(run / 'graph_state.json', state)
            write(run / 'reuse_check.json', dict(callbacks=counts, state_equal=state == reused))
            row, parsed, decisions, responses = run_outcome(run, variant, state)
            row.pop('memory_hits', None)
            metadata = [r['metadata'] for r in responses]
            row.update(selected_skill_ids=[m.get('skill_id') for m in metadata],
                selected_by=[m.get('selected_by') for m in metadata],
                evidence_retrievals=sum(bool(m.get('retrieved_evidence_ids')) for m in metadata),
                learned_skill_invocations=sum(m.get('skill_id', '').startswith('observed_positive_') for m in metadata))
            rows.append(row)
            write(run / 'row.json', row)
            checks[variant + '/outcome_artifacts_readable'] = not row['artifact_issues']
            checks[variant + '/permutation_roundtrip'] = roundtrip_valid(decisions, responses)
            checks[variant + '/rng_preserved'] = bool(decisions) and all(d.get('rng_before') is not None and d['rng_before'] == d['rng_after'] for d in decisions)
            checks[variant + '/same_setup'] = parsed.get('setup') == BEAM_SETUP
            checks[variant + '/graph_complete'] = state['phase'] == 'END' and state['status'] in ('PASS', 'UNKNOWN')
            checks[variant + '/library_frozen_during_run'] = all(sha(library / n) == h for n, h in input_hashes.items())
            checks[variant + '/environment_preserved'] = snapshot('environment_after_' + variant) == environment
            if not args.allow_live_api:
                checks[variant + '/no_live_api'] = bool(metadata) and all(m.get('attempted') is False and m.get('llm_live') is False for m in metadata)
            if state['status'] == 'ERROR':
                raise RuntimeError('graph failed; see graph_state and phase events')
            if index == 0 and not source_library:
                if getattr(args, 'live_development', False):
                    ready = first_live_episode_ready(row, metadata)
                    checks['first_live_episode_ready_for_reuse'] = ready
                    if not ready:
                        raise RuntimeError('first live episode incomplete; second episode not started')
                report = read(run / 'memory_collection.json')
                registered = [memory_store.review_and_register(sid) for sid in report['proposals']]
                write(output / 'offline_registration.json', dict(cards=registered, arbitrary_code=False,
                    review_scope='allowlisted template, exact evidence provenance and permutation contract'))
                memory_store.close()
                memory_store = SkillMemory(db)
                write(output / 'reuse_freeze.json', memory_store.freeze(output / 'frozen_reuse'))
                if getattr(args, 'live_development', False):
                    write(output / 'freeze_transition.json', dict(
                        utc=datetime.now(timezone.utc).isoformat(), source_run_id=run_id,
                        source_trace_sha256=sha(run / 'events.jsonl'),
                        source_evidence_ids=sorted(r['record_id'] for r in memory_store.evidence.records()),
                        frozen_hashes={n: sha(output / 'frozen_reuse' / n)
                                       for n in ('library.json', 'memory.json')},
                        registered_skill_ids=sorted(c['skill_id'] for c in registered)))
                checks['provisional_skills_persisted_after_reopen'] = bool(registered) and len(memory_store.entries('registered_skill')) == len(registered)
            print(json.dumps(dict(phase='run_end', **row)), flush=True)
        write(output / 'memory_final.json', dict(evidence_records=len(memory_store.evidence.records()),
            entry_counts={k: len(memory_store.entries(k)) for k in ('retrieval', 'invocation', 'outcome', 'proposal', 'review', 'registered_skill', 'episode_collection')}))
        if not source_library:
            checks['experience_retrieved_on_revisit'] = rows[0]['evidence_retrievals'] == 0 and rows[1]['evidence_retrievals'] > 0
            if args.selector_mode == 'scripted_test':
                checks['registered_skill_selected_and_executed_on_revisit'] = rows[1]['learned_skill_invocations'] > 0
            checks['development_revisit_same_plan_and_physics_count'] = all(rows[0][k] == rows[1][k] for k in ('sequence', 'tree_sha256', 'verifier_calls', 'plan_status'))
        if source_library:
            checks['source_library_preserved'] = all(sha(source_library / n) == h for n, h in source_library_hashes.items())
    except Exception as error:
        errors.append(dict(error_type=type(error).__name__))
        checks['execution_without_exception'] = False
    finally:
        if memory_store is not None:
            memory_store.close()
        after = preserved_artifacts(output)
        write(output / 'preserved_after.json', after)
        checks['prior_results_preserved'] = before == after
        final_sources = {n: sha(p) for n, p in names.items()}
        write(output / 'source_after.json', final_sources)
        checks['sources_unchanged_during_run'] = source_hashes == final_sources
        checks['frozen_runtime_matches'] = all(sha(frozen / n) == h for n, h in source_hashes.items())
    checks['all_planned_runs_completed'] = len(rows) == len(jobs)
    budget = budget_summary(ledger) if ledger.exists() else None
    if budget and not args.allow_live_api:
        checks['zero_calls_and_cost'] = budget['reserved_calls'] == 0 and Decimal(budget['budget_committed_usd']) == 0
    result = dict(status='PASS' if all(checks.values()) and not errors else 'ERROR',
        status_scope='plumbing_and_integrity_not_model_quality', rows=rows, checks=checks, errors=errors,
        budget=budget, selector_mode=args.selector_mode, live_model_skill_selection='UNVERIFIED' if args.allow_live_api else 'NOT_RUN',
        elapsed_seconds=time.monotonic() - started, preserved_file_count=len(before),
        not_run_variants=[n for n in jobs if n not in {r['variant'] for r in rows}])
    write(output / 'result.json', result)
    print(json.dumps(dict(completed=True, status=result['status'], runs=len(rows))), flush=True)
    return 0 if result['status'] == 'PASS' else 1


def launch(args):
    if args.allow_live_api:
        from local_api_key import key_environment
        with key_environment(ROOT):
            return execute(args)
    return execute(args)


if __name__ == '__main__':
    try:
        raise SystemExit(launch(parser().parse_args()))
    except Exception as error:
        print(json.dumps(dict(status='ERROR', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1)
