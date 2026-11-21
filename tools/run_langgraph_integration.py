"""Real LangGraph/ASAP integration exercise. Live API is explicitly disabled."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from run_recorded import run_recorded
from run_memory_multiseed import preserved_artifacts

ROOT = Path(__file__).absolute().parent.parent
ASAP = '/opt/asap-research/ASAP'
WSL = [r'C:\Windows\System32\wsl.exe', '-d', 'ASAP-Ubuntu', '-u', 'root',
       '--cd', ASAP, '--', '/opt/asap-research/envs/asap/bin/python', '-B']
WORKER = ('rank_contract.py', 'rank_hook.py', 'run_rank_hook.py', 'compare_rank_hook.py',
          'validate_baseline.py', 'remote_rank_client.py', 'run_remote_rank.py',
          'describe_beam.py', 'export_graph_artifacts.py')
ORCHESTRATOR = ('__init__.py', 'file_service.py', 'llm_ranker.py', 'langgraph_workflow.py')
TOOLS = ('run_langgraph_integration.py', 'run_recorded.py', 'run_memory_multiseed.py', 'run_asap_baseline.py')
VARIANTS = ('native_trace', 'remote_noop', 'remote_disabled', 'remote_reverse')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linux(path):
    path = Path(path).absolute()
    if '..' in path.parts or not path.is_relative_to(ROOT):
        raise ValueError('path must stay in C: project')
    return '/mnt/c/dev/disassembly_codex_ready_v2/' + path.relative_to(ROOT).as_posix()


def artifacts(run):
    output = {}
    for name in ('execute', 'validation', 'native', 'ipc/requests', 'ipc/responses'):
        folder = run / name
        if folder.exists():
            for path in sorted(folder.rglob('*')):
                if path.is_symlink() or path.is_junction():
                    raise ValueError('links excluded')
                if path.is_file():
                    output[path.relative_to(run).as_posix()] = sha(path)
    for name in ('events.jsonl', 'remote_audit.jsonl', 'parsed_result.json'):
        path = run / name
        if path.is_file():
            output[name] = sha(path)
    return output


def execute(output):
    sys.path.insert(0, str(ROOT))
    output = Path(output).absolute()
    if '..' in output.parts or output.parent != ROOT / 'runs':
        raise ValueError('use a direct-child project run path')
    if any(path.is_symlink() or path.is_junction() for path in (output,) + tuple(output.parents)):
        raise ValueError('links are excluded')
    if not (output / 'preserved_before.json').is_file():
        raise ValueError('use the prepared run directory')
    before = read(output / 'preserved_before.json')
    if before != preserved_artifacts(output):
        raise ValueError('historical results changed before this experiment')
    frozen = output / 'implementation'
    frozen.mkdir()
    paths = {**{'worker/' + name: ROOT / 'worker' / name for name in WORKER},
             **{'orchestrator/' + name: ROOT / 'orchestrator' / name for name in ORCHESTRATOR},
             **{'tools/' + name: ROOT / 'tools' / name for name in TOOLS}}
    for name, source in paths.items():
        destination = frozen / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    write(output / 'source_before.json', {name: sha(path) for name, path in paths.items()})
    # Use immutable orchestration/worker copies; also rehash current sources.
    import orchestrator
    orchestrator.__path__.insert(0, str(frozen / 'orchestrator'))
    from orchestrator.file_service import FileRankServer
    from orchestrator.langgraph_workflow import Workflow
    started = time.monotonic()
    write(output / 'execution_start.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, python=sys.executable, variants=VARIANTS,
        versions={name: importlib.metadata.version(name) for name in ('langgraph', 'langgraph-checkpoint-sqlite')},
        live_api_authorized=False, live_llm_calls=0, seed=0, base_part='6', budget=400,
        max_gripper=2, max_pose=3, timeout=None, memory_enabled=False,
        purpose='connection verification, not LLM quality or generalization evaluation'))
    worker = frozen / 'worker'
    def record(args, label):
        return run_recorded(WSL + args, ROOT, output / label)
    def snapshot(label):
        result = record([linux(worker / 'compare_rank_hook.py'), '--snapshot', ASAP], label)
        if result['exit_code'] != 0:
            raise RuntimeError('environment snapshot failed')
        return read(output / label / 'stdout.log')
    results = []
    checks = {}
    try:
        environment = snapshot('environment_before')
        described = record([linux(worker / 'describe_beam.py'), '--asap-root', ASAP,
                            '--output', linux(output / 'features.json')], 'feature_preparation')
        if described['exit_code'] != 0:
            raise RuntimeError('geometry description failed')
        features_sha = sha(output / 'features.json')
        previous = record([linux(worker / 'export_graph_artifacts.py'), '--log-dir',
            linux(ROOT / 'runs/m1_wsl_20261016/beam/native/dfs-heur-out/s0/original'),
            '--output', linux(output / 'historical_baseline.json'), '--trusted-local-tree'],
            'historical_baseline_validation')
        if previous['exit_code'] != 0:
            raise RuntimeError('historical baseline validation failed')
        for variant in VARIANTS:
            print(json.dumps({'phase': 'run_start', 'variant': variant, 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)
            run = output / variant
            run.mkdir()
            run_id = output.name + '/' + variant
            counter = {'events': 0}
            callbacks_count = {phase: 0 for phase in ('validate', 'prepare', 'execute', 'collect', 'summarize')}
            def emit(kind, **fields):
                counter['events'] += 1
                event = dict(kind=kind, log_event_id=run_id + ':o' + str(counter['events']),
                             utc=datetime.now(timezone.utc).isoformat(), **fields)
                with (run / 'orchestrator_events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
            def validate(state):
                callbacks_count['validate'] += 1
                if state['config']['live_api_authorized'] or sha(output / 'features.json') != features_sha:
                    raise ValueError('experiment contract changed')
                return {'status': 'RUNNING'}
            def prepare(state):
                callbacks_count['prepare'] += 1
                return {'status': 'RUNNING', 'artifacts': {'run_dir': str(run)}}
            def launch(state):
                callbacks_count['execute'] += 1
                server = None
                if variant == 'native_trace':
                    args = [linux(worker / 'run_rank_hook.py'), '--mode', 'trace']
                else:
                    backend = {'remote_noop': 'noop', 'remote_disabled': 'llm_disabled',
                               'remote_reverse': 'reverse_native'}[variant]
                    server = FileRankServer(run / 'ipc', run_id, backend, emit)
                    args = [linux(worker / 'run_remote_rank.py'), '--ipc-dir', linux(run / 'ipc'),
                            '--remote-trace-file', linux(run / 'remote_audit.jsonl'),
                            '--features-file', linux(output / 'features.json'), '--response-timeout', '30']
                args += ['--asap-root', ASAP, '--native-log-dir', linux(run / 'native'),
                         '--trace-file', linux(run / 'events.jsonl'), '--run-id', run_id, '--seed', '0']
                service_error = None
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pending = pool.submit(record, ['-u'] + args, variant + '/execute')
                        while not pending.done():
                            if server is not None and service_error is None:
                                try:
                                    server.serve_pending()
                                except Exception as error:
                                    service_error = type(error).__name__
                                    emit('service_error', error_type=service_error)
                            time.sleep(0.05)
                        command = pending.result()
                finally:
                    if server is not None:
                        server.close()
                if command['exit_code'] != 0 or service_error:
                    raise RuntimeError('worker or service failed')
                return {'status': 'RUNNING', 'metrics': {'process_seconds': command['elapsed_seconds'],
                        'process_exit': command['exit_code'], 'live_llm_calls': 0}}
            def collect(state):
                callbacks_count['collect'] += 1
                command = record([linux(worker / 'export_graph_artifacts.py'), '--log-dir',
                    linux(run / 'native/dfs-heur-out/s0/original'), '--output', linux(run / 'parsed_result.json'),
                    '--trusted-local-tree'], variant + '/validation')
                parsed = read(run / 'parsed_result.json')
                return {'status': parsed['status'] if command['exit_code'] == 0 else 'ERROR', 'artifacts': dict(state['artifacts'], parsed_result=str(run / 'parsed_result.json')),
                        'metrics': dict(state['metrics'], validation_exit=command['exit_code'],
                                        plan_success=parsed['plan_success'])}
            def summarize(state):
                callbacks_count['summarize'] += 1
                return {'status': state.get('status', 'ERROR')}
            callbacks = dict(validate=validate, prepare=prepare, execute=launch, collect=collect, summarize=summarize)
            initial = dict(run_id=run_id, status='READY', config=dict(variant=variant, feature_sha256=features_sha,
                           seed=0, live_api_authorized=False, memory_enabled=False))
            sink = lambda event: emit('graph_event', graph_event=event)
            with Workflow(run / 'workflow.sqlite', callbacks, sink) as flow:
                state = flow.invoke(initial, thread_id=run_id)
            original_artifacts, original_counts = artifacts(run), dict(callbacks_count)
            with Workflow(run / 'workflow.sqlite', callbacks, sink) as flow:
                reused = flow.invoke(initial, thread_id=run_id)
            checks[variant + '/completed_state_reused'] = state == reused
            checks[variant + '/no_callbacks_reexecuted'] = callbacks_count == original_counts
            checks[variant + '/no_worker_or_rank_artifacts_rewritten'] = original_artifacts == artifacts(run)
            write(run / 'graph_state.json', state)
            write(run / 'reuse_check.json', {'checks': {key: value for key, value in checks.items() if key.startswith(variant + '/')},
                                            'callback_counts': callbacks_count, 'artifacts_sha256': original_artifacts})
            checks[variant + '/environment_unchanged'] = environment == snapshot('environment_after_' + variant)
            row = dict(variant=variant, graph_status=state.get('status'), graph_phase=state.get('phase'),
                       callbacks=callbacks_count, graph_error_type=state.get('error_type'))
            if (run / 'parsed_result.json').exists():
                parsed = read(run / 'parsed_result.json')
                events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
                verifier = [event for event in events if event['kind'] == 'verifier_result']
                generations = [event for event in events if event['kind'] == 'generation']
                decisions = [event for event in events if event['kind'] == 'rank_decision']
                row.update(plan_success=parsed['plan_success'], sequence=parsed['raw_stats'].get('sequence'),
                           verifier_calls=len(verifier), unknown_calls=sum(event['status'] == 'UNKNOWN' for event in verifier),
                           rank_requests=len(decisions), changed_orders=sum(event['native_order'] != event['applied_order'] for event in generations),
                           rank_fallbacks=[event['fallback_reason'] for event in decisions],
                           tree_sha256=parsed['tree_fingerprint_sha256'], native_seconds=parsed['raw_stats']['time'])
                checks[variant + '/plan_verified'] = parsed['status'] == state['status'] == 'PASS'
                checks[variant + '/validation_exit_zero'] = state['metrics'].get('validation_exit') == 0
                checks[variant + '/graph_end'] = state.get('phase') == 'END'
                checks[variant + '/all_candidates_preserved'] = all(sorted(event['native_order']) == sorted(event['applied_order']) for event in generations)
                checks[variant + '/rank_rng_unchanged'] = all(event['rng_before'] == event['rng_after'] for event in decisions)
                checks[variant + '/unique_event_ids'] = len({event['event_id'] for event in events}) == len(events)
                if variant != 'native_trace':
                    requests = [read(path) for path in sorted((run / 'ipc/requests').glob('*.json'))]
                    responses = [read(path) for path in sorted((run / 'ipc/responses').glob('*.json'))]
                    audit = [json.loads(line) for line in (run / 'remote_audit.jsonl').read_text().splitlines()]
                    ids = {event['request']['request_id'] for event in decisions}
                    checks[variant + '/remote_roundtrip_used'] = (len(decisions) == len(requests) == len(responses) == len(audit) == len(generations) > 0
                        and ids == {item['request']['request_id'] for item in requests}
                        == {item['request_id'] for item in responses} == {item['request_id'] for item in audit})
                    expected_backend = {'remote_noop': 'noop', 'remote_disabled': 'llm_disabled', 'remote_reverse': 'reverse_native'}[variant]
                    expected_reason = 'llm_disabled' if variant == 'remote_disabled' else None
                    checks[variant + '/expected_service_backend'] = bool(responses) and all(item['metadata']['backend'] == expected_backend for item in responses)
                    checks[variant + '/expected_service_fallback'] = bool(responses) and all(item['fallback_reason'] == expected_reason for item in responses)
                    checks[variant + '/expected_worker_fallback'] = bool(audit) and all(item['fallback_reason'] == expected_reason for item in audit)
                    checks[variant + '/live_api_zero'] = bool(responses) and all(item['metadata']['llm_live'] is False and item['metadata']['attempted'] is False for item in responses)
                    checks[variant + '/expected_order_change'] = bool(generations) and all(
                        event['applied_order'] == (list(reversed(event['native_order'])) if variant == 'remote_reverse' else event['native_order'])
                        for event in generations)
            else:
                checks[variant + '/plan_verified'] = False
            results.append(row)
            print(json.dumps({'phase': 'run_end', **row}, ensure_ascii=False), flush=True)
        reference = read(output / 'native_trace/parsed_result.json')
        historical = read(output / 'historical_baseline.json')
        checks['original_m1_same_setup'] = historical['setup'] == reference['setup']
        checks['original_m1_same_full_tree'] = historical['tree_fingerprint_sha256'] == reference['tree_fingerprint_sha256']
        checks['original_m1_same_stats_except_time'] = ({k: v for k, v in historical['raw_stats'].items() if k != 'time'} ==
                    {k: v for k, v in reference['raw_stats'].items() if k != 'time'})
        for variant in VARIANTS:
            if not (output / variant / 'parsed_result.json').is_file():
                continue
            parsed = read(output / variant / 'parsed_result.json')
            checks[variant + '/same_setup'] = parsed['setup'] == reference['setup']
            if variant in ('remote_noop', 'remote_disabled'):
                checks[variant + '/same_full_tree'] = parsed['tree_fingerprint_sha256'] == reference['tree_fingerprint_sha256']
                checks[variant + '/same_stats_except_time'] = ({k: v for k, v in parsed['raw_stats'].items() if k != 'time'} ==
                    {k: v for k, v in reference['raw_stats'].items() if k != 'time'})
    except Exception as error:
        checks['execution_completed_without_exception'] = False
        write(output / 'execution_error.json', {'error_type': type(error).__name__,
                                               'completed_variants': len(results)})
    finally:
        after = preserved_artifacts(output)
        write(output / 'preserved_after.json', after)
        final_sources = {name: sha(path) for name, path in paths.items()}
        write(output / 'source_after.json', final_sources)
        checks['historical_results_preserved'] = before == after
        checks['implementation_preserved'] = read(output / 'source_before.json') == final_sources
        checks['frozen_copies_match'] = all(sha(frozen / name) == value for name, value in final_sources.items())
    write(output / 'comparison.json', dict(status='PASS' if checks and all(checks.values()) else 'ERROR',
        checks=checks, rows=results, live_llm_calls=0, llm_quality_evaluation='NOT_RUN',
        preserved_file_count=len(before), elapsed_seconds=time.monotonic() - started))
    print(json.dumps({'completed': True, 'integrity': all(checks.values()), 'runs': len(results)}), flush=True)
    return 0 if checks and all(checks.values()) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    raise SystemExit(execute(parser.parse_args().run_dir))
