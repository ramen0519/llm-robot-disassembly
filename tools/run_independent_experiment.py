"""Bounded real ASAP experiment with independent ranking and existing LangGraph.

Test modes inject explicit ID orders (not an LLM). Live mode requires a separate
enabled configuration and --allow-live-api. No credentials are loaded by this
runner and no API call occurs in test modes. Run directories are exclusive.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'worker'))

from run_recorded import run_recorded
from run_luna_experiment import safe, linux, read, write, sha, validate_config, CONFIG_KEYS

ASAP = '/opt/asap-research/ASAP'
WSL = [r'C:\Windows\System32\wsl.exe', '-d', 'ASAP-Ubuntu', '-u', 'root', '--cd', ASAP, '--']
PYTHON = '/opt/asap-research/envs/asap/bin/python'


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--run-dir', required=True, type=Path)
    value.add_argument('--mode', choices=('both_test_orders', 'test_id_ascending',
                       'test_id_descending', 'deterministic_memory', 'llm'), default='both_test_orders')
    value.add_argument('--memory', type=Path)
    value.add_argument('--llm-config', type=Path)
    value.add_argument('--allow-live-api', action='store_true')
    value.add_argument('--assembly-dir', default=ASAP + '/assets/beam_assembly/original')
    value.add_argument('--base-part', default='6', help='Use none for an unanchored assembly.')
    value.add_argument('--seed', type=int, default=0)
    value.add_argument('--budget', type=int, default=400)
    value.add_argument('--max-gripper', type=int, default=2)
    value.add_argument('--max-pose', type=int, default=3)
    value.add_argument('--wall-seconds', type=int, default=240)
    value.add_argument('--timeout', type=int)
    return value


def source_hashes():
    paths = sorted((ROOT / 'worker').glob('*.py')) + sorted((ROOT / 'orchestrator').glob('*.py'))
    paths += [ROOT / 'tools' / name for name in
              ('run_independent_experiment.py', 'run_recorded.py', 'run_luna_experiment.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def execute(args):
    from orchestrator.independent_ranker import IndependentPipeline
    from orchestrator.file_service import FileRankServer
    from orchestrator.langgraph_workflow import Workflow
    from task_memory import records_from_events, save_snapshot
    output = safe(args.run_dir, True)
    if (not 0 <= args.seed < 2**32 or args.budget <= 0 or args.max_gripper < 2
            or args.max_pose <= 0 or args.wall_seconds <= 0):
        raise ValueError('invalid bounded experiment settings')
    assembly = args.assembly_dir.replace('\\', '/')
    if ('..' in assembly.split('/') or not assembly.startswith((ASAP + '/assets/',
            '/mnt/c/dev/disassembly_codex_ready_v2/runs/'))):
        raise ValueError('assembly must be an approved ASAP asset or project-run input')
    base_part = None if args.base_part == 'none' else args.base_part
    memory_path = safe(args.memory, True) if args.memory else None
    memory_hash = sha(memory_path) if memory_path else None
    live = args.mode == 'llm'
    if live != bool(args.allow_live_api) or (live and args.llm_config is None):
        raise ValueError('live mode requires explicit separate API authorization/config')
    config = read(args.llm_config) if args.llm_config else {'enabled': False, 'max_calls': 0, 'max_usd': 0}
    if type(config) is not dict or not set(config).issubset(CONFIG_KEYS):
        raise ValueError('only documented config fields are allowed; never include inline credentials')
    if live:
        validate_config(config, True)
    elif config.get('enabled') is not False or config.get('max_calls') != 0 or config.get('max_usd') != 0:
        raise ValueError('test/comparator modes require disabled zero API budget')
    output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    write(output / 'source_before.json', hashes)
    for name in hashes:
        target = output / 'implementation' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe(ROOT / name), target)
    write(output / 'config.json', dict(vars(args), run_dir=str(output), memory=str(memory_path) if memory_path else None,
                                     llm_config=str(args.llm_config) if args.llm_config else None,
                                     api_config=config, source_archive_execution=False))
    modes = ['test_id_ascending', 'test_id_descending'] if args.mode == 'both_test_orders' else [args.mode]
    rows = []
    for mode in modes:
        run = output / mode
        run.mkdir()
        run_id = output.name + '/' + mode
        def emit(kind, **fields):
            with (run / 'orchestrator_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(kind=kind, event_id=str(uuid.uuid4()), **fields), allow_nan=False) + '\n')
        def validate(state):
            if hashes != source_hashes() or (memory_path and sha(memory_path) != memory_hash):
                raise ValueError('source_or_frozen_memory_changed')
            return {'status': 'RUNNING'}
        def prepare(state):
            return {'artifacts': {'run_dir': str(run)}}
        def launch(state):
            pipeline = IndependentPipeline(mode, config, output / 'batch_ledger.sqlite', memory_path)
            server = FileRankServer(run / 'ipc', run_id, 'pipeline', emit, rank_service=pipeline)
            argv = WSL + ['/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', str(args.wall_seconds) + 's',
                PYTHON, '-B', '-u', linux(ROOT / 'worker/run_independent_rank.py'), '--asap-root', ASAP,
                '--assembly-dir', assembly, '--output-dir', linux(run), '--run-id', run_id,
                '--seed', str(args.seed), '--budget', str(args.budget), '--max-gripper', str(args.max_gripper),
                '--max-pose', str(args.max_pose)]
            if base_part is not None:
                argv += ['--base-part', base_part]
            if args.timeout is not None:
                argv += ['--timeout', str(args.timeout)]
            service_error = None
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    process = pool.submit(run_recorded, argv, ROOT, run / 'execute')
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
            return {'status': 'RUNNING' if command['exit_code'] == 0 and not service_error else 'ERROR',
                    'metrics': {'process_exit': command['exit_code'], 'seconds': command['elapsed_seconds'],
                                'service_error': service_error}}
        def collect(state):
            native = run / ('native/dfs-independent/s' + str(args.seed)) / assembly.rsplit('/', 1)[-1]
            argv = WSL + [PYTHON, '-B', linux(ROOT / 'worker/validate_assembly.py'),
                    '--log-dir', linux(native), '--assembly-dir', assembly,
                    '--max-gripper', str(args.max_gripper), '--output', linux(run / 'parsed_result.json'),
                    '--trusted-local-tree']
            if base_part is not None:
                argv += ['--base-part', base_part]
            command = run_recorded(argv, ROOT, run / 'validation')
            result = read(run / 'parsed_result.json')
            return {'status': result['status'] if command['exit_code'] == 0 else 'ERROR',
                    'metrics': {'plan_success': result['plan_success']}}
        def summarize(state):
            trace = run / 'events.jsonl'
            if trace.is_file() and (run / 'context.json').is_file():
                events = [json.loads(line) for line in trace.read_text().splitlines()]
                try:
                    records = records_from_events(read(run / 'context.json'), events, str(trace), sha(trace))
                    frozen = save_snapshot(records, run / 'episode_memory.json')
                    write(run / 'memory_collection.json', frozen)
                except ValueError as error:
                    emit('memory_collection_failed', error_type=type(error).__name__)
                    return {'status': 'ERROR', 'metrics': {'memory_collection': 'ERROR_incomplete_or_invalid_evidence'}}
                return {'metrics': {'memory_collection': 'PASS', 'stored_observations': len(records)}}
            return {'metrics': {'memory_collection': 'NOT_RUN'}}
        print(json.dumps(dict(phase='run_start', mode=mode, verifier='actual_ASAP', live_API=live)), flush=True)
        with Workflow(run / 'workflow.sqlite', dict(validate=validate, prepare=prepare, execute=launch,
                         collect=collect, summarize=summarize), lambda event: emit('graph_event', graph_event=event)) as graph:
            state = graph.invoke(dict(run_id=run_id, config=dict(mode=mode, seed=args.seed,
                        memory_sha256=memory_hash, code_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())), run_id)
        write(run / 'graph_state.json', state)
        parsed = read(run / 'parsed_result.json') if (run / 'parsed_result.json').exists() else {}
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()] if (run / 'events.jsonl').exists() else []
        starts = [e for e in events if e['kind'] == 'verifier_start']
        outcomes = [e for e in events if e['kind'] == 'verifier_result']
        decisions = [e for e in events if e['kind'] == 'rank_decision']
        responses = [read(p) for p in sorted((run / 'ipc/responses').glob('*.json'))]
        finish = [e for e in events if e['kind'] == 'run_end']
        row = dict(mode=mode, status=state['status'], plan_success=parsed.get('plan_success'),
                   sequence=parsed.get('raw_stats', {}).get('sequence'),
                   verifier_calls=len(starts), verifier_results=len(outcomes),
                   verifier_statuses={s:sum(e['status']==s for e in outcomes) for s in ('PASS','UNKNOWN','ERROR')},
                   terminal_verifier_calls=sum(e.get('generation_id') is None for e in starts),
                   upstream_total_n_eval=parsed.get('raw_stats', {}).get('total_n_eval'),
                   rank_orders=[e['applied_order'] for e in decisions],
                   rank_fallbacks=sum(e['fallback_reason'] is not None for e in decisions),
                   live_api_attempts=sum(r['metadata'].get('llm_live') is True and r['metadata'].get('attempted') is True for r in responses),
                   heuristic_tripwire_calls=finish[-1].get('heuristic_tripwire_calls') if finish else None,
                   backtracking=None, backtracking_measurement='NOT_INSTRUMENTED',
                   source_preserved=hashes==source_hashes())
        row.update(state['metrics'])
        write(run / 'row.json', row)
        rows.append(row)
        print(json.dumps(row), flush=True)
    write(output / 'source_after.json', source_hashes())
    result = dict(rows=rows, original_sources_preserved=hashes==source_hashes(),
                  memory_frozen=not memory_path or sha(memory_path)==memory_hash,
                  live_llm_verification='RUN' if live else 'NOT_RUN',
                  note='Test orders validate wiring, not LLM performance or cross-task generalization.')
    write(output / 'result.json', result)
    return 0 if result['original_sources_preserved'] and result['memory_frozen'] and all(r['status'] in ('PASS','UNKNOWN') for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(execute(parser().parse_args()))
