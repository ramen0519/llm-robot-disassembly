"""Opt-in order-only LLM/fixture with unchanged native ASAP lower-level search.

No credentials are read in fixture/prepare mode. Existing runs are never reused.
Live mode needs --allow-live-api and a separately authorized enabled config.
"""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(LANGSMITH_TRACING='false', LANGCHAIN_TRACING_V2='false', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from run_recorded import run_recorded
from run_luna_experiment import safe, sha, read, write, linux
from orchestrator.llm_ranker import DEFAULTS, _json
from orchestrator.sequence_ranker import SequenceRankService, FixtureSequenceService, PROTOCOL, MAX_BODY_BYTES
from orchestrator.sequence_knowledge import build_native_snapshot
from orchestrator.file_service import FileRankServer
from orchestrator.langgraph_workflow import Workflow

ASAP = '/opt/asap-research/ASAP'
PYTHON = '/opt/asap-research/envs/asap/bin/python'
WSL = ['C:/Windows/System32/wsl.exe', '-d', 'ASAP-Ubuntu', '-u', 'root', '--cd', ASAP, '--']


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--assembly-dir', default=ASAP + '/assets/beam_assembly/original')
    p.add_argument('--base-part', default=None)
    p.add_argument('--mode', choices=('prepare', 'fixture', 'llm'), default='prepare')
    p.add_argument('--fixture-order', choices=('ascending', 'descending'), default='ascending')
    p.add_argument('--memory', type=Path)
    p.add_argument('--skills', type=Path)
    p.add_argument('--scored-skills', type=Path,
                   help='Explicit frozen registered scorer bridges; also requires --relational-knowledge.')
    p.add_argument('--lessons', type=Path)
    p.add_argument('--relational-knowledge', type=Path,
                   help='Opt-in frozen relational cards; does not authorize candidate skill execution.')
    p.add_argument('--llm-config', type=Path)
    p.add_argument('--allow-live-api', action='store_true')
    p.add_argument('--use-registered-key', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--budget', type=int, default=256)
    p.add_argument('--path-limit', type=int, default=256)
    p.add_argument('--stability-limit', type=int, default=256)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--outer-timeout', type=int, default=3900)
    p.add_argument('--response-timeout', type=int, default=120)
    p.add_argument('--max-gripper', type=int, default=3)
    p.add_argument('--max-pose', type=int, default=128)
    return p


def validate(args):
    output = safe(args.run_dir, True)
    if output.exists():
        raise ValueError('new_run_directory_required')
    assembly = args.assembly_dir.replace('\\', '/')
    if '..' in assembly.split('/') or not assembly.startswith((ASAP + '/assets/',
                 '/mnt/c/dev/disassembly_codex_ready_v2/runs/')):
        raise ValueError('assembly_outside_allowed_roots')
    if args.seed < 0 or args.seed >= 2**32 or args.max_gripper < 2 or any(getattr(args, n) < 1 for n in
        ('budget', 'path_limit', 'stability_limit', 'timeout', 'outer_timeout', 'response_timeout', 'max_pose')):
        raise ValueError('invalid_bounded_settings')
    if args.outer_timeout <= args.timeout or args.response_timeout > 120:
        raise ValueError('invalid_deadline_settings')
    live = args.mode == 'llm'
    if live != args.allow_live_api or (live and not args.llm_config):
        raise ValueError('live_requires_explicit_authorized_configuration')
    if args.use_registered_key and not live:
        raise ValueError('offline_mode_must_not_load_credentials')
    config = read(args.llm_config) if args.llm_config else dict(DEFAULTS)
    if type(config) is not dict or set(config) - set(DEFAULTS):
        raise ValueError('unsupported_config_or_inline_credentials')
    if live:
        if config.get('enabled') is not True or config.get('model') != 'gpt-6.1-sol':
            raise ValueError('live_profile_requires_enabled_sol_configuration')
        if config.get('api_key_env') != 'OPENAI_API_KEY':
            raise ValueError('live_profile_requires_host_only_openai_key')
    elif config.get('enabled') is not False or config.get('max_calls') != 0 or config.get('max_usd') != 0:
        raise ValueError('offline_requires_zero_api_budget')
    pins, knowledge = {}, {}
    for label in ('memory', 'skills', 'lessons', 'scored_skills'):
        path = getattr(args, label, None)
        knowledge[label] = ()
        if path:
            path = safe(path, True)
            pins[str(path)] = sha(path)
            knowledge[label] = read(path)
    knowledge['relational'] = None
    if getattr(args, 'relational_knowledge', None):
        from orchestrator.relational_knowledge import load_library
        knowledge['relational'] = load_library(args.relational_knowledge)
        path = safe(args.relational_knowledge, True)
        pins[str(path)] = sha(path)
    return output, config, pins, knowledge


def source_hashes():
    paths = [p for folder in ('worker', 'orchestrator', 'tools') for p in (ROOT/folder).glob('*.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def execute(args, *, service_factory=None, knowledge_limits=None, allow_fixture_knowledge=False):
    output, config, pins, knowledge = validate(args)
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    write(output/'source_before.json', sources)
    write(output/'frozen_input_pins.json', pins)
    write(output/'configuration.json', dict(profile=PROTOCOL, arguments={k: str(v) if isinstance(v, Path) else v
          for k, v in vars(args).items()}, api_config=config, model_input_byte_cap=MAX_BODY_BYTES))
    run = output/'episode'
    run_id = output.name
    def emit(kind, **fields):
        with (output/'orchestrator_events.jsonl').open('a', encoding='utf8') as f:
            f.write(_json(dict(kind=kind, **fields)) + '\n')
    if allow_fixture_knowledge and args.mode != 'fixture':
        raise ValueError('fixture_knowledge_requires_offline_fixture_mode')
    def no_network(*unused):
        raise RuntimeError('fixture_transport_must_not_be_called')
    service = SequenceRankService(config, output/'api_ledger.sqlite', memory_records=knowledge['memory'],
                                  skills=knowledge['skills'], lessons=knowledge['lessons'],
                                  knowledge_limits=knowledge_limits,
                                  allow_fixture_skills=allow_fixture_knowledge,
                                  relational_knowledge=knowledge['relational'],
                                  scored_skills=knowledge['scored_skills'],
                                  transport=no_network if allow_fixture_knowledge else None)
    if args.mode == 'llm' and service._configuration()[0] is not None:
        raise ValueError('invalid_live_policy')
    if args.mode == 'llm' and not os.environ.get('OPENAI_API_KEY', '').strip():
        raise ValueError('missing_api_key_no_worker_or_call_started')
    pipeline = FixtureSequenceService(service, args.fixture_order) if args.mode == 'fixture' else service
    if service_factory is not None:
        if args.mode != 'fixture':
            raise ValueError('injected_candidate_service_requires_no_api_fixture_mode')
        pipeline = service_factory(service)
    def check_unchanged(state):
        if source_hashes() != sources or any(sha(Path(p)) != h for p, h in pins.items()):
            raise ValueError('source_or_frozen_input_changed')
        return {'status': 'RUNNING'}
    def launch(state):
        server = None if args.mode == 'prepare' else FileRankServer(run/'ipc', run_id, 'pipeline', emit, rank_service=pipeline)
        argv = WSL + ['/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', str(args.outer_timeout)+'s',
            PYTHON, '-B', '-u', linux(ROOT/'worker/run_sequence_native.py'), '--asap-root', ASAP,
            '--assembly-dir', args.assembly_dir, '--output-dir', linux(run), '--run-id', run_id]
        for label in ('seed', 'budget', 'path_limit', 'stability_limit', 'timeout', 'response_timeout', 'max_gripper', 'max_pose'):
            argv += ['--'+label.replace('_', '-'), str(getattr(args, label))]
        if args.base_part is not None:
            argv += ['--base-part', args.base_part]
        if args.mode == 'prepare':
            argv += ['--prepare-only']
        service_error = None
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_recorded, argv, ROOT, output/'execute')
                while not future.done():
                    if server and not service_error:
                        try:
                            server.serve_pending()
                        except Exception as e:
                            service_error = type(e).__name__
                            emit('service_error', error_type=service_error)
                    time.sleep(.05)
                command = future.result()
        finally:
            if server:
                server.close()
        return {'status': 'RUNNING' if command['exit_code'] == 0 and not service_error else 'ERROR',
                'metrics': dict(process_exit=command['exit_code'], service_error=service_error,
                                outer_elapsed_seconds=command['elapsed_seconds'])}
    def collect(state):
        name = 'preparation.json' if args.mode == 'prepare' else 'worker_result.json'
        if not (run/name).is_file():
            return {'status': 'ERROR', 'metrics': {'worker_result': 'MISSING'}}
        result = read(run/name)
        responses = [read(p) for p in sorted((run/'ipc/responses').glob('*.json'))] if (run/'ipc/responses').is_dir() else []
        metrics = dict(worker_status=result['status'], success=result.get('success'),
            max_depth=result.get('connected', {}).get('max_depth'),
            physics_counts=result.get('budget', {}).get('counts', {}),
            actual_API_attempts=sum(r['metadata'].get('llm_live') is True and r['metadata'].get('attempted') is True for r in responses),
            memory_citing_decisions=sum(bool(r['metadata'].get('past_memory_evidence_ids')) for r in responses),
            lesson_citing_decisions=sum(bool(r['metadata'].get('lesson_evidence_ids')) for r in responses),
            episode_citing_decisions=sum(bool(r['metadata'].get('episode_evidence_ids')) for r in responses),
            sequence_skill_selections=sum(r['metadata'].get('selected_skill_id') is not None for r in responses),
            sequence_skill_executions=sum(r['metadata'].get('skill_execution', {}).get('status') == 'PROPOSED'
                                          for r in responses if r['metadata'].get('skill_execution')),
            decision_errors=sum(r.get('fallback_reason') is not None for r in responses),
            fixture=args.mode == 'fixture', live_llm_test='RUN' if args.mode == 'llm' else 'NOT_RUN')
        write(output/'metrics.json', metrics)
        return {'status': 'ERROR' if state['status'] == 'ERROR' else result['status'], 'metrics': metrics}
    def summarize(state):
        preserved = source_hashes() == sources and all(sha(Path(p)) == h for p, h in pins.items())
        if (run/'worker_result.json').is_file():
            result = read(run/'worker_result.json')
            # A native attempt can contain six paths and several support probes.
            # Keep it raw; do not forge a single explicit-action memory record.
            write(output/'native_observations.json', build_native_snapshot(read(run/'context.json'),
                result['attempts'], str(run/'worker_result.json'), sha(run/'worker_result.json')))
        write(output/'source_after.json', source_hashes())
        return {'status': state['status'] if preserved else 'ERROR', 'metrics': {'sources_and_frozen_inputs_preserved': preserved}}
    with Workflow(output/'workflow.sqlite', dict(validate=check_unchanged,
        prepare=lambda state: {'artifacts': {'run_dir': str(run)}}, execute=launch,
        collect=collect, summarize=summarize), lambda e: emit('graph_event', graph_event=e)) as graph:
        state = graph.invoke(dict(run_id=run_id, config=dict(profile=PROTOCOL, mode=args.mode,
            source_sha256=hashlib.sha256(_json(sources).encode()).hexdigest(), frozen_inputs=pins)), run_id)
    write(output/'result.json', state)
    print(json.dumps(dict(status=state['status'], **state['metrics'])), flush=True)
    return 1 if state['status'] == 'ERROR' else 0


def main():
    args = parser().parse_args()
    validate(args)  # Reject mode/path/config mismatches before credential access.
    if args.use_registered_key:
        from local_api_key import key_environment
        context = key_environment()
    else:
        context = nullcontext()
    with context:
        return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
