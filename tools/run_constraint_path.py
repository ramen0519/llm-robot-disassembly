"""One bounded conditional-memory and actual two-leg path repair run.

This runner preserves the established five-node LangGraph, file IPC and real
ASAP worker. Fixture decisions exercise wiring; they are not LLM observations.
Live execution requires an enabled pinned config and --allow-live-api. A shared
--ledger can enforce one approved cap across several exclusive run directories.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
sys.path.insert(0, str(ROOT / 'tools'))

from run_recorded import run_recorded
from run_luna_experiment import CONFIG_KEYS, linux, read, safe, sha, validate_config, write

ASAP = '/opt/asap-research/ASAP'
WSL = [r'C:\Windows\System32\wsl.exe', '-d', 'ASAP-Ubuntu', '-u', 'root', '--cd', ASAP, '--']
PYTHON = '/opt/asap-research/envs/asap/bin/python'
APPROVED_EXT4_ASSEMBLY = '/opt/asap-research/prototype_completion_20261018/07009'


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--run-dir', required=True, type=Path)
    value.add_argument('--mode', choices=('fixture', 'llm'), default='fixture')
    value.add_argument('--candidate-mode', choices=('controlled', 'llm'), default='controlled',
                       help='LLM mode also chooses subsequent part order; controlled initial probe is labelled.')
    value.add_argument('--memory', type=Path)
    value.add_argument('--skill-memory', type=Path, help='Frozen, actually replayed path-skill validation evidence.')
    value.add_argument('--llm-config', type=Path)
    value.add_argument('--allow-live-api', action='store_true')
    value.add_argument('--ledger', type=Path, help='Explicit shared batch ledger; defaults inside this new run.')
    value.add_argument('--controlled-repair', action='store_true',
                       help='Label the initial fixed failure/removal probes separately from LLM choices.')
    value.add_argument('--assembly-dir', default=ASAP + '/assets/beam_assembly/original')
    value.add_argument('--base-part', default='6', help='Use none for an unanchored assembly.')
    value.add_argument('--seed', type=int, default=0)
    value.add_argument('--budget', type=int, default=400)
    value.add_argument('--max-gripper', type=int, default=2)
    value.add_argument('--max-pose', type=int, default=3)
    value.add_argument('--wall-timeout', type=int, default=240)
    value.add_argument('--timeout', type=int, help='ASAP soft planning timeout; the wall timeout is separate.')
    value.add_argument('--response-timeout', type=int, default=90)
    return value


def _excluded(value):
    raw = str(value).replace('\\', '/').lower()
    normalized = '/'.join(part for part in raw.split('/') if part != '.')
    if normalized.startswith(('e:', '/mnt/e')) or '..' in raw.split('/'):
        raise ValueError('excluded_path')
    return value


def checked(value, runs_only=False):
    # This lexical guard runs before Path.absolute, lstat or any other access.
    return safe(_excluded(value), runs_only)


def source_hashes():
    paths = sorted((ROOT / 'worker').glob('*.py')) + sorted((ROOT / 'orchestrator').glob('*.py'))
    paths += [ROOT / 'tools' / name for name in (
        'run_prototype_completion.py', 'run_constraint_path.py', 'run_recorded.py', 'run_luna_experiment.py', 'local_api_key.py')]
    return {path.relative_to(ROOT).as_posix(): sha(checked(path)) for path in paths}


def write_jsonl(path, values):
    with checked(path, True).open('x', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def read_jsonl(path, issues=None):
    path = checked(path, True)
    if not path.is_file():
        return []
    values = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError('event_is_not_an_object')
            values.append(value)
        except (ValueError, TypeError) as error:
            if issues is None:
                raise
            issues.append({'artifact': path.name, 'line': number, 'error_type': type(error).__name__})
    return values


def validate_arguments(args):
    output = checked(args.run_dir, True)
    if output.exists():
        raise FileExistsError('run_directory_must_be_new')
    if (not 0 <= args.seed < 2**32 or args.budget <= 0 or args.max_gripper < 2
            or args.max_pose <= 0 or args.wall_timeout <= 0
            or not 0 < args.response_timeout <= 120
            or (args.timeout is not None and args.timeout <= 0)):
        raise ValueError('invalid_bounded_experiment_settings')
    assembly = str(_excluded(args.assembly_dir)).replace('\\', '/').rstrip('/')
    if (assembly != APPROVED_EXT4_ASSEMBLY
            and not assembly.startswith((ASAP + '/assets/', '/mnt/c/dev/disassembly_codex_ready_v2/runs/'))):
        raise ValueError('assembly_must_be_an_approved_ASAP_asset_or_project_run_input')
    base_part = None if args.base_part == 'none' else args.base_part
    if base_part is not None and (not base_part or '/' in base_part or '\\' in base_part):
        raise ValueError('invalid_base_part_ID')
    memory = checked(args.memory, True) if args.memory else None
    config_path = checked(args.llm_config) if args.llm_config else None
    live = args.mode == 'llm'
    if live != bool(args.allow_live_api) or (live and config_path is None):
        raise ValueError('live_mode_requires_explicit_API_authorization_and_config')
    config = read(config_path) if config_path else {'enabled': False, 'max_calls': 0, 'max_usd': 0}
    if type(config) is not dict or not set(config).issubset(CONFIG_KEYS):
        raise ValueError('only_documented_config_fields_no_inline_credentials')
    if live:
        # Key-dependent validation occurs only inside the narrowly scoped live
        # key context. Reject incorrect policy before reading any stored key.
        if (set(config) != CONFIG_KEYS or config.get('enabled') is not True
                or config.get('model') != 'gpt-6-luna'
                or config.get('base_url') != 'https://api.openai.com/v1'
                or config.get('api_key_env') != 'OPENAI_API_KEY'):
            raise ValueError('live_requires_complete_pinned_Luna_config')
    elif config.get('enabled') is not False or config.get('max_calls') != 0 or config.get('max_usd') != 0:
        raise ValueError('fixture_requires_disabled_zero_API_budget')
    ledger = checked(args.ledger, True) if args.ledger else output / 'batch_ledger.sqlite'
    for suffix in ('-wal', '-shm', '-journal'):
        checked(Path(str(ledger) + suffix), True)
    return output, assembly, base_part, memory, config_path, config, ledger, live


def decision_summary(decisions):
    """Record exposure versus cited selection without claiming causal benefit."""
    offered, selected, source_evidence, current_candidates, kinds = set(), set(), set(), set(), {}
    invocations = []
    citation_requests = 0
    for decision in decisions:
        if type(decision) is not dict:
            raise ValueError('prototype_decision_must_be_object')
        response = decision.get('response', {})
        metadata = response.get('metadata', {}) if type(response) is dict else {}
        for source in (decision, metadata):
            if type(source) is not dict:
                continue
            for key in ('offered_hint_ids', 'offered_transfer_hint_ids', 'transfer_hint_ids', 'exposed_transfer_hint_ids'):
                if type(source.get(key)) is list:
                    offered.update(v for v in source[key] if type(v) is str)
            for key in ('used_hint_ids', 'selected_transfer_hint_ids', 'used_transfer_hint_ids'):
                if type(source.get(key)) is list:
                    selected.update(v for v in source[key] if type(v) is str)
            if type(source.get('used_source_evidence_ids')) is list:
                source_evidence.update(v for v in source['used_source_evidence_ids'] if type(v) is str)
            if type(source.get('used_current_candidate_ids')) is list:
                current_candidates.update(v for v in source['used_current_candidate_ids'] if type(v) is str)
        citation_requests += bool(metadata.get('used_hint_ids'))
        kind = decision.get('decision_source', metadata.get('ranking_source', 'unclassified'))
        kinds[str(kind)] = kinds.get(str(kind), 0) + 1
        invocation = metadata.get('repair_invocation', decision.get('repair_invocation', decision.get('invocation')))
        if type(invocation) is dict and invocation.get('skill_id') == 'retry_after_state_change_v1':
            invocations.append(invocation)
    return {'offered_transfer_hint_ids': sorted(offered), 'selected_transfer_hint_ids': sorted(selected),
            'transfer_exposure_count': len(offered), 'transfer_selection_count': len(selected),
            'requests_citing_transfer': citation_requests,
            'selected_source_evidence_ids': sorted(source_evidence),
            'selected_current_candidate_ids': sorted(current_candidates),
            'decision_sources': kinds, 'repair_invocations': invocations,
            'cross_task_causal_benefit_demonstrated': False}


def assembly_validation_status(parsed, exit_code):
    """Follow validate_assembly: valid artifacts exit zero even without a plan.

    The legacy Beam-only validator has a different UNKNOWN exit convention.
    A process exit alone is never evidence that the actual plan succeeded.
    """
    if type(parsed) is not dict:
        return 'ERROR'
    status, valid, success = parsed.get('status'), parsed.get('validation_ok'), parsed.get('plan_success')
    if valid is True and exit_code == 0:
        if status == 'PASS' and success is True:
            return 'PASS'
        if status == 'UNKNOWN' and success is False:
            return 'UNKNOWN'
    return 'ERROR'


def execute(args):
    from orchestrator.path_repair_ranker import PathRepairPipeline
    from orchestrator.file_service import FileRankServer
    from orchestrator.langgraph_workflow import Workflow
    from orchestrator.repair_library import collect_outcomes
    from task_memory import records_from_events, save_snapshot

    output, assembly, base_part, memory, config_path, config, ledger, live = validate_arguments(args)
    memory_hash = sha(memory) if memory else None
    skill_memory = checked(args.skill_memory, True) if args.skill_memory else None
    skill_memory_hash = sha(skill_memory) if skill_memory else None
    output.mkdir(parents=True, exist_ok=False)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    run_id = 'constraint-path:' + hashlib.sha256(str(output).encode('utf-8')).hexdigest()[:12] + ':' + args.mode
    hashes = source_hashes()
    write(output / 'source_before.json', hashes)
    for name in hashes:
        target = output / 'implementation' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked(ROOT / name), checked(target, True))
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write(output / 'config.json', {
        'arguments': arguments, 'run_id': run_id, 'api_config': config,
        'ledger': str(ledger), 'frozen_memory_sha256': memory_hash,
        'frozen_skill_memory_sha256': skill_memory_hash,
        'source_archive_execution': False, 'physics_backend': 'actual_ASAP_RedMax',
        'decision_backend': 'actual_API_if_successful' if live else 'fixture_mock_model_decision',
        'controlled_initial_probe': args.controlled_repair,
    })

    def emit(kind, **fields):
        with checked(output / 'orchestrator_events.jsonl', True).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(kind=kind, event_id=str(uuid.uuid4()), **fields), allow_nan=False) + '\n')

    def validate(state):
        if hashes != source_hashes() or (memory and sha(memory) != memory_hash):
            raise ValueError('source_or_frozen_memory_changed')
        if skill_memory and sha(skill_memory) != skill_memory_hash:
            raise ValueError('frozen_skill_memory_changed')
        return {'status': 'RUNNING'}

    def prepare(state):
        return {'artifacts': {'run_dir': str(output), 'run_id': run_id}}

    def launch(state):
        if live:
            from local_api_key import key_environment
            key_context = key_environment()
        else:
            key_context = nullcontext('NOT_ACCESSED')
        with key_context as key_source:
            if live:
                validate_config(config, True)
            emit('decision_backend_start', mode=args.mode, key_source=key_source,
                 controlled_initial_probe=args.controlled_repair)
            pipeline = PathRepairPipeline(args.mode, config, ledger, memory_path=memory,
                                          candidate_mode=args.candidate_mode,
                                          controlled_initial=args.controlled_repair,
                                          skill_memory_path=skill_memory)
            server = None
            service_error, command = None, None
            try:
                server = FileRankServer(output / 'ipc', run_id, 'pipeline', emit, rank_service=pipeline)
                argv = WSL + ['/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', str(args.wall_timeout) + 's',
                    PYTHON, '-B', '-u', linux(ROOT / 'worker/run_constraint_path.py'), '--asap-root', ASAP,
                    '--assembly-dir', assembly, '--output-dir', linux(output), '--run-id', run_id,
                    '--seed', str(args.seed), '--budget', str(args.budget), '--max-gripper', str(args.max_gripper),
                    '--max-pose', str(args.max_pose), '--response-timeout', str(args.response_timeout)]
                if args.controlled_repair:
                    argv += ['--controlled-path-probe']
                if base_part is not None:
                    argv += ['--base-part', base_part]
                if args.timeout is not None:
                    argv += ['--timeout', str(args.timeout)]
                with ThreadPoolExecutor(max_workers=1) as pool:
                    process = pool.submit(run_recorded, argv, ROOT, output / 'execute')
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
                try:
                    write_jsonl(output / 'prototype_decisions.jsonl', pipeline.decisions)
                    write_jsonl(output / 'request_body_previews.jsonl', pipeline.previews)
                finally:
                    if server is not None:
                        server.close()
                    else:
                        pipeline.close()
            return {'status': 'RUNNING' if command['exit_code'] == 0 and not service_error else 'ERROR',
                    'metrics': {'process_exit': command['exit_code'], 'seconds': command['elapsed_seconds'],
                                'service_error': service_error, 'key_source': key_source}}

    def collect(state):
        native = output / ('native/dfs-independent/s' + str(args.seed)) / assembly.rsplit('/', 1)[-1]
        argv = WSL + ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', '45s', PYTHON, '-B',
                     linux(ROOT / 'worker/validate_assembly.py'), '--log-dir', linux(native),
                     '--assembly-dir', assembly, '--max-gripper', str(args.max_gripper),
                     '--output', linux(output / 'parsed_result.json'), '--trusted-local-tree']
        if base_part is not None:
            argv += ['--base-part', base_part]
        command = run_recorded(argv, ROOT, output / 'validation')
        result = read(output / 'parsed_result.json') if (output / 'parsed_result.json').is_file() else {}
        return {'status': assembly_validation_status(result, command['exit_code']),
                'metrics': {'plan_success': result.get('plan_success'), 'validator_exit': command['exit_code']}}

    def summarize(state):
        from worker.constraint_memory import load_snapshot
        memory_file = output / 'constraint_memory.json'
        try:
            snapshot = load_snapshot(memory_file)
            paths = read(output / 'path_outcomes.json')
            return {'metrics': {'constraint_memory_collection': 'PASS',
                    'stored_conditional_constraints': snapshot['record_count'],
                    'generic_episode_memory_export': 'NOT_USED_action_field_requires_schedule_sidecar',
                    'repair_invocation_count': len(paths['outcomes'])}}
        except (ValueError, KeyError, TypeError, OSError) as error:
            emit('constraint_collection_failed', error_type=type(error).__name__)
            return {'status': 'ERROR', 'metrics': {'constraint_memory_collection': 'ERROR'}}

    print(json.dumps({'phase': 'run_start', 'run_id': run_id, 'mode': args.mode,
                      'physics': 'actual_ASAP_RedMax', 'live_API_enabled': live,
                      'controlled_initial_probe': args.controlled_repair}), flush=True)
    with Workflow(output / 'workflow.sqlite', dict(validate=validate, prepare=prepare, execute=launch,
                  collect=collect, summarize=summarize), lambda event: emit('graph_event', graph_event=event)) as graph:
        state = graph.invoke({'run_id': run_id, 'config': {
            'mode': args.mode, 'seed': args.seed, 'memory_sha256': memory_hash,
            'assembly_dir': assembly, 'controlled_repair': args.controlled_repair,
            'code_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}}, run_id)
    write(output / 'graph_state.json', state)
    parsed = read(output / 'parsed_result.json') if (output / 'parsed_result.json').is_file() else {}
    final_parse_issues = []
    events = read_jsonl(output / 'events.jsonl', final_parse_issues)
    starts = [e for e in events if e.get('kind') == 'verifier_start']
    outcomes = [e for e in events if e.get('kind') == 'verifier_result']
    decisions = [e for e in events if e.get('kind') == 'rank_decision']
    responses = [read(p) for p in sorted(checked(output / 'ipc/responses', True).glob('*.json'))]
    finish = [e for e in events if e.get('kind') == 'run_end']
    source_after = source_hashes()
    preserved = source_after == hashes
    memory_frozen = memory is None or sha(memory) == memory_hash
    response_metadata = [r.get('metadata', {}) for r in responses]
    row = {
        'run_id': run_id, 'mode': args.mode, 'status': state['status'],
        'plan_success': parsed.get('plan_success'), 'sequence': parsed.get('raw_stats', {}).get('sequence'),
        'verifier_backend': 'actual_ASAP_RedMax', 'model_decision_fixture': args.mode == 'fixture',
        'controlled_initial_probe': args.controlled_repair,
        'verifier_calls': len(starts), 'verifier_results': len(outcomes),
        'verifier_statuses': {s: sum(e.get('status') == s for e in outcomes) for s in ('PASS', 'UNKNOWN', 'ERROR')},
        'terminal_verifier_calls': sum(e.get('generation_id') is None for e in starts),
        'upstream_total_n_eval': parsed.get('raw_stats', {}).get('total_n_eval'),
        'rank_orders': [e['applied_order'] for e in decisions],
        'rank_fallbacks': sum(e.get('fallback_reason') is not None for e in decisions),
        'live_api_attempts': sum(m.get('llm_live') is True and m.get('attempted') is True for m in response_metadata),
        'live_valid_responses': sum(r.get('metadata', {}).get('llm_live') is True
                                    and r.get('metadata', {}).get('attempted') is True
                                    and r.get('metadata', {}).get('ranking_source') in ('path_llm', 'prototype_llm')
                                    and r.get('fallback_reason') is None for r in responses),
        'fixture_responses': sum(m.get('ranking_source') == 'fixture_path_choice' for m in response_metadata),
        'controlled_probe_responses': sum(m.get('ranking_source') == 'controlled_functional_candidate_order'
                                           for m in response_metadata),
        'path_decision_fallbacks': sum(r.get('fallback_reason') is not None and
            ':path_request' in str(r.get('request_id')) for r in responses),
        'heuristic_tripwire_calls': finish[-1].get('heuristic_tripwire_calls') if finish else None,
        'source_preserved': preserved, 'memory_frozen': memory_frozen,
        'backtracking': None, 'backtracking_measurement': 'NOT_INSTRUMENTED',
        'performance_comparison': 'NOT_RUN', 'generalization_effect': 'UNVERIFIED',
    }
    row.update(state['metrics'])
    if final_parse_issues:
        row['artifact_parse_issues'] = final_parse_issues
    if not preserved or not memory_frozen or final_parse_issues:
        row['status'] = 'ERROR'
    path_result = read(output / 'path_outcomes.json') if (output / 'path_outcomes.json').is_file() else {}
    paths = path_result.get('outcomes', [])
    row.update(path_check_success_calls=path_result.get('path_check_success_calls'),
               constraint_records=path_result.get('constraint_records'),
               path_repair_invocations=len(paths),
               path_repair_feasible=sum(p.get('feasible') is True for p in paths),
               physically_noncollinear_repairs=sum(p.get('feasible') is True and p['geometry']['noncollinear'] for p in paths),
               constraint_memory='constraint_memory.json',
               legacy_single_action_replay_supported=False,
               candidate_ranking=('LLM_with_labelled_controlled_initial_probe' if args.candidate_mode == 'llm'
                                  else 'controlled_functional_fixture_not_LLM_ranking'))
    row['skill_memory_frozen'] = skill_memory is None or sha(skill_memory) == skill_memory_hash
    if not row['skill_memory_frozen']:
        row['status'] = 'ERROR'
    write(output / 'source_after.json', source_after)
    write(output / 'result.json', row)
    print(json.dumps(row, allow_nan=False), flush=True)
    return 0 if row['status'] in ('PASS', 'UNKNOWN') and preserved and memory_frozen else 1


if __name__ == '__main__':
    raise SystemExit(execute(parser().parse_args()))
