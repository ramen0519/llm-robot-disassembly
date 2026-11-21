"""Compare unchanged ASAP/rule policies on a preselected official data subset."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

from run_memory_multiseed import ROOT, file_hash, preserved_artifacts, write
from run_recorded import run_recorded

WSL = [r'C:\Windows\System32\wsl.exe', '-d', 'ASAP-Ubuntu', '-u', 'root',
       '--cd', '/opt/asap-research/ASAP', '--', '/opt/asap-research/envs/asap/bin/python', '-B']
ASAP = '/opt/asap-research/ASAP'
VARIANTS = ('native', 'trace', 'id_ascending', 'reverse_native')
WORKER = ('rank_contract.py', 'rank_hook.py', 'compare_rank_hook.py', 'validate_baseline.py',
          'run_assembly_hook.py', 'validate_assembly.py', 'audit_assembly_events.py',
          'prepare_assembly_meshes.py')
TOOLS = ('run_assembly_comparison.py', 'prepare_assembly_dataset.py', 'run_memory_multiseed.py',
         'run_asap_baseline.py', 'run_recorded.py')


def linux(path):
    path = path.resolve()
    return '/mnt/c/dev/disassembly_codex_ready_v2/' + path.relative_to(ROOT).as_posix()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def hashes(folder):
    result = {}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink() or path.is_junction():
            raise ValueError('No links permitted in experiment data')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = file_hash(path)
    return result


def source_map():
    return {name: ROOT / 'worker' / name for name in WORKER} | {name: ROOT / 'tools' / name for name in TOOLS}


def summarize(output, selected):
    spec = importlib.util.spec_from_file_location('assembly_event_audit', output / 'implementation/audit_assembly_events.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks, rows, pairs = {}, [], []
    all_event_ids, all_run_ids = [], []
    for item in selected:
        by_variant, normalized_argv = {}, {}
        for variant in VARIANTS:
            folder = output / 'evaluations' / item['id'] / variant
            parsed = read(folder / 'parsed_result.json')
            executed = read(folder / 'execute/command.json')
            events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
            mode = variant if variant in ('native', 'trace') else 'rank'
            ranker = 'noop' if mode != 'rank' else variant
            audited = module.audit_events(events, mode, ranker, item['part_ids'])
            write(folder / 'event_audit.json', audited)
            prefix = item['id'] + '/' + variant
            checks[prefix + '/process_exit_zero'] = executed['exit_code'] == 0
            checks[prefix + '/artifacts_valid'] = parsed['validation_ok']
            checks[prefix + '/events_valid'] = audited['status'] == 'PASS'
            checks[prefix + '/returned_normally'] = audited['metrics']['run_completed_normally']
            all_event_ids.extend(e['event_id'] for e in events)
            all_run_ids.append(events[0]['run_id'])
            argv = list(events[0]['native_argv'])
            argv[argv.index('--log-dir') + 1] = '<output>'
            normalized_argv[variant] = argv
            stats = parsed['raw_stats'] or {}
            counts = audited['metrics']['verifier_status_counts'] if variant != 'native' else None
            row = {'assembly_id': item['id'], 'parts': item['parts'], 'variant': variant,
                   'status': parsed['status'], 'plan_success': parsed['plan_success'],
                   'process_exit_code': executed['exit_code'], 'stop_msg': stats.get('stop_msg'),
                   'sequence': stats.get('sequence'), 'terminal_parts': parsed['terminal_parts'],
                   'native_n_eval': stats.get('n_eval'), 'native_total_n_eval': stats.get('total_n_eval'),
                   'verifier_calls': audited['metrics']['verifier_calls'], 'verifier_status_counts': counts,
                   'native_seconds': stats.get('time'), 'process_seconds': executed['elapsed_seconds'],
                   'changed_generations': audited['metrics']['changed_generations'],
                   'rank_requests': audited['metrics']['rank_requests'],
                   'tree_sha256': parsed['tree_fingerprint_sha256'],
                   'artifact_validation_ok': parsed['validation_ok'], 'event_integrity': audited['status']}
            rows.append(row)
            by_variant[variant] = row
        native, trace = by_variant['native'], by_variant['trace']
        def stripped(variant):
            folder = output / 'evaluations' / item['id'] / variant / 'native/dfs-heur-out/s0' / item['id']
            return {key: value for key, value in read(folder / 'stats.json').items() if key != 'time'}
        equality = {'assembly_id': item['id'],
                    'native_trace_tree_equal': native['tree_sha256'] == trace['tree_sha256'],
                    'native_trace_stats_except_time_equal': stripped('native') == stripped('trace'),
                    'same_cli_all_variants': all(v == normalized_argv['native'] for v in normalized_argv.values()),
                    'same_setup_all_variants': len({json.dumps(read(output / 'evaluations' / item['id'] / v /
                        'native/dfs-heur-out/s0' / item['id'] / 'setup.json'), sort_keys=True) for v in VARIANTS}) == 1}
        pairs.append(equality)
        checks[item['id'] + '/same_cli'] = equality['same_cli_all_variants']
        checks[item['id'] + '/same_setup'] = equality['same_setup_all_variants']
        # A wall-clock cutoff can legitimately change tree length. Keep that observation explicit.
        if native['stop_msg'] != 'timeout' and trace['stop_msg'] != 'timeout':
            checks[item['id'] + '/native_trace_tree_equal'] = equality['native_trace_tree_equal']
            checks[item['id'] + '/native_trace_stats_equal'] = equality['native_trace_stats_except_time_equal']
    checks['all_event_ids_unique'] = len(all_event_ids) == len(set(all_event_ids))
    checks['all_run_ids_unique'] = len(all_run_ids) == len(set(all_run_ids))
    checks['old_results_preserved'] = read(output / 'preserved_before.json') == read(output / 'preserved_after.json')
    checks['implementation_preserved'] = read(output / 'implementation_sha256.json') == read(output / 'implementation_final_sha256.json')
    inputs = read(output / 'input_hashes.json')
    checks['all_data_preserved'] = all(value == inputs['before'] for value in inputs.values())
    checks['source_binary_beam_preserved'] = read(output / 'environment_before/stdout.log') == read(output / 'environment_after/stdout.log')
    write(output / 'comparison.json', {'status': 'PASS' if all(checks.values()) else 'ERROR', 'checks': checks,
        'rows': rows, 'native_trace_pairs': pairs, 'seed': 0, 'memory_enabled': False, 'llm_calls': 0,
        'internal_geometry_operations': None, 'generalization_claim': None,
        'preserved_file_count': len(read(output / 'preserved_before.json'))})
    return all(checks.values())


def execute(output):
    output = output.resolve()
    if output.parent != ROOT / 'runs' or not (output / 'mesh_preparation.json').is_file():
        raise ValueError('Use the prepared direct-child experiment directory')
    selected = read(output / 'selected_assemblies.json')
    if [(x['id'], x['parts']) for x in selected] != [('00561', 5), ('02628', 7), ('01289', 9)]:
        raise ValueError('Preselected data differs')
    write(output / 'execution_start.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'argv': sys.argv, 'cwd': str(ROOT), 'executable': sys.executable})
    order = [(item, variant) for i, item in enumerate(selected)
             for variant in (VARIANTS if i % 2 == 0 else tuple(reversed(VARIANTS)))]
    write(output / 'protocol.json', {'assemblies': [x['id'] for x in selected], 'variants': VARIANTS,
        'execution_order': [{'assembly': item['id'], 'variant': v} for item, v in order],
        'seed': 0, 'base_part': None, 'budget': 100, 'max_gripper': 2, 'max_pose': 3,
        'pose_reuse': 0, 'timeout_seconds': 180, 'num_proc': 1, 'early_term': True,
        'cache': 'no preexisting SDF, --disable-save-sdf all runs', 'memory_enabled': False,
        'llm_calls': 0, 'planner': 'dfs', 'generator': 'heur-out',
        'goal': 'official complete plan ending at a feasible singleton, no fixed base',
        'rules': {'id_ascending': 'unchanged lexicographic ID order', 'reverse_native': 'unchanged reverse offered heur-out order'},
        'verifier_unit': '_simulate calls including final leaf; not individual geometry operations',
        'budget_note': 'native budget excludes leaf; pose loop may overshoot; timeout is a soft planner cutoff',
        'native_control': 'runpy of official CLI with no hooks; trace retains native order and counts physics',
        'scope': 'three preselected official assemblies, one seed, exploratory comparison; no transfer/memory claim'})
    frozen = output / 'implementation'
    frozen.mkdir()
    sources = source_map()
    for name, source in sources.items():
        shutil.copyfile(source, frozen / name)
    write(output / 'implementation_sha256.json', {name: file_hash(source) for name, source in sources.items()})
    before = {'raw': hashes(output / 'raw_assemblies'), 'dense': hashes(output / 'dense_assemblies'),
              'archive': file_hash(output / 'test_assembly.zip')}
    input_log = {'before': before}
    def record(args, label):
        return run_recorded(WSL + args, ROOT, output / label)
    def progress(kind, **values):
        event = {'kind': kind, 'utc': datetime.now(timezone.utc).isoformat(), **values}
        with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)
    started = time.monotonic()
    try:
        if record([linux(frozen / 'compare_rank_hook.py'), '--snapshot', ASAP], 'environment_before')['exit_code'] != 0:
            raise RuntimeError('Environment snapshot failed')
        for index, (item, variant) in enumerate(order, 1):
            run = output / 'evaluations' / item['id'] / variant
            run.mkdir(parents=True)
            mode = variant if variant in ('native', 'trace') else 'rank'
            ranker = variant if mode == 'rank' else 'noop'
            progress('run_start', index=index, total=len(order), assembly=item['id'], variant=variant)
            write(run / 'config.json', {'mode': mode, 'ranker': ranker, 'assembly': item['id'], 'parts': item['part_ids'],
                                       'seed': 0, 'base_part': None, 'memory_enabled': False})
            executed = record(['-u', linux(frozen / 'run_assembly_hook.py'), '--asap-root', ASAP,
                '--assembly-dir', linux(output / 'dense_assemblies'), '--assembly-id', item['id'],
                '--native-log-dir', linux(run / 'native'), '--trace-file', linux(run / 'events.jsonl'),
                '--run-id', output.name + '/' + item['id'] + '/' + variant, '--mode', mode, '--ranker', ranker,
                '--seed', '0', '--max-gripper', '2', '--budget', '100', '--max-pose', '3', '--pose-reuse', '0', '--timeout', '180'],
                'evaluations/{}/{}/execute'.format(item['id'], variant))
            validated = record([linux(frozen / 'validate_assembly.py'), '--log-dir',
                linux(run / 'native/dfs-heur-out/s0' / item['id']), '--assembly-dir',
                linux(output / 'dense_assemblies' / item['id']), '--max-gripper', '2', '--output',
                linux(run / 'parsed_result.json'), '--trusted-local-tree'], 'evaluations/{}/{}/validation'.format(item['id'], variant))
            observed = {'raw': hashes(output / 'raw_assemblies'), 'dense': hashes(output / 'dense_assemblies'),
                        'archive': before['archive']}
            input_log[item['id'] + '/' + variant] = observed
            if observed != before:
                raise RuntimeError('Experiment data changed during run')
            parsed = read(run / 'parsed_result.json')
            progress('run_end', index=index, total=len(order), assembly=item['id'], variant=variant,
                     process_exit=executed['exit_code'], validation_exit=validated['exit_code'],
                     status=parsed['status'], plan_success=parsed['plan_success'],
                     native_seconds=(parsed['raw_stats'] or {}).get('time'))
            # Include normal bounded failures and execution errors in the complete scheduled table.
        record([linux(frozen / 'compare_rank_hook.py'), '--snapshot', ASAP], 'environment_after')
    finally:
        write(output / 'preserved_after.json', preserved_artifacts(output))
        write(output / 'implementation_final_sha256.json', {name: file_hash(source) for name, source in sources.items()})
        input_log['after'] = {'raw': hashes(output / 'raw_assemblies'), 'dense': hashes(output / 'dense_assemblies'),
                              'archive': file_hash(output / 'test_assembly.zip')}
        write(output / 'input_hashes.json', input_log)
    passed = summarize(output, selected)
    write(output / 'execution_end.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.monotonic() - started, 'integrity': 'PASS' if passed else 'ERROR'})
    progress('comparison_end', integrity='PASS' if passed else 'ERROR')
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    raise SystemExit(execute(parser.parse_args().run_dir))
