"""Read-only completion audit for the three-assembly experiment (host Python).

Plan failure under the declared budget remains UNKNOWN; integrity is separate.
Only the new completion_audit.json is written, exclusively after all checks.
"""
import argparse
from collections import Counter, defaultdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import sys
import time
import zipfile

import audit_seed_memory_completion as safe
import run_memory_multiseed as prior

PROJECT = safe.PROJECT
IDS = ('00561', '02628', '01289')
VARIANTS = ('native', 'trace', 'id_ascending', 'reverse_native')
COMMIT = '3727e716b57e0b508c4e9ecd9fefa6953e8f72da'
URL = 'https://people.csail.mit.edu/yunsheng/ASAP/dataset_2404/test_assembly.zip'
ASAP = '/opt/asap-research/ASAP'
WSL = [r'C:\Windows\System32\wsl.exe', '-d', 'ASAP-Ubuntu', '-u', 'root', '--cd',
       ASAP, '--', '/opt/asap-research/envs/asap/bin/python', '-B']
WORKER = ('rank_contract.py', 'rank_hook.py', 'compare_rank_hook.py', 'validate_baseline.py',
          'run_assembly_hook.py', 'validate_assembly.py', 'audit_assembly_events.py', 'prepare_assembly_meshes.py')
TOOLS = ('run_assembly_comparison.py', 'prepare_assembly_dataset.py', 'run_memory_multiseed.py',
         'run_asap_baseline.py', 'run_recorded.py')
SETUP = dict(budget=100, max_grippers=2, max_poses=3, pose_reuse=0, early_term=True,
             timeout=180, plan_grasp=False, plan_arm=False, gripper_type='robotiq-140',
             gripper_scale=0.4, optimizer='L-BFGS-B')


def linux(path):
    # No resolve: check the lexical C: boundary before touching any path.
    if not PureWindowsPath(str(path)).is_relative_to(PureWindowsPath(str(PROJECT))):
        raise ValueError('Foreign project path')
    return '/mnt/c/dev/disassembly_codex_ready_v2/' + path.relative_to(PROJECT).as_posix()


def tree_hashes(folder):
    """Reject links before descending; do not materialize a recursive glob first."""
    safe.guarded(folder)
    result, pending = {}, [folder]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('Refusing a data link/reparse point')
                path = Path(entry.path)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    result[path.relative_to(folder).as_posix()] = safe.hashed(path)
                else:
                    raise ValueError('Unexpected non-regular experiment data')
    return result


def json_lines(path):
    safe.guarded(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def window(command):
    duration = command['elapsed_seconds']
    if type(duration) not in (float, int) or not math.isfinite(duration) or duration < 0:
        raise ValueError('Invalid subprocess duration')
    start = safe.stamp(command['started_at'])
    return start, start + timedelta(seconds=duration)


def command_timing(begin, finish, command, validation, uncertainty):
    """Compare observed UTC order strictly, mixed-clock duration with precision.

    run_recorded stores a UTC start and a duration from time.monotonic(), not a
    UTC end. Their sum is an estimate. Difference errors for the two separately
    quantized clocks are bounded here by the sum of their reported resolutions.
    This establishes consistency at that precision, not an exact observed end.
    """
    if not math.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError('Invalid clock resolution allowance')
    c_start, c_end = window(command)
    v_start, v_end = window(validation)
    begin_utc, finish_utc = safe.stamp(begin['utc']), safe.stamp(finish['utc'])
    command_gap = (v_start - c_end).total_seconds()
    validation_gap = (finish_utc - v_end).total_seconds()
    observed_order = begin_utc <= c_start <= v_start <= finish_utc
    consistent = observed_order and command_gap >= -uncertainty and validation_gap >= -uncertainty
    return consistent, dict(observed_utc_order_valid=observed_order,
        run_begin_utc=begin['utc'], command_start_utc=command['started_at'],
        command_monotonic_elapsed_seconds=command['elapsed_seconds'], estimated_command_end_utc=c_end.isoformat(),
        validation_start_utc=validation['started_at'], validation_monotonic_elapsed_seconds=validation['elapsed_seconds'],
        estimated_validation_end_utc=v_end.isoformat(), run_finish_utc=finish['utc'],
        estimated_command_end_to_validation_start_seconds=command_gap,
        estimated_validation_end_to_run_finish_seconds=validation_gap,
        exact_utc_command_end_recorded=False, consistency_allowance_seconds=uncertainty)


def audit(root):
    checks, errors, outcomes = {}, [], {}
    expected_pairs = {(assembly, variant) for assembly in IDS for variant in VARIANTS}

    def section(label, function):
        try:
            function()
        except Exception as error:
            checks[label + ':readable'] = False
            errors.append(dict(scope=label, error_type=type(error).__name__, message=str(error)))

    def preservation():
        safe.guarded(PROJECT / 'runs')
        before, after = safe.read(root / 'preserved_before.json'), safe.read(root / 'preserved_after.json')
        checks['all_prior_artifacts_preserved_now'] = bool(before) and before == after == prior.preserved_artifacts(root)
        outcomes['preserved_files'] = len(before)

    def implementation():
        declared = safe.read(root / 'implementation_sha256.json')
        checks['exact_thirteen_implementation_files'] = set(declared) == set(WORKER + TOOLS) and len(declared) == 13
        checks['implementation_final_manifest'] = declared == safe.read(root / 'implementation_final_sha256.json')
        for name in WORKER + TOOLS:
            choices = [PROJECT / folder / name for folder in ('worker', 'tools')]
            found = [path for path in choices if safe.guarded(path, missing=True)]
            if len(found) != 1:
                raise ValueError('Missing or colliding implementation basename: ' + name)
            checks['implementation:' + name] = safe.hashed(found[0]) == safe.hashed(root / 'implementation' / name) == declared[name]

    def selection_and_data():
        policy, amendment = safe.read(root / 'selection_policy.json'), safe.read(root / 'selection_amendment.json')
        inventory, selected = safe.read(root / 'dataset_inventory.json'), safe.read(root / 'selected_assemblies.json')
        start = safe.read(root / 'execution_start.json')
        checks['selection_policy_fixed_before_execution'] = (
            policy.get('url') == URL and policy.get('part_counts') == [3, 5, 7]
            and policy.get('prior_results_used') is False and policy.get('memory_enabled') is False
            and policy.get('seed') == 0
            and policy.get('selection') == 'lexicographically smallest official assembly ID at each exact part count'
            and amendment.get('original_part_counts') == [3, 5, 7] and amendment.get('part_counts') == [5, 7, 9]
            and safe.stamp(policy['created_utc']) <= safe.stamp(amendment['created_utc']) < safe.stamp(start['utc']))
        checks['inventory_has_240_assemblies_and_no_three_part_case'] = len(inventory) == 240 and all(item['parts'] >= 5 for item in inventory)
        chosen = [min((item for item in inventory if item['parts'] == count), key=lambda item: item['id']) for count in (5, 7, 9)]
        checks['selected_by_declared_rule'] = (tuple(item['id'] for item in selected) == IDS and len(selected) == 3
            and all(all(item.get(key) == original[key] for key in ('id', 'parts', 'archive_dir', 'obj_bytes'))
                    for item, original in zip(selected, chosen)))
        raw, dense = tree_hashes(root / 'raw_assemblies'), tree_hashes(root / 'dense_assemblies')
        archive = root / 'test_assembly.zip'
        archive_sha = safe.hashed(archive)
        download = safe.read(root / 'download_manifest.json')
        checks['archive_matches_download'] = download.get('source_url') == URL and download['sha256'] == archive_sha and download['bytes'] == archive.stat().st_size
        grouped = defaultdict(list)
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
                    raise ValueError('Unsafe archive member')
                if path.suffix == '.obj':
                    grouped[str(path.parent)].append(info)
            actual_inventory = [dict(archive_dir=name, id=PurePosixPath(name).name, parts=len(files),
                                     obj_bytes=sum(item.file_size for item in files)) for name, files in sorted(grouped.items())]
            checks['inventory_matches_archive'] = inventory == actual_inventory
            declared_raw = {}
            for item in selected:
                entries = {PurePosixPath(info.filename).name: info for info in bundle.infolist()
                           if str(PurePosixPath(info.filename).parent) == item['archive_dir']
                           and PurePosixPath(info.filename).suffix in ('.obj', '.json')}
                checks[item['id'] + ':extracted_members'] = set(entries) == set(item['files'])
                checks[item['id'] + ':part_ids'] = item['part_ids'] == sorted(name[:-4] for name in entries if name.endswith('.obj')) and len(item['part_ids']) == item['parts']
                for name, info in entries.items():
                    digest = hashlib.sha256(bundle.read(info)).hexdigest()
                    declared_raw[item['id'] + '/' + name] = digest
                    checks[item['id'] + '/' + name + ':raw_archive_identity'] = item['files'][name] == dict(sha256=digest, bytes=info.file_size)
        checks['raw_copies_equal_selected_archive_bytes'] = raw == declared_raw
        prepared = safe.read(root / 'mesh_preparation.json')
        checks['official_mesh_preparation_complete'] = (prepared.get('source_commit') == COMMIT and prepared.get('max_edge') == 0.5
            and prepared.get('rescale') is None and prepared.get('method') == 'official assets.subdivide.subdivide_assembly'
            and prepared.get('dense_sha256') == dense and tuple(item['id'] for item in prepared['results']) == IDS
            and all(item.get('new_verifier_calls') == 0 and item['checks']
                    and all(value is True for value in item['checks'].values()) for item in prepared['results']))
        inputs = safe.read(root / 'input_hashes.json')
        checks['all_data_checkpoints_present'] = set(inputs) == {'before', 'after'} | {a + '/' + v for a, v in expected_pairs}
        for key, snapshot in inputs.items():
            checks['raw_dense_checkpoint:' + key] = snapshot.get('raw') == raw and snapshot.get('dense') == dense
        checks['zip_first_last_current_identical'] = inputs['before']['archive'] == inputs['after']['archive'] == archive_sha
        checks['no_saved_sdf_in_inputs'] = not any(name.lower().endswith('.sdf') for name in list(raw) + list(dense))
        outcomes['zip_hash_measurement_scope'] = 'Actual first/last/current hashes; per-run archive fields were copied by the runner.'

    def completion_and_runs():
        result, protocol = safe.read(root / 'comparison.json'), safe.read(root / 'protocol.json')
        start, end = safe.read(root / 'execution_start.json'), safe.read(root / 'execution_end.json')
        progress = json_lines(root / 'progress.jsonl')
        clocks = {name: vars(time.get_clock_info(name)) for name in ('monotonic', 'time')}
        uncertainty = clocks['monotonic']['resolution'] + clocks['time']['resolution']
        checks['clock_diagnostics_use_same_recorder_interpreter'] = Path(start['executable']) == Path(sys.executable)
        outcomes['command_timing'] = dict(clocks_inspected_at_audit=clocks,
            recorder_executable=start['executable'], audit_executable=sys.executable,
            duration_comparison_allowance_seconds=uncertainty, runs={},
            limitation='UTC command end was not recorded. Observed UTC starts/progress order is strict; estimated ends are checked at reported clock precision.')
        checks['comparison_all_integrity_checks_pass'] = (result.get('status') == 'PASS' and isinstance(result.get('checks'), dict)
            and bool(result['checks']) and all(value is True for value in result['checks'].values()) and result.get('errors', []) == [])
        checks['comparison_declares_no_memory_or_llm'] = result.get('memory_enabled') is False and result.get('llm_calls') == 0 and result.get('seed') == 0
        checks['execution_finished_with_integrity_pass'] = end.get('integrity') == 'PASS' and progress[-1].get('kind') == 'comparison_end' and progress[-1].get('integrity') == 'PASS'
        rows = result['rows']
        checks['all_twelve_unique_rows_present'] = len(rows) == 12 and {(row['assembly_id'], row['variant']) for row in rows} == expected_pairs
        order = [(item['assembly'], item['variant']) for item in protocol['execution_order']]
        checks['all_twelve_planned_runs_once'] = len(order) == 12 and set(order) == expected_pairs
        checks['protocol_fixed_settings'] = all(protocol.get(key) == value for key, value in dict(seed=0, base_part=None, budget=100,
            max_gripper=2, max_pose=3, pose_reuse=0, timeout_seconds=180, num_proc=1, early_term=True,
            memory_enabled=False, llm_calls=0, planner='dfs', generator='heur-out').items())
        observed = [(item['kind'], item.get('assembly'), item.get('variant')) for item in progress]
        expected = [(kind, assembly, variant) for assembly, variant in order for kind in ('run_start', 'run_end')] + [('comparison_end', None, None)]
        checks['actual_order_matches_protocol'] = observed == expected
        stamps = [safe.stamp(item['utc']) for item in progress]
        checks['execution_timeline_monotonic'] = (all(a <= b for a, b in zip(stamps, stamps[1:]))
            and safe.stamp(start['utc']) <= stamps[0] and stamps[-2] <= safe.stamp(end['utc']) <= stamps[-1])
        row_map = {(row['assembly_id'], row['variant']): row for row in rows}
        event_ids, run_ids, request_ids = [], [], []
        for index, (assembly, variant) in enumerate(order):
            label, folder = assembly + '/' + variant, root / 'evaluations' / assembly / variant
            mode, ranker = (variant, 'noop') if variant in ('native', 'trace') else ('rank', variant)
            native = folder / 'native/dfs-heur-out/s0' / assembly
            events, parsed = json_lines(folder / 'events.jsonl'), safe.read(folder / 'parsed_result.json')
            stats, setup = safe.read(native / 'stats.json'), safe.read(native / 'setup.json')
            row, first = row_map[assembly, variant], events[0]
            command, validation = safe.read(folder / 'execute/command.json'), safe.read(folder / 'validation/command.json')
            expected_command = WSL + ['-u', linux(root / 'implementation/run_assembly_hook.py'), '--asap-root', ASAP,
                '--assembly-dir', linux(root / 'dense_assemblies'), '--assembly-id', assembly, '--native-log-dir', linux(folder / 'native'),
                '--trace-file', linux(folder / 'events.jsonl'), '--run-id', root.name + '/' + label, '--mode', mode, '--ranker', ranker,
                '--seed', '0', '--max-gripper', '2', '--budget', '100', '--max-pose', '3', '--pose-reuse', '0', '--timeout', '180']
            expected_validation = WSL + [linux(root / 'implementation/validate_assembly.py'), '--log-dir', linux(native),
                '--assembly-dir', linux(root / 'dense_assemblies' / assembly), '--max-gripper', '2', '--output', linux(folder / 'parsed_result.json'), '--trusted-local-tree']
            checks[label + ':own_exact_commands'] = (command['command'] == expected_command and validation['command'] == expected_validation
                and command['cwd'] == validation['cwd'] == str(PROJECT) and command['exit_code'] == validation['exit_code'] == 0
                and command.get('launch_error') is None and validation.get('launch_error') is None)
            for phase in ('execute', 'validation'):
                for stream in ('stdout.log', 'stderr.log'):
                    safe.guarded(folder / phase / stream)
            begin, finish = progress[2 * index:2 * index + 2]
            consistent, timing = command_timing(begin, finish, command, validation, uncertainty)
            outcomes['command_timing']['runs'][label] = timing
            checks[label + ':commands_ordered_and_elapsed_consistent'] = (consistent
                and begin.get('index') == finish.get('index') == index + 1 and begin.get('total') == finish.get('total') == 12
                and finish.get('process_exit') == command['exit_code'] and finish.get('validation_exit') == validation['exit_code'])
            native_argv = [ASAP + '/plan_sequence/run_seq_plan.py', '--dir', linux(root / 'dense_assemblies'), '--id', assembly,
                '--planner', 'dfs', '--generator', 'heur-out', '--num-proc', '1', '--seed', '0', '--budget', '100',
                '--max-gripper', '2', '--max-pose', '3', '--pose-reuse', '0', '--timeout', '180', '--log-dir', linux(folder / 'native'),
                '--early-term', '--disable-save-sdf']
            checks[label + ':native_settings_and_identity'] = (first.get('native_argv') == native_argv and setup == SETUP
                and first.get('run_id') == root.name + '/' + label and first.get('assembly_id') == assembly and first.get('seed') == 0
                and first.get('mode') == mode and first.get('ranker_name') == (ranker if mode == 'rank' else None)
                and first.get('source_commit') == COMMIT and first.get('tracked_source_clean') is True
                and first.get('base_part') is None and first.get('hooks_installed') is (mode != 'native') and first.get('sdf_writes_enabled') is False)
            checks[label + ':no_memory_or_llm'] = first.get('live_llm_calls') == 0 and all(not any(key == 'memory' or key.startswith('memory_') for key in event) for event in events)
            checks[label + ':parsed_and_reported_outcomes_match'] = (parsed.get('validation_ok') is True and parsed.get('raw_stats') == stats
                and parsed.get('log_dir') == linux(native) and parsed.get('assembly_dir') == linux(root / 'dense_assemblies' / assembly)
                and parsed.get('trusted_local_tree') is True and parsed.get('physics_replayed') is False
                and parsed.get('plan_success') is stats['success'] and parsed.get('status') == ('PASS' if stats['success'] else 'UNKNOWN')
                and row.get('plan_success') is parsed['plan_success'] and row.get('status') == parsed['status']
                and row.get('parts') == {name: count for name, count in zip(IDS, (5, 7, 9))}[assembly] == len(parsed.get('part_ids', []))
                and row.get('stop_msg') == stats.get('stop_msg') and row.get('sequence') == stats.get('sequence')
                and row.get('terminal_parts') == parsed.get('terminal_parts') and row.get('tree_sha256') == parsed.get('tree_fingerprint_sha256')
                and row.get('native_n_eval') == stats.get('n_eval') and row.get('native_total_n_eval') == stats.get('total_n_eval')
                and row.get('native_seconds') == stats.get('time') and row.get('process_seconds') == command['elapsed_seconds'])
            audited = safe.read(folder / 'event_audit.json')
            checks[label + ':event_audit_pass'] = audited.get('status') == 'PASS' and bool(audited.get('checks')) and all(value is True for value in audited['checks'].values())
            calls = sum(event['kind'] == 'verifier_start' for event in events)
            counts = Counter(event['status'] for event in events if event['kind'] == 'verifier_result')
            checks[label + ':verifier_counts_match_trace'] = (row.get('verifier_calls') is None and row.get('verifier_status_counts') is None and len(events) == 2
                if mode == 'native' else row.get('verifier_calls') == calls == sum(counts.values())
                and row.get('verifier_status_counts') == {key: counts[key] for key in ('PASS', 'UNKNOWN', 'ERROR')})
            event_ids.extend(event['event_id'] for event in events)
            run_ids.append(first['run_id'])
            request_ids.extend(event['request']['request_id'] for event in events if event['kind'] == 'rank_decision')
        checks['global_event_run_request_ids_unique'] = (len(event_ids) == len(set(event_ids)) and len(run_ids) == len(set(run_ids)) == 12 and len(request_ids) == len(set(request_ids)))
        outcomes.update(plan_success_count=sum(row['plan_success'] is True for row in rows),
                        plan_unsuccessful_count=sum(row['plan_success'] is False for row in rows),
                        unknown_outcome_count=sum(row['plan_success'] is None for row in rows), run_count=len(rows))

    def environment():
        before, after = safe.read(root / 'environment_before/stdout.log'), safe.read(root / 'environment_after/stdout.log')
        checks['source_binary_and_existing_beam_unchanged'] = before == after and before.get('commit') == COMMIT and before.get('tracked_source_clean') is True
        for phase in ('environment_before', 'environment_after'):
            command = safe.read(root / phase / 'command.json')
            checks[phase + ':successful_own_snapshot'] = command.get('exit_code') == 0 and command.get('launch_error') is None and command.get('command') == WSL + [linux(root / 'implementation/compare_rank_hook.py'), '--snapshot', ASAP]

    for label, function in (('preservation', preservation), ('implementation', implementation),
                            ('selection_and_data', selection_and_data), ('completion_and_runs', completion_and_runs), ('environment', environment)):
        section(label, function)
    errors.extend(dict(scope=name, error_type='CheckFailed', message='Final integrity check is false')
                  for name, value in checks.items() if not value and not name.endswith(':readable'))
    return dict(audit_version=2, status='PASS' if checks and all(checks.values()) else 'ERROR', checks=checks, errors=errors, outcomes=outcomes,
                notes='Read-only completion audit; ordinary unsuccessful plans remain in the denominator; no physics replay.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='New completion_audit*.json filename or absolute path within this run')
    args = parser.parse_args()
    root = args.run_dir
    try:
        safe.guarded(root)
        if root.parent != PROJECT / 'runs' or not root.is_dir():
            raise ValueError('Run must be an existing direct child of the allowed C: project/runs')
        target = args.output or root / 'completion_audit.json'
        if not target.is_absolute():
            if target.parent != Path('.'):
                raise ValueError('Output must be a bare filename or an absolute path within this run')
            target = root / target
        if target.parent != root or not target.name.startswith('completion_audit') or target.suffix != '.json':
            raise ValueError('Output must be a new completion_audit*.json in this run')
        if safe.guarded(target, missing=True):
            raise FileExistsError('Completion audit already exists')
        result = audit(root)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        summary = {key: value for key, value in result['outcomes'].items() if key != 'command_timing'}
        print(json.dumps(dict(status=result['status'], check_count=len(result['checks']), errors=result['errors'], outcomes=summary)))
        return 0 if result['status'] == 'PASS' else 1
    except Exception as error:
        print(json.dumps(dict(status='ERROR', errors=[type(error).__name__ + ': ' + str(error)])))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
