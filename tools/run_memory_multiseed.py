"""Run frozen-memory off/on pairs at five fixed seeds without learning on evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time

import run_asap_baseline as baseline

ROOT = baseline.ROOT
SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ('heur_off', 'heur_on', 'reverse_off', 'reverse_on')
PAIRS = (('heur_off', 'heur_on'), ('reverse_off', 'reverse_on'))
SOURCE_RUN = ROOT / 'runs/m4_memory_20261017'
SOURCE_FILES = {'snapshot': 'memory_snapshot.json', 'source_db': 'source_memory.sqlite',
                'manifest': 'memory_manifest.json'}
WORKER_FILES = ('rank_contract.py', 'rank_hook.py', 'run_rank_hook.py', 'compare_rank_hook.py',
                'compare_rule_rankers.py', 'compare_memory_runs.py', 'compare_memory_multiseed.py',
                'validate_baseline.py', 'evidence_memory.py', 'memory_ranker.py', 'build_evidence.py')
TOOL_FILES = ('run_memory_multiseed.py', 'run_memory_comparison.py', 'run_m3_comparison.py',
              'run_asap_baseline.py', 'run_recorded.py')


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)


def file_hash(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def preserved_artifacts(exclude):
    pending, files = [ROOT / 'runs'], {}
    while pending:
        with os.scandir(str(pending.pop())) as entries:
            for entry in entries:
                path = Path(entry.path)
                if path == exclude:
                    continue
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('Refusing to traverse result link/reparse point: ' + str(path))
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    files[path.relative_to(ROOT).as_posix()] = {'sha256': file_hash(path), 'bytes': info.st_size}
    return files


def memory_hashes(output):
    return {key: file_hash(output / name) for key, name in SOURCE_FILES.items()}


def execution_order():
    return [{'seed': seed, 'variant': variant} for seed in SEEDS
            for variant in (VARIANTS if seed % 2 == 0 else tuple(reversed(VARIANTS)))]


def execute(config, output):
    output = baseline.run_path(output)
    baseline.beam_command(config, output)  # Validate extension-disabled official contract.
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'orchestration_start.json', {
        'argv': sys.argv, 'executable': sys.executable, 'cwd': str(ROOT),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'recording': 'individual subprocess command/stdout/stderr/exit logs plus progress.jsonl',
    })
    order = execution_order()
    write(output / 'config.json', config)
    write(output / 'protocol.json', {
        'schema_version': 1, 'seeds': SEEDS, 'variants': VARIANTS, 'pairs': PAIRS,
        'execution_order': order, 'source_run': 'runs/m4_memory_20261017',
        'policy': 'positive_only_stable_promotion', 'allow_hard_pruning': False,
        'failure_usage': 'UNKNOWN/ERROR retained; no negative score or candidate deletion',
        'source_seed_values': [0], 'source_frozen_before_evaluation': True,
        'lookup_scope': 'unchanged full context equality INCLUDING seed; no cross-seed relaxation',
        'scope': 'one known beam; seed0 revisit and seeds1-4 source-context mismatch reported separately',
        'same_pair_conditions': 'seed/input/cache/solver/native settings/base ordering/budget/goal',
        'goal': 'official full plan terminating with base part6',
        'evaluation_evidence': 'separate archive only after all20 runs; never used during evaluation',
        'preparation_new_verifier_calls': 0, 'historical_source_verifier_calls': 15,
        'run_order': 'serial, alternating pair order and base-order block order by seed parity',
        'live_llm_calls': 0, 'generalization_claim': None, 'exact_verdict_cache': False,
        'skipped_edges': None, 'internal_geometry_operations': None,
    })
    write(output / 'preserved_before.json', preserved_artifacts(output))
    frozen = output / 'implementation'
    frozen.mkdir()
    implementation = [ROOT / 'worker' / name for name in WORKER_FILES]
    implementation += [ROOT / 'tools' / name for name in TOOL_FILES]
    for source in implementation:
        shutil.copyfile(source, frozen / source.name)
    write(output / 'implementation_sha256.json', {path.name: file_hash(path) for path in frozen.iterdir()})
    started = time.perf_counter()
    for name in SOURCE_FILES.values():
        shutil.copyfile(SOURCE_RUN / name, output / name)
    original_hashes = memory_hashes(SOURCE_RUN)
    if memory_hashes(output) != original_hashes:
        raise ValueError('Source memory copy differs from existing frozen artifact')
    write(output / 'source_copy_manifest.json', {
        'source_run': 'runs/m4_memory_20261017', 'hashes': original_hashes,
        'copy_seconds': time.perf_counter() - started, 'new_verifier_calls': 0,
    })
    prefix = [r'C:\Windows\System32\wsl.exe'] + baseline.wsl_prefix(config)[1:]
    worker = baseline.worker_path(frozen, config)
    hashes = {'before': memory_hashes(output)}

    def record(command, name):
        return baseline.recorder.run_recorded(command, ROOT, output / name)

    def snapshot(name):
        return record(prefix + [config['python'], worker + '/compare_rank_hook.py',
                                '--snapshot', config['asap_root']], name)

    def progress(kind, **fields):
        value = {'kind': kind, 'utc': datetime.now(timezone.utc).isoformat(), **fields}
        with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
        print(json.dumps(value, sort_keys=True), flush=True)

    try:
        for index, item in enumerate(order, 1):
            seed, variant = item['seed'], item['variant']
            key = 's{}_{}'.format(seed, variant)
            progress('run_start', index=index, total=len(order), seed=seed, variant=variant)
            if snapshot('inputs_before_' + key)['exit_code'] != 0:
                return 1
            run = output / ('s' + str(seed)) / variant
            run.mkdir(parents=True)
            ranker = 'noop' if variant.startswith('heur_') else 'reverse_native'
            mode = variant.rsplit('_', 1)[1]
            write(run / 'config.json', {'seed': seed, 'ranker': ranker, 'memory_mode': mode,
                                        'baseline': config})
            command = prefix + [config['python'], '-u', worker + '/run_rank_hook.py',
                '--asap-root', config['asap_root'], '--seed', str(seed),
                '--native-log-dir', baseline.worker_path(run / 'native', config),
                '--trace-file', baseline.worker_path(run / 'events.jsonl', config),
                '--run-id', output.relative_to(ROOT).as_posix() + '/s{}/{}'.format(seed, variant),
                '--mode', 'rank', '--ranker', ranker, '--memory-mode', mode,
                '--memory-snapshot', baseline.worker_path(output / 'memory_snapshot.json', config)]
            executed = record(command, 's{}/{}/execute'.format(seed, variant))
            native = run / 'native/dfs-heur-out' / ('s' + str(seed)) / 'original'
            validated = record(prefix + [config['python'], worker + '/validate_baseline.py',
                '--log-dir', baseline.worker_path(native, config),
                '--output', baseline.worker_path(run / 'parsed_result.json', config),
                '--trusted-local-tree'], 's{}/{}/validation'.format(seed, variant))
            if snapshot('inputs_after_' + key)['exit_code'] != 0:
                return 1
            hashes['after_' + key] = memory_hashes(output)
            if hashes['after_' + key] != hashes['before']:
                raise ValueError('Frozen memory changed during evaluation')
            if (output / ('inputs_before_' + key) / 'stdout.log').read_bytes() != (
                    output / ('inputs_after_' + key) / 'stdout.log').read_bytes():
                raise ValueError('Input/cache/source changed during run')
            progress('run_end', index=index, total=len(order), seed=seed, variant=variant,
                     process_exit_code=executed['exit_code'], validation_exit_code=validated['exit_code'])
        archive = record(prefix + [config['python'], worker + '/compare_memory_multiseed.py',
            '--run-dir', baseline.worker_path(output, config), '--archive-only', '--trusted-local-tree'],
            'evaluation_archive')
        hashes['after_evaluation_archive'] = memory_hashes(output)
        if archive['exit_code'] != 0:
            return 1
    finally:
        write(output / 'preserved_after.json', preserved_artifacts(output))
        write(output / 'memory_hashes.json', hashes)
        write(output / 'implementation_final_sha256.json', {path.name: file_hash(path) for path in implementation})
    result = record(prefix + [config['python'], worker + '/compare_memory_multiseed.py',
        '--run-dir', baseline.worker_path(output, config), '--trusted-local-tree'], 'comparison')
    progress('comparison_end', process_exit_code=result['exit_code'])
    return 0 if result['exit_code'] == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/beam_baseline.wsl.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    code = 1
    error_type = None
    output = baseline.run_path(args.run_dir)
    existed = output.exists()
    try:
        code = execute(json.loads(args.config.read_text(encoding='utf-8')), args.run_dir)
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        if not existed and output.is_dir():
            write(output / 'orchestration_end.json', {
                'ended_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': code,
                'elapsed_seconds': time.perf_counter() - started, 'error_type': error_type,
            })
    raise SystemExit(code)
