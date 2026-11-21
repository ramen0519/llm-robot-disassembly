"""Collect development experience at each seed, freeze all, then evaluate pairs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import run_asap_baseline as baseline
import run_memory_multiseed as prior

ROOT = baseline.ROOT
SEEDS = prior.SEEDS
VARIANTS = prior.VARIANTS
DEVELOPMENT = ('baseline', 'id_ascending', 'reverse_native')
RANKERS = {'baseline': 'noop', 'id_ascending': 'id_ascending', 'reverse_native': 'reverse_native'}
MEMORY_FILES = prior.SOURCE_FILES
WORKER_FILES = prior.WORKER_FILES + ('build_seed_memories.py', 'compare_seed_memories.py')
TOOL_FILES = prior.TOOL_FILES + ('run_seed_memory_comparison.py',)
write, file_hash = prior.write, prior.file_hash


def all_memory_hashes(output):
    return {'s' + str(seed): {key: file_hash(output / 'memories' / ('s' + str(seed)) / name)
                             for key, name in MEMORY_FILES.items()} for seed in SEEDS}


def execute(config, output):
    output = baseline.run_path(output)
    baseline.beam_command(config, output)
    implementation = [ROOT / 'worker' / name for name in WORKER_FILES]
    implementation += [ROOT / 'tools' / name for name in TOOL_FILES]
    missing = [str(path) for path in implementation if not path.is_file()]
    if missing:
        raise FileNotFoundError('Implementation files not ready: ' + ', '.join(missing))
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'orchestration_start.json', {
        'argv': sys.argv, 'executable': sys.executable, 'cwd': str(ROOT),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'recording': 'exclusive per-subprocess command/stdout/stderr and append-only progress',
    })
    development_order = [{'seed': seed, 'variant': name} for seed in SEEDS for name in DEVELOPMENT]
    evaluation_order = prior.execution_order()
    write(output / 'config.json', config)
    write(output / 'protocol.json', {
        'schema_version': 1, 'seeds': SEEDS, 'variants': VARIANTS, 'pairs': prior.PAIRS,
        'development_variants': DEVELOPMENT, 'development_order': development_order,
        'execution_order': evaluation_order, 'policy': 'positive_only_stable_promotion',
        'allow_hard_pruning': False, 'failure_usage': 'retain every observed attempt; no negative scoring or pruning',
        'source_policy': 'fresh development runs only; no previous evaluation or current evaluation records',
        'memory_scope': 'one independently frozen snapshot per actual seed, identical full-context matching',
        'freeze_policy': 'all fifteen development runs finish and all five memories freeze before any evaluation',
        'same_pair_conditions': 'input/cache/seed/solver/base-order/budget/goal; only memory mode changes',
        'evaluation_order_policy': 'serial; alternate direction by seed parity, as prior multiseed experiment',
        'goal': 'official full plan terminating at base6', 'development_physics_cost': 'measured separately',
        'scope': 'known beam and seed revisit; independent episodes, not held-out geometry/generalization',
        'evaluation_archive': 'separate archive after all twenty runs; never fed back during evaluation',
        'live_llm_calls': 0, 'exact_verdict_cache': False, 'physics_bypass': False,
        'internal_geometry_operations': None, 'skipped_edges': None,
    })
    write(output / 'preserved_before.json', prior.preserved_artifacts(output))
    frozen = output / 'implementation'
    frozen.mkdir()
    for source in implementation:
        shutil.copyfile(source, frozen / source.name)
    write(output / 'implementation_sha256.json', {path.name: file_hash(path) for path in frozen.iterdir()})
    prefix = [r'C:\Windows\System32\wsl.exe'] + baseline.wsl_prefix(config)[1:]
    worker = baseline.worker_path(frozen, config)
    hashes = {}

    def record(command, name):
        return baseline.recorder.run_recorded(command, ROOT, output / name)

    def progress(kind, **fields):
        value = {'kind': kind, 'utc': datetime.now(timezone.utc).isoformat(), **fields}
        with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
        print(json.dumps(value, sort_keys=True), flush=True)

    def snapshot(name):
        value = record(prefix + [config['python'], worker + '/compare_rank_hook.py', '--snapshot',
                                 config['asap_root']], name)
        if value['exit_code'] != 0:
            raise RuntimeError('input snapshot failed: ' + name)

    def run_one(item, index, development):
        seed, variant = item['seed'], item['variant']
        phase = 'development_' if development else 'run_'
        total = 15 if development else 20
        relative = ('development/' if development else '') + 's{}/{}'.format(seed, variant)
        snapshot_prefix = 'development/' if development else ''
        key = 's{}_{}'.format(seed, variant)
        progress(phase + 'start', index=index, total=total, seed=seed, variant=variant)
        before = snapshot_prefix + 'inputs_before_' + key
        after = snapshot_prefix + 'inputs_after_' + key
        snapshot(before)
        run = output / relative
        run.mkdir(parents=True)
        ranker = RANKERS[variant] if development else ('noop' if variant.startswith('heur_') else 'reverse_native')
        mode = None if development else variant.rsplit('_', 1)[1]
        write(run / 'config.json', {'seed': seed, 'ranker': ranker, 'memory_mode': mode,
                                   'phase': 'development' if development else 'evaluation', 'baseline': config})
        command = prefix + [config['python'], '-u', worker + '/run_rank_hook.py',
            '--asap-root', config['asap_root'], '--seed', str(seed),
            '--native-log-dir', baseline.worker_path(run / 'native', config),
            '--trace-file', baseline.worker_path(run / 'events.jsonl', config),
            '--run-id', output.relative_to(ROOT).as_posix() + '/' + relative,
            '--mode', 'rank', '--ranker', ranker]
        if not development:
            memory = output / 'memories' / ('s' + str(seed)) / 'memory_snapshot.json'
            command += ['--memory-mode', mode, '--memory-snapshot', baseline.worker_path(memory, config)]
        executed = record(command, relative + '/execute')
        native = run / 'native/dfs-heur-out' / ('s' + str(seed)) / 'original'
        validated = record(prefix + [config['python'], worker + '/validate_baseline.py',
            '--log-dir', baseline.worker_path(native, config),
            '--output', baseline.worker_path(run / 'parsed_result.json', config), '--trusted-local-tree'],
            relative + '/validation')
        snapshot(after)
        if (output / before / 'stdout.log').read_bytes() != (output / after / 'stdout.log').read_bytes():
            raise ValueError('input/cache/source changed: ' + relative)
        if not development:
            hashes['after_' + key] = all_memory_hashes(output)
            if hashes['after_' + key] != hashes['before']:
                raise ValueError('seed-specific memories changed during evaluation')
        progress(phase + 'end', index=index, total=total, seed=seed, variant=variant,
                 process_exit_code=executed['exit_code'], validation_exit_code=validated['exit_code'])
        if executed['exit_code'] != 0:
            raise RuntimeError('actual worker failed: ' + relative)

    try:
        for index, item in enumerate(development_order, 1):
            run_one(item, index, True)
        built = record(prefix + [config['python'], worker + '/build_seed_memories.py',
            '--run-dir', baseline.worker_path(output, config)], 'source_build')
        if built['exit_code'] != 0:
            return 1
        hashes['before'] = all_memory_hashes(output)
        freeze = {'utc': datetime.now(timezone.utc).isoformat(), 'hashes': hashes['before'],
                  'index_sha256': file_hash(output / 'memories/index.json')}
        write(output / 'memories_frozen.json', freeze)
        progress('memory_freeze', index_sha256=freeze['index_sha256'])
        for index, item in enumerate(evaluation_order, 1):
            run_one(item, index, False)
        archived = record(prefix + [config['python'], worker + '/compare_seed_memories.py',
            '--run-dir', baseline.worker_path(output, config), '--archive-only', '--trusted-local-tree'],
            'evaluation_archive')
        hashes['after_evaluation_archive'] = all_memory_hashes(output)
        if archived['exit_code'] != 0:
            return 1
    finally:
        write(output / 'memory_hashes.json', hashes)
        write(output / 'preserved_after.json', prior.preserved_artifacts(output))
        write(output / 'implementation_final_sha256.json', {path.name: file_hash(path) for path in implementation})
    compared = record(prefix + [config['python'], worker + '/compare_seed_memories.py',
        '--run-dir', baseline.worker_path(output, config), '--trusted-local-tree'], 'comparison')
    progress('comparison_end', process_exit_code=compared['exit_code'])
    return 0 if compared['exit_code'] == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/beam_baseline.wsl.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    output = baseline.run_path(args.run_dir)
    existed = output.exists()
    started = time.perf_counter()
    code, error_type = 1, None
    try:
        code = execute(json.loads(args.config.read_text(encoding='utf-8')), output)
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        if not existed and output.is_dir():
            write(output / 'orchestration_end.json', {'ended_utc': datetime.now(timezone.utc).isoformat(),
                'exit_code': code, 'elapsed_seconds': time.perf_counter() - started, 'error_type': error_type})
    raise SystemExit(code)
