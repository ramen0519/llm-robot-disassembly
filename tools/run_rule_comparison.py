"""Run fixed rule controls against the native beam order without changing ASAP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import run_asap_baseline as baseline
from run_m3_comparison import write_json

ROOT = baseline.ROOT
VARIANTS = ('baseline', 'id_ascending', 'reverse_native')


def preserved_artifacts() -> dict[str, str]:
    hashes = {}
    for name in ('m1_wsl_20261016', 'm2_wsl_20261016', 'm3_wsl_20261016'):
        folder = ROOT / 'runs' / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def execute(config: dict, output: Path) -> int:
    # Validate the official configuration/command contract before creating a run.
    output = baseline.run_path(output)
    baseline.beam_command(config, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    write_json(output / 'protocol.json', {
        'schema_version': 1, 'variants': VARIANTS,
        'rules_fixed_before_execution': {
            'baseline': 'native lazy heur-out, observation only',
            'id_ascending': 'lexicographic candidate ID order, not numeric conversion',
            'reverse_native': 'reverse each request native heur-out order'},
        'scope': 'official beam original, one seed0, fixed budget400, same warm cache',
        'goal': 'native full plan ending with base part 6',
        'verifier_count_scope': 'SequencePlanner._simulate candidate/pose calls including leaf',
        'internal_geometry_operations': None, 'skipped_edge_count': None,
        'live_llm_calls': 0, 'memory_enabled': False,
        'performance_generalization_claim': None,
        'preexisting_baseline_runs': ['m1_wsl_20261016', 'm2_wsl_20261016', 'm3_wsl_20261016'],
    })
    write_json(output / 'preserved_before.json', preserved_artifacts())
    frozen = output / 'implementation'
    frozen.mkdir()
    source_files = [ROOT / 'worker' / name for name in (
        'rank_contract.py', 'rank_hook.py', 'run_rank_hook.py', 'compare_rank_hook.py',
        'compare_rule_rankers.py', 'validate_baseline.py')]
    source_files += [ROOT / 'tools' / name for name in (
        'run_rule_comparison.py', 'run_m3_comparison.py', 'run_asap_baseline.py', 'run_recorded.py')]
    for source in source_files:
        shutil.copyfile(source, frozen / source.name)
    write_json(output / 'implementation_sha256.json', {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen.iterdir()})
    prefix = baseline.wsl_prefix(config)
    worker = baseline.worker_path(frozen, config)

    def record(command: list[str], name: str) -> dict:
        return baseline.recorder.run_recorded(command, ROOT, output / name)

    def snapshot(name: str) -> dict:
        return record(prefix + [config['python'], worker + '/compare_rank_hook.py',
                                '--snapshot', config['asap_root']], name)

    try:
        for variant in VARIANTS:
            print('{}: snapshot and actual physics start'.format(variant), flush=True)
            before = snapshot('inputs_before_' + variant)
            if before['exit_code'] != 0:
                return 1
            run = output / variant
            run.mkdir()
            mode = 'trace' if variant == 'baseline' else 'rank'
            ranker = 'noop' if variant == 'baseline' else variant
            write_json(run / 'config.json', {'mode': mode, 'ranker': ranker, 'baseline': config})
            command = prefix + [config['python'], '-u', worker + '/run_rank_hook.py',
                                '--asap-root', config['asap_root'],
                                '--native-log-dir', baseline.worker_path(run / 'native', config),
                                '--trace-file', baseline.worker_path(run / 'events.jsonl', config),
                                '--run-id', output.relative_to(ROOT).as_posix() + '/' + variant,
                                '--mode', mode, '--ranker', ranker]
            record(command, variant + '/execute')
            validator = prefix + [config['python'], worker + '/validate_baseline.py',
                                  '--log-dir', baseline.worker_path(run / 'native/dfs-heur-out/s0/original', config),
                                  '--output', baseline.worker_path(run / 'parsed_result.json', config),
                                  '--trusted-local-tree']
            record(validator, variant + '/validation')
            # UNKNOWN/no-plan is an experimental result, not permission to discard a row.
            after = snapshot('inputs_after_' + variant)
            if after['exit_code'] != 0:
                return 1
            if (output / ('inputs_before_' + variant) / 'stdout.log').read_bytes() != (
                    output / ('inputs_after_' + variant) / 'stdout.log').read_bytes():
                print('Input/cache changed; stopping before another treatment', flush=True)
                return 1
            print('{}: execution and artifact check recorded'.format(variant), flush=True)
    finally:
        write_json(output / 'preserved_after.json', preserved_artifacts())
    command = prefix + [config['python'], worker + '/compare_rule_rankers.py',
                        '--run-dir', baseline.worker_path(output, config), '--trusted-local-tree']
    result = record(command, 'comparison')
    return 0 if result['exit_code'] == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/beam_baseline.wsl.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(json.loads(args.config.read_text(encoding='utf-8')), args.run_dir))
