"""Freeze previous beam evidence, then run paired memory off/on real physics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import run_asap_baseline as baseline
from run_m3_comparison import write_json

ROOT = baseline.ROOT
VARIANTS = ('heur_off', 'heur_on', 'reverse_off', 'reverse_on')
PAIRS = (('heur_off', 'heur_on'), ('reverse_off', 'reverse_on'))
SOURCE_RUN = ROOT / 'runs/m4_rules_20261017'
PRESERVE = ('m1_wsl_20261016', 'm2_wsl_20261016', 'm3_wsl_20261016', 'm4_rules_20261017')


def preserved_artifacts() -> dict[str, str]:
    hashes = {}
    for name in PRESERVE:
        folder = ROOT / 'runs' / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def memory_hashes(output: Path) -> dict[str, str]:
    return {key: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for key, name in (('snapshot', 'memory_snapshot.json'), ('source_db', 'source_memory.sqlite'))}


def execute(config: dict, output: Path) -> int:
    output = baseline.run_path(output)
    baseline.beam_command(config, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    write_json(output / 'protocol.json', {
        'schema_version': 1, 'variants': VARIANTS, 'pairs': PAIRS,
        'source_run': 'runs/m4_rules_20261017',
        'source_variants': ['baseline', 'id_ascending', 'reverse_native'],
        'policy': 'positive_only_stable_promotion', 'allow_hard_pruning': False,
        'rules_fixed_before_execution': {'heur': 'noop / native heur-out', 'reverse': 'reverse_native'},
        'scope': 'known beam revisit; seed0 budget400 same warm cache; not held-out evaluation',
        'goal': 'native full plan ending with base part 6',
        'lookup_scope': 'same input/binary/config/seed/remaining-parts/base/pose; soft hint, not exact verdict cache',
        'failure_usage': 'store UNKNOWN/ERROR without negative score or candidate exclusion',
        'positive_weight': 'boolean prior PASS, stable base order inside each group',
        'evaluation_evidence': 'separate database after all four runs; never read during evaluation',
        'verifier_count_scope': 'SequencePlanner._simulate candidate/pose calls including leaf',
        'internal_geometry_operations': None, 'skipped_edge_count': None,
        'live_llm_calls': 0, 'performance_generalization_claim': None,
        'preexisting_baseline_runs': PRESERVE,
    })
    write_json(output / 'preserved_before.json', preserved_artifacts())
    frozen = output / 'implementation'
    frozen.mkdir()
    source_files = [ROOT / 'worker' / name for name in (
        'rank_contract.py', 'rank_hook.py', 'run_rank_hook.py', 'compare_rank_hook.py',
        'compare_rule_rankers.py', 'compare_memory_runs.py', 'validate_baseline.py',
        'evidence_memory.py', 'memory_ranker.py', 'build_evidence.py')]
    source_files += [ROOT / 'tools' / name for name in (
        'run_memory_comparison.py', 'run_m3_comparison.py', 'run_asap_baseline.py', 'run_recorded.py')]
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

    def import_evidence(source: Path, variants: tuple, stem: str, directory: Path) -> dict:
        return record(prefix + [config['python'], worker + '/build_evidence.py',
                      '--source-run', baseline.worker_path(source, config), '--variants'] + list(variants) + [
                      '--db', baseline.worker_path(directory / (stem + '_memory.sqlite'), config),
                      '--snapshot', baseline.worker_path(directory / 'memory_snapshot.json', config),
                      '--manifest', baseline.worker_path(directory / 'memory_manifest.json', config)],
                      stem + '_build')

    hashes = {}
    try:
        result = import_evidence(SOURCE_RUN, ('baseline', 'id_ascending', 'reverse_native'), 'source', output)
        if result['exit_code'] != 0:
            return 1
        hashes['before'] = memory_hashes(output)
        for variant in VARIANTS:
            print('{}: snapshot and actual physics start'.format(variant), flush=True)
            if snapshot('inputs_before_' + variant)['exit_code'] != 0:
                return 1
            run = output / variant
            run.mkdir()
            ranker = 'noop' if variant.startswith('heur_') else 'reverse_native'
            mode = variant.rsplit('_', 1)[1]
            write_json(run / 'config.json', {'mode': 'rank', 'ranker': ranker,
                       'memory_mode': mode, 'baseline': config})
            command = prefix + [config['python'], '-u', worker + '/run_rank_hook.py',
                '--asap-root', config['asap_root'],
                '--native-log-dir', baseline.worker_path(run / 'native', config),
                '--trace-file', baseline.worker_path(run / 'events.jsonl', config),
                '--run-id', output.relative_to(ROOT).as_posix() + '/' + variant,
                '--mode', 'rank', '--ranker', ranker, '--memory-mode', mode,
                '--memory-snapshot', baseline.worker_path(output / 'memory_snapshot.json', config)]
            record(command, variant + '/execute')
            record(prefix + [config['python'], worker + '/validate_baseline.py',
                '--log-dir', baseline.worker_path(run / 'native/dfs-heur-out/s0/original', config),
                '--output', baseline.worker_path(run / 'parsed_result.json', config),
                '--trusted-local-tree'], variant + '/validation')
            if snapshot('inputs_after_' + variant)['exit_code'] != 0:
                return 1
            hashes['after_' + variant] = memory_hashes(output)
            if hashes['before'] != hashes['after_' + variant]:
                raise ValueError('frozen memory changed')
            if (output / ('inputs_before_' + variant) / 'stdout.log').read_bytes() != (
                    output / ('inputs_after_' + variant) / 'stdout.log').read_bytes():
                raise ValueError('input/cache changed')
            print('{}: execution and artifact check recorded'.format(variant), flush=True)
        archive = output / 'evaluation_evidence'
        archive.mkdir()
        if import_evidence(output, VARIANTS, 'evaluation', archive)['exit_code'] != 0:
            return 1
        hashes['after_evaluation_archive'] = memory_hashes(output)
    finally:
        write_json(output / 'preserved_after.json', preserved_artifacts())
        write_json(output / 'memory_hashes.json', hashes)
    result = record(prefix + [config['python'], worker + '/compare_memory_runs.py',
                    '--run-dir', baseline.worker_path(output, config), '--trusted-local-tree'], 'comparison')
    return 0 if result['exit_code'] == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/beam_baseline.wsl.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(json.loads(args.config.read_text(encoding='utf-8')), args.run_dir))
