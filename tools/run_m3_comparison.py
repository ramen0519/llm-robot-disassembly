"""Execute three real beam runs, retaining all M1/M2 artifacts (Windows host)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import run_asap_baseline as baseline

ROOT = baseline.ROOT


def write_json(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def preserved_artifacts() -> dict[str, str]:
    result = {}
    for name in ('m1_wsl_20261016', 'm2_wsl_20261016'):
        folder = ROOT / 'runs' / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def execute(config: dict, output: Path) -> int:
    output = baseline.run_path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    write_json(output / 'preserved_before.json', preserved_artifacts())
    frozen = output / 'implementation'
    frozen.mkdir()
    paths = [ROOT / 'worker' / name for name in (
        'rank_contract.py', 'rank_hook.py', 'run_rank_hook.py', 'compare_rank_hook.py',
        'validate_baseline.py')]
    paths += [ROOT / 'tools' / name for name in (
        'run_m3_comparison.py', 'run_asap_baseline.py', 'run_recorded.py')]
    for path in paths:
        shutil.copyfile(path, frozen / path.name)
    write_json(output / 'implementation_sha256.json', {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen.iterdir()})
    prefix = baseline.wsl_prefix(config)
    worker = baseline.worker_path(frozen, config)

    def record(command: list[str], name: str) -> bool:
        return baseline.recorder.run_recorded(command, ROOT, output / name)['exit_code'] == 0

    def snapshot(name: str) -> bool:
        return record(prefix + [config['python'], worker + '/compare_rank_hook.py',
                                '--snapshot', config['asap_root']], name)

    success = snapshot('inputs_before')
    try:
        if not success:
            return 1
        print('M3 direct: starting original CLI', flush=True)
        success = baseline.execute(config, output / 'direct') == 0
        success = snapshot('inputs_after_direct') and success
        if not success:
            return 1
        for mode in ('trace', 'noop'):
            print('M3 {}: starting actual physics'.format(mode), flush=True)
            run = output / mode
            run.mkdir()
            write_json(run / 'config.json', {'mode': mode, 'baseline': config})
            command = prefix + [config['python'], '-u', worker + '/run_rank_hook.py',
                                '--asap-root', config['asap_root'],
                                '--native-log-dir', baseline.worker_path(run / 'native', config),
                                '--trace-file', baseline.worker_path(run / 'events.jsonl', config),
                                '--run-id', output.name + '_' + mode, '--mode', mode]
            success = record(command, mode + '/execute') and success
            validator = prefix + [config['python'], worker + '/validate_baseline.py',
                                  '--log-dir', baseline.worker_path(run / 'native/dfs-heur-out/s0/original', config),
                                  '--output', baseline.worker_path(run / 'parsed_result.json', config),
                                  '--trusted-local-tree']
            success = record(validator, mode + '/validation') and success
            success = snapshot('inputs_after_' + mode) and success
            if not success:
                return 1
    finally:
        write_json(output / 'preserved_after.json', preserved_artifacts())
    print('M3 comparison: checking artifacts and events', flush=True)
    command = prefix + [config['python'], worker + '/compare_rank_hook.py',
                        '--run-dir', baseline.worker_path(output, config), '--trusted-local-tree']
    success = record(command, 'comparison') and success
    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/beam_baseline.wsl.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(json.loads(args.config.read_text(encoding='utf-8')), args.run_dir))
