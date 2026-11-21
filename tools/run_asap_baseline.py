"""M2 no-op subprocess adapter for the verified official beam command.

Modern Windows Python only; ASAP state remains entirely inside its worker.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('run_recorded', ROOT / 'tools/run_recorded.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def run_path(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (ROOT / 'runs').resolve()
    if not resolved.is_relative_to(allowed) or resolved == allowed:
        raise ValueError('run directory must be a new child of project/runs')
    return resolved


def wsl_prefix(config: dict) -> list[str]:
    return ['wsl', '-d', config['distro'], '-u', config['user'], '--cd', config['asap_root'], '--']


def worker_path(path: Path, config: dict) -> str:
    return config['project_wsl'].rstrip('/') + '/' + path.relative_to(ROOT).as_posix()


def beam_command(config: dict, output: Path) -> list[str]:
    if config.get('extensions_enabled') is not False:
        raise ValueError('this adapter supports no-op execution only')
    return wsl_prefix(config) + [
        config['python'], '-u', 'plan_sequence/run_seq_plan.py',
        '--dir', 'beam_assembly', '--id', 'original', '--planner', 'dfs',
        '--generator', 'heur-out', '--max-gripper', '2', '--base-part', '6',
        '--log-dir', worker_path(output / 'native', config), '--early-term',
    ]


def execute(config: dict, output: Path) -> int:
    output = run_path(output)
    command = beam_command(config, output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    prefix = wsl_prefix(config)
    source = recorder.run_recorded(prefix + ['git', 'rev-parse', 'HEAD'], ROOT, output / 'source')
    if source['exit_code'] != 0:
        return 1
    if (output / 'source/stdout.log').read_text().strip() != config['expected_commit']:
        raise ValueError('upstream commit differs from verified baseline')
    snapshot = (
        'import hashlib,json,pathlib,sys; '
        'p=pathlib.Path(sys.argv[1]); '
        'print(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() '
        'for f in sorted(p.glob("*.sdf"))},sort_keys=True))'
    )
    snapshot_command = prefix + [config['python'], '-c', snapshot, config['asap_root'] + '/assets/beam_assembly/original']
    before = recorder.run_recorded(snapshot_command, ROOT, output / 'cache_before')
    if before['exit_code'] != 0:
        return 1
    process = recorder.run_recorded(command, ROOT, output / 'execute')
    after = recorder.run_recorded(snapshot_command, ROOT, output / 'cache_after')
    validator = prefix + [config['python'], config['project_wsl'] + '/worker/validate_baseline.py',
                          '--log-dir', worker_path(output / 'native/dfs-heur-out/s0/original', config),
                          '--output', worker_path(output / 'parsed_result.json', config), '--trusted-local-tree']
    checked = recorder.run_recorded(validator, ROOT, output / 'validation')
    return 0 if process['exit_code'] == after['exit_code'] == checked['exit_code'] == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/beam_baseline.wsl.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(json.loads(args.config.read_text(encoding='utf-8')), args.run_dir))
