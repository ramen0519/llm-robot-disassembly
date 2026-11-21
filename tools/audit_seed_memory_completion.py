"""Host-only final audit after orchestration; never overwrite existing evidence."""
import argparse
from datetime import datetime
import json
from pathlib import Path, PureWindowsPath
import stat

import run_memory_multiseed as prior

PROJECT = Path(r'C:\dev\disassembly_codex_ready_v2')


def guarded(path, missing=False):
    """Reject foreign paths lexically, then lstat each component before traversal."""
    path = Path(path)
    win, base = PureWindowsPath(str(path)), PureWindowsPath(str(PROJECT))
    if win.drive.upper() != 'C:' or '..' in win.parts or not win.is_relative_to(base):
        raise ValueError('Path outside the explicitly permitted C: project')
    for component in list(path.parents)[::-1] + [path]:
        try:
            info = component.lstat()
        except FileNotFoundError:
            if missing and component == path:
                return False
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Refusing a link/reparse point: ' + str(component))
    return True


def read(path):
    guarded(path)
    return json.loads(path.read_text(encoding='utf-8'))


def hashed(path):
    guarded(path)
    return prior.file_hash(path)


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamp lacks timezone')
    return result


def audit(root):
    checks, errors, outcomes = {}, [], {}

    def section(name, action):
        try:
            action()
        except Exception as error:
            checks[name + ':readable'] = False
            errors.append({'scope': name, 'error_type': type(error).__name__, 'message': str(error)})

    def completion():
        end, result = read(root / 'orchestration_end.json'), read(root / 'comparison.json')
        checks['orchestrator_finished_successfully'] = end.get('exit_code') == 0 and end.get('error_type') is None
        guarded(root / 'progress.jsonl')
        progress = [json.loads(line) for line in (root / 'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        final = progress[-1]
        checks['comparison_finished_before_orchestration'] = (final.get('kind') == 'comparison_end'
            and final.get('process_exit_code') == 0 and stamp(final['utc']) <= stamp(end['ended_utc']))
        comparison = read(root / 'comparison/command.json')
        checks['comparison_process_finished_successfully'] = comparison.get('exit_code') == 0 and comparison.get('launch_error') is None
        checks['comparison_integrity_pass'] = (result.get('status') == result.get('integrity_status') == 'PASS'
            and isinstance(result.get('checks'), dict) and bool(result['checks'])
            and all(value is True for value in result['checks'].values()) and result.get('errors') == [])
        for key, expected in (('rows', 20), ('development_rows', 15)):
            rows = result.get(key, [])
            checks[key + ':complete_count'] = isinstance(rows, list) and len(rows) == expected
            outcomes[key] = {'count': len(rows), 'plan_success_count': sum(row.get('plan_success') is True for row in rows)}

    def preservation():
        guarded(PROJECT / 'runs')
        before, after = read(root / 'preserved_before.json'), read(root / 'preserved_after.json')
        current = prior.preserved_artifacts(root)
        checks['all_previous_artifacts_still_preserved'] = bool(before) and before == after == current
        outcomes['preserved_files'] = len(before)

    def implementation():
        declared, final = read(root / 'implementation_sha256.json'), read(root / 'implementation_final_sha256.json')
        checks['nineteen_implementation_files_declared'] = len(declared) == 19
        checks['implementation_final_manifest_matches'] = declared == final
        for name, expected in declared.items():
            if Path(name).name != name or PureWindowsPath(name).name != name or name in ('.', '..'):
                raise ValueError('Invalid implementation basename')
            choices = [PROJECT / folder / name for folder in ('worker', 'tools')]
            found = [path for path in choices if guarded(path, missing=True)]
            if len(found) != 1:
                raise ValueError('Missing or colliding implementation basename: ' + name)
            checks['implementation:' + name] = hashed(root / 'implementation' / name) == hashed(found[0]) == expected

    def memories():
        index, freeze, checkpoints = read(root / 'memories/index.json'), read(root / 'memories_frozen.json'), read(root / 'memory_hashes.json')
        current = {'s' + str(seed): {key: hashed(root / 'memories' / ('s' + str(seed)) / name)
                   for key, name in prior.SOURCE_FILES.items()} for seed in range(5)}
        checks['five_frozen_memories_match_index_and_freeze'] = current == index['per_seed'] == freeze['hashes']
        checks['frozen_memory_index_unchanged'] = hashed(root / 'memories/index.json') == freeze['index_sha256']
        expected = {'before', 'after_evaluation_archive'} | {
            'after_s{}_{}'.format(seed, variant) for seed in range(5) for variant in prior.VARIANTS}
        checks['all_twenty_evaluation_memory_checkpoints_present'] = set(checkpoints) == expected
        for key, value in checkpoints.items():
            checks['memory_checkpoint:' + key] = value == current

    for name, action in (('completion', completion), ('preservation', preservation),
                         ('implementation', implementation), ('memories', memories)):
        section(name, action)
    errors.extend({'scope': name, 'error_type': 'CheckFailed', 'message': 'Final audit check is false'}
                  for name, value in checks.items() if not value and not name.endswith(':readable'))
    return {'status': 'PASS' if checks and all(checks.values()) else 'ERROR',
            'checks': checks, 'errors': errors, 'outcomes': outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        root = args.run_dir
        guarded(root)
        if root.parent != PROJECT / 'runs' or not root.is_dir():
            raise ValueError('Run must be an existing direct child of project/runs')
        target = root / 'completion_audit.json'
        if guarded(target, missing=True):
            raise FileExistsError('Completion audit already exists')
        result = audit(root)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(json.dumps({'status': result['status'], 'check_count': len(result['checks']), 'errors': result['errors']}))
        return 0 if result['status'] == 'PASS' else 1
    except Exception as error:
        print(json.dumps({'status': 'ERROR', 'check_count': 0, 'errors': [type(error).__name__ + ': ' + str(error)]}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
