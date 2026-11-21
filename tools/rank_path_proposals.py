"""Rank saved path-only proposals by peak and total observed support count."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from worker.path_only_sequence_proposal import PROFILE
from worker.proposal_support_policy import effective_support_cap, parse_support_cap
from worker.proposal_support_score import score_plan
from worker.task_memory import _lexical_path, _safe_path


ASAP_SOURCES = {
    'plan_sequence/planner/base.py', 'plan_sequence/planner/dfs.py',
    'plan_sequence/physics_planner.py', 'plan_sequence/feasibility_check.py',
    'plan_sequence/stable_pose.py', 'plan_sequence/sim_string.py',
}
RUNNER_SOURCES = {'run_path_only_proposal.py', 'path_only_sequence_proposal.py', 'proposal_stability.py'}
PHYSICS_CONSTANTS = {
    'FORCE_MAG', 'FRAME_SKIP', 'MAX_TIME', 'POS_FAR_THRESHOLD', 'MAX_STEP',
    'CONTACT_EPS', 'MIN_SEP', 'COL_TH_PATH', 'COL_TH_STABLE',
}


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _origin_path(value):
    """Compare the recorded WSL/host output location without opening it."""
    if not isinstance(value, str) or not value:
        raise ValueError('missing recorded output directory')
    raw = _lexical_path(value).replace('\\', '/')
    if raw.lower().startswith('/mnt/c/'):
        raw = 'C:/' + raw[7:]
    return PureWindowsPath(raw) if PureWindowsPath(raw).drive else Path(raw)


def _run_path(value):
    # Windows alternate data streams and device aliases are not run files.
    raw = _lexical_path(value)
    win = PureWindowsPath(raw)
    for part in win.parts[1:] if win.anchor else win.parts:
        if ':' in part or part.endswith((' ', '.')) or PureWindowsPath(part).is_reserved():
            raise ValueError('invalid run path component')
    return _safe_path(raw)


def _read_json(path):
    data = _run_path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f'expected a JSON object in {path}')
    return value, hashlib.sha256(data).hexdigest()


def _comparison_settings(settings, result, directory):
    """Check the v1 producer's recorded counterparts before removing run-specific fields."""
    if settings['profile'] != PROFILE or result['profile'] != PROFILE:
        raise ValueError('unsupported proposal profile')
    params = settings['parameters']
    if (not isinstance(settings['assembly'], str) or not settings['assembly']
            or settings['assembly'] != params['assembly_dir']
            or _origin_path(params['output_dir']) != _origin_path(str(directory))):
        raise ValueError('settings do not identify the input/output directories of this run')
    if (type(params['seed']) is not int or not 0 <= params['seed'] < 2**32
            or params['support_cap'] != parse_support_cap(params['support_cap'])
            or any(type(params[k]) is not int or params[k] < 1 for k in ('max_poses', 'budget'))
            or any(not _finite(params[k]) or params[k] <= 0 for k in ('plan_seconds', 'audit_seconds'))):
        raise ValueError('missing or invalid search parameters')
    if (settings['generator'] != result['candidate_policy']
            or settings['api_calls'] != result['api_calls']
            or result['input_unchanged'] is not True or result['asap_sources_unchanged'] is not True):
        raise ValueError('result policy or preservation evidence differs from the run settings')
    for key in ('planner', 'generator', 'terminal', 'cache_policy'):
        if not isinstance(settings[key], str) or not settings[key]:
            raise ValueError(f'missing {key} policy')
    if (type(settings['pose_reuse']) is not int or settings['pose_reuse'] < 0
            or 'base_part' not in settings or type(settings['api_calls']) is not int):
        raise ValueError('missing or invalid run policy')
    for field, required in (('input_sha256', set()), ('asap_path_sources', ASAP_SOURCES),
                            ('runner_sha256', RUNNER_SOURCES)):
        hashes = settings[field]
        if (not isinstance(hashes, dict) or not hashes or not required.issubset(hashes)
                or not all(_sha256(value) for value in hashes.values())):
            raise ValueError(f'incomplete {field} provenance')
    if (not _sha256(settings['redmax_binary_sha256'])
            or not re.fullmatch(r'[a-f0-9]{40}|[a-f0-9]{64}', settings['asap_commit'])):
        raise ValueError('missing verifier build provenance')
    constants = settings['native_physics_constants']
    if not PHYSICS_CONSTANTS.issubset(constants) or not all(_finite(v) for v in constants.values()):
        raise ValueError('missing or invalid native physics constants')

    # Native load_part_ids lists only top-level, lower-case .obj files.
    mesh_ids = sorted(name.replace('.obj', '') for name in settings['input_sha256']
                      if '/' not in name and name.endswith('.obj'))
    if len(mesh_ids) < 2 or mesh_ids != sorted(result['initial_parts']):
        raise ValueError('input mesh IDs differ from initial_parts')
    if result['path_physics']['limits'] != {'path': 6 * params['budget'], 'stability': 1}:
        raise ValueError('recorded path limits differ from budget')
    for step in result['steps']:
        if (type(step['attempt_id']) is not int or not 1 <= step['attempt_id'] <= params['budget']
                or not _finite(step['finished_s']) or not 0 <= step['finished_s'] <= params['plan_seconds']):
            raise ValueError('recorded transition exceeds the configured path budget')
        observation = step.get('stability')
        if observation is None:
            continue
        if 'budget' not in observation and observation['up_to_cap_status'] == 'NOT_RUN':
            continue
        budget = observation['budget']
        if (type(budget['support_cap']) is not int
                or budget['support_cap'] != effective_support_cap(params['support_cap'], len(step['child']))
                or not _finite(budget['wall_time_s'])
                or not 0 < budget['wall_time_s'] <= params['audit_seconds']):
            raise ValueError('recorded step audit budget differs from settings')

    # Compare every recorded parameter, including future policy fields; identical
    # content hashes permit the same assembly to be loaded from another directory.
    group = {key: value for key, value in settings.items() if key not in ('assembly', 'parameters')}
    group['parameters'] = {key: value for key, value in params.items()
                           if key not in ('assembly_dir', 'output_dir', 'seed')}
    return group


def rank_runs(directories):
    paths = [_run_path(directory) for directory in directories]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('supply one or more distinct run directories')
    group = None
    ranked, conditional = [], []
    for directory in paths:
        try:
            settings, settings_hash = _read_json(directory / 'settings.json')
            result, result_hash = _read_json(directory / 'result.json')
            if 'settings_sha256' in result and result['settings_sha256'] != settings_hash:
                raise ValueError('result refers to a different settings.json digest')
            comparison = _comparison_settings(settings, result, directory)
            if group is None:
                group = comparison
            elif comparison != group:
                fields = sorted(k for k in group.keys() | comparison.keys()
                                if k not in group or k not in comparison or group[k] != comparison[k])
                raise ValueError('incomparable settings: ' + ', '.join(fields))
            score = score_plan(result, support_cap=settings['parameters']['support_cap'])
            if 'support_score' in result and result['support_score'] != score:
                raise ValueError('saved support score differs from the recorded trial evidence')
            row = dict(run=str(directory), seed=settings['parameters']['seed'],
                       settings_sha256=settings_hash, result_sha256=result_hash,
                       disassembly_order=result['disassembly_order'], remaining_parts=result['remaining_parts'],
                       complete=result['complete'], execution_status=result['execution_status'], rank=None, **score)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'{directory}: {error}') from error
        (ranked if row['eligible'] else conditional).append(row)
    ranked.sort(key=lambda row: row['score'])
    for index, row in enumerate(ranked):
        row['rank'] = (ranked[index - 1]['rank'] if index and row['score'] == ranked[index - 1]['score']
                       else index + 1)
    return dict(profile='path_proposal_support_ranking/v1', score_fields=['peak', 'total'],
                comparison_settings=group, ranked=ranked, conditional=conditional)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New JSON file under project runs/')
    parser.add_argument('directories', nargs='+', help='Run directories containing settings.json and result.json')
    args = parser.parse_args(argv)
    try:
        output = _run_path(args.output)
        if output.suffix.lower() != '.json':
            raise ValueError('--output requires a .json file')
        report = rank_runs(args.directories)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        output.parent.mkdir(parents=True, exist_ok=True)
        with _run_path(output).open('x', encoding='utf-8') as stream:
            stream.write(payload)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
