"""Exact M3 semantics comparison; pickle loading requires explicit local trust."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

from rank_hook import json_value


SUFFIX = 'native/dfs-heur-out/s0/original'


def read(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def snapshot(root):
    """Source cleanliness, real input/cache and loaded verifier binary identity."""
    import redmax_py
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    files = list((root / 'assets/beam_assembly/original').iterdir())
    files.append(Path(redmax_py.__file__))
    return {
        'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'tracked_source_clean': True,
        'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(files) if path.is_file()},
    }


def tree_data(folder):
    with (folder / 'tree.pkl').open('rb') as stream:
        tree = pickle.load(stream)
    # Preserve node/edge/adjacency order, every attribute and full float precision.
    return json_value({'graph': tree.graph, 'directed': tree.is_directed(),
                       'nodes': list(tree.nodes(data=True)),
                       'edges': list(tree.edges(data=True)),
                       'adjacency': [(node, list(tree.adj[node])) for node in tree.nodes]})


def stats_data(folder):
    return {key: value for key, value in read(folder / 'stats.json').items() if key != 'time'}


def events(path, kind=None):
    result = [json.loads(line) for line in path.read_text().splitlines()]
    if kind:
        result = [event for event in result if event['kind'] == kind]
    return result


def observed(path):
    return [{key: value for key, value in event.items() if key not in ('run_id', 'event_id')}
            for event in events(path) if event['kind'] not in ('run_start', 'run_end', 'rank_decision')]


def native_command(event_path):
    command = events(event_path, 'run_start')[0]['native_argv'][1:]
    command[command.index('--log-dir') + 1] = '<run-output>'
    return command


def compare(root):
    folders = {mode: root / mode / SUFFIX for mode in ('direct', 'trace', 'noop')}
    reference = folders['direct']
    checks = {}
    for mode, folder in folders.items():
        checks[mode + '_process_exit_zero'] = read(root / mode / 'execute/command.json')['exit_code'] == 0
        checks[mode + '_plan_verified'] = read(root / mode / 'parsed_result.json')['status'] == 'PASS'
        checks[mode + '_setup_equal'] = read(reference / 'setup.json') == read(folder / 'setup.json')
        checks[mode + '_stats_except_top_level_time_equal'] = stats_data(reference) == stats_data(folder)
        checks[mode + '_full_ordered_tree_equal'] = tree_data(reference) == tree_data(folder)
    for previous in ('m1_wsl_20261016/beam', 'm2_wsl_20261016/noop'):
        folder = root.parent / previous / SUFFIX
        checks[previous + '_setup_equal'] = read(reference / 'setup.json') == read(folder / 'setup.json')
        checks[previous + '_stats_except_top_level_time_equal'] = stats_data(reference) == stats_data(folder)
        checks[previous + '_full_ordered_tree_equal'] = tree_data(reference) == tree_data(folder)
    before = read(root / 'inputs_before/stdout.log')
    for mode in folders:
        checks[mode + '_input_cache_binary_source_unchanged'] = before == read(root / ('inputs_after_' + mode) / 'stdout.log')
    checks['m1_m2_all_artifact_bytes_preserved'] = read(root / 'preserved_before.json') == read(root / 'preserved_after.json')
    trace, noop = root / 'trace/events.jsonl', root / 'noop/events.jsonl'
    checks['all_observed_events_equal_in_order'] = observed(trace) == observed(noop)
    generations = events(noop, 'generation')
    decisions = events(noop, 'rank_decision')
    verifications = events(noop, 'verifier_result')
    starts = events(noop, 'verifier_start')
    checks['three_generations_three_rank_requests'] = len(generations) == len(decisions) == 3
    checks['noop_kept_every_native_order'] = all(e['native_order'] == e['applied_order'] for e in generations)
    checks['noop_no_fallback'] = all(e['fallback_reason'] is None for e in decisions)
    checks['request_ids_unique'] = len({e['request']['request_id'] for e in decisions}) == len(decisions)
    checks['four_real_verifier_calls_all_pass'] = len(starts) == len(verifications) == 4 and all(e['status'] == 'PASS' for e in verifications)
    checks['leaf_verifier_not_linked_to_rank_request'] = starts[-1]['generation_id'] is None and all(e['generation_id'] is not None for e in starts[:-1])
    for mode, path in (('trace', trace), ('noop', noop)):
        all_events = events(path)
        checks[mode + '_event_ids_unique'] = len({e['event_id'] for e in all_events}) == len(all_events)
        checks[mode + '_upstream_returned'] = all_events[-1]['kind'] == 'run_end' and all_events[-1]['status'] == 'returned_from_upstream'
    direct_command = read(root / 'direct/execute/command.json')['command']
    direct_command = direct_command[direct_command.index('plan_sequence/run_seq_plan.py') + 1:]
    direct_command[direct_command.index('--log-dir') + 1] = '<run-output>'
    checks['same_official_cli_arguments'] = direct_command == native_command(trace) == native_command(noop)
    return {
        'status': 'PASS' if all(checks.values()) else 'ERROR', 'checks': checks,
        'sequence': stats_data(reference)['sequence'],
        'native_times_seconds': {mode: read(folder / 'stats.json')['time'] for mode, folder in folders.items()},
        'upstream_n_eval': stats_data(reference)['n_eval'],
        'rank_requests': len(decisions), 'candidate_pose_verifier_calls': len(verifications),
        'internal_geometry_operations': None, 'performance_claim': None,
        'scope': 'one official beam seed0 warm-cache; Python/NumPy/Torch RNG; no generalization claim',
        'excluded_fields': ['stats.json top-level time', 'event run_id/event_id',
                            'trace run_start/run_end and noop-only rank_decision (checked separately)'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--snapshot', type=Path)
    group.add_argument('--run-dir', type=Path)
    parser.add_argument('--trusted-local-tree', action='store_true')
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(snapshot(args.snapshot), sort_keys=True))
    else:
        if not args.trusted_local_tree:
            parser.error('comparison loads only your own trusted local tree.pkl: explicit trust required')
        with (args.run_dir / 'comparison.json').open('x', encoding='utf-8') as stream:
            result = compare(args.run_dir)
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['status'] == 'PASS' else 1)
