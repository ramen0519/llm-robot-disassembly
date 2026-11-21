"""Bounded, read-only ASAP path replay and exact-mesh visualization export (Py3.8).

No API, candidate search, or modification to upstream/input/old results. Recorded
actions are re-simulated; this does not recover an originally recorded trajectory.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time


def hashes(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()}


def write_new(path: Path, data: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--asap', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--assembly-07009', type=Path, required=True)
    parser.add_argument('--assembly-02628', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.asap), str(args.project / 'worker')]
    import numpy as np
    from assets.load import load_assembly_all_transformed, load_part_ids
    from plan_sequence import physics_planner as physics
    from replay_path_extension import readonly_sdf_cache

    watched = [args.source_run, args.assembly_07009, args.assembly_02628]
    before = {str(p): hashes(p) for p in watched}
    write_new(args.output / 'preservation_before.json', before)
    report = {'api_calls': 0, 'source_run': str(args.source_run),
              'mode': 'new_simulation_of_saved_selected_actions',
              'original_continuous_trajectory_available': False,
              'candidate_search_rerun': False, 'stability_replayed': False,
              'path_checks': 0, 'steps': [], 'status': 'RUNNING'}
    assets = {}
    try:
        for assembly_id, folder in [('07009', args.assembly_07009), ('02628', args.assembly_02628)]:
            meshes = load_assembly_all_transformed(str(folder))
            parts = []
            bounds = []
            for pid, item in meshes.items():
                mesh = item['mesh']
                positions = np.asarray(mesh.vertices, dtype='<f4')
                faces = np.asarray(mesh.faces, dtype='<u2')
                if len(positions) >= 65536 or faces.max() >= len(positions):
                    raise ValueError('mesh_index_range')
                normals = np.asarray(mesh.vertex_normals, dtype='<f4')
                if item.get('final_state') is not None:
                    raise ValueError('nonidentity_configuration_needs_explicit_transform')
                bounds.extend([mesh.bounds[0], mesh.bounds[1]])
                parts.append({'id': pid, 'vertices': base64.b64encode(positions.tobytes()).decode(),
                              'normals': base64.b64encode(normals.tobytes()).decode(),
                              'faces': base64.b64encode(faces.tobytes()).decode(),
                              'vertex_count': len(positions), 'triangle_count': len(faces),
                              'bounds': mesh.bounds.tolist()})
            points = np.array(bounds)
            assets[assembly_id] = {'parts': parts, 'bounds': [points.min(0).tolist(), points.max(0).tolist()],
                                   'geometry': 'all original OBJ faces; float32 display coordinates'}
            print('MESH', assembly_id, len(parts), sum(p['triangle_count'] for p in parts), flush=True)

        parsed = json.loads((args.source_run / 'parsed_result.json').read_text())
        if not parsed['plan_success'] or not parsed['validation_ok']:
            raise ValueError('source_plan_not_validated_success')
        sequence = parsed['raw_stats']['sequence']
        events = [json.loads(line) for line in (args.source_run / 'events.jsonl').read_text().splitlines()]
        requests = {e['call_id']: e['request'] for e in events if e.get('kind') == 'verifier_start'}
        remaining = load_part_ids(str(args.assembly_07009))
        steps = []
        np.random.seed(0)
        with readonly_sdf_cache(physics) as cache_audit:
            for index, pid in enumerate(sequence):
                rest = [p for p in remaining if p != pid]
                matches = [e for e in events if e.get('kind') == 'verifier_result'
                           and e.get('status') == 'PASS' and e['raw']['part_move'] == pid
                           and sorted(requests[e['call_id']]['parts_rest']) == sorted(rest)]
                if len(matches) != 1:
                    raise ValueError('ambiguous_or_missing_selected_pass_event')
                event = matches[0]
                raw = event['raw']
                pose = np.array(raw['pose']) if raw['pose'] is not None else None
                started = time.monotonic()
                planner = physics.MultiPartPathPlanner(str(args.asap / 'assets'), str(args.assembly_07009),
                    rest, pid, pose=pose, save_sdf=True)
                report['path_checks'] += 1
                success, path = planner.check_success(np.array(raw['action']), return_path=True)
                step_report = {'part': pid, 'source_call': event['call_id'], 'source_event': event['event_id'],
                               'path_success': bool(success), 'path_states': len(path),
                               'action': raw['action'], 'pose': raw['pose'], 'parts_rest': rest,
                               'recorded_supports': raw['parts_fix'],
                               'source_stability_success': event['stage_feedback']['support_search_succeeded'],
                               'stability_rechecked': False, 'elapsed_seconds': time.monotonic() - started}
                report['steps'].append(step_report)
                print('REPLAY', index + 1, pid, bool(success), len(path), flush=True)
                if not success:
                    raise RuntimeError('saved_action_did_not_replay_successfully')
                # Preserve full re-simulation states; only display sampling is reduced.
                write_new(args.output / ('step_%d_part_%s_full_path.json' % (index + 1, pid)),
                          dict(step_report, qm=[np.asarray(q).tolist() for q in path]))
                ids = np.unique(np.linspace(0, len(path)-1, min(80, len(path))).astype(int))
                q_his = [planner.sim.get_joint_q_from_qm('part'+pid, path[i]) for i in ids]
                planner.sim.set_state_his(q_his, [np.zeros(6) for _ in q_his])
                matrices = planner.sim.export_replay_matrices()
                # Exported matrices map original OBJ coordinates to world coordinates.
                frames = [{name.replace('part', ''): np.round(mat, 7).tolist()
                           for name, mat in frame.items()} for frame in matrices]
                steps.append(dict(step_report, frames=frames, sampled_indices=ids.tolist()))
                remaining = rest
            report['cache_audit'] = cache_audit
        if remaining != parsed['terminal_parts']:
            raise ValueError('terminal_parts_mismatch')
        assets['07009']['sequence'] = sequence
        assets['07009']['steps'] = steps
        assets['07009']['terminal_parts'] = remaining
        report['status'] = 'PASS'
        report['stability_note'] = 'Original run PASS evidence retained; this visualization rechecks paths only.'
        write_new(args.output / 'scene_data.json', assets)
    finally:
        after = {str(p): hashes(p) for p in watched}
        report['preserved'] = before == after
        write_new(args.output / 'preservation_after.json', after)
        write_new(args.output / 'report.json', report)
        if before != after:
            raise RuntimeError('input_or_original_run_changed')


if __name__ == '__main__':
    main()
