"""Apply pinned official subdivision to new copies, retaining input transforms."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def prepare(root, run):
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '3727e716b57e0b508c4e9ecd9fefa6953e8f72da':
        raise ValueError('Unsupported official source')
    sys.path.insert(0, str(root))
    import numpy as np
    from assets.subdivide import subdivide_assembly
    from assets.load import load_assembly
    selected = json.loads((run / 'selected_assemblies.json').read_text())
    dense = run / 'dense_assemblies'
    dense.mkdir()
    results = []
    for item in selected:
        source, target = run / 'raw_assemblies' / item['id'], dense / item['id']
        started = time.perf_counter()
        print('subdivide ' + item['id'], flush=True)
        subdivide_assembly(str(source), str(target), max_edge=0.5)
        before, after = load_assembly(str(source)), load_assembly(str(target))
        checks = {'same_parts': sorted(before) == sorted(after) == item['part_ids']}
        parts = []
        for part in before:
            old, new = before[part]['mesh'], after[part]['mesh']
            checks[part + '_watertight'] = bool(old.is_watertight and new.is_watertight)
            checks[part + '_bounds'] = bool(np.allclose(old.bounds, new.bounds, atol=1e-6, rtol=1e-6))
            checks[part + '_volume'] = bool(np.isclose(old.volume, new.volume, atol=1e-5, rtol=1e-5))
            checks[part + '_edge_length'] = bool(new.edges_unique_length.max() <= 0.50001)
            parts.append({'id': part, 'vertices_raw': len(old.vertices), 'vertices_dense': len(new.vertices),
                          'faces_raw': len(old.faces), 'faces_dense': len(new.faces),
                          'transformed_bounds': new.bounds.tolist()})
        result = {'id': item['id'], 'parts': parts, 'checks': checks,
                  'elapsed_seconds': time.perf_counter() - started, 'new_verifier_calls': 0}
        results.append(result)
        if not all(checks.values()):
            raise ValueError('Subdivision geometry invariant failed: ' + item['id'])
        print(json.dumps({'id': item['id'], 'status': 'PASS', 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    files = {str(path.relative_to(dense)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(dense.rglob('*')) if path.is_file()}
    with (run / 'mesh_preparation.json').open('x') as stream:
        json.dump({'source_commit': commit, 'method': 'official assets.subdivide.subdivide_assembly',
                   'max_edge': 0.5, 'rescale': None, 'results': results, 'dense_sha256': files}, stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asap-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.asap_root, args.run_dir)
