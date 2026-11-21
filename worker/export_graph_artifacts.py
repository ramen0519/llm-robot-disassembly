"""Revalidate and fingerprint this task's trusted, locally produced beam tree."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_rank_hook import read, tree_data
from validate_baseline import validate_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trusted-local-tree', action='store_true')
    args = parser.parse_args()
    if not args.trusted_local_tree:
        parser.error('explicit local pickle trust is required')
    for path in (args.log_dir, args.output):
        if not str(path).startswith('/mnt/c/dev/disassembly_codex_ready_v2/runs/') or '..' in path.parts:
            raise ValueError('only project run artifacts are allowed')
        if any(item.is_symlink() for item in (path,) + tuple(path.parents)):
            raise ValueError('links are excluded')
    with args.output.open('x', encoding='utf-8') as stream:
        result = validate_baseline(args.log_dir, True)
        result['tree_fingerprint_sha256'] = hashlib.sha256(json.dumps(
            tree_data(args.log_dir), sort_keys=True, allow_nan=False).encode()).hexdigest()
        result['setup'] = read(args.log_dir / 'setup.json')
        json.dump(result, stream, indent=2)
    print(json.dumps({'status': result['status'], 'plan_success': result['plan_success']}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
