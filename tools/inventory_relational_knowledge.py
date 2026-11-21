"""Inventory/import existing historical knowledge without altering any source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.knowledge_import import import_sources, safe_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/relational_knowledge_migration_20261119/imported_sources.json')
    args = parser.parse_args()
    # Output must be repo relative and may only be newly created.
    relative = Path(args.output)
    if relative.is_absolute() or relative.drive or '..' in relative.parts or relative.parts[:1] != ('runs',):
        raise ValueError('new_runs_output_required')
    parent = ROOT
    for part in relative.parts[:-1]:
        target = parent / part
        try:
            target.lstat()
        except FileNotFoundError:
            target.mkdir()
        else:
            safe_path(ROOT, target.relative_to(ROOT))
        parent = target
    output = ROOT / relative
    try:
        output.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError('existing_output_must_be_preserved')
    result = import_sources(ROOT)
    for item in result['inventory']:
        if item.get('sha256'):
            assert hashlib.sha256(safe_path(ROOT, item['path']).read_bytes()).hexdigest() == item['sha256']
    result['summary']['source_hashes_rechecked'] = sum(bool(row.get('sha256')) for row in result['inventory'])
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
