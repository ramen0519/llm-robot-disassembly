"""Acquire an official archive and preselect three assemblies without solver outcomes."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

from run_memory_multiseed import ROOT, file_hash, preserved_artifacts, write

URL = 'https://people.csail.mit.edu/yunsheng/ASAP/dataset_2404/test_assembly.zip'
TARGET_COUNTS = (3, 5, 7)


def prepare(output, resume_pre_download=False, select_available_counts=False):
    output = output.resolve()
    if output.parent != ROOT / 'runs' or not output.is_dir():
        raise ValueError('Use a newly reserved direct child of project/runs')
    if resume_pre_download or select_available_counts:
        policy = json.loads((output / 'selection_policy.json').read_text())
        if policy['url'] != URL or policy['part_counts'] != list(TARGET_COUNTS):
            raise ValueError('Original selection policy differs')
        if json.loads((output / 'preserved_before.json').read_text()) != preserved_artifacts(output):
            raise ValueError('Old results changed before retry')
    else:
        write(output / 'preserved_before.json', preserved_artifacts(output))
        write(output / 'selection_policy.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'url': URL,
        'part_counts': TARGET_COUNTS,
        'selection': 'lexicographically smallest official assembly ID at each exact part count',
        'fallback': 'stop if any requested count absent; do not select by planning outcome',
        'scope': 'exploratory official test subset; no claim of a held-out final benchmark',
        'prior_results_used': False, 'memory_enabled': False, 'seed': 0,
        })
    archive = output / 'test_assembly.zip'
    counts = TARGET_COUNTS
    if select_available_counts:
        manifest = json.loads((output / 'download_manifest.json').read_text())
        if file_hash(archive) != manifest['sha256']:
            raise ValueError('Downloaded archive differs from recorded hash')
        counts = (5, 7, 9)
        write(output / 'selection_amendment.json', {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'original_part_counts': TARGET_COUNTS, 'part_counts': counts,
            'reason': 'Official test inventory starts at five parts; no three-part assembly exists',
            'selection': 'same lexicographically smallest ID rule; amendment before any planning run',
        })
    else:
        print('Downloading official test archive', flush=True)
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('xb') as dest:
            shutil.copyfileobj(response, dest, length=1024 * 1024)
            headers = dict(response.headers.items())
            final_url = response.url
        write(output / 'download_manifest.json', {
            'source_url': URL, 'final_url': final_url, 'headers': headers,
            'sha256': file_hash(archive), 'bytes': archive.stat().st_size,
        })
    assemblies = defaultdict(list)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
                raise ValueError('Unsafe archive name')
            if path.suffix == '.obj':
                assemblies[str(path.parent)].append(info)
        inventory = [{'archive_dir': name, 'id': PurePosixPath(name).name,
                      'parts': len(files), 'obj_bytes': sum(x.file_size for x in files)}
                     for name, files in sorted(assemblies.items())]
        if select_available_counts:
            if inventory != json.loads((output / 'dataset_inventory.json').read_text()):
                raise ValueError('Inventory changed')
        else:
            write(output / 'dataset_inventory.json', inventory)
        selected = []
        data = output / 'raw_assemblies'
        if data.exists():
            if not select_available_counts or any(data.iterdir()):
                raise FileExistsError('Raw extraction destination must be new or empty after the recorded selection failure')
        else:
            data.mkdir()
        for count in counts:
            candidates = sorted((x for x in inventory if x['parts'] == count), key=lambda x: x['id'])
            if not candidates:
                raise ValueError('No assembly at predeclared part count {}'.format(count))
            item = dict(candidates[0])
            target = data / item['id']
            target.mkdir()
            files = {}
            for info in bundle.infolist():
                path = PurePosixPath(info.filename)
                if str(path.parent) != item['archive_dir'] or path.suffix not in ('.obj', '.json'):
                    continue
                # Copy data only. No pickle, executables, links, or archive directory extraction.
                payload = bundle.read(info)
                with (target / path.name).open('xb') as dest:
                    dest.write(payload)
                files[path.name] = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
            item['files'] = files
            item['part_ids'] = sorted(path.stem for path in target.glob('*.obj'))
            selected.append(item)
    write(output / 'selected_assemblies.json', selected)
    print(json.dumps({'assemblies_in_archive': len(inventory),
                      'selected': [{'id': x['id'], 'parts': x['parts']} for x in selected]}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--resume-pre-download', action='store_true')
    parser.add_argument('--select-available-counts', action='store_true')
    args = parser.parse_args()
    raise SystemExit(prepare(args.run_dir, args.resume_pre_download, args.select_available_counts))
