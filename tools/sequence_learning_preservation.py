"""Snapshot only this workspace, refusing links before traversal."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from run_luna_experiment import safe, write, read


def inventory(root):
    result = {}
    for folder, dirs, names in os.walk(safe(root), followlinks=False):
        for name in list(dirs):
            path = Path(folder)/name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
                dirs.remove(name)
        for name in names:
            path = Path(folder)/name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
                continue
            result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    output = safe(ROOT/'runs/sequence_learning_prepare_20261110', True)
    if sys.argv[1] == 'before':
        prior = inventory(ROOT/'runs')
        output.mkdir(exist_ok=False)
        write(output/'protected_before.json', prior)
        before = output/'before'; before.mkdir()
        for name in ('orchestrator/sequence_ranker.py', 'tools/run_sequence_native_experiment.py',
                     'IMPLEMENTATION_STATUS.md', 'CHANGELOG.md', 'docs/12_DECISION_LOG_AND_OPEN_QUESTIONS.md',
                     'LATEST_DECISIONS.md', 'PROJECT_CONTEXT.md', 'docs/SEQUENCE_NATIVE_PROFILE.md'):
            path = before/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(safe(ROOT/name).read_bytes())
        print(json.dumps({'protected_files':len(prior)}))
    else:
        prior = read(output/'protected_before.json')
        changed = [p for p,h in prior.items() if not safe(ROOT/p).is_file() or
                   hashlib.sha256(safe(ROOT/p).read_bytes()).hexdigest() != h]
        write(output/'preservation.json', {'protected_files':len(prior),'changed':changed,'preserved':not changed})
        print(json.dumps({'protected_files':len(prior),'changed':changed}))


if __name__ == '__main__':
    main()
