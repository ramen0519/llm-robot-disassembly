"""Read official CAD geometry for ranking hints; no physics or learned labels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import stat
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asap-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if str(args.asap_root) != '/opt/asap-research/ASAP':
        raise ValueError('use the dedicated ASAP checkout')
    output = str(args.output)
    if not output.startswith('/mnt/c/dev/disassembly_codex_ready_v2/runs/') or '..' in args.output.parts:
        raise ValueError('output must be in project runs')
    for path in reversed((args.output,) + tuple(args.output.parents)):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('output links are excluded')
    commit = subprocess.check_output(['git', '-C', str(args.asap_root), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    if commit != '3727e716b57e0b508c4e9ecd9fefa6953e8f72da':
        raise ValueError('source revision changed')
    sys.path.insert(0, str(args.asap_root))
    from assets.load import load_assembly
    directory = args.asap_root / 'assets/beam_assembly/original'
    assembly = load_assembly(str(directory), transform='final')
    source_files = [path for path in directory.iterdir() if path.suffix in ('.obj', '.json')]
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    parts = {}
    for part, data in assembly.items():
        mesh = data['mesh']
        parts[part] = dict(center_mass=mesh.center_mass.tolist(), bounds=mesh.bounds.tolist(),
                           extents=mesh.extents.tolist(), volume=float(mesh.volume),
                           watertight=bool(mesh.is_watertight))
    result = dict(schema_version=1, source_commit=commit, source_sha256=before,
                  assembly='beam_assembly/original', parts=parts, base_part='6',
                  frame='upstream load_assembly final CAD frame; not a sampled stable pose',
                  role='geometric ranking hints, not proof of collision, blocking, stability or removability',
                  verifier_calls=0, prior_plan_or_verdicts_used=False)
    if before != {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}:
        raise ValueError('input changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'parts': len(parts), 'verifier_calls': 0, 'output': output}))


if __name__ == '__main__':
    main()
