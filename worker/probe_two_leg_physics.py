"""One bounded actual Beam controller probe; no API, mock or DFS comparison."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
from path_repair import execute_single, execute_two_leg
from remote_rank_client import checked_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = checked_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path('/opt/asap-research/ASAP')
    assembly = root / 'assets/beam_assembly/original'
    sys.path.insert(0, str(root))
    from plan_sequence.physics_planner import MultiPartPathPlanner
    from plan_sequence.feasibility_check import get_stable_plan_1pose_serial
    paths = sorted(assembly.glob('*'))
    def hashes():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    before = hashes()
    planner = MultiPartPathPlanner(str(root / 'assets'), str(assembly), ['0', '2', '3', '6'], '1',
                                   pose=None, save_sdf=False)
    single = execute_single(planner, [1, 0, 0])
    repaired = execute_two_leg(planner, [1, 0, 0], [0, 0, 1], 50)
    supports = get_stable_plan_1pose_serial(str(root / 'assets'), str(assembly), ['0', '2', '3', '6'],
                        '6', pose=None, max_fix=1, save_sdf=False, timeout=20)
    result = {'physics': 'actual_ASAP_RedMax', 'single': single, 'two_leg': repaired,
              'supports': supports, 'asset_before': before, 'asset_after': hashes(),
              'no_API': True, 'native_six_axis_failure_recovery': False}
    with (output / 'result.json').open('x') as stream:
        json.dump(result, stream, allow_nan=False)
    print(json.dumps({'single_success': single['raw_success'], 'two_leg_success': repaired['raw_success'],
          'geometry': repaired['geometry'], 'supports': supports, 'assets_preserved': before == hashes()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
