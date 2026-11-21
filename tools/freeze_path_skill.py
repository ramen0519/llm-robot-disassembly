"""Freeze only a locally recorded path skill with a successful fresh replay."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from orchestrator.path_skill_memory import freeze_validated_skills


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    value = freeze_validated_skills(args.run_dir, args.replay, args.output)
    print(json.dumps({'status': 'PASS', 'records': value['record_count'],
                      'api_calls': 0, 'physics_calls': 0, 'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
