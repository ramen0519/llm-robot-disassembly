"""New opt-in run for the ID-hardened two-episode test; no prior approval reused."""
import argparse
from pathlib import Path
import sys

ROOT = Path('C:/dev/disassembly_codex_ready_v2')
sys.path.insert(0, str(ROOT / 'tools'))
import start_skill_two_episode as base

RUN_NAME = 'skill_live_two_episode_idfix_20261017'


def launch(authorize=False, root=ROOT, recorder=base.run_recorded):
    return base.launch(authorize, root, recorder, run_name=RUN_NAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--authorize-six-calls', action='store_true')
    args = parser.parse_args()
    try:
        raise SystemExit(launch(args.authorize_six_calls))
    except Exception as error:
        print('LAUNCH_ERROR_TYPE=' + type(error).__name__, flush=True)
        raise SystemExit(1)
