"""Run official smoke physics unchanged; make its final viewer replay finite.

This is the official entrypoint with a playback-options wrapper, not an
unmodified CLI invocation. Use xvfb-run when a visible window is unwanted.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asap-root', required=True, type=Path)
    parser.add_argument('--summary', required=True, type=Path)
    args = parser.parse_args()
    if args.summary.exists():
        raise FileExistsError(str(args.summary))
    root = args.asap_root.resolve(strict=True)
    sys.path.insert(0, str(root))
    import numpy as np
    from utils.renderer import SimRenderer

    original_replay = SimRenderer.replay

    def replay_once(sim, *positional, **keywords):
        sim.viewer_options.loop = False
        sim.viewer_options.infinite = False
        print('OFFICIAL_PHYSICS_LOOP_FINISHED; replay once', flush=True)
        return original_replay(sim, *positional, **keywords)

    SimRenderer.replay = staticmethod(replay_once)
    script = root / 'test_sim/test_simple_sim.py'
    sys.argv = [str(script), '--model', 'box/box_stack', '--steps', '2000']
    try:
        state = runpy.run_path(str(script), run_name='__main__')
    finally:
        SimRenderer.replay = staticmethod(original_replay)
    data = {key: np.asarray(state[key]).tolist() for key in ('q', 'qdot', 'variables')}
    finite = all(np.isfinite(np.asarray(data[key])).all() for key in data)
    result = {'physics_steps': 2000, 'finite_final_state': bool(finite),
              'viewer_replay_completed': True, 'wrapper': 'loop=False,infinite=False',
              'final_state': data, 'plan_success': None}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    return 0 if finite else 1


if __name__ == '__main__':
    raise SystemExit(main())
