"""Official ASAP entrypoint with a separate non-heuristic Generator (3.8)."""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import runpy
import subprocess
import sys

from independent_features import describe_task
from independent_rank import IndependentSession, installed_independent
from rank_hook import EventLog
from remote_rank_client import RemoteRankClient, checked_path
from run_rank_hook import COMMIT, seed_argument
from task_memory import task_context


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--asap-root', type=Path, required=True)
    value.add_argument('--assembly-dir', type=Path, required=True)
    value.add_argument('--output-dir', type=Path, required=True)
    value.add_argument('--run-id', required=True)
    value.add_argument('--seed', type=seed_argument, default=0)
    value.add_argument('--base-part', default=None)
    value.add_argument('--budget', type=int, default=400)
    value.add_argument('--max-gripper', type=int, default=2)
    value.add_argument('--max-pose', type=int, default=3)
    value.add_argument('--timeout', type=int, default=None)
    value.add_argument('--response-timeout', type=int, default=45)
    return value


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def main():
    args = parser().parse_args()
    root = checked_path(args.asap_root, run_only=False)
    assembly = checked_path(args.assembly_dir, run_only=False)
    output = checked_path(args.output_dir)
    if (args.budget <= 0 or args.max_gripper <= 0 or args.max_pose <= 0
            or not 0 < args.response_timeout <= 120 or
            (args.timeout is not None and args.timeout <= 0)):
        raise ValueError('positive budgets required')
    for name in ('native', 'features.json', 'events.jsonl', 'remote_audit.jsonl', 'context.json'):
        if (output / name).exists():
            raise FileExistsError('worker artifacts must be new')
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                     universal_newlines=True).strip()
    if commit != COMMIT:
        raise ValueError('unsupported_ASAP_revision')
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from plan_sequence.generator.base import Generator
    from plan_sequence.generator import generators
    from plan_sequence.planner.base import SequencePlanner

    def rng_digest():
        payload = pickle.dumps((random.getstate(), np.random.get_state()), protocol=4)
        payload += torch.get_rng_state().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    setup = dict(budget=args.budget, max_grippers=args.max_gripper, max_poses=args.max_pose,
                 pose_reuse=0, early_term=True, timeout=args.timeout, plan_grasp=False,
                 plan_arm=False, gripper_type='robotiq-140', gripper_scale=0.4,
                 optimizer='L-BFGS-B', seed=args.seed, save_sdf=True, num_proc=1)
    # Contact is deliberately null until measured with an audited source; CAD
    # bounds/COM are final-assembly-frame observations, not a sampled pose.
    features = describe_task(root, assembly, args.base_part, setup)
    write_new(output / 'features.json', features)
    write_new(output / 'context.json', task_context(features))
    argv = [str(root / 'plan_sequence/run_seq_plan.py'), '--dir', str(assembly.parent),
            '--id', assembly.name, '--planner', 'dfs', '--generator', 'independent',
            '--max-gripper', str(args.max_gripper), '--max-pose', str(args.max_pose),
            '--budget', str(args.budget), '--seed', str(args.seed), '--early-term',
            '--log-dir', str(output / 'native')]
    if args.base_part is not None:
        argv.extend(['--base-part', args.base_part])
    if args.timeout is not None:
        argv.extend(['--timeout', str(args.timeout)])
    client = RemoteRankClient(output / 'ipc', output / 'remote_audit.jsonl',
                              timeout_seconds=args.response_timeout)
    log = EventLog(output / 'events.jsonl', args.run_id)
    session = IndependentSession(log, client, features, rng_digest)
    previous_argv = sys.argv
    try:
        log.emit('run_start', native_argv=argv, source_commit=commit, tracked_source_clean=True,
                 seed=args.seed, generator='independent', heuristic_fallback=False,
                 terminal_two_parts='unchanged_upstream_mass_or_base_rule',
                 state_owner='upstream_DFS', llm_execution='external_IPC_service')
        with installed_independent(Generator, generators, SequencePlanner, session):
            sys.argv = argv
            runpy.run_path(argv[0], run_name='__main__')
        log.emit('run_end', status='returned_from_upstream',
                 heuristic_tripwire_calls=session.tripwire_calls,
                 verifier_calls=session.verifier_count)
    except BaseException as error:
        log.emit('run_end', status='ERROR', error_type=type(error).__name__,
                 heuristic_tripwire_calls=session.tripwire_calls)
        raise
    finally:
        sys.argv = previous_argv
        log.close()
        client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
