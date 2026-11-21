"""Official beam with local file IPC ranking; no API, network, or key access (3.8)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import hashlib
import math
from pathlib import Path
import pickle
import random
import runpy
import subprocess
import sys

from rank_hook import EventLog, installed_hooks
from remote_rank_client import RemoteRankClient, checked_path, installed_remote_ranker, read_json_bounded
from run_rank_hook import COMMIT, build_run_inputs, seed_argument


def timeout_argument(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return number


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root", type=Path, required=True)
    parser.add_argument("--native-log-dir", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=seed_argument, default=0)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--remote-trace-file", type=Path, required=True)
    parser.add_argument("--features-file", type=Path,
                        help="fixed observation dictionary in the approved runs directory; default {}")
    parser.add_argument("--response-timeout", type=timeout_argument, default=30.0)
    return parser


def main():
    args = create_parser().parse_args()
    root = checked_path(args.asap_root, run_only=False)
    native_log = checked_path(args.native_log_dir)
    trace_path = checked_path(args.trace_file)
    remote_trace = checked_path(args.remote_trace_file)
    ipc_dir = checked_path(args.ipc_dir)
    if native_log.exists() or trace_path.exists() or remote_trace.exists():
        raise FileExistsError("native logs and both traces must be new")
    features, feature_sha256 = {}, None
    if args.features_file is not None:
        features, feature_sha256 = read_json_bounded(args.features_file)
        if not isinstance(features, dict):
            raise ValueError("features-file must contain a dictionary")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("unsupported ASAP source revision")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--"], check=True)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from plan_sequence.generator.heuristic import HeuristicsOutsidenessGenerator
    from plan_sequence.planner.base import SequencePlanner

    def rng_digest():
        payload = pickle.dumps((random.getstate(), np.random.get_state()), protocol=4)
        payload += torch.get_rng_state().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    argv, _ = build_run_inputs(root, native_log, args.seed)
    client = RemoteRankClient(ipc_dir, remote_trace, features, args.response_timeout)
    try:
        log = EventLog(trace_path, args.run_id)
    except BaseException:
        client.close()
        raise
    previous_argv = sys.argv
    try:
        log.emit("run_start", mode="rank", source_commit=commit, tracked_source_clean=True,
                 native_argv=argv, seed=args.seed, ranker_name="remote", remote_only=True,
                 llm_execution="external_service", live_llm_calls=None,
                 scope="candidate_pose_feasibility", internal_geometry_operations=None,
                 state_id_scope="remaining_parts_within_run_only", ipc_dir=str(ipc_dir),
                 response_timeout_seconds=args.response_timeout, features_sha256=feature_sha256)
        with installed_remote_ranker(client):
            with installed_hooks(HeuristicsOutsidenessGenerator, SequencePlanner, log,
                                 "rank", rng_digest, ranker_name="remote"):
                sys.argv = argv
                runpy.run_path(argv[0], run_name="__main__")
        log.emit("run_end", status="returned_from_upstream")
    except BaseException as error:
        log.emit("run_end", status="ERROR", error_type=type(error).__name__)
        raise
    finally:
        sys.argv = previous_argv
        log.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
