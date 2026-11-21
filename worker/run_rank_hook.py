"""Run the original beam CLI with reviewed process-local ranking hooks (Python 3.8)."""
import os
# Must precede NumPy/Torch imports, exactly as in the official entrypoint.
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import runpy
import subprocess
import sys
from typing import List, Optional, Tuple

from rank_hook import EventLog, installed_hooks
from rank_contract import REGISTRY


COMMIT = "3727e716b57e0b508c4e9ecd9fefa6953e8f72da"
MAX_SEED = 2 ** 32 - 1


def seed_argument(value: str) -> int:
    """Accept the range supported by upstream's NumPy RNG seeding."""
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= seed <= MAX_SEED:
        raise argparse.ArgumentTypeError("seed must be between 0 and 2**32 - 1")
    return seed


def build_run_inputs(root: Path, native_log_dir: Path, seed: int = 0,
                     input_snapshot: Optional[dict] = None) -> Tuple[List[str], Optional[dict]]:
    """Bind the native solver and optional memory context to the same seed.

    Seed zero intentionally retains the original argv byte-for-byte: upstream
    already defaults to zero. Memory remains scoped to the actual solver seed.
    """
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    argv = [str(root / "plan_sequence/run_seq_plan.py"), "--dir", "beam_assembly",
            "--id", "original", "--planner", "dfs", "--generator", "heur-out",
            "--max-gripper", "2", "--base-part", "6", "--log-dir", str(native_log_dir),
            "--early-term"]
    if seed != 0:
        argv.extend(["--seed", str(seed)])
    context = None
    if input_snapshot is not None:
        from evidence_memory import beam_context
        from memory_ranker import BEAM_SETUP
        context = beam_context(input_snapshot, BEAM_SETUP, seed=seed)
    return argv, context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root", type=Path, required=True)
    parser.add_argument("--native-log-dir", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("trace", "noop", "rank"), required=True)
    parser.add_argument("--ranker", choices=tuple(REGISTRY), default="noop")
    parser.add_argument("--memory-mode", choices=("off", "on"))
    parser.add_argument("--memory-snapshot", type=Path)
    parser.add_argument("--seed", type=seed_argument, default=0)
    args = parser.parse_args()
    if args.mode != "rank" and args.ranker != "noop":
        parser.error("a non-noop ranker requires --mode rank")
    if bool(args.memory_mode) != bool(args.memory_snapshot) or (args.memory_mode and args.mode != "rank"):
        parser.error("memory requires --mode rank and both memory options")
    root = args.asap_root.resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("unsupported ASAP source revision")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--"], check=True)
    if args.native_log_dir.exists():
        raise FileExistsError("native log directory must be new")
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from plan_sequence.generator.heuristic import HeuristicsOutsidenessGenerator
    from plan_sequence.planner.base import SequencePlanner

    memory = None
    memory_fields = {}
    input_snapshot = None
    if args.memory_mode:
        from compare_rank_hook import snapshot
        input_snapshot = snapshot(root)
    argv, memory_context = build_run_inputs(root, args.native_log_dir, args.seed, input_snapshot)
    if args.memory_mode:
        from memory_ranker import MemoryRanker
        memory = MemoryRanker(args.memory_snapshot, memory_context, args.memory_mode)
        memory_fields = dict(memory_mode=memory.mode, memory_snapshot_sha256=memory.snapshot_sha256,
                             memory_context_digest=memory.context_digest)

    def rng_digest() -> str:
        payload = pickle.dumps((random.getstate(), np.random.get_state()), protocol=4)
        payload += torch.get_rng_state().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    log = EventLog(args.trace_file, args.run_id)
    log.emit("run_start", mode=args.mode, source_commit=commit, native_argv=argv,
             seed=args.seed,
             ranker_name=args.ranker if args.mode != "trace" else None,
             scope="candidate_pose_feasibility", internal_geometry_operations=None,
             live_llm_calls=0, state_id_scope="remaining_parts_within_run_only", **memory_fields)
    previous_argv = sys.argv
    try:
        with installed_hooks(HeuristicsOutsidenessGenerator, SequencePlanner, log,
                             args.mode, rng_digest, ranker_name=args.ranker, memory=memory):
            sys.argv = argv
            runpy.run_path(argv[0], run_name="__main__")
        log.emit("run_end", status="returned_from_upstream")
    except BaseException as error:
        log.emit("run_end", status="ERROR", error_type=type(error).__name__)
        raise
    finally:
        sys.argv = previous_argv
        log.close()
    # Upstream catches planner exceptions: its process return is not plan success.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
