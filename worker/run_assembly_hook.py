"""Run copied official assemblies with native, observed, or fixed-rule ASAP (3.8).

DFS/heur-out, one process, early termination, and disabled SDF writes are fixed.
The timeout is the official planner timeout, not an external process deadline.
This entrypoint does not support memory or modify the beam entrypoint.
"""
import os
# Match the official entrypoint before importing NumPy or Torch.
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import hashlib
from pathlib import Path
import pickle
import random
import runpy
import subprocess
import sys
from typing import List, Optional

from rank_contract import REGISTRY
from rank_hook import EventLog, installed_hooks


COMMIT = "3727e716b57e0b508c4e9ecd9fefa6953e8f72da"
MAX_SEED = 2 ** 32 - 1


def _bounded_integer(value: str, minimum: int, maximum: Optional[int] = None) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number < minimum or (maximum is not None and number > maximum):
        raise argparse.ArgumentTypeError("integer is outside the supported range")
    return number


def positive_argument(value: str) -> int:
    return _bounded_integer(value, 1)


def nonnegative_argument(value: str) -> int:
    return _bounded_integer(value, 0)


def seed_argument(value: str) -> int:
    return _bounded_integer(value, 0, MAX_SEED)


def _single_component(value: str, label: str) -> None:
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value or ":" in value or "\x00" in value
            or value.startswith("-")):
        raise ValueError(label + " must be a single nonempty ID")


def build_native_argv(root: Path, assembly_dir: Path, assembly_id: str,
                      native_log_dir: Path, seed: int = 0, base_part: Optional[str] = None,
                      max_gripper: int = 2, budget: int = 100, max_pose: int = 3,
                      pose_reuse: int = 0, timeout: int = 180) -> List[str]:
    """Create an explicit official invocation without changing rule definitions."""
    if not assembly_dir.is_absolute():
        raise ValueError("assembly-dir must be absolute and identify a copied data directory")
    _single_component(assembly_id, "assembly-id")
    if base_part is not None:
        _single_component(base_part, "base-part")
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    for name, value in (("max-gripper", max_gripper), ("budget", budget),
                        ("max-pose", max_pose), ("timeout", timeout)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if type(pose_reuse) is not int or not 0 <= pose_reuse <= max_pose:
        raise ValueError("pose-reuse must be an integer between zero and max-pose")
    argv = [str(root / "plan_sequence/run_seq_plan.py"), "--dir", str(assembly_dir),
            "--id", assembly_id, "--planner", "dfs", "--generator", "heur-out",
            "--num-proc", "1", "--seed", str(seed), "--budget", str(budget),
            "--max-gripper", str(max_gripper), "--max-pose", str(max_pose),
            "--pose-reuse", str(pose_reuse), "--timeout", str(timeout),
            "--log-dir", str(native_log_dir), "--early-term", "--disable-save-sdf"]
    if base_part is not None:
        argv.extend(["--base-part", base_part])
    return argv


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root", type=Path, required=True)
    parser.add_argument("--assembly-dir", type=Path, required=True,
                        help="absolute parent directory of copied official assembly data")
    parser.add_argument("--assembly-id", required=True)
    parser.add_argument("--native-log-dir", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path, required=True,
                        help="new metadata/event JSONL; native mode records only run boundaries")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("native", "trace", "rank"), required=True)
    parser.add_argument("--ranker", choices=tuple(REGISTRY), default="noop")
    parser.add_argument("--seed", type=seed_argument, default=0)
    parser.add_argument("--base-part", default=None)
    parser.add_argument("--max-gripper", type=positive_argument, default=2)
    parser.add_argument("--budget", type=positive_argument, default=100)
    parser.add_argument("--max-pose", type=positive_argument, default=3)
    parser.add_argument("--pose-reuse", type=nonnegative_argument, default=0)
    parser.add_argument("--timeout", type=positive_argument, default=180,
                        help="official planner timeout in seconds (default: 180)")
    return parser


def invoke_official(argv: List[str]) -> None:
    """Restore process argv even when the official entrypoint raises."""
    previous_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(argv[0], run_name="__main__")
    finally:
        sys.argv = previous_argv


def run_with_mode(argv: List[str], mode: str, ranker: str, log: EventLog) -> None:
    """Native mode calls the official entrypoint without installing any hooks."""
    if mode == "native":
        invoke_official(argv)
        return
    import numpy as np
    import torch
    from plan_sequence.generator.heuristic import HeuristicsOutsidenessGenerator
    from plan_sequence.planner.base import SequencePlanner

    def rng_digest() -> str:
        payload = pickle.dumps((random.getstate(), np.random.get_state()), protocol=4)
        payload += torch.get_rng_state().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    with installed_hooks(HeuristicsOutsidenessGenerator, SequencePlanner, log,
                         mode, rng_digest, ranker_name=ranker):
        invoke_official(argv)


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    if args.mode != "rank" and args.ranker != "noop":
        parser.error("a non-noop ranker requires --mode rank")
    root = args.asap_root.resolve()
    try:
        argv = build_native_argv(root, args.assembly_dir, args.assembly_id, args.native_log_dir,
                                 seed=args.seed, base_part=args.base_part,
                                 max_gripper=args.max_gripper, budget=args.budget,
                                 max_pose=args.max_pose, pose_reuse=args.pose_reuse,
                                 timeout=args.timeout)
    except ValueError as error:
        parser.error(str(error))
    if not (args.assembly_dir / args.assembly_id).is_dir():
        raise FileNotFoundError("copied assembly directory does not exist")
    if args.native_log_dir.exists() or args.trace_file.exists():
        raise FileExistsError("native log directory and trace file must both be new")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("unsupported ASAP source revision")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--"], check=True)
    sys.path.insert(0, str(root))
    log = EventLog(args.trace_file, args.run_id)
    log.emit("run_start", mode=args.mode, source_commit=commit, tracked_source_clean=True,
             native_argv=argv, seed=args.seed, assembly_dir=str(args.assembly_dir),
             assembly_id=args.assembly_id, base_part=args.base_part,
             ranker_name=args.ranker if args.mode == "rank" else None,
             scope="candidate_pose_feasibility", internal_geometry_operations=None,
             live_llm_calls=0, state_id_scope="remaining_parts_within_run_only",
             hooks_installed=args.mode != "native", sdf_writes_enabled=False)
    try:
        run_with_mode(argv, args.mode, args.ranker, log)
        log.emit("run_end", status="returned_from_upstream")
    except BaseException as error:
        log.emit("run_end", status="ERROR", error_type=type(error).__name__)
        raise
    finally:
        log.close()
    # The artifact validator must check plan success independently of process exit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
