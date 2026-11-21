"""Check official beam planning artifacts; this does not rerun physics.

Pickle can execute code. Use --trusted-local-tree only for tree.pkl produced
by your own local ASAP run. Synthetic tests verify this artifact contract only.
"""
import argparse
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional


PARTS = ("0", "1", "2", "3", "6")
REMOVABLE = {"0", "1", "2", "3"}


def _action_valid(action: Any) -> bool:
    try:
        values = list(action)
        return (len(values) == 3
                and all(isinstance(x, Real) and not isinstance(x, bool)
                        and math.isfinite(float(x)) for x in values)
                and any(float(x) != 0 for x in values))
    except (TypeError, ValueError, OverflowError):
        return False


def _tree_errors(tree: Any, sequence: List[str]) -> List[str]:
    """Read the selected path only; never mutate the upstream search state."""
    errors = []
    parent = PARTS
    for index, part in enumerate(sequence):
        child = tuple(p for p in parent if p != part)
        try:
            info = tree.edges[parent, child]["sim_info"]
            if info.get("feasible") is not True:
                errors.append("step {}: edge is not feasible".format(index))
            if info.get("part_move") != part:
                errors.append("step {}: part_move mismatch".format(index))
            if info.get("base_part") != "6":
                errors.append("step {}: base_part is not 6".format(index))
            if not _action_valid(info.get("action")):
                errors.append("step {}: invalid action".format(index))
            support = info.get("parts_fix")
            if (not isinstance(support, list)
                    or not all(isinstance(p, str) for p in support)
                    or len(set(support)) != len(support)
                    or not set(support).issubset(child)
                    or len(support) > 1):
                errors.append("step {}: invalid parts_fix".format(index))
            for node in (parent, child):
                grippers = tree.nodes[node].get("n_gripper")
                if (not isinstance(grippers, Real) or isinstance(grippers, bool)
                        or not math.isfinite(float(grippers))
                        or grippers not in (1, 2)):
                    errors.append("step {}: invalid node n_gripper".format(index))
        except (KeyError, TypeError, AttributeError):
            errors.append("step {}: missing or malformed tree edge/node".format(index))
        parent = child
    if parent != ("6",):
        errors.append("terminal node is not (6,)")
    return errors


def validate_stats(stats: Any, tree: Optional[Any] = None) -> Dict[str, Any]:
    """Validate raw stats and, when provided, an already trusted local tree."""
    result = {
        "schema_version": 1,
        "scope": "official_beam_original_dfs_heur_out_artifact_contract",
        "status": "UNKNOWN", "plan_success": None,
        "tree_checked": False, "physics_replayed": False,
        "terminal_parts": None, "issues": [], "raw_stats": stats,
        "metrics": {
            "upstream_n_eval": None, "upstream_total_n_eval": None,
            "verifier_calls": None, "internal_geometry_operations": None,
        },
    }
    if not isinstance(stats, dict) or type(stats.get("success")) is not bool:
        result["status"] = "ERROR"
        result["issues"].append("stats.success must be a boolean")
        return result
    result["metrics"]["upstream_n_eval"] = stats.get("n_eval")
    result["metrics"]["upstream_total_n_eval"] = stats.get("total_n_eval")
    if stats.get("stop_msg") == "exception":
        result["status"] = "ERROR"
        result["issues"].append("upstream planner reported exception")
        return result
    if stats.get("stop_msg") == "interrupt":
        result["issues"].append("upstream planner was interrupted")
        return result
    if stats["success"] is False:
        result["plan_success"] = False
        result["issues"].append("upstream found no complete plan; impossibility is not established")
        return result
    sequence = stats.get("sequence")
    if (not isinstance(sequence, list) or len(sequence) != 4
            or not all(isinstance(part, str) for part in sequence)
            or set(sequence) != REMOVABLE):
        result["status"] = "ERROR"
        result["issues"].append("sequence must be a permutation of 0, 1, 2, 3")
        return result
    if stats.get("stop_msg") not in ("solution found", "tree fully explored", "budget reached", "timeout"):
        result["status"] = "ERROR"
        result["issues"].append("missing or unsupported upstream stop_msg")
        return result
    if tree is None:
        result["issues"].append("tree was not checked; explicit local pickle trust is required")
        return result
    result["tree_checked"] = True
    result["issues"].extend(_tree_errors(tree, sequence))
    if result["issues"]:
        result["status"] = "ERROR"
    else:
        result["status"] = "PASS"
        result["plan_success"] = True
        result["terminal_parts"] = ["6"]
    return result


def validate_baseline(log_dir: Path, trusted_local_tree: bool = False) -> Dict[str, Any]:
    """Read stats with stdlib; load pickle dependencies only after explicit trust."""
    stats = None
    try:
        with (log_dir / "stats.json").open(encoding="utf-8") as stream:
            stats = json.load(stream)
        tree = None
        if trusted_local_tree:
            import pickle
            # NetworkX is imported by pickle for the upstream DiGraph class.
            with (log_dir / "tree.pkl").open("rb") as stream:
                tree = pickle.load(stream)
        result = validate_stats(stats, tree)
    except Exception as error:
        result = validate_stats(stats)
        result["status"] = "ERROR"
        result["plan_success"] = None
        result["issues"].append("{}: {}".format(type(error).__name__, error))
    result["log_dir"] = str(log_dir.resolve())
    result["trusted_local_tree"] = trusted_local_tree
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON summary file; existing files are never overwritten")
    parser.add_argument("--trusted-local-tree", action="store_true")
    args = parser.parse_args()
    # Reserve before reading pickle; duplicate output paths fail without loading it.
    with args.output.open("x", encoding="utf-8") as stream:
        result = validate_baseline(args.log_dir, args.trusted_local_tree)
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
