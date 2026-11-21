"""Validate generic official ASAP artifacts, without replaying physics (3.8).

The tree argument is the ordered JSON-compatible representation returned by
compare_rank_hook.tree_data. Normal bounded failure is UNKNOWN with a valid
artifact contract; it is never evidence of impossibility.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from compare_rank_hook import read, tree_data
from validate_baseline import _action_valid


NORMAL_STOPS = ("solution found", "budget reached", "tree fully explored", "timeout")


def load_part_ids(assembly_dir):
    """Match pinned assets.load.load_part_ids without importing geometry tools."""
    parts = sorted(path.name.replace(".obj", "") for path in assembly_dir.iterdir()
                   if path.name.endswith(".obj") and path.is_file())
    if len(parts) < 2 or len(parts) != len(set(parts)) or not all(parts):
        raise ValueError("assembly must contain at least two uniquely named OBJ parts")
    return parts


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pose_valid(pose):
    return pose is None or (isinstance(pose, list) and len(pose) == 4
        and all(isinstance(row, list) and len(row) == 4 and all(_finite(value) for value in row)
                for row in pose))


def tree_fingerprint(tree):
    encoded = json.dumps(tree, sort_keys=True, separators=(",", ":"), allow_nan=False,
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_contract(tree, part_ids, base_part, max_gripper, total_n_eval):
    issues = []
    if (not isinstance(tree, dict) or tree.get("directed") is not True
            or not isinstance(tree.get("graph"), dict)
            or not isinstance(tree.get("nodes"), list) or not isinstance(tree.get("edges"), list)
            or not isinstance(tree.get("adjacency"), list)):
        raise ValueError("invalid ordered directed-tree representation")
    initial = tuple(part_ids)
    nodes, edges = {}, {}
    for entry in tree["nodes"]:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("invalid node entry")
        raw_node, attributes = entry
        if (not isinstance(raw_node, list) or not raw_node or not isinstance(attributes, dict)
                or any(type(part) is not str for part in raw_node)
                or raw_node != sorted(set(raw_node)) or not set(raw_node).issubset(part_ids)):
            raise ValueError("invalid node part membership")
        node = tuple(raw_node)
        if node in nodes:
            raise ValueError("duplicate node")
        nodes[node] = attributes
        if (not _integer(attributes.get("n_eval")) or attributes["n_eval"] > total_n_eval
                or (attributes.get("n_gripper") is not None
                    and (not _integer(attributes["n_gripper"], 1) or attributes["n_gripper"] > max_gripper))
                or not isinstance(attributes.get("poses"), list)
                or not all(_pose_valid(pose) for pose in attributes["poses"])):
            issues.append("invalid node evaluation/gripper/pose attributes")
        if base_part is not None and base_part not in node:
            issues.append("anchored part is missing from a node")
    if initial not in nodes or not nodes:
        raise ValueError("initial assembly node is missing")
    if nodes[initial].get("n_eval") != 0 or nodes[initial].get("n_gripper") != 1:
        issues.append("initial node metadata differs from upstream contract")
    successors = {node: [] for node in nodes}
    for entry in tree["edges"]:
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError("invalid edge entry")
        parent, child, attributes = entry
        parent, child = tuple(parent), tuple(child)
        if (parent not in nodes or child not in nodes or (parent, child) in edges
                or len(parent) != len(child) + 1 or not set(child).issubset(parent)
                or not isinstance(attributes, dict)):
            raise ValueError("edge must remove exactly one existing part")
        edges[parent, child] = attributes
        successors[parent].append(child)
        if not _integer(attributes.get("n_eval")) or attributes["n_eval"] > total_n_eval:
            issues.append("invalid edge evaluation index")
        raw = attributes.get("sim_info")
        removed = next(part for part in parent if part not in child)
        if (not isinstance(raw, dict) or type(raw.get("feasible")) is not bool
                or raw.get("part_move") != removed or raw.get("base_part") != base_part
                or "pose" not in raw or not _pose_valid(raw["pose"])):
            issues.append("raw edge result or request identity is invalid")
            continue
        if raw["feasible"]:
            supports = raw.get("parts_fix")
            # Pinned _expand_leaf uses max_grippers=2 independently of main setup.
            allowed = 2 if len(child) == 1 else max_gripper
            if (not _action_valid(raw.get("action")) or not isinstance(supports, list)
                    or any(type(part) is not str for part in supports)
                    or len(supports) != len(set(supports)) or not set(supports).issubset(child)
                    or len(supports) + 1 > allowed):
                issues.append("feasible edge lacks a valid action/support")
            if nodes[parent].get("n_gripper") is None or nodes[child].get("n_gripper") is None:
                issues.append("feasible edge has an infeasible endpoint")
    actual_adjacency = [[list(node), [list(child) for child in successors[node]]] for node in nodes]
    if tree["adjacency"] != actual_adjacency:
        issues.append("ordered adjacency disagrees with ordered edges")
    reached, pending = set(), [initial]
    while pending:
        node = pending.pop()
        if node in reached:
            continue
        reached.add(node)
        pending.extend(successors[node])
    if reached != set(nodes):
        issues.append("tree contains nodes disconnected from the initial assembly")
    return issues, nodes, edges


def validate_stats(stats, part_ids, base_part=None, max_gripper=2, tree=None):
    """Inspect normalized tree and stats; no solver, search or physics mutation."""
    result = {
        "schema_version": 1, "scope": "official_asap_full_plan_artifact_contract",
        "status": "UNKNOWN", "validation_ok": False, "plan_success": None,
        "tree_checked": False, "physics_replayed": False, "terminal_parts": None,
        "part_ids": part_ids, "base_part": base_part, "max_gripper": max_gripper,
        "issues": [], "raw_stats": stats, "outcome_reason": None,
        "tree_fingerprint_sha256": None, "tree_fingerprint_format": "ordered_tree_json_v1",
        "tree_node_count": None, "tree_edge_count": None,
        "metrics": {"upstream_n_eval": None, "upstream_total_n_eval": None,
                    "verifier_calls": None, "internal_geometry_operations": None},
    }
    issues = result["issues"]
    try:
        if (not isinstance(part_ids, (list, tuple)) or len(part_ids) < 2
                or any(type(part) is not str or not part for part in part_ids)
                or list(part_ids) != sorted(set(part_ids))
                or (base_part is not None and base_part not in part_ids)
                or not _integer(max_gripper, 2)):
            raise ValueError("invalid sorted part IDs, base, or supported gripper limit (>=2)")
        if not isinstance(stats, dict) or type(stats.get("success")) is not bool:
            raise ValueError("stats.success must be a boolean")
        result["metrics"].update(upstream_n_eval=stats.get("n_eval"),
                                  upstream_total_n_eval=stats.get("total_n_eval"))
        result["outcome_reason"] = stats.get("stop_msg")
        if stats.get("stop_msg") == "interrupt":
            issues.append("interrupted run does not establish a complete artifact contract")
            return result
        if stats.get("stop_msg") not in NORMAL_STOPS:
            raise ValueError("exception or unsupported planner stop")
        if not _integer(stats.get("total_n_eval")):
            raise ValueError("invalid total_n_eval")
        # Budget is tested outside the upstream pose loop; do not cap this count.
        if tree is None:
            result["plan_success"] = False if stats["success"] is False else None
            issues.append("trusted local tree was not checked")
            return result
        result["tree_fingerprint_sha256"] = tree_fingerprint(tree)
        result["tree_checked"] = True
        found, nodes, edges = _tree_contract(tree, part_ids, base_part, max_gripper, stats["total_n_eval"])
        issues.extend(found)
        result["tree_node_count"], result["tree_edge_count"] = len(nodes), len(edges)
        success_leaves = {node: info for node, info in nodes.items()
                          if len(node) == 1 and info.get("n_gripper") is not None}
        if stats["success"] != bool(success_leaves):
            issues.append("stats success and feasible singleton nodes disagree")
        expected_eval = min(info["n_eval"] for info in success_leaves.values()) if success_leaves else None
        expected_gripper = min(info["n_gripper"] for info in success_leaves.values()) if success_leaves else None
        if stats.get("n_eval") != expected_eval or stats.get("n_gripper") != expected_gripper:
            issues.append("stats n_eval/n_gripper differ from upstream leaf aggregation")
        sequence = stats.get("sequence")
        if stats["success"]:
            if (not isinstance(sequence, list) or len(sequence) != len(part_ids) - 1
                    or any(type(part) is not str for part in sequence)
                    or len(sequence) != len(set(sequence)) or not set(sequence).issubset(part_ids)
                    or (base_part is not None and base_part in sequence)):
                issues.append("sequence must remove N-1 unique non-base initial parts")
            else:
                parent = tuple(part_ids)
                for part in sequence:
                    child = tuple(item for item in parent if item != part)
                    edge = edges.get((parent, child), {})
                    if edge.get("sim_info", {}).get("feasible") is not True:
                        issues.append("reported sequence lacks a connected feasible edge")
                    parent = child
                if len(parent) != 1 or parent not in success_leaves or (base_part is not None and parent != (base_part,)):
                    issues.append("reported sequence terminal is not a valid singleton")
                else:
                    result["terminal_parts"] = list(parent)
        elif sequence is not None:
            issues.append("unsuccessful stats must retain a null sequence")
        if issues:
            result["status"] = "ERROR"
            result["terminal_parts"] = None
        else:
            result["validation_ok"] = True
            result["plan_success"] = stats["success"]
            result["status"] = "PASS" if stats["success"] else "UNKNOWN"
    except Exception as error:
        result["status"], result["plan_success"], result["validation_ok"] = "ERROR", None, False
        issues.append("{}: {}".format(type(error).__name__, error))
    return result


def validate_assembly(log_dir, assembly_dir, base_part=None, max_gripper=2, trusted_local_tree=False):
    stats, parts = None, []
    try:
        parts = load_part_ids(assembly_dir)
        stats = read(log_dir / "stats.json")
        normalized = tree_data(log_dir) if trusted_local_tree else None
        result = validate_stats(stats, parts, base_part, max_gripper, normalized)
    except Exception as error:
        result = validate_stats(stats, parts, base_part, max_gripper)
        result.update(status="ERROR", validation_ok=False, plan_success=None)
        result["issues"].append("{}: {}".format(type(error).__name__, error))
    result.update(log_dir=str(log_dir), assembly_dir=str(assembly_dir),
                  trusted_local_tree=trusted_local_tree)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--assembly-dir", type=Path, required=True)
    parser.add_argument("--base-part")
    parser.add_argument("--max-gripper", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-local-tree", action="store_true")
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        result = validate_assembly(args.log_dir, args.assembly_dir, args.base_part,
                                   args.max_gripper, args.trusted_local_tree)
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "validation_ok": result["validation_ok"],
                      "plan_success": result["plan_success"], "output": str(args.output)}))
    return 0 if result["validation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
