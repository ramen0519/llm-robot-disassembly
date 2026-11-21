"""Post-hoc feasible-prefix summary; no planning or physics replay (Python 3.8).

Only explicitly trusted local tree pickles are read. The recorded prefix is a
path in ASAP's final part-set graph, not a continuously replayed physical motion.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat

from compare_rank_hook import read, tree_data
from validate_assembly import load_part_ids, validate_stats


VARIANTS = ("native", "trace", "id_ascending", "reverse_native")
SOURCE_FILES = ("summarize_assembly_progress.py", "validate_assembly.py",
                "validate_baseline.py", "compare_rank_hook.py", "rank_hook.py",
                "rank_contract.py")


def _safe_path(path):
    """Reject links/reparse points before opening any input or output target."""
    # Path.absolute() does not remove '..'; reject traversal before absolute()
    # or lstat() so an allowed-looking prefix cannot hide an excluded target.
    if ".." in str(path).replace("\\", "/").split("/"):
        raise ValueError("parent traversal path components are not allowed")
    path = Path(path).absolute()
    lowered = str(path).lower().replace("\\", "/")
    if lowered.startswith("e:") or lowered == "/mnt/e" or lowered.startswith("/mnt/e/"):
        raise ValueError("E drive is excluded")
    for entry in reversed((path,) + tuple(path.parents)):
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("links/reparse points are not allowed: {}".format(entry))
    return path


def _hash(path):
    return hashlib.sha256(_safe_path(path).read_bytes()).hexdigest()


def _component(value):
    return (type(value) is str and bool(value) and value not in (".", "..")
            and all(char.isascii() and (char.isalnum() or char in "_-") for char in value))


def summarize_artifacts(stats, tree, part_ids, parsed, base_part=None, max_gripper=2):
    """Revalidate immutable artifact dictionaries, then follow true edges only."""
    checked = validate_stats(stats, part_ids, base_part, max_gripper, tree)
    result = {
        "status": "ERROR", "validation_ok": False,
        "planner_status": checked["status"], "plan_success": None,
        "max_feasible_prefix_steps": None, "full_plan_required_steps": None,
        "witness_removed_parts": None, "witness_remaining_parts": None,
        "witness_edges": None, "tree_fingerprint_sha256": checked["tree_fingerprint_sha256"],
        "tree_fingerprint_format": "ordered_tree_json_v1",
        "issues": list(checked["issues"]),
    }
    issues = result["issues"]
    if not isinstance(parsed, dict):
        issues.append("parsed_result must be an object")
    else:
        # Includes raw_stats, validation_ok, status, counts and ordered-tree hash.
        for key, value in checked.items():
            if key not in parsed or parsed[key] != value:
                issues.append("parsed_result differs from fresh validation: " + key)
    if not checked["validation_ok"] or issues:
        return result

    root = tuple(part_ids)
    successors = {tuple(node): [] for node, _ in tree["nodes"]}
    for parent, child, attributes in tree["edges"]:
        if attributes["sim_info"]["feasible"] is True:
            successors[tuple(parent)].append((tuple(child), attributes))
    # Existing validation guarantees one-part removal on every edge, hence a DAG.
    # First discovery in recorded edge order is the deterministic witness tie-break.
    paths, pending, deepest = {root: []}, deque([root]), root
    while pending:
        parent = pending.popleft()
        for child, attributes in successors[parent]:
            if child in paths:
                continue
            paths[child] = paths[parent] + [(parent, child, attributes)]
            pending.append(child)
            if len(child) < len(deepest):
                deepest = child
    steps, required = len(root) - len(deepest), len(root) - 1
    if checked["plan_success"] != (steps == required):
        issues.append("full feasible-prefix depth disagrees with plan_success")
        return result
    witness = []
    for parent, child, attributes in paths[deepest]:
        raw = attributes["sim_info"]
        witness.append({"parent_parts": list(parent), "remaining_parts": list(child),
                        "part_move": raw["part_move"], "parts_fix": raw["parts_fix"],
                        "action": raw["action"], "pose": raw["pose"],
                        "edge_n_eval": attributes["n_eval"]})
    result.update(status="PASS", validation_ok=True, plan_success=checked["plan_success"],
                  max_feasible_prefix_steps=steps, full_plan_required_steps=required,
                  witness_removed_parts=[edge["part_move"] for edge in witness],
                  witness_remaining_parts=list(deepest), witness_edges=witness)
    return result


def summarize_run(root, trusted_local_tree=False):
    """Read the current assembly experiment layout, preserving every source byte."""
    if not trusted_local_tree:
        raise ValueError("explicit --trusted-local-tree is required to load pickles")
    root = _safe_path(root)
    protocol_path = _safe_path(root / "protocol.json")
    protocol_hash = _hash(protocol_path)
    protocol = read(protocol_path)
    assemblies = protocol.get("assemblies")
    if (not isinstance(assemblies, list) or not 3 <= len(assemblies) <= 5
            or not all(_component(item) for item in assemblies)
            or len(assemblies) != len(set(assemblies))
            or protocol.get("variants") != list(VARIANTS)
            or type(protocol.get("seed")) is not int or protocol["seed"] < 0):
        raise ValueError("invalid assembly/variant/seed protocol")
    worker = Path(__file__).absolute().parent
    sources_before = {name: _hash(worker / name) for name in SOURCE_FILES}
    rows = []
    for assembly in assemblies:
        assembly_dir = _safe_path(root / "dense_assemblies" / assembly)
        for variant in VARIANTS:
            row = {"assembly": assembly, "variant": variant, "seed": protocol["seed"],
                   "status": "ERROR", "validation_ok": False, "planner_status": None,
                   "plan_success": None, "max_feasible_prefix_steps": None,
                   "full_plan_required_steps": None, "witness_removed_parts": None,
                   "witness_remaining_parts": None, "witness_edges": None,
                   "tree_fingerprint_sha256": None, "source_file_sha256": {}, "issues": []}
            try:
                folder = root / "evaluations" / assembly / variant
                native = folder / "native" / "dfs-heur-out" / ("s" + str(protocol["seed"])) / assembly
                files = {"stats.json": native / "stats.json", "tree.pkl": native / "tree.pkl",
                         "parsed_result.json": folder / "parsed_result.json"}
                # Reject OBJ links before load_part_ids can follow their targets.
                for candidate in assembly_dir.iterdir():
                    if candidate.name.endswith(".obj"):
                        _safe_path(candidate)
                parts = load_part_ids(assembly_dir)
                files.update({"assembly/" + part + ".obj": assembly_dir / (part + ".obj") for part in parts})
                before = {name: _hash(path) for name, path in files.items()}
                stats = read(files["stats.json"])
                parsed = read(files["parsed_result.json"])
                normalized = tree_data(native)
                row.update(summarize_artifacts(stats, normalized, parts, parsed,
                                              protocol.get("base_part"), protocol.get("max_gripper")))
                after = {name: _hash(path) for name, path in files.items()}
                row["source_file_sha256"] = before
                row["source_files_unchanged"] = before == after
                if before != after:
                    raise ValueError("source artifacts changed while summarizing")
            except Exception as error:
                row.update(status="ERROR", validation_ok=False, plan_success=None,
                           max_feasible_prefix_steps=None, full_plan_required_steps=None,
                           witness_removed_parts=None, witness_remaining_parts=None, witness_edges=None)
                row["issues"].append("{}: {}".format(type(error).__name__, error))
            rows.append(row)
    source_unchanged = (sources_before == {name: _hash(worker / name) for name in SOURCE_FILES}
                        and protocol_hash == _hash(protocol_path))
    return {
        "schema_version": 1, "status": "PASS" if source_unchanged and all(row["validation_ok"] for row in rows) else "ERROR",
        "post_hoc": True, "physics_replayed": False, "new_verifier_calls": 0,
        "scope": "root-reachable feasible prefix in the recorded final ASAP part-set graph",
        "interpretation": "Witness edges may come from different search branches and poses; continuous physical replay is not established.",
        "metric_is_candidate_attempt_count": False,
        "metric_is_full_plan_success": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(root),
        "assemblies": assemblies, "variants": list(VARIANTS), "seed": protocol["seed"],
        "expected_row_count": len(assemblies) * len(VARIANTS), "rows": rows,
        "protocol_sha256": protocol_hash, "analysis_source_sha256": sources_before,
        "analysis_sources_and_protocol_unchanged": source_unchanged,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trusted-local-tree", action="store_true")
    args = parser.parse_args()
    if not args.trusted_local_tree:
        parser.error("explicit --trusted-local-tree is required")
    root = _safe_path(args.run_dir)
    output = _safe_path(args.output or root / "partial_progress.json")
    if output.parent != root or output.name != "partial_progress.json":
        parser.error("output must be the run directory's new partial_progress.json")
    # Reserve exclusively before reading pickles; an existing result is never overwritten.
    with output.open("x", encoding="utf-8") as stream:
        try:
            result = summarize_run(root, trusted_local_tree=True)
        except Exception as error:
            result = {"status": "ERROR", "post_hoc": True, "physics_replayed": False,
                      "rows": [], "issues": ["{}: {}".format(type(error).__name__, error)]}
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "output": str(output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
