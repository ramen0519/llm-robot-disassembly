"""Post-hoc progress fixtures only; none of these tests runs ASAP physics."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
import summarize_assembly_progress as module
from test_validate_assembly import FixtureGraph, adjacency, fixture, unsuccessful_fixture
from validate_assembly import validate_stats


def parsed(stats, tree, parts=("10", "20", "Z")):
    # A parsed_result read from disk is independent of the separately read stats.
    return deepcopy(validate_stats(stats, list(parts), None, 2, tree))


class AssemblyProgressTests(unittest.TestCase):
    def summarize(self, stats, tree, parts=("10", "20", "Z"), existing=None):
        existing = parsed(stats, tree, parts) if existing is None else existing
        return module.summarize_artifacts(stats, tree, list(parts), existing, None, 2)

    def test_success_and_unknown_prefix_are_separate(self):
        stats, tree = fixture()
        successful = self.summarize(stats, tree)
        self.assertEqual(successful["max_feasible_prefix_steps"], 2)
        self.assertTrue(successful["plan_success"])
        stats, tree = unsuccessful_fixture("timeout")
        original = deepcopy((stats, tree))
        partial = self.summarize(stats, tree)
        self.assertEqual(partial["status"], "PASS")
        self.assertEqual(partial["planner_status"], "UNKNOWN")
        self.assertFalse(partial["plan_success"])
        self.assertEqual(partial["max_feasible_prefix_steps"], 1)
        self.assertEqual(partial["full_plan_required_steps"], 2)
        self.assertEqual(partial["witness_removed_parts"], ["10"])
        self.assertEqual(partial["witness_remaining_parts"], ["20", "Z"])
        self.assertEqual(original, (stats, tree))

    def test_failed_first_edge_is_zero_progress_not_one_attempt(self):
        stats, tree = unsuccessful_fixture()
        tree["edges"][0][2]["sim_info"].update(feasible=False, parts_fix=None)
        tree["nodes"][1][1].update(n_gripper=None, poses=[])
        result = self.summarize(stats, tree)
        self.assertTrue(result["validation_ok"], result["issues"])
        self.assertEqual(result["max_feasible_prefix_steps"], 0)
        self.assertEqual(result["witness_removed_parts"], [])
        self.assertEqual(result["witness_remaining_parts"], ["10", "20", "Z"])

    def test_shared_child_does_not_make_failed_parent_edge_feasible(self):
        parts = ["A", "B", "C", "D"]
        a, b, shared = ["B", "C", "D"], ["A", "C", "D"], ["C", "D"]
        tree = {"graph": {}, "directed": True, "nodes": [
            [parts, {"n_eval": 0, "n_gripper": 1, "poses": []}],
            [a, {"n_eval": 1, "n_gripper": 2, "poses": [None]}],
            [shared, {"n_eval": 2, "n_gripper": 2, "poses": [None]}],
            [b, {"n_eval": 3, "n_gripper": 2, "poses": [None]}]], "edges": []}
        for parent, child, count, feasible in ((parts, a, 1, True), (a, shared, 2, False),
                                               (parts, b, 3, True), (b, shared, 4, True)):
            tree["edges"].append([parent, child, {"n_eval": count, "sim_info": {
                "feasible": feasible, "part_move": next(p for p in parent if p not in child),
                "base_part": None, "action": [1, 0, 0], "parts_fix": ["D"] if feasible else None,
                "pose": None, "grasp": None}}])
        adjacency(tree)
        stats = {"success": False, "sequence": None, "n_eval": None, "n_gripper": None,
                 "total_n_eval": 4, "stop_msg": "timeout", "time": 1.0}
        result = self.summarize(stats, tree, parts)
        self.assertTrue(result["validation_ok"], result["issues"])
        self.assertEqual(result["max_feasible_prefix_steps"], 2)
        self.assertEqual(result["witness_removed_parts"], ["B", "A"])
        self.assertEqual([edge["edge_n_eval"] for edge in result["witness_edges"]], [3, 4])

    def test_unreachable_feasible_metadata_is_not_counted(self):
        stats, tree = unsuccessful_fixture()
        tree["edges"][0][2]["sim_info"].update(feasible=False, parts_fix=None)
        # A non-null child alone never supplies a feasible root-to-child path.
        result = self.summarize(stats, tree)
        self.assertTrue(result["validation_ok"], result["issues"])
        self.assertEqual(result["max_feasible_prefix_steps"], 0)

    def test_raw_stats_hash_and_parsed_validation_cannot_go_stale(self):
        for mutate in (lambda s, t, p: s.update(time=9.0),
                       lambda s, t, p: t["graph"].update(extra="changed"),
                       lambda s, t, p: p.update(validation_ok=False),
                       lambda s, t, p: p.pop("tree_fingerprint_sha256")):
            stats, tree = unsuccessful_fixture()
            old = parsed(stats, tree)
            mutate(stats, tree, old)
            result = self.summarize(stats, tree, existing=old)
            self.assertEqual(result["status"], "ERROR")
            self.assertIsNone(result["max_feasible_prefix_steps"])
            self.assertIsNone(result["witness_removed_parts"])

    def test_exception_and_success_contradiction_have_null_metrics(self):
        for stop in ("exception", "interrupt"):
            stats, tree = unsuccessful_fixture(stop)
            result = self.summarize(stats, tree)
            self.assertEqual(result["status"], "ERROR")
            self.assertIsNone(result["max_feasible_prefix_steps"])
        stats, tree = fixture()
        stats.update(success=False, sequence=None, n_eval=None, n_gripper=None)
        self.assertIsNone(self.summarize(stats, tree)["max_feasible_prefix_steps"])

    def test_two_part_leaf_is_one_step_even_at_zero_native_count(self):
        stats, tree = fixture(parts=("A", "B"))
        result = self.summarize(stats, tree, ("A", "B"))
        self.assertEqual(stats["total_n_eval"], 0)
        self.assertEqual(result["max_feasible_prefix_steps"], 1)
        self.assertTrue(result["plan_success"])

    def test_cli_twelve_fixture_rows_hashes_and_exclusive_output(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as temporary:
            root = Path(temporary)
            protocol = {"assemblies": ["00561", "02628", "01289"], "variants": list(module.VARIANTS),
                        "seed": 0, "base_part": None, "max_gripper": 2}
            (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            stats, tree = unsuccessful_fixture("timeout")
            for assembly in protocol["assemblies"]:
                obj = root / "dense_assemblies" / assembly
                obj.mkdir(parents=True)
                for part in ("10", "20", "Z"):
                    (obj / (part + ".obj")).write_text("# synthetic fixture\n", encoding="utf-8")
                for variant in module.VARIANTS:
                    folder = root / "evaluations" / assembly / variant
                    native = folder / "native/dfs-heur-out/s0" / assembly
                    native.mkdir(parents=True)
                    (native / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
                    (native / "tree.pkl").write_bytes(pickle.dumps(FixtureGraph(tree)))
                    (folder / "parsed_result.json").write_text(json.dumps(parsed(stats, tree)), encoding="utf-8")
            before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file()}
            command = [sys.executable, "-B", str(ROOT / "worker/summarize_assembly_progress.py"),
                       "--run-dir", str(root), "--trusted-local-tree"]
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "tests"))
            executed = subprocess.run(command, capture_output=True, text=True, env=environment)
            self.assertEqual(executed.returncode, 0, executed.stderr + executed.stdout)
            output = root / "partial_progress.json"
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(result["rows"]), 12)
            self.assertEqual(result["expected_row_count"], 12)
            self.assertTrue(result["post_hoc"])
            self.assertFalse(result["physics_replayed"])
            self.assertEqual(result["new_verifier_calls"], 0)
            self.assertTrue(all(row["source_files_unchanged"] for row in result["rows"]))
            self.assertTrue(all(row["max_feasible_prefix_steps"] == 1 for row in result["rows"]))
            self.assertEqual(result["protocol_sha256"], before["protocol.json"])
            for relative, digest in before.items():
                self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), digest)
            saved = output.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True, env=environment).returncode, 0)
            self.assertEqual(output.read_bytes(), saved)

    def test_trust_is_required_before_reading_run_or_pickle(self):
        with self.assertRaisesRegex(ValueError, "trusted-local-tree"):
            module.summarize_run(Path("nonexistent_fixture"))

    def test_parent_traversal_is_rejected_before_any_lstat(self):
        # These are strings only; every lstat is mocked and must remain uncalled.
        for value in ("/mnt/c/../e/private", "C:/dev/../private", "safe\\..\\private"):
            with self.subTest(path=value):
                with patch.object(module.Path, "lstat", side_effect=AssertionError("unexpected filesystem access")) as access:
                    with self.assertRaisesRegex(ValueError, "parent traversal"):
                        module._safe_path(value)
                    access.assert_not_called()


if __name__ == "__main__":
    unittest.main()
