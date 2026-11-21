"""Synthetic artifact-contract tests. These are not ASAP physics validation."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "worker/validate_baseline.py"
spec = importlib.util.spec_from_file_location("validate_baseline", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def synthetic_artifacts():
    stats = {"success": True, "sequence": ["0", "1", "2", "3"],
             "n_eval": 3, "total_n_eval": 3, "n_gripper": 2,
             "stop_msg": "solution found", "extra_raw": {"preserved": True}}
    parent = module.PARTS
    tree = SimpleNamespace(nodes={parent: {"n_gripper": 1}}, edges={})
    for part in stats["sequence"]:
        child = tuple(p for p in parent if p != part)
        tree.nodes[child] = {"n_gripper": 2}
        tree.edges[parent, child] = {"sim_info": {
            "feasible": True, "part_move": part, "base_part": "6",
            "action": [1.0, 0.0, 0.0], "parts_fix": ["6"], "pose": None}}
        parent = child
    return stats, tree


class BaselineContractTests(unittest.TestCase):
    def test_valid_contract_preserves_raw_and_unknown_metrics(self):
        stats, tree = synthetic_artifacts()
        original = copy.deepcopy(stats)
        result = module.validate_stats(stats, tree)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["plan_success"])
        self.assertEqual(result["terminal_parts"], ["6"])
        self.assertEqual(result["raw_stats"], original)
        self.assertEqual(stats, original)
        self.assertFalse(result["physics_replayed"])
        self.assertIsNone(result["metrics"]["verifier_calls"])
        self.assertIsNone(result["metrics"]["internal_geometry_operations"])

    def test_unchecked_tree_and_no_plan_remain_unknown(self):
        stats, _ = synthetic_artifacts()
        result = module.validate_stats(stats)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["plan_success"])
        stats.update(success=False, sequence=None, stop_msg="budget reached")
        result = module.validate_stats(stats)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["plan_success"])

    def test_failures_are_not_success(self):
        cases = [
            {"success": "true"},
            {"sequence": ["0", "1", "2", "2"]},
            {"sequence": ["0", "1", "2", "6"]},
            {"sequence": [0, 1, 2, 3]},
            {"stop_msg": "exception"},
            {"stop_msg": None},
        ]
        for patch in cases:
            with self.subTest(patch=patch):
                stats, tree = synthetic_artifacts()
                stats.update(patch)
                self.assertEqual(module.validate_stats(stats, tree)["status"], "ERROR")

    def test_invalid_edge_action_and_support_are_rejected(self):
        patches = [
            {"feasible": False}, {"part_move": "1"}, {"base_part": "0"},
            {"action": None}, {"action": [0, 0, 0]},
            {"action": [float("nan"), 0, 0]}, {"action": [1, 0]},
            {"action": [True, 0, 0]}, {"parts_fix": None},
            {"parts_fix": ["0"]}, {"parts_fix": ["6", "6"]},
            {"parts_fix": ["1", "6"]},
        ]
        for patch in patches:
            with self.subTest(patch=patch):
                stats, tree = synthetic_artifacts()
                next(iter(tree.edges.values()))["sim_info"].update(patch)
                result = module.validate_stats(stats, tree)
                self.assertEqual(result["status"], "ERROR")
                self.assertIsNone(result["plan_success"])

    def test_missing_edge_and_invalid_terminal_node(self):
        stats, tree = synthetic_artifacts()
        del tree.edges[next(iter(tree.edges))]
        self.assertEqual(module.validate_stats(stats, tree)["status"], "ERROR")
        stats, tree = synthetic_artifacts()
        tree.nodes[("6",)]["n_gripper"] = None
        self.assertEqual(module.validate_stats(stats, tree)["status"], "ERROR")

    def test_no_trust_never_reads_corrupt_pickle_and_trust_loads_local_tree(self):
        stats, tree = synthetic_artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            (root / "tree.pkl").write_bytes(b"invalid pickle")
            self.assertEqual(module.validate_baseline(root)["status"], "UNKNOWN")
            self.assertEqual(module.validate_baseline(root, True)["status"], "ERROR")
            (root / "tree.pkl").write_bytes(pickle.dumps(tree))
            self.assertEqual(module.validate_baseline(root, True)["status"], "PASS")

    def test_cli_records_errors_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "summary.json"
            command = [sys.executable, str(SCRIPT), "--log-dir", str(root),
                       "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(output.read_text())["status"], "ERROR")
            original = output.read_bytes()
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
