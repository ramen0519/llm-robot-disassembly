"""Synthetic generic artifact checks. No test here executes CAD physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
import validate_assembly as module


POSE = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def fixture(base=None, parts=("10", "20", "Z"), sequence=None):
    sequence = list(parts[:-1]) if sequence is None else sequence
    total = max(len(parts) - 2, 0)
    stats = {"success": True, "sequence": sequence, "n_eval": total, "total_n_eval": total,
             "n_gripper": 2, "stop_msg": "solution found", "time": 0.5}
    tree = {"graph": {}, "directed": True, "nodes": [[list(parts), {"n_eval": 0, "n_gripper": 1, "poses": []}]],
            "edges": [], "adjacency": []}
    parent = list(parts)
    for index, part in enumerate(sequence, 1):
        child = [item for item in parent if item != part]
        n_eval = min(index, total)
        tree["nodes"].append([child, {"n_eval": n_eval, "n_gripper": 2, "poses": [deepcopy(POSE)]}])
        tree["edges"].append([parent, child, {"n_eval": n_eval, "sim_info": {
            "feasible": True, "part_move": part, "base_part": base, "action": [1.0, 0.0, 0.0],
            "parts_fix": [parts[-1]], "pose": deepcopy(POSE), "grasp": None}}])
        parent = child
    adjacency(tree)
    return stats, tree


def adjacency(tree):
    tree["adjacency"] = [[node, [child for parent, child, _ in tree["edges"] if parent == node]]
                         for node, _ in tree["nodes"]]


def unsuccessful_fixture(stop="budget reached"):
    stats, tree = fixture()
    tree["nodes"].pop()
    tree["edges"].pop()
    adjacency(tree)
    stats.update(success=False, sequence=None, n_eval=None, n_gripper=None, stop_msg=stop)
    return stats, tree


class NodeView(dict):
    def __call__(self, data=False):
        return list(self.items()) if data else list(self)


class EdgeView(dict):
    def __call__(self, data=False):
        return [(parent, child, attrs) for (parent, child), attrs in self.items()] if data else list(self)


class FixtureGraph:
    """Minimal graph protocol for an explicitly trusted synthetic pickle."""
    def __init__(self, tree):
        self.graph = tree["graph"]
        self.nodes = NodeView((tuple(node), attrs) for node, attrs in tree["nodes"])
        self.edges = EdgeView(((tuple(parent), tuple(child)), attrs) for parent, child, attrs in tree["edges"])
        self.adj = {node: {} for node in self.nodes}
        for (parent, child), attrs in self.edges.items():
            self.adj[parent][child] = attrs

    def is_directed(self):
        return True


class GenericAssemblyTests(unittest.TestCase):
    def validate(self, stats, tree, base=None, parts=("10", "20", "Z")):
        return module.validate_stats(stats, list(parts), base, 2, tree)

    def test_base_none_accepts_non_beam_ids_and_actual_singleton_terminal(self):
        stats, tree = fixture(sequence=["20", "10"])
        original = deepcopy((stats, tree))
        result = self.validate(stats, tree)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["validation_ok"])
        self.assertEqual(result["terminal_parts"], ["Z"])
        self.assertFalse(result["physics_replayed"])
        self.assertIsNone(result["metrics"]["verifier_calls"])
        self.assertEqual((stats, tree), original)

    def test_optional_anchor_must_remain_and_raw_base_must_match(self):
        stats, tree = fixture(base="Z")
        self.assertTrue(self.validate(stats, tree, "Z")["validation_ok"])
        self.assertFalse(self.validate(stats, tree, None)["validation_ok"])
        self.assertFalse(self.validate(stats, tree, "20")["validation_ok"])

    def test_two_part_leaf_can_succeed_at_zero_native_evaluations(self):
        stats, tree = fixture(parts=("A", "B"))
        result = self.validate(stats, tree, parts=("A", "B"))
        self.assertTrue(result["validation_ok"], result["issues"])
        self.assertEqual(result["metrics"]["upstream_total_n_eval"], 0)
        self.assertEqual(result["terminal_parts"], ["B"])

    def test_budget_timeout_and_explored_failures_are_valid_unknowns(self):
        for stop in ("budget reached", "timeout", "tree fully explored", "solution found"):
            with self.subTest(stop=stop):
                stats, tree = unsuccessful_fixture(stop)
                result = self.validate(stats, tree)
                self.assertEqual(result["status"], "UNKNOWN", result["issues"])
                self.assertTrue(result["validation_ok"])
                self.assertFalse(result["plan_success"])
                self.assertTrue(result["tree_checked"])
                self.assertIsNone(result["terminal_parts"])

    def test_exception_and_unknown_stop_are_not_normal_unknown(self):
        for stop in ("exception", "unrecognized", None):
            stats, tree = unsuccessful_fixture(stop)
            self.assertEqual(self.validate(stats, tree)["status"], "ERROR")
        stats, tree = unsuccessful_fixture("interrupt")
        result = self.validate(stats, tree)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["validation_ok"])

    def test_failed_stats_with_a_feasible_singleton_are_rejected(self):
        stats, tree = fixture()
        stats.update(success=False, sequence=None, n_eval=None, n_gripper=None, stop_msg="budget reached")
        self.assertFalse(self.validate(stats, tree)["validation_ok"])

    def test_broken_sequence_action_support_pose_and_missing_edge_are_rejected(self):
        mutations = [
            lambda s, t: s.update(sequence=["10", "10"]),
            lambda s, t: t["edges"][0][2]["sim_info"].update(feasible=False),
            lambda s, t: t["edges"][0][2]["sim_info"].update(action=[0, 0, 0]),
            lambda s, t: t["edges"][0][2]["sim_info"].update(parts_fix=["10"]),
            lambda s, t: t["edges"][0][2]["sim_info"].update(parts_fix=["20", "Z"]),
            lambda s, t: t["edges"][0][2]["sim_info"].update(pose=[[1, 0], [0, 1]]),
            lambda s, t: t["edges"].pop(),
        ]
        for mutate in mutations:
            stats, tree = fixture()
            mutate(stats, tree)
            self.assertEqual(self.validate(stats, tree)["status"], "ERROR")

    def test_native_budget_overshoot_is_not_relabelled_as_corruption(self):
        stats, tree = unsuccessful_fixture()
        stats["total_n_eval"] = 402
        stats["budget"] = 400  # Optional diagnostic only; the pose loop may overshoot.
        result = self.validate(stats, tree)
        self.assertTrue(result["validation_ok"])
        self.assertEqual(result["metrics"]["upstream_total_n_eval"], 402)

    def test_fingerprint_covers_nonselected_state_and_full_float_precision(self):
        stats, tree = fixture()
        original = self.validate(stats, tree)["tree_fingerprint_sha256"]
        changed = deepcopy(tree)
        changed["graph"]["unused_diagnostic"] = [0.1234567890123456]
        self.assertNotEqual(original, self.validate(stats, changed)["tree_fingerprint_sha256"])
        changed["graph"]["unused_diagnostic"][0] = 0.1234567890123457
        self.assertNotEqual(module.tree_fingerprint(tree), module.tree_fingerprint(changed))
        # Node/adjacency order matters even when the selected path remains valid.
        changed = deepcopy(tree)
        changed["nodes"][-2:] = reversed(changed["nodes"][-2:])
        adjacency(changed)
        result = self.validate(stats, changed)
        self.assertTrue(result["validation_ok"])
        self.assertNotEqual(original, result["tree_fingerprint_sha256"])

    def test_no_trust_does_not_open_pickle_and_valid_unknown_cli_exits_zero(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as temporary:
            root = Path(temporary)
            assembly, log = root / "assembly", root / "log"
            assembly.mkdir()
            log.mkdir()
            for part in ("10", "20", "Z"):
                (assembly / (part + ".obj")).write_text("# fixture\n", encoding="utf-8")
            stats, tree = unsuccessful_fixture()
            (log / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            (log / "tree.pkl").write_bytes(b"not a pickle")
            result = module.validate_assembly(log, assembly)
            self.assertEqual(result["status"], "UNKNOWN")
            self.assertFalse(result["validation_ok"])
            (log / "tree.pkl").write_bytes(pickle.dumps(FixtureGraph(tree)))
            output = root / "result.json"
            command = [sys.executable, "-B", str(ROOT / "worker/validate_assembly.py"), "--log-dir", str(log),
                       "--assembly-dir", str(assembly), "--output", str(output), "--trusted-local-tree"]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "tests")
            executed = subprocess.run(command, capture_output=True, text=True, env=environment)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            checked = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(checked["validation_ok"])
            self.assertFalse(checked["plan_success"])
            before = output.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True, env=environment).returncode, 0)
            self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
