"""Synthetic observer contract tests. These fakes are not ASAP physics tests."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "worker"))
from path_evidence import FRAME, collect_endpoint, observed_collision_thresholds


class FakeSimulation:
    def __init__(self, distances=(0.2, 0.3)):
        self.distances = distances
        self.q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.calls = []
        self.contacts = ["partbase"]
        self.contact_override = None

    def get_joint_qm(self, name):
        self.calls.append(("q", name))
        return self.q[:]

    def get_body_distance(self, left, right):
        self.calls.append(("distance", left, right))
        return self.distances[0 if left == "partbase" else 1]

    def body_in_contact(self, left, right, eps):
        self.calls.append(("contact", left, right, eps))
        if self.contact_override is not None:
            return self.contact_override
        return min(self.get_body_distance(left, right),
                   self.get_body_distance(right, left)) < eps

    def get_contact_bodies(self, name):
        self.calls.append(("history", name))
        return self.contacts[:]

    def forward(self, *args, **kwargs):
        raise AssertionError("observer must not forward")

    def reset(self, *args, **kwargs):
        raise AssertionError("observer must not reset")

    def set_joint_qm(self, *args, **kwargs):
        raise AssertionError("observer must not reposition")


class FakePlanner:
    min_sep = 0.5

    def __init__(self, distances=(0.2, 0.3)):
        self.sim = FakeSimulation(distances)
        self.part_move = "arm"
        self.parts_fix = ["base"]
        self.parts_removed = []
        self.native_result = {"arm": 0.02, "base": 0.01}
        self.constructor_result = self._compute_col_th_dict(self.sim, self.parts_fix, self.part_move)

    def _compute_col_th_dict(self, sim, rest, move):
        return self.native_result


class PathEvidenceTests(unittest.TestCase):
    def planner(self, distances=(0.2, 0.3)):
        with observed_collision_thresholds(FakePlanner):
            return FakePlanner(distances)

    def collect(self, planner=None, **changes):
        planner = self.planner() if planner is None else planner
        arguments = {"raw_success": False, "path": [planner.sim.q[:]],
                     "scope": {"part_move": "arm", "parts_rest": ["base"],
                               "physics": {}, "pose": None},
                     "event_id": "run:path:1", "trajectory_ref": "paths/path1.json",
                     "trajectory_sha256": "a" * 64}
        arguments.update(changes)
        return collect_endpoint(planner, **arguments)

    def test_constructor_wrapper_preserves_raw_return_object_and_restores_method(self):
        original = FakePlanner._compute_col_th_dict
        with observed_collision_thresholds(FakePlanner):
            planner = FakePlanner()
            self.assertIs(planner.constructor_result, planner.native_result)
            self.assertIsNot(planner._evidence_col_th, planner.native_result)
            planner.native_result["arm"] = 9
            self.assertEqual(planner._evidence_col_th["arm"], 0.02)
        self.assertIs(FakePlanner._compute_col_th_dict, original)

    def test_module_input_and_exception_restore(self):
        original = FakePlanner._compute_col_th_dict
        with self.assertRaises(RuntimeError):
            with observed_collision_thresholds(SimpleNamespace(MultiPartPathPlanner=FakePlanner)):
                raise RuntimeError("fixture error")
        self.assertIs(FakePlanner._compute_col_th_dict, original)

    def test_positive_distance_can_violate_separation_without_collision_claim(self):
        planner = self.planner()
        result = self.collect(planner)
        pair = result["measurements"]["pairs"][0]
        self.assertEqual(pair["min_distance"], 0.2)
        self.assertTrue(pair["separation_unsatisfied"])
        self.assertIsNone(result["blocker"])
        self.assertEqual(result["termination_reason"], "unknown")
        self.assertEqual(result["measurements"]["contact_history"], ["partbase"])
        self.assertEqual(result["measurements"]["geometry_queries"], 4)
        self.assertEqual(len([c for c in planner.sim.calls if c[0] == "distance"]), 4)
        self.assertEqual(result["trajectory"]["frame"], FRAME)
        self.assertEqual(result["trajectory"]["states"], 1)

    def test_scope_and_simulation_are_not_mutated(self):
        planner = self.planner()
        scope = {"physics": {}, "nested": {"seed": 3}, "frame": FRAME}
        before = deepcopy(scope)
        q_before = planner.sim.q[:]
        self.collect(planner, scope=scope)
        self.assertEqual(scope, before)
        self.assertEqual(planner.sim.q, q_before)
        self.assertEqual(planner.parts_fix, ["base"])

    def test_dfs_removed_history_is_separate_from_path_simulator_obstacles(self):
        scope = {"parts_removed": ["previously_removed"], "physics": {}}
        result = self.collect(scope=scope)
        self.assertEqual(result["scope"]["parts_removed"], ["previously_removed"])
        self.assertEqual(result["scope"]["physics"]["parts_removed_in_path_simulation"], [])
        with self.assertRaisesRegex(ValueError, "removed_parts_mismatch"):
            self.collect(scope={"physics": {"parts_removed_in_path_simulation": ["other"]}})

    def test_success_and_exact_threshold_use_strict_less_than(self):
        result = self.collect(self.planner((0.5, 0.9)), raw_success=True)
        self.assertFalse(result["measurements"]["pairs"][0]["separation_unsatisfied"])
        self.assertEqual(result["termination_reason"], "success")

    def test_explicit_effective_min_sep_is_recorded(self):
        result = self.collect(self.planner((0.5, 0.9)), scope={"physics": {"min_sep": 0.7}})
        self.assertTrue(result["measurements"]["pairs"][0]["separation_unsatisfied"])
        self.assertEqual(result["scope"]["physics"]["min_sep"], 0.7)
        self.assertEqual(result["scope"]["physics"]["col_th_dict"], {"arm": 0.02, "base": 0.01})

    def test_contact_predicate_disagreement_is_rejected(self):
        planner = self.planner()
        planner.sim.contact_override = False
        with self.assertRaisesRegex(ValueError, "predicate_disagreement"):
            self.collect(planner)

    def test_success_with_unsatisfied_endpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "violates_endpoint"):
            self.collect(raw_success=True)

    def test_stale_path_is_rejected_before_geometry_queries(self):
        planner = self.planner()
        with self.assertRaisesRegex(ValueError, "not_current"):
            self.collect(planner, path=[[1, 0, 0, 0, 0, 0]])
        self.assertFalse(any(c[0] == "distance" for c in planner.sim.calls))

    def test_nonfinite_and_malformed_observations_are_rejected(self):
        for bad in (float("nan"), float("inf"), "0.2", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.collect(self.planner((bad, 0.3)))
        for bad_path in ([], [[0, 0, 0]], [[float("nan"), 0, 0, 0, 0, 0]]):
            with self.subTest(path=bad_path), self.assertRaises(ValueError):
                self.collect(path=bad_path)

    def test_metadata_and_scope_mismatches_are_rejected(self):
        cases = [{"raw_success": 1}, {"event_id": ""}, {"trajectory_sha256": "bad"},
                 {"scope": {"physics": {"min_sep": -1}}},
                 {"scope": {"physics": {"min_sep": float("nan")}}},
                 {"scope": {"physics": []}}, {"scope": {"part_move": "other"}},
                 {"scope": {"parts_rest": ["other"]}},
                 {"scope": {"frame": "unknown"}},
                 {"scope": {"physics": {"col_th_dict": {"arm": 8}}}}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.collect(**changes)

    def test_absolute_traversal_and_forbidden_drive_pointers_rejected_lexically(self):
        for ref in ("../path.json", "/tmp/path.json", "E:/path.json", "E:\\path.json",
                    "C:/path.json", "//server/path.json", "paths\\..\\file.json"):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.collect(trajectory_ref=ref)

    def test_missing_constructor_observation_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "missing_observed"):
            self.collect(FakePlanner())

    def test_contact_history_preserves_raw_names_and_rejects_duplicates(self):
        planner = self.planner()
        planner.sim.contacts = ["partbase", "ground"]
        self.assertEqual(self.collect(planner)["measurements"]["contact_history"],
                         ["partbase", "ground"])
        planner.sim.contacts = ["partbase", "partbase"]
        with self.assertRaisesRegex(ValueError, "duplicate_contact"):
            self.collect(planner)


if __name__ == "__main__":
    unittest.main()
