"""Wrapper integration fixtures: no RedMax simulation and no API calls.

NumPy is only needed to import the real controller module. The host without
NumPy explicitly skips these tests; the ASAP Python 3.8 worker can run them.
The fake path result below is never a physical feasibility observation.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "worker"))
if numpy is not None:
    import constraint_path_hook as module
    from independent_rank import installed_independent


class MemoryLog:
    def __init__(self):
        self.run_id, self.count, self.events = "fixture-hook-only", 0, []

    def emit(self, kind, **fields):
        self.count += 1
        self.events.append(json.loads(json.dumps(dict(
            event_id=self.run_id + ":e" + str(self.count), kind=kind, **fields))))


@unittest.skipIf(numpy is None, "NumPy unavailable on host; run this fixture in ASAP Python 3.8")
class ConstraintPathHookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = MemoryLog()
        self.session = SimpleNamespace(
            verifier_count=0, last_generation="g1", last_parts=frozenset(["a", "b", "base"]),
            feedback=[], log=self.log, rng_digest=lambda: "a" * 64)
        self.extension = {"skill_id": "fixture_two_leg", "path_success": True,
                          "controller": {"actions": [{"step": 0, "action": [1, 0, 0]},
                                                      {"step": 50, "action": [0, 0, 1]}]}}
        self.raw = {"feasible": True, "action": [1, 0, 0], "parts_fix": [],
                    "part_move": "a", "base_part": "base", "pose": None, "grasp": None}
        self.failure, self.attach_extension = False, True
        case = self

        class FakePath:
            def check_success(self, action, return_path=False, min_sep=None):
                return True, [[0, 0, 0, 0, 0, 0]]

            def _compute_col_th_dict(self, simulation, parts_fix, part_move):
                return {"a": 0.01, "b": 0.01, "base": 0.01}

        class FakeSequence:
            def _simulate(self, part_move, parts_rest, parts_removed, pose, max_grippers,
                          timeout=None, grasp_planner=None, optimizer="L-BFGS-B", debug=0, render=False):
                if case.failure:
                    raise RuntimeError("synthetic underlying verifier failure")
                FakePath().check_success([1, 0, 0], return_path=True)
                if case.attach_extension:
                    case.hook.selected_path = deepcopy(case.extension)
                return case.raw

        self.sequence, self.path = FakeSequence, FakePath
        self.original_simulate = FakeSequence._simulate
        self.original_path = FakePath.check_success
        self.original_thresholds = FakePath._compute_col_th_dict
        self.original_assemblable = lambda *args, **kwargs: (None, None)
        self.fake_base = ModuleType("plan_sequence.planner.base")
        self.fake_base.check_assemblable = self.original_assemblable
        fake_physics = ModuleType("plan_sequence.physics_planner")
        fake_physics.MultiPartPathPlanner = FakePath
        package, planner_package = ModuleType("plan_sequence"), ModuleType("plan_sequence.planner")
        package.__path__, planner_package.__path__ = [], []
        package.planner, package.physics_planner = planner_package, fake_physics
        planner_package.base = self.fake_base
        self.modules = {"plan_sequence": package, "plan_sequence.planner": planner_package,
                        "plan_sequence.planner.base": self.fake_base,
                        "plan_sequence.physics_planner": fake_physics}
        # Bypass __init__: it fingerprints real native files. This test owns
        # only Python wrappers and uses explicitly synthetic modules above.
        self.hook = module.ConstraintPathHook.__new__(module.ConstraintPathHook)
        self.hook.session = self.session
        self.hook.signature = inspect.signature(FakeSequence._simulate)
        self.hook.output = Path(self.temp.name)
        self.hook.records, self.hook.outcomes = [], []
        self.hook.path_checks, self.hook.repair_requests, self.hook.measurements = 0, 0, 0
        self.hook.decision_requests = 0
        self.hook.current, self.hook.selected_path = None, None
        self.emitted = []
        self.hook.emit = lambda kind, **fields: self.emitted.append(dict(kind=kind, **deepcopy(fields)))
        self.saved = []
        self.written = []
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, self.modules).start()
        patch.object(module, "save_snapshot", side_effect=lambda records, path:
                     self.saved.append((deepcopy(records), path))).start()
        patch.object(module, "write_new", side_effect=lambda path, value:
                     self.written.append((path, deepcopy(value)))).start()

    def call(self):
        return self.sequence()._simulate("a", ["b", "base"], [], None, 2)

    def assert_restored(self):
        self.assertIs(self.sequence._simulate, self.original_simulate)
        self.assertIs(self.path.check_success, self.original_path)
        self.assertIs(self.path._compute_col_th_dict, self.original_thresholds)
        self.assertIs(self.fake_base.check_assemblable, self.original_assemblable)

    def test_hook_retains_native_signature_for_outer_independent_observer(self):
        signature = inspect.signature(self.sequence._simulate)
        with self.hook.installed(self.sequence):
            self.assertEqual(inspect.signature(self.sequence._simulate), signature)
        self.assert_restored()

    def test_outer_observer_logs_schedule_before_tree_receives_result(self):
        registry = {}
        with self.hook.installed(self.sequence):
            with installed_independent(object, registry, self.sequence, self.session):
                result = self.call()
        self.assertIs(result, self.raw)
        observation = [v for v in self.log.events if v["kind"] == "verifier_result"][0]
        self.assertEqual(observation["raw"]["path_extension"], result["path_extension"])
        self.assertEqual(observation["raw"]["path_extension"]["controller"]["actions"],
                         self.extension["controller"]["actions"])
        self.assertTrue(observation["raw"]["path_extension"]["stability_success"])
        self.assertTrue(observation["raw"]["path_extension"]["feasible"])
        self.assertEqual(self.session.feedback[0]["raw"], observation["raw"])
        self.assertEqual(self.session.verifier_count, 1)
        self.assertEqual(self.hook.path_checks, 1)
        self.assertEqual(registry, {})
        self.assert_restored()

    def test_path_success_does_not_override_failed_stability(self):
        self.raw.update(feasible=False, parts_fix=None)
        with self.hook.installed(self.sequence):
            with installed_independent(object, {}, self.sequence, self.session):
                result = self.call()
        observed = [v for v in self.log.events if v["kind"] == "verifier_result"][0]
        self.assertEqual(observed["status"], "UNKNOWN")
        self.assertFalse(result["feasible"])
        self.assertTrue(result["path_extension"]["path_success"])
        self.assertFalse(result["path_extension"]["stability_success"])
        self.assert_restored()

    def test_no_path_extension_leaves_original_raw_object_unchanged(self):
        self.attach_extension = False
        before = deepcopy(self.raw)
        with self.hook.installed(self.sequence):
            with installed_independent(object, {}, self.sequence, self.session):
                result = self.call()
        self.assertIs(result, self.raw)
        self.assertEqual(result, before)
        self.assertEqual(self.emitted, [])
        self.assert_restored()

    def test_underlying_exception_is_logged_and_all_wrappers_are_restored(self):
        self.failure = True
        with self.assertRaisesRegex(RuntimeError, "synthetic underlying"):
            with self.hook.installed(self.sequence):
                with installed_independent(object, {}, self.sequence, self.session):
                    self.call()
        self.assertEqual(self.session.feedback[0]["status"], "ERROR")
        self.assertIsNone(self.session.feedback[0]["raw"])
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(self.saved[0][0], [])
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
