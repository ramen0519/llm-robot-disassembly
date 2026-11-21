"""Unit fixtures for the force controller; NO RedMax/ASAP physics or API calls.

The fake simulation moves by its force vector and has no collision geometry.
Its success flag must never be reported as actual disassembly evidence.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
if np is not None:
    from path_repair import execute_single, execute_two_leg, SKILL_ID, MAX_SWITCH_STEP


class FakeSimulation:
    """Arithmetic fixture, not a physical solver."""
    def __init__(self, fail_step=None, ignore_force=False):
        self.q = np.zeros(6)
        self.force = np.zeros(3)
        self.reset_calls = 0
        self.forward_calls = []
        self.force_history = []
        self.fail_step = fail_step
        self.ignore_force = ignore_force
        self.sentinel = object()

    def reset(self):
        self.reset_calls += 1
        self.q[:] = 0
        return "reset-forwarded"

    def forward(self, steps, verbose=False):
        if self.fail_step == len(self.forward_calls) + 1:
            raise RuntimeError("fake_integration_error")
        self.forward_calls.append((steps, verbose))
        direction = np.array([1., 0., 0.]) if self.ignore_force else self.force
        self.q[:3] += .01 * direction
        return "forward-forwarded"

    def set_body_external_force(self, name, force):
        self.force = np.array(force[-3:], dtype=float)
        self.force_history.append(self.force.tolist())


class FakePlanner:
    """Deterministic verifier-shaped fixture. Success is supplied, not computed."""
    def __init__(self, steps=4, success=True, **sim_options):
        self.sim = FakeSimulation(**sim_options)
        self.steps, self.success = steps, success
        self.check_calls = []
        self.apply_calls = []
        self.check_mode = None

    def get_state(self):
        return SimpleNamespace(q=self.sim.q.copy())

    def apply_action(self, action):
        self.apply_calls.append(action.tolist())
        if self.check_mode == "raise_switch" and len(self.apply_calls) == 2:
            raise RuntimeError("fake_switch_error")
        self.sim.set_body_external_force("fake_part", np.concatenate([np.zeros(3), action]))

    def check_success(self, action, return_path=False, min_sep=None):
        self.check_calls.append((action.tolist(), return_path, min_sep))
        if self.check_mode != "no_reset":
            assert self.sim.reset() == "reset-forwarded"
        self.apply_action(action)
        path = [self.get_state().q]
        for _ in range(self.steps):
            self.sim.forward(2 if self.check_mode == "multi_step" else 1, verbose=False)
            path.append(self.get_state().q)
            if self.check_mode == "double_reset":
                self.sim.reset()
        if self.check_mode == "nan_path":
            path[-1][0] = np.nan
        if self.check_mode == "bad_shape":
            path[-1] = np.zeros(2)
        if self.check_mode == "empty_path":
            path = []
        return self.success, path


@unittest.skipIf(np is None, "Worker NumPy dependency absent; run these 19 fixtures in ASAP Python 3.8")
class PathRepairFixtureTests(unittest.TestCase):
    def test_two_legs_forward_original_check_once_and_restore_sim(self):
        planner = FakePlanner()
        original = planner.sim
        result = execute_two_leg(planner, [1, 0, 0], [0, 0, 1], switch_step=2)
        self.assertIs(planner.sim, original)
        self.assertEqual(planner.check_calls, [([1., 0., 0.], True, None)])
        self.assertEqual(planner.apply_calls, [[1., 0., 0.], [0., 0., 1.]])
        self.assertEqual(original.reset_calls, 1)
        self.assertEqual(original.forward_calls, [(1, False)] * 4)
        self.assertEqual(result["controller"]["skill_id"], SKILL_ID)
        self.assertEqual(result["controller"]["reset_count"], 1)
        self.assertEqual(result["controller"]["switch_count"], 1)
        self.assertEqual(result["controller"]["physics_steps"], 4)
        self.assertEqual(result["controller"]["switch_q"], [.02, 0., 0., 0., 0., 0.])
        self.assertEqual(result["path"][0], [0.] * 6)
        self.assertEqual(result["path"][-1], [.02, 0., .02, 0., 0., 0.])
        self.assertEqual(result["geometry"]["first_leg_displacement"], [.02, 0., 0.])
        self.assertEqual(result["geometry"]["second_leg_displacement"], [0., 0., .02])
        self.assertTrue(result["geometry"]["noncollinear"])
        self.assertAlmostEqual(result["geometry"]["angle_degrees"], 90.)
        self.assertGreater(result["geometry"]["maximum_deviation_from_start_end_line"], 0.)
        self.assertFalse(result["controller"]["stability_checked_by_controller"])
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))

    def test_switch_occurs_after_completed_N_steps(self):
        planner = FakePlanner(steps=3)
        result = execute_two_leg(planner, [0, -1, 0], [-1, 0, 0], switch_step=1)
        self.assertEqual(result["path"][1][:3], [0., -.01, 0.])
        self.assertEqual(result["path"][2][:3], [-.01, -.01, 0.])
        self.assertEqual(result["path"][3][:3], [-.02, -.01, 0.])

    def test_default_switch_is_fifty(self):
        planner = FakePlanner(steps=51)
        result = execute_two_leg(planner, [1, 0, 0], [0, 1, 0])
        self.assertEqual(result["controller"]["actions"][1]["step"], 50)
        self.assertAlmostEqual(result["controller"]["switch_q"][0], .50)
        self.assertAlmostEqual(result["geometry"]["second_leg_displacement"][1], .01)

    def test_short_run_does_not_invent_executed_switch(self):
        for steps in (1, 2):
            with self.subTest(steps=steps):
                result = execute_two_leg(FakePlanner(steps=steps), [1, 0, 0], [0, 1, 0], 2)
                self.assertTrue(result["raw_success"])
                self.assertEqual(result["controller"]["switch_count"], 0)
                self.assertIsNone(result["controller"]["switch_q"])
                self.assertFalse(result["geometry"]["noncollinear"])
                self.assertIsNone(result["geometry"]["second_leg_displacement"])

    def test_failed_path_result_preserved(self):
        result = execute_two_leg(FakePlanner(success=False), [1, 0, 0], [0, 1, 0], 2)
        self.assertIs(result["raw_success"], False)
        self.assertEqual(len(result["path"]), 5)
        self.assertEqual(result["controller"]["switch_count"], 1)

    def test_changed_force_without_changed_geometry_is_not_noncollinear(self):
        result = execute_two_leg(FakePlanner(ignore_force=True), [1, 0, 0], [0, 1, 0], 2)
        self.assertEqual(result["controller"]["switch_count"], 1)
        self.assertFalse(result["geometry"]["noncollinear"])
        self.assertAlmostEqual(result["geometry"]["sin_angle"], 0.)

    def test_single_probe_retains_force_and_checks_no_repair(self):
        planner = FakePlanner()
        original = planner.sim
        result = execute_single(planner, [0, 1, 0], min_sep=.5)
        self.assertIs(planner.sim, original)
        self.assertEqual(planner.check_calls, [([0., 1., 0.], True, .5)])
        self.assertEqual(planner.apply_calls, [[0., 1., 0.]])
        self.assertEqual(result["controller"]["switch_count"], 0)
        self.assertEqual(result["controller"]["skill_id"], "native_single_force_probe")
        self.assertFalse(result["geometry"]["noncollinear"])
        self.assertEqual(result["path"][-1][:3], [0., .04, 0.])

    def test_numpy_axes_and_numpy_boolean_result_supported(self):
        result = execute_two_leg(FakePlanner(success=np.bool_(True)), np.array([1, 0, 0]),
                                 np.array([0., 0., -1.]), np.int64(2))
        self.assertIs(type(result["raw_success"]), bool)
        json.dumps(result, allow_nan=False)

    def test_invalid_actions_rejected_before_verifier(self):
        invalid = ([0, 0, 0], [2, 0, 0], [.5, .5, 0], [1, 1, 0],
                   [True, 0, 0], [1, 0], [1, 0, 0, 0], "100", None,
                   [float("nan"), 0, 0], [float("inf"), 0, 0],
                   [[1], [0], [0]], np.ones((3, 1)), {"x": 1}, ["1", 0, 0])
        for bad in invalid:
            for target in ("first", "second"):
                with self.subTest(bad=repr(bad), target=target):
                    planner = FakePlanner()
                    original = planner.sim
                    with self.assertRaises(ValueError):
                        execute_two_leg(planner, bad if target == "first" else [1, 0, 0],
                                        bad if target == "second" else [0, 1, 0], 2)
                    self.assertIs(planner.sim, original)
                    self.assertEqual(original.reset_calls, 0)
                    self.assertEqual(planner.check_calls, [])

    def test_parallel_and_opposite_axes_rejected(self):
        for action in ([1, 0, 0], [-1, 0, 0]):
            with self.subTest(action=action):
                with self.assertRaisesRegex(ValueError, "nonparallel"):
                    execute_two_leg(FakePlanner(), [1, 0, 0], action)

    def test_invalid_step_rejected_before_reset(self):
        for step in (0, -1, True, 2., "2", None, MAX_SWITCH_STEP + 1, np.bool_(True)):
            with self.subTest(step=step):
                planner = FakePlanner()
                with self.assertRaises(ValueError):
                    execute_two_leg(planner, [1, 0, 0], [0, 1, 0], step)
                self.assertEqual(planner.sim.reset_calls, 0)

    def test_invalid_separation_rejected(self):
        for separation in (-1, True, "0.5", float("nan"), float("inf")):
            with self.subTest(separation=separation):
                with self.assertRaises(ValueError):
                    execute_single(FakePlanner(), [1, 0, 0], min_sep=separation)

    def test_integration_exception_restores_original(self):
        planner = FakePlanner(fail_step=3)
        original = planner.sim
        with self.assertRaisesRegex(RuntimeError, "fake_integration_error"):
            execute_two_leg(planner, [1, 0, 0], [0, 1, 0], 2)
        self.assertIs(planner.sim, original)
        self.assertEqual(original.reset_calls, 1)

    def test_switch_exception_restores_original(self):
        planner = FakePlanner()
        planner.check_mode = "raise_switch"
        original = planner.sim
        with self.assertRaisesRegex(RuntimeError, "fake_switch_error"):
            execute_two_leg(planner, [1, 0, 0], [0, 1, 0], 2)
        self.assertIs(planner.sim, original)

    def test_second_reset_rejected_without_resetting_real_sim(self):
        planner = FakePlanner()
        planner.check_mode = "double_reset"
        original = planner.sim
        with self.assertRaisesRegex(RuntimeError, "intermediate_simulation_reset"):
            execute_two_leg(planner, [1, 0, 0], [0, 1, 0], 2)
        self.assertIs(planner.sim, original)
        self.assertEqual(original.reset_calls, 1)

    def test_reset_required_before_forward(self):
        planner = FakePlanner()
        planner.check_mode = "no_reset"
        original = planner.sim
        with self.assertRaisesRegex(RuntimeError, "reset_once_before_forward"):
            execute_single(planner, [1, 0, 0])
        self.assertIs(planner.sim, original)

    def test_unexpected_multi_step_upstream_call_rejected(self):
        planner = FakePlanner()
        planner.check_mode = "multi_step"
        original = planner.sim
        with self.assertRaisesRegex(ValueError, "forward_one_step"):
            execute_single(planner, [1, 0, 0])
        self.assertIs(planner.sim, original)
        self.assertEqual(original.forward_calls, [])

    def test_malformed_upstream_results_restore_original(self):
        for mode in ("nan_path", "bad_shape", "empty_path"):
            with self.subTest(mode=mode):
                planner = FakePlanner()
                planner.check_mode = mode
                original = planner.sim
                with self.assertRaises(ValueError):
                    execute_single(planner, [1, 0, 0])
                self.assertIs(planner.sim, original)
        for success in (1, "True", None):
            with self.subTest(success=success):
                planner = FakePlanner(success=success)
                original = planner.sim
                with self.assertRaisesRegex(ValueError, "must_be_boolean"):
                    execute_single(planner, [1, 0, 0])
                self.assertIs(planner.sim, original)

    def test_missing_planner_method_rejected_before_mutation(self):
        planner = FakePlanner()
        original = planner.sim
        planner.apply_action = None
        with self.assertRaisesRegex(ValueError, "missing_apply_action"):
            execute_single(planner, [1, 0, 0])
        self.assertIs(planner.sim, original)


if __name__ == "__main__":
    unittest.main()
