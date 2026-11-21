"""Bounded force-turn controller over the existing ASAP path verifier (Py3.8).

This module does not implement a replacement physics checker.  The supplied
planner's ``check_success`` runs once, with its original reset, termination,
contact, and separation tests.  A temporary simulation proxy changes the force
after the specified number of actual ``forward(1)`` calls.  The caller must
still run the upstream stability/grasp checks before committing a DFS edge.

``raw_success`` is the upstream path result, not proof of stability, a useful
two-leg path, or improvement over the upstream six-direction search.  In
particular callers must inspect ``geometry.noncollinear`` and the switch count.
"""
import math
import numbers

import numpy as np


SKILL_ID = "turn_force_two_leg_v1"
VERSION = 1
MAX_SWITCH_STEP = 1000000
_AXES = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
         (0., -1., 0.), (0., 0., 1.), (0., 0., -1.))
_DISPLACEMENT_EPS = 1e-9
_SIN_ANGLE_EPS = 1e-6


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(name + "_must_be_finite_real")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(name + "_must_be_finite_real")
    return number


def _action(value):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("action_must_be_axis_vector")
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError("action_must_be_axis_vector")
    if len(value) != 3:
        raise ValueError("action_must_be_axis_vector")
    vector = tuple(_finite_number(item, "action") for item in value)
    if vector not in _AXES:
        raise ValueError("action_must_be_one_of_six_unit_axes")
    return vector


def _state_q(planner):
    q = planner.get_state().q
    if not isinstance(q, (list, tuple, np.ndarray)):
        raise ValueError("verifier_state_must_be_vector")
    if isinstance(q, np.ndarray) and q.ndim != 1:
        raise ValueError("verifier_state_must_be_vector")
    if len(q) not in (3, 6):
        raise ValueError("verifier_state_dimension_must_be_3_or_6")
    return [_finite_number(item, "verifier_state") for item in q]


def _path_json(path, dimension):
    if not isinstance(path, (list, tuple, np.ndarray)) or len(path) == 0:
        raise ValueError("verifier_path_must_be_nonempty_state_sequence")
    result = []
    for q in path:
        if not isinstance(q, (list, tuple, np.ndarray)) or len(q) != dimension:
            raise ValueError("verifier_path_dimension_mismatch")
        if isinstance(q, np.ndarray) and q.ndim != 1:
            raise ValueError("verifier_path_state_must_be_vector")
        result.append([_finite_number(item, "verifier_path") for item in q])
    return result


class _ScheduledSimulation:
    """Forward all physics operations, intervening only at a force boundary."""

    def __init__(self, simulation, planner, second_action, switch_step, reset_observer=None):
        self._simulation = simulation
        self._planner = planner
        self._second_action = second_action
        self._switch_step = switch_step
        self._reset_observer = reset_observer
        self.reset_count = 0
        self.switch_count = 0
        self.physics_steps = 0
        self.start_q = None
        self.switch_q = None

    def __getattr__(self, name):
        return getattr(self._simulation, name)

    def reset(self, *args, **kwargs):
        if self.reset_count:
            raise RuntimeError("intermediate_simulation_reset_forbidden")
        result = self._simulation.reset(*args, **kwargs)
        self.reset_count = 1
        self.start_q = _state_q(self._planner)
        if self._reset_observer is not None:
            self._reset_observer(self._planner)
        return result

    def forward(self, steps, *args, **kwargs):
        # The pinned upstream check_success uses forward(1).  Do not silently
        # split a changed upstream API call into different integration calls.
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, numbers.Integral) or steps != 1:
            raise ValueError("scheduled_verifier_requires_forward_one_step")
        if self.reset_count != 1:
            raise RuntimeError("verifier_must_reset_once_before_forward")
        if (self._second_action is not None and self.switch_count == 0
                and self.physics_steps == self._switch_step):
            self.switch_q = _state_q(self._planner)
            self._planner.apply_action(np.array(self._second_action, dtype=float))
            self.switch_count = 1
        result = self._simulation.forward(steps, *args, **kwargs)
        self.physics_steps += 1
        return result


def _geometry(start_q, switch_q, end_q, path):
    start = np.array(start_q[:3], dtype=float)
    end = np.array(end_q[:3], dtype=float)
    first_end = end if switch_q is None else np.array(switch_q[:3], dtype=float)
    first = first_end - start
    first_norm = float(np.linalg.norm(first))
    second = None if switch_q is None else end - first_end
    second_norm = None if second is None else float(np.linalg.norm(second))
    sine = None
    angle = None
    if (second is not None and first_norm > _DISPLACEMENT_EPS
            and second_norm > _DISPLACEMENT_EPS):
        sine = float(np.linalg.norm(np.cross(first / first_norm, second / second_norm)))
        cosine = float(np.clip(np.dot(first / first_norm, second / second_norm), -1., 1.))
        angle = float(math.degrees(math.acos(cosine)))
    displacement = end - start
    distance = float(np.linalg.norm(displacement))
    relative = np.array(path, dtype=float)[:, :3] - start
    if distance > _DISPLACEMENT_EPS:
        unit = displacement / distance
        residual = relative - np.outer(relative @ unit, unit)
    else:
        residual = relative
    maximum_deviation = float(np.linalg.norm(residual, axis=1).max())
    return {
        "coordinate_frame": "world_xyz_from_upstream_get_joint_qm",
        "first_leg_displacement": first.tolist(),
        "second_leg_displacement": None if second is None else second.tolist(),
        "first_leg_distance": first_norm,
        "second_leg_distance": second_norm,
        "angle_degrees": angle,
        "sin_angle": sine,
        "noncollinear": bool(sine is not None and sine > _SIN_ANGLE_EPS),
        "noncollinear_displacement_epsilon": _DISPLACEMENT_EPS,
        "noncollinear_sin_angle_epsilon": _SIN_ANGLE_EPS,
        "maximum_deviation_from_start_end_line": maximum_deviation,
        "start_q": start_q,
        "end_q": end_q,
        "path_state_count": len(path),
    }


def _execute(planner, first_action, second_action, switch_step, min_sep, reset_observer=None):
    first = _action(first_action)
    second = None if second_action is None else _action(second_action)
    if second is not None:
        if sum(a * b for a, b in zip(first, second)) != 0:
            raise ValueError("two_leg_actions_must_be_nonparallel_axes")
        if (isinstance(switch_step, (bool, np.bool_))
                or not isinstance(switch_step, numbers.Integral)
                or not 1 <= switch_step <= MAX_SWITCH_STEP):
            raise ValueError("switch_step_must_be_bounded_positive_integer")
        switch_step = int(switch_step)
    if min_sep is not None:
        min_sep = _finite_number(min_sep, "min_sep")
        if min_sep < 0:
            raise ValueError("min_sep_must_be_nonnegative")
    for method in ("check_success", "apply_action", "get_state"):
        if not callable(getattr(planner, method, None)):
            raise ValueError("planner_missing_" + method)
    original_sim = planner.sim
    if isinstance(original_sim, _ScheduledSimulation):
        raise RuntimeError("nested_path_controller_forbidden")
    proxy = _ScheduledSimulation(original_sim, planner, second, switch_step, reset_observer)
    planner.sim = proxy
    try:
        result = planner.check_success(np.array(first, dtype=float),
                                       return_path=True, min_sep=min_sep)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("verifier_result_must_be_success_and_path")
        success, raw_path = result
        if not isinstance(success, (bool, np.bool_)):
            raise ValueError("verifier_success_must_be_boolean")
        if proxy.reset_count != 1 or proxy.start_q is None:
            raise RuntimeError("verifier_did_not_reset_exactly_once")
        end_q = _state_q(planner)
        path = _path_json(raw_path, len(proxy.start_q))
        if len(end_q) != len(proxy.start_q):
            raise ValueError("verifier_state_dimension_changed")
        actions = [{"step": 0, "action": list(first)}]
        if second is not None:
            actions.append({"step": switch_step, "action": list(second)})
        return {
            "raw_success": bool(success),
            "path": path,
            "controller": {
                "skill_id": SKILL_ID if second is not None else "native_single_force_probe",
                "version": VERSION,
                "actions": actions,
                "reset_count": proxy.reset_count,
                "switch_count": proxy.switch_count,
                "physics_steps": proxy.physics_steps,
                "switch_q": proxy.switch_q,
                "switch_semantics": "after_first_N_completed_forward_steps_before_step_N_plus_1",
                "min_sep_argument": min_sep,
                "intermediate_resets": 0,
                "upstream_check_success_calls": 1,
                "stability_checked_by_controller": False,
                "dfs_state_mutations": 0,
            },
            "geometry": _geometry(proxy.start_q, proxy.switch_q, end_q, path),
        }
    finally:
        planner.sim = original_sim


def execute_two_leg(planner, first_action, second_action, switch_step=50, min_sep=None, reset_observer=None):
    """Run one original path check, changing force after N integration steps.

    Only two perpendicular unit axes are accepted.  A result before the switch
    is preserved honestly with switch_count=0 and noncollinear=False; it is not
    a successful execution of a two-leg repair.  Exceptions propagate after
    restoring planner.sim.  This function never writes a trajectory to disk.
    """
    # None is reserved internally for execute_single; a caller cannot silently
    # turn a malformed repair request into a single-direction probe.
    if second_action is None:
        raise ValueError("two_leg_requires_second_action")
    return _execute(planner, first_action, second_action, switch_step, min_sep, reset_observer)


def execute_single(planner, action, min_sep=None, reset_observer=None):
    """Measure one unchanged upstream constant-force probe using the same proxy."""
    return _execute(planner, action, None, None, min_sep, reset_observer)
