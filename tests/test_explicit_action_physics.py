"""Synthetic contracts only: fake runtime flags are NOT physical validation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "worker"))
import explicit_action_physics as executor


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def poison(*args, **kwargs):
    raise AssertionError("hidden search or adaptive support is forbidden")


class FixtureRuntime:
    def __init__(self, source):
        self.calls = []
        self.path_success = True
        self.stable = True
        self.parts_fall = None
        self.switch_executed = True
        self.raise_path = False
        self.write_input = None
        self.source_files = {"fixture_source_sha256": source}
        self.np = SimpleNamespace(array=lambda value, dtype=None: deepcopy(value),
                                  random=SimpleNamespace(get_state=lambda: ["fixture", [0], 0]))
        owner = self

        def xml(*args, **kwargs):
            assert kwargs["save_sdf"] is True
            return '<redmax><body type="SDF" load_sdf="true" save_sdf="true"/></redmax>'

        class PathPlanner:
            frame_skip, max_time, min_sep, force_mag = 100, 60., .5, 100.

            def __init__(self, assets, assembly, fixed, move, parts_removed, pose, save_sdf):
                owner.calls.append(("path_constructor", list(fixed), move, deepcopy(pose), list(parts_removed)))
                self.parts_fix, self.part_move = fixed[:], move
                self.parts_removed, self.pose = parts_removed[:], pose
                self.q = [0.] * 6
                self.sim = SimpleNamespace(options=SimpleNamespace(h=.001),
                    get_joint_qm=lambda name: self.q[:],
                    get_body_distance=lambda a, b: .6 if owner.path_success else .2,
                    body_in_contact=lambda a, b, eps: not owner.path_success,
                    get_contact_bodies=lambda name: [])
                self._compute_col_th_dict(self.sim, fixed, move)
                cache_xml = owner.physics.get_contact_sim_string(assembly, fixed + [move], save_sdf=save_sdf)
                assert 'save_sdf="false"' in cache_xml and 'load_sdf="true"' in cache_xml
                owner.physics.get_path_sim_string(assembly, fixed, move, save_sdf=save_sdf)

            def _compute_col_th_dict(self, sim, fixed, move):
                return {part: .01 for part in fixed + [move]}

            plan_path = poison

        class StabilityPlanner:
            def __init__(self, assets, assembly, fixed, movers, pose, save_sdf, allow_gap):
                owner.calls.append(("stability_constructor", fixed[:], movers[:], deepcopy(pose), allow_gap))
                cache_xml = owner.physics.get_stability_sim_string(assembly, fixed, movers, save_sdf=save_sdf)
                assert 'save_sdf="false"' in cache_xml

            def check_success(self, timeout=None):
                owner.calls.append(("stability_check", timeout))
                return owner.stable, deepcopy(owner.parts_fall)

        self.physics = SimpleNamespace(MultiPartPathPlanner=PathPlanner,
            MultiPartStabilityPlanner=StabilityPlanner, get_contact_sim_string=xml,
            get_path_sim_string=xml, get_stability_sim_string=xml,
            get_stable_plan_1pose_serial=poison, check_assemblable=poison,
            get_body_mass=poison, get_stable_poses=poison,
            MultiPartAdaptiveStabilityPlanner=poison)
        self.execute_single = lambda planner, first: self.control(planner, first)
        self.execute_two_leg = lambda planner, first, second, step: self.control(planner, first, second, step)

    def control(self, planner, first, second=None, step=None):
        self.calls.append(("controller", first[:], None if second is None else second[:], step))
        if self.raise_path:
            raise RuntimeError("fixture_physics_exception")
        if self.write_input:
            self.write_input.write_text("fixture mutation", encoding="utf-8")
        planner.q = [float(v) for v in first] + [0., 0., 0.]
        actions = [{"step": 0, "action": first[:]}]
        if second:
            actions.append({"step": step, "action": second[:]})
        switched = bool(second and self.switch_executed)
        return {"raw_success": self.path_success, "path": [[0.] * 6, planner.q[:]],
                "controller": {"actions": actions, "reset_count": 1, "switch_count": int(switched),
                    "physics_steps": 100, "switch_q": [0.] * 6 if switched else None,
                    "intermediate_resets": 0, "upstream_check_success_calls": 1},
                "geometry": {"start_q": [0.] * 6, "end_q": planner.q[:], "noncollinear": switched}}


class ExplicitActionPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.root = self.home / "asap"
        self.root.mkdir()
        (self.root / "assets").mkdir()
        self.assembly = self.home / "assembly"
        self.assembly.mkdir()
        for part in ("0", "1", "2"):
            (self.assembly / (part + ".obj")).write_text("synthetic fixture only", encoding="utf-8")
        self.source = self.root / "physics_fixture.py"
        self.source.write_text("fixture", encoding="utf-8")
        self.runtime = FixtureRuntime(self.source)
        self.context = {"asap_root": str(self.root), "assembly_dir": str(self.assembly),
                        "remaining_parts": ["0", "1", "2"], "base_part": "0",
                        "max_grippers": 2, "pose_catalog": [{"pose_id": "p0", "matrix": IDENTITY}],
                        "seed": 7, "budget": 20, "stability_timeout": 2.}
        self.choice = {"part_id": "1", "pose_id": "p0", "skill_id": "single_force_v1",
                       "direction_id": "yp", "second_direction_id": None, "support_ids": [],
                       "evidence_ids": []}
        self.output = self.home / "output"
        self.loader = patch.object(executor, "_load_runtime", return_value=self.runtime)
        self.load = self.loader.start()
        self.addCleanup(self.loader.stop)

    def run_action(self):
        return executor.execute_action(self.context, self.choice, self.output)

    def test_selected_direction_pose_and_supports_are_executed_without_search(self):
        before_context, before_choice = deepcopy(self.context), deepcopy(self.choice)
        result = self.run_action()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["feasible"])
        self.assertEqual(self.runtime.calls, [
            ("path_constructor", ["0", "2"], "1", IDENTITY, []),
            ("controller", [0, 1, 0], None, None),
            ("stability_constructor", ["0"], ["2"], IDENTITY, False),
            ("stability_check", 2.),
        ])
        self.assertEqual(result["path_check_calls"], 1)
        self.assertEqual(result["stability_check_calls"], 1)
        self.assertEqual(result["internal_stability_evaluations"], 1)
        self.assertEqual(result["geometry_queries"], 8)
        self.assertFalse(result["native_base_budget_semantics_matched"])
        self.assertEqual(result["parts_fix"], [])
        self.assertEqual(result["action"], [0, 1, 0])
        self.assertEqual(result["pose"], IDENTITY)
        self.assertEqual(self.context, before_context)
        self.assertEqual(self.choice, before_choice)
        self.assertEqual(json.loads((self.output / "result.json").read_text()), result)
        self.assertTrue(result["input_files_unchanged"])
        self.assertTrue(result["source_files_unchanged"])

    def test_terminal_two_parts_use_chosen_part_not_mass(self):
        self.context.update(remaining_parts=["1", "2"], base_part=None)
        self.choice.update(part_id="2", support_ids=["1"], direction_id="xn")
        result = self.run_action()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["part_move"], "2")
        self.assertEqual(result["parts_fix"], ["1"])
        self.assertEqual(self.runtime.calls[0][1:3], (["1"], "2"))
        self.assertEqual(result["stability_check_calls"], 0)
        self.assertTrue(result["stability_success"])
        self.assertIsNone(result["raw_upstream"]["stability_success"])
        self.assertEqual(result["stability_mode"], "all_remaining_rest_explicitly_fixed_no_checker_call")

    def test_selected_additional_supports_are_fixed_only_for_stability(self):
        self.context["base_part"] = None
        self.choice["support_ids"] = ["2"]
        result = self.run_action()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(self.runtime.calls[0][1], ["0", "2"])
        self.assertIn(("stability_constructor", ["2"], ["0"], IDENTITY, False), self.runtime.calls)
        self.assertEqual(result["executed"]["support_ids"], ["2"])

    def test_failed_path_does_not_try_another_direction_or_stability(self):
        self.runtime.path_success = False
        result = self.run_action()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["path_success"])
        self.assertIsNone(result["stability_success"])
        self.assertEqual(len(self.runtime.calls), 2)
        self.assertEqual(result["stability_check_calls"], 0)
        self.assertIsNone(result["blocker"])
        self.assertFalse(result["all_paths_impossible"])
        self.assertEqual(len(json.loads((self.output / "trajectory.json").read_text())["path"]), 2)
        evidence = json.loads((self.output / "endpoint_evidence.json").read_text())
        self.assertEqual(evidence["termination_reason"], "unknown")
        self.assertTrue(all(pair["separation_unsatisfied"] for pair in evidence["measurements"]["pairs"]))

    def test_stability_failure_retains_raw_parts_without_greedy_fixing(self):
        self.runtime.stable, self.runtime.parts_fall = False, ["2"]
        result = self.run_action()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertTrue(result["path_success"])
        self.assertFalse(result["stability_success"])
        self.assertEqual(result["raw_upstream"]["parts_fall"], ["2"])
        self.assertEqual(result["stability_check_calls"], 1)
        self.assertEqual(result["hidden_support_search_calls"], 0)
        self.assertEqual(result["parts_fix"], [])

    def test_stability_false_null_keeps_unknown_cause(self):
        self.runtime.stable = False
        result = self.run_action()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["raw_upstream"]["parts_fall"])
        self.assertIsNone(result["blocker"])
        self.assertNotIn("collision", result["reason"])

    def test_endpoint_projection_preserves_measured_facts_without_extra_queries(self):
        self.runtime.path_success = False
        result = self.run_action()
        event = json.loads((self.output / "endpoint_evidence.json").read_text())
        projection = result["endpoint_observation"]
        self.assertEqual(projection["scope"], "selected_trajectory_terminal_state_only")
        self.assertEqual(projection["min_sep"], event["measurements"]["min_sep"])
        self.assertEqual(projection["evidence_sha256"], executor._sha(self.output / "endpoint_evidence.json"))
        self.assertEqual(projection["termination_reason"], "unknown")
        self.assertIsNone(projection["blocker"])
        self.assertFalse(projection["all_paths_impossible"])
        self.assertEqual(result["path_check_calls"], 1)
        self.assertEqual(result["geometry_queries"], 8)
        for actual, raw in zip(projection["pairs"], event["measurements"]["pairs"]):
            self.assertEqual(actual, {key: raw[key] for key in (
                "moving_part_id", "rest_part_id", "min_distance", "separation_unsatisfied")})
            self.assertGreater(actual["min_distance"], 0.)
            self.assertTrue(actual["separation_unsatisfied"])
        self.assertNotIn("collision", json.dumps(projection))

    def test_endpoint_projection_is_detached_and_does_not_infer_missing_facts(self):
        event = {"measurements": {"min_sep": .5, "pairs": [
            {"moving_part_id": "a", "rest_part_id": "b", "min_distance": .1}]},
            "claimed_blocker": "b", "termination_reason": "invented_collision"}
        before = deepcopy(event)
        with patch.object(Path, "open", side_effect=AssertionError("pure projection cannot read")):
            projected = executor.endpoint_observation(event, None)
        self.assertEqual(event, before)
        self.assertIsNone(projected["pairs"][0]["separation_unsatisfied"])
        self.assertIsNone(projected["termination_reason"])
        self.assertIsNone(projected["evidence_sha256"])
        self.assertIsNone(projected["blocker"])
        projected["pairs"][0]["min_distance"] = 999
        self.assertEqual(event, before)
        empty = executor.endpoint_observation({}, None)
        self.assertIsNone(empty["min_sep"])
        self.assertIsNone(empty["pairs"])
        self.assertIsNone(executor.endpoint_observation(None, None))
        self.load.assert_not_called()

    def test_endpoint_path_success_is_not_promoted_to_whole_action_success(self):
        self.runtime.stable = False
        result = self.run_action()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["endpoint_observation"]["termination_reason"], "success")
        self.assertTrue(result["raw_upstream"]["path_success"])
        self.assertFalse(result["raw_upstream"]["stability_success"])
        self.assertIsNone(result["endpoint_observation"]["blocker"])

    def test_unmeasured_endpoint_is_null_on_physics_exception(self):
        self.runtime.raise_path = True
        result = self.run_action()
        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["endpoint_observation"])

    def test_two_leg_has_one_upstream_path_check_and_requested_force_schedule(self):
        self.choice.update(skill_id="turn_force_two_leg_v1", direction_id="xp", second_direction_id="zn")
        result = self.run_action()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["path_check_calls"], 1)
        self.assertEqual(result["two_leg_controller_calls"], 1)
        self.assertIn(("controller", [1, 0, 0], [0, 0, -1], 50), self.runtime.calls)
        self.assertTrue(result["skill_fully_executed"])
        self.assertTrue(result["repair_geometry_verified"])
        self.assertFalse(result["legacy_single_action_replayer_supported"])
        self.assertEqual(len(result["executed"]["force_actions"]), 2)

    def test_early_separation_does_not_claim_force_switch_or_repair(self):
        self.choice.update(skill_id="turn_force_two_leg_v1", second_direction_id="xp")
        self.runtime.switch_executed = False
        result = self.run_action()
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["skill_fully_executed"])
        self.assertFalse(result["repair_geometry_verified"])
        self.assertEqual(len(result["executed"]["force_actions"]), 1)
        self.assertEqual(len(result["controller"]["actions"]), 2)

    def test_runtime_error_is_error_and_counted_once(self):
        self.runtime.raise_path = True
        result = self.run_action()
        self.assertEqual(result["status"], "ERROR")
        self.assertFalse(result["feasible"])
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["path_check_calls"], 1)
        self.assertEqual(result["stability_check_calls"], 0)
        self.assertTrue(result["input_files_unchanged"])

    def test_sdf_wrapper_is_restored_after_error(self):
        before = self.runtime.physics.get_contact_sim_string
        original_method = self.runtime.physics.MultiPartPathPlanner._compute_col_th_dict
        self.runtime.raise_path = True
        self.run_action()
        self.assertIs(self.runtime.physics.get_contact_sim_string, before)
        self.assertIs(self.runtime.physics.MultiPartPathPlanner._compute_col_th_dict, original_method)

    def test_all_existing_inputs_and_sources_preserved(self):
        cache = self.assembly / "0.sdf"
        cache.write_bytes(b"fixture-cache")
        before = {str(p): p.read_bytes() for p in [*self.assembly.iterdir(), self.source]}
        result = self.run_action()
        self.assertEqual(before, {name: Path(name).read_bytes() for name in before})
        self.assertEqual(result["scope"]["physics"]["sdf_cache_sha256"]["0.sdf"], executor._sha(cache))
        self.assertTrue(all(item["only_save_flag_changed"] for item in result["path_cache_io"]))

    def test_detected_input_mutation_rejects_feasible_result(self):
        self.runtime.write_input = self.assembly / "0.obj"
        result = self.run_action()
        self.assertEqual(result["status"], "ERROR")
        self.assertFalse(result["feasible"])
        self.assertFalse(result["input_files_unchanged"])

    def test_same_output_cannot_be_overwritten_or_reexecuted(self):
        self.run_action()
        raw = (self.output / "result.json").read_bytes()
        calls = len(self.runtime.calls)
        with self.assertRaises(FileExistsError):
            self.run_action()
        self.assertEqual(len(self.runtime.calls), calls)
        self.assertEqual(raw, (self.output / "result.json").read_bytes())

    def test_zero_wall_budget_never_imports_or_calls_physics(self):
        self.context["remaining_wall_seconds"] = 0.
        result = self.run_action()
        self.load.assert_not_called()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["path_check_calls"], 0)

    def test_deadline_caps_native_path_and_stability_timeouts(self):
        self.context["remaining_wall_seconds"] = 1.
        result = self.run_action()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["native_path_timeout_seconds"], 60.)
        self.assertLessEqual(result["effective_path_timeout_seconds"], 1.)
        self.assertLessEqual(result["effective_stability_timeout_seconds"], 1.)
        self.assertGreater(result["effective_stability_timeout_seconds"], 0.)

    def test_zero_stability_budget_is_not_false_or_automatic_support(self):
        self.context["stability_timeout"] = 0.
        result = self.run_action()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertTrue(result["path_success"])
        self.assertIsNone(result["stability_success"])
        self.assertEqual(result["stability_check_calls"], 0)

    def test_condition_scoped_constraints_are_from_actual_fixture_measurements(self):
        self.runtime.path_success = False
        self.context["task_context"] = {
            "schema_version": 2, "usage": "soft_ranking_only",
            "task": {"assembly_id": "fixture", "input_sha256": {
                         item.name: executor._sha(item) for item in self.assembly.iterdir()},
                     "source_commit": "b" * 40, "base_part": "0",
                     "solver_settings": {"seed": 7, "budget": 20},
                     "goal": {"type": "native_full_disassembly"}},
            "initial_parts": ["0", "1", "2"], "geometry_features": {"0": {}, "1": {}, "2": {}}}
        result = self.run_action()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["constraint_record_count"], 2)
        records = json.loads((self.output / "constraint_records.json").read_text())
        for record in records:
            self.assertEqual(record["whole_part_verdict"], "UNKNOWN")
            self.assertFalse(record["hard_pruning"])
            self.assertIsNone(record["blocker"])
            self.assertEqual(record["scope"]["selected_action"], self.choice)
            self.assertEqual(record["scope"]["support_policy"]["actual_parts_fix"], ["0"])

    def test_mismatched_input_identity_cannot_create_reusable_memory(self):
        self.context["task_context"] = {
            "schema_version": 2, "usage": "soft_ranking_only",
            "task": {"assembly_id": "fixture", "input_sha256": {"0.obj": "a" * 64},
                     "source_commit": "b" * 40, "base_part": "0",
                     "solver_settings": {"seed": 7, "budget": 20},
                     "goal": {"type": "native_full_disassembly"}},
            "initial_parts": ["0", "1", "2"], "geometry_features": {"0": {}, "1": {}, "2": {}}}
        result = self.run_action()
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["path_check_calls"], 0)
        self.assertEqual(result["constraint_record_count"], 0)
        self.assertEqual(self.runtime.calls, [])

    def test_implicit_event_ids_do_not_alias_separate_attempts(self):
        self.run_action()
        first = json.loads((self.output / "endpoint_evidence.json").read_text())["event_id"]
        self.output = self.home / "second"
        self.run_action()
        second = json.loads((self.output / "endpoint_evidence.json").read_text())["event_id"]
        self.assertNotEqual(first, second)

    def test_pose_changes_are_explicit_and_not_claimed_validated(self):
        rotated = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        self.context["pose_catalog"].append({"pose_id": "p1", "matrix": rotated})
        self.choice["pose_id"] = "p1"
        result = self.run_action()
        self.assertEqual(result["executed"]["pose"], rotated)
        self.assertEqual(self.runtime.calls[0][3], rotated)
        self.assertFalse(result["pose_transition_validated"])
        self.assertFalse(result["grasp_accessibility_validated"])

    def test_invalid_actions_rejected_without_solver_or_output(self):
        variants = [
            {"part_id": "0"}, {"part_id": "missing"}, {"part_id": ["1"]},
            {"support_ids": ["0"]}, {"support_ids": ["1"]}, {"support_ids": ["2", "2"]},
            {"support_ids": ["missing"]}, {"pose_id": "missing"},
            {"direction_id": "up"}, {"direction_id": []}, {"skill_id": "invented"},
            {"second_direction_id": "xp"}, {"skill_id": "turn_force_two_leg_v1"},
            {"skill_id": "turn_force_two_leg_v1", "second_direction_id": "yn"},
            {"skill_id": "turn_force_two_leg_v1", "second_direction_id": []},
            {"evidence_ids": ["same", "same"]}, {"arbitrary_code": "pass"},
        ]
        for changes in variants:
            choice = dict(self.choice, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                executor.execute_action(self.context, choice, self.output)
        self.load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_contexts_rejected_without_solver(self):
        variants = [
            {"remaining_parts": ["1"]}, {"remaining_parts": ["0", "1", "1"]},
            {"base_part": "missing"}, {"max_grippers": True}, {"max_grippers": 0},
            {"seed": -1}, {"seed": True}, {"budget": 0}, {"budget": float("nan")},
            {"stability_timeout": -1}, {"remaining_wall_seconds": float("inf")},
            {"removed_parts": ["0"]}, {"pose_catalog": []},
            {"pose_catalog": [{"pose_id": "p0", "matrix": [[1]]}]},
        ]
        for changes in variants:
            context = dict(self.context, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                executor.execute_action(context, self.choice, self.output)
        self.load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_gripper_budget_counts_one_mover_plus_additional_supports(self):
        self.context["max_grippers"] = 1
        self.choice["support_ids"] = ["2"]
        with self.assertRaisesRegex(ValueError, "gripper_budget"):
            self.run_action()
        self.choice["support_ids"] = []
        self.assertEqual(self.run_action()["status"], "PASS")

    def test_invalid_pose_scale_reflection_and_nonfinite_rejected(self):
        variants = [deepcopy(IDENTITY) for _ in range(4)]
        variants[0][0][0] = 2
        variants[1][0][0] = -1
        variants[2][3][2] = 1
        variants[3][0][3] = float("nan")
        for matrix in variants:
            self.context["pose_catalog"][0]["matrix"] = matrix
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                self.run_action()
        self.load.assert_not_called()

    def test_forbidden_drive_rejected_lexically_before_any_probe(self):
        self.context["asap_root"] = "E:/must-never-be-probed"
        with patch.object(Path, "lstat", side_effect=AssertionError("no filesystem probe")):
            with self.assertRaises(ValueError):
                self.run_action()
        self.load.assert_not_called()

    def test_removed_history_is_not_invented_path_obstacle(self):
        self.context.update(remaining_parts=["0", "1"], parts_removed=["2"])
        result = self.run_action()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["scope"]["parts_removed"], ["2"])
        self.assertEqual(result["scope"]["physics"]["parts_removed_in_path_simulation"], [])

    def test_null_pose_explicitly_forwards_original_pose(self):
        self.context["pose_catalog"][0]["matrix"] = None
        result = self.run_action()
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(result["pose"])
        self.assertIsNone(self.runtime.calls[0][3])

    def test_simulator_returns_unknown_stability_id_is_error(self):
        self.runtime.stable, self.runtime.parts_fall = False, ["invented"]
        result = self.run_action()
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["stability_check_calls"], 1)
        self.assertFalse(result["feasible"])


if __name__ == "__main__":
    unittest.main()
