"""Saved-schedule contract fixtures only; no actual physics, API or asset reads."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from replay_path_extension import (validate_extension_contract, bind_selected_plan,
                                    readonly_sdf_cache, SDF_BUILDERS)


def fixture():
    controller = {"skill_id": "turn_force_two_leg_v1", "version": 1,
                  "reset_count": 1, "switch_count": 1, "intermediate_resets": 0,
                  "physics_steps": 400, "switch_q": [.01, 0, 0, 0, 0, 0],
                  "actions": [{"step": 0, "action": [1., 0., 0.]},
                              {"step": 50, "action": [0., 0., 1.]}]}
    geometry = {"noncollinear": True}
    saved = {"raw_success": True, "controller": deepcopy(controller), "geometry": geometry}
    outcome = {"feasible": True, "path_success": True, "stability_success": True,
               "controller": controller, "geometry": deepcopy(geometry),
               "scope": {"action_schedule": deepcopy(controller["actions"])},
               "legacy_single_action_replayer_supported": False}
    return outcome, saved


class ReplayContractFixtureTests(unittest.TestCase):
    def test_valid_saved_schedule_returns_complete_controller(self):
        outcome, saved = fixture()
        self.assertEqual(validate_extension_contract(outcome, saved), ([1., 0., 0.], [0., 0., 1.], 50))

    def test_non_feasible_observation_cannot_be_replayed_as_success(self):
        for key in ("feasible", "path_success", "stability_success"):
            with self.subTest(key=key):
                outcome, saved = fixture()
                outcome[key] = False
                with self.assertRaises(ValueError):
                    validate_extension_contract(outcome, saved)

    def test_false_raw_path_success_rejected(self):
        outcome, saved = fixture()
        saved["raw_success"] = False
        with self.assertRaises(ValueError):
            validate_extension_contract(outcome, saved)

    def test_controller_in_file_and_outcome_must_match(self):
        outcome, saved = fixture()
        saved["controller"]["physics_steps"] += 1
        with self.assertRaises(ValueError):
            validate_extension_contract(outcome, saved)

    def test_unsupported_skill_or_version_rejected(self):
        for key, value in (("skill_id", "unknown"), ("version", 2), ("version", True)):
            with self.subTest(key=key, value=value):
                outcome, saved = fixture()
                for record in (outcome, saved):
                    record["controller"][key] = value
                with self.assertRaises(ValueError):
                    validate_extension_contract(outcome, saved)

    def test_reset_and_switch_invariants_required(self):
        for key, value in (("reset_count", 2), ("reset_count", True),
                           ("switch_count", 0), ("switch_count", True), ("intermediate_resets", 1)):
            with self.subTest(key=key, value=value):
                outcome, saved = fixture()
                for record in (outcome, saved):
                    record["controller"][key] = value
                with self.assertRaises(ValueError):
                    validate_extension_contract(outcome, saved)

    def test_missing_or_extra_action_rejected(self):
        for count in (1, 3):
            with self.subTest(count=count):
                outcome, saved = fixture()
                actions = outcome["controller"]["actions"]
                actions = actions[:1] if count == 1 else actions + [actions[1]]
                for record in (outcome, saved):
                    record["controller"]["actions"] = deepcopy(actions)
                with self.assertRaises(ValueError):
                    validate_extension_contract(outcome, saved)

    def test_invalid_switch_step_rejected(self):
        for step in (0, -1, True, 50., "50", 1000001):
            with self.subTest(step=step):
                outcome, saved = fixture()
                for record in (outcome, saved):
                    record["controller"]["actions"][1]["step"] = step
                with self.assertRaises(ValueError):
                    validate_extension_contract(outcome, saved)

    def test_invalid_saved_axis_rejected(self):
        for axis in ([2, 0, 0], [True, 0, 0], [float("nan"), 0, 0], [1, 1, 0]):
            with self.subTest(axis=axis):
                outcome, saved = fixture()
                for record in (outcome, saved):
                    record["controller"]["actions"][0]["action"] = axis
                with self.assertRaises(ValueError):
                    validate_extension_contract(outcome, saved)

    def test_parallel_axis_rejected(self):
        for axis in ([1, 0, 0], [-1, 0, 0]):
            with self.subTest(axis=axis):
                outcome, saved = fixture()
                for record in (outcome, saved):
                    record["controller"]["actions"][1]["action"] = axis
                with self.assertRaises(ValueError):
                    validate_extension_contract(outcome, saved)

    def test_scope_schedule_must_match_saved(self):
        outcome, saved = fixture()
        outcome["scope"]["action_schedule"][1]["action"] = [0, 1, 0]
        with self.assertRaises(ValueError):
            validate_extension_contract(outcome, saved)

    def test_schedule_declared_but_not_executed_is_rejected(self):
        outcome, saved = fixture()
        for record in (outcome, saved):
            record["controller"]["physics_steps"] = 50
        with self.assertRaises(ValueError):
            validate_extension_contract(outcome, saved)

    def test_missing_switch_state_rejected(self):
        outcome, saved = fixture()
        for record in (outcome, saved):
            record["controller"]["switch_q"] = None
        with self.assertRaises(ValueError):
            validate_extension_contract(outcome, saved)

    def test_geometry_without_real_turn_rejected(self):
        outcome, saved = fixture()
        saved["geometry"]["noncollinear"] = False
        outcome["geometry"]["noncollinear"] = False
        with self.assertRaises(ValueError):
            validate_extension_contract(outcome, saved)

    def test_geometry_file_and_outcome_must_match(self):
        outcome, saved = fixture()
        outcome["geometry"]["extra"] = "changed"
        with self.assertRaises(ValueError):
            validate_extension_contract(outcome, saved)

    def test_legacy_single_action_replay_cannot_substitute_schedule(self):
        outcome, saved = fixture()
        outcome["legacy_single_action_replayer_supported"] = True
        with self.assertRaises(ValueError):
            validate_extension_contract(outcome, saved)


def selected_fixture():
    outcome, _ = fixture()
    outcome.update(request_id="fixture:repair1", parent_call_id="v1", trajectory_sha256="a" * 64,
                   parts_fix=[])
    outcome["scope"].update(assembled_parts=["a", "b", "base"],
                            parts_rest=["b", "base"], part_move="a")
    tree = {"edges": [
        [["a", "b", "base"], ["b", "base"],
         {"sim_info": {"feasible": True, "part_move": "a", "parts_fix": [],
                       "path_extension": deepcopy(outcome)}}],
        [["b", "base"], ["base"], {"sim_info": {"feasible": True, "part_move": "b", "parts_fix": []}}]
    ]}
    return outcome, tree


class SelectedPlanBindingFixtureTests(unittest.TestCase):
    def test_exact_selected_edge_bound(self):
        outcome, tree = selected_fixture()
        selected, ignored = bind_selected_plan([outcome], tree, ["a", "b"], ["a", "b", "base"])
        self.assertEqual(ignored, [])
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0]["selected_plan_edge"])
        self.assertEqual(selected[0]["sequence_index"], 0)
        self.assertEqual(selected[0]["parent_call_id"], "v1")

    def test_off_plan_success_is_not_replayed(self):
        outcome, tree = selected_fixture()
        other = deepcopy(outcome)
        other["request_id"], other["parent_call_id"] = "fixture:other", "v99"
        selected, ignored = bind_selected_plan([outcome, other], tree, ["a", "b"], ["a", "b", "base"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(ignored), 1)
        self.assertFalse(ignored[0]["selected_plan_edge"])
        self.assertEqual(ignored[0]["status"], "NOT_SELECTED_NOT_REPLAYED")

    def test_stale_parent_call_id_fails(self):
        outcome, tree = selected_fixture()
        outcome["parent_call_id"] = "v2"
        with self.assertRaises(ValueError):
            bind_selected_plan([outcome], tree, ["a", "b"], ["a", "b", "base"])


class ReadOnlySDFCacheFixtureTests(unittest.TestCase):
    def module(self, xml=None):
        if xml is None:
            xml = ('<redmax><option timestep="1e-3"/>'
                   '<body type="SDF" dx="0.05" col_th="0.125" load_sdf="true" save_sdf="true"/>'
                   '<body type="MultiSDF" load_sdf="true" save_sdf="true"/></redmax>')
        calls = []
        def builder(*args, **kwargs):
            calls.append((args, kwargs))
            return xml
        return SimpleNamespace(**{name: builder for name in SDF_BUILDERS}), builder, calls, xml

    def test_only_save_flags_changed_and_load_true_preserved(self):
        module, original, calls, xml = self.module()
        with readonly_sdf_cache(module) as audit:
            for name in SDF_BUILDERS:
                result = getattr(module, name)("fixture", save_sdf=True, pose=None)
                self.assertEqual(result, xml.replace('save_sdf="true"', 'save_sdf="false"'))
                self.assertIn('load_sdf="true"', result)
                self.assertIn('col_th="0.125"', result)
        self.assertEqual(len(audit), 3)
        self.assertTrue(all(item["only_save_flag_changed"] for item in audit))
        self.assertTrue(all(item["xml_load_sdf"] and not item["xml_save_sdf"] for item in audit))
        self.assertEqual(calls, [(("fixture",), {"save_sdf": True, "pose": None})] * 3)
        self.assertTrue(all(getattr(module, name) is original for name in SDF_BUILDERS))

    def test_context_restores_builders_on_exception(self):
        module, original, _, _ = self.module()
        with self.assertRaisesRegex(RuntimeError, "fixture_error"):
            with readonly_sdf_cache(module):
                raise RuntimeError("fixture_error")
        self.assertTrue(all(getattr(module, name) is original for name in SDF_BUILDERS))

    def test_cache_disabled_input_is_rejected_not_rebuilt(self):
        module, original, _, _ = self.module('<redmax><body type="SDF" dx="0.05"/></redmax>')
        with self.assertRaisesRegex(ValueError, "loading_enabled"):
            with readonly_sdf_cache(module):
                module.get_path_sim_string()
        self.assertTrue(all(getattr(module, name) is original for name in SDF_BUILDERS))

    def test_single_quote_XML_flags_supported(self):
        xml = "<redmax><body type='SDF' load_sdf='true' save_sdf='true'/></redmax>"
        module, _, _, _ = self.module(xml)
        with readonly_sdf_cache(module):
            self.assertEqual(module.get_path_sim_string(), xml.replace("save_sdf='true'", "save_sdf='false'"))

    def test_unknown_nonstring_builder_result_rejected(self):
        module, _, _, _ = self.module({"not": "XML"})
        with self.assertRaisesRegex(ValueError, "did_not_return_xml"):
            with readonly_sdf_cache(module):
                module.get_path_sim_string()

    def test_nested_context_rejected_and_outer_still_restores(self):
        module, original, _, _ = self.module()
        with readonly_sdf_cache(module):
            with self.assertRaisesRegex(ValueError, "nested"):
                with readonly_sdf_cache(module):
                    pass
        self.assertTrue(all(getattr(module, name) is original for name in SDF_BUILDERS))

    def test_changed_trajectory_hash_fails(self):
        outcome, tree = selected_fixture()
        outcome["trajectory_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            bind_selected_plan([outcome], tree, ["a", "b"], ["a", "b", "base"])

    def test_duplicate_manifest_outcome_fails(self):
        outcome, tree = selected_fixture()
        with self.assertRaises(ValueError):
            bind_selected_plan([outcome, deepcopy(outcome)], tree, ["a", "b"], ["a", "b", "base"])

    def test_missing_tree_extension_cannot_prove_selected_use(self):
        outcome, tree = selected_fixture()
        del tree["edges"][0][2]["sim_info"]["path_extension"]
        with self.assertRaisesRegex(ValueError, "no_path_extension_in_selected_plan"):
            bind_selected_plan([outcome], tree, ["a", "b"], ["a", "b", "base"])

    def test_scope_must_match_selected_edge(self):
        outcome, tree = selected_fixture()
        outcome["scope"]["part_move"] = "b"
        tree["edges"][0][2]["sim_info"]["path_extension"] = deepcopy(outcome)
        with self.assertRaisesRegex(ValueError, "scope_or_identity"):
            bind_selected_plan([outcome], tree, ["a", "b"], ["a", "b", "base"])

    def test_selected_edge_supports_must_match_outcome(self):
        outcome, tree = selected_fixture()
        tree["edges"][0][2]["sim_info"]["parts_fix"] = ["b"]
        with self.assertRaisesRegex(ValueError, "scope_or_identity"):
            bind_selected_plan([outcome], tree, ["a", "b"], ["a", "b", "base"])

    def test_non_feasible_selected_edge_rejected(self):
        outcome, tree = selected_fixture()
        tree["edges"][0][2]["sim_info"]["feasible"] = False
        with self.assertRaisesRegex(ValueError, "not_feasible"):
            bind_selected_plan([outcome], tree, ["a", "b"], ["a", "b", "base"])


if __name__ == "__main__":
    unittest.main()
