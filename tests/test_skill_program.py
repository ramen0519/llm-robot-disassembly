"""Offline synthetic fixtures for the IR interpreter; no API/ASAP invocation.

The fixtures label a backend to exercise provenance contracts only. They are
not real physics evidence or examples of autonomous skill discovery.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.skill_program import (
    MAX_RULES, LANGUAGE, program_hash, program_schema, step_program, validate_program,
)
from worker.explicit_action_contract import PROTOCOL, action_id, validate_response
from worker.path_probe_round import validate_path_round_response
from worker.stability_diagnostics import StabilityObservation


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def request():
    return dict(schema_version=1, protocol=PROTOCOL, run_id="unit_fixture_only",
        request_id="fixture:r1", state_id="s1", remaining_parts=["b", "a", "base"],
        candidate_ids=["b", "a"], base_part="base", max_grippers=2,
        pose_catalog=[dict(pose_id="pose_current", matrix=deepcopy(IDENTITY))],
        attempted_action_ids=[], attempted_actions=[], path_stage_observations=[],
        task_context={"assembly_id": "unit_fixture_only", "input_sha256": "a"*64,
                      "solver_settings": {"seed": 0, "force_mag": 100}},
        evidence_ids=[], episode_feedback=[], memory_references=[],
        geometry={"parts": {"a": {"volume": 2, "extents": [1, 2, 3]},
                            "b": {"volume": 5, "extents": [2, 3, 4]}}})


def choice(**changes):
    value = dict(part_id="a", pose_id="pose_current", skill_id="single_force_v1",
                 direction_id="xp", second_direction_id=None, support_ids=[], evidence_ids=[])
    value.update(changes)
    return value


def append_observation(req, selected=None, *, path=False, stability=None,
                       status="UNKNOWN", late=False, accepted=False):
    selected = deepcopy(selected or choice())
    identifier = action_id(req, selected)
    event_id = "fixture:event:" + str(len(req["attempted_actions"]))
    req["attempted_actions"].append(dict(action_id=identifier,
        **{key: value for key, value in selected.items() if key != "evidence_ids"}))
    req["attempted_action_ids"].append(identifier)
    req["path_stage_observations"].append(dict(action_id=identifier,
        path_success=path, status=status, deadline_exceeded=late,
        backend="ASAP_RedMax", source_event_id=event_id))
    pose = next(p for p in req["pose_catalog"] if p["pose_id"] == selected["pose_id"])
    req["episode_feedback"].append(dict(evidence_id=event_id, action_id=identifier,
        remaining_parts=deepcopy(req["remaining_parts"]), choice=selected,
        selected_pose_matrix=deepcopy(pose["matrix"]), path_success=path,
        stability_success=stability, accepted_for_transition=accepted,
        deadline_exceeded=late, status=status))
    req["evidence_ids"].append(event_id)
    return req


def rule(when=None, rank=None, **select):
    selection = dict(primitive="single_force_v1", part="same_as_last", pose="same_as_last",
                     direction="perpendicular_last", second_direction="none", support="empty")
    selection.update(select)
    return dict(when=when or ["last_path_failed"], select=selection,
                rank=[dict(metric="direction_up", order="desc")] if rank is None else rank)


def program(*rules):
    return dict(language=LANGUAGE, rules=list(rules) or [rule()])


class SkillProgramTests(unittest.TestCase):
    def test_conditional_recovery_and_complete_current_permutation(self):
        req = append_observation(request())
        result = step_program(program(), req)
        self.assertTrue(result["triggered"])
        self.assertEqual(result["physical_calls"], 0)
        self.assertEqual(result["response"]["choice"]["direction_id"], "zp")
        self.assertEqual(result["response"]["choice"]["part_id"], "a")
        self.assertEqual(set(result["response"]["candidate_order"]), {"a", "b"})
        self.assertEqual(result["response"]["choice"]["evidence_ids"], req["evidence_ids"])
        validate_response(req, result["response"])
        validate_path_round_response(req, result["response"])

    def test_observation_sensitivity_changes_rule_and_action(self):
        path_failure = append_observation(request())
        stability_failure = append_observation(request(), path=True, stability=False)
        policy = program(rule(), rule(when=["last_path_passed", "last_stability_failed"],
            direction="same_as_last", support="nonempty", rank=[dict(metric="support_count", order="asc")]))
        first, second = step_program(policy, path_failure), step_program(policy, stability_failure)
        self.assertEqual((first["rule_index"], second["rule_index"]), (0, 1))
        self.assertEqual(first["response"]["choice"]["direction_id"], "zp")
        self.assertEqual(second["response"]["choice"]["direction_id"], "xp")
        self.assertEqual(second["response"]["choice"]["support_ids"], ["b"])

    def test_no_trigger_abstains_and_is_not_exhaustion(self):
        result = step_program(program(), request())
        self.assertFalse(result["triggered"])
        self.assertIsNone(result["response"])
        self.assertEqual(result["not_applicable_reason"], "no_trigger")
        self.assertNotIn("status", result)

    def test_step_is_stateless_and_does_not_mutate_request_or_program(self):
        req, policy = append_observation(request()), program()
        before = deepcopy((req, policy))
        first = step_program(policy, req)
        first["response"]["choice"]["support_ids"].append("tampering")
        second = step_program(policy, req)
        self.assertEqual((req, policy), before)
        self.assertEqual(second["response"]["choice"]["support_ids"], [])

    def test_attempt_history_prevents_repeating_selected_action(self):
        req = append_observation(request())
        initial = step_program(program(), req)["response"]
        # A policy with no observation-dependent selector still cannot repeat.
        fixed_axis = program(rule(when=["always"], part="any", pose="any", direction="zp", rank=[]))
        append_observation(req, initial["choice"], path=False)
        response = step_program(fixed_axis, req)["response"]
        self.assertEqual(response["choice"]["part_id"], "b")
        self.assertNotIn(response["action_id"], req["attempted_action_ids"])

    def test_failed_path_support_variants_are_deferred_not_reused(self):
        req = append_observation(request())
        policy = program(rule(direction="same_as_last", support="nonempty"))
        result = step_program(policy, req)
        self.assertTrue(result["triggered"])
        self.assertIsNone(result["response"])
        self.assertFalse(result["candidate_pruning"])
        self.assertEqual(result["not_applicable_reason"], "no_matching_untried_action")
        # Legacy policy can propose the legal support variant but still no PASS.
        self.assertIsNotNone(step_program(policy, req, path_policy="legacy_v1")["response"])

    def test_last_feedback_requires_exact_local_state_and_pose(self):
        req = append_observation(request())
        for mutate in (
            lambda r: r["episode_feedback"][0].update(remaining_parts=["a", "base"]),
            lambda r: r["episode_feedback"][0]["selected_pose_matrix"][0].__setitem__(3, 4),
            lambda r: r["episode_feedback"][0]["choice"].update(part_id="b"),
            lambda r: r["path_stage_observations"][0].update(backend="mock"),
            lambda r: r["path_stage_observations"][0].update(source_event_id="different"),
        ):
            altered = deepcopy(req)
            mutate(altered)
            with self.subTest(altered=altered):
                self.assertIsNone(step_program(program(), altered)["response"])

    def test_task_change_rejects_old_attempt_scope_before_interpretation(self):
        req = append_observation(request())
        req["task_context"]["assembly_id"] = "different_task"
        with self.assertRaisesRegex(ValueError, "attempted_action_scope_mismatch"):
            step_program(program(), req)

    def test_backtracked_state_never_uses_newer_other_state_feedback(self):
        req = append_observation(request())
        unrelated = deepcopy(req["episode_feedback"][0])
        unrelated.update(evidence_id="other_state_event", action_id="different_action",
                         remaining_parts=["a", "base"], path_success=True, stability_success=False)
        req["episode_feedback"].append(unrelated)
        req["evidence_ids"].append("other_state_event")
        result = step_program(program(), req)
        self.assertEqual(result["scoped_feedback_evidence_id"], "fixture:event:0")
        self.assertEqual(result["response"]["choice"]["direction_id"], "zp")

    def test_late_timeout_missing_and_mock_feedback_are_not_negative_facts(self):
        for kwargs in ({"late": True}, {"path": None}):
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(step_program(program(), append_observation(request(), **kwargs))["response"])
        req = append_observation(request())
        req["episode_feedback"] = []
        req["evidence_ids"] = []
        self.assertIsNone(step_program(program(), req)["response"])

    def test_source_memory_never_becomes_local_role(self):
        req = request()
        req["memory_references"] = [dict(evidence_id="source_observation", reference_only=True,
            current_part_id_mapping=None, hard_pruning=False, physics_reuse_allowed=False,
            choice=choice(), path_success=False, remaining_parts=req["remaining_parts"])]
        req["evidence_ids"] = ["source_observation"]
        self.assertIsNone(step_program(program(), req)["response"])

    def test_contradictory_physical_stage_facts_are_not_bound(self):
        for kwargs in ({"path": False, "stability": True},
                       {"path": False, "status": "PASS"}):
            with self.subTest(kwargs=kwargs):
                req = append_observation(request(), **kwargs)
                self.assertIsNone(step_program(program(), req)["response"])

    def test_different_current_ids_rebind_roles_without_program_edits(self):
        req = request()
        req["remaining_parts"] = ["new_piece", "another_piece", "new_base"]
        req["candidate_ids"] = ["another_piece", "new_piece"]
        req["base_part"] = "new_base"
        req["pose_catalog"][0]["pose_id"] = "new_pose"
        append_observation(req, choice(part_id="new_piece", pose_id="new_pose"))
        result = step_program(program(), req)
        self.assertEqual(result["response"]["choice"]["part_id"], "new_piece")
        self.assertEqual(result["response"]["choice"]["pose_id"], "new_pose")
        self.assertEqual(set(result["response"]["candidate_order"]), {"new_piece", "another_piece"})

    def test_measured_unstable_role_is_current_and_diagnostic_validated(self):
        req = append_observation(request(), path=True, stability=False)
        diagnostic = StabilityObservation().data
        diagnostic.update(phase="dynamic_failure", check_success_returned=True,
            returned_success=False, initial_disconnected_parts=[], returned_parts=["b"],
            returned_parts_semantics="upstream_representatives_after_optional_contact_component_grouping",
            dynamic_failure_parts=[dict(part_id="b", displacement=2, displacement_threshold=1,
                                        threshold_exceeded=True, lost_initial_contacts=[])])
        req["episode_feedback"][0]["stability_diagnostics"] = diagnostic
        policy = program(rule(when=["last_stability_failed", "has_unstable_parts"],
                              direction="same_as_last", support="contains_observed_unstable"))
        self.assertEqual(step_program(policy, req)["response"]["choice"]["support_ids"], ["b"])
        diagnostic["dynamic_failure_parts"][0]["part_id"] = "source_only_part"
        self.assertIsNone(step_program(policy, req)["response"])

    def test_program_does_not_execute_arbitrary_or_unknown_fields(self):
        mutations = (
            lambda p: p.update(code="__import__('os').system('never execute')"),
            lambda p: p["rules"][0].update(loop={"while": True}),
            lambda p: p["rules"][0].update(status="PASS"),
            lambda p: p["rules"][0]["select"].update(part="source_part_1"),
            lambda p: p["rules"][0]["select"].update(pose="source_pose_3"),
            lambda p: p["rules"][0]["select"].update(support=["a"]),
            lambda p: p["rules"][0]["rank"].append(dict(metric="outsideness", order="asc")),
            lambda p: p["rules"][0]["when"].append("collision_proven"),
        )
        for mutate in mutations:
            candidate = program()
            mutate(candidate)
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    validate_program(candidate)

    def test_bounded_program_shape_and_scan_limits(self):
        for candidate in (program(*[rule()]*(MAX_RULES+1)), program(dict(when=[], select={}, rank=[]))):
            with self.assertRaises(ValueError):
                validate_program(candidate)
        for value in (0, -1, True, 50001, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    step_program(program(), request(), max_actions=value)
        req = append_observation(request())
        result = step_program(program(), req, max_actions=1)
        self.assertEqual(result["scanned"], 1)
        self.assertEqual(result["not_applicable_reason"], "scan_limit")
        self.assertIsNone(result["response"])

    def test_unknown_ranking_observation_abstains_instead_of_imputing_zero(self):
        req = append_observation(request())
        del req["geometry"]["parts"]["a"]["volume"]
        policy = program(rule(rank=[dict(metric="part_volume", order="desc")]))
        self.assertIsNone(step_program(policy, req)["response"])

    def test_current_geometry_ranking_cannot_drop_unselected_candidates(self):
        req = request()
        policy = program(rule(when=["always"], part="any", pose="any", direction="zp",
                              rank=[dict(metric="part_volume", order="desc")]))
        response = step_program(policy, req)["response"]
        self.assertEqual(response["candidate_order"], ["b", "a"])
        self.assertEqual(response["choice"]["part_id"], "b")

    def test_normalized_hash_ignores_conjunction_order_but_not_rule_order(self):
        one = program(rule(when=["last_path_failed", "last_not_accepted"]))
        two = deepcopy(one)
        two["rules"][0]["when"].reverse()
        self.assertEqual(program_hash(one), program_hash(two))
        two["rules"][0]["select"]["direction"] = "opposite_last"
        self.assertNotEqual(program_hash(one), program_hash(two))
        normalized = validate_program(one)
        normalized["rules"].clear()
        self.assertTrue(one["rules"])

    def test_strict_schema_has_no_open_object_or_task_literals(self):
        schema = program_schema()
        def inspect(value):
            if type(value) is dict:
                if value.get("type") == "object":
                    self.assertIs(value["additionalProperties"], False)
                    self.assertEqual(set(value["required"]), set(value["properties"]))
                for child in value.values():
                    inspect(child)
            elif type(value) is list:
                for child in value:
                    inspect(child)
        inspect(schema)
        schema["properties"]["language"]["enum"].append("bad")
        self.assertEqual(program_schema()["properties"]["language"]["enum"], [LANGUAGE])

    def test_invalid_requests_are_not_silently_fallback_successes(self):
        req = request()
        req["candidate_ids"] = ["a"]
        with self.assertRaisesRegex(ValueError, "incomplete_candidate_set"):
            step_program(program(), req)
        with self.assertRaisesRegex(ValueError, "unknown_path_policy"):
            step_program(program(), request(), path_policy="skip_guards")

    def test_large_support_space_with_missing_role_returns_without_enumerating(self):
        req = request()
        req["remaining_parts"] = ["base"] + [str(i) for i in range(100)]
        req["candidate_ids"] = [str(i) for i in range(100)]
        req["max_grippers"] = 100
        policy = program(rule(when=["always"], part="any", pose="any", direction="zp",
                              support="contains_observed_unstable"))
        result = step_program(policy, req, max_actions=1)
        self.assertEqual(result["scanned"], 0)
        self.assertIsNone(result["response"])


if __name__ == "__main__":
    unittest.main()
