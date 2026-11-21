"""Synthetic contract checks only; no model, simulator, or physical success."""
from copy import deepcopy
import unittest

from orchestrator.strategy_candidates import build_anchor, propose
from tests.test_skill_program import request, choice, append_observation, IDENTITY
from worker.explicit_action_contract import DIRECTIONS, validate_choice, validate_request
from worker.path_probe_round import partition_path_round, semantic_path_key


def strategy(**changes):
    value = dict(part_id="a", procedure="axis_search_v1", pose_policy="select_catalog_once_v1",
                 support_policy="preserve_add_v1", max_attempts=8, evidence_ids=[])
    value.update(changes)
    return value


def state():
    req = request()
    req["remaining_parts"] = ["a", "b", "c", "d", "base"]
    req["candidate_ids"] = ["a", "b", "c", "d"]
    req["max_grippers"] = 3
    req["geometry"]["parts"] = {p: {"bounds_center": [0, 0, 0]} for p in req["remaining_parts"]}
    req["geometry"]["parts"]["a"]["bounds_center"] = [2, 0, 0]
    return req


def observe(req, selected, *, stability=False, reported=None, late=False):
    append_observation(req, selected, path=stability, stability=False if stability else None, late=late)
    if reported is not None:
        req["episode_feedback"][-1]["raw_stability_parts"] = reported
    return req


def reference(req):
    item = req["episode_feedback"][-1]
    return dict(choice=deepcopy(item["choice"]), remaining_parts=deepcopy(item["remaining_parts"]),
                evidence_id=item["evidence_id"], selected_pose_matrix=deepcopy(item["selected_pose_matrix"]))


class StrategyCandidatesTests(unittest.TestCase):
    def test_initial_keep_orientation_needs_known_reference(self):
        result = build_anchor(state(), strategy(pose_policy="keep_orientation_v1"))
        self.assertIsNone(result["anchor"])
        self.assertEqual(result["reason"], "orientation_reference_missing")

    def test_keep_orientation_requires_unique_catalog_match(self):
        req = state()
        req["pose_catalog"].append(dict(pose_id="translated_alias", matrix=deepcopy(IDENTITY)))
        req["pose_catalog"][-1]["matrix"][2][3] = 4
        result = build_anchor(req, strategy(pose_policy="keep_orientation_v1"),
                              dict(selected_pose_matrix=IDENTITY))
        self.assertEqual(result["reason"], "orientation_unavailable_or_ambiguous")

    def test_known_orientation_rebinds_only_ground_translation_and_no_old_supports(self):
        req = state(); req["pose_catalog"][0]["matrix"][2][3] = 2
        prior = dict(selected_pose_matrix=IDENTITY, remaining_parts=["old", "a", "b"],
                     choice=choice(support_ids=["b"]), evidence_id="old-accepted",
                     scope="orientation_only_after_transition")
        result = build_anchor(req, strategy(pose_policy="keep_orientation_v1"), prior)
        self.assertEqual(result["anchor"]["pose_matrix"][2][3], 2)
        self.assertEqual(result["anchor"]["support_ids"], [])
        self.assertEqual(result["diagnostics"]["support_seed"], "orientation_only_after_transition_initial_empty")

    def test_catalog_selection_counts_attempts_then_holds_pose(self):
        req = state()
        rotated = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        req["pose_catalog"].append(dict(pose_id="rotated", matrix=rotated))
        observe(req, choice())
        step = strategy()
        built = build_anchor(req, step)
        self.assertEqual(built["anchor"]["pose_id"], "rotated")
        candidate = propose(req, step, built["anchor"])["choice"]
        self.assertEqual(candidate["pose_id"], "rotated")
        self.assertEqual(candidate["direction_id"], "yp")
        observe(req, candidate)
        self.assertEqual(propose(req, step, built["anchor"])["choice"]["pose_id"], "rotated")

    def test_axis_failures_visit_six_axes_without_changing_selected_part(self):
        req, step = state(), strategy()
        anchor = build_anchor(req, step)["anchor"]
        seen = []
        for _ in range(6):
            result = propose(req, step, anchor)
            selected = result["choice"]
            self.assertEqual(selected["part_id"], "a")
            self.assertEqual(selected["support_ids"], [])
            self.assertFalse(result["diagnostics"]["hard_pruning"])
            validate_choice(validate_request(req), selected)
            seen.append(selected["direction_id"])
            observe(req, selected)
        self.assertEqual(set(seen), set(DIRECTIONS))
        result = propose(req, step, anchor)
        self.assertIsNone(result["choice"])
        self.assertEqual(result["reason"], "local_strategy_options_exhausted")
        self.assertFalse(result["diagnostics"]["local_exhaustion_is_impossibility"])

    def test_two_leg_search_covers_only_24_perpendicular_pairs(self):
        req = state(); step = strategy(procedure="two_leg_search_v1")
        pairs = [(a,b) for a in DIRECTIONS for b in DIRECTIONS
                 if sum(x*y for x,y in zip(DIRECTIONS[a],DIRECTIONS[b])) == 0]
        # Pre-existing attempts do not consume this newly declared step's budget.
        for first, second in pairs[:16]:
            observe(req, choice(skill_id="turn_force_two_leg_v1", direction_id=first,
                                second_direction_id=second))
            req["episode_feedback"] = req["episode_feedback"][-12:]
            req["evidence_ids"] = [r["evidence_id"] for r in req["episode_feedback"]]
        anchor = build_anchor(req, step)["anchor"]
        found = []
        for _ in range(8):
            selected = propose(req, step, anchor)["choice"]
            self.assertEqual(selected["skill_id"], "turn_force_two_leg_v1")
            found.append((selected["direction_id"], selected["second_direction_id"]))
            observe(req, selected)
            req["episode_feedback"] = req["episode_feedback"][-12:]
            req["evidence_ids"] = [r["evidence_id"] for r in req["episode_feedback"]]
        self.assertEqual(set(found), set(pairs[16:]))
        self.assertEqual(propose(req, step, anchor)["reason"], "strategy_attempt_budget_exhausted")

    def test_stability_failure_adds_reported_body_preserving_existing_support(self):
        req = observe(state(), choice(support_ids=["b"]), stability=True, reported=["c"])
        step = strategy(); ref = reference(req)
        anchor = build_anchor(req, step, ref)["anchor"]
        result = propose(req, step, anchor, ref)
        self.assertEqual(result["choice"]["support_ids"], ["b", "c"])
        self.assertEqual(result["choice"]["direction_id"], "xp")
        self.assertEqual(result["diagnostics"]["reported_parts_used"], ["c"])

    def test_replace_once_may_release_only_at_anchor_then_can_only_add(self):
        req = observe(state(), choice(support_ids=["b"]), stability=True, reported=["c"])
        step = strategy(support_policy="replace_once_v1")
        built = build_anchor(req, step, reference(req))
        self.assertEqual(built["diagnostics"]["replaced_supports_at_anchor"], ["b"])
        first = propose(req, step, built["anchor"])["choice"]
        self.assertEqual(first["support_ids"], ["c"])
        observe(req, first, stability=True, reported=["d"])
        second = propose(req, step, built["anchor"], reference(req))["choice"]
        self.assertEqual(second["support_ids"], ["c", "d"])
        observe(req, second, stability=True, reported=["b"])
        self.assertIsNone(propose(req, step, built["anchor"])["choice"])

    def test_same_failed_stability_configuration_does_not_cycle_directions(self):
        req = state(); req["max_grippers"] = 1
        step = strategy(); anchor = build_anchor(req, step)["anchor"]
        first = propose(req, step, anchor)["choice"]
        observe(req, first, stability=True)
        result = propose(req, step, anchor)
        self.assertIsNone(result["choice"])
        self.assertEqual(result["diagnostics"]["stability_configuration_skips"], 1)

    def test_missing_reported_facts_uses_finite_support_combinations(self):
        req = observe(state(), choice(), stability=True)
        step = strategy(); anchor = build_anchor(req, step)["anchor"]
        self.assertEqual(anchor["support_ids"], ["b"])
        selected = propose(req, step, anchor)["choice"]
        self.assertEqual(selected["support_ids"], ["b"])
        self.assertEqual(selected["direction_id"], "xp")

    def test_path_round_deferral_is_preserved_even_after_explicit_support_replacement(self):
        req = observe(state(), choice(support_ids=["b"]))
        step = strategy(support_policy="replace_once_v1")
        anchor = build_anchor(req, step, reference(req))["anchor"]
        result = propose(req, step, anchor)
        self.assertNotEqual(result["choice"]["direction_id"], "xp")
        self.assertGreater(result["diagnostics"]["path_round_skips"], 0)
        self.assertNotIn(semantic_path_key(validate_request(req), result["choice"]),
                         partition_path_round(req)["deferred_path_keys"])

    def test_attempt_and_iteration_limits_are_independent_and_fail_locally(self):
        req = observe(state(), choice())
        step = strategy(max_attempts=2)
        anchor = build_anchor(req, step, max_iterations=1)["anchor"]
        result = propose(req, step, anchor, max_iterations=1)
        self.assertEqual(result["reason"], "strategy_iteration_limit")
        self.assertEqual(result["diagnostics"]["iterations"], 1)
        for bad in (0, 4097, True):
            with self.assertRaises(ValueError): build_anchor(req, step, max_iterations=bad)

    def test_late_or_unbound_new_observation_cannot_drive_next_action(self):
        req, step = state(), strategy()
        anchor = build_anchor(req, step)["anchor"]
        first = propose(req, step, anchor)["choice"]
        observe(req, first, stability=True, reported=["c"], late=True)
        self.assertEqual(propose(req, step, anchor)["reason"], "current_failure_observation_unavailable")

    def test_uncertain_physical_result_does_not_silently_release_known_supports(self):
        for status, late in (("ERROR", False), ("UNKNOWN", True)):
            req = observe(state(), choice(support_ids=["b"]), late=late)
            req["episode_feedback"][-1]["status"] = status
            req["path_stage_observations"][-1]["status"] = status
            built = build_anchor(req, strategy(), reference(req))
            self.assertEqual(built["anchor"]["support_ids"], ["b"])
            self.assertEqual(built["diagnostics"]["support_seed"], "latest_bound_command")
            self.assertEqual(built["diagnostics"]["reported_parts_used"], [])

    def test_missing_or_mismatched_reference_never_becomes_implicit_empty(self):
        req = observe(state(), choice(support_ids=["b"]))
        ref = reference(req); ref["choice"]["support_ids"] = []
        result = build_anchor(req, strategy(), ref)
        self.assertIsNone(result["anchor"])
        self.assertEqual(result["reason"], "support_reference_not_latest_command")
        req["episode_feedback"] = []
        req["evidence_ids"] = []
        self.assertIsNone(build_anchor(req, strategy())["anchor"])

    def test_stale_anchor_or_changed_history_is_rejected_without_mutation(self):
        req, step = state(), strategy()
        before = deepcopy((req, step))
        anchor = build_anchor(req, step)["anchor"]
        first = propose(req, step, anchor)
        self.assertEqual((req, step), before)
        self.assertFalse(first["diagnostics"]["physical_success_known"])
        stale = deepcopy(req); stale["state_id"] = "another-state"
        self.assertEqual(propose(stale, step, anchor)["reason"], "strategy_anchor_scope_mismatch")
        changed = deepcopy(anchor); changed["support_ids"] = ["d"]
        self.assertEqual(propose(req, step, changed)["reason"], "strategy_anchor_scope_mismatch")

    def test_registered_skill_is_delegated_without_program_execution(self):
        req, step = state(), strategy(procedure="registered_skill")
        self.assertEqual(build_anchor(req, step)["reason"], "registered_skill_requires_runtime")
        self.assertEqual(propose(req, step, None)["reason"], "registered_skill_requires_runtime")

    def test_bounded_support_search_requires_model_selected_fixed_procedure(self):
        with self.assertRaisesRegex(ValueError, "bounded_support_search_requires_fixed_procedure"):
            build_anchor(state(), strategy(procedure="registered_skill",
                                          support_policy="bounded_support_search_v1"))
        for policy in ("preserve_add_v1", "replace_once_v1"):
            req, step = state(), strategy(support_policy=policy)
            anchor = build_anchor(req, step)["anchor"]
            first = propose(req, step, anchor)["choice"]
            observe(req, first, stability=True, reported=["b", "c"])
            full = propose(req, step, anchor)["choice"]
            self.assertEqual(full["support_ids"], ["b", "c"])
            observe(req, full, stability=True, reported=["d"])
            result = propose(req, step, anchor)
            self.assertIsNone(result["choice"])
            self.assertFalse(result["diagnostics"]["support_replacement_authorized"])

    def test_bounded_search_replaces_only_after_bound_failure_and_keeps_pose_part_path(self):
        req, step = state(), strategy(support_policy="bounded_support_search_v1", max_attempts=6)
        anchor = build_anchor(req, step)["anchor"]
        first = propose(req, step, anchor)
        self.assertEqual(first["choice"]["support_ids"], [])
        self.assertFalse(first["diagnostics"]["support_replacement_authorized"])
        observe(req, first["choice"], stability=True, reported=["b", "c"])
        full = propose(req, step, anchor)["choice"]
        self.assertEqual(full["support_ids"], ["b", "c"])
        observe(req, full, stability=True, reported=["d"])
        result = propose(req, step, anchor, reference(req))
        self.assertEqual(result["choice"]["support_ids"], ["b", "d"])
        self.assertEqual([result["choice"][k] for k in ("part_id", "pose_id", "direction_id")],
                         [full[k] for k in ("part_id", "pose_id", "direction_id")])
        diag = result["diagnostics"]
        self.assertTrue(diag["support_replacement_authorized"])
        self.assertEqual(diag["prior_support_ids"], ["b", "c"])
        self.assertEqual(diag["proposed_support_ids"], ["b", "d"])
        self.assertEqual(diag["released_support_ids"], ["c"])
        self.assertEqual(diag["support_replacement_source_evidence_id"], req["episode_feedback"][-1]["evidence_id"])
        self.assertFalse(diag["support_replacement_source_observation"]["deadline_exceeded"])
        self.assertFalse(diag["physical_success_known"])
        self.assertEqual(diag["physics_calls"], 0)

    def test_bounded_search_initial_reference_supports_are_preserved_without_stability_failure(self):
        for status, late in (("UNKNOWN", False), ("ERROR", False), ("UNKNOWN", True)):
            req = observe(state(), choice(support_ids=["b", "c"]), late=late)
            req["episode_feedback"][-1]["status"] = status
            req["path_stage_observations"][-1]["status"] = status
            step = strategy(support_policy="bounded_support_search_v1")
            built = build_anchor(req, step, reference(req))
            self.assertEqual(built["anchor"]["support_ids"], ["b", "c"])
            result = propose(req, step, built["anchor"], reference(req))
            self.assertEqual(result["choice"]["support_ids"], ["b", "c"])
            self.assertFalse(result["diagnostics"]["support_replacement_authorized"])

    def test_bounded_search_revalidates_anchor_replacement_authority_before_proposal(self):
        req = observe(state(), choice(support_ids=["b", "c"]), stability=True, reported=["d"])
        step = strategy(support_policy="bounded_support_search_v1")
        built = build_anchor(req, step, reference(req))
        self.assertEqual(built["anchor"]["support_ids"], ["b", "d"])
        req["episode_feedback"][-1]["deadline_exceeded"] = True
        req["path_stage_observations"][-1]["deadline_exceeded"] = True
        result = propose(req, step, built["anchor"], reference(req))
        self.assertIsNone(result["choice"])
        self.assertEqual(result["reason"], "support_replacement_observation_unavailable")
        self.assertFalse(result["diagnostics"]["support_replacement_authorized"])

    def test_bounded_search_does_not_reuse_older_failure_after_latest_error(self):
        req = observe(state(), choice(support_ids=["b", "c"]), stability=True, reported=["d"])
        observe(req, choice(support_ids=["b", "c"], direction_id="yp"))
        req["episode_feedback"][-1]["status"] = "ERROR"
        req["path_stage_observations"][-1]["status"] = "ERROR"
        built = build_anchor(req, strategy(support_policy="bounded_support_search_v1"), reference(req))
        self.assertIsNone(built["anchor"])
        self.assertEqual(built["reason"], "local_support_options_exhausted")
        self.assertFalse(built["diagnostics"]["support_replacement_authorized"])

    def test_bounded_search_blocks_after_late_error_missing_or_unbound_feedback(self):
        req, step = state(), strategy(support_policy="bounded_support_search_v1")
        anchor = build_anchor(req, step)["anchor"]
        full = choice(support_ids=["b", "c"])
        observe(req, full, stability=True, reported=["d"])
        for mutation in ("late", "error", "missing", "unbound", "unknown_stability"):
            altered = deepcopy(req)
            if mutation == "late":
                altered["episode_feedback"][-1]["deadline_exceeded"] = True
                altered["path_stage_observations"][-1]["deadline_exceeded"] = True
            elif mutation == "error":
                altered["episode_feedback"][-1]["status"] = "ERROR"
                altered["path_stage_observations"][-1]["status"] = "ERROR"
            elif mutation == "missing":
                altered["episode_feedback"] = []
                altered["evidence_ids"] = []
            elif mutation == "unbound":
                altered["episode_feedback"][-1]["selected_pose_matrix"][0][3] = 10
            else:
                altered["episode_feedback"][-1]["stability_success"] = None
            with self.subTest(mutation=mutation):
                result = propose(altered, step, anchor)
                self.assertIsNone(result["choice"])
                self.assertEqual(result["reason"], "current_failure_observation_unavailable")
                self.assertFalse(result["diagnostics"]["support_replacement_authorized"])

    def test_bounded_search_cannot_use_other_pose_or_part_failure_to_release(self):
        req = observe(state(), choice(support_ids=["b", "c"]), stability=True, reported=["d"])
        rotated = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        req["pose_catalog"].append(dict(pose_id="rotated", matrix=rotated))
        for part in ("a", "d"):
            step = strategy(part_id=part, support_policy="bounded_support_search_v1")
            result = build_anchor(req, step, reference(req))
            self.assertEqual(result["anchor"]["support_ids"], ["b", "c"])
            self.assertFalse(result["diagnostics"]["support_replacement_authorized"])

    def test_bounded_search_without_raw_parts_is_finite_and_respects_limits(self):
        req = observe(state(), choice(support_ids=["b", "c"]), stability=True)
        step = strategy(support_policy="bounded_support_search_v1", max_attempts=1)
        built = build_anchor(req, step, reference(req))
        self.assertEqual(built["anchor"]["support_ids"], [])
        selected = propose(req, step, built["anchor"])["choice"]
        observe(req, selected, stability=True)
        self.assertEqual(propose(req, step, built["anchor"])["reason"], "strategy_attempt_budget_exhausted")
        limited = build_anchor(req, step, reference(req), max_iterations=1)
        self.assertIsNone(limited["anchor"])
        self.assertEqual(limited["reason"], "strategy_iteration_limit")

    def test_bounded_two_leg_search_keeps_two_leg_successful_path_during_replacement(self):
        req, step = state(), strategy(procedure="two_leg_search_v1", support_policy="bounded_support_search_v1")
        anchor = build_anchor(req, step)["anchor"]
        first = propose(req, step, anchor)["choice"]
        observe(req, first, stability=True, reported=["b", "c"])
        full = propose(req, step, anchor)["choice"]
        observe(req, full, stability=True, reported=["d"])
        selected = propose(req, step, anchor)["choice"]
        self.assertEqual(selected["support_ids"], ["b", "d"])
        for key in ("part_id", "pose_id", "skill_id", "direction_id", "second_direction_id"):
            self.assertEqual(selected[key], full[key])
        self.assertEqual(len(selected["support_ids"]), req["max_grippers"]-1)
        self.assertNotIn(req["base_part"], selected["support_ids"])
        self.assertNotIn(step["part_id"], selected["support_ids"])

    def test_transition_orientation_reference_cannot_hide_current_commanded_supports(self):
        req = observe(state(), choice(support_ids=["b", "c"]))
        prior = dict(selected_pose_matrix=IDENTITY, remaining_parts=["old", "a", "b"],
                     choice=choice(support_ids=[]), evidence_id="old-accepted",
                     scope="orientation_only_after_transition")
        result = build_anchor(req, strategy(support_policy="bounded_support_search_v1"), prior)
        self.assertIsNone(result["anchor"])
        self.assertEqual(result["reason"], "orientation_only_reference_transition_unbound")


if __name__ == "__main__":
    unittest.main()
