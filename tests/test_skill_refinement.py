"""Synthetic contract fixtures only; no API, filesystem experiment or physics."""
from copy import deepcopy
import unittest

from orchestrator.skill_program import program_hash, step_program
from orchestrator.skill_refinement import (
    audit_program_novelty, audit_revision, summarize_feedback, summarize_motion_alignment,
)
from tests.test_skill_program import IDENTITY, append_observation, program, request, rule


def feedback(prior=None, *, path=False, stability=None):
    prior = prior or program()
    req = append_observation(request())
    step = step_program(prior, req)
    trigger = deepcopy(req["episode_feedback"][-1])
    trigger.update(backend="ASAP_RedMax", chosen=trigger.pop("choice"))
    item = dict(case_id="synthetic_debug", phase="debug", current_observation_request=req,
                interpreter_trace=step, trigger=trigger)
    if step["response"] is not None:
        item["recovery"] = dict(backend="ASAP_RedMax", deadline_exceeded=False,
            chosen=deepcopy(step["response"]["choice"]), path_success=path,
            stability_success=stability, status="UNKNOWN")
    return [item]


class SkillRefinementTests(unittest.TestCase):
    def test_identical_failed_action_is_rejected_without_physics(self):
        result = audit_revision(program(), feedback())
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reasons"], ["unchanged_failed_action"])
        self.assertEqual(result["physical_calls"], 0)
        self.assertFalse(result["candidate_pruning"])

    def test_changed_program_hash_with_same_first_choice_is_rejected(self):
        revised = program(rule(), rule(direction="opposite_last"))
        self.assertIn("unchanged_failed_action", audit_revision(revised, feedback())["reasons"])

    def test_changed_action_is_only_unvalidated_proposal(self):
        result = audit_revision(program(rule(direction="opposite_last")), feedback())
        self.assertTrue(result["accepted"])
        self.assertFalse(result["physics_validated"])
        self.assertFalse(result["registered"])

    def test_citation_changes_do_not_make_action_new(self):
        items = feedback()
        items[0]["recovery"]["chosen"]["evidence_ids"] = []
        self.assertIn("unchanged_failed_action", audit_revision(program(), items)["reasons"])

    def test_stability_failure_is_a_bound_failed_attempt(self):
        self.assertIn("unchanged_failed_action", audit_revision(
            program(), feedback(path=True, stability=False))["reasons"])

    def test_unknown_without_known_stage_failure_does_not_ban_action(self):
        self.assertTrue(audit_revision(program(), feedback(path=None))["accepted"])

    def test_error_does_not_become_physics_failure(self):
        items = feedback()
        items[0]["recovery"]["status"] = "ERROR"
        self.assertTrue(audit_revision(program(), items)["accepted"])

    def test_prior_success_does_not_become_failed_action(self):
        items = feedback(path=True, stability=True)
        items[0]["recovery"]["status"] = "PASS"
        self.assertTrue(audit_revision(program(), items)["accepted"])

    def test_same_no_action_rejected(self):
        prior = program(rule(primitive="turn_force_two_leg_v1", second_direction="perpendicular_last"))
        result = audit_revision(prior, feedback(prior))
        self.assertEqual(result["reasons"], ["no_recovery_proposal", "unchanged_no_action"])

    def test_prior_no_action_can_change_to_legal_unvalidated_action(self):
        prior = program(rule(primitive="turn_force_two_leg_v1", second_direction="perpendicular_last"))
        self.assertTrue(audit_revision(program(), feedback(prior))["accepted"])

    def test_unrelated_predicates_cannot_bypass_failed_action_gate(self):
        dead = program(rule(when=["has_unstable_parts", "last_path_failed", "last_single_force"]))
        result = audit_revision(dead, feedback())
        self.assertFalse(result["accepted_for_fresh_validation"])
        self.assertEqual(result["reasons"], ["no_recovery_proposal"])
        self.assertFalse(result["candidate_pruning"])

    def test_real_no_trigger_policy_is_retained_for_next_revision(self):
        dead = program(rule(when=["has_unstable_parts", "last_path_failed", "last_single_force"]))
        items = feedback(dead)
        items[0]["observed_program"] = dead
        summary = summarize_feedback(items, previous_program=dead)
        self.assertTrue(summary["valid"])
        diagnosis = summary["cases"][0]["rule_predicate_diagnostics"][0]
        self.assertEqual(diagnosis["unsatisfied_predicates"], ["has_unstable_parts"])
        self.assertEqual(diagnosis["interpreter_inconsistent_conjunctions"],
                         ["unstable_diagnostics_require_path_passed"])
        self.assertIsNone(diagnosis["physical_failure_cause"])
        self.assertTrue(audit_revision(program(), items)["accepted_for_fresh_validation"])
        self.assertIn("unchanged_no_action", audit_revision(dead, items)["reasons"])

    def test_inconsistent_no_trigger_record_is_rejected(self):
        dead = program(rule(when=["last_stability_failed"]))
        for changes in ({"rule_index": 0}, {"not_applicable_reason": "scan_limit"},
                        {"triggered": True}):
            items = feedback(dead)
            items[0]["interpreter_trace"].update(changes)
            self.assertFalse(audit_revision(program(), items)["accepted"])

    def test_admission_or_transfer_feedback_rejected(self):
        for key in ("phase", "validation_phase"):
            for phase in ("admission", "transfer"):
                with self.subTest(key=key, phase=phase):
                    items = feedback()
                    items[0][key] = phase
                    self.assertFalse(audit_revision(program(), items)["accepted"])
                    self.assertFalse(summarize_feedback(items)["valid"])

    def test_missing_or_late_deadline_not_used_as_failure(self):
        for stage in ("trigger", "recovery"):
            for flag in (None, True):
                items = feedback()
                if flag is None:
                    items[0][stage].pop("deadline_exceeded")
                else:
                    items[0][stage]["deadline_exceeded"] = flag
                self.assertIn("invalid_debug_feedback", audit_revision(program(), items)["reasons"])

    def test_mock_backend_rejected(self):
        items = feedback()
        items[0]["recovery"]["backend"] = "mock"
        self.assertIn("invalid_debug_feedback", audit_revision(program(), items)["reasons"])

    def test_raw_recovery_choice_must_match_saved_proposal(self):
        items = feedback()
        items[0]["recovery"]["chosen"]["direction_id"] = "xn"
        self.assertIn("invalid_debug_feedback", audit_revision(program(), items)["reasons"])

    def test_stale_response_and_scope_are_rejected(self):
        for mutate in (
            lambda f: f[0]["interpreter_trace"]["response"].update(state_id="stale"),
            lambda f: f[0]["current_observation_request"]["episode_feedback"][-1].update(remaining_parts=["a"]),
            lambda f: f[0]["interpreter_trace"].update(scoped_feedback_evidence_id="other"),
            lambda f: f[0]["trigger"].update(path_success=True),
        ):
            items = feedback()
            mutate(items)
            self.assertFalse(audit_revision(program(), items)["accepted"])

    def test_nonfinite_or_invalid_stage_values_rejected(self):
        for value in (0, "false", float("nan")):
            items = feedback()
            items[0]["recovery"]["path_success"] = value
            self.assertFalse(audit_revision(program(), items)["accepted"])

    def test_diagnoses_shadowed_rule_without_recommending_action(self):
        prior = program(rule(), rule(direction="opposite_last"))
        summary = summarize_feedback(feedback(prior), prior)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["cases"][0]["later_available_but_not_reached"], [{
            "rule_index": 1, "physical_execution": False,
            "reason": "earlier_rule_already_returned_one_action"}])
        self.assertIsNone(summary["failure_cause"])
        self.assertEqual(summary["physical_calls"], 0)

    def test_summary_previous_program_must_match_trace(self):
        self.assertFalse(summarize_feedback(feedback(), program(rule(direction="opposite_last")))["valid"])

    def test_cumulative_feedback_binds_each_observed_program(self):
        first, second = program(), program(rule(direction="opposite_last"))
        old, new = feedback(first), feedback(second)
        old[0]["observed_program"], new[0]["observed_program"] = first, second
        summary = summarize_feedback(old + new, previous_program=second)
        self.assertTrue(summary["valid"])
        self.assertEqual(len(summary["cases"]), 2)
        old[0]["observed_program"] = second
        self.assertFalse(summarize_feedback(old + new, previous_program=second)["valid"])
        self.assertIn("invalid_debug_feedback", audit_revision(second, old)["reasons"])

    def test_fresh_validation_alias_matches_acceptance(self):
        for proposal in (program(), program(rule(direction="opposite_last")), {}):
            result = audit_revision(proposal, feedback())
            self.assertEqual(result["accepted"], result["accepted_for_fresh_validation"])

    def test_inputs_not_mutated(self):
        items, policy = feedback(), program()
        old = deepcopy((items, policy))
        audit_revision(policy, items)
        summarize_feedback(items, policy)
        self.assertEqual((items, policy), old)

    def test_invalid_input_never_accepted(self):
        for items in (None, [], {}, [None], [{}], feedback() * 33):
            self.assertFalse(audit_revision(program(), items)["accepted"])
            self.assertFalse(summarize_feedback(items)["valid"])
        self.assertFalse(audit_revision({}, feedback())["accepted"])


class ProgramNoveltyTests(unittest.TestCase):
    def test_empty_history_allows_unvalidated_first_procedure(self):
        result = audit_program_novelty(program(), [])
        self.assertTrue(result['accepted_for_fresh_validation'])
        self.assertEqual(result['program_hash'], program_hash(program()))
        self.assertEqual(result['prior_program_count'], 0)
        self.assertFalse(result['behavioral_novelty_proven'])
        self.assertFalse(result['physics_validated'])
        self.assertFalse(result['registered'])

    def test_exact_program_rejected_even_when_debug_action_previously_passed(self):
        policy = program()
        items = feedback(path=True, stability=True)
        items[0]['recovery']['status'] = 'PASS'
        self.assertTrue(audit_revision(policy, items)['accepted'])
        result = audit_program_novelty(policy, [program_hash(policy)])
        self.assertFalse(result['accepted_for_fresh_validation'])
        self.assertEqual(result['reasons'], ['unchanged_program_no_new_procedure'])
        self.assertTrue(result['exact_program_duplicate'])
        self.assertEqual(result['matched_prior_program_hash'], program_hash(policy))
        self.assertEqual(result['physical_calls'], 0)
        self.assertFalse(result['candidate_pruning'])
        self.assertFalse(result['physics_verdict_reused'])

    def test_predicate_and_object_key_order_cannot_disguise_duplicate(self):
        policy = program(rule(when=['last_path_failed', 'last_not_accepted']))
        reordered = dict(reversed(list(deepcopy(policy).items())))
        reordered['rules'][0]['when'].reverse()
        reordered['rules'][0]['select'] = dict(reversed(list(reordered['rules'][0]['select'].items())))
        result = audit_program_novelty(reordered, [program_hash(policy)])
        self.assertEqual(result['reasons'], ['unchanged_program_no_new_procedure'])

    def test_card_rename_cannot_change_program_only_identity(self):
        original = dict(card=dict(name='old name'), program=program())
        renamed = deepcopy(original)
        renamed['card']['name'] = 'different name and claim'
        result = audit_program_novelty(renamed['program'], [program_hash(original['program'])])
        self.assertFalse(result['accepted'])
        # Metadata is not a permitted way to alter executable identity either.
        renamed['program']['name'] = renamed['card']['name']
        self.assertEqual(audit_program_novelty(renamed['program'], [])['reasons'],
                         ['invalid_program_novelty_input'])

    def test_successful_debug_behavior_can_stay_while_other_condition_changes(self):
        prior = program()
        revised = program(rule(), rule(when=['last_stability_failed'],
            direction='same_as_last', support='nonempty', rank=[]))
        req = append_observation(request())
        self.assertEqual(step_program(prior, req)['response']['choice'],
                         step_program(revised, req)['response']['choice'])
        other = append_observation(request(), path=True, stability=False)
        self.assertIsNone(step_program(prior, other)['response'])
        self.assertIsNotNone(step_program(revised, other)['response'])
        items = feedback(path=True, stability=True)
        items[0]['recovery']['status'] = 'PASS'
        self.assertTrue(audit_revision(revised, items)['accepted'])
        self.assertTrue(audit_program_novelty(revised, [program_hash(prior)])['accepted'])

    def test_semantic_rule_and_rank_order_are_not_normalized_away(self):
        policy = program(rule(), rule(when=['last_stability_failed']))
        changed = deepcopy(policy)
        changed['rules'].reverse()
        self.assertTrue(audit_program_novelty(changed, [program_hash(policy)])['accepted'])
        policy = program(rule(rank=[dict(metric='support_count', order='asc'),
                                    dict(metric='direction_up', order='desc')]))
        changed = deepcopy(policy)
        changed['rules'][0]['rank'].reverse()
        self.assertTrue(audit_program_novelty(changed, [program_hash(policy)])['accepted'])

    def test_prior_hashes_require_unique_lowercase_sha256_strings(self):
        identity = program_hash(program())
        invalid = (None, {}, (), identity, [None], [1], [[]], [''], ['g' * 64],
                   ['A' * 64], ['a' * 63], ['a' * 65], [' ' + identity], [identity, identity])
        for history in invalid:
            with self.subTest(history=history):
                result = audit_program_novelty(program(), history)
                self.assertFalse(result['accepted'])
                self.assertEqual(result['reasons'], ['invalid_program_novelty_input'])
                self.assertEqual(result['physical_calls'], 0)
        self.assertTrue(audit_program_novelty(program(), ['0' * 64, 'f' * 64])['accepted'])

    def test_history_order_does_not_change_identity_gate(self):
        identity = program_hash(program())
        first = audit_program_novelty(program(), ['0' * 64, identity])
        second = audit_program_novelty(program(), [identity, '0' * 64])
        self.assertEqual(first, second)

    def test_invalid_program_is_rejected_and_inputs_are_not_mutated(self):
        self.assertEqual(audit_program_novelty({}, [])['reasons'], ['invalid_program_novelty_input'])
        policy, history = program(), [program_hash(program())]
        before = deepcopy((policy, history))
        result = audit_program_novelty(policy, history)
        self.assertEqual((policy, history), before)
        self.assertEqual(result['accepted'], result['accepted_for_fresh_validation'])


def source_pairs():
    return {"kind": "observed_same_state_failure_later_pass_pairs", "pairs": [{
        "reference_only": True, "physics_reuse_allowed": False, "current_part_id_mapping": None,
        "same_state": True, "same_part": True, "success_evidence_id": "synthetic_source_pass",
        "after_outcome": {"path_success": True, "stability_success": True, "accepted_for_transition": True},
        "after": {"skill_id": "turn_force_two_leg_v1", "direction_id": "zp",
                  "second_direction_id": "xp", "pose_matrix": deepcopy(IDENTITY)}}]}


class MotionAlignmentTests(unittest.TestCase):
    def policy(self):
        return program(rule(primitive="turn_force_two_leg_v1", direction="zp", second_direction="any",
                            rank=[{"metric": "direction_up", "order": "desc"}]))

    def test_reports_unconstrained_secondary_canonical_tie_not_recommendation(self):
        result = summarize_motion_alignment(self.policy(), feedback(), source_pairs())
        self.assertTrue(result["valid"])
        row = result["cases"][0]["comparisons"][0]
        self.assertEqual(row["changed_motion_fields"], ["second_direction_id"])
        self.assertEqual(row["unconstrained_changed_motion_fields"], ["second_direction_id"])
        tie = row["deterministic_resolution_evidence"][0]
        self.assertEqual(tie["resolution"], "canonical_choice_ascending")
        self.assertTrue(tie["chosen_canonical_precedes_source_field_alternative"])
        self.assertFalse(result["recommendation_generated"])
        self.assertFalse(result["transfer_inferred"])
        self.assertFalse(result["admission_gate"])
        self.assertEqual(result["physical_calls"], 0)
        self.assertIsNone(row["source_part_id_mapping"])

    def test_pose_mismatch_and_missing_source_do_not_imply_alignment(self):
        pairs = source_pairs()
        pairs["pairs"][0]["after"]["pose_matrix"][0][3] = 1
        result = summarize_motion_alignment(self.policy(), feedback(), pairs)
        self.assertEqual(result["cases"][0]["comparisons"], [])
        self.assertEqual(result["cases"][0]["unmatched_pose_pair_count"], 1)
        pairs["pairs"] = []
        result = summarize_motion_alignment(self.policy(), feedback(), pairs)
        self.assertTrue(result["valid"])
        self.assertEqual(result["cases"][0]["comparisons"], [])
        self.assertFalse(result["transfer_inferred"])

    def test_equal_motion_is_not_a_transferred_physical_pass(self):
        policy = program(rule(primitive="turn_force_two_leg_v1", direction="zp", second_direction="xp"))
        result = summarize_motion_alignment(policy, feedback(), source_pairs())
        self.assertEqual(result["cases"][0]["comparisons"][0]["changed_motion_fields"], [])
        self.assertFalse(result["physics_validated"])
        self.assertFalse(result["physics_verdict_reused"])

    def test_admission_feedback_and_missing_pass_evidence_rejected(self):
        items = feedback()
        items[0]["phase"] = "admission"
        self.assertFalse(summarize_motion_alignment(self.policy(), items, source_pairs())["valid"])
        pairs = source_pairs()
        pairs["pairs"][0]["after_outcome"]["accepted_for_transition"] = False
        self.assertFalse(summarize_motion_alignment(self.policy(), feedback(), pairs)["valid"])

    def test_inputs_unchanged_and_nonapplicable_program_has_no_alignment(self):
        policy = program(rule(when=["last_stability_failed"]))
        items, pairs = feedback(), source_pairs()
        old = deepcopy((policy, items, pairs))
        result = summarize_motion_alignment(policy, items, pairs)
        self.assertIsNone(result["cases"][0]["proposed_motion"])
        self.assertEqual(result["cases"][0]["comparisons"], [])
        self.assertEqual((policy, items, pairs), old)


if __name__ == "__main__":
    unittest.main()
