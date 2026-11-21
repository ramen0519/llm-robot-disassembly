"""Offline contract fixtures and read-only replay; no new API/physics evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from orchestrator.skill_behavior_contract import (
    PREDICATE_DESCRIPTIONS, audit_behavior_claims, behavior_claim_schema,
    behavior_examples, program_description,
)
from orchestrator.skill_program import PREDICATES, program_hash, step_program
from tests.test_skill_program import append_observation, program, request, rule
from tests.test_skill_refinement import feedback


ROOT = Path(__file__).resolve().parents[1]


def prediction(items, *, part="a", direction="zp", relation="no_observed_pass", index=0):
    return [{"request_id": items[0]["current_observation_request"]["request_id"],
             "predicted_rule_index": index,
             "predicted_choice": {"part_id": part, "pose_id": "pose_current",
                                  "skill_id": "single_force_v1", "direction_id": direction,
                                  "second_direction_id": None, "support_ids": []},
             "relation_to_observed_pass": relation}]


def passed(items=None):
    items = deepcopy(items if items is not None else feedback(path=True, stability=True))
    items[0]["recovery"].update(status="PASS", path_success=True, stability_success=True)
    return items


def rename_request(item, identifier):
    result = deepcopy(item)
    result["current_observation_request"]["request_id"] = identifier
    if result["interpreter_trace"]["response"] is not None:
        result["interpreter_trace"]["response"]["request_id"] = identifier
    return result


class ProgramDescriptionTests(unittest.TestCase):
    def test_deterministic_scope_uses_ir_and_covers_all_predicates(self):
        self.assertEqual(set(PREDICATE_DESCRIPTIONS), set(PREDICATES))
        policy = program(rule(when=["last_path_failed", "last_not_accepted"]),
                         rule(when=["last_single_force"], direction="opposite_last", rank=[]))
        normalized = deepcopy(policy)
        normalized["rules"][0]["when"].reverse()
        first, second = program_description(policy), program_description(normalized)
        self.assertEqual(first, second)
        self.assertEqual(first["program_sha256"], program_hash(policy))
        self.assertEqual(len(first["rules"]), 2)
        self.assertIn("last_single_force", first["rules"][1]["rendered_text"])
        self.assertIn("opposite_last", first["rules"][1]["rendered_text"])
        self.assertEqual(first["raw_model_prose_status"], "unverified")
        self.assertFalse(first["physics_validated"])

    def test_empty_rank_and_first_return_do_not_claim_an_optimum_or_next_step(self):
        result = program_description(program(rule(rank=[])))
        text = result["rendered_text"]
        self.assertIn("FIRST canonical_filtered", text)
        self.assertIn("not a geometry-ranked optimum", text)
        self.assertIn("Later rules are not physical follow-up steps", text)
        self.assertIn("scan-limit result abstains", text)
        self.assertEqual(result["rules"][0]["ranking"], [])

    def test_mover_support_and_pose_semantics_are_explicit(self):
        result = program_description(program(rule(part="observed_initial_disconnected",
            support="contains_observed_initial_disconnected")))
        values = {row["field"]: row["meaning"] for row in result["rules"][0]["selectors"]}
        self.assertIn("mover", values["part"])
        self.assertIn("at least one", values["support"])
        self.assertIn("not necessarily all", values["support"])
        self.assertIn("pose matrix", values["pose"])

    def test_rank_order_and_missing_metric_semantics_preserved(self):
        policy = program(rule(rank=[dict(metric="part_volume", order="asc"),
                                    dict(metric="support_count", order="desc")]))
        result = program_description(policy)
        self.assertEqual([r["metric"] for r in result["rules"][0]["ranking"]],
                         ["part_volume", "support_count"])
        self.assertIn("Missing metric values skip", result["rendered_text"])
        with self.assertRaises(ValueError):
            program_description({**policy, "card": "invented claim"})


class BehaviorClaimTests(unittest.TestCase):
    def test_correct_concrete_prediction_is_only_offline_contract_success(self):
        items = feedback()
        result = audit_behavior_claims(program(), items, prediction(items))
        self.assertTrue(result["accepted_for_fresh_validation"])
        self.assertEqual(result["cases"][0]["expected_choice"]["part_id"], "a")
        self.assertEqual(result["physical_calls"], 0)
        self.assertFalse(result["physics_validated"])
        self.assertFalse(result["registered"])
        self.assertFalse(result["physics_verdict_reused"])
        self.assertIsNone(result["failure_cause"])

    def test_correct_preserved_pass_is_not_a_new_physical_pass(self):
        items = passed()
        result = audit_behavior_claims(program(), items,
                                      prediction(items, relation="preserve_observed_pass"))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["cases"][0]["previous_pass_differences"][0]["different_fields"], [])
        self.assertFalse(result["physics_validated"])

    def test_declared_change_to_pass_is_allowed_and_difference_reported(self):
        items = passed()
        policy = program(rule(direction="opposite_last"))
        claim = prediction(items, direction="xn", relation="change_observed_pass")
        result = audit_behavior_claims(policy, items, claim)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["cases"][0]["previous_pass_differences"][0]["different_fields"],
                         ["direction_id"])

    def test_false_preservation_and_incorrect_rule_or_choice_rejected(self):
        items = passed()
        policy = program(rule(direction="opposite_last"))
        wrong = prediction(items, relation="preserve_observed_pass", index=1)
        result = audit_behavior_claims(policy, items, wrong)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reasons"], ["behavior_claim_mismatch"])
        self.assertEqual(result["cases"][0]["mismatched_fields"],
                         ["rule_index", "choice", "observed_pass_relation"])

    def test_no_observed_pass_does_not_invent_one(self):
        items = feedback()
        for relation in ("preserve_observed_pass", "change_observed_pass"):
            result = audit_behavior_claims(program(), items, prediction(items, relation=relation))
            self.assertFalse(result["accepted"])
            self.assertEqual(result["cases"][0]["expected_relation_to_observed_pass"], "no_observed_pass")

    def test_matching_without_action_keeps_interpreter_rule_index(self):
        policy = program(rule(primitive="turn_force_two_leg_v1", second_direction="perpendicular_last"))
        items = feedback(policy)
        claim = prediction(items)
        claim[0]["predicted_choice"] = None
        result = audit_behavior_claims(policy, items, claim)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["cases"][0]["expected_rule_index"], 0)
        self.assertEqual(result["cases"][0]["not_applicable_reason"], "no_matching_untried_action")

    def test_no_trigger_uses_null_index_and_choice(self):
        policy = program(rule(when=["last_stability_failed"]))
        items = feedback(policy)
        claim = prediction(items, index=None)
        claim[0]["predicted_choice"] = None
        self.assertTrue(audit_behavior_claims(policy, items, claim)["accepted"])

    def test_unique_request_dedup_collects_multiple_bound_observations(self):
        items = feedback() + passed()
        examples = behavior_examples(items)
        self.assertEqual(len(examples), 1)
        self.assertEqual(len(examples[0]["observed_pass_choices"]), 1)
        self.assertTrue(audit_behavior_claims(program(), items,
            prediction(items, relation="preserve_observed_pass"))["accepted"])

    def test_distinct_replay_ids_share_pass_only_with_same_context(self):
        items = feedback() + [rename_request(passed()[0], "fixture:other")]
        examples = behavior_examples(items)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["context_sha256"], examples[1]["context_sha256"])
        self.assertEqual(examples[0]["observed_pass_choices"], examples[1]["observed_pass_choices"])
        # A changed geometry observation is a different condition, even at the
        # same state label. Historical PASS does not silently migrate there.
        items[1]["current_observation_request"]["geometry"]["parts"]["a"]["volume"] = 9
        examples = behavior_examples(items)
        self.assertEqual(examples[0]["observed_pass_choices"], [])
        self.assertNotEqual(examples[0]["context_sha256"], examples[1]["context_sha256"])

    def test_conflicting_reused_request_id_rejected(self):
        items = feedback() + passed()
        items[1]["current_observation_request"]["geometry"]["parts"]["a"]["volume"] = 9
        result = audit_behavior_claims(program(), items, prediction(items))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["errors"], ["conflicting_behavior_request_id"])

    def test_exact_coverage_rejects_missing_extra_duplicate_and_unknown_claims(self):
        items = feedback() + [rename_request(passed()[0], "fixture:other")]
        first = prediction(items, relation="preserve_observed_pass")[0]
        second = {**deepcopy(first), "request_id": "fixture:other"}
        self.assertTrue(audit_behavior_claims(program(), items, [second, first])["accepted"])
        for claims in ([], [first], [first, first], [first, second, second],
                       [first, {**second, "request_id": "unknown"}], None):
            with self.subTest(claims=claims):
                self.assertFalse(audit_behavior_claims(program(), items, claims)["accepted"])

    def test_unknown_fields_and_illegal_concrete_choices_never_accepted(self):
        items = feedback()
        for mutate in (
            lambda c: c.update(physics_pass=True),
            lambda c: c.update(predicted_rule_index=True),
            lambda c: c.update(predicted_rule_index=-1),
            lambda c: c.update(predicted_rule_index=None),
            lambda c: c.update(relation_to_observed_pass="PASS"),
            lambda c: c["predicted_choice"].update(evidence_ids=[]),
            lambda c: c["predicted_choice"].update(part_id="foreign-source-part"),
            lambda c: c["predicted_choice"].update(support_ids=["base"]),
            lambda c: c["predicted_choice"].update(second_direction_id="xp"),
        ):
            claims = prediction(items)
            mutate(claims[0])
            self.assertFalse(audit_behavior_claims(program(), items, claims)["accepted"])

    def test_non_debug_unbound_late_and_mock_feedback_rejected(self):
        for mutate in (
            lambda i: i.update(phase="admission"),
            lambda i: i["trigger"].update(backend="mock"),
            lambda i: i["recovery"].update(deadline_exceeded=True),
            lambda i: i["interpreter_trace"].update(scoped_feedback_evidence_id="foreign"),
        ):
            items = feedback()
            mutate(items[0])
            self.assertFalse(audit_behavior_claims(program(), items, prediction(items))["accepted"])

    def test_first_generation_empty_examples_is_explicit_no_evidence(self):
        self.assertEqual(behavior_examples([]), [])
        schema = behavior_claim_schema([])
        self.assertEqual((schema["minItems"], schema["maxItems"]), (0, 0))
        result = audit_behavior_claims(program(), [], [])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["example_count"], 0)
        self.assertFalse(result["physics_validated"])
        self.assertFalse(audit_behavior_claims(program(), [], [{}])["accepted"])

    def test_schema_has_bound_ids_all_required_and_no_verdict_field(self):
        items = feedback()
        schema = behavior_claim_schema(items)
        self.assertEqual(schema["minItems"], 1)
        claim = schema["items"]["anyOf"][0]
        self.assertFalse(claim["additionalProperties"])
        self.assertEqual(set(claim["required"]), set(prediction(items)[0]))
        self.assertEqual(claim["properties"]["request_id"]["enum"], ["fixture:r1"])
        choice = claim["properties"]["predicted_choice"]["anyOf"][0]
        self.assertEqual(choice["properties"]["part_id"]["enum"], ["a", "b"])
        self.assertNotIn("physics_pass", choice["properties"])
        self.assertNotIn("evidence_ids", choice["properties"])

    def test_functions_do_not_mutate_inputs(self):
        items, policy = feedback(), program()
        claims = prediction(items)
        before = deepcopy((items, policy, claims))
        behavior_examples(items)
        behavior_claim_schema(items)
        program_description(policy)
        audit_behavior_claims(policy, items, claims)
        self.assertEqual((items, policy, claims), before)


class HistoricalBehaviorRegressionTests(unittest.TestCase):
    def test_actual_candidate_02_empty_rank_does_not_preserve_recorded_part_4_pass(self):
        path = ROOT / "runs/initial_contact_skill_followup_20261019/live_01/candidate_02"
        packet = json.loads((path / "generation_packet.json").read_text(encoding="utf-8"))
        proposal = json.loads((path / "provider_proposal.json").read_text(encoding="utf-8"))
        items, policy = packet["validation_feedback"], proposal["program"]
        examples = behavior_examples(items)
        self.assertEqual(len(examples), 3)
        self.assertEqual(len({e["context_sha256"] for e in examples}), 1)
        self.assertTrue(all(e["observed_pass_choices"][0]["part_id"] == "4" for e in examples))
        claims = [{"request_id": row["request_id"], "predicted_rule_index": 0,
                   "predicted_choice": deepcopy(row["observed_pass_choices"][0]),
                   "relation_to_observed_pass": "preserve_observed_pass"} for row in examples]
        result = audit_behavior_claims(policy, items, claims)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reasons"], ["behavior_claim_mismatch"])
        for row in result["cases"]:
            self.assertEqual(row["expected_choice"]["part_id"], "1")
            self.assertEqual(row["expected_choice"]["direction_id"], "zp")
            self.assertEqual(row["expected_relation_to_observed_pass"], "change_observed_pass")
            self.assertEqual(row["previous_pass_differences"][0]["different_fields"], ["part_id"])
            self.assertIsNone(row["failure_cause"])
        # This is retrospective CPU replay; the old response had no structured
        # behavior_claims. Do not relabel it as a new model response or physics.
        self.assertNotIn("behavior_claims", proposal)
        self.assertEqual(result["physical_calls"], 0)
        text = program_description(policy)["rendered_text"]
        self.assertIn("last_turn_force", text)
        self.assertIn("last_single_force", text)
        self.assertIn("FIRST canonical_filtered", text)


if __name__ == "__main__":
    unittest.main()
