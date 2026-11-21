"""Fixture tests of recovery contracts; these are not physics executions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.repair_library import (SKILL_ID, collect_outcomes, execute,
                                         feedback_from_events, list_applicable)


def request(parts=("a", "b", "base"), run_id="fixture"):
    return {"run_id": run_id, "request_id": run_id + ":r3", "state_id": "fixture-state",
            "assembled_parts": list(parts), "candidate_ids": [p for p in parts if p != "base"],
            "base_part": "base"}


def observation(number, move, rest, status, removed=(), run_id="fixture"):
    passed = status == "PASS"
    return {"event_id": run_id + ":e" + str(number), "call_id": "v" + str(number),
            "status": status,
            "request": {"part_move": move, "parts_rest": list(rest), "parts_removed": list(removed),
                        "pose": None, "max_grippers": 2, "timeout": None},
            "raw": {"part_move": move, "base_part": "base", "feasible": passed,
                    "action": [0, 0, 1] if passed else None,
                    "parts_fix": [] if passed else None, "pose": None},
            "observed_stage": "passed" if passed else "path_search_returned_none"}


def feedback():
    return [observation(2, "a", ("b", "c", "base"), "UNKNOWN"),
            observation(4, "c", ("a", "b", "base"), "PASS")]


class RepairLibraryTests(unittest.TestCase):
    def test_real_state_change_makes_failed_candidate_applicable(self):
        cards = list_applicable(request(), feedback())
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["eligible_part_ids"], ["a"])
        self.assertEqual(cards[0]["lifecycle"], "provisional")
        self.assertFalse(cards[0]["generalization_validated"])
        self.assertIsNone(cards[0]["candidates"][0]["blocker"])

    def test_execute_preserves_all_candidates_and_does_not_mutate_inputs(self):
        req, events, order = request(), feedback(), ["b", "a"]
        before = deepcopy((req, events, order))
        result = execute(SKILL_ID, req, order, {"failed_part_id": "a"}, events)
        self.assertEqual(result["candidate_ids"], ["a", "b"])
        self.assertEqual((req, events, order), before)
        self.assertEqual(result["invocation"]["internal_verifier_calls"], 0)
        self.assertFalse(result["invocation"]["state_mutations"])
        self.assertEqual(result["invocation"]["physical_result"], "PENDING")
        self.assertEqual(result, execute(SKILL_ID, req, order, {"failed_part_id": "a"}, events))

    def test_empty_memory_or_feedback_is_not_a_repair(self):
        self.assertEqual(list_applicable(request(), []), [])

    def test_unknown_without_changed_state_is_not_applicable(self):
        self.assertEqual(list_applicable(request(("a", "b", "c", "base")), feedback()[:1]), [])

    def test_removed_part_must_have_actual_pass(self):
        events = feedback()
        events[1]["status"] = "UNKNOWN"
        events[1]["raw"]["feasible"] = False
        self.assertEqual(list_applicable(request(), events), [])

    def test_prior_pass_promotion_is_not_repair(self):
        events = feedback()
        events[0]["status"] = "PASS"
        events[0]["raw"].update(feasible=True, action=[0, 0, 1], parts_fix=[])
        self.assertEqual(list_applicable(request(), events), [])

    def test_wrong_run_ids_never_transfer(self):
        events = feedback()
        events[0]["event_id"] = "different-task:e2"
        self.assertEqual(list_applicable(request(), events), [])

    def test_raw_identity_or_feasibility_mismatch_rejected(self):
        for change in ({"part_move": "not-a"}, {"base_part": "wrong"}, {"feasible": True}):
            events = feedback()
            events[0]["raw"].update(change)
            self.assertEqual(list_applicable(request(), events), [])

    def test_error_is_not_evidence_of_physical_failure(self):
        events = feedback()
        events[0]["status"] = "ERROR"
        self.assertEqual(list_applicable(request(), events), [])

    def test_rejected_is_not_promoted_to_supported_unknown(self):
        events = feedback()
        events[0]["status"] = "REJECTED"
        self.assertEqual(list_applicable(request(), events), [])

    def test_cannot_repeat_the_same_visited_candidate_state(self):
        for status in ("UNKNOWN", "PASS"):
            events = feedback() + [observation(6, "a", ("b", "base"), status, removed=("c",))]
            self.assertEqual(list_applicable(request(), events), [])

    def test_missing_intermediate_transition_rejected(self):
        events = [observation(2, "a", ("b", "c", "d", "base"), "UNKNOWN"),
                  observation(4, "c", ("a", "b", "base"), "PASS", removed=("d",))]
        self.assertEqual(list_applicable(request(), events), [])

    def test_multiple_verified_removals_supported(self):
        events = [observation(2, "a", ("b", "c", "d", "base"), "UNKNOWN"),
                  observation(4, "d", ("a", "b", "c", "base"), "PASS"),
                  observation(6, "c", ("a", "b", "base"), "PASS", removed=("d",))]
        card = list_applicable(request(), events)[0]
        self.assertEqual(card["candidates"][0]["state_change_event_ids"], ["fixture:e4", "fixture:e6"])

    def test_inconsistent_removed_history_is_not_a_chain(self):
        events = feedback()
        events[1]["request"]["parts_removed"] = ["different"]
        self.assertEqual(list_applicable(request(), events), [])

    def test_duplicate_or_reversed_event_order_rejected(self):
        for events in (list(reversed(feedback())), [feedback()[0], feedback()[0], feedback()[1]]):
            with self.assertRaisesRegex(ValueError, "feedback_event_order"):
                list_applicable(request(), events)

    def test_request_cannot_hide_a_candidate_or_include_base(self):
        for ids in (["a"], ["a", "b", "base"]):
            req = request()
            req["candidate_ids"] = ids
            with self.assertRaisesRegex(ValueError, "complete_candidate"):
                list_applicable(req, feedback())

    def test_proposed_order_must_be_full_permutation(self):
        for order in (["a"], ["a", "a"], ["a", "foreign"]):
            with self.assertRaises(ValueError):
                execute(SKILL_ID, request(), order, {"failed_part_id": "a"}, feedback())

    def test_only_allowlisted_skill_and_exact_parameters(self):
        for skill, args in (("exec_python", {"failed_part_id": "a"}),
                            (SKILL_ID, {"failed_part_id": "a", "code": "anything"}),
                            (SKILL_ID, {"failed_part_id": "b"}), (SKILL_ID, None)):
            with self.assertRaises(ValueError):
                execute(skill, request(), ["a", "b"], args, feedback())

    def test_metadata_preserves_actual_failure_and_transition_pointers(self):
        result = execute(SKILL_ID, request(), ["b", "a"], {"failed_part_id": "a"}, feedback())
        invocation = result["invocation"]
        self.assertEqual(invocation["source_failure_event_id"], "fixture:e2")
        self.assertEqual(invocation["state_change_event_ids"], ["fixture:e4"])
        self.assertFalse(invocation["causal_benefit_demonstrated"])

    def test_actual_existing_log_can_be_read_as_scoped_evidence_without_simulation(self):
        path = ROOT / "runs/independent_ranker_20261018/physics/test_id_descending/events.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        decision = next(v for v in events if v.get("kind") == "rank_decision" and v["generation_id"] == "g4")
        preceding = [v for v in events if int(v["event_id"].rsplit(":e", 1)[1]) < 20]
        cards = list_applicable(decision["request"], feedback_from_events(preceding))
        self.assertEqual(cards[0]["eligible_part_ids"], ["2", "3"])
        result = execute(SKILL_ID, decision["request"], ["2", "0", "3"],
                         {"failed_part_id": "3"}, feedback_from_events(preceding))
        self.assertEqual(result["candidate_ids"], ["3", "2", "0"])
        # This is a read-only replay of an existing order, NOT a historical
        # claim that this newly implemented skill caused that old run's result.
        outcome = collect_outcomes([result["invocation"]], events)[0]
        self.assertEqual(outcome["status"], "PASS")
        self.assertTrue(outcome["recovery_chain_observed"])
        self.assertFalse(outcome["causal_benefit_demonstrated"])
        self.assertFalse(outcome["generalization_validated"])

    def test_collect_records_not_run_when_no_matching_worker_decision(self):
        invocation = execute(SKILL_ID, request(), ["b", "a"], {"failed_part_id": "a"}, feedback())["invocation"]
        outcome = collect_outcomes([invocation], [])[0]
        self.assertEqual(outcome["status"], "NOT_RUN")
        self.assertFalse(outcome["recovery_chain_observed"])

    def test_collection_rejects_fallback_attribution(self):
        invocation = execute(SKILL_ID, request(), ["b", "a"], {"failed_part_id": "a"}, feedback())["invocation"]
        decision = {"kind": "rank_decision", "request": request(), "event_id": "fixture:e5",
                    "generation_id": "g3", "applied_order": ["a", "b"], "fallback_reason": "budget"}
        with self.assertRaisesRegex(ValueError, "worker_did_not_apply"):
            collect_outcomes([invocation], [decision])


if __name__ == "__main__":
    unittest.main()
