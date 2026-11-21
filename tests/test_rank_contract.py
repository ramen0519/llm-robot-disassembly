"""Local ranking-contract tests; these do not exercise ASAP physics."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

from rank_contract import REGISTRY, RankRequest, RankResponse, rank_candidates


class RankContractTests(unittest.TestCase):
    def setUp(self):
        self.request = RankRequest(
            run_id="unit-run", request_id="unit-run:r1", state_id="state-1",
            assembled_parts=("1", "0", "3", "2", "6"),
            candidate_ids=("1", "0", "3", "2"), base_part="6")

    def decide_with(self, response):
        return rank_candidates(self.request, "test", {"test": lambda _: response})

    def test_noop_preserves_order_and_input_snapshot(self):
        before = replace(self.request)
        decision = rank_candidates(self.request, "noop")
        self.assertEqual(decision.candidate_ids, ("1", "0", "3", "2"))
        self.assertIsNone(decision.fallback_reason)
        self.assertEqual(self.request, before)
        with self.assertRaises(FrozenInstanceError):
            self.request.state_id = "mutated"
        with self.assertRaises(TypeError):
            self.request.candidate_ids[0] = "mutated"
        with self.assertRaises(TypeError):
            REGISTRY["unreviewed"] = lambda request: request

    def test_empty_and_single_candidate_are_valid(self):
        for candidates in ((), ("2",)):
            with self.subTest(candidates=candidates):
                request = replace(self.request, candidate_ids=candidates)
                decision = rank_candidates(request, "noop")
                self.assertEqual(decision.candidate_ids, candidates)
                self.assertIsNone(decision.fallback_reason)

    def test_valid_permutation_is_accepted_without_mutating_input(self):
        response = RankResponse(self.request.request_id, self.request.state_id,
                                ("2", "3", "0", "1"))
        decision = self.decide_with(response)
        self.assertEqual(decision.candidate_ids, response.candidate_ids)
        self.assertIsNone(decision.fallback_reason)
        self.assertEqual(self.request.candidate_ids, ("1", "0", "3", "2"))

    def test_missing_duplicate_extra_and_foreign_ids_fall_back(self):
        for candidates in (("1", "0", "3"), ("1", "0", "3", "3"),
                           ("1", "0", "3", "2", "9"), ("1", "0", "3", "9")):
            with self.subTest(candidates=candidates):
                response = RankResponse(self.request.request_id,
                                        self.request.state_id, candidates)
                decision = self.decide_with(response)
                self.assertEqual(decision.candidate_ids, self.request.candidate_ids)
                self.assertEqual(decision.fallback_reason, "invalid_permutation")

    def test_stale_request_or_state_falls_back(self):
        for request_id, state in (("old-request", self.request.state_id),
                                  (self.request.request_id, "old-state")):
            with self.subTest(request_id=request_id, state=state):
                decision = self.decide_with(RankResponse(
                    request_id, state, tuple(reversed(self.request.candidate_ids))))
                self.assertEqual(decision.candidate_ids, self.request.candidate_ids)
                self.assertEqual(decision.fallback_reason, "stale_request_or_state")

    def test_wrong_response_and_candidate_types_fall_back_without_coercion(self):
        for response in (None, {}, list(self.request.candidate_ids), "1,0,3,2"):
            with self.subTest(response=response):
                decision = self.decide_with(response)
                self.assertEqual(decision.candidate_ids, self.request.candidate_ids)
                self.assertEqual(decision.fallback_reason, "invalid_response_type")
        for candidates in (["1", "0", "3", "2"], (1, "0", "3", "2"),
                           (True, "0", "3", "2"), "1032"):
            with self.subTest(candidates=candidates):
                decision = self.decide_with(RankResponse(
                    self.request.request_id, self.request.state_id, candidates))
                self.assertEqual(decision.candidate_ids, self.request.candidate_ids)
                self.assertEqual(decision.fallback_reason, "invalid_candidate_type")

    def test_unknown_ranker_and_exception_preserve_native_order(self):
        calls = []

        def failing(request):
            calls.append(request.request_id)
            raise RuntimeError("local test failure")

        unknown = rank_candidates(self.request, "missing", {"failing": failing})
        self.assertEqual(unknown.candidate_ids, self.request.candidate_ids)
        self.assertEqual(unknown.fallback_reason, "unknown_ranker")
        self.assertEqual(calls, [])
        failure = rank_candidates(self.request, "failing", {"failing": failing})
        self.assertEqual(failure.candidate_ids, self.request.candidate_ids)
        self.assertEqual(failure.fallback_reason, "ranker_exception:RuntimeError")
        self.assertEqual(calls, [self.request.request_id])

    def test_attempt_to_mutate_snapshot_falls_back_without_corruption(self):
        def mutating(request):
            request.candidate_ids += ("foreign",)

        decision = rank_candidates(self.request, "mutating", {"mutating": mutating})
        self.assertEqual(decision.candidate_ids, ("1", "0", "3", "2"))
        self.assertEqual(decision.fallback_reason, "ranker_exception:FrozenInstanceError")
        self.assertEqual(self.request.candidate_ids, ("1", "0", "3", "2"))

    def test_bad_upstream_candidates_are_errors_before_ranker_runs(self):
        calls = []

        def observing(request):
            calls.append(request)
            return RankResponse(request.request_id, request.state_id, request.candidate_ids)

        for candidates in (["1", "0"], ("1", "1"), (1,), (True,),
                           ("foreign",), ("6",)):
            with self.subTest(candidates=candidates):
                request = replace(self.request, candidate_ids=candidates)
                with self.assertRaises(ValueError):
                    rank_candidates(request, "observing", {"observing": observing})
        self.assertEqual(calls, [])

    def test_cancellation_is_not_swallowed_as_a_ranker_failure(self):
        def interrupted(_):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            rank_candidates(self.request, "interrupted", {"interrupted": interrupted})


if __name__ == "__main__":
    unittest.main()
