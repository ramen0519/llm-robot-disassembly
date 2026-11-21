"""Registered rule-order tests only; these do not exercise ASAP physics."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

from rank_contract import (
    REGISTRY, RankRequest, RankResponse, id_ascending_ranker,
    rank_candidates, reverse_native_ranker,
)


def request_for(candidates):
    return RankRequest(
        run_id="rule-unit-run", request_id="rule-unit-run:r1", state_id="state-1",
        assembled_parts=candidates + ("6",), candidate_ids=candidates, base_part="6")


class RuleRankerTests(unittest.TestCase):
    def test_registered_functions_produce_expected_order_without_fallback(self):
        request = request_for(("1", "0", "3", "2"))
        cases = (
            ("id_ascending", id_ascending_ranker, ("0", "1", "2", "3")),
            ("reverse_native", reverse_native_ranker, ("2", "3", "0", "1")),
        )
        for name, function, expected in cases:
            with self.subTest(name=name):
                self.assertIs(REGISTRY[name], function)
                response = REGISTRY[name](request)
                self.assertIs(type(response), RankResponse)
                self.assertEqual(response.candidate_ids, expected)
                self.assertEqual(response.request_id, request.request_id)
                self.assertEqual(response.state_id, request.state_id)
                decision = rank_candidates(request, name)
                self.assertEqual(decision.candidate_ids, expected)
                self.assertIsNone(decision.fallback_reason)

    def test_rules_preserve_exact_candidates_and_input_snapshot(self):
        request = request_for(("part-B", "10", "part-A", "2"))
        before = replace(request)
        for name in ("id_ascending", "reverse_native"):
            with self.subTest(name=name):
                response = REGISTRY[name](request)
                self.assertIs(type(response.candidate_ids), tuple)
                self.assertCountEqual(response.candidate_ids, request.candidate_ids)
                self.assertEqual(len(set(response.candidate_ids)), len(request.candidate_ids))
                self.assertEqual(request, before)
                with self.assertRaises(FrozenInstanceError):
                    response.state_id = "changed"
                with self.assertRaises(FrozenInstanceError):
                    request.state_id = "changed"
                self.assertEqual(request, before)

    def test_empty_and_single_candidate_have_no_fallback(self):
        for candidates in ((), ("bolt",)):
            for name in ("id_ascending", "reverse_native"):
                with self.subTest(candidates=candidates, name=name):
                    request = request_for(candidates)
                    response = REGISTRY[name](request)
                    decision = rank_candidates(request, name)
                    self.assertEqual(response.candidate_ids, candidates)
                    self.assertEqual(decision.candidate_ids, candidates)
                    self.assertIsNone(decision.fallback_reason)

    def test_id_ascending_uses_lexical_string_order_including_non_numeric_ids(self):
        request = request_for(("2", "10", "part-2", "bolt", "A"))
        expected = ("10", "2", "A", "bolt", "part-2")
        self.assertEqual(id_ascending_ranker(request).candidate_ids, expected)
        decision = rank_candidates(request, "id_ascending")
        self.assertEqual(decision.candidate_ids, expected)
        self.assertIsNone(decision.fallback_reason)

    def test_reverse_native_reverses_given_order_without_sorting(self):
        request = request_for(("2", "part-A", "10", "bolt"))
        expected = ("bolt", "10", "part-A", "2")
        self.assertEqual(reverse_native_ranker(request).candidate_ids, expected)
        decision = rank_candidates(request, "reverse_native")
        self.assertEqual(decision.candidate_ids, expected)
        self.assertIsNone(decision.fallback_reason)

    def test_rule_calls_do_not_consume_python_random_state(self):
        request = request_for(("1", "0", "3", "2"))
        initial_state = random.getstate()
        try:
            for name in ("id_ascending", "reverse_native"):
                with self.subTest(name=name):
                    REGISTRY[name](request)
                    self.assertEqual(random.getstate(), initial_state)
                    decision = rank_candidates(request, name)
                    self.assertIsNone(decision.fallback_reason)
                    self.assertEqual(random.getstate(), initial_state)
        finally:
            random.setstate(initial_state)


if __name__ == "__main__":
    unittest.main()
