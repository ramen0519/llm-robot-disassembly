"""Synthetic log-contract checks only; no fixture is evidence of ASAP physics."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

from compare_rule_rankers import _audit_events


def successful_ascending_fixture():
    """A separate feasible fixture path: 0,1,2,3 versus native 1,0,3,2."""
    run_id = "synthetic-rule-comparison"
    events, edges = [], []
    nodes = [[["0", "1", "2", "3", "6"], {"n_eval": 0, "n_gripper": 1}]]
    parts = ["0", "1", "2", "3", "6"]
    initial_parts = list(parts)
    native_orders = [["1", "0", "3", "2"], ["3", "1", "2"], ["3", "2"]]

    def emit(kind, **fields):
        events.append(dict(kind=kind, run_id=run_id,
                           event_id="{}:e{}".format(run_id, len(events) + 1), **fields))

    emit("run_start", mode="rank", ranker_name="id_ascending", live_llm_calls=0,
         internal_geometry_operations=None)
    for index, part in enumerate(("0", "1", "2", "3")):
        generation_id = "g{}".format(index + 1) if index < 3 else None
        if generation_id is not None:
            native = list(native_orders[index])
            applied = sorted(native)
            membership_id = "remaining-parts-{}".format(index)
            request = {
                "run_id": run_id, "request_id": "{}:r{}".format(run_id, index + 1),
                "state_id": membership_id, "assembled_parts": list(parts),
                "candidate_ids": list(native), "base_part": "6",
            }
            emit("rank_decision", generation_id=generation_id, request=request,
                 ranker_name="id_ascending", applied_order=list(applied),
                 fallback_reason=None, rng_before="fixture-rng", rng_after="fixture-rng",
                 rank_seconds=0.00001)
            emit("generation", generation_id=generation_id, state_id=membership_id,
                 assembled_parts=list(parts), native_order=list(native), applied_order=list(applied),
                 rng_before="fixture-native-before", rng_after="fixture-rng")
            emit("candidate_yield", generation_id=generation_id, part=part)
        rest = [candidate for candidate in parts if candidate != part]
        call_id = "v{}".format(index + 1)
        emit("verifier_start", call_id=call_id, generation_id=generation_id,
             request={"part_move": part, "parts_rest": list(rest),
                      "parts_removed": [candidate for candidate in initial_parts if candidate not in parts],
                      "pose": None, "max_grippers": 2, "timeout": None,
                      "optimizer": "L-BFGS-B", "debug": 1, "render": False},
             rng_before="fixture-simulation-before")
        raw = {"feasible": True, "action": [0, 0, 1], "base_part": "6",
               "parts_fix": [], "part_move": part, "pose": None, "grasp": None}
        emit("verifier_result", call_id=call_id, status="PASS", raw=deepcopy(raw),
             blocker=None, path=None, rng_after="fixture-simulation-after")
        n_eval = min(index + 1, 3)
        edges.append([list(parts), list(rest), {"n_eval": n_eval, "sim_info": deepcopy(raw)}])
        nodes.append([list(rest), {"n_eval": n_eval, "n_gripper": 1}])
        parts = rest
    emit("run_end", status="returned_from_upstream")
    return {
        "events": events,
        "stats": {"success": True, "sequence": ["0", "1", "2", "3"],
                  "n_eval": 3, "total_n_eval": 3, "stop_msg": "solution found", "time": 0.01},
        "parsed": {"status": "PASS"}, "command": {"elapsed_seconds": 0.02},
        "tree": {"graph": {}, "directed": True, "nodes": nodes, "edges": edges},
    }


class RuleComparisonTests(unittest.TestCase):
    def audit(self, data):
        checks = {}
        row = _audit_events(data, "id_ascending", checks)
        return row, checks

    def assert_all_checks_pass(self, checks):
        self.assertTrue(checks)
        self.assertEqual([key for key, value in checks.items() if not value], [])

    def test_different_successful_rule_plan_is_allowed(self):
        row, checks = self.audit(successful_ascending_fixture())
        self.assert_all_checks_pass(checks)
        self.assertEqual(row["sequence"], ["0", "1", "2", "3"])
        self.assertNotEqual(row["sequence"], ["1", "0", "3", "2"])
        self.assertEqual(row["first_native_order"], ["1", "0", "3", "2"])
        self.assertEqual(row["first_applied_order"], ["0", "1", "2", "3"])
        self.assertTrue(row["plan_success"])
        self.assertEqual(row["verifier_calls"], 4)
        self.assertEqual(row["verifier_PASS"], 4)
        self.assertEqual(row["leaf_verifier_calls"], 1)
        self.assertEqual(row["native_total_n_eval"], 3)
        self.assertEqual(row["rank_requests"], 3)
        self.assertEqual(row["changed_order_generations"], 3)
        self.assertIsNone(row["skipped_edges"])

    def test_missing_verifier_result_is_detected(self):
        data = successful_ascending_fixture()
        data["events"] = [event for event in data["events"]
                          if not (event["kind"] == "verifier_result" and event["call_id"] == "v2")]
        _, checks = self.audit(data)
        self.assertFalse(checks["id_ascending:starts_results_same_order"])
        self.assertFalse(checks["id_ascending:serial_start_result_pairs"])
        self.assertFalse(checks["id_ascending:raw_verifier_order_matches_stored_edges"])

    def test_duplicate_event_request_and_call_ids_are_detected(self):
        cases = (
            ("event", "event_ids_unique"),
            ("request", "request_ids_unique"),
            ("call", "result_call_ids_unique"),
        )
        for kind, check_name in cases:
            with self.subTest(kind=kind):
                data = successful_ascending_fixture()
                if kind == "event":
                    data["events"][1]["event_id"] = data["events"][0]["event_id"]
                elif kind == "request":
                    ranks = [event for event in data["events"] if event["kind"] == "rank_decision"]
                    ranks[1]["request"]["request_id"] = ranks[0]["request"]["request_id"]
                else:
                    results = [event for event in data["events"] if event["kind"] == "verifier_result"]
                    results[1]["call_id"] = results[0]["call_id"]
                _, checks = self.audit(data)
                self.assertFalse(checks["id_ascending:" + check_name])

    def test_missing_candidate_cannot_be_hidden_by_consistent_rank_logs(self):
        data = successful_ascending_fixture()
        generation = next(event for event in data["events"] if event["kind"] == "generation")
        rank = next(event for event in data["events"] if event["kind"] == "rank_decision")
        generation["applied_order"].pop()
        rank["applied_order"] = list(generation["applied_order"])
        _, checks = self.audit(data)
        self.assertFalse(checks["id_ascending:candidate_sets_preserved"])
        self.assertFalse(checks["id_ascending:expected_rule_and_request_response"])

    def test_raw_result_and_stored_tree_disagreement_is_detected(self):
        data = successful_ascending_fixture()
        data["tree"]["edges"][0][2]["sim_info"]["action"] = [0, -1, 0]
        _, checks = self.audit(data)
        self.assertFalse(checks["id_ascending:raw_verifier_order_matches_stored_edges"])
        # A plausible-looking action on a feasible edge must still match its raw event.
        self.assertTrue(checks["id_ascending:reported_plan_is_connected_feasible_path"])

    def test_local_rank_rng_mutation_is_detected(self):
        data = successful_ascending_fixture()
        rank = next(event for event in data["events"] if event["kind"] == "rank_decision")
        rank["rng_after"] = "changed-by-ranker"
        _, checks = self.audit(data)
        self.assertFalse(checks["id_ascending:ranker_does_not_consume_rng"])

    def test_unknown_leaf_without_stored_edge_is_valid_unsuccessful_outcome(self):
        data = successful_ascending_fixture()
        result = [event for event in data["events"] if event["kind"] == "verifier_result"][-1]
        result["status"] = "UNKNOWN"
        result["raw"].update(feasible=False, action=None, parts_fix=None)
        data["tree"]["edges"].pop()
        data["tree"]["nodes"].pop()
        data["stats"].update(success=False, sequence=None, n_eval=None)
        data["parsed"]["status"] = "UNKNOWN"
        row, checks = self.audit(data)
        self.assert_all_checks_pass(checks)
        self.assertFalse(row["plan_success"])
        self.assertEqual(row["plan_status"], "UNKNOWN")
        self.assertEqual(row["verifier_PASS"], 3)
        self.assertEqual(row["verifier_UNKNOWN"], 1)
        self.assertEqual(row["verifier_calls"], 4)
        self.assertEqual(row["native_total_n_eval"], 3)


if __name__ == "__main__":
    unittest.main()
