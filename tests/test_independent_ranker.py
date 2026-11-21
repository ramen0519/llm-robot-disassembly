"""Offline injected-provider fixtures; no paid API or physical simulator calls."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.independent_ranker import IndependentPipeline, IndependentRankService, PROTOCOL
from worker.independent_features import describe_loaded, filter_features
from worker.task_memory import task_context


class MeshFixture:
    bounds = [[0, 0, 0], [1, 1, 1]]
    center_mass = [0.5, 0.5, 0.5]
    volume = 1
    is_watertight = True
    is_winding_consistent = True


def envelope(request_id="fixture:r1"):
    features = describe_loaded({p: {"mesh": MeshFixture(), "transform": "final"} for p in ("2", "1", "base")},
                               {p + ".obj": "a" * 64 for p in ("2", "1", "base")},
                               "fixture", "c" * 40, "base", {"seed": 0})
    return {"schema_version": 1, "kind": "rank_request", "timeout_seconds": 4,
            "request": {"run_id": "fixture", "request_id": request_id, "state_id": "s1",
                        "assembled_parts": ["2", "base", "1"], "candidate_ids": ["2", "1"], "base_part": "base"},
            "features": {"protocol": PROTOCOL, "cad": filter_features(features, ["2", "base", "1"]),
                         "memory_context": task_context(features), "episode_feedback": [],
                         "validated_repair_skills": []}}


def config(**changes):
    return {"enabled": True, "model": "fixture-model", "api_key_env": "INDEPENDENT_FIXTURE_KEY",
            "max_calls": 10, "max_usd": "10", "input_usd_per_million": "1",
            "output_usd_per_million": "1", **changes}


def response(request, **changes):
    chosen = {"candidate_ids": ["2", "1"], "request_id": request["request_id"],
              "state_id": request["state_id"], **changes}
    return {"status": "completed", "error": None, "id": "fixture-response", "model": "fixture-model",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(chosen)}]}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}


class IndependentRankerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="independent_ranker_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0
        self.env = patch.dict(os.environ, {"INDEPENDENT_FIXTURE_KEY": "fake-fixture-only-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network_forbidden"))
        self.network_mock = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.network_mock.assert_not_called)

    def service(self, cfg=None, transport=None, path=None):
        self.serial += 1
        return IndependentRankService(config() if cfg is None else cfg,
                                      path or self.root / ("ledger%d.sqlite" % self.serial), transport)

    def transport(self, **changes):
        return Mock(side_effect=lambda url, headers, body, timeout:
                    response(json.loads(body["input"][0]["content"])["request"], **changes))

    def pipeline(self, mode, transport=None):
        self.serial += 1
        return IndependentPipeline(mode, config(enabled=mode == "llm"),
                                   self.root / ("pipeline%d.sqlite" % self.serial), transport=transport)

    def test_empty_memory_allows_independent_provider_choice(self):
        transport = self.transport()
        offered = envelope()
        saved = deepcopy(offered)
        result = self.pipeline("llm", transport).rank(offered)
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["candidate_ids"], ["2", "1"])
        self.assertTrue(result["metadata"]["attempted"])
        self.assertFalse(result["metadata"]["llm_live"])
        self.assertFalse(result["metadata"]["memory_enabled"])
        self.assertEqual(offered, saved)
        body = transport.call_args.args[2]
        sent = json.loads(body["input"][0]["content"])
        self.assertEqual(sent["request"]["candidate_ids"], ["1", "2"])
        self.assertNotIn("memory_context", sent["features"])
        self.assertNotIn("geometry_features", body["input"][0]["content"])
        self.assertEqual(sent["features"]["memory"]["exact_observations"], [])
        self.assertIn("Empty memory is normal", body["instructions"])

    def test_schema_is_bound_to_request_ids_and_full_count(self):
        transport = self.transport()
        self.service(transport=transport).rank(envelope())
        schema = transport.call_args.args[2]["text"]["format"]["schema"]
        self.assertEqual(schema["properties"]["request_id"]["enum"], ["fixture:r1"])
        self.assertEqual(schema["properties"]["state_id"]["enum"], ["s1"])
        self.assertEqual(schema["properties"]["candidate_ids"]["items"]["enum"], ["1", "2"])
        self.assertEqual(schema["properties"]["candidate_ids"]["minItems"], 2)
        self.assertEqual(schema["properties"]["candidate_ids"]["maxItems"], 2)
        self.assertFalse(schema["additionalProperties"])

    def test_bad_provider_ids_fall_back_to_full_canonical_order(self):
        for changes, reason in (({"candidate_ids": ["2", "2"]}, "invalid_permutation"),
                                ({"candidate_ids": ["2"]}, "invalid_permutation"),
                                ({"candidate_ids": ["2", "invented"]}, "invalid_permutation"),
                                ({"request_id": "stale"}, "stale_request_or_state"),
                                ({"state_id": "stale"}, "stale_request_or_state")):
            with self.subTest(changes=changes):
                result = self.service(transport=self.transport(**changes)).rank(envelope())
                self.assertEqual(result["fallback_reason"], reason)
                self.assertEqual(result["candidate_ids"], ["1", "2"])
                self.assertEqual(result["metadata"]["ranking_source"], "deterministic_fallback")

    def test_incomplete_candidate_input_rejected_before_api(self):
        offered = envelope()
        offered["request"]["candidate_ids"] = ["2"]
        transport = self.transport()
        result = self.service(transport=transport).rank(offered)
        self.assertEqual(result["fallback_reason"], "invalid_request")
        self.assertEqual(result["candidate_ids"], ["1", "2"])
        transport.assert_not_called()

    def test_hidden_priors_and_unknown_fields_rejected_before_api(self):
        for place, key in (("request", "native_order"), ("features", "heuristic_scores"),
                           ("nested", "gnn_scores"), ("parts", "vertices"), ("cad", "recommended_order")):
            with self.subTest(place=place, key=key):
                offered = envelope()
                if place == "nested":
                    offered["features"]["memory_context"]["nested"] = {key: [2, 1]}
                elif place == "parts":
                    offered["features"]["cad"]["parts"]["1"][key] = []
                elif place == "cad":
                    offered["features"]["cad"][key] = []
                else:
                    offered[place][key] = []
                transport = self.transport()
                self.assertEqual(self.service(transport=transport).rank(offered)["fallback_reason"], "invalid_request")
                transport.assert_not_called()

    def test_removed_cad_part_rejected(self):
        offered = envelope()
        offered["features"]["cad"]["parts"]["removed"] = deepcopy(offered["features"]["cad"]["parts"]["1"])
        self.assertEqual(self.service({}).rank(offered)["fallback_reason"], "invalid_request")

    def test_nonempty_repairs_cannot_be_claimed_implemented(self):
        offered = envelope()
        offered["features"]["validated_repair_skills"] = [{"name": "unverified"}]
        self.assertEqual(self.pipeline("llm").rank(offered)["fallback_reason"], "invalid_request")

    def test_worker_cannot_inject_host_memory(self):
        offered = envelope()
        offered["features"]["memory"] = {}
        self.assertEqual(self.pipeline("llm").rank(offered)["fallback_reason"], "invalid_request")

    def test_api_error_and_incomplete_are_not_success(self):
        for side_effect, expected in ((TimeoutError("fixture"), "api_timeout"),
                                      (RuntimeError("fixture"), "api_error")):
            transport = Mock(side_effect=side_effect)
            result = self.service(transport=transport).rank(envelope())
            self.assertEqual(result["fallback_reason"], expected)
            self.assertEqual(result["candidate_ids"], ["1", "2"])
            self.assertEqual(transport.call_count, 1)
        incomplete = Mock(return_value={"status": "incomplete"})
        self.assertEqual(self.service(transport=incomplete).rank(envelope())["fallback_reason"], "llm_incomplete")

    def test_duplicate_response_json_key_is_rejected(self):
        raw = response(envelope()["request"])
        raw["output"][0]["content"][0]["text"] = '{"candidate_ids":["1","2"],"request_id":"fixture:r1","request_id":"fixture:r1","state_id":"s1"}'
        self.assertEqual(self.service(transport=Mock(return_value=raw)).rank(envelope())["fallback_reason"], "invalid_response")

    def test_disabled_model_and_budget_remain_safe(self):
        transport = self.transport()
        result = self.service({}, transport).rank(envelope())
        self.assertEqual(result["fallback_reason"], "llm_disabled")
        transport.assert_not_called()
        one = self.service(config(max_calls=1), transport)
        self.assertIsNone(one.rank(envelope())["fallback_reason"])
        self.assertEqual(one.rank(envelope("fixture:r2"))["fallback_reason"], "call_budget_exhausted")
        self.assertEqual(transport.call_count, 1)

    def test_shared_ledger_and_cache_do_not_repeat_calls(self):
        transport = self.transport()
        ledger = self.root / "shared.sqlite"
        first = self.service(config(max_calls=1), transport, ledger)
        second = self.service(config(max_calls=1), transport, ledger)
        first.rank(envelope())
        repeated = second.rank(envelope())
        self.assertTrue(repeated["metadata"]["cached"])
        self.assertFalse(repeated["metadata"]["attempted"])
        self.assertEqual(second.rank(envelope("fixture:r2"))["fallback_reason"], "call_budget_exhausted")
        self.assertEqual(transport.call_count, 1)

    def test_test_modes_force_different_full_orders_without_api(self):
        for mode, expected in (("test_id_ascending", ["1", "2"]), ("test_id_descending", ["2", "1"]),
                               ("deterministic_memory", ["1", "2"])):
            with self.subTest(mode=mode):
                transport = self.transport()
                pipeline = self.pipeline(mode, transport)
                result = pipeline.rank(envelope())
                self.assertIsNone(result["fallback_reason"])
                self.assertEqual(result["candidate_ids"], expected)
                self.assertFalse(result["metadata"]["attempted"])
                self.assertFalse(result["metadata"]["llm_live"])
                self.assertEqual(result["metadata"]["ranking_source"], mode)
                transport.assert_not_called()
                pipeline.close()

    def test_feedback_is_bounded_and_projected_without_grasp(self):
        offered = envelope()
        observed = {"event_id": "e1", "call_id": "v1", "status": "UNKNOWN", "observed_stage": "path_search_returned_none",
                    "request": {"part_move": "1", "parts_rest": ["2", "base"], "parts_removed": [],
                                "pose": None, "max_grippers": 2, "timeout": None, "optimizer": "L-BFGS-B", "debug": 0, "render": False},
                    "raw": {"feasible": False, "action": None, "part_move": "1", "base_part": "base", "pose": None,
                            "grasp": {"fixture": "not-send"}}, "blocker": None, "path": None}
        offered["features"]["episode_feedback"] = [observed]
        transport = self.transport()
        self.pipeline("llm", transport).rank(offered)
        sent = json.loads(transport.call_args.args[2]["input"][0]["content"])
        self.assertEqual(sent["features"]["episode_feedback"][0]["status"], "UNKNOWN")
        self.assertNotIn("grasp", sent["features"]["episode_feedback"][0]["raw"])
        offered["features"]["episode_feedback"] = [observed] * 13
        self.assertEqual(self.pipeline("llm").rank(offered)["fallback_reason"], "invalid_request")

    def test_nested_order_hints_rejected_and_context_identity_matches_without_memory(self):
        for key in ("deterministic_fallback_order", "offered_positive_candidate_ids", "recommended_order", "preferred_order"):
            offered = envelope()
            offered["features"]["cad"]["task"]["solver_settings"]["nested"] = {key: ["2", "1"]}
            self.assertEqual(self.service({}).rank(offered)["fallback_reason"], "invalid_request")
        for mutation in ("identity", "geometry"):
            offered = envelope()
            if mutation == "identity":
                offered["features"]["memory_context"]["task"]["assembly_id"] = "wrong"
            else:
                offered["features"]["memory_context"]["geometry_features"]["1"]["volume"] = 9
            self.assertEqual(self.service({}).rank(offered)["fallback_reason"], "invalid_request")

    def test_corrupted_episode_status_is_rejected(self):
        offered = envelope()
        offered["features"]["episode_feedback"] = [{"event_id": "e1", "call_id": "v1", "status": "PASS",
            "observed_stage": "passed", "request": {"part_move": "1", "parts_rest": ["2", "base"],
                "parts_removed": [], "pose": None, "max_grippers": 2, "timeout": None,
                "optimizer": "L-BFGS-B", "debug": 0, "render": False},
            "raw": {"feasible": False, "action": None, "part_move": "1", "base_part": "base", "pose": None},
            "blocker": None, "path": None}]
        transport = self.transport()
        self.assertEqual(self.service(transport=transport).rank(offered)["fallback_reason"], "invalid_request")
        transport.assert_not_called()

    def frozen_fixture(self, offered):
        from worker.task_memory import task_context, records_from_events, save_snapshot, FrozenTaskMemory
        context = task_context(offered["features"]["cad"])
        offered["features"]["memory_context"] = context
        events = []
        for index, part, status in ((1, "2", "PASS"), (2, "1", "UNKNOWN")):
            call = "v%d" % index
            request = {"part_move": part, "parts_rest": [p for p in ("1", "2", "base") if p != part],
                       "parts_removed": [], "pose": None, "max_grippers": 2, "timeout": None,
                       "optimizer": "L-BFGS-B", "debug": 0, "render": False}
            events.append({"kind": "verifier_start", "run_id": "fixture", "event_id": "s%d" % index,
                           "call_id": call, "request": request})
            events.append({"kind": "verifier_result", "run_id": "fixture", "event_id": "e%d" % index,
                           "call_id": call, "status": status, "blocker": None, "path": None,
                           "raw": {"feasible": status == "PASS", "part_move": part, "base_part": "base",
                                   "action": [0, 0, 1] if status == "PASS" else None,
                                   "parts_fix": ["base"] if status == "PASS" else None, "pose": None}})
        records = records_from_events(context, events, "fixture/events.jsonl", "d" * 64)
        path = self.root / "frozen_fixture.json"
        save_snapshot(records, path, allow_test_root=True)
        return FrozenTaskMemory(path, allow_test_root=True)

    def test_same_frozen_memory_reaches_llm_and_deterministic_comparator(self):
        offered = envelope()
        memory = self.frozen_fixture(offered)
        comparator = self.pipeline("deterministic_memory")
        comparator.memory = memory
        compared = comparator.rank(offered)
        self.assertIsNone(compared["fallback_reason"])
        self.assertEqual(compared["candidate_ids"], ["2", "1"])
        transport = self.transport()
        llm = self.pipeline("llm", transport)
        llm.memory = memory
        selected = llm.rank(offered)
        self.assertIsNone(selected["fallback_reason"])
        self.assertEqual(selected["metadata"]["retrieved_evidence_ids"], compared["metadata"]["retrieved_evidence_ids"])
        sent = json.loads(transport.call_args.args[2]["input"][0]["content"])
        self.assertEqual(len(sent["features"]["memory"]["exact_observations"]), 2)
        self.assertEqual({item["status"] for item in sent["features"]["memory"]["exact_observations"]}, {"PASS", "UNKNOWN"})
        self.assertNotIn("geometry_features", transport.call_args.args[2]["input"][0]["content"])

    def test_cross_task_same_id_success_is_not_automatic_id_transfer(self):
        from worker.task_memory import task_context
        offered = envelope()
        memory = self.frozen_fixture(offered)
        offered["features"]["cad"]["task"]["assembly_id"] = "different-fixture-task"
        offered["features"]["memory_context"] = task_context(offered["features"]["cad"])
        comparator = self.pipeline("deterministic_memory")
        comparator.memory = memory
        result = comparator.rank(offered)
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["candidate_ids"], ["1", "2"])
        self.assertEqual(len(result["metadata"]["retrieved_evidence_ids"]), 2)

    def test_memory_read_failure_is_logged_fallback_without_api(self):
        transport = self.transport()
        pipeline = self.pipeline("llm", transport)
        pipeline.memory = Mock()
        pipeline.memory.retrieve.side_effect = ValueError("fixture-invalid-memory")
        result = pipeline.rank(envelope())
        self.assertEqual(result["fallback_reason"], "memory_retrieval_error")
        self.assertEqual(result["candidate_ids"], ["1", "2"])
        self.assertEqual(result["metadata"]["memory_error_type"], "ValueError")
        transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
