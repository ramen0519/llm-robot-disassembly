"""Offline prototype contracts; injected provider is not an actual LLM call.

The historical-log test is explicitly read-only replay. It performs no new
physics and never attributes an earlier PASS to this newly written procedure.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_independent_ranker import config, MeshFixture
from test_transfer_memory import record as source_record
from orchestrator.independent_ranker import PROTOCOL as INDEPENDENT_PROTOCOL
from orchestrator.prototype_ranker import PrototypePipeline, PrototypeRankService, enrich, FIELDS
from orchestrator.repair_library import SKILL_ID
from worker.independent_features import describe_loaded, filter_features
from worker.task_memory import save_snapshot, task_context


def make_envelope(request_id="fixture:r3", repair=False, beam=False):
    initial = ["0", "1", "2", "3", "6"] if beam else ["1", "2", "base"] + (["gone"] if repair else [])
    base = "6" if beam else "base"
    assembled = list(initial) if beam else ["1", "2", "base"]
    features = describe_loaded({part: {"mesh": MeshFixture(), "transform": "final"} for part in initial},
                               {part + ".obj": "d" * 64 for part in initial},
                               "prototype-synthetic-target", "c" * 40, base, {"seed": 0})
    value = {"schema_version": 1, "kind": "rank_request", "timeout_seconds": 4,
             "request": {"run_id": "fixture", "request_id": request_id, "state_id": "s3",
                         "assembled_parts": assembled,
                         "candidate_ids": [part for part in assembled if part != base], "base_part": base},
             "features": {"protocol": INDEPENDENT_PROTOCOL,
                          "cad": filter_features(features, assembled),
                          "memory_context": task_context(features), "episode_feedback": [],
                          "validated_repair_skills": []}}
    if repair:
        value["features"]["episode_feedback"] = [
            observation(2, "2", ["1", "gone", "base"], "UNKNOWN"),
            observation(4, "gone", ["1", "2", "base"], "PASS")]
    return value


def observation(number, move, rest, status, base="base", removed=()):
    passed = status == "PASS"
    return {"event_id": "fixture:e" + str(number), "call_id": "v" + str(number),
            "status": status, "observed_stage": "passed" if passed else "path_search_returned_none",
            "request": {"part_move": move, "parts_rest": list(rest), "parts_removed": list(removed),
                        "pose": None, "max_grippers": 2, "timeout": 10,
                        "optimizer": "L-BFGS-B", "debug": 0, "render": False},
            "raw": {"feasible": passed, "part_move": move, "base_part": base, "pose": None,
                    "action": [0, 0, 1] if passed else None,
                    "parts_fix": [base] if passed else None,
                    "grasp": {"fixture_data_not_for_prompt": True}},
            "blocker": None, "path": None}


def provider_response(choice, **changes):
    return {"status": "completed", "error": None, "id": "injected-fixture-response", "model": "fixture-model",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(choice)}]}],
            "usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130}, **changes}


class PrototypeRankerTests(unittest.TestCase):
    def setUp(self):
        # Snapshot's production guard permits only project runs. This exclusively
        # created temporary directory never reuses or deletes earlier evidence.
        self.temp = tempfile.TemporaryDirectory(prefix="prototype_ranker_test_", dir=ROOT / "runs")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0
        self.environment = patch.dict(os.environ, {"INDEPENDENT_FIXTURE_KEY": "fixture-secret-not-a-real-key"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network_forbidden"))
        self.network_mock = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.network_mock.assert_not_called)

    def pipeline(self, mode="llm", transport=None, cfg=None, memory=False, ledger=None, controlled=False):
        self.serial += 1
        memory_path = None
        if memory:
            memory_path = self.root / ("frozen%d.json" % self.serial)
            save_snapshot([source_record()], memory_path)
        return PrototypePipeline(mode, config(enabled=mode == "llm") if cfg is None else cfg,
                                 ledger or self.root / ("ledger%d.sqlite" % self.serial),
                                 memory_path=memory_path, transport=transport, controlled_repair=controlled)

    def transport(self, repair=False, use_hint=False, changes=None, wrapper_changes=None):
        def choose(url, headers, body, timeout):
            value = json.loads(body["input"][0]["content"])
            request, features = value["request"], value["features"]
            cards = features["repair_cards"]
            hints = features["cross_task_geometry_references"]
            chosen = {"request_id": request["request_id"], "state_id": request["state_id"],
                      "candidate_ids": list(request["candidate_ids"]),
                      "skill_id": SKILL_ID if repair else "normal_rank",
                      "repair_part_id": cards[0]["eligible_part_ids"][0] if repair and cards else None,
                      "used_hint_ids": [hints[0]["hint_id"]] if use_hint and hints else []}
            chosen.update(changes or {})
            return provider_response(chosen, **(wrapper_changes or {}))
        return Mock(side_effect=choose)

    def test_empty_memory_actual_transport_seam_still_ranks_independently(self):
        transport = self.transport(changes={"candidate_ids": ["2", "1"]})
        offered = make_envelope()
        before = deepcopy(offered)
        result = self.pipeline(transport=transport).rank(offered)
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["candidate_ids"], ["2", "1"])
        self.assertTrue(result["metadata"]["attempted"])
        self.assertFalse(result["metadata"]["llm_live"])
        self.assertEqual(result["metadata"]["selected_skill_id"], "normal_rank")
        self.assertEqual(result["metadata"]["transfer_reference_count"], 0)
        self.assertEqual(offered, before)

    def test_disabled_config_has_zero_transport_calls(self):
        transport = self.transport()
        result = self.pipeline(transport=transport, cfg={}).rank(make_envelope())
        self.assertEqual(result["fallback_reason"], "llm_disabled")
        self.assertEqual(result["candidate_ids"], ["1", "2"])
        self.assertFalse(result["metadata"]["attempted"])
        self.assertNotIn("selected_skill_id", result["metadata"])
        transport.assert_not_called()

    def test_live_parser_receives_feedback_and_executes_only_registered_procedure(self):
        transport = self.transport(repair=True)
        result = self.pipeline(transport=transport).rank(make_envelope(repair=True))
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["candidate_ids"], ["2", "1"])
        invocation = result["metadata"]["repair_invocation"]
        self.assertEqual(invocation["source_failure_event_id"], "fixture:e2")
        self.assertEqual(invocation["state_change_event_ids"], ["fixture:e4"])
        self.assertEqual(invocation["physical_result"], "PENDING")
        self.assertFalse(invocation["state_mutations"])
        self.assertEqual(invocation["internal_verifier_calls"], 0)
        sent = json.loads(transport.call_args.args[2]["input"][0]["content"])
        self.assertEqual([item["status"] for item in sent["features"]["episode_feedback"]], ["UNKNOWN", "PASS"])
        self.assertNotIn("grasp", sent["features"]["episode_feedback"][0]["raw"])

    def test_normal_rank_can_decline_offered_repair(self):
        result = self.pipeline(transport=self.transport()).rank(make_envelope(repair=True))
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["metadata"]["offered_skill_ids"], [SKILL_ID])
        self.assertEqual(result["metadata"]["selected_skill_id"], "normal_rank")
        self.assertIsNone(result["metadata"]["repair_invocation"])

    def test_schema_enum_ids_skills_arguments_hints_and_full_candidate_count(self):
        transport = self.transport(repair=True, use_hint=True)
        result = self.pipeline(transport=transport, memory=True).rank(make_envelope(repair=True))
        self.assertIsNone(result["fallback_reason"])
        schema = transport.call_args.args[2]["text"]["format"]["schema"]
        props = schema["properties"]
        self.assertEqual(set(schema["required"]), FIELDS)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(props["request_id"]["enum"], ["fixture:r3"])
        self.assertEqual(props["candidate_ids"]["items"]["enum"], ["1", "2"])
        self.assertEqual(props["candidate_ids"]["minItems"], 2)
        self.assertEqual(props["candidate_ids"]["maxItems"], 2)
        self.assertEqual(props["skill_id"]["enum"], ["normal_rank", SKILL_ID])
        self.assertEqual(props["repair_part_id"]["anyOf"][0]["enum"], ["2"])
        self.assertEqual(props["used_hint_ids"]["items"]["enum"], result["metadata"]["offered_hint_ids"])

    def test_without_repair_or_memory_schema_allows_only_null_and_empty_evidence(self):
        transport = self.transport()
        self.pipeline(transport=transport).rank(make_envelope())
        props = transport.call_args.args[2]["text"]["format"]["schema"]["properties"]
        self.assertEqual(props["skill_id"]["enum"], ["normal_rank"])
        self.assertEqual(props["repair_part_id"], {"type": "null"})
        self.assertEqual(props["used_hint_ids"]["maxItems"], 0)

    def test_invalid_id_evidence_skill_and_arguments_fall_back_without_metadata_claims(self):
        cases = [({"candidate_ids": ["1", "1"]}, "invalid_permutation"),
                 ({"candidate_ids": ["1"]}, "invalid_permutation"),
                 ({"candidate_ids": ["1", "ghost"]}, "invalid_permutation"),
                 ({"request_id": "stale"}, "stale_request_or_state"),
                 ({"state_id": "stale"}, "stale_request_or_state"),
                 ({"used_hint_ids": ["invented-evidence"]}, "invalid_evidence_id"),
                 ({"used_hint_ids": "not-a-list"}, "invalid_evidence_id"),
                 ({"skill_id": "execute_python"}, "unregistered_skill"),
                 ({"repair_part_id": "2"}, "unexpected_repair_argument"),
                 ({"skill_id": SKILL_ID, "repair_part_id": "1"}, "repair_not_applicable"),
                 ({"skill_id": SKILL_ID, "repair_part_id": None}, "repair_not_applicable"),
                 ({"code": "invented code"}, "invalid_skill_response")]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                transport = self.transport(changes=changes)
                result = self.pipeline(transport=transport, memory=True).rank(make_envelope(repair=True))
                self.assertEqual(result["fallback_reason"], expected)
                self.assertEqual(result["candidate_ids"], ["1", "2"])
                self.assertEqual(result["metadata"]["ranking_source"], "deterministic_fallback")
                self.assertNotIn("selected_skill_id", result["metadata"])
                self.assertNotIn("repair_invocation", result["metadata"])
                self.assertEqual(transport.call_count, 1)

    def test_duplicate_offered_evidence_is_rejected(self):
        transport = self.transport(use_hint=True)
        original = transport.side_effect
        def duplicate(*args):
            raw = original(*args)
            text = raw["output"][0]["content"][0]
            value = json.loads(text["text"])
            value["used_hint_ids"] *= 2
            text["text"] = json.dumps(value)
            return raw
        transport.side_effect = duplicate
        result = self.pipeline(transport=transport, memory=True).rank(make_envelope())
        self.assertEqual(result["fallback_reason"], "invalid_evidence_id")

    def test_fresh_frozen_geometry_snapshot_provides_different_task_hints(self):
        transport = self.transport(use_hint=True)
        pipeline = self.pipeline(transport=transport, memory=True)
        frozen = next(self.root.glob("frozen*.json"))
        before = frozen.read_bytes()
        result = pipeline.rank(make_envelope())
        self.assertIsNone(result["fallback_reason"])
        self.assertTrue(result["metadata"]["used_hint_ids"])
        self.assertEqual(result["metadata"]["used_source_evidence_ids"], ["source:r1"])
        self.assertEqual(result["metadata"]["experience_usage_claim"], "model_cited_offered_reference_not_causal_proof")
        payload = json.loads(transport.call_args.args[2]["input"][0]["content"])
        hint = payload["features"]["cross_task_geometry_references"][0]
        self.assertTrue(hint["task_scope"]["mesh_input_bytes_differ"])
        self.assertTrue(hint["task_scope"]["geometry_descriptor_multiset_differs"])
        self.assertFalse(hint["current_attempt_verified"])
        self.assertIsNone(hint["current_physical_facts"])
        self.assertEqual(frozen.read_bytes(), before)

    def test_frozen_memory_context_cannot_be_mutated_by_worker(self):
        transport = self.transport()
        service = PrototypeRankService(config(), self.root / "service.sqlite", [source_record()], transport)
        offered = enrich(make_envelope(), [source_record()])
        offered["features"]["prototype"]["transfer"]["allow_hard_pruning"] = True
        result = service.rank(offered)
        self.assertEqual(result["fallback_reason"], "invalid_request")
        transport.assert_not_called()

    def test_worker_cannot_prepopulate_host_namespaces_or_hidden_priorities(self):
        for key in ("prototype", "memory", "recommended_order", "heuristic_score"):
            with self.subTest(key=key):
                offered = make_envelope()
                offered["features"][key] = {} if key in ("prototype", "memory") else ["2", "1"]
                transport = self.transport()
                result = self.pipeline(transport=transport).rank(offered)
                self.assertEqual(result["fallback_reason"], "invalid_request")
                transport.assert_not_called()

    def test_missing_candidate_or_fabricated_feedback_never_reaches_transport(self):
        for mutate in (lambda v: v["request"].update(candidate_ids=["1"]),
                       lambda v: v["features"]["episode_feedback"][0].update(blocker="gone"),
                       lambda v: v["features"]["episode_feedback"][1]["raw"].update(feasible=False)):
            offered = make_envelope(repair=True)
            mutate(offered)
            transport = self.transport()
            result = self.pipeline(transport=transport).rank(offered)
            self.assertEqual(result["fallback_reason"], "invalid_request")
            transport.assert_not_called()

    def test_shared_budget_and_cached_skill_metadata_do_not_repeat_provider_calls(self):
        transport = self.transport(repair=True)
        ledger = self.root / "shared.sqlite"
        settings = config(max_calls=1)
        first = self.pipeline(transport=transport, cfg=settings, ledger=ledger)
        second = self.pipeline(transport=transport, cfg=settings, ledger=ledger)
        original = first.rank(make_envelope(repair=True))
        cached = second.rank(make_envelope(repair=True))
        self.assertIsNone(original["fallback_reason"])
        self.assertTrue(cached["metadata"]["cached"])
        self.assertFalse(cached["metadata"]["attempted"])
        self.assertFalse(cached["metadata"]["llm_live"])
        self.assertEqual(cached["metadata"]["repair_invocation"], original["metadata"]["repair_invocation"])
        exhausted = second.rank(make_envelope("fixture:r4", repair=True))
        self.assertEqual(exhausted["fallback_reason"], "call_budget_exhausted")
        self.assertNotIn("repair_invocation", exhausted["metadata"])
        self.assertEqual(transport.call_count, 1)

    def test_duplicate_request_with_different_state_is_conflict(self):
        transport = self.transport()
        pipeline = self.pipeline(transport=transport)
        first = make_envelope()
        pipeline.rank(first)
        changed = deepcopy(first)
        changed["request"]["state_id"] = "different"
        result = pipeline.rank(changed)
        self.assertEqual(result["fallback_reason"], "request_id_conflict")
        self.assertEqual(transport.call_count, 1)

    def test_cost_cap_prevents_transport(self):
        transport = self.transport()
        result = self.pipeline(transport=transport, cfg=config(max_usd="0.00000001")).rank(make_envelope())
        self.assertEqual(result["fallback_reason"], "cost_budget_exhausted")
        transport.assert_not_called()

    def test_provider_errors_do_not_retry_and_do_not_expose_secrets(self):
        for failure, reason in ((TimeoutError("fixture-secret-not-a-real-key"), "api_timeout"),
                                (RuntimeError("fixture-secret-not-a-real-key"), "api_error")):
            with self.subTest(reason=reason):
                transport = Mock(side_effect=failure)
                result = self.pipeline(transport=transport).rank(make_envelope())
                self.assertEqual(result["fallback_reason"], reason)
                self.assertEqual(transport.call_count, 1)
                self.assertNotIn("fixture-secret-not-a-real-key", json.dumps(result))

    def test_incomplete_and_refusal_do_not_execute_skill(self):
        result = self.pipeline(transport=self.transport(repair=True, wrapper_changes={"status": "incomplete"})).rank(make_envelope(repair=True))
        self.assertEqual(result["fallback_reason"], "llm_incomplete")
        self.assertNotIn("repair_invocation", result["metadata"])
        refusal = {"status": "completed", "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "fixture"}]}]}
        result = self.pipeline(transport=Mock(return_value=refusal)).rank(make_envelope(repair=True))
        self.assertEqual(result["fallback_reason"], "llm_refusal")
        self.assertNotIn("selected_skill_id", result["metadata"])

    def test_duplicate_json_keys_rejected(self):
        transport = self.transport()
        original = transport.side_effect
        def duplicate(*args):
            raw = original(*args)
            content = raw["output"][0]["content"][0]
            content["text"] = content["text"][:-1] + ', "skill_id":"normal_rank"}'
            return raw
        transport.side_effect = duplicate
        result = self.pipeline(transport=transport).rank(make_envelope())
        self.assertEqual(result["fallback_reason"], "invalid_response")

    def test_fixture_mode_marks_choice_and_previews_without_provider_attempt(self):
        transport = self.transport()
        pipeline = self.pipeline("fixture", transport=transport, memory=True)
        result = pipeline.rank(make_envelope(repair=True))
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["metadata"]["ranking_source"], "fixture_decision")
        self.assertTrue(result["metadata"]["fixture"])
        self.assertFalse(result["metadata"]["attempted"])
        self.assertFalse(result["metadata"]["llm_live"])
        self.assertEqual(pipeline.previews[0]["status"], "PREPARED_NOT_PROOF_OF_SEND")
        self.assertEqual(result["metadata"]["repair_invocation"]["physical_result"], "PENDING")
        transport.assert_not_called()

    def test_fixture_mode_rejects_enabled_paid_configuration(self):
        with self.assertRaisesRegex(ValueError, "fixture_must_disable_api"):
            self.pipeline("fixture", cfg=config())

    def test_controlled_probes_are_explicitly_not_llm_or_fixture_skill_validation(self):
        transport = self.transport()
        pipeline = self.pipeline(transport=transport, controlled=True)
        first = pipeline.rank(make_envelope("fixture:r1", beam=True))
        second_envelope = make_envelope("fixture:r2", beam=True)
        second_envelope["features"]["episode_feedback"] = [observation(4, "3", ["0", "1", "2", "6"], "UNKNOWN", base="6")]
        second = pipeline.rank(second_envelope)
        self.assertEqual(first["candidate_ids"][0], "3")
        self.assertEqual(second["candidate_ids"][0], "1")
        for result in (first, second):
            self.assertEqual(result["metadata"]["ranking_source"], "controlled_physical_probe")
            self.assertFalse(result["metadata"]["attempted"])
            self.assertFalse(result["metadata"]["llm_live"])
            self.assertEqual(result["metadata"]["selected_skill_id"], "normal_rank")
            self.assertNotIn("repair_invocation", result["metadata"])
        transport.assert_not_called()

    def test_controlled_probe_without_real_unknown_fails_closed(self):
        transport = self.transport()
        pipeline = self.pipeline(transport=transport, controlled=True)
        pipeline.rank(make_envelope("fixture:r1", beam=True))
        result = pipeline.rank(make_envelope("fixture:r2", beam=True))
        self.assertEqual(result["fallback_reason"], "controlled_probe_precondition_failed")
        transport.assert_not_called()

    def test_controlled_probe_redelivery_is_cached_without_advancing_step(self):
        transport = self.transport()
        pipeline = self.pipeline(transport=transport, controlled=True)
        offered = make_envelope("fixture:r1", beam=True)
        original = pipeline.rank(offered)
        repeated = pipeline.rank(deepcopy(offered))
        self.assertEqual(original["candidate_ids"], repeated["candidate_ids"])
        self.assertTrue(repeated["metadata"]["cached"])
        self.assertEqual(repeated["metadata"]["ranking_source"], "controlled_physical_probe")
        self.assertEqual(len(pipeline.decisions), 1)
        self.assertEqual(len(pipeline.previews), 0)
        transport.assert_not_called()

    def test_malformed_provider_containers_fail_closed_without_skill_execution(self):
        for output in (None, {}, [None], [{"type": "message", "content": None}]):
            with self.subTest(output=output):
                transport = Mock(return_value={"status": "completed", "output": output})
                result = self.pipeline(transport=transport).rank(make_envelope(repair=True))
                self.assertIsNotNone(result["fallback_reason"])
                self.assertEqual(result["candidate_ids"], ["1", "2"])
                self.assertNotIn("repair_invocation", result["metadata"])
                self.assertEqual(transport.call_count, 1)

    def test_historical_actual_log_fixture_replay_is_not_a_new_physics_run(self):
        folder = ROOT / "runs/independent_ranker_20261018/physics/test_id_descending"
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        previous = next(item for item in events if item.get("kind") == "rank_decision" and item["generation_id"] == "g4")
        offered = next(json.loads(path.read_text(encoding="utf-8")) for path in (folder / "ipc/requests").glob("*.json")
                       if json.loads(path.read_text(encoding="utf-8"))["request"]["request_id"] == previous["request"]["request_id"])
        transport = self.transport()
        result = self.pipeline("fixture", transport=transport).rank(offered)
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual(result["metadata"]["ranking_source"], "fixture_decision")
        invocation = result["metadata"]["repair_invocation"]
        self.assertEqual(invocation["physical_result"], "PENDING")
        self.assertFalse(invocation["causal_benefit_demonstrated"])
        self.assertFalse(invocation["generalization_validated"])
        self.assertEqual(invocation["internal_verifier_calls"], 0)
        transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
