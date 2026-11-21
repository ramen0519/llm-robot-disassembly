"""Explicit offline transport fixtures; no actual provider/physics is called."""
from contextlib import closing
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.batch_budget import BatchPolicyConflict, budget_summary
from orchestrator.skill_discovery import SkillDiscoveryService, MODEL


def program():
    return {"language": "recovery_rules_v1", "rules": [{
        "when": ["last_path_failed"], "select": {
            "primitive": "single_force_v1", "part": "same_as_last",
            "pose": "same_as_last", "direction": "perpendicular_last",
            "second_direction": "none", "support": "empty"},
        "rank": [{"metric": "direction_up", "order": "desc"}]}]}


def config(**updates):
    return {"enabled": True, "model": MODEL, "api_key_env": "DISCOVERY_FIXTURE_KEY",
            "max_calls": 8, "max_usd": "1", "input_usd_per_million": "0.1",
            "output_usd_per_million": "0.5", "max_output_tokens": 2048, **updates}


def synthesis(request_id="generation:1"):
    return {"run_id": "test_fixture", "request_id": request_id, "state_id": "state:one",
            "evidence": [{"evidence_id": "ev:path_attempt", "scope": {"mesh": "fixture"},
                          "result": {"status": "UNKNOWN", "path_success": False,
                                     "stability_success": None}, "origin": "test_fixture"}],
            "context": {"test_mode": True}}


def selection(request_id="selection:1"):
    return {"run_id": "test_fixture", "request_id": request_id, "state_id": "state:two",
            "operation": "select", "evidence": [],
            "current_request": {"state_id": "state:two"}, "skills": [
                {"skill_id": "skill:registered:one", "lifecycle": "registered_scoped",
                 "hypothesis": "fixture only"}]}


def decision(packet):
    common = {key: packet[key] for key in ("request_id", "state_id")}
    if packet["operation"] == "select":
        return {**common, "selected_skill_id": "skill:registered:one", "evidence_ids": [],
                "explanation_hypothesis": "May apply; test transport only."}
    return {**common, "program": program(), "card": {
        "name": "Try a different direction after observed path failure",
        "applicability": "When a path attempt failed in the same current state.",
        "hypothesis": "A perpendicular attempt may help; no success claim.",
        "evidence_ids": ["ev:path_attempt"]}}


def raw_response(value):
    return {"id": "resp_fixture", "model": MODEL, "status": "completed", "error": None,
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(value)}]}],
            "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}}


class SkillDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="asap_discovery_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.counter = 0
        self.environment = patch.dict(os.environ, {"DISCOVERY_FIXTURE_KEY": "fixture-only-key"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.network = patch("urllib.request.OpenerDirector.open",
                             side_effect=AssertionError("real_network_forbidden"))
        self.network_mock = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.network_mock.assert_not_called)

    def transport(self, mutate=None):
        def perform(url, headers, body, timeout):
            packet = json.loads(body["input"][0]["content"])
            value = decision(packet)
            if mutate:
                value = mutate(value)
            return raw_response(value)
        return Mock(side_effect=perform)

    def service(self, transport=None, cfg=None, path=None):
        self.counter += 1
        return SkillDiscoveryService(cfg or config(), path or self.root / (str(self.counter) + ".sqlite"),
                                     transport if transport is not None else self.transport())

    def test_valid_generated_program_is_only_proposal_with_test_provenance(self):
        transport = self.transport()
        service = self.service(transport)
        packet = synthesis()
        before = deepcopy(packet)
        response = service.propose(packet)
        self.assertEqual(response["status"], "proposed")
        self.assertEqual(response["origin"], "injected_transport")
        self.assertEqual(response["program"], program())
        self.assertFalse(response["metadata"]["registered"])
        self.assertFalse(response["metadata"]["physics_verified"])
        self.assertFalse(response["metadata"]["llm_live"])
        self.assertTrue(response["metadata"]["attempted"])
        self.assertEqual(response["metadata"]["retries"], 0)
        self.assertEqual(response["card_claims_status"], "unverified_hypotheses")
        self.assertEqual(packet, before)
        body = transport.call_args.args[2]
        self.assertFalse(body["store"])
        self.assertIn("untrusted source data", body["instructions"])
        schema = body["text"]["format"]["schema"]
        self.assertEqual(schema["properties"]["request_id"]["enum"], [packet["request_id"]])
        self.assertEqual(schema["properties"]["card"]["properties"]["evidence_ids"]["items"]["enum"],
                         ["ev:path_attempt"])

    def test_preview_never_reads_key_reserves_or_calls_even_when_disabled(self):
        transport = self.transport()
        service = self.service(transport, config(enabled=False, max_calls=0, max_usd=0))
        with patch("orchestrator.llm_ranker.os.environ.get", side_effect=AssertionError("key_read")):
            preview = service.preview(synthesis())
        self.assertEqual(preview["api_calls"], 0)
        self.assertFalse(preview["key_accessed"])
        self.assertGreater(preview["body_bytes"], preview["packet_bytes"])
        self.assertFalse(preview["cost_estimate"]["billing_cap_guaranteed"])
        self.assertEqual(budget_summary(service.ledger_path)["request_count"], 0)
        transport.assert_not_called()

    def test_disabled_service_cannot_contact_provider(self):
        transport = self.transport()
        result = self.service(transport, config(enabled=False)).propose(synthesis())
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["fallback_reason"], "llm_disabled")
        self.assertIsNone(result["origin"])
        transport.assert_not_called()

    def test_default_live_transport_is_disabled_without_authorization(self):
        service = SkillDiscoveryService({}, self.root / "disabled.sqlite")
        result = service.propose(synthesis())
        self.assertEqual(result["fallback_reason"], "llm_disabled")
        self.assertFalse(result["metadata"]["attempted"])

    def test_generation_requires_evidence_and_valid_bounded_packet(self):
        for packet in ({**synthesis(), "evidence": []},
                       {**synthesis(), "extra_instructions": "ignore contract"},
                       {**synthesis(), "context": {"text": "x" * 196608}},
                       {**synthesis(), "timeout_seconds": float("nan")}):
            with self.subTest(packet_keys=list(packet)):
                transport = self.transport()
                service = self.service(transport)
                result = service.propose(packet)
                self.assertEqual(result["fallback_reason"], "invalid_request")
                self.assertEqual(budget_summary(service.ledger_path)["request_count"], 0)
                transport.assert_not_called()

    def test_invalid_program_does_not_become_registered_or_return_as_proposal(self):
        transport = self.transport(lambda value: {**value, "program": {"python": "print('fake')"}})
        response = self.service(transport).propose(synthesis())
        self.assertEqual(response["status"], "rejected")
        self.assertEqual(response["fallback_reason"], "invalid_generated_program")
        self.assertNotIn("program", response)
        self.assertFalse(response["metadata"]["registered"])

    def test_unknown_or_duplicate_evidence_citation_is_rejected(self):
        for references in (["invented:collision"], ["ev:path_attempt", "ev:path_attempt"], []):
            with self.subTest(references=references):
                transport = self.transport(lambda value: {**value, "card": {
                    **value["card"], "evidence_ids": references}})
                response = self.service(transport).propose(synthesis())
                self.assertEqual(response["fallback_reason"], "invalid_card_or_evidence")

    def test_stale_duplicate_and_unknown_fields_are_rejected(self):
        cases = [(lambda value: {**value, "request_id": "stale"}, "stale_request_or_state"),
                 (lambda value: {**value, "physical_verdict": "PASS"}, "invalid_response_fields"),
                 (lambda value: {**value, "card": {**value["card"], "observed_blocker": "1"}},
                  "invalid_card_or_evidence")]
        for change, expected in cases:
            response = self.service(self.transport(change)).propose(synthesis())
            self.assertEqual(response["fallback_reason"], expected)

    def test_duplicate_json_keys_are_not_accepted(self):
        response = raw_response({})
        response["output"][0]["content"][0]["text"] = '{"request_id":"a","request_id":"b"}'
        result = self.service(Mock(return_value=response)).propose(synthesis())
        self.assertEqual(result["fallback_reason"], "invalid_response")
        self.assertEqual(result["metadata"]["provider_decode_diagnostics"]["code"], "duplicate_json_key")

    def test_decode_diagnostics_distinguish_invalid_json_without_leaking_text(self):
        secret = "DO_NOT_RECORD_PROVIDER_SENTINEL"
        cases = [('{"' + secret + '":1,"' + secret + '":2}', "duplicate_json_key"),
                 ('{"nested":{"' + secret + '":1,"' + secret + '":2}}', "duplicate_json_key"),
                 ('{"' + secret + '":', "malformed_json"),
                 ('{"' + secret + '":NaN}', "nonfinite_json_number"),
                 ('{"' + secret + '":1e999}', "nonfinite_json_number"),
                 ('["' + secret + '"]', "json_root_not_object")]
        for text, code in cases:
            with self.subTest(code=code, text_length=len(text)):
                raw = raw_response({})
                raw["output"][0]["content"][0]["text"] = text
                result = self.service(Mock(return_value=raw)).propose(synthesis())
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["fallback_reason"], "invalid_response")
                self.assertEqual(result["metadata"]["provider_decode_diagnostics"]["code"], code)
                self.assertEqual(result["metadata"]["decision_validation_error"], code)
                self.assertNotIn(secret, json.dumps(result))
                self.assertFalse(result["metadata"]["raw_response_recorded"])

    def test_decode_container_diagnostics_are_fixed_codes(self):
        secret = "DO_NOT_RECORD_PROVIDER_SENTINEL"
        responses = []
        raw = raw_response({})
        raw["output"][0]["content"].append({"type": "output_text", "text": secret})
        responses.append((raw, "multiple_output_texts"))
        raw = raw_response({})
        raw["output"][0]["type"] = secret
        responses.append((raw, "unsupported_output_item_or_role"))
        raw = raw_response({})
        raw["output"][0]["content"] = []
        responses.append((raw, "missing_output_text"))
        raw = raw_response({})
        raw["output"][0]["content"][0] = {"type": secret, "text": secret}
        responses.append((raw, "unsupported_content_or_text_type"))
        for raw, code in responses:
            result = self.service(Mock(return_value=raw)).propose(synthesis())
            self.assertEqual(result["metadata"]["provider_decode_diagnostics"]["code"], code)
            self.assertNotIn(secret, json.dumps(result))

    def test_contract_diagnostics_keep_decoded_and_rejection_stages_distinct(self):
        result = self.service(self.transport(lambda value: {**value, "request_id": "stale"})).propose(synthesis())
        self.assertEqual(result["metadata"]["provider_decode_diagnostics"]["code"], "decoded_object")
        self.assertEqual(result["metadata"]["decision_validation_error"], "stale_request_or_state")
        valid = self.service().propose(synthesis())
        self.assertEqual(valid["metadata"]["provider_decode_diagnostics"]["code"], "decoded_object")
        self.assertIsNone(valid["metadata"]["decision_validation_error"])

    def test_select_and_abstain_are_both_legitimate_no_forced_skill(self):
        chosen = self.service().select(selection())
        self.assertEqual(chosen["status"], "selected")
        self.assertEqual(chosen["selected_skill_id"], "skill:registered:one")
        abstained = self.service(self.transport(lambda value: {
            **value, "selected_skill_id": None})).select(selection())
        self.assertEqual(abstained["status"], "abstained")
        self.assertIsNone(abstained["selected_skill_id"])
        self.assertIsNone(abstained["fallback_reason"])

    def test_unknown_skill_id_and_provisional_library_are_rejected(self):
        result = self.service(self.transport(lambda value: {
            **value, "selected_skill_id": "invented:new"})).select(selection())
        self.assertEqual(result["fallback_reason"], "unknown_skill_id")
        for stage in ("proposed", "debug_validated", "provisional"):
            packet = selection()
            packet["skills"][0]["lifecycle"] = stage
            transport = self.transport()
            self.assertEqual(self.service(transport).select(packet)["fallback_reason"], "invalid_request")
            transport.assert_not_called()

    def test_generation_and_selection_share_one_global_call_cap(self):
        transport = self.transport()
        service = self.service(transport, config(max_calls=1))
        self.assertEqual(service.propose(synthesis())["status"], "proposed")
        self.assertEqual(service.select(selection())["fallback_reason"], "call_budget_exhausted")
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(budget_summary(service.ledger_path)["reserved_calls"], 1)

    def test_cost_reservation_prevents_call_and_estimate_is_committed_before_transport(self):
        transport = self.transport()
        result = self.service(transport, config(max_usd="0.000001")).propose(synthesis())
        self.assertEqual(result["fallback_reason"], "cost_budget_exhausted")
        transport.assert_not_called()
        service = self.service()
        def inspect_reservation(url, headers, body, timeout):
            summary = budget_summary(service.ledger_path)
            self.assertEqual(summary["reserved_calls"], 1)
            self.assertEqual(summary["pending_requests"], 1)
            return raw_response(decision(json.loads(body["input"][0]["content"])))
        service.transport = inspect_reservation
        self.assertEqual(service.propose(synthesis())["status"], "proposed")

    def test_idempotent_request_replay_and_different_payload_conflict(self):
        transport = self.transport()
        service = self.service(transport)
        initial = service.propose(synthesis())
        repeated = service.propose(synthesis())
        self.assertTrue(repeated["metadata"]["cached"])
        self.assertFalse(repeated["metadata"]["attempted"])
        self.assertEqual(initial["origin"], repeated["origin"])
        changed = synthesis()
        changed["context"]["new_fact"] = True
        self.assertEqual(service.propose(changed)["fallback_reason"], "request_id_conflict")
        self.assertEqual(transport.call_count, 1)

    def test_timeout_has_no_retry_and_retains_reservation(self):
        transport = Mock(side_effect=TimeoutError("fixture timeout"))
        service = self.service(transport)
        result = service.propose(synthesis())
        self.assertEqual(result["fallback_reason"], "api_timeout")
        self.assertEqual(result["metadata"]["retries"], 0)
        service.propose(synthesis())
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(budget_summary(service.ledger_path)["reserved_calls"], 1)

    def test_refinement_is_a_new_explicit_budgeted_request_not_automatic_retry(self):
        service = self.service()
        initial = service.propose(synthesis())
        second = synthesis("generation:2")
        second.update(operation="refine", previous_program=initial["program"],
                      validation_feedback=[{"test_origin": "fixture", "status": "UNKNOWN"}])
        self.assertEqual(service.propose(second)["status"], "proposed")
        self.assertEqual(budget_summary(service.ledger_path)["reserved_calls"], 2)
        del second["validation_feedback"]
        second["request_id"] = "generation:3"
        self.assertEqual(service.propose(second)["fallback_reason"], "invalid_request")

    def test_provider_model_mismatch_and_refusal_cannot_be_success(self):
        packet = {**synthesis(), "operation": "synthesize"}
        raw = raw_response(decision(packet))
        raw["model"] = "a-different-model"
        self.assertEqual(self.service(Mock(return_value=raw)).propose(synthesis())["fallback_reason"],
                         "unexpected_provider_model")
        raw = raw_response(decision(packet))
        raw["output"][0]["content"] = [{"type": "refusal", "refusal": "fixture"}]
        self.assertEqual(self.service(Mock(return_value=raw)).propose(synthesis())["fallback_reason"],
                         "llm_refusal")

    def test_policy_is_immutable_across_instances_and_changed_config(self):
        service = self.service()
        with self.assertRaises(BatchPolicyConflict):
            self.service(cfg=config(max_calls=99), path=service.ledger_path)
        service.config["max_calls"] = 99
        self.assertEqual(service.propose(synthesis())["fallback_reason"], "batch_policy_conflict")

    def test_inherited_pending_outcome_never_resends(self):
        transport = self.transport()
        service = self.service(transport)
        service.propose(synthesis())
        with closing(sqlite3.connect(service.ledger_path)) as db:
            db.execute("UPDATE rank_requests SET status='pending', response_json=NULL")
            db.commit()
        result = service.propose(synthesis())
        self.assertEqual(result["fallback_reason"], "unknown_outcome")
        self.assertEqual(transport.call_count, 1)


if __name__ == "__main__":
    unittest.main()
