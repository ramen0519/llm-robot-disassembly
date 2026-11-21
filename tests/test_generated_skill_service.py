"""Offline adapter/transport fixtures, not actual skill admission or physics.

Registry retrieval is injected at the adapter boundary so these tests exercise
runtime decisions independently of filesystem admission tests. The interpreter,
action guards, provider parser and shared SQLite budget are real local code.
"""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.batch_budget import budget_summary
from orchestrator.explicit_action_service import ExplicitActionService
from orchestrator.explicit_action_provider_v4 import PROVIDER_CONTRACT_V4
from orchestrator.generated_skill_service import GeneratedSkillService
from orchestrator.skill_discovery import SkillDiscoveryService
from orchestrator.skill_program import program_hash
from tests.test_skill_program import append_observation, request, program
from tests.test_skill_discovery import config, raw_response
from worker.explicit_action_contract import validate_response
from worker.path_probe_round import validate_path_round_response


def row(fixture=False):
    policy = program()
    from orchestrator.skill_execution_roles import summarize_execution_roles
    return {"strategy_id": program_hash(policy), "program_hash": program_hash(policy),
            "program": policy, "card": {"description": "Fixture procedure, not discovered by LLM"},
            "lifecycle": "fixture_registered_scoped" if fixture else "registered_scoped",
            "validation_scope": [{"origin": "unit_fixture_only"}], "fixture": fixture,
            "execution_roles": summarize_execution_roles([]),
            "physics_reuse_allowed": False, "hard_pruning": False}


def snapshot(fixture=False):
    return {"content_sha256": "a"*64, "fixture": fixture}


class GeneratedSkillServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="generated_service_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.environment = patch.dict(os.environ, {"DISCOVERY_FIXTURE_KEY": "fixture-only-key"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.network = patch("urllib.request.OpenerDirector.open",
                             side_effect=AssertionError("actual_network_forbidden"))
        self.network_mock = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.network_mock.assert_not_called)
        self.rows = [row()]
        self.retrieve = patch("orchestrator.generated_skill_service.retrieve",
                              side_effect=lambda *args, **kwargs: deepcopy(self.rows))
        self.retrieve.start()
        self.addCleanup(self.retrieve.stop)

    def fallback(self, mode="deterministic", cfg=None, ledger=None, transport=None):
        return ExplicitActionService(cfg or config(), ledger or self.root / "actor.sqlite",
            transport=transport, mode=mode, provider_contract=PROVIDER_CONTRACT_V4,
            fallback_policy="path_stage_v1")

    def transport(self, selected="first"):
        def reply(url, headers, body, timeout):
            packet = json.loads(body["input"][0]["content"])
            identity = packet["skills"][0]["skill_id"] if selected == "first" else selected
            return raw_response({"request_id": packet["request_id"], "state_id": packet["state_id"],
                "selected_skill_id": identity, "evidence_ids": [],
                "explanation_hypothesis": "Fixture selector, no physics verdict."})
        return Mock(side_effect=reply)

    def selector(self, transport=None, cfg=None):
        return SkillDiscoveryService(cfg or config(), self.root / "shared.sqlite",
                                     transport or self.transport())

    def test_generated_action_uses_primitive_and_preserves_complete_candidates(self):
        selected = self.selector()
        fallback = self.fallback()
        service = GeneratedSkillService(snapshot(), fallback, selected)
        req = append_observation(request())
        before = deepcopy(req)
        response = service.handle(req)
        self.assertEqual(req, before)
        self.assertEqual(response["choice"]["skill_id"], "single_force_v1")
        self.assertEqual(set(response["candidate_order"]), set(req["candidate_ids"]))
        self.assertEqual(response["metadata"]["generated_skill_id"], self.rows[0]["strategy_id"])
        self.assertTrue(response["metadata"]["generated_program_executed"])
        self.assertEqual(response["metadata"]["selection_source"], "injected_provider_fixture")
        self.assertEqual(response["metadata"]["decision_origin"], "generated_program")
        self.assertFalse(response["metadata"]["llm_live"])
        self.assertFalse(response["metadata"]["physics_bypass"])
        self.assertEqual(fallback.decisions, [])
        validate_response(req, response)
        validate_path_round_response(req, response)

    def test_selector_abstention_retains_original_actor_and_records_it(self):
        selector = self.selector(self.transport(None))
        fallback = self.fallback()
        service = GeneratedSkillService(snapshot(), fallback, selector)
        response = service.handle(append_observation(request()))
        self.assertFalse(response["metadata"]["generated_program_executed"])
        self.assertEqual(response["metadata"]["generated_skill_fallback_reason"], "skill_selector_abstained")
        self.assertEqual(response["metadata"]["selection_source"], "deterministic_control")
        self.assertEqual(len(fallback.decisions), 1)

    def test_selector_card_exposes_development_scope_separately(self):
        self.rows[0]["development_scope"] = [{"phase": "debug", "source_task": "target_fixture",
                                               "outcome": "UNKNOWN"}]
        transport = self.transport()
        service = GeneratedSkillService(snapshot(), self.fallback(), self.selector(transport))
        service.handle(append_observation(request()))
        packet = json.loads(transport.call_args.args[2]["input"][0]["content"])
        card = packet["skills"][0]
        self.assertEqual(card["development_scope"], self.rows[0]["development_scope"])
        self.assertEqual(card["validation_scope"], self.rows[0]["validation_scope"])
        self.assertFalse(card["target_task_verified"])

    def test_unknown_selection_rejected_before_existing_actor_fallback(self):
        service = GeneratedSkillService(snapshot(), self.fallback(), self.selector(self.transport("unknown")))
        response = service.handle(append_observation(request()))
        self.assertEqual(response["metadata"]["generated_skill_fallback_reason"], "skill_selector_rejected")
        self.assertEqual(response["metadata"]["generated_selector_response"]["fallback_reason"], "unknown_skill_id")
        self.assertFalse(response["metadata"]["generated_program_executed"])

    def test_no_current_trigger_skips_paid_selection(self):
        transport = self.transport()
        selector = self.selector(transport)
        service = GeneratedSkillService(snapshot(), self.fallback(), selector)
        response = service.handle(request())
        self.assertEqual(response["metadata"]["generated_skill_fallback_reason"], "no_applicable_registered_skill")
        self.assertEqual(budget_summary(selector.ledger_path)["reserved_calls"], 0)
        transport.assert_not_called()

    def test_selector_then_actor_share_global_cap(self):
        transport = self.transport(None)
        cfg = config(max_calls=1)
        selector = self.selector(transport, cfg)
        actor_transport = Mock(side_effect=AssertionError("call_cap_must_block_fallback_http"))
        fallback = self.fallback("llm", cfg, selector.ledger_path, actor_transport)
        service = GeneratedSkillService(snapshot(), fallback, selector)
        response = service.handle(append_observation(request()))
        self.assertEqual(response["fallback_reason"], "call_budget_exhausted")
        self.assertEqual(budget_summary(selector.ledger_path)["reserved_calls"], 1)
        self.assertEqual(response["metadata"]["generated_skill_fallback_reason"], "skill_selector_abstained")
        self.assertEqual(transport.call_count, 1)
        actor_transport.assert_not_called()

    def test_different_actor_budget_is_rejected_at_construction(self):
        selector = self.selector()
        fallback = self.fallback("llm", transport=Mock())
        with self.assertRaisesRegex(ValueError, "share_immutable_budget"):
            GeneratedSkillService(snapshot(), fallback, selector)

    def test_fixture_registry_requires_explicit_offline_mode(self):
        self.rows = [row(fixture=True)]
        with self.assertRaises(ValueError):
            GeneratedSkillService(snapshot(True), self.fallback(), mode="deterministic")
        with self.assertRaisesRegex(ValueError, "wholly_offline"):
            GeneratedSkillService(snapshot(True), self.fallback(), self.selector(), allow_fixture=True)
        service = GeneratedSkillService(snapshot(True), self.fallback(), mode="deterministic", allow_fixture=True)
        response = service.handle(append_observation(request()))
        self.assertEqual(response["metadata"]["selection_source"], "deterministic_control")
        self.assertTrue(response["metadata"]["generated_library_fixture"])
        self.assertFalse(response["metadata"]["llm_live"])

    def test_provisional_or_wrong_program_hash_cannot_run(self):
        self.rows[0]["lifecycle"] = "provisional"
        with self.assertRaises(ValueError):
            GeneratedSkillService(snapshot(), self.fallback(), mode="deterministic")
        self.rows = [row()]
        self.rows[0]["program_hash"] = "b" * 64
        with self.assertRaises(ValueError):
            GeneratedSkillService(snapshot(), self.fallback(), mode="deterministic")

    def test_snapshot_copy_and_identity_cannot_be_mutated_by_caller(self):
        source = snapshot()
        service = GeneratedSkillService(source, self.fallback(), mode="deterministic")
        source["content_sha256"] = "b"*64
        response = service.handle(append_observation(request()))
        self.assertEqual(response["metadata"]["generated_library_sha256"], "a"*64)
        service._snapshot["content_sha256"] = "c"*64
        with self.assertRaisesRegex(ValueError, "identity_changed"):
            service.handle(request())

    def test_selector_operation_id_does_not_conflict_with_action_id(self):
        service = GeneratedSkillService(snapshot(), self.fallback(), self.selector())
        req = append_observation(request())
        service.handle(req)
        packet = service.selector_decisions[0]["packet"]
        self.assertNotEqual(packet["request_id"], req["request_id"])
        self.assertEqual(packet["current_request"]["request_id"], req["request_id"])
        self.assertEqual(packet["state_id"], req["state_id"])
        self.assertEqual(packet["skills"][0]["target_task_verified"], False)
        preview = service.discovery_service.preview(packet)
        self.assertEqual(preview["operation"], "select")
        self.assertEqual(preview["api_calls"], 0)

    def test_expired_selector_does_not_launch_another_actor_call(self):
        clock = [0.0]
        def delayed_abstention(url, headers, body, timeout):
            self.assertEqual(timeout, 5.0)
            clock[0] = 11.0
            packet = json.loads(body["input"][0]["content"])
            return raw_response({"request_id": packet["request_id"], "state_id": packet["state_id"],
                "selected_skill_id": None, "evidence_ids": [], "explanation_hypothesis": "Abstain."})
        fallback = self.fallback()
        fallback.handle = Mock(wraps=fallback.handle)
        service = GeneratedSkillService(snapshot(), fallback, self.selector(delayed_abstention),
                                        timeout_seconds=10)
        with patch("orchestrator.generated_skill_service.time.monotonic", side_effect=lambda: clock[0]):
            response = service.handle(append_observation(request()))
        fallback.handle.assert_not_called()
        self.assertEqual(response["fallback_reason"], "generated_service_deadline")
        self.assertTrue(response["metadata"]["generated_actor_skipped_deadline"])
        self.assertFalse(response["metadata"]["llm_live"])

    def test_fallback_receives_only_remaining_total_deadline(self):
        clock = [0.0]
        def delayed_abstention(url, headers, body, timeout):
            self.assertEqual(timeout, 5.0)
            clock[0] = 4.0
            packet = json.loads(body["input"][0]["content"])
            return raw_response({"request_id": packet["request_id"], "state_id": packet["state_id"],
                "selected_skill_id": None, "evidence_ids": [], "explanation_hypothesis": "Abstain."})
        fallback = self.fallback()
        original_handle, original_timeout = fallback.handle, fallback.timeout_seconds
        observed = []
        def check_remaining(req):
            observed.append(fallback.timeout_seconds)
            return original_handle(req)
        fallback.handle = check_remaining
        service = GeneratedSkillService(snapshot(), fallback, self.selector(delayed_abstention),
                                        timeout_seconds=10)
        with patch("orchestrator.generated_skill_service.time.monotonic", side_effect=lambda: clock[0]):
            response = service.handle(append_observation(request()))
        self.assertEqual(observed, [5.95])
        self.assertEqual(fallback.timeout_seconds, original_timeout)
        self.assertFalse(response["metadata"]["generated_actor_skipped_deadline"])

    def test_live_actor_transport_uses_remaining_time_without_breaking_replay(self):
        clock = [0.0]
        def delayed_abstention(url, headers, body, timeout):
            clock[0] = 4.0
            packet = json.loads(body["input"][0]["content"])
            return raw_response({"request_id": packet["request_id"], "state_id": packet["state_id"],
                "selected_skill_id": None, "evidence_ids": [], "explanation_hypothesis": "Abstain."})
        selector = self.selector(delayed_abstention)
        observed = []
        def actor_transport(url, headers, body, timeout):
            observed.append(timeout)
            raise TimeoutError("fixture failure, no actual API")
        fallback = self.fallback("llm", ledger=selector.ledger_path, transport=actor_transport)
        service = GeneratedSkillService(snapshot(), fallback, selector, timeout_seconds=10)
        req = append_observation(request())
        with patch("orchestrator.generated_skill_service.time.monotonic", side_effect=lambda: clock[0]):
            response = service.handle(req)
            clock[0] = 1.0  # Different elapsed time still reuses immutable requests.
            repeated = service.handle(req)
        self.assertEqual(observed, [5.95])
        self.assertEqual(response["fallback_reason"], "api_timeout")
        self.assertEqual(repeated["fallback_reason"], "api_timeout")
        self.assertTrue(repeated["metadata"]["cached"])
        self.assertEqual(budget_summary(selector.ledger_path)["reserved_calls"], 2)

    def test_scan_limit_abstains_without_hidden_physics(self):
        service = GeneratedSkillService(snapshot(), self.fallback(), mode="deterministic", max_scan_actions=1)
        response = service.handle(append_observation(request()))
        self.assertFalse(response["metadata"]["generated_program_executed"])
        self.assertEqual(response["metadata"]["generated_skill_applicability"][0]["not_applicable_reason"], "scan_limit")


if __name__ == "__main__":
    unittest.main()
