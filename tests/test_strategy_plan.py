"""Offline symbolic-contract/compiler/provider fixtures; no API or physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.conditional_plan import task_scope, verify_geometry
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchy_budget import HierarchyBudget
from orchestrator.knowledge_retrieval import KnowledgeRetrievalService
from orchestrator.skill_program import program_hash
from orchestrator.strategy_plan import CONTRACT, STEP_KEYS, compile_plan, schema
from orchestrator.strategy_planner import StrategyPlanService
from tests.test_hierarchical_planner import request, policy, feedback
from tests.test_explicit_action_service import provider
from tests.test_skill_program import program
from worker.explicit_action_contract import canonical_response, digest, validate_request


def value(req=None):
    req = request() if req is None else req
    return dict(request_id=req["request_id"], state_id=req["state_id"],
        candidate_order=list(req["candidate_ids"]), recovery_skill_ids=[],
        steps=[dict(part_id=part, procedure="axis_search_v1", registered_skill_id=None,
                    pose_policy="select_catalog_once_v1" if i == 0 else "keep_orientation_v1",
                    support_policy="preserve_add_v1",
                    max_attempts=3, evidence_ids=[]) for i, part in enumerate(req["candidate_ids"])])


def card():
    code = program()
    pin = program_hash(code)
    return dict(skill_id=pin, program_hash=pin, program=code,
                lifecycle="registered_scoped", currently_proposed_choice=None,
                card={"description": "Offline IR fixture, no physics claim"})


def registered_value(req=None, selected=None):
    result = value(req)
    result["steps"][0].update(procedure="registered_skill", registered_skill_id=(selected or card())["skill_id"])
    return result


class StrategyContractTests(unittest.TestCase):
    def test_wire_schema_contains_only_symbolic_choices(self):
        result = schema(request(), [card()["skill_id"]])
        self.assertFalse(result["additionalProperties"])
        self.assertEqual(result["properties"]["candidate_order"]["minItems"], 2)
        self.assertEqual(result["properties"]["candidate_order"]["maxItems"], 2)
        self.assertEqual(result["properties"]["recovery_skill_ids"]["maxItems"], 0)
        variants = result["properties"]["steps"]["items"]["anyOf"]
        for variant in variants:
            self.assertEqual(set(variant["required"]), STEP_KEYS)
            self.assertFalse(variant["additionalProperties"])
            self.assertEqual(set(variant["properties"]), STEP_KEYS)
            self.assertEqual(variant["properties"]["max_attempts"],
                             dict(type="integer", minimum=1, maximum=8))
        self.assertEqual(variants[0]["properties"]["registered_skill_id"], {"type": "null"})
        self.assertEqual(variants[1]["properties"]["procedure"]["enum"], ["registered_skill"])
        empty = schema(request(), [])
        self.assertNotIn("registered_skill", empty["properties"]["steps"]["items"]["properties"]["procedure"]["enum"])

    def test_compiler_derives_conditional_sets_and_preserves_inputs(self):
        req, wire, info = request(), value(), {}
        original = deepcopy((req, wire))
        result = compile_plan(wire, req, [], diagnostics=info)
        self.assertEqual((req, wire), original)
        self.assertEqual(result["steps"][0]["expected_remaining"], ["1", "2", "base"])
        self.assertEqual(result["steps"][1]["expected_remaining"], ["1", "base"])
        self.assertEqual([s["mode"] for s in result["steps"]], ["strategy", "strategy"])
        self.assertEqual(result["task_scope"], task_scope(req))
        self.assertEqual(result["source_geometry_parts"], req["geometry"]["parts"])
        self.assertEqual(result["first_request_sha256"], digest(validate_request(req)))
        self.assertEqual(result["source_poses"]["p0"], [[1,0,0], [0,1,0], [0,0,1]])
        self.assertFalse(result["physics_success_implied"])
        self.assertEqual(result["plan_contract"], CONTRACT)
        self.assertEqual(info["phase"], "compiled")
        self.assertEqual(compile_plan(wire, req, [card()])["plan_id"], result["plan_id"])
        wire["steps"][0]["max_attempts"] = 4
        self.assertNotEqual(compile_plan(wire, req, [])["plan_id"], result["plan_id"])

    def test_horizon_and_final_body_are_host_bounded(self):
        short = value(); short["steps"] = short["steps"][:1]
        self.assertEqual(len(compile_plan(short, request(), [], max_steps=1)["steps"]), 1)
        with self.assertRaisesRegex(ValueError, "length"):
            compile_plan(value(), request(), [], max_steps=1)
        for limit in (True, 0, 65, 1.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "horizon"):
                compile_plan(short, request(), [], max_steps=limit)
        req = request(); req.update(base_part=None, remaining_parts=["1", "2"], candidate_ids=["1", "2"])
        with self.assertRaisesRegex(ValueError, "length"):
            compile_plan(value(req), req, [])

    def test_rejects_stale_identity_incomplete_order_and_unapproved_fields(self):
        changes = [lambda v:v.update(request_id="stale"), lambda v:v.update(state_id="stale"),
                   lambda v:v.update(candidate_order=["1"]), lambda v:v.update(candidate_order=["1", "1"]),
                   lambda v:v.update(candidate_order=["1", "unknown"]),
                   lambda v:v.update(physics_success=True)]
        for mutate in changes:
            candidate = value(); mutate(candidate)
            with self.subTest(value=candidate), self.assertRaises(ValueError):
                compile_plan(candidate, request(), [])

    def test_rejects_physical_tuple_in_every_wire_form(self):
        for name, forbidden in (("direction", "xp"), ("direction_id", "xp"),
                ("second_direction", "yn"), ("pose_reference", "p0"), ("pose_id", "p0"),
                ("support_ids", ["1"]), ("expected_remaining", ["1", "2", "base"]),
                ("mode", "action"), ("vector", [1,0,0])):
            candidate = value(); candidate["steps"][0][name] = forbidden
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "step_fields"):
                compile_plan(candidate, request(), [])

    def test_rejects_repeated_base_unknown_movers_and_bad_vocabularies(self):
        cases = [("part_id", "base"), ("part_id", "invented"), ("part_id", None),
                 ("procedure", "single_force_v1"), ("procedure", ["axis_search_v1"]),
                 ("pose_policy", "p0"), ("support_policy", "release_all"),
                 ("registered_skill_id", card()["skill_id"])]
        for name, bad in cases:
            candidate = value(); candidate["steps"][0][name] = bad
            with self.subTest(field=name, value=bad), self.assertRaises(ValueError):
                compile_plan(candidate, request(), [card()])
        repeated = value(); repeated["steps"][1]["part_id"] = repeated["steps"][0]["part_id"]
        with self.assertRaisesRegex(ValueError, "repeated_strategy_mover"):
            compile_plan(repeated, request(), [])

    def test_small_attempt_limit_requires_real_integer(self):
        for cap in (True, False, 0, 9, -1, 1.0, "2", None):
            candidate = value(); candidate["steps"][0]["max_attempts"] = cap
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "attempt_limit"):
                compile_plan(candidate, request(), [])
        for cap in (1, 8):
            candidate = value(); candidate["steps"][0]["max_attempts"] = cap
            self.assertEqual(compile_plan(candidate, request(), [])["steps"][0]["max_attempts"], cap)

    def test_citations_must_be_current_and_unique(self):
        req = request()
        req = feedback(req, canonical_response(req, None), False)
        candidate = value(req); candidate["steps"][0]["evidence_ids"] = list(req["evidence_ids"])
        compile_plan(candidate, req, [])
        for ids in (["historical-lesson-id"], req["evidence_ids"]*2, [None]):
            candidate["steps"][0]["evidence_ids"] = ids
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "evidence_ids"):
                compile_plan(candidate, req, [])

    def test_no_separate_automatic_recovery_authorization(self):
        for recovery in ([card()["skill_id"]], None, ""):
            candidate = value(); candidate["recovery_skill_ids"] = recovery
            with self.subTest(value=recovery), self.assertRaisesRegex(ValueError, "new_step"):
                compile_plan(candidate, request(), [card()])

    def test_bounded_support_search_is_fixed_procedure_only_and_keeps_attempt_cap(self):
        definitions = schema(request(), [card()["skill_id"]])["properties"]["steps"]["items"]["anyOf"]
        self.assertIn("bounded_support_search_v1", definitions[0]["properties"]["support_policy"]["enum"])
        self.assertNotIn("bounded_support_search_v1", definitions[1]["properties"]["support_policy"]["enum"])
        for procedure in ("axis_search_v1", "two_leg_search_v1"):
            candidate = value()
            candidate["steps"][0].update(procedure=procedure, support_policy="bounded_support_search_v1")
            compiled = compile_plan(candidate, request(), [])
            self.assertEqual(compiled["steps"][0]["max_attempts"], 3)
        selected = registered_value()
        selected["steps"][0]["support_policy"] = "bounded_support_search_v1"
        with self.assertRaisesRegex(ValueError, "requires_fixed_procedure"):
            compile_plan(selected, request(), [card()])

    def test_selected_program_is_pinned_without_execution_or_forced_card_mover(self):
        selected = card()
        selected["currently_proposed_choice"] = canonical_response(request(), None)["choice"]
        candidate = registered_value(selected=selected)
        proposed = selected["currently_proposed_choice"]["part_id"]
        other = next(part for part in request()["candidate_ids"] if part != proposed)
        candidate["steps"][0]["part_id"] = other
        candidate["steps"][1]["part_id"] = proposed
        with patch("orchestrator.skill_program.step_program", side_effect=AssertionError("VM forbidden in compiler")):
            result = compile_plan(candidate, request(), [selected])
        self.assertEqual(result["registered_program_hashes"], {selected["skill_id"]: selected["program_hash"]})
        self.assertEqual(result["steps"][0]["part_id"], other)

    def test_source_omitted_card_requires_consistent_frozen_hash(self):
        selected = card()
        selected["program"] = dict(language="python_policy_v1", source_omitted_for_planning=True,
                                   program_hash=selected["program_hash"])
        result = compile_plan(registered_value(selected=selected), request(), [selected])
        self.assertIn(selected["skill_id"], result["registered_program_hashes"])
        selected["program"]["program_hash"] = "f"*64
        with self.assertRaisesRegex(ValueError, "program_binding"):
            compile_plan(registered_value(selected=selected), request(), [selected])

    def test_rejects_unknown_duplicate_and_changed_program_bindings(self):
        selected = card()
        with self.assertRaisesRegex(ValueError, "unknown_strategy"):
            compile_plan(registered_value(), request(), [])
        with self.assertRaisesRegex(ValueError, "skill_cards"):
            compile_plan(registered_value(), request(), [selected, deepcopy(selected)])
        for mutate in (lambda c:c.update(program_hash="0"*64),
                       lambda c:c.update(skill_id="0"*64),
                       lambda c:c.update(program=None),
                       lambda c:c["program"]["rules"][0].update(when=["always"])):
            changed = card(); mutate(changed)
            with self.subTest(card=changed), self.assertRaisesRegex(ValueError, "program_binding"):
                compile_plan(registered_value(selected=changed), request(), [changed])

    def test_existing_geometry_scope_checks_remain_compatible(self):
        req = request(); result = compile_plan(value(req), req, [])
        verify_geometry(result["source_geometry_parts"], req)
        req["geometry"]["parts"]["1"]["extents"][0] += 1
        with self.assertRaisesRegex(ValueError, "geometry_features"):
            verify_geometry(result["source_geometry_parts"], req)
        req["task_context"]["solver_settings"]["seed"] += 1
        self.assertNotEqual(result["task_scope"], task_scope(req))


class StrategyProviderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="strategy_plan_fixture_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.serial = 0
        environment = patch.dict(os.environ, STRATEGY_FIXTURE_KEY="fixture-only-not-a-real-key",
                                 OPENAI_API_KEY="fixture-only-not-a-real-key")
        environment.start(); self.addCleanup(environment.stop)
        for target in ("urllib.request.OpenerDirector.open", "socket.create_connection"):
            blocked = patch(target, side_effect=AssertionError("actual_network_forbidden"))
            network = blocked.start(); self.addCleanup(blocked.stop)
            self.addCleanup(network.assert_not_called)

    def config(self, role="planner", **changes):
        return {**policy()["roles"][role], "api_key_env": "STRATEGY_FIXTURE_KEY", **changes}

    def service(self, role="planner", transport=None, config=None):
        self.serial += 1
        cfg = config or self.config(role)
        if transport is None:
            raw = provider(value()); raw["model"] = cfg["model"]
            transport = Mock(return_value=raw)
        return StrategyPlanService(cfg, self.root/("role%d.sqlite" % self.serial), transport, role=role)

    def envelope(self, skills=None, escalation=None):
        return dict(request=request(), skills=skills or [], escalation=escalation or {}, max_steps=64,
                    timeout_seconds=30, plan_provider_contract=CONTRACT,
                    planning_feedback_policy="repair_evidence_v2")

    def test_provider_uses_strategy_only_instructions_with_v5_facts_and_knowledge(self):
        svc = self.service()
        selected = card()
        envelope = self.envelope([selected], dict(retrieved_knowledge={"audit": {"status": "partial"},
            "lessons": [{"description": "fixture hypothesis"}]}, knowledge_context={"status": "partial"}))
        original = deepcopy(envelope)
        body = svc._body(envelope)
        self.assertEqual(envelope, original)
        self.assertIsInstance(svc, SolPlanService)
        self.assertEqual(body["text"]["format"]["name"], "disassembly_strategy_plan")
        self.assertFalse(body["store"])
        for absent in ("For direct action steps use mode=action", "pose_reference names", "Return action steps only"):
            self.assertNotIn(absent, body["instructions"])
        for required in ("first proposal", "hold that selected", "physics attempts", "recovery_skill_ids must be empty"):
            self.assertIn(required, body["instructions"])
        for required in ("bounded_support_search_v1", "timely bound", "stability FAIL",
                         "pose stay fixed", "top-k retrieval changes", "never become new current observations"):
            self.assertIn(required, body["instructions"])
        context = json.loads(body["input"][0]["content"])
        self.assertEqual(context["execution_contract"], CONTRACT)
        self.assertEqual(context["retrieved_knowledge"]["audit"]["status"], "partial")
        self.assertNotIn("retrieved_knowledge", context["escalation"])
        for name in ("observation", "decision_brief", "projection_audit", "stability_repair", "path_failure_evidence"):
            self.assertIn(name, context)
        self.assertEqual(context["repair_review"]["skills"][0]["future_authorization_by"],
                         "new_strategy_plan_registered_skill_step")
        self.assertNotIn("recovery_skill_ids_after_a_new_failure", json.dumps(context))

    def test_scout_and_planner_share_contract_with_role_specific_model_and_ledger(self):
        for role in ("scout", "planner"):
            with self.subTest(role=role):
                cfg = self.config(role)
                raw = provider(value()); raw["model"] = cfg["model"]
                transport = Mock(return_value=raw)
                svc = self.service(role, transport)
                result = svc.plan(request(), [], {}, timeout_seconds=30)
                self.assertEqual(result["status"], "planned", result)
                self.assertEqual(result["metadata"]["planning_role"], role)
                self.assertFalse(result["metadata"]["llm_live"])
                self.assertEqual(svc.plan(request(), [], {}, timeout_seconds=30)["status"], "planned")
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(result["plan"]["plan_contract"], CONTRACT)

    def test_wrong_role_model_is_rejected_before_transport(self):
        for role, model in (("scout", "gpt-6.1-sol"), ("planner", "gpt-6-luna")):
            transport = Mock()
            cfg = self.config(role); cfg["model"] = model
            result = self.service(role, transport, cfg).plan(request(), [], {})
            self.assertEqual(result["fallback_reason"], "unsupported_strategy_planning_model")
            transport.assert_not_called()
        cfg = self.config(); cfg["reasoning_effort"] = "none"
        result = self.service(config=cfg).plan(request(), [], {})
        self.assertEqual(result["fallback_reason"], "unsupported_strategy_planning_model")

    def test_contract_and_role_are_explicit_opt_in(self):
        for options in ({"role": "learner"}, {"provider_contract": "conditional_plan_v4"},
                        {"planning_feedback_policy": "legacy_v1"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                StrategyPlanService(self.config(), self.root/"invalid.sqlite", Mock(), **options)
        svc = self.service()
        for field in ("plan_provider_contract", "planning_feedback_policy"):
            envelope = self.envelope(); envelope[field] = "legacy_v1"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                svc._body(envelope)

    def test_invalid_model_output_has_diagnostics_and_no_retry(self):
        bad = value(); bad["steps"][0]["direction_id"] = "xp"
        raw = provider(bad); raw["model"] = self.config()["model"]
        transport = Mock(return_value=raw)
        svc = self.service(transport=transport)
        result = svc.plan(request(), [], {})
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["fallback_reason"], "invalid_strategy_plan")
        self.assertEqual(result["metadata"]["plan_contract_error"], "invalid_strategy_step_fields")
        self.assertEqual(result["metadata"]["plan_contract_diagnostics"]["step_index"], 0)
        self.assertEqual(result["metadata"]["retries"], 0)
        svc.plan(request(), [], {})
        self.assertEqual(transport.call_count, 1)

    def test_stale_returned_identity_and_wrong_returned_model_are_rejected(self):
        for kind in ("identity", "model"):
            wire = value()
            if kind == "identity": wire["request_id"] = "old"
            raw = provider(wire); raw["model"] = self.config()["model"] if kind == "identity" else "gpt-6-luna"
            result = self.service(transport=Mock(return_value=raw)).plan(request(), [], {})
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["fallback_reason"], "invalid_conditional_plan_response")

    def test_timeout_cannot_trigger_transport_retry(self):
        transport = Mock(side_effect=TimeoutError("untrusted raw provider message"))
        svc = self.service(transport=transport)
        result = svc.plan(request(), [], {})
        self.assertEqual(result["fallback_reason"], "api_timeout")
        self.assertNotIn("untrusted raw", json.dumps(result))
        svc.plan(request(), [], {})
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result["metadata"]["retries"], 0)

    def test_shared_hierarchy_and_local_budget_bound_each_role(self):
        p = policy()
        p["roles"]["scout"]["max_calls"] = 1
        budget = HierarchyBudget(self.root/"global.sqlite", p)
        for role in ("scout", "planner"):
            raw = provider(value()); raw["model"] = p["roles"][role]["model"]
            backend = Mock(return_value=raw)
            svc = self.service(role, budget.transport(role, backend), p["roles"][role])
            self.assertEqual(svc.plan(request(), [], {})["status"], "planned")
            if role == "scout":
                req = request(); req["request_id"] += ":second"
                rejected = svc.plan(req, [], {})
                self.assertEqual(rejected["fallback_reason"], "call_budget_exhausted")
            self.assertEqual(backend.call_count, 1)
        summary = budget.summary()
        self.assertEqual(summary["roles"]["scout"], 1)
        self.assertEqual(summary["roles"]["planner"], 1)
        self.assertEqual(summary["live_api_attempts"], 0)

    def test_knowledge_retrieval_keeps_same_owner_config_and_budget(self):
        for role in ("scout", "planner"):
            owner = self.service(role)
            retrieval = KnowledgeRetrievalService.for_owner(owner)
            self.assertEqual(retrieval.ledger_path, owner.ledger_path)
            self.assertEqual(retrieval.config, owner.config)
            self.assertIs(retrieval.transport, owner.transport)
            self.assertIsNone(retrieval._configuration()[0])
            owner.config["model"] = "wrong-model"
            self.assertEqual(retrieval._configuration()[0], "retrieval_owner_configuration_changed")


if __name__ == "__main__":
    unittest.main()
