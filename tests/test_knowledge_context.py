"""Offline joins, admission fixtures and interpreter checks; NO physical success claims."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.knowledge_context import KnowledgeContext, skill_knowledge
from orchestrator.lesson_memory import freeze_lessons, make_lesson
from orchestrator.skill_registry import SkillRegistry
from orchestrator.generated_skill_service import GeneratedSkillService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.skill_discovery import SkillDiscoveryService
from orchestrator.python_skill_program import step_program
from tests.test_lesson_pipeline import claim
from tests.test_skill_discovery import synthesis, config
from tests.test_skill_validation import fixture_case
from tests.test_skill_program import request, append_observation, choice
from tests.test_hierarchical_planner import policy
from tools.python_skill_fixture import PROGRAM
from worker.explicit_action_contract import canonical_response, validate_request, validate_response


class KnowledgeContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="knowledge_OFFLINE_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        net = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("HTTP_FORBIDDEN"))
        network = net.start(); self.addCleanup(net.stop); self.addCleanup(network.assert_not_called)
        self.req = request()
        for direction in ("xp", "xn", "yp", "yn", "zp", "zn"):
            append_observation(self.req, choice(direction_id=direction))
        packet = synthesis()
        self.lesson = make_lesson(claim(packet), packet, {"origin": "manual_authored"})
        registry = SkillRegistry(self.root/"registry", True, fixture=True)
        candidate = registry.propose(PROGRAM, dict(description="Offline only", lesson_id=self.lesson["content_sha256"]),
                                     {"origin": "fixture_authored"})
        cases = [fixture_case(self.root, "debug", "debug", "a", policy=PROGRAM),
                 fixture_case(self.root, "a1", "admission", "b", policy=PROGRAM),
                 fixture_case(self.root, "a2", "admission", "c", policy=PROGRAM)]
        registry.set_manifests(candidate["candidate_id"], {"phase": "debug", "cases": [cases[0][0]]},
                              {"phase": "admission", "cases": [r[0] for r in cases[1:]]})
        for case, artifacts, _ in cases:
            registry.add_validation(candidate["candidate_id"], case, artifacts)
        self.assertTrue(registry.admit(candidate["candidate_id"])["admitted"])
        self.skills = registry.freeze(self.root/"skills.json")
        self.lessons = freeze_lessons([self.lesson], self.skills, allow_fixture=True)
        self.context = KnowledgeContext(self.skills, self.lessons, allow_fixture=True)

    def hierarchy(self, **extra):
        return HierarchicalDecisionService(Mock(), Mock(), self.skills, policy()["episode"],
            allow_fixture=True, lesson_library=self.lessons, knowledge_policy="unified_v1", **extra)

    def generated(self):
        actor = Mock(mode="deterministic", timeout_seconds=60)
        actor.handle.side_effect = lambda r: canonical_response(r, None)
        return GeneratedSkillService(self.skills, actor, mode="deterministic", allow_fixture=True,
            lesson_library=self.lessons, knowledge_policy="unified_v1")

    def test_code_lesson_evidence_scope_join_and_input_immutability(self):
        before = deepcopy((self.skills, self.lessons, self.req))
        view = self.context.query(self.req)
        bundle = view["skills"][0]
        self.assertEqual(bundle["program"], PROGRAM)
        self.assertEqual(bundle["lessons"][0]["lesson_id"], self.lesson["content_sha256"])
        self.assertTrue(bundle["validation_scope"])
        self.assertEqual(bundle["lessons"][0]["observed_facts"], self.lesson["observed_facts"])
        self.assertFalse(bundle["current_attempt_verified"])
        self.assertIsNone(bundle["current_part_id_mapping"])
        self.assertTrue(bundle["requires_fresh_physics"])
        self.assertEqual(bundle["execution_roles"]["status"], "RECORDED")
        self.assertFalse(bundle["execution_roles"]["explicit_plan"]["explicit_plan_execution_verified"])
        self.assertFalse(bundle["execution_roles"]["automatic_recovery"]["authorized"])
        self.assertEqual(skill_knowledge(bundle)["execution_roles"], bundle["execution_roles"])
        self.assertEqual(before, (self.skills, self.lessons, self.req))
        bundle["program"]["source"] = "tamper"
        self.assertEqual(self.context.query(self.req)["skills"][0]["program"], PROGRAM)

    def test_unlinked_observations_and_citation_boundary_preserved(self):
        view = self.context.query(self.req)
        self.assertEqual(view["audit"]["unlinked_observation_ids"], sorted(self.req["evidence_ids"]))
        self.assertEqual([r["record"] for r in view["observations"]], self.req["episode_feedback"])
        self.assertTrue(all(not link["citable_in_current_response"] for link in view["skills"][0]["evidence_links"]))
        self.assertEqual(view["audit"]["request_evidence_ids"], self.req["evidence_ids"])

    def test_legacy_skill_without_lesson_stays_available(self):
        view = KnowledgeContext(self.skills, allow_fixture=True).query(self.req)
        self.assertEqual(view["skills"][0]["lesson_status"], "NO_LINKED_LESSON")

    def test_tampered_lesson_and_fixture_live_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeContext(self.skills, self.lessons)
        changed = deepcopy(self.lessons); changed["entries"][0]["lesson"]["claim"]["hypothesis"] = "tampered"
        with self.assertRaises(ValueError):
            KnowledgeContext(self.skills, changed, allow_fixture=True)

    def test_no_old_failure_trigger_after_new_pass(self):
        req = deepcopy(self.req)
        append_observation(req, choice(part_id="b"), path=True, stability=True, status="PASS", accepted=True)
        self.assertEqual(self.context.query(req)["audit"]["trigger"], "unknown")

    def test_byte_limit_and_count_limit_are_explicit_not_pruning(self):
        view = self.context.query(self.req, max_skill_bytes=1024)
        self.assertEqual(view["skills"], [])
        self.assertEqual(view["audit"]["excluded_skills"][0]["reason"], "skill_context_byte_limit")
        self.assertEqual(len(view["observations"]), 6)
        # Synthetic large registry after validated fixture load, not physical admissions.
        bundle = deepcopy(self.context._bundles[next(iter(self.context._bundles))])
        self.context._bundles = {}
        for i in range(40):
            value = deepcopy(bundle); value["skill_id"] = str(i).zfill(64)
            self.context._bundles[value["skill_id"]] = value
        view = self.context.query(self.req, max_skills=2)
        self.assertEqual(len(view["skills"]), 2)
        self.assertEqual(len(view["audit"]["excluded_skills"]), 38)
        self.assertTrue(all(e["reason"] == "skill_count_limit" for e in view["audit"]["excluded_skills"]))
        with self.assertRaises(ValueError):
            self.context.query(self.req, max_skills=33)

    def test_same_context_reaches_both_runtime_selectors(self):
        hierarchical = self.hierarchy()
        cards = hierarchical._cards(self.req, time.monotonic()+20)
        generated = self.generated()
        row = generated._rows(self.req)[0]
        step = step_program(row["program"], self.req)
        self.assertIsNotNone(step["response"])
        packet = generated.selection_packet(self.req, [(row, step)])
        self.assertEqual(cards[0]["knowledge"], packet["skills"][0]["knowledge"])
        self.assertEqual(hierarchical.knowledge_queries[-1]["audit"], packet["context"]["knowledge_context"])
        self.assertTrue(packet["skills"][0]["knowledge"]["lessons"])
        self.assertIn("execution_roles", packet["skills"][0]["knowledge"])
        selector = SkillDiscoveryService(config(), self.root/"preview.sqlite")
        # Fixture admission must never become a live selectable skill. Test the
        # encoder separately without weakening that existing public gate.
        with self.assertRaisesRegex(ValueError, "invalid_registered_skill_options"):
            selector.preview(packet)
        body = selector._body({"request": packet})
        self.assertEqual(json.loads(body["input"][0]["content"])["skills"][0]["knowledge"], cards[0]["knowledge"])
        self.assertNotIn("knowledge_context", self.req)

    def test_offline_execution_preserves_action_guards_and_audit(self):
        service = self.generated()
        response = service.handle(self.req)
        validate_response(self.req, response)
        self.assertTrue(response["metadata"]["generated_program_executed"])
        self.assertFalse(response["metadata"]["llm_live"])
        self.assertFalse(response["metadata"]["physics_bypass"])
        self.assertEqual(set(response["candidate_order"]), set(self.req["candidate_ids"]))
        self.assertEqual(response["metadata"]["knowledge_context"]["policy"], "unified_v1")

    def test_v4_sol_projection_keeps_linked_lesson_and_audit(self):
        service = self.hierarchy()
        cards = service._cards(self.req, time.monotonic()+20)
        planner = SolPlanService(policy()["roles"]["planner"], self.root/"sol.sqlite",
            provider_contract="conditional_plan_v4", planning_feedback_policy="repair_evidence_v2")
        envelope = dict(request=self.req, skills=cards,
            escalation={"knowledge_context": service.knowledge_queries[-1]["audit"]},
            max_steps=4, timeout_seconds=30, plan_provider_contract="conditional_plan_v4",
            planning_feedback_policy="repair_evidence_v2")
        planner._validate_envelope(envelope)
        body = planner._body(envelope)
        packet = json.loads(body["input"][0]["content"])
        self.assertEqual(packet["registered_skills"][0]["knowledge"], cards[0]["knowledge"])
        self.assertEqual(packet["escalation"]["knowledge_context"]["policy"], "unified_v1")
        self.assertIn("NOT additional allowed citations", body["instructions"])

    def test_missing_acceptance_is_unknown_not_pass_or_transition(self):
        req = deepcopy(self.req)
        record = req["episode_feedback"][0]
        record.update(status="PASS", path_success=True, stability_success=True)
        del record["accepted_for_transition"]
        del record["deadline_exceeded"]
        view = self.context.query(req)
        observation = view["observations"][0]
        self.assertIsNone(observation["accepted_for_transition"])
        self.assertIsNone(observation["deadline_exceeded"])
        self.assertEqual(observation["record"]["status"], "PASS")
        self.assertEqual(view["audit"]["observations_with_missing_acceptance"], [record["evidence_id"]])

    def test_conflicting_evidence_id_rejected(self):
        req = deepcopy(self.req)
        req["episode_feedback"].append(dict(req["episode_feedback"][0], status="ERROR"))
        with self.assertRaisesRegex(ValueError, "conflicting_knowledge_evidence_id"):
            self.context.query(req)

    def test_hierarchy_handle_delivers_view_and_keeps_fallback(self):
        service = self.hierarchy()
        service.limits["scout_calls"] = 0
        service.planner.provider_contract = "conditional_plan_v1"
        service.planner.plan.return_value = dict(status="rejected", fallback_reason="offline_fixture")
        response = service.handle(self.req)
        validate_response(self.req, response)
        req, cards, escalation = service.planner.plan.call_args.args
        self.assertEqual(req, validate_request(self.req))
        self.assertEqual(cards[0]["knowledge"]["lessons"][0]["lesson_id"], self.lesson["content_sha256"])
        self.assertEqual(escalation["knowledge_context"]["policy"], "unified_v1")
        self.assertNotIn("lesson_references", escalation)
        self.assertEqual(response["metadata"]["hierarchy_role"], "deterministic_fallback")
        service.handle(self.req)
        self.assertEqual(service.planner.plan.call_count, 1)
        self.assertEqual(len(service.knowledge_queries), 1)


if __name__ == "__main__":
    unittest.main()
