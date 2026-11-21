"""Offline fixtures. Transport/physics outcomes below are NOT live evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.test_skill_discovery import synthesis, config, raw_response
from tests.test_hierarchical_planner import request, policy, plan_value, feedback
from tests.test_pose_guard_recovery import source, PROGRAM as GUARD_PROGRAM
from tests.test_generated_skill_service import row
from tests.test_skill_validation import fixture_case
from tools.python_skill_fixture import PROGRAM
from orchestrator.lesson_memory import (evidence_view, make_lesson, validate_lesson,
    freeze_lessons, validate_library, lesson_cards, sign, freeze_bundle, load_bundle)
from orchestrator.lesson_discovery import LessonDiscoveryService, knowledge_only_packet
from orchestrator.skill_registry import SkillRegistry
from orchestrator.skill_program import program_hash
from orchestrator.pose_guard_recovery import make_guard
from orchestrator.python_skill_program import step_program
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.conditional_plan import validate_plan


def claim(packet):
    facts = evidence_view(packet)
    return dict(title="Fixture lesson", hypothesis="An alternative may help, not a proven cause.",
        when_to_apply="Observed bounded path failure.", when_not_to_apply="Absent observation.",
        evidence_ids=[facts[0]["evidence_id"]], counterexample_ids=[],
        roles=["current mover"], trigger_kind=facts[0]["observed_kind"])


class LessonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lesson_OFFLINE_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        net = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("HTTP_FORBIDDEN"))
        self.network = net.start(); self.addCleanup(net.stop); self.addCleanup(self.network.assert_not_called)
        env = patch.dict(os.environ, DISCOVERY_FIXTURE_KEY="OFFLINE_NOT_A_KEY")
        env.start(); self.addCleanup(env.stop)
        self.packet = synthesis(); self.packet["operation"] = "synthesize"
        self.packet["context"]["lesson_stage"] = "extract"
        self.lesson = make_lesson(claim(self.packet), self.packet, {"origin": "manual_authored"})

    def test_facts_are_projected_and_causes_remain_unknown(self):
        facts = evidence_view(self.packet)
        self.assertEqual(facts[0]["observed_kind"], "path_failed")
        self.assertIsNone(facts[0]["failure_cause"])
        self.assertIsNone(facts[0]["blocker"])
        self.assertTrue(validate_lesson(self.lesson)["claims_are_hypotheses"])

    def test_invented_ids_false_stage_and_extra_facts_rejected(self):
        for update in ({"evidence_ids": ["invented"]}, {"trigger_kind": "stability_failed"},
                       {"facts": "collision proven"}, {"counterexample_ids": ["invented"]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                make_lesson(dict(claim(self.packet), **update), self.packet, {"origin": "manual_authored"})

    def test_changed_lesson_hash_and_forged_live_origin_rejected(self):
        bad = deepcopy(self.lesson); bad["claim"]["title"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_lesson(bad)
        bad = {k:v for k,v in self.lesson.items() if k != "content_sha256"}
        bad["provenance"] = {"origin":"actual_llm"}
        with self.assertRaises((ValueError, KeyError)):
            validate_lesson(sign(bad))

    def test_extract_transport_and_cache_no_real_api(self):
        send = Mock(return_value=raw_response(dict(request_id=self.packet["request_id"],
            state_id=self.packet["state_id"], lesson=claim(self.packet))))
        service = LessonDiscoveryService(config(), self.root/"lesson.sqlite", send)
        self.addCleanup(service.close)
        response = service.propose(self.packet)
        self.assertEqual(response["status"], "proposed", response)
        self.assertEqual(response["origin"], "injected_transport")
        service.propose(self.packet)
        self.assertEqual(send.call_count, 1)

    def test_rejected_lesson_does_not_retry(self):
        wrong = dict(claim(self.packet), evidence_ids=["invented"])
        send = Mock(return_value=raw_response(dict(request_id=self.packet["request_id"],
            state_id=self.packet["state_id"], lesson=wrong)))
        service = LessonDiscoveryService(config(), self.root/"bad.sqlite", send)
        self.addCleanup(service.close)
        self.assertEqual(service.propose(self.packet)["status"], "rejected")
        self.assertEqual(send.call_count, 1)

    def test_program_prompt_has_lesson_and_no_prescribed_repair(self):
        packet = deepcopy(self.packet)
        packet["context"].update(lesson_stage="program", lesson=self.lesson)
        service = LessonDiscoveryService(config(enabled=False, max_calls=0, max_usd=0), self.root/"preview.sqlite")
        self.addCleanup(service.close)
        body = service.preview(packet)["body"]
        self.assertIn("hypothesis", body["instructions"])
        self.assertNotIn("capacity-sufficient", body["instructions"])
        self.assertIn(self.lesson["content_sha256"], body["input"][0]["content"])
        packet["context"]["coordinate_contract"] = {}
        with self.assertRaisesRegex(ValueError, "legacy_solution"):
            service.preview(packet)

    def test_admission_feedback_cannot_enter_extraction(self):
        packet = deepcopy(self.packet)
        packet["operation"] = "refine"; packet["previous_program"] = PROGRAM
        packet["validation_feedback"] = [dict(phase="admission", outcome="PASS")]
        service = LessonDiscoveryService(config(), self.root/"hidden.sqlite")
        self.addCleanup(service.close)
        with self.assertRaisesRegex(ValueError, "debug"):
            service.preview(packet)

    def test_independent_reconstruction_excludes_original_solution_and_paths(self):
        original = deepcopy(self.packet)
        original["previous_program"] = {"source":"DO_NOT_SEND_ORIGINAL_SOLUTION"}
        original["validation_feedback"] = [{"secret":"DO_NOT_SEND_SUCCESS_ACTION"}]
        packet = knowledge_only_packet(original, self.lesson, request())
        self.assertNotIn("previous_program", packet)
        self.assertNotIn("validation_feedback", packet)
        service = LessonDiscoveryService(config(), self.root/"transfer.sqlite")
        self.addCleanup(service.close)
        body = service.preview(packet)["body"]
        sent = json.dumps(body)
        self.assertNotIn("DO_NOT_SEND", sent)
        self.assertNotIn("response_path", sent)
        self.assertTrue(packet["context"]["lesson_transfer_test"])

    def registered(self):
        registry = SkillRegistry(self.root/"registry", True, fixture=True)
        candidate = registry.propose(PROGRAM, dict(description="Fixture only", lesson_id=self.lesson["content_sha256"]),
                                     {"origin":"fixture_authored"})
        cases = [fixture_case(self.root, "debug", "debug", "a", policy=PROGRAM),
                 fixture_case(self.root, "a1", "admission", "b", policy=PROGRAM),
                 fixture_case(self.root, "a2", "admission", "c", policy=PROGRAM)]
        registry.set_manifests(candidate["candidate_id"], {"phase":"debug", "cases":[cases[0][0]]},
                              {"phase":"admission", "cases":[r[0] for r in cases[1:]]})
        for case, artifacts, _ in cases:
            registry.add_validation(candidate["candidate_id"], case, artifacts)
        self.assertTrue(registry.admit(candidate["candidate_id"])["admitted"])
        return registry.freeze(self.root/"skills.json")

    def test_registered_code_and_lesson_freeze_without_generalization_claim(self):
        skills = self.registered()
        library = freeze_lessons([self.lesson], skills, allow_fixture=True)
        self.assertEqual(len(library["entries"]), 1)
        self.assertEqual(library["entries"][0]["lesson_transfer_test"], "NOT_RUN")
        self.assertFalse(library["entries"][0]["causal_claim_verified"])
        self.assertEqual(validate_library(library, skills, allow_fixture=True), library)
        card = lesson_cards(library)[0]
        self.assertNotIn("source", card)
        self.assertFalse(card["target_verified"])
        with self.assertRaisesRegex(ValueError, "fixture_snapshot"):
            validate_library(library, skills)

    def test_unregistered_lesson_is_not_published(self):
        skills = SkillRegistry(self.root/"empty", True, fixture=True).freeze(self.root/"empty.json")
        self.assertEqual(freeze_lessons([self.lesson], skills, allow_fixture=True)["entries"], [])

    def test_bundle_pins_raw_memory_lessons_code_and_rejects_changed_file(self):
        from orchestrator.explicit_action_memory import append_action_observation, freeze_memory
        from tests.test_explicit_action_memory import outcome
        from worker.explicit_action_contract import canonical_response
        r = request()
        append_action_observation(self.root/"events.jsonl", r, canonical_response(r,None), outcome(), allow_test_root=True)
        freeze_memory(self.root/"events.jsonl", self.root/"memory.json", allow_test_root=True)
        skills = self.registered()
        library = freeze_lessons([self.lesson], skills, allow_fixture=True)
        (self.root/"lessons.json").write_text(json.dumps(library), encoding="utf-8")
        args = dict(allow_fixture=True, allow_test_root=True)
        freeze_bundle(self.root/"memory.json", self.root/"skills.json", self.root/"lessons.json",
                      self.root/"bundle.json", **args)
        self.assertEqual(load_bundle(self.root/"bundle.json", **args)["lessons"], library)
        with self.assertRaises(FileExistsError):
            freeze_bundle(self.root/"memory.json", self.root/"skills.json", self.root/"lessons.json",
                          self.root/"bundle.json", **args)
        with (self.root/"lessons.json").open("a") as f:
            f.write(" ")
        with self.assertRaisesRegex(ValueError, "file_changed"):
            load_bundle(self.root/"bundle.json", **args)

    def test_runtime_guard_uses_common_python_entrypoint(self):
        s = source(); guard = make_guard(s["plan"], 0, s["request"])
        ext = dict(guard=guard, plan=s["plan"], index=0, lessons=[])
        result = step_program(GUARD_PROGRAM, s["request"], extension=ext)
        self.assertIsNotNone(result["response"], result)
        self.assertEqual(result["guard_evidence_id"], guard["evidence_id"])
        self.assertEqual(result["physical_calls"], 0)
        bad = deepcopy(ext); bad["guard"]["state_id"] = "stale"
        with self.assertRaisesRegex(ValueError, "tampered"):
            step_program(GUARD_PROGRAM, s["request"], extension=bad)

    def test_general_hierarchy_executes_authorized_guard_skill(self):
        s = source(); entry = row(fixture=True)
        entry.update(program=GUARD_PROGRAM, program_hash=program_hash(GUARD_PROGRAM),
                     strategy_id=program_hash(GUARD_PROGRAM))
        s["plan"]["recovery_skill_ids"] = [entry["strategy_id"]]
        limits = policy()["episode"]; limits["scout_calls"] = 0
        planner = Mock(); scout = Mock()
        with patch("orchestrator.hierarchical_service.retrieve", return_value=[entry]):
            svc = HierarchicalDecisionService(scout, planner, {}, limits, allow_fixture=True,
                                             lesson_runtime=True, max_skill_attempts=3)
        svc.plan = s["plan"]
        response = svc.handle(s["request"])
        self.assertEqual(response["metadata"]["hierarchy_role"], "recovery")
        self.assertEqual(response["metadata"]["generated_skill_id"], entry["strategy_id"])
        planner.plan.assert_not_called(); scout.handle.assert_not_called()
        n = feedback(s["request"], response, True)
        svc.handle(n)
        self.assertTrue(any(e["kind"] == "physics_feedback" for e in svc.events))

    def test_unapproved_guard_skill_is_not_executed(self):
        s = source(); limits = policy()["episode"]; limits["scout_calls"] = 0; limits["planner_calls"] = 0
        with patch("orchestrator.hierarchical_service.retrieve", return_value=[]):
            svc = HierarchicalDecisionService(Mock(), Mock(), {}, limits, lesson_runtime=True)
        svc.plan = s["plan"]
        response = svc.handle(s["request"])
        self.assertEqual(response["metadata"]["hierarchy_role"], "deterministic_fallback")
        self.assertFalse(response["metadata"].get("generated_program_executed", False))

    def test_model_can_choose_guard_skill_after_previous_plan_did_not_authorize_it(self):
        s = source(); entry = row(fixture=True)
        entry.update(program=GUARD_PROGRAM, program_hash=program_hash(GUARD_PROGRAM),
                     strategy_id=program_hash(GUARD_PROGRAM))
        planner = Mock(); planner.provider_contract = "conditional_plan_v3"
        def reply(r, cards, escalation, **kwargs):
            self.assertIsNotNone(cards[0]["currently_proposed_choice"])
            self.assertFalse(escalation["lesson_runtime"]["guard"]["physical_attempt_executed"])
            p = plan_value(r)
            p["steps"] = [dict(p["steps"][0], mode="registered_skill", pose_reference=None,
                primitive=None, direction=None, second_direction=None, support_ids=[],
                registered_skill_id=entry["strategy_id"])]
            return dict(status="planned", plan=validate_plan(p, r, [entry["strategy_id"]]), metadata={})
        planner.plan.side_effect = reply
        limits = policy()["episode"]; limits["scout_calls"] = 0
        with patch("orchestrator.hierarchical_service.retrieve", return_value=[entry]):
            svc = HierarchicalDecisionService(Mock(), planner, {}, limits, allow_fixture=True,
                                             lesson_runtime=True)
        svc.plan = s["plan"]
        response = svc.handle(s["request"])
        self.assertEqual(response["metadata"]["hierarchy_role"], "plan_skill")
        self.assertEqual(planner.plan.call_count, 1)
        self.assertIsNone(svc.planning_guard_context)

    def test_lessons_reach_planner_without_changing_current_candidate_set(self):
        skills = self.registered()
        library = freeze_lessons([self.lesson], skills, allow_fixture=True)
        planner = Mock(); planner.provider_contract = "conditional_plan_v1"
        planner.plan.side_effect = lambda r, *a, **kw: dict(status="planned",
            plan=validate_plan(plan_value(r), r, []), metadata={"llm_live":False})
        scout = Mock(); limits = policy()["episode"]; limits["scout_calls"] = 0
        svc = HierarchicalDecisionService(scout, planner, skills, limits, allow_fixture=True,
            lesson_runtime=True, lesson_library=library)
        r = request(); original = deepcopy(r)
        answer = svc.handle(r)
        supplied = planner.plan.call_args.args[2]["lesson_references"]
        self.assertEqual(supplied[0]["lesson_id"], self.lesson["content_sha256"])
        self.assertEqual(set(answer["candidate_order"]), set(r["candidate_ids"]))
        self.assertEqual(r, original)
        self.assertFalse(supplied[0]["target_verified"])

    def test_failed_recovery_continues_bounded_without_extra_model_calls(self):
        entry = row(fixture=True)
        program = dict(language="python_policy_v1", source='''def propose(ctx):
    last = ctx.get("last")
    if last is None:
        return None
    for axis in ctx["directions"]:
        c = {"part_id":last["choice"]["part_id"], "pose_id":last["choice"]["pose_id"],
             "skill_id":"single_force_v1", "direction_id":axis, "second_direction_id":None,
             "support_ids":[], "evidence_ids":[last["feedback"]["evidence_id"]]}
        if eligible(c):
            return c
    return None
''')
        entry.update(program=program, program_hash=program_hash(program), strategy_id=program_hash(program))
        planner = Mock(); planner.provider_contract = "conditional_plan_v1"
        def reply(r, *a, **kw):
            p = plan_value(r); p["recovery_skill_ids"] = [entry["strategy_id"]]
            return dict(status="planned", plan=validate_plan(p, r, [entry["strategy_id"]]), metadata={})
        planner.plan.side_effect = reply
        limits = policy()["episode"]; limits["scout_calls"] = 0
        with patch("orchestrator.hierarchical_service.retrieve", return_value=[entry]):
            svc = HierarchicalDecisionService(Mock(), planner, {}, limits, allow_fixture=True,
                                             lesson_runtime=True, max_skill_attempts=2)
        r = request(); a = svc.handle(r)
        n = feedback(r, a, False); b = svc.handle(n)
        self.assertEqual(b["metadata"]["hierarchy_role"], "recovery")
        n2 = feedback(n, b, False)
        n2["attempted_actions"] = n["attempted_actions"] + n2["attempted_actions"]
        n2["attempted_action_ids"] = n["attempted_action_ids"] + n2["attempted_action_ids"]
        n2["path_stage_observations"] = n["path_stage_observations"] + n2["path_stage_observations"]
        c = svc.handle(n2)
        self.assertEqual(c["metadata"]["hierarchy_role"], "recovery")
        self.assertNotEqual(b["action_id"], c["action_id"])
        self.assertEqual(planner.plan.call_count, 1)
        n3 = feedback(n2, c, False); svc.handle(n3)
        self.assertEqual(planner.plan.call_count, 2)


if __name__ == "__main__":
    unittest.main()
