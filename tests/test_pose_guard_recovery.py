"""OFFLINE fixtures: forbid HTTP; none of these tests establish physical success."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.test_hierarchical_planner import request, plan_value
from tests.test_plan_pose_continuity import wire, tilted
from tests.test_explicit_action_service import provider
from orchestrator.conditional_plan_v3 import compile_plan
from orchestrator.plan_pose_binding import EXACT
from orchestrator.pose_guard_case import prepare, validate_replay
from orchestrator.pose_guard_recovery import make_guard, validate_guard, guard_packet
from orchestrator.pose_guard_runtime import evaluate, step_guard_program
from orchestrator.pose_guard_registry import contract_probe, PoseGuardSkillRegistry
from orchestrator.skill_registry import retrieve
from tools.run_pose_guard_cycle import (BoundedDiscovery, BoundedSol, policy, envelope,
    evaluate_arm, feedback_after, build_observation_view, BODY_CAP)
from worker.explicit_action_contract import digest, validate_request

# Deliberately simple authored contract fixture, NEVER supplied to live Luna.
PROGRAM = dict(language="python_policy_v1", source="""def propose(ctx):
    guard = ctx.get("plan_guard")
    if guard is None:
        return None
    if guard["reason"] != "planned_orientation_unavailable":
        return None
    intended = guard["intended"]
    for pose in ctx["pose_catalog"]:
        choice = {"part_id": intended["part_id"], "pose_id": pose["pose_id"],
                  "skill_id": intended["skill_id"], "direction_id": intended["direction_id"],
                  "second_direction_id": intended["second_direction_id"],
                  "support_ids": intended["support_ids"], "evidence_ids": []}
        if eligible(choice):
            return choice
    return None
""")


def source():
    r = request()
    p = compile_plan(wire(plan_value(r), EXACT), r, [])
    r["pose_catalog"] = [tilted(3)]
    return dict(request=r, plan=p, step_index=0)


def wire_step(r, registered=None):
    value = wire(plan_value(r, parts=[r["candidate_ids"][0]]), EXACT)
    if registered:
        step = value["steps"][0]
        step.update(mode="registered_skill", registered_skill_id=registered, pose_reference=None,
                    primitive=None, direction=None, second_direction=None, support_ids=[], pose_policy=None)
    return value


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pose_guard_OFFLINE_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = prepare(source(), "offline")
        net = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("HTTP_FORBIDDEN"))
        self.network = net.start()
        self.addCleanup(net.stop)
        self.addCleanup(self.network.assert_not_called)
        env = patch.dict(os.environ, OPENAI_API_KEY="OFFLINE_INVALID_NOT_A_KEY")
        env.start()
        self.addCleanup(env.stop)

    def test_guard_is_reproducible_and_explicitly_nonphysical(self):
        b = self.bundle
        g = validate_guard(b["guard"], b["plan"], b["index"], b["request"])
        self.assertFalse(g["physical_attempt_executed"])
        self.assertIsNone(g["path_success"])
        self.assertIsNone(g["stability_success"])
        self.assertIsNone(g["blocker"])
        self.assertFalse(g["physics_reuse_allowed"])
        bad = deepcopy(g); bad["path_success"] = False
        with self.assertRaisesRegex(ValueError, "tampered"):
            validate_guard(bad, b["plan"], b["index"], b["request"])

    def test_present_orientation_or_changed_state_cannot_supply_guard(self):
        s = source()
        s["request"]["pose_catalog"] = request()["pose_catalog"]
        with self.assertRaisesRegex(ValueError, "not_present"):
            make_guard(s["plan"], 0, s["request"])
        s = source(); s["request"]["remaining_parts"].pop()
        with self.assertRaises(ValueError):
            make_guard(s["plan"], 0, s["request"])

    def test_declared_yaws_are_bound_and_source_unchanged(self):
        s = source(); original = deepcopy(s)
        for yaw in (0, 90, 180):
            b = prepare(s, "yaw", yaw)
            validate_replay(b, s)
            self.assertEqual(b["guard"]["reason"], "planned_orientation_unavailable")
            changed = deepcopy(b)
            changed["request"]["pose_catalog"][0]["matrix"][2][3] += 1
            with self.assertRaisesRegex(ValueError, "source_transform"):
                validate_replay(changed, s)
        self.assertEqual(s, original)

    def test_actual_vm_proposes_but_cannot_run_physics(self):
        b = self.bundle; original = deepcopy(b)
        result = step_guard_program(PROGRAM, b["request"], b["guard"], b["plan"], b["index"])
        self.assertIsNotNone(result["response"])
        self.assertFalse(result["response"]["metadata"]["guard_is_physical_failure"])
        self.assertEqual(result["physical_calls"], 0)
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(original, b)
        probe = contract_probe(PROGRAM, b)
        for key in ("not_applicable", "duplicate_action", "invalid_output_rejected",
                    "observation_sensitivity", "no_literal_source_ids"):
            self.assertTrue(probe[key], probe)

    def test_code_not_just_host_abstains_without_guard(self):
        b = self.bundle
        self.assertIsNone(evaluate(PROGRAM, b["request"], None, b["plan"], b["index"])["choice"])
        unconditional = dict(PROGRAM, source=PROGRAM["source"].replace(
            'if guard is None:\n        return None', 'if guard is None:\n        return {"fake": 1}'))
        self.assertFalse(contract_probe(unconditional, b)["not_applicable"])

    def test_generated_code_rejects_imports_and_cannot_cite_guard_as_physics(self):
        b = self.bundle
        with self.assertRaises(ValueError):
            step_guard_program(dict(PROGRAM, source="import os\n" + PROGRAM["source"]),
                               b["request"], b["guard"], b["plan"], b["index"])
        bad = dict(PROGRAM, source=PROGRAM["source"].replace('"evidence_ids": []',
            '"evidence_ids": [guard["evidence_id"]]'))
        result = step_guard_program(bad, b["request"], b["guard"], b["plan"], b["index"])
        self.assertIsNone(result["response"])

    def test_generation_parse_uses_injected_response_never_claims_live(self):
        b = self.bundle
        packet = guard_packet(b["guard"], b["request"], "fixture-generation")
        card = dict(name="Offline fixture", applicability="Guard present", hypothesis="Unverified",
                    evidence_ids=[b["guard"]["evidence_id"]])
        raw = provider(dict(request_id=packet["request_id"], state_id=packet["state_id"],
                            program=PROGRAM, card=card))
        raw["model"] = "gpt-6-luna"
        send = Mock(return_value=raw)
        service = BoundedDiscovery(policy()["roles"]["scout"], self.root / "generation.sqlite", send)
        result = service.propose(packet)
        self.assertEqual(result["status"], "proposed", result)
        self.assertEqual(result["origin"], "injected_transport")
        self.assertFalse(result["metadata"]["llm_live"])
        self.assertEqual(send.call_count, 1)

    def test_preview_and_refinement_do_not_call_transport(self):
        b = self.bundle
        packet = guard_packet(b["guard"], b["request"], "fixture-preview")
        send = Mock(side_effect=AssertionError("TRANSPORT_FORBIDDEN"))
        service = BoundedDiscovery(policy()["roles"]["scout"], self.root / "preview.sqlite", send)
        service.preview(packet)
        packet.update(operation="refine", previous_program=PROGRAM,
                      validation_feedback=[dict(phase="debug", outcome="UNKNOWN")])
        packet.pop("existing_skills")
        service.preview(packet)
        send.assert_not_called()

    def test_overlong_provider_request_rejected_before_transport(self):
        b = self.bundle
        service = BoundedSol(policy()["roles"]["planner"], self.root / "body.sqlite")
        env = envelope(b, [])
        env["escalation"]["oversized"] = "x" * BODY_CAP
        with self.assertRaisesRegex(ValueError, "body_cap|body_too_large"):
            service._body(env)

    def test_without_admission_artifacts_program_cannot_register(self):
        registry = PoseGuardSkillRegistry(self.root / "registry", allow_test_root=True, fixture=True)
        candidate = registry.propose(PROGRAM, dict(description="OFFLINE fixture"), dict(origin="fixture_authored"))
        identity = candidate["candidate_id"]
        cases = [dict(case_id=n, phase=p, program_hash=identity)
                 for n, p in (("d", "debug"), ("a", "admission"))]
        registry.set_manifests(identity, dict(phase="debug", cases=cases[:1]),
                              dict(phase="admission", cases=cases[1:]))
        decision = registry.admit(identity)
        self.assertFalse(decision["admitted"])
        frozen = registry.freeze(self.root / "empty.json")
        self.assertEqual(retrieve(frozen, allow_fixture=True), [])

    def test_shared_build_observations_have_no_code_or_provider_claims(self):
        record = dict(evidence_id="fixture", source_run_id="fixture",
            scope=dict(remaining_parts=["a"], pose_matrix=tilted(3)["matrix"]),
            choice={"part_id": "a"}, result=dict(status="UNKNOWN", path_success=False,
            stability_success=None, evidence_ref="fixture"), program=PROGRAM)
        view = build_observation_view(dict(records=[record]))
        self.assertNotIn("program", view[0])
        self.assertFalse(view[0]["path_success"])
        self.assertIsNone(view[0]["stability_success"])

    def test_failed_physics_feedback_has_exact_action_and_no_success_transition(self):
        b = self.bundle
        response = step_guard_program(PROGRAM, b["request"], b["guard"], b["plan"], b["index"])["response"]
        raw = dict(status="UNKNOWN", path_success=False, stability_success=None,
                   backend="fixture_not_actual_physics", elapsed_seconds=1)
        new = feedback_after(b, response, raw, "offline-failure")
        self.assertEqual(new["request"]["remaining_parts"], b["request"]["remaining_parts"])
        self.assertEqual(new["request"]["attempted_action_ids"], [response["action_id"]])
        self.assertFalse(new["request"]["episode_feedback"][-1]["accepted_for_transition"])
        self.assertFalse(new["request"]["episode_feedback"][-1]["path_success"])

    def test_second_decision_keeps_feedback_within_existing_twelve_record_contract(self):
        b = deepcopy(self.bundle)
        b["request"]["episode_feedback"] = [dict(evidence_id="old:%d" % i,
            status="UNKNOWN") for i in range(12)]
        b["request"]["evidence_ids"] = [r["evidence_id"] for r in b["request"]["episode_feedback"]]
        b["guard"] = make_guard(b["plan"], b["index"], b["request"])
        response = step_guard_program(PROGRAM, b["request"], b["guard"], b["plan"], b["index"])["response"]
        new = feedback_after(b, response, dict(status="UNKNOWN", path_success=False,
            stability_success=None, elapsed_seconds=1, backend="fixture_not_actual_physics"), "new")
        self.assertEqual(len(new["request"]["episode_feedback"]), 12)
        self.assertNotIn("old:0", new["request"]["evidence_ids"])
        self.assertIn("new", new["request"]["evidence_ids"])

    def test_free_model_selection_direct_success_is_not_skill_reuse(self):
        b = self.bundle
        def plan(r, cards, escalation, max_steps=1):
            return dict(status="planned", plan=compile_plan(wire_step(r), r, [], 1))
        service = Mock(); service.plan.side_effect = plan
        physics = Mock()
        physics.run.return_value = dict(status="PASS", backend="fixture_not_actual_physics",
            path_success=True, stability_success=True, elapsed_seconds=1)
        with patch("tools.run_pose_guard_cycle.save") as store:
            result = evaluate_arm({}, b, [], service, physics, self.root / "without_skill")
        self.assertFalse(result["reused"])
        self.assertEqual(result["status"], "PASS_ONE_REMOVAL")
        self.assertFalse(result["whole_assembly_success"])
        self.assertEqual(physics.run.call_count, 1)

    def test_registered_selection_runs_program_and_control_gets_same_evidence(self):
        b = self.bundle
        b["shared_build_observations"] = [dict(evidence_id="OFFLINE_same_fact")]
        row = dict(strategy_id=digest(PROGRAM), program=PROGRAM,
                   card=dict(description="OFFLINE"), lifecycle="registered_scoped")
        identities = []
        def plan(r, cards, escalation, max_steps=1):
            identities.append((r["request_id"], deepcopy(escalation["shared_build_observations"]),
                               deepcopy(r["memory_references"])))
            sid = cards[0]["skill_id"] if cards else None
            value = wire_step(r, sid)
            if sid:
                value["steps"][0]["part_id"] = cards[0]["current_proposal"]["part_id"]
            return dict(status="planned", plan=compile_plan(value, r, [sid] if sid else [], 1))
        service = Mock(); service.plan.side_effect = plan
        physics = Mock()
        physics.run.return_value = dict(status="PASS", backend="fixture_not_actual_physics",
            path_success=True, stability_success=True, elapsed_seconds=1)
        original = deepcopy(b)
        with patch("tools.run_pose_guard_cycle.save"):
            on = evaluate_arm({}, b, [row], service, physics, self.root / "with_skill")
            off = evaluate_arm({}, b, [], service, physics, self.root / "without_skill")
        self.assertTrue(on["reused"])
        self.assertFalse(off["reused"])
        self.assertNotEqual(identities[0][0], identities[1][0])
        self.assertEqual(identities[0][1:], identities[1][1:])
        self.assertEqual(b, original)


if __name__ == "__main__":
    unittest.main()
