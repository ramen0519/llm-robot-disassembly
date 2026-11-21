"""Offline provenance fixtures and saved-input replay; no model/physics calls."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from orchestrator.strategy_evidence import (
    capture_plan_evidence, project_plan_evidence, validate_plan_evidence,
)
from orchestrator.strategy_plan import compile_plan
from tests.test_hierarchical_planner import request, feedback
from tests.test_strategy_plan import value
from worker.explicit_action_contract import canonical_response, digest, validate_choice, validate_request


def memory_request():
    req = request()
    record = dict(evidence_id="memory:fixture", source_run_id="source-fixture",
        source_request_id="source:1", source_task_context=deepcopy(req["task_context"]),
        source_context_sha256=digest(req["task_context"]), source_remaining_parts=list(req["remaining_parts"]),
        source_pose_matrix=deepcopy(req["pose_catalog"][0]["matrix"]),
        source_choice=deepcopy(canonical_response(req, None)["choice"]),
        status="PASS", path_success=True, stability_success=True, accepted_for_transition=True,
        deadline_exceeded=False, verifier_origin="synthetic_fixture_only", record_sha256="b"*64,
        same_task=True, same_conditions=True, reference_only=False,
        current_part_id_mapping=None, current_attempt_verified=False,
        hard_pruning=False, physics_reuse_allowed=False)
    req.update(memory_references=[record], evidence_ids=[record["evidence_id"]])
    return validate_request(req)


def pinned(req=None, cited=None):
    req = memory_request() if req is None else req
    wire = value(req)
    wire["steps"][0]["evidence_ids"] = list(req["evidence_ids"] if cited is None else cited)
    return compile_plan(wire, req, [])


class StrategyEvidenceTests(unittest.TestCase):
    def setUp(self):
        for target in ("urllib.request.OpenerDirector.open", "socket.create_connection", "subprocess.Popen"):
            guard = patch(target, side_effect=AssertionError("external_execution_forbidden"))
            blocked = guard.start(); self.addCleanup(guard.stop); self.addCleanup(blocked.assert_not_called)

    def test_capture_is_independent_full_record_copy_with_source_pins(self):
        req = memory_request(); original = deepcopy(req)
        plan = pinned(req)
        binding = validate_plan_evidence(plan, req)
        record = binding["records"]["memory:fixture"]
        self.assertEqual(record["snapshot"], req["memory_references"][0])
        self.assertEqual(record["snapshot_sha256"], digest(record["snapshot"]))
        self.assertEqual(binding["first_request_sha256"], digest(req))
        self.assertEqual(binding["source_request_id"], req["request_id"])
        self.assertEqual(req, original)
        binding["records"]["memory:fixture"]["snapshot"]["status"] = "ERROR"
        self.assertEqual(plan["evidence_binding"]["records"]["memory:fixture"]["snapshot"]["status"], "PASS")

    def test_top_k_drop_preserves_provenance_without_injection_or_current_citation(self):
        req = memory_request(); plan = pinned(req)
        req.update(request_id="request:later", memory_references=[], evidence_ids=[])
        original = deepcopy(req)
        result = project_plan_evidence(plan, req, ["memory:fixture"])
        self.assertEqual(result["evidence_ids"], [])
        self.assertEqual(result["provenance"]["pinned_only_evidence_ids"], ["memory:fixture"])
        self.assertFalse(result["provenance"]["snapshot_reinjected"])
        self.assertFalse(result["provenance"]["current_success_implied"])
        self.assertEqual(req, original)
        choice = canonical_response(req, None)["choice"]
        choice["evidence_ids"] = result["evidence_ids"]
        validate_choice(req, choice)
        choice["evidence_ids"] = ["memory:fixture"]
        with self.assertRaisesRegex(ValueError, "unknown_evidence_id"):
            validate_choice(req, choice)

    def test_current_citations_are_only_current_pinned_intersection(self):
        req = memory_request(); plan = pinned(req)
        result = project_plan_evidence(plan, req, ["memory:fixture"])
        self.assertEqual(result["evidence_ids"], ["memory:fixture"])
        self.assertEqual(result["provenance"]["pinned_only_evidence_ids"], [])
        newer = feedback(req, canonical_response(req, None), False)
        with self.assertRaisesRegex(ValueError, "not_pinned"):
            project_plan_evidence(plan, newer, [newer["episode_feedback"][-1]["evidence_id"]])

    def test_dynamic_memory_query_decorations_may_change(self):
        req = memory_request(); plan = pinned(req)
        req["memory_references"][0].update(same_task=False, same_conditions=False, reference_only=True)
        result = project_plan_evidence(plan, req, ["memory:fixture"])
        self.assertEqual(result["evidence_ids"], ["memory:fixture"])
        snapshot = plan["evidence_binding"]["records"]["memory:fixture"]["snapshot"]
        self.assertTrue(snapshot["same_conditions"])
        self.assertFalse(snapshot["reference_only"])

    def test_same_id_source_result_and_safety_changes_are_rejected(self):
        for field, changed in (("status", "UNKNOWN"), ("path_success", False),
                ("accepted_for_transition", False), ("deadline_exceeded", True),
                ("source_request_id", "another-source"), ("record_sha256", "c"*64),
                ("current_attempt_verified", True)):
            req = memory_request(); plan = pinned(req)
            req["memory_references"][0][field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "current_record_conflict"):
                project_plan_evidence(plan, req, ["memory:fixture"])

    def test_source_context_hash_and_same_id_source_changes_are_checked(self):
        req = memory_request(); plan = pinned(req)
        req["memory_references"][0]["source_task_context"]["assembly_id"] = "other-product"
        with self.assertRaisesRegex(ValueError, "source_context_mismatch"):
            validate_plan_evidence(plan, req)
        req["memory_references"][0]["source_context_sha256"] = digest(req["memory_references"][0]["source_task_context"])
        with self.assertRaisesRegex(ValueError, "current_record_conflict"):
            validate_plan_evidence(plan, req)

    def test_changed_target_product_solver_or_run_cannot_reuse_pin(self):
        for field in ("product", "solver", "run"):
            req = memory_request(); plan = pinned(req)
            if field == "product": req["task_context"]["assembly_id"] = "another-product"
            elif field == "solver": req["task_context"]["solver_settings"]["seed"] += 1
            else: req["run_id"] = "another-run"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "task_scope_changed"):
                validate_plan_evidence(plan, req)

    def test_other_product_historical_memory_remains_reference_only(self):
        req = memory_request()
        source = req["memory_references"][0]
        source["source_task_context"]["assembly_id"] = "source-product-only"
        source["source_context_sha256"] = digest(source["source_task_context"])
        source.update(same_task=False, same_conditions=False, reference_only=True)
        plan = pinned(req)
        result = project_plan_evidence(plan, req, ["memory:fixture"])
        self.assertTrue(result["provenance"]["historical_reference_only"])
        self.assertFalse(result["provenance"]["physics_reuse_allowed"])

    def test_pin_snapshot_mutation_and_recomputed_self_digest_are_rejected(self):
        req = memory_request()
        for rehash in (False, True):
            plan = pinned(req)
            binding = plan["evidence_binding"]
            binding["records"]["memory:fixture"]["snapshot"]["status"] = "UNKNOWN"
            if rehash:
                binding["binding_sha256"] = digest({k:v for k,v in binding.items() if k != "binding_sha256"})
            with self.subTest(rehash=rehash), self.assertRaisesRegex(ValueError, "tampered"):
                validate_plan_evidence(plan, req)
        plan = pinned(req)
        altered = deepcopy(req); altered["memory_references"][0]["status"] = "UNKNOWN"
        # Replacing the complete self-consistent pin still changes the plan ID.
        plan["evidence_binding"] = capture_plan_evidence(altered, ["memory:fixture"])
        plan["first_request_sha256"] = plan["evidence_binding"]["first_request_sha256"]
        with self.assertRaisesRegex(ValueError, "plan_identity_mismatch"):
            validate_plan_evidence(plan, altered)

    def test_plan_source_request_state_and_first_request_sha_must_match_pin(self):
        req = memory_request()
        for field in ("request_id", "state_id", "first_request_sha256"):
            plan = pinned(req); plan[field] = "changed-source"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "source_mismatch"):
                validate_plan_evidence(plan, req)

    def test_ghost_missing_record_and_changed_step_citations_are_rejected(self):
        req = memory_request()
        with self.assertRaisesRegex(ValueError, "not_supplied"):
            capture_plan_evidence(req, ["ghost"])
        plan = pinned(req); del plan["evidence_binding"]
        with self.assertRaisesRegex(ValueError, "binding_missing"):
            project_plan_evidence(plan, req, ["memory:fixture"])
        plan = pinned(req); binding = plan["evidence_binding"]
        binding["records"] = {}
        binding["binding_sha256"] = digest({k:v for k,v in binding.items() if k != "binding_sha256"})
        with self.assertRaisesRegex(ValueError, "record_set_mismatch"):
            validate_plan_evidence(plan, req)
        plan = pinned(req); plan["steps"][0]["evidence_ids"] = []
        with self.assertRaisesRegex(ValueError, "citation_set_mismatch"):
            validate_plan_evidence(plan, req)

    def test_duplicate_or_memory_episode_identity_is_fail_closed(self):
        req = memory_request()
        req["episode_feedback"] = [dict(evidence_id="memory:fixture", status="PASS")]
        with self.assertRaisesRegex(ValueError, "identity_conflict"):
            capture_plan_evidence(req, ["memory:fixture"])
        req = feedback(request(), canonical_response(request(), None), False)
        req["episode_feedback"].append(deepcopy(req["episode_feedback"][0]))
        with self.assertRaisesRegex(ValueError, "identity_conflict"):
            capture_plan_evidence(req, list(req["evidence_ids"]))
        req = memory_request(); plan = pinned(req)
        req["memory_references"] = []
        req["episode_feedback"] = [dict(evidence_id="memory:fixture", status="PASS")]
        with self.assertRaisesRegex(ValueError, "current_record_conflict"):
            validate_plan_evidence(plan, req)

    def test_episode_observation_mutation_is_not_a_memory_decoration(self):
        req = feedback(request(), canonical_response(request(), None), False)
        plan = pinned(req)
        req["episode_feedback"][0]["reference_only"] = True
        with self.assertRaisesRegex(ValueError, "current_record_conflict"):
            validate_plan_evidence(plan, req)

    def test_new_failure_does_not_rewrite_historical_success_or_imply_current_success(self):
        req = memory_request(); plan = pinned(req)
        newer = feedback(req, canonical_response(req, None), False)
        result = project_plan_evidence(plan, newer, ["memory:fixture"])
        self.assertFalse(result["provenance"]["current_success_implied"])
        self.assertTrue(result["provenance"]["requires_fresh_physics"])
        self.assertEqual(plan["evidence_binding"]["records"]["memory:fixture"]["snapshot"]["status"], "PASS")
        self.assertEqual(newer["episode_feedback"][-1]["path_success"], False)

    def test_empty_citations_have_empty_pins_but_still_bind_plan_identity(self):
        req = request(); plan = pinned(req, [])
        result = project_plan_evidence(plan, req, [])
        self.assertEqual(plan["evidence_binding"]["records"], {})
        self.assertEqual(result["evidence_ids"], [])
        self.assertEqual(result["provenance"]["record_sha256_by_id"], {})
        with self.assertRaisesRegex(ValueError, "not_pinned"):
            project_plan_evidence(plan, req, ["ghost"])

    def test_saved_live_top_k_rotation_keeps_original_plan_provenance(self):
        root = Path(__file__).resolve().parents[1]
        base = root / "runs/strategy_live_20261109_1631/episode"
        decisions = json.loads((base/"hierarchy_decisions.json").read_text(encoding="utf-8"))
        saved = json.loads((base/"hierarchy_plans.json").read_text(encoding="utf-8"))[0]["result"]["plan"]
        from orchestrator.conditional_plan import PLAN_KEYS
        from orchestrator.strategy_plan import STEP_KEYS
        wire = {key:deepcopy(saved[key]) for key in PLAN_KEYS}
        wire["steps"] = [{key:deepcopy(step[key]) for key in STEP_KEYS} for step in saved["steps"]]
        first = decisions[0]["request"]
        next_state = decisions[3]["request"]
        plan = compile_plan(wire, first, [])
        cited = saved["steps"][1]["evidence_ids"]
        self.assertTrue(set(cited) - set(next_state["evidence_ids"]))
        original = deepcopy((first, next_state))
        result = project_plan_evidence(plan, next_state, cited)
        self.assertEqual(result["evidence_ids"], [])
        self.assertEqual(result["provenance"]["pinned_only_evidence_ids"], cited)
        self.assertEqual((first, next_state), original)
        self.assertFalse(result["provenance"]["physics_reuse_allowed"])


if __name__ == "__main__":
    unittest.main()
