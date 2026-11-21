"""Offline Python generation/runtime/lifecycle tests. Physics fixtures are mock."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
from test_skill_program import request, append_observation
from test_skill_validation import fixture_case, save
from test_skill_discovery import config, synthesis, decision, raw_response
from python_skill_fixture import PROGRAM
from orchestrator.python_skill_vm import VM, PolicyError
from orchestrator.python_skill_program import evaluate
from orchestrator.python_skill_discovery import PythonSkillDiscoveryService
from orchestrator.skill_program import step_program, validate_program, program_hash
from orchestrator.skill_registry import SkillRegistry, retrieve
from orchestrator.generated_skill_service import GeneratedSkillService
from worker.explicit_action_contract import canonical_response, validate_response
from worker.path_probe_round import canonical_path_round_response


def python(source):
    return {"language": "python_policy_v1", "source": source}


class RuntimeTests(unittest.TestCase):
    def test_real_python_functions_loop_calculation_and_fresh_action_guard(self):
        req = append_observation(request())
        result = step_program(PROGRAM, req)
        self.assertIsNotNone(result["response"], result)
        self.assertEqual(result["response"]["choice"]["direction_id"], "zp")
        self.assertGreater(result["vm_steps"], 50)
        self.assertGreater(result["scanned"], 1)
        self.assertEqual(result["physical_calls"], 0)
        self.assertEqual(set(result["response"]["candidate_order"]), set(req["candidate_ids"]))
        self.assertEqual(result["scoped_feedback_evidence_id"], req["evidence_ids"][-1])

    def test_matrix_changes_computed_world_direction(self):
        req = request()
        req["pose_catalog"][0]["matrix"] = [[0, 0, 1, 0], [0, 1, 0, 0],
                                           [-1, 0, 0, 0], [0, 0, 0, 1]]
        append_observation(req)
        result = step_program(PROGRAM, req)
        self.assertEqual(result["response"]["choice"]["direction_id"], "xp")

    def test_no_feedback_and_late_feedback_do_not_trigger(self):
        for req in (request(), append_observation(request(), late=True)):
            result = step_program(PROGRAM, req)
            self.assertIsNone(result["response"])
            self.assertFalse(result["triggered"])

    def test_same_code_reuses_new_task_current_ids_without_mapping_source_ids(self):
        req = request()
        req["candidate_ids"] = ["new_part", "new_other"]
        req["remaining_parts"] = ["new_part", "new_other", "base"]
        req["geometry"] = {"parts": {}}
        req["task_context"]["assembly_id"] = "unseen_mock_geometry"
        from test_skill_program import choice
        append_observation(req, choice(part_id="new_part"))
        output = step_program(PROGRAM, req)["response"]["choice"]
        self.assertEqual(output["part_id"], "new_part")
        self.assertNotIn("a", output.values())

    def test_request_and_program_unchanged_and_output_not_replayed_as_verdict(self):
        req = append_observation(request())
        before = deepcopy(req)
        output = step_program(PROGRAM, req)
        self.assertEqual(req, before)
        self.assertFalse(output["response"]["metadata"]["physics_bypass"])
        self.assertFalse(output["physics_verdict_reused"])
        self.assertNotIn("status", output["response"])

    def test_identical_action_cannot_repeat(self):
        req = append_observation(request())
        first = step_program(PROGRAM, req)["response"]
        append_observation(req, first["choice"])
        result = step_program(PROGRAM, req)
        if result["response"]:
            self.assertNotEqual(first["action_id"], result["response"]["action_id"])

    def test_forged_pass_wrong_id_and_metadata_rejected(self):
        for source in (
            'def propose(ctx):\n    return {"status": "PASS"}',
            'def propose(ctx):\n    return {"part_id": "source_only"}',
            'def propose(ctx):\n    return ctx["last"]["choice"]',  # exact duplicate
        ):
            result = step_program(python(source), append_observation(request()))
            self.assertIsNone(result["response"])
            self.assertEqual(result["not_applicable_reason"], "python_execution_rejected")

    def test_forbidden_syntax_and_host_access(self):
        sources = [
            "import os\ndef propose(ctx):\n    return None",
            'def propose(ctx):\n    return open("secret")',
            'def propose(ctx):\n    return eval("1")',
            'def propose(ctx):\n    return ctx.__class__',
            'def propose(ctx):\n    return ctx["last"].pop("choice")',
            'def propose(ctx):\n    while True:\n        pass',
            "def propose(ctx):\n    return 2 ** 9999999",
            "def propose(ctx):\n    ctx['candidate_ids'] = []",
        ]
        for source in sources:
            with self.subTest(source=source):
                with self.assertRaises((ValueError, SyntaxError)):
                    validate_program(python(source))

    def test_step_recursion_allocation_and_range_budgets(self):
        sources = [
            "def propose(ctx):\n    return propose(ctx)",
            "def propose(ctx):\n    return [0] * 999999999",
            "def propose(ctx):\n    return range(999999999)",
            "def propose(ctx):\n    return '%999999999s' % 'a'",
            "def propose(ctx):\n    for a in range(4096):\n        for b in range(4096):\n            pass",
        ]
        for source in sources:
            with self.subTest(source=source):
                result = step_program(python(source), request())
                self.assertIsNone(result["response"])
                self.assertEqual(result["not_applicable_reason"], "python_execution_rejected")

    def test_child_timeout_fails_closed(self):
        with patch("orchestrator.python_skill_program.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(["fixture"], 1)):
            result = step_program(PROGRAM, request())
        self.assertIsNone(result["response"])
        self.assertEqual(result["not_applicable_reason"], "python_wall_timeout")

    def test_no_credential_environment_in_child(self):
        real = subprocess.run
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unit_test_not_real"}):
            with patch("orchestrator.python_skill_program.subprocess.run", wraps=real) as run:
                step_program(PROGRAM, request())
        self.assertNotIn("OPENAI_API_KEY", run.call_args.kwargs["env"])
        self.assertIn("-I", run.call_args.args[0])

    def test_stale_response_rejected_by_existing_host_contract(self):
        req = append_observation(request())
        response = step_program(PROGRAM, req)["response"]
        changed = deepcopy(req)
        changed["state_id"] = "different"
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_response(changed, response)

    def test_oversized_request_rejected_by_host_before_child_launch(self):
        req = append_observation(request())
        req["geometry"]["padding"] = "x" * 512001
        with patch("orchestrator.python_skill_program.subprocess.run") as child:
            with self.assertRaisesRegex(ValueError, "request_too_large"):
                step_program(PROGRAM, req)
        child.assert_not_called()

    def test_explicit_eligibility_budget(self):
        with self.assertRaisesRegex(ValueError, "eligibility_budget"):
            evaluate(PROGRAM, append_observation(request()), max_actions=1)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="python_skill_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"DISCOVERY_FIXTURE_KEY": "fixture"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.net = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("NETWORK_FORBIDDEN"))
        self.net.start()
        self.addCleanup(self.net.stop)

    def service(self, transform=None, **updates):
        def transport(url, headers, body, timeout):
            packet = json.loads(body["input"][0]["content"])
            value = decision(packet)
            value["program"] = deepcopy(PROGRAM)
            raw = raw_response(value if transform is None else transform(value))
            raw["model"] = body["model"]
            return raw
        self.transport = Mock(side_effect=transport)
        return PythonSkillDiscoveryService(config(**updates), self.root / "ledger.sqlite", self.transport)

    def test_injected_generation_is_not_live_or_registration(self):
        service = self.service()
        result = service.propose(synthesis())
        self.assertEqual(result["status"], "proposed", result)
        self.assertEqual(result["origin"], "injected_transport")
        self.assertFalse(result["metadata"]["python_executed"])
        self.assertFalse(result["metadata"]["registered"])
        self.assertEqual(result["metadata"]["retries"], 0)
        self.transport.assert_called_once()

    def test_sol_generator_contract_without_api(self):
        result = self.service(model="gpt-6.1-sol", reasoning_effort="low").propose(synthesis())
        self.assertEqual(result["status"], "proposed", result)
        self.assertEqual(result["origin"], "injected_transport")

    def test_invalid_code_rejected_without_retry(self):
        def bad(value):
            value["program"]["source"] = "import os"
            return value
        result = self.service(bad).propose(synthesis())
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["metadata"]["decision_validation_error"], "invalid_generated_python")
        self.assertEqual(result["metadata"]["python_validation_error"],
                         "python_module_requires_unique_functions_and_propose")
        self.assertFalse(result["metadata"]["python_executed"])
        self.transport.assert_called_once()

    def test_list_comprehension_is_static_checked_but_not_executed_at_generation(self):
        source = "def propose(ctx):\n    return [x for x in ctx]"
        def bad(value):
            value["program"]["source"] = source
            return value
        result = self.service(bad).propose(synthesis())
        self.assertEqual(result["status"], "proposed", result)
        self.assertEqual(result["program"]["source"], source)
        self.assertFalse(result["metadata"]["python_executed"])
        self.transport.assert_called_once()

    def test_documented_commentary_is_not_treated_as_generated_code(self):
        packet = synthesis()
        packet["operation"] = "synthesize"
        value = decision(packet)
        value["program"] = deepcopy(PROGRAM)
        raw = raw_response(value)
        raw["model"] = "gpt-6-luna"
        raw["output"][0]["phase"] = "final_answer"
        commentary = {"type": "message", "role": "assistant", "status": "completed",
                      "phase": "commentary", "content": [
                          {"type": "output_text", "text": "untrusted commentary, not Python"}]}
        raw["output"].insert(0, commentary)
        metadata = {}
        result, reason = self.service()._parse_response(raw, packet, metadata)
        self.assertIsNone(reason)
        self.assertEqual(result["program"], PROGRAM)
        self.assertEqual(metadata["python_commentary_messages_ignored"], 1)
        self.transport.assert_not_called()
        for change in ("missing_final", "two_final", "unlabeled", "late_commentary", "unknown_phase"):
            bad = deepcopy(raw)
            if change == "missing_final":
                bad["output"].pop()
            elif change == "two_final":
                bad["output"].append(deepcopy(bad["output"][-1]))
            elif change == "unlabeled":
                bad["output"][-1].pop("phase")
            elif change == "unknown_phase":
                bad["output"] = [bad["output"][-1]]
                bad["output"][0]["phase"] = "unknown_untrusted_phase"
            else:
                bad["output"].reverse()
            result, reason = self.service()._parse_response(bad, packet, {})
            self.assertIsNone(result, change)
            self.assertIsNotNone(reason, change)

    def test_preview_zero_calls_disabled_config(self):
        service = self.service(enabled=False, max_calls=0, max_usd=0)
        preview = service.preview(synthesis())
        self.assertEqual(preview["api_calls"], 0)
        self.assertFalse(preview["key_accessed"])
        schema = preview["body"]["text"]["format"]["schema"]
        self.assertEqual(schema["properties"]["program"]["properties"]["language"]["enum"], ["python_policy_v1"])
        self.assertEqual(service.propose(synthesis())["status"], "rejected")
        self.transport.assert_not_called()

    def test_duplicate_card_evidence_rejected_with_safe_diagnostic(self):
        def bad(value):
            value["card"]["evidence_ids"] *= 2
            return value
        result = self.service(bad).propose(synthesis())
        self.assertEqual(result["fallback_reason"], "invalid_card_or_evidence")
        self.assertFalse(result["metadata"]["python_card_diagnostics"]["unique_nonempty_evidence"])
        self.transport.assert_called_once()

    def test_budget_does_not_reset_for_refinement(self):
        service = self.service(max_calls=1)
        service.propose(synthesis())
        packet = synthesis("refine2")
        packet.update(operation="refine", previous_program=PROGRAM,
                      validation_feedback=[{"phase": "debug", "outcome": "UNKNOWN",
                                            "source": "debug_mock"}])
        result = service.propose(packet)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["fallback_reason"], "call_budget_exhausted")
        self.transport.assert_called_once()

    def test_cli_accepts_approved_sol_config_and_rejects_disabled(self):
        from run_python_skill_cycle import require_live_config
        service = self.service(model="gpt-6.1-sol", reasoning_effort="low",
                               api_key_env="OPENAI_API_KEY")
        require_live_config(service.config, service)
        self.transport.assert_not_called()
        disabled = dict(service.config, enabled=False)
        other = PythonSkillDiscoveryService(disabled, self.root / "off.sqlite", self.transport)
        with self.assertRaisesRegex(ValueError, "llm_disabled"):
            require_live_config(disabled, other)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="python_registry_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare(self, failed=False):
        registry = SkillRegistry(self.root / "registry", True, fixture=True)
        candidate = registry.propose(PROGRAM, {"description": "human authored fixture"},
                                     {"origin": "fixture_authored"})
        identity = candidate["candidate_id"]
        bundles = [fixture_case(self.root, "debug", "debug", "a", policy=PROGRAM),
                   fixture_case(self.root, "a1", "admission", "b", policy=PROGRAM),
                   fixture_case(self.root, "a2", "admission", "c", policy=PROGRAM, success=not failed)]
        registry.set_manifests(identity, {"phase": "debug", "cases": [bundles[0][0]]},
                              {"phase": "admission", "cases": [b[0] for b in bundles[1:]]})
        for case, artifacts, _ in bundles:
            registry.add_validation(identity, case, artifacts)
        return registry, identity

    def test_registration_freeze_next_request_reuse_and_no_llm_calls(self):
        registry, identity = self.prepare()
        admission = registry.admit(identity)
        self.assertTrue(admission["admitted"], admission)
        snapshot = registry.freeze(self.root / "frozen.json")
        with self.assertRaisesRegex(ValueError, "fixture_snapshot"):
            retrieve(snapshot)

        class Fallback:
            mode, timeout_seconds = "deterministic", 60
            def handle(self, req):
                return canonical_path_round_response(req, "fixture")
        service = GeneratedSkillService(snapshot, Fallback(), mode="deterministic", allow_fixture=True)
        response = service.handle(append_observation(request()))
        self.assertTrue(response["metadata"]["generated_program_executed"])
        self.assertFalse(response["metadata"]["llm_live"])
        self.assertEqual(response["metadata"]["generated_skill_id"], identity)
        from orchestrator.skill_usage import summarize_usage
        usage = summarize_usage(service.decisions, None, [], snapshot, allow_fixture=True)
        self.assertIsInstance(usage, dict)
        # Many Python entries must not each receive a new full service timeout.
        req = append_observation(request())
        precomputed = step_program(PROGRAM, req)
        short = GeneratedSkillService(snapshot, Fallback(), mode="deterministic",
                                      allow_fixture=True, timeout_seconds=0.5)
        with patch("orchestrator.python_skill_program.step_program", return_value=precomputed) as evaluator:
            short.handle(req)
        self.assertLessEqual(evaluator.call_args.kwargs["timeout_seconds"], 0.45)
        self.assertGreater(evaluator.call_args.kwargs["timeout_seconds"], 0)

    def test_failed_admission_cannot_enter_library(self):
        registry, identity = self.prepare(failed=True)
        decision = registry.admit(identity)
        self.assertFalse(decision["admitted"])
        self.assertIn("triggered_admission_condition_failed", decision["reasons"])
        self.assertEqual(registry.freeze(self.root / "empty.json")["entries"], [])

    def test_forged_live_origin_rejected(self):
        registry = SkillRegistry(self.root / "real", True)
        path = self.root / "provider.json"
        hashed = save(path, {"status": "proposed", "origin": "injected_transport",
                            "metadata": {"llm_live": False}, "program": PROGRAM})
        with self.assertRaisesRegex(ValueError, "actual_provider"):
            registry.propose(PROGRAM, {"description": "fixture"}, {"origin": "live_llm",
                "model": "gpt-6-luna", "generation_id": "fixture",
                "provider_response_path": str(path), "provider_response_sha256": hashed})

    def test_existing_legacy_revision_still_readable(self):
        from orchestrator.knowledge_store import load_revision
        store = ROOT / "runs/knowledge_store"
        if not (store / "revisions/rev_0001/manifest.json").exists():
            self.skipTest("historical local knowledge revision unavailable")
        loaded = load_revision(store, "rev_0001")
        self.assertEqual(loaded["manifest"]["registered_skills"], 2)

    def test_python_index_records_interpreter_not_invented_predicates(self):
        from orchestrator.knowledge_index import build_index
        from orchestrator.explicit_action_memory import freeze_memory
        registry, identity = self.prepare()
        registry.admit(identity)
        snapshot = registry.freeze(self.root / "frozen.json")
        # Use an existing immutable real memory snapshot; no new observation.
        memory = json.loads((ROOT / "runs/knowledge_store/revisions/rev_0001/memory.json").read_text())
        index = build_index(memory, snapshot, allow_fixture=True)
        self.assertEqual(index["skills"][0]["trigger_rules"], [])
        self.assertEqual(index["skills"][0]["program_language"], "python_policy_v1")


class RefinementTests(unittest.TestCase):
    def test_rejection_feedback_is_same_ledger_and_no_untrusted_source(self):
        from run_python_skill_cycle import contract_rejection_feedback
        with tempfile.TemporaryDirectory(prefix="python_reject_", dir=ROOT / "runs") as temp:
            root = Path(temp)
            ledger = ROOT / "runs" / "not_opened_ledger.sqlite"
            save(root / "config.json", {"mode": "live", "ledger": str(ledger)})
            save(root / "provider_proposal.json", {
                "status": "rejected", "fallback_reason": "invalid_generated_python",
                "program": {"source": "untrusted never forwarded"},
                "metadata": {"llm_live": True, "python_validation_error": "python_forbidden_syntax:While"}})
            value = contract_rejection_feedback(root, ledger)
            self.assertEqual(value["python_validation_error"], "python_forbidden_syntax:While")
            self.assertNotIn("untrusted", json.dumps(value))
            with self.assertRaisesRegex(ValueError, "bound_live_contract"):
                contract_rejection_feedback(root, root / "wrong.sqlite")

    def test_python_projection_retains_observed_failure_not_raw_ir_or_admission(self):
        from orchestrator.python_skill_evidence import generation_view
        original = {
            "context": {"language_semantics": {"when": "legacy IR"}, "development_only": True},
            "validation_feedback": [{"phase": "debug", "case_id": "debug",
                "outcome": "UNKNOWN", "validation_record_sha256": "bound",
                "current_observation_request": {"candidate_ids": ["current"],
                    "pose_catalog": [], "episode_feedback": [], "geometry": {"large": "raw"}},
                "original_trigger": {"status": "UNKNOWN", "path_success": False,
                    "chosen": {"direction_id": "xp"}, "source_hashes": {"large": "raw"}},
                "recovery": None}]}
        before = deepcopy(original)
        compact = generation_view(original)
        self.assertEqual(original, before)
        self.assertNotIn("language_semantics", compact["context"])
        row = compact["validation_feedback"][0]
        self.assertFalse(row["original_trigger"]["path_success"])
        self.assertEqual(row["original_trigger"]["chosen"], {"direction_id": "xp"})
        self.assertEqual(row["validation_record_sha256"], "bound")
        self.assertNotIn("source_hashes", row["original_trigger"])
        original["validation_feedback"][0]["phase"] = "admission"
        with self.assertRaisesRegex(ValueError, "debug_feedback_only"):
            generation_view(original)

    def test_only_bound_debug_results_supplied_not_admission(self):
        from run_python_skill_cycle import refinement_packet
        with tempfile.TemporaryDirectory(prefix="python_refine_fixture_", dir=ROOT / "runs") as temp:
            root = Path(temp)
            registry = SkillRegistry(root / "registry", fixture=True)
            candidate = registry.propose(PROGRAM, {"description": "fixture"}, {"origin": "fixture_authored"})
            identity = candidate["candidate_id"]
            bundles = [fixture_case(root, "debug", "debug", "a", policy=PROGRAM),
                       fixture_case(root, "admission_secret", "admission", "b", policy=PROGRAM)]
            registry.set_manifests(identity, {"phase": "debug", "cases": [bundles[0][0]]},
                                  {"phase": "admission", "cases": [bundles[1][0]]})
            for case, artifacts, _ in bundles:
                registry.add_validation(identity, case, artifacts)
            manifest = {"fixture": "predeclared"}
            save(root / "precommitted_cases.json", manifest)
            save(root / "config.json", {"ledger": None})
            save(root / "result.json", {"actual_llm_generation": False, "candidate_id": identity})
            packet = refinement_packet(synthesis(), root, manifest)
            self.assertEqual(packet["operation"], "refine")
            self.assertEqual([r["case_id"] for r in packet["validation_feedback"]], ["debug"])
            self.assertNotIn("admission_secret", json.dumps(packet))
            with self.assertRaisesRegex(ValueError, "same_approved_ledger"):
                refinement_packet(synthesis(), root, manifest, live=True, ledger=root / "bad.sqlite")


if __name__ == "__main__":
    unittest.main()
