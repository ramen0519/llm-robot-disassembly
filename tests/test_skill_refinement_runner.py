"""Offline refinement-artifact tests; copied logs are NOT new physics runs.

Only temporary copies of preserved development artifacts are changed. No API,
WSL worker, provider transport, or simulator is invoked by these tests.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_skill_discovery as runner


HISTORY = ROOT / "runs/skill_discovery_implementation_20261019"
DEBUG_ID = "beam_debug_yp"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class SkillRefinementRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="skill_refinement_artifacts_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

        def checked_temporary(path, *args, **kwargs):
            path = Path(path).resolve()
            if path != self.root and self.root not in path.parents:
                raise ValueError("test_artifact_outside_temporary_root")
            return path

        guard = patch.object(runner, "checked", side_effect=checked_temporary)
        guard.start()
        self.addCleanup(guard.stop)
        reader = patch.object(runner, "read", side_effect=lambda path: read_json(checked_temporary(path)))
        reader.start()
        self.addCleanup(reader.stop)
        hasher = patch.object(runner, "sha", side_effect=lambda path:
                              hashlib.sha256(checked_temporary(path).read_bytes()).hexdigest())
        hasher.start()
        self.addCleanup(hasher.stop)
        # Fail loudly if a future helper implementation unexpectedly launches
        # a worker instead of checking saved development observations.
        physics = patch.object(runner, "actual_action", side_effect=AssertionError("no_new_physics"))
        physics.start()
        self.addCleanup(physics.stop)

    def copy_prior(self, name="prior", historical="live_02"):
        source = HISTORY / historical
        target = self.root / name
        target.mkdir()
        for filename in ("result.json", "precommitted_cases.json", "provider_proposal.json"):
            shutil.copyfile(source / filename, target / filename)
        # Inherited old projections are intentionally excluded unless a test
        # explicitly constructs a provenance-bound inheritance chain.
        write_json(target / "generation_packet.json", {"validation_feedback": []})
        relative_files = ["program_step.json", "program_request.json"]
        for stage in ("trigger", "recovery"):
            relative_files += [stage + "/physics/request.json", stage + "/physics/result.json"]
        for relative in relative_files:
            path = source / "cases" / DEBUG_ID / relative
            if path.is_file():
                output = target / "cases" / DEBUG_ID / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
        return target

    @staticmethod
    def inputs(prior):
        return read_json(prior / "result.json"), read_json(prior / "precommitted_cases.json")

    def feedback(self, prior, result=None):
        saved_result, manifest = self.inputs(prior)
        return runner.build_refinement_feedback(prior, saved_result if result is None else result, manifest)

    def hashes(self):
        return {path.relative_to(self.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.root.rglob("*") if path.is_file()}

    def test_only_predeclared_debug_is_supplied_and_artifacts_are_unchanged(self):
        prior = self.copy_prior()
        result, manifest = self.inputs(prior)
        before_inputs = deepcopy((result, manifest))
        before_files = self.hashes()
        feedback = runner.build_refinement_feedback(prior, result, manifest)
        self.assertEqual(len(feedback), 1)
        item = feedback[0]
        self.assertEqual(item["case_id"], DEBUG_ID)
        self.assertEqual(item["phase"], "debug")
        self.assertEqual(item["observed_program"], read_json(prior / "provider_proposal.json")["program"])
        self.assertTrue(item["trigger"]["bound_actual_scope_checked"])
        self.assertTrue(item["recovery"]["bound_actual_scope_checked"])
        self.assertFalse(item["recovery"]["deadline_exceeded"])
        self.assertEqual(item["recovery"]["chosen"],
                         read_json(prior / "cases" / DEBUG_ID / "recovery/physics/result.json")["chosen"])
        self.assertEqual((result, manifest), before_inputs)
        self.assertEqual(self.hashes(), before_files)
        self.assertFalse(any("admission" in item["case_id"] for item in feedback))

    def test_no_action_history_survives_development_rejection_without_new_cases(self):
        parent = self.copy_prior("parent", "live_01")
        child = self.copy_prior("child", "live_02")
        write_json(child / "generation_packet.json", {"validation_feedback": [{
            "case_id": DEBUG_ID, "phase": "debug", "source_run": str(parent),
            "outcome": "NOT_APPLICABLE", "recovery": {"status": "PASS", "invented": True},
            "interpreter_trace": {"response": {"invented": True}},
            "observed_program": {"invented": True},
        }]})
        before = self.hashes()
        feedback = self.feedback(child, {"status": "DEVELOPMENT_REJECTED", "cases": []})
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["source_run"], str(parent))
        self.assertIsNone(feedback[0]["interpreter_trace"]["response"])
        self.assertEqual(feedback[0]["observed_program"],
                         read_json(parent / "provider_proposal.json")["program"])
        self.assertFalse(feedback[0]["trigger"]["path_success"])
        self.assertNotIn("recovery", feedback[0])
        self.assertEqual(self.hashes(), before)

    def test_inherited_admission_and_unbound_projection_are_not_supplied(self):
        parent = self.copy_prior("parent")
        child = self.copy_prior("child")
        write_json(child / "generation_packet.json", {"validation_feedback": [
            {"case_id": "beam_admission_xn", "phase": "debug", "source_run": str(parent)},
            {"case_id": DEBUG_ID, "phase": "admission", "source_run": str(parent)},
            {"case_id": DEBUG_ID, "phase": "debug", "trigger": {"status": "PASS"}},
        ]})
        feedback = self.feedback(child)
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["source_run"], str(child))
        self.assertEqual(feedback[0]["trigger"]["status"], "UNKNOWN")

    def test_inherited_case_must_be_debug_in_its_own_source_manifest(self):
        parent = self.copy_prior("parent")
        child = self.copy_prior("child")
        manifest = read_json(parent / "precommitted_cases.json")
        next(case for case in manifest["cases"] if case["case_id"] == DEBUG_ID)["phase"] = "admission"
        write_json(parent / "precommitted_cases.json", manifest)
        write_json(child / "generation_packet.json", {"validation_feedback": [
            {"case_id": DEBUG_ID, "phase": "debug", "source_run": str(parent)}]})
        with self.assertRaisesRegex(ValueError, "feedback_must_be_predeclared_debug"):
            self.feedback(child, {"status": "DEVELOPMENT_REJECTED", "cases": []})

    def test_deadline_is_derived_from_worker_allowance_not_saved_projection(self):
        prior = self.copy_prior()
        path = prior / "cases" / DEBUG_ID / "recovery/physics/result.json"
        raw = read_json(path)
        raw.update(elapsed_seconds=31, deadline_exceeded=False)
        write_json(path, raw)
        feedback = self.feedback(prior)
        self.assertTrue(feedback[0]["recovery"]["deadline_exceeded"])

    def test_nonfinite_negative_boolean_or_missing_elapsed_is_rejected(self):
        prior = self.copy_prior()
        for stage in ("trigger", "recovery"):
            path = prior / "cases" / DEBUG_ID / stage / "physics/result.json"
            original = read_json(path)
            for value in (float("nan"), float("inf"), -1, True, None, "1"):
                with self.subTest(stage=stage, elapsed=value):
                    raw = deepcopy(original)
                    raw["elapsed_seconds"] = value
                    write_json(path, raw)
                    with self.assertRaisesRegex(ValueError, "invalid_development_elapsed_or_budget"):
                        self.feedback(prior)
            write_json(path, original)

    def test_nonfinite_or_unbounded_allowance_is_rejected(self):
        prior = self.copy_prior()
        path = prior / "cases" / DEBUG_ID / "recovery/physics/request.json"
        original = read_json(path)
        for value in (float("nan"), float("inf"), -1, 0, 46, True, None):
            with self.subTest(allowance=value):
                worker_request = deepcopy(original)
                worker_request["context"]["remaining_wall_seconds"] = value
                write_json(path, worker_request)
                with self.assertRaisesRegex(ValueError, "invalid_development_elapsed_or_budget"):
                    self.feedback(prior)

    def test_stale_recovery_raw_cannot_be_relabelled_as_current_feedback(self):
        prior = self.copy_prior()
        path = prior / "cases" / DEBUG_ID / "recovery/physics/result.json"
        original = read_json(path)
        for mutate in (
            lambda raw: raw["chosen"].update(direction_id="zp"),
            lambda raw: raw["scope"]["task_context"]["task"].update(assembly_id="stale_assembly"),
            lambda raw: raw.update(backend="fixture_not_actual_physics"),
        ):
            with self.subTest(mutation=mutate):
                raw = deepcopy(original)
                mutate(raw)
                write_json(path, raw)
                with self.assertRaisesRegex(ValueError, "fresh_actual_trigger_scope_mismatch"):
                    self.feedback(prior)

    def test_empty_or_unbound_history_does_not_claim_feedback_available(self):
        prior = self.copy_prior()
        with self.assertRaisesRegex(ValueError, "refinement_requires_bound_debug_observations"):
            self.feedback(prior, {"status": "DEVELOPMENT_REJECTED", "cases": []})

    def test_fixture_proposal_cannot_be_relabelled_as_live_development(self):
        prior = self.copy_prior()
        path = prior / "provider_proposal.json"
        proposal = read_json(path)
        proposal["origin"] = "fixture_authored"
        write_json(path, proposal)
        with self.assertRaisesRegex(ValueError, "development_feedback_requires_actual_proposal"):
            self.feedback(prior)


if __name__ == "__main__":
    unittest.main()
