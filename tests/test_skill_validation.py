"""Synthetic integrity fixtures only; no API, ASAP or actual physics execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_skill_program import request, append_observation, program, choice, rule
from orchestrator.skill_program import program_hash, step_program
from orchestrator.skill_validation import record_validation, check_evidence
from worker.task_memory import canonical, digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_case(root, case_id="debug", phase="debug", condition="a", *, success=True,
                 late=False, boundary=False, seed=0, not_triggered=False, policy=None,
                 stability_trigger=False, trigger_choice=None):
    """Every emitted worker backend is deliberately fixture_not_actual_physics."""
    root = Path(root) / case_id
    policy = policy or (program(rule(when=['last_stability_failed'])) if stability_trigger else program())
    identity = program_hash(policy)
    req = request()
    if boundary:
        req["remaining_parts"] = ["a", "base"]
        req["candidate_ids"] = ["a"]
    req["task_context"]["solver_settings"]["seed"] = seed
    source = "fixture_" + condition
    maps = {"a.obj": digest({"mesh_fixture": condition})}
    sources = {"physics_planner": "f" * 64, "redmax": "e" * 64}
    case = dict(case_id=case_id, phase=phase, source_task=source, program_hash=identity,
                input_hashes=maps, source_hashes=sources,
                trigger_kind="stability_failed" if stability_trigger else "path_failed")
    if not_triggered:
        request_hash = save(root / "program_request.json", req)
        selection = dict(case_id=case_id, source_task=source, program_hash=identity,
                         program_status="NOT_APPLICABLE", program_request_path=str(root / "program_request.json"),
                         program_request_sha256=request_hash)
        save(root / "selection.json", selection)
        return case, dict(selection_path=str(root / "selection.json"), attempt_dir=None,
                          trigger_attempt_dir=None), policy
    append_observation(req, selected=trigger_choice, path=stability_trigger,
                       stability=False if stability_trigger else None)
    selected = step_program(policy, req)["response"]["choice"]
    current_request_hash = save(root / "program_request.json", req)
    context = {k: deepcopy(req[k]) for k in ("remaining_parts", "base_part", "max_grippers", "pose_catalog")}
    context.update(remaining_wall_seconds=10, removed_parts=[], seed=seed)

    def attempt(name, selected_choice, passed, is_late=False, failed_stability=False):
        directory = root / name
        worker_request = dict(context=deepcopy(context), choice=deepcopy(selected_choice))
        request_sha = save(directory / "request.json", worker_request)
        path_passed = passed or failed_stability
        trajectory = dict(raw_success=path_passed, controller={"physics_steps": 8}, path=[[0] * 6])
        trajectory_sha = save(directory / "trajectory.json", trajectory)
        fixed = [context["base_part"]] + selected_choice["support_ids"]
        rest = [p for p in context["remaining_parts"] if p != selected_choice["part_id"]]
        endpoint = dict(kind="path_endpoint_measurement", raw_success=path_passed,
            trajectory={"sha256": trajectory_sha}, scope={"selected_action": deepcopy(selected_choice),
                "physics": {"source_and_binary_hashes": sources},
                "assembled_parts": context["remaining_parts"], "part_move": selected_choice["part_id"],
                "parts_rest": rest, "parts_removed": [], "base_part": context["base_part"],
                "pose": context["pose_catalog"][0]["matrix"], "pose_id": selected_choice["pose_id"],
                "seed": seed, "task_context": None,
                "support_policy": {"selected_support_ids": selected_choice["support_ids"],
                    "actual_parts_fix": fixed, "actual_parts_move": [p for p in rest if p not in fixed]}},
            terminal_q=[0] * 6, measurements={"pairs": []})
        endpoint_sha = save(directory / "endpoint_evidence.json", endpoint)
        result = dict(protocol="explicit_action_physics_v1", backend="fixture_not_actual_physics",
            chosen=deepcopy(selected_choice), status="PASS" if passed else "UNKNOWN", feasible=passed,
            input_hashes=maps, source_hashes=sources, input_files_unchanged=True, source_files_unchanged=True,
            path_check_calls=1, stability_check_calls=1 if failed_stability else (0 if boundary else 1) if passed else 0,
            path_success=path_passed, stability_success=False if failed_stability else True if passed else None,
            raw_upstream={"path_success": path_passed, "stability_success": False if failed_stability else
                (None if boundary else True) if passed else None},
            trajectory_file="trajectory.json", trajectory_sha256=trajectory_sha,
            evidence_file="endpoint_evidence.json", evidence_sha256=endpoint_sha,
            executed={"stability_moving_parts": [] if boundary else ["b"]},
            stability_mode=("all_remaining_rest_explicitly_fixed_no_checker_call" if boundary else
                            "one_exact_selected_support_configuration") if path_passed else None,
            elapsed_seconds=11 if is_late else 1)
        result_sha = save(directory / "result.json", result)
        return directory, request_sha, result_sha

    trigger, _, trigger_sha = attempt("trigger", trigger_choice or choice(), False,
                                     failed_stability=stability_trigger)
    recovery, worker_request_sha, worker_result_sha = attempt("recovery", selected, success, late)
    selection = dict(case_id=case_id, source_task=source, program_hash=identity, program_status="ACTION",
        worker_request_sha256=worker_request_sha, worker_result_sha256=worker_result_sha,
        action_sha256=digest(selected), trigger_result_sha256=trigger_sha,
        program_request_path=str(root / "program_request.json"), program_request_sha256=current_request_hash)
    save(root / "selection.json", selection)
    return case, dict(selection_path=str(root / "selection.json"), attempt_dir=str(recovery),
                      trigger_attempt_dir=str(trigger)), policy


class SkillValidationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="skill_validation_fixture_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def record(self, case=None, artifacts=None, policy=None):
        if case is None:
            case, artifacts, policy = fixture_case(self.root)
        return record_validation(case, artifacts, program_hash(policy), allow_test_root=True,
                                 fixture=True, program=policy)

    def test_natural_trigger_program_and_worker_hashes_bound(self):
        record = self.record()
        self.assertEqual(record["outcome"], "PASS")
        self.assertEqual(record["verification_class"], "fixture_not_actual_physics")
        self.assertTrue(record["program_execution_checked"])
        self.assertEqual(check_evidence(record, True), record)
        self.assertFalse(record["physics_reuse_allowed"])

    def test_physical_pass_with_support_release_is_explicit_plan_only(self):
        record = self.record(*fixture_case(self.root, trigger_choice=choice(support_ids=["b"])))
        self.assertEqual(record["outcome"], "PASS")
        self.assertEqual(record["classification_version"], 4)
        roles = record["execution_roles"]
        self.assertEqual(roles["validation_mode"], "explicit_action_physics")
        self.assertFalse(roles["automatic_recovery"]["allowed"])
        self.assertEqual(roles["automatic_recovery"]["guard"]["released_support_ids"], ["b"])
        self.assertEqual(check_evidence(record, True), record)

    def test_rehashed_execution_role_tampering_is_recomputed_and_rejected(self):
        record = self.record(*fixture_case(self.root, trigger_choice=choice(support_ids=["b"])))
        record["execution_roles"]["automatic_recovery"]["allowed"] = True
        record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "validation_evidence_replay_mismatch"):
            check_evidence(record, True)

    def test_old_v3_record_remains_unmodified_without_invented_role_evidence(self):
        record = self.record()
        record["classification_version"] = 3
        record.pop("execution_roles")
        record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
        before = deepcopy(record)
        self.assertEqual(check_evidence(record, True), before)
        self.assertEqual(record, before)

    def test_actual_stage_shaped_stability_fixture_can_validate_but_is_not_physics(self):
        record = self.record(*fixture_case(self.root, stability_trigger=True))
        self.assertEqual(record['classification_version'], 4)
        self.assertTrue(record['natural_trigger'])
        self.assertEqual(record['trigger_stability_calls'], 1)
        self.assertEqual(record['outcome'], 'PASS')
        self.assertEqual(record['verification_class'], 'fixture_not_actual_physics')
        self.assertEqual(check_evidence(record, True), record)

    def test_error_result_is_not_a_natural_failure_trigger(self):
        case, artifacts, policy = fixture_case(self.root, stability_trigger=True)
        raw_path = Path(artifacts['trigger_attempt_dir']) / 'result.json'
        raw = json.loads(raw_path.read_text())
        raw['status'] = 'ERROR'
        selection_path = Path(artifacts['selection_path'])
        selection = json.loads(selection_path.read_text())
        selection['trigger_result_sha256'] = save(raw_path, raw)
        save(selection_path, selection)
        with self.assertRaisesRegex(ValueError, 'natural_physical_failure'):
            self.record(case, artifacts, policy)

    def test_natural_failure_with_false_program_predicate_is_recorded_and_legacy_replays(self):
        case, artifacts, _ = fixture_case(self.root)
        policy = program(rule(when=['last_stability_failed']))
        identity = program_hash(policy)
        case['program_hash'] = identity
        selection_path = Path(artifacts['selection_path'])
        selected = json.loads(selection_path.read_text())
        selected.update(program_status='NOT_APPLICABLE', program_hash=identity)
        save(selection_path, selected)
        artifacts['attempt_dir'] = None
        record = self.record(case, artifacts, policy)
        self.assertTrue(record['natural_trigger'])
        self.assertFalse(record['program_predicate_triggered'])
        self.assertEqual(record['outcome'], 'NOT_TRIGGERED')
        self.assertEqual(check_evidence(record, True), record)
        # Frozen v2 bytes retain their earlier natural_trigger classification.
        old = deepcopy(record)
        old.update(classification_version=2, natural_trigger=False)
        old.pop('execution_roles')
        old['record_sha256'] = digest({k: v for k, v in old.items() if k != 'record_sha256'})
        self.assertEqual(check_evidence(old, True), old)

    def test_fixture_cannot_enter_actual_validation(self):
        case, artifacts, policy = fixture_case(self.root)
        with self.assertRaisesRegex(ValueError, "backend"):
            record_validation(case, artifacts, program_hash(policy), True, program=policy)

    def test_candidate_program_mismatch(self):
        case, artifacts, policy = fixture_case(self.root)
        with self.assertRaisesRegex(ValueError, "program_hash"):
            record_validation(case, artifacts, "a" * 64, True, True, policy)

    def test_result_tamper_rejected_after_validation(self):
        record = self.record()
        result = Path(record["artifacts"]["attempt_dir"]) / "result.json"
        content = json.loads(result.read_text())
        content["status"] = "UNKNOWN"
        save(result, content)
        with self.assertRaisesRegex(ValueError, "changed"):
            check_evidence(record, True)

    def test_fake_pass_missing_stability_rejected(self):
        case, artifacts, policy = fixture_case(self.root)
        result = Path(artifacts["attempt_dir"]) / "result.json"
        content = json.loads(result.read_text())
        content["stability_success"] = None
        save(result, content)
        with self.assertRaisesRegex(ValueError, "fake_pass"):
            self.record(case, artifacts, policy)

    def test_missing_raw_backend_rejected(self):
        case, artifacts, policy = fixture_case(self.root)
        result = Path(artifacts["attempt_dir"]) / "result.json"
        content = json.loads(result.read_text())
        content.pop("backend")
        save(result, content)
        with self.assertRaisesRegex(ValueError, "backend"):
            self.record(case, artifacts, policy)

    def test_stale_input_and_solver_rejected(self):
        for field in ("input_hashes", "source_hashes"):
            case, artifacts, policy = fixture_case(self.root, case_id=field)
            case[field] = {"changed": "f" * 64}
            with self.assertRaisesRegex(ValueError, "stale_or_mismatched"):
                self.record(case, artifacts, policy)

    def test_late_pass_preserved_but_not_registration_pass(self):
        record = self.record(*fixture_case(self.root, late=True))
        self.assertEqual(record["outcome"], "LATE")

    def test_boundary_is_not_gravity_simulation(self):
        record = self.record(*fixture_case(self.root, boundary=True))
        self.assertEqual(record["stability_calls"], 0)
        self.assertEqual(record["stability_evidence_kind"], "explicit_all_rest_fixed_boundary")

    def test_not_triggered_retained_without_success(self):
        record = self.record(*fixture_case(self.root, not_triggered=True))
        self.assertEqual(record["outcome"], "NOT_TRIGGERED")
        self.assertEqual(record["path_calls"], 0)
        self.assertFalse(record["program_predicate_triggered"])

    def test_true_predicate_with_no_legal_action_is_not_applicable(self):
        policy = program(rule(direction="same_as_last", support="nonempty"))
        case, artifacts, _ = fixture_case(self.root, not_triggered=True, policy=policy)
        selection_path = Path(artifacts["selection_path"])
        selected = json.loads(selection_path.read_text())
        current = append_observation(request())
        request_path = Path(selected["program_request_path"])
        selected["program_request_sha256"] = save(request_path, current)
        save(selection_path, selected)
        record = self.record(case, artifacts, policy)
        self.assertEqual(record["outcome"], "NOT_APPLICABLE")
        self.assertTrue(record["program_predicate_triggered"])
        self.assertEqual(record["not_applicable_reason"], "no_matching_untried_action")
        self.assertEqual(record["path_calls"], 0)
        self.assertEqual(check_evidence(record, True), record)
        # Old files remain readable and keep their historical outcome label.
        old = deepcopy(record)
        old.pop("classification_version")
        old.pop("program_predicate_triggered")
        old.pop("execution_roles")
        old["outcome"] = "NOT_TRIGGERED"
        old["record_sha256"] = digest({k: v for k, v in old.items() if k != "record_sha256"})
        self.assertEqual(check_evidence(old, True), old)

    def test_seed_and_label_changes_do_not_invent_distinct_condition(self):
        first = self.record(*fixture_case(self.root, case_id="one", condition="a", seed=0))
        case, artifacts, policy = fixture_case(self.root, case_id="two", condition="a", seed=9)
        second = self.record(case, artifacts, policy)
        self.assertEqual(first["condition_hash"], second["condition_hash"])

    def test_missing_trajectory_is_not_accepted(self):
        case, artifacts, policy = fixture_case(self.root)
        (Path(artifacts["attempt_dir"]) / "trajectory.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.record(case, artifacts, policy)

    def test_semantic_scope_tamper_rejected_even_with_rehashed_artifacts(self):
        case, artifacts, policy = fixture_case(self.root)
        directory = Path(artifacts["attempt_dir"])
        endpoint = json.loads((directory / "endpoint_evidence.json").read_text())
        endpoint["scope"]["assembled_parts"] = ["a", "base"]
        endpoint_sha = save(directory / "endpoint_evidence.json", endpoint)
        result = json.loads((directory / "result.json").read_text())
        result["evidence_sha256"] = endpoint_sha
        result_sha = save(directory / "result.json", result)
        selection_path = Path(artifacts["selection_path"])
        selection = json.loads(selection_path.read_text())
        selection["worker_result_sha256"] = result_sha
        save(selection_path, selection)
        with self.assertRaisesRegex(ValueError, "physical_context_mismatch"):
            self.record(case, artifacts, policy)

    def test_program_output_cannot_be_relabelled_with_correct_hash(self):
        case, artifacts, policy = fixture_case(self.root)
        selection_path = Path(artifacts["selection_path"])
        selection = json.loads(selection_path.read_text())
        request_path = Path(selection["program_request_path"])
        supplied = json.loads(request_path.read_text())
        supplied["episode_feedback"] = []
        supplied["evidence_ids"] = []
        selection["program_request_sha256"] = save(request_path, supplied)
        save(selection_path, selection)
        with self.assertRaisesRegex(ValueError, "program_replay_action_mismatch"):
            self.record(case, artifacts, policy)

    def test_late_trigger_cannot_be_recovery_evidence(self):
        case, artifacts, policy = fixture_case(self.root)
        directory = Path(artifacts["trigger_attempt_dir"])
        raw = json.loads((directory / "result.json").read_text())
        raw["elapsed_seconds"] = 11
        trigger_sha = save(directory / "result.json", raw)
        selection_path = Path(artifacts["selection_path"])
        selection = json.loads(selection_path.read_text())
        selection["trigger_result_sha256"] = trigger_sha
        save(selection_path, selection)
        with self.assertRaisesRegex(ValueError, "natural_physical_failure"):
            self.record(case, artifacts, policy)

    def test_e_drive_rejected_before_file_access(self):
        case, artifacts, policy = fixture_case(self.root)
        artifacts["selection_path"] = "E:/not_opened/selection.json"
        with self.assertRaisesRegex(ValueError, "exclude_non|exclude non"):
            self.record(case, artifacts, policy)

    def test_cli_refinement_payload_uses_only_debug_cases(self):
        # Copied historical development artifacts plus an injected preview
        # service. This checks CLI wiring, never new API or physics execution.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import run_skill_discovery as cli
        from worker.task_memory import _safe_path
        prior, output = self.root / "prior_fixture", self.root / "preview_fixture"
        ledger = self.root / "fixture_ledger.sqlite"
        manifest = {"cases": [{"case_id": "debug", "phase": "debug"},
                              {"case_id": "heldout", "phase": "admission"}]}
        manifest_path = self.root / "fixture_manifest.json"
        save(manifest_path, manifest)
        save(prior / "precommitted_cases.json", manifest)
        save(prior / "config.json", {"ledger": str(ledger)})
        history = Path(__file__).resolve().parents[1] / "runs/skill_discovery_implementation_20261019/live_02"
        save(prior / "provider_proposal.json",
             json.loads((history / "provider_proposal.json").read_text(encoding="utf-8")))
        save(prior / "generation_packet.json", {"validation_feedback": []})
        save(prior / "result.json", {"cases": [{"case_id": "debug", "outcome": "UNKNOWN"},
                                                {"case_id": "heldout", "outcome": "HOLDOUT_SECRET"}]})
        source_case_dir = history / "cases/beam_debug_yp"
        for relative in ("program_request.json", "program_step.json",
                         "trigger/physics/request.json", "trigger/physics/result.json",
                         "recovery/physics/request.json", "recovery/physics/result.json"):
            artifact = json.loads((source_case_dir / relative).read_text(encoding="utf-8"))
            if relative == "program_step.json":
                artifact["marker"] = "DEBUG_ALLOWED"
            save(prior / "cases/debug" / relative, artifact)
        save(prior / "cases/heldout/program_step.json", {"marker": "HOLDOUT_SECRET"})
        original_checked = cli.checked

        def check(path, runs_only=False):
            path = Path(path)
            if path.is_absolute() and self.root in path.parents:
                return _safe_path(path, True)
            return original_checked(path, runs_only)

        service = Mock()
        service.preview.return_value = {"api_calls": 0, "unit_fixture_only": True}
        with patch.object(cli, "checked", check), patch.object(cli, "source_case", return_value={}), \
                patch.object(cli, "read", side_effect=lambda p: json.loads(_safe_path(p, True).read_text(encoding="utf-8"))), \
                patch.object(cli, "sha", side_effect=lambda p: hashlib.sha256(check(p).read_bytes()).hexdigest()), \
                patch.object(cli, "write", side_effect=lambda p, v: save(_safe_path(p, True), v)), \
                patch.object(cli, "synthesis_packet", return_value={"operation": "synthesize", "evidence": [],
                    "context": {"automatically_extracted_same_condition_pairs": {"pairs": []}}}), \
                patch.object(cli, "SkillDiscoveryService", return_value=service), \
                patch.object(cli, "actual_action", side_effect=AssertionError("physics not allowed")):
            code = cli.main(["--run-dir", str(output), "--manifest", str(manifest_path),
                             "--ledger", str(ledger), "--refine-from", str(prior), "--mode", "preview"])
        self.assertEqual(code, 0)
        packet = json.loads((output / "generation_packet.json").read_text())
        self.assertEqual([c["case_id"] for c in packet["validation_feedback"]], ["debug"])
        self.assertIn("DEBUG_ALLOWED", canonical(packet))
        self.assertNotIn("HOLDOUT_SECRET", canonical(packet))
        self.assertTrue(packet["validation_feedback"][0]["recovery"]["bound_actual_scope_checked"])
        development_ids = {packet['validation_feedback'][0][stage]['evidence_id']
                           for stage in ('trigger', 'recovery')}
        self.assertEqual({e['evidence_id'] for e in packet['evidence']}, development_ids)
        self.assertTrue(all(e['source_phase'] == 'debug' and e['reference_only']
                            and e['failure_cause'] is None for e in packet['evidence']))
        service.preview.assert_called_once()
        service.propose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
