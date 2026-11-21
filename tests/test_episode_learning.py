"""Saved-artifact integrity tests, NOT fresh ASAP physics or live LLM trials.

Each test uses an owned temporary copy of historical actual Beam artifacts.
The memory pointer is rehashed to describe that copy; injected mutations never
touch the original episode or its baseline. No API/WSL process is launched.
"""
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prepare_episode_learning as preparation
from worker.task_memory import digest

HISTORICAL = ROOT / "runs/explicit_action_v2_live_20261018/beam"
TEMPLATE = ROOT / "configs/skill_discovery_cases_v1.json"


class EpisodeLearningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="episode_learning_artifact_checks_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).absolute()
        self.episode, self.output = self.root / "historical_artifact_copy", self.root / "new_learning"
        shutil.copytree(HISTORICAL, self.episode)
        memory = self.read("action_memory_frozen.json")
        for record in memory["records"]:
            record["result"]["evidence_ref"] = str(self.episode / "planner/planner_report.json")
            record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
        memory["content_sha256"] = digest(memory["records"])
        self.write("action_memory_frozen.json", memory)
        self.original_report = preparation._sha(HISTORICAL / "worker_result.json")

    def read(self, relative):
        return json.loads((self.episode / relative).read_text(encoding="utf-8"))

    def write(self, relative, value):
        (self.episode / relative).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def prepare(self):
        return preparation.prepare_episode_learning(self.episode, TEMPLATE, self.output, allow_test_root=True)

    def reject(self, expression=None):
        with self.assertRaisesRegex((ValueError, FileNotFoundError), expression or ".*"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(preparation._sha(HISTORICAL / "worker_result.json"), self.original_report)

    def mutate_report(self, callback):
        for name in ("worker_result.json", "planner/planner_report.json"):
            value = self.read(name)
            callback(value)
            self.write(name, value)

    def test_oldest_failure_selected_with_unchanged_admission_and_frozen_provenance(self):
        with patch("subprocess.Popen", side_effect=AssertionError("no subprocess permitted")):
            result = self.prepare()
        self.assertEqual(result["status"], "PREPARED")
        self.assertEqual(result["selected"]["attempt_index"], 1)
        self.assertEqual(result["selected"]["trigger_kind"], "path_failed")
        self.assertEqual(len(result["considered"]), 8)
        self.assertEqual(sum(row["selected"] for row in result["considered"]), 1)
        self.assertEqual(result["new_api_calls"], 0)
        self.assertEqual(result["new_physical_calls"], 0)
        self.assertEqual(result["evidence_class"], "saved_actual_asap_artifacts_checked_not_new_physics")
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        original = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(manifest["cases"][1:], [c for c in original["cases"] if c["phase"] == "admission"])
        self.assertEqual(manifest["experience_report"], str(self.episode / "planner/planner_report.json"))
        self.assertEqual(manifest["experience_memory"], str(self.episode / "action_memory_frozen.json"))
        self.assertEqual(manifest["experience_task"], "beam_assembly/original")
        self.assertIsNone(result["source_to_target_part_mapping"])
        self.assertFalse(result["hard_pruning"])
        self.assertEqual(result["memory_file_sha256"], preparation._sha(self.episode / "action_memory_frozen.json"))
        self.assertEqual(len(result["admission_inputs"]), 2)
        self.assertNotIn("admission_results", manifest)
        for path, expected in result["original_artifact_sha256"].items():
            self.assertEqual(preparation._sha(Path(path)), expected)

    def test_chronological_selection_skips_late_first_without_best_success_selection(self):
        self.mutate_report(lambda report: report["attempts"][0].update(deadline_exceeded=True))
        result = self.prepare()
        self.assertEqual(result["selected"]["attempt_index"], 2)
        self.assertEqual(result["considered"][0]["reason"], "deadline_exceeded_observation_not_eligible")

    def test_no_eligible_failure_stops_without_old_trace_or_fake_manifest(self):
        def make_late(report):
            for attempt in report["attempts"]:
                if not attempt["accepted_for_transition"]:
                    attempt["deadline_exceeded"] = True
        self.mutate_report(make_late)
        result = self.prepare()
        self.assertEqual(result["status"], "NO_ELIGIBLE_EXPERIENCE")
        self.assertIsNone(result["manifest_path"])
        self.assertIsNone(result["selected"])
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertTrue((self.output / "preparation.json").is_file())

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("unchanged", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_empty_observation_episode_has_explicit_no_experience_stop(self):
        def empty(report):
            report.update(attempts=[], decisions=[], total_n_eval=0, final_pair_calls=0)
        self.mutate_report(empty)
        self.write("audit.json", preparation.audit_report(self.read("worker_result.json")))
        memory = self.read("action_memory_frozen.json")
        memory.update(records=[], record_count=0, content_sha256=digest([]))
        self.write("action_memory_frozen.json", memory)
        result = self.prepare()
        self.assertEqual(result["status"], "NO_ELIGIBLE_EXPERIENCE")
        self.assertEqual(result["considered"], [])
        self.assertEqual(result["memory_record_count"], 0)
        self.assertIsNone(result["manifest_path"])

    def test_worker_planner_mismatch_rejected(self):
        report = self.read("worker_result.json")
        report["attempts"][0]["status"] = "ERROR"
        self.write("worker_result.json", report)
        self.reject("worker_planner")

    def test_report_cannot_claim_mock_is_actual(self):
        report = self.read("worker_result.json")
        report["actual_physics"] = False
        self.write("worker_result.json", report)
        self.reject("actual_asap_full_episode")

    def test_stored_audit_tampering_rejected(self):
        audit = self.read("audit.json")
        audit["verifier_calls"] = 777
        self.write("audit.json", audit)
        self.reject("episode_audit")

    def test_unpreserved_host_rejected(self):
        self.write("host_preservation.json", {"source_unchanged": False, "memory_unchanged": True})
        self.reject("not_preserved")

    def test_changed_archived_controller_rejected_without_comparing_current_source(self):
        archived = self.episode / "implementation/worker/explicit_action_physics.py"
        archived.write_bytes(archived.read_bytes() + b"\n# injected test mutation\n")
        self.reject("archived_episode_source_hash")

    def test_memory_content_hash_tampering_rejected(self):
        snapshot = self.read("action_memory_frozen.json")
        snapshot["content_sha256"] = "0" * 64
        self.write("action_memory_frozen.json", snapshot)
        self.reject()

    def test_rehashed_memory_raw_tampering_still_rejected(self):
        snapshot = self.read("action_memory_frozen.json")
        record = snapshot["records"][0]
        record["result"]["raw"]["reason"] = "invented failure reason"
        record["record_sha256"] = digest({k: v for k, v in record.items() if k != "record_sha256"})
        snapshot["content_sha256"] = digest(snapshot["records"])
        self.write("action_memory_frozen.json", snapshot)
        self.reject("episode_memory_raw_binding")

    def test_raw_file_and_report_mismatch_rejected(self):
        raw = self.read("planner/attempt_000001/result.json")
        raw["path_success"] = True
        self.write("planner/attempt_000001/result.json", raw)
        self.reject("episode_raw_file_binding")

    def test_physical_request_changed_scope_rejected(self):
        physical = self.read("planner/attempt_000001/request.json")
        physical["context"]["state_id"] = "different-state"
        self.write("planner/attempt_000001/request.json", physical)
        self.reject("episode_physical_request_binding")

    def test_trajectory_hash_tampering_rejected(self):
        trajectory = self.episode / "planner/attempt_000001/trajectory.json"
        trajectory.write_bytes(trajectory.read_bytes() + b"\n")
        self.reject("path_evidence_hash_mismatch")

    def test_input_hash_preservation_mismatch_rejected(self):
        preservation = self.read("preservation.json")
        preservation["after"]["0.obj"] = "0" * 64
        self.write("preservation.json", preservation)
        self.reject("input_hashes_changed")

    def test_non_c_and_traversal_rejected_before_filesystem(self):
        for excluded in ("E:/must-not-read/episode", "/mnt/e/must-not-read", "runs/../unsafe"):
            with self.subTest(path=excluded), patch.object(preparation, "_load", side_effect=AssertionError("read forbidden")):
                with self.assertRaises(ValueError):
                    preparation.prepare_episode_learning(excluded, TEMPLATE, self.output)

    def test_admission_source_frozen_but_outcomes_never_added_to_synthesis_manifest(self):
        result = self.prepare()
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["experience_report"], result["experience_report"])
        for row in result["admission_inputs"]:
            self.assertEqual(row["report_sha256"], preparation._sha(Path(row["report_path"])))
        self.assertNotIn("admission_inputs", manifest)
        self.assertTrue(result["admission_cases_unchanged"])

    def test_cli_has_no_test_root_escape(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as raised:
            preparation.main(["--episode-dir", str(self.episode), "--admission-manifest", str(TEMPLATE),
                "--output-dir", str(self.output), "--allow-test-root"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
