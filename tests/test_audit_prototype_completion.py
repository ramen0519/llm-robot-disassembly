"""Audit checks fail closed on incomplete or invented evidence; no API/physics."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).parents[1] / "tools/audit_prototype_completion.py"
SPEC = importlib.util.spec_from_file_location("audit_prototype_test", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class PrototypeAuditTests(unittest.TestCase):
    def retry_fixture(self, declared="2026-10-18T00:00:00+00:00"):
        run = audit.ROOT / "runs/prototype_completion_20261018"
        task = {"input_sha256": {"0.obj": "a" * 64}, "solver_settings": {"seed": 0},
                "base_part": None, "goal": {"type": "native_full_disassembly"}}
        config = {"frozen_memory_sha256": "b" * 64, "arguments": {"seed": 0, "mode": "fixture"}}
        manifest_path = run / "retry_test_manifest.json"
        files = {
            str(manifest_path): {"source_label": "transfer_fixture", "target_label": audit.EXT4_LABEL,
                "reason": "I/O-only relocation", "declared_before_execution": True, "created_utc": declared},
            str(run / "transfer_fixture/result.json"): {"process_exit": 124, "verifier_calls": 0,
                "verifier_results": 0, "plan_success": None},
            str(run / "transfer_fixture/execute/command.json"): {"exit_code": 124},
            str(run / audit.EXT4_LABEL / "execute/command.json"): {"started_at": "2026-10-18T01:00:00+00:00"},
        }
        for label in ("transfer_fixture", audit.EXT4_LABEL):
            files[str(run / label / "config.json")] = deepcopy(config)
            files[str(run / label / "features.json")] = {"task": deepcopy(task)}
        return run, manifest_path, files

    def test_retry_requires_predeclared_identical_geometry_and_solver(self):
        run, manifest, files = self.retry_fixture()
        with patch.object(audit, "read", side_effect=lambda path, default=None: files.get(str(path), default)), \
             patch.object(audit, "sha", return_value="c" * 64), \
             patch.object(audit, "source_archive_checks", return_value={"archive": True}):
            self.assertEqual(audit.retry_gate_checks(run, manifest)["status"], "PASS")
            files[str(run / audit.EXT4_LABEL / "features.json")]["task"]["input_sha256"]["0.obj"] = "d" * 64
            report = audit.retry_gate_checks(run, manifest)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["checks"]["same_nonempty_mesh_and_JSON_hashes"])

    def test_posthoc_retry_selection_cannot_pass_gate(self):
        run, manifest, files = self.retry_fixture(declared="2026-10-18T02:00:00+00:00")
        with patch.object(audit, "read", side_effect=lambda path, default=None: files.get(str(path), default)), \
             patch.object(audit, "sha", return_value="c" * 64), \
             patch.object(audit, "source_archive_checks", return_value={"archive": True}):
            report = audit.retry_gate_checks(run, manifest)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["checks"]["declared_before_execution"])

    def test_full_permutation_checks_allowed_set_and_canonical_input(self):
        request = {"assembled_parts": ["0", "1", "6"], "candidate_ids": ["0", "1"], "base_part": "6"}
        self.assertTrue(audit.full_permutation(request, ["1", "0"]))
        for order in (["0"], ["0", "0"], ["0", "6"], ["0", "new"], "01"):
            self.assertFalse(audit.full_permutation(request, order))
        self.assertFalse(audit.full_permutation({**request, "candidate_ids": ["1", "0"]}, ["0", "1"]))

    def test_raw_feasible_must_match_status(self):
        start = {"kind": "verifier_start", "run_id": "run", "call_id": "v1", "event_id": "run:e1",
                 "request": {"part_move": "0"}}
        bad = {"kind": "verifier_result", "run_id": "run", "call_id": "v1", "event_id": "run:e2",
               "status": "PASS", "raw": {"feasible": False, "part_move": "0", "action": None, "parts_fix": None}}
        checks = audit.evidence_pairs([start, bad])[2]
        self.assertFalse(checks["raw_status_binding_v1"])
        self.assertTrue(checks["all_verifier_calls_completed"])

    def test_incomplete_and_duplicate_events_are_not_complete_evidence(self):
        start = {"kind": "verifier_start", "run_id": "run", "call_id": "v1", "event_id": "run:e1"}
        checks = audit.evidence_pairs([start, start])[2]
        self.assertFalse(checks["event_ids_unique"])
        self.assertFalse(checks["unique_verifier_start_v1"])
        self.assertFalse(checks["all_verifier_calls_completed"])

    def test_absent_live_run_is_not_run(self):
        path = audit.ROOT / "runs/prototype_completion_20261018/nonexistent_audit_fixture"
        with patch.object(Path, "is_dir", return_value=False):
            result, ledger = audit.audit_case(path, "transfer_live")
        self.assertEqual(result["status"], "NOT_RUN")
        self.assertIsNone(ledger)

    def test_forbidden_paths_rejected_before_lstat(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("filesystem touched")):
            for path in ("E:/private", "e:\\private", "/mnt/e", "/mnt/e/private", "//host/share", "../private"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    audit.safe(path)

    def test_unoffered_reference_citation_is_rejected(self):
        decision = {"request": {"request_id": "r1"}, "response": {"metadata": {"used_hint_ids": ["invented"]}},
                    "transfer": {"tentative_correspondences": [], "source_records_sha256": audit.digest([]),
                                 "allow_hard_pruning": False, "physics_reuse_allowed": False, "constraints": []}}
        checks, chains = audit.transfer_chains([decision], [], {"records": []})
        self.assertFalse(checks["r1/cites_offered_only"])
        self.assertEqual(chains, [])


if __name__ == "__main__":
    unittest.main()
