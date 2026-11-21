"""Offline audit of the recorded bridge error; no credential or HTTP access."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.pose_guard_provenance import verified_proposal
from orchestrator.hierarchy_budget import HierarchyBudget
from tools.run_pose_guard_cycle import BoundedDiscovery, policy, services, linux, cycle_status
from worker.explicit_action_contract import digest

ROOT = Path(__file__).absolute().parents[1]
PRIOR = ROOT / "runs/01256_pose_skill_20261021/live_01"


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="guard_bridge_offline_")
        self.addCleanup(self.tmp.cleanup)
        self.temp = Path(self.tmp.name)
        net = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("HTTP_FORBIDDEN"))
        self.network = net.start()
        self.addCleanup(net.stop); self.addCleanup(self.network.assert_not_called)
        self.service = BoundedDiscovery(policy()["roles"]["scout"], self.temp / "preview.sqlite")
        packet = json.loads((PRIOR / "revision_0/generation_packet.json").read_text(encoding="utf-8"))
        self.proposal = json.loads((PRIOR / "revision_0/provider_proposal.json").read_text(encoding="utf-8"))
        self.env = self.service._envelope(packet)
        self.body = self.service._body(self.env)
        with sqlite3.connect("file:" + (PRIOR / "luna.sqlite").as_posix() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            self.local = dict(db.execute("SELECT * FROM rank_requests WHERE request_id=?",
                                        (packet["request_id"],)).fetchone())
        with sqlite3.connect("file:" + (PRIOR / "global_budget.sqlite").as_posix() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            self.shared = dict(db.execute("SELECT * FROM hierarchy_calls WHERE id=?",
                (digest(dict(role="scout", body=self.body)),)).fetchone())

    def verify(self, proposal=None, shared=None, body=None):
        return verified_proposal(proposal or self.proposal, self.local, shared or self.shared,
                                 self.env, body or self.body)

    def test_completed_actual_record_relabels_without_changing_code_or_original(self):
        original = deepcopy(self.proposal)
        value = self.verify()
        self.assertEqual(value["origin"], "actual_llm")
        self.assertFalse(value["metadata"]["new_api_call"])
        self.assertFalse(value["metadata"]["attempted"])
        self.assertTrue(value["metadata"]["original_llm_live"])
        self.assertEqual(value["program"], original["program"])
        self.assertEqual(value["card"], original["card"])
        self.assertEqual(self.proposal, original)

    def test_fixture_unknown_or_different_response_cannot_be_promoted(self):
        for change in (dict(live=0), dict(status="unknown"), dict(body_sha256="0"*64)):
            with self.assertRaises(ValueError):
                self.verify(shared=dict(self.shared, **change))
        bad = deepcopy(self.shared)
        meta = json.loads(bad["metadata"])
        meta["response_id"] = "other"
        bad["metadata"] = json.dumps(meta)
        with self.assertRaisesRegex(ValueError, "identity_mismatch"):
            self.verify(shared=bad)

    def test_modified_program_or_input_cannot_be_recovered(self):
        modified = deepcopy(self.proposal)
        modified["program"]["source"] += "\n"
        with self.assertRaises(ValueError):
            self.verify(proposal=modified)
        body = deepcopy(self.body)
        body["input"][0]["content"] += " "
        with self.assertRaises(ValueError):
            self.verify(body=body)

    def test_service_propagates_only_wrapper_live_mode_without_calling_it(self):
        for flag in (True, False):
            transport = Mock()
            transport.live = flag
            pool = Mock(); pool.transport.return_value = transport
            root = self.temp / str(flag)
            generator, planner = services(pool, policy(True), root)
            self.assertIs(generator._live_transport, flag)
            self.assertIs(planner._live_transport, flag)
            transport.assert_not_called()

    def test_wsl_source_path_allowed_but_excluded_drive_never_inspected(self):
        self.assertEqual(linux(ROOT / "worker/run_pose_guard_action.py"),
            "/mnt/c/dev/disassembly_codex_ready_v2/worker/run_pose_guard_action.py")
        self.assertTrue(linux(PRIOR / "result.json").startswith("/mnt/c/dev/"))
        with patch.object(Path, "lstat", side_effect=AssertionError("MUST_NOT_INSPECT")):
            for bad in ("E:/not_accessed", "/mnt/e/not_accessed", "C:/dev/../outside"):
                with self.assertRaises(ValueError):
                    linux(bad)

    def test_api_rejection_cannot_be_reported_as_completed_physical_comparison(self):
        on = dict(reused=True)
        self.assertEqual(cycle_status(on, dict(history=[], fallback_reason="api_error",
            provider_status_code=429)), "REUSED_CONTROL_API_BLOCKED")
        self.assertEqual(cycle_status(on, "NOT_RUN"), "REUSED_CONTROL_NOT_RUN")
        self.assertEqual(cycle_status(on, dict(history=[])), "REUSED_CONTROL_NO_PHYSICS")
        self.assertEqual(cycle_status(on, dict(history=[dict(outcome={"status":"UNKNOWN"})])),
                         "REUSED_AND_CONTROL_EXECUTED")


if __name__ == "__main__":
    unittest.main()
