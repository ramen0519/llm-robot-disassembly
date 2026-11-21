"""Actual StateGraph/SQLite tests with local callbacks; no API or physics calls."""
from copy import deepcopy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.langgraph_workflow import PHASES, Workflow, _checkpoint_path


INITIAL = {"run_id": "fixture-run", "config": {"seed": 0, "ranker": "deterministic"}}


def callbacks(calls, overrides=None):
    def callback(name):
        def run(state):
            calls.append(name)
            return {"artifacts": {name: "fixture-" + name}, "metrics": {name: 1}}
        return run
    result = {name: callback(name) for name in PHASES}
    result.update(overrides or {})
    return result


class LangGraphWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=str(ROOT))
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "checkpoints.sqlite"
        self.calls, self.events = [], []

    def workflow(self, overrides=None):
        return Workflow(self.path, callbacks(self.calls, overrides), self.events.append)

    def test_real_graph_order_end_checkpoint_and_metadata_events(self):
        initial = deepcopy(INITIAL)
        with self.workflow() as flow:
            result = flow.invoke(initial, "t1")
            saved = flow.graph.get_state({"configurable": {"thread_id": "t1"}})
            self.assertEqual(saved.next, ())
            self.assertEqual(saved.values, result)
        self.assertEqual(self.calls, list(PHASES))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["phase"], "END")
        self.assertEqual(set(result["artifacts"]), set(PHASES))
        self.assertEqual(set(result["metrics"]), set(PHASES))
        self.assertEqual(initial, INITIAL)
        self.assertEqual([(e["kind"], e["phase"]) for e in self.events],
                         [(kind, phase) for phase in PHASES for kind in ("phase_before", "phase_after")])
        self.assertEqual(len({e["event_id"] for e in self.events}), 10)
        self.assertTrue(all(e["run_id"] == INITIAL["run_id"] and e["thread_id"] == "t1" for e in self.events))
        self.assertEqual(self.path.read_bytes()[:16], b"SQLite format 3\x00")
        with closing(sqlite3.connect(self.path)) as db:
            self.assertGreater(db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)

    def test_repeated_and_reopened_completed_thread_never_runs_callbacks_again(self):
        with self.workflow() as flow:
            first = flow.invoke(INITIAL, "t1")
            again = flow.invoke(INITIAL, "t1")
            self.assertEqual(first, again)
        with self.workflow() as reopened:
            self.assertEqual(reopened.invoke(INITIAL, "t1"), first)
        self.assertEqual(self.calls, list(PHASES))
        self.assertEqual(sum(e["kind"] == "checkpoint_reuse" for e in self.events), 2)
        self.assertEqual(len({e["event_id"] for e in self.events}), len(self.events))

    def test_same_thread_rejects_changed_identity_before_callbacks(self):
        with self.workflow() as flow:
            flow.invoke(INITIAL, "t1")
            for changed in ({**INITIAL, "run_id": "another"},
                            {**INITIAL, "config": {"seed": 1}},
                            {**INITIAL, "config": {"seed": False, "ranker": "deterministic"}}):
                with self.assertRaisesRegex(ValueError, "thread_identity_mismatch"):
                    flow.invoke(changed, "t1")
        self.assertEqual(self.calls, list(PHASES))

    def test_phase_exception_routes_to_summary_and_preserves_only_type(self):
        secret = "PRIVATE_EXCEPTION_MARKER_DO_NOT_RECORD"
        def fail(state):
            self.calls.append("execute")
            raise ValueError(secret)
        def summary(state):
            self.calls.append("summarize")
            self.assertEqual(state["status"], "ERROR")
            return {"status": "PASS", "error_type": None, "artifacts": {"summary": "available"}}
        with self.workflow({"execute": fail, "summarize": summary}) as flow:
            result = flow.invoke(INITIAL, "t1")
            self.assertEqual(flow.invoke(INITIAL, "t1"), result)
        self.assertEqual(self.calls, ["validate", "prepare", "execute", "summarize"])
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertEqual(result["phase"], "END")
        self.assertNotIn(secret, json.dumps(self.events))
        self.assertNotIn(secret.encode(), self.path.read_bytes())

    def test_fallback_callback_continues_all_phases_and_unknown_is_preserved(self):
        def fallback(state):
            self.calls.append("prepare")
            return {"status": "UNKNOWN", "metrics": {"fallback_count": 1},
                    "artifacts": {"ranker": "deterministic"}}
        with self.workflow({"prepare": fallback}) as flow:
            result = flow.invoke(INITIAL, "t1")
        self.assertEqual(self.calls, list(PHASES))
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["metrics"]["fallback_count"], 1)
        self.assertEqual(result["phase"], "END")

    def test_summary_exception_still_ends_without_error_text(self):
        def fail(state):
            self.calls.append("summarize")
            raise OSError("PRIVATE_DETAIL")
        with self.workflow({"summarize": fail}) as flow:
            result = flow.invoke(INITIAL, "t1")
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["phase"], "END")
        self.assertEqual(result["error_type"], "OSError")
        self.assertNotIn("PRIVATE_DETAIL", json.dumps(result) + json.dumps(self.events))

    def test_callback_receives_private_copy_and_cannot_change_identity(self):
        def mutate_copy(state):
            self.calls.append("validate")
            state["config"]["seed"] = 99
            state["artifacts"]["hidden_mutation"] = True
            return {}
        with self.workflow({"validate": mutate_copy}) as flow:
            result = flow.invoke(INITIAL, "t1")
        self.assertEqual(result["config"], INITIAL["config"])
        self.assertNotIn("hidden_mutation", result["artifacts"])
        with self.workflow({"validate": lambda s: {"config": {"seed": 99}}}) as flow:
            result = flow.invoke({**INITIAL, "run_id": "fixture-2"}, "t2")
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertEqual(result["config"], INITIAL["config"])

    def test_partial_checkpoint_is_rejected_without_worker_resume(self):
        state = {**deepcopy(INITIAL), "status": "RUNNING", "artifacts": {},
                 "metrics": {}, "error_type": None, "phase": "START"}
        with self.workflow() as flow:
            # Actual LangGraph interruption before the execute callback.
            flow.graph.invoke(state, {"configurable": {"thread_id": "partial"}},
                              interrupt_before=["execute"], durability="sync")
            self.assertEqual(self.calls, ["validate", "prepare"])
            with self.assertRaisesRegex(RuntimeError, "unsupported_partial_resume"):
                flow.invoke(INITIAL, "partial")
        with self.workflow() as reopened:
            with self.assertRaisesRegex(RuntimeError, "unsupported_partial_resume"):
                reopened.invoke(INITIAL, "partial")
        self.assertEqual(self.calls, ["validate", "prepare"])

    def test_claim_blocks_retry_even_if_crash_precedes_first_checkpoint(self):
        with self.workflow() as flow:
            with patch.object(flow.graph, "invoke", side_effect=KeyboardInterrupt("PRIVATE_INTERRUPT")):
                with self.assertRaisesRegex(RuntimeError, "unsupported_partial_resume"):
                    flow.invoke(INITIAL, "t1")
        with self.workflow() as reopened:
            with self.assertRaisesRegex(RuntimeError, "unsupported_partial_resume"):
                reopened.invoke(INITIAL, "t1")
        self.assertEqual(self.calls, [])
        self.assertNotIn("PRIVATE_INTERRUPT", json.dumps(self.events))

    def test_two_instances_cannot_execute_one_thread_concurrently(self):
        entered, release = threading.Event(), threading.Event()
        failures = []
        def execute(state):
            self.calls.append("execute")
            entered.set()
            if not release.wait(5):
                raise TimeoutError("fixture did not release")
            return {}
        with self.workflow({"execute": execute}) as first, self.workflow() as second:
            def run():
                try:
                    first.invoke(INITIAL, "t1")
                except Exception as error:
                    failures.append(type(error).__name__)
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(RuntimeError, "unsupported_partial_resume"):
                    second.invoke(INITIAL, "t1")
            finally:
                release.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(self.calls.count("execute"), 1)

    def test_bad_update_is_terminal_error_and_close_is_idempotent(self):
        flow = self.workflow({"prepare": lambda state: None})
        result = flow.invoke(INITIAL, "t1")
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["error_type"], "TypeError")
        self.assertNotIn("execute", self.calls)
        flow.close()
        flow.close()
        with self.assertRaisesRegex(RuntimeError, "workflow_closed"):
            flow.invoke(INITIAL, "t1")

    def test_event_sink_failure_skips_unlogged_worker_and_reaches_end(self):
        def sink(event):
            self.events.append(event)
            if event["kind"] == "phase_before" and event["phase"] == "execute":
                raise OSError("PRIVATE_SINK_DETAIL")
        with Workflow(self.path, callbacks(self.calls), sink) as flow:
            result = flow.invoke(INITIAL, "t1")
        self.assertEqual(self.calls, ["validate", "prepare", "summarize"])
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["phase"], "END")
        self.assertEqual(result["error_type"], "_EventSinkError")
        self.assertNotIn("PRIVATE_SINK_DETAIL", json.dumps(self.events))

    def test_worker_interruption_is_sanitized_and_cannot_be_reexecuted(self):
        def interrupted(state):
            self.calls.append("execute")
            raise KeyboardInterrupt("PRIVATE_WORKER_INTERRUPT")
        with self.workflow({"execute": interrupted}) as flow:
            with self.assertRaisesRegex(RuntimeError, "unsupported_partial_resume"):
                flow.invoke(INITIAL, "t1")
        with self.workflow() as reopened:
            with self.assertRaisesRegex(RuntimeError, "unsupported_partial_resume"):
                reopened.invoke(INITIAL, "t1")
        self.assertEqual(self.calls, ["validate", "prepare", "execute"])
        self.assertNotIn("PRIVATE_WORKER_INTERRUPT", json.dumps(self.events))
        self.assertNotIn(b"PRIVATE_WORKER_INTERRUPT", self.path.read_bytes())

    def test_path_traversal_rejected_before_filesystem_access(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("must not access")) as access:
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                _checkpoint_path("/mnt/c/../e/checkpoints.sqlite")
            access.assert_not_called()


if __name__ == "__main__":
    unittest.main()
