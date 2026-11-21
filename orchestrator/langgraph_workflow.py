"""Whole-run orchestration with real LangGraph and local SQLite checkpoints.

This graph owns orchestration metadata only; callbacks own external work and
ASAP remains the sole search-state owner. Checkpoints do not resume its solver.
No LLM client or network request is implemented here. Callers must supply
JSON-compatible configuration/artifacts without credentials or secret values.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import stat
import threading
from typing import Callable, TypedDict
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context


PHASES = ("validate", "prepare", "execute", "collect", "summarize")
STATUSES = {"PENDING", "RUNNING", "COMPLETE", "PASS", "UNKNOWN", "ERROR", "NOT_RUN"}


class WorkflowState(TypedDict):
    run_id: str
    status: str
    config: dict
    artifacts: dict
    metrics: dict
    error_type: str | None
    phase: str


class _EventSinkError(RuntimeError):
    pass


class _WorkflowInterrupted(RuntimeError):
    """Only a sanitized type name, never the original exception text."""


def _json_copy(value):
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise TypeError("workflow state must contain finite JSON values")
    check(value)
    return deepcopy(value)


def _config_hash(config):
    encoded = json.dumps(_json_copy(config), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _type_name(error):
    name = type(error).__name__
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,127}", name) else "Exception"


def _checkpoint_path(value):
    raw = str(value).replace("\\", "/")
    if ".." in raw.split("/"):
        raise ValueError("parent traversal is not allowed")
    if raw.lower().startswith("e:") or raw.lower() == "/mnt/e" or raw.lower().startswith("/mnt/e/"):
        raise ValueError("excluded checkpoint location")
    if raw == ":memory:" or raw.lower().startswith("file:"):
        raise ValueError("a persistent local checkpoint file is required")
    path = Path(value).absolute()
    for target in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"), Path(str(path) + "-journal")):
        for entry in reversed((target,) + tuple(target.parents)):
            try:
                info = entry.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("checkpoint links/reparse points are not allowed")
    if not path.parent.is_dir():
        raise ValueError("checkpoint parent must already exist")
    return path


class Workflow:
    """Execute once per thread identity; reuse END and reject partial resume.

    Each callback accepts a private state copy and returns a partial update.
    Artifacts/metrics updates are shallow-merged. Identity is immutable; ERROR
    is sticky. COMPLETE means orchestration finished, not that a plan succeeded.
    The event sink receives metadata and exception types, never exception text.
    """
    def __init__(self, checkpoint_path, callbacks: dict[str, Callable], event_sink: Callable):
        if (not isinstance(callbacks, dict) or set(callbacks) != set(PHASES)
                or not all(callable(callbacks[name]) for name in PHASES)
                or not callable(event_sink)):
            raise TypeError("five phase callbacks and a callable event sink are required")
        self.checkpoint_path = _checkpoint_path(checkpoint_path)
        self.callbacks = dict(callbacks)
        self.event_sink = event_sink
        self._lock = threading.RLock()
        self._closed = False
        self._invoking = False
        self._active_thread = None
        self._ledger = None
        self._saver_context = SqliteSaver.from_conn_string(str(self.checkpoint_path))
        self.checkpointer = self._saver_context.__enter__()
        try:
            self.checkpointer.setup()
            self._ledger = sqlite3.connect(str(self.checkpoint_path), timeout=5.0,
                                           isolation_level=None, check_same_thread=False)
            self._ledger.execute("""CREATE TABLE IF NOT EXISTS workflow_invocations (
                thread_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                config_sha256 TEXT NOT NULL, state TEXT NOT NULL
                CHECK(state IN ('ACTIVE', 'COMPLETED')))""")
            builder = StateGraph(WorkflowState)
            for name in PHASES:
                builder.add_node(name, self._node(name))
            builder.add_edge(START, "validate")
            for name, following in zip(PHASES[:-1], PHASES[1:]):
                builder.add_conditional_edges(name, self._route(following),
                                              {"next": following, "error": "summarize"})
            builder.add_edge("summarize", END)
            self.graph = builder.compile(checkpointer=self.checkpointer)
        except BaseException:
            if self._ledger is not None:
                self._ledger.close()
            self._saver_context.__exit__(None, None, None)
            self._closed = True
            raise

    @staticmethod
    def _route(following):
        def route(state):
            return "error" if state["status"] == "ERROR" else "next"
        return route

    def _emit(self, kind, state, phase=None, error_type=None):
        event = {"event_id": str(uuid4()), "kind": kind,
                 "utc": datetime.now(timezone.utc).isoformat(),
                 "run_id": state["run_id"], "thread_id": self._active_thread,
                 "phase": phase or state["phase"], "status": state["status"],
                 "error_type": error_type}
        try:
            self.event_sink(event)
        except BaseException:
            raise _EventSinkError("event_sink_failed") from None

    @staticmethod
    def _updated(state, update):
        if type(update) is not dict or not set(update).issubset(WorkflowState.__annotations__):
            raise TypeError("callback must return a workflow state update")
        update = _json_copy(update)
        if "run_id" in update and update["run_id"] != state["run_id"]:
            raise ValueError("callback cannot change run identity")
        if "config" in update and _config_hash(update["config"]) != _config_hash(state["config"]):
            raise ValueError("callback cannot change run configuration")
        if "status" in update and update["status"] not in STATUSES:
            raise ValueError("unsupported workflow status")
        if update.get("error_type") is not None and (
                type(update["error_type"]) is not str
                or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,127}", update["error_type"])):
            raise ValueError("error_type must be a type name, not exception text")
        result = _json_copy(state)
        for field in ("artifacts", "metrics"):
            if field in update:
                if type(update[field]) is not dict:
                    raise TypeError("artifact and metric updates must be objects")
                update[field] = {**result[field], **update[field]}
        result.update(update)
        if state["status"] == "ERROR":
            result["status"], result["error_type"] = "ERROR", state["error_type"]
        elif result["status"] == "ERROR" and result["error_type"] is None:
            result["error_type"] = "CallbackReportedError"
        return result

    def _node(self, phase):
        def node(state):
            result = _json_copy(state)
            result["phase"] = phase
            failed_type = None
            try:
                self._emit("phase_before", result, phase)
                update = self.callbacks[phase](_json_copy(result))
                result = self._updated(result, update)
            except GraphInterrupt:
                # Dynamic solver continuation is deliberately unsupported. Do not
                # persist an interrupt payload that could contain private data.
                raise _WorkflowInterrupted("GraphInterrupt") from None
            except Exception as error:
                failed_type = _type_name(error)
                result.update(status="ERROR", error_type=state.get("error_type") or failed_type)
            except BaseException as error:
                raise _WorkflowInterrupted(_type_name(error)) from None
            result["phase"] = "END" if phase == "summarize" else phase
            if phase == "summarize" and result["status"] in ("PENDING", "RUNNING"):
                result["status"] = "COMPLETE"
            try:
                self._emit("phase_after", result, phase, failed_type or result["error_type"])
            except Exception as error:
                result.update(status="ERROR", error_type=result.get("error_type") or _type_name(error))
            return result
        return node

    @staticmethod
    def _identity(initial, thread_id):
        if (type(initial) is not dict or type(initial.get("run_id")) is not str
                or not initial["run_id"] or type(initial.get("config")) is not dict
                or type(thread_id) is not str or not thread_id):
            raise ValueError("run_id, config object, and thread_id are required")
        return initial["run_id"], _config_hash(initial["config"])

    def _claim(self, thread_id, run_id, digest, state="ACTIVE"):
        try:
            self._ledger.execute("INSERT INTO workflow_invocations VALUES (?, ?, ?, ?)",
                                 (thread_id, run_id, digest, state))
        except sqlite3.IntegrityError:
            identity = self._ledger.execute("SELECT run_id, config_sha256 FROM workflow_invocations WHERE thread_id=?",
                                            (thread_id,)).fetchone()
            if identity != (run_id, digest):
                raise ValueError("thread_identity_mismatch") from None
            raise RuntimeError("unsupported_partial_resume") from None

    def invoke(self, initial: dict, thread_id: str) -> dict:
        with self._lock, tracing_context(enabled=False):
            if self._closed:
                raise RuntimeError("workflow_closed")
            if self._invoking:
                raise RuntimeError("workflow_busy")
            run_id, digest = self._identity(initial, thread_id)
            config = {"configurable": {"thread_id": thread_id}, "callbacks": []}
            identity = self._ledger.execute("SELECT run_id, config_sha256 FROM workflow_invocations WHERE thread_id=?",
                                            (thread_id,)).fetchone()
            if identity is not None and identity != (run_id, digest):
                raise ValueError("thread_identity_mismatch")
            saved = self.graph.get_state(config)
            if saved.created_at is not None:
                values = saved.values
                if values.get("run_id") is not None and (
                        values["run_id"] != run_id or _config_hash(values.get("config")) != digest):
                    raise ValueError("thread_identity_mismatch")
                if (not saved.next and values.get("phase") == "END"
                        and values.get("run_id") == run_id and type(values.get("config")) is dict):
                    if identity is None:
                        self._claim(thread_id, run_id, digest, "COMPLETED")
                    self._ledger.execute("UPDATE workflow_invocations SET state='COMPLETED' WHERE thread_id=?", (thread_id,))
                    self._active_thread = thread_id
                    try:
                        self._emit("checkpoint_reuse", values)
                        return _json_copy(values)
                    finally:
                        self._active_thread = None
                raise RuntimeError("unsupported_partial_resume")
            if identity is not None:
                # A crash before the first checkpoint can still have launched work.
                raise RuntimeError("unsupported_partial_resume")
            if not set(initial).issubset(WorkflowState.__annotations__):
                raise ValueError("unknown initial state fields")
            state = {"run_id": run_id, "config": _json_copy(initial["config"]),
                     "artifacts": _json_copy(initial.get("artifacts", {})),
                     "metrics": _json_copy(initial.get("metrics", {})),
                     "status": "RUNNING", "error_type": None, "phase": "START"}
            if type(state["artifacts"]) is not dict or type(state["metrics"]) is not dict:
                raise TypeError("initial artifacts and metrics must be objects")
            self._claim(thread_id, run_id, digest)
            self._invoking, self._active_thread = True, thread_id
            try:
                self.graph.invoke(state, config, durability="sync")
                completed = self.graph.get_state(config)
                if completed.next or completed.values.get("phase") != "END":
                    raise _WorkflowInterrupted("IncompleteCheckpoint")
                self._ledger.execute("UPDATE workflow_invocations SET state='COMPLETED' WHERE thread_id=?", (thread_id,))
                return _json_copy(completed.values)
            except BaseException as error:
                try:
                    self._emit("workflow_interrupted", state, error_type=_type_name(error))
                except _EventSinkError:
                    pass
                raise RuntimeError("unsupported_partial_resume") from None
            finally:
                self._invoking, self._active_thread = False, None

    def close(self):
        with self._lock:
            if self._invoking:
                raise RuntimeError("workflow_busy")
            if not self._closed:
                try:
                    self._ledger.close()
                finally:
                    self._saver_context.__exit__(None, None, None)
                    self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("workflow_closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
