"""Bounded file IPC ranker for a Python 3.8 ASAP worker; no network or keys."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
import uuid

import rank_contract
import rank_hook
from rank_contract import RankRequest, RankResponse


MAX_BYTES = 1024 * 1024
IPC_ROOT = Path("/mnt/c/dev/disassembly_codex_ready_v2/runs")
POLL_SECONDS = 0.05


class RemoteRankingError(RuntimeError):
    """The original rank contract converts this into native-order fallback."""


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def request_filename(request_id):
    if type(request_id) is not str or not request_id:
        raise ValueError("request_id must be a nonempty string")
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest() + ".json"


def checked_path(value, run_only=True, allow_test_root=False):
    """Reject lexical escape/E:/links before opening any target content."""
    raw = str(value)
    normalized = raw.replace("\\", "/")
    lower = normalized.lower()
    if ("\x00" in raw or any(part == ".." for part in normalized.split("/"))
            or lower.startswith("e:") or lower == "/mnt/e" or lower.startswith("/mnt/e/")
            or normalized.startswith("//")):
        raise ValueError("unsafe path is not permitted")
    path = Path(value)
    normalized_path = path.as_posix().lower()
    if (normalized_path.startswith("e:") or normalized_path == "/mnt/e"
            or normalized_path.startswith("/mnt/e/")):
        raise ValueError("E: paths are not permitted")
    if not path.is_absolute():
        raise ValueError("absolute paths are required")
    if run_only:
        allowed = Path(tempfile.gettempdir()) if allow_test_root else IPC_ROOT
        try:
            relative = path.relative_to(allowed)
        except ValueError:
            raise ValueError("IPC paths must remain within the approved run directory")
        if not relative.parts:
            raise ValueError("use a dedicated run subdirectory")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("symlink/reparse paths are not permitted")
    return path


def read_json_bounded(path, allow_test_root=False):
    path = checked_path(path, allow_test_root=allow_test_root)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("IPC input must be a regular file no larger than 1 MiB")
        payload = stream.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("IPC input exceeds 1 MiB")
    return json.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def write_json_new(path, value, allow_test_root=False):
    """Publish one complete file using exclusive hard-link creation, never replace."""
    path = checked_path(path, allow_test_root=allow_test_root)
    payload = canonical_bytes(value)
    if len(payload) > MAX_BYTES:
        raise ValueError("IPC request exceeds 1 MiB")
    temporary = checked_path(path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp"),
                             allow_test_root=allow_test_root)
    created = False
    try:
        descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        checked_path(path, allow_test_root=allow_test_root)
        os.link(str(temporary), str(path), follow_symlinks=False)
    finally:
        if created:
            checked_path(temporary, allow_test_root=allow_test_root)
            temporary.unlink()  # Only the uniquely named temporary file created above.
    return hashlib.sha256(payload).hexdigest()


class RemoteRankClient:
    def __init__(self, ipc_dir, remote_trace_file, features=None, timeout_seconds=30.0,
                 allow_test_root=False):
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be positive and finite")
        if features is None:
            features = {}
        if not isinstance(features, dict):
            raise ValueError("features must be a fixed observation dictionary")
        encoded = canonical_bytes(features)
        if len(encoded) > MAX_BYTES:
            raise ValueError("features exceed 1 MiB")
        self.features = json.loads(encoded.decode("utf-8"))
        self.timeout_seconds = float(timeout_seconds)
        self.allow_test_root = allow_test_root
        self.root = checked_path(ipc_dir, allow_test_root=allow_test_root)
        self.trace_path = checked_path(remote_trace_file, allow_test_root=allow_test_root)
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("requests", "responses"):
            checked_path(self.root / name, allow_test_root=allow_test_root).mkdir(exist_ok=True)
        self.stream = self.trace_path.open("x", encoding="utf-8")
        self.cache = {}
        self.audit_count = 0

    def close(self):
        self.stream.close()

    def _audit(self, request, started, reason, metadata, **extra):
        self.audit_count += 1
        event = dict(schema_version=1, kind="remote_rank_result", run_id=request.run_id,
                     event_id=request.run_id + ":ipc" + str(self.audit_count),
                     request_id=request.request_id, state_id=request.state_id,
                     filename=request_filename(request.request_id),
                     outcome="fallback" if reason else "accepted", fallback_reason=reason,
                     response_metadata=metadata, elapsed_seconds=time.monotonic() - started, **extra)
        self.stream.write(canonical_bytes(event).decode("utf-8") + "\n")
        self.stream.flush()

    def __call__(self, request):
        started = time.monotonic()
        if type(request) is not RankRequest:
            raise RemoteRankingError("invalid_request_type")
        envelope = dict(schema_version=1, kind="rank_request", request=asdict(request),
                        features=self.features, timeout_seconds=self.timeout_seconds)
        payload = canonical_bytes(envelope)
        fingerprint = hashlib.sha256(payload).hexdigest()
        previous = self.cache.get(request.request_id)
        if previous is not None:
            if previous["fingerprint"] != fingerprint:
                self._audit(request, started, "duplicate_request_id_conflict", {}, duplicate_cached=True)
                raise RemoteRankingError("duplicate_request_id_conflict")
            self._audit(request, started, previous["reason"], previous["metadata"], duplicate_cached=True)
            if previous["reason"]:
                raise RemoteRankingError(previous["reason"])
            return previous["response"]
        reason, metadata, response = None, {}, None
        request_created, response_sha256 = False, None
        try:
            if len(payload) > MAX_BYTES:
                raise RemoteRankingError("request_too_large")
            filename = request_filename(request.request_id)
            target = self.root / "requests" / filename
            try:
                write_json_new(target, envelope, self.allow_test_root)
                request_created = True
            except FileExistsError:
                existing, _ = read_json_bounded(target, self.allow_test_root)
                if canonical_bytes(existing) != payload:
                    raise RemoteRankingError("existing_request_id_conflict")
            response_path = self.root / "responses" / filename
            while True:
                try:
                    raw, response_sha256 = read_json_bounded(response_path, self.allow_test_root)
                    break
                except FileNotFoundError:
                    remaining = self.timeout_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise RemoteRankingError("response_timeout")
                    time.sleep(min(POLL_SECONDS, remaining))
            required = {"schema_version", "kind", "request_id", "state_id", "candidate_ids", "fallback_reason", "metadata"}
            if (not isinstance(raw, dict) or set(raw) != required or type(raw["schema_version"]) is not int
                    or raw["schema_version"] != 1 or raw["kind"] != "rank_response"
                    or not isinstance(raw["metadata"], dict)
                    or (raw["fallback_reason"] is not None and (type(raw["fallback_reason"]) is not str or not raw["fallback_reason"]))):
                raise RemoteRankingError("invalid_response_schema")
            metadata = json.loads(canonical_bytes(raw["metadata"]).decode("utf-8"))
            if raw["request_id"] != request.request_id or raw["state_id"] != request.state_id:
                raise RemoteRankingError("stale_request_or_state")
            if (type(raw["candidate_ids"]) is not list
                    or any(type(part) is not str for part in raw["candidate_ids"])):
                raise RemoteRankingError("invalid_candidate_type")
            response = RankResponse(raw["request_id"], raw["state_id"], tuple(raw["candidate_ids"]))
            checked = rank_contract.rank_candidates(request, "received", {"received": lambda _: response})
            if checked.fallback_reason is not None:
                raise RemoteRankingError(checked.fallback_reason)
            if raw["fallback_reason"] is not None:
                raise RemoteRankingError(raw["fallback_reason"])
        except RemoteRankingError as error:
            reason = str(error)
        except Exception as error:
            reason = "ipc_error:" + type(error).__name__
        self.cache[request.request_id] = dict(fingerprint=fingerprint, response=response, reason=reason, metadata=metadata)
        self._audit(request, started, reason, metadata, request_created=request_created,
                    request_sha256=fingerprint, response_sha256=response_sha256, duplicate_cached=False)
        if reason is not None:
            raise RemoteRankingError(reason)
        return response


@contextmanager
def installed_remote_ranker(client):
    """Temporarily route only rank_hook's lookup through the original contract."""
    original = rank_hook.rank_candidates
    def choose(request, name):
        return rank_contract.rank_candidates(request, name, {"remote": client})
    rank_hook.rank_candidates = choose
    try:
        yield
    finally:
        rank_hook.rank_candidates = original
