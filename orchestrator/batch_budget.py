"""Immutable policy binding and read-only accounting for a shared rank ledger.

All run IDs in this SQLite file share one policy and reservation pool. Monetary
amounts use configured-rate estimates; they are not a provider billing limit.
This module never opens a network connection or reads an API-key environment.
"""
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import sqlite3
import stat


class BatchPolicyConflict(ValueError):
    """The ledger has a different immutable configuration; never raise its cap."""


def guard_ledger_path(value: Path) -> Path:
    raw = str(value).replace("\\", "/")
    if ((re.match(r"^[A-Za-z]:", raw) and raw[:2].lower() != "c:") or
            raw.lower() == "/mnt/e" or raw.lower().startswith("/mnt/e/")):
        raise ValueError("ledger_drive_not_allowed")
    path = Path(os.path.abspath(value))
    if os.name == "nt" and path.drive.lower() != "c:":
        raise ValueError("ledger_drive_not_allowed")
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
            raise ValueError("ledger_link_not_allowed")
    return path


def _money(value):
    if type(value) not in (str, int, float):
        raise ValueError("invalid_ledger_amount")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("invalid_ledger_amount") from None
    if not amount.is_finite() or amount < 0:
        raise ValueError("invalid_ledger_amount")
    return amount


def _fingerprint(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BatchPolicyConflict("invalid_policy_fingerprint")


def bind_batch_policy(db: sqlite3.Connection, config_fingerprint: str, config: dict) -> None:
    """Bind once inside the caller's BEGIN IMMEDIATE transaction.

Older one-policy ledgers can be adopted without changing request records.
Mixed-policy legacy ledgers are rejected. Only the fingerprint and safe cap
scalars are saved; configuration values and credentials are never serialized.
"""
    if not db.in_transaction:
        raise ValueError("policy_binding_requires_transaction")
    _fingerprint(config_fingerprint)
    historical = db.execute("SELECT DISTINCT config_fingerprint FROM rank_requests").fetchall()
    if any(row[0] != config_fingerprint for row in historical):
        raise BatchPolicyConflict("legacy_policy_conflict")
    db.execute("""CREATE TABLE IF NOT EXISTS batch_policy (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1), schema_version INTEGER NOT NULL,
        config_fingerprint TEXT NOT NULL, max_calls INTEGER, max_usd TEXT,
        created_utc TEXT NOT NULL)""")
    previous = db.execute("SELECT schema_version,config_fingerprint FROM batch_policy WHERE singleton=1").fetchone()
    if previous is not None:
        if tuple(previous) != (1, config_fingerprint):
            raise BatchPolicyConflict("batch_policy_conflict")
    else:
        max_calls = config.get("max_calls")
        if type(max_calls) is not int or not 0 <= max_calls <= 1000000:
            max_calls = None
        try:
            max_usd = str(_money(config.get("max_usd")))
        except ValueError:
            max_usd = None
        db.execute("INSERT INTO batch_policy VALUES(1,1,?,?,?,?)", (
            config_fingerprint, max_calls, max_usd, datetime.now(timezone.utc).isoformat()))
    # Protect against accidental SQL updates, including a later service instance.
    db.execute("""CREATE TRIGGER IF NOT EXISTS batch_policy_no_update
        BEFORE UPDATE ON batch_policy BEGIN SELECT RAISE(ABORT,'immutable_batch_policy'); END""")
    db.execute("""CREATE TRIGGER IF NOT EXISTS batch_policy_no_delete
        BEFORE DELETE ON batch_policy BEGIN SELECT RAISE(ABORT,'immutable_batch_policy'); END""")
    db.execute("""CREATE TRIGGER IF NOT EXISTS batch_policy_no_replace
        BEFORE INSERT ON batch_policy WHEN EXISTS(SELECT 1 FROM batch_policy)
        BEGIN SELECT RAISE(ABORT,'immutable_batch_policy'); END""")


def assert_batch_policy(db: sqlite3.Connection, config_fingerprint: str) -> None:
    """Check the binding again before using any cached response or reservation."""
    _fingerprint(config_fingerprint)
    row = db.execute("SELECT schema_version,config_fingerprint FROM batch_policy WHERE singleton=1").fetchone()
    if row is None or tuple(row) != (1, config_fingerprint):
        raise BatchPolicyConflict("batch_policy_conflict")


def budget_summary(ledger_path: Path) -> dict:
    """Read a consistent, payload-free summary, with no creation or DB writes.

`unknown_outcomes` includes pending reservations (possibly still in flight) and
completed unknown_outcome fallbacks. `budget_committed_usd` sums the larger of
each call's reserved estimate and configured-rate usage cost.
"""
    path = guard_ledger_path(ledger_path)
    for suffix in ("-journal", "-wal", "-shm"):
        guard_ledger_path(Path(str(path) + suffix))
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        policy = db.execute("SELECT * FROM batch_policy WHERE singleton=1").fetchone()
        if policy is None or policy["schema_version"] != 1:
            raise ValueError("missing_or_unsupported_batch_policy")
        _fingerprint(policy["config_fingerprint"])
        result = {"schema_version": 1, "kind": "batch_budget_summary",
                  "policy_fingerprint": policy["config_fingerprint"],
                  "max_calls": policy["max_calls"], "max_usd": policy["max_usd"],
                  "request_count": 0, "run_count": 0, "reserved_calls": 0,
                  "completed_requests": 0, "pending_requests": 0,
                  "completed_unknown_outcomes": 0, "unknown_outcomes": 0,
                  "usage_reported_calls": 0, "input_tokens": 0, "output_tokens": 0,
                  "total_tokens": 0, "billing_cap_guaranteed": False}
        runs = set()
        reserved, usage_cost, committed = Decimal(0), Decimal(0), Decimal(0)
        for row in db.execute("SELECT * FROM rank_requests"):
            if (row["config_fingerprint"] != policy["config_fingerprint"] or
                    row["status"] not in ("pending", "completed") or
                    row["reserved_call"] not in (0, 1)):
                raise ValueError("invalid_ledger_record")
            payload = json.loads(row["payload_json"])
            run_id = payload["request"]["run_id"]
            if type(run_id) is not str:
                raise ValueError("invalid_ledger_record")
            runs.add(run_id)
            result["request_count"] += 1
            result["reserved_calls"] += row["reserved_call"]
            result[row["status"] + "_requests"] += 1
            reserved_amount = _money(row["reserved_usd"])
            actual_amount = _money(row["actual_usd"]) if row["actual_usd"] is not None else Decimal(0)
            reserved += reserved_amount
            usage_cost += actual_amount
            committed += max(reserved_amount, actual_amount)
            if row["status"] == "completed":
                response = json.loads(row["response_json"])
                if response.get("fallback_reason") == "unknown_outcome":
                    result["completed_unknown_outcomes"] += 1
            if row["usage_json"] is not None:
                usage = json.loads(row["usage_json"])
                if usage is not None:
                    if (type(usage) is not dict or any(type(usage.get(key)) is not int or usage[key] < 0
                            for key in ("input_tokens", "output_tokens", "total_tokens")) or
                            usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]):
                        raise ValueError("invalid_ledger_usage")
                    result["usage_reported_calls"] += 1
                    for key in ("input_tokens", "output_tokens", "total_tokens"):
                        result[key] += usage[key]
        result.update(run_count=len(runs), estimated_reservations_usd=str(reserved),
                      usage_cost_usd=str(usage_cost), budget_committed_usd=str(committed),
                      unknown_outcomes=result["pending_requests"] + result["completed_unknown_outcomes"])
        cap = policy["max_calls"]
        if cap is not None and (type(cap) is not int or cap < 0):
            raise ValueError("invalid_ledger_policy")
        result["remaining_calls"] = max(0, cap - result["reserved_calls"]) if cap is not None else None
        result["remaining_estimated_usd"] = (str(max(Decimal(0), _money(policy["max_usd"]) - committed))
                                              if policy["max_usd"] is not None else None)
        return result
