"""Separate immutable, multi-model budget; never adopts a legacy rank ledger.

Role ledgers retain ordinary request caching. This transport gate adds one
shared reservation pool and protected role allocations before any HTTP call.
Unknown outcomes are never automatically sent again. No credentials or raw
provider text are persisted here.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import time

from orchestrator.batch_budget import guard_ledger_path
from orchestrator.llm_ranker import BASE_URL, DEFAULTS, RankService, _decimal, _http_transport, _json
from worker.explicit_action_contract import digest
from orchestrator.transport_diagnostics import diagnose, TransportCircuitOpen

TRANSPORT_POLICIES = ('legacy_v1', 'latch_on_error_v1')

MODELS = {"scout": "gpt-6-luna", "planner": "gpt-6.1-sol", "learner": "gpt-6-luna"}
CAPACITY_PROFILE = 'part_capacity_v1'


class HierarchyBudgetError(ValueError):
    pass


def validate_policy(policy):
    policy = json.loads(_json(policy))
    if type(policy) is not dict or type(policy.get('schema_version')) is not int:
        raise ValueError("invalid_hierarchy_policy")
    capacity = policy['schema_version'] == 2
    keys = {"schema_version", "enabled", "max_calls", "max_usd", "roles", "episode"}
    if capacity:
        keys.add('execution_profile')
    if (set(policy) != keys or policy['schema_version'] not in (1, 2)
            or capacity and policy.get('execution_profile') != CAPACITY_PROFILE
            or type(policy["enabled"]) is not bool
            or type(policy["max_calls"]) is not int or not 0 < policy["max_calls"] <= 10000
            or type(policy['roles']) is not dict or set(policy["roles"]) != set(MODELS)):
        raise ValueError("invalid_hierarchy_policy")
    total = _decimal(policy["max_usd"])
    if total <= 0:
        raise ValueError("positive_hierarchy_budget_required")
    learning_only = (all(type(policy['roles'][r]) is dict for r in MODELS)
        and type(policy['episode']) is dict
        and policy['roles']['learner'].get('model') == 'gpt-6-astra'
        and policy['roles']['scout'].get('max_calls') == 0
        and policy['roles']['planner'].get('max_calls') == 0
        and policy['episode'].get('planner_calls') == 0)
    for role, cfg in policy["roles"].items():
        astra = type(cfg) is dict and role == 'learner' and cfg.get('model') == 'gpt-6-astra'
        if (type(cfg) is not dict or set(cfg) != set(DEFAULTS)
                or cfg["enabled"] is not True or (cfg["model"] != MODELS[role] and not astra)
                or cfg["base_url"] != BASE_URL or cfg["api_key_env"] != "OPENAI_API_KEY"
                or type(cfg["max_calls"]) is not int or not 0 <= cfg["max_calls"] <= 10000
                or type(cfg["max_output_tokens"]) is not int or not 16 <= cfg["max_output_tokens"] <= (25000 if astra else 4096)
                or cfg["reasoning_effort"] not in ("none", "low", "medium", "high")
                or ((role == "planner" or astra) and cfg["reasoning_effort"] == "none")
                or type(cfg["request_overhead_tokens"]) is not int
                or not 2048 <= cfg["request_overhead_tokens"] <= 65536):
            raise ValueError("invalid_hierarchy_role")
        if astra and (_decimal(cfg['input_usd_per_million']) < 10
                      or _decimal(cfg['output_usd_per_million']) < 50):
            raise ValueError('astra_explicit_standard_rates_required')
        # A planning-only test must not reserve an unrequested learning call.
        # Existing transport quotas reject a zero-call role before any HTTP.
        inactive = (role in ("learner", "scout") or learning_only) and cfg["max_calls"] == 0
        if (role == "scout" and inactive and
                (type(policy["episode"]) is not dict or policy["episode"].get("scout_calls") != 0
                 or policy["episode"].get("fallback_calls") != 0)):
            raise ValueError("inactive_scout_requires_zero_episode_calls")
        if (not inactive and cfg["max_calls"] == 0
                or (inactive and _decimal(cfg["max_usd"]) != 0)
                or (not inactive and _decimal(cfg["max_usd"]) <= 0)
                or any(_decimal(cfg[key]) <= 0 for key in
                       ("input_usd_per_million", "output_usd_per_million"))):
            raise ValueError("explicit_positive_role_rates_required")
    # Reserved sub-budgets protect the planner from scout/learning expenditure.
    if (sum(c["max_calls"] for c in policy["roles"].values()) > policy["max_calls"]
            or sum((_decimal(c["max_usd"]) for c in policy["roles"].values()), Decimal(0)) > total):
        raise ValueError("role_allocations_exceed_global_budget")
    limits = policy["episode"]
    if (type(limits) is not dict or set(limits) != {
            "scout_calls", "planner_calls", "fallback_calls", "max_plan_steps", "recovery_per_step"}
            or any(type(v) is not int for v in limits.values())
            or not 0 <= limits["scout_calls"] <= 4
            or not (1 <= limits["planner_calls"] <= (24 if capacity else 6)
                    or (learning_only and limits['planner_calls'] == 0))
            or not 0 <= limits["fallback_calls"] <= 4
            or not 1 <= limits["max_plan_steps"] <= 64
            or limits["recovery_per_step"] != 1):
        raise ValueError("invalid_hierarchy_episode_limits")
    if capacity:
        # An explicit, separately authorized completion trial, not new defaults
        # or a larger learning/query pool. The CLI binds this policy to its
        # matching capacity execution profile before reading any credential.
        if (policy['max_calls'] > 25 or total > Decimal('15.00')
                or policy['roles']['scout']['max_calls'] > 1
                or policy['roles']['planner']['max_calls'] > 24
                or policy['roles']['learner']['max_calls'] != 0
                or any(policy['roles'][role]['model'] != model for role, model in MODELS.items())
                or limits['scout_calls'] > 1 or limits['fallback_calls'] != 0
                or limits['max_plan_steps'] > 8
                or limits['planner_calls'] > policy['roles']['planner']['max_calls']
                or limits['scout_calls'] > policy['roles']['scout']['max_calls']):
            raise ValueError('capacity_hierarchy_scope_exceeded')
    return policy


def estimate_request(policy, role, body):
    cfg = policy["roles"][role]
    size = len(_json(body).encode("utf-8"))
    if (size > 262144 or body.get("model") != cfg["model"] or body.get("store") is not False
            or body.get("reasoning") != {"effort": cfg["reasoning_effort"]}
            or set(body) - {"model", "store", "instructions", "input", "max_output_tokens", "reasoning", "text"}
            or type(body.get("max_output_tokens")) is not int
            or not 16 <= body["max_output_tokens"] <= cfg["max_output_tokens"]):
        raise HierarchyBudgetError("invalid_role_request_body")
    amount = ((size + cfg["request_overhead_tokens"]) * _decimal(cfg["input_usd_per_million"])
              + body["max_output_tokens"] * _decimal(cfg["output_usd_per_million"])) / 1000000
    return dict(usd=str(amount), body_utf8_bytes=size, output_cap=body["max_output_tokens"],
                method="utf8_bytes_plus_overhead_and_output_cap", billing_cap_guaranteed=False)


class HierarchyBudget:
    def __init__(self, path, policy, *, transport_failure_policy='legacy_v1'):
        if transport_failure_policy not in TRANSPORT_POLICIES:
            raise ValueError('unknown_transport_failure_policy')
        self.transport_failure_policy = transport_failure_policy
        self._transport_policy_binding = transport_failure_policy
        self.path = guard_ledger_path(Path(path))
        self.policy = validate_policy(policy)
        self.binding = digest(self.policy)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='rank_requests'").fetchone():
                raise HierarchyBudgetError("legacy_ledger_must_not_be_repurposed")
            db.execute("CREATE TABLE IF NOT EXISTS hierarchy_policy (id INTEGER PRIMARY KEY CHECK(id=1), binding TEXT, policy TEXT)")
            prior = db.execute("SELECT binding,policy FROM hierarchy_policy WHERE id=1").fetchone()
            if prior and tuple(prior) != (self.binding, _json(self.policy)):
                raise HierarchyBudgetError("immutable_hierarchy_policy_conflict")
            guard_table = db.execute("SELECT 1 FROM sqlite_master WHERE name='hierarchy_transport_policy'").fetchone()
            if guard_table:
                guard = db.execute('SELECT policy FROM hierarchy_transport_policy WHERE id=1').fetchone()
                if guard is None or guard[0] != transport_failure_policy:
                    raise HierarchyBudgetError('immutable_transport_policy_conflict')
            elif prior and transport_failure_policy != 'legacy_v1':
                raise HierarchyBudgetError('new_ledger_required_for_transport_policy')
            elif transport_failure_policy != 'legacy_v1':
                db.execute('CREATE TABLE hierarchy_transport_policy (id INTEGER PRIMARY KEY CHECK(id=1), policy TEXT)')
                db.execute('INSERT INTO hierarchy_transport_policy VALUES(1,?)', (transport_failure_policy,))
                for op in ('UPDATE', 'DELETE'):
                    db.execute(f"CREATE TRIGGER hierarchy_transport_no_{op.lower()} BEFORE {op} ON hierarchy_transport_policy BEGIN SELECT RAISE(ABORT,'immutable_transport_policy'); END")
            if not prior:
                db.execute("INSERT INTO hierarchy_policy VALUES(1,?,?)", (self.binding, _json(self.policy)))
            db.execute("""CREATE TABLE IF NOT EXISTS hierarchy_calls (
                id TEXT PRIMARY KEY, role TEXT, model TEXT, body_sha256 TEXT,
                estimated_usd TEXT, actual_usd TEXT, status TEXT, live INTEGER,
                metadata TEXT, created_utc TEXT)""")
            for op in ("UPDATE", "DELETE"):
                db.execute(f"""CREATE TRIGGER IF NOT EXISTS hierarchy_policy_no_{op.lower()}
                    BEFORE {op} ON hierarchy_policy BEGIN SELECT RAISE(ABORT,'immutable_hierarchy_policy'); END""")

    @contextmanager
    def _db(self):
        for suffix in ("", "-journal", "-wal", "-shm"):
            guard_ledger_path(Path(str(self.path) + suffix))
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def _blocked(self, rows):
        if self.transport_failure_policy != self._transport_policy_binding:
            raise HierarchyBudgetError('mutable_transport_policy_rejected')
        if self.transport_failure_policy == 'legacy_v1':
            return None
        for row in rows:
            meta = json.loads(row['metadata'])
            detail = meta.get('transport_diagnostics', {})
            if row['status'] == 'pending' or (row['status'] == 'unknown' and detail.get('stop_paid_requests')):
                return dict(policy=self.transport_failure_policy, blocked=True,
                    source_call_id=row['id'], source_role=row['role'],
                    category=detail.get('category', 'pending_outcome'), automatic_reset=False,
                    new_attempt_reserved=False)
        return None

    def transport_blocked(self):
        if self.transport_failure_policy != self._transport_policy_binding:
            raise HierarchyBudgetError('mutable_transport_policy_rejected')
        if self.transport_failure_policy == 'legacy_v1':
            return None
        with self._db() as db:
            return self._blocked(db.execute('SELECT * FROM hierarchy_calls ORDER BY created_utc,id').fetchall())

    def transport(self, role, backend=None):
        if role not in MODELS:
            raise ValueError("unknown_model_role")
        live = backend is None
        send = backend or _http_transport

        def call(url, headers, body, timeout):
            if url != BASE_URL + "/responses":
                raise HierarchyBudgetError("unexpected_provider_endpoint")
            if live and self.policy["enabled"] is not True:
                raise HierarchyBudgetError("live_hierarchy_not_authorized")
            if digest(self.policy) != self.binding:
                raise HierarchyBudgetError("mutable_policy_rejected")
            estimate = estimate_request(self.policy, role, body)
            identity = digest(dict(role=role, body=body))
            cfg = self.policy["roles"][role]
            with self._db() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT binding FROM hierarchy_policy WHERE id=1").fetchone()[0] != self.binding:
                    raise HierarchyBudgetError("immutable_hierarchy_policy_conflict")
                if db.execute("SELECT 1 FROM hierarchy_calls WHERE id=?", (identity,)).fetchone():
                    raise HierarchyBudgetError("transport_already_attempted_no_resend")
                rows = db.execute("SELECT * FROM hierarchy_calls").fetchall()
                if self._blocked(rows):
                    raise TransportCircuitOpen('transport_circuit_open_no_new_attempt')
                group = [r for r in rows if r["role"] == role]
                for records, max_calls, max_usd in (
                    (rows, self.policy["max_calls"], self.policy["max_usd"]),
                    (group, cfg["max_calls"], cfg["max_usd"])):
                    committed = sum((max(_decimal(r["estimated_usd"]), _decimal(r["actual_usd"] or "0"))
                                     for r in records), Decimal(0))
                    if len(records) >= max_calls or committed + _decimal(estimate["usd"]) > _decimal(max_usd):
                        raise HierarchyBudgetError("hierarchy_call_or_cost_limit")
                db.execute("INSERT INTO hierarchy_calls VALUES(?,?,?,?,?,NULL,'pending',?,?,?)", (
                    identity, role, cfg["model"], digest(body), estimate["usd"], int(live),
                    _json(dict(estimate=estimate, retries=0)),
                    datetime.now(timezone.utc).isoformat()))
            meta = dict(estimate=estimate, retries=0, llm_live=live,
                        usage=None, actual_cost_from_usage_usd=None)
            started = time.monotonic()
            try:
                raw = send(url, headers, body, timeout)
                if type(raw) is dict:
                    key = headers.get("Authorization", "").removeprefix("Bearer ")
                    rates = {k: _decimal(cfg[k]) for k in ("input_usd_per_million", "output_usd_per_million")}
                    RankService._usage(raw, meta, rates, key or "unavailable-key-placeholder")
                with self._db() as db:
                    db.execute("UPDATE hierarchy_calls SET status='completed',metadata=?,actual_usd=? WHERE id=?", (
                        _json(meta), meta["actual_cost_from_usage_usd"], identity))
                return raw
            except Exception as error:
                meta["error_type"] = type(error).__name__
                meta['transport_diagnostics'] = diagnose(error)
                meta['transport_elapsed_seconds'] = round(time.monotonic()-started, 6)
                # Sending may have reached the provider: keep full reservation.
                with self._db() as db:
                    db.execute("UPDATE hierarchy_calls SET status='unknown',metadata=? WHERE id=?",
                               (_json(meta), identity))
                raise
        call.live = live
        call.circuit_status = self.transport_blocked
        return call

    def summary(self):
        with self._db() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM hierarchy_calls ORDER BY created_utc,id")]
        used = sum((_decimal(r["actual_usd"] or "0") for r in rows if r["live"]), Decimal(0))
        simulated = sum((_decimal(r["actual_usd"] or "0") for r in rows if not r["live"]), Decimal(0))
        committed = sum((max(_decimal(r["estimated_usd"]), _decimal(r["actual_usd"] or "0"))
                         for r in rows), Decimal(0))
        result = dict(policy_sha256=self.binding, max_calls=self.policy["max_calls"],
            max_usd=self.policy["max_usd"], transport_attempts=len(rows),
            live_api_attempts=sum(r["live"] for r in rows),
            injected_provider_attempts=sum(not r["live"] for r in rows),
            pending_or_unknown=sum(r["status"] != "completed" for r in rows),
            usage_cost_usd=str(used), fixture_simulated_usage_usd=str(simulated), committed_usd=str(committed),
            remaining_calls=self.policy["max_calls"] - len(rows),
            remaining_usd=str(_decimal(self.policy["max_usd"]) - committed),
            roles={role: sum(r["role"] == role for r in rows) for role in MODELS},
            billing_cap_guaranteed=False, automatic_http_retries=0, calls=rows)
        if self.transport_failure_policy != 'legacy_v1':
            result.update(transport_failure_policy=self.transport_failure_policy, transport_guard=self._blocked(rows))
        return result
