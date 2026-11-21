"""Pinned plan provenance, separate from the worker's current action citations.

Snapshots are historical source data, never injected into a new observation or
used as a physical verdict. Changing a bounded retrieval list does not change a
plan's provenance; changing the content of the same evidence ID is an error.
"""
from copy import deepcopy

from orchestrator.conditional_plan import task_scope
from worker.explicit_action_contract import digest, validate_request

POLICY = "strategy_evidence_pin_v1"
# These three values are recomputed by FrozenActionMemory.retrieve for each
# current request. Every source/result/safety field remains in the source core.
MEMORY_QUERY_DECORATIONS = frozenset(("same_task", "same_conditions", "reference_only"))
_BINDING_KEYS = {"policy", "task_scope", "first_request_sha256", "source_request_id",
                 "source_state_id", "evidence_ids", "records", "binding_sha256"}
_RECORD_KEYS = {"origin", "snapshot", "snapshot_sha256", "core_sha256"}


def _ids(value):
    if (type(value) is not list or len(value) > 128
            or any(type(identity) is not str or not identity for identity in value)
            or len(value) != len(set(value))):
        raise ValueError("invalid_strategy_evidence_ids")


def _index(request):
    records = {}
    for origin in ("memory_references", "episode_feedback"):
        for snapshot in request[origin]:
            identity = snapshot["evidence_id"]
            # Even equal duplicate rows have ambiguous occurrence/origin. The
            # current explicit contract permits some such lists; pins do not.
            if identity in records:
                raise ValueError("strategy_evidence_identity_conflict")
            records[identity] = (origin, snapshot)
    return records


def _core(origin, snapshot):
    if origin not in ("memory_references", "episode_feedback") or type(snapshot) is not dict:
        raise ValueError("invalid_strategy_evidence_record")
    if origin == "memory_references":
        for key in MEMORY_QUERY_DECORATIONS:
            if key in snapshot and type(snapshot[key]) is not bool:
                raise ValueError("invalid_strategy_evidence_decoration")
        if "source_task_context" in snapshot and (
                snapshot.get("source_context_sha256") != digest(snapshot["source_task_context"])):
            raise ValueError("strategy_evidence_source_context_mismatch")
        return {key:deepcopy(value) for key,value in snapshot.items() if key not in MEMORY_QUERY_DECORATIONS}
    return deepcopy(snapshot)


def capture_plan_evidence(request, evidence_ids):
    """Copy only cited supplied records and bind them to the planning request."""
    request = validate_request(request)
    _ids(evidence_ids)
    indexed = _index(request)
    if not set(evidence_ids).issubset(indexed):
        raise ValueError("strategy_evidence_not_supplied")
    records = {}
    for identity in sorted(evidence_ids):
        origin, snapshot = indexed[identity]
        records[identity] = dict(origin=origin, snapshot=deepcopy(snapshot),
            snapshot_sha256=digest(snapshot), core_sha256=digest(_core(origin, snapshot)))
    binding = dict(policy=POLICY, task_scope=task_scope(request), first_request_sha256=digest(request),
        source_request_id=request["request_id"], source_state_id=request["state_id"],
        evidence_ids=sorted(evidence_ids), records=records)
    binding["binding_sha256"] = digest(binding)
    return binding


def evidence_bound_plan_id(wire_value, registered_program_hashes, binding):
    """Content identity includes the particular historical evidence snapshots."""
    return "plan:" + digest(dict(contract="strategy_plan_v1", value=wire_value,
        registered_program_hashes=registered_program_hashes,
        evidence_binding_sha256=binding["binding_sha256"]))


def validate_plan_evidence(plan, request):
    """Check source identity/integrity and any reappearing current evidence core.

    A new observation with another ID does not erase a historical fact. Runtime
    failure/attempt guards still use the fresh request and can reject a concrete
    action. Historical success is never promoted to current acceptance.
    """
    request = validate_request(request)
    if type(plan) is not dict or type(plan.get("evidence_binding")) is not dict:
        raise ValueError("strategy_evidence_binding_missing")
    binding = plan["evidence_binding"]
    if set(binding) != _BINDING_KEYS or binding.get("policy") != POLICY:
        raise ValueError("invalid_strategy_evidence_binding")
    if binding["binding_sha256"] != digest({k:v for k,v in binding.items() if k != "binding_sha256"}):
        raise ValueError("strategy_evidence_binding_tampered")
    if (binding["task_scope"] != plan.get("task_scope")
            or binding["task_scope"] != task_scope(request)):
        raise ValueError("strategy_evidence_task_scope_changed")
    if (binding["first_request_sha256"] != plan.get("first_request_sha256")
            or binding["source_request_id"] != plan.get("request_id")
            or binding["source_state_id"] != plan.get("state_id")):
        raise ValueError("strategy_evidence_source_mismatch")
    _ids(binding["evidence_ids"])
    records = binding["records"]
    if type(records) is not dict or set(records) != set(binding["evidence_ids"]):
        raise ValueError("strategy_evidence_record_set_mismatch")
    for identity, entry in records.items():
        if (type(entry) is not dict or set(entry) != _RECORD_KEYS
                or type(entry["snapshot"]) is not dict or entry["snapshot"].get("evidence_id") != identity
                or entry["snapshot_sha256"] != digest(entry["snapshot"])
                or entry["core_sha256"] != digest(_core(entry["origin"], entry["snapshot"]))):
            raise ValueError("strategy_evidence_snapshot_tampered")
    # The plan ID anchors the pin too: simply recomputing a modified binding's
    # self-digest must not make it the original model-authorized plan.
    from orchestrator.strategy_plan import CONTRACT, wire_step
    from orchestrator.strategy_failover import LEGACY, validate_policy
    from orchestrator.conditional_plan import PLAN_KEYS
    try:
        wire = {key:deepcopy(plan[key]) for key in PLAN_KEYS}
        failure_policy = validate_policy(plan.get('strategy_failure_policy', LEGACY))
        wire["steps"] = [wire_step(step, failure_policy) for step in plan["steps"]]
        cited = set()
        for step in wire["steps"]:
            _ids(step["evidence_ids"])
            cited.update(step["evidence_ids"])
            for other in (step.get('stability_failover') or {}).get('alternatives', []):
                _ids(other['evidence_ids'])
                cited.update(other['evidence_ids'])
        expected = evidence_bound_plan_id(wire, plan["registered_program_hashes"], binding)
    except (KeyError, TypeError):
        raise ValueError("invalid_strategy_evidence_plan_binding") from None
    if cited != set(records):
        raise ValueError("strategy_evidence_citation_set_mismatch")
    if plan.get("plan_contract") != CONTRACT or plan.get("plan_id") != expected:
        raise ValueError("strategy_evidence_plan_identity_mismatch")
    for identity, (origin, snapshot) in _index(request).items():
        if identity in records:
            previous = records[identity]
            if origin != previous["origin"] or digest(_core(origin, snapshot)) != previous["core_sha256"]:
                raise ValueError("strategy_evidence_current_record_conflict")
    return deepcopy(binding)


def project_plan_evidence(plan, request, evidence_ids):
    """Return legal current citations plus compact provenance for omitted pins."""
    binding = validate_plan_evidence(plan, request)
    _ids(evidence_ids)
    if not set(evidence_ids).issubset(binding["records"]):
        raise ValueError("strategy_evidence_citation_not_pinned")
    current = [identity for identity in evidence_ids if identity in request["evidence_ids"]]
    historical = [identity for identity in evidence_ids if identity not in request["evidence_ids"]]
    return dict(evidence_ids=current, provenance=dict(policy=POLICY,
        binding_sha256=binding["binding_sha256"], cited_evidence_ids=list(evidence_ids),
        current_evidence_ids=current, pinned_only_evidence_ids=historical,
        record_sha256_by_id={identity:binding["records"][identity]["snapshot_sha256"] for identity in evidence_ids},
        historical_reference_only=True, snapshot_reinjected=False, current_success_implied=False,
        physics_reuse_allowed=False, requires_fresh_physics=True))
