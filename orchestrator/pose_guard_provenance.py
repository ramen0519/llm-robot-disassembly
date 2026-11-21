"""Recover a typed response mislabeled at a verified live quota wrapper.

Never resend HTTP or edit old artifacts. Both ledgers must bind the same
completed request, provider identity and typed response before relabeling.
"""
from copy import deepcopy
import json
from orchestrator.llm_ranker import _json, _hash
from worker.explicit_action_contract import digest


def verified_proposal(proposal, local, shared, envelope, body):
    meta = json.loads(shared["metadata"])
    if (proposal.get("status") != "proposed" or proposal.get("origin") != "injected_transport"
            or proposal.get("metadata", {}).get("llm_live") is not False
            or local["status"] != "completed"
            or local["payload_json"] != _json(envelope)
            or local["payload_hash"] != _hash(_json(envelope))
            or json.loads(local["response_json"]) != proposal
            or shared["id"] != digest(dict(role="scout", body=body))
            or shared["body_sha256"] != digest(body)
            or shared["role"] != "scout" or shared["status"] != "completed"
            or shared["live"] != 1 or meta.get("llm_live") is not True
            or meta.get("retries") != 0):
        raise ValueError("completed_live_bridge_proof_required")
    for key in ("response_id", "model_returned", "usage", "actual_cost_from_usage_usd"):
        if meta.get(key) is None or meta[key] != proposal["metadata"].get(key):
            raise ValueError("live_bridge_provider_identity_mismatch")
    if shared["model"] != body["model"] or meta["model_returned"] != body["model"]:
        raise ValueError("live_bridge_model_mismatch")
    value = deepcopy(proposal)
    value["origin"] = "actual_llm"
    value["metadata"].update(llm_live=False, original_llm_live=True,
        attempted=False, cached=True, transport_kind="verified_completed_live_bridge",
        provenance_correction_only=True, new_api_call=False,
        prior_response_sha256=digest(proposal),
        shared_call_id=shared["id"], shared_body_sha256=shared["body_sha256"])
    return value
