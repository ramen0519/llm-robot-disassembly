"""Derived observations; never rewrite evidence or turn failure into a constraint."""
from copy import deepcopy

from orchestrator.explicit_action_memory import FrozenActionMemory
from orchestrator.skill_registry import retrieve
from worker.stability_diagnostics import project_stability_diagnostics

DERIVATION_VERSION = "knowledge_observations_v1"


def observed_signature(result: dict, choice: dict) -> dict:
    """Project measured stages using the observer vocabulary already in use.

    A failed bounded path search is not a collision or impossibility proof.
    Direction labels are retained without inferring a world/CAD frame.
    """
    raw = result.get("raw") or {}
    diagnostic, diagnostic_error = None, None
    if raw.get("stability_diagnostics") is not None:
        try:
            diagnostic = project_stability_diagnostics(raw["stability_diagnostics"])
        except (ValueError, TypeError, KeyError):
            diagnostic_error = "invalid_stability_diagnostics"
    path, stable = result.get("path_success"), result.get("stability_success")
    stage = ("path" if path is False else "stability" if path is True and stable is False
             else "none" if result.get("status") == "PASS" and path is True and stable is True else None)
    value = dict(status=result.get("status"), observed_failure_stage=stage,
        path_success=path, stability_success=stable,
        stability_phase=diagnostic["phase"] if diagnostic else None,
        diagnostic_error=diagnostic_error, verifier_origin=result.get("verifier_origin"),
        deadline_exceeded=raw.get("deadline_exceeded") if type(raw.get("deadline_exceeded")) is bool else None,
        primitive=choice.get("skill_id"), direction_id=choice.get("direction_id"),
        second_direction_id=choice.get("second_direction_id"),
        direction_class=None, direction_frame="not_inferred_from_direction_label",
        support_count=len(choice["support_ids"]) if type(choice.get("support_ids")) is list else None,
        failure_cause=None, blocker=None, hard_pruning=False, physics_reuse_allowed=False)
    from orchestrator.execution_observation import project_execution
    value.update(project_execution(result))
    return value


def build_index(memory: dict, skills: dict, *, allow_fixture: bool = False) -> dict:
    """Keep original record and program hashes as references, not new verdicts."""
    records = FrozenActionMemory(memory).snapshot["records"]
    programs = retrieve(skills, allow_fixture=allow_fixture)
    return dict(schema_version=1, kind="knowledge_derived_index_v1",
        derivation_version=DERIVATION_VERSION,
        memory_content_sha256=memory["content_sha256"],
        skills_content_sha256=skills["content_sha256"],
        observations=[dict(evidence_id=row["evidence_id"], source_record_sha256=row["record_sha256"],
            source_run_id=row["source_run_id"], source_request_id=row["source_request_id"],
            source_action_id=row["source_action_id"], source_evidence_ref=row["result"]["evidence_ref"],
            observed_signature=observed_signature(row["result"], row["choice"]),
            skill_usage=dict(generated_skill_id=None, program_hash=None, decision_origin=None,
                             binding_status="not_encoded_in_memory_record; see episode usage report"))
            for row in records],
        skills=[dict(program_hash=row["program_hash"], lifecycle=row["lifecycle"],
            trigger_rules=[dict(rule_index=i, all_of=deepcopy(rule["when"]),
                                select=deepcopy(rule["select"]), rank=deepcopy(rule["rank"]))
                           for i, rule in enumerate(row["program"].get("rules", []))],
            **(dict(program_language="python_policy_v1",
                    python_trigger_scope="evaluate_current_context; no static predicate claim")
               if row["program"]["language"] == "python_policy_v1" else {}),
            validation_scope=deepcopy(row["validation_scope"]),
            development_scope=deepcopy(row.get("development_scope", [])),
            requires_current_request_binding=True, requires_fresh_physics=True)
            for row in programs],
        hard_pruning=False, physics_reuse_allowed=False)
