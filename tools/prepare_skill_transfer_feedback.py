"""Explicitly promote an observed failed transfer into adaptive development.

This copies immutable local evidence into a NEW run. It never calls an API or
simulator, never creates a physical PASS, and never leaves the promoted target
labelled held-out. The generated program remains the original provider output.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tools"))

from run_integrated_repair import checked
from orchestrator.skill_case import validate_trigger_scope
from orchestrator.skill_program import program_hash, step_program, validate_program
from orchestrator.skill_registry import retrieve
from orchestrator.skill_validation import check_evidence
from worker.explicit_action_contract import action_id, validate_request, validate_response


def _load(path):
    return json.loads(checked(path, True).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(checked(path, True).read_bytes()).hexdigest()


def _new_json(path, value):
    path = checked(path, True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _actual_luna(response, status):
    metadata = response.get("metadata") or {}
    returned = metadata.get("model_returned")
    if (response.get("origin") != "actual_llm" or response.get("status") != status
            or not (metadata.get("llm_live") is True or metadata.get("original_llm_live") is True)
            or metadata.get("transport_kind") != "live_http"
            or metadata.get("model_requested") != "gpt-6-luna"
            or not isinstance(returned, str)
            or not (returned == "gpt-6-luna" or returned.startswith("gpt-6-luna-"))
            or response.get("fallback_reason") is not None):
        raise ValueError("actual_luna_response_required")


def _known_failure(raw, worker):
    elapsed = raw.get("elapsed_seconds")
    limit = worker.get("context", {}).get("remaining_wall_seconds")
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
            or type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit <= 45
            or elapsed > limit):
        raise ValueError("nonlate_actual_failure_required")
    if (raw.get("status") not in ("UNKNOWN", "REJECTED") or not (
            raw.get("path_success") is False or raw.get("path_success") is True
            and raw.get("stability_success") is False)):
        raise ValueError("observed_failed_attempt_required")


def prepare_transfer_feedback(transfer_run, generation_run, run_dir, manifest_output):
    """Validate first, then copy exact bytes and add explicit development labels."""
    transfer = checked(transfer_run, True)
    generation = checked(generation_run, True)
    output = checked(run_dir, True)
    manifest_path = checked(manifest_output, True)
    if output.exists():
        raise FileExistsError("new_run_required")
    if output not in manifest_path.parents or manifest_path.suffix.lower() != ".json":
        raise ValueError("new_manifest_must_be_inside_new_run")
    reserved = {"provider_proposal.json", "precommitted_cases.json", "result.json", "config.json",
                "generation_packet.json", "promotion.json"}
    if manifest_path.relative_to(output).as_posix() in reserved:
        raise ValueError("manifest_output_conflicts_with_bundle_artifact")
    originals = {}

    def load(path):
        path = checked(path, True)
        originals[str(path)] = _sha(path)
        return _load(path)

    proposal = load(generation / "provider_proposal.json")
    _actual_luna(proposal, "proposed")
    program = validate_program(proposal["program"])
    identity = program_hash(program)
    generation_result = load(generation / "result.json")
    if (generation_result.get("status") != "REGISTERED_SCOPED"
            or generation_result.get("candidate_id") != identity
            or generation_result.get("admission", {}).get("admitted") is not True):
        raise ValueError("registered_source_generation_required")
    source_preservation = load(generation / "source_preservation.json")
    if source_preservation.get("unchanged") is not True:
        raise ValueError("source_generation_not_preserved")
    library_path = generation / "skills_frozen.json"
    snapshot = load(library_path)
    rows = retrieve(snapshot)
    matching = [row for row in rows if row["program_hash"] == identity]
    if snapshot.get("fixture") is not False or len(matching) != 1:
        raise ValueError("actual_registered_program_required")
    entry = next(item for item in snapshot["entries"] if item["candidate"]["program_hash"] == identity)
    provenance = entry["candidate"]["generation"]
    if (provenance.get("origin") != "live_llm"
            or checked(provenance["provider_response_path"], True) != generation / "provider_proposal.json"
            or provenance.get("provider_response_sha256") != _sha(generation / "provider_proposal.json")):
        raise ValueError("generation_library_provenance_mismatch")
    for record in entry["validations"]:
        check_evidence(record)
    precommit = load(transfer / "precommitted_case.json")
    if (checked(precommit["library_path"], True) != library_path
            or precommit.get("library_file_sha256") != _sha(library_path)):
        raise ValueError("transfer_used_different_library")
    preserved = load(transfer / "preservation.json")
    if preserved.get("source_unchanged") is not True or preserved.get("library_unchanged") is not True:
        raise ValueError("transfer_source_or_library_not_preserved")
    report = load(transfer / "case_report.json")
    if (report.get("workflow_completed") is not True or report.get("fixture_library") is not False
            or report.get("generated_reuse_executed") is not True
            or report.get("llm_selection_used") is not True
            or report.get("generated_skill_id") != identity
            or report.get("recovery_accepted") is not False
            or report.get("generated_skill_fallback_reason") is not None):
        raise ValueError("actual_failed_generated_transfer_required")
    request = validate_request(load(transfer / "program_request.json"))
    response = validate_response(request, load(transfer / "response.json"))
    trace = step_program(program, request)
    if (trace.get("response") is None or trace["response"]["choice"] != response["choice"]
            or action_id(request, response["choice"]) != response["action_id"]
            or response["metadata"].get("generated_program_hash") != identity
            or response["metadata"].get("generated_skill_id") != identity
            or response["metadata"].get("generated_program_executed") is not True
            or response["metadata"].get("generated_library_fixture") is not False
            or response["metadata"].get("generated_library_sha256") != snapshot["content_sha256"]):
        raise ValueError("program_replay_response_mismatch")
    selections = load(transfer / "selector_decisions.json")
    selected = [item for item in selections if item["response"].get("selected_skill_id") == identity
                and item["packet"].get("current_request") == request]
    if len(selected) != 1:
        raise ValueError("bound_actual_selector_required")
    selector = selected[0]["response"]
    _actual_luna(selector, "selected")
    if response["metadata"].get("generated_selector_response") != selector:
        raise ValueError("selector_response_binding_mismatch")
    options = selected[0]["packet"].get("skills", [])
    option = [item for item in options if item.get("skill_id") == identity]
    if (len(option) != 1 or option[0].get("program_hash") != identity
            or option[0].get("program") != program
            or option[0].get("proposed_action") != response["choice"]):
        raise ValueError("selector_program_binding_mismatch")

    raw_stages, copy_pairs = {}, []
    for stage in ("trigger", "recovery"):
        directory = transfer / stage / "physics"
        worker = load(directory / "request.json")
        raw = load(directory / "result.json")
        validate_trigger_scope(request, raw, worker["choice"])
        _known_failure(raw, worker)
        if raw.get("source_hashes") != precommit.get("source_hashes"):
            raise ValueError("changed_physical_source_hashes")
        if stage == "recovery" and worker["choice"] != response["choice"]:
            raise ValueError("recovery_worker_action_mismatch")
        if stage == "trigger":
            feedback = request.get("episode_feedback", [])
            if (len(feedback) != 1 or feedback[0].get("choice") != worker["choice"]
                    or feedback[0].get("path_success") != raw.get("path_success")
                    or feedback[0].get("stability_success") != raw.get("stability_success")):
                raise ValueError("current_trigger_feedback_mismatch")
        if (raw.get("trajectory_file") != "trajectory.json"
                or raw.get("evidence_file") != "endpoint_evidence.json"):
            raise ValueError("bound_physical_evidence_required")
        for filename, hash_key in (("trajectory.json", "trajectory_sha256"),
                                   ("endpoint_evidence.json", "evidence_sha256")):
            source = directory / filename
            artifact = load(source)
            if _sha(source) != raw.get(hash_key) or artifact.get("raw_success") is not raw["path_success"]:
                raise ValueError("physical_artifact_hash_or_status_mismatch")
            if filename == "endpoint_evidence.json" and artifact.get("scope", {}).get("selected_action") != worker["choice"]:
                raise ValueError("physical_endpoint_action_mismatch")
        raw_stages[stage] = raw
        for filename in ("request.json", "result.json", "trajectory.json", "endpoint_evidence.json"):
            copy_pairs.append((directory / filename, stage + "/physics/" + filename))
        optional = directory / "constraint_records.json"
        if optional.is_file():
            load(optional)
            copy_pairs.append((optional, stage + "/physics/constraint_records.json"))
    if raw_stages["trigger"]["input_hashes"] != raw_stages["recovery"]["input_hashes"]:
        raise ValueError("changed_physical_input_hashes")
    spec = deepcopy(precommit["spec"])
    if (spec.get("source_task") != report.get("target_task") or spec.get("phase") != "transfer"
            or spec.get("source_task") != "07009"):
        raise ValueError("declared_07009_transfer_required")
    source_report = checked(ROOT / spec["report"], True)
    originals[str(source_report)] = _sha(source_report)
    if originals[str(source_report)] != precommit.get("source_report_sha256"):
        raise ValueError("declared_source_report_changed")
    spec.update(case_id="07009_promoted_transfer_debug", phase="debug", trigger_kind="path_failed")
    if raw_stages["trigger"]["path_success"] is not False:
        raise ValueError("declared_path_failure_trigger_required")
    generation_manifest = load(generation / "precommitted_cases.json")
    admissions = [deepcopy(case) for case in generation_manifest["cases"] if case.get("phase") == "admission"]
    if len(admissions) != 2 or any(case.get("source_task") != "beam" for case in admissions):
        raise ValueError("two_original_beam_admission_cases_required")
    config = load(generation / "config.json")
    transfer_config = load(transfer / "config.json")
    if (checked(config["ledger"], True) != checked(transfer_config["ledger"], True)
            or config.get("api") != transfer_config.get("api")
            or transfer_config.get("fixture_library_explicit") is not False):
        raise ValueError("shared_approved_ledger_and_config_required")
    for path, expected in originals.items():
        if _sha(path) != expected:
            raise ValueError("source_changed_during_validation")

    # Every validation above finishes before the first output directory exists.
    output.mkdir(parents=True, exist_ok=False)
    copied = []

    def copy_exact(source, target):
        source, target = checked(source, True), checked(target, True)
        before = originals.get(str(source), _sha(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        if _sha(target) != before or _sha(source) != before:
            raise ValueError("copy_preservation_failed")
        copied.append({"source_path": str(source), "source_sha256": before,
                       "bundle_path": str(target), "copied_bytes_identical": True})

    case_dir = output / "cases" / spec["case_id"]
    for source, relative in copy_pairs:
        copy_exact(source, case_dir / relative)
    copy_exact(transfer / "program_request.json", case_dir / "program_request.json")
    copy_exact(generation / "provider_proposal.json", output / "provider_proposal.json")
    copy_exact(generation / "config.json", output / "config.json")
    trace.update(replayed_from_saved_observations=True, new_physical_execution=False,
                 replay_purpose="explicit_transfer_promotion_to_adaptive_development")
    _new_json(case_dir / "program_step.json", trace)
    _new_json(output / "precommitted_cases.json", {
        "schema_version": 1, "purpose": "explicit_post_observation_development_promotion",
        "cases": [spec], "not_precommitted_before_original_transfer": True,
        "promoted_target_is_held_out": False})
    _new_json(output / "result.json", {
        "status": "TRANSFER_PROMOTED_TO_DEVELOPMENT", "generation_origin": "live_llm",
        "candidate_id": identity, "cases": [{"case_id": spec["case_id"],
            "outcome": raw_stages["recovery"]["status"]}],
        "new_api_calls": 0, "new_physical_calls": 0, "adaptive_development": True,
        "performance_comparison": "NOT_RUN", "generalization_claim": False})
    _new_json(output / "generation_packet.json", {
        "validation_feedback": [], "artifact_role": "promotion_bundle_not_a_new_provider_request",
        "new_api_calls": 0, "source_generation_run": str(generation)})
    future_manifest = {"schema_version": 1,
        "purpose": "adaptive development after explicit transfer-to-debug promotion; not held-out evaluation",
        "experience_task": generation_manifest["experience_task"],
        "experience_report": generation_manifest["experience_report"],
        "selection_rationale": "Reuse the observed failed 07009 case as openly declared development; retain original two Beam admission cases without replacement.",
        "cases": [spec] + admissions,
        "adaptive_development": True, "promoted_target_is_held_out": False,
        "source_promotion_bundle": str(output)}
    _new_json(manifest_path, future_manifest)
    unchanged = all(_sha(path) == expected for path, expected in originals.items())
    audit = {"schema_version": 1, "status": "PROMOTED_TO_DEVELOPMENT",
        "source_transfer_run": str(transfer), "source_generation_run": str(generation),
        "program_hash": identity, "promoted_case_id": spec["case_id"],
        "manifest_output": str(manifest_path), "adaptive_development": True,
        "target_was_already_observed": True, "promoted_target_is_held_out": False,
        "performance_or_generalization_claim": False,
        "new_api_calls": 0, "new_physical_calls": 0,
        "original_files_unchanged": unchanged,
        "original_artifact_sha256": originals, "copies": copied,
        "program_step_origin": "local_interpreter_replay_of_saved_request_not_new_physics"}
    _new_json(output / "promotion.json", audit)
    if not unchanged:
        raise ValueError("source_changed_during_promotion")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-run", type=Path, required=True)
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = prepare_transfer_feedback(args.transfer_run, args.generation_run, args.run_dir, args.manifest_output)
    print(json.dumps({key: audit[key] for key in ("status", "program_hash", "manifest_output",
        "adaptive_development", "promoted_target_is_held_out", "new_api_calls", "new_physical_calls",
        "original_files_unchanged")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
