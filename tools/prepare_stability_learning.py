"""Predeclare chronological measured stability development conditions.

This preparation reads saved actual artifacts; it runs neither a model nor a
simulator. All failures remain in the selection audit. Initial contact-graph
disconnection is not relabeled as observed dynamic movement. The nested generic
episode preparation is an integrity gate only; its manifest is NEVER the
stability experiment manifest returned by this module.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import prepare_episode_learning as episode_preparation
from orchestrator.skill_validation import _condition
from worker.stability_diagnostics import project_stability_diagnostics
from worker.task_memory import digest

INTEGRITY_TEMPLATE = ROOT / "configs/skill_discovery_cases_v1.json"
POLICY = "first_three_distinct_measured_dynamic_stability_failures_in_recorded_order_v1"
INITIAL_POLICY = "first_three_distinct_measured_initial_contact_failures_in_recorded_order_v1"
FAILURE_SCOPES = ("dynamic_failure", "initial_disconnected")


def _failure_scope(value):
    if type(value) is not str or value not in FAILURE_SCOPES:
        raise ValueError("unsupported_stability_learning_failure_scope")
    return value


def _initial_contact_condition(attempt, physical, row, diagnostic):
    """Bind the initial checker branch without treating it as a fall or cause.

    More disconnected parts than available supports is an observed difficulty,
    not a selection exclusion or evidence that other removal choices cannot work.
    """
    raw, choice, context = attempt["raw"], physical["choice"], physical["context"]
    if diagnostic["phase"] != "initial_disconnected":
        row["reason"] = "non_initial_contact_stability_failure"
        return row
    full = raw["stability_diagnostics"]
    if (diagnostic["check_success_returned"] is not True
            or diagnostic["returned_success"] is not False
            or diagnostic["physics_steps"] != 0
            or diagnostic["dynamic_failure_parts"] != []
            or full.get("instrumentation_errors") != []):
        row["reason"] = "incomplete_initial_contact_observation"
        return row
    initial = diagnostic["initial_disconnected_parts"]
    returned = diagnostic["returned_parts"]
    upstream = raw.get("raw_upstream") or {}
    if (not initial or returned != initial or upstream.get("parts_fall") != returned
            or upstream.get("path_success") is not True
            or upstream.get("stability_success") is not False):
        row["reason"] = "initial_contact_returned_parts_binding_mismatch"
        return row
    rest = set(attempt["remaining_parts"]) - {choice["part_id"]}
    fixed = set(choice["support_ids"])
    if context.get("base_part") is not None:
        fixed.add(context["base_part"])
    free = rest - fixed
    moving = (raw.get("executed") or {}).get("stability_moving_parts")
    if (not fixed.issubset(rest) or type(moving) is not list
            or len(moving) != len(set(moving)) or set(moving) != free
            or not set(initial).issubset(free)):
        row["reason"] = "initial_contact_parts_not_current_free_parts"
        return row
    graph = full.get("initial_contact_graph") or {}
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if (type(nodes) is not list or any(type(n) is not str for n in nodes)
            or len(nodes) != len(set(nodes)) or not free.issubset(nodes)
            or type(edges) is not list
            or any(type(edge) is not list or len(edge) != 2
                   or any(type(n) is not str or n not in nodes for n in edge) for edge in edges)):
        row["reason"] = "initial_contact_graph_evidence_invalid"
        return row
    incident = {node for edge in edges for node in edge}
    if set(initial) != free - incident:
        row["reason"] = "initial_contact_graph_returned_parts_mismatch"
        return row
    budget = context["max_grippers"] - 1
    condition = _condition({"request": {"value": physical}, "result": {"value": raw}}, "unused_alias")
    row.update(eligible=True, reason="measured_initial_contact_disconnection",
               condition_hash=digest(condition),
               observed_initial_disconnected_parts=deepcopy(initial),
               current_free_stability_parts=sorted(free), available_support_budget=budget,
               disconnected_count_exceeds_support_budget=len(initial) > budget,
               excluded_for_support_budget=False,
               dynamic_fall_observed=False, all_alternative_actions_impossible=False)
    return row


def classify_condition(attempt, integrity_item, physical, *, failure_scope="dynamic_failure"):
    """Pure classification after the full raw/Memory integrity gate succeeded."""
    _failure_scope(failure_scope)
    raw, choice, context = attempt["raw"], physical["choice"], physical["context"]
    row = {"attempt_index": integrity_item["attempt_index"],
           "event_id": attempt["event_id"], "request_id": attempt["request_id"],
           "state_id": attempt["state_id"], "choice": deepcopy(choice),
           "remaining_parts": deepcopy(attempt["remaining_parts"]),
           "memory_evidence_id": integrity_item["memory_evidence_id"],
           "raw_result_path": integrity_item["raw_result_path"],
           "raw_result_sha256": integrity_item["raw_result_sha256"],
           "path_success": raw.get("path_success"),
           "stability_success": raw.get("stability_success"),
           "stability_failure_observed": raw.get("stability_success") is False,
           "diagnostic_phase": None, "diagnostic": None,
           "eligible": False, "selected": False, "phase": None,
           "condition_hash": None, "reason": None}
    if not integrity_item["eligible"]:
        row["reason"] = integrity_item["reason"]
        return row
    if raw.get("path_success") is not True or raw.get("stability_success") is not False:
        row["reason"] = "not_observed_path_pass_stability_failure"
        return row
    if (raw.get("stability_check_calls") != 1
            or raw.get("stability_mode") != "one_exact_selected_support_configuration"):
        row["reason"] = "missing_actual_stability_checker"
        return row
    try:
        diagnostic = project_stability_diagnostics(raw.get("stability_diagnostics"))
    except (ValueError, TypeError, KeyError):
        row["reason"] = "invalid_stability_diagnostic"
        return row
    if diagnostic is None:
        row["reason"] = "missing_stability_diagnostic"
        return row
    row.update(diagnostic_phase=diagnostic["phase"], diagnostic=diagnostic)
    if failure_scope == "initial_disconnected":
        return _initial_contact_condition(attempt, physical, row, diagnostic)
    if diagnostic["phase"] != "dynamic_failure":
        row["reason"] = ("initial_disconnection_outside_dynamic_skill_scope"
                         if diagnostic["phase"] == "initial_disconnected"
                         else "non_dynamic_stability_failure")
        return row
    if (diagnostic["check_success_returned"] is not True
            or diagnostic["returned_success"] is not False
            or diagnostic["physics_steps"] <= 0
            or raw["stability_diagnostics"].get("instrumentation_errors")):
        row["reason"] = "incomplete_dynamic_observation"
        return row
    unstable = sorted({d["part_id"] for d in diagnostic["dynamic_failure_parts"]})
    possible = set(attempt["remaining_parts"]) - {choice["part_id"], context.get("base_part")}
    budget = context["max_grippers"] - 1
    row.update(observed_dynamic_parts=unstable, available_support_budget=budget)
    if not unstable or not set(unstable).issubset(possible):
        row["reason"] = "observed_dynamic_parts_not_current_support_candidates"
        return row
    if len(unstable) > budget:
        row["reason"] = "observed_dynamic_parts_exceed_declared_support_budget"
        return row
    condition = _condition({"request": {"value": physical}, "result": {"value": raw}}, "unused_alias")
    row.update(eligible=True, reason="measured_dynamic_failure_within_support_budget",
               condition_hash=digest(condition))
    return row


def select_conditions(attempts, considered, physical_requests, *, failure_scope="dynamic_failure"):
    """Keep the entire denominator; choose by chronology, never repair outcome."""
    _failure_scope(failure_scope)
    if len(attempts) != len(considered) or len(attempts) != len(physical_requests):
        raise ValueError("stability_selection_input_count_mismatch")
    rows, selected, seen = [], [], set()
    for attempt, checked, physical in zip(attempts, considered, physical_requests):
        row = classify_condition(attempt, checked, physical, failure_scope=failure_scope)
        if row["eligible"]:
            condition = row["condition_hash"]
            if condition in seen:
                row.update(eligible=False, reason="duplicate_physical_trigger_condition")
            else:
                seen.add(condition)
                if len(selected) < 3:
                    row.update(selected=True, phase="debug" if not selected else "admission")
                    selected.append(row)
                else:
                    row["reason"] = "eligible_beyond_predeclared_three_condition_limit"
        rows.append(row)
    return rows, selected


def prepare_stability_learning(episode_dir, output_dir, *, failure_scope="dynamic_failure", allow_test_root=False):
    _failure_scope(failure_scope)
    def safe(value, runs_only=True):
        return episode_preparation._path(value, allow_test_root=allow_test_root, runs_only=runs_only)

    episode, output = safe(episode_dir), safe(output_dir)
    if output.exists():
        raise FileExistsError("new_stability_learning_output_required")
    # Reuse the complete archived-source/raw/trajectory/Memory binding audit.
    # Its historical admission template is not used to select or execute our
    # conditions, and no candidate or physical result is created by this call.
    integrity = episode_preparation.prepare_episode_learning(
        episode, INTEGRITY_TEMPLATE, output / "episode_integrity", allow_test_root=allow_test_root)
    report_path = safe(episode / "planner/planner_report.json")
    report = episode_preparation._load(report_path)
    physicals = [episode_preparation._load(safe(report_path.parent / ("attempt_%06d" % i) / "request.json"))
                 for i in range(1, len(report["attempts"]) + 1)]
    # Preserve the original three-argument call for existing dynamic consumers.
    rows, selected = (select_conditions(report["attempts"], integrity["considered"], physicals)
        if failure_scope == "dynamic_failure" else
        select_conditions(report["attempts"], integrity["considered"], physicals, failure_scope=failure_scope))
    originals = integrity["original_artifact_sha256"]
    if any(episode_preparation._sha(safe(path, False)) != expected for path, expected in originals.items()):
        raise ValueError("stability_episode_changed_during_preparation")
    ready = len(selected) == 3
    manifest_path = output / "manifest.json" if ready else None
    reference = episode_preparation._project_reference(report_path, allow_test_root)
    initial_scope = failure_scope == "initial_disconnected"
    policy = INITIAL_POLICY if initial_scope else POLICY
    result = {"schema_version": 1,
        "status": "PREPARED" if ready else ("INSUFFICIENT_DISTINCT_INITIAL_CONTACT_FAILURES" if initial_scope
                                             else "INSUFFICIENT_DISTINCT_DYNAMIC_FAILURES"),
        "failure_scope": failure_scope,
        "selection_policy": policy, "selected": selected, "considered": rows,
        "selection_counts": {"all_attempts": len(rows),
            "all_stability_failures": sum(r["stability_failure_observed"] for r in rows),
            "initial_disconnection": sum(r["diagnostic_phase"] == "initial_disconnected" for r in rows),
            "dynamic_failure": sum(r["diagnostic_phase"] == "dynamic_failure" for r in rows),
            "eligible_distinct_conditions": sum(r["eligible"] for r in rows),
            "selected_conditions": len(selected)},
        "manifest_path": str(manifest_path) if manifest_path else None,
        "episode_dir": str(episode), "source_run_id": integrity["source_run_id"],
        "experience_task": integrity["experience_task"], "experience_report": reference,
        "memory_path": integrity["memory_path"], "memory_file_sha256": integrity["memory_file_sha256"],
        "memory_content_sha256": integrity["memory_content_sha256"],
        "integrity_audit": str(output / "episode_integrity/preparation.json"),
        "generic_integrity_manifest_used_for_experiment": False,
        "original_artifact_sha256": originals, "original_files_unchanged": True,
        "development_only": True, "target_is_held_out": False,
        "fresh_admission_results_available_to_synthesis": False,
        "new_api_calls": 0, "new_physical_calls": 0,
        "source_to_target_part_mapping": None, "hard_pruning": False,
        "performance_comparison": "NOT_RUN",
        "evidence_class": "saved_actual_asap_artifacts_checked_not_new_physics"}
    if ready:
        prefix = "initial_contact" if initial_scope else "stability"
        cases = [{"case_id": "%s_%s_%06d" % (prefix, r["phase"], r["attempt_index"]),
                  "phase": r["phase"], "source_task": integrity["experience_task"],
                  "report": reference, "attempt_index": r["attempt_index"],
                  "trigger_kind": "stability_failed"} for r in selected]
        if initial_scope:
            for case in cases:
                case["expected_diagnostic_phase"] = "initial_disconnected"
        memory_path = safe(integrity["memory_path"])
        manifest = {"schema_version": 1,
            "purpose": ("measured initial-contact development; not held-out evaluation" if initial_scope
                        else "measured dynamic-stability development; not held-out evaluation"),
            "experience_task": integrity["experience_task"], "experience_report": reference,
            "experience_memory": episode_preparation._project_reference(memory_path, allow_test_root),
            "frozen_episode_memory": {"path": str(memory_path),
                "file_sha256": integrity["memory_file_sha256"],
                "content_sha256": integrity["memory_content_sha256"]},
            "synthesis_failure_scope": "initial_disconnected" if initial_scope else "dynamic_stability_failed",
            "selection_rationale": policy, "cases": cases,
            "episode_learning_audit": str(output / "preparation.json"),
            "condition_hashes": {case["case_id"]: row["condition_hash"] for case, row in zip(cases, selected)},
            "scope_exclusions": (["dynamic_failure_is_not_initial_contact_disconnection",
                                  "missing_or_unbound_initial_contact_diagnostic"] if initial_scope else
                                 ["initial_contact_graph_disconnection_not_dynamic_failure",
                                  "more_observed_dynamic_parts_than_declared_support_budget"]),
            "development_inputs": [reference, episode_preparation._project_reference(memory_path, allow_test_root)]}
        episode_preparation._json_new(manifest_path, manifest)
    episode_preparation._json_new(output / "preparation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--failure-scope", choices=FAILURE_SCOPES, default="dynamic_failure")
    args = parser.parse_args(argv)
    result = prepare_stability_learning(args.episode_dir, args.output_dir, failure_scope=args.failure_scope)
    print(json.dumps({key: result[key] for key in ("status", "manifest_path", "selection_counts",
          "new_api_calls", "new_physical_calls", "original_files_unchanged")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
