"""One declared cross-assembly skill-reuse case with fresh actual physics.

The default is deterministic; --allow-fixture-library is an explicit wiring
test, never learned-skill evidence. Live selection requires an approved shared
config/ledger and --allow-live-api. This is NOT a full disassembly or performance
comparison. The old failure specifies only the trigger; recovery is not fixed.
"""
from contextlib import nullcontext
from copy import deepcopy
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tools"))

from run_skill_discovery import source_case, actual_action, preview_config
from run_integrated_repair import checked
from run_luna_experiment import read, write, sha, validate_config
from orchestrator.batch_budget import budget_summary
from orchestrator.explicit_action_memory import append_action_observation, freeze_memory
from orchestrator.explicit_action_provider_v5 import PROVIDER_CONTRACT_V5
from orchestrator.explicit_action_service import ExplicitActionService
from orchestrator.generated_skill_service import GeneratedSkillService
from orchestrator.skill_case import (physical_context, request_after_trigger, validate_trigger_scope,
                                    is_reproduced_trigger)
from orchestrator.skill_discovery import SkillDiscoveryService
from orchestrator.skill_registry import retrieve
from worker.explicit_action_contract import (PROTOCOL, action_id, digest, validate_request,
                                            validate_response)


DEFAULT_SOURCE = {
    "case_id": "07009_declared_transfer", "source_task": "07009", "phase": "transfer",
    "report": "runs/explicit_action_v2_live_20261018/07009/planner/planner_report.json",
    "attempt_index": 3, "trigger_kind": "path_failure",
}
CASE_SECONDS = 30


def _bounded_elapsed(raw):
    value = raw.get("elapsed_seconds")
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid_actual_elapsed_seconds")
    return value <= CASE_SECONDS


def _memory_result(raw, path):
    return {"status": raw["status"], "path_success": raw["path_success"],
            "stability_success": raw["stability_success"],
            "verifier_origin": raw["backend"], "raw": deepcopy(raw),
            "evidence_ref": str(path)}


def _observation_response(request, choice):
    selected = deepcopy(choice)
    selected["evidence_ids"] = []
    return validate_response(request, {"schema_version": 1, "protocol": PROTOCOL,
        "request_id": request["request_id"], "state_id": request["state_id"],
        "candidate_order": [selected["part_id"]] + [p for p in request["candidate_ids"]
                                                    if p != selected["part_id"]],
        "choice": selected, "action_id": action_id(request, selected), "fallback_reason": None,
        "metadata": {"selection_source": "declared_state_replay_trigger", "llm_live": False}})


def _summary(raw):
    return {key: deepcopy(raw.get(key)) for key in (
        "status", "backend", "chosen", "path_success", "stability_success", "elapsed_seconds",
        "path_check_calls", "stability_check_calls", "stability_mode")}


def _geometry_identity(hashes):
    """Conservative identity from immutable meshes/config, excluding SDF caches.

    This is evidence classification only, not an alignment of source part IDs.
    Unknown inputs never establish cross-assembly reuse. Equal mesh content
    with renamed files also cannot establish novel geometry.
    """
    if type(hashes) is not dict or not hashes:
        return None
    if any(type(k) is not str or type(v) is not str or re.fullmatch(r'[0-9a-f]{64}', v) is None
           for k, v in hashes.items()):
        return None
    meshes = {k.replace('\\', '/'): v for k, v in hashes.items() if k.lower().endswith('.obj')}
    if not meshes:
        return None
    configs = {k.replace('\\', '/'): v for k, v in hashes.items()
               if k.lower().endswith('.json') and Path(k).name.lower() != 'contact_graph.json'}
    return {'mesh_and_config_sha256': digest(dict(meshes=meshes, configs=configs)),
            'mesh_content_sha256': digest(sorted(meshes.values()))}


def classify_task_reuse(target_label, target_hashes, selected_rows):
    """Do not mistake a human task alias for an unseen geometry evaluation."""
    target = _geometry_identity(target_hashes)
    comparisons = []
    for row in selected_rows:
        for phase, scope in (('debug', row.get('development_scope', [])),
                             ('admission', row.get('validation_scope', []))):
            for case in scope:
                source = _geometry_identity(case.get('input_hashes'))
                same_label = case.get('source_task') == target_label
                same_geometry = (source is not None and target is not None
                    and (source['mesh_and_config_sha256'] == target['mesh_and_config_sha256']
                         or source['mesh_content_sha256'] == target['mesh_content_sha256']))
                relation = ('SAME_INPUT_OR_MESH' if same_geometry else
                            'SAME_LABEL_NOT_PROVEN_DIFFERENT' if same_label else
                            'DIFFERENT_GEOMETRY' if source is not None and target is not None else 'UNKNOWN')
                comparisons.append(dict(phase=phase, source_task=case.get('source_task'),
                    relation=relation, source_geometry=source))
    same = {'SAME_INPUT_OR_MESH', 'SAME_LABEL_NOT_PROVEN_DIFFERENT'}
    used = any(c['phase'] == 'debug' and c['relation'] in same for c in comparisons)
    seen = any(c['relation'] in same for c in comparisons)
    cross = bool(comparisons) and all(c['relation'] == 'DIFFERENT_GEOMETRY' for c in comparisons)
    return dict(cross_task_eligible=cross, target_used_in_development=used,
        target_used_for_skill_validation=seen, task_identity_relation=(
            'SAME_OR_PREVIOUSLY_USED' if seen else 'DIFFERENT_GEOMETRY' if cross else 'UNKNOWN'),
        task_identity_evidence=dict(target_geometry=target, comparisons=comparisons,
                                    part_id_mapping_inferred=False))


def load_case_spec(path=None):
    spec = read(checked(path, True)) if path is not None else deepcopy(DEFAULT_SOURCE)
    required = {'case_id', 'source_task', 'phase', 'report', 'attempt_index', 'trigger_kind'}
    if (type(spec) is not dict or not required.issubset(spec)
            or set(spec) - required - {'trigger_override', 'pose_yaw_degrees'}
            or spec.get('phase') != 'transfer'
            or not isinstance(spec.get('case_id'), str)
            or re.fullmatch(r'[A-Za-z0-9_]{1,100}', spec['case_id']) is None
            or not isinstance(spec.get('source_task'), str) or not spec['source_task']
            or spec.get('trigger_kind') not in ('path_failed', 'path_failure',
                                               'stability_failed', 'stability_failure')):
        raise ValueError('invalid_declared_transfer_case_spec')
    if 'pose_yaw_degrees' in spec and (type(spec['pose_yaw_degrees']) is not int
                                     or spec['pose_yaw_degrees'] not in (90, 180, 270)):
        raise ValueError('invalid_declared_pose_yaw')
    return spec


def run_transfer_case(prepared, snapshot, service, output, run_id, *, live=False,
                      config=None, allow_test_root=False):
    """Run only a fresh trigger and one freely selected recovery action.

    ``allow_test_root`` solely supports offline unit wiring tests. Main does not
    expose it and keeps every output inside the existing C-drive runs boundary.
    """
    template = deepcopy(prepared["request"])
    trigger_request = deepcopy(template)
    trigger_request.update(run_id=run_id, request_id=run_id + ":trigger",
        attempted_action_ids=[], attempted_actions=[], path_stage_observations=[],
        episode_feedback=[], memory_references=[], evidence_ids=[])
    trigger_request = validate_request(trigger_request)
    event_id = run_id + ":fresh-trigger"
    trigger_context = physical_context(prepared["context"], trigger_request,
                                      prepared["choice"], event_id=event_id, seconds=CASE_SECONDS)
    write(output / "trigger_context.json", trigger_context)
    raw_trigger = actual_action({"context": trigger_context, "choice": prepared["choice"]},
                                output / "trigger")
    validate_trigger_scope(trigger_request, raw_trigger, prepared["choice"])
    if raw_trigger.get("source_hashes") != prepared["expected_source_hashes"]:
        raise ValueError("fresh_trigger_source_hashes_changed")
    write(output / "trigger_request.json", trigger_request)
    trigger_response = _observation_response(trigger_request, prepared["choice"])
    write(output / "trigger_response.json", trigger_response)
    report = {"status": "TRIGGER_NOT_REPRODUCED", "workflow_completed": False,
              "case_type": "declared_state_replay_then_free_recovery",
              "target_task": prepared["spec"]["source_task"], "case_wall_seconds": CASE_SECONDS,
              "trigger": _summary(raw_trigger), "recovery": None,
              "generated_reuse_executed": False, "recovery_accepted": False,
              "cross_task_reuse_observed": False, "learned_skill_reuse_observed": False,
              "cross_task_reuse_pass": False, "learned_program_executed": False,
              "generated_source_tasks": [], "generated_development_tasks": [],
              "target_used_in_development": False,
              "llm_selection_used": False, "full_plan_test": "NOT_RUN",
              "performance_comparison": "NOT_RUN", "generalization_claim": False,
              "fixture_library": snapshot["fixture"], "library_mutated_by_runner": False,
              "physical_calls": 1, "path_check_calls": raw_trigger.get("path_check_calls"),
              "stability_check_calls": raw_trigger.get("stability_check_calls"),
              "memory_export": {"status": "NOT_RUN"}}
    observation_path = output / "observations.jsonl"
    trigger_record = None
    memory_errors = []
    try:
        trigger_record = append_action_observation(observation_path, trigger_request, trigger_response,
            _memory_result(raw_trigger, output / "trigger/physics/result.json"),
            allow_test_root=allow_test_root)
    except (ValueError, TypeError, OSError) as error:
        memory_errors.append({"stage": "trigger", "error_type": type(error).__name__})
    trigger_kind = prepared['spec'].get('trigger_kind', 'path_failed')
    trigger_reproduced = is_reproduced_trigger(raw_trigger, trigger_kind, CASE_SECONDS)
    report['declared_trigger_kind'] = trigger_kind
    report['trigger_reproduced'] = trigger_reproduced
    if trigger_reproduced:
        request = request_after_trigger(template, raw_trigger, prepared["choice"],
            run_id=run_id, event_id=event_id, wall_seconds=CASE_SECONDS)
        write(output / "program_request.json", request)
        if live:
            from local_api_key import key_environment
            scope = key_environment()
        else:
            scope = nullcontext()
        # Credentials are scoped to API selection only, not inherited by a
        # physics worker or serialized into the immutable request artifacts.
        with scope:
            if live:
                validate_config(config, True)
            response = service.handle(request)
        response = validate_response(request, response)
        write(output / "response.json", response)
        write(output / "selector_decisions.json", service.selector_decisions)
        report["llm_selection_used"] = response["metadata"].get("llm_live") is True
        if response["choice"] is None:
            report.update(status="NO_RECOVERY_ACTION", workflow_completed=True)
        else:
            recovery_context = physical_context(prepared["context"], request, response["choice"],
                event_id=run_id + ":fresh-recovery", seconds=CASE_SECONDS)
            write(output / "recovery_context.json", recovery_context)
            raw_recovery = actual_action({"context": recovery_context, "choice": response["choice"]},
                                         output / "recovery")
            validate_trigger_scope(request, raw_recovery, response["choice"])
            if raw_recovery.get("source_hashes") != prepared["expected_source_hashes"]:
                raise ValueError("fresh_recovery_source_hashes_changed")
            matches = raw_recovery["chosen"] == response["choice"]
            generated = response["metadata"].get("generated_program_executed") is True and matches
            accepted = (raw_recovery["status"] == "PASS" and raw_recovery["path_success"] is True
                        and raw_recovery["stability_success"] is True and _bounded_elapsed(raw_recovery))
            rows = retrieve(snapshot, request["task_context"], allow_fixture=snapshot["fixture"])
            selected_id = response["metadata"].get("generated_skill_id")
            selected_rows = [row for row in rows if row["strategy_id"] == selected_id]
            generated = generated and len(selected_rows) == 1
            development_tasks = sorted({case["source_task"] for row in selected_rows
                                        for case in row.get("development_scope", []) if "source_task" in case})
            source_tasks = sorted(set(development_tasks) | {case["source_task"] for row in selected_rows
                                   for case in row["validation_scope"] if "source_task" in case})
            target = prepared["spec"]["source_task"]
            identity = classify_task_reuse(target, raw_trigger.get('input_hashes'), selected_rows)
            cross = generated and identity.pop('cross_task_eligible')
            report.update(identity)
            report.update(status="RECOVERY_PASS" if accepted else "RECOVERY_NOT_ACCEPTED",
                workflow_completed=True, recovery=_summary(raw_recovery), physical_calls=2,
                generated_reuse_executed=generated, recovery_accepted=accepted,
                generated_skill_id=selected_id, generated_source_tasks=source_tasks,
                generated_development_tasks=development_tasks,
                learned_program_executed=bool(generated and not snapshot["fixture"]),
                cross_task_reuse_observed=cross,
                cross_task_reuse_pass=bool(cross and accepted),
                learned_skill_reuse_observed=bool(cross and not snapshot["fixture"]),
                generated_reuse_pass=bool(generated and accepted),
                generated_selector_status=(response["metadata"].get("generated_selector_response") or {}).get("status"),
                generated_skill_fallback_reason=response["metadata"].get("generated_skill_fallback_reason"))
            for count_key in ("path_check_calls", "stability_check_calls"):
                counts = (raw_trigger.get(count_key), raw_recovery.get(count_key))
                report[count_key] = sum(counts) if all(type(value) is int for value in counts) else None
            try:
                parent_id = (trigger_record["evidence_id"] if trigger_record is not None
                    and trigger_record["result"]["status"] in ("UNKNOWN", "REJECTED")
                    and response["choice"]["part_id"] == prepared["choice"]["part_id"]
                    and response["action_id"] != trigger_response["action_id"] else None)
                append_action_observation(observation_path, request, response,
                    _memory_result(raw_recovery, output / "recovery/physics/result.json"),
                    parent_failure_evidence_id=parent_id, allow_test_root=allow_test_root)
            except (ValueError, TypeError, OSError) as error:
                memory_errors.append({"stage": "recovery", "error_type": type(error).__name__})
    else:
        report["workflow_completed"] = True
    if observation_path.exists():
        try:
            memory = freeze_memory(observation_path, output / "memory_frozen.json",
                                   allow_test_root=allow_test_root)
            report["memory_export"] = {"status": "PARTIAL" if memory_errors else "FROZEN",
                "record_count": memory["record_count"], "content_sha256": memory["content_sha256"],
                "hard_pruning": False, "physics_reuse_allowed": False}
        except (ValueError, TypeError, OSError) as error:
            memory_errors.append({"stage": "freeze", "error_type": type(error).__name__})
    report["memory_export"]["errors"] = memory_errors
    write(output / "case_report.json", report)
    return report


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-dir", required=True, type=Path)
    value.add_argument("--skill-library", required=True, type=Path)
    value.add_argument("--mode", choices=("deterministic", "llm"), default="deterministic")
    value.add_argument("--allow-fixture-library", action="store_true")
    value.add_argument("--llm-config", type=Path)
    value.add_argument("--allow-live-api", action="store_true")
    value.add_argument("--ledger", type=Path)
    value.add_argument("--case-spec", type=Path,
        help="Predeclared transfer case JSON; default preserves the original path-failure case")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    live = args.mode == "llm"
    if live != args.allow_live_api or (live and (not args.llm_config or not args.ledger)):
        raise ValueError("live_mode_requires_explicit_approved_config_and_shared_ledger")
    if live and args.allow_fixture_library:
        raise ValueError("fixture_library_for_offline_wiring_only")
    output = checked(args.run_dir, True)
    library = checked(args.skill_library, True)
    ledger = checked(args.ledger, True) if args.ledger else None
    if output.exists():
        raise FileExistsError("new_run_required")
    config = read(checked(args.llm_config)) if args.llm_config else preview_config()
    if not live and (config.get("enabled") or config.get("max_calls") or config.get("max_usd")):
        raise ValueError("deterministic_mode_requires_zero_api_authorization")
    snapshot = read(library)
    rows = retrieve(snapshot, allow_fixture=args.allow_fixture_library)
    if not rows:
        raise ValueError("registered_skill_library_is_empty")
    library_before = sha(library)
    spec = load_case_spec(args.case_spec)
    spec_before = sha(checked(args.case_spec, True)) if args.case_spec else None
    prepared = source_case(spec)
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT / "orchestrator").glob("*.py")) + sorted((ROOT / "worker").glob("*.py"))
    sources += [Path(__file__), ROOT / "tools/run_skill_discovery.py", ROOT / "tools/run_recorded.py",
                ROOT / "tools/run_integrated_repair.py", ROOT / "tools/run_luna_experiment.py"]
    before = {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}
    write(output / "source_before.json", before)
    for name in before:
        target = output / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked(ROOT / name), checked(target, True))
    write(output / "precommitted_case.json", {"spec": spec,
        "case_spec_path": str(args.case_spec) if args.case_spec else None,
        "case_spec_file_sha256": spec_before,
        "source_report_sha256": prepared["source_report_sha256"],
        "library_path": str(library), "library_file_sha256": library_before,
        "source_hashes": prepared["expected_source_hashes"], "case_seconds": CASE_SECONDS,
        "full_plan_test": "NOT_RUN", "performance_comparison": "NOT_RUN"})
    write(output / "config.json", {"mode": args.mode, "api": config, "ledger": str(ledger) if ledger else None,
        "fixture_library_explicit": args.allow_fixture_library, "automatic_retries": 0})
    write(output / "budget_before.json", budget_summary(ledger) if ledger and ledger.exists() else
          {"reserved_calls": 0, "budget_committed_usd": "0", "usage_cost_usd": "0"})
    service = None
    try:
        selector = SkillDiscoveryService(config, ledger) if live else None
        fallback = ExplicitActionService(config, ledger or output / "unused_offline_ledger.sqlite",
            mode=args.mode, timeout_seconds=60, provider_contract=PROVIDER_CONTRACT_V5,
            fallback_policy="path_stage_v1")
        service = GeneratedSkillService(snapshot, fallback, selector, mode=args.mode,
            allow_fixture=args.allow_fixture_library, timeout_seconds=60)
        run_id = "generated-transfer:" + digest(str(output))[:20]
        report = run_transfer_case(prepared, snapshot, service, output, run_id,
                                   live=live, config=config)
        print(json.dumps({"status": report["status"], "generated_reuse_executed": report["generated_reuse_executed"],
                          "recovery_accepted": report["recovery_accepted"],
                          "memory_export": report["memory_export"]["status"]}), flush=True)
        return 0
    except Exception as error:
        diagnostic = str(error)
        diagnostic = diagnostic if re.fullmatch(r"[a-z0-9_]{1,120}", diagnostic) else None
        write(output / "case_error.json", {"status": "ERROR", "error_type": type(error).__name__,
            "diagnostic": diagnostic,
            "workflow_completed": False, "full_plan_test": "NOT_RUN", "performance_comparison": "NOT_RUN"})
        return 1
    finally:
        if service is not None:
            service.close()
        write(output / "budget_after.json", budget_summary(ledger) if ledger and ledger.exists() else
              {"reserved_calls": 0, "budget_committed_usd": "0", "usage_cost_usd": "0"})
        write(output / "preservation.json", {"library_unchanged": sha(library) == library_before,
            "case_spec_unchanged": (args.case_spec is None or sha(checked(args.case_spec, True)) == spec_before),
            "source_unchanged": all(sha(ROOT / name) == value for name, value in before.items()),
            "baseline_and_previous_runs_modified": False})


if __name__ == "__main__":
    raise SystemExit(main())
