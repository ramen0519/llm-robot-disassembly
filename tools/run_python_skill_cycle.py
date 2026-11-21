"""Experience -> Python -> fresh ASAP cases -> scoped registry -> frozen reuse.

Preview is API/physics-free. Fixture uses explicitly human-authored Python and
real ASAP when available. Live requires a new approved config and shared ledger.
No automatic HTTP retries, no automatic paid revision, no ACTIVE library update.
"""
import argparse
from copy import deepcopy
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from run_skill_discovery import source_case, synthesis_packet, run_case, preview_config
from run_integrated_repair import checked
from run_luna_experiment import read, write, sha
from orchestrator.llm_ranker import DEFAULTS
from orchestrator.python_skill_discovery import PythonSkillDiscoveryService
from orchestrator.python_skill_program import API_DESCRIPTION
from orchestrator.python_skill_diagnostics import save_rejection_artifacts, validate_record
from orchestrator.initial_contact_trace_summary import summarize_initial_contact_transitions
from orchestrator.skill_program import program_hash, step_program
from orchestrator.skill_registry import SkillRegistry
from worker.task_memory import digest
from python_skill_fixture import PROGRAM


def require_live_config(config, service):
    """Check the explicit learner config without modifying legacy Luna runners."""
    if (type(config) is not dict or set(config) != set(DEFAULTS)
            or config["api_key_env"] != "OPENAI_API_KEY"):
        raise ValueError("complete_python_config_without_inline_secrets_required")
    reason, _ = service._configuration()
    if reason is not None:
        raise ValueError("python_live_config_rejected:" + reason)


def python_generation_packet(manifest, run_id):
    """Opt-in stability synthesis with bound evidence, not legacy IR semantics."""
    if 'experience_sources' in manifest:
        from tools.multitask_skill_evidence import build_packet
        from tools.support_experience import attach_support_experiences
        return attach_support_experiences(build_packet(manifest, run_id), manifest)
    scope = manifest.get("python_failure_scope")
    if scope not in (None, "remaining_body_stability_failed"):
        raise ValueError("unsupported_python_failure_scope")
    if scope and (manifest.get("synthesis_failure_scope") or not manifest.get("experience_memory")):
        raise ValueError("python_stability_requires_bound_memory_without_ir_scope")
    packet = synthesis_packet(manifest, run_id)
    packet["context"]["generated_language"] = "python_policy_v1"
    if scope:
        measured = [item["evidence_id"] for item in packet["evidence"]
            if item.get("physically_checked_stability_failure") is True
            and item.get("path_success") is True and item.get("stability_success") is False
            and item.get("deadline_exceeded") is False
            and (item.get("stability_diagnostics") or {}).get("phase")
                in ("dynamic_failure", "initial_disconnected")]
        if not measured:
            raise ValueError("python_stability_requires_measured_failure")
        packet["context"].update(python_failure_scope=scope,
            measured_stability_evidence_ids=measured,
            observations_are_not_causes=True, admission_results_exposed=False)
        # The preceding synthesis binding checked this development report against
        # its frozen memory. Project observed mover/support transitions only;
        # admission reports and their outcomes never enter the generation packet.
        packet["context"]["initial_contact_observed_transitions"] = (
            summarize_initial_contact_transitions(
                read(checked(ROOT / manifest["experience_report"], True))))
    from tools.support_experience import attach_support_experiences
    return attach_support_experiences(packet, manifest)


def refinement_packet(packet, prior, manifest, *, live=False, ledger=None):
    """Bind actual saved DEBUG feedback; never send admission results to model."""
    prior = checked(prior, True)
    old_config, old_result = read(prior / "config.json"), read(prior / "result.json")
    if read(prior / "precommitted_cases.json") != manifest:
        raise ValueError("refinement_predeclared_conditions_changed")
    if live and (not old_result.get("actual_llm_generation")
                 or old_config.get("ledger") != str(ledger)):
        raise ValueError("live_refinement_requires_same_approved_ledger_and_live_parent")
    registry = SkillRegistry(prior / "registry", fixture=not old_result["actual_llm_generation"])
    candidate = registry.candidate(old_result["candidate_id"])
    feedback = []
    if old_result.get('status') == 'PREFLIGHT_REJECTED':
        from orchestrator.python_skill_preflight import audit_program, debug_rows, preflight_rows
        saved = read(prior / 'generation_packet.json')
        from orchestrator.python_skill_revision import enabled, audit_revision
        audit = (audit_revision(candidate['program'], saved, read(prior / 'provider_proposal.json').get('behavior_checks'))
                 if enabled(saved) else audit_program(candidate['program'], preflight_rows(saved)))
        if (digest(audit) != old_result.get('preflight_sha256')
                or read(prior / 'development_preflight.json') != audit):
            raise ValueError('preflight_feedback_binding_mismatch')
        result = deepcopy(packet)
        result.update(operation='refine', previous_program=candidate['program'],
                      validation_feedback=deepcopy(debug_rows(saved)))
        result['context']['development_preflight_feedback'] = audit
        if old_result.get('support_preflight_sha256'):
            from orchestrator.support_learning import audit_support_program
            support_audit = audit_support_program(candidate['program'], saved)
            if (digest(support_audit) != old_result['support_preflight_sha256']
                    or read(prior/'support_development_preflight.json') != support_audit):
                raise ValueError('support_preflight_feedback_binding_mismatch')
            result['context']['support_development_preflight'] = support_audit
        if saved.get('context', {}).get('python_applicability_requests'):
            result['context']['python_applicability_requests'] = deepcopy(saved['context']['python_applicability_requests'])
        return result
    for record in registry.validations(candidate["candidate_id"]):
        if record["case"]["phase"] != "debug":
            continue
        from orchestrator.recovery_sequence import debug_steps
        for index, transition in enumerate(debug_steps(record)):
            artifacts = transition['artifacts']
            selection = read(artifacts['selection_path'])
            observed = dict(case_id=record['case']['case_id'] + ('_step_%02d' % (index+1) if index else ''),
                phase='debug', outcome=transition['outcome'],
                validation_record_sha256=transition['record_sha256'],
                current_observation_request=read(selection['program_request_path']),
                original_trigger=read(Path(artifacts['trigger_attempt_dir'])/'result.json'), recovery=None)
            if artifacts['attempt_dir']:
                observed['recovery'] = read(Path(artifacts['attempt_dir'])/'result.json')
            feedback.append(observed)
    if not feedback:
        raise ValueError("refinement_requires_bound_debug_results")
    result = deepcopy(packet)
    result.update(operation="refine", previous_program=candidate["program"], validation_feedback=feedback)
    if candidate.get('card', {}).get('runtime_contract') == 'lesson_runtime_v2':
        from orchestrator.python_skill_preflight import audit_program
        result['context']['development_preflight_feedback'] = audit_program(candidate['program'], feedback)
    return result


def attach_debug_inputs(packet, prepared, run_id, *, recorded_clock=False, source_observation=False):
    """Bind existing source observations. No admission data or new simulation."""
    from orchestrator.skill_case import request_after_trigger
    rows = []
    for item in prepared:
        spec = item['spec']
        if spec['phase'] != 'debug':
            continue
        event = run_id + ':' + spec['case_id'] + ':saved_trigger'
        binding = None
        if source_observation:
            from tools.python_source_budget import observation_request
            request, binding = observation_request(item, event, event)
        elif recorded_clock:
            from tools.python_recorded_clock import replay_request
            request=replay_request(item,event,event)
        else:
            request = request_after_trigger(item['request'], item['raw'], item['choice'],
                                           run_id=event, event_id=event)
        rows.append(dict(case_id=spec['case_id'], phase='debug',
            origin='saved_observation_replay_NOT_new_physics',
            current_observation_request=request, original_trigger=deepcopy(item['raw']), recovery=None))
        if binding is not None:
            rows[-1]['observation_binding'] = binding
    result = deepcopy(packet)
    result['context']['python_debug_requests'] = rows
    return result


def attach_applicability_inputs(packet, manifest, run_id, *, source_observation=False):
    """Saved development inputs only: no admission outcomes or physics calls."""
    specs = manifest.get('python_applicability_cases', [])
    if not specs:
        return packet
    if type(specs) is not list or not 1 <= len(specs) <= 4:
        raise ValueError('bounded_applicability_cases_required')
    from orchestrator.skill_case import request_after_trigger
    required = {'case_id', 'phase', 'source_task', 'trigger_kind', 'report',
                'attempt_index', 'report_sha256', 'purpose'}
    admissions = {(c['report'], c['attempt_index']) for c in manifest['cases']
                  if c['phase'] == 'admission'}
    rows = []
    for spec in specs:
        if (set(spec) != required or spec['phase'] != 'debug'
                or spec['purpose'] != 'development_applicability_NOT_evaluation'
                or (spec['report'], spec['attempt_index']) in admissions):
            raise ValueError('applicability_must_be_declared_development_not_admission')
        if sha(checked(ROOT / spec['report'], True)) != spec['report_sha256']:
            raise ValueError('applicability_source_hash_mismatch')
        item = source_case(spec)
        raw = item['raw']
        if (raw.get('status') not in ('UNKNOWN', 'REJECTED')
                or raw.get('deadline_exceeded') is True
                or not (raw.get('path_success') is False or
                        (raw.get('path_success') is True and raw.get('stability_success') is False))):
            raise ValueError('applicability_requires_observed_failure')
        event = run_id + ':' + spec['case_id'] + ':saved_applicability'
        binding = None
        if source_observation:
            from tools.python_source_budget import observation_request
            request, binding = observation_request(item, event, event)
        else:
            request = request_after_trigger(item['request'], raw, item['choice'],
                                           run_id=event, event_id=event)
        rows.append(dict(case_id=spec['case_id'], phase='debug', purpose=spec['purpose'],
            origin='saved_development_input_NOT_physics_or_heldout_evaluation',
            source_report_sha256=spec['report_sha256'],
            current_observation_request=request, original_trigger=deepcopy(raw), recovery=None))
        if binding is not None:
            rows[-1]['observation_binding'] = binding
    result = deepcopy(packet)
    result['context']['python_applicability_requests'] = rows
    from orchestrator.python_skill_preflight import preflight_rows
    preflight_rows(result)  # Check count/identity before any model request.
    return result


def development_parent_packet(packet, manifest, *, live, ledger):
    """Explicit new development split; historical admissions are never held-out again."""
    declaration = manifest["development_parent"]
    prior = checked(ROOT / declaration["run_dir"], True)
    if (declaration.get("former_admission_is_now_development") is not True
            or declaration.get("previous_validation_is_not_new_admission") is not True
            or sha(prior / "result.json") != declaration["result_sha256"]):
        raise ValueError("development_reclassification_declaration_required")
    config, result = read(prior / "config.json"), read(prior / "result.json")
    if not live or not result.get("actual_llm_generation") or config.get("ledger") != str(ledger):
        raise ValueError("development_parent_requires_same_live_ledger")
    old_manifest = read(prior / "precommitted_cases.json")
    selected = declaration["cases"]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("invalid_development_parent_cases")
    signature = lambda c: digest({k: c.get(k) for k in
        ("report", "attempt_index", "pose_yaw_degrees", "trigger_override")})
    old_cases = {c["case_id"]: c for c in old_manifest["cases"]}
    source_conditions = {signature(old_cases[c]) for c in selected}
    debug_conditions = {signature(c) for c in manifest["cases"] if c["phase"] == "debug"}
    admission_conditions = {signature(c) for c in manifest["cases"] if c["phase"] == "admission"}
    if source_conditions != debug_conditions or source_conditions & admission_conditions:
        raise ValueError("development_and_new_admission_conditions_must_be_separate")
    registry = SkillRegistry(prior / "registry", fixture=False)
    candidate = registry.candidate(result["candidate_id"])
    records = {r["case"]["case_id"]: r for r in registry.validations(result["candidate_id"])}
    feedback = []
    for identity in selected:
        record = records[identity]  # validations() rechecks hashes and actual artifacts
        selection = read(record["artifacts"]["selection_path"])
        feedback.append(dict(case_id=identity, phase="debug",
            previous_phase=record["case"]["phase"], explicitly_reclassified_development=True,
            outcome=record["outcome"], validation_record_sha256=record["record_sha256"],
            current_observation_request=read(selection["program_request_path"]),
            original_trigger=read(Path(record["artifacts"]["trigger_attempt_dir"]) / "result.json"),
            recovery=(read(Path(record["artifacts"]["attempt_dir"]) / "result.json")
                      if record["artifacts"]["attempt_dir"] else None)))
    value = deepcopy(packet)
    value.update(operation="refine", previous_program=candidate["program"], validation_feedback=feedback)
    value["context"]["development_reclassification"] = deepcopy(declaration)
    return value


def contract_rejection_feedback(prior, ledger):
    """A new budgeted request may carry inert, sanitized rejection evidence."""
    prior = checked(prior, True)
    config = read(prior / "config.json")
    proposal = read(prior / "provider_proposal.json")
    metadata = proposal.get("metadata", {})
    result = read(prior / "result.json") if (prior / "result.json").is_file() else {}
    if (config.get("ledger") == str(ledger) and config.get("mode") == "live"
            and metadata.get("llm_live") is True and proposal.get("status") == "proposed"
            and result.get("status") == "STABILITY_POSE_CONTRACT_REJECTED"
            and result.get("physical_cases_not_run") is True):
        audit = read(prior / "stability_coordinate_check.json")
        checks = audit.get("checks", [audit])
        return dict(kind="prior_synthetic_coordinate_rejection_NOT_PHYSICS",
            source_response_sha256=sha(prior / "provider_proposal.json"),
            coordinate_check_sha256=sha(prior / "stability_coordinate_check.json"),
            prior_program_hash=program_hash(proposal["program"]),
            checked=sum(c["checked"] for c in checks), passed=sum(c["passed"] for c in checks),
            physical_calls=0, admission_feedback=False,
            instruction="Preserve primitive and BOTH directions only in the capacity-sufficient SAME-mover branch. In the capacity-insufficient mover-switch branch retain single_force_v1 with second_direction_id null and the transformed source CAD axis. Do not carry a failed old two-leg primitive into that new-mover branch.")
    allowed_rejections = {"invalid_generated_python", "invalid_card_or_evidence", "invalid_response"}
    packet_path = prior / 'generation_packet.json'
    if packet_path.is_file():
        from orchestrator.python_skill_revision import enabled
        if enabled(read(packet_path)):
            allowed_rejections.update(('invalid_revision_predictions', 'invalid_response_fields'))
    if (config.get("ledger") != str(ledger) or config.get("mode") != "live"
            or proposal.get("status") != "rejected" or metadata.get("llm_live") is not True
            or proposal.get("fallback_reason") not in allowed_rejections):
        raise ValueError("bound_live_contract_rejection_required")
    code = metadata.get("python_validation_error")
    if type(code) is not str or not re.fullmatch(r"python_[a-z_]+(?::[A-Za-z]+)?", code):
        code = None
    feedback = dict(kind="prior_generation_contract_rejection",
        fallback_reason=proposal["fallback_reason"], python_validation_error=code,
        source_response_sha256=sha(prior / "provider_proposal.json"),
        instruction="Correct this contract violation in the NEW candidate; no prior code was executed.")
    record = metadata.get("python_rejected_source")
    if record is not None:
        # Bound to the existing same-ledger response, not an arbitrary source file.
        feedback["rejected_source_diagnostic"] = deepcopy(validate_record(record))
    return feedback


def registered_parent_packet(packet, manifest):
    """Read a previously admitted version and fresh DEBUG replay, independent of budget.

    This copies evidence, never authorization. The new run still requires its own
    approved config/ledger and allow-live-api. Historical admissions stay excluded.
    """
    from orchestrator.skill_registry import retrieve
    from orchestrator.skill_validation import check_evidence
    declaration = manifest["registered_parent"]
    prior = checked(ROOT / declaration["run_dir"], True)
    if sha(prior / "result.json") != declaration["result_sha256"]:
        raise ValueError("registered_parent_result_hash_mismatch")
    result = read(prior / "result.json")
    entries = retrieve(read(prior / "skills_frozen.json"))
    entry = next((e for e in entries if e["program_hash"] == result.get("candidate_id")), None)
    if not entry or not result.get("actual_llm_generation") or not result.get("admission", {}).get("admitted"):
        raise ValueError("actual_registered_parent_required")
    registry = SkillRegistry(prior / "registry", fixture=False)
    records = {r["case"]["case_id"]: r for r in registry.validations(entry["program_hash"])}
    selected = declaration["cases"]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("invalid_registered_parent_debug_cases")
    chosen = [records[c] for c in selected]
    if any(r["case"]["phase"] != "debug" for r in chosen):
        raise ValueError("registered_parent_admission_not_generation_input")
    signature = lambda c: digest({k: c.get(k) for k in
        ("report", "attempt_index", "pose_yaw_degrees", "trigger_override")})
    old_cases = {c["case_id"]: c for c in read(prior / "precommitted_cases.json")["cases"]}
    old_conditions = {signature(old_cases[c]) for c in selected}
    replay_path = checked(ROOT / declaration["replay_record"], True)
    replay = check_evidence(read(replay_path))
    replay_protocol = read(replay_path.parent / "protocol.json")
    if replay["case"]["program_hash"] != entry["program_hash"] or replay["case"]["phase"] != "debug":
        raise ValueError("registered_parent_replay_program_mismatch")
    declared_debug = {signature(c) for c in manifest["cases"] if c["phase"] == "debug"}
    declared_admission = {signature(c) for c in manifest["cases"] if c["phase"] == "admission"}
    bound_conditions = old_conditions | {signature(replay_protocol["spec"])}
    if bound_conditions != declared_debug or bound_conditions & declared_admission:
        raise ValueError("registered_parent_debug_split_mismatch")
    chosen.append(replay)
    feedback = []
    for record in chosen:
        selection = read(record["artifacts"]["selection_path"])
        feedback.append(dict(case_id=record["case"]["case_id"], phase="debug",
            outcome=record["outcome"], validation_record_sha256=record["record_sha256"],
            current_observation_request=read(selection["program_request_path"]),
            original_trigger=read(Path(record["artifacts"]["trigger_attempt_dir"]) / "result.json"),
            recovery=(read(Path(record["artifacts"]["attempt_dir"]) / "result.json")
                      if record["artifacts"]["attempt_dir"] else None)))
    value = deepcopy(packet)
    value.update(operation="refine", previous_program=entry["program"], validation_feedback=feedback)
    value["context"]["registered_parent"] = dict(program_hash=entry["program_hash"],
        frozen_library_sha256=sha(prior / "skills_frozen.json"), authorization_inherited=False,
        historical_admission_results_exposed=False, development_replay_record_sha256=replay["record_sha256"])
    return value


def add_development_probes(packet, directory):
    """Bind all human-exploration observations to already declared DEBUG triggers."""
    from orchestrator.skill_validation import check_evidence, _condition
    directory = checked(directory, True)
    protocol = read(directory / "protocol.json")
    if (protocol.get("generated_skill_origin") is not False
            or protocol.get("stop_on_success") is not False
            or protocol.get("registration") is not False
            or protocol.get("probe_indices") != [0, 1, 2]):
        raise ValueError("explicit_complete_development_probe_protocol_required")
    allowed = set()
    for row in packet["validation_feedback"]:
        if row["phase"] != "debug":
            raise ValueError("probe_parent_requires_debug_only")
        raw, req = row["original_trigger"], row["current_observation_request"]
        allowed.add(digest(_condition({"request": {"value": {"choice": raw["chosen"], "context": req}},
                                      "result": {"value": raw}}, "07009")))
    result = deepcopy(packet)
    for index in protocol["probe_indices"]:
        record = check_evidence(read(directory / ("record_%d.json" % index)))
        if record["case"]["phase"] != "debug" or record["condition_hash"] not in allowed:
            raise ValueError("probe_not_from_declared_debug_condition")
        artifacts = record["artifacts"]
        selected = read(artifacts["selection_path"])
        raw = read(Path(artifacts["attempt_dir"]) / "result.json") if artifacts["attempt_dir"] else None
        result["validation_feedback"].append(dict(case_id="development_probe_%d" % index,
            phase="debug", outcome=record["outcome"], validation_record_sha256=record["record_sha256"],
            origin="human exploration operator; actual ASAP physics; NOT LLM-generated skill",
            retain_successful_probe=record["outcome"] == "PASS",
            current_observation_request=read(selected["program_request_path"]),
            original_trigger=read(Path(artifacts["trigger_attempt_dir"]) / "result.json"), recovery=raw))
        result["evidence"].append(dict(evidence_id="development-probe:" + record["record_sha256"],
            source_task="07009", status=record["outcome"], choice=raw.get("chosen") if raw else None,
            path_success=raw.get("path_success") if raw else None,
            stability_success=raw.get("stability_success") if raw else None,
            origin="human bounded support exploration, actual verifier",
            reference_only=True, source_part_ids_are_not_target_ids=True, physics_reuse_allowed=False))
    result["context"]["additional_development_probe_origin"] = protocol
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/skill_discovery_cases_v1.json")
    parser.add_argument("--mode", choices=("preview", "fixture", "live"), default="preview")
    parser.add_argument("--llm-config", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument('--recorded-trigger-clock', action='store_true',
                        help='V2 roles: retain saved observation clock; fresh validation uses existing 45s harness cap.')
    parser.add_argument('--source-case-clock', action='store_true',
                        help='Opt-in fresh validation with hash-bound original allowance including initialization.')
    parser.add_argument('--observation-contract', choices=('legacy_v1', 'source_bound_v1'), default='legacy_v1',
                        help='Source-bound historical feedback and fresh-case budget for new learning runs.')
    parser.add_argument('--replay-generated-from', type=Path,
                        help='Revalidate an unchanged actual prior candidate without another generation call.')
    parser.add_argument("--lesson", type=Path, help="Opt-in evidence-grounded lesson; legacy generation remains unchanged.")
    parser.add_argument("--lesson-only", action="store_true", help="Independent code reconstruction; exclude original solution and trace.")
    parser.add_argument('--python-revision-contract', action='store_true',
                        help='Opt-in revision checks: v1 predictions, v2 executed action, v3 direction dependence.')
    parser.add_argument('--python-revision-version', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--generation-timeout-seconds', type=int, choices=(30, 60, 90, 120, 600),
                        help='Explicit generation response wait; no retry and no change to physics timeouts.')
    parser.add_argument('--prior-library', type=Path)
    parser.add_argument('--prior-lessons', type=Path)
    parser.add_argument("--hierarchy-policy", type=Path)
    parser.add_argument("--hierarchy-ledger", type=Path)
    parser.add_argument('--transport-failure-policy', choices=('legacy_v1', 'latch_on_error_v1'), default='legacy_v1')
    parser.add_argument("--refine-from", type=Path,
                        help="Explicit new candidate from saved DEBUG feedback; not an HTTP retry.")
    parser.add_argument('--historical-debug-parent', type=Path,
                        help='Explicit hash-bound old debug evidence; never inherits old API budget.')
    parser.add_argument('--abstention-feedback', action='store_true',
                        help='Opt-in bounded return/condition/value evidence from verified old preflight, no physics.')
    parser.add_argument("--generation-feedback-from", type=Path,
                        help="Include a bound prior generation rejection in this explicit new request.")
    parser.add_argument("--require-pose-equivariance", action="store_true",
                        help="Require source/current-frame contract before fresh physics and registration.")
    parser.add_argument("--development-probes", type=Path,
                        help="Explicit complete probes on existing DEBUG conditions; never admission.")
    parser.add_argument("--generation-output-tokens", type=int, choices=(1024, 2048, 4096, 25000),
                        help="Explicit request ceiling; actual ceiling is reserved in the same global budget.")
    args = parser.parse_args(argv)
    if args.python_revision_version != 1 and not args.python_revision_contract:
        raise ValueError('revision_version_requires_explicit_contract')
    if args.python_revision_contract and (not args.lesson or args.lesson_only or args.mode == 'fixture'):
        raise ValueError('revision_contract_requires_regular_lesson_preview_or_live')
    if args.historical_debug_parent and (args.refine_from or not args.python_revision_contract):
        raise ValueError('historical_parent_requires_revision_contract_without_refine_from')
    if args.abstention_feedback and (not args.python_revision_contract or args.python_revision_version != 2
                                    or not (args.refine_from or args.historical_debug_parent)):
        raise ValueError('abstention_feedback_requires_bound_parent_and_revision_v2')
    live = args.mode == "live"
    if live != args.allow_live_api or (live and (not args.llm_config or not args.ledger)):
        raise ValueError("explicit_approved_config_and_shared_ledger_required")
    output = checked(args.run_dir, True)
    if output.exists():
        raise FileExistsError("new_run_required")
    config = read(checked(args.llm_config)) if args.llm_config else preview_config()
    if not args.llm_config:
        config["max_output_tokens"] = 4096
    if not live and (config.get("enabled") or config.get("max_calls") or config.get("max_usd")):
        raise ValueError("offline_requires_zero_budget")
    manifest = read(checked(args.manifest))
    source_observation = args.observation_contract == 'source_bound_v1'
    if source_observation:
        if not args.lesson or args.lesson_only:
            raise ValueError('source_observation_requires_regular_lesson_runtime')
        args.recorded_trigger_clock = args.source_case_clock = True
    if args.source_case_clock and not args.recorded_trigger_clock:
        raise ValueError('source_clock_requires_recorded_observation_clock')
    recovery_protocol = manifest.get('recovery_protocol')
    if recovery_protocol is not None:
        from orchestrator.recovery_sequence import validate_protocol
        recovery_protocol = validate_protocol(recovery_protocol)
        if not source_observation or args.lesson_only:
            raise ValueError('bounded_recovery_requires_source_bound_lesson_runtime')
    if args.recorded_trigger_clock and not source_observation and (manifest.get('multi_task_schema') not in ('development_roles_v2','development_states_v3')
                                      or not args.lesson or args.lesson_only):
        raise ValueError('recorded_clock_requires_roles_v2_lesson_runtime')
    if args.replay_generated_from and (not live or not args.recorded_trigger_clock
            or args.refine_from or args.generation_feedback_from or args.historical_debug_parent):
        raise ValueError('explicit_live_clock_corrected_revalidation_required')
    if (not 3 <= len(manifest["cases"]) <= 8
            or len({c["case_id"] for c in manifest["cases"]}) != len(manifest["cases"])
            or any(not c["case_id"].replace("_", "").isalnum() for c in manifest["cases"])
            or any(c["phase"] not in ("debug", "admission") for c in manifest["cases"])
            or sum(c["phase"] == "debug" for c in manifest["cases"]) < 1
            or sum(c["phase"] == "admission" for c in manifest["cases"]) < 2):
        raise ValueError("predeclare_debug_and_two_admission_cases")
    # Specialized legacy IR behavior contracts are not silently treated as
    # Python validation. The new Python scope has its own evidence gate.
    if manifest.get("synthesis_failure_scope") == "initial_disconnected":
        raise ValueError("specialized_initial_contact_python_cycle_not_supported_yet")
    prepared = [source_case(spec) for spec in manifest["cases"]]
    observation_bindings = None
    if source_observation:
        from tools.python_source_budget import observation_binding
        observation_bindings = {p['spec']['case_id']: observation_binding(p) for p in prepared}
    from tools.skill_nonapplicability import bind_cases, audit as audit_nonapplicability
    healthy_cases = bind_cases(manifest)
    if healthy_cases and (not args.lesson or args.lesson_only):
        raise ValueError('healthy_contract_requires_regular_lesson_runtime')
    output.mkdir(parents=True, exist_ok=False)
    write(output / "precommitted_cases.json", manifest)
    if observation_bindings is not None:
        write(output / 'observation_bindings.json', observation_bindings)
    sources = sorted((ROOT / "orchestrator").glob("*.py")) + sorted((ROOT / "worker").glob("*.py"))
    sources += [Path(__file__), ROOT / "tools/python_skill_fixture.py", ROOT / "tools/run_skill_discovery.py"]
    if healthy_cases:
        sources += [ROOT / 'tools/skill_nonapplicability.py', ROOT / 'tools/multitask_skill_evidence.py']
    if args.recorded_trigger_clock:
        sources.append(ROOT/'tools/python_recorded_clock.py')
    if args.source_case_clock:sources.append(ROOT/'tools/python_source_budget.py')
    if manifest.get('support_probe_sources'):
        sources.append(ROOT/'tools/support_experience.py')
    if recovery_protocol is not None:
        sources.append(ROOT/'tools/run_support_recovery_case.py')
    source_before = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    write(output / "source_before.json", source_before)
    run_id = "python-cycle:" + digest(str(output))[:20]
    packet = python_generation_packet(manifest, run_id)
    if recovery_protocol is not None:
        packet['context']['recovery_protocol'] = recovery_protocol
    ledger = checked(args.ledger, True) if args.ledger else None
    if args.historical_debug_parent:
        from python_revision_parent import historical_packet
        packet = historical_packet(packet, read(checked(args.historical_debug_parent)), manifest)
    elif args.refine_from:
        packet = refinement_packet(packet, args.refine_from, manifest, live=live, ledger=ledger)
    elif manifest.get("development_parent"):
        packet = development_parent_packet(packet, manifest, live=live, ledger=ledger)
    elif manifest.get("registered_parent"):
        packet = registered_parent_packet(packet, manifest)
    if args.development_probes:
        packet = add_development_probes(packet, args.development_probes)
    if args.generation_output_tokens is not None:
        packet["context"]["python_output_tokens"] = args.generation_output_tokens
    if manifest.get("python_stability_pose_contract"):
        from orchestrator.python_stability_pose import stability_source_contract, failed_debug_requirements
        packet["context"]["stability_coordinate_contract"] = stability_source_contract(packet)
        if manifest.get("python_support_preserve_contract"):
            from orchestrator.python_stability_pose import support_preservation_scope
            packet["context"]["stability_coordinate_contract"]["preserve_path_when_all_remaining_fit"] = True
            packet["context"]["support_preservation_observations"] = [
                dict(case_id=r["case_id"], **scope) for r in packet["validation_feedback"]
                if (scope := support_preservation_scope(r["current_observation_request"])) is not None]
        packet["context"]["failed_debug_change_requirements"] = failed_debug_requirements(packet)
    if args.generation_feedback_from:
        if not live:
            raise ValueError("live_generation_feedback_requires_live_mode")
        prior_manifest = read(checked(args.generation_feedback_from, True) / 'precommitted_cases.json')
        if prior_manifest != manifest:
            raise ValueError('generation_feedback_predeclared_conditions_changed')
        packet["context"]["prior_generation_rejection"] = contract_rejection_feedback(
            args.generation_feedback_from, ledger)
    if args.require_pose_equivariance:
        if not args.refine_from:
            raise ValueError("pose_contract_requires_bound_debug_reference")
        from orchestrator.python_skill_pose import source_contract
        packet["context"]["coordinate_contract"] = source_contract(packet)
    lesson = None
    if args.lesson:
        from orchestrator.lesson_memory import validate_lesson
        lesson = validate_lesson(read(checked(args.lesson)))
        packet["context"].update(lesson_stage="program", lesson=lesson)
    if args.prior_library or args.prior_lessons:
        if not args.lesson or not args.prior_library or args.lesson_only:
            raise ValueError('prior_knowledge_requires_regular_lesson_generation')
        from orchestrator.lesson_memory import with_prior_knowledge
        packet=with_prior_knowledge(packet,checked(args.prior_library),
            checked(args.prior_lessons) if args.prior_lessons else None,allow_fixture=not live)
    if args.lesson_only:
        if (lesson is None or args.refine_from or args.require_pose_equivariance
                or args.development_probes or args.generation_feedback_from
                or manifest.get("python_stability_pose_contract")):
            raise ValueError("lesson_only_requires_independent_unspecialized_generation")
        from orchestrator.lesson_discovery import knowledge_only_packet
        packet = knowledge_only_packet(packet, lesson, prepared[0]["request"])
    if lesson is not None and not args.lesson_only:
        if source_observation:
            packet = attach_debug_inputs(packet, prepared, run_id, source_observation=True)
            packet['context']['saved_observation_clock'] = 'actual_source_request_allowance'
            packet['context']['observation_contract'] = args.observation_contract
        elif args.recorded_trigger_clock:
            packet = attach_debug_inputs(packet, prepared, run_id, recorded_clock=True)
            packet['context']['saved_observation_clock']='actual_source_request_allowance'
        else:
            packet = attach_debug_inputs(packet, prepared, run_id)
        packet = attach_applicability_inputs(packet, manifest, run_id, source_observation=source_observation)
    elif manifest.get('python_applicability_cases'):
        raise ValueError('applicability_requires_regular_lesson_runtime')
    if args.python_revision_contract:
        packet['context']['python_revision_contract_version'] = args.python_revision_version
    if args.abstention_feedback:
        from orchestrator.python_abstention import attach_feedback
        packet = attach_feedback(packet)
    if args.generation_timeout_seconds is not None:
        packet['timeout_seconds'] = args.generation_timeout_seconds
    write(output / "generation_packet.json", packet)
    write(output / "config.json", dict(mode=args.mode, api=config, retries=0, key_accessed=live and not args.replay_generated_from,
        recorded_trigger_clock=args.recorded_trigger_clock,
        observation_contract=args.observation_contract,
        fresh_validation_wall_seconds='bound_source_per_case' if args.source_case_clock else (45 if args.recorded_trigger_clock else 30),
        replay_generated_from=str(args.replay_generated_from) if args.replay_generated_from else None,
        ledger=str(ledger) if ledger else None,
        generation_timeout_seconds=packet.get('timeout_seconds', 30),
        require_pose_equivariance=args.require_pose_equivariance,
        lesson_only_reconstruction=args.lesson_only,
        preparation_cost=True, automatic_active_promotion=False,
        body_source="debug_experience_only; admission excluded"))
    transport = None
    if args.hierarchy_policy or args.hierarchy_ledger:
        if not live or not args.hierarchy_policy or not args.hierarchy_ledger:
            raise ValueError("hierarchy_learner_requires_explicit_live_policy_and_ledger")
        from orchestrator.hierarchy_budget import HierarchyBudget, validate_policy
        hierarchy_policy = validate_policy(read(checked(args.hierarchy_policy)))
        if config != hierarchy_policy["roles"]["learner"] or not hierarchy_policy["enabled"]:
            raise ValueError("hierarchy_learner_config_mismatch")
        pool = HierarchyBudget(checked(args.hierarchy_ledger, True), hierarchy_policy,
            transport_failure_policy=args.transport_failure_policy)
        transport = pool.transport("learner")
    Service = PythonSkillDiscoveryService
    if lesson is not None:
        from orchestrator.lesson_discovery import LessonDiscoveryService
        Service = LessonDiscoveryService
    service = Service(config, ledger or output / "preview_ledger.sqlite", transport)
    if transport is not None:
        service._live_transport = transport.live
    preview = service.preview(packet)
    write(output / "api_preview.json", preview)
    if args.mode == "preview":
        print(json.dumps({k: v for k, v in preview.items() if k != "body"}), flush=True)
        return 0
    if live and args.replay_generated_from:
        from tools.python_recorded_clock import replay_candidate
        require_live_config(config,service)
        proposal,generation,replay = replay_candidate(args.replay_generated_from,manifest,ledger,config['model'])
        write(output/'generation_replay.json',replay)
        write(output/'provider_proposal.json',proposal)
        program,card=proposal['program'],proposal['card']
    elif live:
        from local_api_key import key_environment
        require_live_config(config, service)
        with key_environment():
            proposal = service.propose(packet)
        write(output / "provider_proposal.json", proposal)
        save_rejection_artifacts(output, proposal)
        if proposal["status"] != "proposed" or proposal["origin"] != "actual_llm":
            write(output / "result.json", dict(status="GENERATION_REJECTED", registered=False,
                actual_llm_generation=proposal.get('metadata', {}).get('llm_live') is True,
                fallback_reason=proposal.get('fallback_reason'), physical_cases_not_run=True,
                source_unchanged=all(sha(ROOT / name) == value for name,value in source_before.items())))
            return 1
        program, card = proposal["program"], proposal["card"]
        generation = dict(origin="live_llm", model=config["model"], generation_id=packet["request_id"],
            provider_response_path=str(output / "provider_proposal.json"),
            provider_response_sha256=sha(output / "provider_proposal.json"))
    else:
        program = deepcopy(PROGRAM)
        if lesson is not None:
            # Explicit handwritten integration fixture exercises the NEW field;
            # it cannot accidentally pass using only the legacy ctx.last path.
            program['source'] = program['source'].replace('def propose(ctx):\n',
                'def propose(ctx):\n    if ctx.get("failure") is None:\n        return None\n', 1)
        card = dict(name="HANDWRITTEN pose-relative upward turn fixture",
                    hypothesis="Wiring fixture, not LLM-generated or generalization evidence.",
                    evidence_ids=[packet["evidence"][0]["evidence_id"]])
        generation = dict(origin="fixture_authored", generation_id=run_id)
    if manifest.get("python_stability_pose_contract"):
        from orchestrator.python_stability_pose import check_stability_pose_contract, check_stability_revision
        from orchestrator.skill_program import _scoped_last
        revision = check_stability_revision(program, packet)
        write(output / "stability_revision_check.json", revision)
        if not revision["accepted"]:
            with (output / "candidate.py").open("x", encoding="utf-8") as stream:
                stream.write(program["source"])
            write(output / "result.json", dict(status="STABILITY_REVISION_REJECTED",
                registered=False, actual_llm_generation=live, physical_cases_not_run=True))
            return 1
        templates = [r["current_observation_request"] for r in packet["validation_feedback"]
            if len(r["current_observation_request"]["remaining_parts"]) > 2
            and _scoped_last(r["current_observation_request"])["initial_disconnected"]]
        if manifest.get("python_support_preserve_contract"):
            checks = [check_stability_pose_contract(program, template,
                packet["context"]["stability_coordinate_contract"]) for template in templates]
            audit = dict(kind="synthetic_stability_support_and_coordinates_NOT_PHYSICS",
                accepted=all(c["accepted"] for c in checks), checks=checks,
                checked=sum(c["checked"] for c in checks), passed=sum(c["passed"] for c in checks),
                api_calls=0, physical_calls=0)
        else:
            audit = check_stability_pose_contract(program, templates[0],
                                                  packet["context"]["stability_coordinate_contract"])
        write(output / "stability_coordinate_check.json", audit)
        if not audit["accepted"]:
            with (output / "candidate.py").open("x", encoding="utf-8") as stream:
                stream.write(program["source"])
            write(output / "result.json", dict(status="STABILITY_POSE_CONTRACT_REJECTED",
                registered=False, actual_llm_generation=live, physical_cases_not_run=True))
            return 1
        card["coordinate_contract_sha256"] = sha(output / "stability_coordinate_check.json")
        card["coordinate_contract_scope"] = "%d synthetic rotations, initial multi-body branches; NOT physics" % audit["checked"]
    if args.require_pose_equivariance:
        from orchestrator.python_skill_pose import check_pose_contract
        audit = check_pose_contract(program,
            packet["validation_feedback"][0]["current_observation_request"],
            packet["context"]["coordinate_contract"])
        write(output / "coordinate_contract_check.json", audit)
        if not audit["accepted"]:
            with (output / "candidate.py").open("x", encoding="utf-8") as stream:
                stream.write(program["source"])
            write(output / "result.json", dict(status="POSE_CONTRACT_REJECTED",
                registered=False, actual_llm_generation=live,
                physical_cases_not_run=True, coordinate_contract_passed=False))
            return 1
        card["coordinate_contract_sha256"] = sha(output / "coordinate_contract_check.json")
        card["coordinate_contract_scope"] = "24 synthetic axis rotations; NOT physics"
    card["description"] = API_DESCRIPTION + "\nCode:\n" + program["source"]
    if healthy_cases:
        healthy_audit = audit_nonapplicability(program, healthy_cases)
        write(output / 'nonapplicability_check.json', healthy_audit)
        if not healthy_audit['accepted']:
            with (output / 'candidate.py').open('x', encoding='utf-8') as stream:
                stream.write(program['source'])
            write(output / 'result.json', dict(status='NONAPPLICABILITY_REJECTED',
                registered=False, actual_llm_generation=live, physical_cases_not_run=True,
                new_physical_success=False))
            return 1
        card['healthy_training_contract_sha256'] = sha(output / 'nonapplicability_check.json')
        card['healthy_training_contract_scope'] = 'saved actual PASS states; API0/physics0; NOT repair success'
    if lesson is not None:
        from orchestrator.lesson_runtime import CONTRACT
        card.update(lesson_id=lesson["content_sha256"], lesson_claims_are_hypotheses=True)
        card["runtime_contract"] = CONTRACT
    if recovery_protocol is not None:
        card['recovery_protocol'] = deepcopy(recovery_protocol)
    registry = SkillRegistry(output / "registry", fixture=not live)
    candidate = registry.propose(program, card, generation)
    identity = candidate["candidate_id"]
    with (output / "candidate.py").open("x", encoding="utf-8") as stream:
        stream.write(program["source"])
    cases = [dict(case_id=p["spec"]["case_id"], phase=p["spec"]["phase"],
        source_task=p["spec"]["source_task"], program_hash=identity,
        input_hashes=p["raw"]["input_hashes"], source_hashes=p["expected_source_hashes"],
        trigger_kind=p["spec"]["trigger_kind"],
        **({"expected_diagnostic_phase": p["spec"]["expected_diagnostic_phase"]}
           if "expected_diagnostic_phase" in p["spec"] else {})) for p in prepared]
    if lesson is not None:
        for case in cases:
            case["runtime_contract"] = CONTRACT
    if recovery_protocol is not None:
        for case in cases:
            case['recovery_protocol'] = deepcopy(recovery_protocol)
    if args.recorded_trigger_clock:
        for original,case in zip(prepared,cases):
            if args.source_case_clock:
                from tools.python_source_budget import source_budget
                case['source_budget']=source_budget(original)
                case['case_wall_seconds']=case['source_budget']['seconds']
            else:case['case_wall_seconds']=45
    registry.set_manifests(identity,
        {"phase": "debug", "cases": [c for c in cases if c["phase"] == "debug"]},
        {"phase": "admission", "cases": [c for c in cases if c["phase"] == "admission"]})
    if lesson is not None and not args.lesson_only:
        from orchestrator.python_skill_preflight import audit_program, preflight_rows
        if args.python_revision_contract:
            from orchestrator.python_skill_revision import audit_revision
            preflight = audit_revision(program, packet, proposal.get('behavior_checks'))
        else:
            preflight = audit_program(program, preflight_rows(packet))
        write(output / 'development_preflight.json', preflight)
        if not preflight['accepted']:
            write(output / 'result.json', dict(status='PREFLIGHT_REJECTED', registered=False,
                candidate_id=identity, actual_llm_generation=live, generation_origin=generation['origin'],
                preflight_sha256=digest(preflight), cases=[], physical_cases_not_run=True,
                source_unchanged=all(sha(ROOT / name) == value for name,value in source_before.items()),
                full_plan='NOT_RUN', cross_task_transfer='NOT_RUN', performance='NOT_RUN'))
            return 1
        if packet['context'].get('support_development_experiences'):
            from orchestrator.support_learning import audit_support_program
            support_audit = audit_support_program(program, packet)
            write(output/'support_development_preflight.json',support_audit)
            if not support_audit['accepted']:
                write(output/'result.json',dict(status='PREFLIGHT_REJECTED',registered=False,
                    candidate_id=identity,actual_llm_generation=live,physical_cases_not_run=True,
                    preflight_sha256=digest(preflight),support_preflight_sha256=digest(support_audit),
                    cases=[], rejection_stage='support_development',
                    source_unchanged=all(sha(ROOT/name)==h for name,h in source_before.items())))
                return 1
    outcomes = []
    for original, case in zip(prepared, cases):
        print(json.dumps({"phase": "fresh_actual_physics", "case": case["case_id"]}), flush=True)
        try:
            if recovery_protocol is not None:
                from tools.run_support_recovery_case import run_case as run_sequence_case
                artifacts = run_sequence_case(original, case, program, output, run_id)
            else:
                artifacts = run_case(original, case, program, output, run_id)
            record = registry.add_validation(identity, case, artifacts)
            outcomes.append({"case_id": case["case_id"], "outcome": record["outcome"]})
        except Exception as error:
            issue = dict(case_id=case["case_id"], outcome="ERROR",
                         error_type=type(error).__name__, error=str(error))
            write(output / (case["case_id"] + "_error.json"), issue)
            outcomes.append(issue)
    decision = registry.admit(identity)
    snapshot = registry.freeze(output / "skills_frozen.json")
    unchanged = all(sha(ROOT / name) == value for name, value in source_before.items())
    write(output / "result.json", dict(
        status="REGISTERED_SCOPED" if decision["admitted"] else "NOT_REGISTERED",
        generation_origin=generation["origin"], actual_llm_generation=live,
        candidate_id=identity, cases=outcomes, admission=decision,
        frozen_library=str(output / "skills_frozen.json"), entries=len(snapshot["entries"]),
        source_unchanged=unchanged, physics_backend="ASAP_RedMax; inspect per-case results",
        lesson_only_reconstruction=args.lesson_only,
        full_plan="NOT_RUN", cross_task_transfer="NOT_RUN", performance="NOT_RUN"))
    print(json.dumps({"admitted": decision["admitted"], "cases": outcomes,
                      "generation_origin": generation["origin"], "source_unchanged": unchanged}), flush=True)
    return 0 if decision["admitted"] and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
