"""Prepare observed pose/support recovery for Python learning, without API/physics.

This opt-in development policy leaves the older three-independent-failure policy
unchanged. One observed failure/PASS pair is NOT three independent examples:
the two admission cases are predeclared yaw transformations of the same CAD.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import prepare_episode_learning as evidence
from prepare_stability_learning import INTEGRITY_TEMPLATE, classify_condition
from orchestrator.skill_trace_summary import summarize_recovery_pairs
from worker.stability_diagnostics import project_stability_diagnostics

POLICY = 'first_measured_dynamic_failure_later_checked_pass_with_pose_support_change_v1'
YAW_ADMISSIONS = (90, 180)


def classify_transition_failure(attempt, checked, physical):
    """Do not confuse an over-capacity observation with an impossible action."""
    row = classify_condition(attempt, checked, physical)
    if not row['eligible'] and row['reason'] != 'observed_dynamic_parts_exceed_declared_support_budget':
        return row
    row['eligible'] = False
    raw, choice, context = attempt['raw'], physical['choice'], physical['context']
    full = raw['stability_diagnostics']
    diagnostic = row['diagnostic']
    fixed = set(choice['support_ids'])
    if context.get('base_part') is not None:
        fixed.add(context['base_part'])
    rest = set(attempt['remaining_parts']) - {choice['part_id']}
    free = rest - fixed
    moving = (raw.get('executed') or {}).get('stability_moving_parts')
    observed = set(row['observed_dynamic_parts'])
    upstream = raw.get('raw_upstream') or {}
    if (full.get('instrumentation_errors') != [] or not fixed.issubset(rest)
            or type(moving) is not list or len(moving) != len(set(moving))
            or set(moving) != free or not observed.issubset(free)
            or upstream.get('parts_fall') != diagnostic['returned_parts']
            or not set(diagnostic['returned_parts'] or []).issubset(observed)
            or upstream.get('path_success') is not True or upstream.get('stability_success') is not False):
        row['reason'] = 'dynamic_observation_not_bound_to_actual_free_bodies'
        return row
    row.update(eligible=True, reason='measured_dynamic_failure_including_over_capacity',
        current_free_stability_parts=sorted(free),
        observed_count_exceeds_support_budget=len(observed) > row['available_support_budget'],
        all_alternative_actions_impossible=False)
    return row


def _checked_gravity_pass(attempt):
    raw = attempt['raw']
    try:
        d = project_stability_diagnostics(raw.get('stability_diagnostics'))
    except (ValueError, TypeError, KeyError):
        return False
    return bool(d and raw.get('stability_check_calls') == 1
        and raw.get('stability_mode') == 'one_exact_selected_support_configuration'
        and d['phase'] == 'passed' and d['check_success_returned'] is True
        and d['returned_success'] is True and d['physics_steps'] > 0
        and raw['stability_diagnostics'].get('instrumentation_errors') == []
        and (raw.get('raw_upstream') or {}).get('stability_success') is True)


def select_transition(report, considered, physicals):
    attempts = report['attempts']
    if not (len(attempts) == len(considered) == len(physicals)):
        raise ValueError('transition_selection_input_count_mismatch')
    rows = [classify_transition_failure(a, c, p) for a, c, p in zip(attempts, considered, physicals)]
    summary = summarize_recovery_pairs(report)
    if summary['truncated']:
        raise ValueError('transition_pair_projection_truncated_no_silent_selection')
    pairs = []
    for pair in summary['pairs']:
        before, after = pair['failure_attempt_index'], pair['success_attempt_index']
        reason = rows[before]['reason'] if not rows[before]['eligible'] else None
        if reason is None and not _checked_gravity_pass(attempts[after]):
            reason = 'later_pass_without_measured_gravity_check'
        if reason is None and not {'pose_matrix', 'support_ids'}.intersection(pair['changed_choice_fields']):
            reason = 'no_observed_pose_or_support_change'
        pairs.append(dict(pair=deepcopy(pair), eligible=reason is None, reason=reason,
                          failure_attempt_number=before + 1, success_attempt_number=after + 1))
    selected = next((p for p in pairs if p['eligible']), None)
    return dict(considered=rows, pair_summary=summary, pair_decisions=pairs, selected=selected)


def prepare_transition(episode_dir, output_dir, *, allow_test_root=False):
    def safe(value, runs_only=True):
        return evidence._path(value, allow_test_root=allow_test_root, runs_only=runs_only)
    episode, output = safe(episode_dir), safe(output_dir)
    if output.exists():
        raise FileExistsError('new_transition_output_required')
    # Full archived source, raw result, trajectory, deadline and memory binding.
    # The generic historical admission template is an integrity input only.
    integrity = evidence.prepare_episode_learning(episode, INTEGRITY_TEMPLATE,
        output / 'episode_integrity', allow_test_root=allow_test_root)
    report_path = safe(episode / 'planner/planner_report.json')
    report = evidence._load(report_path)
    physicals = [evidence._load(safe(report_path.parent / ('attempt_%06d' % i) / 'request.json'))
                 for i in range(1, len(report['attempts']) + 1)]
    selection = select_transition(report, integrity['considered'], physicals)
    selected = selection['selected']
    manifest_path = output / 'manifest.json' if selected else None
    result = dict(schema_version=1, status='PREPARED' if selected else 'NO_ELIGIBLE_OBSERVED_TRANSITION',
        selection_policy=POLICY, **selection, episode_dir=str(episode),
        experience_task=integrity['experience_task'], source_run_id=integrity['source_run_id'],
        memory_path=integrity['memory_path'], memory_record_count=integrity['memory_record_count'],
        manifest_path=str(manifest_path) if manifest_path else None,
        integrity_audit=str(output / 'episode_integrity/preparation.json'),
        generic_integrity_manifest_used_for_experiment=False,
        original_artifact_sha256=integrity['original_artifact_sha256'], original_files_unchanged=True,
        development_only=True, target_is_held_out=False, independent_source_tasks=1,
        selected_observed_pairs=int(selected is not None),
        admission_yaw_degrees=list(YAW_ADMISSIONS),
        admission_meaning='same CAD rotation robustness; NOT independent tasks or generalization',
        new_api_calls=0, new_physical_calls=0, new_skill_registered=False,
        causal_effect_established=False, performance_comparison='NOT_RUN')
    if any(evidence._sha(safe(p, False)) != h for p, h in integrity['original_artifact_sha256'].items()):
        raise ValueError('transition_source_changed_during_preparation')
    if selected:
        reference = evidence._project_reference(report_path, allow_test_root)
        common = dict(report=reference, source_task=integrity['experience_task'],
            attempt_index=selected['failure_attempt_number'], trigger_kind='stability_failed',
            expected_diagnostic_phase='dynamic_failure')
        cases = [dict(common, case_id='transition_debug_original', phase='debug')]
        cases += [dict(common, case_id='transition_yaw_%d' % yaw, phase='admission', pose_yaw_degrees=yaw)
                  for yaw in YAW_ADMISSIONS]
        manifest = dict(schema_version=1, purpose='observed transition development; same CAD rotated admission',
            experience_task=integrity['experience_task'], experience_report=reference,
            experience_memory=evidence._project_reference(safe(integrity['memory_path']), allow_test_root),
            frozen_episode_memory=dict(path=integrity['memory_path'], file_sha256=integrity['memory_file_sha256'],
                content_sha256=integrity['memory_content_sha256']),
            python_failure_scope='remaining_body_stability_failed',
            selection_rationale=POLICY, cases=cases, episode_learning_audit=str(output / 'preparation.json'),
            admission_results_exposed=False, support_budget_unchanged=True,
            observations_are_not_causes=True)
        evidence._json_new(manifest_path, manifest)
    evidence._json_new(output / 'preparation.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode-dir', required=True, type=Path)
    p.add_argument('--run-dir', required=True, type=Path)
    args = p.parse_args()
    result = prepare_transition(args.episode_dir, args.run_dir)
    print(json.dumps({k: result[k] for k in ('status', 'manifest_path', 'memory_record_count',
                                           'new_api_calls', 'new_physical_calls')}), flush=True)
    return 0 if result['status'] == 'PREPARED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
