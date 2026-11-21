"""Compact facts for model reconsideration after same-state stability failures.

This does not select a mover, authorize a new orientation, alter support guards,
prune a candidate, or infer feasibility. The original request and raw evidence
remain authoritative. Registered Python programs retain their own pose choice.
"""
from orchestrator.strategy_candidates import (
    _observations, catalog_pose_choice, pose_policy_context,
)
from worker.explicit_action_contract import digest, validate_request


def stability_reconsideration_context(request, orientation_reference=None):
    current = validate_request(request)
    if 'attempted_actions' not in current or 'path_stage_observations' not in current:
        # These readable histories are optional in the general worker contract.
        # Missing records must not become evidence that a pose is untried.
        return dict(version='strategy_stability_reconsideration_v1',
            request_id=current['request_id'], state_id=current['state_id'], request_sha256=digest(current),
            available=False, reason='readable_attempt_history_unavailable',
            recorded_action_count=len(current['attempted_action_ids']), bound_failure_count=None,
            recorded_pose_count=None, poses_without_recorded_actions=None,
            select_catalog_once_preview=None, reconsider_mover_pose_or_support=None,
            coverage_is_complete_episode_history=False,
            preview_is_not_action_permission=True, selection_forced=False,
            candidate_pruning=False, automatic_pose_change=False, physics_calls=0, api_calls=0,
            robot_motion_authorized=False)
    availability = pose_policy_context(current, orientation_reference)
    facts = _observations(current)
    recorded = current['attempted_actions']
    pose_ids = sorted(p['pose_id'] for p in current['pose_catalog'])
    attempted_pose_ids = {row['pose_id'] for row in recorded}
    observed = []
    for pose_id in sorted({item['choice']['pose_id'] for item in facts}):
        rows = [item for item in facts if item['choice']['pose_id'] == pose_id]
        stability = [item for item in rows if item['feedback'].get('stability_success') is False]
        paths = [item for item in rows if item['feedback'].get('path_success') is False]
        observed.append(dict(pose_id=pose_id,
            bound_path_failures=len(paths), bound_stability_failures=len(stability),
            tested_mover_ids=sorted({item['choice']['part_id'] for item in rows}),
            distinct_failed_stability_configurations=len({
                (item['choice']['part_id'], tuple(sorted(item['choice']['support_ids'])))
                for item in stability}),
            path_failure_evidence_ids=[item['feedback']['evidence_id'] for item in paths],
            stability_failure_evidence_ids=[item['feedback']['evidence_id'] for item in stability]))
    matches = availability['matching_pose_ids']
    reference_pose = matches[0] if availability['keep_orientation_available'] else None
    previews = {}
    for part in sorted(current['candidate_ids']):
        selected = catalog_pose_choice(current, part)['pose_id']
        previews[part] = dict(pose_id=selected,
            same_as_current_reference=(selected == reference_pose if reference_pose is not None else None))
    return dict(version='strategy_stability_reconsideration_v1',
        request_id=current['request_id'], state_id=current['state_id'], request_sha256=digest(current),
        available=True,
        scope='current_request_retained_bound_failure_history',
        recorded_action_count=len(recorded), bound_failure_count=len(facts),
        recorded_actions_without_bound_failure_count=len(recorded) - len(facts),
        coverage_is_complete_episode_history=False,
        current_reference_pose_id=reference_pose,
        current_reference_legal_to_keep=availability['keep_orientation_available'],
        catalog_pose_count=len(pose_ids), recorded_pose_count=len(attempted_pose_ids),
        poses_without_recorded_actions=[p for p in pose_ids if p not in attempted_pose_ids],
        bound_failures_by_pose=observed,
        reconsider_mover_pose_or_support=any(row['bound_stability_failures'] for row in observed),
        select_catalog_once_preview=previews,
        preview_procedures=['axis_search_v1', 'two_leg_search_v1'],
        registered_skill_pose_is_program_selected=True,
        preview_rule='fewest_selected_part_attempts_then_identity_rotation_distance_then_pose_id',
        preview_is_not_action_permission=True, future_state_preview_valid=False,
        untried_pose_is_not_success=True, stability_failure_is_not_all_pose_failure=True,
        support_guards_unchanged=True, selection_forced=False, candidate_pruning=False,
        automatic_pose_change=False, physics_calls=0, api_calls=0,
        robot_motion_authorized=False)
