"""One explicit, bounded part-level profile; legacy defaults are unchanged.

This module specifies executable policy, not permission to spend or simulate.
It performs no filesystem access, model call, or scene mutation.
"""
from copy import deepcopy
from pathlib import PurePosixPath

PROFILE_ID = 'part_research_v1'
EPISODE_SETTINGS = {
    'provider_contract': 'explicit_action_provider_v5',
    'plan_provider_contract': 'strategy_plan_v1',
    'decision_failure_policy': 'halt_v1',
    'fallback_policy': 'path_stage_v1',
    'path_selection_policy': 'path_probe_round_v1',
    'recovery_support_policy': 'preserve_existing_v1',
    'planning_feedback_policy': 'repair_evidence_v2',
    'transport_failure_policy': 'latch_on_error_v1',
    'pose_catalog_policy': 'complete_v1',
    'pose_transition_policy': 'accepted_pose_carry_v1',
    'strategy_failure_policy': 'model_ordered_failover_v1',
    'physics_scene_policy': 'path_scene_reuse_v1',
    'memory_policy': 'assembly_balanced_v1',
    'memory_max_records': 8,
    'knowledge_policy': 'unified_v1',
    'knowledge_runtime_feedback': True,
    'knowledge_max_skills': 16,
    'lesson_runtime': True,
    'max_skill_attempts': 3,
    'stability_diagnostics': True,
}
TASK_FIELDS = {'assembly_dir', 'base_part', 'seed', 'budget', 'state_action_budget',
               'max_gripper', 'max_pose', 'timeout', 'wall_timeout', 'response_timeout'}
# Source-bound learning already supports <=600 s original observations. A larger
# capacity run is a different legacy profile, never silently made learnable here.
LIMITS = {'budget': 128, 'state_action_budget': 24, 'max_gripper': 3,
          'max_pose': 128, 'timeout': 600, 'wall_timeout': 900, 'response_timeout': 120}


def validate_profile(value):
    if value != PROFILE_ID:
        raise ValueError('unknown_research_profile')
    return value


def validate_task(task):
    if type(task) is not dict or not TASK_FIELDS <= set(task) or set(task) - TASK_FIELDS - {'evaluation_manifest'}:
        raise ValueError('invalid_research_task_fields')
    raw = task['assembly_dir']
    if (type(raw) is not str or '\\' in raw or ':' in raw
            or '..' in PurePosixPath(raw).parts
            or not raw.startswith(('/opt/asap-research/', '/mnt/c/dev/disassembly_codex_ready_v2/'))):
        raise ValueError('research_task_path_outside_supported_roots')
    if (type(task['base_part']) is not str or not task['base_part']
            or any(c in task['base_part'] for c in '/\\:')):
        raise ValueError('invalid_base_id')
    for name in TASK_FIELDS - {'assembly_dir', 'base_part'}:
        if type(task[name]) is not int or not (0 if name == 'seed' else 1) <= task[name]:
            raise ValueError('invalid_research_task_integer')
        if name in LIMITS and task[name] > LIMITS[name]:
            raise ValueError('research_task_bound_exceeded:' + name)
    if task['seed'] >= 2**32 or task['state_action_budget'] > task['budget']:
        raise ValueError('invalid_research_seed_or_state_budget')
    if task['wall_timeout'] <= task['timeout'] or task['response_timeout'] > task['timeout']:
        raise ValueError('research_time_contract_invalid')
    if task.get('evaluation_manifest') is not None and type(task['evaluation_manifest']) is not str:
        raise ValueError('invalid_research_evaluation_manifest')
    return deepcopy(task)


def episode_flags():
    result = ['--research-profile', PROFILE_ID]
    for name, value in EPISODE_SETTINGS.items():
        result.append('--' + name.replace('_', '-'))
        if value is not True:
            result.append(str(value))
    return result


def learning_flags():
    return ['--observation-contract', 'source_bound_v1',
            '--transport-failure-policy', 'latch_on_error_v1']


def apply_episode_profile(args, argv):
    """Pin options; reject conflicts and omitted task budgets before side effects."""
    if not getattr(args, 'research_profile', None):
        return None
    validate_profile(args.research_profile)
    provided = [str(x).split('=', 1)[0] for x in argv if str(x).startswith('--')]
    for name, value in EPISODE_SETTINGS.items():
        flag = '--' + name.replace('_', '-')
        if provided.count(flag) > 1:
            raise ValueError('duplicate_research_option:' + name)
        if flag in provided and getattr(args, name) != value:
            raise ValueError('research_profile_conflict:' + name)
        setattr(args, name, value)
    for name in TASK_FIELDS:
        flag = '--' + name.replace('_', '-')
        if provided.count(flag) != 1:
            raise ValueError('explicit_unique_research_task_option_required:' + name)
    task = validate_task({name: getattr(args, name) for name in TASK_FIELDS})
    return dict(profile_id=PROFILE_ID, policy=deepcopy(EPISODE_SETTINGS), task=task,
                learning_observation_contract='source_bound_v1',
                retrieval='local_bounded_cards_model_selects', extra_retrieval_api_calls=0,
                physical_verifier='ASAP_RedMax_part_level_NOT_robot',
                approval_inferred=False, automatic_http_retries=0)
