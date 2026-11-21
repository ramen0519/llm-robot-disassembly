"""Opt-in long part-level episode; no automatic learning or extra authority.

This is an execution bound, never permission to spend. The ordinary research
profile and its source-bound learning contract retain their original limits.
"""
from copy import deepcopy
from decimal import Decimal
from pathlib import PurePosixPath

from orchestrator.research_profile import EPISODE_SETTINGS as RESEARCH_SETTINGS, TASK_FIELDS

PROFILE_ID = 'part_capacity_v1'
EPISODE_SETTINGS = deepcopy(RESEARCH_SETTINGS)
LIMITS = {'budget': 256, 'state_action_budget': 48, 'max_gripper': 3,
          'max_pose': 128, 'timeout': 3600, 'wall_timeout': 3900, 'response_timeout': 120}


def validate_task(task):
    if type(task) is not dict or not TASK_FIELDS <= set(task) or set(task) - TASK_FIELDS - {'evaluation_manifest'}:
        raise ValueError('invalid_capacity_task_fields')
    raw = task['assembly_dir']
    if (type(raw) is not str or '\\' in raw or ':' in raw
            or '..' in PurePosixPath(raw).parts
            or not raw.startswith(('/opt/asap-research/', '/mnt/c/dev/disassembly_codex_ready_v2/'))):
        raise ValueError('capacity_task_path_outside_supported_roots')
    if (type(task['base_part']) is not str or not task['base_part']
            or any(c in task['base_part'] for c in '/\\:')):
        raise ValueError('invalid_capacity_base_id')
    for name in TASK_FIELDS - {'assembly_dir', 'base_part'}:
        if type(task[name]) is not int or not (0 if name == 'seed' else 1) <= task[name]:
            raise ValueError('invalid_capacity_task_integer')
        if name in LIMITS and task[name] > LIMITS[name]:
            raise ValueError('capacity_task_bound_exceeded:' + name)
    if task['seed'] >= 2**32 or task['state_action_budget'] > task['budget']:
        raise ValueError('invalid_capacity_seed_or_state_budget')
    if task['wall_timeout'] <= task['timeout'] or task['response_timeout'] > task['timeout']:
        raise ValueError('capacity_time_contract_invalid')
    if task.get('evaluation_manifest') is not None and type(task['evaluation_manifest']) is not str:
        raise ValueError('invalid_capacity_evaluation_manifest')
    return deepcopy(task)


def episode_flags():
    result = ['--research-profile', PROFILE_ID]
    for name, value in EPISODE_SETTINGS.items():
        result.append('--' + name.replace('_', '-'))
        if value is not True:
            result.append(str(value))
    return result


def apply_episode_profile(args, argv):
    if getattr(args, 'research_profile', None) != PROFILE_ID:
        raise ValueError('explicit_capacity_profile_required')
    provided = [str(x).split('=', 1)[0] for x in argv if str(x).startswith('--')]
    for name, value in EPISODE_SETTINGS.items():
        flag = '--' + name.replace('_', '-')
        if provided.count(flag) > 1:
            raise ValueError('duplicate_capacity_option:' + name)
        if flag in provided and getattr(args, name) != value:
            raise ValueError('capacity_profile_conflict:' + name)
        setattr(args, name, value)
    for name in TASK_FIELDS:
        if provided.count('--' + name.replace('_', '-')) != 1:
            raise ValueError('explicit_unique_capacity_task_option_required:' + name)
    task = validate_task({name: getattr(args, name) for name in TASK_FIELDS})
    return dict(profile_id=PROFILE_ID, policy=deepcopy(EPISODE_SETTINGS), task=task,
        learning_observation_contract=None, automatic_learning_enabled=False,
        learning_eligibility='NOT_VALIDATED_FOR_AUTOMATIC_LEARNING',
        observations_saved_for_review=True, new_skill_generation=False,
        retrieval='local_bounded_cards_model_selects', extra_retrieval_api_calls=0,
        physical_verifier='ASAP_RedMax_part_level_NOT_robot',
        approval_inferred=False, automatic_http_retries=0)


def validate_policy_binding(policy, profile_id):
    """Bind an already validated budget to the explicit runner profile.

    A schema-2 long budget cannot be used through the legacy or short runner;
    the capacity runner cannot silently adopt an old budget or learning role.
    """
    capacity_policy = (policy.get('schema_version') == 2
                       and policy.get('execution_profile') == PROFILE_ID)
    if profile_id != PROFILE_ID:
        if policy.get('schema_version') == 2 or policy.get('execution_profile') is not None:
            raise ValueError('capacity_policy_requires_explicit_capacity_profile')
        return
    if not capacity_policy:
        raise ValueError('capacity_profile_requires_bound_capacity_policy')
    limits, roles = policy['episode'], policy['roles']
    if (limits['scout_calls'] > 1 or limits['planner_calls'] > 24
            or limits['fallback_calls'] != 0 or limits['max_plan_steps'] > 8
            or limits['recovery_per_step'] != 1
            or roles['scout']['max_calls'] > 1 or roles['planner']['max_calls'] > 24
            or roles['learner']['max_calls'] != 0 or policy['max_calls'] > 25
            or Decimal(str(policy['max_usd'])) > Decimal('15')):
        raise ValueError('capacity_profile_policy_scope_exceeded')
