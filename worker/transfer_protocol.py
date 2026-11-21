"""Read-only opt-in evaluation contract shared by host and legacy worker."""
import hashlib
import json
from pathlib import Path


def validate_manifest(path, assembly, args):
    raw=str(path).replace('\\','/')
    if raw.lower().startswith(('e:','/mnt/e','//')) or '..' in raw.split('/'):
        raise ValueError('excluded_manifest_path')
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    # Old manifests describe fresh construction. A new execution profile must
    # be declared, and a declared reuse experiment must not silently run fresh.
    scene_policy = getattr(args, 'physics_scene_policy', 'fresh_v1')
    declared_scene_policy = value.get('settings', {}).get('physics_scene_policy', 'fresh_v1')
    if (scene_policy not in ('fresh_v1', 'path_scene_reuse_v1')
            or declared_scene_policy not in ('fresh_v1', 'path_scene_reuse_v1')
            or scene_policy != declared_scene_policy):
        raise ValueError('evaluation_physics_scene_policy_mismatch')
    for field, allowed in (
            ('pose_transition_policy', ('legacy_v1', 'accepted_pose_carry_v1')),
            ('strategy_failure_policy', ('legacy_v1', 'model_ordered_failover_v1'))):
        # Strategy failover belongs to the host; the worker has no such flag.
        # A new host/worker policy must be declared in new manifests, while old
        # manifests and default legacy invocations keep their original meaning.
        if field == 'strategy_failure_policy' and not hasattr(args, field):
            continue
        selected = getattr(args, field, 'legacy_v1')
        declared = value.get('settings', {}).get(field, 'legacy_v1')
        if selected not in allowed or declared not in allowed or selected != declared:
            raise ValueError('evaluation_' + field + '_mismatch')
    if value.get('schema')=='multitask_learning_v1':
        if (value.get('purpose') not in ('training','evaluation') or value.get('automatic_retries')!=0
                or type(value.get('seed')) is not int or args.seed!=value['seed']
                or len(set(value.get('training_ids',[])))!=3 or len(set(value.get('evaluation_ids',[])))!=2
                or set(value['training_ids']) & set(value['evaluation_ids'])):
            raise ValueError('invalid_multitask_manifest')
        allowed=value['training_ids'] if value['purpose']=='training' else value['evaluation_ids']
        if (assembly not in value['assembly_dirs'] or Path(assembly).name not in allowed
                or any(Path(p).name not in allowed for p in value['assembly_dirs'])):
            raise ValueError('task_outside_predeclared_split')
        for field in ('budget','state_action_budget','max_gripper','max_pose','timeout','response_timeout'):
            if getattr(args,field)!=value['settings'][field]:raise ValueError('multitask_setting_mismatch:'+field)
        if hasattr(args,'wall_timeout') and args.wall_timeout!=value['settings']['wall_timeout']:
            raise ValueError('multitask_wall_mismatch')
        if getattr(args,'base_part',None) not in (None,'none'):
            raise ValueError('multitask_requires_no_fixed_base')
        if value.get('path_limit')!=96 or value.get('stability_limit')!=48:
            raise ValueError('multitask_physics_limit_mismatch')
        return value
    if value.get('schema')!='unseen_transfer_v1' or value.get('automatic_retries')!=0:
        raise ValueError('invalid_transfer_manifest')
    if assembly not in value['assembly_dirs']:
        raise ValueError('assembly_not_predeclared')
    for field in ('budget','state_action_budget','max_gripper','max_pose','timeout','response_timeout'):
        if getattr(args,field)!=value['settings'][field]:raise ValueError('evaluation_setting_mismatch:'+field)
    if hasattr(args,'wall_timeout') and args.wall_timeout!=value['settings']['wall_timeout']:
        raise ValueError('wall_setting_mismatch')
    if args.seed!=0:raise ValueError('pilot_seed_mismatch')
    if getattr(args,'base_part',None) not in (None,'none',value.get('development_base_part')):
        raise ValueError('fixed_base_not_declared')
    return value


def verify_input(assembly, manifest):
    expected=manifest['assembly_dirs'][str(assembly)]
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(assembly).iterdir() if p.is_file()}
    if expected!=actual:raise ValueError('evaluation_input_hash_changed')
    return actual
