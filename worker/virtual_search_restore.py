"""A worker-owned DFS pre-action snapshot, never a robot reset or a new verdict.

Python 3.8, pure JSON. This context is separate from citable failure feedback.
Action IDs and path history deliberately exclude the node visit revision.
"""
from copy import deepcopy

FIELD = 'virtual_search_restore'
VERSION = 'virtual_dfs_restore_v1'
SCOPE = 'planning_pre_action_snapshot_not_robot_state'
KEYS = {'version', 'scope', 'owner', 'run_id', 'request_id', 'state_id',
        'node_visit_revision', 'remaining_parts', 'returned_from_state_id',
        'returned_from_parts', 'source', 'request_binding_sha256'}
SOURCE_KEYS = {'event_id', 'action_id', 'choice', 'pose_matrix', 'status',
               'accepted_for_transition', 'deadline_exceeded', 'path_success',
               'stability_success', 'backend'}


def _binding(request):
    from worker.explicit_action_contract import digest
    value = {k: deepcopy(request[k]) for k in ('run_id', 'request_id', 'state_id',
        'task_context', 'remaining_parts', 'candidate_ids', 'base_part', 'max_grippers',
        'attempted_action_ids', 'attempted_actions', 'path_stage_observations')}
    value['pose_catalog'] = [{k:p[k] for k in ('pose_id', 'matrix')} for p in request['pose_catalog']]
    return digest(value)


def snapshot(edge, child_state, revision):
    """Called only when worker pops the accepted edge leading to this child."""
    if (edge['status'] != 'PASS' or edge['accepted_for_transition'] is not True
            or edge['deadline_exceeded'] is not False
            or edge['raw'].get('path_success') is not True
            or edge['raw'].get('stability_success') is not True
            or edge['context']['parts_rest'] != child_state['remaining_parts']):
        raise ValueError('invalid_dfs_restore_edge')
    choice=deepcopy(edge['choice']); choice['evidence_ids']=[]
    return dict(version=VERSION, scope=SCOPE, owner='explicit_action_dfs',
        run_id=edge['context']['run_id'], state_id=edge['state_id'],
        node_visit_revision=revision, remaining_parts=list(edge['remaining_parts']),
        returned_from_state_id=child_state['state_id'],
        returned_from_parts=list(child_state['remaining_parts']),
        source=dict(event_id=edge['event_id'], action_id=edge['action_id'], choice=choice,
            pose_matrix=deepcopy(edge['context']['pose_matrix']), status=edge['status'],
            accepted_for_transition=edge['accepted_for_transition'], deadline_exceeded=edge['deadline_exceeded'],
            path_success=edge['raw']['path_success'], stability_success=edge['raw']['stability_success'],
            backend=edge['raw'].get('backend')))


def bind(request, state_snapshot):
    value=deepcopy(state_snapshot)
    value.update(request_id=request['request_id'], request_binding_sha256=_binding(request))
    return value


def validate(request):
    from worker.explicit_action_contract import action_id, validate_choice
    value=request.get(FIELD)
    if value is None and FIELD not in request:
        return None
    def require(ok):
        if not ok:raise ValueError('invalid_virtual_search_restore')
    require(type(value) is dict and set(value)==KEYS)
    require(value['version']==VERSION and value['scope']==SCOPE and value['owner']=='explicit_action_dfs')
    require(all(value[k]==request[k] for k in ('run_id','request_id','state_id','remaining_parts')))
    require(type(value['node_visit_revision']) is int and value['node_visit_revision']>0)
    require(type(value['returned_from_state_id']) is str and 0<len(value['returned_from_state_id'])<=512
            and value['returned_from_state_id']!=request['state_id'])
    require('attempted_actions' in request and 'path_stage_observations' in request)
    require(value['request_binding_sha256']==_binding(request))
    source=value['source']
    require(type(source) is dict and set(source)==SOURCE_KEYS)
    require(source['status']=='PASS' and source['accepted_for_transition'] is True
        and source['deadline_exceeded'] is False and source['path_success'] is True
        and source['stability_success'] is True)
    choice=validate_choice(request,source['choice'])
    require(choice['evidence_ids']==[] and action_id(request,choice)==source['action_id'])
    require(value['returned_from_parts']==[p for p in request['remaining_parts'] if p!=choice['part_id']])
    require(source['pose_matrix']==next(p['matrix'] for p in request['pose_catalog'] if p['pose_id']==choice['pose_id']))
    matches=[(a,s) for a,s in zip(request['attempted_actions'],request['path_stage_observations'])
             if a['action_id']==source['action_id']]
    require(len(matches)==1)
    action,stage=matches[0]
    require({k:v for k,v in action.items() if k!='action_id'}=={k:v for k,v in choice.items() if k!='evidence_ids'})
    require(stage['source_event_id']==source['event_id'] and stage['status']=='PASS'
        and stage['deadline_exceeded'] is False and stage['path_success'] is True
        and stage['backend']==source['backend'])
    # If the full observation is still visible, it must agree. Never edit it.
    for row in request['episode_feedback']:
        if row.get('action_id')==source['action_id'] or row.get('evidence_id')==source['event_id']:
            require(row.get('action_id')==source['action_id'] and row.get('evidence_id')==source['event_id']
                and row.get('remaining_parts')==request['remaining_parts']
                and row.get('accepted_for_transition') is True and row.get('deadline_exceeded') is False
                and row.get('status')=='PASS' and row.get('path_success') is True
                and row.get('stability_success') is True
                and row.get('selected_pose_matrix')==source['pose_matrix']
                and type(row.get('choice')) is dict
                and {k:v for k,v in row['choice'].items() if k!='evidence_ids'}
                    =={k:v for k,v in choice.items() if k!='evidence_ids'})
    return deepcopy(value)


def restored_reference(request):
    """Only the entry to the restored visit can use this pre-action snapshot.

    Once another attempt exists, its normal bound feedback owns the conditions,
    including ERROR/late cases. An old restoration never releases new supports.
    """
    value=validate(request)
    if value is None or request['attempted_action_ids'][-1:] != [value['source']['action_id']]:
        return None
    source=value['source']
    return dict(scope=SCOPE, node_visit_revision=value['node_visit_revision'],
        remaining_parts=list(value['remaining_parts']), action_id=source['action_id'],
        evidence_id=source['event_id'], choice=deepcopy(source['choice']),
        selected_pose_matrix=deepcopy(source['pose_matrix']))
