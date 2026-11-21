"""Merge independently bound development traces; never read evaluation inputs."""
from copy import deepcopy
from pathlib import Path
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read,sha
from worker.task_memory import digest


def validate_roles(training,evaluation):
    if (type(training) is not list or type(evaluation) is not list
            or len(training)!=3 or len(evaluation)!=2 or len(set(training+evaluation))!=5
            or any(type(i) is not str or len(i)!=5 or not i.isdigit() for i in training+evaluation)):
        raise ValueError('three_training_two_disjoint_evaluation_tasks_required')


def manifest_roles(manifest, train):
    """Keep v1 frozen; opt in to failure learning plus healthy abstention v2."""
    if manifest.get('multi_task_schema') == 'development_sources_v1':
        return train[:2], [train[2], train[2]], []
    state_roles = manifest.get('multi_task_schema') == 'development_states_v3'
    if manifest.get('multi_task_schema') not in ('development_roles_v2', 'development_states_v3'):
        raise ValueError('unknown_multi_task_schema')
    roles = manifest.get('role_assignment', {})
    generators = roles.get('generation_tasks', [])
    healthy = roles.get('nonapplicability_tasks', [])
    if (type(generators) is not list or type(healthy) is not list
            or len(generators) != 2 or len(set(generators)) != 2
            or (not state_roles and (len(healthy) != 1 or len(set(generators + healthy)) != 3
                                    or set(generators + healthy) != set(train)))
            or (state_roles and (not 1 <= len(healthy) <= 2 or len(set(healthy)) != len(healthy)
                or not set(generators + healthy) <= set(train)
                or roles.get('nonapplicability_scope') != 'observed_post_PASS_states_NOT_held_out_products'))
            or roles.get('positive_admission_scope') != 'same_products_yaw90_NOT_new_product_transfer'
            or roles.get('assignment_after_collection') is not True
            or roles.get('evaluation_split_unchanged') is not True
            or manifest.get('require_shared_failure_family') is not True):
        raise ValueError('explicit_training_roles_with_unchanged_evaluation_required')
    negatives = manifest.get('nonapplicability_cases')
    if (type(negatives) is not list or len(negatives) != 2
            or len({c['case_id'] for c in negatives}) != 2
            or any(c.get('task_id') not in healthy for c in negatives)):
        raise ValueError('two_real_healthy_training_cases_required')
    cases = manifest['cases']
    if len(cases) != 4:
        raise ValueError('two_debug_and_two_yaw_admission_cases_required')
    for debug, admission in zip(cases[:2], cases[2:]):
        # Same source state, predeclared transformed condition; neither its
        # future outcome nor the negative cases is sent as generation evidence.
        if (set(debug) & {'pose_yaw_degrees', 'trigger_override'}
                or admission != dict(debug, case_id=admission['case_id'],
                                     phase='admission', pose_yaw_degrees=90)):
            raise ValueError('positive_admission_must_be_predeclared_yaw90')
    return generators, generators, healthy


def build_packet(manifest,run_id):
    from tools.run_python_skill_cycle import python_generation_packet
    from orchestrator.skill_discovery import MAX_EVIDENCE
    split_path=checked(manifest['split']['path'],True)
    if sha(split_path)!=manifest['split']['sha256']:raise ValueError('split_changed')
    split=read(split_path)
    train=[x['id'] for x in split['training']];evaluation=[x['id'] for x in split['evaluation']]
    validate_roles(train,evaluation)
    generators, admission, healthy = manifest_roles(manifest, train)
    refs=manifest['experience_sources']
    if type(refs) is not list or len(refs)!=2:raise ValueError('two_independent_generation_sources_required')
    if [r.get('task_id') for r in refs]!=generators:raise ValueError('source_task_not_precommitted_generation_task')
    cases=manifest['cases']
    if (type(cases) is not list or len(cases)!=4 or len({c['case_id'] for c in cases})!=4
            or [c['phase'] for c in cases]!=['debug','debug','admission','admission']
            or [c['source_task'] for c in cases]!=generators+admission):
        raise ValueError('two_debug_tasks_and_reserved_training_admission_required')
    packets=[];provenance=[]
    for ref in refs:
        p=checked(ref['manifest_path'],True)
        if sha(p)!=ref['manifest_sha256']:raise ValueError('source_manifest_changed')
        child=read(p)
        if ('experience_sources' in child or child.get('experience_task')!=ref['task_id']
                or not child.get('frozen_episode_memory') or not child.get('experience_memory')):
            raise ValueError('independent_bound_source_manifest_required')
        if healthy and child.get('cases') != [cases[len(packets)]]:
            raise ValueError('generation_debug_source_binding_mismatch')
        # This reuses full existing source report/request/result/memory identity
        # checks. A task label alone never authorizes an observation.
        packet=python_generation_packet(child,run_id+':source:'+ref['task_id'])
        packets.append(packet)
        provenance.append(dict(task_id=ref['task_id'],manifest_sha256=ref['manifest_sha256'],
            packet_sha256=digest(packet),memory=packet['context']['frozen_episode_memory'],
            evidence_ids=[e['evidence_id'] for e in packet['evidence']],
            part_ids_are_local_to_source=True,current_target_mapping=None))
    facts=[deepcopy(e) for p in packets for e in p['evidence']]
    if len(facts)>MAX_EVIDENCE:raise ValueError('multi_task_evidence_limit_no_silent_truncation')
    if len({e['evidence_id'] for e in facts})!=len(facts):raise ValueError('cross_source_evidence_identity_collision')
    for ref,packet in zip(refs,packets):
        if any(e.get('source_task')!=ref['task_id'] for e in packet['evidence']):raise ValueError('cross_source_task_mismatch')
    if any(p['existing_skills']!=packets[0]['existing_skills'] for p in packets):raise ValueError('source_primitive_contract_mismatch')
    context=deepcopy(packets[0]['context'])
    for name in ('frozen_episode_memory','initial_contact_observed_transitions','measured_stability_evidence_ids'):
        context.pop(name,None)
    if any(p['context'].get('python_failure_scope')!=packets[0]['context'].get('python_failure_scope') for p in packets):
        raise ValueError('different_generation_scopes_require_separate_skill')
    context['multi_source_provenance']=provenance
    context['multi_task_learning']=dict(version=1,source_tasks=generators,
        independent_task_count=2,admission_task=None if healthy else train[2],evaluation_tasks_exposed=False,
        source_to_target_part_mapping=None,all_observations_are_conditional=True,
        no_success_or_causality_invented=True,duplicate_trace_memory_views_are_not_new_trials=True,
        instructions='Compare matching failure stages across tasks; preserve contradictory cases and limits. Do not merge local part IDs or treat all moving bodies as necessary simultaneous supports. A new action remains a hypothesis requiring actual physics.')
    if healthy:
        context['multi_task_learning'].update(role_assignment_version=3 if manifest.get('multi_task_schema') == 'development_states_v3' else 2,
            positive_admission_scope='same_products_yaw90_NOT_new_product_transfer',
            healthy_training_contract_checks_hidden=True, role_assignment_after_collection=True)
        if manifest.get('multi_task_schema') == 'development_states_v3':
            context['multi_task_learning'].update(negative_contract_is_held_out_evaluation=False,
                negative_source_observations_may_overlap_generation=True)
    context['automatically_extracted_same_condition_pairs']=dict(kind='per_source_pairs_v1',
        source_local_only=True,causal_effect_established=False,
        sources=[dict(task_id=r['task_id'],summary=deepcopy(p['context']['automatically_extracted_same_condition_pairs']))
                 for r,p in zip(refs,packets)])
    context['multi_source_initial_contact_transitions']=[dict(task_id=r['task_id'],
        summary=deepcopy(p['context'].get('initial_contact_observed_transitions')))
        for r,p in zip(refs,packets)]
    result=dict(run_id=run_id,request_id=run_id+':synthesis',state_id='offline-multi-task-traces',
        operation='synthesize',context=context,evidence=facts,existing_skills=deepcopy(packets[0]['existing_skills']))
    if manifest.get('require_shared_failure_family') is True:
        gate=common_failure_families(result)
        if not gate['eligible_for_generation']:
            raise ValueError('no_measured_shared_failure_family_no_generation')
        result['context']['shared_failure_family_gate']=gate
    return result


def training_memory_sources(split,sources):
    """Gate cumulative-memory inputs before the existing freezer reads files."""
    train=[x['id'] for x in split['training']];evaluation=[x['id'] for x in split['evaluation']]
    validate_roles(train,evaluation)
    if not sources:raise ValueError('actual_training_memory_required')
    from orchestrator.explicit_action_memory import load_snapshot
    paths=[]
    for ref in sources:
        if ref.get('task_id') not in train or ref.get('purpose')!='training':raise ValueError('evaluation_memory_must_not_enter_training')
        p=checked(ref['path'],True)
        if sha(p)!=ref['sha256']:raise ValueError('training_memory_changed')
        snapshot=load_snapshot(p)
        if not snapshot['records']:raise ValueError('no_observations_in_training_snapshot')
        # The task identity comes from the frozen observed scope, not a file name.
        for record in snapshot['records']:
            if record['scope']['task_context']['task']['assembly_id']!=ref['task_id']:
                raise ValueError('training_memory_scope_task_mismatch')
        paths.append(p)
    return paths


def common_failure_families(packet):
    """Choose a measured shared failure stage, never infer a common cause.

    Input must be the bound multi-source packet above. Count original trace
    observations once; the same events' frozen-memory views are not extra trials.
    This is only a collection gate, not permission to register a repair skill.
    """
    scope=packet.get('context',{}).get('multi_task_learning',{})
    tasks=scope.get('source_tasks',[])
    if (scope.get('version')!=1 or len(tasks)!=2 or len(set(tasks))!=2
            or scope.get('evaluation_tasks_exposed') is not False):
        raise ValueError('bound_two_source_packet_required')
    priority=('dynamic_stability_failed','path_failed','initial_disconnected')
    counts={family:{task:[] for task in tasks} for family in priority}
    seen=set()
    for fact in packet['evidence']:
        task=fact.get('source_task')
        if task not in tasks:raise ValueError('unexpected_source_in_learning_gate')
        if fact.get('evidence_origin')!='episode_trace':continue
        if fact['evidence_id'] in seen:raise ValueError('duplicate_trace_evidence')
        seen.add(fact['evidence_id'])
        if fact.get('deadline_exceeded') is not False or fact.get('status') not in ('UNKNOWN','REJECTED'):continue
        phase=(fact.get('stability_diagnostics') or {}).get('phase')
        family=None
        if fact.get('path_success') is False:family='path_failed'
        elif (fact.get('path_success') is True and fact.get('stability_success') is False
                and fact.get('physically_checked_stability_failure') is True):
            family={'dynamic_failure':'dynamic_stability_failed','initial_disconnected':'initial_disconnected'}.get(phase)
        if family:counts[family][task].append(fact['evidence_id'])
    common=[f for f in priority if all(counts[f][t] for t in tasks)]
    return dict(common_families=common,first_family=common[0] if common else None,
        eligible_for_generation=bool(common),evidence_by_family=counts,
        independent_trace_observations=len(seen),shared_cause_established=False,
        no_common_family_action='STOP_WITHOUT_GENERATION; retain failed runs and fixed task split')
