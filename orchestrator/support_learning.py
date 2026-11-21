"""Development-only support evidence and behavioral checks, never a verifier."""
from copy import deepcopy
from worker.task_memory import digest

INSTRUCTIONS = """
Support development protocol v1:
Read EVERY success, failure, LATE/ERROR and NOT_RUN row and the shared geometry.
The grid is human-declared diagnostic exploration, not model skill success.
It is development data, not admission or generalization evidence. The raw
instability list is the FIRST observed failure; it is not a sufficient support
set, a list of all future failures, or a causal explanation. Small displacement
differences and large center distances alone do not establish useful supports.
All legal remaining parts except mover/base may be support hypotheses, including
parts absent from the failure list. Use current geometry/contact relationships
and exact-state attempted choices. Preserve successful measured behavior where
applicable, with explicit limits; do not copy source part IDs or action tables.
If a best candidate is already attempted, consider another justified eligible
candidate before abstaining. A failure rules out neither that part globally nor
other poses/directions. Describe the actual bounded procedure, its stop/abstain
conditions, and uncertainty. The host supplies feedback after each action.
No generated code owns physics, state transitions, or registration. A changed
action, geometry calculation, renamed-ID check or replay is NOT physical PASS.
When recovery_protocol.action_scope is support_only, retain the current trigger's
mover, pose, primitive and both directions; propose only eligible support changes.
"""


def compact_bundle(bundle):
    if bundle.get('kind') != 'support_development_experiences_v1' or bundle.get('registration_evidence') is not False:
        raise ValueError('invalid_support_development_bundle')
    groups=[]
    for g in bundle['groups']:
        if g['phase'] != 'debug' or len(g['rows']) != g['declared_count']:
            raise ValueError('debug_only_complete_support_bundle_required')
        row={k:deepcopy(g[k]) for k in ('case_id','source_task','phase',
            'fixed_choice','declared_count','attempted_count','not_run','selection_policy',
            'generalization','diagnostic_origin','recommended_action','causal_effect_established',
            'runtime_observation_injection')}
        example=g['source_runtime_example']
        # The saved source happened to use one support pair. It is not fixed
        # across this grid and must not look like a recommended support choice.
        row['fixed_choice']={k:v for k,v in row['fixed_choice'].items()
                             if k not in ('support_ids','evidence_ids')}
        row['source_observation']={k:deepcopy(example[k]) for k in
            ('remaining_parts','base_part','max_grippers')}
        row['source_observation']['geometry']={k:deepcopy(example['geometry'][k]) for k in ('parts','relations')}
        row['source_observation']['pose_matrix']=deepcopy(example['last']['pose_matrix'])
        row['source_observation']['trigger_diagnostics']=deepcopy(example['last']['feedback'].get('stability_diagnostics'))
        row['rows']=[]
        for r in g['rows']:
            d=r.get('stability_diagnostics') or {}
            row['rows'].append({k:deepcopy(r[k]) for k in ('index','support_ids','raw_status','outcome',
                'path_success','stability_success','elapsed_seconds','evidence_id','source_result_sha256') if k in r} |
                dict(diagnostic_phase=d.get('phase'),first_failure_parts=deepcopy(d.get('dynamic_failure_parts')),
                     initial_disconnected_parts=d.get('initial_disconnected_parts'),
                     physics_steps=d.get('physics_steps'),failure_cause=None))
        groups.append(row)
    return dict(kind=bundle['kind'],groups=groups,registration_evidence=False,
                supplied_for='development_learning_only',new_physical_evidence=False)


def _rename(value, mapping):
    if isinstance(value,list):return [_rename(x,mapping) for x in value]
    if isinstance(value,dict):return {mapping.get(k,k):_rename(v,mapping) for k,v in value.items()}
    return mapping.get(value,value) if isinstance(value,str) else value


def deduplicate_observations(value):
    """Lossless references for exact duplicate trace/memory fields only."""
    rows=value.get('evidence',[]);by_id={r['evidence_id']:deepcopy(r) for r in rows}
    for row in rows:
        other=by_id.get(row.get('same_observation_as_event_id'))
        if other is None:continue
        shared=[k for k in ('choice','remaining_parts','stability_diagnostics','execution')
                if k in other and k in row and other[k]==row[k]]
        if shared:
            row['identical_fields_reference']=dict(evidence_id=other['evidence_id'],fields=shared)
            for k in shared:row.pop(k)
    return value


def share_runtime_examples(contract):
    """Provider-only references; local VM requests and actual ctx are unchanged."""
    pool={}
    for example in contract['examples']:
        context=example['runtime_input_example'];refs={}
        for key in ('geometry','pose_catalog'):
            if key not in context:continue
            subtree=context.pop(key);identity=digest(subtree);pool[identity]=subtree;refs[key]=identity
        example['runtime_input_shared_fields']=refs
    contract['shared_input_subtrees']=pool
    contract['shared_input_rule']='Expand runtime_input_shared_fields into runtime_input_example before interpreting ctx. References are provider formatting, never actual ctx keys.'
    return contract


def share_repeated_data(value):
    """Lossless provider formatting, retaining all observations and outcomes.

Only identical object/array subtrees of >=256 UTF-8 bytes are interned. These
references are input-document notation, never fields added to executable ctx.
"""
    import json
    from collections import Counter
    counts=Counter()
    def identity(node):
        return digest(node) if len(json.dumps(node,ensure_ascii=False,sort_keys=True,
                    separators=(',',':')).encode('utf-8')) >= 256 else None
    def collect(node):
        if not isinstance(node,(dict,list)):return
        key=identity(node)
        if key:counts[key]+=1
        for child in (node.values() if isinstance(node,dict) else node):collect(child)
    collect(value);pool={}
    def encode(node,root=False):
        if not isinstance(node,(dict,list)):return deepcopy(node)
        key=identity(node)
        if not root and key and counts[key]>1:
            if key not in pool:pool[key]=children(node)
            return {'$support_shared':key}
        return children(node)
    def children(node):
        return {k:encode(v) for k,v in node.items()} if isinstance(node,dict) else [encode(v) for v in node]
    encoded=encode(value,root=True)
    if pool:
        encoded['shared_data']=pool
        encoded['shared_data_rule']='Recursively replace an object containing only $support_shared with shared_data[that ID] to recover the exact input. These are document references, NOT Python ctx fields. No evidence or outcomes are omitted.'
    return encoded


def audit_support_program(program, packet):
    """Execute code against original/renamed data, retaining counterexamples.

Renaming checks legal current IDs and applicability, not a general proof of
label invariance: symmetric geometry can have several equally legal outputs.
Known outcomes remain source-state replay evidence, not fresh skill validation.
"""
    from orchestrator.python_skill_preflight import preflight_rows
    from orchestrator.python_skill_program import build_context, step_program, _response
    from orchestrator.python_skill_vm import VM
    from orchestrator.lesson_runtime import extension_for
    bundle=packet['context']['support_development_experiences']
    checks=[]
    for row in preflight_rows(packet):
        request=row['current_observation_request'];ctx,_=build_context(request,extension_for())
        step=step_program(program,request,extension=extension_for(),diagnose=True)
        choice=(step.get('response') or {}).get('choice');errors=[];matched=[]
        if choice is not None:
            if packet.get('context', {}).get('recovery_protocol', {}).get('action_scope') == 'support_only':
                from orchestrator.recovery_sequence import same_support_scope
                if not same_support_scope(ctx['last']['choice'],choice):
                    errors.append('outside_declared_support_recovery_scope')
            for group in bundle['groups']:
                source=group['source_runtime_example'];fixed=group['fixed_choice']
                if digest(request['task_context']) != group.get('source_task_context_sha256'):
                    continue
                # Exact geometry and full pose catalog only; IDs alone never bind a source.
                if any(ctx[k]!=source[k] for k in ('remaining_parts','base_part','max_grippers','geometry','pose_catalog')):
                    continue
                if any(choice[k]!=fixed[k] for k in ('part_id','pose_id','direction_id','skill_id','second_direction_id')):
                    continue
                for observed in group['rows']:
                    if sorted(observed['support_ids'])==sorted(choice['support_ids']):
                        matched.append(dict(evidence_id=observed.get('evidence_id'),outcome=observed['outcome']))
                        if observed['outcome'] in ('UNKNOWN','REJECTED') and observed['path_success'] is True and observed['stability_success'] is False:
                            errors.append('repeats_measured_failed_support_in_exact_development_state')
        renamed_checks=[]
        if choice is not None:
            for reverse in (False,True):
                ids=list(ctx['remaining_parts']);labels=list(reversed(ids)) if reverse else ids
                mapping={part:'probe_part_'+str(labels.index(part)) for part in ids}
                inverse={v:k for k,v in mapping.items()};renamed=_rename(ctx,mapping)
                def eligible(proposal):
                    try:_response(request,_rename(proposal,inverse),digest(program));return True
                    except (ValueError,KeyError,TypeError):return False
                try:
                    proposed=VM(program['source'],eligible,max_steps=50000).call('propose',[renamed])
                    if proposed is None:raise ValueError('renamed_input_abstained')
                    restored=_rename(proposed,inverse)
                    _response(request,restored,digest(program))
                    renamed_checks.append(dict(accepted=True,action_equivalent=
                        all(restored[k]==choice[k] for k in choice if k!='evidence_ids')))
                except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,RecursionError) as error:
                    renamed_checks.append(dict(accepted=False,error_type=type(error).__name__))
                    errors.append('source_id_or_renamed_input_contract_failure')
        checks.append(dict(case_id=row['case_id'],proposed_action=choice,known_source_outcomes=matched,
            renamed_checks=renamed_checks,errors=errors,accepted=not errors))
    return dict(kind='support_development_program_audit_v1',checks=checks,
        accepted=all(c['accepted'] for c in checks),api_calls=0,physical_calls=0,
        registration_evidence=False,new_physical_success=False,
        coverage='executed_development_branches_and_two_ID_relabelings_only')
