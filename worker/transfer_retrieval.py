"""Source-scoped relative observations; no part ID mapping or verdict reuse."""
from copy import deepcopy


def select_references(references, request, max_records):
    if type(max_records) is not int or not 0<=max_records<=12:
        raise ValueError('invalid_retrieval_limit')
    from orchestrator.skill_program import _scoped_last
    last=_scoped_last(request)
    def stage(row):
        return ('path_failed' if row.get('path_success') is False else
                'stability_failed' if row.get('path_success') is True and row.get('stability_success') is False else
                'pass' if row.get('status')=='PASS' else 'unobserved')
    current=stage(last['feedback']) if last else None
    by_id={r['evidence_id']:r for r in references}
    def priority(row):
        parent=by_id.get(row.get('parent_failure_evidence_id'))
        applicable=current is not None and stage(parent or row)==current
        return (not applicable,not (parent and row.get('status')=='PASS'),
                abs(len(row.get('source_remaining_parts',[]))-len(request['remaining_parts'])),
                row['evidence_id'])
    selected=[];seen=set()
    for row in sorted(references,key=priority):
        chain=[];item=row;visiting=set()
        while item and item['evidence_id'] not in seen:
            if item['evidence_id'] in visiting:raise ValueError('cyclic_reference')
            visiting.add(item['evidence_id']);chain.append(item)
            parent=item.get('parent_failure_evidence_id')
            if parent and parent not in by_id:raise ValueError('missing_parent_reference')
            item=by_id.get(parent)
        if len(selected)+len(chain)>max_records:continue
        for item in reversed(chain):
            value=deepcopy(item)
            # Descriptive only; original raw observations and hashes unchanged.
            value['relative_observation']={'observed_stage':stage(item),
                'source_part_count':len(item.get('source_remaining_parts',[])),
                'support_count':len(item['source_choice']['support_ids']),
                'primitive':item['source_choice']['skill_id'],
                'current_part_id_mapping':None,'current_verdict':None,
                'direction_frame':'source_simulation_world_not_target_correspondence'}
            selected.append(value);seen.add(item['evidence_id'])
    return selected
