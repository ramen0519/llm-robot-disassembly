"""Explicit current skill review without compelling any skill selection."""
from copy import deepcopy

from orchestrator.conditional_plan import PLAN_KEYS
from orchestrator.conditional_plan_v3 import schema_v3, compile_plan as compile_v3
from worker.explicit_action_contract import digest

CONTRACT = 'conditional_plan_v4'
DISPOSITIONS = ('execute_now', 'defer', 'unavailable')


def schema_v4(request, library_ids, max_steps=64):
    schema = schema_v3(request, library_ids, max_steps)
    props = dict(skill_id={'type':'string', **({'enum':list(library_ids)} if library_ids else {})},
        disposition={'type':'string', 'enum':list(DISPOSITIONS)},
        reason={'type':'string', 'minLength':1, 'maxLength':400})
    schema['properties']['skill_review'] = dict(type='array', minItems=len(library_ids), maxItems=len(library_ids),
        items=dict(type='object', properties=props, required=list(props), additionalProperties=False))
    schema['required'].append('skill_review')
    return schema


def compile_plan(value, request, cards, max_steps=64, diagnostics=None):
    info = diagnostics if diagnostics is not None else {}
    info.update(contract=CONTRACT, phase='skill_review', step_index=None)
    if type(value) is not dict or set(value) != PLAN_KEYS | {'skill_review'}:
        raise ValueError('invalid_v4_plan_fields')
    by_id = {c['skill_id']:c for c in cards}
    if len(by_id) != len(cards):
        raise ValueError('duplicate_skill_cards')
    reviews = value['skill_review']
    if type(reviews) is not list or len(reviews) != len(by_id):
        raise ValueError('incomplete_skill_review')
    ids = []
    for row in reviews:
        if type(row) is not dict or set(row) != {'skill_id', 'disposition', 'reason'}:
            raise ValueError('invalid_skill_review_fields')
        identity = row['skill_id']
        if type(identity) is not str or identity not in by_id or identity in ids:
            raise ValueError('unknown_or_duplicate_skill_review')
        ids.append(identity)
        if (row['disposition'] not in DISPOSITIONS or type(row['reason']) is not str
                or not row['reason'].strip() or len(row['reason']) > 400):
            raise ValueError('invalid_skill_review_reason_or_disposition')
    clean = deepcopy(value)
    clean.pop('skill_review')
    base = {}
    try:
        plan = compile_v3(clean, request, list(by_id), max_steps, base)
    finally:
        info['base_compiler'] = base
    first = plan['steps'][0]
    for row in reviews:
        choice = by_id[row['skill_id']].get('currently_proposed_choice')
        selected = first['mode'] == 'registered_skill' and first['registered_skill_id'] == row['skill_id']
        expected = 'unavailable' if choice is None else 'execute_now' if selected else 'defer'
        if row['disposition'] != expected or selected and choice is None:
            raise ValueError('skill_review_action_mismatch')
        if selected and first['part_id'] != choice['part_id']:
            raise ValueError('selected_skill_proposal_mover_mismatch')
    plan.update(plan_contract=CONTRACT, skill_review=deepcopy(reviews),
        skill_review_is_model_self_report=True,
        plan_id='plan:' + digest(dict(contract=CONTRACT, value=value)))
    info.update(phase='validated', steps_compiled=len(plan['steps']),
                review_count=len(reviews), wire_value_sha256=digest(value))
    return plan
