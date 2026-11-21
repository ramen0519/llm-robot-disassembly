"""Finite, lossless provider encoding of untried explicit actions.

Slots are serialization addresses, not geometric preferences. Disjoint schema
rectangles cover every legal action except exact previously submitted actions.
No physics, API client, heuristic, search state or success cache lives here.
"""
from copy import deepcopy
from itertools import combinations
import json
import math

from worker.explicit_action_contract import (DIRECTIONS, SKILLS, PROTOCOL,
    canonical, validate_request, validate_choice, action_id)

PROVIDER_CONTRACT_V3 = 'explicit_action_provider_v3'
WIRE_FIELDS = {'request_id', 'state_id', 'decision', 'candidate_tail', 'evidence_selection'}
DECISION_FIELDS = {'part_id', 'pose_slot', 'force_slot', 'support_slot'}
MAX_SUPPORT_OPTIONS = 4096
MAX_SCHEMA_BYTES = 196608

WIRE_INSTRUCTIONS = (
    'Use the provider_v3 wire format. decision.part_id is the one moving part. '
    'Select pose_slot, force_slot and support_slot by their complete lookup tables; '
    'slots and ascending ID lists are serialization addresses, not recommendations. '
    'All legal untried combinations remain representable. Schema exclusions mark '
    'only exact actions already submitted in this state, not failed parts. '
    'candidate_tail encodes YOUR ranking of EVERY other candidate: start with the '
    'provided candidate_address_list minus decision.part_id; pick_1 is the zero-based '
    'index of your next preferred part in that remaining list, remove it, then use '
    'pick_2 in the shortened list, and continue. Every complete candidate permutation '
    'is expressible; do not treat slot zero as a recommendation. Do not return a '
    'candidate_order or choice field in the provider wire response. '
    'evidence_selection marks each offered evidence slot true only if used; false '
    'otherwise. Support sets are post-removal stability conditions. Changing only '
    'supports after path failure repeats the same path conditions, although the '
    'support variant remains legal. Path success plus stability failure may motivate '
    'a different support set. These are stage semantics, not a supplied ranking or '
    'a failure-cause inference. Only the actual verifier may accept a transition.'
)


def _object(properties):
    return {'type': 'object', 'additionalProperties': False,
            'properties': properties, 'required': list(properties)}


def _integer_set(values):
    values = sorted(set(values))
    if not values:
        raise ValueError('empty_provider_integer_domain')
    ranges, start, end = [], values[0], values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            ranges.append({'type': 'integer', 'minimum': start, 'maximum': end})
            start = end = value
    ranges.append({'type': 'integer', 'minimum': start, 'maximum': end})
    return ranges[0] if len(ranges) == 1 else {'anyOf': ranges}


def build_domain(request):
    request = validate_request(request)
    if request['attempted_action_ids'] and 'attempted_actions' not in request:
        raise ValueError('provider_requires_readable_attempted_actions')
    parts = request['candidate_ids']
    support_parts = sorted(set(request['remaining_parts']) - {request['base_part']})
    maximum = min(request['max_grippers'] - 1, len(support_parts))
    if sum(math.comb(len(support_parts), size) for size in range(maximum + 1)) > MAX_SUPPORT_OPTIONS:
        raise ValueError('provider_support_domain_too_large')
    supports = [list(items) for size in range(maximum + 1)
                for items in combinations(support_parts, size)]
    forces = []
    for skill in SKILLS:
        for direction in DIRECTIONS:
            seconds = [None] if skill == 'single_force_v1' else [other for other in DIRECTIONS
                if sum(a*b for a, b in zip(DIRECTIONS[direction], DIRECTIONS[other])) == 0]
            for second in seconds:
                forces.append(dict(skill_id=skill, direction_id=direction, second_direction_id=second))
    support_index = {tuple(items): i for i, items in enumerate(supports)}
    force_index = {(row['skill_id'], row['direction_id'], row['second_direction_id']): i
                   for i, row in enumerate(forces)}
    poses = request['pose_catalog']
    pose_index = {p['pose_id']: i for i, p in enumerate(poses)}
    # Physical action identity ignores pose labels. Exclude all exact matrix
    # aliases of a tried pose, while keeping every pose entry in the lookup.
    aliases = {}
    for index, pose in enumerate(poses):
        aliases.setdefault(canonical(pose['matrix']), []).append(index)
    forbidden = set()
    for row in request.get('attempted_actions', []):
        pose_slots = aliases[canonical(poses[pose_index[row['pose_id']]]['matrix'])]
        force = force_index[(row['skill_id'], row['direction_id'], row['second_direction_id'])]
        support = support_index[tuple(sorted(row['support_ids']))]
        forbidden.update((row['part_id'], pose, force, support) for pose in pose_slots)
    rectangles = []
    all_poses, all_forces = set(range(len(poses))), set(range(len(forces)))
    full_count = 0

    def add(part, pose_slots, force_slots, support_slots):
        if pose_slots and force_slots and support_slots:
            rectangles.append(dict(part_id=part, pose_slots=sorted(pose_slots),
                                   force_slots=sorted(force_slots), support_slots=sorted(support_slots)))

    for part in parts:
        legal_supports = {i for i, values in enumerate(supports) if part not in values}
        full_count += len(all_poses) * len(all_forces) * len(legal_supports)
        tried = {(pose, force, support) for p, pose, force, support in forbidden if p == part}
        tried_poses = {pose for pose, _, _ in tried}
        add(part, all_poses - tried_poses, all_forces, legal_supports)
        for pose in sorted(tried_poses):
            tried_forces = {force for p, force, _ in tried if p == pose}
            add(part, {pose}, all_forces - tried_forces, legal_supports)
            for force in sorted(tried_forces):
                used_supports = {support for p, f, support in tried if p == pose and f == force}
                add(part, {pose}, {force}, legal_supports - used_supports)
    available = sum(len(r['pose_slots']) * len(r['force_slots']) * len(r['support_slots']) for r in rectangles)
    if available != full_count - len(forbidden):
        raise ValueError('provider_domain_coverage_mismatch')
    return dict(candidate_address_list=parts, pose_catalog=deepcopy(poses),
        force_catalog=forces, support_catalog=supports, evidence_ids=sorted(request['evidence_ids']),
        forbidden_action_slots=[list(row) for row in sorted(forbidden)], rectangles=rectangles,
        full_action_serializations=full_count, forbidden_action_serializations=len(forbidden),
        available_action_serializations=available)


def schema_statistics(schema):
    totals = dict(property_count=0, enum_values=0, string_characters=0, max_object_depth=0)
    def visit(value, depth):
        if isinstance(value, dict):
            current = depth + int(value.get('type') == 'object')
            totals['max_object_depth'] = max(totals['max_object_depth'], current)
            properties = value.get('properties', {})
            totals['property_count'] += len(properties)
            totals['string_characters'] += sum(len(k) for k in properties)
            totals['string_characters'] += sum(len(k) for k in value.get('$defs', {}))
            enums = value.get('enum', [])
            totals['enum_values'] += len(enums)
            totals['string_characters'] += sum(len(v) for v in enums if isinstance(v, str))
            if isinstance(value.get('const'), str):
                totals['string_characters'] += len(value['const'])
            for child in value.values():
                visit(child, current)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth)
    visit(schema, 0)
    totals['schema_bytes'] = len(canonical(schema).encode('utf-8'))
    return totals


def _schema(request, domain):
    branches = [_object({'part_id': {'type': 'string', 'enum': [row['part_id']]},
        'pose_slot': _integer_set(row['pose_slots']), 'force_slot': _integer_set(row['force_slots']),
        'support_slot': _integer_set(row['support_slots'])}) for row in domain['rectangles']]
    if not branches:
        raise ValueError('provider_action_space_exhausted')
    count = len(domain['candidate_address_list'])
    schema = _object({'request_id': {'type': 'string', 'enum': [request['request_id']]},
        'state_id': {'type': 'string', 'enum': [request['state_id']]},
        'decision': {'anyOf': branches},
        'candidate_tail': _object({'pick_' + str(i): {'type': 'integer', 'minimum': 0,
            'maximum': count-i-1} for i in range(1, count)}),
        'evidence_selection': _object({'e'+str(i): {'type': 'boolean'} for i in range(len(domain['evidence_ids']))})})
    stats = schema_statistics(schema)
    if (stats['property_count'] > 5000 or stats['enum_values'] > 1000
            or stats['string_characters'] > 120000 or stats['max_object_depth'] > 10
            or stats['schema_bytes'] > MAX_SCHEMA_BYTES):
        raise ValueError('provider_schema_limit_exceeded')
    return schema


def response_schema_v3(request):
    return _schema(request, build_domain(request))


def provider_input_v3(request):
    domain = build_domain(request)
    schema = _schema(request, domain)
    lookup = {key: deepcopy(domain[key]) for key in
              ('candidate_address_list', 'force_catalog', 'support_catalog', 'forbidden_action_slots')}
    lookup['pose_slots'] = [dict(pose_slot=i, pose_id=p['pose_id']) for i, p in enumerate(domain['pose_catalog'])]
    lookup['evidence_slots'] = {'e'+str(i): identity for i, identity in enumerate(domain['evidence_ids'])}
    lookup['slot_semantics'] = 'Zero-based lookup addresses only. No geometry priority or recommended choices.'
    audit = {key: domain[key] for key in ('full_action_serializations', 'forbidden_action_serializations',
                                        'available_action_serializations')}
    audit.update(schema_statistics(schema), branch_count=len(domain['rectangles']),
                 pose_entries_preserved=len(domain['pose_catalog']), candidate_parts_preserved=len(domain['candidate_address_list']),
                 exact_action_exclusions_only=True, path_equivalent_support_variants_allowed=True)
    return lookup, schema, audit


def _slot(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError('invalid_provider_slot')
    return value


def expand_provider_decision_v3(request, value):
    domain = build_domain(request)
    if type(value) is not dict or set(value) != WIRE_FIELDS:
        raise ValueError('invalid_provider_wire_fields')
    if value['request_id'] != request['request_id'] or value['state_id'] != request['state_id']:
        raise ValueError('stale_or_invalid_response_identity')
    decision = value['decision']
    if type(decision) is not dict or set(decision) != DECISION_FIELDS:
        raise ValueError('invalid_provider_decision_fields')
    part = decision['part_id']
    if type(part) is not str or part not in domain['candidate_address_list']:
        raise ValueError('invalid_selected_part')
    pose = _slot(decision['pose_slot'], len(domain['pose_catalog']))
    force = _slot(decision['force_slot'], len(domain['force_catalog']))
    support = _slot(decision['support_slot'], len(domain['support_catalog']))
    count = len(domain['candidate_address_list'])
    tail = value['candidate_tail']
    if type(tail) is not dict or set(tail) != {'pick_'+str(i) for i in range(1, count)}:
        raise ValueError('invalid_provider_permutation_code')
    pool = [candidate for candidate in domain['candidate_address_list'] if candidate != part]
    order = [part]
    for index in range(1, count):
        order.append(pool.pop(_slot(tail['pick_'+str(index)], len(pool))))
    evidence = value['evidence_selection']
    keys = ['e'+str(i) for i in range(len(domain['evidence_ids']))]
    if type(evidence) is not dict or set(evidence) != set(keys) or any(type(evidence[k]) is not bool for k in keys):
        raise ValueError('invalid_provider_evidence_selection')
    choice = dict(part_id=part, pose_id=domain['pose_catalog'][pose]['pose_id'],
        **domain['force_catalog'][force], support_ids=list(domain['support_catalog'][support]),
        evidence_ids=[identity for key, identity in zip(keys, domain['evidence_ids']) if evidence[key]])
    validate_choice(request, choice)
    # The ordinary worker validator still checks this returned identity against
    # attempted_action_ids. Do not silently repair unsupported provider output.
    return dict(schema_version=1, protocol=PROTOCOL, request_id=request['request_id'], state_id=request['state_id'],
                candidate_order=order, choice=choice, action_id=action_id(request, choice), fallback_reason=None, metadata={})


def encode_provider_decision_v3(request, response):
    """Inverse serialization helper for offline fixtures and audit projections."""
    domain = build_domain(request)
    order, choice = response['candidate_order'], response['choice']
    if (len(order) != len(domain['candidate_address_list']) or len(set(order)) != len(order)
            or set(order) != set(domain['candidate_address_list']) or order[0] != choice['part_id']):
        raise ValueError('invalid_candidate_permutation')
    validate_choice(request, choice)
    pool = [part for part in domain['candidate_address_list'] if part != order[0]]
    tail = {}
    for index, part in enumerate(order[1:], 1):
        tail['pick_'+str(index)] = pool.index(part)
        pool.remove(part)
    force = {key: choice[key] for key in ('skill_id', 'direction_id', 'second_direction_id')}
    return dict(request_id=request['request_id'], state_id=request['state_id'],
        decision=dict(part_id=order[0], pose_slot=[p['pose_id'] for p in domain['pose_catalog']].index(choice['pose_id']),
                      force_slot=domain['force_catalog'].index(force), support_slot=domain['support_catalog'].index(sorted(choice['support_ids']))),
        candidate_tail=tail, evidence_selection={'e'+str(i): identity in choice['evidence_ids']
                                               for i, identity in enumerate(domain['evidence_ids'])})
