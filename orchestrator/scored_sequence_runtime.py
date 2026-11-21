"""Bind explicitly frozen registered scorers to relational search results.

Search does not grant execution rights. Both an eligible provenance card and
an independently checked, explicitly loaded bridge are required. No API here.
"""
from copy import deepcopy

from .relational_runtime import size
from .scored_sequence_bridge import validate_bridge, compact_bridge
from .scored_order_skill import feature_rows
from worker.task_memory import digest

FROZEN_PROTOCOL = 'frozen_scored_sequence_bridges/v1'


def load_bridges(value):
    if isinstance(value, (list, tuple)):
        rows = value
    elif type(value) is dict and value.get('protocol') == FROZEN_PROTOCOL:
        if value.get('content_sha256') != digest({k: v for k, v in value.items() if k != 'content_sha256'}):
            raise ValueError('scored_bridge_library_hash_mismatch')
        rows = value.get('entries')
        if type(rows) is not list:
            raise ValueError('invalid_scored_bridge_library')
    else:
        raise ValueError('invalid_scored_bridge_library')
    result = [validate_bridge(row) for row in rows]
    if len(result) > 4 or len({b['skill_id'] for b in result}) != len(result):
        raise ValueError('duplicate_or_excess_scored_bridges')
    return result


def searched_bridges(runtime, bridges, envelope, excluded_products):
    """Use the same relevance search while keeping source evidence eligible."""
    if not bridges:
        return [], [], dict(matches=[], excluded=[])
    if runtime is None:
        raise ValueError('scored_bridges_require_relational_library')
    excluded = set(excluded_products)
    by_card, audit = {}, dict(matches=[], excluded=[])
    # Check data availability before spending a model request. This is feature
    # construction only, not VM execution or a physical feasibility prediction.
    cad, request = envelope['features']['cad'], envelope['request']
    try:
        contacts = cad.get('relations', {}).get('contact_edges')
        if type(contacts) is not list:
            raise ValueError('contact_graph_not_observed')
        if len(request['assembled_parts']) <= 2:
            raise ValueError('native_terminal_rule_owns_last_two')
        feature_rows(dict(parts=cad['parts'], contact_edges=contacts), request['assembled_parts'])
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as error:
        audit['excluded'] = [dict(skill_id=b['skill_id'], reason='scorer_input_unavailable',
                                  diagnostic=str(error)[:180]) for b in bridges]
        return [], [], audit
    blocked = runtime.excluded_ids(excluded)
    for bridge in bridges:
        if excluded.intersection(bridge['source_task_ids']):
            audit['excluded'].append(dict(skill_id=bridge['skill_id'], reason='evaluation_task_excluded'))
            continue
        candidates = [c for c in runtime.cards if c['kind'] == 'skill'
            and c['status'] == 'REGISTERED_SCOPED'
            and c['origin'].get('source_id') == bridge['skill_id']
            and c['origin'].get('source_sha256') == bridge['original_registry_sha256']]
        eligible = []
        for card in candidates:
            proof = card.get('provenance_validation', {})
            if (card['card_id'] not in blocked
                    and proof.get('status') == 'VERIFIED_HISTORICAL_REGISTRATION'
                    and proof.get('registration_path') == bridge['original_registry_path']
                    and proof.get('registration_sha256') == bridge['original_registry_sha256']
                    and proof.get('registration_content_sha256') == digest(bridge['registration'])
                    and proof.get('source_sha256') == bridge['program_sha256']
                    and set(proof.get('training_products', [])) | set(proof.get('validation_products', []))
                        == set(bridge['source_task_ids'])
                    and card['content'] == bridge['registration']):
                eligible.append(card)
        if len(eligible) != 1:
            audit['excluded'].append(dict(skill_id=bridge['skill_id'],
                reason='registered_relational_provenance_missing_ambiguous_or_excluded'))
            continue
        by_card[eligible[0]['card_id']] = bridge
    matches = runtime.search(envelope['request'], cad=envelope['features']['cad'],
        kind='skill', limit=4, excluded_products=excluded_products, return_card_ids=list(by_card))
    offers, views = [], []
    for match in matches['results']:
        identity = match['card_id']
        bridge = by_card[identity]
        offered = compact_bridge(bridge)
        offered['relational_card_id'] = identity
        offers.append(offered)
        # Source is included once, intact, in sequence_skills. This small
        # relational view retains its identity, evidence links and scope.
        card = runtime.by_id[identity]
        evidence = list(card.get('evidence_ids', []))
        view = dict(card=dict(card_id=identity, kind='skill', status=card['status'],
            dataset_role=card['dataset_role'], fixture=card['fixture'],
            origin={k: deepcopy(card['origin'].get(k)) for k in
                    ('source_id', 'source_path', 'source_sha256', 'author_model')},
            evidence_ids=evidence[:4], evidence_ids_omitted=max(0, len(evidence)-4),
            evidence_ids_sha256=digest(evidence),
            content=dict(skill_id=bridge['skill_id'], description=offered.get('description'),
                source_location='sequence_skills.program.source',
                original_registration_status='REGISTERED_SCOPED',
                current_physics_verified=False),
            projection=dict(original_immutable=True, code_in_full_executable_card=True,
                evidence_details_available_via_read=True)),
            execution_authorized=False, current_physics_verified=False,
            requires_fresh_physics=True, current_part_mapping=None,
            library_sha256=runtime.fingerprint)
        compact_match = {k: deepcopy(v) for k, v in match.items() if k != 'channels'}
        compact_match['executable_skill_id'] = bridge['skill_id']
        views.append(dict(match=compact_match, knowledge=view))
        audit['matches'].append(dict(skill_id=bridge['skill_id'], card_id=identity,
            matched_evidence_card_id=match.get('matched_evidence_card_id'),
            original_registry_sha256=bridge['original_registry_sha256'],
            bridge_sha256=bridge['content_sha256'], selection_forced=False))
    found = {x['skill_id'] for x in offers}
    for bridge in by_card.values():
        if bridge['skill_id'] not in found:
            audit['excluded'].append(dict(skill_id=bridge['skill_id'], reason='no_relevant_registered_card'))
    audit['retrieval'] = matches['audit']
    return offers, views, audit


def reserve_registered_views(packet, required_views, max_bytes):
    """Keep searched executable references; discard whole optional references."""
    if not required_views:
        return packet
    value = deepcopy(packet)
    required_ids = {v['match']['card_id'] for v in required_views}
    restored = [r for r in value['excluded'] if r.get('card_id') in required_ids]
    value['excluded'] = [r for r in value['excluded'] if r.get('card_id') not in required_ids]
    value['excluded_count'] -= len(restored)
    value['restored_after_registered_reservation'] = [r['card_id'] for r in restored]
    old = [v for v in value['cards'] if v['match']['card_id'] not in required_ids]
    value['cards'] = deepcopy(required_views)
    value['card_ids'] = [v['match']['card_id'] for v in value['cards']]
    value['registered_code_location'] = 'sequence_skills.program.source'
    if size(value) > max_bytes:
        raise ValueError('registered_scored_reference_exceeds_input_limit')
    for view in old:
        candidate = deepcopy(value)
        candidate['cards'].append(view)
        candidate['card_ids'].append(view['match']['card_id'])
        if size(candidate) <= max_bytes - 1024:
            value = candidate
        else:
            value['excluded_count'] += 1
            if len(value['excluded']) < 6:
                value['excluded'].append(dict(card_id=view['match']['card_id'],
                                             reason='registered_skill_input_reservation'))
    if size(value) > max_bytes:
        raise ValueError('registered_scored_reference_exceeds_input_limit')
    return value
