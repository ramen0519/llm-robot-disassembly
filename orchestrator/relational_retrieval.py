"""Read-only retrieval over relational knowledge cards.

Scores describe relevance, never physical feasibility or permission to execute a
skill. Three separately ranked channels are fused without model/API calls. Raw
product/part IDs are used only for provenance, explicit exclusion and tie-breaks.
"""
from copy import deepcopy
import json
import math
import re


POLICY = 'relational_three_channel_v1'
CHANNELS = ('shape', 'relations', 'situation')
KINDS = ('experience', 'lesson', 'skill')
RRF_OFFSET = 60
MAX_NEIGHBORS = 32
MAX_LINK_DEPTH = 6
_UNKNOWN = {'', 'unknown', 'unspecified', 'not_observed', 'missing', 'unverified', 'null', 'none'}
_UNITS = {'m': 1., 'meter': 1., 'metre': 1., 'cm': .01, 'mm': .001,
          'millimeter': .001, 'millimetre': .001, 'inch': .0254, 'in': .0254}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _known(value):
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        return text not in _UNKNOWN and not any(t in text for t in ('unknown', 'unconfirmed', 'not_confirmed', 'unobserved'))
    if type(value) in (int, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return any(_known(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_known(item) for item in value)
    return True


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value, *, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_number(v) for v in value):
        return None
    if positive and (min(value) < 0 or max(value) <= 0):
        return None
    return list(value)


def _ratio(shape):
    shape = _mapping(shape)
    value = _vector(shape.get('ratio_sorted'), positive=True)
    if value is None:
        value = _vector(shape.get('extents_xyz'), positive=True)
    if value is None:
        return None
    longest = max(value)
    return sorted(v / longest for v in value)


def _tokenize(value):
    words = set(re.findall(r'\b[a-zA-Z가-힣][a-zA-Z가-힣_-]*\b', str(value).lower()))
    # Deliberately no raw ID fields, numeric labels, code-token matching or embedding calls.
    for korean, english in (('안정성', 'stability'), ('지지', 'support'), ('경로', 'path'),
                            ('실패', 'failed'), ('복구', 'recovery'), ('접촉', 'contact')):
        if korean in words:
            words.add(english)
    return words


def _text(card):
    content = _mapping(card.get('content'))
    values = []
    for obj in (card, content):
        for key in ('title', 'description', 'statement', 'scope', 'limitation', 'text', 'trigger'):
            if isinstance(obj.get(key), str):
                values.append(obj[key])
    return ' '.join(values)[:16384]


def _trigger(card, context):
    content = _mapping(card.get('content'))
    for obj in (context, _mapping(context.get('conditions')), card, content):
        for key in ('trigger', 'trigger_kind', 'failure_phase', 'failure_stage'):
            if _known(obj.get(key)):
                return obj[key]
    # Do not turn generic False, timeout, parts_fall or a narrative into a failure cause.
    return None


def _channel(parts):
    if not parts:
        return None
    return dict(distance=sum(value for _, value in parts) / len(parts),
                matched_fields=[key for key, _ in parts],
                feature_distances={key: value for key, value in parts})


def _shape(query, context):
    qshape, shape = _mapping(query.get('shape')), _mapping(context.get('shape'))
    parts, missing = [], []
    a, b = _ratio(qshape), _ratio(shape)
    if a is not None and b is not None:
        parts.append(('bbox_sorted_ratio', sum(abs(x-y) for x, y in zip(a, b)) / 3))
    else:
        missing.append('shape.bbox_sorted_ratio')
    a, b = _vector(qshape.get('extents_xyz'), positive=True), _vector(shape.get('extents_xyz'), positive=True)
    qa, qb = qshape.get('length_unit'), shape.get('length_unit')
    ua = _UNITS.get(str(qa).lower()) if _known(qa) else None
    ub = _UNITS.get(str(qb).lower()) if _known(qb) else None
    if a is not None and b is not None and min(a) > 0 and min(b) > 0 and ua and ub:
        ratios = [abs(math.log((x*ua)/(y*ub))) for x, y in zip(sorted(a), sorted(b))]
        distance = sum(ratios) / 3
        parts.append(('absolute_size_known_units', distance / (1+distance)))
    else:
        missing.append('shape.absolute_size_comparable_units')
    return _channel(parts), missing


def _neighbors(context):
    neighbors = context.get('neighbors')
    if not isinstance(neighbors, list):
        return None
    descriptors = []
    for entry in neighbors:
        if not isinstance(entry, dict):
            continue
        ratio = _ratio(entry.get('shape'))
        if ratio is None:
            # Old scalar extent forms may be upgraded by the card builder; no guessing here.
            continue
        descriptors.append(tuple(ratio))
    # Content sorting, never a source part ID. No empty list -> 'perfect match' inference.
    return sorted(descriptors)[:MAX_NEIGHBORS] or None


def _relations(query, context):
    qa, qb = _mapping(query.get('topology')), _mapping(context.get('topology'))
    parts, missing = [], []
    for key in ('contact_degree', 'degree_fraction', 'component_delta', 'remainder_isolated_nodes'):
        a, b = qa.get(key), qb.get(key)
        if _number(a) and _number(b):
            distance = abs(a-b) / (1+max(abs(a), abs(b)))
            parts.append(('topology.' + key, distance))
        else:
            missing.append('topology.' + key)
    a, b = _neighbors(query), _neighbors(context)
    if a is not None and b is not None:
        def nearest(left, right):
            return sum(min(sum(abs(x-y) for x, y in zip(p, q))/3 for q in right)
                       for p in left) / len(left)
        parts.append(('neighbor_shape_distribution', (nearest(a,b) + nearest(b,a))/2))
    else:
        missing.append('neighbors.shape_distribution')
    # Radial distances retain coarse local spacing, not full arrangement/direction.
    # Common per-assembly axis labels do not establish aligned CAD frames.
    for key in ('neighbor_radial_distribution',):
        def radial(ctx):
            scale = _vector(_mapping(ctx.get('shape')).get('extents_xyz'), positive=True)
            if scale is None:
                return None
            values = []
            for item in ctx.get('neighbors') or []:
                if not isinstance(item, dict):
                    continue
                pos = _vector(item.get('center_delta_xyz'))
                if pos is not None:
                    r = math.sqrt(sum(v*v for v in pos)) / max(scale)
                    values.append(r/(1+r))
            return sorted(values)[:MAX_NEIGHBORS] or None
        a, b = radial(query), radial(context)
        if a is not None and b is not None:
            d1 = sum(min(abs(x-y) for y in b) for x in a)/len(a)
            d2 = sum(min(abs(x-y) for x in a) for y in b)/len(b)
            parts.append((key, (d1+d2)/2))
        else:
            missing.append('neighbors.relative_distance_distribution')
    return _channel(parts), missing


def _situation(query, context, card, requested_trigger, words):
    a, b = _mapping(query.get('conditions')), _mapping(context.get('conditions'))
    a, b = deepcopy(a), deepcopy(b)
    for conditions in (a, b):
        known = conditions.get('known_fields', [])
        if 'fixed_base_present' not in conditions and 'fixed_base' in known:
            base = conditions.get('fixed_base')
            if base == 'none':
                conditions['fixed_base_present'] = False
            elif _known(base):
                conditions['fixed_base_present'] = True
    parts, missing, differences = [], [], []
    for key in ('verifier_domain', 'verifier_profile', 'support_policy', 'fixed_base_present'):
        qa, qb = a.get(key), b.get(key)
        if _known(qa) and _known(qb):
            equal = qa == qb
            parts.append(('conditions.' + key, 0. if equal else 1.))
            if not equal:
                differences.append(dict(field=key, current=deepcopy(qa), historical=deepcopy(qb)))
        else:
            missing.append('conditions.' + key)
    # A raw fixed-base ID is local to a product. Only presence can be compared.
    # Missing/null is ambiguous and must be disambiguated by the card importer.
    if 'fixed_base_present' not in a or 'fixed_base_present' not in b:
        if _known(a.get('fixed_base')) and _known(b.get('fixed_base')):
            parts.append(('conditions.fixed_base_known_present', 0.))
    if _known(requested_trigger):
        previous = _trigger(card, context)
        if _known(previous):
            equal = requested_trigger == previous
            parts.append(('observed_trigger', 0. if equal else 1.))
            if not equal:
                differences.append(dict(field='trigger', current=requested_trigger, historical=previous))
        else:
            missing.append('observed_trigger')
    if a.get('pose') is None or b.get('pose') is None:
        missing.append('conditions.pose')
    elif a['pose'] != b['pose']:
        differences.append(dict(field='pose', current='recorded', historical='different_recorded_pose',
                                meaning='transform_equivalence_not_established'))
    if words:
        matched = sorted(words & _tokenize(_text(card)))
        if matched:
            parts.append(('text_terms:' + ','.join(matched)[:160], 1-len(matched)/len(words)))
        # No token match does not exclude a structural match.
    return _channel(parts), missing, differences


def _pair(query, context, card, requested_trigger, words, geometry=None):
    if geometry is None:
        shape, ms = _shape(query, context)
        relations, mr = _relations(query, context)
    else:
        shape, ms, relations, mr = deepcopy(geometry)
    situation, mt, differences = _situation(query, context, card, requested_trigger, words)
    channels = {key: val for key, val in zip(CHANNELS, (shape, relations, situation)) if val is not None}
    return dict(channels=channels, missing_features=sorted(set(ms+mr+mt)), condition_differences=differences)


def _origins(card):
    origin = _mapping(card.get('origin'))
    key = card.get('evidence_group_id') or origin.get('dedup_key') or card.get('source_group_id')
    if isinstance(key, str) and key:
        return key
    # Only an exact location/hash pair identifies duplicate evidence. Product/part
    # alone must never collapse successful and failed attempts under different conditions.
    if origin.get('path') and origin.get('record_pointer') is not None and origin.get('source_hash'):
        return json.dumps([origin['path'], origin['record_pointer'], origin['source_hash']], sort_keys=True)
    return card['card_id']


def source_products(card):
    """All known generation exposure, not only the model's selected citations."""
    context = _mapping(card.get('context'))
    proof = _mapping(card.get('provenance_validation'))
    products = set()
    for values in (card.get('source_products'), context.get('source_products'),
                   proof.get('training_products'), proof.get('validation_products')):
        if isinstance(values, (list, tuple)):
            products.update(p for p in values if isinstance(p, str))
    return products


def search_cards(cards, query, limit=8, offset=0, exclude_products=(), allowed_card_ids=None):
    """Retrieve contextual card families without granting execution permission.

    query may be a context, or {'context': context, 'kind': ..., 'text': ...,
    'trigger': ..., 'channels': ['shape', 'relations', 'situation']}. A
    lesson/skill's source context is resolved only through
    canonical evidence_ids in the supplied (already frozen) collection.
    """
    if type(limit) is not int or not 1 <= limit <= 32 or type(offset) is not int or offset < 0:
        raise ValueError('invalid_relational_page')
    if not isinstance(query, dict):
        raise ValueError('invalid_relational_query')
    kind = query.get('kind', 'all')
    if kind not in ('all',) + KINDS:
        raise ValueError('invalid_relational_kind')
    return_ids = query.get('return_card_ids')
    if return_ids is not None:
        if (type(return_ids) is not list or any(type(v) is not str for v in return_ids)
                or len(return_ids) != len(set(return_ids))):
            raise ValueError('invalid_relational_return_ids')
        return_ids = set(return_ids)
    channels = query.get('channels', list(CHANNELS))
    if (not isinstance(channels, list) or not channels or len(channels) > len(CHANNELS)
            or any(not isinstance(c, str) or c not in CHANNELS for c in channels)
            or len(set(channels)) != len(channels)):
        raise ValueError('invalid_relational_channels')
    context = _mapping(query.get('context', query))
    requested_trigger = query.get('trigger', context.get('trigger'))
    words = _tokenize(query.get('text', ''))
    allowed = None if allowed_card_ids is None else set(allowed_card_ids)
    excluded_products = set(exclude_products)
    indexed, excluded, duplicates = {}, [], []
    for card in cards:
        if not isinstance(card, dict) or not isinstance(card.get('card_id'), str) or card.get('kind') not in KINDS:
            raise ValueError('invalid_relational_card')
        identity = card['card_id']
        if identity in indexed:
            if indexed[identity] != card:
                raise ValueError('conflicting_relational_card_id')
            duplicates.append(identity)
            continue
        indexed[identity] = card
    # Both returned cards and their evidence must be from the permitted set.
    excluded = {identity for identity, card in indexed.items()
                if (allowed is not None and identity not in allowed)
                or card.get('fixture') is True
                or _mapping(card.get('context')).get('product_id') in excluded_products
                or bool(source_products(card) & excluded_products)}
    # A lesson trained on an excluded product can leak that product's answer in
    # its own text even when its closest remaining evidence belongs to another
    # product. Exclude the whole dependent document, not just the anchor score.
    changed = True
    while changed:
        dependencies = {identity for identity, card in indexed.items()
                        if identity not in excluded and any(eid in excluded for eid in card.get('evidence_ids') or [])}
        changed = bool(dependencies)
        excluded.update(dependencies)
    eligible = {identity: card for identity, card in indexed.items() if identity not in excluded}
    excluded = sorted(excluded)

    context_cache = {}
    def evidence_contexts(identity, seen=(), depth=0):
        if identity in seen or depth > MAX_LINK_DEPTH or identity not in eligible:
            return []
        if identity in context_cache:
            return context_cache[identity]
        card = eligible[identity]
        current = _mapping(card.get('context'))
        values = []
        if card['kind'] == 'experience':
            # A selected step cannot borrow its parent episode's initial geometry:
            # those are different states even when they share one evidence group.
            return [(identity, current)] if current else []
        for eid in card.get('evidence_ids') or []:
            if isinstance(eid, str):
                values.extend(evidence_contexts(eid, seen+(identity,), depth+1))
        # A non-experience context is usable only when no evidence context exists;
        # it never establishes validation or fills a missing original observation.
        if not values and current and card['kind'] != 'experience':
            values.append((identity, current))
        unique = {}
        for eid, value in values:
            unique.setdefault(eid, value)
        context_cache[identity] = list(unique.items())
        return context_cache[identity]

    scored, unmatched, geometry_cache = [], [], {}
    for identity, card in eligible.items():
        if kind != 'all' and card['kind'] != kind:
            continue
        # A return shortlist never makes its evidence ineligible. In contrast,
        # allowed_card_ids controls both the returned records and their sources.
        if return_ids is not None and identity not in return_ids:
            continue
        possible = evidence_contexts(identity)
        if not possible:
            possible = [(identity, _mapping(card.get('context')))]
        choices = []
        for anchor, historical in possible:
            if anchor not in geometry_cache:
                sh, ms = _shape(context, historical)
                rel, mr = _relations(context, historical)
                geometry_cache[anchor] = (sh, ms, rel, mr)
            row = _pair(context, historical, card, requested_trigger, words, geometry_cache[anchor])
            row['channels'] = {key: value for key, value in row['channels'].items() if key in channels}
            if row['channels']:
                # Prefer context that matches more independent channels, then the
                # mean distance. Multiple citations do not add relevance votes.
                distance = sum(x['distance'] for x in row['channels'].values())/len(row['channels'])
                choices.append(((-len(row['channels']), distance, anchor), anchor, row))
        if not choices:
            unmatched.append(identity)
            continue
        _, anchor, row = min(choices, key=lambda v: v[0])
        row.update(card_id=identity, kind=card['kind'], matched_evidence_card_id=anchor,
                   evidence_group_id=_origins(eligible[anchor]), score=0.)
        scored.append(row)

    # Dense ranks ensure changing an ID cannot change the relevance score of a
    # numerically tied record. ID is only a deterministic presentation tie-break.
    for channel in channels:
        values = sorted({round(r['channels'][channel]['distance'], 12) for r in scored if channel in r['channels']})
        ranks = {value: i+1 for i, value in enumerate(values)}
        for row in scored:
            if channel in row['channels']:
                item = row['channels'][channel]
                item['rank'] = ranks[round(item['distance'], 12)]
                row['score'] += 1/(RRF_OFFSET+item['rank'])
    scored.sort(key=lambda r: (-r['score'], r['card_id']))
    # Preserve all returnable members in a family without consuming three slots
    # or calling one observation three independent pieces of physical evidence.
    groups = {}
    for row in scored:
        key = row['matched_evidence_card_id'] if kind == 'all' else row['card_id']
        groups.setdefault(key, []).append(row)
    ranked = []
    for members in groups.values():
        members.sort(key=lambda r: (-r['score'], KINDS.index(r['kind']), r['card_id']))
        row = deepcopy(members[0])
        row['related_card_ids'] = [r['card_id'] for r in members[1:]]
        row['related_card_count'] = len(members)-1
        row['matched_reasons'] = [dict(channel=channel, distance=value['distance'],
                                       fields=value['matched_fields']) for channel, value in row['channels'].items()]
        row['execution_authorized'] = False
        row['requires_fresh_physics'] = True
        row['source_part_ids_are_current_ids'] = False
        row['status'] = deepcopy(eligible[row['card_id']].get('status'))
        ranked.append(row)
    ranked.sort(key=lambda r: (-r['score'], r['card_id']))
    # Keep all distinct states but prevent many aliases/stages of one episode
    # from occupying the entire first page. Later pages retain the remainder.
    evidence_queues = {}
    for row in ranked:
        evidence_queues.setdefault(row['evidence_group_id'], []).append(row)
    ranked = []
    for position in range(max((len(v) for v in evidence_queues.values()), default=0)):
        tier = [items[position] for items in evidence_queues.values() if position < len(items)]
        ranked.extend(sorted(tier, key=lambda r: (-r['score'], r['card_id'])))
    page = ranked[offset:offset+limit]
    # Bounded metadata. Complete original cards are fetched separately by ID.
    for row in page:
        row['related_card_ids'] = row['related_card_ids'][:32]
    return dict(results=page, total_matches=len(ranked), offset=offset,
                next_offset=offset+limit if offset+limit < len(ranked) else None,
                audit=dict(policy=POLICY, channels=list(channels), available_channels=list(CHANNELS),
                           fusion='dense_rank_reciprocal_rank_fusion',
                           rrf_offset=RRF_OFFSET, input_cards=len(indexed), eligible_cards=len(eligible),
                           matched_cards=len(scored), grouped_results=len(ranked),
                           excluded_count=len(excluded), excluded_ids=excluded[:64],
                           duplicate_input_count=len(duplicates), unmatched_count=len(unmatched),
                           diversity='one_evidence_group_per_round_preserve_later_states',
                           returned_unique_evidence_groups=len({r['evidence_group_id'] for r in page}),
                           geometry_contexts_evaluated=len(geometry_cache),
                           max_neighbor_descriptors=MAX_NEIGHBORS,
                           geometry_scope='static_reference_features_not_dynamic_cause',
                           bbox_ratio_invariance='axis_permutation_only_not_general_rotation',
                           outcome_used_for_ranking=False, unknown_is_match=False,
                           result_is_execution_permission=False, api_calls=0,
                           evaluation='NOT_PERFORMANCE_VALIDATED'))
