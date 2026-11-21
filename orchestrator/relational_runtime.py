"""Bounded, read-only relational cards for existing planning providers.

Retrieval never authorizes code execution, imports a source part ID into the
current task, adds an action citation, or certifies current physical success.
"""
from copy import deepcopy
import json

from orchestrator.relational_knowledge import context_from_geometry, validate_library
from orchestrator.relational_retrieval import search_cards, source_products
from orchestrator.independent_ranker import FORBIDDEN_KEYS


def size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8'))


def product_id(request, cad=None):
    task = (cad or {}).get('task', request.get('task_context', {}))
    if type(task) is dict:
        task = task.get('task', task)
        return task.get('assembly_id')
    return None


def provider_projection(value):
    """Preserve original records on disk; withhold prohibited priors in prompts."""
    withheld = []
    def visit(item, path):
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                if key.lower().replace('-', '_') in FORBIDDEN_KEYS:
                    withheld.append(path + '/' + key)
                else:
                    result[key] = visit(child, path + '/' + key)
            return result
        if isinstance(item, list):
            return [visit(child, path + '/' + str(i)) for i, child in enumerate(item)]
        return deepcopy(item)
    projected = visit(value, '')
    return projected, withheld


def compact_card(card, *, max_neighbors=3):
    """Exact selected fields, not generated prose; raw records remain in archive.

    Whole-episode arrays, duplicate aliases and solver traces must not crowd out
    the shape, observed result, limitation or actual Python the model needs.
    Missing/unrepresented fields are explicitly listed, never inferred.
    """
    content = card.get('content', {})
    effective = content.get('candidate', content.get('lesson', content))
    effective = effective if isinstance(effective, dict) else content
    wanted = (
        'observation_id', 'record_id', 'id', 'protocol', 'schema_version',
        'scope', 'product', 'first', 'mover', 'stage', 'parent_parts',
        'terminal_by_native_mass_rule', 'held_count_in_selected_pose',
        'outcome', 'status', 'error_type', 'accepted_for_transition',
        'score', 'max_and_total_support', 'step_support_counts',
        'removed_count', 'remaining', 'budget', 'budget_profile',
        'direction_checks', 'direction_count_basis', 'stability_checks',
        'stability_starts', 'stability_finishes', 'stability_count_basis',
        'statement', 'claim', 'limitation', 'generalization', 'claims_are_hypotheses',
        'description', 'applicability', 'preconditions', 'lifecycle',
        'validation_scope', 'registered', 'physics_validated', 'performance_effect',
        'generalization_claim', 'observed_facts', 'causal_claim_verified',
        'path_success', 'stability_success', 'failure_stage', 'failure_reason',
        'choice', 'action', 'support', 'stage_feedback', 'result',
    )
    selected, omitted = {}, []
    for key in wanted:
        if key not in effective:
            continue
        value = effective[key]
        # Keep small values verbatim. Large detailed result trees stay local;
        # preserve their scalar observed status without inventing an explanation.
        if size(value) <= 4096:
            selected[key] = deepcopy(value)
        elif isinstance(value, dict):
            small = {k: deepcopy(v) for k, v in value.items()
                     if k in wanted and size(v) <= 512}
            if small:
                selected[key] = small
            omitted.append(key + '/noncompact_fields')
        else:
            omitted.append(key)
    skill_card = effective.get('card')
    if isinstance(skill_card, dict):
        selected['card'] = {k: deepcopy(v) for k, v in skill_card.items()
                            if k in wanted and size(v) <= 4096}
    source = effective.get('source')
    if not isinstance(source, str) and isinstance(effective.get('program'), dict):
        source = effective['program'].get('source')
    if isinstance(source, str):
        # Never truncate code into a plausible-looking incomplete program.
        selected['source'] = source
    represented = set(selected) | {'evidence_ids', 'source_evidence_ids', 'lesson_ids'}
    omitted += [k for k in effective if k not in represented]
    context = card.get('context') or {}
    context_keys = ('product_id', 'target_part_id', 'remaining_parts', 'shape', 'topology',
                    'conditions', 'relation_scope', 'dynamic_contacts_observed',
                    'contact_source', 'contact_eps', 'evidence_status', 'missing_features',
                    'outcome', 'source_geometry_status')
    compact_context = {k: deepcopy(context[k]) for k in context_keys if k in context}
    observations = context.get('observations')
    if isinstance(observations, dict):
        compact_observations = {}
        omitted_observations = []
        for key, item in observations.items():
            if size(item) <= 1024:
                compact_observations[key] = deepcopy(item)
            elif isinstance(item, dict):
                compact_observations[key] = {k: deepcopy(v) for k,v in item.items()
                    if (v is None or isinstance(v, (bool, int, float, str))) and size(v) <= 256}
                omitted_observations.append(key + '/noncompact_fields')
            else:
                omitted_observations.append(key)
        compact_context['observations'] = compact_observations
        compact_context['observation_fields_omitted'] = omitted_observations
    neighbors = context.get('neighbors', [])
    compact_neighbors = []
    for neighbor in neighbors[:max_neighbors]:
        row = {k: deepcopy(neighbor[k]) for k in
               ('part_id', 'center_delta_xyz', 'contact', 'relation') if k in neighbor}
        shape = neighbor.get('shape') or {}
        row['shape'] = {k: deepcopy(shape[k]) for k in
                        ('extents_xyz', 'ratio_xyz', 'length_unit', 'frame') if k in shape}
        relational = ('contact_position', 'contact_normal', 'contact_area',
                      'fastening', 'supports_target', 'blocks_motion')
        row.update({k: deepcopy(neighbor[k]) for k in relational if neighbor.get(k) is not None})
        row['unobserved_relation_fields'] = [k for k in relational if neighbor.get(k) is None]
        compact_neighbors.append(row)
    compact_context['neighbors'] = compact_neighbors
    compact_context['neighbors_omitted_count'] = max(0, len(neighbors) - max_neighbors)
    target = context.get('target_part_id')
    prechoice = context.get('stored_prechoice_features')
    if isinstance(prechoice, dict):
        features = prechoice.get(target, prechoice)
        if size(features) <= 2048:
            compact_context['target_prechoice_features'] = deepcopy(features)
    origin = card.get('origin', {})
    compact_origin = {k: deepcopy(origin[k]) for k in
                      ('source_path', 'json_pointer', 'source_id', 'source_sha256', 'author_model') if k in origin}
    value = {k: deepcopy(card[k]) for k in ('card_id', 'kind', 'status', 'dataset_role',
             'fixture', 'evidence_ids', 'original_program_sha256', 'original_status_preserved') if k in card}
    value.update(content=selected, context=compact_context, origin=compact_origin,
        execution_authorized=False, physics_reuse_allowed=False,
        projection=dict(protocol='relational_provider_view/v1', original_immutable=True,
                        original_content_fields_omitted=sorted(set(omitted)),
                        original_aliases_omitted=len(origin.get('aliases', [])),
                        no_new_claims_generated=True))
    return value


def contexts_for_request(request, cad=None):
    """Calculate queries for every current candidate; missing fields stay missing."""
    geometry = deepcopy(cad if cad is not None else request.get('geometry', {}))
    if 'contact_edges' not in geometry:
        geometry['contact_edges'] = geometry.get('relations', {}).get('contact_edges')
    candidates = request.get('candidate_ids', [])
    remaining = request.get('remaining_parts', request.get('assembled_parts', []))
    if not remaining:
        remaining = list(geometry.get('parts', {}))
    base = request.get('base_part')
    task = request.get('task_context', {})
    if cad:
        task = cad.get('task', task)
    conditions = dict(verifier_domain='ASAP_RedMax',
                      verifier_profile='sequence_native_v1' if cad else 'explicit_action_v1')
    if 'base_part' in request:
        conditions['fixed_base'] = base if base is not None else 'none'
        conditions['fixed_base_present'] = base is not None
        conditions['known_fields'] = ['fixed_base']
    if type(task) is dict:
        settings = task.get('solver_settings', {})
        if type(settings) is dict and 'support_policy' in settings:
            conditions['support_policy'] = settings['support_policy']
    contexts = []
    for part in candidates:
        context = context_from_geometry(geometry, part, remaining_parts=remaining,
                                        conditions=conditions)
        context['product_id'] = product_id(request, cad)
        context['target_part_id'] = part
        contexts.append(context)
    return contexts


class RelationalRuntime:
    def __init__(self, library):
        self.library = deepcopy(validate_library(library))
        self.cards = self.library['cards']
        self.by_id = {card['card_id']: card for card in self.cards}
        self.fingerprint = self.library['content_sha256']
        self.ineligible = {card['card_id'] for card in self.cards
                           if card.get('fixture') is not False or card.get('dataset_role') != 'training'}

    def excluded_ids(self, excluded_products):
        """Exclude a derived card too when its readable evidence leaks a product."""
        excluded = set(excluded_products)
        blocked = self.ineligible | {key for key, card in self.by_id.items()
                                    if (card.get('context') or {}).get('product_id') in excluded
                                    or bool(source_products(card) & excluded)}
        changed = True
        while changed:
            old = len(blocked)
            blocked.update(key for key, card in self.by_id.items()
                           if set(card.get('evidence_ids', [])) & blocked)
            changed = len(blocked) != old
        return blocked

    def search(self, request, *, cad=None, query='', kind='all', trigger='all',
               offset=0, limit=6, excluded_products=(), return_card_ids=None):
        if type(limit) is not int or not 1 <= limit <= 8 or type(offset) is not int or offset < 0:
            raise ValueError('invalid_relational_page')
        if type(query) is not str or len(query) > 240:
            raise ValueError('invalid_relational_query')
        if kind not in ('all', 'memory', 'experience', 'lesson', 'skill'):
            raise ValueError('invalid_relational_kind')
        if return_card_ids is not None and (type(return_card_ids) is not list
                or any(type(v) is not str for v in return_card_ids)
                or len(return_card_ids) != len(set(return_card_ids))):
            raise ValueError('invalid_relational_return_ids')
        channels = None
        for channel in ('shape', 'relations', 'situation'):
            prefix = channel + ':'
            if query.startswith(prefix):
                channels, query = [channel], query[len(prefix):].strip()
                break
        blocked = self.excluded_ids(excluded_products)
        allowed = set(self.by_id) - blocked
        groups, traces = [], []
        contexts = contexts_for_request(request, cad)
        for context in contexts or [{}]:
            query_value = dict(context=context,
                kind='experience' if kind == 'memory' else kind,
                trigger=None if trigger == 'all' else trigger, text=query)
            if return_card_ids is not None:
                query_value['return_card_ids'] = return_card_ids
            if channels:
                query_value['channels'] = channels
            result = search_cards(self.cards, query_value,
                limit=limit, offset=offset, exclude_products=excluded_products,
                allowed_card_ids=allowed)
            rows = deepcopy(result['results'])
            for row in rows:
                row['matched_current_part_ids'] = [context.get('target_part_id')]
            groups.append(rows)
            # Per-candidate full excluded-ID lists would multiply archive size
            # and disclose evaluation catalog identities in a model prompt.
            compact_audit = {k: deepcopy(v) for k,v in result.get('audit', {}).items()
                             if k != 'excluded_ids'}
            traces.append(dict(target_part_id=context.get('target_part_id'),
                               audit=compact_audit, total_matches=result.get('total_matches'),
                               next_offset=result.get('next_offset')))
        # Each candidate gets a chance; do not silently keep only the first part.
        selected, seen = [], {}
        for rank in range(max((len(rows) for rows in groups), default=0)):
            for rows in groups:
                if rank >= len(rows):
                    continue
                row = rows[rank]
                identity = row['card_id']
                if identity in seen:
                    prior = seen[identity]['matched_current_part_ids']
                    prior.extend(p for p in row['matched_current_part_ids'] if p not in prior)
                elif len(selected) < limit:
                    selected.append(row)
                    seen[identity] = row
        return dict(policy='relational_v1', results=selected,
                    total_matches=sum(t['total_matches'] or 0 for t in traces),
                    next_offset=offset+limit if any(t['next_offset'] is not None for t in traces) else None,
                    audit=dict(library_sha256=self.fingerprint,
                        per_candidate_queries=traces, candidate_merge='rank_round_robin_unique_card',
                        pagination='offset_applied_per_current_candidate', excluded_products=list(excluded_products),
                        excluded_cards=len(blocked), ineligible_not_confirmed_training=len(self.ineligible),
                        requested_channels=channels, execution_authorized=False,
                        relevance_is_not_validation=True))

    def read(self, card_id, *, excluded_products=(), max_bytes=32768, include_evidence=True):
        if card_id not in self.by_id:
            return dict(error='unknown_relational_card')
        blocked = self.excluded_ids(excluded_products)
        if card_id in blocked:
            return dict(error='relational_evaluation_product_excluded')
        card, withheld = provider_projection(compact_card(self.by_id[card_id],
            max_neighbors=3 if max_bytes <= 16384 else 12))
        value = dict(card=card, linked_evidence=[], omitted_evidence=[],
                     library_sha256=self.fingerprint, execution_authorized=False,
                     action_citation_ids_unchanged=True, current_part_mapping=None,
                     current_physics_verified=False, requires_fresh_physics=True,
                     withheld_prior_fields=withheld)
        if size(value) > max_bytes:
            return dict(error='relational_card_byte_limit', card_id=card_id, bytes=size(value))
        if not include_evidence:
            value['evidence_details_omitted_count'] = len(card.get('evidence_ids', []))
            value['evidence_details_policy'] = 'linked_by_id; available_via_read; no_implicit_validation'
            return value
        for identity in card.get('evidence_ids', []):
            evidence = self.by_id.get(identity)
            if evidence is None:
                value['omitted_evidence'].append(dict(card_id=identity, reason='unresolved_reference'))
            elif identity in blocked:
                value['omitted_evidence'].append(dict(card_id=identity, reason='evaluation_product_excluded'))
            else:
                candidate = deepcopy(value)
                projected, hidden = provider_projection(compact_card(evidence, max_neighbors=3))
                candidate['linked_evidence'].append(projected)
                candidate['withheld_prior_fields'].extend(hidden)
                if size(candidate) <= max_bytes - 2048:
                    value = candidate
                else:
                    value['omitted_evidence'].append(dict(card_id=identity, reason='context_byte_limit'))
        if size(value) > max_bytes:
            return dict(error='relational_linked_card_byte_limit', card_id=card_id)
        return value

    def packet(self, request, *, cad=None, max_bytes=16384, excluded_products=()):
        if type(max_bytes) is not int or max_bytes < 2048:
            raise ValueError('relational_packet_limit_too_small')
        # A nearest stage experience may have no derived lesson. Reserve a
        # relevance shortlist for each type so code/lessons are discoverable;
        # this is input coverage, never compulsory model use or execution.
        by_kind = {kind: self.search(request, cad=cad, kind=kind, limit=2,
                                    excluded_products=excluded_products)
                   for kind in ('experience', 'lesson', 'skill')}
        matches = []
        for rank in range(2):
            for result in by_kind.values():
                if rank < len(result['results']):
                    matches.append(result['results'][rank])
        audit = dict(policy='relational_v1', library_sha256=self.fingerprint,
                     shortlist_policy='up_to_two_relevant_per_kind_interleaved',
                     by_kind={kind: value['audit'] for kind,value in by_kind.items()},
                     execution_authorized=False, relevance_is_not_validation=True)
        value = dict(policy='relational_v1', library_sha256=self.fingerprint,
                     cards=[], excluded=[], excluded_count=0, current_part_ids=list(request.get('candidate_ids', [])),
                     citation_namespace='relational_card_ids', execution_authorized=False,
                     relevance_is_not_validation=True)
        considered = set()
        def exclude(identity, reason):
            value['excluded_count'] += 1
            if len(value['excluded']) < 6:
                value['excluded'].append(dict(card_id=identity, reason=reason))
        for match in matches:
            # Search deliberately groups an experience and its derived knowledge
            # into one hit. Read that whole family, not only the representative
            # experience: otherwise lessons/Python silently disappear in prompts.
            family = [match['card_id']] + match.get('related_card_ids', [])
            for identity in family:
                if identity in considered:
                    continue
                considered.add(identity)
                if len(value['cards']) >= 9:
                    exclude(identity, 'provider_card_count_limit_use_agent_read_for_more')
                    continue
                view = self.read(identity, excluded_products=excluded_products,
                                 max_bytes=min(32768, max_bytes), include_evidence=False)
                if 'error' in view:
                    exclude(identity, view['error'])
                    continue
                related_match = {k: deepcopy(v) for k,v in match.items() if k != 'channels'}
                related_match['card_id'] = identity
                related_match['kind'] = view['card']['kind']
                if identity != match['card_id']:
                    related_match['retrieved_via_evidence_family'] = match['card_id']
                    related_match['score_is_shared_anchor_relevance'] = True
                row = dict(match=related_match, knowledge=view)
                if size(dict(value, cards=value['cards']+[row])) <= max_bytes - 1024:
                    value['cards'].append(row)
                else:
                    exclude(identity, 'context_byte_limit')
        value['card_ids'] = [row['match']['card_id'] for row in value['cards']]
        if size(value) > max_bytes:
            raise ValueError('relational_packet_byte_limit')
        return value, audit
