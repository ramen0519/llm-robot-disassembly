"""Optional, bounded model declarations about supplied strategy skills.

This is audit metadata, not a policy, new action permission, or a physics result.
The executable strategy is compiled independently from the unchanged v1 wire.
"""
from copy import deepcopy

from worker.explicit_action_contract import digest


LEGACY = 'legacy_v1'
POLICY = 'skill_selection_review_v1'
POLICIES = (LEGACY, POLICY)
REASON_LIMIT = 240
SELECTED_REASONS = ('planned_procedure',)
NOT_SELECTED_REASONS = ('prefer_other_procedure', 'no_current_proposal',
                        'no_trigger', 'scope_uncertain', 'defer')
ROW_KEYS = {'skill_id', 'decision', 'reason_code', 'reason'}
_MISSING = object()

INSTRUCTIONS = """
skill_selection_review_v1 adds audit declarations only. Return skill_review with
exactly one row per supplied registered_skills card, or [] if none were supplied.
Each row has skill_id, decision (selected or not_selected), reason_code and reason.
selected means that your returned primary steps or stability_failover alternatives
explicitly name that registered skill; otherwise use not_selected. Selection is
conditional plan authorization, not actual execution, attention proof or success.
For selected use reason_code=planned_procedure. For not_selected use one of
prefer_other_procedure, no_current_proposal, no_trigger, scope_uncertain, defer.
no_current_proposal requires an explicitly null currently_proposed_choice; no_trigger also
requires the card's applicability to be exactly no_trigger. A currently available
proposal may be deferred or rejected in favor of another procedure. A currently
absent proposal does not forbid a future conditional skill step. No skill is
preferred or required. Give one short declarative reason, at most 240 characters
on one line, not a thinking transcript or invented physical evidence. Automatic
recovery being unauthorized does not mean a new explicit plan cannot authorize
the registered skill under its existing pose/support/VM guards. Reasons neither
change these guards nor cause another model call. Actual execution and fresh
physics are recorded separately by the host.
"""


def validate_policy(value):
    if value not in POLICIES:
        raise ValueError('unknown_strategy_skill_review_policy')
    return value


def _cards(cards):
    if type(cards) is not list or len(cards) > 32:
        raise ValueError('invalid_strategy_skill_review_cards')
    result = {}
    for card in cards:
        identity = card.get('skill_id') if type(card) is dict else None
        if type(identity) is not str or not identity or len(identity) > 128:
            raise ValueError('invalid_strategy_skill_review_card_id')
        if identity in result:
            raise ValueError('duplicate_strategy_skill_review_card')
        result[identity] = card
    return result


def _proposal_available(card):
    # Missing card facts are unknown, not evidence of an absent proposal.
    if 'currently_proposed_choice' not in card:
        return None
    return card['currently_proposed_choice'] is not None


def extend_schema(base, cards):
    """Extend only the optional provider output, never the v1 compiler schema."""
    identities = list(_cards(cards))
    properties = dict(skill_id=(dict(type='string', enum=identities) if identities
                               else dict(type='string', minLength=1, maxLength=128)),
        decision=dict(type='string', enum=['selected', 'not_selected']),
        reason_code=dict(type='string', enum=list(SELECTED_REASONS + NOT_SELECTED_REASONS)),
        reason=dict(type='string', minLength=1, maxLength=REASON_LIMIT))
    result = deepcopy(base)
    result['properties']['skill_review'] = dict(type='array', minItems=len(identities),
        maxItems=len(identities), items=dict(type='object', properties=properties,
            required=list(properties), additionalProperties=False))
    result['required'].append('skill_review')
    return result


def context(cards):
    return dict(policy=POLICY, supplied_skills=[dict(skill_id=identity,
        current_proposal_available=_proposal_available(card),
        applicability=deepcopy(card.get('applicability')))
        for identity, card in _cards(cards).items()],
        coverage='every_supplied_card_once', selection_forced=False,
        declaration_is_not_execution=True, physical_success_implied=False)


def split_response(value):
    """Separate the self-report so it cannot change executable plan identity."""
    if type(value) is not dict:
        return value, _MISSING
    result = deepcopy(value)
    return result, result.pop('skill_review', _MISSING)


def _planned_uses(wire):
    """Read a compiled/validated wire without authorizing or changing any step."""
    result = {}
    for index, step in enumerate(wire['steps']):
        if step['procedure'] == 'registered_skill':
            result.setdefault(step['registered_skill_id'], []).append(dict(
                step_index=index, alternative_index=None, part_id=step['part_id']))
        failover = step.get('stability_failover')
        if failover is not None:
            for alternative, item in enumerate(failover['alternatives']):
                if item['procedure'] == 'registered_skill':
                    result.setdefault(item['registered_skill_id'], []).append(dict(
                        step_index=index, alternative_index=alternative, part_id=item['part_id']))
    return result


def validate_review(rows, wire, request, cards, plan_id):
    """Bind bounded declarations to actual selected steps and supplied facts.

    Semantic truth of the free text is not inferred. Only coverage, the selected
    IDs and objective no-proposal/no-trigger claims are checked. The full v1
    compiler must succeed before this function is called.
    """
    if rows is _MISSING:
        raise ValueError('missing_strategy_skill_review')
    by_id = _cards(cards)
    if type(rows) is not list or len(rows) != len(by_id):
        raise ValueError('invalid_strategy_skill_review_coverage')
    uses = _planned_uses(wire)
    seen, output = set(), []
    for row in rows:
        if type(row) is not dict or set(row) != ROW_KEYS:
            raise ValueError('invalid_strategy_skill_review_fields')
        identity = row['skill_id']
        if type(identity) is not str or identity not in by_id:
            raise ValueError('unknown_strategy_skill_review_id')
        if identity in seen:
            raise ValueError('duplicate_strategy_skill_review_id')
        seen.add(identity)
        selected = identity in uses
        if row['decision'] != ('selected' if selected else 'not_selected'):
            raise ValueError('strategy_skill_review_selection_mismatch')
        allowed = SELECTED_REASONS if selected else NOT_SELECTED_REASONS
        if type(row['reason_code']) is not str or row['reason_code'] not in allowed:
            raise ValueError('invalid_strategy_skill_review_reason_code')
        reason = row['reason']
        if (type(reason) is not str or not 1 <= len(reason) <= REASON_LIMIT
                or reason != reason.strip() or not reason.isprintable()):
            raise ValueError('invalid_strategy_skill_review_reason')
        card = by_id[identity]
        available = _proposal_available(card)
        if row['reason_code'] in ('no_current_proposal', 'no_trigger') and available is not False:
            raise ValueError('strategy_skill_review_proposal_fact_mismatch')
        if row['reason_code'] == 'no_trigger' and card.get('applicability') != 'no_trigger':
            raise ValueError('strategy_skill_review_trigger_fact_mismatch')
        output.append(dict(deepcopy(row), current_proposal_available=available,
            applicability=deepcopy(card.get('applicability')),
            planned_uses=deepcopy(uses.get(identity, []))))
    if seen != set(by_id):
        raise ValueError('invalid_strategy_skill_review_coverage')
    return dict(policy=POLICY, request_id=request['request_id'], state_id=request['state_id'],
        request_sha256=digest(request), skills_sha256=digest(cards), plan_id=plan_id,
        review_sha256=digest(rows), reviews=output, model_self_report=True,
        selection_forced=False, physical_success_implied=False)
