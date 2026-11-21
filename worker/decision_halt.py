"""Explicit decision stop, never an action-space or physical impossibility claim.

This Python 3.8 module only constructs/checks protocol data. The worker must
separately opt in before accepting this response as a stop instruction.
"""

HALT_KEY = 'decision_halt_v1'
HALT_FIELDS = {'reason', 'status', 'physical_attempt_executed',
               'physical_verdict', 'action_space_exhausted'}


def is_halt_response(response):
    return (type(response) is dict and type(response.get('metadata')) is dict
            and HALT_KEY in response['metadata'])


def validate_halt_fields(response):
    """Validate halt-specific facts after the ordinary envelope/permutation."""
    metadata = response['metadata']
    halt = metadata.get(HALT_KEY)
    if type(halt) is not dict or set(halt) != HALT_FIELDS:
        raise ValueError('invalid_decision_halt_fields')
    reason = halt['reason']
    if (type(reason) is not str or not 1 <= len(reason) <= 160
            or reason != reason.strip() or not reason.isprintable()):
        raise ValueError('invalid_decision_halt_reason')
    if (response['choice'] is not None or response['action_id'] is not None
            or response['fallback_reason'] != reason
            or halt['status'] != 'UNKNOWN'
            or halt['physical_attempt_executed'] is not False
            or halt['physical_verdict'] is not None
            or halt['action_space_exhausted'] is not False
            or metadata.get('exhausted') is not False
            or metadata.get('candidate_pruning') is not False
            or metadata.get('physics_bypass') is not False
            or metadata.get('selection_source') != 'decision_halt'):
        raise ValueError('invalid_decision_halt_claim')


def halt_response(request, reason):
    """Build a no-physics stop without pretending all legal actions were tried."""
    try:
        from .explicit_action_contract import PROTOCOL, validate_request, validate_response
    except ImportError:
        from explicit_action_contract import PROTOCOL, validate_request, validate_response
    request = validate_request(request)
    response = dict(schema_version=1, protocol=PROTOCOL,
        request_id=request['request_id'], state_id=request['state_id'],
        candidate_order=list(request['candidate_ids']), choice=None, action_id=None,
        fallback_reason=reason, metadata=dict(selection_source='decision_halt',
            llm_live=False, exhausted=False, candidate_pruning=False, physics_bypass=False,
            **{HALT_KEY: dict(reason=reason, status='UNKNOWN',
                physical_attempt_executed=False, physical_verdict=None,
                action_space_exhausted=False)}))
    return validate_response(request, response)
