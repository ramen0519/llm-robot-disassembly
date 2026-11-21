"""Provider encoding for an explicit, temporary current-episode probe round.

The full catalog and candidate permutation remain present. Only the active
round is selectable now; deferred support variants remain in the retained
action pool and return when untried paths are exhausted. No physics/cache/API.
"""
from copy import deepcopy

from worker.explicit_action_contract import canonical
from worker.path_probe_round import partition_path_round, validate_path_round_response
from orchestrator.explicit_action_provider_v3 import (build_domain, _schema,
    schema_statistics, expand_provider_decision_v3, encode_provider_decision_v3)

PROVIDER_CONTRACT_V4 = 'explicit_action_provider_v4'
WIRE_INSTRUCTIONS_V4 = (
    'Use the provider_v4 wire format. decision.part_id is the one moving part. '
    'Select pose_slot, force_slot and support_slot by their full lookup tables. '
    'Slots and ascending IDs are serialization addresses, not recommendations. '
    'An explicit path_probe_round_v1 scheduling policy separates currently active '
    'actions from temporarily deferred support variants. The schema admits active '
    'actions only. An observed current-episode path failure can defer its other '
    'support variants while an untried path remains; this is an exploration-order '
    'policy, not an impossibility claim. Observed path success, conflicting or '
    'ambiguous observations do not justify that deferral. Deferred actions remain '
    'in the full catalog and become selectable when untried paths are exhausted. '
    'A valid initial sweep always has an active action while an untried path '
    'remains. A new state has its own observations. '
    'Exact actions already attempted in this state remain separately excluded. '
    'candidate_tail encodes YOUR ranking of EVERY other candidate, including '
    'parts without a currently active action. Start with candidate_address_list '
    'minus decision.part_id; pick_1 is the zero-based index of your next preferred '
    'part, remove it, then use pick_2 in the shortened list, and continue. '
    'Every candidate must be retained, but the first selected part must have an '
    'active action in this round. Do not return candidate_order or choice fields. '
    'evidence_selection marks each supplied evidence slot true only if used; '
    'otherwise false. Path verification fixes all other remaining parts; support '
    'sets affect the later stability stage. Path success with stability failure '
    'may motivate another support set. LLM chooses the active action directly; '
    'no heuristic or deterministic replacement chooses the successful response. '
    'Only fresh actual physics may authorize a transition.'
)


def _count(rectangles):
    return sum(len(r['pose_slots']) * len(r['force_slots']) * len(r['support_slots'])
               for r in rectangles)


def build_domain_v4(request):
    """Split the complete v3 untried domain, without enumerating support tuples."""
    round_state = partition_path_round(request)
    domain = build_domain(request)
    forces = {tuple(force[k] for k in ('skill_id', 'direction_id', 'second_direction_id')): i
              for i, force in enumerate(domain['force_catalog'])}
    poses = {p['pose_id']: p for p in domain['pose_catalog']}
    aliases = {}
    for i, pose in enumerate(domain['pose_catalog']):
        aliases.setdefault(canonical(pose['matrix']), []).append(i)
    deferred_cells = set()
    for item in round_state['deferred_paths']:
        choice = item['choice']
        force = forces[tuple(choice[k] for k in ('skill_id', 'direction_id', 'second_direction_id'))]
        for pose in aliases[canonical(poses[choice['pose_id']]['matrix'])]:
            deferred_cells.add((choice['part_id'], pose, force))
    active, deferred = [], []
    for row in domain['rectangles']:
        # Equal force domains share a pose interval, retaining a compact sparse
        # representation instead of expanding every support combination.
        groups = ({}, {})
        for pose in row['pose_slots']:
            available_forces, delayed_forces = [], []
            for force in row['force_slots']:
                target = delayed_forces if (row['part_id'], pose, force) in deferred_cells else available_forces
                target.append(force)
            for grouping, values in zip(groups, (available_forces, delayed_forces)):
                if values:
                    grouping.setdefault(tuple(values), []).append(pose)
        for output, grouping in zip((active, deferred), groups):
            for force_slots, pose_slots in grouping.items():
                output.append(dict(part_id=row['part_id'], pose_slots=pose_slots,
                    force_slots=list(force_slots), support_slots=list(row['support_slots'])))
    active_count, deferred_count = _count(active), _count(deferred)
    if active_count + deferred_count != domain['available_action_serializations']:
        raise ValueError('provider_round_partition_coverage_mismatch')
    release_reason = round_state['release_reason']
    # An untried path always has its legal empty-support action. Thus a valid
    # initial sweep cannot strand all actions. Do not invent a provider-only
    # release that the shared worker admission rule would reject.
    if active_count == 0 and deferred_count:
        raise ValueError('provider_round_active_space_mismatch')
    domain['rectangles'] = active
    domain['deferred_rectangles'] = deferred
    domain['active_action_serializations'] = active_count
    domain['deferred_action_serializations'] = deferred_count
    domain['round_state'] = deepcopy(round_state)
    domain['round_state']['release_reason'] = release_reason
    domain['round_state']['deferral_active'] = bool(deferred_count)
    domain['round_state']['round_audit']['release_reason'] = release_reason
    domain['round_state']['round_audit']['deferral_active'] = bool(deferred_count)
    domain['active_first_candidate_ids'] = sorted({r['part_id'] for r in active})
    return domain


def response_schema_v4(request):
    return _schema(request, build_domain_v4(request))


def provider_input_v4(request):
    domain = build_domain_v4(request)
    schema = _schema(request, domain)
    lookup = {key: deepcopy(domain[key]) for key in
        ('candidate_address_list', 'force_catalog', 'support_catalog', 'forbidden_action_slots')}
    lookup['pose_slots'] = [dict(pose_slot=i, pose_id=p['pose_id']) for i, p in enumerate(domain['pose_catalog'])]
    lookup['evidence_slots'] = {'e'+str(i): identity for i, identity in enumerate(domain['evidence_ids'])}
    lookup['slot_semantics'] = 'Serialization addresses only; no geometric priority or recommended choice.'
    lookup['path_probe_round'] = deepcopy(domain['round_state']['round_audit'])
    lookup['active_first_candidate_ids'] = list(domain['active_first_candidate_ids'])
    lookup['deferred_action_rectangles'] = deepcopy(domain['deferred_rectangles'])
    audit = {key: domain[key] for key in ('full_action_serializations', 'forbidden_action_serializations',
        'available_action_serializations', 'active_action_serializations', 'deferred_action_serializations')}
    audit.update(schema_statistics(schema), branch_count=len(domain['rectangles']),
        deferred_branch_count=len(domain['deferred_rectangles']),
        pose_entries_preserved=len(domain['pose_catalog']),
        candidate_parts_preserved=len(domain['candidate_address_list']),
        active_first_candidate_ids=list(domain['active_first_candidate_ids']),
        all_candidates_have_active_first_choice=(len(domain['active_first_candidate_ids']) == len(domain['candidate_address_list'])),
        full_catalog_preserved=True, candidate_permutation_preserved=True,
        temporary_action_deferral=True, exact_action_exclusions_only=False,
        permanent_action_exclusions_exact_attempts_only=True,
        all_untried_actions_immediately_selectable=domain['deferred_action_serializations'] == 0,
        active_plus_deferred_equals_untried=(domain['active_action_serializations'] + domain['deferred_action_serializations']
                                           == domain['available_action_serializations']),
        scheduling_policy='path_probe_round_v1', path_round=deepcopy(domain['round_state']['round_audit']),
        scheduler_override_count=0, physics_cache_used=False,
        active_pool_nonempty_while_untried_paths_remain=(domain['round_state']['untried_path_count'] == 0
                                                      or domain['active_action_serializations'] > 0),
        release_conditions=['untried_paths_exhausted', 'new_state_new_scope'])
    return lookup, schema, audit


def expand_provider_decision_v4(request, value):
    """Decode the existing wire syntax, then enforce the active round explicitly."""
    response = expand_provider_decision_v3(request, value)
    validate_path_round_response(request, response)
    return response


def encode_provider_decision_v4(request, response):
    """Same lossless wire projection; deliberately permits invalid test injections."""
    return encode_provider_decision_v3(request, response)
