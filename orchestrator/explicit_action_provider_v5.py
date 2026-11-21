"""Lossless evidence presentation and auditable self-report over the v4 domain.

No action ranking, physics verdict, cache or new pruning is introduced. Evidence
review is a model self-report, not proof of attention or causal benefit.
"""
from copy import deepcopy
import re

from worker.explicit_action_contract import canonical, digest, validate_request
from worker.path_probe_round import validate_path_round_response
from orchestrator.explicit_action_provider_v3 import (MAX_SCHEMA_BYTES, WIRE_FIELDS,
    schema_statistics, expand_provider_decision_v3, encode_provider_decision_v3)
from orchestrator.explicit_action_provider_v4 import provider_input_v4, WIRE_INSTRUCTIONS_V4

PROVIDER_CONTRACT_V5 = 'explicit_action_provider_v5'
REVIEW_VALUES = ('used_for_choice', 'reviewed_not_used', 'not_reviewed')
WIRE_FIELDS_V5 = (WIRE_FIELDS - {'evidence_selection'}) | {'evidence_review'}
PROJECTION_VERSION = 'shared_source_context_v1'
WIRE_INSTRUCTIONS_V5 = WIRE_INSTRUCTIONS_V4.replace('provider_v4', 'provider_v5').replace(
    'evidence_selection marks each supplied evidence slot true only if used; otherwise false.',
    'evidence_review reports one value for each supplied evidence slot: used_for_choice '
    'only when that observation informed your current choice; reviewed_not_used if '
    'you reviewed it but did not use it; not_reviewed otherwise. No citation or '
    'review is mandatory. All not_reviewed is legal. These are self-reports, not '
    'physical conclusions or proof of memory benefit.') + (
    ' The decision_brief presents the supplied evidence without ranking it. '
    'source_task_context references are losslessly shared by canonical SHA256; '
    'the full source context remains available in shared_source_contexts. '
    'same_mesh_id_correspondence proves only equal declared OBJ bytes and IDs, '
    'not solver, pose, support, RNG, trajectory or outcome equivalence. Original '
    'reference_only, same_conditions and physics_reuse_allowed flags still apply. '
    'Both force directions are in simulation world coordinates after the selected '
    'pose. gravity_unit_in_input_frame is R-transpose times world minus-Z for '
    'each supplied pose, not a stability score or recommended pose. '
    'raw_stability_parts alone can combine initial disconnection and observed '
    'falling. A missing detailed stability diagnostic is unknown, not success. '
    'The stability checker is constructed afresh from the remaining bodies, selected '
    'pose and fixed supports; it does not continue the moving part trajectory. '
    'Changing only force direction or path primitive therefore does not change '
    'the stability configuration. After path PASS / stability failure, reason '
    'about the support SET, pose, or which part is removed before spending another '
    'attempt on a different path with identical stability conditions. This is '
    'checker semantics, not pruning or a guarantee that any repair succeeds. '
    'Initial_disconnected means zero initial contact-graph degree, not measured '
    'falling. Dynamic_failure requires the reported measured displacement or lost '
    'contacts. Support can hold remaining parts, never the part being removed. '
    'memory_alignment compares declared geometry, remaining bodies and pose only; '
    'reference_only does not prohibit using an observation as a hypothesis. '
    'Use relevant successes AND failures to inform a fresh choice when justified; '
    'do not copy IDs without their geometry correspondence. No citation is required.'
)


def project_request_v5(request):
    """Replace only repeated source contexts, with exact reversible references.

    A context already present as current task_context is referenced there rather
    than copied. Distinct source contexts remain fully present once. Every other
    request field, safety flag, candidate and history row stays unchanged.
    """
    validate_request(request)
    projected, contexts, occurrences = deepcopy(request), {}, 0
    current_digest = digest(request['task_context'])
    for row in projected['memory_references']:
        if 'source_task_context' not in row:
            continue
        context = row['source_task_context']
        context_digest = digest(context)
        if row.get('source_context_sha256') != context_digest:
            raise ValueError('source_context_digest_mismatch')
        if context_digest not in contexts:
            contexts[context_digest] = ({'location': 'request.task_context'}
                if context_digest == current_digest and context == request['task_context']
                else {'value': deepcopy(context)})
        row['source_task_context'] = {'shared_source_context_sha256': context_digest}
        occurrences += 1
    bundle = {'projection_version': PROJECTION_VERSION, 'request': projected,
              'shared_source_contexts': contexts}
    if restore_request_v5(bundle) != request:
        raise ValueError('provider_projection_not_lossless')
    original_bytes = len(canonical(request).encode('utf-8'))
    projected_bytes = len(canonical(bundle).encode('utf-8'))
    audit = dict(source_context_occurrences=occurrences,
        unique_source_contexts=len(contexts),
        shared_contexts_using_current_task=sum('location' in v for v in contexts.values()),
        original_request_bytes=original_bytes, projected_bundle_bytes=projected_bytes,
        bytes_saved=original_bytes-projected_bytes, lossless_roundtrip_verified=True,
        candidates_changed=False, histories_changed=False, safety_flags_changed=False)
    return bundle, audit


def restore_request_v5(bundle):
    """Validate hashes and restore exactly the original provider input request."""
    if (type(bundle) is not dict or set(bundle) != {'projection_version', 'request', 'shared_source_contexts'}
            or bundle['projection_version'] != PROJECTION_VERSION
            or type(bundle['shared_source_contexts']) is not dict):
        raise ValueError('invalid_provider_projection')
    request = deepcopy(bundle['request'])
    contexts, used = bundle['shared_source_contexts'], set()
    for row in request['memory_references']:
        if 'source_task_context' not in row:
            continue
        reference = row['source_task_context']
        if type(reference) is not dict or set(reference) != {'shared_source_context_sha256'}:
            raise ValueError('invalid_source_context_reference')
        identity = reference['shared_source_context_sha256']
        if type(identity) is not str or identity not in contexts:
            raise ValueError('unknown_source_context_reference')
        entry = contexts[identity]
        if type(entry) is not dict:
            raise ValueError('invalid_shared_source_context')
        if entry == {'location': 'request.task_context'}:
            value = request['task_context']
        elif set(entry) == {'value'}:
            value = entry['value']
        else:
            raise ValueError('invalid_shared_source_context')
        if digest(value) != identity or row.get('source_context_sha256') != identity:
            raise ValueError('source_context_digest_mismatch')
        row['source_task_context'] = deepcopy(value)
        used.add(identity)
    if used != set(contexts):
        raise ValueError('unused_shared_source_context')
    # Validation normalizes candidate/remaining order in its returned copy.
    # Restoration must retain the original serialized lists exactly instead.
    validate_request(request)
    return request


def _obj_hashes(context):
    task = context.get('task', {}) if type(context) is dict else {}
    hashes = task.get('input_sha256', {}) if type(task) is dict else {}
    if type(hashes) is not dict:
        return {}
    return {name[:-4]: value for name, value in hashes.items()
        if type(name) is str and name.endswith('.obj') and '/' not in name and '\\' not in name
        and type(value) is str and re.fullmatch('[0-9a-f]{64}', value)}


def same_mesh_id_correspondence(request, reference):
    """Geometry-only same-ID proof, never cross-task matching or cache admission."""
    source, current = _obj_hashes(reference.get('source_task_context')), _obj_hashes(request['task_context'])
    source_parts = reference.get('source_remaining_parts')
    if (not source or type(source_parts) is not list
            or any(type(p) is not str for p in source_parts)):
        return None
    pairs = {part: {'current_part_id': part, 'obj_sha256': source[part]}
        for part in sorted(set(source_parts) & set(request['remaining_parts']))
        if part in source and source[part] == current.get(part)}
    if not pairs:
        return None
    return {'scope': 'same_part_id_and_equal_declared_obj_sha256_only', 'parts': pairs,
        'source_task_context_sha256': digest(reference['source_task_context']),
        'current_task_context_sha256': digest(request['task_context']),
        'physics_reuse_allowed': False, 'current_attempt_verified': False,
        'original_memory_safety_flags_unchanged': True}


def memory_alignment(request, reference):
    """Expose comparison facts, never a proposed action or a cached verdict."""
    proof = same_mesh_id_correspondence(request, reference)
    source_parts = reference.get('source_remaining_parts')
    choice = reference.get('source_choice', {})
    current = set(request['remaining_parts'])
    equal_state = type(source_parts) is list and set(source_parts) == current
    equal_meshes = bool(equal_state and proof and set(proof['parts']) == current)
    supports = choice.get('support_ids')
    return dict(
        remaining_part_ids_equal=equal_state,
        all_remaining_obj_bytes_equal=equal_meshes,
        source_pose_in_current_catalog=reference.get('source_pose_matrix') in [
            p['matrix'] for p in request['pose_catalog']],
        source_mover_is_current_candidate=bool(equal_meshes and
            choice.get('part_id') in request['candidate_ids']),
        source_supports_present=bool(equal_meshes and type(supports) is list and
            set(supports) <= current - {choice.get('part_id')}),
        source_support_count=len(supports) if type(supports) is list else None,
        current_support_capacity=request['max_grippers'] - 1,
        scope='comparison_facts_only_not_solver_seed_budget_or_outcome_equivalence',
        current_attempt_verified=False, physics_reuse_allowed=False)


def current_failure_brief(request):
    """Only a bound current failure can supply current unstable-body IDs."""
    from orchestrator.skill_program import _scoped_last
    last = _scoped_last(validate_request(request))
    if last is None:
        return None
    feedback = last['feedback']
    if feedback.get('path_success') is False:
        stage = 'path_not_successful'
    elif feedback.get('path_success') is True and feedback.get('stability_success') is False:
        stage = 'remaining_body_stability_not_successful'
    else:
        return None
    return dict(
        evidence_id=feedback['evidence_id'], stage=stage,
        observed_choice=deepcopy(last['choice']),
        measured_dynamic_failure_parts=sorted(last['unstable']),
        observed_initial_disconnected_parts=sorted(last['initial_disconnected']),
        support_capacity=request['max_grippers'] - 1,
        diagnosis_missing_does_not_mean_no_problem=True,
        action_ranking_supplied=False, hard_pruning=False)


def decision_brief_v5(request):
    """Facts adjacent to their review slot, with no priority or invented causes."""
    rows = {}
    for kind, collection in (('frozen_memory', request['memory_references']),
                             ('current_episode', request['episode_feedback'])):
        for record in collection:
            rows.setdefault(record['evidence_id'], []).append((kind, record))
    evidence = []
    for i, identity in enumerate(sorted(request['evidence_ids'])):
        sources = []
        for kind, record in rows[identity]:
            frozen = kind == 'frozen_memory'
            diagnostic = record.get('stability_diagnostics')
            if diagnostic is not None:
                from worker.stability_diagnostics import project_stability_diagnostics
                diagnostic = project_stability_diagnostics(diagnostic)
            sources.append(dict(source_kind=kind,
                observed_choice=deepcopy(record.get('source_choice' if frozen else 'choice')),
                status=record.get('status'), path_success=record.get('path_success'),
                stability_success=record.get('stability_success'),
                raw_stability_parts=deepcopy(record.get('raw_stability_parts')),
                stability_diagnostics=diagnostic,
                parent_failure_evidence_id=record.get('parent_failure_evidence_id'),
                scope={key: deepcopy(record.get(key)) for key in (
                    ('source_context_sha256', 'source_remaining_parts', 'source_pose_matrix',
                     'same_task', 'same_conditions', 'condition_match_scope', 'reference_only',
                     'current_attempt_verified', 'physics_reuse_allowed', 'hard_pruning')
                    if frozen else ('remaining_parts', 'selected_pose_matrix', 'deadline_exceeded',
                                    'accepted_for_transition'))},
                same_mesh_id_correspondence=same_mesh_id_correspondence(request, record) if frozen else None,
                memory_alignment=memory_alignment(request, record) if frozen else None))
        evidence.append(dict(slot='e'+str(i), evidence_id=identity, observations=sources))
    return {'evidence_order': 'slot_address_only_not_priority', 'evidence': evidence,
        'review_semantics': 'model_self_report_not_proof_of_attention_or_causal_effect',
        'empty_citations_allowed': True,
        'current_failure': current_failure_brief(request),
        'stability_configuration_fields': ['remaining_parts_minus_mover', 'pose_matrix',
                                           'base_part', 'support_set'],
        'pose_gravity': [dict(pose_id=pose['pose_id'],
            gravity_unit_in_input_frame=[-pose['matrix'][2][j] for j in range(3)])
            for pose in request['pose_catalog']],
        'gravity_scope': 'R_transpose_times_world_minus_Z_algebra_only_no_stability_verdict'}


def provider_input_v5(request):
    """Keep the v4 action domain exactly; change evidence syntax/presentation only."""
    lookup, schema, audit = provider_input_v4(request)
    schema['properties'].pop('evidence_selection')
    schema['required'] = [k if k != 'evidence_selection' else 'evidence_review' for k in schema['required']]
    schema['properties']['evidence_review'] = {'type': 'object', 'additionalProperties': False,
        'properties': {key: {'type': 'string', 'enum': list(REVIEW_VALUES)} for key in lookup['evidence_slots']},
        'required': list(lookup['evidence_slots'])}
    stats = schema_statistics(schema)
    if (stats['property_count'] > 5000 or stats['enum_values'] > 1000
            or stats['string_characters'] > 120000 or stats['max_object_depth'] > 10
            or stats['schema_bytes'] > MAX_SCHEMA_BYTES):
        raise ValueError('provider_schema_limit_exceeded')
    _, projection_audit = project_request_v5(request)
    # Validate optional diagnostic facts before transport or reservation, even
    # when a caller performs only a provider_input preflight.
    brief = decision_brief_v5(request)
    audit.update(stats, request_projection=projection_audit,
        decision_brief_bytes=len(canonical(brief).encode('utf-8')),
        evidence_review_kind='model_self_report', evidence_citation_required=False,
        action_domain_unchanged_from_v4=True)
    return lookup, schema, audit


def response_schema_v5(request):
    return provider_input_v5(request)[1]


def review_ids_v5(request, review):
    slots = {'e'+str(i): identity for i, identity in enumerate(sorted(request['evidence_ids']))}
    if (type(review) is not dict or set(review) != set(slots)
            or any(type(value) is not str or value not in REVIEW_VALUES for value in review.values())):
        raise ValueError('invalid_provider_evidence_review')
    return {value: [identity for slot, identity in slots.items() if review[slot] == value]
            for value in REVIEW_VALUES}


def decode_provider_decision_v5(request, value):
    """Decode prior to duplicate/round validation so diagnostics retain ordering."""
    if type(value) is not dict or set(value) != WIRE_FIELDS_V5:
        raise ValueError('invalid_provider_wire_fields')
    review_ids_v5(request, value['evidence_review'])
    old = deepcopy(value)
    old['evidence_selection'] = {key: status == 'used_for_choice'
                               for key, status in old.pop('evidence_review').items()}
    return expand_provider_decision_v3(request, old)


def expand_provider_decision_v5(request, value):
    response = decode_provider_decision_v5(request, value)
    validate_path_round_response(request, response)
    return response


def encode_provider_decision_v5(request, response, evidence_review=None):
    """Fixture/audit encoder; unused slots default to not_reviewed, never inferred."""
    wire = encode_provider_decision_v3(request, response)
    wire['evidence_review'] = {key: 'used_for_choice' if used else 'not_reviewed'
                              for key, used in wire.pop('evidence_selection').items()}
    if evidence_review is not None:
        ids = review_ids_v5(request, evidence_review)
        if set(ids['used_for_choice']) != set(response['choice']['evidence_ids']):
            raise ValueError('review_citation_mismatch')
        wire['evidence_review'] = deepcopy(evidence_review)
    return wire
