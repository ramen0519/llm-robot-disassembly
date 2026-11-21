"""Bound explicit requests without changing search scope or physical evidence.

This Python 3.8 module performs JSON validation only: no filesystem, network,
solver, or physics calls. Retrieval rows and their parent chains are atomic.
"""
from copy import deepcopy
from itertools import islice

from worker.explicit_action_contract import (action_id, canonical,
    enumerate_actions, validate_request)


MAX_REQUEST_BYTES = 196608


def request_bytes(value):
    return len(canonical(value).encode('utf-8'))


class RequestCoreTooLarge(ValueError):
    """The mandatory request cannot fit; never silently remove its fields."""
    def __init__(self, audit):
        super().__init__('request_core_too_large')
        self.audit = deepcopy(audit)


def bounded_memory_request(request, references, require_parent_pairs=False):
    """Return (request, audit), adding complete reference chains within 192 KiB.

    ``request`` is the complete mandatory request including recent episode
    feedback, all candidates, poses and attempted actions. Existing memory is
    mandatory too. New references retain their full source scope and raw facts.
    In legacy mode standalone references remain legal as in the old retriever.
    """
    if type(require_parent_pairs) is not bool or type(references) is not list:
        raise ValueError('invalid_bounded_memory_options')
    core = deepcopy(request)
    audit = dict(schema_version=1, policy='whole_reference_parent_chain_v1',
        request_id=core.get('request_id'), state_id=core.get('state_id'),
        limit_bytes=MAX_REQUEST_BYTES, core_bytes=request_bytes(core),
        source_reference_count=len(references), provided_evidence_ids=[],
        excluded_references=[], parent_pairs_required=require_parent_pairs,
        search_scope_changed=False, physics_called=False, api_called=False)
    try:
        validate_request(core)
    except ValueError as error:
        if str(error) == 'request_too_large':
            audit.update(status='ERROR', reason='mandatory_core_exceeds_limit',
                         request_bytes=audit['core_bytes'])
            raise RequestCoreTooLarge(audit)
        raise
    by_id = {}
    for row in references:
        if type(row) is not dict or type(row.get('evidence_id')) is not str or not row['evidence_id']:
            raise ValueError('invalid_retrieval_reference')
        if row['evidence_id'] in by_id:
            raise ValueError('duplicate_retrieval_reference')
        by_id[row['evidence_id']] = deepcopy(row)
    existing = {row['evidence_id']: row for row in core['memory_references']}
    for identity in set(existing) & set(by_id):
        if existing[identity] != by_id[identity]:
            raise ValueError('conflicting_retrieval_reference')

    def closure(identity, visiting):
        if identity in existing:
            return []
        if identity in visiting:
            raise ValueError('cyclic_retrieval_parent_chain')
        row = by_id.get(identity)
        if row is None:
            return None
        parent = row.get('parent_failure_evidence_id')
        parents = []
        if require_parent_pairs and parent is not None:
            if type(parent) is not str or not parent:
                raise ValueError('invalid_retrieval_parent_identity')
            parents = closure(parent, visiting | {identity})
            if parents is None:
                return None
        return parents + [row]

    for identity, row in by_id.items():
        if identity in existing:
            continue
        group = closure(identity, set())
        if group is None:
            audit['excluded_references'].append(dict(evidence_id=identity,
                reason='parent_reference_not_available', source_bytes=request_bytes(row)))
            continue
        trial = deepcopy(core)
        trial['memory_references'] += deepcopy(group)
        trial['evidence_ids'] = sorted(set(trial['evidence_ids']) |
            {item['evidence_id'] for item in group})
        proposed_bytes = request_bytes(trial)
        # Validate all retained facts, including hidden-prior and safety guards.
        # Only size/count exhaustion is a retrieval exclusion, never bad facts.
        try:
            validate_request(trial)
        except ValueError as error:
            if str(error) not in ('request_too_large', 'unbounded_memory_references'):
                raise
            audit['excluded_references'].append(dict(evidence_id=identity,
                reason=('request_byte_limit' if proposed_bytes > MAX_REQUEST_BYTES
                        else 'reference_count_limit'), proposed_request_bytes=proposed_bytes,
                group_evidence_ids=[item['evidence_id'] for item in group],
                source_bytes=request_bytes(row)))
            continue
        core = trial
        existing.update({item['evidence_id']: item for item in group})
    audit.update(status='PASS', request_bytes=request_bytes(core),
        provided_evidence_ids=[row['evidence_id'] for row in core['memory_references']],
        provided_reference_count=len(core['memory_references']),
        excluded_reference_count=len(by_id.keys() - existing.keys()),
        candidate_count=len(core['candidate_ids']), pose_count=len(core['pose_catalog']),
        attempted_action_count=len(core['attempted_action_ids']),
        episode_count=len(core['episode_feedback']))
    # No sorting/copy normalization of the mandatory request is performed.
    return core, audit


def capacity_probe(request, attempted_count=60, episode_count=12, memory_count=8):
    """Make a clearly synthetic payload fixture from real CAD/pose observations.

    Synthetic histories exercise every part-pair endpoint and repeated source
    context. They are never observations or memory for an actual experiment.
    """
    if (type(attempted_count) is not int or not 0 <= attempted_count <= 10000
            or type(episode_count) is not int or not 0 <= episode_count <= 12
            or type(memory_count) is not int or not 0 <= memory_count <= 12):
        raise ValueError('invalid_capacity_probe_counts')
    core = deepcopy(request)
    core.update(memory_references=[], episode_feedback=[], evidence_ids=[],
                attempted_action_ids=[], attempted_actions=[])
    # Synthetic replacement histories cannot retain a real worker restoration
    # authority. Actual restored requests are sized separately, without edits.
    core.pop('virtual_search_restore', None)
    core.pop('accepted_pose_carry', None)
    if 'path_stage_observations' in core:
        core['path_stage_observations'] = []
    choices = list(islice(enumerate_actions(core), attempted_count))
    if len(choices) != attempted_count or episode_count > len(choices):
        raise ValueError('insufficient_actions_for_capacity_probe')
    core['attempted_actions'] = [dict(action_id=action_id(core, choice),
        **{key: deepcopy(value) for key, value in choice.items() if key != 'evidence_ids'})
        for choice in choices]
    core['attempted_action_ids'] = [row['action_id'] for row in core['attempted_actions']]
    if 'path_stage_observations' in core:
        core['path_stage_observations'] = [dict(action_id=identifier,
            path_success=None, status='ERROR', deadline_exceeded=False, backend=None,
            source_event_id='synthetic-capacity:unobserved-path:'+str(index))
            for index, identifier in enumerate(core['attempted_action_ids'])]

    def endpoint(choice):
        return dict(scope='SYNTHETIC_CAPACITY_FIXTURE_NOT_OBSERVED', min_sep=0.0123456789012345,
            pairs=[dict(moving_part_id=choice['part_id'], rest_part_id=part,
                min_distance=0.0123456789012345, separation_unsatisfied=True)
                for part in core['remaining_parts'] if part != choice['part_id']],
            termination_reason='unknown', evidence_sha256='f'*64, blocker=None,
            all_paths_impossible=False)

    for index, choice in enumerate(choices[-episode_count:] if episode_count else []):
        core['episode_feedback'].append(dict(evidence_id='synthetic-capacity:episode:'+str(index),
            status='UNKNOWN', action_id=action_id(core, choice),
            remaining_parts=deepcopy(core['remaining_parts']), choice=deepcopy(choice),
            decision_source='test_fixture', selection_feedback=dict(
                scope='selection_validation_not_physics', provider_contract='explicit_action_provider_v2',
                decision_validation_error=None, action_repeated=False, rejected_action_id=None,
                raw_response_recorded=False), selected_pose_matrix=core['pose_catalog'][0]['matrix'],
            path_success=False, stability_success=None, measured_endpoint=endpoint(choice),
            raw_stability_parts=None,
            raw_stability_parts_semantics='upstream_combines_initial_disconnection_and_observed_falling_no_cause_inference',
            accepted_for_transition=False, deadline_exceeded=False))
    core['evidence_ids'] = [row['evidence_id'] for row in core['episode_feedback']]
    refs = []
    for index in range(memory_count):
        choice = choices[index % len(choices)] if choices else next(enumerate_actions(core))
        refs.append(dict(evidence_id='synthetic-capacity:memory:'+str(index),
            source_run_id='SYNTHETIC_CAPACITY_FIXTURE_NOT_OBSERVED',
            source_request_id='synthetic-request:'+str(index), source_context_sha256='e'*64,
            source_task_context=deepcopy(core['task_context']),
            source_remaining_parts=deepcopy(core['remaining_parts']),
            source_pose_matrix=deepcopy(core['pose_catalog'][0]['matrix']),
            source_choice=deepcopy(choice), source_action_id=action_id(core, choice),
            status='UNKNOWN', path_success=False, stability_success=None,
            verifier_origin='synthetic_fixture_no_verifier', measured_endpoint=endpoint(choice),
            raw_stability_parts=None, source_evidence_ref='synthetic-not-a-file', record_sha256='e'*64,
            parent_failure_evidence_id=None, same_task=False, same_conditions=False,
            condition_match_scope='serialized_declared_conditions_only', reference_only=True,
            current_part_id_mapping=None, current_attempt_verified=False, failure_cause=None,
            blocker=None, hard_pruning=False, physics_reuse_allowed=False, repair_lifecycle=None))
    result, audit = bounded_memory_request(core, refs, require_parent_pairs=True)
    audit.update(synthetic_history=True, synthetic_memory=True,
                 observed_physics=False, observed_api=False)
    if 'path_stage_observations' in core:
        audit.update(synthetic_path_stage_observations=True,
                     path_stage_observations_are_real_physics=False,
                     path_stage_observation_count=len(core['path_stage_observations']))
    return result, audit
