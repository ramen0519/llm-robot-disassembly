"""Compact reasoning inputs and exact local request audits for generation.

Only initial-contact synthesis/refinement is compacted. Full local packets are
unchanged. This module reads no files, credentials or ledger and calls neither
an API nor physics. Debug observations must pass the existing replay binding.
References factor repeated data; they do not rank or recommend an action.
"""
from copy import deepcopy
import hashlib
import json

from orchestrator.skill_refinement import _bound_case

MAX_FULL_BYTES = 1024 * 1024
MAX_PROVIDER_BYTES = 96 * 1024
MAX_NODES = 100000
VERSION = 'initial_contact_evidence_packet_v1'


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _bounded_json(value):
    count = 0
    def visit(item, depth=0):
        nonlocal count
        count += 1
        if count > MAX_NODES or depth > 64:
            raise ValueError('generation_packet_structure_limit')
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError('generation_packet_string_keys_required')
            for child in item.values():
                visit(child, depth + 1)
        elif type(item) is list:
            for child in item:
                visit(child, depth + 1)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError('generation_packet_json_values_required')
        elif type(item) is str and len(item) > MAX_FULL_BYTES:
            raise ValueError('generation_packet_string_limit')
    visit(value)
    raw = _encoded(value)
    if len(raw) > MAX_FULL_BYTES:
        raise ValueError('full_generation_packet_byte_limit')
    return raw


def packet_size_components(packet):
    """Canonical UTF-8 byte counts; nested counts overlap, never add them."""
    result = {'total_bytes': len(_encoded(packet)),
              'top_level_bytes': {key: len(_encoded(value)) for key, value in packet.items()},
              'nested_component_counts_overlap': True}
    context = packet.get('context')
    if type(context) is dict:
        result['context_component_bytes'] = {key: len(_encoded(value)) for key, value in context.items()}
    return result


class _Tables:
    def __init__(self):
        self.tables = {}
        self.seen = {}

    def intern(self, kind, value):
        key = _encoded(value)
        table, seen = self.tables.setdefault(kind, {}), self.seen.setdefault(kind, {})
        if key not in seen:
            ref = kind + ':' + str(len(table))
            table[ref], seen[key] = deepcopy(value), ref
        return seen[key]

    def pose(self, value):
        return self.intern('poses', value)

    def observed_fields(self, value):
        """Factor only named observations, retaining every original field/value."""
        if type(value) is list:
            return [self.observed_fields(child) for child in value]
        if type(value) is not dict:
            return deepcopy(value)
        result = {}
        for key, child in value.items():
            if key == 'stability_diagnostics' and type(child) is dict:
                result[key + '_ref'] = self.intern('stability_diagnostics', child)
            elif key == 'selected_pose_matrix' and type(child) is list:
                result[key + '_ref'] = self.pose(child)
            elif key in ('measured_endpoint', 'endpoint_observation') and type(child) is dict:
                endpoint = deepcopy(child)
                if 'pairs' in endpoint:
                    endpoint['pairs_ref'] = self.intern('endpoint_pairs', endpoint.pop('pairs'))
                result[key + '_ref'] = self.intern('endpoints', endpoint)
            else:
                result[key] = self.observed_fields(child)
        return result

    def task(self, value):
        value = deepcopy(value)
        if 'input_sha256' in value:
            value['input_sha256_ref'] = self.intern('input_hashes', value.pop('input_sha256'))
        return self.intern('tasks', value)

    def geometry(self, value):
        value = deepcopy(value)
        for key in ('parts',):
            if key in value:
                value[key + '_ref'] = self.intern('part_features', value.pop(key))
        if 'task' in value:
            value['task_ref'] = self.task(value.pop('task'))
        if 'input_integrity' in value:
            integrity = value['input_integrity']
            for key in ('before_sha256', 'after_sha256'):
                if key in integrity:
                    integrity[key + '_ref'] = self.intern('input_hashes', integrity.pop(key))
        return self.intern('geometry', value)

    def request(self, request):
        value = deepcopy(request)
        context = value.pop('task_context')
        if 'task' in context:
            context['task_ref'] = self.task(context.pop('task'))
        if 'geometry_features' in context:
            context['geometry_features_ref'] = self.intern('part_features', context.pop('geometry_features'))
        state = {key: value.pop(key) for key in ('state_id', 'remaining_parts', 'candidate_ids', 'base_part', 'max_grippers')}
        state['task_context_ref'] = self.intern('task_contexts', context)
        state['geometry_ref'] = self.geometry(value.pop('geometry'))
        catalog = deepcopy(value.pop('pose_catalog'))
        for pose in catalog:
            pose['matrix_ref'] = self.pose(pose.pop('matrix'))
        state['pose_catalog_ref'] = self.intern('pose_catalogs', catalog)
        value['state_ref'] = self.intern('states', state)
        # Dynamic feedback/guards remain intact; their IDs are not interchangeable.
        return self.intern('requests', self.observed_fields(value))


def restore_debug_request(provider_packet, request_id, audit=None):
    """Reconstruct the exact normalized request, for audits/tests, no physics."""
    if audit is not None:
        request = deepcopy(audit['local_debug_requests'][request_id])
        if _hash(request) != audit['local_debug_request_hashes'][request_id]:
            raise ValueError('local_debug_request_hash_mismatch')
        return request
    if 'provider_request_projection' in provider_packet:
        raise ValueError('exact_request_requires_local_audit')
    tables = provider_packet['tables']
    rows = [row for row in tables['requests'].values() if row['request_id'] == request_id]
    if len(rows) != 1:
        raise ValueError('unique_compact_debug_request_required')
    request = deepcopy(rows[0])
    def observed_fields(value):
        if type(value) is list:
            return [observed_fields(child) for child in value]
        if type(value) is not dict:
            return deepcopy(value)
        result = {}
        for key, child in value.items():
            if key == 'stability_diagnostics_ref':
                result['stability_diagnostics'] = deepcopy(tables['stability_diagnostics'][child])
            elif key == 'selected_pose_matrix_ref':
                result['selected_pose_matrix'] = deepcopy(tables['poses'][child])
            elif key in ('measured_endpoint_ref', 'endpoint_observation_ref'):
                endpoint = deepcopy(tables['endpoints'][child])
                if 'pairs_ref' in endpoint:
                    endpoint['pairs'] = deepcopy(tables['endpoint_pairs'][endpoint.pop('pairs_ref')])
                result[key[:-4]] = endpoint
            else:
                result[key] = observed_fields(child)
        return result
    request = observed_fields(request)
    state = deepcopy(tables['states'][request.pop('state_ref')])
    context = deepcopy(tables['task_contexts'][state.pop('task_context_ref')])
    def task(ref):
        value = deepcopy(tables['tasks'][ref])
        if 'input_sha256_ref' in value:
            value['input_sha256'] = deepcopy(tables['input_hashes'][value.pop('input_sha256_ref')])
        return value
    if 'task_ref' in context:
        context['task'] = task(context.pop('task_ref'))
    if 'geometry_features_ref' in context:
        context['geometry_features'] = deepcopy(tables['part_features'][context.pop('geometry_features_ref')])
    geometry = deepcopy(tables['geometry'][state.pop('geometry_ref')])
    if 'parts_ref' in geometry:
        geometry['parts'] = deepcopy(tables['part_features'][geometry.pop('parts_ref')])
    if 'task_ref' in geometry:
        geometry['task'] = task(geometry.pop('task_ref'))
    for key in ('before_sha256', 'after_sha256'):
        integrity = geometry.get('input_integrity', {})
        if key + '_ref' in integrity:
            integrity[key] = deepcopy(tables['input_hashes'][integrity.pop(key + '_ref')])
    catalog = deepcopy(tables['pose_catalogs'][state.pop('pose_catalog_ref')])
    for pose in catalog:
        pose['matrix'] = deepcopy(tables['poses'][pose.pop('matrix_ref')])
    return {**request, **state, 'task_context': context, 'geometry': geometry, 'pose_catalog': catalog}


def _reasoning_projection(packet, full_feedback):
    """Remove non-selector raw payloads only from the provider copy.

    Exact requests remain in the local audit. This is a reasoning projection,
    deliberately not a runnable or lossless replacement for those requests.
    """
    from orchestrator.skill_behavior_contract import behavior_examples
    examples = behavior_examples(full_feedback) if full_feedback else []
    packet['context']['behavior_examples'] = examples
    examples_by_id = {row['request_id']: row for row in examples}
    reasoning = []
    for index, row in enumerate(packet.get('validation_feedback', [])):
        actual = row['recovery']
        observed = examples_by_id[row['request_id']]['observed_pass_choices']
        changes = []
        if actual is not None:
            changes = [key for key in row['trigger']['chosen']
                       if row['trigger']['chosen'][key] != actual['chosen'][key]]
        reasoning.append(dict(case_id=row['case_id'], request_id=row['request_id'],
            request_ref=row['request_ref'], previous_rule_index=row['selected_rule_index'],
            previous_program_sha256=row['program_sha256'], trigger=row['trigger'],
            observed_program=deepcopy(row['observed_program']),
            attempted_recovery=actual, changed_choice_fields=changes,
            observed_same_context_pass_choices=deepcopy(observed),
            new_physical_calls=0, recommendation=None))
    # Each source transition body joins the same primary evidence row by its
    # original ID; relation edges and all denominator/failure rows remain.
    indexed = {row['evidence_id']: row for row in packet['evidence']}
    summary = packet['context'].get('initial_contact_observed_transitions')
    if summary:
        for row in summary.get('observations', []):
            target = indexed[row['evidence_id']]
            if any(target.get(key) != row[key] for key in ('source_task', 'remaining_parts',
                    'path_success', 'stability_success') if key in row):
                raise ValueError('source_transition_evidence_mismatch')
            if any(target['choice'].get(key) != value for key, value in row['choice'].items()):
                raise ValueError('source_transition_choice_mismatch')
            if target['choice']['pose_id'] != row['pose_id']:
                raise ValueError('source_transition_pose_mismatch')
            target['recorded_condition'] = {key: deepcopy(value) for key, value in row.items()
                if key not in ('evidence_id', 'source_task', 'remaining_parts', 'choice',
                               'path_success', 'stability_success', 'pose_id')}
        summary['observations'] = [{'evidence_id': row['evidence_id']}
                                   for row in summary.get('observations', [])]
    # Retain actual result hashes and condition hashes, but source filenames,
    # per-mesh provenance and endpoint distance arrays stay local. They are not
    # inputs to any skill selector/predicate/metric in this language version.
    omit = {'measured_endpoint', 'measured_endpoint_ref', 'endpoint_observation',
        'endpoint_observation_ref', 'raw_upstream', 'raw_result_path',
        'input_sha256_ref', 'before_sha256_ref', 'after_sha256_ref', 'record_sha256',
        'source_request_id', 'source_run_id', 'source_action_id', 'source_state_id',
        'source_commit', 'source_fields', 'source_geometry_sha256', 'source_run',
        'run_id', 'input_integrity'}
    def trim(value):
        if type(value) is list:
            return [trim(child) for child in value]
        if type(value) is not dict:
            return deepcopy(value)
        return {key: trim(child) for key, child in value.items() if key not in omit}
    packet = trim(packet)
    for key in ('endpoints', 'endpoint_pairs', 'input_hashes'):
        packet['tables'].pop(key, None)
    # Raw debug observations are already in the compact reasoning table; keep
    # citations and their source scope bound to an explicit row/stage.
    debug_evidence = {}
    for index, row in enumerate(reasoning):
        for stage, key in (('trigger', 'trigger'), ('attempted_recovery', 'recovery')):
            observed = row[stage]
            if observed and observed.get('evidence_id'):
                debug_evidence[observed['evidence_id']] = (index, stage)
    for row in packet['evidence']:
        if row['evidence_id'] in debug_evidence and 'observation' in row:
            index, stage = debug_evidence[row['evidence_id']]
            row.pop('observation')
            row['debug_reasoning_row'] = index
            row['debug_reasoning_stage'] = stage
    packet.pop('validation_feedback', None)
    packet['provider_request_projection'] = ('Reasoning fields only; not executable requests. '
        'Full requests, raw endpoints, source filenames and per-file hashes stay in local audit. '
        'Omission does not imply a physical result or an available blocker measurement.')
    # Put the verified comparison table before source context for the model.
    return {'provider_packet_schema': VERSION, 'debug_reasoning_table': trim(reasoning), **packet}


def _check_citations(value, allowed):
    single = {'evidence_id', 'failure_evidence_id', 'success_evidence_id', 'source_event_id',
              'same_observation_as_event_id', 'frozen_memory_evidence_id', 'parent_failure_evidence_id',
              'scoped_feedback_evidence_id', 'memory_evidence_id'}
    plural = {'evidence_ids', 'exposed_evidence_ids'}
    def walk(item):
        if type(item) is dict:
            for key, child in item.items():
                if key in ('phase', 'source_phase', 'validation_phase') and child in ('admission', 'transfer'):
                    raise ValueError('fresh_non_debug_evidence_forbidden')
                if key in single and child is not None and (type(child) is not str or child not in allowed):
                    raise ValueError('unknown_citation_reference')
                if key in plural and (type(child) is not list or any(type(c) is not str or c not in allowed for c in child)):
                    raise ValueError('unknown_citation_reference')
                walk(child)
        elif type(item) is list:
            for child in item:
                walk(child)
    walk(value)


def _observations(evidence, tables):
    indexed = {row['evidence_id']: row for row in evidence}
    if len(indexed) != len(evidence):
        raise ValueError('duplicate_evidence_id')
    aliases, merged = {}, {}
    for row in evidence:
        alias = row.get('same_observation_as_event_id')
        if alias is None:
            continue
        original = indexed.get(alias)
        if (original is None or row.get('evidence_origin') != 'frozen_episode_memory'
                or original.get('evidence_origin') != 'episode_trace'
                or original.get('frozen_memory_evidence_id') != row['evidence_id']
                or row.get('source_event_id') != alias):
            raise ValueError('unbound_memory_trace_alias')
        shared = set(row) & set(original) - {'evidence_id', 'evidence_origin'}
        if any(row[key] != original[key] for key in shared):
            raise ValueError('memory_trace_observation_disagrees')
        aliases[row['evidence_id']] = alias
        merged[alias] = {**deepcopy(row), **deepcopy(original), 'evidence_origin': 'episode_trace_and_frozen_memory'}
        for key in ('same_observation_as_event_id', 'source_event_id', 'frozen_memory_evidence_id'):
            merged[alias].pop(key, None)
    output = []
    for original in evidence:
        identifier = original['evidence_id']
        if identifier in aliases:
            output.append({'evidence_id': identifier, 'alias_of_evidence_id': aliases[identifier],
                           'evidence_origin': 'frozen_episode_memory_same_observation'})
            continue
        value = deepcopy(merged.get(identifier, original))
        if value.get('source_phase') not in (None, 'debug'):
            raise ValueError('fresh_non_debug_evidence_forbidden')
        if type(value.get('observation')) is dict:
            observed = value['observation']
            observed.pop('raw_result_path', None)
        output.append(tables.observed_fields(value))
    return output, aliases


def _initial_transitions(summary, tables):
    value = deepcopy(summary)
    observations = value.get('observations', [])
    by_id = {row['evidence_id']: row for row in observations}
    if len(by_id) != len(observations):
        raise ValueError('duplicate_transition_observation')
    for pair in value.get('pairs', []):
        for key in ('before', 'after'):
            original = pair.pop(key)
            identifier = original['evidence_id']
            if by_id.get(identifier) != original:
                raise ValueError('transition_pair_observation_mismatch')
            pair[key + '_evidence_id'] = identifier
    for row in observations:
        if 'pose_matrix' in row:
            row['pose_matrix_ref'] = tables.pose(row.pop('pose_matrix'))
    for context in value.get('geometry_contexts', {}).values():
        for key, kind in (('parts', 'part_features'), ('input_sha256', 'input_hashes')):
            if key in context:
                context[key + '_ref'] = tables.intern(kind, context.pop(key))
    return value


def compact_generation_packet(full_packet):
    """Return an independent provider projection and deterministic audit."""
    if type(full_packet) is not dict:
        raise ValueError('generation_packet_object_required')
    context = full_packet.get('context', {})
    target = (full_packet.get('operation') in ('synthesize', 'refine')
        and type(context) is dict and type(context.get('target_failure_scope')) is dict
        and context['target_failure_scope'].get('kind') == 'initial_disconnected')
    if not target:
        return deepcopy(full_packet), {'schema_version': 1, 'applied': False,
                                      'reason': 'outside_initial_contact_generation'}
    original_bytes = _bounded_json(full_packet)
    packet = deepcopy(full_packet)
    evidence = packet.get('evidence')
    feedback = packet.get('validation_feedback', [])
    if (type(evidence) is not list or not 1 <= len(evidence) <= 64
            or type(feedback) is not list or len(feedback) > 32
            or any(type(row) is not dict or type(row.get('evidence_id')) is not str for row in evidence)):
        raise ValueError('bounded_evidence_and_feedback_required')
    allowed = {row['evidence_id'] for row in evidence}
    _check_citations(packet, allowed)
    tables, request_map, debug = _Tables(), {}, []
    for item in feedback:
        if item.get('phase') != 'debug' or item.get('source_phase', 'debug') != 'debug':
            raise ValueError('fresh_non_debug_feedback_forbidden')
        request, trace, trigger, recovery = _bound_case(item)
        identifier = request['request_id']
        if identifier in request_map and request_map[identifier] != request:
            raise ValueError('debug_request_identity_conflict')
        request_map[identifier] = request
        ref = tables.request(request)
        row = {'case_id': item['case_id'], 'phase': 'debug', 'source_run': item.get('source_run'),
            'request_id': identifier, 'request_ref': ref, 'outcome': item.get('outcome'),
            'selected_rule_index': trace['rule_index'], 'triggered': trace['triggered'],
            'not_applicable_reason': trace.get('not_applicable_reason'),
            'program_sha256': trace['program_sha256'], 'observed_program': deepcopy(item.get('observed_program')),
            'trigger': trigger, 'recovery': recovery, 'physics_calls_by_interpreter': 0,
            'candidate_pruning': False, 'physics_verdict_reused': False}
        for stage in ('trigger', 'recovery'):
            if row[stage] is not None:
                original = item[stage]
                for key in ('evidence_id', 'stability_diagnostics', 'elapsed_seconds', 'raw_result_sha256', 'bound_actual_scope_checked'):
                    if key in original:
                        row[stage][key] = deepcopy(original[key])
        debug.append(tables.observed_fields(row))
    packet['evidence'], aliases = _observations(evidence, tables)
    if aliases and packet['context'].get('frozen_episode_memory', {}).get('binding_checked') is not True:
        raise ValueError('memory_trace_alias_requires_bound_source_context')
    if 'validation_feedback' in packet:
        packet['validation_feedback'] = debug
    context = packet['context']
    dropped = []
    if 'frozen_episode_memory' in context:
        original = context['frozen_episode_memory']
        context['frozen_episode_memory'] = {key: deepcopy(original[key]) for key in (
            'file_sha256', 'content_sha256', 'record_count', 'exposed_record_count', 'binding_checked',
            'hard_pruning', 'source_to_target_part_mapping', 'physics_reuse_allowed',
            'duplicate_views_of_same_observations', 'additional_physical_observation_count') if key in original}
        dropped.append('context.frozen_episode_memory.redundant_paths_ids_and_links')
    if 'initial_contact_observed_transitions' in context:
        context['initial_contact_observed_transitions'] = _initial_transitions(context['initial_contact_observed_transitions'], tables)
    if 'development_replay_diagnostics' in context:
        context.pop('development_replay_diagnostics')
        context['development_replay_representation'] = 'Bound validation_feedback rows preserve programs, reached rules, actions and outcomes.'
        dropped.append('context.development_replay_diagnostics.duplicate_debug_results')
    packet['tables'] = tables.tables
    packet['provider_packet_schema'] = VERSION
    packet['reference_semantics'] = ('*_ref fields index typed tables. Evidence aliases name the same observation, '
        'not extra trials. Debug request rows and referenced states contain reasoning inputs; '
        'exact original requests require the local audit. No observations, candidates, matrices or numeric precision were inferred.')
    for identifier, request in request_map.items():
        if restore_debug_request(packet, identifier) != request:
            raise ValueError('compact_debug_request_roundtrip_mismatch')
    packet = _reasoning_projection(packet, feedback)
    if {row['evidence_id'] for row in packet['evidence']} != allowed:
        raise ValueError('compact_citation_ids_changed')
    provider_bytes = _encoded(packet)
    if len(provider_bytes) > MAX_PROVIDER_BYTES:
        raise ValueError('compact_generation_packet_byte_limit')
    if _encoded(full_packet) != original_bytes:
        raise ValueError('local_full_packet_was_mutated')
    audit = {'schema_version': 1, 'applied': True, 'kind': VERSION,
        'full_sha256': hashlib.sha256(original_bytes).hexdigest(), 'provider_sha256': hashlib.sha256(provider_bytes).hexdigest(),
        'full_bytes': len(original_bytes), 'provider_bytes': len(provider_bytes),
        'saved_bytes': len(original_bytes) - len(provider_bytes),
        'size_components_before': packet_size_components(full_packet),
        'size_components_after': packet_size_components(packet),
        'bound_debug_case_count': len(debug), 'unique_debug_requests': len(request_map),
        'unique_state_contexts': len(tables.tables.get('states', {})),
        'evidence_count': len(evidence), 'same_observation_aliases': aliases,
        'citation_ids_unchanged': True, 'all_debug_requests_roundtrip_equal': True,
        'provider_request_is_lossless': False,
        'local_debug_requests': deepcopy(request_map),
        'local_debug_request_hashes': {key: _hash(value) for key, value in request_map.items()},
        'provider_omissions': ['raw endpoint observations/distances', 'per-file input hashes/paths',
            'repeated source provenance fields', 'raw upstream debug copies'],
        'dropped_redundant_provider_fields': dropped, 'full_local_packet_unchanged': True,
        'new_api_calls': 0, 'new_physical_calls': 0, 'recommended_action': None,
        'hard_pruning': False, 'fresh_admission_feedback_included': False}
    return packet, audit
