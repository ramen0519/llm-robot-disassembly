"""Python 3.8 append-only action observations and frozen reference retrieval.

No stored result authorizes pruning or replaces fresh physics. Cross-task IDs
remain source IDs with no inferred part/pose correspondence. This is scoped
observation memory, not proof of general blockers or validated repair skills.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from worker.explicit_action_contract import (PROTOCOL, canonical, digest,
    validate_request, validate_response, action_scope)
from worker.task_memory import _safe_path, _source_ref

RESULT_KEYS = {'status', 'path_success', 'stability_success', 'verifier_origin', 'raw', 'evidence_ref'}
RECORD_KEYS = {'schema_version', 'protocol', 'evidence_id', 'source_run_id', 'source_request_id',
    'source_state_id', 'source_action_id', 'scope', 'choice', 'result', 'parent_failure_evidence_id',
    'created_utc', 'record_sha256'}
MEMORY_BYTE_LIMIT = 32 * 1024 * 1024
SNAPSHOT_KEYS = {'schema_version', 'protocol', 'records', 'record_count', 'content_sha256',
    'hard_pruning', 'physics_reuse_allowed', 'frozen_utc'}


def _validate_parent_links(records):
    """Validate the observed recovery relation, without inferring a failure cause."""
    indexed = {record['evidence_id']: record for record in records}
    for record in records:
        parent_id = record['parent_failure_evidence_id']
        if parent_id is None:
            continue
        parent = indexed.get(parent_id)
        if parent is None or parent['result']['status'] not in ('UNKNOWN', 'REJECTED'):
            raise ValueError('parent_must_be_recorded_failed_observation')
        if (parent['source_run_id'] != record['source_run_id']
                or parent['source_state_id'] != record['source_state_id']
                or any(parent['scope'][key] != record['scope'][key] for key in
                       ('task_context', 'remaining_parts', 'base_part', 'max_grippers'))
                or parent['choice']['part_id'] != record['choice']['part_id']):
            raise ValueError('parent_failure_scope_mismatch')
        if parent['source_action_id'] == record['source_action_id']:
            raise ValueError('repair_link_requires_changed_action_conditions')
    # A failed retry can itself refer to an earlier failure. Cycles never describe
    # an observed chronology and could otherwise hang reference closure retrieval.
    finished = set()
    for record in records:
        chain, identity = set(), record['evidence_id']
        while identity is not None and identity not in finished:
            if identity in chain:
                raise ValueError('cyclic_parent_failure_links')
            chain.add(identity)
            identity = indexed[identity]['parent_failure_evidence_id']
        finished.update(chain)


def _validate_merge_provenance(value, unique_count):
    keys = {'schema_version', 'method', 'sources', 'source_record_count', 'duplicate_record_count'}
    if (type(value) is not dict or set(value) != keys or value['schema_version'] != 1
            or value['method'] != 'append_only_union_by_evidence_id_v1'
            or type(value['sources']) is not list
            or type(value['source_record_count']) is not int
            or type(value['duplicate_record_count']) is not int
            or value['duplicate_record_count'] < 0
            or value['source_record_count'] != unique_count + value['duplicate_record_count']):
        raise ValueError('invalid_merge_provenance')
    count = 0
    for source in value['sources']:
        if (type(source) is not dict or set(source) != {'kind', 'path', 'file_sha256',
                'file_bytes', 'record_count', 'content_sha256'}
                or source['kind'] not in ('snapshot', 'observations')
                or type(source['file_bytes']) is not int or not 0 <= source['file_bytes'] <= MEMORY_BYTE_LIMIT
                or type(source['record_count']) is not int or source['record_count'] < 0):
            raise ValueError('invalid_merge_source')
        _source_ref(source['path'])
        for key in ('file_sha256', 'content_sha256'):
            if (type(source[key]) is not str or len(source[key]) != 64
                    or any(c not in '0123456789abcdef' for c in source[key])):
                raise ValueError('invalid_merge_source_hash')
        count += source['record_count']
    if count != value['source_record_count']:
        raise ValueError('invalid_merge_source_count')


def _result(result):
    if (type(result) is not dict or not RESULT_KEYS.issubset(result)
            or not set(result).issubset(RESULT_KEYS | {'execution'})):
        raise ValueError('invalid_action_result_fields')
    status, path, stability = result['status'], result['path_success'], result['stability_success']
    if (status not in ('PASS', 'REJECTED', 'UNKNOWN', 'ERROR')
            or type(result['verifier_origin']) is not str or not result['verifier_origin']
            or type(result['raw']) is not dict
            or (path is not None and type(path) is not bool)
            or (stability is not None and type(stability) is not bool)):
        raise ValueError('invalid_observed_result')
    if status == 'PASS' and (path is not True or stability is not True):
        raise ValueError('pass_requires_both_actual_stages')
    if path is not True and stability is not None:
        raise ValueError('stability_cannot_be_claimed_before_path')
    if status != 'PASS' and path is True and stability is True:
        raise ValueError('inconsistent_result_status')
    _source_ref(result['evidence_ref'])
    if 'execution' in result:
        from orchestrator.execution_observation import validate_execution
        validate_execution(result['execution'], status)
    canonical(result)


def _record(record):
    if type(record) is not dict or set(record) != RECORD_KEYS:
        raise ValueError('invalid_action_record')
    if type(record['schema_version']) is not int or record['schema_version'] != 1 or record['protocol'] != PROTOCOL:
        raise ValueError('unsupported_action_record')
    _result(record['result'])
    scope, choice = record['scope'], record['choice']
    if (type(choice.get('evidence_ids')) is not list
            or any(type(x) is not str or not x for x in choice['evidence_ids'])
            or len(choice['evidence_ids']) != len(set(choice['evidence_ids']))):
        raise ValueError('invalid_recorded_citations')
    request = {'schema_version': 1, 'protocol': PROTOCOL, 'run_id': record['source_run_id'],
        'request_id': record['source_request_id'], 'state_id': record['source_state_id'],
        'remaining_parts': scope['remaining_parts'],
        'candidate_ids': [p for p in scope['remaining_parts'] if p != scope['base_part']],
        'base_part': scope['base_part'], 'max_grippers': scope['max_grippers'],
        'pose_catalog': [{'pose_id': choice['pose_id'], 'matrix': scope['pose_matrix']}],
        'attempted_action_ids': [], 'task_context': scope['task_context'], 'evidence_ids': [],
        'episode_feedback': [], 'memory_references': [], 'geometry': {}}
    validate_request(request)
    # Citations are provenance only and do not contribute to physical conditions.
    check_choice = deepcopy(choice)
    check_choice['evidence_ids'] = []
    if action_scope(request, check_choice) != scope:
        raise ValueError('action_record_scope_mismatch')
    if record['source_action_id'] != 'action:' + digest(scope):
        raise ValueError('action_record_identity_mismatch')
    if record['evidence_id'] != 'action-evidence:' + digest({k: record[k] for k in
            ('source_run_id', 'source_request_id', 'source_action_id')}):
        raise ValueError('action_evidence_id_mismatch')
    parent = record['parent_failure_evidence_id']
    if parent is not None and (type(parent) is not str or not parent.startswith('action-evidence:')):
        raise ValueError('invalid_parent_failure_id')
    if record['record_sha256'] != digest({k: v for k, v in record.items() if k != 'record_sha256'}):
        raise ValueError('action_record_hash_mismatch')
    return deepcopy(record)


def load_observations(path, allow_test_root=False):
    path = _safe_path(path, allow_test_root)
    if not path.exists():
        return []
    if path.stat().st_size > MEMORY_BYTE_LIMIT:
        raise ValueError('observation_log_limit')
    records = [_record(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line]
    if len({r['evidence_id'] for r in records}) != len(records):
        raise ValueError('duplicate_action_evidence')
    return records


def append_action_observation(path, request, response, result, parent_failure_evidence_id=None,
                              allow_test_root=False):
    request, response = validate_request(request), validate_response(request, response)
    _result(result)
    if response['choice'] is None:
        raise ValueError('no_action_to_observe')
    record = {'schema_version': 1, 'protocol': PROTOCOL,
        'source_run_id': request['run_id'], 'source_request_id': request['request_id'],
        'source_state_id': request['state_id'], 'source_action_id': response['action_id'],
        'scope': action_scope(request, response['choice']), 'choice': deepcopy(response['choice']),
        'result': deepcopy(result), 'parent_failure_evidence_id': parent_failure_evidence_id,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    record['evidence_id'] = 'action-evidence:' + digest({k: record[k] for k in
        ('source_run_id', 'source_request_id', 'source_action_id')})
    record['record_sha256'] = digest(record)
    _record(record)
    path = _safe_path(path, allow_test_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _safe_path(Path(str(path) + '.lock'), allow_test_root)
    # Serialized run-local append; fail closed on a concurrent writer or stale lock.
    with lock.open('x', encoding='utf-8') as lock_stream:
        lock_stream.write('explicit_action_append_lock')
    try:
        old = load_observations(path, allow_test_root)
        _validate_parent_links(old + [record])
        for previous in old:
            if previous['evidence_id'] == record['evidence_id']:
                if any(previous[k] != record[k] for k in record if k not in ('created_utc', 'record_sha256')):
                    raise ValueError('conflicting_action_observation')
                return previous
        content = canonical(record).encode('utf-8') + b'\n'
        if len(content) > 262144:
            raise ValueError('observation_too_large')
        with path.open('ab') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        return record
    finally:
        lock.unlink()


def freeze_memory(path, snapshot_path, allow_test_root=False):
    records = load_observations(path, allow_test_root)
    snapshot = {'schema_version': 1, 'protocol': PROTOCOL, 'records': records,
        'record_count': len(records), 'content_sha256': digest(records),
        'hard_pruning': False, 'physics_reuse_allowed': False,
        'frozen_utc': datetime.now(timezone.utc).isoformat()}
    return _write_snapshot(snapshot, snapshot_path, allow_test_root)


def _write_snapshot(snapshot, snapshot_path, allow_test_root):
    FrozenActionMemory(snapshot)
    content = canonical(snapshot).encode('utf-8')
    if len(content) > MEMORY_BYTE_LIMIT:
        raise ValueError('snapshot_output_limit')
    target = _safe_path(snapshot_path, allow_test_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return snapshot


def freeze_cumulative_memory(snapshot_paths, observation_paths, snapshot_path, allow_test_root=False):
    """Freeze an exact union of prior snapshots and new episode observations.

    The source rows and their hashes remain byte-for-byte equivalent as JSON
    values. Only identical evidence records are deduplicated; conflicting IDs
    fail closed. Sources are read-only and the destination must not exist.
    This union is not a learned constraint set, plan cache, or physics cache.
    """
    if (not isinstance(snapshot_paths, (list, tuple))
            or not isinstance(observation_paths, (list, tuple))):
        raise ValueError('cumulative_sources_must_be_path_lists')
    records, indexed, sources, total_bytes, total_records = [], {}, [], 0, 0
    for kind, paths in (('snapshot', snapshot_paths), ('observations', observation_paths)):
        for source_path in paths:
            source_path = _safe_path(source_path, allow_test_root)
            file_bytes = source_path.stat().st_size
            total_bytes += file_bytes
            if file_bytes > MEMORY_BYTE_LIMIT or total_bytes > MEMORY_BYTE_LIMIT:
                raise ValueError('cumulative_input_limit')
            with source_path.open('rb') as stream:
                raw = stream.read(MEMORY_BYTE_LIMIT + 1)
            if len(raw) != file_bytes:
                raise ValueError('cumulative_source_changed_during_read')
            if kind == 'snapshot':
                loaded = FrozenActionMemory(json.loads(raw.decode('utf-8'))).snapshot['records']
            else:
                loaded = [_record(json.loads(line)) for line in raw.decode('utf-8').splitlines() if line]
            sources.append({'kind': kind, 'path': str(source_path),
                'file_sha256': hashlib.sha256(raw).hexdigest(), 'file_bytes': len(raw),
                'record_count': len(loaded), 'content_sha256': digest(loaded)})
            total_records += len(loaded)
            for record in loaded:
                previous = indexed.get(record['evidence_id'])
                if previous is not None:
                    if previous != record:
                        raise ValueError('conflicting_cumulative_action_evidence')
                    continue
                indexed[record['evidence_id']] = record
                records.append(record)
    _validate_parent_links(records)
    snapshot = {'schema_version': 1, 'protocol': PROTOCOL, 'records': records,
        'record_count': len(records), 'content_sha256': digest(records),
        'hard_pruning': False, 'physics_reuse_allowed': False,
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'merge_provenance': {'schema_version': 1, 'method': 'append_only_union_by_evidence_id_v1',
            'sources': sources, 'source_record_count': total_records,
            'duplicate_record_count': total_records - len(records)}}
    return _write_snapshot(snapshot, snapshot_path, allow_test_root)


def load_snapshot(path, allow_test_root=False):
    path = _safe_path(path, allow_test_root)
    if path.stat().st_size > MEMORY_BYTE_LIMIT:
        raise ValueError('snapshot_limit')
    value = json.loads(path.read_text(encoding='utf-8'))
    return FrozenActionMemory(value).snapshot


def _state_diverse_references(references, request, max_records, balanced=False):
    """Select scoped observations, retaining both outcomes and recovery parents.

    Similarity only orders evidence: it never validates a current action, maps
    cross-assembly IDs, discards a candidate, or changes an observation record.
    Parent closure may retain two equivalent observations when their actual IDs
    are needed to show the recorded failure/recovery relation honestly.
    """
    by_id = {ref['evidence_id']: ref for ref in references}
    current = request['task_context']
    # Production CAD context is schema 2 with identity under ``task``. Legacy
    # fixtures/clients use a flat identity; keep both representations readable.
    current_identity = current['task'] if type(current.get('task')) is dict else current
    remaining = set(request['remaining_parts'])
    poses = [pose['matrix'] for pose in request['pose_catalog']]

    def priority(ref):
        source = ref['source_task_context']
        source_identity = source['task'] if type(source.get('task')) is dict else source
        same_assembly = (bool(current_identity.get('assembly_id')) and bool(current_identity.get('input_sha256'))
            and source_identity.get('assembly_id') == current_identity.get('assembly_id')
            and source_identity.get('input_sha256') == current_identity.get('input_sha256'))
        # IDs and pose frames from different CAD inputs are not correspondences.
        state_distance = (len(remaining.symmetric_difference(ref['source_remaining_parts']))
            if same_assembly else abs(len(remaining) - len(ref['source_remaining_parts'])))
        part_present = same_assembly and ref['source_choice']['part_id'] in request['candidate_ids']
        pose_present = same_assembly and ref['source_pose_matrix'] in poses
        return (not ref['same_conditions'], not ref['same_task'], not same_assembly,
                not part_present, state_distance, not pose_present)

    def outcome_key(ref):
        return (ref['source_action_id'], ref['status'], ref['path_success'], ref['stability_success'],
                digest(ref['execution']) if 'execution' in ref else None)

    representatives = {}
    for ref in sorted(references, key=lambda item: (
            not (item['status'] == 'PASS' and item['parent_failure_evidence_id']), item['evidence_id'])):
        representatives.setdefault(outcome_key(ref), ref)
    buckets = {}
    for ref in representatives.values():
        buckets.setdefault(priority(ref), []).append(ref)
    selected, selected_ids, represented_outcomes = [], set(), set()

    def include(ref):
        if outcome_key(ref) in represented_outcomes:
            return
        closure, item = [], ref
        while item['evidence_id'] not in selected_ids:
            closure.append(item)
            parent_id = item['parent_failure_evidence_id']
            if parent_id is None:
                break
            item = by_id[parent_id]
        if len(selected) + len(closure) > max_records:
            return
        for item in reversed(closure):
            selected.append(item)
            selected_ids.add(item['evidence_id'])
            represented_outcomes.add(outcome_key(item))

    for key in sorted(buckets):
        ordered = sorted(buckets[key], key=lambda ref: ref['evidence_id'])
        if balanced:
            # Reserve representation of both outcomes at equal relevance before
            # long failed-retry chains consume the window. Prefer the smallest
            # complete provenance bundle; never truncate its parent chain.
            def closure_size(ref):
                size, item = 0, ref
                while item['evidence_id'] not in selected_ids:
                    size += 1
                    parent = item['parent_failure_evidence_id']
                    if parent is None:
                        break
                    item = by_id[parent]
                return size

            positives = sorted([r for r in ordered if r['status'] == 'PASS'],
                               key=lambda r: (closure_size(r), r['evidence_id']))
            negatives = sorted([r for r in ordered if r['status'] != 'PASS'],
                               key=lambda r: (closure_size(r), r['evidence_id']))
            for index in range(max(len(positives), len(negatives))):
                for group in (positives, negatives):
                    if index < len(group):
                        include(group[index])
            continue
        # Show actual scoped failure/recovery together before unrelated outcomes
        # of equal relevance. A child is never returned with a missing parent.
        linked = [ref for ref in ordered if ref['parent_failure_evidence_id'] is not None]
        for ref in sorted(linked, key=lambda item: (item['status'] != 'PASS', item['evidence_id'])):
            include(ref)
        failures = [ref for ref in ordered if ref['status'] != 'PASS']
        successes = [ref for ref in ordered if ref['status'] == 'PASS']
        # Balance positive/negative evidence within the same relevance level;
        # do not bias toward PASS or erase observed failed actions.
        for index in range(max(len(failures), len(successes))):
            if index < len(failures):
                include(failures[index])
            if index < len(successes):
                include(successes[index])
    return deepcopy(selected)


class FrozenActionMemory:
    def __init__(self, snapshot=None):
        if snapshot is None:
            snapshot = {'schema_version': 1, 'protocol': PROTOCOL, 'records': [],
                'record_count': 0, 'content_sha256': digest([]), 'hard_pruning': False,
                'physics_reuse_allowed': False, 'frozen_utc': None}
        if (type(snapshot) is not dict or not SNAPSHOT_KEYS.issubset(snapshot)
                or not set(snapshot).issubset(SNAPSHOT_KEYS | {'merge_provenance'})
                or snapshot['schema_version'] != 1 or snapshot['protocol'] != PROTOCOL
                or type(snapshot['records']) is not list
                or type(snapshot['record_count']) is not int
                or snapshot['record_count'] != len(snapshot['records'])
                or snapshot['hard_pruning'] is not False or snapshot['physics_reuse_allowed'] is not False
                or snapshot['content_sha256'] != digest(snapshot['records'])):
            raise ValueError('invalid_frozen_action_memory')
        for record in snapshot['records']:
            _record(record)
        if len({r['evidence_id'] for r in snapshot['records']}) != len(snapshot['records']):
            raise ValueError('duplicate_frozen_action_evidence')
        _validate_parent_links(snapshot['records'])
        if 'merge_provenance' in snapshot:
            _validate_merge_provenance(snapshot['merge_provenance'], snapshot['record_count'])
        self.snapshot = deepcopy(snapshot)

    def retrieve(self, request, max_records=4, policy='legacy_v1'):
        request = validate_request(request)
        if type(max_records) is not int or not 0 <= max_records <= 12:
            raise ValueError('invalid_retrieval_limit')
        if policy not in ('legacy_v1', 'state_diverse_v1', 'state_balanced_v2', 'transfer_v1', 'assembly_balanced_v1'):
            raise ValueError('invalid_retrieval_policy')
        references = []
        for record in self.snapshot['records']:
            scope = record['scope']
            same_task = scope['task_context'] == request['task_context']
            same_conditions = (same_task and scope['remaining_parts'] == sorted(request['remaining_parts'])
                and scope['base_part'] == request['base_part']
                and scope['max_grippers'] == request['max_grippers']
                and scope['pose_matrix'] in [p['matrix'] for p in request['pose_catalog']])
            result = record['result']
            raw_upstream = result['raw'].get('raw_upstream')
            raw_stability_parts = (raw_upstream.get('parts_fall')
                                   if type(raw_upstream) is dict else None)
            references.append({'evidence_id': record['evidence_id'],
                'source_run_id': record['source_run_id'], 'source_request_id': record['source_request_id'],
                'source_context_sha256': digest(scope['task_context']),
                'source_task_context': deepcopy(scope['task_context']),
                'source_remaining_parts': deepcopy(scope['remaining_parts']),
                'source_pose_matrix': deepcopy(scope['pose_matrix']),
                'source_choice': deepcopy(record['choice']),
                'source_action_id': record['source_action_id'],
                'status': result['status'], 'path_success': result['path_success'],
                'stability_success': result['stability_success'], 'verifier_origin': result['verifier_origin'],
                'measured_endpoint': deepcopy(result['raw'].get('endpoint_observation')),
                'raw_stability_parts': deepcopy(raw_stability_parts),
                'source_evidence_ref': result['evidence_ref'], 'record_sha256': record['record_sha256'],
                'parent_failure_evidence_id': record['parent_failure_evidence_id'],
                'same_task': same_task, 'same_conditions': same_conditions,
                'condition_match_scope': 'serialized_declared_conditions_only',
                'reference_only': not same_conditions, 'current_part_id_mapping': None,
                'current_attempt_verified': False, 'failure_cause': None, 'blocker': None,
                'hard_pruning': False, 'physics_reuse_allowed': False,
                'repair_lifecycle': ('provisional_observed_recovery_only'
                    if result['status'] == 'PASS' and record['parent_failure_evidence_id'] else None)})
            from orchestrator.execution_observation import project_execution
            references[-1].update(project_execution(result))
            if result['raw'].get('stability_diagnostics') is not None:
                from worker.stability_diagnostics import project_stability_diagnostics
                references[-1]['stability_diagnostics'] = project_stability_diagnostics(
                    result['raw']['stability_diagnostics'])
        # Stable retrieval control; retain success and failure equally, never prune.
        references.sort(key=lambda x: (not x['same_conditions'], not x['same_task'], x['evidence_id']))
        if policy == 'assembly_balanced_v1':
            from worker.assembly_memory_retrieval import select_references
            return select_references(references, request, max_records)
        if policy == 'transfer_v1':
            from worker.transfer_retrieval import select_references
            return select_references(references, request, max_records)
        if policy in ('state_diverse_v1', 'state_balanced_v2'):
            return _state_diverse_references(references, request, max_records,
                                             balanced=policy == 'state_balanced_v2')
        return deepcopy(references[:max_records])
