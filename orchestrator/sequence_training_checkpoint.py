"""Immutable review checkpoint for five tasks after two completed waves.

No model, process, simulator or automatic resume is invoked here. All opened
artifacts stay inside this project's runs directory. Continuation consumes a
single exclusive claim in a separate directory; an uncertain claim is never
deleted or reused automatically.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from orchestrator.sequence_knowledge import validate_native_snapshot
from orchestrator.sequence_learning import (validate_sequence_lesson_library,
    validate_sequence_skill_library)
from worker.task_memory import PROJECT, _safe_path, digest


PROTOCOL = 'sequence_training_checkpoint_v1'
CLAIM_PROTOCOL = 'sequence_training_continuation_claim_v1'
CLAIM_ROOT = PROJECT / 'runs' / 'sequence_iterative_claims'
LEARNING_TERMINALS = frozenset(('REGISTERED', 'DUPLICATE_PROGRAM', 'NO_OBSERVATIONS',
    'LESSON_REJECTED', 'CANDIDATE_REJECTED', 'VALIDATION_FAILED', 'POSTLESSON_REJECTED'))
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


def _hash(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('checkpoint_invalid_sha256')
    return value


def _read_bytes(path):
    path = _safe_path(path)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError('checkpoint_artifact_too_large')
    return path.read_bytes()


def _decode(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('checkpoint_duplicate_json_key')
            result[key] = value
        return result
    return json.loads(raw.decode('utf8'), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('checkpoint_nonfinite_json')))


def _object(path):
    return _decode(_read_bytes(path))


def artifact_ref(path):
    """Return a content-addressed local reference after lexical path checks."""
    path = _safe_path(path)
    return dict(path=str(path), sha256=hashlib.sha256(_read_bytes(path)).hexdigest())


def _open_ref(ref):
    if type(ref) is not dict or set(ref) != {'path', 'sha256'}:
        raise ValueError('checkpoint_invalid_artifact_ref')
    _hash(ref['sha256'])
    raw = _read_bytes(ref['path'])
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('checkpoint_artifact_hash_changed')
    return _decode(raw)


def _pins(value):
    if type(value) is not dict or not value:
        raise ValueError('checkpoint_source_pins_required')
    for key, val in value.items():
        if type(key) is not str or not key or '\x00' in key:
            raise ValueError('checkpoint_invalid_source_pin_name')
        _hash(val)
    return deepcopy(value)


def _split(manifest, supplied_held):
    tasks = manifest.get('tasks') if type(manifest) is dict else None
    if type(tasks) is not list or len(tasks) != 5:
        raise ValueError('checkpoint_requires_five_tasks')
    ids = [t.get('assembly_id') if type(t) is dict else None for t in tasks]
    if any(type(i) is not str or not i for i in ids) or len(set(ids)) != 5:
        raise ValueError('checkpoint_invalid_task_ids')
    if any(t.get('role', 'training') != 'training' for t in tasks):
        raise ValueError('checkpoint_nontraining_task_role')
    held = manifest.get('evaluation_task_ids', []) + manifest.get('development_holdout_ids', ['00335'])
    if (type(supplied_held) not in (list, tuple) or any(type(i) is not str or not i for i in held)
            or set(supplied_held) != set(held) or set(ids).intersection(held)):
        raise ValueError('checkpoint_split_or_holdout_mismatch')
    return ids, sorted(set(held))


def _completed(rows, ids):
    if type(rows) is not list or len(rows) != 10:
        raise ValueError('checkpoint_requires_ten_completed_rounds')
    expected = [(task, wave) for wave in (1, 2) for task in ids]
    actual, paths = [], set()
    for row in rows:
        if (type(row) is not dict or row.get('status') != 'COMPLETED'
                or type(row.get('round_index')) is not int
                or row.get('learning_status') not in LEARNING_TERMINALS
                or row.get('halt_required') or row.get('stop_reason')
                or row.get('episode_status') not in ('PASS', 'UNKNOWN')):
            raise ValueError('checkpoint_incomplete_or_uncertain_round')
        host = _open_ref(row.get('episode_ref'))
        episode_path = str(_safe_path(row['episode_ref']['path']))
        if episode_path in paths:
            raise ValueError('checkpoint_duplicate_episode_reference')
        paths.add(episode_path)
        metrics = host.get('metrics', {}) if type(host) is dict else {}
        if (host.get('status') not in ('PASS', 'UNKNOWN') or metrics.get('process_exit') != 0
                or metrics.get('service_error') is not None
                or metrics.get('sources_and_frozen_inputs_preserved') is not True):
            raise ValueError('checkpoint_episode_host_not_accepted')
        actual.append((row.get('assembly_id'), row['round_index']))
    if actual != expected:
        raise ValueError('checkpoint_completed_wave_order_or_tasks_mismatch')
    return deepcopy(rows)


def _knowledge(refs, ids, held, allow_fixture):
    if type(refs) is not dict or set(refs) != {'memory', 'lessons', 'skills'}:
        raise ValueError('checkpoint_requires_three_frozen_artifacts')
    snapshots = _open_ref(refs['memory'])
    lessons = validate_sequence_lesson_library(_open_ref(refs['lessons']))
    skills = validate_sequence_skill_library(_open_ref(refs['skills']))
    if type(snapshots) is not list:
        raise ValueError('checkpoint_memory_must_be_native_snapshot_list')
    if not allow_fixture and (lessons['fixture'] or skills['fixture']):
        raise ValueError('checkpoint_fixture_knowledge_not_live')
    seen, source_tasks, sources = set(), set(), {}
    for snapshot in snapshots:
        validate_native_snapshot(snapshot)
        for row in snapshot['records']:
            if row['evidence_id'] in seen:
                raise ValueError('checkpoint_duplicate_native_evidence')
            seen.add(row['evidence_id'])
            context = row['source_task_context']
            source_tasks.add(context.get('task', context).get('assembly_id'))
            ref = dict(path=row['source_ref'], sha256=row['source_sha256'])
            identity = digest(ref)
            if identity not in sources:
                source = _open_ref(ref)
                if (source.get('profile') != 'sequence_native_v1'
                        or type(source.get('attempts')) is not list):
                    raise ValueError('checkpoint_native_worker_source_required')
                if not allow_fixture and (source.get('actual_physics') is not True or source.get('fixture')):
                    raise ValueError('checkpoint_fixture_memory_not_live')
                sources[identity] = {a['event_id']: a for a in source['attempts']}
            if sources[identity].get(row['source_observation_id']) != row['attempt']:
                raise ValueError('checkpoint_native_observation_binding_mismatch')
    for lesson in lessons['entries']:
        source_tasks.update(f['source_task']['assembly_id'] for f in lesson['facts'])
        if not {f['evidence_id'] for f in lesson['facts']}.issubset(seen):
            raise ValueError('checkpoint_lesson_observations_missing')
    for skill in skills['entries']:
        source_tasks.update(skill['source_task_ids'])
        validation = skill['validation']
        for path in validation['run_refs']:
            if artifact_ref(path)['sha256'] not in validation['source_hashes']:
                raise ValueError('checkpoint_skill_proof_hash_changed')
    if None in source_tasks or source_tasks.intersection(held) or not source_tasks.issubset(ids):
        raise ValueError('checkpoint_knowledge_task_leakage')
    return dict(snapshots=deepcopy(snapshots), lessons=deepcopy(lessons['entries']),
                skills=deepcopy(skills['entries']))


def make_checkpoint(checkpoint_path, *, manifest_path, source_pins, budget_profiles,
                    frozen, completed_rounds, held_out_ids, coordinator_status='REVIEW_REQUIRED',
                    max_rounds=5, review_after_round=2, allow_fixture=False):
    """Validate all frozen artifacts, then write a new immutable review gate."""
    if coordinator_status != 'REVIEW_REQUIRED' or max_rounds != 5 or review_after_round != 2:
        raise ValueError('checkpoint_only_after_complete_first_two_waves')
    path = _safe_path(checkpoint_path)
    manifest_ref = artifact_ref(manifest_path)
    manifest = _open_ref(manifest_ref)
    ids, held = _split(manifest, held_out_ids)
    rows = _completed(completed_rounds, ids)
    if type(budget_profiles) is not dict or not budget_profiles:
        raise ValueError('checkpoint_budget_profiles_required')
    refs = {key: artifact_ref(val) for key, val in frozen.items()}
    _knowledge(refs, ids, held, allow_fixture)
    value = dict(schema_version=1, protocol=PROTOCOL, state='REVIEW_REQUIRED',
        manifest=manifest_ref, source_pins=_pins(source_pins), budget_profiles=deepcopy(budget_profiles),
        task_ids=ids, held_out_ids=held, max_rounds=5, completed_through_round=2,
        completed_rounds=rows, completed_rounds_sha256=digest(rows), knowledge=refs,
        fixture_allowed=bool(allow_fixture), automatic_retries=0, automatic_resume=False)
    value['content_sha256'] = digest(value)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return deepcopy(value)


def load_checkpoint(checkpoint_path, *, manifest_path, source_pins, budget_profiles, allow_fixture=False):
    """Reopen every hash-bound input; caller supplies authoritative current pins."""
    value = _object(checkpoint_path)
    if (type(value) is not dict or value.get('schema_version') != 1 or value.get('protocol') != PROTOCOL
            or value.get('content_sha256') != digest({k:v for k,v in value.items() if k != 'content_sha256'})
            or value.get('state') != 'REVIEW_REQUIRED' or value.get('max_rounds') != 5
            or value.get('completed_through_round') != 2 or value.get('automatic_retries') != 0
            or value.get('automatic_resume') is not False):
        raise ValueError('checkpoint_hash_or_state_invalid')
    if value['source_pins'] != _pins(source_pins):
        raise ValueError('checkpoint_source_pins_changed')
    if value['budget_profiles'] != budget_profiles:
        raise ValueError('checkpoint_budget_profile_changed')
    if value['manifest'] != artifact_ref(manifest_path):
        raise ValueError('checkpoint_manifest_changed')
    ids, held = _split(_open_ref(value['manifest']), value['held_out_ids'])
    if ids != value['task_ids']:
        raise ValueError('checkpoint_task_order_changed')
    rows = _completed(value['completed_rounds'], ids)
    if digest(rows) != value['completed_rounds_sha256']:
        raise ValueError('checkpoint_completed_round_digest_changed')
    if value['fixture_allowed'] and not allow_fixture:
        raise ValueError('checkpoint_fixture_checkpoint_not_live')
    knowledge = _knowledge(value['knowledge'], ids, held, allow_fixture)
    return dict(checkpoint=deepcopy(value), **knowledge, completed_rounds=rows,
                next_round_by_task={task:3 for task in ids})


def claim_continuation(checkpoint_path, *, new_run_dir, manifest_path, source_pins,
                       budget_profiles, continue_after_review=False, claim_root=CLAIM_ROOT,
                       allow_fixture=False):
    """Consume a one-shot review gate. No stale/failed claim is auto-released."""
    if continue_after_review is not True:
        raise ValueError('checkpoint_explicit_review_continuation_required')
    checkpoint_path = _safe_path(checkpoint_path)
    root = _safe_path(claim_root)
    if root != _safe_path(CLAIM_ROOT):
        raise ValueError('checkpoint_claim_root_is_fixed')
    destination = _safe_path(new_run_dir)
    if destination == checkpoint_path.parent or checkpoint_path.parent in destination.parents:
        raise ValueError('checkpoint_continuation_requires_new_run')
    loaded = load_checkpoint(checkpoint_path, manifest_path=manifest_path, source_pins=source_pins,
                             budget_profiles=budget_profiles, allow_fixture=allow_fixture)
    root.mkdir(parents=True, exist_ok=True)
    identity = loaded['checkpoint']['content_sha256']
    path = _safe_path(root/(identity+'.json'))
    claim = dict(schema_version=1, protocol=CLAIM_PROTOCOL,
        checkpoint=artifact_ref(checkpoint_path), checkpoint_content_sha256=identity,
        new_run_dir=str(destination), state='CLAIMED_NO_AUTOMATIC_RESUME',
        first_round=3, last_round=5, explicit_review=True)
    claim['content_sha256'] = digest(claim)
    try:
        with path.open('x', encoding='utf8') as stream:
            json.dump(claim, stream, indent=2, ensure_ascii=False, allow_nan=False)
    except FileExistsError as exc:
        raise ValueError('checkpoint_continuation_already_claimed_or_uncertain') from exc
    return dict(**loaded, claim=claim, claim_path=str(path))
