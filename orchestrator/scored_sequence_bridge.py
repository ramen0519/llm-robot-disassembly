"""Explicit, read-only admission of historical scored Python into native ranking.

This bridge grants only a data-only permutation proposal. Historical path-only
RedMax registration is retained verbatim and is NOT native-profile validation.
No API, simulator, generated import, filesystem action or fallback is executed.
"""
from copy import deepcopy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from orchestrator.python_skill_vm import LANGUAGE
from orchestrator.relational_knowledge import ROOT, digest, read_json, safe_path
from orchestrator.scored_order_skill import feature_rows, run_skill, validate_source
from worker.proposal_support_score import score_plan


PROTOCOL = 'scored_sequence_bridge/v1'
REGISTRATION_PROTOCOL = 'scored_path_order_python_registration/v1'
SOURCE_PROFILE = 'path_only_sequence_proposal/v1'
TARGET_PROFILE = 'sequence_native_v1'
MAX_FILE_BYTES = 16 * 1024 * 1024
RUNTIME_SOURCES = ('orchestrator/scored_sequence_bridge.py',
                   'orchestrator/scored_order_skill.py', 'orchestrator/python_skill_vm.py')


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _read_bytes(path):
    path = safe_path(path)
    _require(path.is_file() and path.stat().st_size <= MAX_FILE_BYTES,
             'bridge_evidence_file_missing_or_oversize')
    return path.read_bytes()


def _ids(value, label):
    _require(type(value) in (list, tuple) and 1 <= len(value) <= 128,
             'invalid_bridge_' + label)
    _require(all(type(p) is str and 0 < len(p) <= 128 for p in value)
             and len(set(value)) == len(value), 'invalid_bridge_' + label)
    return list(value)


def _registration(card):
    _require(type(card) is dict and card.get('protocol') == REGISTRATION_PROTOCOL
             and card.get('status') == 'REGISTERED_SCOPED' and not card.get('fixture'),
             'bridge_requires_registered_nonfixture_skill')
    candidate = card.get('candidate', {})
    _require(candidate.get('protocol') == 'scored_path_order_candidate/v1'
             and candidate.get('skill_id') == card.get('skill_id')
             and candidate.get('provenance', {}).get('origin') == 'actual_llm'
             and not candidate.get('fixture'), 'bridge_requires_actual_model_candidate')
    source = validate_source(candidate.get('source'))
    _require(_sha(source.encode('utf8')) == candidate.get('source_sha256'),
             'bridge_program_hash_mismatch')
    _ids(card.get('training_products'), 'training_products')
    validation = _ids(card.get('validation_products'), 'validation_products')
    proofs = card.get('validation_summary')
    _require(type(proofs) is list and len(proofs) == len(validation)
             and {p.get('product') for p in proofs} == set(validation),
             'bridge_missing_registration_proofs')
    for proof in proofs:
        _require(proof.get('qualifies') is True and proof.get('complete') is True
                 and proof.get('domain') == SOURCE_PROFILE
                 and proof.get('skill_id') == candidate['skill_id']
                 and proof.get('candidate_digest') == digest(candidate),
                 'bridge_registration_proof_mismatch')
    return candidate


def _required_source_refs(registry, card):
    run = registry.parent.parent
    paths = [registry, safe_path(card['generation_artifact'], root=run),
             safe_path(card['generation_ledger'], root=run)]
    paths.extend(safe_path(p) for p in RUNTIME_SOURCES)
    for proof in card['validation_summary']:
        folder = safe_path(run / 'validation' / proof['product'])
        paths.extend(safe_path(folder / p) for p in
                     ('verified.json', 'physics/result.json', 'physics/settings.json', 'physics/events.jsonl'))
        for step in read_json(folder / 'physics/result.json')['steps']:
            paths.append(safe_path(step['trajectory']['file'], root=folder / 'physics'))
    return {p.relative_to(ROOT).as_posix() for p in paths}


def build_bridge(registry_path):
    """Verify old on-disk evidence and return a separately frozen bridge artifact.

    Caller owns exclusive writing/freezing of the returned artifact. Only paths
    inside this repository are opened; every referenced path passes safe_path.
    """
    registry = safe_path(registry_path)
    card = read_json(registry)
    candidate = _registration(card)
    run = registry.parent.parent
    pins = {}

    def pin(path):
        path = safe_path(path)
        raw = _read_bytes(path)
        pins[path.relative_to(ROOT).as_posix()] = _sha(raw)
        return path

    pin(registry)
    generation = pin(safe_path(card['generation_artifact'], root=run))
    _require(read_json(generation) == candidate, 'bridge_generation_artifact_mismatch')
    ledger = pin(safe_path(card['generation_ledger'], root=run))
    for suffix in ('-wal', '-shm', '-journal'):
        _require(not safe_path(str(ledger) + suffix).exists(), 'bridge_generation_ledger_not_frozen')
    # immutable/read-only prevents SQLite from writing sidecars or recovering a journal.
    with closing(sqlite3.connect(ledger.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.row_factory = sqlite3.Row
        records = db.execute('SELECT * FROM rank_requests WHERE request_id=?',
                             (candidate['provenance']['request_id'],)).fetchall()
    _require(len(records) == 1, 'bridge_missing_live_generation_receipt')
    row = records[0]
    response = json.loads(row['response_json'] or '{}')
    meta = response.get('metadata', {})
    _require(row['status'] == 'completed' and row['reserved_call'] == 1
             and response.get('fallback_reason') is None
             and meta.get('candidate') == candidate and meta.get('llm_live') is True
             and meta.get('attempted') is True and meta.get('retries') == 0
             and meta.get('response_id') == candidate['provenance']['provider_response_id']
             and meta.get('call_id') == candidate['provenance']['call_id']
             and meta.get('model_returned') == candidate['provenance']['model'],
             'bridge_live_generation_receipt_mismatch')
    for proof in card['validation_summary']:
        folder = safe_path(run / 'validation' / proof['product'])
        saved = pin(folder / 'verified.json')
        _require(read_json(saved) == proof, 'bridge_saved_proof_changed')
        result_path = pin(folder / 'physics/result.json')
        result = read_json(result_path)
        settings_path = pin(folder / 'physics/settings.json')
        settings = read_json(settings_path)
        events_path = pin(folder / 'physics/events.jsonl')
        events = [json.loads(line) for line in _read_bytes(events_path).splitlines()]
        paths = {e['attempt_id']: e for e in events if e['kind'] == 'path_result'}
        audits = {e['attempt_id']: e for e in events if e['kind'] == 'stability_result'}
        _require(_sha(_read_bytes(result_path)) == proof['result_sha256']
                 and result.get('settings_sha256') == _sha(_read_bytes(settings_path))
                 and result.get('profile') == SOURCE_PROFILE and result.get('complete') is True
                 and result.get('input_unchanged') is True and result.get('asap_sources_unchanged') is True
                 and result.get('api_calls') == settings.get('api_calls') == 0,
                 'bridge_physics_result_mismatch')
        score = score_plan(result, support_cap='auto')
        _require(score == result.get('support_score') and score['eligible']
                 and score['score'] == proof['score'], 'bridge_support_witness_mismatch')
        for step in result['steps']:
            idx = step['attempt_id']
            _require(idx in paths and idx in audits
                     and all(paths[idx].get(k) == v for k, v in step.items() if k != 'stability')
                     and all(audits[idx].get(k) == v for k, v in step['stability'].items()),
                     'bridge_raw_physics_evidence_mismatch')
            trajectory = step['trajectory']
            traj_path = pin(safe_path(trajectory['file'], root=folder / 'physics'))
            _require(_sha(_read_bytes(traj_path)) == trajectory['sha256'],
                     'bridge_trajectory_hash_mismatch')
    for runtime in RUNTIME_SOURCES:
        pin(safe_path(runtime))
    value = dict(schema_version=1, protocol=PROTOCOL, skill_id=card['skill_id'],
        registration=deepcopy(card), registry_ref=registry.relative_to(ROOT).as_posix(),
        original_registry_path=registry.relative_to(ROOT).as_posix(),
        original_registry_sha256=pins[registry.relative_to(ROOT).as_posix()],
        source_task_ids=sorted(set(card['training_products']) | set(card['validation_products'])),
        source_file_hashes=pins, fixture=False,
        program=dict(language=LANGUAGE, source=candidate['source']),
        program_sha256=candidate['source_sha256'], source_profile=SOURCE_PROFILE,
        target_profile=TARGET_PROFILE, authority='candidate_order_only',
        native_physics_validation='NOT_RUN', transfer_effect='UNVERIFIED',
        hard_pruning=False, physics_reuse_allowed=False)
    value['content_sha256'] = digest(value)
    return validate_bridge(value)


def validate_bridge(value):
    """Reject altered artifacts/evidence; no legacy physics claim is promoted."""
    fields = {'schema_version', 'protocol', 'skill_id', 'registration', 'registry_ref',
              'original_registry_path', 'original_registry_sha256', 'source_task_ids',
              'source_file_hashes', 'fixture', 'program', 'program_sha256', 'source_profile',
              'target_profile', 'authority', 'native_physics_validation', 'transfer_effect',
              'hard_pruning', 'physics_reuse_allowed', 'content_sha256'}
    _require(type(value) is dict and set(value) == fields and value['schema_version'] == 1
             and value['protocol'] == PROTOCOL and value['fixture'] is False
             and value['content_sha256'] == digest({k: v for k, v in value.items() if k != 'content_sha256'}),
             'invalid_scored_sequence_bridge')
    _require(value['source_profile'] == SOURCE_PROFILE and value['target_profile'] == TARGET_PROFILE
             and value['authority'] == 'candidate_order_only'
             and value['native_physics_validation'] == 'NOT_RUN'
             and value['transfer_effect'] == 'UNVERIFIED'
             and value['hard_pruning'] is False and value['physics_reuse_allowed'] is False,
             'bridge_cannot_promote_physics_authority')
    candidate = _registration(value['registration'])
    _require(value['skill_id'] == candidate['skill_id']
             and value['program'] == dict(language=LANGUAGE, source=candidate['source'])
             and value['program_sha256'] == candidate['source_sha256'], 'bridge_program_mismatch')
    pins = value['source_file_hashes']
    _require(type(pins) is dict and 1 <= len(pins) <= 128 and value['registry_ref'] in pins,
             'bridge_missing_source_pins')
    _require(value['original_registry_path'] == value['registry_ref']
             and value['original_registry_sha256'] == pins[value['registry_ref']]
             and value['source_task_ids'] == sorted(set(value['registration']['training_products'])
                                                    | set(value['registration']['validation_products'])),
             'bridge_registry_binding_mismatch')
    for relative, checksum in pins.items():
        _require(_sha(_read_bytes(safe_path(relative))) == checksum, 'bridge_source_hash_changed')
    _require(read_json(safe_path(value['registry_ref'])) == value['registration'],
             'bridge_registry_snapshot_mismatch')
    _require(set(pins) == _required_source_refs(safe_path(value['registry_ref']), value['registration']),
             'bridge_source_pin_set_mismatch')
    return deepcopy(value)


def compact_bridge(bridge):
    bridge = validate_bridge(bridge)
    card = bridge['registration']
    candidate = card['candidate']
    return dict(protocol=PROTOCOL, skill_id=bridge['skill_id'], name=candidate['name'],
        description=candidate['description'], program=deepcopy(bridge['program']),
        program_sha256=bridge['program_sha256'], content_sha256=bridge['content_sha256'],
        source_task_ids=deepcopy(bridge['source_task_ids']),
        original_registry_path=bridge['original_registry_path'],
        original_registry_sha256=bridge['original_registry_sha256'],
        fixture=False, original_scope=card['scope'], source_profile=SOURCE_PROFILE,
        target_profile=TARGET_PROFILE, authority='candidate_order_only',
        native_physics_validation='NOT_RUN', performance_effect='UNVERIFIED',
        provider=deepcopy(candidate['provenance']),
        validation=dict(status=card['status'], profile=SOURCE_PROFILE,
                        products=deepcopy(card['validation_products']),
                        native_physics_validation='NOT_RUN'),
        feature_scope='all remaining parts including fixed base; base never proposed')


def execute_bridge(bridge, cad, request):
    """Run identical historical Python on full-state features, filter only base.

    Full rows preserve original parent_count/contact/geometry semantics and the
    unchanged run_skill contract. The base is evaluated internally but never
    appears in proposed candidates or exported priorities. No fallback is chosen.
    """
    result = dict(protocol=PROTOCOL, status='REJECTED', candidate_order=None,
        priorities={}, diagnostics=[], vm_steps=0, vm_executed=False,
        physics_executed=False, current_attempt_verified=False,
        source_profile=SOURCE_PROFILE, native_physics_validation='NOT_RUN',
        fallback_required=True, fallback_applied=False, base_score_evaluated=False)
    try:
        bridge = validate_bridge(bridge)
        result.update(skill_id=bridge['skill_id'], bridge_sha256=bridge['content_sha256'],
                      program_sha256=bridge['program_sha256'])
        _require(type(cad) is dict and type(request) is dict, 'invalid_bridge_request')
        assembled = _ids(request.get('assembled_parts', request.get('remaining_parts')), 'assembled_parts')
        candidates = _ids(request.get('candidate_ids', request.get('candidates')), 'candidate_ids')
        base = request.get('base_part')
        _require(base is None or type(base) is str and base in assembled, 'invalid_bridge_base')
        _require(set(candidates) == set(assembled) - ({base} if base is not None else set()),
                 'bridge_requires_all_nonbase_candidates')
        task = cad.get('task', {})
        _require(type(task) is dict, 'invalid_bridge_task_metadata')
        identity = task.get('assembly_id', request.get('assembly_id'))
        _require(type(identity) is str and identity
                 and request.get('assembly_id', identity) == identity, 'bridge_task_identity_mismatch')
        purpose = request.get('purpose', 'evaluation')
        _require(purpose in ('evaluation', 'training_replay'), 'invalid_bridge_execution_purpose')
        result['transfer_evaluation'] = purpose == 'evaluation'
        excluded = set(bridge['registration']['training_products']) | set(bridge['registration']['validation_products'])
        if purpose == 'evaluation' and identity in excluded:
            return dict(result, status='DECLINED', diagnostics=['current_task_in_source_products'])
        if len(assembled) <= 2:
            return dict(result, status='DECLINED', diagnostics=['native_terminal_rule_owns_last_two'])
        relations = cad.get('relations', {})
        _require(type(relations) is dict, 'invalid_bridge_relations')
        contacts = cad.get('contact_edges', relations.get('contact_edges'))
        _require(type(contacts) is list and type(cad.get('parts')) is dict
                 and len(cad['parts']) <= 128 and set(assembled) <= cad['parts'].keys(),
                 'bridge_missing_full_state_geometry')
        _require(all(type(edge) in (list, tuple) and len(edge) == 2
                     and all(type(p) is str and p in cad['parts'] for p in edge)
                     and edge[0] != edge[1] for edge in contacts), 'bridge_invalid_contact_edges')
        snapshot = dict(parts=cad['parts'], contact_edges=contacts)
        json.dumps(snapshot, allow_nan=False)
        rows = feature_rows(snapshot, assembled)
        raw = run_skill(bridge['program']['source'], rows)
        result.update(vm_executed=True, vm_steps=raw['vm_steps'],
                      base_score_evaluated=(False if base is None else True if raw['status'] == 'PROPOSED' else None),
                      full_feature_rows_sha256=digest(rows),
                      feature_parent_count=len(assembled), candidate_count=len(candidates))
        if raw['status'] != 'PROPOSED':
            return dict(result, status='DECLINED' if raw['status'] == 'ABSTAIN' else 'REJECTED',
                        diagnostics=deepcopy(raw['diagnostics']))
        full_order = raw.get('candidate_order')
        _require(type(full_order) is list and len(full_order) == len(assembled)
                 and len(set(full_order)) == len(full_order) and set(full_order) == set(assembled),
                 'bridge_vm_invalid_full_permutation')
        order = [p for p in full_order if p != base]
        _require(len(order) == len(candidates) and set(order) == set(candidates),
                 'bridge_vm_invalid_candidate_permutation')
        priorities = {p: raw['priorities'][p] for p in candidates}
        diagnostics = deepcopy(raw['diagnostics'])
        if len(set(priorities.values())) == 1 and 'all_candidate_priorities_tied' not in diagnostics:
            diagnostics.append('all_candidate_priorities_tied')
        return dict(result, status='PROPOSED', candidate_order=order, priorities=priorities,
                    diagnostics=diagnostics, fallback_required=False)
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError, OSError) as exc:
        return dict(result, diagnostics=[getattr(exc, 'diagnostic', {'code': str(exc)[:180]})])
