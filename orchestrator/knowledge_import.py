"""Read-only discovery of historical knowledge; no execution or registration.

Every imported record retains its source bytes' hash and JSON pointer. Similar
looking local IDs are never treated as globally unique. Reparse points are not
followed, including before path resolution.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat

MAX_FILE_BYTES = 64 * 1024 * 1024
REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024)
SKIP_DIRS = frozenset({'.git', '.secrets', '.codex', '.claude', '.codex-attachments',
    '.venv-orchestrator', '__pycache__', 'node_modules', 'requests', 'responses',
    'source_snapshot', 'code_snapshot', 'before', 'backup', 'backups', 'archive',
    'external', 'relational_knowledge_migration_20261119'})
NAME_PARTS = ('memory', 'experience', 'lesson', 'skill', 'knowledge', 'observations')
EXACT_NAMES = frozenset({'candidate.json', 'candidate_bundle.json', 'common_start.json',
                         'registry.json', 'library.json'})
METADATA_PARTS = ('request', 'response', 'authorization', 'budget', 'secret', 'api_key')
SECRET = re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _regular(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & REPARSE:
        raise ValueError('link_or_reparse_point_not_allowed')
    return info


def safe_path(root, relative):
    # Reject prohibited or absolute strings before resolving or inspecting them.
    value = str(relative).replace('\\', '/')
    if PureWindowsPath(value).drive or value.startswith('/') or '..' in value.split('/'):
        raise ValueError('repository_relative_path_required')
    root = Path(root).absolute()
    if PureWindowsPath(str(root)).drive.upper() == 'E:':
        raise ValueError('prohibited_drive')
    for ancestor in reversed((root,) + tuple(root.parents)):
        _regular(ancestor)
    current = root
    for part in value.split('/'):
        if part in ('', '.'):
            continue
        current = current / part
        _regular(current)
    return current


def _candidate(name):
    low = name.lower()
    return low.endswith(('.json', '.jsonl')) and (low in EXACT_NAMES or any(t in low for t in NAME_PARTS))


def discover_sources(root, roots=('runs', 'reference')):
    """Return inventory rows, including lexically excluded metadata files."""
    root = Path(root).absolute()
    found = []
    for prefix in roots:
        try:
            start = safe_path(root, prefix)
        except FileNotFoundError:
            continue
        for base, dirs, names in os.walk(start, followlinks=False):
            retained = []
            for name in sorted(dirs):
                if name in SKIP_DIRS or name.startswith('.'):
                    continue
                try:
                    _regular(Path(base) / name)
                except (ValueError, OSError):
                    continue
                retained.append(name)
            dirs[:] = retained
            for name in sorted(names):
                if not _candidate(name):
                    continue
                relative = (Path(base) / name).relative_to(root).as_posix()
                if any(t in name.lower() for t in METADATA_PARTS):
                    found.append(dict(path=relative, status='EXCLUDED',
                                      reason='request_response_budget_or_secret_metadata', sha256=None))
                    continue
                try:
                    path = safe_path(root, relative)
                    info = _regular(path)
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError('not_regular_file')
                    found.append(dict(path=relative, bytes=info.st_size, status='DISCOVERED'))
                except (OSError, ValueError) as error:
                    found.append(dict(path=relative, status='EXCLUDED', reason=str(error), sha256=None))
    return sorted(found, key=lambda row: row['path'])


def _source_id(kind, record):
    for key in ('evidence_id', 'observation_id', 'lesson_id', 'skill_id', 'candidate_id', 'record_id',
                'strategy_id', 'id', 'content_sha256', 'program_hash'):
        if isinstance(record.get(key), (str, int)):
            return str(record[key])
    if kind == 'skill' and isinstance(record.get('candidate'), dict):
        return _source_id(kind, record['candidate'])
    if kind == 'lesson' and isinstance(record.get('lesson'), dict):
        return _source_id(kind, record['lesson'])
    return None


def _fixture(record, document, path):
    flags = [document.get('fixture'), document.get('fixture_registry'),
             record.get('fixture'), record.get('fixture_registry')]
    for key in ('candidate', 'admission', 'lesson', 'generation', 'provenance'):
        sub = record.get(key)
        if isinstance(sub, dict):
            flags.extend((sub.get('fixture'), sub.get('fixture_registry')))
            if str(sub.get('origin', '')).lower() in ('fixture', 'mock', 'injected_fixture'):
                return True
    if any(flag is True for flag in flags):
        return True
    if any('fixture' in segment.lower() or 'synthetic' in segment.lower() or segment.lower() in ('mock', 'mocks', 'tests')
           for segment in path.split('/')):
        return True
    return False if any(flag is False for flag in flags) else None


def _envelope(document):
    return {key: deepcopy(document[key]) for key in (
        'protocol', 'kind', 'schema_version', 'fixture', 'fixture_registry',
        'registered', 'physics_validated', 'usable_for_actual_planning', 'hard_pruning',
        'physics_reuse_allowed', 'packet_sha256', 'profile', 'condition',
        'source_sha256', 'content_sha256', 'provenance') if key in document}


def extract_records(document):
    """Yield (kind, JSON pointer, original record) for known historical shapes."""
    def usable(kind, row):
        if not isinstance(row, dict):
            return False
        if kind == 'experience':
            return any(k in row for k in ('result', 'raw_result', 'outcome', 'observation_id', 'evidence_id',
                                          'evidence_event_id', 'path_success', 'path_status', 'first_features'))
        if kind == 'lesson':
            return any(k in row for k in ('claim', 'statement', 'hypothesis', 'lesson', 'text'))
        return any(k in row for k in ('source', 'program', 'candidate', 'python_source',
                                     'strategy_id', 'skill_id'))

    if isinstance(document, list):
        for index, row in enumerate(document):
            if not isinstance(row, dict):
                continue
            if isinstance(row.get('records'), list) or isinstance(row.get('entries'), list):
                for nested_kind, pointer, nested in extract_records(row):
                    yield nested_kind, '/' + str(index) + pointer, nested
                continue
            if 'claim' in row or 'statement' in row or 'lesson_id' in row:
                kind = 'lesson'
            elif 'program' in row or 'candidate' in row or 'skill_id' in row:
                kind = 'skill'
            elif 'source' in row and ('skill_id' in row or 'description' in row):
                kind = 'skill'
            elif usable('experience', row):
                kind = 'experience'
            else:
                continue
            if usable(kind, row):
                yield kind, '/' + str(index), row
        return
    if not isinstance(document, dict):
        return
    # Standalone generated candidates/lessons (not log views or prompt packets).
    if 'program' in document and ('candidate_id' in document or 'program_hash' in document):
        yield 'skill', '', document
        return
    if 'source' in document and ('skill_id' in document or 'source_sha256' in document):
        yield 'skill', '', document
        return
    if 'candidate' in document and 'skill_id' in document:
        yield 'skill', '', document
        return
    if 'python_registered' in document and 'intended_use' in document:
        yield 'skill', '', document
        return
    if 'claim' in document and ('observed_facts' in document or 'facts' in document):
        yield 'lesson', '', document
        return
    if 'statement' in document and 'lesson_id' in document:
        yield 'lesson', '', document
        return
    if 'experience_id' in document and 'result' in document:
        yield 'experience', '', document
        return
    if 'actual_order' in document and 'verifier_profile' in document:
        yield 'experience', '', document
        return
    if 'entries' in document and isinstance(document['entries'], list):
        for index, row in enumerate(document['entries']):
            if not isinstance(row, dict):
                continue
            if 'candidate' in row:
                yield 'skill', f'/entries/{index}', row
            elif 'program' in row or 'skill_id' in row:
                yield 'skill', f'/entries/{index}', row
            elif 'lesson' in row or 'claim' in row:
                yield 'lesson', f'/entries/{index}', row
            elif usable('experience', row):
                yield 'experience', f'/entries/{index}', row
        return
    for field, kind in (('records', 'experience'), ('observations', 'experience'),
                        ('examples', 'experience'), ('evidence', 'experience'),
                        ('lessons', 'lesson'), ('skills', 'skill'), ('cards', 'skill')):
        rows = document.get(field)
        if isinstance(rows, list):
            for index, row in enumerate(rows):
                row_kind = 'skill' if field == 'records' and isinstance(row, dict) and 'skill_id' in row else kind
                if usable(row_kind, row):
                    yield row_kind, f'/{field}/{index}', row


def _namespace(kind, row, envelope, path):
    prov = row.get('provenance') or {}
    packet = prov.get('packet_sha256') or envelope.get('packet_sha256')
    if not packet:
        packet = (envelope.get('provenance') or {}).get('packet_sha256')
    if packet:
        return 'packet:' + str(packet)
    source_id = _source_id(kind, row)
    if source_id and re.search(r'[a-f0-9]{40,64}', source_id):
        return 'content-addressed'
    if row.get('source_run_id'):
        return 'run:' + str(row['source_run_id'])
    # Local eXX / pXX and unspecified IDs cannot join across arbitrary files.
    return 'file:' + path


def import_sources(root, inventory=None):
    """Return {inventory, records, summary}; no writes and no schema promotion."""
    root = Path(root).absolute()
    inventory = deepcopy(inventory if inventory is not None else discover_sources(root))
    imported = {}
    rejected_rows = []
    for item in inventory:
        if item['status'] != 'DISCOVERED':
            continue
        try:
            path = safe_path(root, item['path'])
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                item.update(status='NOT_IMPORTED', reason='file_exceeds_64_mib', sha256=None)
                continue
            blob = path.read_bytes()
            item['sha256'] = hashlib.sha256(blob).hexdigest()
            text = blob.decode('utf-8-sig')
            if SECRET.search(text):
                item.update(status='NOT_IMPORTED', reason='credential_pattern_detected_no_content_export')
                continue
            def load_json(value):
                def pairs(items):
                    result = {}
                    for key, val in items:
                        if key in result:
                            raise ValueError('duplicate_json_key')
                        result[key] = val
                    return result
                return json.loads(value, object_pairs_hook=pairs,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite_json')))
            if path.suffix.lower() == '.jsonl':
                doc = [load_json(line) for line in text.splitlines() if line.strip()]
            else:
                doc = load_json(text)
            if isinstance(doc, dict) and doc.get('kind') in ('knowledge_context_v1', 'agent_knowledge_catalog_v1'):
                item.update(status='NOT_IMPORTED', reason='derived_runtime_view_not_source_library')
                continue
            # Packet, audit and summary files are evidence indexes, not new memories.
            low = path.name.lower()
            if any(t in low for t in ('packet', 'audit', 'summary', 'manifest', 'preview', 'contexts', 'selections', 'hashes', 'started')):
                item.update(status='NOT_IMPORTED', reason='derived_packet_audit_or_metadata')
                continue
            envelope = _envelope(doc) if isinstance(doc, dict) else {}
            item_counts = Counter()
            for kind, pointer, row in extract_records(doc):
                namespace = _namespace(kind, row, envelope, item['path'])
                claim_context = {k: envelope[k] for k in ('fixture', 'fixture_registry', 'registered',
                    'physics_validated', 'usable_for_actual_planning', 'condition', 'profile') if k in envelope}
                key = digest(dict(kind=kind, namespace=namespace, record=row, claim_context=claim_context))
                origin = dict(source_path=item['path'], json_pointer=pointer,
                              sha256=item['sha256'], source_id=_source_id(kind, row),
                              fixture=_fixture(row, doc if isinstance(doc, dict) else {}, item['path']),
                              format='jsonl_parsed_as_array' if path.suffix.lower() == '.jsonl' else 'json',
                              envelope=envelope)
                if key not in imported:
                    imported[key] = dict(record_id=key, kind=kind, namespace=namespace,
                        source_id=origin['source_id'], record=deepcopy(row),
                        source_path=item['path'], json_pointer=pointer, sha256=item['sha256'],
                        envelope=envelope, origins=[])
                imported[key]['origins'].append(origin)
                item_counts[kind] += 1
            if item_counts:
                item.update(status='IMPORTED', records=dict(item_counts))
            else:
                empty = isinstance(doc, dict) and any(k in doc and doc[k] == [] for k in ('entries', 'records', 'observations', 'skills', 'lessons'))
                item.update(status='EMPTY_SOURCE' if empty else 'NOT_IMPORTED', reason='empty_knowledge_library' if empty else 'no_supported_knowledge_records',
                            document_keys=list(doc)[:32] if isinstance(doc, dict) else type(doc).__name__)
        except (OSError, ValueError, TypeError, KeyError) as error:
            item.update(status='ERROR', reason=type(error).__name__ + ':' + str(error)[:180])
    records = sorted(imported.values(), key=lambda row: (row['kind'], row['record_id']))
    return dict(protocol='historical_knowledge_import/v1', inventory=inventory, records=records,
                summary=dict(files=len(inventory), file_statuses=dict(Counter(i['status'] for i in inventory)),
                             unique_records=dict(Counter(r['kind'] for r in records)),
                             record_occurrences=sum(len(r['origins']) for r in records),
                             fixtures=sum(all(o['fixture'] is True for o in r['origins']) for r in records),
                             api_calls=0, physics_calls=0, existing_files_modified=False))
