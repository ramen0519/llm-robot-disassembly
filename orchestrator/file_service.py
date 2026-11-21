"""Local file mailbox service. It never owns or advances ASAP search state."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

# Stable even when this module is loaded from a frozen run/implementation copy.
PROJECT = Path('C:/dev/disassembly_codex_ready_v2')
LIMIT = 1024 * 1024


def guarded(path, allow_test_root=False):
    text = str(path).replace('\\', '/')
    normalized = '/'.join(part for part in text.lower().split('/') if part != '.')
    if '..' in text.split('/') or normalized.startswith(('e:', '/mnt/e')):
        raise ValueError('excluded path')
    path = Path(path).absolute()
    if not allow_test_root and not path.is_relative_to(PROJECT / 'runs'):
        raise ValueError('mailbox artifacts must stay inside project runs')
    for item in reversed((path,) + tuple(path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('links are excluded')
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def atomic_new(path, value, allow_test_root=False):
    path = guarded(path, allow_test_root)
    content = canonical(value)
    if len(content) > LIMIT:
        raise ValueError('message too large')
    temporary = guarded(path.parent / ('.' + path.name + '.' + uuid.uuid4().hex + '.tmp'), allow_test_root)
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hardlink publishes a complete file and refuses replacement.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_message(path, allow_test_root=False):
    path = guarded(path, allow_test_root)
    with path.open('rb') as stream:
        content = stream.read(LIMIT + 1)
    if len(content) > LIMIT:
        raise ValueError('message too large')
    return json.loads(content)


def validate_request(envelope, expected_run_id):
    if (type(envelope) is not dict or envelope.get('schema_version') != 1
            or envelope.get('kind') != 'rank_request'):
        raise ValueError('unsupported envelope')
    request = envelope['request']
    if type(request) is not dict or request.get('run_id') != expected_run_id:
        raise ValueError('run identity mismatch')
    for key in ('request_id', 'state_id'):
        if type(request.get(key)) is not str or not 1 <= len(request[key]) <= 256:
            raise ValueError('invalid identity')
    for key in ('candidate_ids', 'assembled_parts'):
        values = request.get(key)
        if (type(values) is not list or (key == 'assembled_parts' and not values) or len(values) > 256
                or any(type(value) is not str or not value or len(value) > 128 for value in values)
                or len(values) != len(set(values))):
            raise ValueError('invalid part set')
    if (not set(request['candidate_ids']).issubset(request['assembled_parts'])
            or request.get('base_part') in request['candidate_ids']):
        raise ValueError('invalid candidates')
    if type(envelope.get('features')) is not dict:
        raise ValueError('features must be observations in an object')
    return request


class FileRankServer:
    def __init__(self, root, run_id, backend, emit, allow_test_root=False,
                 llm_config=None, transport=None, rank_service=None):
        if backend not in ('noop', 'reverse_native', 'llm_disabled', 'llm', 'pipeline'):
            raise ValueError('unknown ranking backend')
        if (backend == 'pipeline') != (rank_service is not None):
            raise ValueError('pipeline backend requires an explicit ranking service')
        if rank_service is not None and not callable(getattr(rank_service, 'rank', None)):
            raise ValueError('ranking service must expose rank(envelope)')
        if backend == 'llm' and type(llm_config) is not dict:
            raise ValueError('LLM backend requires explicit configuration')
        if backend != 'llm' and (llm_config is not None or transport is not None):
            raise ValueError('LLM configuration only applies to LLM backend')
        self.root = guarded(root, allow_test_root)
        self.test_root = allow_test_root
        self.run_id, self.backend, self.emit = run_id, backend, emit
        self.handled = {}
        for name in ('requests', 'responses'):
            guarded(self.root / name, allow_test_root).mkdir(parents=True, exist_ok=True)
        self.service = rank_service
        if backend in ('llm_disabled', 'llm'):
            from orchestrator.llm_ranker import RankService
            config = llm_config if backend == 'llm' else {'enabled': False, 'max_calls': 0, 'max_usd': 0}
            self.service = RankService(config, self.root / 'llm_ledger.sqlite', transport=transport)

    def serve_pending(self):
        for path in sorted((self.root / 'requests').glob('*.json')):
            if not re.fullmatch(r'[0-9a-f]{64}\.json', path.name):
                raise ValueError('invalid request filename')
            envelope = read_message(path, self.test_root)
            request = validate_request(envelope, self.run_id)
            if hashlib.sha256(request['request_id'].encode()).hexdigest() + '.json' != path.name:
                raise ValueError('request filename does not match identity')
            digest = hashlib.sha256(canonical(envelope)).hexdigest()
            if path.name in self.handled:
                if self.handled[path.name] != digest:
                    raise ValueError('request changed after publication')
                continue
            target = guarded(self.root / 'responses' / path.name, self.test_root)
            if target.exists():
                previous = read_message(target, self.test_root)
                if (previous.get('request_id') != request['request_id']
                        or previous.get('state_id') != request['state_id']
                        or previous.get('metadata', {}).get('envelope_sha256') != digest):
                    raise ValueError('existing response does not match request')
                self.handled[path.name] = digest
                self.emit('service_response_reused', request_id=request['request_id'])
                continue
            if self.service is not None:
                response = self.service.rank(envelope)
            else:
                candidates = list(request['candidate_ids'])
                if self.backend == 'reverse_native':
                    candidates.reverse()
                response = dict(schema_version=1, kind='rank_response',
                    request_id=request['request_id'], state_id=request['state_id'],
                    candidate_ids=candidates, fallback_reason=None,
                    metadata={'llm_live': False, 'attempted': False})
            response['metadata'].update(backend=self.backend, envelope_sha256=digest)
            atomic_new(target, response, self.test_root)
            self.handled[path.name] = digest
            self.emit('service_response', request_id=request['request_id'],
                      state_id=request['state_id'], backend=self.backend,
                      fallback_reason=response.get('fallback_reason'),
                      candidate_ids=response['candidate_ids'],
                      llm_live=response['metadata'].get('llm_live', False))

    def close(self):
        if self.service is not None and hasattr(self.service, 'close'):
            self.service.close()
