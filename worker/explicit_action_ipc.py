"""Explicit-action payloads over the existing guarded file mailbox (Python 3.8)."""
import hashlib
import time

from remote_rank_client import (canonical_bytes, checked_path, read_json_bounded,
                                request_filename, write_json_new)


class ExplicitActionClient:
    def __init__(self, root, timeout_seconds=90, allow_test_root=False):
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 120:
            raise ValueError('invalid_response_timeout')
        self.root = checked_path(root, allow_test_root=allow_test_root)
        self.timeout_seconds = timeout_seconds
        self.allow_test_root = allow_test_root
        self.cache = {}
        for name in ('requests', 'responses'):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def __call__(self, request):
        envelope = {'schema_version': 1, 'kind': 'rank_request',
                    'request': {'run_id': request['run_id'], 'request_id': request['request_id'],
                                'state_id': request['state_id'],
                                'assembled_parts': request['remaining_parts'],
                                'candidate_ids': request['candidate_ids'], 'base_part': request['base_part']},
                    'features': {'explicit_action': request}, 'timeout_seconds': self.timeout_seconds}
        digest = hashlib.sha256(canonical_bytes(envelope)).hexdigest()
        rid = request['request_id']
        if rid in self.cache:
            old_digest, response = self.cache[rid]
            if digest != old_digest:
                raise ValueError('duplicate_request_id_conflict')
            return response
        filename = request_filename(rid)
        write_json_new(self.root / 'requests' / filename, envelope, self.allow_test_root)
        started = time.monotonic()
        while True:
            try:
                outer, _ = read_json_bounded(self.root / 'responses' / filename, self.allow_test_root)
                break
            except FileNotFoundError:
                if time.monotonic() - started >= self.timeout_seconds:
                    raise TimeoutError('explicit_action_response_timeout')
                time.sleep(.05)
        if (outer.get('request_id') != rid or outer.get('state_id') != request['state_id']
                or outer.get('metadata', {}).get('envelope_sha256') != digest):
            raise ValueError('explicit_action_envelope_mismatch')
        response = outer.get('metadata', {}).get('explicit_action_response')
        if type(response) is not dict:
            raise ValueError('explicit_action_response_missing')
        self.cache[rid] = (digest, response)
        return response
