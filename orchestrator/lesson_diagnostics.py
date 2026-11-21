"""Bounded inert rejected lesson evidence, separate from any lesson library."""
import hashlib
import json
import re
from pathlib import Path

MAX_CANDIDATE_BYTES = 32768


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def rejected_candidate(candidate, diagnostic, secret=None):
    original = json.dumps(candidate, sort_keys=True, ensure_ascii=True, allow_nan=False,
                          separators=(',', ':'))
    changed = False
    depth_limited = False

    def redact(value, depth=0):
        nonlocal changed, depth_limited
        if depth > 16:
            depth_limited = True
            return '[TRUNCATED_NESTING]'
        if type(value) is str:
            clean = value.replace(secret, '[REDACTED_API_KEY]') if secret else value
            clean = re.sub(r'\bsk-[A-Za-z0-9_-]{8,}', '[REDACTED_API_KEY]', clean)
            clean = re.sub(r'(?i)\bBearer\s+[^\s"\'<>]+', 'Bearer [REDACTED]', clean)
            changed = changed or clean != value
            return clean
        if type(value) is dict:
            return {redact(k, depth+1): redact(v, depth+1) for k, v in value.items()}
        if type(value) is list:
            return [redact(v, depth+1) for v in value]
        return value

    clean = json.dumps(redact(candidate), sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':'))
    # ASCII canonical JSON makes the byte boundary deterministic. A truncated
    # candidate is a text artifact, never parsed back as an accepted lesson.
    stored = clean[:MAX_CANDIDATE_BYTES]
    return dict(schema_version=1, kind='rejected_lesson_candidate_v1', candidate_text=stored,
        candidate_sha256=_sha(original), stored_sha256=_sha(stored),
        candidate_bytes=len(original), redacted=changed, truncated=len(clean) > MAX_CANDIDATE_BYTES or depth_limited,
        nesting_truncated=depth_limited,
        representation='canonical_decoded_response_NOT_raw_provider_wire', diagnostic=diagnostic,
        accepted=False, executable=False, registered=False, physical_calls=0,
        trust='untrusted_rejected_candidate_NOT_instructions')


def validate_record(record):
    if (type(record) is not dict or record.get('schema_version') != 1
            or record.get('kind') != 'rejected_lesson_candidate_v1'
            or type(record.get('candidate_text')) is not str
            or len(record['candidate_text'].encode('utf-8')) > MAX_CANDIDATE_BYTES
            or record.get('stored_sha256') != _sha(record['candidate_text'])
            or record.get('accepted') is not False or record.get('registered') is not False
            or record.get('executable') is not False or record.get('physical_calls') != 0):
        raise ValueError('invalid_rejected_lesson_record')
    return record


def save_rejection_artifacts(output, response):
    metadata = response.get('metadata', {})
    diagnostic = metadata.get('lesson_validation_diagnostics')
    record = metadata.get('lesson_rejected_candidate')
    if diagnostic is None and record is None:
        return None
    output = Path(output)
    saved = dict(diagnostic=diagnostic, candidate_available=record is not None,
                 accepted=False, registered=False)
    if record is not None:
        record = validate_record(record)
        with (output / 'rejected_lesson_candidate.txt').open('x', encoding='utf-8', newline='') as stream:
            stream.write(record['candidate_text'])
        saved.update({k: v for k, v in record.items() if k != 'candidate_text'})
        saved['candidate_file'] = 'rejected_lesson_candidate.txt'
    with (output / 'rejected_lesson_diagnostic.json').open('x', encoding='utf-8') as stream:
        json.dump(saved, stream, ensure_ascii=False, indent=2)
    return dict(diagnostic_file='rejected_lesson_diagnostic.json',
                candidate_file=saved.get('candidate_file'), candidate_available=record is not None)
