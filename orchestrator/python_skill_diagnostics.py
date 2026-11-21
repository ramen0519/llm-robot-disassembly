"""Inert, bounded rejected-source evidence; never executable registry input."""
import hashlib
import json
import re
from pathlib import Path

from orchestrator.python_skill_vm import MAX_SOURCE


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()


def rejected_source(program, error, secret=None):
    source = program.get('source') if type(program) is dict else None
    if type(source) is not str:
        return None
    text = source
    if secret:
        text = text.replace(secret, '[REDACTED_API_KEY]')
    text = re.sub(r'\bsk-[A-Za-z0-9_-]{8,}', '[REDACTED_API_KEY]', text)
    redacted = text != source
    raw = text.encode('utf-8', errors='replace')
    stored = raw[:MAX_SOURCE].decode('utf-8', errors='ignore')
    code = str(error)
    if not re.fullmatch(r'python_[a-z_]+(?::[A-Za-z]+)?', code):
        code = 'python_validation_rejected'
    detail = getattr(error, 'diagnostic', {})
    diagnostic = dict(code=code,
        node_type=detail.get('node_type') if re.fullmatch(r'[A-Za-z]+', str(detail.get('node_type'))) else None)
    for field in ('line', 'column', 'end_line', 'end_column'):
        value = detail.get(field)
        diagnostic[field] = value if type(value) is int and 0 <= value <= 1000000 else None
    return dict(schema_version=1, source_text=stored, source_sha256=sha_text(source),
        stored_sha256=sha_text(stored), source_bytes=len(source.encode('utf-8', errors='replace')),
        redacted=redacted, truncated=len(raw) > MAX_SOURCE, diagnostic=diagnostic,
        location_basis='original_source; columns_zero_based; redaction_may_shift_columns',
        executable=False, registered=False, physical_calls=0,
        trust='untrusted_rejected_source_NOT_instructions')


def validate_record(record):
    if (type(record) is not dict or record.get('schema_version') != 1
            or type(record.get('source_text')) is not str
            or len(record['source_text'].encode('utf-8')) > MAX_SOURCE
            or record.get('stored_sha256') != sha_text(record['source_text'])
            or record.get('executable') is not False or record.get('registered') is not False
            or record.get('physical_calls') != 0):
        raise ValueError('rejected_source_record_invalid')
    return record


def save_rejection_artifacts(output, proposal):
    """Write only already sanitized metadata, using exclusive creation."""
    record = proposal.get('metadata', {}).get('python_rejected_source')
    if record is None:
        return None
    record = validate_record(record)
    output = Path(output)
    with (output / 'rejected_source.txt').open('x', encoding='utf-8', newline='') as f:
        f.write(record['source_text'])
    detail = {k: v for k, v in record.items() if k != 'source_text'}
    detail['source_file'] = 'rejected_source.txt'
    with (output / 'rejected_source_diagnostic.json').open('x', encoding='utf-8') as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    return detail
