"""Small append-only batch records; no jobs, retries, APIs, or secret access.

The caller owns the phase lock. Metadata is validated and the whole JSON value
is serialized before any file is opened. An I/O interruption can leave a
partial new file; it is preserved and must never be interpreted as completion.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import time
from typing import Any, Callable, Mapping


class EventWriteFailure(OSError):
    """The important record was saved but its trailing event was not."""
    def __init__(self, record_path: Path, cause: Exception):
        super().__init__('record_saved_event_append_failed')
        self.record_path = str(record_path)
        self.record_saved = True
        self.event_error_type = type(cause).__name__


def checked_path(value: Any) -> Path:
    """Exclude the prohibited drive before resolving, probing or opening it."""
    text = str(value)
    normalized = text.replace('\\', '/').lower()
    if (PureWindowsPath(text).drive.lower().endswith('e:') or
            re.match(r'^(?:[\\/]{2}[?.][\\/])?e:', text, re.I) or
            normalized == '/mnt/e' or normalized.startswith('/mnt/e/')):
        raise ValueError('forbidden_drive')
    return Path(value)


def _serialized(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      indent=2 if pretty else None) + '\n'


def make_event(event_kind: str, fields: Mapping[str, Any] | None = None, *,
               protocol_version: str, unix_time: float | None = None,
               clock: Callable[[], float] = time.time) -> dict:
    """Accept equal reserved fields, reject conflicting ones before I/O.

    `fields` is a mapping, not **kwargs, so a duplicate `kind` is inspectable.
    Use an explicit unix_time when replaying already timestamped metadata.
    """
    if type(event_kind) is not str or not event_kind:
        raise ValueError('nonempty_event_kind_required')
    if type(protocol_version) is not str or not protocol_version:
        raise ValueError('nonempty_protocol_version_required')
    timestamp = clock() if unix_time is None else unix_time
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
        raise ValueError('finite_event_time_required')
    payload = dict(fields or {})
    if any(type(key) is not str for key in payload):
        raise ValueError('string_event_fields_required')
    metadata = {'kind': event_kind, 'unix_time': timestamp,
                'protocol_version': protocol_version}
    for key, expected in metadata.items():
        if key not in payload:
            continue
        actual = payload.pop(key)
        if key == 'unix_time':
            same = type(actual) in (int, float) and math.isfinite(actual) and actual == expected
        else:
            same = type(actual) is type(expected) and actual == expected
        if not same:
            raise ValueError('event_metadata_conflict_' + key)
    result = dict(metadata)
    result.update(payload)
    _serialized(result)
    return result


def write_new_json(path: Any, value: Any) -> Path:
    """Serialize before exclusively creating one immutable result file."""
    target = checked_path(path)
    payload = _serialized(value, pretty=True)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def append_event(path: Any, event_kind: str, fields: Mapping[str, Any] | None = None,
                 *, protocol_version: str, unix_time: float | None = None,
                 clock: Callable[[], float] = time.time) -> dict:
    """Append one validated event; caller must hold its single-writer lock."""
    target = checked_path(path)
    event = make_event(event_kind, fields, protocol_version=protocol_version,
                       unix_time=unix_time, clock=clock)
    payload = _serialized(event)
    with target.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return event


def write_record_and_event(record_path: Any, event_path: Any, event_kind: str,
                           value: Mapping[str, Any], *, protocol_version: str,
                           clock: Callable[[], float] = time.time) -> dict:
    """Save the gate/result/error before its event, never rerun its producer.

    EventWriteFailure distinguishes a saved outcome from an append failure.
    No retry is performed, including on an interrupted write.
    """
    target = checked_path(record_path)
    event_target = checked_path(event_path)
    event = make_event(event_kind, value, protocol_version=protocol_version, clock=clock)
    write_new_json(target, value)
    try:
        return append_event(event_target, event_kind, event,
                            protocol_version=protocol_version, unix_time=event['unix_time'])
    except Exception as error:
        raise EventWriteFailure(target, error) from error
