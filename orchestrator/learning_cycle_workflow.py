"""Thin LangGraph lifecycle with immutable, fail-closed stage artifacts.

This graph owns no DFS/physics state. COMPLETE means the stage callbacks ended,
not that a plan or generated skill succeeded. A completed stage may be reused;
an interrupted callback is never automatically relaunched. Artifacts declared
by callbacks must be immutable files (do not declare a shared mutable ledger).
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Callable, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from worker.task_memory import PROJECT, _lexical_path, _safe_path, canonical, digest


class _CycleState(TypedDict):
    stage_results: dict
    completed_artifact_reuse: list
    executed_stages: list


def _json_copy(value):
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(k) is str for k in item):
            for child in item.values():
                check(child)
            return
        raise ValueError('finite_json_required')
    check(value)
    return deepcopy(value)


def _error_type(error):
    name = type(error).__name__
    return name if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,127}', name) else 'Exception'


class LearningCycleWorkflow:
    """Run ordered callbacks, reusing only verified complete stage artifacts.

    Each callback receives a deep copy of prior stage results and returns a JSON
    object. Its optional ``artifacts`` maps labels to immutable file paths. A
    callback can report NO_ELIGIBLE/NOT_REGISTERED normally; only exceptions
    stop the graph. Run identity and stage order must match on later invocations.
    """

    def __init__(self, run_dir, identity: dict,
                 stages: dict[str, Callable[[dict], dict]], allow_test_root=False):
        if type(identity) is not dict or not identity:
            raise ValueError('nonempty_cycle_identity_required')
        if (type(stages) is not dict or not 1 <= len(stages) <= 32
                or any(type(name) is not str
                       or re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name) is None
                       or name in (START, END) or not callable(fn)
                       for name, fn in stages.items())):
            raise ValueError('bounded_named_stage_callbacks_required')
        self.allow_test_root = allow_test_root
        self.run_dir = _safe_path(run_dir, allow_test_root)
        self.journal = _safe_path(self.run_dir/'learning_cycle_journal', allow_test_root)
        self.identity = _json_copy(identity)
        self.stages = dict(stages)
        self.order = list(stages)
        self.binding = {'schema_version': 1, 'identity': self.identity,
                        'stage_order': self.order}
        self.identity_sha256 = digest(self.binding)
        self.binding['identity_sha256'] = self.identity_sha256
        builder = StateGraph(_CycleState)
        for name in self.order:
            builder.add_node(name, self._node(name))
        builder.add_edge(START, self.order[0])
        for before, after in zip(self.order, self.order[1:]):
            builder.add_edge(before, after)
        builder.add_edge(self.order[-1], END)
        self.graph = builder.compile()

    def _path(self, value):
        return _safe_path(value, self.allow_test_root)

    def _write_new(self, path, value):
        path = self._path(path)
        payload = canonical(_json_copy(value)).encode('utf-8') + b'\n'
        with path.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def _read(self, path):
        path = self._path(path)
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('duplicate_journal_key')
                result[key] = value
            return result
        try:
            with path.open('rb') as stream:
                raw = stream.read(32 * 1024 * 1024 + 1)
            if len(raw) > 32 * 1024 * 1024:
                raise ValueError('journal_byte_limit')
            result = _json_copy(json.loads(raw.decode('utf-8'), object_pairs_hook=pairs))
            if type(result) is not dict:
                raise ValueError('journal_object_required')
            return result
        except (ValueError, TypeError, UnicodeError, OSError):
            raise RuntimeError('invalid_or_partial_stage_journal') from None

    def _event(self, kind, stage=None, **fields):
        event = dict(event_id=str(uuid4()), utc=datetime.now(timezone.utc).isoformat(),
                     kind=kind, stage=stage, identity_sha256=self.identity_sha256,
                     **fields)
        with self._path(self.journal/'events.jsonl').open('ab') as stream:
            stream.write(canonical(event).encode('utf-8') + b'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _file_hash(self, path):
        # Journal writes stay in runs. Read-only provenance may also bind the
        # predeclared config and archived/current code inside this project.
        path = Path(_lexical_path(path)).absolute()
        if path.is_relative_to(PROJECT):
            for ancestor in reversed((path,) + tuple(path.parents)):
                try:
                    info = ancestor.lstat()
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('linked_artifact_paths_excluded')
        else:
            path = self._path(path)
        if not path.is_file():
            raise RuntimeError('completed_artifact_missing')
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(chunk)
        return str(path), value.hexdigest()

    def _artifact_hashes(self, result):
        artifacts = result.get('artifacts', {})
        if (type(artifacts) is not dict or any(type(k) is not str or not k
                or type(v) is not str or not v for k, v in artifacts.items())):
            raise ValueError('artifact_path_mapping_required')
        output = {}
        for label, path in artifacts.items():
            safe, hashed = self._file_hash(path)
            output[label] = {'path': safe, 'sha256': hashed}
        return output

    def _base_record(self, stage, prior):
        return {'schema_version': 1, 'stage': stage,
                'identity_sha256': self.identity_sha256,
                'prior_results_sha256': digest(prior)}

    def _complete(self, stage, prior):
        directory = self._path(self.journal/stage)
        started, completed = directory/'stage_started.json', directory/'stage_completed.json'
        if not directory.exists():
            return None
        if not directory.is_dir():
            raise RuntimeError('unknown_partial_stage_state')
        names = {self._path(p).name for p in directory.iterdir()}
        if names - {'stage_started.json', 'stage_completed.json', 'stage_failed.json'}:
            raise RuntimeError('unknown_partial_stage_state')
        if not started.exists():
            # Even an empty stage directory may be left by an interrupted owner.
            raise RuntimeError('unsupported_partial_resume')
        expected = self._base_record(stage, prior)
        if self._read(started) != expected:
            raise RuntimeError('stage_identity_or_history_mismatch')
        if not completed.exists():
            raise RuntimeError('unsupported_partial_resume')
        if 'stage_failed.json' in names:
            raise RuntimeError('conflicting_completed_stage')
        record = self._read(completed)
        if (set(record) != set(expected) | {'result', 'artifact_hashes', 'record_sha256'}
                or any(record.get(k) != v for k, v in expected.items())
                or record['record_sha256'] != digest({k: v for k, v in record.items()
                                                      if k != 'record_sha256'})
                or type(record['result']) is not dict):
            raise RuntimeError('completed_stage_record_mismatch')
        if self._artifact_hashes(record['result']) != record['artifact_hashes']:
            raise RuntimeError('completed_artifact_hash_mismatch')
        return _json_copy(record['result'])

    def _preflight(self):
        known = {'identity.json', 'events.jsonl', 'session.lock', *self.order}
        if {self._path(p).name for p in self.journal.iterdir()} - known:
            raise RuntimeError('unknown_partial_cycle_state')
        events_path = self._path(self.journal/'events.jsonl')
        if events_path.exists():
            try:
                with events_path.open('rb') as stream:
                    raw = stream.read(32 * 1024 * 1024 + 1)
                if (len(raw) > 32 * 1024 * 1024
                        or raw and not raw.endswith(b'\n')):
                    raise ValueError('partial_event_line')
                seen = set()
                kinds = {'cycle_started', 'cycle_completed', 'stage_started',
                         'stage_completed', 'stage_failed', 'completed_artifact_reuse'}
                for line in raw.splitlines():
                    event = _json_copy(json.loads(line.decode('utf-8')))
                    if (type(event) is not dict
                            or event.get('identity_sha256') != self.identity_sha256
                            or event.get('kind') not in kinds
                            or event.get('stage') not in [None, *self.order]
                            or type(event.get('event_id')) is not str
                            or not event['event_id'] or event['event_id'] in seen):
                        raise ValueError('invalid_event_identity')
                    seen.add(event['event_id'])
            except (ValueError, TypeError, UnicodeError, OSError):
                raise RuntimeError('invalid_or_partial_event_journal') from None
        prior, gap = {}, False
        for stage in self.order:
            exists = self._path(self.journal/stage).exists()
            if gap and exists:
                raise RuntimeError('noncontiguous_stage_journal')
            result = self._complete(stage, prior)
            if result is None:
                gap = True
            else:
                prior[stage] = result

    def _verify_prior(self, prior):
        verified = {}
        for stage, expected in prior.items():
            if self._complete(stage, verified) != expected:
                raise RuntimeError('prior_stage_changed')
            verified[stage] = expected

    def _node(self, stage):
        def node(state):
            prior = _json_copy(state['stage_results'])
            self._verify_prior(prior)
            existing = self._complete(stage, prior)
            if existing is not None:
                self._event('completed_artifact_reuse', stage)
                return {'stage_results': {**prior, stage: existing},
                        'completed_artifact_reuse': state['completed_artifact_reuse'] + [stage]}
            directory = self._path(self.journal/stage)
            directory.mkdir(exist_ok=False)
            record = self._base_record(stage, prior)
            self._write_new(directory/'stage_started.json', record)
            self._event('stage_started', stage)
            try:
                result = self.stages[stage](_json_copy(prior))
                if type(result) is not dict:
                    raise ValueError('stage_result_object_required')
                result = _json_copy(result)
                completed = dict(record, result=result,
                                 artifact_hashes=self._artifact_hashes(result))
                completed['record_sha256'] = digest(completed)
                self._verify_prior(prior)
                self._write_new(directory/'stage_completed.json', completed)
            except BaseException as error:
                failed = dict(record, error_type=_error_type(error))
                self._write_new(directory/'stage_failed.json', failed)
                self._event('stage_failed', stage, error_type=failed['error_type'])
                raise RuntimeError('learning_cycle_stage_failed') from None
            self._event('stage_completed', stage)
            return {'stage_results': {**prior, stage: result},
                    'executed_stages': state['executed_stages'] + [stage]}
        return node

    def run(self) -> dict:
        self._path(self.run_dir).mkdir(parents=True, exist_ok=True)
        self._path(self.journal).mkdir(exist_ok=True)
        lock = self._path(self.journal/'session.lock')
        token = {'session_id': str(uuid4()), 'identity_sha256': self.identity_sha256}
        try:
            self._write_new(lock, token)
        except FileExistsError:
            raise RuntimeError('cycle_active_or_stale_session_lock') from None
        try:
            identity_path = self._path(self.journal/'identity.json')
            if identity_path.exists():
                if self._read(identity_path) != self.binding:
                    raise ValueError('cycle_identity_mismatch')
            else:
                if any(p.name != 'session.lock' for p in self.journal.iterdir()):
                    raise RuntimeError('unknown_partial_cycle_state')
                self._write_new(identity_path, self.binding)
            self._preflight()
            self._event('cycle_started')
            with tracing_context(enabled=False):
                state = self.graph.invoke({'stage_results': {},
                    'completed_artifact_reuse': [], 'executed_stages': []},
                    {'recursion_limit': len(self.order) + 5, 'callbacks': []})
            self._verify_prior(state['stage_results'])
            self._event('cycle_completed',
                        completed_artifact_reuse=state['completed_artifact_reuse'])
            return dict(status='COMPLETE', identity_sha256=self.identity_sha256,
                        **_json_copy(state))
        finally:
            # Never remove a different session's or externally changed lock.
            if self._read(lock) != token:
                raise RuntimeError('cycle_session_lock_changed')
            self._path(lock).unlink()
