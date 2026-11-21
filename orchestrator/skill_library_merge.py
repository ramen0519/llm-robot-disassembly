"""Append-only union of frozen, admitted generated-skill snapshots.

This is storage orchestration, not another admission gate or a physics cache.
Whole entries are preserved. Identical program hashes with different cards,
generation provenance, admission decisions or validation records fail closed;
combining those histories requires a separately specified evidence merge.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from typing import Any, Iterable

from orchestrator.skill_registry import retrieve
from orchestrator.skill_validation import MAX_ARTIFACT_BYTES, read_artifact
from worker.task_memory import _lexical_path, _safe_path, canonical, digest


def _check_snapshot(snapshot: dict, allow_fixture: bool) -> list:
    """Keep the established frozen-snapshot contract and its origin labels."""
    if (type(snapshot.get("schema_version")) is not int
            or snapshot["schema_version"] != 1
            or type(snapshot.get("entries")) is not list):
        raise ValueError("unsupported_frozen_skill_snapshot")
    rows = retrieve(snapshot, allow_fixture=allow_fixture)
    for entry in snapshot["entries"]:
        candidate, admission = entry["candidate"], entry["admission"]
        if (candidate.get("fixture_registry") is not snapshot["fixture"]
                or admission.get("fixture") is not snapshot["fixture"]):
            raise ValueError("inconsistent_skill_fixture_provenance")
    return rows


def freeze_skill_union(snapshot_paths: Iterable[Any], output_path: Any, *,
                       allow_fixture: bool = False,
                       allow_test_root: bool = False) -> dict:
    """Freeze complete entries from admitted libraries, without overwriting.

    Only project ``runs`` paths are allowed (explicit temporary roots in unit
    tests). Non-C drives, traversal and links are rejected before opening files.
    An empty input list produces a valid, empty actual-planning library, leaving
    the caller free to use its ordinary actor. Fixture and actual snapshots may
    never be mixed, including snapshots with no entries. Fixture-only output
    stays fixture-only even when there are zero admitted entries.

    Returning a skill does not authorize a new action: consumers still select
    a current-state action and require fresh physics. No source file is changed.
    """
    if type(allow_fixture) is not bool or type(allow_test_root) is not bool:
        raise ValueError("skill_union_flags_must_be_bool")
    if isinstance(snapshot_paths, (str, bytes)):
        raise ValueError("skill_union_requires_snapshot_path_sequence")
    paths = list(snapshot_paths)
    # Validate every textual boundary before even inspecting a safe C: path.
    for value in [output_path] + paths:
        _lexical_path(value)
    destination = _safe_path(output_path, allow_test_root)
    sources = [_safe_path(path, allow_test_root) for path in paths]
    if destination in sources:
        raise ValueError("skill_union_output_must_not_replace_input")
    if destination.exists():
        raise FileExistsError(str(destination))

    entries, provenance, originals = {}, [], []
    modes = set()
    source_count = duplicate_count = 0
    for path in sources:
        artifact = read_artifact(path, allow_test_root)
        snapshot = artifact["value"]
        _check_snapshot(snapshot, allow_fixture)
        modes.add(snapshot["fixture"])
        if len(modes) > 1:
            raise ValueError("fixture_and_actual_skill_snapshots_cannot_mix")
        for entry in snapshot["entries"]:
            identity = entry["candidate"]["program_hash"]
            source_count += 1
            if identity in entries:
                if canonical(entries[identity]) != canonical(entry):
                    raise ValueError("conflicting_frozen_skill_program:" + identity)
                duplicate_count += 1
            else:
                entries[identity] = deepcopy(entry)
        provenance.append({"path": str(path), "file_sha256": artifact["sha256"],
                           "content_sha256": snapshot["content_sha256"],
                           "entry_count": len(snapshot["entries"]),
                           "fixture": snapshot["fixture"]})
        originals.append((path, artifact["sha256"]))

    fixture = modes == {True}
    snapshot = {"schema_version": 1, "kind": "frozen_generated_skills_v1",
                "entries": [entries[key] for key in sorted(entries)],
                "fixture": fixture, "usable_for_actual_planning": not fixture,
                "physics_reuse_allowed": False, "hard_pruning": False,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "performance_effect": "NOT_RUN",
                "merge_provenance": {
                    "schema_version": 1,
                    "method": "whole_entry_union_by_program_hash_v1",
                    "conflict_policy": "reject_nonidentical_entries",
                    "sources": provenance, "source_entry_count": source_count,
                    "duplicate_entry_count": duplicate_count,
                    "unique_entry_count": len(entries),
                    "new_api_calls": 0, "new_physics_calls": 0}}
    snapshot["content_sha256"] = digest(snapshot)
    _check_snapshot(snapshot, allow_fixture)
    encoded = (canonical(snapshot) + "\n").encode("utf-8")
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("skill_union_artifact_byte_limit")

    # Do not freeze a provenance statement about inputs changed during merging.
    # This reads bytes only: it never follows raw worker-artifact pointers.
    for path, expected_sha256 in originals:
        checked = _safe_path(path, allow_test_root)
        with checked.open("rb") as stream:
            raw = stream.read(MAX_ARTIFACT_BYTES + 1)
        if (len(raw) > MAX_ARTIFACT_BYTES
                or hashlib.sha256(raw).hexdigest() != expected_sha256):
            raise ValueError("skill_union_source_changed_during_merge")
    destination = _safe_path(destination, allow_test_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return snapshot
