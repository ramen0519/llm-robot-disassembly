"""Immutable bundles of existing Memory/Skill formats, explicit CAS activation.

No model, simulator, database or automatic discovery. All paths use the existing
runs-only guard. A revision commits by writing its manifest last; incomplete
directories remain uncommitted evidence. ACTIVE is opt-in, never a runner default.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from orchestrator.explicit_action_memory import FrozenActionMemory
from orchestrator.knowledge_index import build_index, DERIVATION_VERSION
from orchestrator.skill_registry import retrieve
from worker.task_memory import _safe_path, _lexical_path, canonical, digest

ROOT = Path(__file__).absolute().parents[1]
MAX_BYTES = 32 * 1024 * 1024
REVISION = re.compile(r"rev_[a-zA-Z0-9_-]{1,80}\Z")
SOURCE_FILES = ("orchestrator/skill_program.py", "worker/explicit_action_contract.py",
                "worker/path_probe_round.py", "worker/stability_diagnostics.py",
                "orchestrator/knowledge_index.py", "orchestrator/python_skill_vm.py",
                "orchestrator/python_skill_program.py", "orchestrator/python_skill_child.py")


def _memory_origins(memory, allow_fixture):
    # These are the two actual labels emitted by the existing episode and
    # declared-case exporters. Executor errors remain valid negative evidence.
    if not allow_fixture and any(row["result"]["verifier_origin"] not in (
            "actual_ASAP_RedMax", "ASAP_RedMax", "executor_ERROR") for row in memory["records"]):
        raise ValueError("nonactual_memory_requires_explicit_fixture_mode")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _revision(value: str) -> str:
    if type(value) is not str or not REVISION.fullmatch(value):
        raise ValueError("invalid_knowledge_revision")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def read_json(path, allow_test_root=False):
    path = _safe_path(path, allow_test_root)
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("knowledge_artifact_limit")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    return value, raw


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _new(path, blob: bytes, allow_test_root=False):
    path = _safe_path(path, allow_test_root)
    if len(blob) > MAX_BYTES:
        raise ValueError("knowledge_artifact_limit")
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def _encoded(value: dict) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def source_fingerprints() -> dict:
    return {name: _sha((ROOT / name).read_bytes()) for name in SOURCE_FILES}


def _signed(value: dict, field="content_sha256") -> dict:
    return dict(value, **{field: digest(value)})


def _check_signed(value: dict, kind: str):
    if (type(value) is not dict or value.get("kind") != kind or value.get("schema_version") != 1
            or value.get("content_sha256") != digest({k: v for k, v in value.items() if k != "content_sha256"})):
        raise ValueError("invalid_knowledge_artifact:" + kind)


def create_revision(store, revision: str, memory_path, skills_path, *, reason: str,
                    purpose="development", allow_fixture=False, allow_test_root=False) -> dict:
    # Validate every supplied path lexically before reading any of them.
    for value in (store, memory_path, skills_path):
        _lexical_path(value)
    base = _safe_path(store, allow_test_root)
    name = _revision(revision)
    if purpose not in ("development", "evaluation") or type(reason) is not str or not reason.strip():
        raise ValueError("knowledge_reason_and_purpose_required")
    memory, memory_bytes = read_json(memory_path, allow_test_root)
    skills, skills_bytes = read_json(skills_path, allow_test_root)
    FrozenActionMemory(memory)
    retrieve(skills, allow_fixture=allow_fixture)
    _memory_origins(memory, allow_fixture)
    index = build_index(memory, skills, allow_fixture=allow_fixture)
    index_bytes = _encoded(index)
    fingerprints = source_fingerprints()
    directory = _safe_path(base / "revisions" / name, allow_test_root)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    files = {}
    for filename, data in (("memory.json", memory_bytes), ("skills.json", skills_bytes),
                           ("observations_index.json", index_bytes)):
        _new(directory / filename, data, allow_test_root)
        files[filename] = {"sha256": _sha(data), "bytes": len(data)}
    # Original bytes remain authoritative; do not silently bind a changed source.
    if (read_json(memory_path, allow_test_root)[1] != memory_bytes
            or read_json(skills_path, allow_test_root)[1] != skills_bytes
            or source_fingerprints() != fingerprints):
        raise ValueError("knowledge_source_changed_during_freeze")
    manifest = _signed(dict(schema_version=1, kind="knowledge_bundle_v1", revision=name,
        created_utc=_utc(), reason=reason, purpose=purpose, fixture=bool(allow_fixture or skills["fixture"]),
        files=files, sources={"memory": str(_safe_path(memory_path, allow_test_root)),
                              "skills": str(_safe_path(skills_path, allow_test_root))},
        memory_content_sha256=memory["content_sha256"], skills_content_sha256=skills["content_sha256"],
        memory_records=memory["record_count"], registered_skills=len(skills["entries"]),
        derivation_version=DERIVATION_VERSION, interpreter_and_derivation_sources=fingerprints,
        hard_pruning=False, physics_reuse_allowed=False, active_changed=False,
        split_validity="not_inferred; protocol must declare construction and evaluation tasks"))
    _new(directory / "manifest.json", _encoded(manifest), allow_test_root)
    return manifest


def load_revision(store, revision: str, *, allow_fixture=False, allow_test_root=False) -> dict:
    base = _safe_path(store, allow_test_root)
    directory = _safe_path(base / "revisions" / _revision(revision), allow_test_root)
    manifest, _ = read_json(directory / "manifest.json", allow_test_root)
    _check_signed(manifest, "knowledge_bundle_v1")
    if (manifest.get("revision") != revision or manifest.get("hard_pruning") is not False
            or manifest.get("physics_reuse_allowed") is not False
            or type(manifest.get("fixture")) is not bool
            or manifest.get("purpose") not in ("development", "evaluation")
            or set(manifest.get("files", {})) != {"memory.json", "skills.json", "observations_index.json"}):
        raise ValueError("invalid_bundle_policy")
    if manifest["fixture"] and not allow_fixture:
        raise ValueError("fixture_bundle_not_for_actual_planning")
    values = {}
    for filename, expected in manifest["files"].items():
        value, raw = read_json(directory / filename, allow_test_root)
        if {"sha256": _sha(raw), "bytes": len(raw)} != expected:
            raise ValueError("knowledge_file_hash_mismatch:" + filename)
        values[filename] = value
    memory, skills = values["memory.json"], values["skills.json"]
    FrozenActionMemory(memory)
    _memory_origins(memory, allow_fixture)
    retrieve(skills, allow_fixture=allow_fixture)
    if (memory["content_sha256"] != manifest["memory_content_sha256"]
            or skills["content_sha256"] != manifest["skills_content_sha256"]
            or memory["record_count"] != manifest["memory_records"]
            or len(skills["entries"]) != manifest["registered_skills"]):
        raise ValueError("knowledge_content_identity_mismatch")
    index = values["observations_index.json"]
    if (type(index) is not dict or index.get("kind") != "knowledge_derived_index_v1"
            or index.get("memory_content_sha256") != manifest["memory_content_sha256"]
            or index.get("skills_content_sha256") != manifest["skills_content_sha256"]
            or index.get("hard_pruning") is not False or index.get("physics_reuse_allowed") is not False):
        raise ValueError("knowledge_index_binding_mismatch")
    if manifest["interpreter_and_derivation_sources"] == source_fingerprints():
        if index != build_index(memory, skills, allow_fixture=allow_fixture):
            raise ValueError("knowledge_index_derivation_mismatch")
    return dict(manifest=manifest, directory=str(directory), memory=str(directory / "memory.json"),
                skills=str(directory / "skills.json"), index=values["observations_index.json"])


def active_pointer(store, *, allow_test_root=False):
    path = _safe_path(Path(store) / "ACTIVE.json", allow_test_root)
    if not path.exists():
        return None
    value, _ = read_json(path, allow_test_root)
    _check_signed(value, "knowledge_active_v1")
    _revision(value["revision"])
    return value


@contextmanager
def _lock(base, allow_test_root):
    lock = _safe_path(base / ".promotion.lock", allow_test_root)
    _new(lock, b"single_writer; stale lock requires explicit inspection\n", allow_test_root)
    try:
        yield
    finally:
        lock.unlink()


def promote(store, revision: str, *, expected_active_sha256, reason: str,
            allow_fixture=False, allow_test_root=False) -> dict:
    base = _safe_path(store, allow_test_root)
    if not reason or type(reason) is not str:
        raise ValueError("promotion_reason_required")
    with _lock(base, allow_test_root):
        bundle = load_revision(base, revision, allow_fixture=allow_fixture, allow_test_root=allow_test_root)
        if bundle["manifest"]["purpose"] == "evaluation":
            raise ValueError("evaluation_bundle_cannot_update_active")
        previous = active_pointer(base, allow_test_root=allow_test_root)
        observed = previous["content_sha256"] if previous else None
        if observed != expected_active_sha256:
            raise ValueError("active_compare_and_swap_conflict")
        pointer = _signed(dict(schema_version=1, kind="knowledge_active_v1", revision=revision,
            manifest_sha256=bundle["manifest"]["content_sha256"], previous_active_sha256=observed,
            reason=reason, created_utc=_utc(), promotion_id=uuid.uuid4().hex))
        history = _safe_path(base / "history", allow_test_root)
        history.mkdir(exist_ok=True)
        # Intent is recorded before replace. An interrupted intent is not a committed promotion.
        _new(history / (pointer["promotion_id"] + "_intent.json"), _encoded(pointer), allow_test_root)
        temporary = _safe_path(base / (".active-" + pointer["promotion_id"] + ".json"), allow_test_root)
        _new(temporary, _encoded(pointer), allow_test_root)
        destination = _safe_path(base / "ACTIVE.json", allow_test_root)
        os.replace(temporary, destination)
        _new(history / (pointer["promotion_id"] + "_committed.json"), _encoded(pointer), allow_test_root)
        return pointer


def resolve_bundle(store, revision=None, *, purpose="development", allow_fixture=False,
                   allow_test_root=False) -> dict:
    if purpose not in ("development", "evaluation"):
        raise ValueError("invalid_bundle_purpose")
    if purpose == "evaluation" and revision is None:
        raise ValueError("evaluation_requires_explicit_revision")
    pointer = None
    if revision is None:
        pointer = active_pointer(store, allow_test_root=allow_test_root)
        if pointer is None:
            raise ValueError("no_active_knowledge_revision")
        revision = pointer["revision"]
    bundle = load_revision(store, revision, allow_fixture=allow_fixture, allow_test_root=allow_test_root)
    if pointer and bundle["manifest"]["content_sha256"] != pointer["manifest_sha256"]:
        raise ValueError("active_manifest_mismatch")
    if bundle["manifest"]["interpreter_and_derivation_sources"] != source_fingerprints():
        raise ValueError("knowledge_interpreter_or_derivation_changed")
    return dict(bundle, pin=_signed(dict(schema_version=1, kind="knowledge_run_pin_v1",
        store=str(_safe_path(store, allow_test_root)), revision=revision, purpose=purpose,
        manifest_sha256=bundle["manifest"]["content_sha256"],
        active_sha256_at_resolution=pointer["content_sha256"] if pointer else None,
        memory=bundle["memory"], skills=bundle["skills"],
        files=bundle["manifest"]["files"], active_updates_allowed=False)))


def verify_pin(pin: dict, *, allow_fixture=False, allow_test_root=False) -> dict:
    """Revalidate a run without reading ACTIVE; safe against later promotion."""
    _check_signed(pin, "knowledge_run_pin_v1")
    if pin.get("active_updates_allowed") is not False:
        raise ValueError("invalid_pin_policy")
    bundle = resolve_bundle(pin["store"], pin["revision"], purpose=pin["purpose"],
        allow_fixture=allow_fixture, allow_test_root=allow_test_root)
    if (bundle["manifest"]["content_sha256"] != pin["manifest_sha256"]
            or any(bundle[key] != pin[key] for key in ("memory", "skills"))
            or bundle["manifest"]["files"] != pin["files"]):
        raise ValueError("knowledge_pin_mismatch")
    return bundle
