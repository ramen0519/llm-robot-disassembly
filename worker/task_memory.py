"""Frozen JSON observations for generic tasks; never constraints or a physics cache.

This module deliberately does not open historical SQLite databases. A caller
collects a *new* episode separately and explicitly freezes records before use.
All observations, including PASS, remain conditional on their source request.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import stat
import tempfile
from typing import Any, Iterable, Tuple


STATUSES = ("PASS", "UNKNOWN", "ERROR")
POLICY = "scoped_observations_no_constraints"
PROJECT = Path(__file__).absolute().parents[1]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(canonical(value))


def _ids(value: Any) -> bool:
    return (type(value) is list and all(type(x) is str and x for x in value)
            and len(set(value)) == len(value))


def _text(value: Any) -> bool:
    return type(value) is str and bool(value) and "\x00" not in value


def _sha(value: Any) -> bool:
    return type(value) is str and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _input_hash(value: Any) -> bool:
    return (_sha(value) or (type(value) is dict and bool(value)
                           and all(_text(k) and _sha(v) for k, v in value.items())))


def _lexical_path(value: Any) -> str:
    """Reject forbidden locations before resolve/stat/open or directory scans."""
    text = str(value)
    normalized = text.replace("\\", "/").lower()
    win = PureWindowsPath(text)
    if (not text or "\x00" in text or normalized.startswith("//")
            or (win.drive and win.drive.lower() != "c:")
            or normalized == "/mnt/e" or normalized.startswith("/mnt/e/")
            or ".." in text.replace("\\", "/").split("/")):
        raise ValueError("memory paths must exclude non-C drives and traversal")
    return text


def _source_ref(value: Any) -> None:
    if not _text(value) or len(value) > 4096 or "\n" in value or "\r" in value:
        raise ValueError("invalid source log pointer")
    # A pointer is descriptive only: this function never opens it.
    _lexical_path(value)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_path(value: Any, allow_test_root: bool = False) -> Path:
    path = Path(_lexical_path(value)).absolute()
    allowed = _within(path, PROJECT / "runs")
    if allow_test_root and not allowed:
        test_root = Path(_lexical_path(tempfile.gettempdir())).absolute()
        allowed = _within(path, test_root)
    if not allowed:
        raise ValueError("memory files must stay within project runs")
    for ancestor in reversed((path,) + tuple(path.parents)):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("linked memory paths are excluded")
    return path


def task_context(features: dict) -> dict:
    """Snapshot schema-2 geometry/task metadata before filtering candidates.

    Solver settings must include the actual seed/budget/physics configuration
    supplied by the producer. This module does not infer missing settings.
    """
    if (not isinstance(features, dict) or features.get("schema_version") != 2
            or type(features.get("schema_version")) is not int
            or not isinstance(features.get("task"), dict)
            or not isinstance(features.get("parts"), dict) or not features["parts"]):
        raise ValueError("schema-2 task and complete initial parts are required")
    task = features["task"]
    required = {"assembly_id", "input_sha256", "source_commit", "base_part",
                "solver_settings", "goal"}
    if not required.issubset(task):
        raise ValueError("incomplete task identity")
    context = {"schema_version": 2, "usage": "soft_ranking_only",
               "task": _copy(task), "initial_parts": sorted(features["parts"]),
               "geometry_features": _copy(features["parts"])}
    _validate_context(context)
    return context


def _validate_context(context: Any) -> None:
    if (not isinstance(context, dict) or type(context.get("schema_version")) is not int
            or context["schema_version"] != 2 or context.get("usage") != "soft_ranking_only"
            or not _ids(context.get("initial_parts")) or not context["initial_parts"]
            or context["initial_parts"] != sorted(context["initial_parts"])
            or not isinstance(context.get("geometry_features"), dict)
            or sorted(context["geometry_features"]) != context["initial_parts"]):
        raise ValueError("invalid generic task context")
    task = context.get("task")
    if (not isinstance(task, dict)
            or not all(_text(task.get(k)) for k in ("assembly_id", "source_commit"))
            or not _input_hash(task.get("input_sha256"))
            or "base_part" not in task or "goal" not in task
            or task["goal"] in (None, "", {})
            or not isinstance(task.get("solver_settings"), dict)
            or not task["solver_settings"]
            or (task["base_part"] is not None and task["base_part"] not in context["initial_parts"])):
        raise ValueError("invalid task identity or solver settings")
    canonical(context)


def _validate_record(record: Any) -> None:
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != 2 or record.get("status") not in STATUSES
            or record.get("usage") != "soft_ranking_only"
            or record.get("allow_hard_pruning") is not False
            or record.get("constraints") != [] or record.get("blocker") is not None
            or record.get("path") is not None):
        raise ValueError("invalid observation schema or non-pruning policy")
    source = record.get("source")
    if (not isinstance(source, dict)
            or not all(_text(source.get(k)) for k in
                       ("run_id", "call_id", "start_event_id", "result_event_id", "ref"))
            or source["start_event_id"] == source["result_event_id"]
            or record.get("record_id") != source["result_event_id"]
            or not _sha(source.get("sha256"))):
        raise ValueError("invalid observation provenance")
    _source_ref(source["ref"])
    if record.get("imported_legacy") is True:
        if record.get("reference_only") is not True:
            raise ValueError("legacy observations are reference-only")
        original = record.get("legacy_record")
        _validate_legacy(original)
        if (record.get("context") != original["context"]
                or source != original["source"] or record["status"] != original["status"]
                or record.get("request") != original["request"]
                or record.get("raw_result") != original["raw_result"]):
            raise ValueError("legacy provenance or raw observation changed")
        canonical(record)
        return
    _validate_context(record.get("context"))
    request = record.get("request")
    required = {"part_move", "parts_rest", "parts_removed", "pose", "max_grippers",
                "timeout", "optimizer", "debug", "render"}
    if (not isinstance(request, dict) or not required.issubset(request)
            or not _text(request["part_move"]) or not _ids(request["parts_rest"])
            or not _ids(request["parts_removed"]) or request["part_move"] in request["parts_rest"]
            or type(request["max_grippers"]) is not int or request["max_grippers"] < 1
            or not _text(request["optimizer"])):
        raise ValueError("invalid bounded verifier request")
    initial = set(record["context"]["initial_parts"])
    assembled = set(request["parts_rest"]) | {request["part_move"]}
    base = record["context"]["task"]["base_part"]
    if (not assembled.issubset(initial) or set(request["parts_removed"]) != initial - assembled
            or (base is not None and (base not in request["parts_rest"] or base == request["part_move"]))):
        raise ValueError("inconsistent assembled/removed parts or base")
    raw = record.get("raw_result")
    if record["status"] == "ERROR":
        if raw is not None or not _text(record.get("error_type")):
            raise ValueError("ERROR requires null raw result and error type")
    elif (not isinstance(raw, dict) or type(raw.get("feasible")) is not bool
          or raw["feasible"] != (record["status"] == "PASS")
          or raw.get("part_move") != request["part_move"] or raw.get("base_part") != base
          or "pose" not in raw or canonical(raw["pose"]) != canonical(request["pose"])):
        raise ValueError("raw result contradicts status or request")
    if record["status"] == "PASS":
        action, support = raw.get("action"), raw.get("parts_fix")
        if (type(action) is not list or len(action) != 3
                or any(type(v) not in (int, float) for v in action) or not any(action)
                or not _ids(support) or not set(support).issubset(request["parts_rest"])
                or len(support) + 1 > request["max_grippers"]):
            raise ValueError("PASS requires observed action and support")
    if (record.get("action") != (raw.get("action") if isinstance(raw, dict) else None)
            or record.get("support") != (raw.get("parts_fix") if isinstance(raw, dict) else None)
            or record.get("pose") != request["pose"]):
        raise ValueError("projected action, support or pose disagrees with raw evidence")
    canonical(record)


def records_from_events(context: dict, events: Iterable[dict], source_ref: str,
                        source_sha256: str) -> list:
    """Normalize complete verifier pairs; reject conflicts and incomplete calls.

    A repeated identical event is idempotent. No failure cause is synthesized
    from feasible=False; optional stage_feedback is preserved as supplied.
    """
    _validate_context(context)
    _source_ref(source_ref)
    if not _sha(source_sha256):
        raise ValueError("source log requires a SHA-256")
    seen, pending, completed, records = {}, {}, set(), []
    for original in events:
        event = _copy(original)
        if not isinstance(event, dict):
            raise ValueError("events must be dictionaries")
        if event.get("kind") not in ("verifier_start", "verifier_result"):
            continue
        if not all(_text(event.get(k)) for k in ("event_id", "run_id", "call_id")):
            raise ValueError("verifier event lacks identity")
        event_id, payload = event["event_id"], canonical(event)
        if event_id in seen:
            if seen[event_id] != payload:
                raise ValueError("conflicting event ID")
            continue
        seen[event_id] = payload
        key = (event["run_id"], event["call_id"])
        if event["kind"] == "verifier_start":
            if key in pending or key in completed:
                raise ValueError("duplicate verifier call")
            pending[key] = event
            continue
        if key not in pending or key in completed:
            raise ValueError("result has no unique prior start")
        start = pending.pop(key)
        if event.get("blocker") is not None or event.get("path") is not None:
            raise ValueError("unsupported blocker/path evidence must not become a fact")
        raw = event.get("raw")
        record = {"schema_version": 2, "record_id": event_id,
                  "context": _copy(context), "request": start.get("request"),
                  "source": {"run_id": key[0], "call_id": key[1],
                             "start_event_id": start["event_id"], "result_event_id": event_id,
                             "ref": source_ref, "sha256": source_sha256},
                  "status": event.get("status"), "raw_result": raw,
                  "error_type": event.get("error_type"),
                  "stage_feedback": _copy(event.get("stage_feedback")),
                  "action": _copy(raw.get("action")) if isinstance(raw, dict) else None,
                  "pose": _copy(start.get("request", {}).get("pose")),
                  "support": _copy(raw.get("parts_fix")) if isinstance(raw, dict) else None,
                  "scope": "bounded_verifier_attempt", "usage": "soft_ranking_only",
                  "blocker": None, "path": None, "constraints": [],
                  "allow_hard_pruning": False, "reference_only": False}
        _validate_record(record)
        records.append(record)
        completed.add(key)
    if pending:
        raise ValueError("incomplete verifier calls are not completed observations")
    return _normal_records(records)


def _validate_legacy(record: dict) -> None:
    # Existing validator is pure JSON checking; constructing EvidenceStore is
    # deliberately forbidden here. The import does not connect to SQLite.
    try:
        from .evidence_memory import _validate_record as validate
    except ImportError:
        from evidence_memory import _validate_record as validate
    validate(record)


def from_legacy_records(records: Iterable[dict]) -> list:
    """Explicitly import supplied schema-1 JSON records, never legacy databases.

    Legacy scope is kept intact. Missing generic identity makes every imported
    record reference-only, including a numerically identical target part ID.
    """
    imported = []
    for original in records:
        record = _copy(original)
        _validate_legacy(record)
        result = {"schema_version": 2, "record_id": record["record_id"],
                  "context": record["context"], "request": record["request"],
                  "source": record["source"], "status": record["status"],
                  "raw_result": record["raw_result"], "error_type": record.get("error_type"),
                  "stage_feedback": None, "scope": record["scope"],
                  "usage": "soft_ranking_only", "constraints": [],
                  "blocker": None, "path": None, "allow_hard_pruning": False,
                  "reference_only": True, "imported_legacy": True,
                  "legacy_record": record}
        _validate_record(result)
        imported.append(result)
    return _normal_records(imported)


def _normal_records(records: Iterable[dict]) -> list:
    unique = {}
    for value in records:
        record = _copy(value)
        _validate_record(record)
        key = record["record_id"]
        if key in unique and canonical(unique[key]) != canonical(record):
            raise ValueError("conflicting observation ID")
        unique[key] = record
    return [unique[key] for key in sorted(unique)]


def save_snapshot(records: Iterable[dict], path: Path, allow_test_root: bool = False) -> dict:
    path = _safe_path(path, allow_test_root)
    normalized = _normal_records(records)
    snapshot = {"schema_version": 2, "policy": POLICY, "allow_hard_pruning": False,
                "records": normalized, "record_count": len(normalized),
                "source_record_ids": [r["record_id"] for r in normalized],
                "content_sha256": digest(normalized)}
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(snapshot) + "\n")
    return snapshot


def load_snapshot(path: Path, allow_test_root: bool = False) -> dict:
    path = _safe_path(path, allow_test_root)
    with path.open(encoding="utf-8") as stream:
        snapshot = json.load(stream)
    if (not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int
            or snapshot["schema_version"] != 2 or snapshot.get("policy") != POLICY
            or snapshot.get("allow_hard_pruning") is not False
            or type(snapshot.get("records")) is not list):
        raise ValueError("invalid frozen task memory")
    records = snapshot["records"]
    if (records != _normal_records(records)
            or snapshot.get("source_record_ids") != [r["record_id"] for r in records]
            or type(snapshot.get("record_count")) is not int or snapshot["record_count"] != len(records)
            or snapshot.get("content_sha256") != digest(records)):
        raise ValueError("frozen memory count, IDs, order or hash changed")
    return snapshot


@dataclass(frozen=True, init=False)
class FrozenTaskMemory:
    _records: Tuple[str, ...]
    content_sha256: str

    def __init__(self, path: Path, allow_test_root: bool = False):
        snapshot = load_snapshot(path, allow_test_root)
        object.__setattr__(self, "_records", tuple(canonical(r) for r in snapshot["records"]))
        object.__setattr__(self, "content_sha256", snapshot["content_sha256"])

    def retrieve(self, context: dict, assembled_parts: Iterable[str],
                 candidate_ids: Iterable[str], max_records: int = 12) -> dict:
        _validate_context(context)
        assembled, candidates = list(assembled_parts), list(candidate_ids)
        if (not _ids(assembled) or not _ids(candidates)
                or not set(assembled).issubset(context["initial_parts"])
                or not set(candidates).issubset(assembled)
                or context["task"]["base_part"] in candidates
                or (context["task"]["base_part"] is not None
                    and context["task"]["base_part"] not in assembled)
                or type(max_records) is not int or not 0 <= max_records <= 100):
            raise ValueError("invalid retrieval state, candidates or limit")
        exact, references = [], []
        context_json = canonical(context)
        for payload in self._records:
            record = json.loads(payload)
            source_assembled = set(record["request"]["parts_rest"]) | {record["request"]["part_move"]}
            matches = (record.get("reference_only") is False
                       and canonical(record["context"]) == context_json
                       and source_assembled == set(assembled)
                       and record["request"]["part_move"] in candidates)
            view = {"record_id": record["record_id"], "status": record["status"],
                    "source": record["source"], "source_context": record["context"],
                    "source_request": record["request"], "raw_result": record["raw_result"],
                    "stage_feedback": record.get("stage_feedback"),
                    "error_type": record.get("error_type"),
                    "reference_only": not matches, "constraints": [],
                    "allow_hard_pruning": False, "current_attempt_verified": False,
                    "current_part_id_mapping": None,
                    "applicability": "same_task_state_observation" if matches else "cross_task_reference",
                    "use": "conditional_source_observation_only" if matches else "LLM_hint_only_no_ID_transfer"}
            (exact if matches else references).append(view)
        exact = exact[:max_records]
        references = references[:max_records - len(exact)]
        selected = exact + references
        return {"exact_observations": exact, "cross_task_references": references,
                "evidence_ids": [r["record_id"] for r in selected],
                "status_counts": {s: sum(r["status"] == s for r in selected) for s in STATUSES},
                "constraints": [], "allow_hard_pruning": False,
                "content_sha256": self.content_sha256,
                "current_candidate_ids": candidates,
                "truncated": len(self._records) > len(selected),
                "physics_reuse_allowed": False}


def project_for_model(retrieved: dict) -> dict:
    """Project facts only; no stored ordering, recommendations or raw free text.

    Raw snapshots and full geometry are retained for provenance outside the
    model prompt. The caller separately supplies current filtered CAD features.
    Reference source part IDs must never be interpreted as current candidate IDs.
    """
    if (not isinstance(retrieved, dict) or retrieved.get("allow_hard_pruning") is not False
            or retrieved.get("physics_reuse_allowed") is not False
            or retrieved.get("constraints") != []):
        raise ValueError("unsafe memory projection policy")
    projected = {"exact_observations": [], "cross_task_references": [],
                 "allow_hard_pruning": False, "physics_reuse_allowed": False,
                 "constraints": [], "content_sha256": retrieved["content_sha256"]}
    for category in ("exact_observations", "cross_task_references"):
        for value in retrieved[category]:
            if value["status"] not in STATUSES:
                raise ValueError("invalid observation status")
            context, request, source = value["source_context"], value["source_request"], value["source"]
            task = context.get("task", {})
            identity = {"context_sha256": digest(context),
                        "assembly_id": task.get("assembly_id"),
                        "input_sha256": _copy(task.get("input_sha256")),
                        "source_commit": task.get("source_commit"),
                        "base_part": task.get("base_part", context.get("base_part")),
                        "goal": _copy(task.get("goal")),
                        "legacy_scope": context.get("scope"),
                        "solver_settings_sha256": digest(task.get("solver_settings", context.get("setup", {}))),
                        "run_id": source["run_id"], "call_id": source["call_id"],
                        "log_ref": source["ref"], "log_sha256": source["sha256"]}
            observed_settings = {k: _copy(request[k]) for k in
                                 ("max_grippers", "timeout", "optimizer", "debug", "render") if k in request}
            seed = task.get("solver_settings", {}).get("seed", context.get("seed"))
            if type(seed) is int:
                observed_settings["seed"] = seed
            raw = value["raw_result"] if isinstance(value["raw_result"], dict) else {}
            stages = value.get("stage_feedback")
            stage_facts = ({k: stages[k] for k in
                           ("path_found", "support_search_succeeded", "stability_passed",
                            "path_search_attempted", "support_search_attempted")
                           if type(stages.get(k)) is bool} if isinstance(stages, dict) else {})
            projected[category].append({
                "record_id": value["record_id"], "status": value["status"],
                "source_part_id": request["part_move"],
                "source_assembled_parts": sorted([request["part_move"]] + request["parts_rest"]),
                "source_parts_removed": _copy(request["parts_removed"]),
                "source_pose": _copy(request["pose"]),
                "source_action": _copy(raw.get("action")),
                "source_support": _copy(raw.get("parts_fix")),
                "source_request_settings": observed_settings, "source_identity": identity,
                "stage_feedback": stage_facts or None,
                "reference_only": category != "exact_observations",
                "current_part_id_mapping": None, "current_attempt_verified": False,
                "constraints": [], "blocker": None,
            })
    selected = projected["exact_observations"] + projected["cross_task_references"]
    projected["evidence_ids"] = [r["record_id"] for r in selected]
    projected["status_counts"] = {s: sum(r["status"] == s for r in selected) for s in STATUSES}
    return projected
