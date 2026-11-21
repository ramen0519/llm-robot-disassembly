"""Measured, conditional endpoint constraints; never general blocking facts.

Only new path endpoint measurement events are accepted.  This module performs
JSON validation and frozen-memory retrieval: it does not open trajectory files,
run physics, infer a failure cause, prune a candidate or reuse a verdict.
Python 3.8 compatible.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Tuple

try:
    from .task_memory import (_copy, _ids, _safe_path, _sha, _source_ref, _text,
                              _validate_context, canonical, digest)
except ImportError:
    from task_memory import (_copy, _ids, _safe_path, _sha, _source_ref, _text,
                             _validate_context, canonical, digest)


POLICY = "conditional_endpoint_measurements_no_pruning_v1"
KIND = "required_final_separation_unsatisfied"
SCOPE_KEYS = {"task_context", "assembled_parts", "part_move", "parts_rest",
              "parts_removed", "base_part", "pose", "support_policy",
              "action_schedule", "physics", "solver_fingerprint", "seed",
              "budget", "rng_before", "initial_q", "collector_version"}
EVENT_KEYS = {"schema_version", "event_id", "kind", "scope", "raw_success",
              "trajectory", "terminal_q", "measurements", "blocker",
              "termination_reason"}
PAIR_KEYS = {"rest_part_id", "moving_part_id", "d_rest_move", "d_move_rest",
             "min_distance", "separation_unsatisfied"}


def _number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _vector(value: Any) -> bool:
    return (type(value) is list and len(value) in (3, 6)
            and all(_number(item) for item in value))


def _pose(value: Any) -> bool:
    return (value is None or (type(value) is list and len(value) == 4
            and all(type(row) is list and len(row) == 4
                    and all(_number(v) for v in row) for row in value)))


def _relative_ref(value: Any) -> None:
    if not _text(value) or len(value) > 4096 or "\n" in value or "\r" in value:
        raise ValueError("invalid trajectory relative path")
    normalized = value.replace("\\", "/")
    if (PureWindowsPath(value).drive or normalized.startswith("/")
            or any(part in ("", ".", "..") for part in normalized.split("/"))):
        raise ValueError("trajectory path must be relative without traversal")


def validate_scope(scope: Any, context: Any = None) -> None:
    """Validate every declared scope component without interpreting unknowns."""
    if type(scope) is not dict or not SCOPE_KEYS.issubset(scope):
        raise ValueError("incomplete constraint scope")
    _validate_context(scope["task_context"])
    if context is not None and canonical(scope["task_context"]) != canonical(context):
        raise ValueError("evidence context disagrees with supplied task")
    task = scope["task_context"]["task"]
    initial = set(scope["task_context"]["initial_parts"])
    assembled, rest, removed = (scope[key] for key in
                                ("assembled_parts", "parts_rest", "parts_removed"))
    move, base = scope["part_move"], scope["base_part"]
    if (not all(_ids(ids) for ids in (assembled, rest, removed))
            or not assembled or not _text(move) or move not in assembled
            or set(rest) != set(assembled) - {move} or not rest
            or not set(assembled).issubset(initial)
            or set(removed) != initial - set(assembled)
            or base != task["base_part"]
            or (base is not None and base not in rest)):
        raise ValueError("invalid assembled, removed, moved or base scope")
    if (not _pose(scope["pose"]) or not _vector(scope["initial_q"])
            or type(scope["seed"]) is not int
            or not _number(scope["budget"]) or scope["budget"] <= 0
            or not _sha(scope["rng_before"]) or not _sha(scope["solver_fingerprint"])
            or type(scope["physics"]) is not dict
            or not _number(scope["physics"].get("min_sep"))
            or scope["physics"]["min_sep"] < 0
            or not _text(scope["collector_version"])
            or type(scope["action_schedule"]) is not list or not scope["action_schedule"]
            or type(scope["support_policy"]) not in (dict, str)
            or not scope["support_policy"]):
        raise ValueError("invalid pose, solver, action or measurement scope")
    canonical(scope)


def validate_event(event: Any, context: Any = None) -> None:
    """Validate a measured endpoint, never convert a bare failure to a cause."""
    if (type(event) is not dict or set(event) != EVENT_KEYS
            or type(event["schema_version"]) is not int or event["schema_version"] != 1
            or not _text(event["event_id"])
            or event["kind"] != "path_endpoint_measurement"
            or type(event["raw_success"]) is not bool
            or event["blocker"] is not None
            or event["termination_reason"] != ("success" if event["raw_success"] else "unknown")):
        raise ValueError("only measured endpoint events may create constraints")
    validate_scope(event["scope"], context)
    trajectory = event["trajectory"]
    if (type(trajectory) is not dict
            or not {"sha256", "relative_path", "states"}.issubset(trajectory)
            or not _sha(trajectory["sha256"])
            or type(trajectory["states"]) is not int or trajectory["states"] < 1):
        raise ValueError("invalid measured trajectory provenance")
    _relative_ref(trajectory["relative_path"])
    if (not _vector(event["terminal_q"])
            or len(event["terminal_q"]) != len(event["scope"]["initial_q"])):
        raise ValueError("invalid measured endpoint state")
    measured = event["measurements"]
    keys = {"min_sep", "pairs", "contact_history", "geometry_queries", "measurement_seconds"}
    if (type(measured) is not dict or set(measured) != keys
            or not _number(measured["min_sep"]) or measured["min_sep"] < 0
            or type(measured["pairs"]) is not list
            or not _ids(measured["contact_history"])
            or type(measured["geometry_queries"]) is not int
            or measured["geometry_queries"] < 4 * len(measured["pairs"])
            or not _number(measured["measurement_seconds"])
            or measured["measurement_seconds"] < 0):
        raise ValueError("invalid endpoint measurements")
    if measured["min_sep"] != event["scope"]["physics"]["min_sep"]:
        raise ValueError("measured threshold disagrees with physics scope")
    seen = set()
    for pair in measured["pairs"]:
        if (type(pair) is not dict or set(pair) != PAIR_KEYS
                or not _text(pair["rest_part_id"])
                or pair["rest_part_id"] in seen
                or pair["moving_part_id"] != event["scope"]["part_move"]
                or pair["rest_part_id"] not in event["scope"]["parts_rest"]
                or not all(_number(pair[key]) for key in
                           ("d_rest_move", "d_move_rest", "min_distance"))
                or pair["min_distance"] != min(pair["d_rest_move"], pair["d_move_rest"])
                or type(pair["separation_unsatisfied"]) is not bool
                or pair["separation_unsatisfied"] != (pair["min_distance"] < measured["min_sep"])):
            raise ValueError("endpoint pair contradicts measured separation predicate")
        if event["raw_success"] and pair["separation_unsatisfied"]:
            raise ValueError("successful path contradicts endpoint separation")
        seen.add(pair["rest_part_id"])
    if seen != set(event["scope"]["parts_rest"]):
        raise ValueError("endpoint measurements must cover every remaining rest part")
    canonical(event)


def _record(event: dict, pair: dict, source: dict) -> dict:
    scope = event["scope"]
    identity = {"event_id": event["event_id"], "scope_sha256": digest(scope),
                "rest_part_id": pair["rest_part_id"]}
    return {"schema_version": 1, "record_id": "constraint:" + digest(identity),
            "kind": KIND, "scope": _copy(scope), "scope_sha256": digest(scope),
            "source": _copy(source), "evidence_event_id": event["event_id"],
            "evidence_sha256": digest(event), "evidence": _copy(event),
            "predicate": {"name": "endpoint_pair_min_distance_at_least_min_sep",
                          "moving_part_id": pair["moving_part_id"],
                          "rest_part_id": pair["rest_part_id"],
                          "measured_min_distance": pair["min_distance"],
                          "required_min_sep": event["measurements"]["min_sep"],
                          "satisfied": False},
            "verdict": "REJECTED", "verdict_scope": "measured_endpoint_predicate_only",
            "whole_part_verdict": "UNKNOWN", "blocker": None,
            "all_paths_impossible": False, "hard_pruning": False, "reuse_verdict": False}


def records_from_evidence(context: dict, events: Iterable[dict], source_ref: str,
                          source_sha256: str) -> list:
    """Create one conditional constraint per measured unsatisfied endpoint pair.

    Input must consist exclusively of the new measurement events. Duplicate
    identical event IDs are idempotent; conflicting IDs fail. A successful or
    unseparated=False measurement yields no constraint, not a fabricated cause.
    Log and trajectory references are provenance pointers, never opened here.
    """
    _validate_context(context)
    _source_ref(source_ref)
    if not _sha(source_sha256):
        raise ValueError("source log requires a SHA-256")
    seen, records = {}, []
    source = {"ref": source_ref, "sha256": source_sha256}
    for value in events:
        event = _copy(value)
        validate_event(event, context)
        identity, payload = event["event_id"], canonical(event)
        if identity in seen:
            if seen[identity] != payload:
                raise ValueError("conflicting measurement event ID")
            continue
        seen[identity] = payload
        for pair in event["measurements"]["pairs"]:
            if pair["separation_unsatisfied"]:
                records.append(_record(event, pair, source))
    return _normal_records(records)


def _validate_record(record: Any) -> None:
    if type(record) is not dict:
        raise ValueError("invalid conditional constraint record")
    event, source = record.get("evidence"), record.get("source")
    validate_event(event)
    if (type(source) is not dict or set(source) != {"ref", "sha256"}
            or not _sha(source["sha256"])):
        raise ValueError("invalid constraint source provenance")
    _source_ref(source["ref"])
    expected = [_record(event, pair, source) for pair in event["measurements"]["pairs"]
                if pair["separation_unsatisfied"]]
    if not any(canonical(record) == canonical(candidate) for candidate in expected):
        raise ValueError("constraint is not derived from its exact measured evidence")


def _normal_records(records: Iterable[dict]) -> list:
    unique = {}
    for value in records:
        record = _copy(value)
        _validate_record(record)
        identity = record["record_id"]
        if identity in unique and canonical(unique[identity]) != canonical(record):
            raise ValueError("conflicting constraint record ID")
        unique[identity] = record
    return [unique[key] for key in sorted(unique)]


def save_snapshot(records: Iterable[dict], path: Path, allow_test_root: bool = False) -> dict:
    path = _safe_path(path, allow_test_root)
    normalized = _normal_records(records)
    snapshot = {"schema_version": 1, "policy": POLICY, "records": normalized,
                "record_count": len(normalized), "content_sha256": digest(normalized),
                "hard_pruning": False, "reuse_verdict": False}
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(snapshot) + "\n")
    return snapshot


def load_snapshot(path: Path, allow_test_root: bool = False) -> dict:
    path = _safe_path(path, allow_test_root)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    keys = {"schema_version", "policy", "records", "record_count", "content_sha256",
            "hard_pruning", "reuse_verdict"}
    if (type(value) is not dict or set(value) != keys
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["policy"] != POLICY or type(value["records"]) is not list
            or value["hard_pruning"] is not False or value["reuse_verdict"] is not False):
        raise ValueError("invalid frozen constraint memory")
    if (value["records"] != _normal_records(value["records"])
            or type(value["record_count"]) is not int
            or value["record_count"] != len(value["records"])
            or value["content_sha256"] != digest(value["records"])):
        raise ValueError("frozen constraint memory integrity mismatch")
    return value


@dataclass(frozen=True, init=False)
class FrozenConstraintMemory:
    _records: Tuple[str, ...]
    content_sha256: str

    def __init__(self, records: Iterable[dict]):
        values = _normal_records(records)
        object.__setattr__(self, "_records", tuple(canonical(v) for v in values))
        object.__setattr__(self, "content_sha256", digest(values))

    def retrieve(self, context: dict, assembled_parts: Iterable[str],
                 candidate_ids: Iterable[str], request_scope: dict = None,
                 max_records: int = 12) -> dict:
        """Return source-scoped references, never a candidate ordering or filter."""
        _validate_context(context)
        assembled, candidates = list(assembled_parts), list(candidate_ids)
        base = context["task"]["base_part"]
        if (not _ids(assembled) or not assembled or not _ids(candidates)
                or not set(assembled).issubset(context["initial_parts"])
                or (base is not None and base not in assembled)
                or set(candidates) != set(assembled) - ({base} if base is not None else set())
                or type(max_records) is not int or not 0 <= max_records <= 12):
            raise ValueError("invalid full candidate retrieval state or limit")
        if request_scope is not None:
            validate_scope(request_scope, context)
            if (set(request_scope["assembled_parts"]) != set(assembled)
                    or request_scope["part_move"] not in candidates):
                raise ValueError("request scope disagrees with current retrieval state")
        groups = {"exact_scope_observation": [], "same_task_state_reference": [],
                  "source_conditions_reference": []}
        for payload in self._records:
            record = json.loads(payload)
            scope = record["scope"]
            same_task_state = (canonical(scope["task_context"]) == canonical(context)
                               and set(scope["assembled_parts"]) == set(assembled)
                               and scope["part_move"] in candidates)
            exact = request_scope is not None and canonical(scope) == canonical(request_scope)
            category = ("exact_scope_observation" if exact else
                        "same_task_state_reference" if same_task_state else
                        "source_conditions_reference")
            view = {"record_id": record["record_id"], "kind": record["kind"],
                    "applicability": category, "reference_only": not exact,
                    "source_scope": scope, "source_scope_sha256": record["scope_sha256"],
                    "evidence_origin": {"source": record["source"],
                                        "event_id": record["evidence_event_id"],
                                        "event_sha256": record["evidence_sha256"],
                                        "trajectory": record["evidence"]["trajectory"]},
                    "observed_predicate": record["predicate"],
                    "source_verdict": record["verdict"],
                    "source_verdict_scope": record["verdict_scope"],
                    "whole_part_verdict": "UNKNOWN", "blocker": None,
                    "current_part_id_mapping": None, "current_attempt_verified": False,
                    "all_paths_impossible": False, "hard_pruning": False,
                    "reuse_verdict": False}
            groups[category].append(view)
        selected = []
        for category in groups:
            selected.extend(groups[category])
        selected = selected[:max_records]
        return {"schema_version": 1, "policy": POLICY, "records": selected,
                "record_ids": [value["record_id"] for value in selected],
                "content_sha256": self.content_sha256,
                "truncated": len(self._records) > len(selected),
                "scope_counts": {category: sum(value["applicability"] == category
                                               for value in selected) for category in groups},
                "hard_pruning": False, "reuse_verdict": False,
                "current_attempt_verified": False}
