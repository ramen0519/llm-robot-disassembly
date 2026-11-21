"""Append-only, scoped ASAP observations for frozen soft ranking, not caching.

These records describe bounded local verifier attempts. A failed attempt never
establishes impossibility, and this module never removes a search candidate.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple


INITIAL_PARTS = ["0", "1", "2", "3", "6"]
STATUSES = ("PASS", "UNKNOWN", "ERROR")
POLICY = "positive_only_stable_promotion"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _copy(value: Any) -> Any:
    # Both copy and assert that the value is finite JSON, without solver objects.
    return json.loads(canonical(value))


def beam_context(input_snapshot: dict, setup: dict, seed: int = 0) -> dict:
    context = {
        "version": 1, "scope": "official_beam_original",
        "usage": "soft_ranking_only", "input_snapshot": deepcopy(input_snapshot),
        "setup": deepcopy(setup), "seed": seed, "base_part": "6",
        "initial_parts": INITIAL_PARTS.copy(),
    }
    _validate_context(context)
    return _copy(context)


def _validate_context(context: Any) -> None:
    if (not isinstance(context, dict) or type(context.get("version")) is not int
            or context["version"] != 1
            or context.get("scope") != "official_beam_original"
            or context.get("usage") != "soft_ranking_only"
            or context.get("base_part") != "6"
            or context.get("initial_parts") != INITIAL_PARTS
            or type(context.get("seed")) is not int
            or not isinstance(context.get("input_snapshot"), dict)
            or not isinstance(context.get("setup"), dict)):
        raise ValueError("invalid official beam ranking context")
    canonical(context)


def _ids(value: Any) -> bool:
    return (type(value) is list and all(type(p) is str for p in value)
            and len(set(value)) == len(value))


def _nonempty(value: Any) -> bool:
    return type(value) is str and bool(value)


def _validate_record(record: Any) -> None:
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1
            or record.get("scope") != "bounded_search"
            or record.get("usage") != "soft_ranking_only"
            or record.get("status") not in STATUSES
            or record.get("blocker") is not None or record.get("path") is not None):
        raise ValueError("invalid evidence record schema or policy")
    context = record.get("context")
    _validate_context(context)
    source, request = record.get("source"), record.get("request")
    if not isinstance(source, dict) or not isinstance(request, dict):
        raise ValueError("missing evidence source or request")
    if (not all(_nonempty(source.get(key)) for key in
                ("run_id", "call_id", "start_event_id", "result_event_id", "ref", "sha256"))
            or record.get("record_id") != source["result_event_id"]
            or source["start_event_id"] == source["result_event_id"]
            or len(source["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in source["sha256"])):
        raise ValueError("inconsistent evidence provenance")
    required = {"part_move", "parts_rest", "parts_removed", "pose", "max_grippers",
                "timeout", "optimizer", "debug", "render"}
    if (not required.issubset(request) or not _nonempty(request["part_move"])
            or not _ids(request["parts_rest"]) or not _ids(request["parts_removed"])
            or request["part_move"] in request["parts_rest"]
            or request["part_move"] == context["base_part"]
            or context["base_part"] not in request["parts_rest"]
            or type(request["max_grippers"]) is not int or request["max_grippers"] < 1
            or not _nonempty(request["optimizer"])):
        raise ValueError("invalid bounded verifier request")
    assembled = set(request["parts_rest"]) | {request["part_move"]}
    if (not assembled.issubset(context["initial_parts"])
            or set(request["parts_removed"]) != set(context["initial_parts"]) - assembled):
        raise ValueError("inconsistent assembled and removed parts")
    raw, status = record.get("raw_result"), record["status"]
    if status == "ERROR":
        if raw is not None or not _nonempty(record.get("error_type")):
            raise ValueError("ERROR must retain a null raw result and error type")
    else:
        if (not isinstance(raw, dict) or type(raw.get("feasible")) is not bool
                or raw["feasible"] != (status == "PASS")
                or raw.get("part_move") != request["part_move"]
                or raw.get("base_part") != context["base_part"]
                or "pose" not in raw or canonical(raw["pose"]) != canonical(request["pose"])):
            raise ValueError("status and raw verifier result disagree")
        if status == "PASS":
            supports = raw.get("parts_fix")
            action = raw.get("action")
            if (not _ids(supports) or not set(supports).issubset(request["parts_rest"])
                    or len(supports) + 1 > request["max_grippers"]
                    or type(action) is not list or len(action) != 3
                    or any(type(v) not in (int, float) for v in action)
                    or not any(v != 0 for v in action)):
                raise ValueError("PASS lacks valid action or support evidence")
    canonical(record)


def record_from_pair(context: dict, start_event: dict, result_event: dict,
                     source_ref: str, source_sha256: str) -> dict:
    if (start_event.get("kind") != "verifier_start"
            or result_event.get("kind") != "verifier_result"
            or not _nonempty(start_event.get("run_id"))
            or start_event.get("run_id") != result_event.get("run_id")
            or not _nonempty(start_event.get("call_id"))
            or start_event.get("call_id") != result_event.get("call_id")):
        raise ValueError("verifier events do not form the same run/call pair")
    record = {
        "schema_version": 1, "record_id": result_event.get("event_id"),
        "source": {"run_id": start_event["run_id"], "call_id": start_event["call_id"],
                   "start_event_id": start_event.get("event_id"),
                   "result_event_id": result_event.get("event_id"),
                   "ref": source_ref, "sha256": source_sha256},
        "context": deepcopy(context), "request": deepcopy(start_event.get("request")),
        "status": result_event.get("status"), "raw_result": deepcopy(result_event.get("raw")),
        "error_type": result_event.get("error_type"),
        "scope": "bounded_search", "usage": "soft_ranking_only",
        "blocker": None, "path": None,
    }
    _validate_record(record)
    return _copy(record)


class EvidenceStore:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(str(path))
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY, payload TEXT NOT NULL, hash TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS records_no_update
            BEFORE UPDATE ON records BEGIN
                SELECT RAISE(ABORT, 'evidence records are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS records_no_delete
            BEFORE DELETE ON records BEGIN
                SELECT RAISE(ABORT, 'evidence records are append-only');
            END;
        """)

    def insert_many(self, records: Iterable[dict]) -> Dict[str, int]:
        inserted, duplicates = 0, 0
        with self.connection:
            for record in records:
                _validate_record(record)
                payload = canonical(record)
                found = self.connection.execute(
                    "SELECT payload, hash FROM records WHERE record_id = ?",
                    (record["record_id"],)).fetchone()
                if found is not None:
                    if found != (payload, digest(record)):
                        raise ValueError("record ID conflict: " + record["record_id"])
                    duplicates += 1
                else:
                    self.connection.execute("INSERT INTO records VALUES (?, ?, ?)",
                                            (record["record_id"], payload, digest(record)))
                    inserted += 1
        return {"inserted": inserted, "duplicates": duplicates}

    def records(self) -> List[dict]:
        result = []
        for record_id, payload, stored_hash in self.connection.execute(
                "SELECT record_id, payload, hash FROM records ORDER BY record_id"):
            record = json.loads(payload)
            _validate_record(record)
            if record["record_id"] != record_id or digest(record) != stored_hash:
                raise ValueError("stored evidence hash or ID mismatch")
            result.append(record)
        return result

    def close(self) -> None:
        self.connection.close()


def export_snapshot(store: EvidenceStore, path: Path) -> dict:
    records = store.records()
    snapshot = {"schema_version": 1, "records": records, "record_count": len(records),
                "source_record_ids": [record["record_id"] for record in records],
                "content_sha256": digest(records), "policy": POLICY,
                "allow_hard_pruning": False}
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical(snapshot) + "\n")
    return snapshot


@dataclass(frozen=True, init=False)
class FrozenMemory:
    _records: Tuple[str, ...]
    content_sha256: str

    def __init__(self, path: Path):
        with Path(path).open(encoding="utf-8") as stream:
            snapshot = json.load(stream)
        if (not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int
                or snapshot["schema_version"] != 1
                or snapshot.get("policy") != POLICY
                or snapshot.get("allow_hard_pruning") is not False
                or type(snapshot.get("records")) is not list):
            raise ValueError("invalid frozen evidence schema or policy")
        records = snapshot["records"]
        for record in records:
            _validate_record(record)
        ids = [record["record_id"] for record in records]
        if (ids != sorted(set(ids)) or snapshot.get("source_record_ids") != ids
                or type(snapshot.get("record_count")) is not int
                or snapshot["record_count"] != len(records)
                or snapshot.get("content_sha256") != digest(records)):
            raise ValueError("frozen evidence content, count or source IDs mismatch")
        object.__setattr__(self, "_records", tuple(canonical(record) for record in records))
        object.__setattr__(self, "content_sha256", snapshot["content_sha256"])

    def lookup(self, context: dict, assembled_parts: tuple, base_part: str,
               pose: Any = None, max_grippers: int = 2) -> dict:
        result = {"evidence_ids": [], "positive_candidate_ids": [],
                  "status_counts": {status: 0 for status in STATUSES}}
        _validate_context(context)
        if (type(assembled_parts) is not tuple
                or not all(type(p) is str for p in assembled_parts)
                or len(set(assembled_parts)) != len(assembled_parts)):
            raise ValueError("assembled parts must be a tuple of unique string IDs")
        assembled = set(assembled_parts)
        if (base_part != context["base_part"] or base_part not in assembled
                or not assembled.issubset(context["initial_parts"])):
            return result
        context_json, pose_json = canonical(context), canonical(pose)
        positive = set()
        for payload in self._records:
            record = json.loads(payload)
            request = record["request"]
            if (canonical(record["context"]) != context_json
                    or set(request["parts_rest"]) | {request["part_move"]} != assembled
                    or set(request["parts_removed"]) != set(context["initial_parts"]) - assembled
                    or canonical(request["pose"]) != pose_json
                    or request["max_grippers"] != max_grippers
                    or request["optimizer"] != context["setup"].get("optimizer")
                    or request["timeout"] is not None):
                continue
            result["evidence_ids"].append(record["record_id"])
            result["status_counts"][record["status"]] += 1
            if record["status"] == "PASS":
                positive.add(request["part_move"])
        result["positive_candidate_ids"] = sorted(positive)
        return result
