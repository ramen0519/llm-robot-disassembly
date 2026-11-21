"""Extract only the preselected official 07009 task into this prototype run.

This host-side preparation performs no physics, networking, or API calls.  It
does not rescale, rename, rotate, simplify, or otherwise change the OBJ payloads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


ROOT = Path("C:/dev/disassembly_codex_ready_v2")
RUN = ROOT / "runs/prototype_completion_20261018"
ARCHIVE = ROOT / "runs/assembly_rules_20261017/test_assembly.zip"
ARCHIVE_SHA256 = "c89ea240b1962e3230b00487bf23f3fcbb8b5cd7b6ce1a6816e043a31cac09c2"
TASK_ID = "07009"
MEMBER_PREFIX = "test_assembly/07009/"
EXPECTED_SHA256 = {
    "0.obj": "f40642c67a0e04dc10a7d94007aa032315098632ff0bb17d203f515f2ef9a406",
    "1.obj": "e4feac510f40f93648369b433a3c6958907a265c3b8671c1a24c78ac59edd901",
    "2.obj": "4a0c7627b94f5ea87c5c85fbb47a082bee27f089bf7b05af05208115f17a5d64",
    "3.obj": "cb3a35d894562a84c296aa1e35f37206fda455541c03e8f973978384223251f5",
    "4.obj": "40aa44922ab87309f540c9e2ea9d022dd9853aa8c967a75631e896084b95f98c",
}


def checked_path(value, expected):
    """Reject disallowed strings before any filesystem operation, then links."""
    raw = str(value)
    normalized = raw.replace("\\", "/")
    lower = normalized.lower()
    if ("\x00" in raw or lower.startswith("e:") or lower == "/mnt/e"
            or lower.startswith("/mnt/e/") or normalized.startswith("//")
            or any(part in (".", "..") for part in normalized.split("/"))):
        raise ValueError("Unsafe path; E drive and path escapes are forbidden")
    path = Path(value)
    if not path.is_absolute() or path != expected:
        raise ValueError("Only the predeclared C-drive prototype paths are allowed")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse paths are forbidden")
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_members(infos):
    """Validate the complete ZIP catalog and return the one task's six files."""
    seen = set()
    selected = {}
    expected = set(EXPECTED_SHA256) | {"contact_graph.json"}
    for info in infos:
        name = info.filename
        # ZipInfo normalizes Windows backslashes and truncates at NUL before
        # exposing filename. Validate the original archive spelling as well.
        original_name = info.orig_filename
        if original_name != name:
            raise ValueError("ZIP member names must not require normalization")
        path = PurePosixPath(name)
        if ("\x00" in name or "\\" in name or ":" in name
                or path.is_absolute() or any(x in (".", "..") for x in name.split("/"))
                or name in seen):
            raise ValueError("Unsafe or duplicate ZIP member")
        seen.add(name)
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise ValueError("ZIP symlinks are forbidden")
        if name.startswith(MEMBER_PREFIX) and not info.is_dir():
            leaf = name[len(MEMBER_PREFIX):]
            if leaf not in expected or "/" in leaf:
                raise ValueError("Unexpected task member")
            if not 0 < info.file_size <= 1024 * 1024:
                raise ValueError("Task member size is outside the expected bound")
            selected[leaf] = info
    if set(selected) != expected:
        raise ValueError("The task member set differs from the preselection")
    return selected


def prepare(run_dir, archive_path=ARCHIVE):
    """Create RUN/raw_assemblies/07009 once; existing outputs are never replaced."""
    run = checked_path(run_dir, RUN)
    archive = checked_path(archive_path, ARCHIVE)
    if not run.is_dir():
        raise ValueError("Reserve the prototype run before preparing its input")
    selection_path = checked_path(run / "dataset_selection.json", RUN / "dataset_selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("selected_id") != TASK_ID
            or selection.get("archive_sha256") != ARCHIVE_SHA256):
        raise ValueError("The predeclared selection does not match this extraction")
    destination = checked_path(run / "raw_assemblies" / TASK_ID, RUN / "raw_assemblies" / TASK_ID)
    report = checked_path(run / "dataset_preparation.json", RUN / "dataset_preparation.json")
    if destination.exists() or report.exists():
        raise FileExistsError("Task destination/report already exists; nothing overwritten")
    if file_sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Official archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        members = validated_members(bundle.infolist())
        payloads = {name: bundle.read(info) for name, info in sorted(members.items())}
    file_hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    if any(file_hashes[name] != digest for name, digest in EXPECTED_SHA256.items()):
        raise ValueError("Official mesh payload hash mismatch")
    manifest_hashes = selection["selected_files"]
    if any(file_hashes[name] != manifest_hashes[name]["sha256"] for name in payloads):
        raise ValueError("Task payload differs from the predeclared selection")
    destination.parent.mkdir(exist_ok=True)
    checked_path(destination, RUN / "raw_assemblies" / TASK_ID)
    destination.mkdir()  # atomic refusal if another process already prepared it
    for name, payload in payloads.items():
        output = checked_path(destination / name, RUN / "raw_assemblies" / TASK_ID / name)
        with output.open("xb") as stream:
            stream.write(payload)
    result = {
        "schema_version": 1,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "selected_id": TASK_ID,
        "archive_path": str(archive),
        "archive_sha256": ARCHIVE_SHA256,
        "destination": str(destination),
        "files": {name: {"sha256": file_hashes[name], "bytes": len(payload)}
                  for name, payload in payloads.items()},
        "method": "byte-exact copy of predeclared OBJ/JSON members; no transformations",
        "api_calls": 0,
        "verifier_calls": 0,
        "physics_status": "NOT_RUN",
    }
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.run_dir)
    print(json.dumps({"selected_id": TASK_ID, "destination": result["destination"],
                      "api_calls": 0, "verifier_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
