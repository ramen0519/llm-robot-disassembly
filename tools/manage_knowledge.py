"""Offline immutable Memory/Skill bundles and saved skill-stage statistics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.knowledge_store import (create_revision, load_revision, promote, resolve_bundle,
    read_json, _new, _encoded, source_fingerprints)
from orchestrator.skill_usage import summarize_usage, summarize_declared_case
from worker.task_memory import _lexical_path, _safe_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("create", "inspect", "activate", "pin"):
        child = sub.add_parser(name)
        child.add_argument("--store", type=Path, required=True)
        child.add_argument("--revision", required=name != "pin")
        child.add_argument("--allow-fixture", action="store_true")
        if name in ("create", "activate"):
            child.add_argument("--reason", required=True)
        if name == "create":
            child.add_argument("--memory", type=Path, required=True)
            child.add_argument("--skills", type=Path, required=True)
            child.add_argument("--purpose", choices=("development", "evaluation"), default="development")
        if name == "activate":
            child.add_argument("--expected-active-sha256", required=True, help="none for the first activation")
        if name == "pin":
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--purpose", choices=("development", "evaluation"), default="development")
    usage = sub.add_parser("summarize")
    usage.add_argument("--episode", type=Path, required=True)
    usage.add_argument("--library", type=Path)
    usage.add_argument("--output", type=Path, required=True)
    usage.add_argument("--allow-fixture", action="store_true")
    usage.add_argument("--format", choices=("dfs", "declared-case"), default="dfs")
    args = p.parse_args(argv)
    for value in vars(args).values():
        if isinstance(value, Path):
            _lexical_path(value)
    if args.command == "create":
        result = create_revision(args.store, args.revision, args.memory, args.skills,
            reason=args.reason, purpose=args.purpose, allow_fixture=args.allow_fixture)
    elif args.command == "inspect":
        result = load_revision(args.store, args.revision, allow_fixture=args.allow_fixture)
    elif args.command == "activate":
        result = promote(args.store, args.revision,
            expected_active_sha256=None if args.expected_active_sha256 == "none" else args.expected_active_sha256,
            reason=args.reason, allow_fixture=args.allow_fixture)
    elif args.command == "pin":
        result = resolve_bundle(args.store, args.revision, purpose=args.purpose, allow_fixture=args.allow_fixture)["pin"]
        _new(args.output, _encoded(result))
    else:
        episode = _safe_path(args.episode)
        sources = {}
        def read(name, optional=False):
            path = _safe_path(episode / name)
            if optional and not path.exists():
                return None
            value, raw = read_json(path)
            from hashlib import sha256
            sources[str(path)] = sha256(raw).hexdigest()
            return value
        library, library_bytes = read_json(args.library) if args.library else (None, None)
        fingerprints = source_fingerprints()
        if args.format == "dfs":
            decisions = read("host_decisions.json")
            report = read("worker_result.json", True)
            selections = read("generated_skill_selections.json", True)
            result = summarize_usage(decisions, report, selections, library, allow_fixture=args.allow_fixture)
            result["execution_mode"] = "free_dfs"
        else:
            result = summarize_declared_case(read("program_request.json"), read("response.json"),
                read("selector_decisions.json", True), read("recovery/physics/result.json", True),
                read("case_report.json"), library, allow_fixture=args.allow_fixture)
        if args.library:
            from hashlib import sha256
            sources[str(_safe_path(args.library))] = sha256(library_bytes).hexdigest()
        result["source_file_sha256"] = sources
        result["analysis_sources"] = fingerprints
        from hashlib import sha256
        if any(sha256(read_json(path)[1]).hexdigest() != expected for path, expected in sources.items()):
            raise ValueError("usage_source_changed_during_analysis")
        if source_fingerprints() != fingerprints:
            raise ValueError("usage_interpreter_changed_during_analysis")
        result["source_files_unchanged_during_analysis"] = True
        # Never overwrite a source log, even if the user supplied its name as output.
        _new(args.output, _encoded(result))
    summary = result.get("manifest", result)
    print(json.dumps({"command": args.command, "kind": summary.get("kind"),
                      "revision": summary.get("revision"), "status": "PASS",
                      **{key: summary[key] for key in ("memory_records", "registered_skills", "content_sha256")
                         if key in summary}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
