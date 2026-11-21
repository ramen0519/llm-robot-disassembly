"""Build a read-only linked knowledge view from a SAVED request; no API/physics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator.knowledge_context import KnowledgeContext
from orchestrator.lesson_memory import write_new
from worker.task_memory import _safe_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--skill-library", required=True)
    parser.add_argument("--lesson-library")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--max-skills", type=int, default=16)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(_safe_path(path).read_text(encoding="utf-8"))
    view = KnowledgeContext(read(args.skill_library), read(args.lesson_library) if args.lesson_library else None,
        allow_fixture=args.allow_fixture).query(read(args.request), max_skills=args.max_skills)
    write_new(args.output, view)
    print(json.dumps(dict(output=args.output, selected_skills=len(view["skills"]),
        linked_lessons=sum(len(s["lessons"]) for s in view["skills"]),
        observations=len(view["observations"]), api_calls=0, physics_calls=0), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
