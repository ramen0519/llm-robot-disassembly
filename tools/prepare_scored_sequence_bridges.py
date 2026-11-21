"""Freeze explicit historical scorer registrations for order-only use. No API."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from orchestrator.relational_knowledge import safe_path
from orchestrator.scored_sequence_bridge import build_bridge
from orchestrator.scored_sequence_runtime import FROZEN_PROTOCOL, load_bridges
from worker.task_memory import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = safe_path(args.output)
    if output.exists():
        raise ValueError('new_output_required')
    value = dict(protocol=FROZEN_PROTOCOL,
        entries=[build_bridge(safe_path(path)) for path in args.registry],
        API_calls=0, physics_calls=0, native_physics_validation='NOT_RUN')
    value['content_sha256'] = digest(value)
    load_bridges(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps(dict(skills=len(value['entries']), output=str(output),
                         content_sha256=value['content_sha256'], API_calls=0, physics_calls=0)))


if __name__ == '__main__':
    main()
