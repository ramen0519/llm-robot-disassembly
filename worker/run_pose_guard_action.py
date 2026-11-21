"""One fresh action in a declared pose-guard state; never a DFS transition."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
from remote_rank_client import checked_path
from explicit_action_physics import execute_action, validate_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    packet = json.loads(checked_path(args.input).read_text(encoding="utf-8"))
    if set(packet) != {"context", "choice"}:
        raise ValueError("exact_worker_packet_required")
    context, choice = packet["context"], packet["choice"]
    validate_action(context, choice)
    if context.get("remaining_wall_seconds") != 65:
        raise ValueError("declared_65_second_case_required")
    sys.path.insert(0, str(checked_path(context["asap_root"], run_only=False)))
    checked_path(context["assembly_dir"], run_only=False)
    import numpy as np
    random.seed(context["seed"])
    np.random.seed(context["seed"])
    result = execute_action(context, choice, checked_path(args.output))
    print(json.dumps(dict(status=result["status"], backend=result["backend"],
        path_success=result["path_success"], stability_success=result["stability_success"],
        api_calls=0)), flush=True)
    return 1 if result["status"] == "ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
