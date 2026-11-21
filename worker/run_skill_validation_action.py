"""One bounded real ASAP action for offline generated-procedure validation.

This is a physical case harness, not a second search algorithm. It loads a
declared state snapshot and never accepts a removal on behalf of the DFS.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'worker'))

from remote_rank_client import checked_path
from explicit_action_physics import execute_action, validate_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-request')
    parser.add_argument('--source-request-sha256')
    args = parser.parse_args()
    source = checked_path(args.input)
    output = checked_path(args.output)
    packet = json.loads(source.read_text(encoding='utf-8'))
    if set(packet) != {'context', 'choice'}:
        raise ValueError('exact_worker_packet_required')
    context, choice = packet['context'], packet['choice']
    validate_action(context, choice)
    from skill_case_budget import worker_allowance
    worker_allowance(context, args.source_request, args.source_request_sha256, checked=checked_path)
    # All source/input paths are checked before runtime import.
    root = checked_path(context['asap_root'], run_only=False)
    checked_path(context['assembly_dir'], run_only=False)
    sys.path.insert(0, str(root))
    import numpy as np
    random.seed(context['seed'])
    np.random.seed(context['seed'])
    result = execute_action(context, choice, output)
    print(json.dumps({'status': result['status'], 'path_success': result['path_success'],
                     'stability_success': result['stability_success'],
                     'backend': result['backend'], 'api_calls': 0}), flush=True)
    return 1 if result['status'] == 'ERROR' else 0


if __name__ == '__main__':
    raise SystemExit(main())
