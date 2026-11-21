"""Run the fixed offline regression gate. Never starts API or ASAP/WSL physics."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import subprocess
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.research_release import (PROFILE_ID, COVERED_PROFILES, SCHEMA, TEST_MODULES,
    source_inventory, replay_inventory, workspace_path, sha, gate_tests)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    output = workspace_path(args.output)
    if output.relative_to(ROOT).parts[0] != 'runs':
        raise ValueError('release_output_must_be_in_runs')
    output.mkdir(parents=True, exist_ok=False)
    sources, replays = source_inventory(), replay_inventory()
    tests = gate_tests()
    started = time.monotonic()
    log = output/'tests.log'
    original_popen = subprocess.Popen
    def only_skill_vm(command, *args, **kwargs):
        if (not isinstance(command, list) or len(command) != 5
                or command[0] != sys.executable or command[1:4] != ['-I', '-S', '-B']
                or Path(command[4]).absolute() != ROOT/'orchestrator/python_skill_child.py'):
            raise AssertionError('OFFLINE_GATE_ONLY_LOCAL_SKILL_VM_SUBPROCESS')
        return original_popen(command, *args, **kwargs)
    with log.open('x', encoding='utf-8') as stream, ExitStack() as stack:
        stack.enter_context(patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('OFFLINE_GATE_NETWORK_FORBIDDEN')))
        stack.enter_context(patch('socket.create_connection', side_effect=AssertionError('OFFLINE_GATE_NETWORK_FORBIDDEN')))
        stack.enter_context(patch('subprocess.Popen', side_effect=only_skill_vm))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(tests.values()))
    stats = dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses))
    unchanged = source_inventory() == sources and replay_inventory() == replays
    passed = result.wasSuccessful() and not any(stats[k] for k in stats if k != 'run') and unchanged
    value = dict(schema=SCHEMA, profile_id=PROFILE_ID, covered_profiles=list(COVERED_PROFILES),
        status='PASS' if passed else 'FAIL',
        created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
        python_version=sys.version, test_modules=list(TEST_MODULES), test_ids=list(tests), tests=stats,
        source_sha256=sources, replay_sha256=replays, source_and_replays_unchanged=unchanged,
        test_log=log.relative_to(ROOT).as_posix(), test_log_sha256=sha(log),
        API_calls=0, physics_calls=0, authorizes_paid_execution=False,
        physical_performance='UNVERIFIED', evidence_level='offline_contracts_saved_input_replay_and_injected_dependencies')
    with (output/'certificate.json').open('x', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    print(json.dumps({k:value[k] for k in ('status','tests','elapsed_seconds','source_and_replays_unchanged',
        'API_calls','physics_calls','physical_performance')}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
