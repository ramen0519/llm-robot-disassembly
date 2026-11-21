"""Explicit opt-in launcher for ONE new three-call skill-selection smoke run.

Default --check-only never reads a key or calls an API. Paid execution requires
--authorize-three-calls, which is a NEW authorization, not reuse of the old cap.
After one-time DPAPI registration it can run from the same user's fresh process.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path('C:/dev/disassembly_codex_ready_v2')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from run_luna_experiment import safe, sha, read, write
from run_recorded import run_recorded
from run_memory_multiseed import preserved_artifacts
from local_api_key import key_environment

RUN_NAME = 'skill_live_smoke_20261017'
DEVELOPMENT = 'skill_loop_20261017'
CONFIG = dict(enabled=True, model='gpt-6-luna', base_url='https://api.openai.com/v1',
    api_key_env='OPENAI_API_KEY', max_calls=3, max_usd=0.01,
    input_usd_per_million=0.10, output_usd_per_million=0.50,
    max_output_tokens=256, reasoning_effort='none', request_overhead_tokens=2048)


def launch(authorize=False, root=ROOT, recorder=run_recorded):
    root = safe(root)
    output = safe(root / 'runs' / RUN_NAME, True)
    development = safe(root / 'runs' / DEVELOPMENT, True)
    library = safe(development / 'frozen_reuse', True)
    report = read(development / 'result.json')
    if report.get('status') != 'PASS':
        raise ValueError('development physics/integrity did not pass')
    frozen = read(library / 'library.json')
    if frozen['source_code_sha256'] != sha(root / 'orchestrator/skill_library.py'):
        raise ValueError('skill code changed; revalidation required')
    for name in ('library.json', 'memory.json'):
        if not safe(library / name, True).is_file():
            raise ValueError('frozen development files missing')
    if output.exists():
        print('ALREADY_STARTED: this fixed run cannot be retried.', flush=True)
        return 3
    print('CONFIG: gpt-6-luna; maximum 3 attempts total; estimated USD 0.01; no retries.', flush=True)
    if not authorize:
        print('CHECK_ONLY: no key read, API call, physics run, or output folder creation.', flush=True)
        return 0
    with key_environment(root) as credential_source:
        if credential_source == 'missing':
            print('KEY_NOT_REGISTERED: run tools/local_api_key.py --register once. No request sent.', flush=True)
            return 2
        return _launch_ready(root, output, library, recorder, credential_source)


def _launch_ready(root, output, library, recorder, credential_source):
    output.mkdir(exist_ok=False)
    write(output / 'approved_llm_config.json', CONFIG)
    write(output / 'task_start.json', dict(authorization='explicit --authorize-three-calls invocation',
        max_calls=3, max_usd=0.01, prior_one_call_budget_reused=False,
        scope='new actual skill-selection smoke; frozen development memory', credential_source=credential_source,
        source_library=str(library), source_hashes={n: sha(library / n) for n in ('library.json', 'memory.json')}))
    write(output / 'preserved_before.json', preserved_artifacts(output))
    command = [str(safe(root / '.venv-orchestrator/Scripts/python.exe')), '-B', '-u',
        str(safe(root / 'tools/run_skill_experiment.py')), '--run-dir', str(output), '--prepared-run',
        '--llm-config', str(output / 'approved_llm_config.json'), '--selector-mode', 'live',
        '--frozen-library', str(library), '--seed', '0', '--allow-live-api']
    print('Running actual official beam physics. Logs: ' + str(output), flush=True)
    result = recorder(command, root, safe(output / 'integration_command', True))
    print('PROCESS_EXIT=' + str(result['exit_code']), flush=True)
    print('Check result.json and response metadata; exit 0 alone does not prove live skill selection.', flush=True)
    return result['exit_code'] if result['exit_code'] is not None else 127


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--authorize-three-calls', action='store_true')
    args = parser.parse_args()
    try:
        raise SystemExit(launch(args.authorize_three_calls))
    except Exception as error:
        print('LAUNCH_ERROR_TYPE=' + type(error).__name__, flush=True)
        raise SystemExit(1)
