"""Launch the authorized Luna smoke test from the key-bearing PowerShell session.

This fixed run cannot be launched twice. No key value is printed or saved.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('C:/dev/disassembly_codex_ready_v2')
sys.path.insert(0, str(ROOT))
from tools.run_luna_experiment import safe
from tools.run_recorded import run_recorded

RUN_NAME = 'luna_live_smoke_20261017'
CONFIG_SHA256 = 'b54eec9b649770aaba7a6c43deec6d0475490de83307ef7f7ee787f0655c9fcf'


def launch(root: Path = ROOT, recorder=run_recorded) -> int:
    """Validate the fixed authorization and claim one execution before starting."""
    root = safe(root)
    run = safe(root / 'runs' / RUN_NAME, runs_only=True)
    config = safe(run / 'approved_llm_config.json', runs_only=True)
    if hashlib.sha256(config.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError('approved configuration changed; refusing execution')
    if not safe(run / 'task_start.json', runs_only=True).is_file() or not safe(run / 'preserved_before.json', runs_only=True).is_file():
        raise ValueError('prepared run evidence is missing')
    for name in ('integration_command', 'execution_start.json', 'implementation', 'batch_ledger.sqlite'):
        if safe(run / name, runs_only=True).exists():
            print('ALREADY_STARTED: no second launch or API retry is allowed.', flush=True)
            return 3
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        print('KEY_NOT_IN_THIS_SESSION: run this command in the PowerShell where you set the key.', flush=True)
        print('No API request sent. No launch claimed.', flush=True)
        return 2
    command = [str(safe(root / '.venv-orchestrator/Scripts/python.exe')), '-B', '-u',
        str(safe(root / 'tools/run_luna_experiment.py')), '--run-dir', str(run),
        '--prepared-run', '--llm-config', str(config), '--base-order', 'native',
        '--memory', 'off', '--seed', '0', '--allow-live-api']
    print('Starting: gpt-6-luna; maximum 1 API attempt; estimated budget USD 0.01; no retries.', flush=True)
    print('Running official beam physics. Keep this window open; logs: ' + str(run), flush=True)
    # mkdir(exist_ok=False) in the recorder atomically claims this fixed run.
    result = recorder(command, root, safe(run / 'integration_command', runs_only=True))
    print('PROCESS_EXIT=' + str(result['exit_code']), flush=True)
    result_path = safe(run / 'result.json', runs_only=True)
    if result_path.is_file():
        report = json.loads(result_path.read_text(encoding='utf-8'))
        print('Execution report saved: ' + str(result_path), flush=True)
        print('Execution status is not proof of a successful LLM response.', flush=True)
        print('RUN_STATUS=' + str(report.get('status')), flush=True)
    return result['exit_code'] if result['exit_code'] is not None else 127


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print('No arguments accepted; this launcher is pinned to the authorized run.')
        raise SystemExit(2)
    try:
        raise SystemExit(launch())
    except Exception as error:
        # Arbitrary exception messages may contain external data; retain type only.
        print('LAUNCH_ERROR_TYPE=' + type(error).__name__)
        raise SystemExit(1)
