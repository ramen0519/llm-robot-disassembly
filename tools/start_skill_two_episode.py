"""Explicit opt-in for one new Luna development-and-reuse experiment.

Check-only validates local inputs without reading a key or creating a run.
The six-call / USD 0.01 cap is shared by both episodes in one runner ledger.
Old one-call and three-call approvals are not reused by this launcher.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path('C:/dev/disassembly_codex_ready_v2')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from run_luna_experiment import safe, sha, write, validate_config
from run_recorded import run_recorded
from run_memory_multiseed import preserved_artifacts
from local_api_key import key_environment

RUN_NAME = 'skill_live_two_episode_20261017'
CONFIG = dict(enabled=True, model='gpt-6-luna', base_url='https://api.openai.com/v1',
    api_key_env='OPENAI_API_KEY', max_calls=6, max_usd=0.01,
    input_usd_per_million=0.10, output_usd_per_million=0.50,
    max_output_tokens=256, reasoning_effort='none', request_overhead_tokens=2048)
REQUIRED_FILES = ('.venv-orchestrator/Scripts/python.exe',
    'tools/run_skill_experiment.py', 'tools/local_api_key.py',
    'orchestrator/skill_library.py', 'orchestrator/skill_selector.py',
    'orchestrator/skill_pipeline.py')


def launch(authorize=False, root=ROOT, recorder=run_recorded, *, run_name=RUN_NAME):
    root = safe(root)
    if (type(run_name) is not str or not run_name or
            any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in run_name)):
        raise ValueError('invalid exclusive run name')
    output = safe(root / 'runs' / run_name, True)
    for name in REQUIRED_FILES:
        if not safe(root / name).is_file():
            raise ValueError('required local input is missing')
    if output.exists():
        print('ALREADY_STARTED: this fixed run cannot be retried.', flush=True)
        return 3
    print('CONFIG: gpt-6-luna; 2 consecutive episodes; maximum 6 attempts TOTAL; '
          'estimated USD 0.01 TOTAL; no retries.', flush=True)
    if not authorize:
        print('CHECK_ONLY: no key read, API call, physics run, or output folder creation.', flush=True)
        return 0
    with key_environment(root) as credential_source:
        if credential_source == 'missing':
            print('KEY_NOT_REGISTERED: no request sent and no run created.', flush=True)
            return 2
        validate_config(CONFIG, True)
        return _launch_ready(root, output, recorder, credential_source)


def _launch_ready(root, output, recorder, credential_source):
    output.mkdir(exist_ok=False)
    write(output / 'approved_llm_config.json', CONFIG)
    write(output / 'task_start.json', dict(
        authorization='explicit --authorize-six-calls invocation',
        max_calls=6, max_usd=0.01, max_calls_scope='both episodes combined',
        prior_one_call_budget_reused=False, prior_three_call_budget_reused=False,
        prior_six_call_budget_reused=False,
        scope='live development from empty memory, then frozen reuse of this run only',
        episodes=['development_empty', 'reuse_frozen'], seed=0,
        credential_source=credential_source,
        source_library=None,
        required_source_hashes={n: sha(root / n) for n in REQUIRED_FILES
                                if n.endswith('.py')}))
    write(output / 'preserved_before.json', preserved_artifacts(output))
    command = [str(safe(root / '.venv-orchestrator/Scripts/python.exe')), '-B', '-u',
        str(safe(root / 'tools/run_skill_experiment.py')), '--run-dir', str(output),
        '--prepared-run', '--llm-config', str(output / 'approved_llm_config.json'),
        '--selector-mode', 'live', '--live-development', '--seed', '0', '--allow-live-api']
    print('Running two official beam episodes with one shared API budget. Logs: '
          + str(output), flush=True)
    result = recorder(command, root, safe(output / 'integration_command', True))
    print('PROCESS_EXIT=' + str(result['exit_code']), flush=True)
    print('Check result.json and reuse audit; exit 0 alone does not prove experience reuse.', flush=True)
    return result['exit_code'] if result['exit_code'] is not None else 127


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--authorize-six-calls', action='store_true')
    args = parser.parse_args()
    try:
        raise SystemExit(launch(args.authorize_six_calls))
    except Exception as error:
        print('LAUNCH_ERROR_TYPE=' + type(error).__name__, flush=True)
        raise SystemExit(1)
