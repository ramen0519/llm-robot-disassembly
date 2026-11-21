"""Recheck request-dependent counts without rerunning or changing raw experiments."""
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/dev/disassembly_codex_ready_v2')
RUN = ROOT / 'runs/llm_langgraph_20261017'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    original = read(RUN / 'comparison.json')
    failed = [key for key, value in original['checks'].items() if not value]
    allowed = ['remote_reverse/remote_roundtrip_used', 'remote_reverse/expected_order_change']
    if failed != allowed or original['status'] != 'ERROR':
        raise ValueError('unexpected original comparison outcome')
    result = json.loads(json.dumps(original))
    for variant in ('remote_noop', 'remote_disabled', 'remote_reverse'):
        folder = RUN / variant
        events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
        decisions = [item for item in events if item['kind'] == 'rank_decision']
        generations = [item for item in events if item['kind'] == 'generation']
        requests = [read(path) for path in sorted((folder / 'ipc/requests').glob('*.json'))]
        responses = [read(path) for path in sorted((folder / 'ipc/responses').glob('*.json'))]
        audits = [json.loads(line) for line in (folder / 'remote_audit.jsonl').read_text().splitlines()]
        ids = {item['request']['request_id'] for item in decisions}
        count = len(decisions)
        result['checks'][variant + '/remote_roundtrip_used'] = (
            count == len(ids) == len(requests) == len(responses) == len(audits) == len(generations) > 0
            and ids == {item['request']['request_id'] for item in requests}
            == {item['request_id'] for item in responses} == {item['request_id'] for item in audits})
        result['checks'][variant + '/expected_order_change'] = bool(generations) and all(
            item['applied_order'] == (list(reversed(item['native_order']))
                if variant == 'remote_reverse' else item['native_order']) for item in generations)
        by_id = {item['request_id']: item for item in responses}
        result['checks'][variant + '/response_matches_each_requested_order'] = bool(requests) and all(
            by_id[item['request']['request_id']]['state_id'] == item['request']['state_id'] and
            by_id[item['request']['request_id']]['candidate_ids'] == (
                list(reversed(item['request']['candidate_ids'])) if variant == 'remote_reverse'
                else item['request']['candidate_ids']) for item in requests)
        result['checks'][variant + '/worker_applied_service_order'] = bool(decisions) and all(
            item['applied_order'] == by_id[item['request']['request_id']]['candidate_ids'] for item in decisions)
    source = 'tools/run_langgraph_integration.py'
    before = read(RUN / 'source_before.json')
    after = read(RUN / 'source_after.json')
    if before != after:
        raise ValueError('source changed during experiment')
    amendment = dict(reason='fixed count 3 was invalid for reverse traversal with repeated generation; check actual IDs and exact permutations',
        original_failed_checks=failed, physics_rerun=False, original_comparison_sha256=sha(RUN / 'comparison.json'),
        root_changes={source: {'old_sha256': before[source], 'new_sha256': sha(ROOT / source)}},
        recheck_source_sha256=sha(Path(__file__)), preserved_frozen_sources=True)
    result.update(status='PASS' if all(result['checks'].values()) else 'ERROR',
                  original_status=original['status'], original_run_exit_code=read(RUN / 'integration_command/command.json')['exit_code'],
                  physics_rerun=False, posthoc_correction=amendment['reason'])
    write(RUN / 'postrun_amendment.json', amendment)
    write(RUN / 'comparison_v2.json', result)
    print(json.dumps({'status': result['status'], 'passed': sum(result['checks'].values()),
                      'total': len(result['checks']), 'physics_rerun': False}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
