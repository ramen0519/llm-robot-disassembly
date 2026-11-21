"""Read completed local artifacts; no provider or physics calls."""
import json
from pathlib import Path
import sqlite3
import sys
from run_luna_experiment import ROOT, safe, sha, read, write


def audit(run):
    run = safe(run, True)
    result = read(run / 'result.json')
    baseline = read(ROOT / 'runs/llm_langgraph_20261017/historical_baseline.json')
    checks = dict(result['checks'])
    original = read(run / 'original_sources.json')
    checks['original_82_sources_preserved'] = len(original) == 82 and all(sha(ROOT / n) == h for n, h in original.items())
    source_records = read(run / 'frozen_reuse/memory.json')['records']
    source_ids = {r['record_id'] for r in source_records}
    checks['reuse_memory_contains_only_first_episode'] = len(source_records) == 4 and all(r['source']['run_id'].endswith('/development_empty') for r in source_records)
    db = sqlite3.connect('file:' + (run / 'skill_memory.sqlite').as_posix() + '?mode=ro', uri=True)
    try:
        entries = {}
        for kind, key, payload in db.execute('SELECT kind,entry_id,payload FROM skill_entries'):
            entries.setdefault(kind, {})[key] = json.loads(payload)
        records = {key: json.loads(payload) for key, payload in db.execute('SELECT record_id,payload FROM records')}
        checks['all_8_actual_verifier_results_saved'] = len(records) == 8 and all(r['status'] == 'PASS' for r in records.values())
        checks['all_6_skill_invocations_logged'] = len(entries['invocation']) == len(entries['retrieval']) == len(entries['outcome']) == 6
        checks['three_provisional_reviewed_cards'] = len(entries['review']) == len(entries['registered_skill']) == 3 and all(c['lifecycle'] == 'provisional' for c in entries['registered_skill'].values())
        for row in result['rows']:
            name = row['variant']
            parsed = read(run / name / 'parsed_result.json')
            checks[name + '/matches_M1_setup_tree_terminal'] = all(parsed[k] == baseline[k] for k in ('setup', 'tree_fingerprint_sha256', 'terminal_parts'))
            checks[name + '/matches_M1_stats_except_time'] = {k: v for k, v in parsed['raw_stats'].items() if k != 'time'} == {k: v for k, v in baseline['raw_stats'].items() if k != 'time'}
            events = [json.loads(line) for line in (run / name / 'events.jsonl').read_text().splitlines()]
            decisions = [e for e in events if e['kind'] == 'rank_decision']
            checks[name + '/actual_code_and_worker_order_match'] = bool(decisions) and all(entries['invocation'][e['request']['request_id']]['applied_order'] == e['applied_order'] for e in decisions)
            report = read(run / name / 'memory_collection.json')
            checks[name + '/forced_terminal_not_attributed_to_skill'] = len(report['unassociated_evidence_ids']) == 1 and all(i in records and i not in entries['outcome'] for i in report['unassociated_evidence_ids'])
            for decision in decisions:
                rid = decision['request']['request_id']
                invocation, retrieval = entries['invocation'][rid], entries['retrieval'][rid]
                checks[rid + '/no_candidate_deletion'] = sorted(invocation['applied_order']) == sorted(decision['request']['candidate_ids'])
                checks[rid + '/explicit_fixture_not_live_model'] = invocation['selected_by'] == 'scripted_test'
                if name == 'reuse_frozen':
                    checks[rid + '/only_development_evidence_used'] = bool(retrieval['evidence']['evidence_ids']) and set(retrieval['evidence']['evidence_ids']).issubset(source_ids)
                    checks[rid + '/selected_registered_callable'] = invocation['skill_id'] in entries['registered_skill'] and invocation['callable_id'] == 'promote_verified'
        checks['new_episode_evidence_excluded_from_frozen_memory'] = len(set(records) - source_ids) == 4
    finally:
        db.close()
    checks['execution_exit_zero'] = read(run / 'execution_log/command.json')['exit_code'] == 0
    report = dict(status='PASS' if all(checks.values()) else 'ERROR', check_count=len(checks), checks=checks,
        actual_api_calls=0, original_result_files=result['preserved_file_count'], original_source_files=len(original),
        scope='artifact provenance, executable skills and physics; no model quality or generalization claim')
    write(run / 'completion_audit.json', report)
    print(json.dumps(dict(status=report['status'], checks=len(checks))))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(audit(Path(sys.argv[1])))
