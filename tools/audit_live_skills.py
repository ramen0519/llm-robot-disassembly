"""Audit a completed real skill-selection smoke run without new API/physics.

Separates live selection, skill execution, evidence linkage, and baseline
comparison. Supplying memory does not establish its causal influence on an LLM.
"""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import sys
from run_luna_experiment import ROOT, safe, sha, read, write
from run_memory_multiseed import preserved_artifacts


def jsonl(path):
    return [json.loads(line) for line in safe(path).read_text(encoding='utf-8').splitlines()]


def audit(run):
    run = safe(run, True)
    if not (run / 'result.json').is_file() or read(run / 'integration_command/command.json')['exit_code'] is None:
        print('NOT_READY: execution is absent or incomplete; no audit output created.')
        return 2
    result = read(run / 'result.json')
    if len(result.get('rows', [])) != 1:
        raise ValueError('expected exactly one frozen evaluation')
    row = result['rows'][0]
    episode = run / row['variant']
    parsed = read(episode / 'parsed_result.json')
    baseline = read(ROOT / 'runs/llm_langgraph_20261017/historical_baseline.json')
    native = read(ROOT / 'runs/skill_loop_20261017/development_empty/row.json')
    config = read(run / 'llm_config.json')
    events = jsonl(episode / 'events.jsonl')
    decisions = [e for e in events if e['kind'] == 'rank_decision']
    starts = {e['call_id']: e for e in events if e['kind'] == 'verifier_start'}
    verifiers = {e['call_id']: e for e in events if e['kind'] == 'verifier_result'}
    responses = {v['request_id']: v for p in sorted((episode / 'ipc/responses').glob('*.json')) for v in [read(p)]}
    library = read(run / 'frozen_initial/library.json')
    cards = {c['skill_id']: c for c in library['cards']}
    development_records = read(run / 'frozen_initial/memory.json')['records']
    development_ids = {r['record_id'] for r in development_records}
    with sqlite3.connect((run / 'skill_memory.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        entries = {}
        for kind, key, payload in db.execute('SELECT kind,entry_id,payload FROM skill_entries'):
            entries.setdefault(kind, {})[key] = json.loads(payload)
        records = {key: json.loads(payload) for key, payload in db.execute('SELECT record_id,payload FROM records')}
    with sqlite3.connect((run / 'batch_ledger.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.row_factory = sqlite3.Row
        ledger = {r['request_id']: dict(r) for r in db.execute('SELECT * FROM rank_requests')}
    raw_checks = dict(result['checks'])
    checks = {k: v for k, v in raw_checks.items() if k != 'prior_results_preserved'}
    launcher_exit = read(run / 'integration_command/command.json')['exit_code']
    checks['launcher_exit_matches_raw_runtime_status'] = launcher_exit == (0 if result['status'] == 'PASS' else 1)
    checks['live_mode_requested'] = result['selector_mode'] == 'live'
    checks['approved_endpoint_model_and_budget'] = (config['enabled'] is True and config['model'] == 'gpt-6-luna'
        and config['base_url'] == 'https://api.openai.com/v1' and config['max_calls'] == 3 and config['max_usd'] == .01)
    budget = result['budget']
    checks['call_and_estimated_cost_limits_respected'] = (budget['reserved_calls'] <= 3
        and Decimal(budget['budget_committed_usd']) <= Decimal('.01'))
    checks['no_unknown_or_pending_calls'] = budget['unknown_outcomes'] == 0
    checks['one_ledger_entry_per_rank_request'] = len(ledger) == len(decisions) == len(responses)
    checks['no_duplicate_call_ids'] = len({r['call_id'] for r in ledger.values() if r['reserved_call']}) == budget['reserved_calls']
    checks['every_verifier_result_stored'] = len(records) == len(verifiers) and {
        r['source']['call_id'] for r in records.values()} == set(verifiers)
    checks['evaluation_excluded_from_frozen_development'] = not (set(records) & development_ids)
    collection = read(episode / 'memory_collection.json')
    checks['no_new_evaluation_skill_registration'] = not entries.get('registered_skill') and not entries.get('review')
    checks['unassociated_terminal_not_attributed_to_skill'] = all(
        rid in records and rid not in entries.get('outcome', {}) for rid in collection['unassociated_evidence_ids'])
    before = read(run / 'preserved_before.json')
    after = read(run / 'preserved_after.json')
    current = preserved_artifacts(run)
    diff = dict(added=sorted(set(after) - set(before)), removed=sorted(set(before) - set(after)),
        modified=sorted(n for n in set(before) & set(after) if before[n] != after[n]))
    checks['all_preexisting_files_preserved_at_exit_and_audit'] = not diff['removed'] and not diff['modified'] and all(current.get(n) == v for n, v in before.items())
    # This smoke run overlapped the separately authorized key-storage tests.
    # Their three NEW logs changed inventory equality; no prior file changed.
    allowed_new_logs = {'runs/persistent_key_setup_20261017/tests/' + n for n in ('command.json', 'stdout.log', 'stderr.log')}
    checks['inventory_difference_is_only_concurrent_key_test_logs'] = set(diff['added']).issubset(allowed_new_logs)
    raw_failures = [k for k, v in raw_checks.items() if not v]
    reconciled = (raw_failures == ['prior_results_preserved'] and diff['added']
        and all(checks[k] for k in ('all_preexisting_files_preserved_at_exit_and_audit', 'inventory_difference_is_only_concurrent_key_test_logs'))
        and not result['errors'] and launcher_exit == 1)
    checks['frozen_runtime_matches_executed_source'] = all(
        sha(run / 'implementation' / n) == h for n, h in read(run / 'source_before.json').items())
    checks['actual_code_matches_library'] = sha(run / 'implementation/orchestrator/skill_library.py') == library['source_code_sha256']
    live_checks, detail = {}, []
    for d in decisions:
        request = d['request']
        rid = request['request_id']
        response = responses[rid]
        metadata = response['metadata']
        invocation = entries['invocation'][rid]
        retrieval = entries['retrieval'][rid]
        logged = ledger[rid]
        payload = json.loads(logged['payload_json'])
        provider = json.loads(logged['response_json'])
        selected = provider['metadata'].get('selected_skill_id')
        sid = invocation['skill_id']
        card = cards[sid]
        prefix = rid + '/'
        checks[prefix + 'same_request_state_library'] = (response['state_id'] == request['state_id']
            and invocation['request'] == request and payload['request']['library_sha256'] == library['content_sha256'])
        checks[prefix + 'prompt_contained_frozen_applicable_skills_and_memory'] = (
            [c['skill_id'] for c in payload['features']['registered_skills']] == retrieval['offered_skill_ids']
            and all(c == cards[c['skill_id']] for c in payload['features']['registered_skills'])
            and payload['features']['evidence_memory'] == retrieval['evidence']
            and bool(retrieval['evidence']['evidence_ids'])
            and set(retrieval['evidence']['evidence_ids']).issubset(development_ids))
        checks[prefix + 'selected_registered_function_executed'] = (sid in retrieval['offered_skill_ids']
            and invocation['callable_id'] == card['callable_id'] and invocation['code_sha256'] == card['code_sha256']
            and invocation['execution_error_type'] is None and metadata['skill_applied'] is True
            and invocation['applied_order'] == response['candidate_ids'] == d['applied_order'])
        checks[prefix + 'complete_permutation_no_state_mutation_or_physics_bypass'] = (
            sorted(d['applied_order']) == sorted(request['candidate_ids'])
            and invocation['state_mutations'] is False and invocation['internal_verifier_calls'] == 0
            and metadata['candidate_pruning'] is False and metadata['physics_bypass'] is False)
        expected = list(request['candidate_ids'])
        if card['callable_id'] == 'promote_verified':
            preferred = card['parameters']['preferred_ids']
            expected = [p for p in preferred if p in expected] + [p for p in expected if p not in preferred]
        checks[prefix + 'applied_order_matches_registered_procedure'] = expected == d['applied_order']
        linked = [r for r in records.values() if entries.get('outcome', {}).get(r['record_id'], {}).get('invocation_id') == rid]
        checks[prefix + 'outcomes_link_actual_physics'] = bool(linked) and all(
            starts[r['source']['call_id']]['generation_id'] == d['generation_id']
            and verifiers[r['source']['call_id']]['status'] == r['status']
            and entries['outcome'][r['record_id']]['skill_id'] == sid for r in linked)
        live_checks[prefix + 'actual_luna_successful_selection'] = (
            logged['reserved_call'] == 1 and logged['status'] == 'completed'
            and metadata['attempted'] is True and metadata['llm_live'] is True
            and metadata.get('model_returned') == 'gpt-6-luna'
            and str(metadata.get('response_id', '')).startswith('resp_')
            and type(metadata.get('usage')) is dict and metadata['usage']['total_tokens'] > 0
            and not metadata.get('offline_fixture') and metadata['selected_by'] == 'llm'
            and selected == sid and metadata['llm_fallback_reason'] is None)
        detail.append(dict(request_id=rid, state_id=request['state_id'], response_id=metadata.get('response_id'),
            model_returned=metadata.get('model_returned'), usage=metadata.get('usage'),
            retrieved_evidence_ids=retrieval['evidence']['evidence_ids'], offered_skill_ids=retrieval['offered_skill_ids'],
            selected_skill_id=sid, callable_id=invocation['callable_id'], applied_order=d['applied_order'],
            selected_by=metadata['selected_by'], fallback_reason=metadata['llm_fallback_reason'],
            linked_verifier_results=[dict(call_id=r['source']['call_id'], status=r['status']) for r in linked]))
    live_checks['three_real_selections'] = len(detail) == budget['reserved_calls'] == budget['usage_reported_calls'] == 3
    comparison = dict(plan_pass=parsed['status'] == 'PASS' and parsed['plan_success'] is True,
        matches_M1_setup_tree_terminal=all(parsed[k] == baseline[k] for k in ('setup', 'tree_fingerprint_sha256', 'terminal_parts')),
        matches_M1_stats_except_time={k:v for k,v in parsed['raw_stats'].items() if k != 'time'} == {k:v for k,v in baseline['raw_stats'].items() if k != 'time'},
        verifier_calls=len(verifiers), original_native_verifier_calls=native['verifier_calls'],
        sequence=parsed['raw_stats'].get('sequence'), process_seconds=row['process_seconds'],
        native_seconds=parsed['raw_stats'].get('time'))
    passed = all(checks.values()) and all(live_checks.values()) and comparison['plan_pass']
    report = dict(utc=datetime.now(timezone.utc).isoformat(),
        integrity_status='PASS' if all(checks.values()) else 'ERROR',
        live_selection_status='PASS' if all(live_checks.values()) else 'INCOMPLETE_OR_FAILED',
        status=('PASS_WITH_RUNTIME_CHECK_EXCEPTION' if reconciled else 'PASS') if passed else 'ERROR',
        raw_runtime_status=result['status'], raw_launcher_exit=launcher_exit,
        raw_failed_checks=raw_failures, preservation_diff=diff, runtime_error_reconciled=bool(reconciled),
        checks=checks, live_checks=live_checks, check_count=len(checks)+len(live_checks),
        detail=detail, comparison=comparison, budget=budget,
        learned_skill_selections=sum(d['callable_id'] == 'promote_verified' for d in detail),
        current_evidence_records=len(records), linked_outcomes=len(entries.get('outcome', {})),
        unassociated_records=len(collection['unassociated_evidence_ids']),
        prior_result_files=result['preserved_file_count'],
        memory_causal_effect='UNVERIFIED', generalization='NOT_RUN', billed_cost='UNVERIFIED')
    write(run / 'live_skill_audit.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'integrity_status', 'live_selection_status', 'check_count', 'learned_skill_selections', 'budget', 'comparison')}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(audit(Path(sys.argv[1])))
