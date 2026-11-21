"""Independently audit two completed live episodes; never call API or physics.

Integrity, real model selection, and demonstrated learned-skill reuse are
separate outcomes. A valid native choice is not evidence of learned-skill reuse.
Only the exclusive audit output is written; all SQLite connections are read-only.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from run_luna_experiment import ROOT, safe, sha, read, write
from run_memory_multiseed import preserved_artifacts


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in safe(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def unique(values, key):
    result = {value[key]: value for value in values}
    if len(result) != len(values):
        raise ValueError('duplicate_artifact_identity')
    return result


def actual_selection(logged, metadata, selected_skill):
    """A fallback or offline fixture can never count as a real model decision."""
    if not logged or logged.get('status') != 'completed' or not logged.get('response_json'):
        return False
    provider = json.loads(logged['response_json'])
    original = provider.get('metadata', {})
    return (logged.get('reserved_call') == 1 and metadata.get('attempted') is True
            and metadata.get('llm_live') is True and metadata.get('selected_by') == 'llm'
            and metadata.get('model_returned') == 'gpt-6-luna'
            and str(metadata.get('response_id', '')).startswith('resp_')
            and type(metadata.get('usage')) is dict and metadata['usage'].get('total_tokens', 0) > 0
            and metadata.get('offline_fixture') is not True
            and metadata.get('llm_fallback_reason') is None
            and metadata.get('skill_error_type') is None
            and original.get('selected_skill_id') == selected_skill
            and original.get('model_returned') == metadata['model_returned']
            and original.get('response_id') == metadata['response_id']
            and original.get('usage') == metadata['usage']
            and provider.get('fallback_reason') is None)


def card_lineage(card, first_records, first_run_id, invocations, outcomes):
    """Validate every source, not only a matching ID or a convenient PASS label."""
    if card.get('callable_id') != 'promote_verified' or card.get('lifecycle') != 'provisional':
        return False
    ids = card.get('source_evidence_ids', [])
    if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(first_records):
        return False
    invocation_id = card.get('source_invocation_id')
    invocation = invocations.get(invocation_id, {})
    request = invocation.get('request', {})
    scope = card.get('preconditions', {})
    if request.get('run_id') != first_run_id or scope.get('base_part') != request.get('base_part'):
        return False
    if scope.get('assembled_parts') != sorted(request.get('assembled_parts', [])):
        return False
    linked = [first_records[key] for key in ids]
    for record in linked:
        evidence_request = record.get('request', {})
        if (record.get('status') != 'PASS' or record.get('source', {}).get('run_id') != first_run_id
                or record.get('raw_result', {}).get('feasible') is not True
                or digest(record.get('context')) != scope.get('context_sha256')
                or sorted(evidence_request.get('parts_rest', []) + [evidence_request.get('part_move')]) != scope.get('assembled_parts')
                or outcomes.get(record.get('record_id'), {}).get('invocation_id') != invocation_id):
            return False
    return set(card.get('parameters', {}).get('preferred_ids', [])) == {
        record['request']['part_move'] for record in linked}


def reuse_demonstrated(detail):
    return any(item.get('variant') == 'reuse_frozen' and item.get('actual_luna_selection') is True
               and item.get('callable_id') == 'promote_verified'
               and item.get('first_episode_card_lineage') is True
               and item.get('execution_and_outcomes_valid') is True for item in detail)


def load_database(run):
    entries, records = {}, {}
    path = safe(run / 'skill_memory.sqlite')
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        for kind, key, payload, stored_hash in db.execute('SELECT kind,entry_id,payload,sha256 FROM skill_entries'):
            value = json.loads(payload)
            if digest(value) != stored_hash:
                raise ValueError('skill_entry_hash_mismatch')
            entries.setdefault(kind, {})[key] = value
        for key, payload, stored_hash in db.execute('SELECT record_id,payload,hash FROM records'):
            value = json.loads(payload)
            if value['record_id'] != key or digest(value) != stored_hash:
                raise ValueError('evidence_hash_mismatch')
            records[key] = value
    with sqlite3.connect(safe(run / 'batch_ledger.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.row_factory = sqlite3.Row
        ledger = {row['request_id']: dict(row) for row in db.execute('SELECT * FROM rank_requests')}
    return entries, records, ledger


def build_report(run):
    sys.path.insert(0, str(ROOT))
    from orchestrator.episode_archive import _collect_run
    from orchestrator.batch_budget import budget_summary
    from evidence_memory import FrozenMemory
    result = read(run / 'result.json')
    config = read(run / 'llm_config.json')
    context = read(run / 'context.json')
    entries, records, ledger = load_database(run)
    budget = budget_summary(run / 'batch_ledger.sqlite')
    baseline = read(ROOT / 'runs/llm_langgraph_20261017/historical_baseline.json')
    expected_jobs = ['development_empty', 'reuse_frozen']
    launcher = read(run / 'integration_command/command.json')
    execution = read(run / 'execution_start.json')
    checks = {'raw_runtime_pass_and_exit_zero': result['status'] == 'PASS' and launcher['exit_code'] == 0,
        'runtime_checks_all_pass': bool(result['checks']) and all(result['checks'].values()) and not result['errors'],
        'exact_two_live_jobs': execution['jobs'] == expected_jobs and result['selector_mode'] == 'live'
            and [row['variant'] for row in result['rows']] == expected_jobs,
        'approved_endpoint_model_budget': config['enabled'] is True and config['model'] == 'gpt-6-luna'
            and config['base_url'] == 'https://api.openai.com/v1' and config['max_calls'] == 6
            and Decimal(str(config['max_usd'])) == Decimal('.01'),
        'one_global_ledger_two_episodes': budget['run_count'] == 2 and budget['max_calls'] == 6
            and Decimal(budget['max_usd']) == Decimal('.01'),
        'budget_recomputed_from_ledger': result['budget'] == budget,
        'global_call_and_estimated_cost_limit': budget['reserved_calls'] <= 6
            and Decimal(budget['budget_committed_usd']) <= Decimal('.01'),
        'no_pending_or_unknown_call_outcomes': budget['unknown_outcomes'] == 0,
    }
    before, after = read(run / 'preserved_before.json'), read(run / 'preserved_after.json')
    current = preserved_artifacts(run)
    checks['preexisting_results_preserved_at_exit_and_audit'] = before == after and all(current.get(n) == h for n, h in before.items())
    source_before, source_after = read(run / 'source_before.json'), read(run / 'source_after.json')
    checks['source_and_frozen_implementation_preserved'] = source_before == source_after and all(
        sha(ROOT / n) == h == sha(run / 'implementation' / n) for n, h in source_before.items())
    snapshots = {}
    for folder in ('frozen_initial', 'frozen_reuse'):
        library, memory = read(run / folder / 'library.json'), read(run / folder / 'memory.json')
        snapshots[folder] = library, memory
        checks[folder + '/valid_content_hashes'] = (library['content_sha256'] == digest(library['cards'])
            and memory['content_sha256'] == digest(memory['records'])
            and memory['source_record_ids'] == [r['record_id'] for r in memory['records']]
            and memory['record_count'] == len(memory['records'])
            and memory['allow_hard_pruning'] is False and library['evaluation_updates_allowed'] is False)
        checks[folder + '/code_matches_frozen_implementation'] = library['source_code_sha256'] == sha(run / 'implementation/orchestrator/skill_library.py')
    first_library, empty_memory = snapshots['frozen_initial']
    reuse_library, reused_memory = snapshots['frozen_reuse']
    checks['first_episode_starts_without_prior_experience'] = empty_memory['records'] == []
    checks['first_episode_only_native_registered_skill'] = (len(first_library['cards']) == 1
        and first_library['cards'][0]['skill_id'] == 'native_order_v1'
        and first_library['cards'][0]['callable_id'] == 'native_order'
        and first_library['cards'][0]['source_evidence_ids'] == [])
    event_ids, run_ids, episode_records, traces = set(), set(), {}, {}
    for variant in expected_jobs:
        collected, source = _collect_run(run / variant, context, event_ids, run_ids)
        episode_records[variant] = unique(collected, 'record_id')
        traces[variant] = jsonl(run / variant / 'events.jsonl')
        checks[variant + '/records_reconstructed_from_actual_raw_physics'] = all(records.get(k) == v for k, v in episode_records[variant].items())
        checks[variant + '/source_collection_matches_trace'] = read(run / variant / 'memory_collection.json')['source'] == source
    first_records, second_records = episode_records['development_empty'], episode_records['reuse_frozen']
    first_run_id, second_run_id = (run.name + '/' + name for name in expected_jobs)
    checks['all_and_only_both_episodes_stored'] = set(records) == set(first_records) | set(second_records) and not set(first_records) & set(second_records)
    checks['frozen_reuse_contains_exact_first_new_episode'] = unique(reused_memory['records'], 'record_id') == first_records
    registered = entries.get('registered_skill', {})
    reviews = entries.get('review', {})
    invocations, outcomes = entries.get('invocation', {}), entries.get('outcome', {})
    cards = unique(reuse_library['cards'], 'skill_id')
    first_proposals = read(run / 'development_empty/memory_collection.json')['proposals']
    checks['only_first_episode_proposals_registered'] = set(registered) == set(reviews) == set(first_proposals) and bool(registered)
    checks['frozen_library_equals_offline_registration'] = (
        cards == {first_library['cards'][0]['skill_id']: first_library['cards'][0], **registered}
        and unique(read(run / 'offline_registration.json')['cards'], 'skill_id') == registered)
    for sid, card in registered.items():
        checks[sid + '/review_provenance'] = (card_lineage(card, first_records, first_run_id, invocations, outcomes)
            and reviews[sid]['proposal_sha256'] == digest(entries['proposal'][sid])
            and card == {**entries['proposal'][sid], 'lifecycle': 'provisional'}
            and reviews[sid]['human_review_claimed'] is False
            and reviews[sid]['generalization_validated'] is False)
    transition = read(run / 'freeze_transition.json')
    checks['freeze_transition_binds_new_experience_and_cards'] = (
        transition['source_run_id'] == first_run_id
        and transition['source_trace_sha256'] == sha(run / 'development_empty/events.jsonl')
        and set(transition['source_evidence_ids']) == set(first_records)
        and set(transition['registered_skill_ids']) == set(registered)
        and transition['frozen_hashes'] == {n: sha(run / 'frozen_reuse' / n) for n in ('library.json', 'memory.json')})
    phases = {name: [event['graph_event'] for event in jsonl(run / name / 'orchestrator_events.jsonl') if event['kind'] == 'graph_event'] for name in expected_jobs}
    first_end = [e['utc'] for e in phases['development_empty'] if e['phase'] == 'summarize' and e['kind'] == 'phase_after']
    second_start = [e['utc'] for e in phases['reuse_frozen'] if e['phase'] == 'validate' and e['kind'] == 'phase_before']
    checks['first_collection_then_freeze_then_second_validation'] = bool(first_end and second_start) and (
        datetime.fromisoformat(max(first_end)) <= datetime.fromisoformat(transition['utc']) <= datetime.fromisoformat(min(second_start)))
    detail, comparison, live_checks, all_decisions, associated_ids = [], [], {}, set(), set()
    for variant, folder in zip(expected_jobs, ('frozen_initial', 'frozen_reuse')):
        episode = run / variant
        library, memory = snapshots[folder]
        frozen_memory = FrozenMemory(safe(run / folder / 'memory.json'))
        offered_cards = unique(library['cards'], 'skill_id')
        decisions = [event for event in traces[variant] if event['kind'] == 'rank_decision']
        starts = unique([event for event in traces[variant] if event['kind'] == 'verifier_start'], 'call_id')
        responses = unique([read(p) for p in sorted(safe(episode / 'ipc/responses').glob('*.json'))], 'request_id')
        graph = read(episode / 'graph_state.json')
        checks[variant + '/frozen_inputs_bound_before_execution'] = graph['config']['input_hashes'] == {n: sha(run / folder / n) for n in ('library.json', 'memory.json')}
        checks[variant + '/all_decisions_have_responses'] = set(responses) == {d['request']['request_id'] for d in decisions}
        collection = read(episode / 'memory_collection.json')
        associated_ids.update(set(episode_records[variant]) - set(collection['unassociated_evidence_ids']))
        checks[variant + '/unassociated_terminal_not_claimed_as_skill'] = all(
            key in episode_records[variant] and key not in outcomes for key in collection['unassociated_evidence_ids'])
        parsed = read(episode / 'parsed_result.json')
        checks[variant + '/actual_worker_and_plan_pass'] = read(episode / 'execute/command.json')['exit_code'] == 0 and parsed['status'] == 'PASS' and parsed['plan_success'] is True
        checks[variant + '/same_official_baseline_setup'] = parsed['setup'] == baseline['setup'] == context['setup']
        comparison.append(dict(variant=variant, sequence=parsed['raw_stats'].get('sequence'),
            verifier_calls=len(episode_records[variant]), plan_status=parsed['status'], native_seconds=parsed['raw_stats'].get('time'),
            same_baseline_tree=parsed['tree_fingerprint_sha256'] == baseline['tree_fingerprint_sha256'],
            same_baseline_stats_except_time={k:v for k,v in parsed['raw_stats'].items() if k != 'time'} == {k:v for k,v in baseline['raw_stats'].items() if k != 'time'}))
        live_checks[variant + '/three_actual_selections'] = len(decisions) == 3
        for decision in decisions:
            request, rid = decision['request'], decision['request']['request_id']
            all_decisions.add(rid)
            invocation, retrieval, response = invocations[rid], entries['retrieval'][rid], responses[rid]
            metadata, logged = response['metadata'], ledger[rid]
            payload = json.loads(logged['payload_json'])
            card = offered_cards[invocation['skill_id']]
            prefix = rid + '/'
            evidence_ids = retrieval['evidence']['evidence_ids']
            checks[prefix + 'same_request_state_and_library'] = (invocation['request'] == retrieval['request'] == request
                and response['state_id'] == request['state_id']
                and all(payload['request'].get(key) == value for key, value in request.items())
                and invocation['library_sha256'] == retrieval['library_sha256'] == payload['request']['library_sha256'] == library['content_sha256']
                and retrieval['memory_sha256'] == memory['content_sha256'])
            checks[prefix + 'actual_prompt_contains_frozen_retrieval'] = (
                payload['features']['evidence_memory'] == retrieval['evidence']
                and retrieval['evidence'] == frozen_memory.lookup(context, tuple(request['assembled_parts']), request['base_part'])
                and [c['skill_id'] for c in payload['features']['registered_skills']] == retrieval['offered_skill_ids']
                and all(c == offered_cards[c['skill_id']] for c in payload['features']['registered_skills'])
                and (evidence_ids == [] if variant == 'development_empty' else bool(evidence_ids) and set(evidence_ids).issubset(first_records)))
            linked = [r for r in episode_records[variant].values() if outcomes.get(r['record_id'], {}).get('invocation_id') == rid]
            expected_order = list(request['candidate_ids'])
            if card['callable_id'] == 'promote_verified':
                preferred = card['parameters']['preferred_ids']
                expected_order = [p for p in preferred if p in expected_order] + [p for p in expected_order if p not in preferred]
            execution_valid = (invocation['skill_id'] in retrieval['offered_skill_ids']
                and invocation['callable_id'] == card['callable_id'] and invocation['code_sha256'] == card['code_sha256']
                and invocation['execution_error_type'] is None and metadata['skill_applied'] is True
                and invocation['source_evidence_ids'] == card['source_evidence_ids']
                and invocation['applied_order'] == response['candidate_ids'] == decision['applied_order'] == expected_order
                and bool(linked) and all(starts[r['source']['call_id']]['generation_id'] == decision['generation_id']
                    and outcomes[r['record_id']]['skill_id'] == invocation['skill_id']
                    and outcomes[r['record_id']]['raw_source'] == r['source']
                    and outcomes[r['record_id']]['status'] == r['status'] for r in linked))
            checks[prefix + 'registered_execution_and_actual_outcome_lineage'] = execution_valid
            checks[prefix + 'full_permutation_no_state_mutation_or_verifier_bypass'] = (
                sorted(decision['applied_order']) == sorted(request['candidate_ids'])
                and invocation['state_mutations'] is False and invocation['internal_verifier_calls'] == 0
                and metadata['candidate_pruning'] is False and metadata['physics_bypass'] is False)
            live = actual_selection(logged, metadata, invocation['skill_id'])
            live_checks[prefix + 'actual_luna_selection_not_fallback'] = live
            lineage = card_lineage(card, first_records, first_run_id, invocations, outcomes)
            if card['callable_id'] == 'promote_verified':
                checks[prefix + 'selected_card_only_first_new_episode_evidence'] = lineage and set(card['source_evidence_ids']).issubset(evidence_ids)
                checks[prefix + 'selected_card_matches_current_state_and_context'] = (
                    card['preconditions']['context_sha256'] == digest(context)
                    and card['preconditions']['assembled_parts'] == sorted(request['assembled_parts'])
                    and card['preconditions']['base_part'] == request['base_part']
                    and set(card['parameters']['preferred_ids']).issubset(request['candidate_ids']))
            detail.append(dict(variant=variant, request_id=rid, response_id=metadata.get('response_id'),
                model_returned=metadata.get('model_returned'), usage=metadata.get('usage'),
                selected_skill_id=invocation['skill_id'], callable_id=card['callable_id'],
                retrieved_evidence_ids=evidence_ids, selected_card_evidence_ids=card['source_evidence_ids'],
                source_invocation_id=card['source_invocation_id'], applied_order=decision['applied_order'],
                actual_luna_selection=live, first_episode_card_lineage=lineage,
                execution_and_outcomes_valid=execution_valid, fallback_reason=metadata.get('llm_fallback_reason'),
                linked_result_ids=[r['record_id'] for r in linked]))
    checks['exactly_one_ledger_entry_per_both_episode_decisions'] = set(ledger) == all_decisions
    checks['no_duplicate_provider_call_or_response_ids'] = (
        len({row['call_id'] for row in ledger.values() if row['reserved_call']}) == budget['reserved_calls']
        and len({d['response_id'] for d in detail if d['actual_luna_selection']}) == sum(d['actual_luna_selection'] for d in detail))
    checks['all_invocations_and_retrievals_belong_to_new_episodes'] = set(invocations) == set(entries.get('retrieval', {})) == all_decisions
    checks['all_and_only_associated_actual_results_have_outcomes'] = set(outcomes) == associated_ids
    live_checks['six_calls_with_usage_and_no_extra_calls'] = budget['reserved_calls'] == budget['usage_reported_calls'] == len(detail) == 6
    integrity, live = all(checks.values()), all(live_checks.values())
    demonstrated = integrity and live and reuse_demonstrated(detail)
    return dict(utc=datetime.now(timezone.utc).isoformat(), status='ERROR' if not integrity else 'PASS' if demonstrated else 'NOT_DEMONSTRATED',
        integrity_status='PASS' if integrity else 'ERROR', live_selection_status='PASS' if live else 'INCOMPLETE_OR_FAILED',
        learned_skill_reuse_status='DEMONSTRATED' if demonstrated else 'NOT_DEMONSTRATED',
        raw_runtime_status=result['status'], raw_launcher_exit=launcher['exit_code'],
        checks=checks, live_checks=live_checks, check_count=len(checks) + len(live_checks),
        detail=detail, comparison=comparison, budget=budget, previous_result_files=len(before),
        first_episode_records=len(first_records), second_episode_records=len(second_records),
        learned_skill_selections=sum(d['variant'] == 'reuse_frozen' and d['actual_luna_selection'] and d['callable_id'] == 'promote_verified' for d in detail),
        memory_causal_effect='UNVERIFIED', generalization='NOT_RUN', billed_cost='UNVERIFIED',
        scope='Known beam revisit: trace-linked experience reuse, not model learning or performance improvement')


def audit(run):
    run = safe(run, True)
    if run.parent != ROOT / 'runs':
        raise ValueError('direct_child_run_required')
    output = safe(run / 'live_skill_reuse_audit.json')
    if output.exists():
        raise FileExistsError('audit_already_exists')
    if not (run / 'result.json').is_file() or read(run / 'integration_command/command.json')['exit_code'] is None:
        print('NOT_READY: no completed two-episode result; no audit output created.')
        return 2
    try:
        report = build_report(run)
    except (KeyError, ValueError, TypeError, OSError, sqlite3.Error) as error:
        report = dict(status='ERROR', integrity_status='ERROR', error_type=type(error).__name__,
            scope='Incomplete or inconsistent artifacts; no live reuse success claimed')
    write(output, report)
    print(json.dumps({key: report.get(key) for key in ('status', 'integrity_status', 'live_selection_status',
        'learned_skill_reuse_status', 'check_count', 'learned_skill_selections', 'budget')}))
    return 0 if report['status'] == 'PASS' else 2 if report['status'] == 'NOT_DEMONSTRATED' else 1


if __name__ == '__main__':
    raise SystemExit(audit(Path(sys.argv[1])))
