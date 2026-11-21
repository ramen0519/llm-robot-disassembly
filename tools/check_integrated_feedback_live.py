"""Two independent live decisions over saved actual evidence, without physics.

--prepare-only validates inputs, hashes and provider bodies without accessing a
key or sending an API request. --allow-live-api permits at most TWO decisions
under the existing, immutable shared batch policy. This does not resume DFS,
execute the newly selected path, or validate the newly proposed permutation.
"""
import argparse
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from run_integrated_repair import checked, write_jsonl, source_hashes
from run_luna_experiment import CONFIG_KEYS, read, sha, write, validate_config
from orchestrator.batch_budget import budget_summary
from orchestrator.integrated_path_ranker import IntegratedPathPipeline, validate_worker_envelope
from orchestrator.path_repair_ranker import PROTOCOL as PATH_PROTOCOL
from orchestrator.repair_library import feedback_from_events
from worker.constraint_memory import FrozenConstraintMemory, load_snapshot


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--source-run', required=True, type=Path)
    value.add_argument('--run-dir', required=True, type=Path)
    value.add_argument('--llm-config', required=True, type=Path)
    value.add_argument('--ledger', required=True, type=Path)
    value.add_argument('--memory', type=Path)
    value.add_argument('--skill-memory', type=Path)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--allow-live-api', action='store_true')
    return value


def _jsonlines(path):
    return [json.loads(line) for line in checked(path, True).read_text(encoding='utf-8').splitlines() if line.strip()]


def _hashes(paths):
    return {str(checked(path)): sha(checked(path)) for path in sorted(set(paths), key=str)}


def build_requests(source, namespace):
    """Construct diagnostic requests from real artifacts, never fake outcomes."""
    source = checked(source, True)
    files = sorted(checked(source / 'ipc/requests', True).glob('*.json'))
    requests = [(path, read(path)) for path in files]
    path_requests = [(p, r) for p, r in requests if r['features'].get('protocol') == PATH_PROTOCOL]
    if len(path_requests) != 1:
        raise ValueError('source_requires_one_saved_path_request')
    path_file, original_path = path_requests[0]
    state = original_path['request']['assembled_parts']
    candidates = [(p, r) for p, r in requests if r['features'].get('protocol') != PATH_PROTOCOL
                  and r['request']['assembled_parts'] == state and r['features'].get('episode_feedback') == []]
    if len(candidates) != 1:
        raise ValueError('source_requires_unique_initial_candidate_at_path_state')
    candidate_file, original_candidate = candidates[0]
    result_file = checked(source / 'result.json', True)
    source_result = read(result_file)
    if (source_result.get('verifier_backend') != 'actual_ASAP_RedMax'
            or source_result.get('model_decision_fixture') is not True
            or source_result.get('controlled_initial_probe') is not False):
        raise ValueError('source_must_be_labelled_actual_physics_with_uncontrolled_fixture_selection')
    feedback_file = checked(source / 'integrated_feedback.json', True)
    completed = read(feedback_file)['completed_path_feedback']
    if (len(completed) != 1 or completed[0]['outcome']['path_success'] is not False
            or completed[0]['outcome']['native_six_axis_failure_recovery'] is not True
            or completed[0]['outcome']['scope']['assembled_parts'] != state
            or completed[0]['outcome']['request_id'] != original_path['request']['request_id']):
        raise ValueError('source_requires_matching_completed_natural_path_failure')
    events_file = checked(source / 'events.jsonl', True)
    episode = feedback_from_events(_jsonlines(events_file))
    if not episode or len(episode) > 12:
        raise ValueError('source_feedback_must_be_nonempty_and_fit_untruncated_protocol')
    # We synthesize a request at the observed unchanged state, not a new DFS
    # generation. Do not silently roll back actual successful transitions.
    if any(event['status'] == 'PASS' for event in episode):
        raise ValueError('diagnostic_initial_state_requires_no_successful_removal')
    memory_file = checked(source / 'constraint_memory.json', True)
    memory = FrozenConstraintMemory(load_snapshot(memory_file)['records'])
    path_request, candidate_request = deepcopy(original_path), deepcopy(original_candidate)
    candidate_request['features']['episode_feedback'] = episode
    candidate_request['features']['path_feedback'] = deepcopy(completed)
    refs = memory.retrieve(candidate_request['features']['memory_context'], state,
        candidate_request['request']['candidate_ids'], max_records=4)['records']
    candidate_request['features']['constraint_history'] = [dict(r, origin='current_episode_history') for r in refs]
    lineage = []
    for route, request, original, file in (
        ('path', path_request, original_path, path_file),
        ('candidate', candidate_request, original_candidate, candidate_file)):
        original_id = request['request']['request_id']
        request['request'].update(run_id=namespace, request_id=namespace + ':' + route)
        validate_worker_envelope(request)
        lineage.append({'route': route, 'source_request_file': str(file),
            'source_request_id': original_id, 'source_run_id': original['request']['run_id'],
            'diagnostic_request_id': request['request']['request_id'],
            'original_state_id': original['request']['state_id'],
            'state_id_preserved': request['request']['state_id'] == original['request']['state_id'],
            'diagnostic_candidate_request_generated': route == 'candidate',
            'historical_event_ids_renamed': False,
            'feedback_files': [str(feedback_file), str(events_file), str(memory_file)] if route == 'candidate' else [],
            'solver_not_resumed': True, 'path_response_caused_stored_outcome': False})
    files += [result_file, feedback_file, events_file, memory_file]
    return [path_request, candidate_request], lineage, files


def _policy(config):
    if (type(config) is not dict or set(config) != CONFIG_KEYS
            or config.get('enabled') is not True or config.get('model') != 'gpt-6-luna'
            or config.get('base_url') != 'https://api.openai.com/v1'
            or config.get('api_key_env') != 'OPENAI_API_KEY'
            or type(config.get('max_calls')) is not int or config['max_calls'] != 12
            or Decimal(str(config.get('max_usd'))) != Decimal('.05')):
        raise ValueError('diagnostic_requires_existing_pinned_12_call_005_USD_batch_policy')


def execute(args, transport=None, key_context_factory=None):
    """The injectable seams are solely for offline tests; CLI exposes neither."""
    output, source = checked(args.run_dir, True), checked(args.source_run, True)
    config_file, ledger = checked(args.llm_config, True), checked(args.ledger, True)
    memory = checked(args.memory, True) if args.memory is not None else None
    skill_memory = checked(args.skill_memory, True) if args.skill_memory is not None else None
    if bool(args.prepare_only) == bool(args.allow_live_api):
        raise ValueError('choose_exactly_one_prepare_or_authorized_live_mode')
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('diagnostic_output_must_be_separate_from_source_run')
    config = read(config_file)
    _policy(config)
    namespace = 'saved-evidence-diagnostic:' + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    requests, lineage, inputs = build_requests(source, namespace)
    inputs += [config_file, Path(__file__)]
    inputs += [ROOT / name for name in source_hashes()]
    for path in (memory, skill_memory):
        if path is not None:
            inputs.append(path)
    if skill_memory is not None:
        # The production loader additionally validates these source hashes.
        snapshot = read(skill_memory)
        inputs.extend(checked(item['path'], True) for record in snapshot['records']
                      for item in record['source_files'].values())
    before = _hashes(inputs)
    budget_before = budget_summary(ledger)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'source_before.json', before)
    write(output / 'budget_before.json', budget_before)
    write(output / 'lineage.json', {'schema_version': 1, 'requests': lineage,
        'requests_independent': True, 'diagnostic_only': True, 'full_DFS': False,
        'solver_not_resumed': True, 'source_model_decision_fixture': True,
        'source_physics_backend': 'actual_ASAP_RedMax'})
    write(output / 'config.json', {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'run_id': namespace, 'api_config': config, 'ledger': str(ledger),
        'maximum_new_requests': 2, 'automatic_retries': 0, 'diagnostic_only': True, 'full_DFS': False,
        'selected_path_executions': 0, 'candidate_orders_consumed_by_DFS': 0, 'new_physics_calls': 0,
        'solver_not_resumed': True, 'source_model_decision_fixture': True,
        'source_physics_backend': 'actual_ASAP_RedMax'})
    write_jsonl(output / 'requests.jsonl', requests)
    # Never open the shared live ledger for write in --prepare-only mode.
    prepare = IntegratedPathPipeline('llm', config, output / 'preflight_only.sqlite',
        memory_path=memory, skill_memory_path=skill_memory, transport=lambda *a: (_ for _ in ()).throw(AssertionError('prepare cannot send')))
    enriched, previews, estimates = [], [], []
    reason, rates = prepare.service._configuration()
    if reason:
        raise ValueError('invalid_preflight_configuration')
    for request in requests:
        value = prepare.enrich(request)
        body = prepare.service._body(value)
        enriched.append(value)
        estimates.append(prepare.service._estimate(body, rates))
    previews = deepcopy(prepare.previews)
    write_jsonl(output / 'prepared_requests.jsonl', enriched)
    write_jsonl(output / 'preflight_body_previews.jsonl', previews)
    required_cost = sum((Decimal(item['usd']) for item in estimates), Decimal(0))
    eligible = (budget_before['policy_fingerprint'] == prepare.service.config_fingerprint
        and budget_before['unknown_outcomes'] == 0
        and budget_before['reserved_calls'] + 2 <= config['max_calls']
        and Decimal(budget_before['budget_committed_usd']) + required_cost <= Decimal(str(config['max_usd'])))
    write(output / 'preflight.json', {'schema_version': 1, 'requests': 2, 'estimates': estimates,
        'estimated_additional_reservation_usd': str(required_cost), 'within_remaining_batch_budget': eligible,
        'key_accessed': False, 'API_calls': 0, 'billing_cap_guaranteed': False})
    responses, decisions, actual_previews, error_type = [], [], [], None
    pipeline = None
    try:
        if before != _hashes(inputs):
            raise ValueError('source_changed_before_execution')
        if not eligible:
            raise ValueError('preflight_batch_policy_or_remaining_budget_rejected')
        if args.allow_live_api:
            if key_context_factory is None:
                from local_api_key import key_environment
                key_context_factory = key_environment
            with key_context_factory():
                validate_config(config, True)
                pipeline = IntegratedPathPipeline('llm', config, ledger,
                    memory_path=memory, skill_memory_path=skill_memory, transport=transport)
                for request in requests:  # Exactly two, independent requests; never retry.
                    if before != _hashes(inputs):
                        raise ValueError('source_changed_before_request')
                    response = pipeline.rank(request)
                    responses.append(deepcopy(response))
                    write(output / ('response_' + str(len(responses)) + '.json'), response)
    except Exception as error:
        # No provider exception messages or key material enter diagnostics.
        error_type = type(error).__name__
    finally:
        if pipeline is not None:
            decisions, actual_previews = pipeline.decisions, pipeline.previews
            pipeline.close()
        prepare.close()
        write_jsonl(output / 'responses.jsonl', responses)
        write_jsonl(output / 'decisions.jsonl', decisions)
        write_jsonl(output / 'request_body_previews.jsonl', actual_previews if args.allow_live_api else previews)
        after = _hashes(inputs)
        budget_after = budget_summary(ledger)
        write(output / 'source_after.json', after)
        write(output / 'budget_after.json', budget_after)
        valid = sum(response['fallback_reason'] is None and response['metadata'].get('llm_live') is True
                    and response['metadata'].get('attempted') is True for response in responses)
        attempted = sum(response['metadata'].get('attempted') is True and response['metadata'].get('llm_live') is True
                        for response in responses)
        status = ('ERROR' if error_type or before != after else 'PREPARED' if args.prepare_only
                  else 'PASS' if valid == 2 else 'PARTIAL')
        result = {'kind': 'saved_evidence_live_contract_diagnostic', 'run_id': namespace,
            'status': status, 'error_type': error_type, 'diagnostic_only': True, 'full_DFS': False,
            'response_count': len(responses), 'new_api_attempts': attempted, 'live_valid_responses': valid,
            'transport_attempts_including_injected_tests': sum(r['metadata'].get('attempted') is True for r in responses),
            'selected_path_executions': 0, 'candidate_orders_consumed_by_DFS': 0, 'new_physics_calls': 0,
            'solver_not_resumed': True, 'requests_independent': True, 'source_model_decision_fixture': True,
            'source_physics_backend': 'actual_ASAP_RedMax', 'source_preserved': before == after,
            'diagnostic_candidate_request_generated': True, 'path_response_caused_stored_outcome': False,
            'budget_delta_calls': budget_after['reserved_calls'] - budget_before['reserved_calls'],
            'billing_cap_guaranteed': False}
        write(output / 'result.json', result)
    return result


def main(argv=None):
    try:
        result = execute(parser().parse_args(argv))
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result['status'] in ('PASS', 'PREPARED') else 1
    except Exception as error:
        print('DIAGNOSTIC_SETUP_ERROR_TYPE=' + type(error).__name__, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
