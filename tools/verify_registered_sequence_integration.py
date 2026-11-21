"""Offline provider-contract replay of real saved CAD and registered scorers.

The responses are explicitly injected fixtures, not model decisions. No CAD is
loaded into a simulator and no network request is permitted. Historical physical
validation is never promoted to current native-ASAP physical validation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator.relational_knowledge import read_json, safe_path
from orchestrator.scored_order_skill import feature_rows, run_skill
from orchestrator.scored_sequence_runtime import load_bridges
from orchestrator.sequence_ranker import PROTOCOL, SequenceRankService
from worker.independent_features import filter_features
from worker.task_memory import digest, task_context


SOURCES = {
    '20014': {
        'features': 'runs/native_pair_20014_00709_20261119/episodes/20014/features.json',
        'result': 'runs/native_pair_20014_00709_20261119/episodes/20014/worker_result.json',
    },
    '00709': {
        'features': 'runs/remaining32_prep_timeout_retry_20261116/live/p26/episodes/gnn/p26-gnn/features.json',
        'result': 'runs/remaining32_prep_timeout_retry_20261116/live/p26/episodes/gnn/p26-gnn/worker_result.json',
    },
    '07009': {
        'features': 'runs/multitask_learning_prep_20261105/request_inputs/07009/features.json',
        'snapshot': 'runs/scored_order_skill_pilot_20261111/transfer/07009/actual_snapshot.json',
    },
}


def sha_file(path):
    hasher = hashlib.sha256()
    with safe_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_sha(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def write_json(path, value):
    path = safe_path(path)
    # Every artifact is new. Existing evidence is never replaced.
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')


def require(value, reason):
    if not value:
        raise AssertionError(reason)


def saved_states(inputs):
    states = []
    for product, paths in SOURCES.items():
        features = deepcopy(inputs[paths['features']])
        provenance = dict(product_label=product,
                          source_assembly_id=features['task']['assembly_id'],
                          source_files=deepcopy(paths), geometry_recomputed=False,
                          observed_pose_replayed=False,
                          frame='saved reference CAD frame, not settled simulation coordinates')
        if 'snapshot' in paths:
            snapshot = inputs[paths['snapshot']]
            require(snapshot['parts'] == features['parts'], 'saved_geometry_sources_do_not_match')
            require(type(snapshot.get('contact_edges')) is list, 'saved_contact_graph_missing')
            require(snapshot.get('contact_source') == 'RedMax body_in_contact on original fixed CAD pairs',
                    'unexpected_saved_contact_source')
            # Copy a previously measured graph only after exact geometry equality.
            # The original features/snapshot files remain untouched.
            features['relations']['contact_edges'] = deepcopy(snapshot['contact_edges'])
            features['relations']['contact_source'] = 'saved_RedMax_contact_graph_replay'
            provenance['contact_graph_replay'] = dict(
                parts_exactly_equal=True, source=paths['snapshot'],
                contact_source=snapshot['contact_source'], contact_eps=snapshot.get('contact_eps'),
                contact_units=snapshot.get('contact_units'), additional_physics_calls=0)
        require(type(features['relations']['contact_edges']) is list, 'saved_contact_graph_missing')
        initial = sorted(features['parts'])
        states.append(dict(name=product + '_initial', parts=initial,
                           features=features, provenance=deepcopy(provenance)))
        if 'result' in paths:
            result = inputs[paths['result']]
            chain = result['connected']['best_partial_chain']
            require(type(chain) is list and len(chain) > 0, 'saved_partial_chain_missing')
            parent = initial
            for edge in chain:
                require(sorted(edge['parent']) == parent, 'saved_chain_not_connected')
                require(edge['sim_info'].get('feasible') is True, 'saved_chain_not_physics_passed')
                require(set(edge['child']) == set(parent) - {edge['part_move']}, 'invalid_saved_chain_transition')
                parent = sorted(edge['child'])
            require(len(parent) > 2, 'terminal_state_is_not_rankable')
            states.append(dict(name=product + '_last', parts=parent, features=features,
                provenance=dict(provenance, state_source='connected.best_partial_chain[-1].child',
                                saved_removal_count=len(chain))))
    require(len(states) == 5, 'five_saved_states_required')
    return states


def envelope(state, request_id):
    features = state['features']
    assembled = sorted(state['parts'])
    base = features['task']['base_part']
    return dict(schema_version=1, kind='rank_request', timeout_seconds=30,
        request=dict(run_id='offline_registered_sequence_integration', request_id=request_id,
            state_id='saved-state:' + digest(dict(task=features['task'], parts=assembled)),
            assembled_parts=assembled, candidate_ids=sorted(p for p in assembled if p != base),
            base_part=base),
        features=dict(protocol=PROTOCOL, cad=filter_features(features, assembled),
            memory_context=task_context(features), episode_feedback=[], validated_repair_skills=[],
            decision_scope='complete_part_permutation_only',
            lower_level_owner='original_ASAP_pose_direction_support_search',
            terminal_policy='unchanged_upstream_mass_or_base_rule'))


def provider_audit(body, request, bridges):
    payload = json.loads(body['input'][0]['content'])
    schema = body['text']['format']['schema']
    properties = schema['properties']
    require(payload['request'] == request, 'provider_request_changed')
    require(body['text']['format']['strict'] is True, 'strict_schema_missing')
    require(schema['additionalProperties'] is False, 'schema_allows_extra_properties')
    require(properties['request_id']['enum'] == [request['request_id']], 'schema_request_id_mismatch')
    require(properties['state_id']['enum'] == [request['state_id']], 'schema_state_id_mismatch')
    candidate_schema = properties['candidate_ids']['anyOf'][0]
    require(candidate_schema['items']['enum'] == request['candidate_ids'], 'schema_candidate_id_mismatch')
    require(candidate_schema['minItems'] == candidate_schema['maxItems'] == len(request['candidate_ids']),
            'schema_permutation_length_mismatch')
    offered = payload['knowledge']['sequence_skills']
    offered_by_id = {s['skill_id']: s for s in offered}
    expected_ids = [bridge['skill_id'] for bridge in bridges]
    require(len(offered) == len(offered_by_id), 'duplicate_provider_skill')
    require(set(properties['selected_skill_id']['enum']) == set(offered_by_id) | {None},
            'schema_skill_id_mismatch')
    require(set(expected_ids) <= set(offered_by_id), 'registered_skill_missing_from_actual_provider_body')
    rows = []
    for bridge in bridges:
        skill = offered_by_id[bridge['skill_id']]
        require(skill['program']['source'] == bridge['program']['source'], 'provider_python_source_changed')
        require(source_sha(skill['program']['source']) == bridge['program_sha256'], 'provider_source_bytes_hash_changed')
        require(skill['content_sha256'] == bridge['content_sha256'], 'provider_bridge_hash_changed')
        require(skill['original_registry_sha256'] == bridge['original_registry_sha256'], 'provider_registry_hash_changed')
        require(skill['native_physics_validation'] == 'NOT_RUN', 'historical_physics_promoted_to_native')
        require(skill['relational_card_id'] in payload['knowledge']['relational_context']['card_ids'],
                'executable_skill_missing_relational_reference')
        rows.append(dict(skill_id=bridge['skill_id'], source_utf8_sha256=source_sha(skill['program']['source']),
            bridge_sha256=bridge['content_sha256'], original_registry_sha256=bridge['original_registry_sha256'],
            relational_card_id=skill['relational_card_id'], source_bytes_equal=True))
    return dict(strict_schema_checked=True, offered_skills=rows,
                provider_body_sha256=digest(body), provider_body_bytes=len(json.dumps(body, ensure_ascii=False).encode('utf-8')))


def fixture_response(body, selection):
    payload = json.loads(body['input'][0]['content'])
    request, knowledge = payload['request'], payload['knowledge']
    chosen = None if selection == 'direct' else selection
    reference = next((s['relational_card_id'] for s in knowledge['sequence_skills']
                      if s['skill_id'] == chosen), None)
    decision = dict(request_id=request['request_id'], state_id=request['state_id'],
        decision_kind='direct' if chosen is None else 'skill',
        candidate_ids=list(reversed(request['candidate_ids'])) if chosen is None else None,
        selected_skill_id=chosen, evidence_ids=[],
        relational_card_ids=[] if reference is None else [reference])
    return dict(status='completed', error=None, id='fixture-response:' + request['request_id'],
        model='offline-provider-fixture', output=[dict(type='message', role='assistant', status='completed',
            content=[dict(type='output_text', text=json.dumps(decision))])],
        usage=dict(input_tokens=100, output_tokens=20, total_tokens=120))


def run_case(directory, state, selection, library, frozen, bridges):
    directory.mkdir()
    audit = dict(status='FAIL', state=state['name'], selection=selection,
                 provider_response_fixture=True, actual_model_decision=False,
                 actual_network_calls=0, physics_calls=0, current_task_physics_validation='NOT_RUN',
                 native_physics_validation='NOT_RUN', performance_effect='UNVERIFIED')
    transport_calls = []
    try:
        request = envelope(state, 'offline:' + directory.name)
        input_hash = digest(request)
        write_json(directory / 'envelope.json', request)
        write_json(directory / 'state_provenance.json', state['provenance'])

        def transport(url, headers, body, timeout):
            # No real credentials are read or saved, even in this injected path.
            require(headers.get('Authorization') == 'Bearer fixture-key-no-network', 'fixture_key_not_used')
            require(len(transport_calls) == 0, 'automatic_provider_retry_detected')
            transport_calls.append(deepcopy(body))
            write_json(directory / 'provider_body_actual.json', body)
            response = fixture_response(body, selection)
            write_json(directory / 'provider_response_injected.json', response)
            return response

        config = dict(enabled=True, model='offline-provider-fixture', api_key_env='REGISTERED_SEQUENCE_FIXTURE_KEY',
            max_calls=1, max_usd='1', input_usd_per_million='1', output_usd_per_million='1')
        service = SequenceRankService(config, directory / 'fixture_ledger.sqlite', transport=transport,
                                      relational_knowledge=library, scored_skills=frozen)
        prepared = service.prepare(request)
        body = service._body(prepared)
        write_json(directory / 'provider_body_prepared.json', body)
        write_json(directory / 'knowledge_audit_prepared.json', prepared['features']['knowledge_audit'])
        audit['provider'] = provider_audit(body, request['request'], bridges)
        result = service.rank(request)
        write_json(directory / 'rank_response.json', result)
        require(len(transport_calls) == 1, 'injected_transport_not_called_once')
        require(transport_calls[0] == body, 'prepared_and_actual_provider_body_differ')
        require(result['fallback_reason'] is None, 'rank_failed:' + str(result['fallback_reason']))
        require(result['metadata']['llm_live'] is False, 'fixture_reported_as_live_llm')
        require(result['metadata']['retries'] == 0, 'retry_count_not_zero')
        require(digest(request) == input_hash, 'input_envelope_mutated')
        order = result['candidate_ids']
        require(len(order) == len(set(order)) and set(order) == set(request['request']['candidate_ids']),
                'result_not_complete_permutation')
        require(request['request']['base_part'] not in order, 'fixed_base_in_output')
        if selection == 'direct':
            require(order == list(reversed(request['request']['candidate_ids'])), 'direct_response_not_preserved')
            require(result['metadata']['skill_execution'] is None, 'direct_choice_executed_skill')
            audit['old_vm_comparison'] = 'NOT_APPLICABLE_DIRECT_RESPONSE'
        else:
            bridge = next(b for b in bridges if b['skill_id'] == selection)
            cad = request['features']['cad']
            rows = feature_rows(dict(parts=cad['parts'], contact_edges=cad['relations']['contact_edges']),
                                request['request']['assembled_parts'])
            legacy = run_skill(bridge['program']['source'], rows)
            write_json(directory / 'legacy_feature_rows.json', rows)
            write_json(directory / 'legacy_vm_result.json', legacy)
            expected = [p for p in legacy['candidate_order'] if p != request['request']['base_part']]
            execution = result['metadata']['skill_execution']
            require(legacy['status'] == execution['status'] == 'PROPOSED', 'vm_did_not_propose')
            require(order == expected == execution['candidate_order'], 'original_vm_order_changed')
            require(legacy['vm_steps'] == execution['vm_steps'], 'original_vm_steps_changed')
            require(digest(rows) == execution['full_feature_rows_sha256'], 'original_full_geometry_rows_changed')
            require(execution['physics_executed'] is False and execution['current_attempt_verified'] is False,
                    'fixture_reported_physics_success')
            require(execution['native_physics_validation'] == 'NOT_RUN', 'bridge_reported_native_physics')
            audit['old_vm_comparison'] = dict(order_equal=True, vm_steps_equal=True,
                full_geometry_rows_equal=True, vm_steps=execution['vm_steps'],
                base_score_evaluated=execution['base_score_evaluated'])
        audit.update(status='PASS', candidate_order=order, remaining_parts=len(request['request']['assembled_parts']),
                     injected_provider_calls=1, actual_api_calls=0)
    except Exception as error:
        audit.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                     injected_provider_calls=len(transport_calls), actual_api_calls=0)
    write_json(directory / 'audit.json', audit)
    print(json.dumps(dict(case=directory.name, status=audit['status'], error=audit.get('error')), ensure_ascii=True), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory under this workspace; existing directories are rejected.')
    parser.add_argument('--library', required=True, help='Frozen relational library JSON; no migration is performed.')
    parser.add_argument('--bridges', required=True, help='Frozen explicit scored bridge JSON.')
    args = parser.parse_args()
    # All user paths are lexically checked before filesystem operations.
    output, library_path, bridges_path = (safe_path(v) for v in (args.output, args.library, args.bridges))
    require(not output.exists(), 'output_directory_already_exists')
    require(output.parent.is_dir(), 'output_parent_must_exist')
    paths = {str(library_path): library_path, str(bridges_path): bridges_path}
    for sources in SOURCES.values():
        for name in sources.values():
            paths[name] = safe_path(name)
    for name in ('orchestrator/sequence_ranker.py', 'orchestrator/scored_sequence_runtime.py',
                 'orchestrator/relational_runtime.py', 'orchestrator/relational_retrieval.py',
                 'orchestrator/scored_sequence_bridge.py', 'orchestrator/sequence_knowledge.py',
                 'orchestrator/scored_order_skill.py', 'tools/verify_registered_sequence_integration.py'):
        paths[name] = safe_path(name)
    pins = {name: sha_file(path) for name, path in paths.items()}
    output.mkdir()
    write_json(output / 'input_pins_before.json', pins)
    library = read_json(library_path)
    frozen = read_json(bridges_path)
    bridges = load_bridges(frozen)
    require(len(bridges) == 2, 'exactly_two_registered_bridges_required')
    inputs = {name: read_json(path) for name, path in paths.items()
              if any(name in sources.values() for sources in SOURCES.values())}
    states = saved_states(inputs)
    write_json(output / 'manifest.json', dict(protocol='registered_sequence_provider_replay/v1',
        source_files=SOURCES, library_sha256=library['content_sha256'], bridges_sha256=frozen['content_sha256'],
        states=[dict(name=s['name'], parts=s['parts'], provenance=s['provenance']) for s in states],
        skill_ids=[b['skill_id'] for b in bridges], expected_cases=15,
        actual_api_calls=0, actual_network_calls=0, physics_calls=0,
        provider_response_fixture=True, actual_llm_selection_verified=False,
        native_physics_validation='NOT_RUN', performance_effect='UNVERIFIED'))
    blocked_attempts = []

    def deny(*args, **kwargs):
        blocked_attempts.append('network_or_process_attempt')
        raise RuntimeError('offline_replay_forbids_network_and_subprocesses')

    audits = []
    with ExitStack() as stack:
        stack.enter_context(patch.dict('os.environ', {'REGISTERED_SEQUENCE_FIXTURE_KEY': 'fixture-key-no-network'}))
        for target in ('urllib.request.urlopen', 'urllib.request.OpenerDirector.open',
                       'socket.create_connection', 'socket.socket.connect', 'socket.socket.connect_ex',
                       'subprocess.Popen'):
            stack.enter_context(patch(target, side_effect=deny))
        for state in states:
            for index, selection in enumerate(['direct'] + [b['skill_id'] for b in bridges]):
                audits.append(run_case(output / (state['name'] + '_' + str(index)), state, selection,
                                       library, frozen, bridges))
    after = {name: sha_file(path) for name, path in paths.items()}
    write_json(output / 'input_pins_after.json', after)
    unchanged = pins == after
    passed = sum(row['status'] == 'PASS' for row in audits)
    success = len(audits) == 15 and passed == 15 and unchanged and not blocked_attempts
    summary = dict(status='PASS' if success else 'FAIL', expected_cases=15, cases=len(audits),
        passed=passed, failed=len(audits)-passed, input_files_unchanged=unchanged,
        changed_inputs=[name for name in pins if pins[name] != after[name]],
        actual_api_calls=0, actual_network_calls=0, physics_calls=0, blocked_attempts=len(blocked_attempts),
        injected_provider_calls=sum(a['injected_provider_calls'] for a in audits),
        provider_response_fixture=True, actual_llm_selection_verified=False,
        current_native_physics_validation='NOT_RUN', performance_effect='UNVERIFIED',
        limitations=['Injected responses prove wiring/contract behavior, not actual model selection.',
                     'Saved contact graphs describe reference CAD, not simulated settled poses.',
                     'Historical path-first registration does not prove current native physics or transfer benefit.'],
        results=audits)
    write_json(output / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, ensure_ascii=True), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
