"""Audit saved real-physics logs and replay ranking only; no API or simulation.

Existing databases are never opened. Offline replays create fresh ledgers only
inside a new audit output directory. A passing audit verifies contracts and
provenance; it does not claim LLM performance, repair skills, or generalization.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import stat
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'worker'))

from orchestrator.independent_ranker import IndependentPipeline, IndependentRankService
from worker.task_memory import (FrozenTaskMemory, from_legacy_records, load_snapshot,
                                project_for_model, save_snapshot)


MODES = ('test_id_ascending', 'test_id_descending')
DISABLED = {'enabled': False, 'max_calls': 0, 'max_usd': 0}
LEGACY = ROOT / 'runs/skill_live_two_episode_idfix_20261017/frozen_reuse/memory.json'


def guarded(value):
    text = str(value)
    normalized = text.replace('\\', '/')
    drive = PureWindowsPath(text).drive
    if ('\x00' in text or '..' in normalized.split('/') or normalized.startswith('//')
            or (drive and drive.lower() != 'c:') or normalized.lower().startswith('/mnt/e')):
        raise ValueError('excluded_audit_path')
    path = Path(value).absolute()
    if not path.is_relative_to(ROOT / 'runs'):
        raise ValueError('audit_paths_must_stay_inside_project_runs')
    for item in reversed((path,) + tuple(path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('linked_audit_path')
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def sha(path):
    return hashlib.sha256(guarded(path).read_bytes()).hexdigest()


def write_new(path, value):
    with guarded(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class Audit:
    def __init__(self, output):
        self.output = output
        self.checks = []
        self.inputs = {}
        self.api_attempts = 0

    def read(self, path, lines=False):
        path = guarded(path)
        if path.suffix in ('.sqlite', '.db') or path.name.endswith(('-wal', '-shm')):
            raise ValueError('existing_database_reads_forbidden')
        content = path.read_bytes()
        self.inputs[str(path)] = hashlib.sha256(content).hexdigest()
        if lines:
            return [json.loads(line) for line in content.decode('utf-8').splitlines() if line.strip()]
        return json.loads(content.decode('utf-8'))

    def check(self, name, passed, evidence=None):
        self.checks.append({'name': name, 'passed': bool(passed), 'evidence': evidence})
        return bool(passed)

    def trap_transport(self, *args, **kwargs):
        self.api_attempts += 1
        raise AssertionError('audit_must_never_call_provider_transport')

    def protected(self, name, callback):
        try:
            return callback()
        except Exception as error:
            self.check(name + '/read_or_contract_error', False, {'error_type': type(error).__name__})
            return None


def complete_permutation(order, expected):
    return (type(order) is list and all(type(p) is str for p in order)
            and len(order) == len(expected) and len(set(order)) == len(order)
            and set(order) == set(expected))


def audit_mode(audit, root, config, mode):
    folder = guarded(root / mode)
    row = audit.read(folder / 'row.json')
    parsed = audit.read(folder / 'parsed_result.json')
    state = audit.read(folder / 'graph_state.json')
    events = audit.read(folder / 'events.jsonl', True)
    remote = audit.read(folder / 'remote_audit.jsonl', True)
    execute = audit.read(folder / 'execute/command.json')
    validation = audit.read(folder / 'validation/command.json')
    features = audit.read(folder / 'features.json')
    context = audit.read(folder / 'context.json')
    episode = audit.read(folder / 'episode_memory.json')
    load_snapshot(folder / 'episode_memory.json')
    assembly_name = config['assembly_dir'].replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]
    native = folder / 'native/dfs-independent' / ('s' + str(config['seed'])) / assembly_name
    setup = audit.read(native / 'setup.json')
    by_kind = lambda kind: [e for e in events if e.get('kind') == kind]
    generations, decisions = by_kind('generation'), by_kind('rank_decision')
    starts, outcomes = by_kind('verifier_start'), by_kind('verifier_result')
    begin, finish = by_kind('run_start'), by_kind('run_end')
    audit.check(mode + '/events_have_unique_ids', bool(events)
                and len({e['event_id'] for e in events}) == len(events))
    audit.check(mode + '/independent_run_completed', len(begin) == len(finish) == 1
                and begin[0].get('generator') == 'independent'
                and begin[0].get('state_owner') == 'upstream_DFS'
                and begin[0].get('heuristic_fallback') is False
                and finish[0].get('status') == 'returned_from_upstream'
                and finish[0].get('heuristic_tripwire_calls') == 0
                and row.get('heuristic_tripwire_calls') == 0)
    audit.check(mode + '/real_worker_command_and_validated_artifacts', execute['exit_code'] == 0
                and any(str(arg).endswith('/worker/run_independent_rank.py') for arg in execute['command'])
                and validation['exit_code'] == 0 and parsed.get('validation_ok') is True
                and parsed.get('tree_checked') is True and parsed.get('physics_replayed') is False
                and parsed.get('status') in ('PASS', 'UNKNOWN'))
    audit.check(mode + '/plan_status_matches_saved_stats_not_exit_alone',
                parsed.get('plan_success') is parsed.get('raw_stats', {}).get('success')
                and row.get('plan_success') is parsed.get('plan_success')
                and row.get('status') == state.get('status') == parsed.get('status'),
                {'plan_success': parsed.get('plan_success'), 'status': parsed.get('status')})
    audit.check(mode + '/generation_and_decision_counts', bool(generations)
                and len(generations) == len(decisions)
                and len({e['generation_id'] for e in generations}) == len(generations)
                and len({e['generation_id'] for e in decisions}) == len(decisions))
    generation_map = {e['generation_id']: e for e in generations}
    requests, responses = {}, {}
    for filename in sorted((folder / 'ipc/requests').glob('*.json')):
        envelope = audit.read(filename)
        request = envelope['request']
        audit.check(mode + '/request_filename/' + request['request_id'],
                    filename.name == hashlib.sha256(request['request_id'].encode()).hexdigest() + '.json')
        audit.check(mode + '/request_unique/' + request['request_id'], request['request_id'] not in requests)
        IndependentRankService._validate_envelope(envelope)
        requests[request['request_id']] = envelope
    for filename in sorted((folder / 'ipc/responses').glob('*.json')):
        response = audit.read(filename)
        audit.check(mode + '/response_unique/' + response['request_id'], response['request_id'] not in responses)
        responses[response['request_id']] = response
    audit.check(mode + '/ipc_counts', len(requests) == len(responses) == len(decisions)
                and set(requests) == set(responses))
    ordered = []
    for decision in decisions:
        request = decision['request']
        rid, gid = request['request_id'], decision['generation_id']
        generation, envelope, response = generation_map[gid], requests[rid], responses[rid]
        expected = sorted(p for p in request['assembled_parts'] if p != request['base_part'])
        wanted_order = expected if mode == 'test_id_ascending' else list(reversed(expected))
        audit.check(mode + '/complete_candidates/' + gid,
                    request['assembled_parts'] == sorted(set(request['assembled_parts']))
                    and request['candidate_ids'] == expected
                    and generation['enumerated_candidates'] == expected
                    and generation.get('full_candidate_set') is True
                    and generation.get('heuristic_or_gnn_invoked') is False
                    and set(envelope['features']['cad']['parts']) == set(request['assembled_parts']))
        audit.check(mode + '/roundtrip/' + gid, request == envelope['request']
                    and response['state_id'] == request['state_id']
                    and response['request_id'] == rid
                    and complete_permutation(response['candidate_ids'], expected)
                    and response['candidate_ids'] == decision['applied_order'] == generation['applied_order'] == wanted_order
                    and response.get('fallback_reason') is None and decision.get('fallback_reason') is None)
        audit.check(mode + '/rank_rng_unchanged/' + gid,
                    decision['rng_before'] == decision['rng_after']
                    and generation['rng_before'] == generation['rng_after'])
        metadata = response['metadata']
        audit.check(mode + '/no_api/' + gid, metadata.get('attempted') is False
                    and metadata.get('llm_live') is False and metadata.get('offline_fixture') is True
                    and metadata.get('pipeline_mode') == mode)
        ordered.append(envelope)
    audit.check(mode + '/remote_all_accepted', len(remote) == len(decisions)
                and all(e.get('outcome') == 'accepted' and e.get('fallback_reason') is None for e in remote))
    paired = {e['call_id']: e for e in outcomes}
    audit.check(mode + '/verifier_calls_pair_and_match_metrics', bool(starts)
                and len(starts) == len(outcomes) == len(paired)
                and len({e['call_id'] for e in starts}) == len(starts)
                and set(paired) == {e['call_id'] for e in starts}
                and row.get('verifier_calls') == finish[0].get('verifier_calls') == len(starts)
                and row.get('verifier_results') == len(outcomes))
    terminal = [e for e in starts if e.get('generation_id') is None]
    audit.check(mode + '/terminal_handling_not_claimed_as_LLM', row.get('terminal_verifier_calls') == len(terminal)
                and all(e.get('selection_scope') == 'upstream_terminal_completion'
                        and len(e['request']['parts_rest']) == 1 for e in terminal)
                and all(e.get('generation_id') in generation_map for e in starts if e not in terminal))
    audit.check(mode + '/raw_physics_status_and_unknowns', all(
        e.get('blocker') is None and e.get('path') is None and (
            (e['status'] == 'PASS' and e.get('raw', {}).get('feasible') is True)
            or (e['status'] == 'UNKNOWN' and e.get('raw', {}).get('feasible') is False)
            or (e['status'] == 'ERROR' and e.get('raw') is None)) for e in outcomes))
    audit.check(mode + '/frozen_episode_count_and_sources',
                episode['record_count'] == len(episode['records']) == len(outcomes)
                and row.get('stored_observations') == len(outcomes)
                and row.get('memory_collection') == 'PASS'
                and {r['record_id'] for r in episode['records']} == {e['event_id'] for e in outcomes}
                and all(r['source']['sha256'] == sha(folder / 'events.jsonl')
                        and r['context'] == context and r['allow_hard_pruning'] is False
                        and r['constraints'] == [] for r in episode['records']))
    audit.check(mode + '/source_preserved_and_API_zero', row.get('source_preserved') is True
                and row.get('live_api_attempts') == 0)
    return dict(mode=mode, row=row, requests=ordered, features=features, context=context,
                setup=setup, memory_path=folder / 'episode_memory.json')


def body_for(pipeline, envelope):
    """Construct the same reviewed request body without calling RankService.rank."""
    from orchestrator.independent_ranker import _empty_memory
    value = json.loads(pipeline.service._validate_envelope(envelope))
    value['features']['memory'] = (_empty_memory() if pipeline.memory is None else
        project_for_model(pipeline.memory.retrieve(value['features']['memory_context'],
            value['request']['assembled_parts'], value['request']['candidate_ids'])))
    value = json.loads(pipeline.service._validate_envelope(value))
    return pipeline.service._body(value)


def offline_replay(audit, source):
    requests = source['requests']
    memory_path = source['memory_path']
    before_memory = sha(memory_path)
    legacy = audit.read(LEGACY)
    imported = from_legacy_records(legacy['records'])
    imported_path = audit.output / 'legacy_reference_memory.json'
    save_snapshot(imported, imported_path)
    audit.check('legacy/import_preserves_scope_and_raw', bool(imported)
                and len(imported) == len(legacy['records'])
                and all(r.get('reference_only') is True and r.get('imported_legacy') is True
                        and r['raw_result'] == r['legacy_record']['raw_result']
                        and r['source'] == r['legacy_record']['source']
                        and r['constraints'] == [] for r in imported))
    replays = {name: [] for name in ('off', 'on', 'legacy_reference')}
    previews = {name: [] for name in replays}
    for name, path in (('off', None), ('on', memory_path), ('legacy_reference', imported_path)):
        pipeline = IndependentPipeline('deterministic_memory', dict(DISABLED),
            audit.output / (name + '_new_replay_ledger.sqlite'), path, transport=audit.trap_transport)
        try:
            for index, envelope in enumerate(requests):
                before = canonical(envelope)
                response = pipeline.rank(deepcopy(envelope))
                body = body_for(pipeline, deepcopy(envelope))
                payload = json.loads(body['input'][0]['content'])
                memory = payload['features']['memory']
                metadata = response['metadata']
                expected = envelope['request']['candidate_ids']
                audit.check('replay/' + name + '/complete_no_api/' + str(index),
                            canonical(envelope) == before and response.get('fallback_reason') is None
                            and complete_permutation(response['candidate_ids'], expected)
                            and metadata.get('attempted') is False and metadata.get('llm_live') is False
                            and metadata.get('ranking_source') == 'deterministic_memory')
                audit.check('replay/' + name + '/memory_provenance/' + str(index),
                            metadata.get('retrieved_evidence_ids') == memory['evidence_ids']
                            and memory['allow_hard_pruning'] is False
                            and memory['physics_reuse_allowed'] is False and memory['constraints'] == [])
                if name == 'off':
                    audit.check('replay/off/empty/' + str(index), memory['evidence_ids'] == [])
                elif name == 'on':
                    audit.check('replay/on/exact_observation_available/' + str(index),
                                bool(memory['exact_observations'])
                                and all(r['current_attempt_verified'] is False
                                        for r in memory['exact_observations']))
                else:
                    references = memory['cross_task_references']
                    audit.check('replay/legacy/reference_only_no_ID_transfer/' + str(index),
                                memory['exact_observations'] == [] and bool(references)
                                and all(r['reference_only'] is True and r['current_part_id_mapping'] is None
                                        and r['current_attempt_verified'] is False for r in references)
                                and response['candidate_ids'] == replays['off'][index]['response']['candidate_ids'])
                replays[name].append({'request_id': envelope['request']['request_id'],
                    'input_sha256': digest(envelope), 'response': response,
                    'body_sha256': digest(body), 'physics_replayed': False})
                previews[name].append({'request_id': envelope['request']['request_id'],
                                      'body': body, 'provider_called': False})
            invalid = deepcopy(requests[0])
            invalid['features']['deterministic_fallback_order'] = list(invalid['request']['candidate_ids'])
            rejected = pipeline.rank(invalid)
            audit.check('replay/' + name + '/hidden_order_rejected',
                        rejected.get('fallback_reason') == 'invalid_request'
                        and rejected['metadata'].get('attempted') is False)
        finally:
            pipeline.close()
    for index in range(len(requests)):
        payloads = [json.loads(previews[name][index]['body']['input'][0]['content']) for name in replays]
        audit.check('replay/same_input_and_prompt_shape/' + str(index),
                    len({replays[name][index]['input_sha256'] for name in replays}) == 1
                    and all(p['request'] == payloads[0]['request'] for p in payloads)
                    and all(set(p['features']) == set(payloads[0]['features']) for p in payloads)
                    and all(p['features']['cad'] == payloads[0]['features']['cad'] for p in payloads)
                    and all(p['features']['episode_feedback'] == payloads[0]['features']['episode_feedback'] for p in payloads))
    audit.check('replay/frozen_original_memory_unchanged', sha(memory_path) == before_memory)
    audit.check('replay/provider_transport_never_called', audit.api_attempts == 0)
    write_new(audit.output / 'offline_replay.json', replays)
    write_new(audit.output / 'model_body_previews_NOT_SENT.json', previews)
    return {'replay_requests_per_condition': len(requests), 'conditions': list(replays),
            'new_physics_calls': 0, 'new_API_calls': audit.api_attempts,
            'legacy_records_imported': len(imported),
            'scope': 'Saved-request offline ranking replay, not a new plan or causal memory evaluation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root, output = guarded(args.run_dir), guarded(args.output)
    if output == root or output.is_relative_to(root):
        raise ValueError('audit_output_must_be_separate_from_existing_physics_run')
    output.mkdir(parents=True, exist_ok=False)
    audit = Audit(output)
    result, replay, completed = {}, None, []
    config = audit.protected('configuration', lambda: audit.read(root / 'config.json'))
    result = audit.protected('experiment_result', lambda: audit.read(root / 'result.json')) or {}
    def preservation():
        before, after = audit.read(root / 'source_before.json'), audit.read(root / 'source_after.json')
        audit.check('source/before_after_equal', bool(before) and before == after
                    and result.get('original_sources_preserved') is True)
        audit.check('experiment/no_live_LLM', result.get('live_llm_verification') == 'NOT_RUN'
                    and config['api_config'].get('enabled') is False
                    and config['api_config'].get('max_calls') == 0 and config['api_config'].get('max_usd') == 0)
    if config is not None:
        audit.protected('preservation', preservation)
        for mode in MODES:
            data = audit.protected(mode, lambda mode=mode: audit_mode(audit, root, config, mode))
            if data is not None:
                completed.append(data)
    audit.check('both_test_orders/auditable', len(completed) == len(MODES))
    if len(completed) == 2:
        ascending, descending = completed
        audit.check('both_test_orders/same_initial_candidates_and_task',
                    ascending['requests'][0]['request']['candidate_ids'] == descending['requests'][0]['request']['candidate_ids']
                    and ascending['features'] == descending['features']
                    and ascending['context'] == descending['context'])
        audit.check('both_test_orders/same_native_physics_setup', ascending['setup'] == descending['setup'],
                    ascending['setup'])
        audit.check('both_test_orders/different_forced_first_order',
                    ascending['row']['rank_orders'][0] != descending['row']['rank_orders'][0])
        replay = audit.protected('offline_replay', lambda: offline_replay(audit, ascending))
    unchanged = all(sha(Path(path)) == digest_value for path, digest_value in audit.inputs.items())
    audit.check('audit/original_read_files_unchanged', unchanged, {'files': len(audit.inputs)})
    audit.check('audit/new_API_attempts_zero', audit.api_attempts == 0)
    summary = {'status': 'PASS' if audit.checks and all(c['passed'] for c in audit.checks) else 'FAIL',
        'checks_passed': sum(c['passed'] for c in audit.checks), 'checks_total': len(audit.checks),
        'source_run': str(root), 'new_API_calls': audit.api_attempts, 'new_physics_calls': 0,
        'actual_plans': [{'mode': data['mode'], 'plan_success': data['row']['plan_success'],
                          'status': data['row']['status'], 'verifier_calls': data['row']['verifier_calls']}
                         for data in completed], 'offline_replay': replay,
        'LLM_performance': 'NOT_RUN', 'repair_skills': 'NOT_IMPLEMENTED',
        'cross_task_effectiveness': 'UNVERIFIED',
        'note': 'Audit PASS means saved-log contracts and offline replay passed; actual plan outcomes are reported separately.'}
    write_new(output / 'checks.json', audit.checks)
    write_new(output / 'read_file_hashes.json', audit.inputs)
    write_new(output / 'result.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
