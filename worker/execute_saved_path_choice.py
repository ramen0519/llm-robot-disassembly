"""Execute one saved live decision at an actual recorded failure state (Py3.8).

This component test neither resumes DFS nor labels a fixture outcome as an LLM
outcome. It sends no API request. The only selectable controller parameters are
those in the accepted saved live response; no_repair performs zero physics.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import pickle
import random
import subprocess
import sys
import time

from constraint_memory import (FrozenConstraintMemory, load_snapshot, records_from_evidence,
                               save_snapshot, validate_event)
from task_memory import _lexical_path, _safe_path, canonical, digest
from replay_path_extension import (read_json, sha, readonly_sdf_cache, _solver_hashes,
                                    EXPECTED_SOLVER_KEYS, ATOL, RTOL)

AXES = {'+X': [1, 0, 0], '+Y': [0, 1, 0], '-Y': [0, -1, 0], '+Z': [0, 0, 1], '-Z': [0, 0, -1]}


def local_path(value):
    raw = _lexical_path(value)
    normalized = raw.replace('\\', '/')
    if os.name == 'nt' and normalized.lower().startswith('/mnt/c/'):
        raw = 'C:/' + normalized[7:]
    elif os.name != 'nt' and PureWindowsPath(raw).drive.lower() == 'c:':
        raw = '/mnt/c/' + normalized[3:]
    return _safe_path(raw)


def relative(root, value):
    if (type(value) is not str or PureWindowsPath(value).drive or value.startswith(('/', '\\'))
            or ':' in value or any(p in ('', '.', '..') for p in value.replace('\\', '/').split('/'))):
        raise ValueError('safe_relative_artifact_required')
    return local_path(root / value)


def jsonlines(path):
    values = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    canonical(values)
    return values


def write_new(path, value):
    with local_path(path).open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')


def _ids(value, offered):
    return (type(value) is list and all(type(v) is str for v in value)
            and len(set(value)) == len(value) and set(value).issubset(offered))


def validate_choice(original, request, response, prepared, lineage):
    """Pure binding/decision checks; recorded labels alone never run physics."""
    if (lineage.get('route') != 'path' or lineage.get('state_id_preserved') is not True
            or lineage.get('solver_not_resumed') is not True
            or lineage.get('path_response_caused_stored_outcome') is not False
            or lineage.get('source_request_id') != original['request']['request_id']
            or lineage.get('source_run_id') != original['request']['run_id']
            or lineage.get('original_state_id') != original['request']['state_id']
            or lineage.get('diagnostic_request_id') != request['request']['request_id']):
        raise ValueError('diagnostic_lineage_does_not_bind_original_request')
    reconstructed = deepcopy(request)
    reconstructed['request'].update(run_id=original['request']['run_id'], request_id=original['request']['request_id'])
    if canonical(reconstructed) != canonical(original):
        raise ValueError('diagnostic_request_changed_original_failure_conditions')
    raw = deepcopy(prepared)
    context = raw['features'].pop('integrated_context')
    if canonical(raw) != canonical(request) or context.get('route') != 'path':
        raise ValueError('prepared_request_not_bound_to_path_request')
    metadata, identity = response.get('metadata', {}), request['request']
    if (response.get('schema_version') != 1 or response.get('kind') != 'rank_response'
            or response.get('fallback_reason') is not None
            or response.get('request_id') != identity['request_id']
            or response.get('state_id') != identity['state_id']
            or not _ids(response.get('candidate_ids'), identity['candidate_ids'])
            or len(response['candidate_ids']) != len(identity['candidate_ids'])
            or metadata.get('llm_live') is not True or metadata.get('attempted') is not True
            or metadata.get('decision_route') != 'path' or metadata.get('ranking_source') != 'integrated_llm'
            or metadata.get('model_requested') != 'gpt-6-luna'
            or type(metadata.get('model_returned')) is not str or not metadata['model_returned']):
        raise ValueError('requires_matching_accepted_live_path_response')
    features = request['features']
    if (features.get('protocol') != 'constraint_path_repair_v1' or features.get('first_axis') != '+X'
            or features.get('allowed_second_axes') != ['+Y', '-Y', '+Z', '-Z']
            or type(features.get('switch_step')) is not int or features['switch_step'] != 50):
        raise ValueError('unsupported_saved_controller_domain')
    choice = metadata.get('path_skill')
    if (type(choice) is not dict or type(choice.get('switch_step')) is not int or choice['switch_step'] != 50
            or choice.get('skill_id') not in ('no_repair', 'turn_force_two_leg_v1')
            or (choice['skill_id'] == 'no_repair' and choice.get('second_axis') != 'none')
            or (choice['skill_id'] == 'turn_force_two_leg_v1' and choice.get('second_axis') not in features['allowed_second_axes'])):
        raise ValueError('saved_choice_has_unregistered_skill_or_parameters')
    for key, offered in (('used_constraint_ids', [r['record_id'] for r in context['constraint_references']]),
                         ('used_skill_evidence_ids', [r['skill_evidence_id'] for r in context['frozen_skill_evidence']]),
                         ('used_path_feedback_ids', [r['feedback_id'] for r in context['path_feedback']])):
        if not _ids(choice.get(key), offered) or metadata.get(key) != choice[key]:
            raise ValueError('saved_choice_cites_unoffered_or_inconsistent_evidence')
    return deepcopy(choice)


def validate_failure(event, trajectory, request, context):
    validate_event(event, context)
    scope = event['scope']
    if (event['raw_success'] is not False or event['event_id'] != request['features']['failure_event_id']
            or scope['part_move'] != request['features']['moving_part_id']
            or scope['assembled_parts'] != request['request']['assembled_parts']
            or scope['base_part'] != request['request']['base_part']
            or canonical(scope['task_context']) != canonical(request['features']['memory_context'])
            or scope['action_schedule'] != [{'step': 0, 'action': [1, 0, 0]}]
            or context['task']['assembly_id'] != '02628' or scope['part_move'] != '0'):
        raise ValueError('requires_recorded_02628_part0_single_force_failure')
    expected_ids = sorted(p for p in scope['assembled_parts'] if p != scope['base_part'])
    expected_state = hashlib.sha256(json.dumps(sorted(scope['assembled_parts'])).encode()).hexdigest()
    if request['request']['candidate_ids'] != expected_ids or request['request']['state_id'] != expected_state:
        raise ValueError('recorded_candidate_or_state_identity_mismatch')
    controller, path = trajectory.get('controller', {}), trajectory.get('path')
    if (trajectory.get('raw_success') is not False or controller.get('skill_id') != 'native_single_force_probe'
            or controller.get('actions') != scope['action_schedule'] or controller.get('reset_count') != 1
            or controller.get('switch_count') != 0 or controller.get('intermediate_resets') != 0
            or controller.get('upstream_check_success_calls') != 1
            or type(path) is not list or not path
            or any(type(q) is not list or len(q) != 6 or any(type(x) not in (int, float) or not math.isfinite(x) for x in q) for q in path)
            or path[0] != scope['initial_q'] or path[-1] != event['terminal_q']
            or len(path) != event['trajectory']['states']):
        raise ValueError('failed_measurement_does_not_bind_raw_single_force_trajectory')
    if not any(pair['separation_unsatisfied'] for pair in event['measurements']['pairs']):
        raise ValueError('requires_actual_unsatisfied_endpoint_predicate')


def load_choice(source_run, decision_run):
    """Cross-link immutable actual-failure evidence and the new live decision."""
    source_run, decision_run = local_path(source_run), local_path(decision_run)
    source_files = {name: relative(source_run, name) for name in
                    ('result.json', 'context.json', 'path_outcomes.json', 'path_events.jsonl', 'constraint_memory.json')}
    diagnostic_files = {name: relative(decision_run, name) for name in
                        ('result.json', 'lineage.json', 'requests.jsonl', 'prepared_requests.jsonl',
                         'response_1.json', 'responses.jsonl', 'decisions.jsonl', 'source_before.json', 'source_after.json')}
    report, lineage_report = read_json(diagnostic_files['result.json']), read_json(diagnostic_files['lineage.json'])
    if (report.get('kind') != 'saved_evidence_live_contract_diagnostic' or report.get('status') not in ('PASS', 'PARTIAL')
            or report.get('source_preserved') is not True or report.get('full_DFS') is not False
            or report.get('new_physics_calls') != 0 or report.get('selected_path_executions') != 0
            or lineage_report.get('full_DFS') is not False or lineage_report.get('requests_independent') is not True):
        raise ValueError('source_decision_run_not_a_preserved_live_diagnostic')
    lineages = [value for value in lineage_report['requests'] if value.get('route') == 'path']
    if len(lineages) != 1:
        raise ValueError('requires_one_path_lineage')
    lineage = lineages[0]
    original_file = local_path(lineage['source_request_file'])
    try:
        original_file.relative_to(source_run / 'ipc' / 'requests')
    except ValueError:
        raise ValueError('lineage_request_outside_source_IPC')
    before, after = [read_json(diagnostic_files[name]) for name in ('source_before.json', 'source_after.json')]
    if before != after:
        raise ValueError('diagnostic_input_hashes_changed')
    # Verify the original request's recorded hash without following unrelated
    # entries in the diagnostic hash inventory or inspecting another drive.
    location = lambda value: str(value).replace('\\', '/').lower().replace('c:/', '/mnt/c/', 1)
    matches = [value for key, value in before.items() if location(key) == location(original_file)]
    if len(matches) != 1 or matches[0] != sha(original_file):
        raise ValueError('original_request_not_bound_to_diagnostic_source_hash')
    original = read_json(original_file)
    request = jsonlines(diagnostic_files['requests.jsonl'])[0]
    prepared = jsonlines(diagnostic_files['prepared_requests.jsonl'])[0]
    response = read_json(diagnostic_files['response_1.json'])
    if response != jsonlines(diagnostic_files['responses.jsonl'])[0]:
        raise ValueError('saved_response_copies_disagree')
    decision = jsonlines(diagnostic_files['decisions.jsonl'])[0]
    if (decision.get('request') != request or decision.get('enriched_request') != prepared
            or decision.get('response') != response):
        raise ValueError('actual_decision_audit_does_not_bind_saved_response')
    choice = validate_choice(original, request, response, prepared, lineage)
    result, context = read_json(source_files['result.json']), read_json(source_files['context.json'])
    if (result.get('verifier_backend') != 'actual_ASAP_RedMax' or result.get('model_decision_fixture') is not True
            or result.get('controlled_initial_probe') is not False or result.get('run_id') != original['request']['run_id']):
        raise ValueError('source_fixture_physics_provenance_invalid')
    refs = [ref for ref in original['features']['constraint_references'] if ref.get('origin') == 'current_episode_measurement']
    if not refs:
        raise ValueError('source_request_missing_current_measured_failure')
    event_files = {local_path(ref['evidence_origin']['source']['ref']) for ref in refs}
    if len(event_files) != 1:
        raise ValueError('source_request_must_bind_one_failure_event')
    event_file = next(iter(event_files))
    event_file.relative_to(source_run / 'paths')
    event = read_json(event_file)
    trajectory_file = relative(source_run, event['trajectory']['relative_path'])
    if sha(trajectory_file) != event['trajectory']['sha256']:
        raise ValueError('source_failed_trajectory_hash_mismatch')
    trajectory = read_json(trajectory_file)
    validate_failure(event, trajectory, original, context)
    records = [r for r in load_snapshot(source_files['constraint_memory.json'])['records'] if r['evidence_event_id'] == event['event_id']]
    expected_refs = FrozenConstraintMemory(records).retrieve(context, original['request']['assembled_parts'],
                        original['request']['candidate_ids'], max_records=12)['records']
    expected_refs = {ref['record_id']: dict(ref, origin='current_episode_measurement') for ref in expected_refs}
    for ref in refs:
        origin = ref['evidence_origin']
        if (ref != expected_refs.get(ref['record_id']) or origin['event_sha256'] != digest(event)
                or origin['source']['sha256'] != sha(event_file) or ref['source_scope'] != event['scope']):
            raise ValueError('source_constraint_not_derived_from_bound_failure_measurement')
    outcomes = [o for o in read_json(source_files['path_outcomes.json'])['outcomes']
                if o.get('request_id') == original['request']['request_id']]
    if (len(outcomes) != 1 or outcomes[0].get('failed_evidence_id') != event['event_id']
            or outcomes[0].get('failed_path') != event['trajectory']['relative_path']
            or outcomes[0].get('native_six_axis_failure_recovery') is not True
            or outcomes[0].get('single_direction_failure_recovery') is not False):
        raise ValueError('failure_not_linked_to_original_natural_trigger')
    source_outcome = outcomes[0]
    trigger = [e for e in jsonlines(source_files['path_events.jsonl']) if e.get('kind') == 'repair_trigger'
               and e.get('call_id') == source_outcome['parent_call_id']]
    if (len(trigger) != 1 or trigger[0].get('trigger') != 'native_six_axis_search_returned_none'
            or trigger[0].get('part_move') != event['scope']['part_move']
            or trigger[0].get('parts_rest') != event['scope']['parts_rest']):
        raise ValueError('actual_native_failure_trigger_not_found')
    files = list(source_files.values()) + list(diagnostic_files.values()) + [original_file, event_file, trajectory_file]
    return {'request': request, 'response': response, 'choice': choice, 'scope': event['scope'],
            'source_outcome': source_outcome, 'source_failed_event': event, 'source_probe': trajectory,
            'input_hashes': {str(path): sha(path) for path in files}}


def base_report(bundle):
    return {'schema_version': 1, 'status': 'RUNNING', 'API_calls': 0, 'actual_physics': False,
            'full_DFS': False, 'selected_plan_edge': False, 'solver_not_resumed': True,
            'recorded_failure_state_component_test': True, 'source_fixture_result_relabelled': False,
            'hard_pruning': False, 'physics_result_reuse': False, 'performance_comparison': 'NOT_RUN',
            'path_check_calls': 0, 'stability_check_calls': 0, 'path_success': None,
            'stability_success': None, 'parts_fix': None, 'feasible': False,
            'source_decision_request_id': bundle['source_outcome']['request_id'],
            'new_decision_request_id': bundle['request']['request']['request_id'],
            'source_failed_evidence_id': bundle['source_failed_event']['event_id'],
            'decision': deepcopy(bundle['choice']), 'used_skill_evidence_ids': bundle['choice']['used_skill_evidence_ids'],
            'source_fixture_path_success': bundle['source_outcome']['path_success'],
            'source_fixture_second_axis': bundle['source_outcome']['decision']['second_axis'],
            'current_second_axis': bundle['choice']['second_axis'],
            'source_failed_probe_not_reexecuted': True, 'candidate_diagnostic_response_consumed': False,
            'source_skill_card_citation_claim': bool(bundle['choice']['used_skill_evidence_ids'])}


def execute_physics(bundle, root, assembly, output, report):
    from remote_rank_client import checked_path
    from independent_features import describe_task, _source_hashes
    from task_memory import task_context
    root, assembly = checked_path(root, run_only=False), checked_path(assembly, run_only=False)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import plan_sequence.physics_planner as physics_module
    from plan_sequence.physics_planner import MultiPartPathPlanner
    from plan_sequence.feasibility_check import get_stable_plan_1pose_serial
    from path_repair import execute_two_leg
    from path_evidence import collect_endpoint, observed_collision_thresholds
    scope = deepcopy(bundle['scope'])
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', 'HEAD', '--'],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    actual_context = task_context(describe_task(root, assembly, scope['base_part'],
                                  scope['task_context']['task']['solver_settings']))
    if canonical(actual_context) != canonical(scope['task_context']):
        raise ValueError('fresh_task_assets_pose_settings_geometry_mismatch')
    actual_solver = _solver_hashes()
    if (set(scope['physics']['source_and_binary_hashes']) != EXPECTED_SOLVER_KEYS
            or actual_solver != scope['physics']['source_and_binary_hashes']
            or hashlib.sha256(json.dumps(actual_solver, sort_keys=True).encode()).hexdigest() != scope['solver_fingerprint']):
        raise ValueError('fresh_solver_binary_controller_observer_mismatch')
    def caches():
        return {p.name: sha(checked_path(p, run_only=False)) for p in sorted(assembly.iterdir()) if p.suffix.lower() == '.sdf'}
    inputs_before, caches_before = _source_hashes(assembly), caches()
    if caches_before != scope['physics'].get('sdf_cache_sha256'):
        raise ValueError('fresh_SDF_cache_mismatch')
    report.update(task_context_equal=True, source_binary_controller_hashes_equal=True, source_sdf_cache_hashes_equal=True)
    try:
        # Reinitialize this isolated component, never claim restoration of the
        # original DFS RNG state (only its digest was recorded).
        random.seed(scope['seed']); np.random.seed(scope['seed']); torch.manual_seed(scope['seed'])
        pose = None if scope['pose'] is None else np.asarray(scope['pose'], dtype=float)
        physics, support = scope['physics'], scope['support_policy']
        if (type(support) is not dict or support.get('stability_function') != 'upstream_get_stable_plan_1pose_serial'
                or 'timeout' not in support or support.get('allow_gap') is not False
                or type(support.get('max_grippers')) is not int or support['max_grippers'] < 2):
            raise ValueError('unsupported_recorded_support_policy')
        with readonly_sdf_cache(physics_module) as cache_io:
            with observed_collision_thresholds(MultiPartPathPlanner):
                planner = MultiPartPathPlanner(str(root / 'assets'), str(assembly), scope['parts_rest'], scope['part_move'],
                    parts_removed=physics['parts_removed_in_path_simulation'], pose=pose,
                    force_mag=physics['force_mag'], save_sdf=True)
        report['path_cache_io'] = cache_io
        if (planner.frame_skip != physics['frame_skip'] or planner.max_time != physics['max_time']
                or planner.min_sep != physics['min_sep'] or planner._evidence_col_th != physics['col_th_dict']
                or not np.allclose(planner.get_state().q, scope['initial_q'], rtol=RTOL, atol=ATOL)):
            raise ValueError('fresh_physics_threshold_or_initial_state_mismatch')
        report['collision_thresholds_equal'] = True
        report['rng_restored_from_source'] = False
        scope['rng_before'] = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()), protocol=4)
                                            + torch.get_rng_state().numpy().tobytes()).hexdigest()
        scope['collector_version'] = 'saved_live_choice_component_endpoint_v1'
        report.update(actual_physics=True, path_check_calls=1)
        started = time.perf_counter()
        trajectory = execute_two_leg(planner, AXES['+X'], AXES[bundle['choice']['second_axis']], 50,
                                     min_sep=bundle['source_probe']['controller']['min_sep_argument'])
        report['path_seconds'] = time.perf_counter() - started
        if not np.allclose(trajectory['path'][0], bundle['scope']['initial_q'], rtol=RTOL, atol=ATOL):
            raise ValueError('reset_start_differs_from_recorded_failure_state')
        scope['action_schedule'] = deepcopy(trajectory['controller']['actions'])
        scope['initial_q'] = deepcopy(trajectory['geometry']['start_q'])
        write_new(output / 'trajectory.json', trajectory)
        identity = bundle['request']['request']['request_id'] + ':new-component-endpoint'
        evidence = collect_endpoint(planner, trajectory['raw_success'], trajectory['path'], scope,
                                    identity, 'trajectory.json', sha(output / 'trajectory.json'))
        write_new(output / 'evidence.json', evidence)
        records = records_from_evidence(actual_context, [evidence], str(output / 'evidence.json'), sha(output / 'evidence.json'))
        save_snapshot(records, output / 'constraint_memory.json')
        report.update(path_success=trajectory['raw_success'], geometry=trajectory['geometry'],
            controller=trajectory['controller'], stored_conditional_constraints=len(records),
            trajectory_sha256=sha(output / 'trajectory.json'), evidence_sha256=sha(output / 'evidence.json'),
            endpoint_geometry_queries=evidence['measurements']['geometry_queries'],
            new_scope=scope, path_state_start_matches_source=bool(np.allclose(trajectory['path'][0], bundle['scope']['initial_q'], rtol=RTOL, atol=ATOL)))
        if trajectory['raw_success']:
            report['stability_check_calls'] = 1
            started = time.perf_counter()
            with readonly_sdf_cache(physics_module) as stable_io:
                fixed, evaluations = get_stable_plan_1pose_serial(str(root / 'assets'), str(assembly),
                    scope['parts_rest'], scope['base_part'], pose, max_fix=support['max_grippers'] - 1,
                    save_sdf=True, timeout=support['timeout'], allow_gap=False, debug=0, render=False, return_count=True)
            report.update(stability_success=fixed is not None, parts_fix=fixed, stability_cache_io=stable_io,
                          internal_stability_evaluations=evaluations, stability_seconds=time.perf_counter() - started)
        report['feasible'] = report['path_success'] is True and report['stability_success'] is True
        report['completed_noncollinear_recovery'] = (report['feasible'] and trajectory['controller']['switch_count'] == 1
                                                    and trajectory['geometry']['noncollinear'] is True)
        report['status'] = 'PASS' if report['feasible'] else 'UNKNOWN'
    finally:
        report.update(input_files_unchanged=_source_hashes(assembly) == inputs_before,
                      sdf_cache_unchanged=caches() == caches_before,
                      source_and_binary_unchanged=_solver_hashes() == actual_solver)
        if not all(report[k] for k in ('input_files_unchanged', 'sdf_cache_unchanged', 'source_and_binary_unchanged')):
            report['status'] = 'ERROR'


def run(args):
    output = local_path(args.output_dir)
    source, diagnostic = local_path(args.source_run), local_path(args.decision_run)
    if any(output == p or p in output.parents or output in p.parents for p in (source, diagnostic)):
        raise ValueError('component_output_must_be_separate_from_existing_inputs')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'ERROR', 'API_calls': 0, 'actual_physics': False,
              'full_DFS': False, 'selected_plan_edge': False, 'recorded_failure_state_component_test': True,
              'path_check_calls': 0, 'stability_check_calls': 0}
    started, bundle = time.perf_counter(), None
    try:
        bundle = load_choice(source, diagnostic)
        report = base_report(bundle)
        write_new(output / 'input_hashes_before.json', bundle['input_hashes'])
        if bundle['choice']['skill_id'] == 'no_repair':
            report.update(status='SKIPPED', skip_reason='live_model_selected_no_repair')
        else:
            execute_physics(bundle, args.asap_root, args.assembly_dir, output, report)
    except Exception as error:
        report.update(status='ERROR', error_type=type(error).__name__, error=str(error))
    finally:
        if bundle is not None:
            try:
                after = {str(local_path(path)): sha(local_path(path)) for path in bundle['input_hashes']}
                write_new(output / 'input_hashes_after.json', after)
                report['saved_source_and_decision_files_unchanged'] = after == bundle['input_hashes']
                if not report['saved_source_and_decision_files_unchanged']:
                    report['status'] = 'ERROR'
            except Exception as error:
                report.update(status='ERROR', saved_source_and_decision_files_unchanged=False,
                              preservation_error_type=type(error).__name__)
        report['elapsed_seconds'] = time.perf_counter() - started
        write_new(output / 'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('asap-root', 'assembly-dir', 'source-run', 'decision-run', 'output-dir'):
        parser.add_argument('--' + flag, required=True, type=Path)
    report = run(parser.parse_args(argv))
    print(json.dumps({k: report.get(k) for k in ('status', 'API_calls', 'path_success', 'stability_success', 'feasible')}))
    return 1 if report['status'] == 'ERROR' else 0


if __name__ == '__main__':
    raise SystemExit(main())
