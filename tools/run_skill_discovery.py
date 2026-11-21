"""Experience -> bounded program -> actual cases -> scoped registry -> freeze.

Default preview never reads credentials or runs physics. Fixture mode tests the
same physical pipeline with an explicitly HANDWRITTEN program. Live mode needs
an approved config and --allow-live-api; failed generation is never replaced by
the fixture. Every invocation requires a new output directory.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import sys

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'worker'))
sys.path.insert(0, str(ROOT / 'tools'))

from run_integrated_repair import checked, WSL, PYTHON
from run_luna_experiment import read, write, sha, linux, validate_config
from run_recorded import run_recorded
from orchestrator.skill_case import (physical_context, request_after_trigger, validate_trigger_scope,
                                    matches_diagnostic_phase)
from orchestrator.skill_discovery import SkillDiscoveryService
from orchestrator.skill_program import program_hash, step_program, validate_program
from orchestrator.skill_refinement import summarize_feedback, audit_revision, summarize_motion_alignment
from worker.task_memory import digest
from worker.stability_diagnostics import project_stability_diagnostics


FIXTURE_PROGRAM = {'language': 'recovery_rules_v1', 'rules': [{
    'when': ['last_path_failed'],
    'select': {'primitive': 'turn_force_two_leg_v1', 'part': 'same_as_last',
               'pose': 'same_as_last', 'direction': 'zp', 'second_direction': 'xp',
               'support': 'same_as_last'}, 'rank': []}]}


def preview_config():
    return dict(enabled=False, model='gpt-6-luna', base_url='https://api.openai.com/v1',
                api_key_env='OPENAI_API_KEY', max_calls=0, max_usd=0,
                input_usd_per_million=.10, output_usd_per_million=.50,
                max_output_tokens=2048, reasoning_effort='none', request_overhead_tokens=2048)


def _checked_stability_failure_stage(raw):
    upstream = raw.get('raw_upstream') or {}
    return (raw.get('backend') == 'ASAP_RedMax'
            and raw.get('status') in ('UNKNOWN', 'REJECTED')
            and raw.get('path_check_calls') == 1 and raw.get('stability_check_calls') == 1
            and raw.get('stability_mode') == 'one_exact_selected_support_configuration'
            and raw.get('path_success') is upstream.get('path_success') is True
            and raw.get('stability_success') is upstream.get('stability_success') is False)


def source_case(spec):
    report_path = checked(ROOT / spec['report'], True)
    report = read(report_path)
    index = spec['attempt_index']
    if type(index) is not int or not 1 <= index <= len(report['attempts']):
        raise ValueError('invalid_declared_attempt_index')
    attempt = report['attempts'][index - 1]
    directory = report_path.parent / ('attempt_%06d' % index)
    physical = read(directory / 'request.json')
    raw = read(directory / 'result.json')
    if raw.get('backend') != 'ASAP_RedMax' or raw['chosen'] != attempt['choice']:
        raise ValueError('actual_source_artifact_required')
    request = next(d['request'] for d in report['decisions']
                   if d['request']['request_id'] == attempt['request_id'])
    choice = deepcopy(physical['choice'])
    choice['evidence_ids'] = []
    override = spec.get('trigger_override', {})
    if set(override) - {'direction_id', 'second_direction_id', 'skill_id', 'support_ids'}:
        raise ValueError('unsupported_trigger_override')
    choice.update(deepcopy(override))
    # Historical traces can precede an already-installed executor correction.
    # Precommit CURRENT reviewed local controller/observer/executor hashes;
    # keep the pinned upstream and binary hashes, then require fresh results to
    # match all of them. Never rewrite the historical result or its hashes.
    expected_sources = deepcopy(raw['source_hashes'])
    local_sources = {'explicit_executor_sha256': 'explicit_action_physics.py',
                     'controller_sha256': 'path_repair.py',
                     'observer_sha256': 'path_evidence.py',
                     'stability_observer_sha256': 'stability_diagnostics.py'}
    for key, filename in local_sources.items():
        if key in expected_sources:
            expected_sources[key] = sha(ROOT/'worker'/filename)
    prepared = dict(spec=spec, request=request, context=physical['context'],
                choice=choice, raw=raw, source_path=str(directory),
                expected_source_hashes=expected_sources,
                source_report_sha256=sha(report_path))
    if 'pose_yaw_degrees' in spec:
        from orchestrator.python_skill_pose import apply_declared_yaw
        prepared = apply_declared_yaw(prepared, spec['pose_yaw_degrees'])
    return prepared


def _frozen_synthesis_memory(manifest, source, source_path, trace_evidence):
    """Expose frozen record IDs only after binding them to the source episode.

    Trace-event and memory-record IDs name the SAME observation. Neither a
    second physical trial nor a source-to-target ID correspondence is created.
    This optional path does not change historical manifest behavior.
    """
    from orchestrator.explicit_action_memory import load_snapshot
    from worker.explicit_action_contract import action_scope, validate_request, validate_response
    memory_path = checked(ROOT / manifest['experience_memory'], True)
    expected = manifest.get('frozen_episode_memory')
    if (type(expected) is not dict
            or checked(ROOT / expected.get('path', ''), True) != memory_path
            or sha(memory_path) != expected.get('file_sha256')):
        raise ValueError('synthesis_memory_file_provenance_mismatch')
    before = sha(memory_path)
    snapshot = load_snapshot(memory_path)
    if snapshot['content_sha256'] != expected.get('content_sha256'):
        raise ValueError('synthesis_memory_content_provenance_mismatch')
    attempts, decisions = source['attempts'], source['decisions']
    by_request = {d['request']['request_id']: d for d in decisions}
    by_attempt = {a['request_id']: a for a in attempts}
    by_record = {r['source_request_id']: r for r in snapshot['records']}
    if (len(by_request) != len(decisions) or len(by_attempt) != len(attempts)
            or len(by_record) != len(snapshot['records'])
            or set(by_record) != set(by_attempt)):
        raise ValueError('synthesis_memory_episode_identity_mismatch')
    trace_by_event = {e['evidence_id']: e for e in trace_evidence}
    memory_evidence, links, excluded, raw_file_hashes = [], [], [], {}
    for index, attempt in enumerate(attempts, 1):
        decision = by_request.get(attempt['request_id'])
        if decision is None:
            raise ValueError('synthesis_memory_missing_decision')
        request = validate_request(decision['request'])
        response = validate_response(request, decision['response'])
        record = by_record[attempt['request_id']]
        raw, observed = attempt['raw'], record['result']
        if (record['source_run_id'] != request['run_id']
                or record['source_state_id'] != request['state_id']
                or record['source_action_id'] != response['action_id']
                or record['choice'] != response['choice']
                or record['choice'] != attempt['choice']
                or record['scope'] != action_scope(request, response['choice'])
                or request['task_context']['task']['assembly_id'] != manifest['experience_task']
                or attempt['state_id'] != request['state_id']
                or attempt['action_id'] != response['action_id']
                or attempt['remaining_parts'] != request['remaining_parts']
                or checked(ROOT / observed['evidence_ref'], True) != source_path
                or observed['raw'] != raw or observed['status'] != attempt['status']
                or observed['path_success'] is not raw.get('path_success')
                or observed['stability_success'] is not raw.get('stability_success')):
            raise ValueError('synthesis_memory_report_binding_mismatch')
        directory = checked(source_path.parent / ('attempt_%06d' % index), True)
        physical_path, raw_path = directory / 'request.json', directory / 'result.json'
        physical = read(physical_path)
        if (read(raw_path) != raw or physical.get('choice') != record['choice']
                or physical.get('context', {}).get('task_context') != request['task_context']
                or physical['context'].get('request_id') != request['request_id']
                or physical['context'].get('action_id') != response['action_id']):
            raise ValueError('synthesis_memory_actual_file_binding_mismatch')
        raw_file_hashes[str(raw_path)] = sha(raw_path)
        raw_file_hashes[str(physical_path)] = sha(physical_path)
        if raw.get('backend') != 'ASAP_RedMax':
            excluded.append(dict(evidence_id=record['evidence_id'],
                source_event_id=attempt['event_id'], reason='not_actual_ASAP_observation'))
            continue
        if observed['verifier_origin'] != 'actual_ASAP_RedMax':
            raise ValueError('synthesis_memory_actual_origin_mismatch')
        if raw.get('path_check_calls') == 1 and type(raw.get('path_success')) is bool:
            validate_trigger_scope(request, raw, record['choice'])
        trace = trace_by_event.get(attempt['event_id'])
        if trace is None or record['evidence_id'] in trace_by_event:
            raise ValueError('synthesis_memory_event_identity_collision')
        trace.update(evidence_origin='episode_trace', frozen_memory_evidence_id=record['evidence_id'])
        memory_evidence.append(dict(evidence_id=record['evidence_id'],
            evidence_origin='frozen_episode_memory', source_event_id=attempt['event_id'],
            same_observation_as_event_id=attempt['event_id'],
            source_task=manifest['experience_task'], source_run_id=record['source_run_id'],
            source_request_id=record['source_request_id'], source_state_id=record['source_state_id'],
            source_action_id=record['source_action_id'], source_scope_sha256=digest(record['scope']),
            record_sha256=record['record_sha256'],
            parent_failure_evidence_id=record['parent_failure_evidence_id'],
            remaining_parts=deepcopy(record['scope']['remaining_parts']),
            selected_pose_matrix=deepcopy(record['scope']['pose_matrix']),
            base_part=record['scope']['base_part'], max_grippers=record['scope']['max_grippers'],
            choice=deepcopy(record['choice']), status=observed['status'],
            path_success=observed['path_success'], stability_success=observed['stability_success'],
            stability_diagnostics=project_stability_diagnostics(raw.get('stability_diagnostics')),
            physically_checked_stability_failure=_checked_stability_failure_stage(raw),
            deadline_exceeded=attempt['deadline_exceeded'],
            accepted_for_transition=attempt['accepted_for_transition'],
            raw_result_sha256=raw_file_hashes[str(raw_path)],
            reference_only=True, current_part_id_mapping=None,
            failure_cause=None, all_paths_impossible=False, physics_reuse_allowed=False))
        links.append(dict(source_event_id=attempt['event_id'], memory_evidence_id=record['evidence_id']))
        if 'execution' in observed:
            from orchestrator.execution_observation import from_attempt, project_execution
            if observed['execution'] != from_attempt(attempt):
                raise ValueError('synthesis_memory_execution_binding_mismatch')
            memory_evidence[-1].update(project_execution(observed))
            trace.update(project_execution(observed))
    if sha(memory_path) != before or any(sha(path) != value for path, value in raw_file_hashes.items()):
        raise ValueError('synthesis_memory_or_raw_changed_during_read')
    return memory_evidence, dict(path=str(memory_path), file_sha256=before,
        content_sha256=snapshot['content_sha256'], record_count=snapshot['record_count'],
        exposed_record_count=len(memory_evidence), excluded_records=excluded,
        exposed_evidence_ids=[e['evidence_id'] for e in memory_evidence],
        linked_trace_events=links, binding_checked=True, hard_pruning=False,
        source_to_target_part_mapping=None, physics_reuse_allowed=False,
        duplicate_views_of_same_observations=True, additional_physical_observation_count=0,
        description='Frozen record IDs are citeable stored experience; their linked trace event IDs describe the same observations, not independent extra trials.')


def synthesis_packet(manifest, run_id):
    source_path = checked(ROOT / manifest['experience_report'], True)
    source = read(source_path)
    from orchestrator.skill_trace_summary import summarize_recovery_pairs
    paired_observations = summarize_recovery_pairs(source)
    failure_scope = manifest.get('synthesis_failure_scope')
    if failure_scope not in (None, 'dynamic_stability_failed', 'initial_disconnected'):
        raise ValueError('unsupported_synthesis_failure_scope')
    if failure_scope and not manifest.get('experience_memory'):
        raise ValueError('stability_synthesis_requires_bound_frozen_memory')
    evidence = []
    for item in source['attempts']:
        raw = item['raw']
        if raw.get('backend') != 'ASAP_RedMax':
            continue
        evidence.append(dict(evidence_id=item['event_id'], source_task=manifest['experience_task'],
            remaining_parts=item['remaining_parts'], choice=item['choice'],
            path_success=raw.get('path_success'), stability_success=raw.get('stability_success'),
            status=item['status'], deadline_exceeded=item['deadline_exceeded'],
            accepted_for_transition=item['accepted_for_transition'],
            measured_endpoint=raw.get('endpoint_observation'),
            stability_diagnostics=project_stability_diagnostics(raw.get('stability_diagnostics')),
            physically_checked_stability_failure=_checked_stability_failure_stage(raw),
            raw_stability_parts=raw.get('raw_upstream', {}).get('parts_fall'),
            reference_only=True, current_part_id_mapping=None,
            failure_cause=None, all_paths_impossible=False))
    packet = dict(run_id=run_id, request_id=run_id+':synthesis', state_id='offline-source-trace',
        operation='synthesize', evidence=evidence,
        context={'purpose': 'propose_conditional_recovery_procedure',
                 'automatically_extracted_same_condition_pairs': paired_observations,
                 'development_only': True, 'full_plan_success_not_implied': True,
                 'must_abstain_without_matching_current_failure_feedback': True,
                 'no_coverage_or_performance_claim_from_registration': True,
                 'language_semantics': {
                     'when': 'AND of predicates; FIRST matching rule with a legal action RETURNS immediately. Later rules are NOT physical fallback stages. Each new call starts at rule zero with fresh latest feedback.',
                     'last': 'latest exact-current-state actual nonlate episode feedback, never source IDs',
                     'direction': 'relative selectors compare to previous primary direction',
                     'second_direction': 'relative selectors compare to previous SECOND direction. A prior single_force has null second direction: opposite_last and perpendicular_last cannot match it. any allows the legal perpendicular options to the NEW first direction; literals xp/xn/yp/yn/zp/zn are world axes.',
                     'rank': 'lexicographic listed metrics, skip missing observations, canonical tie break; empty list takes first match',
                     'no_match': 'abstain without pruning; no physical execution',
                     'support': 'same_as_last is a relative current-state set; never copy source literal IDs'},
                 'part_ids_are_source_labels_not_current_task_ids': True},
        existing_skills=[{'primitive_id': 'single_force_v1', 'description': 'one fixed force direction'},
                        {'primitive_id': 'turn_force_two_leg_v1',
                         'description': 'fixed first direction then perpendicular second at step 50'}])
    if manifest.get('experience_memory') is not None:
        stored, provenance = _frozen_synthesis_memory(manifest, source, source_path, packet['evidence'])
        packet['evidence'].extend(stored)
        packet['context']['frozen_episode_memory'] = provenance
    if failure_scope:
        initial_contact = failure_scope == 'initial_disconnected'
        target_phase = 'initial_disconnected' if initial_contact else 'dynamic_failure'
        scoped = [item for item in packet['evidence']
                  if item.get('path_success') is True and item.get('stability_success') is False
                  and item.get('physically_checked_stability_failure') is True
                  and item.get('status') in ('UNKNOWN', 'REJECTED')
                  and item.get('deadline_exceeded') is False
                  and (item.get('stability_diagnostics') or {}).get('phase') == target_phase]
        if not scoped:
            raise ValueError('no_bound_' + target_phase + '_observation')
        packet['context']['target_failure_scope'] = dict(
            kind=failure_scope, evidence_ids=[item['evidence_id'] for item in scoped],
            initial_disconnected_not_in_scope=not initial_contact,
            description=('Develop a conditional recovery for a zero-step initial contact-graph disconnection after a successful path. The original checker found degree-zero FREE remaining parts in its part-to-part/fixed-part contact graph; ground is not a graph node. This is NOT observed falling, dynamic displacement or a collision. A rigid common pose change is not evidence that part-to-part connectivity changes. Holding one isolated part does not connect the other isolated parts. Do not require all isolated parts to fit the support budget merely to consider a different mover. Each proposed action requires fresh path and stability checks.' if initial_contact else
                'Develop a conditional recovery for measured dynamic stability failure after a successful path. Initial contact-graph disconnection and path failures are different observations. No successful recovery is supplied or implied; any proposed change remains a hypothesis until fresh validation.'),
            observations_are_not_causes=True, literal_source_ids_forbidden=True)
        packet['context']['language_semantics']['support'] += (
            '; contains_observed_unstable binds to dynamic-failure part IDs from the NEW exact-current-state feedback, not stored source IDs. It selects only legal support sets within the existing gripper budget. Support selection is a hypothesis requiring fresh path and stability checks.')
        packet['context']['language_semantics']['part'] = (
            'observed_unstable selects the MOVER from current legal candidates measured unstable in the latest exact-current-state dynamic failure. This is different from support:contains_observed_unstable, which HOLDS an unstable part while a different part moves. any is canonical candidate enumeration and does not select by instability. None of these selectors supplies a physics verdict or transfers source IDs.')
        if initial_contact:
            from orchestrator.initial_contact_trace_summary import summarize_initial_contact_transitions
            packet['context']['initial_contact_observed_transitions'] = summarize_initial_contact_transitions(source)
            packet['context']['language_semantics'].update(
                initial_contact_predicates='last_initial_disconnected and has_initial_disconnected_parts bind only validated zero-step current-state initial-contact feedback. has_unstable_parts is dynamic-only and is FALSE for initial disconnection. Every rule in this scoped proposal must include at least one initial-contact predicate.',
                part='observed_initial_disconnected chooses a MOVER from the current candidates observed isolated in the latest exact-state initial contact graph. This differs from holding a part as support. No source IDs transfer. any uses canonical enumeration; it does not choose by isolation. Choosing an isolated mover remains a hypothesis, not guaranteed recovery.',
                support='contains_observed_initial_disconnected means at least ONE current observed isolated part is held; it does not hold ALL isolated parts. Legal supports remain limited by max_grippers minus the mover. Holding one isolated part does not establish contact for the others. empty/nonempty/same_as_last retain their literal meanings.')
            packet['context']['card_scope_rule'] = 'Describe only the predicates/selectors actually encoded. Do not claim a single-force, empty-support, direction or pose restriction unless the program encodes it. Natural-language applicability is unverified; executable predicates control applicability.'
            packet['context']['coordinate_semantics'] = {
                'pose': 'The recorded pose matrix maps original CAD coordinates into world coordinates. A fixed world force can correspond to different CAD directions after a pose change. Pose/direction facts alone do not prove path clearance.',
                'direction_selectors': 'xp/xn/yp/yn/zp/zn are WORLD axes. same_as_last/opposite_last/perpendicular_last compare to the last WORLD force, not a CAD axis or a surface normal. Current IR has no pose-matrix-dependent direction expression.',
                'cad_metrics': 'part_extent_z is an extent in the original CAD frame, not world height under the chosen pose. part_volume is invariant under rigid rotation but close values can order by numerical differences. Neither metric is a physics verdict; material density is unknown.',
                'pose_selector_limits': 'any/same_as_last/different_from_last select only from the current pose catalog. any uses canonical enumeration; it does not optimize contact or gravity.',
                'recommended_program': None}
    return packet


def actual_action(packet, folder, *, source_budget=None):
    extra=[];watchdog=55
    if source_budget is not None:
        from tools.python_source_budget import validate_binding
        seconds=validate_binding(packet['context'],source_budget)
        watchdog=math.ceil(seconds)+20
        extra=['--source-request',linux(checked(source_budget['source_request_path'],True)),
               '--source-request-sha256',source_budget['source_request_sha256']]
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'input.json', packet)
    if source_budget is not None:write(folder/'source_budget.json',source_budget)
    argv = WSL + ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', str(watchdog)+'s',
                  PYTHON, '-B', '-u', linux(ROOT / 'worker/run_skill_validation_action.py'),
                  '--input', linux(folder / 'input.json'), '--output', linux(folder / 'physics'),*extra]
    command = run_recorded(argv, ROOT, folder / 'execute')
    if command['exit_code'] != 0 or not (folder / 'physics/result.json').is_file():
        raise RuntimeError('actual_worker_failed_see_command_logs')
    return read(folder / 'physics/result.json')


def executable_card(program, original_card):
    """A new card states executable scope, not an unverified prose promise.

    The provider response remains separately hash-bound and unchanged. This
    never edits an existing library or changes interpreter/physics semantics.
    """
    from orchestrator.skill_behavior_contract import program_description
    description = program_description(program)
    return dict(name='Generated recovery ' + program_hash(program)[:12],
        applicability='The executable predicates and selectors in description determine applicability; this card supplies no physical verdict.',
        description=description['rendered_text'],
        hypothesis='This executable procedure is a hypothesis; fresh physical validation is required.',
        evidence_ids=deepcopy(original_card['evidence_ids']),
        executable_description=dict(schema_version=1, program_sha256=description['program_sha256'],
                                    rule_count=len(description['rules']), source='validated_program'),
        provider_prose_status='unverified_preserved_in_provider_response')


def build_refinement_feedback(prior, result, previous_manifest):
    """Rehydrate only debug artifacts, retaining failed revisions' evidence.

    Admission outcomes never enter synthesis. Saved projection fields are not
    authoritative: physical results are rebound to the saved request before
    deriving a nonlate failure for the behavioral novelty check.
    """
    prior = checked(prior, True)
    debug_ids = {c['case_id'] for c in previous_manifest['cases'] if c['phase'] == 'debug'}
    previous_packet = read(prior/'generation_packet.json')
    inherited = [deepcopy(c) for c in previous_packet.get('validation_feedback', [])
                 if c.get('case_id') in debug_ids and c.get('phase') == 'debug'
                 and c.get('source_run')]
    fresh = [dict(deepcopy(c), phase='debug', source_run=str(prior))
             for c in result.get('cases', []) if c['case_id'] in debug_ids]
    unique = {}
    for item in inherited + fresh:
        case_id = item['case_id']
        if not case_id.replace('_', '').isalnum():
            raise ValueError('simple_case_id_required')
        source = checked(item['source_run'], True)
        source_manifest = read(source/'precommitted_cases.json')
        if not any(c['case_id'] == case_id and c['phase'] == 'debug'
                   for c in source_manifest['cases']):
            raise ValueError('feedback_must_be_predeclared_debug')
        directory = source/'cases'/case_id
        request_path = directory/'program_request.json'
        step_path = directory/'program_step.json'
        # A process error before an observation remains recorded, but supplies
        # no imagined physics evidence to synthesis or the offline gate.
        if not request_path.is_file() or not step_path.is_file():
            continue
        request = read(request_path)
        proposal = read(source/'provider_proposal.json')
        if proposal.get('origin') != 'actual_llm' or proposal.get('status') != 'proposed':
            raise ValueError('development_feedback_requires_actual_proposal')
        item['observed_program'] = validate_program(proposal['program'])
        item['current_observation_request'] = request
        item['interpreter_trace'] = read(step_path)
        for stage in ('trigger', 'recovery'):
            item.pop(stage, None)
            raw_path = directory/stage/'physics/result.json'
            if not raw_path.is_file():
                continue
            worker_request = read(directory/stage/'physics/request.json')
            raw = read(raw_path)
            validate_trigger_scope(request, raw, worker_request['choice'])
            elapsed, limit = raw.get('elapsed_seconds'), worker_request['context'].get('remaining_wall_seconds')
            if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
                    or type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit <= 45):
                raise ValueError('invalid_development_elapsed_or_budget')
            item[stage] = {key: raw.get(key) for key in (
                'status', 'chosen', 'path_success', 'stability_success',
                'endpoint_observation', 'raw_upstream', 'backend', 'elapsed_seconds')}
            item[stage]['stability_diagnostics'] = project_stability_diagnostics(raw.get('stability_diagnostics'))
            item[stage].update(deadline_exceeded=elapsed > limit,
                               bound_actual_scope_checked=True,
                               evidence_id=worker_request['context']['event_id'],
                               raw_result_sha256=sha(raw_path),
                               raw_result_path=str(raw_path))
        unique[(case_id, request['request_id'])] = item
    feedback = list(unique.values())[-6:]
    if not feedback:
        raise ValueError('refinement_requires_bound_debug_observations')
    return feedback


def run_case(prepared, case, program, output, run_id):
    """Fresh failed physical trigger, stateless execution, fresh verification."""
    seconds=case.get('case_wall_seconds',30)
    binding=case.get('source_budget')
    wall_limit=45;clock_args={};physical_args={};worker_args={}
    if binding is not None:
        from tools.python_source_budget import source_budget
        if binding!=source_budget(prepared) or seconds!=binding['seconds']:
            raise ValueError('precommitted_source_budget_mismatch')
        wall_limit=seconds
        clock_args=dict(wall_limit=wall_limit)
        physical_args=dict(wall_limit=wall_limit,preserve_stability_policy=True)
        worker_args=dict(source_budget=binding)
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=wall_limit:
        raise ValueError('existing_bounded_45_second_harness_required')
    folder = output / 'cases' / case['case_id']
    folder.mkdir(parents=True, exist_ok=False)
    event = run_id + ':' + case['case_id'] + ':trigger'
    template = prepared['request']
    trigger_context = physical_context(prepared['context'], template, prepared['choice'], event_id=event,seconds=seconds,**physical_args)
    raw = actual_action({'context': trigger_context, 'choice': prepared['choice']}, folder / 'trigger',**worker_args)
    if not matches_diagnostic_phase(raw, case.get('expected_diagnostic_phase')):
        raise ValueError('fresh_trigger_diagnostic_phase_mismatch')
    request = request_after_trigger(template, raw, prepared['choice'], run_id=event,
                                    event_id=event,wall_seconds=seconds,**clock_args)
    write(folder / 'program_request.json', request)
    runtime = None
    if case.get('runtime_contract') is not None:
        from orchestrator.lesson_runtime import CONTRACT, bind_execution, execute_bound
        if case['runtime_contract'] != CONTRACT or program['language'] != 'python_policy_v1':
            raise ValueError('unsupported_case_runtime_contract')
        runtime = bind_execution(program, request)
        step = execute_bound(program, request, runtime['extension'])
    else:
        step = step_program(program, request)
    write(folder / 'program_step.json', step)
    selected = dict(case_id=case['case_id'], source_task=case['source_task'],
        program_hash=program_hash(program), program_status='NOT_APPLICABLE',
        program_request_path=str(folder / 'program_request.json'),
        program_request_sha256=sha(folder / 'program_request.json'),
        trigger_result_sha256=sha(folder / 'trigger/physics/result.json'))
    if runtime is not None:
        selected['program_runtime'] = runtime
    artifacts = dict(attempt_dir=None, selection_path=str(folder / 'selection.json'),
                     trigger_attempt_dir=str(folder / 'trigger/physics'))
    if step['response'] is not None:
        choice = step['response']['choice']
        context = physical_context(prepared['context'], request, choice, event_id=event+':recovery',seconds=seconds,**physical_args)
        actual_action({'context': context, 'choice': choice}, folder / 'recovery',**worker_args)
        artifacts['attempt_dir'] = str(folder / 'recovery/physics')
        selected.update(program_status='ACTION', action_sha256=digest(choice),
            worker_request_sha256=sha(folder / 'recovery/physics/request.json'),
            worker_result_sha256=sha(folder / 'recovery/physics/result.json'))
    write(folder / 'selection.json', selected)
    return artifacts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT/'configs/skill_discovery_cases_v1.json')
    p.add_argument('--mode', choices=('preview', 'fixture', 'live'), default='preview')
    p.add_argument('--llm-config', type=Path)
    p.add_argument('--allow-live-api', action='store_true')
    p.add_argument('--ledger', type=Path)
    p.add_argument('--refine-from', type=Path,
                   help='Explicit new proposal using a completed prior live validation; no HTTP retry')
    args = p.parse_args(argv)
    output = checked(args.run_dir, True)
    if output.exists():
        raise FileExistsError('new_run_required')
    live = args.mode == 'live'
    if live != args.allow_live_api or (live and not args.llm_config):
        raise ValueError('approved_config_and_explicit_live_flag_required')
    if live and not args.ledger:
        raise ValueError('shared_approved_ledger_required_for_live_generation')
    config = read(checked(args.llm_config)) if args.llm_config else preview_config()
    if not live and (config.get('enabled') or config.get('max_calls') or config.get('max_usd')):
        raise ValueError('offline_requires_zero_budget')
    manifest = read(checked(args.manifest))
    if len(manifest['cases']) > 6 or len({c['case_id'] for c in manifest['cases']}) != len(manifest['cases']):
        raise ValueError('bounded_unique_cases_required')
    for case in manifest['cases']:
        if not case['case_id'].replace('_', '').isalnum():
            raise ValueError('simple_case_id_required')
    prepared = [source_case(c) for c in manifest['cases']]
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT/'orchestrator').glob('*.py')) + sorted((ROOT/'worker').glob('*.py'))
    sources += [Path(__file__), ROOT/'tools/run_recorded.py', ROOT/'tools/run_integrated_repair.py',
                ROOT/'tools/run_luna_experiment.py']
    hashes = {f.relative_to(ROOT).as_posix(): sha(f) for f in sources}
    write(output/'source_before.json', hashes)
    for name in hashes:
        target = output/'implementation'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked(ROOT/name), checked(target, True))
    write(output/'precommitted_cases.json', manifest)
    ledger = checked(args.ledger, True) if args.ledger else output/'batch_ledger.sqlite'
    write(output/'config.json', dict(mode=args.mode, api=config, ledger=str(ledger), automatic_retries=0,
                                    performance_comparison=False))
    run_id = 'discovery:' + digest(str(output))[:20]
    packet = synthesis_packet(manifest, run_id)
    if args.refine_from:
        prior = checked(args.refine_from, True)
        previous = read(prior/'provider_proposal.json')
        result = read(prior/'result.json')
        previous_manifest = read(prior/'precommitted_cases.json')
        previous_config = read(prior/'config.json')
        if live and previous_config.get('ledger') and checked(previous_config['ledger'], True) != ledger:
            raise ValueError('refinement_must_share_original_approved_ledger')
        if previous.get('origin') != 'actual_llm' or previous.get('status') != 'proposed':
            raise ValueError('refinement_requires_actual_prior_proposal')
        packet.update(operation='refine', previous_program=previous['program'],
                      validation_feedback=build_refinement_feedback(prior, result, previous_manifest))
        # Newly observed failed attempts are citeable evidence too. Preserve
        # their exact source scope rather than attributing them to the original
        # Beam source trace or turning a target observation into a global fact.
        seen_evidence = {item['evidence_id'] for item in packet['evidence']}
        for feedback in packet['validation_feedback']:
            for stage in ('trigger', 'recovery'):
                observed = feedback.get(stage)
                if observed and observed['evidence_id'] not in seen_evidence:
                    packet['evidence'].append(dict(evidence_id=observed['evidence_id'],
                        source_task=feedback['current_observation_request']['task_context']['task'].get('assembly_id'),
                        source_scope_sha256=digest(feedback['current_observation_request']['task_context']),
                        source_phase='debug', observation=deepcopy(observed),
                        reference_only=True, current_part_id_mapping=None,
                        failure_cause=None, all_paths_impossible=False))
                    seen_evidence.add(observed['evidence_id'])
        packet['context']['development_replay_diagnostics'] = summarize_feedback(
            packet['validation_feedback'], previous_program=previous['program'])
        if not packet['context']['development_replay_diagnostics']['valid']:
            raise ValueError('invalid_bound_development_diagnostics')
        packet['context']['source_motion_replay_diagnostics'] = summarize_motion_alignment(
            previous['program'], packet['validation_feedback'],
            packet['context']['automatically_extracted_same_condition_pairs'])
        packet['context']['previous_development_gate'] = result.get('development_gate')
        if manifest.get('synthesis_failure_scope') == 'initial_disconnected':
            prior_packet = read(prior/'generation_packet.json')
            programs = [previous['program']]
            if prior_packet.get('previous_program') is not None:
                programs.append(prior_packet['previous_program'])
            programs.extend(item['observed_program'] for item in packet['validation_feedback']
                            if item.get('observed_program') is not None)
            packet['context']['prior_development_program_hashes'] = sorted({program_hash(p) for p in programs})
            packet['context']['new_program_required'] = (
                'An exact normalized program already submitted is not a new procedure. '
                'Do not resubmit it or only rename the card. Preserve observed debug successes '
                'where justified while changing the executable hypothesis. The returned program '
                'must still pass the same predeclared physical checks; no successful action is prescribed.')
            packet['context']['behavior_contract_version'] = 1
    write(output/'generation_packet.json', packet)
    service = SkillDiscoveryService(config, ledger)
    try:
        preview = service.preview(packet)
        write(output/'api_preview.json', preview)
        if packet['context'].get('behavior_contract_version') == 1:
            from orchestrator.skill_evidence_packet import compact_generation_packet
            compact, compact_audit = compact_generation_packet(packet)
            write(output/'provider_evidence_packet.json', compact)
            write(output/'evidence_compaction_audit.json', compact_audit)
        if args.mode == 'preview':
            print(json.dumps({k: v for k, v in preview.items() if k != 'body'}), flush=True)
            return 0
        from orchestrator.skill_registry import SkillRegistry
        if live:
            from local_api_key import key_environment
            with key_environment():
                validate_config(config, True)
                proposal = service.propose(packet)
            write(output/'provider_proposal.json', proposal)
            if proposal['status'] != 'proposed' or proposal['origin'] != 'actual_llm':
                write(output/'result.json', dict(status='GENERATION_REJECTED', proposal=proposal))
                return 1
            program, card = proposal['program'], proposal['card']
            generation = dict(origin='live_llm', model=config['model'], generation_id=packet['request_id'],
                provider_response_path=str(output/'provider_proposal.json'),
                provider_response_sha256=sha(output/'provider_proposal.json'))
        else:
            program = deepcopy(FIXTURE_PROGRAM)
            card = dict(name='fixture_vertical_turn', applicability='actual same-state failed path',
                        hypothesis='handwritten wiring fixture; NOT LLM-discovered',
                        evidence_ids=[packet['evidence'][0]['evidence_id']])
            generation = dict(origin='fixture_authored', generation_id=run_id)
        if packet['context'].get('behavior_contract_version') == 1 and live:
            from orchestrator.skill_behavior_contract import audit_behavior_claims
            behavior = audit_behavior_claims(program, packet['validation_feedback'],
                                            proposal.get('behavior_claims'))
            write(output/'behavior_contract_audit.json', behavior)
            if not behavior['accepted_for_fresh_validation']:
                write(output/'result.json', dict(status='DEVELOPMENT_REJECTED',
                    generation_origin=generation['origin'], cases=[],
                    reason='behavior_prediction_mismatch', physical_cases_not_run=True,
                    live_generation_tested=True, registered=False))
                return 0
            write(output/'provider_card_unverified.json', card)
            from orchestrator.skill_behavior_contract import program_description
            write(output/'executable_description.json', program_description(program))
            card = executable_card(program, card)
            write(output/'executable_card.json', card)
        else:
            card['description'] = card['name'] + ': ' + card['hypothesis']
        registry = SkillRegistry(output/'registry', fixture=not live)
        candidate = registry.propose(validate_program(program), card, generation)
        identity = candidate['candidate_id']
        cases = [dict(case_id=c['spec']['case_id'], phase=c['spec']['phase'],
                      source_task=c['spec']['source_task'], program_hash=identity,
                      input_hashes=c['raw']['input_hashes'], source_hashes=c['expected_source_hashes'],
                      trigger_kind=c['spec']['trigger_kind'],
                      **({'expected_diagnostic_phase': c['spec']['expected_diagnostic_phase']}
                         if 'expected_diagnostic_phase' in c['spec'] else {})) for c in prepared]
        debug = {'phase': 'debug', 'cases': [c for c in cases if c['phase'] == 'debug']}
        admission = {'phase': 'admission', 'cases': [c for c in cases if c['phase'] == 'admission']}
        registry.set_manifests(identity, debug, admission)
        if manifest.get('synthesis_failure_scope') == 'initial_disconnected':
            guard = {'last_initial_disconnected', 'has_initial_disconnected_parts'}
            narrow = all(guard.intersection(rule['when']) for rule in program['rules'])
            write(output/'executable_scope_check.json', dict(
                scope='initial_disconnected', all_rules_guarded=narrow,
                executable_predicates=[rule['when'] for rule in program['rules']],
                natural_language_card_is_unverified=True, new_physical_calls=0))
            if not narrow:
                snapshot = registry.freeze(output/'skills_frozen.json')
                write(output/'result.json', dict(status='DEVELOPMENT_REJECTED',
                    generation_origin=generation['origin'], candidate_id=identity,
                    cases=[], snapshot=snapshot, live_generation_tested=live,
                    physical_cases_not_run=True, reason='initial_contact_scope_not_encoded',
                    performance_benefit='NOT_RUN', full_plan_test='NOT_RUN'))
                return 0
        if live and packet['operation'] == 'refine':
            if manifest.get('synthesis_failure_scope') == 'initial_disconnected':
                from orchestrator.skill_refinement import audit_program_novelty
                novelty = audit_program_novelty(program, packet['context']['prior_development_program_hashes'])
                write(output/'program_novelty_gate.json', novelty)
                if not novelty['accepted_for_fresh_validation']:
                    snapshot = registry.freeze(output/'skills_frozen.json')
                    write(output/'result.json', dict(status='DEVELOPMENT_REJECTED',
                        generation_origin=generation['origin'], candidate_id=identity,
                        cases=[], development_gate=novelty, snapshot=snapshot,
                        live_generation_tested=True, physical_cases_not_run=True,
                        reason='unchanged_program_no_new_procedure',
                        performance_benefit='NOT_RUN', full_plan_test='NOT_RUN'))
                    print(json.dumps({'phase': 'duplicate_program_rejected', 'gate': novelty}), flush=True)
                    return 0
            gate = audit_revision(program, packet['validation_feedback'])
            write(output/'development_gate.json', gate)
            if not gate['accepted_for_fresh_validation']:
                # This is an offline proposal-novelty check, not a physical
                # rejection or pruning rule. Keep the proposal unregistered.
                snapshot = registry.freeze(output/'skills_frozen.json')
                write(output/'result.json', dict(status='DEVELOPMENT_REJECTED',
                    generation_origin=generation['origin'], candidate_id=identity,
                    cases=[], development_gate=gate, snapshot=snapshot,
                    live_generation_tested=True, physical_cases_not_run=True,
                    reason='unchanged_or_invalid_development_behavior',
                    performance_benefit='NOT_RUN', full_plan_test='NOT_RUN'))
                print(json.dumps({'phase': 'development_rejected', 'gate': gate}), flush=True)
                return 0
        outcomes = []
        for original, case in zip(prepared, cases):
            print(json.dumps({'phase': 'actual_case', 'case_id': case['case_id'], 'API_calls': 0}), flush=True)
            try:
                artifacts = run_case(original, case, program, output, run_id)
                record = registry.add_validation(identity, case, artifacts)
                outcomes.append({'case_id': case['case_id'], 'outcome': record['outcome']})
            except Exception as error:
                # A failed physical case cannot disappear from the denominator.
                issue = {'case_id': case['case_id'], 'outcome': 'ERROR',
                         'error_type': type(error).__name__, 'error': str(error)}
                write(output/(case['case_id']+'_error.json'), issue)
                outcomes.append(issue)
        # Admission rechecks every evidence hash and interpreter output itself.
        decision = registry.admit(identity, debug, admission, contract_checks={
            'program_validated': True, 'bounded_interpreter': True})
        snapshot = registry.freeze(output/'skills_frozen.json')
        write(output/'result.json', dict(status='REGISTERED_SCOPED' if decision['admitted'] else 'NOT_REGISTERED',
            generation_origin=generation['origin'],
            candidate_id=identity, cases=outcomes, admission=decision,
            snapshot=snapshot, performance_benefit='NOT_RUN', full_plan_test='NOT_RUN',
            live_generation_tested=live))
        print(json.dumps({'phase': 'end', 'cases': outcomes,
            'admitted': decision['admitted'], 'lifecycle': decision['lifecycle'],
            'reasons': decision['reasons']}), flush=True)
        return 0 if not any(x['outcome']=='ERROR' for x in outcomes) else 1
    finally:
        service.close()
        from orchestrator.batch_budget import budget_summary
        write(output/'budget_after.json', budget_summary(ledger) if ledger.exists() else
              {'reserved_calls': 0, 'budget_committed_usd': '0', 'usage_cost_usd': '0'})
        write(output/'source_preservation.json',
              {'unchanged': all(sha(ROOT/name)==value for name,value in hashes.items())})


if __name__ == '__main__':
    raise SystemExit(main())
