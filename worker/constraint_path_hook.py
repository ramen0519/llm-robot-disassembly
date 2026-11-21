"""Process-local path-repair extension; ASAP alone owns the search tree (3.8)."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path

from constraint_memory import records_from_evidence, save_snapshot, FrozenConstraintMemory
from independent_features import filter_features
from path_evidence import collect_endpoint, observed_collision_thresholds
from path_repair import execute_single, execute_two_leg
from rank_contract import RankRequest
from rank_hook import json_value, state_id

AXES = {'+X': [1, 0, 0], '-X': [-1, 0, 0], '+Y': [0, 1, 0],
        '-Y': [0, -1, 0], '+Z': [0, 0, 1], '-Z': [0, 0, -1]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(json_value(value), stream, sort_keys=True, allow_nan=False, indent=2)


class ConstraintPathHook:
    def __init__(self, session, signature, controlled_probe=False):
        self.session, self.signature = session, signature
        self.output = Path(session.log.stream.name).parent
        self.output.joinpath('paths').mkdir(exist_ok=False)
        self.controlled_probe = controlled_probe
        self.current, self.selected_path = None, None
        self.records, self.outcomes = [], []
        self.path_checks, self.repair_requests, self.measurements = 0, 0, 0
        self.decision_requests = 0
        import redmax_py
        import plan_sequence.physics_planner as physics
        import plan_sequence.sim_string as sim_string
        self.solver = {'redmax_binary_sha256': sha(redmax_py.__file__),
                       'physics_source_sha256': sha(physics.__file__),
                       'sim_string_sha256': sha(sim_string.__file__),
                       'controller_sha256': sha(Path(__file__).with_name('path_repair.py')),
                       'observer_sha256': sha(Path(__file__).with_name('path_evidence.py'))}
        self.solver_hash = hashlib.sha256(json.dumps(self.solver, sort_keys=True).encode()).hexdigest()

    def emit(self, kind, **fields):
        with self.output.joinpath('path_events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(json_value(dict(kind=kind, **fields)), sort_keys=True, allow_nan=False) + '\n')

    def evidence(self, planner, result, label, rng_before):
        self.measurements += 1
        identity = self.session.log.run_id + ':path' + str(self.measurements)
        relative = 'paths/' + str(self.measurements) + '_' + label + '.json'
        path_file = self.output / relative
        write_new(path_file, result)
        current = self.current
        context = self.session.context
        settings = context['task']['solver_settings']
        scope = {
            'task_context': context, 'assembled_parts': sorted(current['parts_rest'] + [current['part_move']]),
            'part_move': current['part_move'], 'parts_rest': current['parts_rest'],
            'parts_removed': current['parts_removed'], 'base_part': context['task']['base_part'],
            'pose': current['pose'],
            'support_policy': {'base_part': context['task']['base_part'], 'max_grippers': current['max_grippers'],
                               'timeout': current['timeout'], 'allow_gap': False,
                               'stability_function': 'upstream_get_stable_plan_1pose_serial'},
            'action_schedule': result['controller']['actions'],
            'physics': {'integrator': 'BDF1', 'timestep': .001, 'gravity': [0, 0, 1e-12],
                        'friction': 0, 'contact_kn': 1e3, 'contact_damping': 1e3, 'sdf_dx': .05,
                        'sdf_res': 20, 'min_sep': .5, 'force_mag': float(planner.force_mag),
                        'frame_skip': planner.frame_skip, 'max_time': planner.max_time,
                        'stability_gravity': [0, 0, -9.8], 'stability_friction': .5,
                        'parts_removed_in_path_simulation': list(planner.parts_removed),
                        'sdf_cache_sha256': planner._constraint_cache_hashes,
                        'source_and_binary_hashes': self.solver},
            'solver_fingerprint': self.solver_hash, 'seed': settings['seed'], 'budget': settings['budget'],
            'rng_before': rng_before, 'initial_q': result['geometry']['start_q'],
            'collector_version': 'conditional_endpoint_v1',
        }
        event = collect_endpoint(planner, result['raw_success'], result['path'], scope,
                                 identity, relative, sha(path_file))
        event_file = self.output / ('paths/' + str(self.measurements) + '_evidence.json')
        write_new(event_file, event)
        new = records_from_evidence(context, [event], str(event_file), sha(event_file))
        self.records.extend(new)
        self.emit('measurement_saved', event_id=identity, evidence_file=str(event_file),
                  record_ids=[r['record_id'] for r in new], raw_success=result['raw_success'],
                  physics_steps=result['controller']['physics_steps'], label=label)
        return event, new, relative

    def assemblable(self, original, asset_folder, assembly_dir, parts_fix, part_move, **kwargs):
        import numpy as np
        from plan_sequence.physics_planner import MultiPartPathPlanner
        # A per-run cap prevents this procedure becoming an unbounded hidden search.
        controlled = (self.controlled_probe and self.repair_requests == 0 and part_move == '1'
                      and set(parts_fix) == {'0', '2', '3', '6'})
        if not controlled:
            native = original(asset_folder, assembly_dir, parts_fix, part_move, **kwargs)
            native_action = native[0] if kwargs.get('return_path') else native
            if native_action is not None or self.repair_requests >= 1:
                return native
        self.repair_requests += 1
        self.emit('repair_trigger', trigger='controlled_single_direction_failure_probe' if controlled
                  else 'native_six_axis_search_returned_none', call_id='v' + str(self.session.verifier_count),
                  part_move=part_move, parts_rest=parts_fix, performance_test=False)
        planner = MultiPartPathPlanner(asset_folder, assembly_dir, parts_fix, part_move,
                    pose=kwargs.get('pose'), save_sdf=kwargs.get('save_sdf', False))
        planner._constraint_cache_hashes = {p.name: sha(p) for p in sorted(Path(assembly_dir).glob('*.sdf'))}
        before = self.session.rng_digest()
        failed = execute_single(planner, AXES['+X'])
        failure_event, new_records, failed_path = self.evidence(planner, failed, 'single_probe', before)
        if failed['raw_success'] or not new_records:
            self.emit('repair_not_applicable', reason='probe_not_failed_with_measured_endpoint_violation')
            return original(asset_folder, assembly_dir, parts_fix, part_move, **kwargs) if controlled else native
        request = RankRequest(self.session.log.run_id,
            self.session.log.run_id + ':path_request' + str(self.repair_requests),
            state_id(tuple(sorted(parts_fix + [part_move]))), tuple(sorted(parts_fix + [part_move])),
            tuple(sorted(p for p in parts_fix + [part_move] if p != self.session.context['task']['base_part'])),
            self.session.context['task']['base_part'])
        refs = FrozenConstraintMemory(new_records).retrieve(self.session.context, request.assembled_parts,
                                                           request.candidate_ids, max_records=2)['records']
        self.session.client.features = {'protocol': 'constraint_path_repair_v1',
            'cad': filter_features(self.session.features, request.assembled_parts),
            'memory_context': self.session.context, 'moving_part_id': part_move,
            'constraint_references': [dict(r, origin='current_episode_measurement') for r in refs],
            'failure_event_id': failure_event['event_id'], 'first_axis': '+X',
            'allowed_second_axes': ['+Y', '-Y', '+Z', '-Z'], 'switch_step': 50,
            'registered_skill': {'skill_id': 'turn_force_two_leg_v1', 'code_sha256': self.solver['controller_sha256'],
                                'status': 'provisional', 'mid_path_resets': 0},
            'hard_pruning': False, 'verifier_reuse': False}
        try:
            self.decision_requests += 1
            self.session.client(request)
            response = self.session.client.cache[request.request_id]
            decision = response['metadata'].get('path_skill')
        except Exception as error:
            decision = None
            self.emit('path_decision_fallback', error_type=type(error).__name__)
        if not decision or decision.get('skill_id') != 'turn_force_two_leg_v1':
            self.emit('path_decision_no_repair', request_id=request.request_id)
            return original(asset_folder, assembly_dir, parts_fix, part_move, **kwargs) if controlled else native
        # Defense in depth: do not trust metadata merely because candidate IDs validated.
        if decision.get('second_axis') not in ('+Y', '-Y', '+Z', '-Z') or decision.get('switch_step') != 50:
            raise ValueError('unregistered_path_controller_argument')
        before = self.session.rng_digest()
        repaired = execute_two_leg(planner, AXES['+X'], AXES[decision['second_axis']], 50)
        repaired_event, _, relative = self.evidence(planner, repaired, 'two_leg', before)
        invocation = {'skill_id': 'turn_force_two_leg_v1', 'request_id': request.request_id,
            'parent_call_id': 'v' + str(self.session.verifier_count), 'decision': decision,
            'failed_evidence_id': failure_event['event_id'], 'repaired_evidence_id': repaired_event['event_id'],
            'failed_path': failed_path, 'trajectory_file': relative, 'trajectory_sha256': sha(self.output / relative),
            'scope': repaired_event['scope'], 'controller': repaired['controller'], 'geometry': repaired['geometry'],
            'path_success': repaired['raw_success'], 'stability_success': None,
            'single_direction_failure_recovery': controlled, 'native_six_axis_failure_recovery': not controlled,
            'legacy_single_action_replayer_supported': False,
            'action_field_semantics': 'initial_force_only_full_schedule_required'}
        self.outcomes.append(invocation)
        self.emit('path_skill_execution', **invocation)
        if repaired['raw_success']:
            self.selected_path = invocation
            # The real full trajectory reaches the original _simulate stability
            # and optional grasp stage; no state transition occurs in this hook.
            path = np.array(repaired['path'])
            action = np.array(AXES['+X'])
            return (action, path) if kwargs.get('return_path') else action
        return original(asset_folder, assembly_dir, parts_fix, part_move, **kwargs) if controlled else native

    @contextmanager
    def installed(self, planner_class):
        import plan_sequence.planner.base as base
        from plan_sequence.physics_planner import MultiPartPathPlanner
        original_simulate, original_assemblable = planner_class._simulate, base.check_assemblable
        original_check = MultiPartPathPlanner.check_success

        def counted(planner, *args, **kwargs):
            self.path_checks += 1
            return original_check(planner, *args, **kwargs)

        @wraps(original_simulate)
        def simulate(*args, **kwargs):
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            self.current = json_value({k: bound.arguments[k] for k in (
                'part_move', 'parts_rest', 'parts_removed', 'pose', 'max_grippers', 'timeout')})
            self.selected_path = None
            raw = original_simulate(*args, **kwargs)
            if self.selected_path is not None:
                extension = self.selected_path
                extension['stability_success'] = raw['parts_fix'] is not None
                extension['parts_fix'] = json_value(raw['parts_fix'])
                extension['feasible'] = bool(raw['feasible'])
                # The outer IndependentSession observes this annotated return.
                # Both raw feedback and tree.pkl retain the complete schedule.
                raw['path_extension'] = deepcopy(extension)
                self.emit('repair_verifier_result', **extension)
            return raw

        base.check_assemblable = lambda *args, **kwargs: self.assemblable(original_assemblable, *args, **kwargs)
        planner_class._simulate, MultiPartPathPlanner.check_success = simulate, counted
        try:
            with observed_collision_thresholds(MultiPartPathPlanner):
                yield self
        finally:
            planner_class._simulate, base.check_assemblable = original_simulate, original_assemblable
            MultiPartPathPlanner.check_success = original_check
            save_snapshot(self.records, self.output / 'constraint_memory.json')
            write_new(self.output / 'path_outcomes.json', {'outcomes': self.outcomes,
                'path_check_success_calls': self.path_checks, 'new_measurements': self.measurements,
                'repair_decision_requests': self.decision_requests, 'repair_trigger_count': self.repair_requests,
                'constraint_records': len(self.records),
                'hard_pruning': False, 'performance_comparison': 'NOT_RUN'})
