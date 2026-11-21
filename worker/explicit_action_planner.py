"""ASAP DFS adapter with explicit, budgeted actions through the final pair.

This is an extended search adapter, not the unchanged upstream DFS. It retains
ASAP's G_path stack, successful transition updater, graph and statistics, while
replacing part-edge exhaustion and the unbudgeted mass-selected terminal step.
Failure belongs to one selected action; it never removes a candidate part.
The finite pose/skill/direction/support pool is a declared controller scope,
not a claim of physically complete motion planning. No API is called here.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
from time import monotonic
from typing import Any, Callable, Dict, Optional

try:
    from .explicit_action_contract import (DIRECTIONS, action_id, canonical_response,
        enumerate_actions, validate_request, validate_response)
    from .path_probe_round import (partition_path_round, canonical_path_round_response,
        validate_path_round_response, semantic_path_key)
    from .stability_diagnostics import project_stability_diagnostics
    from .decision_halt import halt_response, is_halt_response
except ImportError:
    from explicit_action_contract import (DIRECTIONS, action_id, canonical_response,
        enumerate_actions, validate_request, validate_response)
    from path_probe_round import (partition_path_round, canonical_path_round_response,
        validate_path_round_response, semantic_path_key)
    from stability_diagnostics import project_stability_diagnostics
    from decision_halt import halt_response, is_halt_response


def _json(value: Any) -> Any:
    """Copy observations, including upstream NumPy arrays, into JSON values."""
    if hasattr(value, 'tolist'):
        return _json(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return json.loads(json.dumps(value, allow_nan=False))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_json(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return (type(a) in (int, float) and type(b) in (int, float)
            and abs(a - b) <= 1e-8)


def planner_error_details(error, phase, remaining_parts, n_eval):
    """Allowlisted local diagnostics only; no raw callback messages or payloads."""
    codes = {'empty_geometric_pose_catalog', 'configured_pose_limit_exceeds_contract',
        'complete_pose_catalog_overflow_no_truncation',
        'evaluation_pose_catalog_overflow_no_truncation', 'request_core_too_large',
        'request_too_large', 'invalid_pose_catalog',
        'invalid_accepted_pose_carry', 'accepted_pose_catalog_overflow_no_truncation',
        'request_enricher_changed_search_scope', 'request_enricher_changed_pose_pool'}
    code = str(error) if isinstance(error, ValueError) else None
    detail = dict(error_type=type(error).__name__, phase=phase,
        remaining_parts=list(remaining_parts), n_eval=n_eval,
        code=code if code in codes else 'unclassified_planner_error')
    audit = getattr(error, 'audit', None)
    if code in codes and type(audit) is dict:
        for key in ('generated_count', 'configured_limit', 'contract_limit',
                    'core_bytes', 'limit_bytes', 'request_bytes'):
            value = audit.get(key)
            if type(value) is int and value >= 0:
                detail[key] = value
    return detail


SELECTION_ERRORS = frozenset({
    'action_already_attempted', 'invalid_candidate_permutation',
    'invalid_provider_choice_fields', 'stale_or_invalid_response_identity',
    'invalid_decision_fields', 'duplicate_json_key', 'invalid_response',
    'invalid_provider_wire_fields', 'invalid_provider_decision_fields',
    'invalid_provider_slot', 'invalid_selected_part', 'invalid_provider_permutation_code',
    'invalid_provider_evidence_selection', 'illegal_support_set_or_budget',
    'llm_incomplete', 'llm_refusal', 'api_error', 'action_deferred_for_probe_round'})
DECISION_SOURCES = frozenset({
    'selector', 'canonical_fallback', 'live_llm', 'injected_provider_fixture',
    'deterministic_fallback', 'deterministic_control', 'declared_fixture',
    'test_fixture', 'fixture'})


def _selection_feedback(response, request):
    """Project fixed diagnostics, never provider prose or physical conclusions."""
    metadata = response.get('metadata', {})
    error = metadata.get('decision_validation_error')
    error = (None if error is None else error if type(error) is str
             and error in SELECTION_ERRORS else 'invalid_response')
    rejected = metadata.get('rejected_action_id')
    repeated = (metadata.get('action_repeated') is True
                and error == 'action_already_attempted'
                and type(rejected) is str
                and rejected in request['attempted_action_ids'])
    return dict(scope='selection_validation_not_physics',
        provider_contract=(metadata.get('provider_contract') if
            metadata.get('provider_contract') in ('explicit_action_provider_v2', 'explicit_action_provider_v3',
                                                'explicit_action_provider_v4', 'explicit_action_provider_v5')
            else None), decision_validation_error=error,
        action_repeated=repeated, rejected_action_id=rejected if repeated else None,
        raw_response_recorded=(False if metadata.get('raw_response_recorded', False)
                               is False else None))


def _safe_output(value: Any) -> Path:
    """Reject forbidden drives before any filesystem call."""
    text = str(value)
    normalized = text.replace('\\', '/').lower()
    drive = PureWindowsPath(text).drive.lower()
    if (not text or '\x00' in text or normalized.startswith('//')
            or (drive and drive != 'c:') or normalized == '/mnt/e'
            or normalized.startswith('/mnt/e/') or '..' in normalized.split('/')):
        raise ValueError('unsupported_output_path')
    path = Path(text)
    if not path.is_absolute():
        raise ValueError('absolute_output_path_required')
    return path


def make_explicit_action_planner(dfs_class: type, sequence_class: type) -> type:
    """Build an actual upstream DFS subclass without importing physics on host.

    Production passes ASAP DFSSequencePlanner and SequencePlanner. Tests inject
    the pinned upstream method bodies and a graph fixture, never physics.
    The factory does not monkey-patch either upstream class.
    """
    if not issubclass(dfs_class, sequence_class):
        raise TypeError('dfs_class_must_extend_sequence_class')

    class ExplicitActionDFS(dfs_class):
        """Single owner of the search tree, stack and exact episode attempts."""

        def __init__(self, *, parts, decision_fn: Callable,
                     verifier_fn: Callable, pose_provider: Callable,
                     run_id: str, base_part: Optional[str] = None,
                     max_grippers: int = 3, task_context=None, geometry=None,
                     memory_references=None, evidence_ids=None,
                     request_enricher: Optional[Callable] = None,
                     event_sink: Optional[Callable] = None,
                     graph_factory: Optional[Callable] = None,
                     clock: Callable = monotonic, fallback_policy: str = 'legacy_v1',
                     path_selection_policy: str = 'legacy_v1',
                     decision_failure_policy: str = 'legacy_v1',
                     pose_transition_policy: str = 'legacy_v1', max_pose: int = 128):
            # Avoid upstream __init__: its get_body_mass is not needed for this
            # controller and no terminal mass rule is permitted on this route.
            if (type(parts) not in (list, tuple) or len(parts) < 2
                    or any(type(part) is not str or not part for part in parts)
                    or len(set(parts)) != len(parts)):
                raise ValueError('parts_must_be_unique_nonempty_ids')
            if base_part is not None and base_part not in parts:
                raise ValueError('base_must_be_present')
            if type(run_id) is not str or not run_id:
                raise ValueError('run_id_required')
            if type(max_grippers) is not int or max_grippers < 1:
                raise ValueError('positive_max_grippers_required')
            if fallback_policy not in ('legacy_v1', 'path_stage_v1'):
                raise ValueError('unknown_fallback_policy')
            self.fallback_policy = fallback_policy
            if path_selection_policy not in ('legacy_v1', 'path_probe_round_v1'):
                raise ValueError('unknown_path_selection_policy')
            self.path_selection_policy = path_selection_policy
            if decision_failure_policy not in ('legacy_v1', 'halt_v1'):
                raise ValueError('unknown_decision_failure_policy')
            self.decision_failure_policy = decision_failure_policy
            from worker.accepted_pose_carry import POLICIES
            if pose_transition_policy not in POLICIES or type(max_pose) is not int or not 1 <= max_pose <= 128:
                raise ValueError('invalid_pose_transition_policy_or_limit')
            self.pose_transition_policy, self.max_pose = pose_transition_policy, max_pose
            if not all(callable(fn) for fn in (decision_fn, verifier_fn,
                                               pose_provider, clock)):
                raise TypeError('callbacks_must_be_callable')
            self.parts = sorted(parts)
            self.base_part = base_part
            self.max_grippers = max_grippers
            self.run_id = run_id
            self.decision_fn, self.verifier_fn = decision_fn, verifier_fn
            self.pose_provider = pose_provider
            self.task_context = _json(task_context or {})
            if not self.task_context:
                raise ValueError('nonempty_task_context_required')
            self.geometry = _json(geometry or {})
            self.memory_references = _json(memory_references or [])
            self.evidence_ids = _json(evidence_ids or [])
            self.event_sink = event_sink
            self.request_enricher = request_enricher
            self.graph_factory, self.clock = graph_factory, clock
            self.tree = None
            self.n_eval = 0
            self.stop_msg = None
            self.attempts = []
            self.decisions = []
            self.events = []
            self.states = {}
            self.started = None
            self.elapsed = None
            self._event_file = None
            self._output_dir = None
            self._ran = False
            self._solution_choices = []
            self._path_choices = []
            self._request_counter = 0
            self._last_error = None

        def _emit(self, kind: str, **fields) -> None:
            record = dict(fields)
            record.setdefault('event_id', self.run_id + ':e' + str(len(self.events) + 1))
            record.update(run_id=self.run_id, kind=kind)
            record = _json(record)
            self.events.append(record)
            if self._event_file is not None:
                self._event_file.write(json.dumps(record, ensure_ascii=False,
                    allow_nan=False, sort_keys=True) + '\n')
                self._event_file.flush()
            if self.event_sink is not None:
                fields = deepcopy(record)
                fields.pop('kind')
                self.event_sink(kind, **fields)

        def _remaining_time(self) -> Optional[float]:
            if self.timeout is None:
                return None
            return max(0.0, self.timeout - (self.clock() - self.started))

        def _time_exhausted(self) -> bool:
            remaining = self._remaining_time()
            return remaining is not None and remaining <= 0

        def _state(self, parts):
            key = tuple(parts)
            if key not in self.states:
                catalog = _json(self.pose_provider(list(parts)))
                self.states[key] = dict(
                    state_id='state:' + _digest(dict(parts=parts,
                        base_part=self.base_part, task_context=self.task_context,
                        max_grippers=self.max_grippers, pose_catalog=catalog)),
                    remaining_parts=list(parts), pose_catalog=catalog,
                    attempted_action_ids=[], attempted_actions=[],
                    n_attempts=0, stop_reason=None)
            state = self.states[key]
            if self.pose_transition_policy != 'legacy_v1' and self._path_choices:
                from worker.accepted_pose_carry import snapshot, append_to_state
                edge = self._path_choices[-1]
                if edge['context']['parts_rest'] == list(parts):
                    proof = snapshot(edge, run_id=self.run_id, task_context=self.task_context,
                        base_part=self.base_part, max_grippers=self.max_grippers, remaining_parts=list(parts))
                    revision = state.get('pose_catalog_revision', 0)
                    identity = append_to_state(state, proof, self.max_pose)
                    if state['pose_catalog_revision'] != revision:
                        self._emit('accepted_pose_catalog_appended', state_id=state['state_id'],
                            catalog_revision=state['pose_catalog_revision'], pose_id=identity,
                            source_event_id=edge['event_id'], exact_full_matrix_preserved=True,
                            attempts_preserved=state['n_attempts'], new_physics_pass_implied=False,
                            robot_state_changed=False)
            return state

        def _request(self, parts, state) -> Dict[str, Any]:
            from orchestrator.execution_observation import from_attempt
            self._request_counter += 1
            feedback = [dict(evidence_id=item['event_id'],
                status=item['status'], action_id=item['action_id'],
                remaining_parts=item['remaining_parts'], choice=item['choice'],
                decision_source=(item['decision_source'] if
                    type(item['decision_source']) is str and
                    item['decision_source'] in DECISION_SOURCES else 'unknown'),
                selection_feedback=deepcopy(item['selection_feedback']),
                selected_pose_matrix=item['context']['pose_matrix'],
                path_success=item['raw'].get('path_success'),
                stability_success=item['raw'].get('stability_success'),
                # Only the executor's measured projection is forwarded. Model
                # explanations and arbitrary raw text are not physical facts.
                measured_endpoint=deepcopy(item['raw'].get('endpoint_observation')),
                raw_stability_parts=deepcopy(item['raw'].get(
                    'raw_upstream', {}).get('parts_fall')),
                raw_stability_parts_semantics=(
                    'upstream_combines_initial_disconnection_and_observed_falling_no_cause_inference'),
                accepted_for_transition=item['accepted_for_transition'],
                deadline_exceeded=item['deadline_exceeded'],
                # Forward the same measured clock/acceptance contract used by
                # episode memory. Missing values remain unknown; no new PASS.
                execution=from_attempt(item))
                for item in self.attempts[-12:]]
            for projected, item in zip(feedback, self.attempts[-12:]):
                diagnostic = project_stability_diagnostics(item['raw'].get('stability_diagnostics'))
                if diagnostic is not None:
                    projected['stability_diagnostics'] = diagnostic
            # Neither attempted choices nor historical failures filter this set.
            request = dict(schema_version=1, protocol='explicit_action_v1',
                run_id=self.run_id, request_id=self.run_id + ':request:' +
                str(self._request_counter), state_id=state['state_id'],
                remaining_parts=list(parts), candidate_ids=[part for part in parts
                    if part != self.base_part], base_part=self.base_part,
                max_grippers=self.max_grippers,
                pose_catalog=deepcopy(state['pose_catalog']),
                attempted_action_ids=list(state['attempted_action_ids']),
                attempted_actions=deepcopy(state['attempted_actions']),
                task_context=deepcopy(self.task_context),
                evidence_ids=sorted(set(self.evidence_ids) |
                    {item['evidence_id'] for item in self.memory_references} |
                    {item['evidence_id'] for item in feedback}),
                episode_feedback=deepcopy(feedback),
                memory_references=deepcopy(self.memory_references),
                geometry=deepcopy(self.geometry))
            if self.path_selection_policy == 'path_probe_round_v1':
                request['path_stage_observations'] = [dict(
                    action_id=item['action_id'], path_success=item['raw'].get('path_success'),
                    status=item['status'], deadline_exceeded=item['deadline_exceeded'],
                    backend=item['raw'].get('backend'), source_event_id=item['event_id'])
                    for item in self.attempts if item['state_id'] == state['state_id']]
                if state.get('virtual_restore_snapshot') is not None:
                    from worker.virtual_search_restore import bind, FIELD
                    request[FIELD] = bind(request, state['virtual_restore_snapshot'])
            if self.pose_transition_policy != 'legacy_v1' and state.get('active_pose_carry') is not None:
                from worker.accepted_pose_carry import bind, FIELD
                request[FIELD] = bind(request, state)
            if self.request_enricher is not None:
                enriched = self.request_enricher(deepcopy(request))
                protected = set(request) - {'memory_references', 'evidence_ids',
                                            'geometry', 'pose_catalog'}
                if (type(enriched) is not dict or any(enriched.get(key) !=
                        request[key] for key in protected)):
                    raise ValueError('request_enricher_changed_search_scope')
                original_poses = [{k: p[k] for k in ('pose_id', 'matrix')}
                                  for p in request['pose_catalog']]
                enriched_poses = [{k: p[k] for k in ('pose_id', 'matrix')}
                                  for p in enriched.get('pose_catalog', [])]
                if original_poses != enriched_poses:
                    raise ValueError('request_enricher_changed_pose_pool')
                request = enriched
            request = validate_request(request)
            if self.path_selection_policy == 'path_probe_round_v1':
                # Fail before the external decision boundary for incomplete or
                # inconsistent history. Never downgrade to a paid legacy call.
                partition_path_round(request)
            return request

        def _backtrack(self, state, reason: str) -> None:
            state['stop_reason'] = reason
            popped = self.G_path.pop()
            edge = None
            if self._path_choices:
                edge = self._path_choices.pop()
            if self.G_path and edge is not None and self.path_selection_policy == 'path_probe_round_v1':
                from worker.virtual_search_restore import snapshot
                parent = self._state(self.G_path[-1])
                if edge['state_id'] != parent['state_id']:
                    raise ValueError('dfs_restore_parent_mismatch')
                revision = parent.get('node_visit_revision', 0) + 1
                parent['node_visit_revision'] = revision
                if (edge['raw'].get('path_success') is True
                        and edge['raw'].get('stability_success') is True):
                    parent['virtual_restore_snapshot'] = snapshot(edge, state, revision)
                    self._emit('virtual_search_restored', state_id=parent['state_id'],
                        node_visit_revision=revision, source_event_id=edge['event_id'],
                        source_action_id=edge['action_id'], returned_from_state_id=state['state_id'],
                        robot_state_changed=False, physical_success_implied=False)
                else:
                    # Legacy custom verifiers may omit subverdicts. Preserve
                    # that profile without manufacturing restoration evidence.
                    parent.pop('virtual_restore_snapshot', None)
                    self._emit('virtual_search_restore_unavailable', state_id=parent['state_id'],
                        node_visit_revision=revision, reason='complete_subverdicts_required',
                        source_event_id=edge['event_id'], physical_success_implied=False)
            self._emit('backtrack', state_id=state['state_id'],
                remaining_parts=popped, reason=reason,
                verdict='UNKNOWN', proves_impossible=False)

        def _choose(self, request):
            source = 'selector'
            reply = None
            valid_reply = None
            try:
                reply = self.decision_fn(deepcopy(request))
                response = validate_response(request, reply)
                if is_halt_response(response) and self.decision_failure_policy != 'halt_v1':
                    raise ValueError('decision_halt_requires_opt_in')
                valid_reply = response
                if self.path_selection_policy == 'path_probe_round_v1':
                    response = validate_path_round_response(request, response)
            except Exception as error:
                reason = 'decision_exception:' + type(error).__name__
                response = (halt_response(request, reason) if self.decision_failure_policy == 'halt_v1' else
                    canonical_path_round_response(request, reason, self.fallback_policy)
                    if self.path_selection_policy == 'path_probe_round_v1' else
                    canonical_response(request, reason, policy=self.fallback_policy))
                diagnostic = str(error) if type(error) is ValueError else None
                response['metadata']['decision_validation_error'] = (
                    diagnostic if diagnostic in SELECTION_ERRORS else 'invalid_response')
                # Only a complete valid reply rejected at the final duplicate
                # guard can identify a rejected action. Callback exceptions or
                # malformed replies never fabricate such an identity.
                if (diagnostic == 'action_already_attempted' and type(reply) is dict
                        and reply.get('action_id') in request['attempted_action_ids']):
                    clean = deepcopy(request)
                    clean['attempted_action_ids'] = []
                    clean['attempted_actions'] = []
                    if 'path_stage_observations' in clean:
                        clean['path_stage_observations'] = []
                    try:
                        validated = validate_response(clean, reply)
                    except Exception:
                        pass
                    else:
                        response['metadata'].update(action_repeated=True,
                            rejected_action_id=validated['action_id'])
                if diagnostic == 'action_deferred_for_probe_round' and valid_reply is not None:
                    response['metadata'].update(path_selection_deferred=True,
                        deferred_action_id=valid_reply['action_id'],
                        deferred_choice=deepcopy(valid_reply['choice']),
                        deferred_is_not_physical_failure=True)
                source = 'canonical_fallback'
            if self.path_selection_policy == 'path_probe_round_v1':
                response['metadata'].update(path_selection_policy=self.path_selection_policy,
                    path_probe_round=partition_path_round(request)['round_audit'],
                    semantic_path_key=semantic_path_key(request, response['choice'])
                        if response['choice'] is not None else None)
            if is_halt_response(response):
                source = 'decision_halt'
            elif response.get('fallback_reason') is not None:
                source = 'canonical_fallback'
            elif isinstance(response.get('metadata'), dict):
                source = response['metadata'].get('selection_source', source)
            decision = dict(request=deepcopy(request), response=deepcopy(response),
                            decision_source=source)
            self.decisions.append(decision)
            self._emit('action_decision', **decision)
            return response, source

        def _sim_info(self, result, choice, pose):
            if type(result) is not dict:
                raise ValueError('invalid_verifier_response')
            status = result.get('status')
            if status not in ('PASS', 'REJECTED', 'UNKNOWN', 'ERROR'):
                raise ValueError('invalid_verifier_status')
            if type(result.get('feasible')) is not bool:
                raise ValueError('verifier_feasible_must_be_bool')
            if (status == 'PASS') != result['feasible']:
                raise ValueError('inconsistent_verifier_pass')
            if result.get('part_move', choice['part_id']) != choice['part_id']:
                raise ValueError('verifier_changed_selected_part')
            if ('pose' in result and result['pose'] is not None and
                    not _close(result['pose'], pose)):
                raise ValueError('verifier_changed_selected_pose')
            sim_info = dict(feasible=result['feasible'],
                action=deepcopy(result.get('action')),
                parts_fix=deepcopy(result.get('parts_fix')),
                base_part=self.base_part, part_move=choice['part_id'],
                pose=deepcopy(pose), grasp=deepcopy(result.get('grasp')),
                explicit_action=deepcopy(choice),
                action_extension=deepcopy(result.get('action_extension')),
                legacy_single_action_replayer_supported=(
                    choice['skill_id'] == 'single_force_v1'),
                raw_explicit_verifier=deepcopy(result))
            if sim_info['feasible']:
                if sorted(sim_info['parts_fix'] or []) != sorted(choice['support_ids']):
                    raise ValueError('verifier_changed_selected_supports')
                if sim_info['parts_fix'] is None or sim_info['action'] is None:
                    raise ValueError('pass_requires_observed_action_and_supports')
                if not _close(sim_info['action'], DIRECTIONS[choice['direction_id']]):
                    raise ValueError('verifier_changed_selected_initial_direction')
                if (result.get('path_success') is False or
                        result.get('stability_success') is False):
                    raise ValueError('inconsistent_verifier_substage')
            return status, sim_info

        def plan(self, *, budget: int, state_action_budget: int = 12,
                 timeout: Optional[float] = 120.0, output_dir=None):
            """Run budgeted DFS; every selected physics call counts, final2 too.

            The timeout is a cooperative deadline. A running verifier must obey
            context.time_remaining_seconds; an overrun is recorded and cannot
            yield a transition accepted within this plan's budget.
            """
            if self._ran:
                raise RuntimeError('planner_instances_are_single_run')
            if (type(budget) is not int or budget < 0 or
                    type(state_action_budget) is not int or state_action_budget < 1):
                raise ValueError('invalid_attempt_budget')
            if timeout is not None and (type(timeout) not in (int, float)
                                        or not math.isfinite(timeout) or timeout <= 0):
                raise ValueError('timeout_must_be_positive')
            self._ran = True
            self.budget, self.state_action_budget, self.timeout = (
                budget, state_action_budget, timeout)
            if output_dir is not None:
                self._output_dir = _safe_output(output_dir)
                self._output_dir.mkdir(parents=True, exist_ok=False)
                self._event_file = (self._output_dir / 'planner_events.jsonl').open(
                    'x', encoding='utf-8')
            self.started = self.clock()
            # The actual ASAP DFS owns this stack. Its edge-exhaustion selector
            # is explicitly replaced below with action-level exhaustion.
            dfs_class._reset(self)
            if self.graph_factory is None:
                import networkx as nx
                self.tree = nx.DiGraph()
            else:
                self.tree = self.graph_factory()
            self.tree.add_node(tuple(self.parts), n_eval=0, n_gripper=1, poses=[])
            self._emit('planner_start', parts=self.parts, base_part=self.base_part,
                budget=budget, state_action_budget=state_action_budget,
                timeout=timeout, search_owner='extended_asap_dfs_adapter',
                decision_failure_policy=self.decision_failure_policy,
                pose_transition_policy=self.pose_transition_policy,
                final_pair_policy='same_explicit_selector_and_budget',
                hard_pruning=False, hidden_heuristic=False,
                scope='registered_finite_action_pool_not_all_physical_paths')
            phase, parts = 'start', list(self.parts)
            try:
                while self.G_path:
                    if self.n_eval >= budget:
                        self.stop_msg = 'budget reached'
                        break
                    if self._time_exhausted():
                        self.stop_msg = 'timeout'
                        break
                    parts = self.G_path[-1]
                    phase = 'pose_catalog'
                    state = self._state(parts)
                    if state['n_attempts'] >= state_action_budget:
                        self._backtrack(state, 'state_action_budget_reached')
                        continue
                    phase = 'request_construction'
                    request = self._request(parts, state)
                    phase = 'action_enumeration'
                    if next(iter(enumerate_actions(request, policy=self.fallback_policy)), None) is None:
                        self._backtrack(state, 'registered_action_pool_exhausted')
                        continue
                    if self._time_exhausted():
                        self.stop_msg = 'timeout'
                        break
                    phase = 'decision'
                    response, source = self._choose(request)
                    if is_halt_response(response):
                        reason = response['metadata']['decision_halt_v1']['reason']
                        self.stop_msg = 'decision halted: ' + reason
                        self._emit('decision_halted', state_id=state['state_id'],
                            request_id=request['request_id'], reason=reason,
                            status='UNKNOWN', physical_attempt_executed=False,
                            action_space_exhausted=False, backtracked=False)
                        break
                    choice = response['choice']
                    if choice is None:
                        self._backtrack(state, 'registered_action_pool_exhausted')
                        continue
                    if self._time_exhausted():
                        self.stop_msg = 'timeout'
                        break
                    phase = 'action_context'
                    identifier = action_id(request, choice)
                    # Do not let a selector mutate any search-state reference.
                    pose = next(item['matrix'] for item in state['pose_catalog']
                                if item['pose_id'] == choice['pose_id'])
                    child = [part for part in parts if part != choice['part_id']]
                    context = dict(run_id=self.run_id,
                        request_id=request['request_id'], state_id=state['state_id'],
                        action_id=identifier, remaining_parts=list(parts),
                        parts_rest=child, parts_removed=[part for part in self.parts
                                                       if part not in parts],
                        base_part=self.base_part, max_grippers=self.max_grippers,
                        pose_matrix=deepcopy(pose),
                        pose_catalog=deepcopy(state['pose_catalog']),
                        task_context=deepcopy(self.task_context),
                        time_remaining_seconds=self._remaining_time())
                    state['attempted_action_ids'].append(identifier)
                    state['attempted_actions'].append(dict(action_id=identifier,
                        **{key: deepcopy(value) for key, value in choice.items()
                           if key != 'evidence_ids'}))
                    state['n_attempts'] += 1
                    self.n_eval += 1
                    attempt_dir = (self._output_dir / ('attempt_%06d' % self.n_eval)
                                   if self._output_dir is not None else None)
                    start = self.clock()
                    self._emit('verifier_start', call_id='v' + str(self.n_eval),
                        context=context, choice=choice, decision_source=source,
                        terminal_pair=len(parts) == 2)
                    phase = 'verification'
                    raw = None
                    try:
                        raw = _json(self.verifier_fn(deepcopy(context),
                            deepcopy(choice), attempt_dir))
                        status, sim_info = self._sim_info(raw, choice, pose)
                    except Exception as error:
                        raw = dict(status='ERROR', feasible=False,
                                   error_type=type(error).__name__,
                                   raw_returned_result=raw)
                        status, sim_info = 'ERROR', None
                    late = self._time_exhausted()
                    accepted = bool(sim_info is not None and
                                    sim_info['feasible'] and not late)
                    attempt = dict(call_id='v' + str(self.n_eval),
                        event_id=self.run_id + ':attempt:' + str(self.n_eval),
                        request_id=request['request_id'], state_id=state['state_id'],
                        action_id=identifier, remaining_parts=list(parts),
                        choice=deepcopy(choice), status=status, raw=raw,
                        elapsed_seconds=self.clock() - start,
                        terminal_pair=len(parts) == 2, decision_source=source,
                        selection_feedback=_selection_feedback(response, request),
                        deadline_exceeded=late, accepted_for_transition=accepted,
                        context=context)
                    self.attempts.append(attempt)
                    self._emit('verifier_result', **attempt)
                    if accepted:
                        # Only real feasible transitions become structural
                        # edges. UNKNOWN/REJECTED/ERROR remain attempt records.
                        phase = 'accepted_transition'
                        sequence_class._update_tree(self, self.tree, parts,
                            child, self.n_eval, sim_info)
                        self._path_choices.append(deepcopy(attempt))
                        if len(child) == 1:
                            self._solution_choices = deepcopy(self._path_choices)
                            self.stop_msg = 'solution found'
                            break
                        self.G_path.append(child)
                    if late:
                        self.stop_msg = 'timeout'
                        break
                if self.stop_msg is None:
                    self.stop_msg = 'bounded action search exhausted'
            except Exception as error:
                self.stop_msg = 'exception'
                self._last_error = planner_error_details(error, phase, parts, self.n_eval)
                self._emit('planner_error', **self._last_error)
            finally:
                self.elapsed = self.clock() - self.started
                self._emit('planner_stop', stop_msg=self.stop_msg,
                    n_eval=self.n_eval, elapsed_seconds=self.elapsed,
                    status=('ERROR' if self.stop_msg == 'exception' else
                            'PASS' if self._solution_choices else 'UNKNOWN'))
                if self._event_file is not None:
                    self._event_file.close()
                    self._event_file = None
                if self._output_dir is not None:
                    with (self._output_dir / 'planner_report.json').open(
                            'x', encoding='utf-8') as handle:
                        json.dump(self.report(), handle, ensure_ascii=False,
                                  indent=2, allow_nan=False)
            return self.tree

        @staticmethod
        def get_stats(tree):
            return sequence_class.get_stats(tree)

        def report(self) -> Dict[str, Any]:
            stats = self.get_stats(self.tree) if self.tree is not None else {}
            continuity_errors = []
            remaining = list(self.parts)
            for index, attempt in enumerate(self._solution_choices):
                choice = attempt['choice']
                if attempt['remaining_parts'] != remaining:
                    continuity_errors.append('remaining_mismatch_at_' + str(index))
                if (not attempt['accepted_for_transition'] or
                        attempt['status'] != 'PASS' or attempt['deadline_exceeded']):
                    continuity_errors.append('unaccepted_attempt_at_' + str(index))
                if choice['part_id'] not in remaining or choice['part_id'] == self.base_part:
                    continuity_errors.append('invalid_removal_at_' + str(index))
                    break
                child = [part for part in remaining if part != choice['part_id']]
                if (attempt['context']['parts_rest'] != child or
                        not self.tree.has_edge(tuple(remaining), tuple(child))):
                    continuity_errors.append('missing_transition_at_' + str(index))
                remaining = child
            if self._solution_choices and len(remaining) != 1:
                continuity_errors.append('not_singleton_terminal')
            if self._solution_choices and stats.get('sequence') != [
                    item['choice']['part_id'] for item in self._solution_choices]:
                continuity_errors.append('upstream_sequence_differs_from_selected_plan')
            success = bool(self._solution_choices) and not continuity_errors
            counter_keys = ('path_check_calls', 'stability_check_calls',
                            'two_leg_controller_calls', 'internal_stability_evaluations',
                            'geometry_queries', 'path_physics_steps')
            counters = {key: (sum(item['raw'][key] for item in self.attempts)
                if all(type(item['raw'].get(key)) is int and item['raw'][key] >= 0
                       for item in self.attempts) else None) for key in counter_keys}
            return _json(dict(schema_version=1, run_id=self.run_id,
                status=('ERROR' if self.stop_msg == 'exception' or continuity_errors
                        else 'PASS' if success else 'UNKNOWN'),
                success=success, stats=stats,
                total_n_eval=self.n_eval, stop_msg=self.stop_msg,
                planner_error=deepcopy(self._last_error),
                elapsed_seconds=self.elapsed, budget=getattr(self, 'budget', None),
                state_action_budget=getattr(self, 'state_action_budget', None),
                deadline_seconds=getattr(self, 'timeout', None),
                final_pair_calls=sum(a['terminal_pair'] for a in self.attempts),
                terminal_parts=([part for part in self.parts if part not in
                    [a['choice']['part_id'] for a in self._solution_choices]]
                    if self._solution_choices else None),
                selected_plan=self._solution_choices,
                selected_plan_continuity_checked=bool(self._solution_choices),
                selected_plan_continuity_errors=continuity_errors,
                decisions=self.decisions, attempts=self.attempts,
                states=list(self.states.values()),
                counters=counters,
                n_eval_semantics='selected_verifier_dispatch_attempts_including_errors_and_final_pair',
                deadline_semantics='cooperative_checks_around_callbacks_late_pass_never_accepted',
                geometry_query_count_scope='endpoint_observer_only_not_all_geometry_operations',
                search_owner='extended_asap_dfs_adapter',
                unchanged_upstream_dfs=False, hard_pruning=False,
                candidate_policy='all_remaining_minus_fixed_base',
                fallback_policy=('canonical_untried_explicit_action' if self.fallback_policy == 'legacy_v1'
                                 else self.fallback_policy),
                path_selection_policy=self.path_selection_policy,
                decision_failure_policy=self.decision_failure_policy,
                pose_transition_policy=self.pose_transition_policy,
                action_pool_scope='registered_pose_direction_support_skill_pool',
                completeness_claim=False))

    ExplicitActionDFS.__name__ = 'ExplicitActionDFS'
    return ExplicitActionDFS
