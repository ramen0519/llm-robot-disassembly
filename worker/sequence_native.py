"""Part-order-only adapter for original ASAP, compatible with Python 3.8.

This module does not import ASAP or execute physics. Native DFS owns its tree;
the session owns immutable observations and rank-request identity only.
"""
from contextlib import contextmanager
from dataclasses import asdict
import inspect
import json
import time

from evaluation_budget import EvaluationLimit
from independent_features import filter_features
from independent_rank import enumerate_candidates, validate_complete_request
from rank_contract import RankRequest, RankResponse
from rank_hook import json_value, state_id
from task_memory import task_context


PROFILE = 'sequence_native_v1'
TERMINAL_POLICY = 'unchanged_upstream_mass_or_base_rule'


class DecisionHalted(RuntimeError):
    """A failed decision never becomes a hidden canonical-order continuation."""


def snapshot(value):
    return json.loads(json.dumps(json_value(value), allow_nan=False))


def connected_path(tree, parts):
    """Count a root-connected feasible chain, never sum distinct DFS branches."""
    root = tuple(sorted(parts))
    if tree is None or root not in tree:
        return dict(success=False, max_depth=0, complete_chain=None,
                    best_partial_chain=[], reachable_feasible_nodes=0)
    queue, seen, best, full = [(root, [])], {root}, [], None
    while queue:
        parent, chain = queue.pop(0)
        if len(chain) > len(best):
            best = chain
        if len(parent) == 1 and full is None:
            full = chain
        for child in tree.successors(parent):
            info = tree.edges[parent, child].get('sim_info', {})
            if not info.get('feasible') or tree.nodes[child].get('n_gripper') is None:
                continue
            mover = info['part_move']
            if (set(parent) - set(child) != {mover} or
                    len(child) != len(parent) - 1 or not set(child).issubset(parent)):
                raise ValueError('invalid_native_removal_edge')
            if child not in seen:
                seen.add(child)
                queue.append((child, chain + [dict(parent=list(parent), child=list(child),
                    part_move=mover, sim_info=snapshot(info))]))
    return dict(success=full is not None, max_depth=len(best), complete_chain=full,
                best_partial_chain=best, reachable_feasible_nodes=len(seen))


def tree_snapshot(tree):
    return snapshot(dict(nodes=[dict(parts=list(n), data=d) for n, d in tree.nodes(data=True)],
        edges=[dict(parent=list(a), child=list(b), data=d) for a, b, d in tree.edges(data=True)]))


class SequenceNativeSession:
    def __init__(self, log, client, features, guard, composite_limit, rng_digest,
                 source_hashes=None, clock=time.monotonic):
        if type(composite_limit) is not int or composite_limit < 1:
            raise ValueError('invalid_composite_limit')
        self.log, self.client, self.guard = log, client, guard
        self.features = snapshot(features)
        self.context = task_context(self.features)
        self.source_hashes = snapshot(source_hashes or {})
        self.composite_limit = composite_limit
        self.rng_digest, self.clock = rng_digest, clock
        self.generation_count = 0
        self.rank_decisions, self.attempts, self.transitions, self.feedback = [], [], [], []
        self.halt_reason = None
        self.tripwire_calls = 0
        self.active = None
        self.last_tree = None
        self.completion_elapsed_seconds = None

    def elapsed(self):
        return self.clock() - self.guard.started

    def completed_in_budget(self):
        """Use accepted goal time; report serialization cannot revoke a success."""
        return (self.completion_elapsed_seconds is not None and
                self.completion_elapsed_seconds < self.guard.deadline-self.guard.started)

    def emit(self, kind, **fields):
        value = dict(elapsed_seconds=self.elapsed())
        value.update(fields)
        self.log.emit(kind, **value)

    def halt(self, reason):
        self.halt_reason = reason
        self.emit('decision_halted', reason=reason, canonical_fallback=False)
        raise DecisionHalted(reason)

    def generate(self, assembled_parts, base_part):
        self.guard.check()
        if self.halt_reason:
            raise DecisionHalted(self.halt_reason)
        parts = tuple(sorted(assembled_parts))
        if base_part != self.features['task']['base_part']:
            raise ValueError('base_part_differs_from_task')
        candidates = enumerate_candidates(parts, base_part)
        self.generation_count += 1
        request = RankRequest(self.log.run_id, self.log.run_id + ':r' + str(self.generation_count),
                              state_id(parts), parts, candidates, base_part)
        validate_complete_request(request)
        self.client.features = dict(protocol=PROFILE, cad=filter_features(self.features, parts),
            memory_context=snapshot(self.context), episode_feedback=snapshot(self.feedback[-12:]),
            validated_repair_skills=[],
            decision_scope='complete_part_permutation_only',
            lower_level_owner='original_ASAP_pose_direction_support_search',
            terminal_policy=TERMINAL_POLICY)
        before = self.rng_digest()
        start = self.clock()
        old_timeout = getattr(self.client, 'timeout_seconds', None)
        if old_timeout is not None:
            self.client.timeout_seconds = min(old_timeout, self.guard.remaining())
        try:
            response = self.client(request)  # Deliberately bypass legacy fallback wrapper.
        except Exception as exc:
            reason = 'rank_request_failed:' + type(exc).__name__
            # RemoteRankingError is produced locally after bounded schema validation.
            from remote_rank_client import RemoteRankingError
            if isinstance(exc, RemoteRankingError):
                reason += ':' + str(exc)[:256]
            self.halt(reason)
        finally:
            if old_timeout is not None:
                self.client.timeout_seconds = old_timeout
        self.guard.check()
        if (type(response) is not RankResponse or response.request_id != request.request_id
                or response.state_id != request.state_id):
            self.halt('invalid_or_stale_rank_response')
        order = response.candidate_ids
        if (type(order) is not tuple or any(type(p) is not str for p in order)
                or len(order) != len(candidates) or len(set(order)) != len(order)
                or set(order) != set(candidates)):
            self.halt('invalid_complete_permutation')
        after = self.rng_digest()
        if after != before:
            self.halt('ranking_mutated_solver_rng')
        row = dict(request=asdict(request), applied_order=list(order), fallback_reason=None,
            rank_seconds=self.clock()-start, rng_before=before, rng_after=after,
            full_candidate_set=True, heuristic_or_gnn_invoked=False)
        self.rank_decisions.append(snapshot(row))
        self.emit('rank_decision', **row)
        for part in order:
            self.guard.check()
            self.emit('candidate_yield', request_id=request.request_id, part=part)
            yield part

    def forbidden_ranker(self, *args, **kwargs):
        self.tripwire_calls += 1
        self.halt('heuristic_or_gnn_called_in_sequence_route')


@contextmanager
def installed_sequence(generator_base, registry, planner_class, session):
    """Replace candidate ordering; otherwise only observe/bound original calls."""
    original_simulate = planner_class._simulate
    original_update = planner_class._update_tree
    signature = inspect.signature(original_simulate)
    saved = []

    class SequenceCandidateGenerator(generator_base):
        def generate_candidate_part(self, assembled_parts):
            return session.generate(assembled_parts, self.base_part)

    def simulate(*args, **kwargs):
        session.guard.check()
        if len(session.attempts) >= session.composite_limit:
            session.guard.stop_reason = 'composite_budget'
            raise EvaluationLimit('composite_budget')
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        req = snapshot({key: bound.arguments[key] for key in
            ('part_move', 'parts_rest', 'parts_removed', 'pose', 'max_grippers', 'timeout')})
        idx = len(session.attempts) + 1
        row = dict(index=idx, call_id='v'+str(idx), event_id=session.log.run_id+':attempt:'+str(idx),
            part_move=req['part_move'], parts_rest=req['parts_rest'], request=req,
            remaining_parts=sorted(req['parts_rest']+[req['part_move']]),
            returned=False, feasible=False, status='UNKNOWN', accepted_for_transition=False,
            deadline_exceeded=False, started=session.elapsed(), counts_before=dict(session.guard.counts),
            selection_scope='upstream_terminal_completion' if len(req['parts_rest']) == 1 else 'ranked_candidate',
            source_hashes=session.source_hashes, verifier_domain='ASAP_RedMax')
        session.attempts.append(row)
        session.active = row
        session.emit('simulate_start', index=idx, part_move=row['part_move'],
                     remaining_count=len(row['remaining_parts']))
        try:
            raw = original_simulate(*args, **kwargs)
            row.update(returned=True, feasible=bool(raw['feasible']), raw_sim_info=snapshot(raw),
                       status='PASS' if raw['feasible'] else 'UNKNOWN')
            session.guard.check()
            return raw
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            if not isinstance(exc, EvaluationLimit):
                row['status'] = 'ERROR'
            raise
        finally:
            row.update(elapsed_seconds=session.elapsed(), counts_after=dict(session.guard.counts),
                deadline_exceeded=session.guard.remaining() <= 0)
            raw = row.get('raw_sim_info', {})
            feedback = dict(event_id=row['event_id'], call_id=row['call_id'], status=row['status'],
                request=req, raw=raw or None, accepted_for_transition=False,
                deadline_exceeded=row['deadline_exceeded'], remaining_parts=row['remaining_parts'],
                part_id=row['part_move'], source_hashes=session.source_hashes,
                verifier_domain='ASAP_RedMax', observed_stage=(
                    'passed' if row['feasible'] else 'exception' if row.get('error_type') else
                    'path_search_returned_none' if raw.get('action') is None else 'stability_returned_none'),
                blocker=None, path=None, wall_seconds=session.guard.deadline-session.guard.started,
                elapsed_seconds=row['elapsed_seconds'])
            session.feedback.append(feedback)
            # event_id is the EventLog-owned identity; source identity is separate.
            logged = dict(row); logged['observation_id'] = logged.pop('event_id')
            session.emit('simulate_end', **logged)

    def update(instance, tree, parts_parent, parts_child, n_eval, sim_info):
        session.guard.check()  # Late raw PASS never changes the native tree.
        if session.active is None or not session.feedback:
            raise RuntimeError('native_transition_without_observed_simulation')
        accepted_at = session.elapsed()
        original_update(instance, tree, parts_parent, parts_child, n_eval, sim_info)
        session.last_tree = tree
        accepted = bool(sim_info['feasible'])
        session.active['accepted_for_transition'] = accepted
        session.feedback[-1]['accepted_for_transition'] = accepted
        transition = dict(parent=list(parts_parent), child=list(parts_child),
            sim_info=snapshot(sim_info), attempt_index=session.active['index'],
            accepted_for_transition=accepted, deadline_exceeded=False,
            accepted_at_elapsed_seconds=accepted_at if accepted else None,
            elapsed_seconds=session.elapsed())
        session.transitions.append(transition)
        chain = connected_path(tree, instance.parts)
        if accepted and chain['success'] and session.completion_elapsed_seconds is None:
            session.completion_elapsed_seconds = accepted_at
        session.emit('tree_transition', **transition)
        session.emit('tree_checkpoint', max_depth=chain['max_depth'], success=chain['success'],
                     counts=dict(session.guard.counts))

    try:
        for cls in set(registry.values()):
            for name in ('__init__', 'generate_candidate_part', 'heuristic_func'):
                if hasattr(cls, name):
                    saved.append((cls, name, name in cls.__dict__, getattr(cls, name)))
                    setattr(cls, name, session.forbidden_ranker)
        planner_class._simulate = simulate
        planner_class._update_tree = update
        yield SequenceCandidateGenerator
    finally:
        planner_class._simulate = original_simulate
        planner_class._update_tree = original_update
        for cls, name, owned, value in reversed(saved):
            if owned:
                setattr(cls, name, value)
            else:
                delattr(cls, name)
