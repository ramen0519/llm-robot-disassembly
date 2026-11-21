"""Independent enumeration/ranking, with unchanged ASAP search and verifier (3.8).

The session holds observations only. It never owns the DFS stack or updates a
search node. Canonical ID order is serialization/fallback, not a CAD heuristic.
"""
from contextlib import contextmanager
from dataclasses import asdict
import inspect
import json
from time import perf_counter

from independent_features import filter_features
from rank_contract import RankRequest, rank_candidates
from rank_hook import json_value, state_id
from task_memory import task_context


def enumerate_candidates(assembled_parts, base_part=None):
    if isinstance(assembled_parts, (str, bytes, dict)):
        raise ValueError('remaining_parts_must_be_an_ID_collection')
    parts = tuple(assembled_parts)
    if (any(type(p) is not str or not p for p in parts)
            or len(parts) != len(set(parts))
            or (base_part is not None and base_part not in parts)):
        raise ValueError("invalid_remaining_parts_or_base")
    return tuple(sorted(p for p in parts if p != base_part))


def validate_complete_request(request):
    expected = enumerate_candidates(request.assembled_parts, request.base_part)
    if request.candidate_ids != expected:
        raise ValueError("independent_input_requires_all_candidates_in_canonical_order")
    if request.assembled_parts != tuple(sorted(request.assembled_parts)):
        raise ValueError("remaining_parts_must_be_canonical")


class IndependentSession:
    def __init__(self, log, client, features, rng_digest):
        self.log, self.client = log, client
        self.features = json.loads(json.dumps(features, allow_nan=False))
        self.context = task_context(self.features)
        self.rng_digest = rng_digest
        self.generation_count = 0
        self.verifier_count = 0
        self.last_generation = None
        self.last_parts = None
        self.feedback = []
        self.tripwire_calls = 0

    def generate(self, assembled_parts, base_part):
        parts = tuple(sorted(assembled_parts))
        candidates = enumerate_candidates(parts, base_part)
        if base_part != self.features['task']['base_part']:
            raise ValueError("base_part_differs_from_cad_task")
        self.generation_count += 1
        generation = 'g' + str(self.generation_count)
        request = RankRequest(self.log.run_id,
                              self.log.run_id + ':r' + str(self.generation_count),
                              state_id(parts), parts, candidates, base_part)
        validate_complete_request(request)
        before = self.rng_digest()
        # Single-thread worker: refresh the existing file client's observation
        # snapshot before each request. No search-state references cross IPC.
        self.client.features = {
            'protocol': 'independent_rank_v1',
            'cad': filter_features(self.features, parts),
            'memory_context': self.context,
            'episode_feedback': self.feedback[-12:],
            'validated_repair_skills': [],
        }
        started = perf_counter()
        decision = rank_candidates(request, 'independent', {'independent': self.client})
        self.log.emit('rank_decision', generation_id=generation, request=asdict(request),
                      applied_order=decision.candidate_ids,
                      fallback_reason=decision.fallback_reason,
                      fallback_policy='canonical_id_order', ranker_name='independent',
                      rank_seconds=perf_counter() - started,
                      rng_before=before, rng_after=self.rng_digest())
        self.log.emit('generation', generation_id=generation, state_id=request.state_id,
                      assembled_parts=parts, enumerated_candidates=candidates,
                      applied_order=decision.candidate_ids, full_candidate_set=True,
                      heuristic_or_gnn_invoked=False,
                      rng_before=before, rng_after=self.rng_digest())
        self.last_generation, self.last_parts = generation, frozenset(parts)
        for part in decision.candidate_ids:
            self.log.emit('candidate_yield', generation_id=generation, part=part)
            yield part

    def forbid_ranker(self, *args, **kwargs):
        self.tripwire_calls += 1
        raise RuntimeError('heuristic_or_gnn_called_in_independent_route')


@contextmanager
def installed_independent(generator_base, registry, planner_class, session):
    """Register a new Generator and observe _simulate; restore everything.

Poison the old generator classes during this route. A completed run with zero
tripwire calls proves they were not silently used as a ranker or fallback.
"""
    if 'independent' in registry:
        raise ValueError('independent_generator_already_registered')
    original_simulate = planner_class._simulate
    signature = inspect.signature(original_simulate)
    saved = []

    class IndependentCandidateGenerator(generator_base):
        def generate_candidate_part(self, assembled_parts):
            return session.generate(assembled_parts, self.base_part)

    def simulate(*args, **kwargs):
        session.verifier_count += 1
        call_id = 'v' + str(session.verifier_count)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        request = json_value({key: bound.arguments[key] for key in (
            'part_move', 'parts_rest', 'parts_removed', 'pose', 'max_grippers',
            'timeout', 'optimizer', 'debug', 'render')})
        membership = frozenset([request['part_move']] + request['parts_rest'])
        generation = session.last_generation if membership == session.last_parts else None
        session.log.emit('verifier_start', call_id=call_id, generation_id=generation,
                         request=request, rng_before=session.rng_digest(),
                         selection_scope='ranker' if generation else 'upstream_terminal_completion')
        started = perf_counter()
        try:
            raw = original_simulate(*args, **kwargs)
        except Exception as error:
            session.log.emit('verifier_result', call_id=call_id, status='ERROR', raw=None,
                             error_type=type(error).__name__, blocker=None, path=None,
                             elapsed_seconds=perf_counter()-started,
                             rng_after=session.rng_digest())
            session.feedback.append(dict(event_id=session.log.run_id + ':e' + str(session.log.count),
                call_id=call_id, status='ERROR', request=request, raw=None,
                observed_stage='exception', blocker=None, path=None))
            raise
        snapshot = json_value(raw)
        status = 'PASS' if snapshot.get('feasible') is True else 'UNKNOWN'
        stage = ('passed' if status == 'PASS' else 'path_search_returned_none'
                 if snapshot.get('action') is None else 'stability_returned_none'
                 if snapshot.get('parts_fix') is None else 'unclassified')
        session.log.emit('verifier_result', call_id=call_id, status=status, raw=snapshot,
                         observed_stage=stage, blocker=None, path=None,
                         stage_feedback={'path_search_attempted': True,
                                         'path_found': snapshot.get('action') is not None,
                                         'support_search_attempted': snapshot.get('action') is not None,
                                         'support_search_succeeded': snapshot.get('parts_fix') is not None},
                         elapsed_seconds=perf_counter()-started, rng_after=session.rng_digest())
        session.feedback.append(dict(event_id=session.log.run_id + ':e' + str(session.log.count),
            call_id=call_id, status=status, request=request, raw=snapshot,
            observed_stage=stage, blocker=None, path=None))
        return raw

    try:
        for cls in set(registry.values()):
            for name in ('__init__', 'generate_candidate_part', 'heuristic_func'):
                if hasattr(cls, name):
                    saved.append((cls, name, name in cls.__dict__, getattr(cls, name)))
                    setattr(cls, name, session.forbid_ranker)
        registry['independent'] = IndependentCandidateGenerator
        planner_class._simulate = simulate
        yield IndependentCandidateGenerator
    finally:
        planner_class._simulate = original_simulate
        registry.pop('independent', None)
        for cls, name, owned, value in reversed(saved):
            if owned:
                setattr(cls, name, value)
            else:
                delattr(cls, name)
