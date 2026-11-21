"""Process-local instrumentation for the pinned ASAP heur-out generator.

The caller checks the source revision. Trace mode retains the lazy iterator;
noop/rank modes materialize it once, then offer an immutable snapshot to a ranker.
Only this generator is supported: it finishes RNG/sorting before first yield.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, Optional, Type

from rank_contract import RankRequest, rank_candidates


def json_value(value: Any) -> Any:
    """Snapshot NumPy values without retaining references to solver objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    raise TypeError("unsupported trace value: " + type(value).__name__)


class EventLog:
    def __init__(self, path: Path, run_id: str):
        self.run_id = run_id
        self.count = 0
        self.stream = path.open("x", encoding="utf-8")

    def emit(self, kind: str, **fields: Any) -> None:
        self.count += 1
        event = dict(run_id=self.run_id, event_id="{}:e{}".format(self.run_id, self.count),
                     kind=kind, **fields)
        self.stream.write(json.dumps(json_value(event), sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def state_id(parts: tuple) -> str:
    # Within-run membership ID only, never an exact physical-state/cache key.
    return hashlib.sha256(json.dumps(sorted(parts)).encode("utf-8")).hexdigest()


@contextmanager
def installed_hooks(generator_class: Type, planner_class: Type, log: EventLog,
                    mode: str, rng_digest: Callable[[], str],
                    ranker_name: str = "noop", memory: Any = None) -> Iterator[None]:
    if mode not in ("trace", "noop", "rank"):
        raise ValueError("unsupported hook mode")
    if mode != "rank" and ranker_name != "noop":
        raise ValueError("a non-noop ranker requires rank mode")
    if memory is not None and mode != "rank":
        raise ValueError("memory requires rank mode")
    original_generate = generator_class.generate_candidate_part
    original_simulate = planner_class._simulate
    simulate_signature = inspect.signature(original_simulate)
    generation_count = 0
    verifier_count = 0
    last_generation = None  # only observational membership, never a search stack
    last_parts = None

    def generate(instance: Any, assembled_parts: list) -> Iterator[str]:
        nonlocal generation_count, last_generation, last_parts
        generation_count += 1
        generation_id = "g{}".format(generation_count)
        parts = tuple(assembled_parts)
        before = rng_digest()
        native_iterator = original_generate(instance, assembled_parts)
        if mode == "trace":
            # The pinned upstream computes sorted_parts before its first yield.
            # Observe locals without draining unused candidates/RNG/work.
            try:
                first = next(native_iterator)
            except StopIteration:
                native_order = ()
                first = None
            else:
                native_order = tuple(native_iterator.gi_frame.f_locals["sorted_parts"])
                if not native_order or native_order[0] != first:
                    raise RuntimeError("pinned generator trace contract changed")
            applied_order = native_order
            fallback_reason = None
        else:
            native_order = tuple(native_iterator)
            request = RankRequest(log.run_id, "{}:r{}".format(log.run_id, generation_count),
                                  state_id(parts), parts, native_order, instance.base_part)
            rank_rng_before = rng_digest()
            rank_started = perf_counter()
            decision = rank_candidates(request, ranker_name)
            extra = {}
            applied_order, fallback_reason = decision.candidate_ids, decision.fallback_reason
            if memory is not None:
                base_order = applied_order
                applied_order, memory_metadata = memory.apply(request, base_order)
                extra = dict(base_order=base_order, memory=memory_metadata)
            rank_seconds = perf_counter() - rank_started
            log.emit("rank_decision", generation_id=generation_id, request=asdict(request),
                     applied_order=applied_order, fallback_reason=fallback_reason,
                     ranker_name=ranker_name, rank_seconds=rank_seconds,
                     rng_before=rank_rng_before, rng_after=rng_digest(), **extra)
        log.emit("generation", generation_id=generation_id, state_id=state_id(parts),
                 assembled_parts=parts, native_order=native_order, applied_order=applied_order,
                 rng_before=before, rng_after=rng_digest())
        last_generation, last_parts = generation_id, frozenset(parts)
        if mode == "trace":
            if first is None:
                return
            log.emit("candidate_yield", generation_id=generation_id, part=first)
            yield first
            for part in native_iterator:
                log.emit("candidate_yield", generation_id=generation_id, part=part)
                yield part
        else:
            for part in applied_order:
                log.emit("candidate_yield", generation_id=generation_id, part=part)
                yield part

    def simulate(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        nonlocal verifier_count
        verifier_count += 1
        call_id = "v{}".format(verifier_count)
        bound = simulate_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        # Deliberately exclude planner and mutable solver objects from the snapshot.
        request = json_value({name: arguments[name] for name in (
            "part_move", "parts_rest", "parts_removed", "pose", "max_grippers",
            "timeout", "optimizer", "debug", "render")})
        membership = frozenset([request["part_move"]] + request["parts_rest"])
        generation_id = last_generation if membership == last_parts else None
        log.emit("verifier_start", call_id=call_id, generation_id=generation_id,
                 request=request, rng_before=rng_digest())
        try:
            raw = original_simulate(*args, **kwargs)
        except Exception as error:
            log.emit("verifier_result", call_id=call_id, status="ERROR", raw=None,
                     error_type=type(error).__name__, blocker=None, path=None,
                     rng_after=rng_digest())
            raise
        log.emit("verifier_result", call_id=call_id,
                 status="PASS" if raw.get("feasible") is True else "UNKNOWN",
                 raw=json_value(raw), blocker=None, path=None, rng_after=rng_digest())
        return raw

    generator_class.generate_candidate_part = generate
    planner_class._simulate = simulate
    try:
        yield
    finally:
        generator_class.generate_candidate_part = original_generate
        planner_class._simulate = original_simulate
