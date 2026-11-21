"""Python 3.8 local ranking contract. Owns no ASAP search/physics state."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class RankRequest:
    run_id: str
    request_id: str
    state_id: str
    assembled_parts: Tuple[str, ...]
    candidate_ids: Tuple[str, ...]
    base_part: Optional[str]


@dataclass(frozen=True)
class RankResponse:
    request_id: str
    state_id: str
    candidate_ids: Tuple[str, ...]


@dataclass(frozen=True)
class RankDecision:
    candidate_ids: Tuple[str, ...]
    fallback_reason: Optional[str]


def noop_ranker(request: RankRequest) -> RankResponse:
    return RankResponse(request.request_id, request.state_id, request.candidate_ids)


def id_ascending_ranker(request: RankRequest) -> RankResponse:
    """Lexicographic ID control, not a geometric feasibility heuristic."""
    return RankResponse(request.request_id, request.state_id,
                        tuple(sorted(request.candidate_ids)))


def reverse_native_ranker(request: RankRequest) -> RankResponse:
    """Reverse the offered heur-out order as a fixed sensitivity control."""
    return RankResponse(request.request_id, request.state_id,
                        tuple(reversed(request.candidate_ids)))


# Reviewed local functions only. No imports/eval from ranker output or CLI text.
REGISTRY = MappingProxyType({"noop": noop_ranker, "id_ascending": id_ascending_ranker,
                             "reverse_native": reverse_native_ranker})


def rank_candidates(request: RankRequest, name: str,
                    registry: Mapping[str, Callable[[RankRequest], RankResponse]] = REGISTRY
                    ) -> RankDecision:
    """Reject invalid output; fall back to this request's seeded native order.

    Synchronous local calls only: this is not an API timeout/cost controller.
    Invalid upstream input is an integration error, not a ranker fallback.
    """
    candidates = request.candidate_ids
    if (type(candidates) is not tuple
            or any(type(p) is not str for p in candidates)
            or len(set(candidates)) != len(candidates)
            or not set(candidates).issubset(request.assembled_parts)
            or request.base_part in candidates):
        raise ValueError("upstream candidates must be unique non-base part IDs")
    if name not in registry:
        return RankDecision(candidates, "unknown_ranker")
    try:
        response = registry[name](request)
    except Exception as error:
        return RankDecision(candidates, "ranker_exception:" + type(error).__name__)
    reason = None
    if type(response) is not RankResponse:
        reason = "invalid_response_type"
    elif response.request_id != request.request_id or response.state_id != request.state_id:
        reason = "stale_request_or_state"
    elif (type(response.candidate_ids) is not tuple
          or any(type(p) is not str for p in response.candidate_ids)):
        reason = "invalid_candidate_type"
    elif (len(response.candidate_ids) != len(candidates)
          or len(set(response.candidate_ids)) != len(response.candidate_ids)
          or set(response.candidate_ids) != set(candidates)):
        reason = "invalid_permutation"
    return RankDecision(candidates if reason else response.candidate_ids, reason)
