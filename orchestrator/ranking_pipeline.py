"""Compose existing frozen evidence ranking with the bounded LLM service.

This host-side adapter owns no search state or physical verdicts. Memory only
promotes observed positive candidates within the offered permutation. A service
failure selects that deterministic order; it never prunes or bypasses ASAP.
"""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional

from .llm_ranker import RankService, _guard_path


# These three unchanged worker modules also travel with a frozen implementation.
# Their legacy absolute imports need the worker directory during module loading.
_WORKER = str(Path(__file__).absolute().parent.parent / "worker")
sys.path.insert(0, _WORKER)
try:
    from evidence_memory import canonical, digest
    from memory_ranker import MemoryRanker
    from rank_contract import RankRequest, rank_candidates
finally:
    sys.path.remove(_WORKER)


HINT_SCOPE = (
    "Bounded local verifier observations under the exact frozen context, not "
    "physical verdicts for this request or proof of impossibility. Positive "
    "evidence may promote offered candidates only. UNKNOWN and ERROR never "
    "penalize, remove, or exclude candidates. Every candidate still needs the "
    "existing ASAP verifier."
)


def _error_type(value: Any) -> Optional[str]:
    """Do not send exception messages, file contents, or paths to the service."""
    if value is None:
        return None
    name = str(value).split(":", 1)[0]
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else "MemoryError"


def _snapshot_path(value: Any) -> Path:
    # Lexical checks precede all filesystem calls, including on non-Windows hosts.
    raw = str(value).replace("\\", "/")
    normalized = "/".join(part for part in raw.lower().split("/") if part != ".")
    if (".." in raw.split("/") or normalized.startswith(("e:", "/mnt/e"))
            or raw.startswith("//")):
        raise ValueError("excluded_memory_path")
    return _guard_path(Path(value))


class RankingPipeline:
    """A seven-key worker response with auditable LLM and memory decisions.

    ``off`` never opens the evidence snapshot. With ``on``, the unchanged
    MemoryRanker loads FrozenMemory once and performs full-context matching.
    The caller validates and freezes the source snapshot/context before a run.
    Invalid evidence falls back to the registered base order, with a type-only
    error in metadata. Invalid requests remain invalid service responses.
    """

    def __init__(self, llm_config: dict, ledger_path: Path,
                 memory_snapshot: Optional[Path], context: dict,
                 memory_mode: str, base_ranker: str, transport=None):
        if memory_mode not in ("off", "on"):
            raise ValueError("memory_mode must be off or on")
        if base_ranker not in ("noop", "reverse_native"):
            raise ValueError("base_ranker must be noop or reverse_native")
        if type(context) is not dict:
            raise ValueError("context must be a finite JSON object")
        self.context = json.loads(canonical(context))
        self.context_digest = digest(self.context)
        self.memory_mode = memory_mode
        self.base_ranker = base_ranker
        self.memory_ranker = None
        self.memory_error = None
        if memory_mode == "on":
            try:
                if memory_snapshot is None:
                    raise ValueError("memory_snapshot_required")
                snapshot = _snapshot_path(memory_snapshot)
                self.memory_ranker = MemoryRanker(snapshot, self.context, "on")
            except (ValueError, TypeError, OSError) as error:
                self.memory_error = type(error).__name__
        self.service = RankService(llm_config, ledger_path, transport)

    def _memory_metadata(self) -> Dict[str, Any]:
        return dict(mode=self.memory_mode, snapshot_sha256=None,
                    context_digest=self.context_digest, matched_evidence_ids=[],
                    positive_candidate_ids=[], status_counts={}, changed=False,
                    fallback_reason=self.memory_error, evidence_loaded=False,
                    policy="positive_only_stable_promotion", scope="bounded_search",
                    usage="soft_ranking_only", allow_hard_pruning=False,
                    hint_scope=HINT_SCOPE)

    def _ordering(self, request: RankRequest):
        base = rank_candidates(request, self.base_ranker)
        if base.fallback_reason is not None:
            raise ValueError("registered_base_ranker_failed")
        metadata = self._memory_metadata()
        order = base.candidate_ids
        if self.memory_ranker is not None:
            order, observed = self.memory_ranker.apply(request, order)
            metadata.update(observed)
            metadata["fallback_reason"] = _error_type(observed["fallback_reason"])
            metadata["evidence_loaded"] = self.memory_ranker.memory is not None
        # The original metadata describes all matching state observations. This
        # subset makes explicit that evidence cannot introduce absent candidates.
        metadata["offered_positive_candidate_ids"] = [
            part for part in request.candidate_ids
            if part in metadata["positive_candidate_ids"]]
        return order, metadata

    def rank(self, envelope: dict) -> dict:
        try:
            # Reuse the service's validation/finite JSON boundary before lookup.
            payload = self.service._validate_envelope(envelope)
            enriched = json.loads(payload)
        except (ValueError, TypeError, RecursionError):
            response = self.service.rank(envelope)
            response["metadata"].update(
                llm_fallback_reason=response["fallback_reason"],
                ranking_source="deterministic_fallback", base_ranker=self.base_ranker,
                memory=self._memory_metadata(), input_envelope_sha256=None,
                llm_enriched_envelope_sha256=None)
            return response
        incoming = enriched["request"]
        request = RankRequest(
            incoming["run_id"], incoming["request_id"], incoming["state_id"],
            tuple(incoming["assembled_parts"]), tuple(incoming["candidate_ids"]),
            incoming["base_part"])
        order, memory = self._ordering(request)
        input_hash = digest(enriched)
        # Copy metadata separately so an injected transport/caller cannot alter
        # the selected fallback order or the response's local evidence record.
        enriched["features"]["memory"] = deepcopy(memory)
        enriched["features"]["deterministic_fallback_order"] = list(order)
        worker_timeout = float(enriched["timeout_seconds"])
        # The worker deadline also includes IPC, ledger work and response writing.
        # Leave five seconds for that work, or half of a short worker deadline.
        enriched["timeout_seconds"] = min(
            30.0, worker_timeout - 5.0 if worker_timeout > 5.0 else worker_timeout * 0.5)
        enriched_hash = digest(enriched)
        response = self.service.rank(enriched)
        reason = response["fallback_reason"]
        if reason is not None:
            response["candidate_ids"] = list(order)
            # A non-null worker fallback would undo this valid deterministic
            # permutation. Preserve the service reason in metadata instead.
            response["fallback_reason"] = None
        response["metadata"].update(
            llm_fallback_reason=reason,
            ranking_source="deterministic_fallback" if reason else "llm",
            base_ranker=self.base_ranker, memory=memory,
            input_envelope_sha256=input_hash,
            llm_enriched_envelope_sha256=enriched_hash,
            worker_timeout_seconds=worker_timeout,
            llm_timeout_seconds=enriched["timeout_seconds"])
        return response
