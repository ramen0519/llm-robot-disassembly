"""Frozen positive evidence supplies ordering hints, never physical verdicts."""
import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

from evidence_memory import FrozenMemory, digest
from rank_contract import RankRequest, RankResponse, rank_candidates


BEAM_SETUP = {
    "budget": 400, "max_grippers": 2, "max_poses": 3, "pose_reuse": 0,
    "early_term": True, "timeout": None, "plan_grasp": False, "plan_arm": False,
    "gripper_type": "robotiq-140", "gripper_scale": 0.4, "optimizer": "L-BFGS-B",
}


class MemoryRanker:
    def __init__(self, snapshot: Path, context: dict, mode: str):
        if mode not in ("off", "on"):
            raise ValueError("memory mode must be off or on")
        self.mode = mode
        self.context = context
        self.context_digest = digest(context)
        self.snapshot_sha256 = None
        self.memory = None
        self.load_error = None
        try:
            self.snapshot_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            self.memory = FrozenMemory(snapshot)
        except Exception as error:
            self.load_error = type(error).__name__ + ": " + str(error)

    def apply(self, request: RankRequest, base_order: tuple) -> Tuple[tuple, Dict[str, Any]]:
        metadata = dict(mode=self.mode, snapshot_sha256=self.snapshot_sha256,
                        context_digest=self.context_digest, matched_evidence_ids=[],
                        positive_candidate_ids=[], status_counts={}, changed=False,
                        fallback_reason=self.load_error)
        if self.mode == "off" or self.load_error is not None:
            return base_order, metadata
        try:
            found = self.memory.lookup(self.context, request.assembled_parts, request.base_part,
                                       pose=None, max_grippers=BEAM_SETUP["max_grippers"])
            positives = set(found["positive_candidate_ids"])
            proposed = tuple(part for part in base_order if part in positives)
            proposed += tuple(part for part in base_order if part not in positives)
            # Reuse the same strict permutation contract for the final response.
            def propose(current: RankRequest) -> RankResponse:
                return RankResponse(current.request_id, current.state_id, proposed)
            checked = rank_candidates(request, "memory", {"memory": propose})
            if checked.fallback_reason is not None:
                raise ValueError(checked.fallback_reason)
            metadata.update(matched_evidence_ids=found["evidence_ids"],
                            positive_candidate_ids=found["positive_candidate_ids"],
                            status_counts=found["status_counts"], changed=proposed != base_order)
            return checked.candidate_ids, metadata
        except Exception as error:
            # A lookup/format error keeps the already validated deterministic base order.
            metadata["fallback_reason"] = type(error).__name__ + ": " + str(error)
            return base_order, metadata
