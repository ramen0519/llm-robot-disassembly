"""Finite model-candidate allocation; transport errors never get another call."""
from dataclasses import dataclass, field


@dataclass
class SplitRevisionBudget:
    physical_revisions: int = 2
    contract_corrections: int = 2
    used: dict = field(default_factory=lambda: {'physical_revision': 0, 'contract_correction': 0})
    next_kind: str | None = 'physical_revision'
    pending: bool = False
    stop_reason: str | None = None
    single_candidate: bool = False

    def __post_init__(self):
        if type(self.single_candidate) is not bool:
            raise ValueError('explicit_single_candidate_mode_required')
        counts=(self.physical_revisions,self.contract_corrections)
        if (any(type(n) is not int for n in counts)
                or (self.single_candidate and counts != (1,0))
                or (not self.single_candidate and any(not 1 <= n <= 3 for n in counts))):
            raise ValueError('bounded_separate_allocations_required')

    def allocate(self):
        if self.pending:
            raise ValueError('previous_candidate_outcome_required')
        if self.next_kind is None:
            return None
        kind = self.next_kind
        cap = self.physical_revisions if kind == 'physical_revision' else self.contract_corrections
        if self.used[kind] >= cap:
            self.stop_reason = kind + '_allocation_exhausted'
            self.next_kind = None
            return None
        self.used[kind] += 1
        self.pending = True
        return kind

    def observe(self, outcome):
        if not self.pending:
            raise ValueError('candidate_allocation_required')
        self.pending = False
        if outcome == 'contract_rejected':
            self.next_kind = 'contract_correction'
        elif outcome == 'measured_debug_failure':
            self.next_kind = 'physical_revision'
        elif outcome in ('registered', 'stop'):
            self.next_kind = None
            self.stop_reason = outcome
        else:
            raise ValueError('unknown_candidate_outcome')

    def summary(self):
        return dict(used=dict(self.used), physical_revisions=self.physical_revisions,
                    contract_corrections=self.contract_corrections, stop_reason=self.stop_reason,
                    http_retries=0, allocations_are_candidate_attempts_not_successes=True)
