"""Opt-in automatic-recovery policy; a refusal is never a physics verdict."""

POLICIES = ("legacy_v1", "preserve_existing_v1")


def check_recovery_supports(reference, choice, request):
    """Do not silently release existing supports during preauthorized recovery.

    A fresh explicit LLM plan may choose a different support set. No candidate
    is removed, no action is rewritten, and no unavailable support is invented.
    """
    facts = dict(policy="preserve_existing_v1", allowed=False, reason=None,
                 physical_verdict=None, hard_pruning=False, action_rewritten=False,
                 requires_new_explicit_plan=False)
    if not reference or set(reference["remaining_parts"]) != set(request["remaining_parts"]):
        facts.update(reason="recovery_support_context_unavailable", requires_new_explicit_plan=True)
        return facts
    previous = set(reference["choice"]["support_ids"])
    proposed = set(choice["support_ids"])
    released = previous - proposed
    capacity = request["max_grippers"] - 1
    facts.update(previous_support_ids=sorted(previous), proposed_support_ids=sorted(proposed),
        released_support_ids=sorted(released), added_support_ids=sorted(proposed-previous),
        union_support_ids=sorted(previous | proposed), support_capacity=capacity,
        union_exceeds_capacity=len(previous | proposed) > capacity,
        source_evidence_id=reference["evidence_id"])
    if released:
        facts.update(reason="recovery_support_release_requires_new_plan", requires_new_explicit_plan=True)
    else:
        facts.update(allowed=True, reason="existing_supports_preserved")
    return facts
