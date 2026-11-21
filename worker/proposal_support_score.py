"""Rank path proposals by the supports witnessed at their selected poses."""
try:
    from .path_only_sequence_proposal import PROFILE, build_proposal
    from .explicit_action_physics import _ids, _pose
    from .proposal_support_policy import effective_support_cap, parse_support_cap
except ImportError:  # WSL worker launched as a script
    from path_only_sequence_proposal import PROFILE, build_proposal
    from explicit_action_physics import _ids, _pose
    from proposal_support_policy import effective_support_cap, parse_support_cap


SCORE_PROFILE = 'witnessed_support_cost/v1'
AUDIT_STATUSES = {'PASS', 'REJECTED', 'UNKNOWN', 'ERROR', 'NOT_RUN'}


def _support_count(step, cap):
    audit = step.get('stability')
    if audit is None:
        return 'NOT_RUN', None
    status = audit.get('up_to_cap_status')
    if status not in AUDIT_STATUSES:
        raise ValueError('invalid support audit status')
    child = set(step['child'])
    effective_cap = effective_support_cap(cap, len(child))
    if 'budget' in audit and (type(audit['budget'].get('support_cap')) is not int or
                              audit['budget']['support_cap'] != effective_cap):
        raise ValueError('audit performed at a different support cap')
    if status != 'PASS':
        if audit.get('held_parts') is not None:
            raise ValueError('unsuccessful audit reports a support witness')
        return status, None

    held = set(_ids(audit.get('held_parts'), 'held_parts'))
    trials = audit.get('trials')
    if 'budget' not in audit or not held <= child or len(held) > effective_cap or not trials:
        raise ValueError('missing or out-of-cap support witness')
    first, last = trials[0], trials[-1]
    if (first.get('parts_fix') != [] or audit.get('no_support_status') != first.get('status') or
            audit['no_support_status'] != ('REJECTED' if held else 'PASS')):
        raise ValueError('inconsistent zero-support trial')
    if (last.get('status') != 'PASS' or last.get('success') is not True or
            last.get('deadline_exhausted') is not False or 'parts_fall' not in last or
            last['parts_fall'] not in (None, []) or last.get('all_remaining_fixed') is not (child == held) or
            set(_ids(last.get('parts_fix'), 'trial_fixed')) != held or
            set(_ids(last.get('parts_move'), 'trial_free')) != child - held or
            audit.get('all_remaining_fixed') is not (child == held)):
        raise ValueError('support witness differs from final successful native trial')
    return status, len(held)


def score_plan(result, *, support_cap):
    """Smaller (peak held parts, sum of held parts) wins among eligible plans."""
    support_cap = parse_support_cap(support_cap)
    if result.get('profile') != PROFILE:
        raise ValueError('unsupported proposal profile')
    steps = result['steps']
    for step in steps:
        if step.get('accepted_for_transition') is not True or step.get('deadline_exceeded') is not False:
            raise ValueError('selected path lacks timely transition acceptance')
        _pose(step['pose'])
    chain = build_proposal(result['initial_parts'], steps, profile=result['profile'])
    for field in ('complete', 'steps', 'disassembly_order', 'remaining_parts', 'assembly_order_candidate'):
        if chain[field] != result[field]:
            raise ValueError('proposal does not match the connected removal chain: ' + field)

    statuses, counts = [], []
    for step in steps:
        status, count = _support_count(step, support_cap)
        statuses.append(status)
        counts.append(count)
    prefix = next((i for i, v in enumerate(counts) if v is None), len(counts))
    passed = len(counts) - counts.count(None)
    eligible = (chain['complete'] and passed == len(steps) and
                result.get('execution_status') == 'OK' and result.get('input_unchanged') is True and
                result.get('asap_sources_unchanged') is True)
    return dict(profile=SCORE_PROFILE, support_cap=support_cap, eligible=eligible,
                score=[max(counts), sum(counts)] if eligible else None,
                step_counts=counts, step_statuses=statuses, stable_prefix=prefix,
                passed_steps=passed, step_count=len(steps),
                first_blocking_step=prefix+1 if prefix < len(steps) else None,
                first_blocking_part=steps[prefix]['part_move'] if prefix < len(steps) else None)
