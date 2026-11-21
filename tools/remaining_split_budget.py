"""Read-only residual proposal. Does not authorize, fork or resume paid calls."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import sqlite3
from tools.run_integrated_repair import checked
from orchestrator.hierarchy_budget import validate_policy
from worker.task_memory import digest
from pathlib import Path

CLAIM_ROOT = Path(__file__).absolute().parents[1]/'runs/budget_continuation_claims'


def remaining_proposal(ledger):
    path = checked(ledger, True)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(path.absolute().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        policy_row = db.execute('SELECT binding,policy FROM hierarchy_policy WHERE id=1').fetchone()
        rows = [dict(r) for r in db.execute('SELECT * FROM hierarchy_calls ORDER BY id')]
    policy = validate_policy(json.loads(policy_row['policy']))
    if policy_row['binding'] != digest(policy) or not policy['enabled']:
        raise ValueError('approved_parent_policy_binding_required')
    if any(r['role'] not in policy['roles'] or r['live'] != 1 for r in rows):
        raise ValueError('actual_shared_budget_required')
    def remaining(records, cfg):
        amounts = [max(Decimal(r['estimated_usd']), Decimal(r['actual_usd'] or '0')) for r in records]
        if any(not n.is_finite() or n < 0 for n in amounts):
            raise ValueError('invalid_accounting_amount')
        calls = cfg['max_calls'] - len(records)
        usd = Decimal(cfg['max_usd']) - sum(amounts, Decimal(0))
        if calls < 0 or usd < 0:
            raise ValueError('parent_budget_exhausted_or_invalid')
        return calls, str(usd)
    proposed = deepcopy(policy)
    proposed['enabled'] = False
    proposed['max_calls'], proposed['max_usd'] = remaining(rows, policy)
    for role,cfg in proposed['roles'].items():
        cfg['max_calls'],cfg['max_usd'] = remaining([r for r in rows if r['role']==role],cfg)
    validate_policy(proposed)
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError('parent_ledger_changed_during_read')
    return dict(kind='read_only_remaining_budget_proposal_NOT_authorization',
        parent_ledger=str(path), parent_ledger_sha256=before, parent_policy_sha256=policy_row['binding'],
        parent_rows_sha256=digest(rows), prior_calls=len(rows),
        pending_or_unknown=sum(r['status']!='completed' for r in rows),
        unknown_reservations_retained=True, parent_unchanged=True, live_execution_enabled=False,
        new_calls=0, policy=proposed)


def validate_remaining_binding(binding, policy, *, reallocation=None, restriction=False):
    if set(binding) != {'parent_ledger', 'parent_ledger_sha256', 'parent_policy_sha256', 'parent_rows_sha256'}:
        raise ValueError('explicit_remaining_budget_binding_required')
    current = remaining_proposal(binding['parent_ledger'])
    if any(current[k] != v for k,v in binding.items()):
        raise ValueError('remaining_budget_parent_changed')
    policy=validate_policy(policy)
    expected = deepcopy(current['policy'])
    expected['enabled'] = policy['enabled']
    if type(restriction) is not bool:
        raise ValueError('explicit_boolean_restriction_required')
    if restriction:
        if reallocation is not None:
            raise ValueError('restriction_cannot_reallocate_roles')
        # Opt-in smaller continuation: only unused call/USD caps may decrease.
        # All prices, models, endpoints, per-episode limits and unknown-cost
        # reservations remain bound to the immutable parent ledger.
        for before, after in [(expected, policy)] + [
                (expected['roles'][r], policy['roles'][r]) for r in expected['roles']]:
            if (after['max_calls'] > before['max_calls']
                    or Decimal(after['max_usd']) > Decimal(before['max_usd'])):
                raise ValueError('restricted_budget_cannot_increase_any_cap')
            before['max_calls'], before['max_usd'] = after['max_calls'], after['max_usd']
    if type(reallocation) is dict and reallocation.get('kind') == 'abstention_candidate_5_v1':
        if set(reallocation) != {'kind','authorization_path','authorization_sha256'}:
            raise ValueError('explicit_five_call_reallocation_required')
        path = checked(reallocation['authorization_path'], True)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reallocation['authorization_sha256']:
            raise ValueError('reallocation_authorization_changed')
        approval = json.loads(raw)
        if (approval.get('approved') is not True
                or approval.get('scope') != 'remaining_five_calls_abstention_v1'
                or approval.get('budget_parent') != binding
                or approval.get('remaining_policy_sha256') != digest(current['policy'])
                or type(approval.get('maximum_new_api_attempts')) is not int
                or approval['maximum_new_api_attempts'] != 5
                or approval.get('maximum_program_estimated_usd') != current['policy']['max_usd']
                or approval.get('role_calls') != dict(learner=2, scout=0, planner=3)
                or approval.get('automatic_http_retries') != 0
                or approval.get('no_revision_after_failure') is not True):
            raise ValueError('bound_explicit_reallocation_approval_required')
        if (expected['max_calls'] != 5
                or {k:c['max_calls'] for k,c in expected['roles'].items()} != dict(learner=1,scout=1,planner=3)
                or expected['episode']['scout_calls'] != 1
                or expected['episode']['planner_calls'] != 3
                or expected['episode']['fallback_calls'] != 0):
            raise ValueError('unsupported_five_call_parent_allocation')
        # Reassign only the unused scout call. Preserve all other role/global
        # money caps, prices, model IDs and unknown-cost reservations.
        expected['roles']['learner']['max_calls'] = 2
        expected['roles']['scout'].update(max_calls=0, max_usd='0')
        expected['episode']['scout_calls'] = 0
    elif type(reallocation) is dict and reallocation.get('kind') == 'single_candidate_3_v1':
        if set(reallocation) != {'kind','authorization_path','authorization_sha256'}:
            raise ValueError('explicit_three_call_reallocation_required')
        path = checked(reallocation['authorization_path'], True)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reallocation['authorization_sha256']:
            raise ValueError('reallocation_authorization_changed')
        approval = json.loads(raw)
        if (approval.get('approved') is not True
                or approval.get('scope') != 'remaining_three_calls_reallocated_v1'
                or approval.get('budget_parent') != binding
                or approval.get('remaining_policy_sha256') != digest(current['policy'])
                or type(approval.get('maximum_new_api_attempts')) is not int
                or approval.get('maximum_new_api_attempts') != 3
                or approval.get('maximum_program_estimated_usd') != current['policy']['max_usd']
                or approval.get('role_calls') != dict(learner=2,scout=0,planner=1)
                or approval.get('automatic_http_retries') != 0
                or approval.get('no_revision_after_failure') is not True):
            raise ValueError('bound_explicit_reallocation_approval_required')
        if (expected['max_calls'] != 3
                or {k:c['max_calls'] for k,c in expected['roles'].items()} != dict(learner=1,scout=1,planner=1)
                or expected['episode']['scout_calls'] != 1
                or expected['episode']['planner_calls'] != 1
                or expected['episode']['fallback_calls'] != 0):
            raise ValueError('unsupported_three_call_parent_allocation')
        expected['roles']['learner']['max_calls'] = 2
        expected['roles']['scout']['max_calls'] = 0
        expected['roles']['scout']['max_usd'] = '0'
        expected['episode']['scout_calls'] = 0
        # The unused scout dollars stay unallocated. Global/learner/planner
        # caps and all rates/models are unchanged; unknown costs stay reserved.
    elif reallocation is not None:
        if (type(reallocation) is not dict or set(reallocation) != {'kind','authorization_path','authorization_sha256'}
                or reallocation['kind'] != 'single_candidate_4_v1'):
            raise ValueError('explicit_four_call_reallocation_required')
        path=checked(reallocation['authorization_path'],True)
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=reallocation['authorization_sha256']:
            raise ValueError('reallocation_authorization_changed')
        approval=json.loads(raw)
        if (approval.get('approved') is not True
                or approval.get('scope')!='remaining_four_calls_reallocated_v1'
                or approval.get('budget_parent')!=binding
                or approval.get('remaining_policy_sha256')!=digest(current['policy'])
                or type(approval.get('maximum_new_api_attempts')) is not int
                or approval.get('maximum_new_api_attempts')!=4
                or approval.get('maximum_program_estimated_usd')!=current['policy']['max_usd']
                or approval.get('role_calls')!=dict(learner=2,scout=1,planner=1)
                or approval.get('automatic_http_retries')!=0):
            raise ValueError('bound_explicit_reallocation_approval_required')
        if (expected['max_calls']!=4
                or {k:c['max_calls'] for k,c in expected['roles'].items()}!=dict(learner=1,scout=1,planner=2)
                or expected['episode']['scout_calls']!=1
                or expected['episode']['planner_calls']!=2
                or expected['episode']['fallback_calls']!=0):
            raise ValueError('unsupported_four_call_parent_allocation')
        # Only the exact user-approved call transfer. No USD, rate, model,
        # endpoint, output ceiling, or unknown reservation is changed/refunded.
        expected['roles']['learner']['max_calls']=2
        expected['roles']['planner']['max_calls']=1
        expected['episode']['planner_calls']=1
    if expected != policy:
        raise ValueError('remaining_budget_policy_cannot_increase_or_reallocate')
    return current


def claim_remaining_binding(binding, policy, output, *, reallocation=None, restriction=False):
    """One exclusive continuation per parent snapshot, only at explicit live launch."""
    if policy['enabled'] is not True:
        raise ValueError('disabled_proposal_cannot_claim_budget')
    validate_remaining_binding(binding, policy,reallocation=reallocation,restriction=restriction)
    output = checked(output, True)
    claim_dir = checked(CLAIM_ROOT, True)
    claim_dir.mkdir(parents=True, exist_ok=True)
    identity = digest({k:binding[k] for k in ('parent_ledger','parent_policy_sha256','parent_rows_sha256')})
    claim = dict(parent=deepcopy(binding), destination=str(output), policy_sha256=digest(policy),
                 unknown_reservations_retained=True, automatic_resend=False)
    if reallocation is not None:
        claim['explicit_reallocation']=deepcopy(reallocation)
    if restriction:
        claim['strictly_bounded_suballocation'] = True
    # Exclusive creation is also the concurrent-launch guard. A crash does not
    # silently release this claim or grant another copy of the remaining budget.
    with checked(claim_dir/(identity+'.json'), True).open('x', encoding='utf-8') as f:
        json.dump(claim,f,ensure_ascii=False,indent=2)
    return claim
