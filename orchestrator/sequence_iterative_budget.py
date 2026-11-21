"""Pure finite budgets for explicitly gated sequence-learning stages.

No file, environment, key, HTTP, or simulator access. These calculations are
proposals and contract checks, never proof of user authorization. The runner
owns persisted checkpoint review and the live-execution permission gates.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation

PROFILE = 'sequence_iterative_learning_v1'
TASK_COUNT = 5
STAGES = {'initial': (1, 2), 'continuation': (3, 4, 5)}
ROLES = {
    'sol': dict(model='gpt-6.1-sol', calls_per_round=25, usd_per_round='4.00',
                max_body_bytes=65536, max_output_tokens=2048, request_overhead_tokens=2048,
                input_usd_per_million='2', output_usd_per_million='10',
                pricing_source='https://developers.openai.com/api/docs/models/gpt-6.1-sol'),
    'luna': dict(model='gpt-6-luna', calls_per_round=4, usd_per_round='0.10',
                 max_body_bytes=131072, max_output_tokens=4096, request_overhead_tokens=2048,
                 input_usd_per_million='0.10', output_usd_per_million='0.50',
                 pricing_source='https://developers.openai.com/api/docs/models/gpt-6-luna'),
}
PHYSICS = dict(free=dict(composite_limit=256,path_limit=256,stability_limit=256,
                       timeout_seconds=3600,outer_timeout_seconds=3900),
               validation=dict(composite_limit=128,path_limit=128,stability_limit=128,
                               timeout_seconds=600,outer_timeout_seconds=660))


def _number(value, name):
    if isinstance(value,bool) or not isinstance(value,(str,int,float,Decimal)):
        raise ValueError('invalid_finite_'+name)
    try:
        result=Decimal(str(value))
    except (InvalidOperation,ValueError):
        raise ValueError('invalid_finite_'+name) from None
    if not result.is_finite() or result<=0:
        raise ValueError('invalid_finite_'+name)
    return result


def role_reservation(role):
    """UTF-8 byte ceiling plus overhead and output cap; not invoice pricing."""
    if type(role) is not str or role not in ROLES:
        raise ValueError('unknown_iterative_role')
    row=ROLES[role]
    return (Decimal(row['max_body_bytes']+row['request_overhead_tokens'])*Decimal(row['input_usd_per_million'])
            + Decimal(row['max_output_tokens'])*Decimal(row['output_usd_per_million']))/Decimal(1000000)


def _duration(seconds):
    hours, rest=divmod(seconds,3600)
    minutes, seconds=divmod(rest,60)
    return '%02d:%02d:%02d' % (hours,minutes,seconds)


def stage_budget(stage):
    """Return an isolated proposal. ``full`` is summary-only, never a stage."""
    if type(stage) is not str or stage not in set(STAGES)|{'full'}:
        raise ValueError('unknown_iterative_stage')
    rounds=tuple(range(1,6)) if stage=='full' else STAGES[stage]
    episodes=TASK_COUNT*len(rounds)
    roles={}
    for role,settings in ROLES.items():
        row=deepcopy(settings)
        row.update(max_calls=episodes*settings['calls_per_round'],
                   max_usd=str(Decimal(settings['usd_per_round'])*episodes),
                   conservative_reservation_per_call=str(role_reservation(role)),
                   conservative_reservation_total=str(role_reservation(role)*episodes*settings['calls_per_round']))
        roles[role]=row
    validations=episodes*4  # at most two candidate versions, two seeds each
    physical=dict(settings=deepcopy(PHYSICS),free_episodes=episodes,
        max_validation_runs=validations,
        max_composite_attempts=episodes*256+validations*128,
        max_path_checks=episodes*256+validations*128,
        max_stability_checks=episodes*256+validations*128,
        max_worker_seconds=episodes*3600+validations*600,
        max_outer_seconds=episodes*3900+validations*660,
        physics_domain='ASAP_RedMax_part_level_not_robot')
    physical['max_worker_hhmmss']=_duration(physical['max_worker_seconds'])
    physical['max_outer_hhmmss']=_duration(physical['max_outer_seconds'])
    return dict(profile=PROFILE,stage=stage,rounds=list(rounds),products=TASK_COUNT,
        free_episodes=episodes,learning_rounds=episodes,roles=roles,
        max_calls=sum(row['max_calls'] for row in roles.values()),
        max_usd=str(sum((Decimal(row['max_usd']) for row in roles.values()),Decimal(0))),
        conservative_reservation_total=str(sum((Decimal(row['conservative_reservation_total']) for row in roles.values()),Decimal(0))),
        physics=physical,automatic_http_retries=0,automatic_episode_retries=0,
        learn_after_every_round=True,candidate_versions_per_round_max=2,
        validation_seeds=[0,1],initial_lesson_calls_per_round_max=1,
        postvalidation_lesson_calls_per_round_max=1,
        stop_for_review_after_round=2,continuation_is_automatic=False,
        execution_stage=stage!='full',live_enabled=False,authorization_inferred=False,
        billing_cap_guaranteed=False,cache_discounts_assumed=False,
        success_or_registration_guaranteed=False)


def validate_stage_request(stage, max_calls, max_usd):
    """Validate numeric ceilings only; caller must check approval/checkpoint.

    A full 25-episode summary cannot be supplied as an execution stage. The
    continuation stage still requires the runner's explicit reviewed checkpoint
    and live-execution flags; returning this value does not open those gates.
    """
    if type(stage) is not str or stage not in STAGES:
        raise ValueError('iterative_execution_stage_must_be_initial_or_continuation')
    budget=stage_budget(stage)
    if type(max_calls) is not int or max_calls<=0 or max_calls>budget['max_calls']:
        raise ValueError('iterative_call_cap_exceeded_or_invalid')
    usd=_number(max_usd,'usd')
    if usd>Decimal(budget['max_usd']):
        raise ValueError('iterative_cost_cap_exceeded')
    return dict(stage=stage,max_calls=max_calls,max_usd=str(usd),
                authorization_inferred=False,checkpoint_validated=False)


def build_budget_proposal(manifest_path, manifest_sha256):
    if (type(manifest_path) is not str or not manifest_path
            or type(manifest_sha256) is not str or len(manifest_sha256)!=64
            or any(c not in '0123456789abcdef' for c in manifest_sha256)):
        raise ValueError('invalid_pinned_iterative_manifest')
    initial,continuation,total=(stage_budget(stage) for stage in ('initial','continuation','full'))
    return dict(schema_version=1,protocol='sequence_iterative_budget_proposal_v1',
        status='PROPOSED_NOT_AUTHORIZED_NOT_RUN',
        endpoint='https://api.openai.com/v1/responses',
        dataset_manifest=dict(path=manifest_path,sha256=manifest_sha256),
        stages=dict(initial=initial,continuation=continuation),full_summary=total,
        active_proposed_stage='initial',
        continuation_gate='STOP after first 2 rounds per product; separate review, checkpoint and fresh stage authorization required.',
        full_summary_is_not_executable=True,pricing_checked_date='2026-11-10',
        pricing_basis='Existing configured Standard short-context rates, verified against official model pages on 2026-11-10.',
        reservation_basis='UTF-8 body-byte ceiling + 2048 request overhead + output cap; not a billed-token forecast.',
        cost_discounts_assumed=False,actual_invoice_guaranteed=False,
        API_calls_executed=0,physics_calls_executed=0,earlier_budget_residual_reused=False)
