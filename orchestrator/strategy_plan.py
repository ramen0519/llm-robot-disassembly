"""Symbolic disassembly strategies; the host derives guards, never physics results.

The wire contract contains no direction, pose ID or support ID. A strategy only
authorizes a bounded procedure; each proposed action still needs runtime guards
and fresh verification. Compilation does not run registered programs.
"""
from copy import deepcopy
import re

from orchestrator.conditional_plan import PLAN_KEYS, rotation, task_scope
from orchestrator.skill_program import program_hash
from orchestrator.strategy_evidence import capture_plan_evidence, evidence_bound_plan_id
from worker.explicit_action_contract import digest, validate_request
from orchestrator.strategy_failover import (LEGACY, POLICY as FAILOVER_POLICY,
    MAX_ALTERNATIVES_PER_STEP, MAX_ALTERNATIVES_PER_PLAN, validate_policy)

CONTRACT = "strategy_plan_v1"
PROCEDURES = ("axis_search_v1", "two_leg_search_v1", "registered_skill")
POSE_POLICIES = ("keep_orientation_v1", "select_catalog_once_v1")
SUPPORT_POLICIES = ("preserve_add_v1", "replace_once_v1", "bounded_support_search_v1")
STEP_KEYS = {"part_id", "procedure", "registered_skill_id", "pose_policy",
             "support_policy", "max_attempts", "evidence_ids"}
_UNBOUND_REFERENCE = object()


def wire_step(step, strategy_failure_policy=LEGACY):
    """Reconstruct immutable model authorization, excluding host-only guards."""
    validate_policy(strategy_failure_policy)
    result = {key: deepcopy(step[key]) for key in STEP_KEYS}
    if strategy_failure_policy == FAILOVER_POLICY:
        failover = step['stability_failover']
        result['stability_failover'] = (None if failover is None else dict(
            after_failures=failover['after_failures'],
            alternatives=[{key:deepcopy(item[key]) for key in STEP_KEYS}
                          for item in failover['alternatives']]))
    return result


def _horizon(max_steps):
    if type(max_steps) is not int or not 1 <= max_steps <= 64:
        raise ValueError("invalid_strategy_plan_horizon")


def _ids(value, permitted, code):
    if (type(value) is not list or any(type(item) is not str for item in value)
            or len(set(value)) != len(value) or not set(value).issubset(permitted)):
        raise ValueError(code)


def _object(properties):
    return dict(type="object", properties=properties, required=sorted(properties),
                additionalProperties=False)


def schema(request, library_ids, max_steps=64, *, orientation_reference=_UNBOUND_REFERENCE,
           strategy_failure_policy=LEGACY):
    """Strict provider schema; dynamic cross-field guards also run in compile_plan."""
    request = validate_request(request)
    validate_policy(strategy_failure_policy)
    _horizon(max_steps)
    _ids(library_ids, library_ids, "invalid_strategy_skill_ids")
    if len(library_ids) > 32:
        raise ValueError("invalid_strategy_skill_ids")
    def ids(values):
        return dict(type="array", maxItems=len(values),
                    items=dict(type="string", **({"enum": list(values)} if values else {})))
    pose_policies = list(POSE_POLICIES)
    if orientation_reference is not _UNBOUND_REFERENCE:
        from orchestrator.strategy_candidates import pose_policy_context
        availability = pose_policy_context(request, orientation_reference)
        if not availability['planning_ready']:
            raise ValueError('strategy_reference_preflight_unavailable')
        # Strict provider arrays do not use positional tuple schemas. With a
        # multi-step plan the first-step constraint is in the host brief and
        # compiler; future keep-orientation remains legal but conditional.
        if min(max_steps, len(request['remaining_parts']) - 1) == 1:
            pose_policies = availability['allowed_first_step_pose_policies']
    common = dict(part_id=dict(type="string", enum=list(request["candidate_ids"])),
        pose_policy=dict(type="string", enum=pose_policies),
        support_policy=dict(type="string", enum=list(SUPPORT_POLICIES)),
        max_attempts=dict(type="integer", minimum=1, maximum=8),
        evidence_ids=ids(request["evidence_ids"]))
    variants = [_object(dict(common,
        procedure=dict(type="string", enum=list(PROCEDURES[:2])),
        registered_skill_id=dict(type="null")))]
    if library_ids:
        variants.append(_object(dict(common,
            procedure=dict(type="string", enum=["registered_skill"]),
            support_policy=dict(type="string", enum=list(SUPPORT_POLICIES[:2])),
            registered_skill_id=dict(type="string", enum=list(library_ids)))))
    if strategy_failure_policy == FAILOVER_POLICY:
        alternative_variants = deepcopy(variants)
        for item in alternative_variants:
            item['properties']['pose_policy'] = dict(type='string', enum=list(POSE_POLICIES))
        alternative = dict(anyOf=alternative_variants) if len(variants)>1 else alternative_variants[0]
        failover = dict(anyOf=[dict(type='null'), _object(dict(
            after_failures=dict(type='integer', enum=[2,3]),
            alternatives=dict(type='array', minItems=1, maxItems=MAX_ALTERNATIVES_PER_STEP,
                              items=alternative)))])
        variants = [_object(dict(item['properties'], stability_failover=deepcopy(failover)))
                    for item in variants]
    order = ids(request["candidate_ids"])
    order["minItems"] = len(request["candidate_ids"])
    return _object(dict(request_id=dict(type="string", enum=[request["request_id"]]),
        state_id=dict(type="string", enum=[request["state_id"]]), candidate_order=order,
        steps=dict(type="array", minItems=1,
            maxItems=min(max_steps, len(request["remaining_parts"]) - 1),
            items=dict(anyOf=variants) if len(variants) > 1 else variants[0]),
        recovery_skill_ids=ids([])))


def _cards(cards):
    if type(cards) is not list or len(cards) > 32:
        raise ValueError("invalid_strategy_skill_cards")
    result = {}
    for card in cards:
        if type(card) is not dict or type(card.get("skill_id")) is not str:
            raise ValueError("invalid_strategy_skill_cards")
        identity = card["skill_id"]
        if not identity or identity in result:
            raise ValueError("invalid_strategy_skill_cards")
        result[identity] = card
    return result


def _program_pin(card):
    """Bind the frozen content ID even when the planner receives a compact card."""
    pin, program = card.get("program_hash"), card.get("program")
    if (type(pin) is not str or re.fullmatch(r"[0-9a-f]{64}", pin) is None
            or card["skill_id"] != pin or type(program) is not dict):
        raise ValueError("invalid_strategy_program_binding")
    if program.get("source_omitted_for_planning") is True:
        if (set(program) != {"language", "source_omitted_for_planning", "program_hash"}
                or program.get("language") not in ("recovery_rules_v1", "python_policy_v1")
                or program.get("program_hash") != pin):
            raise ValueError("invalid_strategy_program_binding")
    else:
        try:
            actual = program_hash(program)
        except (ValueError, TypeError, KeyError):
            raise ValueError("invalid_strategy_program_binding") from None
        if actual != pin:
            raise ValueError("invalid_strategy_program_binding")
    return pin


def compile_plan(value, request, cards, max_steps=64, diagnostics=None, *,
                 orientation_reference=_UNBOUND_REFERENCE, strategy_failure_policy=LEGACY):
    """Validate the symbolic choices and attach host-owned conditional guards.

    Registered-program mover/applicability checks belong after VM evaluation on
    the current runtime request. A card's earlier proposal cannot certify a
    future step, and this compiler neither executes nor rewrites that proposal.
    """
    request = validate_request(request)
    validate_policy(strategy_failure_policy)
    _horizon(max_steps)
    library = _cards(cards)
    info = diagnostics if diagnostics is not None else {}
    info.update(contract=CONTRACT, phase="identity", step_index=None,
                raw_response_recorded=False, wire_value_sha256=digest(value),
                expected_remaining_source="current_state_minus_preceding_proposed_movers")
    if (type(value) is not dict or set(value) != PLAN_KEYS
            or value["request_id"] != request["request_id"]
            or value["state_id"] != request["state_id"]):
        raise ValueError("stale_or_invalid_strategy_plan_identity")
    _ids(value["candidate_order"], request["candidate_ids"], "invalid_strategy_candidate_order")
    if set(value["candidate_order"]) != set(request["candidate_ids"]):
        raise ValueError("incomplete_strategy_candidate_order")
    if type(value["recovery_skill_ids"]) is not list or value["recovery_skill_ids"]:
        raise ValueError("strategy_recovery_requires_new_step")
    info["phase"] = "length"
    steps = value["steps"]
    if (type(steps) is not list
            or not 1 <= len(steps) <= min(max_steps, len(request["remaining_parts"]) - 1)):
        raise ValueError("invalid_strategy_plan_length")
    remaining, compiled, pins = set(request["remaining_parts"]), [], {}
    pose_context = None
    from orchestrator.strategy_candidates import support_policy_context
    support_reference = None if orientation_reference is _UNBOUND_REFERENCE else orientation_reference
    support_context = support_policy_context(request, support_reference)
    info['support_preflight'] = support_context
    if not support_context['planning_ready']:
        raise ValueError('strategy_reference_preflight_unavailable')
    if orientation_reference is not _UNBOUND_REFERENCE:
        from orchestrator.strategy_candidates import pose_policy_context
        pose_context = pose_policy_context(request, orientation_reference)
        info['pose_preflight'] = pose_context
        if not pose_context['planning_ready']:
            raise ValueError('strategy_reference_preflight_unavailable')
    failover_count = 0
    def compile_step(step, index, allowed_parts, *, alternative=False):
        nonlocal failover_count
        keys = STEP_KEYS | ({'stability_failover'}
            if strategy_failure_policy == FAILOVER_POLICY and not alternative else set())
        if type(step) is not dict or set(step) != keys:
            raise ValueError("invalid_strategy_step_fields")
        part = step['part_id']
        if type(part) is not str or part not in allowed_parts:
            raise ValueError('invalid_or_repeated_strategy_mover')
        if step['procedure'] not in PROCEDURES:
            raise ValueError('invalid_strategy_procedure')
        if step['pose_policy'] not in POSE_POLICIES:
            raise ValueError('invalid_strategy_pose_policy')
        if (index == 0 and not alternative and pose_context is not None
                and step['pose_policy'] not in pose_context['allowed_first_step_pose_policies']):
            raise ValueError('strategy_first_step_pose_policy_unavailable')
        if step['support_policy'] not in SUPPORT_POLICIES:
            raise ValueError('invalid_strategy_support_policy')
        if type(step['max_attempts']) is not int or not 1 <= step['max_attempts'] <= 8:
            raise ValueError('invalid_strategy_attempt_limit')
        _ids(step['evidence_ids'], request['evidence_ids'], 'invalid_strategy_evidence_ids')
        identity = step['registered_skill_id']
        if step['procedure'] == 'registered_skill':
            if step['support_policy'] == 'bounded_support_search_v1':
                raise ValueError('bounded_support_search_requires_fixed_procedure')
            if type(identity) is not str or identity not in library:
                raise ValueError('unknown_strategy_registered_skill')
            pins[identity] = _program_pin(library[identity])
        elif identity is not None:
            raise ValueError('unexpected_strategy_registered_skill')
        if index == 0 and not alternative:
            first_support = support_policy_context(request, support_reference, step=step)
            info['support_preflight'] = first_support
            if not first_support['support_policy_available']:
                info['phase'] = 'support_preflight'
                raise ValueError('strategy_first_step_support_conflict')
        item = dict(deepcopy(step), mode='strategy', expected_remaining=sorted(remaining))
        failover = step.get('stability_failover')
        if failover is not None:
            if (type(failover) is not dict or set(failover) != {'after_failures','alternatives'}
                    or type(failover['after_failures']) is not int or failover['after_failures'] not in (2,3)
                    or type(failover['alternatives']) is not list
                    or not 1 <= len(failover['alternatives']) <= MAX_ALTERNATIVES_PER_STEP):
                raise ValueError('invalid_strategy_stability_failover')
            if failover['after_failures'] > step['max_attempts']:
                raise ValueError('strategy_failover_threshold_exceeds_attempt_limit')
            seen = {part}
            alternatives = []
            for other in failover['alternatives']:
                compiled_other = compile_step(other, index, allowed_parts - seen, alternative=True)
                alternatives.append(compiled_other)
                seen.add(compiled_other['part_id'])
            failover_count += len(alternatives)
            if failover_count > MAX_ALTERNATIVES_PER_PLAN:
                raise ValueError('too_many_strategy_failover_alternatives')
            item['stability_failover'] = dict(after_failures=failover['after_failures'],
                                             alternatives=alternatives)
        return item
    for index, step in enumerate(steps):
        info.update(phase="step", step_index=index, expected_remaining=sorted(remaining))
        compiled.append(compile_step(step, index, remaining & set(request['candidate_ids']) - {request['base_part']}))
        remaining.remove(step['part_id'])
    all_steps = compiled + [other for step in compiled if step.get('stability_failover')
                           for other in step['stability_failover']['alternatives']]
    evidence = capture_plan_evidence(request, sorted({identity for step in all_steps for identity in step['evidence_ids']}))
    bound = deepcopy(value)
    bound.update(steps=compiled, plan_contract=CONTRACT,
        plan_id=evidence_bound_plan_id(value, pins, evidence), evidence_binding=evidence,
        task_scope=task_scope(request),
        source_poses={p["pose_id"]: rotation(p) for p in request["pose_catalog"]},
        source_geometry_parts=deepcopy(request["geometry"].get("parts", {})),
        first_request_sha256=digest(request),
        pose_rule="per_step_strategy_policy_current_catalog",
        registered_program_hashes=pins, physics_success_implied=False)
    if strategy_failure_policy != LEGACY:
        bound['strategy_failure_policy'] = strategy_failure_policy
    info.update(phase="compiled", step_count=len(compiled), plan_id=bound["plan_id"])
    return bound
