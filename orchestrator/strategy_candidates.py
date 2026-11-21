"""Bounded low-level candidates for one LLM-selected part and strategy.

No API, physics, file access, state transition, feasibility inference or part
ranking. Local exhaustion means only this bounded strategy found no next action.
"""
from copy import deepcopy
from itertools import combinations
import math

from worker.explicit_action_contract import (
    DIRECTIONS, action_id, digest, validate_choice, validate_matrix, validate_request,
)
from worker.path_probe_round import partition_path_round, semantic_path_key

MAX_ITERATIONS = 4096
PROCEDURES = ("axis_search_v1", "two_leg_search_v1", "registered_skill")
POSE_POLICIES = ("keep_orientation_v1", "select_catalog_once_v1")
SUPPORT_POLICIES = ("preserve_add_v1", "replace_once_v1", "bounded_support_search_v1")


def _inputs(request, step, max_iterations):
    current = validate_request(request)
    if type(max_iterations) is not int or not 1 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError("invalid_strategy_iteration_limit")
    if (type(step) is not dict or step.get("part_id") not in current["candidate_ids"]
            or step.get("procedure") not in PROCEDURES
            or step.get("pose_policy") not in POSE_POLICIES
            or step.get("support_policy") not in SUPPORT_POLICIES
            or type(step.get("max_attempts")) is not int or not 1 <= step["max_attempts"] <= 8
            or type(step.get("evidence_ids")) is not list
            or any(type(x) is not str for x in step["evidence_ids"])
            or len(set(step["evidence_ids"])) != len(step["evidence_ids"])
            or not set(step["evidence_ids"]).issubset(current["evidence_ids"])):
        raise ValueError("invalid_strategy_step_or_evidence")
    if "attempted_actions" not in current or "path_stage_observations" not in current:
        raise ValueError("strategy_requires_readable_attempt_history")
    if (step["support_policy"] == "bounded_support_search_v1"
            and step["procedure"] == "registered_skill"):
        raise ValueError("bounded_support_search_requires_fixed_procedure")
    return current


def _scope(request):
    value = {k: request[k] for k in
        ("run_id", "state_id", "task_context", "remaining_parts", "base_part", "max_grippers")}
    if 'virtual_search_restore' in request:
        value['node_visit_revision'] = request['virtual_search_restore']['node_visit_revision']
    if 'accepted_pose_carry' in request:
        value['pose_carry_activation_revision'] = request['accepted_pose_carry']['activation_revision']
    return digest(value)


def _diagnostics(limit):
    return dict(max_iterations=limit, iterations=0, physics_calls=0, api_calls=0,
        candidate_pruning=False, hard_pruning=False, physical_success_known=False,
        geometry_ordering_is_unverified=True, local_exhaustion_is_impossibility=False,
        supports_are_action_scoped_not_live_robot_hold_state=True,
        support_replacement_authorized=False, support_replacement_authority=None,
        support_replacement_source_evidence_id=None, support_replacement_source_observation=None,
        prior_support_ids=None, proposed_support_ids=None, released_support_ids=[])


def _observations(request):
    """Only current-state, actually bound, timely failure facts affect ordering."""
    observed = []
    poses = {p["pose_id"]: p["matrix"] for p in request["pose_catalog"]}
    for attempt, stage in zip(request["attempted_actions"], request["path_stage_observations"]):
        matches = [f for f in request["episode_feedback"]
                   if f.get("action_id") == attempt["action_id"]]
        if len(matches) != 1:
            continue
        row = matches[0]
        if (stage["backend"] != "ASAP_RedMax" or stage["deadline_exceeded"]
                or stage["status"] not in ("UNKNOWN", "REJECTED")
                or row.get("status") != stage["status"]
                or row.get("accepted_for_transition") is not False
                or row.get("deadline_exceeded") is not False
                or row.get("evidence_id") != stage["source_event_id"]
                or row.get("path_success") is not stage["path_success"]
                or type(row.get("remaining_parts")) is not list
                or any(type(p) is not str for p in row["remaining_parts"])
                or sorted(row["remaining_parts"]) != request["remaining_parts"]):
            continue
        choice = deepcopy(row.get("choice"))
        if type(choice) is not dict:
            continue
        choice["evidence_ids"] = []
        try:
            validate_choice(request, choice)
            expected = {k: v for k, v in attempt.items() if k != "action_id"}
            expected["evidence_ids"] = []
            if (choice != expected or action_id(request, choice) != attempt["action_id"]
                    or row.get("selected_pose_matrix") != poses[choice["pose_id"]]):
                continue
        except (ValueError, KeyError, TypeError):
            continue
        if (row.get("path_success") is False and row.get("stability_success") is None
                or row.get("path_success") is True and row.get("stability_success") is False):
            observed.append(dict(choice=choice, feedback=row, pose_matrix=poses[choice["pose_id"]]))
    return observed


def _support_reference(request, reference):
    if 'accepted_pose_carry' in request:
        from worker.accepted_pose_carry import entry_reference
        entry = entry_reference(request)
        if entry is not None:
            if reference is not None and reference != entry:
                return None, None, 'accepted_entry_reference_mismatch'
            # A supported parent removal is a new virtual child visit, not
            # persistent robot hold state. Old visits' command history remains
            # for budgets/novelty but cannot supply current supports here.
            return [], 'accepted_pose_carry_new_visit', None
    from worker.virtual_search_restore import restored_reference
    restored = restored_reference(request)
    if restored is not None:
        if reference is not None and reference != restored:
            return None, None, "restored_support_reference_mismatch"
        return list(restored['choice']['support_ids']), "worker_restored_pre_action", None
    """Known command supports survive an uncertain/late physical verdict.

    Support policy is distinct from failure classification. A missing binding
    is a stop, never permission to reinterpret existing supports as empty.
    """
    if reference is not None and type(reference) is not dict:
        return None, None, "support_reference_unavailable"
    if reference is not None and reference.get("scope") == "orientation_only_after_transition":
        remaining = reference.get("remaining_parts")
        if (request["attempted_actions"] or type(remaining) is not list
                or set(remaining) == set(request["remaining_parts"])):
            return None, None, "orientation_only_reference_transition_unbound"
        return [], "orientation_only_after_transition_initial_empty", None
    if not request["attempted_actions"]:
        if reference is not None:
            return None, None, "support_reference_unavailable"
        return [], "explicit_initial_empty", None
    attempt, stage = request["attempted_actions"][-1], request["path_stage_observations"][-1]
    rows = [f for f in request["episode_feedback"] if f.get("action_id") == attempt["action_id"]]
    if len(rows) != 1:
        return None, None, "support_reference_unavailable"
    row = rows[0]
    if (stage["backend"] != "ASAP_RedMax" or row.get("evidence_id") != stage["source_event_id"]
            or type(row.get("remaining_parts")) is not list
            or any(type(p) is not str for p in row["remaining_parts"])
            or sorted(row["remaining_parts"]) != request["remaining_parts"]):
        return None, None, "support_reference_unavailable"
    if row.get("accepted_for_transition") is True:
        return None, None, "reference_transition_not_reflected"
    selected = deepcopy(row.get("choice"))
    if type(selected) is not dict:
        return None, None, "support_reference_unavailable"
    selected["evidence_ids"] = []
    try:
        validate_choice(request, selected)
        expected = {k:v for k,v in attempt.items() if k != "action_id"}
        expected["evidence_ids"] = []
        pose = next(p for p in request["pose_catalog"] if p["pose_id"] == selected["pose_id"])
        if (selected != expected or action_id(request, selected) != attempt["action_id"]
                or row.get("selected_pose_matrix") != pose["matrix"]):
            return None, None, "support_reference_unavailable"
        if reference is not None:
            supplied = deepcopy(reference.get("choice"))
            if type(supplied) is not dict:
                return None, None, "support_reference_unavailable"
            supplied["evidence_ids"] = []
            if (supplied != selected or reference.get("evidence_id") != row["evidence_id"]
                    or sorted(reference.get("remaining_parts", [])) != request["remaining_parts"]):
                return None, None, "support_reference_not_latest_command"
    except (ValueError, TypeError, KeyError, StopIteration):
        return None, None, "support_reference_unavailable"
    return list(selected["support_ids"]), "latest_bound_command", None


def _reference_rotation(reference, request):
    if type(reference) is not dict:
        return None
    matrix = reference.get("selected_pose_matrix", reference.get("pose_matrix"))
    if matrix is None and reference.get("rotation") is not None:
        rotation = reference["rotation"]
        if type(rotation) is not list or len(rotation) != 3 or any(
                type(row) is not list or len(row) != 3 for row in rotation):
            return None
        matrix = [list(row) + [0] for row in rotation] + [[0, 0, 0, 1]]
    if matrix is None:
        identity = (reference.get("choice") or {}).get("pose_id")
        matches = [p["matrix"] for p in request["pose_catalog"] if p["pose_id"] == identity]
        matrix = matches[0] if len(matches) == 1 else None
    try:
        validate_matrix(matrix)
    except (ValueError, TypeError, KeyError):
        return None
    return [row[:3] for row in matrix[:3]]


def resolve_reference(request, orientation_reference=None):
    """Reuse the executor's known command/DFS context, never infer a new pose.

    A prior accepted transition is supplied by the service which observed it.
    This function does not search unrelated historical PASS records for an anchor.
    """
    if orientation_reference is not None and type(orientation_reference) is not dict:
        raise ValueError('invalid_strategy_orientation_reference')
    if 'accepted_pose_carry' in request:
        from worker.accepted_pose_carry import entry_reference
        entry = entry_reference(request)
        if entry is not None:
            return entry
    from worker.virtual_search_restore import restored_reference
    restored = restored_reference(request)
    if restored is not None:
        return restored
    ref = deepcopy(orientation_reference)
    if ref is None and request['attempted_action_ids']:
        last = request['attempted_action_ids'][-1]
        matches = [row for row in request['episode_feedback'] if row.get('action_id') == last
                   and set(row.get('remaining_parts', [])) == set(request['remaining_parts'])]
        stages = [row for row in request['path_stage_observations'] if row['action_id'] == last]
        if len(matches) != 1 or len(stages) != 1:
            return dict(reference_error='current_support_reference_unavailable')
        ref = deepcopy(matches[0])
        try:
            validate_choice(request, ref['choice'])
            if (action_id(request, ref['choice']) != last
                    or stages[0]['source_event_id'] != ref['evidence_id']
                    or ref['selected_pose_matrix'] != next(p['matrix'] for p in request['pose_catalog']
                        if p['pose_id'] == ref['choice']['pose_id'])):
                raise ValueError('current_support_reference_mismatch')
        except (ValueError, KeyError, TypeError, StopIteration):
            return dict(reference_error='current_support_reference_mismatch')
    if ref and set(ref.get('remaining_parts', [])) != set(request['remaining_parts']):
        if 'reference_error' not in ref:
            ref['scope'] = 'orientation_only_after_transition'
    return ref


def pose_policy_context(request, orientation_reference=None):
    """Host facts for first-step legality, not a physics result or pose choice.

    Future catalogs are unknown. The first step can be checked now; subsequent
    keep-orientation steps remain conditional and are checked again at runtime.
    """
    current = validate_request(request)
    reference = resolve_reference(current, orientation_reference)
    _, _, support_error = _support_reference(current, reference)
    if reference and reference.get('reference_error'):
        support_error = reference['reference_error']
    rotation = _reference_rotation(reference, current)
    matches = _matching_reference_poses(current, reference)
    reason = ('orientation_reference_missing' if rotation is None else
              'orientation_not_in_current_catalog' if not matches else
              'orientation_match_ambiguous' if len(matches) != 1 else None)
    allowed = ['select_catalog_once_v1'] if support_error is None else []
    if reason is None and support_error is None:
        allowed.insert(0, 'keep_orientation_v1')
    return dict(version='strategy_pose_preflight_v1', request_id=current['request_id'],
        state_id=current['state_id'], request_sha256=digest(current),
        planning_ready=support_error is None, preflight_error=support_error,
        allowed_first_step_pose_policies=allowed,
        keep_orientation_available=reason is None and support_error is None,
        keep_orientation_unavailable_reason=reason or support_error,
        matching_pose_count=len(matches), matching_pose_ids=matches,
        reference_scope=reference.get('scope', 'latest_bound_command') if reference else 'missing',
        future_steps_require_current_catalog_recheck=True,
        selection_is_new_model_permission=True, automatic_pose_fallback=False,
        physics_success_implied=False, robot_motion_authorized=False)


def support_policy_context(request, orientation_reference=None, *, step=None):
    """Expose current first-step support permissions without choosing an action.

    These are command-state guards, not measured robot holds or feasibility
    findings. Future steps and failure alternatives may have different supports
    and remain conditional on a fresh runtime check.
    """
    current = validate_request(request)
    reference = resolve_reference(current, orientation_reference)
    previous, source, error = _support_reference(current, reference)
    if reference and reference.get('reference_error'):
        error = reference['reference_error']
    ready = error is None
    supports = sorted(previous) if ready else None
    allowed = {
        part: ([policy for policy in SUPPORT_POLICIES
                if policy == 'replace_once_v1' or part not in supports] if ready else [])
        for part in sorted(current['candidate_ids'])
    }
    result = dict(version='strategy_support_preflight_v1',
        request_id=current['request_id'], state_id=current['state_id'],
        request_sha256=digest(current), planning_ready=ready, preflight_error=error,
        current_support_ids=supports, reference_source=source if ready else None,
        reference_evidence_id=reference.get('evidence_id') if ready and reference else None,
        allowed_first_step_support_policies_by_part=allowed,
        first_proposal_release_requires='replace_once_v1',
        bounded_search_first_proposal_preserves_current_supports=True,
        bounded_replacement_requires_same_mover_pose_timely_stability_failure=True,
        future_steps_and_alternatives_require_runtime_recheck=True,
        candidate_pruning=False, selection_forced=False, automatic_support_release=False,
        physics_calls=0, physics_success_implied=False, robot_motion_authorized=False)
    if step is not None:
        if (type(step) is not dict or step.get('part_id') not in current['candidate_ids']
                or step.get('support_policy') not in SUPPORT_POLICIES
                or step.get('procedure') not in PROCEDURES):
            raise ValueError('invalid_strategy_support_preflight_step')
        part, policy = step['part_id'], step['support_policy']
        policies = list(allowed[part])
        if step['procedure'] == 'registered_skill':
            policies = [value for value in policies if value != 'bounded_support_search_v1']
        conflict = ([part] if ready and part in supports and policy != 'replace_once_v1' else [])
        valid = ready and policy in policies
        result.update(selected_part_id=part, selected_procedure=step['procedure'], selected_support_policy=policy,
            allowed_support_policies=policies, conflicting_support_ids=conflict,
            support_policy_available=valid,
            reason=(error if not ready else 'preserved_support_conflicts_with_selected_part'
                    if conflict else 'support_policy_not_available_for_procedure'
                    if not valid else 'support_policy_available'))
    return result


def _matching_reference_poses(request, reference):
    """An opted-in worker carry must resolve the full bound transform exactly.

    A geometrically generated pose with the same rotation but a different
    translation is not the carried current transform. Legacy matching is kept.
    """
    if 'accepted_pose_carry' in request:
        from worker.accepted_pose_carry import resolve_pose
        identity = resolve_pose(request, reference)
        return [identity] if identity is not None else []
    rotation = _reference_rotation(reference, request)
    return [] if rotation is None else [p['pose_id'] for p in request['pose_catalog'] if all(
        abs(p['matrix'][i][j] - rotation[i][j]) <= 1e-6 for i in range(3) for j in range(3))]


def _center(features):
    if type(features) is not dict:
        return None
    value = features.get("bounds_center", features.get("center_mass"))
    if (type(value) is list and len(value) == 3 and
            all(type(v) in (int, float) and math.isfinite(v) for v in value)):
        return value
    return None


def _axes(request, part, matrix):
    parts = request.get("geometry", {}).get("parts", {})
    center = _center(parts.get(part))
    other = [_center(parts.get(p)) for p in request["remaining_parts"] if p != part]
    other = [p for p in other if p is not None]
    if center is None or not other:
        return list(DIRECTIONS), "registered_axis_order_missing_geometry"
    outward = [center[i] - sum(p[i] / len(other) for p in other) for i in range(3)]
    world = [sum(matrix[i][j] * outward[j] for j in range(3)) for i in range(3)]
    if not all(math.isfinite(v) for v in world):
        return list(DIRECTIONS), "registered_axis_order_nonfinite_geometry"
    return sorted(DIRECTIONS, key=lambda d: (-sum(world[i] * DIRECTIONS[d][i]
        for i in range(3)), list(DIRECTIONS).index(d))), "centroid_outward_world_axis_order_unverified"


def _directions(procedure, axes):
    for first in axes:
        if procedure == "axis_search_v1":
            yield first, None
        else:
            for second in axes:
                if sum(a*b for a, b in zip(DIRECTIONS[first], DIRECTIONS[second])) == 0:
                    yield first, second


def _reported_parts(observation, request, part):
    if observation is None or observation["feedback"].get("stability_success") is not False:
        return []
    raw = observation["feedback"].get("raw_stability_parts")
    available = set(request["remaining_parts"]) - {part, request["base_part"]}
    # An upstream representative is a preference only, never a complete blocker
    # or instability set. No LLM explanation fills in missing measurements.
    if (type(raw) is not list or any(type(p) is not str for p in raw)
            or len(set(raw)) != len(raw)):
        return []
    return sorted(set(raw) & available)


def _support_sets(available, preserved, capacity, reported):
    preserved = set(preserved)
    seen = set()
    preferred = [preserved | set(reported)] if reported else []
    preferred += [preserved | {part} for part in reported]
    preferred.append(preserved)
    for support in preferred:
        key = tuple(sorted(support))
        if support.issubset(available) and len(support) <= capacity and key not in seen:
            seen.add(key)
            yield list(key)
    extras = sorted(set(available) - preserved)
    for size in range(max(0, capacity-len(preserved)) + 1):
        for extra in combinations(extras, size):
            key = tuple(sorted(preserved | set(extra)))
            if key not in seen:
                seen.add(key)
                yield list(key)


def _same_configuration(observed, part, matrix, supports):
    return (observed["choice"]["part_id"] == part and observed["pose_matrix"] == matrix
            and set(observed["choice"]["support_ids"]) == set(supports)
            and observed["feedback"].get("path_success") is True
            and observed["feedback"].get("stability_success") is False)


def _support_options(available, previous, capacity, reported, replacement_allowed):
    """Preserve/add first; opt-in replacements remain a finite candidate search.

    Reported bodies only order candidates. At full capacity keep as many prior
    supports as possible while adding a reported body; no support is diagnosed
    as unnecessary and none of these alternatives is assumed to pass physics.
    """
    yield from _support_sets(available, previous, capacity, reported)
    if not replacement_allowed:
        return
    previous = set(previous)
    seen = set()
    for part in reported:
        keep = sorted(previous - {part})
        for size in range(min(len(keep), capacity - 1), -1, -1):
            for retained in combinations(keep, size):
                key = tuple(sorted(set(retained) | {part}))
                if not previous.issubset(key) and key not in seen:
                    seen.add(key)
                    yield list(key)
    for support in _support_sets(available, [], capacity, reported):
        key = tuple(support)
        if not previous.issubset(support) and key not in seen:
            seen.add(key)
            yield support


def _replacement_diagnostics(step, relevant, previous, proposed=None):
    allowed = (step["support_policy"] == "bounded_support_search_v1"
               and relevant is not None
               and relevant["feedback"].get("path_success") is True
               and relevant["feedback"].get("stability_success") is False)
    row = relevant["feedback"] if allowed else None
    return dict(support_policy=step["support_policy"],
        support_replacement_authorized=allowed,
        support_replacement_authority=("model_selected_bounded_support_search_after_bound_stability_failure"
                                       if allowed else None),
        support_replacement_source_evidence_id=row["evidence_id"] if row else None,
        support_replacement_source_observation=(dict(
            evidence_id=row["evidence_id"], action_id=row["action_id"], backend="ASAP_RedMax",
            status=row["status"], path_success=True, stability_success=False,
            accepted_for_transition=False, deadline_exceeded=False) if row else None),
        prior_support_ids=sorted(previous),
        proposed_support_ids=sorted(proposed) if proposed is not None else None,
        released_support_ids=sorted(set(previous)-set(proposed)) if proposed is not None else [])


def catalog_pose_choice(request, part_id):
    """Preview the existing select-once rule; this is not physics or permission.

    The caller supplies a validated current request. Sharing the implementation
    with the planning summary prevents its preview from promising a different
    orientation while the runtime silently selects the original one.
    """
    counts = {p["pose_id"]: sum(a["part_id"] == part_id and
        a["pose_id"] == p["pose_id"] for a in request["attempted_actions"])
        for p in request["pose_catalog"]}
    return min(request["pose_catalog"], key=lambda p: (counts[p["pose_id"]],
        sum((p["matrix"][i][j] - int(i == j))**2 for i in range(3) for j in range(3)), p["pose_id"]))


def build_anchor(request, step, reference=None, *, max_iterations=MAX_ITERATIONS):
    current = _inputs(request, step, max_iterations)
    diagnostics = _diagnostics(max_iterations)
    diagnostics["support_policy"] = step["support_policy"]
    if step["procedure"] == "registered_skill":
        return dict(anchor=None, reason="registered_skill_requires_runtime", diagnostics=diagnostics)
    observations = _observations(current)
    if step["pose_policy"] == "keep_orientation_v1":
        rotation = _reference_rotation(reference, current)
        if rotation is None:
            return dict(anchor=None, reason="orientation_reference_missing", diagnostics=diagnostics)
        identities = _matching_reference_poses(current, reference)
        matches = [p for p in current['pose_catalog'] if p['pose_id'] in identities]
        if len(matches) != 1:
            return dict(anchor=None, reason="orientation_unavailable_or_ambiguous", diagnostics=diagnostics)
        pose = matches[0]
    else:
        pose = catalog_pose_choice(current, step["part_id"])
        diagnostics["pose_selection"] = "fewest_selected_part_attempts_then_identity_rotation_distance"
    latest = observations[-1] if observations else None
    if latest and latest["feedback"]["action_id"] != current["attempted_action_ids"][-1]:
        latest = None
    previous, support_source, support_error = _support_reference(current, reference)
    if support_error:
        return dict(anchor=None, reason=support_error, diagnostics=diagnostics)
    preserved = [] if step["support_policy"] == "replace_once_v1" else previous
    available = set(current["remaining_parts"]) - {step["part_id"], current["base_part"]}
    if not set(preserved).issubset(available):
        diagnostics['support_preflight'] = support_policy_context(current, reference, step=step)
        return dict(anchor=None, reason="preserved_support_conflicts_with_selected_part", diagnostics=diagnostics)
    relevant = (latest if latest and latest["choice"]["part_id"] == step["part_id"]
                and latest["pose_matrix"] == pose["matrix"] else None)
    reported = _reported_parts(relevant, current, step["part_id"])
    diagnostics.update(_replacement_diagnostics(step, relevant, previous))
    supports = None
    for support in _support_options(available, preserved, current["max_grippers"]-1, reported,
                                   diagnostics["support_replacement_authorized"]):
        if diagnostics["iterations"] >= max_iterations:
            return dict(anchor=None, reason="strategy_iteration_limit", diagnostics=diagnostics)
        diagnostics["iterations"] += 1
        if any(_same_configuration(o, step["part_id"], pose["matrix"], support) for o in observations):
            continue
        supports = support
        break
    if supports is None:
        return dict(anchor=None, reason="local_support_options_exhausted", diagnostics=diagnostics)
    axes, ordering = _axes(current, step["part_id"], pose["matrix"])
    diagnostics.update(axis_ordering=ordering, reported_parts_used=reported,
        support_seed=support_source,
        replaced_supports_at_anchor=sorted(set(previous)-set(supports)))
    diagnostics.update(_replacement_diagnostics(step, relevant, previous, supports))
    anchor = dict(version="strategy_anchor_v1", scope_sha256=_scope(current), step_sha256=digest(step),
        part_id=step["part_id"], pose_id=pose["pose_id"], pose_matrix=deepcopy(pose["matrix"]),
        support_ids=supports, axes=axes, initial_attempt_ids=list(current["attempted_action_ids"]),
        max_iterations=max_iterations, diagnostics=deepcopy(diagnostics))
    anchor["anchor_sha256"] = digest(anchor)
    return dict(anchor=anchor, reason="anchored", diagnostics=diagnostics)


def propose(request, step, anchor, reference=None, *, max_iterations=MAX_ITERATIONS):
    current = _inputs(request, step, max_iterations)
    diagnostics = _diagnostics(max_iterations)
    diagnostics["support_policy"] = step["support_policy"]
    if step["procedure"] == "registered_skill":
        return dict(choice=None, reason="registered_skill_requires_runtime", diagnostics=diagnostics)
    if (type(anchor) is not dict or anchor.get("version") != "strategy_anchor_v1"
            or anchor.get("anchor_sha256") != digest({k:v for k,v in anchor.items() if k != "anchor_sha256"})
            or anchor.get("scope_sha256") != _scope(current)
            or anchor.get("step_sha256") != digest(step) or anchor.get("max_iterations") != max_iterations):
        return dict(choice=None, reason="strategy_anchor_scope_mismatch", diagnostics=diagnostics)
    attempted = set(current["attempted_action_ids"])
    initial = set(anchor["initial_attempt_ids"])
    if not initial.issubset(attempted):
        return dict(choice=None, reason="strategy_attempt_history_changed", diagnostics=diagnostics)
    diagnostics["attempts_since_anchor"] = len(attempted-initial)
    if len(attempted-initial) >= step["max_attempts"]:
        return dict(choice=None, reason="strategy_attempt_budget_exhausted", diagnostics=diagnostics)
    if not any(p["pose_id"] == anchor["pose_id"] and p["matrix"] == anchor["pose_matrix"]
               for p in current["pose_catalog"]):
        return dict(choice=None, reason="anchored_pose_unavailable", diagnostics=diagnostics)
    observations = _observations(current)
    latest = observations[-1] if observations else None
    if latest and latest["feedback"]["action_id"] != current["attempted_action_ids"][-1]:
        latest = None
    if attempted-initial and latest is None:
        return dict(choice=None, reason="current_failure_observation_unavailable", diagnostics=diagnostics)
    relevant = (latest if latest and latest["choice"]["part_id"] == step["part_id"]
                and latest["pose_matrix"] == anchor["pose_matrix"] else None)
    # Legacy replace_once permission ends at the anchor. Repeated replacement
    # needs the distinct model-selected policy and the latest bound failure.
    previous = (relevant["choice"]["support_ids"] if relevant and attempted-initial
                else anchor["support_ids"])
    stability_failure = relevant and relevant["feedback"].get("stability_success") is False
    reported = _reported_parts(relevant, current, step["part_id"]) if stability_failure else []
    available = set(current["remaining_parts"]) - {step["part_id"], current["base_part"]}
    commanded = (relevant["choice"]["support_ids"] if relevant else
                 anchor["diagnostics"].get("prior_support_ids", previous))
    diagnostics.update(_replacement_diagnostics(step, relevant, commanded))
    if step["support_policy"] == "bounded_support_search_v1":
        commanded, _, support_error = _support_reference(current, reference)
        if support_error:
            return dict(choice=None, reason=support_error, diagnostics=diagnostics)
        diagnostics.update(_replacement_diagnostics(step, relevant, commanded))
        if (not diagnostics["support_replacement_authorized"]
                and not set(commanded).issubset(previous)):
            return dict(choice=None, reason="support_replacement_observation_unavailable", diagnostics=diagnostics)
    support_sets = (_support_options(available, previous, current["max_grippers"]-1, reported,
                                    diagnostics["support_replacement_authorized"])
                    if stability_failure else iter([previous]))
    deferred = set(partition_path_round(current)["deferred_path_keys"])
    directions = list(_directions(step["procedure"], anchor["axes"]))
    if relevant and relevant["feedback"].get("path_success") is True:
        known = (relevant["choice"]["direction_id"], relevant["choice"]["second_direction_id"])
        if known in directions:
            directions.remove(known)
            directions.insert(0, known)
    diagnostics.update(reported_parts_used=reported, duplicate_skips=0, path_round_skips=0,
        stability_configuration_skips=0, selected_part_id=step["part_id"])
    for supports in support_sets:
        if any(_same_configuration(o, step["part_id"], anchor["pose_matrix"], supports) for o in observations):
            diagnostics["stability_configuration_skips"] += 1
            diagnostics["iterations"] += 1
            if diagnostics["iterations"] >= max_iterations:
                return dict(choice=None, reason="strategy_iteration_limit", diagnostics=diagnostics)
            continue
        for first, second in directions:
            if diagnostics["iterations"] >= max_iterations:
                return dict(choice=None, reason="strategy_iteration_limit", diagnostics=diagnostics)
            diagnostics["iterations"] += 1
            candidate = dict(part_id=step["part_id"], pose_id=anchor["pose_id"],
                skill_id="single_force_v1" if step["procedure"] == "axis_search_v1" else "turn_force_two_leg_v1",
                direction_id=first, second_direction_id=second,
                support_ids=list(supports), evidence_ids=list(step["evidence_ids"]))
            validate_choice(current, candidate)
            if action_id(current, candidate) in attempted:
                diagnostics["duplicate_skips"] += 1
                continue
            if semantic_path_key(current, candidate) in deferred:
                diagnostics["path_round_skips"] += 1
                continue
            diagnostics.update(_replacement_diagnostics(step, relevant, commanded, supports))
            return dict(choice=candidate, reason="candidate", diagnostics=diagnostics)
    return dict(choice=None, reason="local_strategy_options_exhausted", diagnostics=diagnostics)
