"""Opt-in endpoint semantics checks. Synthetic diagnostics are never evidence."""
from copy import deepcopy
import math

from orchestrator.python_direction_probe import TraceVM, action_sequence, sequence
from orchestrator.python_skill_vm import parse
from orchestrator.python_skill_program import build_context
from orchestrator.lesson_runtime import extension_for
from worker.task_memory import digest

HELPER = 'unseparated_parts'
SCOPE = 'selected_trajectory_terminal_state_only'
MAX_PROBES = 32


def unmet(context, part_id, pose_id):
    """Reference interpretation of measured clearance, NOT a path predictor.

    build_context has already scoped last to this task/state/configuration.
    A measurement for a different mover/pose cannot describe this action.
    """
    last = context.get('last')
    if not last or last['choice']['part_id'] != part_id or last['choice']['pose_id'] != pose_id:
        return None
    measured = last['feedback'].get('measured_endpoint')
    if not isinstance(measured, dict) or measured.get('scope') != SCOPE:
        return None
    threshold, pairs = measured.get('min_sep'), measured.get('pairs')
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0 or type(pairs) is not list:
        return None
    expected = sorted(p for p in context['remaining_parts'] if p != part_id)
    seen, result = [], []
    for pair in pairs:
        if type(pair) is not dict:
            return None
        other, distance = pair.get('rest_part_id'), pair.get('min_distance')
        flag = pair.get('separation_unsatisfied')
        if (pair.get('moving_part_id') != part_id or other not in expected or other in seen
                or type(distance) not in (int, float) or not math.isfinite(distance)
                or type(flag) is not bool or flag != (distance < threshold)):
            return None
        seen.append(other)
        if distance < threshold:
            result.append(other)
    return sorted(result) if sorted(seen) == expected else None


class SeparationVM(TraceVM):
    def __init__(self, source, context, override=None):
        super().__init__(source, lambda choice: False, context)
        self.separation_override = override
        self.separation_calls = []
        self.in_separation = False

    def call(self, name, values, keywords=None):
        if self.in_separation and name in ('eligible', 'select_directions', HELPER):
            raise ValueError('python_separation_helper_must_be_pure')
        if name != HELPER:
            return super().call(name, values, keywords)
        if len(self.separation_calls) >= 32:
            raise ValueError('python_separation_helper_call_limit')
        args = dict(zip(('ctx', 'part_id', 'pose_id'), values))
        args.update(keywords or {})
        before = digest(args.get('ctx'))
        self.in_separation = True
        try:
            value = (super().call(name, values, keywords) if self.separation_override is None
                     else deepcopy(self.separation_override))
        finally:
            self.in_separation = False
        if value is not None and (type(value) is not list or any(type(p) is not str for p in value)):
            raise ValueError('python_separation_helper_result')
        self.separation_calls.append(dict(context_sha256=before,
            context_unchanged=before == digest(args.get('ctx')), part_id=args.get('part_id'),
            pose_id=args.get('pose_id'), result=value, inside_direction=bool(self.in_direction)))
        return value


def measured_case(context, distances, threshold=0.5):
    changed = deepcopy(context)
    measured = changed['last']['feedback']['measured_endpoint']
    measured['min_sep'] = threshold
    for pair, distance in zip(measured['pairs'], distances):
        pair.update(min_distance=distance, separation_unsatisfied=distance < threshold)
    return changed


def semantic_cases(context, part_id, pose_id):
    pairs = context['last']['feedback']['measured_endpoint']['pairs']
    count = len(pairs)
    yield 'actual', context, part_id, pose_id
    # Positive distances below a positive threshold are STILL unsatisfied.
    for label, distances, threshold in (
        ('one_small_positive_among_far', [0.25] + [100.0] * (count-1), 0.5),
        ('same_distances_lower_threshold', [0.25] + [100.0] * (count-1), 0.1),
        ('at_boundary', [0.5] * count, 0.5),
        ('below_boundary', [0.5-1e-6] * count, 0.5),
        ('all_above', [1.0] * count, 0.5),
        ('all_below', [-0.1] * count, 0.5)):
        yield label, measured_case(context, distances, threshold), part_id, pose_id
    reordered = deepcopy(context)
    reordered['last']['feedback']['measured_endpoint']['pairs'].reverse()
    yield 'pair_order', reordered, part_id, pose_id
    for label in ('missing_endpoint', 'missing_pair', 'duplicate_pair', 'wrong_mover', 'inconsistent_flag', 'wrong_scope'):
        changed = deepcopy(context)
        measured = changed['last']['feedback']['measured_endpoint']
        if label == 'missing_endpoint':
            changed['last']['feedback']['measured_endpoint'] = None
        elif label == 'missing_pair':
            measured['pairs'].pop()
        elif label == 'duplicate_pair':
            measured['pairs'].append(deepcopy(measured['pairs'][0]))
        elif label == 'wrong_mover':
            measured['pairs'][0]['moving_part_id'] = '__synthetic_different_mover'
        elif label == 'inconsistent_flag':
            measured['pairs'][0]['separation_unsatisfied'] = not measured['pairs'][0]['separation_unsatisfied']
        else:
            measured['scope'] = 'different_state'
        yield label, changed, part_id, pose_id
    yield 'different_selected_mover', context, pairs[0]['rest_part_id'], pose_id
    yield 'different_selected_pose', context, part_id, '__synthetic_different_pose'


def inspect(program, request, choice):
    result = dict(version='separation_use_probe_v1', accepted=False, errors=[],
        api_calls=0, physical_calls=0, memory_writes=0,
        scope='synthetic_semantics_and_sensitivity_NOT_physics',
        new_physical_success=False)
    functions = {f.name:f for f in parse(program['source']).body}
    helper = functions.get(HELPER)
    if helper is None or [a.arg for a in helper.args.args] != ['ctx','part_id','pose_id']:
        result['errors'].append('separation_helper_contract_missing')
        return result
    context, _ = build_context(request, extension_for())
    part, pose = choice['part_id'], choice['pose_id']
    expected = unmet(context, part, pose)
    if expected is None or not expected:
        result['errors'].append('scoped_unmet_separation_not_available')
        return result
    original_hash = digest(context)
    rows = []
    for label, ctx, mover, attitude in semantic_cases(context, part, pose):
        before = digest(ctx)
        vm = SeparationVM(program['source'], ctx)
        observed = vm.call(HELPER, [ctx, mover, attitude])
        correct = unmet(ctx, mover, attitude)
        rows.append(dict(name=label, expected=correct, actual=observed,
            accepted=observed == correct and before == digest(ctx), synthetic_NOT_observed=label != 'actual'))
    result['semantics'] = rows
    if not all(row['accepted'] for row in rows):
        result['errors'].append('per_pair_separation_semantics_incorrect')
        return result
    vm = SeparationVM(program['source'], context)
    actual = sequence(vm.call('select_directions', [context, part, pose]), context)
    bound = any(c['context_sha256'] == original_hash and c['context_unchanged']
        and c['part_id'] == part and c['pose_id'] == pose and c['result'] == expected
        and c['inside_direction'] for c in vm.separation_calls)
    result['actual_helper_bound_to_direction'] = bound
    if not bound or actual != action_sequence(choice) or digest(context) != original_hash:
        result['errors'].append('separation_helper_not_bound_to_actual_direction')
        return result
    pairs = context['last']['feedback']['measured_endpoint']['pairs']
    endpoint_rows, injected_rows = [], []
    # Keep geometry, failed force and the distance multiset fixed. Only which
    # pair falls short changes. Positive sum/sign tricks cannot distinguish it.
    for index, pair in enumerate(pairs[:4]):
        changed = measured_case(context, [0.25 if i == index else 100.0 for i in range(len(pairs))])
        before = digest(changed)
        probe = SeparationVM(program['source'], changed)
        directions = sequence(probe.call('select_directions', [changed, part, pose]), changed)
        if digest(changed) != before:
            raise ValueError('python_separation_context_mutated')
        endpoint_rows.append(dict(rest_part_id=pair['rest_part_id'], directions=directions, synthetic_NOT_observed=True))
        injected_ctx = deepcopy(context)
        injected = SeparationVM(program['source'], injected_ctx, [pair['rest_part_id']])
        value = sequence(injected.call('select_directions', [injected_ctx, part, pose]), injected_ctx)
        if digest(injected_ctx) != original_hash:
            raise ValueError('python_separation_context_mutated')
        injected_rows.append(dict(injected_unmet_parts=[pair['rest_part_id']], directions=value, synthetic_NOT_observed=True))
    for label, probes in [('endpoint_pair', endpoint_rows), ('helper_result', injected_rows)]:
        concrete = {tuple(r['directions']) for r in probes if r['directions'] is not None}
        result[label + '_direction_sensitive'] = len(concrete) >= 2
        if len(concrete) < 2:
            result['errors'].append(label + '_direction_dependence_not_demonstrated')
    result.update(endpoint_probes=endpoint_rows, helper_return_probes=injected_rows,
        probe_count=len(rows)+1+len(endpoint_rows)+len(injected_rows), accepted=not result['errors'])
    if result['probe_count'] > MAX_PROBES:
        raise ValueError('python_separation_probe_limit')
    return result
