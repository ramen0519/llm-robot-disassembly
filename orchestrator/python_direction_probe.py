"""Bounded synthetic input interventions, NOT physical evidence or action ranking."""
from copy import deepcopy
import math

from orchestrator.python_skill_vm import VM, parse
from orchestrator.python_skill_program import build_context, _response
from orchestrator.lesson_runtime import extension_for
from worker.explicit_action_contract import digest

HELPER = 'select_directions'
MAX_PROBES = 24
MAX_HELPER_CALLS = 32


def sequence(value, context):
    if value is None:
        return None
    if (type(value) is not list or not 1 <= len(value) <= 2
            or any(type(x) is not str or x not in context['directions'] for x in value)):
        raise ValueError('python_direction_helper_result')
    return value


def action_sequence(choice):
    return [choice['direction_id']] + ([choice['second_direction_id']]
        if choice['second_direction_id'] is not None else [])


class TraceVM(VM):
    def __init__(self, source, eligible, context, override=None):
        super().__init__(source, eligible)
        self.context = context
        self.direction_calls = []
        self.in_direction = 0
        self.override = override

    def call(self, name, values, keywords=None):
        if name == 'eligible' and self.in_direction:
            raise ValueError('python_direction_helper_must_be_pure')
        if name != HELPER:
            return super().call(name, values, keywords)
        if len(self.direction_calls) >= MAX_HELPER_CALLS or self.in_direction:
            raise ValueError('python_direction_helper_call_limit')
        args = dict(zip(('ctx', 'part_id', 'pose_id'), values))
        args.update(keywords or {})
        before = digest(args.get('ctx'))
        self.in_direction += 1
        try:
            value = super().call(name, values, keywords) if self.override is None else deepcopy(self.override)
        finally:
            self.in_direction -= 1
        self.direction_calls.append(dict(context_sha256=before,
            context_unchanged=before == digest(args.get('ctx')),
            part_id=args.get('part_id'), pose_id=args.get('pose_id'),
            directions=sequence(value, self.context)))
        return value


def center(record):
    value = record.get('bounds_center')
    if (type(value) is not list or len(value) != 3 or
            any(type(n) not in (int, float) or not math.isfinite(n) for n in value)):
        return None
    return value


def interventions(context, part_id):
    """One family changes geometry, the other failure details. Trigger stays fixed.

    These are intentionally counterfactual data, never a verified CAD/trajectory.
    They do not leave this disposable diagnostic process or enter any memory.
    """
    output = []
    records = context['geometry'].get('parts', {})
    origin = center(records.get(part_id, {}))
    other = [p for p in context['remaining_parts'] if p != part_id and p in records]
    if origin is not None and other:
        ext = records[part_id].get('extents') or [1, 1, 1]
        scale = max([abs(float(x)) for x in ext if type(x) in (int, float) and math.isfinite(x)] + [1.0]) * 3
        for axis in range(3):
            for sign in (-1, 1):
                changed = deepcopy(context)
                for offset, identifier in enumerate(other):
                    record = changed['geometry']['parts'][identifier]
                    prior = center(record)
                    if prior is None:
                        continue
                    dest = list(origin)
                    dest[axis] += sign * scale * (1 + offset / 10)
                    delta = [dest[i] - prior[i] for i in range(3)]
                    record['bounds_center'] = dest
                    if center({'bounds_center':record.get('center_mass')}) is not None:
                        record['center_mass'] = [record['center_mass'][i] + delta[i] for i in range(3)]
                    bounds = record.get('bounds')
                    if type(bounds) is list and len(bounds) == 2:
                        record['bounds'] = [[v[i] + delta[i] for i in range(3)] for v in bounds]
                output.append(('geometry', 'relative_position_%s_%s' % (axis, sign), changed))
    if context.get('last') is not None:
        # Do not count failure-trigger on/off as directional evidence use.
        for direction in context['directions']:
            changed = deepcopy(context)
            changed['last']['choice']['direction_id'] = direction
            if type(changed['last']['feedback'].get('choice')) is dict:
                changed['last']['feedback']['choice']['direction_id'] = direction
            output.append(('failure', 'failed_force_' + direction, changed))
        endpoint = context['last']['feedback'].get('measured_endpoint')
        if type(endpoint) is dict and endpoint.get('pairs'):
            for index in range(min(4, len(endpoint['pairs']))):
                changed = deepcopy(context)
                measured = changed['last']['feedback']['measured_endpoint']
                minimum = measured.get('min_sep')
                if type(minimum) not in (int, float) or not math.isfinite(minimum):
                    continue
                for i, pair in enumerate(measured['pairs']):
                    pair['separation_unsatisfied'] = i == index
                    pair['min_distance'] = minimum - 1 if i == index else minimum + 1
                output.append(('failure', 'endpoint_pair_' + str(index), changed))
    if len(output) > MAX_PROBES:
        raise ValueError('python_direction_probe_limit')
    return output


def inspect(program, request, expected):
    tree = parse(program['source'])
    functions = {f.name:f for f in tree.body}
    result = dict(version='direction_use_probe_v1', accepted=False, errors=[],
        scope='synthetic_sensitivity_NOT_physics_or_causal_success',
        fixed_part_id=expected['part_id'], fixed_pose_id=expected['pose_id'],
        api_calls=0, physical_calls=0, memory_writes=0, probe_count=0,
        sensitivity_is_sufficient_for_physical_success=False)
    if (HELPER not in functions or [a.arg for a in functions[HELPER].args.args] != ['ctx','part_id','pose_id']):
        result['errors'].append('direction_helper_contract_missing')
        return result
    context, _ = build_context(request, extension_for())
    original = digest(context)
    calls = [0]
    def eligible(choice):
        calls[0] += 1
        if calls[0] > 512:
            raise ValueError('python_eligibility_budget')
        try:
            _response(request, choice, digest(program))
            return True
        except (ValueError, TypeError, KeyError):
            return False
    vm = TraceVM(program['source'], eligible, context)
    actual = vm.call('propose', [context])
    if digest(context) != original:
        result['errors'].append('direction_context_mutated')
    if actual != expected:
        result['errors'].append('direction_diagnostic_differs_from_runtime')
    seq = action_sequence(expected)
    binding = any(c['context_sha256'] == original and c['context_unchanged'] and
        c['part_id'] == expected['part_id'] and c['pose_id'] == expected['pose_id'] and
        c['directions'] == seq for c in vm.direction_calls)
    result['actual_helper_bound_to_choice'] = binding
    if not binding:
        result['errors'].append('direction_helper_not_bound_to_actual_choice')
    if result['errors']:
        return result
    # An unused helper can accidentally agree on the original input. Intervene
    # on its return and verify that the real propose output follows it. These
    # executions are diagnostics only; no intervened action reaches physics.
    overrides=[]
    for direction in context['directions']:
        for second in ([None] if len(seq)==1 else context['directions']):
            changed=[direction]+([second] if second is not None else [])
            if changed==seq:continue
            candidate=deepcopy(expected)
            candidate.update(direction_id=direction,second_direction_id=second)
            if not eligible(candidate):continue
            overrides.append(changed)
            if len(overrides)==2:break
        if len(overrides)==2:break
    followed=[]
    for override in overrides:
        ctx=deepcopy(context)
        tracer=TraceVM(program['source'],eligible,ctx,override)
        choice=tracer.call('propose',[ctx])
        follows=(type(choice) is dict and choice.get('part_id')==expected['part_id']
            and choice.get('pose_id')==expected['pose_id'] and action_sequence(choice)==override
            and digest(ctx)==original and bool(tracer.direction_calls))
        followed.append(dict(injected_helper_return=override,actual_choice_follows=follows))
    result['helper_return_interventions']=followed
    if len(followed)!=2 or not all(r['actual_choice_follows'] for r in followed):
        result['errors'].append('direction_helper_result_not_driving_actual_choice')
        return result
    # Direct helper calls hold mover/pose/support decisions outside the test.
    def execute(ctx):
        before = digest(ctx)
        child = TraceVM(program['source'], eligible, ctx)
        value = child.call(HELPER, [ctx, expected['part_id'], expected['pose_id']])
        if digest(ctx) != before:
            raise ValueError('python_direction_context_mutated')
        return sequence(value, ctx)
    if execute(deepcopy(context)) != seq:
        result['errors'].append('direction_helper_not_repeatable')
        return result
    rows = []
    for family, name, ctx in interventions(context, expected['part_id']):
        value = execute(ctx)
        rows.append(dict(family=family, name=name, directions=value,
            changed_from_actual=value != seq, synthetic_NOT_observed=True))
    # Require two concrete direction choices per family. Merely abstaining or
    # changing the trigger's existence is NOT evidence of directional reasoning.
    for family in ('geometry','failure'):
        unique = {tuple(seq)} | {tuple(r['directions']) for r in rows if r['family']==family and r['directions'] is not None}
        sensitive = len(unique) > 1
        result[family + '_direction_sensitive'] = sensitive
        if not sensitive:
            result['errors'].append(family + '_direction_dependence_not_demonstrated')
    reordered = deepcopy(context)
    reordered['directions'] = dict(reversed(list(reordered['directions'].items())))
    reordered['geometry']['parts'] = dict(reversed(list(reordered['geometry']['parts'].items())))
    same_order = execute(reordered) == seq
    result['map_order_invariant'] = same_order
    if not same_order:
        result['errors'].append('direction_depends_on_map_serialization_order')
    result.update(probe_count=len(rows)+2, propose_executions=1+len(followed),
                  probes=rows, original_direction_sequence=seq,
                  accepted=not result['errors'])
    return result
