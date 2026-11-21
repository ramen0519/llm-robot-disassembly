"""Bounded, anonymous part scoring for the path-only support-cost experiment."""
import ast
import math

from orchestrator.python_skill_vm import PolicyError, VM, parse


FEATURES = {
    'parent_count': 'integer number of parts in the current assembly',
    'contact_degree': 'current neighbors in the initial native contact graph restricted to the remaining parts',
    'degree_fraction': 'contact_degree divided by parent_count minus one',
    'components_before': 'number of connected components in the current induced contact graph',
    'components_if_removed': 'number of connected components after deleting this candidate',
    'component_delta': 'components_if_removed minus components_before',
    'remainder_isolated_nodes': 'number of vertices with degree zero after deleting this candidate',
    'distance_from_center_over_diagonal': 'distance between candidate and assembly bounding-box centers, divided by assembly diagonal, or null',
    'shortest_over_longest_extent': 'ratio of the shortest and longest candidate bounding-box sides, or null',
    'volume_fraction_if_valid': 'fraction of remaining mesh volume if all mass properties are valid, otherwise null',
}
INTEGER_FIELDS = {'parent_count', 'contact_degree', 'components_before', 'components_if_removed',
                  'component_delta', 'remainder_isolated_nodes'}
OPTIONAL = {'distance_from_center_over_diagonal', 'shortest_over_longest_extent', 'volume_fraction_if_valid'}
MAX_SOURCE_BYTES = 4096
MAX_PARTS = 128
MAX_STEPS_PER_PART = 2048


def _components(nodes, edges):
    adjacent = {node: set() for node in nodes}
    for a, b in edges:
        if a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)
    unseen = set(nodes)
    count = 0
    while unseen:
        count += 1
        todo = [unseen.pop()]
        while todo:
            neighbors = adjacent[todo.pop()] & unseen
            unseen -= neighbors
            todo.extend(neighbors)
    return count


def validate_rows(rows):
    if type(rows) is not dict or not 2 <= len(rows) <= MAX_PARTS:
        raise ValueError('invalid_candidate_set')
    for part, row in rows.items():
        if type(part) is not str or not 1 <= len(part) <= 128 or type(row) is not dict or set(row) != set(FEATURES):
            raise ValueError('invalid_feature_row')
        if row['parent_count'] != len(rows):
            raise ValueError('candidate_set_mismatch')
        for field, value in row.items():
            if value is None and field in OPTIONAL:
                continue
            if field in INTEGER_FIELDS and type(value) is not int:
                raise ValueError('invalid_integer_feature')
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('invalid_numeric_feature')
    return rows


def feature_rows(snapshot, parent):
    """Compute the pre-choice fields used by the historical geometry study."""
    parts = list(parent)
    nodes = set(parts)
    meshes = snapshot['parts']
    if len(nodes) != len(parts) or len(nodes) < 2 or not nodes <= meshes.keys():
        raise ValueError('invalid_parent_parts')
    edges = {frozenset(edge) for edge in snapshot['contact_edges'] if set(edge) <= nodes}
    if any(len(edge) != 2 for edge in edges):
        raise ValueError('invalid_contact_edge')
    components = _components(nodes, edges)
    lo = [min(meshes[p]['bounds'][0][i] for p in parts) for i in range(3)]
    hi = [max(meshes[p]['bounds'][1][i] for p in parts) for i in range(3)]
    center = [(a+b)/2 for a,b in zip(lo,hi)]
    diagonal = math.dist(lo,hi)
    volumes_valid = all(meshes[p]['validity']['center_mass_status']=='uniform_density_estimate' for p in parts)
    total_volume = math.fsum(meshes[p]['volume'] for p in parts) if volumes_valid else None
    rows = {}
    for part in sorted(parts):
        child = nodes - {part}
        child_edges = [edge for edge in edges if part not in edge]
        child_components = _components(child, child_edges)
        degree = sum(part in edge for edge in edges)
        extent = meshes[part]['extents']
        rows[part] = dict(parent_count=len(parts), contact_degree=degree,
            degree_fraction=degree/(len(parts)-1), components_before=components,
            components_if_removed=child_components, component_delta=child_components-components,
            remainder_isolated_nodes=len(child-set().union(*child_edges)),
            distance_from_center_over_diagonal=math.dist(meshes[part]['bounds_center'],center)/diagonal if diagonal else None,
            shortest_over_longest_extent=min(extent)/max(extent) if max(extent) else None,
            volume_fraction_if_valid=meshes[part]['volume']/total_volume if total_volume else None)
    return validate_rows(rows)


def validate_source(source):
    if type(source) is not str or not 1 <= len(source.encode('utf8')) <= MAX_SOURCE_BYTES:
        raise ValueError('invalid_scored_source_size')
    tree = parse(source)
    # Numeric program over one feature row: no name, label, product-specific string
    # or ordered candidate index can ever be supplied to the callable.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is str and node.value not in FEATURES:
            raise PolicyError('unknown_scored_feature_literal', node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'eligible':
            raise PolicyError('no_eligibility_api_in_scored_skill', node)
    return source


def run_skill(source, rows):
    """Run Luna's numeric policy on all candidates or return no usable decision."""
    base = dict(candidate_order=None, priorities={}, diagnostics=[], vm_steps=0)
    try:
        validate_source(source)
        validate_rows(rows)
        steps = 0
        scores = {}
        for part, row in rows.items():
            vm = VM(source, lambda _: False, max_steps=MAX_STEPS_PER_PART)
            score = vm.call('propose', [row.copy()])
            steps += vm.steps
            if score is None:
                return dict(base, status='ABSTAIN', diagnostics=['candidate_abstained'], vm_steps=steps)
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError('priority_must_be_finite_number_or_none')
            scores[part] = score
        order = sorted(rows, key=lambda p:(-scores[p],p))
        diagnostics = ['all_candidate_priorities_tied'] if len(set(scores.values())) == 1 else []
        return dict(base, status='PROPOSED', candidate_order=order, priorities=scores,
                    diagnostics=diagnostics, vm_steps=steps)
    except (ValueError, ArithmeticError, TypeError, KeyError, IndexError, RecursionError) as err:
        diagnostic = getattr(err, 'diagnostic', {'code': str(err)[:180]})
        return dict(base, status='REJECTED', diagnostics=[diagnostic])
