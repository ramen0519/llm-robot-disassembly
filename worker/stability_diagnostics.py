"""Observe the pinned ASAP stability checker without another physics query.

Python 3.8 compatible.  This is diagnostics, not a verifier or a repair rule.
The original ``MultiPartStabilityPlanner.check_success`` remains the caller's
only stability invocation.  The temporary subclass records existing results;
it never supplies a new verdict, support, contact, or simulation step.
"""
from contextlib import contextmanager
from copy import deepcopy
import math


OBSERVER = 'asap_stability_diagnostics_v1'
PHASES = frozenset(('initial_disconnected', 'dynamic_failure', 'timeout',
                    'passed', 'error', 'unverified_return'))


def _ids(value):
    if (type(value) is not list or len(value) > 10000 or
            any(type(x) is not str or not x or len(x) > 512 or
                any(c in x for c in '\x00\r\n/\\') for x in value) or
            len(set(value)) != len(value)):
        raise ValueError('invalid_stability_diagnostic_ids')
    return value[:]


def _number(value, nullable=False):
    if nullable and value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid_stability_diagnostic_number')
    return value


def project_stability_diagnostics(value):
    """Validate and project measured facts for episode/frozen-memory inputs.

    Missing legacy diagnostics stay missing.  These local part IDs belong to
    the source observation; this function does not map them to another task.
    Graph details and arbitrary strings are deliberately not model inputs.
    """
    if value is None:
        return None
    if (type(value) is not dict or type(value.get('schema_version')) is not int or
            value['schema_version'] != 1 or value.get('observer') != OBSERVER or
            value.get('phase') not in PHASES):
        raise ValueError('invalid_stability_diagnostic_schema')
    result = {key: value[key] for key in ('schema_version', 'observer', 'phase')}
    for key in ('physics_steps', 'upstream_contact_queries_observed',
                'extra_physics_calls', 'extra_geometry_queries'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError('invalid_stability_diagnostic_counter')
        result[key] = value[key]
    if result['extra_physics_calls'] or result['extra_geometry_queries']:
        raise ValueError('stability_observer_must_not_query')
    for key in ('check_success_returned',):
        if type(value.get(key)) is not bool:
            raise ValueError('invalid_stability_diagnostic_boolean')
        result[key] = value[key]
    success = value.get('returned_success')
    if success is not None and type(success) is not bool:
        raise ValueError('invalid_stability_diagnostic_result')
    result['returned_success'] = success
    for key in ('initial_disconnected_parts', 'returned_parts'):
        result[key] = None if value.get(key) is None else _ids(value[key])
    rows = value.get('dynamic_failure_parts')
    if type(rows) is not list or len(rows) > 10000:
        raise ValueError('invalid_stability_diagnostic_failures')
    result['dynamic_failure_parts'] = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError('invalid_stability_diagnostic_failure')
        part = _ids([row.get('part_id')])[0]
        exceeded = row.get('threshold_exceeded')
        if type(exceeded) is not bool:
            raise ValueError('invalid_stability_diagnostic_boolean')
        lost = _ids(row.get('lost_initial_contacts'))
        displacement = _number(row.get('displacement'))
        threshold = _number(row.get('displacement_threshold'), nullable=True)
        if not exceeded and not lost:
            raise ValueError('stability_failure_requires_observed_condition')
        if exceeded and (threshold is None or displacement <= threshold):
            raise ValueError('stability_displacement_condition_mismatch')
        result['dynamic_failure_parts'].append(dict(part_id=part,
            displacement=displacement, displacement_threshold=threshold,
            threshold_exceeded=exceeded, lost_initial_contacts=lost))
    _ids([row['part_id'] for row in result['dynamic_failure_parts']])
    semantics = value.get('returned_parts_semantics')
    if semantics not in ('not_observed', 'initial_contact_graph_degree_zero',
            'upstream_representatives_after_optional_contact_component_grouping',
            'no_returned_parts'):
        raise ValueError('invalid_stability_diagnostic_semantics')
    result['returned_parts_semantics'] = semantics
    if value.get('contact_query_observation_scope') != 'check_fallen_parts_only':
        raise ValueError('invalid_stability_query_scope')
    result['contact_query_observation_scope'] = 'check_fallen_parts_only'
    if value['phase'] == 'initial_disconnected':
        if (success is not False or not result['initial_disconnected_parts'] or
                result['returned_parts'] != result['initial_disconnected_parts']):
            raise ValueError('stability_initial_branch_mismatch')
    elif value['phase'] == 'dynamic_failure':
        if (success is not False or not result['returned_parts'] or
                not set(result['returned_parts']).issubset(
                    row['part_id'] for row in result['dynamic_failure_parts'])):
            raise ValueError('stability_dynamic_branch_mismatch')
    elif value['phase'] in ('timeout', 'passed'):
        if success is not (value['phase'] == 'passed') or result['returned_parts'] is not None:
            raise ValueError('stability_terminal_branch_mismatch')
    return deepcopy(result)


class StabilityObservation:
    def __init__(self):
        self.data = dict(schema_version=1, observer=OBSERVER,
            phase='unverified_return', check_success_returned=False,
            returned_success=None, returned_parts=None, physics_steps=0,
            initial_disconnected_parts=None, dynamic_failure_parts=[],
            returned_parts_semantics='not_observed',
            upstream_contact_queries_observed=0, extra_physics_calls=0,
            extra_geometry_queries=0, instrumentation_errors=[],
            contact_query_observation_scope='check_fallen_parts_only',
            dynamic_checks=0, initial_contact_graph=None)
        self.checker_count = 0
        self.last_fallen = None

    def capture(self, fn, *args):
        # Diagnostics failures must never replace an upstream result/exception.
        try:
            fn(*args)
        except Exception as error:
            self.data['instrumentation_errors'].append(type(error).__name__)

    def returned(self, success, parts):
        data = self.data
        data['check_success_returned'] = True
        data['returned_success'] = bool(success)
        data['returned_parts'] = None if parts is None else list(parts)
        if self.checker_count != 1 or data['instrumentation_errors']:
            return
        initial = data['initial_disconnected_parts']
        if initial is None:
            return
        if not success and initial and data['returned_parts'] == initial:
            data['phase'] = 'initial_disconnected'
            data['returned_parts_semantics'] = 'initial_contact_graph_degree_zero'
        elif not success and self.last_fallen and data['returned_parts'] == self.last_fallen:
            data['phase'] = 'dynamic_failure'
            data['returned_parts_semantics'] = (
                'upstream_representatives_after_optional_contact_component_grouping')
        elif not initial and parts is None:
            data['phase'] = 'passed' if success else 'timeout'
            data['returned_parts_semantics'] = 'no_returned_parts'


@contextmanager
def observe_stability(physics):
    """Temporarily observe the exact module-global checker used by ASAP.

    A worker executes stability serially.  Nested observers are rejected.  The
    original class and each C++ simulation reference are restored in finally.
    Wall-clock instrumentation overhead is nonzero even with zero extra
    simulation/geometry calls; a near-deadline execution may therefore differ.
    """
    original = physics.StabilityChecker
    if getattr(original, '_stability_observer_active', False):
        raise ValueError('nested_stability_observer')
    observation = StabilityObservation()
    data = observation.data

    class SimProxy:
        def __init__(self, sim, checker, contacts):
            self._sim, self._checker, self._contacts = sim, checker, contacts

        def __getattr__(self, name):
            return getattr(self._sim, name)

        def body_in_contact(self, first, second, epsilon):
            result = self._sim.body_in_contact(first, second, epsilon)
            data['upstream_contact_queries_observed'] += 1
            if not self._checker._deriving_graph:
                self._contacts.append((first, second, bool(result)))
            return result

    class ObservedChecker(original):
        _stability_observer_active = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._deriving_graph = False
            observation.checker_count += 1

        def derive_contact_graph(self):
            previous = self._deriving_graph
            self._deriving_graph = True
            try:
                return super().derive_contact_graph()
            finally:
                self._deriving_graph = previous

        def update_sim(self, *args, **kwargs):
            result = super().update_sim(*args, **kwargs)
            def record():
                data['initial_contact_graph'] = dict(
                    nodes=sorted(str(x) for x in self.G.nodes),
                    edges=sorted(sorted((str(a), str(b))) for a, b in self.G.edges))
            observation.capture(record)
            return result

        def update_status(self, *args, **kwargs):
            result = super().update_status(*args, **kwargs)
            observation.capture(lambda: data.update(physics_steps=int(self.n_step)))
            return result

        def check_disconnected_parts(self, *args, **kwargs):
            result = super().check_disconnected_parts(*args, **kwargs)
            observation.capture(lambda: data.update(initial_disconnected_parts=list(result)))
            return result

        def check_fallen_parts(self, *args, **kwargs):
            sim, contacts = self.sim, []
            self.sim = SimProxy(sim, self, contacts)
            try:
                result = super().check_fallen_parts(*args, **kwargs)
            finally:
                self.sim = sim
            def record():
                data['dynamic_checks'] += 1
                observation.last_fallen = list(result)
                if not result:
                    return
                failures = []
                threshold = float(self.pos_far_threshold)
                for part in self.parts_move:
                    distance = float(self.dist_his_map[part][-1])
                    exceeded = distance > threshold
                    # These are original checker queries, not a new contact test.
                    lost = [first[4:] for first, second, present in contacts
                            if second == 'part' + part and first.startswith('part')
                            and not present]
                    if exceeded or lost:
                        failures.append(dict(part_id=part, displacement=distance,
                            displacement_threshold=threshold if math.isfinite(threshold) else None,
                            threshold_exceeded=exceeded, lost_initial_contacts=lost))
                data['dynamic_failure_parts'] = failures
            observation.capture(record)
            return result

    physics.StabilityChecker = ObservedChecker
    try:
        yield observation
    except BaseException:
        data['phase'] = 'error'
        raise
    finally:
        physics.StabilityChecker = original
