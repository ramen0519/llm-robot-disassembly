"""Pinned ASAP method bodies + synthetic simulation; not physical success.

No ASAP/RedMax import, API call, WSL, or simulation is performed by this test.
The synthetic simulator records every call so observer on/off can be compared.
"""
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
from stability_diagnostics import observe_stability, project_stability_diagnostics
from tests.test_explicit_action_planner import make_planner, result_for
from tests import test_explicit_action_physics as physics_fixture


class Vector(list):
    def __sub__(self, other):
        return Vector(a - b for a, b in zip(self, other))

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Vector(result) if isinstance(key, slice) else result


class Graph:
    def __init__(self):
        self.adj = {}

    def add_node(self, node):
        self.adj.setdefault(node, [])

    def add_edge(self, first, second):
        self.add_node(first)
        self.add_node(second)
        if second not in self.adj[first]:
            self.adj[first].append(second)
            self.adj[second].append(first)

    @property
    def nodes(self):
        return list(self.adj)

    @property
    def edges(self):
        return [(a, b) for a in self.adj for b in self.adj[a] if a < b]

    def degree(self, node):
        return len(self.adj[node])

    def neighbors(self, node):
        return iter(self.adj[node])

    def subgraph(self, nodes):
        graph = Graph()
        for node in nodes:
            graph.add_node(node)
        for a, b in self.edges:
            if a in nodes and b in nodes:
                graph.add_edge(a, b)
        return graph


def components(graph):
    remaining = set(graph.nodes)
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            node = pending.pop()
            if node not in component:
                component.add(node)
                pending.extend(graph.adj[node])
        remaining -= component
        yield component


class SimulatorFixture:
    def __init__(self, mode):
        self.mode, self.step, self.calls = mode, 0, []

    def reset(self):
        self.calls.append(('reset',))
        self.step = 0

    def forward(self, steps, verbose=False):
        self.calls.append(('forward', steps, verbose))
        self.step += steps
        if self.mode == 'exception':
            raise RuntimeError('synthetic_simulator_failure')

    def get_joint_qm(self, name):
        self.calls.append(('joint', name))
        displacement = 0.
        if self.step and self.mode in ('displacement', 'grouped'):
            displacement = .3 if name == 'parta' else .4
        return Vector([displacement, 0., 0., 0., 0., 0.])

    def body_in_contact(self, first, second, epsilon):
        self.calls.append(('contact', first, second, epsilon))
        pair = frozenset((first, second))
        if self.mode == 'disconnected' and 'partb' in pair:
            return False
        if self.mode == 'contact_loss' and self.step and pair == {'parta', 'partf'}:
            return False
        if pair == {'parta', 'partb'}:
            return self.mode == 'grouped'
        return True

    def get_body_mass(self, name):
        self.calls.append(('mass', name))
        return 2. if name == 'partb' else 1.


def upstream_fixture(mode):
    """Load only actual audited classes; imports/constructors are not executed."""
    tree = ast.parse((ROOT / 'external/ASAP/plan_sequence/physics_planner.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
             node.name in ('StabilityChecker', 'MultiPartStabilityPlanner')]
    clock = [0.]
    def time():
        clock[0] += 1.
        return clock[0]
    namespace = dict(CONTACT_EPS=.1, POS_FAR_THRESHOLD=.2, POS_NEAR_THRESHOLD=.05,
        NEAR_STEP=100, MAX_STEP=1000, COL_TH_STABLE=0., CHECK_FREQ=20, DEBUG_SIM=False,
        np=SimpleNamespace(inf=float('inf'), linalg=SimpleNamespace(
            norm=lambda value: sum(x*x for x in value) ** .5)),
        nx=SimpleNamespace(Graph=Graph, connected_components=components), time=time)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 'actual_upstream_stability_method_fixture', 'exec'), namespace)
    # The original method resolves module globals in this dictionary.
    class ModuleProxy:
        def __getattr__(self, name):
            return namespace[name]
        def __setattr__(self, name, value):
            namespace[name] = value
    module = ModuleProxy()
    planner = module.MultiPartStabilityPlanner.__new__(module.MultiPartStabilityPlanner)
    planner.sim = SimulatorFixture(mode)
    planner.parts_move, planner.parts_fix, planner.allow_gap = ['a', 'b'], ['f'], False
    return module, planner


def run_fixture(mode, observed):
    module, planner = upstream_fixture(mode)
    original = module.StabilityChecker
    timeout = .5 if mode == 'timeout' else None
    diagnostic = None
    if observed:
        with observe_stability(module) as collector:
            result = planner.check_success(max_step=40, timeout=timeout)
            collector.returned(*result)
        diagnostic = collector.data
    else:
        result = planner.check_success(max_step=40, timeout=timeout)
    assert module.StabilityChecker is original
    return result, planner.sim.calls, diagnostic


class StabilityDiagnosticsTests(unittest.TestCase):
    def test_all_branches_preserve_exact_return_and_every_simulator_call(self):
        expected = {'disconnected': 'initial_disconnected', 'displacement': 'dynamic_failure',
            'contact_loss': 'dynamic_failure', 'grouped': 'dynamic_failure',
            'timeout': 'timeout', 'stable': 'passed'}
        for mode, phase in expected.items():
            with self.subTest(mode=mode):
                plain = run_fixture(mode, False)
                observed = run_fixture(mode, True)
                self.assertEqual(plain[:2], observed[:2])
                self.assertEqual(observed[2]['phase'], phase)
                self.assertEqual(observed[2]['instrumentation_errors'], [])
                self.assertEqual(project_stability_diagnostics(observed[2])['phase'], phase)
                self.assertEqual(observed[2]['extra_physics_calls'], 0)
                self.assertEqual(observed[2]['extra_geometry_queries'], 0)

    def test_initial_disconnection_has_zero_physics_steps_and_no_dynamic_cause(self):
        result, _, data = run_fixture('disconnected', True)
        self.assertEqual(result, (False, ['b']))
        self.assertEqual(data['initial_disconnected_parts'], ['b'])
        self.assertEqual(data['physics_steps'], 0)
        self.assertEqual(data['dynamic_failure_parts'], [])

    def test_displacement_is_3d_norm_not_invented_vertical_fall(self):
        data = run_fixture('displacement', True)[2]
        self.assertEqual(data['physics_steps'], 20)
        self.assertEqual(data['dynamic_failure_parts'][0], dict(part_id='a', displacement=.3,
            displacement_threshold=.2, threshold_exceeded=True, lost_initial_contacts=[]))

    def test_contact_loss_records_existing_query_result_only(self):
        data = run_fixture('contact_loss', True)[2]
        self.assertEqual(data['dynamic_failure_parts'], [dict(part_id='a', displacement=0.,
            displacement_threshold=.2, threshold_exceeded=False, lost_initial_contacts=['f'])])
        self.assertEqual(data['contact_query_observation_scope'], 'check_fallen_parts_only')

    def test_grouped_return_is_not_all_failed_parts(self):
        result, _, data = run_fixture('grouped', True)
        self.assertEqual(result, (False, ['b']))
        self.assertEqual([x['part_id'] for x in data['dynamic_failure_parts']], ['a', 'b'])
        self.assertEqual(data['returned_parts'], ['b'])

    def test_timeout_not_reported_as_contact_or_displacement_failure(self):
        data = run_fixture('timeout', True)[2]
        self.assertEqual(data['physics_steps'], 1)
        self.assertIsNone(data['returned_parts'])
        self.assertEqual(data['dynamic_failure_parts'], [])

    def test_exception_restores_original_class_and_preserves_exception(self):
        module, planner = upstream_fixture('exception')
        original = module.StabilityChecker
        with self.assertRaisesRegex(RuntimeError, 'synthetic_simulator_failure'):
            with observe_stability(module) as collector:
                planner.check_success(max_step=40)
        self.assertIs(module.StabilityChecker, original)
        self.assertEqual(collector.data['phase'], 'error')
        self.assertFalse(collector.data['check_success_returned'])

    def test_observation_failure_does_not_change_upstream_verdict(self):
        module, planner = upstream_fixture('stable')
        # A graph diagnostic property fails while verifier graph methods still work.
        with patch.object(Graph, 'edges', property(lambda self: (_ for _ in ()).throw(ValueError()))):
            with observe_stability(module) as collector:
                result = planner.check_success(max_step=20)
                collector.returned(*result)
        self.assertEqual(result, (True, None))
        self.assertEqual(collector.data['phase'], 'unverified_return')
        self.assertEqual(collector.data['instrumentation_errors'], ['ValueError'])

    def test_nested_observer_rejected_and_outer_restored(self):
        module, _ = upstream_fixture('stable')
        original = module.StabilityChecker
        with observe_stability(module):
            with self.assertRaisesRegex(ValueError, 'nested_stability_observer'):
                with observe_stability(module):
                    pass
        self.assertIs(module.StabilityChecker, original)

    def test_projection_absent_legacy_and_arbitrary_text_omission(self):
        self.assertIsNone(project_stability_diagnostics(None))
        data = run_fixture('contact_loss', True)[2]
        data['model_invented_blocker'] = 'not a physical observation'
        projected = project_stability_diagnostics(data)
        self.assertNotIn('model_invented_blocker', projected)
        self.assertNotIn('initial_contact_graph', projected)

    def test_projection_rejects_type_nonfinite_and_fake_branch_errors(self):
        data = run_fixture('contact_loss', True)[2]
        mutations = [lambda x: x.update(physics_steps=True),
            lambda x: x.update(extra_geometry_queries=1),
            lambda x: x.update(phase='COLLISION'),
            lambda x: x['dynamic_failure_parts'][0].update(displacement=float('nan')),
            lambda x: x['dynamic_failure_parts'][0].update(lost_initial_contacts=[]),
            lambda x: x.update(returned_parts=['unobserved']),
            lambda x: x.update(returned_success=True)]
        for mutate in mutations:
            broken = deepcopy(data)
            mutate(broken)
            with self.assertRaises(ValueError):
                project_stability_diagnostics(broken)

    def test_planner_feedback_only_adds_existing_diagnostics(self):
        requests = []
        data = run_fixture('contact_loss', True)[2]
        def verifier(context, choice, output):
            result = result_for(context, choice, feasible=False)
            result.update(path_success=True, stability_success=False,
                          stability_diagnostics=deepcopy(data))
            return result
        planner = make_planner(verifier=verifier,
            request_enricher=lambda request: requests.append(deepcopy(request)) or request)
        planner.plan(budget=2)
        self.assertEqual(requests[0]['episode_feedback'], [])
        self.assertEqual(requests[1]['episode_feedback'][0]['stability_diagnostics'],
                         project_stability_diagnostics(data))

    def test_provider_v5_label_preserved_as_selection_metadata(self):
        from explicit_action_planner import _selection_feedback
        response = {'metadata': {'provider_contract': 'explicit_action_provider_v5'}}
        self.assertEqual(_selection_feedback(response, {'attempted_action_ids': []})[
            'provider_contract'], 'explicit_action_provider_v5')

    def test_python38_syntax(self):
        for file in ('worker/stability_diagnostics.py', 'worker/explicit_action_physics.py',
                     'worker/explicit_action_planner.py'):
            ast.parse((ROOT / file).read_text(), feature_version=(3, 8))


class StabilityExecutorOptInTests(unittest.TestCase):
    def setUp(self):
        self.fixture = physics_fixture.ExplicitActionPhysicsTests(methodName='test_failed_path_does_not_try_another_direction_or_stability')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_failed_path_leaves_diagnostics_null_and_records_observer_hash(self):
        fixture = self.fixture
        fixture.context['stability_diagnostics'] = True
        fixture.runtime.path_success = False
        result = fixture.run_action()
        self.assertIsNone(result['stability_diagnostics'])
        self.assertEqual(result['stability_check_calls'], 0)
        self.assertIn('stability_observer_sha256', result['source_hashes'])

    def test_default_keeps_legacy_fields_absent(self):
        result = self.fixture.run_action()
        self.assertNotIn('stability_diagnostics', result)
        self.assertNotIn('stability_observer_sha256', result['source_hashes'])

    def test_context_flag_rejects_truthy_string_before_runtime(self):
        self.fixture.context['stability_diagnostics'] = 'yes'
        with self.assertRaisesRegex(ValueError, 'stability_diagnostics_must_be_boolean'):
            self.fixture.run_action()
        self.fixture.load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
