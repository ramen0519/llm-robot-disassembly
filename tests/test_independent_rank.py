"""Independent worker contract fixtures: API calls 0; no actual ASAP physics.

FakeGenerator/FakePlanner below exercise registration and observer contracts.
Their result dictionaries must never be cited as RedMax feasibility evidence.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker'))

from independent_features import describe_loaded
from independent_rank import (IndependentSession, enumerate_candidates,
                              installed_independent, validate_complete_request)
from rank_contract import RankRequest, RankResponse


class MeshFixture:
    """Fixed test observations, not a loaded official CAD model."""
    bounds = [[0., 0., 0.], [1., 2., 3.]]
    center_mass = [.5, 1., 1.5]
    volume = 6.
    is_watertight = True
    is_winding_consistent = True


def feature_fixture():
    return describe_loaded(
        {part: {'mesh': MeshFixture(), 'transform': 'final'}
         for part in ('base', 'a', 'b', 'c')},
        {'fixture.obj': 'a' * 64}, 'test_fixture_only', 'c' * 40, 'base',
        {'seed': 0, 'budget': 20, 'max_grippers': 2, 'optimizer': 'L-BFGS-B'})


class MemoryLog:
    """In-memory event recorder; does not write experiment evidence."""
    def __init__(self):
        self.run_id = 'fixture-independent'
        self.count = 0
        self.events = []

    def emit(self, kind, **fields):
        self.count += 1
        self.events.append(json.loads(json.dumps(dict(
            run_id=self.run_id, event_id=self.run_id + ':e' + str(self.count),
            kind=kind, **fields), allow_nan=False)))

    def kind(self, kind):
        return [event for event in self.events if event['kind'] == kind]


class TestRankClient:
    """Deterministic local fixture; never contacts a provider or file service."""
    def __init__(self, mode='reverse'):
        self.mode = mode
        self.features = {}
        self.requests = []
        self.observations = []

    def __call__(self, request):
        self.requests.append(asdict(request))
        self.observations.append(deepcopy(self.features))
        if self.mode == 'error':
            raise RuntimeError('fixture error')
        order = tuple(reversed(request.candidate_ids))
        if self.mode == 'ascending':
            order = request.candidate_ids
        elif self.mode == 'missing':
            order = order[:-1]
        elif self.mode == 'duplicate':
            order = (order[0],) * len(order)
        elif self.mode == 'invented':
            order = ('not-a-part',) + order[1:]
        rid = 'old-request' if self.mode == 'stale' else request.request_id
        return RankResponse(rid, request.state_id, order)


def fixture_classes():
    class FakeGenerator:
        def __init__(self, asset_folder, assembly_dir, base_part=None, save_sdf=False):
            self.asset_folder = asset_folder
            self.assembly_dir = assembly_dir
            self.base_part = base_part
            self.save_sdf = save_sdf

    class FakeLegacyGenerator(FakeGenerator):
        """Any successful execution of this legacy scorer is a test failure."""
        native_calls = 0

        def generate_candidate_part(self, assembled_parts):
            type(self).native_calls += 1
            raise AssertionError('legacy generator was executed')

        def heuristic_func(self, part):
            type(self).native_calls += 1
            raise AssertionError('legacy score was executed')

    class FakePlanner:
        def __init__(self):
            self.raw = {'feasible': False, 'action': None, 'parts_fix': None,
                        'fixture_only': True}
            self.error = None
            self.received = None
            self.calls = 0

        def _simulate(self, part_move, parts_rest, parts_removed, pose, max_grippers,
                      timeout=None, grasp_planner=None, optimizer='L-BFGS-B',
                      debug=0, render=False):
            self.calls += 1
            self.received = dict(part_move=part_move, parts_rest=parts_rest,
                parts_removed=parts_removed, pose=pose, max_grippers=max_grippers,
                timeout=timeout, grasp_planner=grasp_planner, optimizer=optimizer,
                debug=debug, render=render)
            if self.error is not None:
                raise self.error
            return self.raw

    return FakeGenerator, FakeLegacyGenerator, FakePlanner


class EnumerationTests(unittest.TestCase):
    def test_complete_set_and_canonical_order_ignore_input_order(self):
        parts = ['c', 'base', 'a', 'b']
        before = parts.copy()
        self.assertEqual(enumerate_candidates(parts, 'base'), ('a', 'b', 'c'))
        self.assertEqual(enumerate_candidates(reversed(parts), 'base'), ('a', 'b', 'c'))
        self.assertEqual(parts, before)

    def test_no_base_means_every_part_is_retained(self):
        self.assertEqual(enumerate_candidates(['c', 'a', 'b']), ('a', 'b', 'c'))

    def test_empty_and_base_only_have_no_candidates(self):
        self.assertEqual(enumerate_candidates([], None), ())
        self.assertEqual(enumerate_candidates(['base'], 'base'), ())

    def test_numeric_looking_ids_are_strings_not_integer_positions(self):
        self.assertEqual(enumerate_candidates(['2', '10', '0']), ('0', '10', '2'))

    def test_invalid_state_or_base_is_rejected(self):
        for parts, base in ((['a', 'a'], None), (['a', ''], None),
                            (['a', 2], None), (['a'], 'absent'),
                            ('12', None), (b'12', None), (None, None)):
            with self.subTest(parts=parts, base=base):
                with self.assertRaises((ValueError, TypeError)):
                    enumerate_candidates(parts, base)

    def test_request_rejects_missing_candidate_even_when_subset_is_valid(self):
        request = RankRequest('run', 'r1', 'state', ('a', 'b', 'base'), ('a',), 'base')
        with self.assertRaises(ValueError):
            validate_complete_request(request)

    def test_request_requires_canonical_candidates_and_remaining_parts(self):
        for remaining, candidates in ((('a', 'b', 'base'), ('b', 'a')),
                                      (('base', 'b', 'a'), ('a', 'b'))):
            with self.subTest(remaining=remaining, candidates=candidates):
                with self.assertRaises(ValueError):
                    validate_complete_request(RankRequest('run', 'r1', 's',
                        remaining, candidates, 'base'))


class IndependentSessionTests(unittest.TestCase):
    def setUp(self):
        self.log = MemoryLog()
        self.client = TestRankClient()
        self.features = feature_fixture()
        self.session = IndependentSession(self.log, self.client, self.features,
            lambda: repr(random.getstate()))

    def test_empty_memory_does_not_prevent_direct_ranking(self):
        remaining = ['c', 'base', 'b', 'a']
        before = remaining.copy()
        self.assertEqual(list(self.session.generate(remaining, 'base')), ['c', 'b', 'a'])
        self.assertEqual(remaining, before)
        observed = self.client.observations[0]
        self.assertEqual(observed['episode_feedback'], [])
        self.assertEqual(observed['validated_repair_skills'], [])
        self.assertIsNone(self.log.kind('rank_decision')[0]['fallback_reason'])
        self.assertEqual(self.session.tripwire_calls, 0)

    def test_different_test_rankers_change_order_preserve_full_candidate_set(self):
        ascending = TestRankClient('ascending')
        other = IndependentSession(MemoryLog(), ascending, self.features, lambda: 'rng')
        reverse_order = list(self.session.generate(['a', 'b', 'c', 'base'], 'base'))
        ascending_order = list(other.generate(['a', 'b', 'c', 'base'], 'base'))
        self.assertNotEqual(reverse_order, ascending_order)
        self.assertEqual(set(reverse_order), set(ascending_order))
        self.assertEqual(set(ascending_order), {'a', 'b', 'c'})

    def test_invalid_responses_use_complete_canonical_fallback(self):
        for mode in ('missing', 'duplicate', 'invented', 'stale', 'error'):
            with self.subTest(mode=mode):
                log = MemoryLog()
                session = IndependentSession(log, TestRankClient(mode), self.features, lambda: 'rng')
                self.assertEqual(list(session.generate(['base', 'c', 'a', 'b'], 'base')),
                                 ['a', 'b', 'c'])
                decision = log.kind('rank_decision')[0]
                self.assertTrue(decision['fallback_reason'])
                self.assertEqual(decision['fallback_policy'], 'canonical_id_order')

    def test_remaining_geometry_is_refreshed_without_mutating_initial_features(self):
        original = deepcopy(self.features)
        list(self.session.generate(['base', 'a', 'b', 'c'], 'base'))
        list(self.session.generate(['base', 'b', 'c'], 'base'))
        self.assertEqual(set(self.client.observations[0]['cad']['parts']), {'base', 'a', 'b', 'c'})
        self.assertEqual(set(self.client.observations[1]['cad']['parts']), {'base', 'b', 'c'})
        self.assertEqual(self.features, original)
        self.assertEqual(self.session.features, original)
        self.assertEqual(self.client.requests[1]['candidate_ids'], ('b', 'c'))

    def test_generator_does_not_consume_global_python_rng(self):
        before = random.getstate()
        list(self.session.generate(['base', 'a', 'b', 'c'], 'base'))
        self.assertEqual(random.getstate(), before)
        event = self.log.kind('generation')[0]
        self.assertEqual(event['rng_before'], event['rng_after'])
        self.assertFalse(event['heuristic_or_gnn_invoked'])

    def test_mismatched_task_base_is_rejected_before_ranker(self):
        with self.assertRaises(ValueError):
            list(self.session.generate(['base', 'a', 'b', 'c'], None))
        self.assertEqual(self.client.requests, [])


class InstalledIndependentTests(unittest.TestCase):
    def setUp(self):
        self.base, self.legacy, self.planner_cls = fixture_classes()
        self.registry = {'heur-out': self.legacy, 'learn': self.legacy}
        self.log = MemoryLog()
        self.client = TestRankClient()
        self.session = IndependentSession(self.log, self.client, feature_fixture(), lambda: 'rng')
        self.original_simulate = self.planner_cls._simulate
        self.original_legacy_dict = dict(self.legacy.__dict__)

    def installed(self):
        return installed_independent(self.base, self.registry, self.planner_cls, self.session)

    def assert_restored(self):
        self.assertEqual(set(self.registry), {'heur-out', 'learn'})
        self.assertIs(self.registry['heur-out'], self.legacy)
        self.assertIs(self.planner_cls._simulate, self.original_simulate)
        self.assertEqual(dict(self.legacy.__dict__), self.original_legacy_dict)

    def test_separate_generator_in_registry_never_calls_legacy(self):
        with self.installed() as generator_cls:
            self.assertIs(self.registry['independent'], generator_cls)
            self.assertTrue(issubclass(generator_cls, self.base))
            self.assertFalse(issubclass(generator_cls, self.legacy))
            generator = generator_cls('fixture-assets', 'fixture-assembly', base_part='base')
            self.assertEqual(list(generator.generate_candidate_part(['a', 'b', 'c', 'base'])),
                             ['c', 'b', 'a'])
            self.assertEqual(self.legacy.native_calls, 0)
            self.assertEqual(self.session.tripwire_calls, 0)
        self.assert_restored()

    def test_tripwire_blocks_constructor_generator_and_scoring(self):
        legacy_instance = self.legacy('fixture-assets', 'fixture-assembly')
        with self.installed():
            for action in (lambda: self.legacy('fixture-assets', 'fixture-assembly'),
                           lambda: legacy_instance.generate_candidate_part(['a']),
                           lambda: legacy_instance.heuristic_func('a')):
                with self.assertRaisesRegex(RuntimeError, 'heuristic_or_gnn'):
                    action()
            self.assertEqual(self.session.tripwire_calls, 3)
            self.assertEqual(self.legacy.native_calls, 0)
        self.assert_restored()

    def test_registry_and_owned_inherited_attributes_restore_on_exception(self):
        self.assertNotIn('__init__', self.legacy.__dict__)
        with self.assertRaisesRegex(RuntimeError, 'fixture body failure'):
            with self.installed():
                raise RuntimeError('fixture body failure')
        self.assert_restored()

    def test_existing_independent_registration_is_not_overwritten(self):
        marker = object()
        self.registry['independent'] = marker
        with self.assertRaises(ValueError):
            with self.installed():
                self.fail('conflicting registration should reject before body')
        self.assertIs(self.registry['independent'], marker)
        self.assertIs(self.planner_cls._simulate, self.original_simulate)

    def test_observer_preserves_actual_argument_objects_and_raw_return_identity(self):
        planner = self.planner_cls()
        rest, removed, pose = ['a', 'b', 'base'], [], [[1., 0.], [0., 1.]]
        grasp = object()
        before = deepcopy((rest, removed, pose, planner.raw))
        with self.installed():
            list(self.session.generate(['a', 'b', 'c', 'base'], 'base'))
            result = planner._simulate('c', rest, removed, pose, 2,
                timeout=3.5, grasp_planner=grasp, optimizer='fixture-opt', debug=4, render=True)
            self.assertIs(result, planner.raw)
            self.assertIs(planner.received['parts_rest'], rest)
            self.assertIs(planner.received['parts_removed'], removed)
            self.assertIs(planner.received['pose'], pose)
            self.assertIs(planner.received['grasp_planner'], grasp)
            self.assertEqual(planner.received['timeout'], 3.5)
            self.assertEqual(planner.received['optimizer'], 'fixture-opt')
            self.assertEqual(planner.received['debug'], 4)
            self.assertIs(planner.received['render'], True)
        self.assertEqual((rest, removed, pose, planner.raw), before)
        self.assertEqual(planner.calls, 1)
        self.assertEqual(self.session.verifier_count, 1)
        self.assertEqual(self.log.kind('verifier_start')[0]['generation_id'], 'g1')
        result_event = self.log.kind('verifier_result')[0]
        self.assertEqual(result_event['status'], 'UNKNOWN')
        self.assertIsNone(result_event['blocker'])
        self.assertIsNone(result_event['path'])
        planner.raw['action'] = [1, 0, 0]
        self.assertIsNone(result_event['raw']['action'])
        self.assert_restored()

    def test_current_episode_feedback_reaches_next_ranking_without_future_leak(self):
        planner = self.planner_cls()
        planner.raw.update(feasible=True, action=[1, 0, 0], parts_fix=['base'])
        with self.installed():
            list(self.session.generate(['a', 'b', 'c', 'base'], 'base'))
            self.assertEqual(self.client.observations[0]['episode_feedback'], [])
            planner._simulate('c', ['a', 'b', 'base'], [], None, 2)
            list(self.session.generate(['a', 'b', 'base'], 'base'))
        feedback = self.client.observations[1]['episode_feedback']
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]['status'], 'PASS')
        self.assertEqual(feedback[0]['request']['part_move'], 'c')
        self.assertTrue(feedback[0]['raw']['fixture_only'])
        self.assertEqual(self.client.observations[0]['episode_feedback'], [])

    def test_exception_is_logged_and_rethrown_without_physics_verdict(self):
        planner = self.planner_cls()
        planner.error = RuntimeError('fixture solver exception')
        with self.installed():
            with self.assertRaisesRegex(RuntimeError, 'fixture solver exception'):
                planner._simulate('c', ['a', 'b', 'base'], [], None, 2)
        event = self.log.kind('verifier_result')[0]
        self.assertEqual(event['status'], 'ERROR')
        self.assertEqual(event['error_type'], 'RuntimeError')
        self.assertIsNone(event['raw'])
        self.assertIsNone(event['blocker'])
        self.assertIsNone(event['path'])
        self.assertEqual(self.session.feedback[0]['status'], 'ERROR')
        self.assert_restored()

    def test_upstream_terminal_pair_observation_does_not_create_rank_request(self):
        planner = self.planner_cls()
        with self.installed():
            list(self.session.generate(['a', 'b', 'base'], 'base'))
            count = len(self.client.requests)
            planner._simulate('a', ['base'], ['b', 'c'], None, 2)
            self.assertEqual(len(self.client.requests), count)
        event = self.log.kind('verifier_start')[0]
        self.assertIsNone(event['generation_id'])
        self.assertEqual(event['selection_scope'], 'upstream_terminal_completion')
        self.assertEqual(self.session.generation_count, 1)


if __name__ == '__main__':
    unittest.main()
