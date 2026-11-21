"""Offline cache-contract tests using fake planners; no RedMax or physics runs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.path_scene_reuse import PathSceneSession, SUPPORTED_BACKEND


class FakeSimulation:
    """Model only reset-visible state; this does not validate native RedMax."""

    def __init__(self):
        self.options = SimpleNamespace(h=0.005)
        self.initial_q = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        self.initial_qdot = [0.0] * 6
        self.reset()

    def reset(self):
        self.q = list(self.initial_q)
        self.qdot = list(self.initial_qdot)


class FakePlanner:
    def __init__(self):
        self.sim = FakeSimulation()
        self.parts_fix = ['0', '2']
        self.part_move = '1'
        self.parts_removed = []
        self.pose = [[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]]
        self.force_mag = 100.0
        self.frame_skip = 100
        self.min_sep = 0.5
        self.contact_eps = 0.1
        self.col_th_base = 0.01
        self._evidence_col_th = {'0': 0.01, '1': 0.01, '2': 0.01}
        self.max_time = 60.0

    def get_state(self):
        return SimpleNamespace(q=list(self.sim.q), qdot=list(self.sim.qdot))


class PathSceneReuseTests(unittest.TestCase):
    def setUp(self):
        self.session = PathSceneSession()
        self.addCleanup(self.session.close)
        self.key = dict(source_hashes=deepcopy(SUPPORTED_BACKEND),
                        assets={'1.obj': 'mesh-a', '1.sdf': 'sdf-a'},
                        part_move='1', parts_fix=['0', '2'],
                        pose=[[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0], [0, 0, 0, 1]])
        self.created = []
        self.cache_io = [{'builder': 'fake_path', 'xml_load_sdf': True,
                          'xml_save_sdf': False, 'readonly_xml_sha256': 'xml-a'}]

    def factory(self):
        planner = FakePlanner()
        self.created.append(planner)
        return planner, deepcopy(self.cache_io)

    def acquire(self, key=None, factory=None, rng=None, session=None):
        return (session or self.session).acquire(
            self.key if key is None else key,
            self.factory if factory is None else factory,
            (lambda: 'stable-rng') if rng is None else rng)

    def reset_and_finish(self, lease, reusable=True):
        lease.planner.sim.reset()
        lease.after_reset(lease.planner)
        lease.finish(reusable)

    def test_repeated_attempt_reuses_scene_but_resets_q_and_qdot(self):
        first = self.acquire()
        first.planner.sim.reset()
        first.after_reset(first.planner)
        first.planner.sim.q[0] = 5.0
        first.planner.sim.qdot[1] = 8.0
        first.finish(True)
        second = self.acquire()
        self.assertIs(second.planner, first.planner)
        self.assertTrue(second.audit['cache_hit'])
        self.assertEqual(second.audit['constructor_calls'], 0)
        self.assertFalse(second.audit['outcome_reused'])
        self.assertFalse(second.audit['stability_scene_reused'])
        self.reset_and_finish(second)
        self.assertEqual(len(self.created), 1)
        self.assertEqual(self.session.summary()['reset_checks'], 2)

    def test_max_time_restored_from_native_limit_on_each_acquire(self):
        first = self.acquire()
        first.planner.max_time = 0.25
        self.reset_and_finish(first)
        second = self.acquire()
        self.assertEqual(second.planner.max_time, 60.0)
        self.assertEqual(second.audit['native_path_timeout_seconds'], 60.0)
        self.reset_and_finish(second)

    def test_q_reset_must_be_exact(self):
        lease = self.acquire()
        lease.planner.sim.reset()
        lease.planner.sim.q[0] += 1e-12
        with self.assertRaisesRegex(ValueError, 'scene_reset_did_not_restore'):
            lease.after_reset(lease.planner)
        lease.finish(False)
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_qdot_reset_must_be_exact(self):
        lease = self.acquire()
        lease.planner.sim.reset()
        lease.planner.sim.qdot[0] = 1e-12
        with self.assertRaisesRegex(ValueError, 'scene_reset_did_not_restore'):
            lease.after_reset(lease.planner)
        lease.finish(False)

    def test_unobserved_reset_cannot_retain_scene(self):
        lease = self.acquire()
        lease.finish(True)
        self.assertFalse(lease.audit['retained_for_next_attempt'])
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_explicit_discard_builds_fresh_next_time(self):
        first = self.acquire()
        self.reset_and_finish(first, False)
        second = self.acquire()
        self.assertFalse(second.audit['cache_hit'])
        self.assertIsNot(second.planner, first.planner)
        self.reset_and_finish(second)

    def test_fingerprint_changes_evict_one_entry(self):
        first = self.acquire()
        self.reset_and_finish(first)
        changed = deepcopy(self.key)
        changed['assets']['1.obj'] = 'mesh-b'
        second = self.acquire(changed)
        self.assertFalse(second.audit['cache_hit'])
        self.assertNotEqual(first.audit['key_sha256'], second.audit['key_sha256'])
        self.reset_and_finish(second)
        third = self.acquire()
        self.assertFalse(third.audit['cache_hit'])
        self.reset_and_finish(third)
        self.assertEqual(len(self.created), 3)
        self.assertEqual(self.session.summary()['evictions'], 2)
        self.assertEqual(self.session.summary()['cached_entries'], 1)

    def test_each_constructor_input_change_misses(self):
        variants = []
        for name, value in [('part_move', '2'), ('parts_fix', ['2', '0'])]:
            key = deepcopy(self.key)
            key[name] = value
            variants.append(key)
        pose_key = deepcopy(self.key)
        pose_key['pose'][0][3] = 0.01
        variants.append(pose_key)
        source_key = deepcopy(self.key)
        source_key['source_hashes']['controller_sha256'] = 'controller-b'
        variants.append(source_key)
        for key in [self.key] + variants:
            lease = self.acquire(key)
            self.assertFalse(lease.audit['cache_hit'])
            self.reset_and_finish(lease)

    def test_dictionary_order_does_not_prevent_hit(self):
        first = self.acquire()
        self.reset_and_finish(first)
        reordered = dict(reversed(list(self.key.items())))
        second = self.acquire(reordered)
        self.assertTrue(second.audit['cache_hit'])
        self.reset_and_finish(second)

    def test_no_cross_session_reuse(self):
        first = self.acquire()
        self.reset_and_finish(first)
        other = PathSceneSession()
        self.addCleanup(other.close)
        second = self.acquire(session=other)
        self.assertIsNot(first.planner, second.planner)
        self.assertFalse(second.audit['cache_hit'])
        self.reset_and_finish(second)

    def test_stochastic_constructor_is_never_cached_or_rewound(self):
        rng_state = {'value': 0}

        def stochastic_factory():
            rng_state['value'] += 1
            return self.factory()

        for expected in (1, 2):
            lease = self.acquire(factory=stochastic_factory,
                                 rng=lambda: str(rng_state['value']))
            self.assertFalse(lease.audit['cache_hit'])
            self.assertFalse(lease.audit['constructor_rng_unchanged'])
            self.assertFalse(lease.audit['global_rng_rewound'])
            self.reset_and_finish(lease)
            self.assertFalse(lease.audit['retained_for_next_attempt'])
            self.assertEqual(rng_state['value'], expected)
        self.assertEqual(self.session.summary()['stochastic_builds'], 2)
        self.assertEqual(self.session.summary()['hits'], 0)

    def test_backend_change_rejects_and_discards_existing_scene(self):
        first = self.acquire()
        self.reset_and_finish(first)
        bad = deepcopy(self.key)
        bad['source_hashes']['redmax_binary_sha256'] = 'unvalidated-binary'
        with self.assertRaisesRegex(ValueError, 'native_backend_not_validated'):
            self.acquire(bad)
        self.assertEqual(self.session.summary()['cached_entries'], 0)
        self.assertFalse(self.session.summary()['busy'])
        fresh = self.acquire()
        self.assertFalse(fresh.audit['cache_hit'])
        self.reset_and_finish(fresh)

    def test_mutated_retained_configuration_rejected(self):
        first = self.acquire()
        self.reset_and_finish(first)
        first.planner.force_mag = 1.0
        with self.assertRaisesRegex(ValueError, 'cached_scene_mutated'):
            self.acquire()
        self.assertEqual(self.session.summary()['cached_entries'], 0)
        self.assertFalse(self.session.summary()['busy'])

    def test_configuration_change_at_release_is_an_error(self):
        lease = self.acquire()
        lease.planner.sim.reset()
        lease.after_reset(lease.planner)
        lease.planner.sim.options.h = 0.01
        with self.assertRaises(ValueError):
            lease.finish(True)
        self.assertFalse(lease.audit['retained_for_next_attempt'])
        self.assertEqual(self.session.summary()['cached_entries'], 0)
        self.assertFalse(self.session.summary()['busy'])

    def test_configuration_change_at_reset_rejected(self):
        lease = self.acquire()
        lease.planner.sim.reset()
        lease.planner._evidence_col_th['0'] = 0.2
        with self.assertRaisesRegex(ValueError, 'physics_configuration_changed'):
            lease.after_reset(lease.planner)
        # Restore only the fake fixture to permit ordinary error cleanup.
        lease.planner._evidence_col_th['0'] = 0.01
        lease.finish(False)

    def test_simulation_identity_change_at_acquire_rejected(self):
        first = self.acquire()
        self.reset_and_finish(first)
        first.planner.sim = FakeSimulation()
        with self.assertRaisesRegex(ValueError, 'cached_scene_mutated'):
            self.acquire()
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_simulation_identity_change_at_release_is_an_error(self):
        lease = self.acquire()
        lease.planner.sim.reset()
        lease.after_reset(lease.planner)
        lease.planner.sim = FakeSimulation()
        with self.assertRaises(ValueError):
            lease.finish(True)
        self.assertFalse(self.session.summary()['busy'])
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_factory_exception_drops_old_entry_and_releases_busy(self):
        first = self.acquire()
        self.reset_and_finish(first)
        changed = deepcopy(self.key)
        changed['assets']['1.sdf'] = 'sdf-b'

        def broken_factory():
            raise RuntimeError('fake constructor failed')

        with self.assertRaisesRegex(RuntimeError, 'fake constructor failed'):
            self.acquire(changed, factory=broken_factory)
        self.assertEqual(self.session.summary()['cached_entries'], 0)
        self.assertFalse(self.session.summary()['busy'])
        fresh = self.acquire()
        self.assertFalse(fresh.audit['cache_hit'])
        self.reset_and_finish(fresh)

    def test_busy_session_rejects_overlapping_acquire(self):
        lease = self.acquire()
        with self.assertRaisesRegex(ValueError, 'closed_or_busy'):
            self.acquire()
        self.assertEqual(len(self.created), 1)
        self.reset_and_finish(lease)
        next_lease = self.acquire()
        self.assertTrue(next_lease.audit['cache_hit'])
        self.reset_and_finish(next_lease)

    def test_close_discards_and_prevents_future_acquire(self):
        lease = self.acquire()
        self.reset_and_finish(lease)
        self.session.close()
        self.session.close()
        self.assertTrue(self.session.summary()['closed'])
        self.assertEqual(self.session.summary()['cached_entries'], 0)
        with self.assertRaisesRegex(ValueError, 'closed_or_busy'):
            self.acquire()

    def test_close_invalidates_an_outstanding_reset_observer(self):
        lease = self.acquire()
        self.session.close()
        lease.planner.sim.reset()
        with self.assertRaisesRegex(ValueError, 'invalid_scene_reset_observation'):
            lease.after_reset(lease.planner)
        self.assertFalse(lease.audit['reset_state_verified'])
        lease.finish(False)
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_one_entry_configuration_is_strict(self):
        for value in (0, 2, True, 1.0, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'requires_one_entry'):
                    PathSceneSession(max_entries=value)

    def test_wrong_planner_or_duplicate_reset_observation_rejected(self):
        lease = self.acquire()
        with self.assertRaisesRegex(ValueError, 'invalid_scene_reset_observation'):
            lease.after_reset(FakePlanner())
        lease.planner.sim.reset()
        lease.after_reset(lease.planner)
        with self.assertRaisesRegex(ValueError, 'invalid_scene_reset_observation'):
            lease.after_reset(lease.planner)
        lease.finish(False)
        with self.assertRaisesRegex(ValueError, 'invalid_scene_reset_observation'):
            lease.after_reset(lease.planner)

    def test_finish_is_idempotent(self):
        lease = self.acquire()
        self.reset_and_finish(lease, False)
        before = self.session.summary()
        lease.finish(True)
        self.assertEqual(self.session.summary(), before)

    def test_audit_payloads_do_not_mutate_retained_construction_evidence(self):
        first = self.acquire()
        first.audit['initial_state']['q'][0] = 999
        first.audit['construction_cache_io'][0]['readonly_xml_sha256'] = 'tampered'
        self.reset_and_finish(first)
        second = self.acquire()
        self.assertEqual(second.audit['initial_state']['q'][0], 0.1)
        self.assertEqual(second.audit['construction_cache_io'][0]['readonly_xml_sha256'], 'xml-a')
        self.assertEqual(second.audit['construction_evidence_scope'],
                         'original_construction_not_current_disk_load')
        self.reset_and_finish(second)

    def test_invalid_native_limit_never_enters_cache(self):
        for value in (0, -1, float('inf'), float('nan')):
            with self.subTest(value=value):
                def factory():
                    planner, audit = self.factory()
                    planner.max_time = value
                    return planner, audit
                with self.assertRaisesRegex(ValueError, 'invalid_native_path_time_limit'):
                    self.acquire(factory=factory)
                self.assertEqual(self.session.summary()['cached_entries'], 0)
                self.assertFalse(self.session.summary()['busy'])

    def test_invalid_initial_state_never_enters_cache(self):
        for field, value in [('q', [float('nan')] * 6),
                             ('qdot', [True] * 6), ('q', [0.0] * 4)]:
            with self.subTest(field=field, value=value):
                def factory():
                    planner, audit = self.factory()
                    setattr(planner.sim, field, value)
                    return planner, audit
                with self.assertRaisesRegex(ValueError, 'scene_initial_state_invalid'):
                    self.acquire(factory=factory)
                self.assertEqual(self.session.summary()['cached_entries'], 0)
                self.assertFalse(self.session.summary()['busy'])


if __name__ == '__main__':
    unittest.main()
