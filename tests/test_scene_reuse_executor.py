"""Real executor/session integration over an injected synthetic physics runtime.

Backend fingerprints and native source discovery are explicitly patched here.
These tests are contracts, NOT evidence of actual RedMax physics equivalence.
"""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
import explicit_action_physics as executor
import path_scene_reuse as reuse
from task_memory import task_context as build_task_context
from tests.test_explicit_action_physics import FixtureRuntime, IDENTITY
from tests.test_task_memory import features


class ReusableFixtureRuntime(FixtureRuntime):
    def __init__(self, source):
        super().__init__(source)
        owner = self
        base = self.physics.MultiPartPathPlanner
        self.scenes = []
        self.control_limits = []
        self.reset_bad = False
        self.mutate_configuration = False
        self.write_source = False

        class PathPlanner(base):
            contact_eps, col_th_base = 0.01, 0.01

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.qdot = [0.] * 6
                owner.scenes.append(self)

            def get_state(self):
                return SimpleNamespace(q=self.q[:], qdot=self.qdot[:])

        self.physics.MultiPartPathPlanner = PathPlanner
        self.execute_single = lambda planner, first, **kwargs: self.reuse_control(planner, first, **kwargs)
        self.execute_two_leg = lambda planner, first, second, step, **kwargs: self.reuse_control(
            planner, first, second, step, **kwargs)

    def reuse_control(self, planner, first, second=None, step=None, reset_observer=None):
        self.control_limits.append(planner.max_time)
        planner.q = [0.] * 6
        planner.qdot = [0.] * 6
        if self.reset_bad:
            planner.qdot[0] = 1.
        if reset_observer is not None:
            reset_observer(planner)
        result = self.control(planner, first, second, step)
        planner.qdot = [0.1] * 6  # Dirty previous state must be reset next time.
        if self.mutate_configuration:
            planner.force_mag = 999.
        if self.write_source:
            next(iter(self.source_files.values())).write_text('changed fixture source', encoding='utf-8')
        return result


class SceneReuseExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='scene_executor_OFFLINE_', dir=ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.root = self.home / 'asap'; self.root.mkdir(); (self.root / 'assets').mkdir()
        self.assembly = self.home / 'assembly'; self.assembly.mkdir()
        for part in ('0', '1', '2', '3'):
            (self.assembly / (part + '.obj')).write_text('synthetic fixture only', encoding='utf-8')
        self.source = self.root / 'fixture_source.py'
        self.source.write_text('fixture only', encoding='utf-8')
        self.runtime = ReusableFixtureRuntime(self.source)
        self.context = dict(asap_root=str(self.root), assembly_dir=str(self.assembly),
            remaining_parts=['0', '1', '2', '3'], base_part='0', max_grippers=3,
            pose_catalog=[dict(pose_id='p0', matrix=deepcopy(IDENTITY))], seed=7, budget=20,
            stability_timeout=2., physics_scene_policy='path_scene_reuse_v1')
        self.choice = dict(part_id='1', pose_id='p0', skill_id='single_force_v1', direction_id='yp',
                           second_direction_id=None, support_ids=[], evidence_ids=[])
        self.session = reuse.PathSceneSession()
        self.addCleanup(self.session.close)
        self.index = 0
        for guard in (
            patch.object(executor, '_load_runtime', return_value=self.runtime),
            patch.object(executor, '_reuse_source_files', return_value={}),
            patch.object(reuse, 'validate_backend', return_value=None),
            patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK')),
            patch('subprocess.run', side_effect=AssertionError('NO_SUBPROCESS')),
        ):
            guard.start(); self.addCleanup(guard.stop)

    def action(self, context=None, choice=None, fresh=False):
        self.index += 1
        context = deepcopy(self.context if context is None else context)
        choice = deepcopy(self.choice if choice is None else choice)
        if fresh:
            context.pop('physics_scene_policy', None)
            return executor.execute_action(context, choice, self.home / ('action_' + str(self.index)))
        return executor.execute_action(context, choice, self.home / ('action_' + str(self.index)),
                                       scene_session=self.session)

    def calls(self, kind):
        return [call for call in self.runtime.calls if call[0] == kind]

    def context_with_task_policy(self, policy=None):
        """Build valid provenance; only the policy differs in rejection tests."""
        context = deepcopy(self.context)
        source = features('scene-reuse-fixture')
        source['task'].update(
            input_sha256={item.name: executor._sha(item) for item in self.assembly.iterdir()},
            base_part=context['base_part'])
        source['task']['solver_settings'].update(seed=context['seed'], budget=context['budget'],
                                                max_grippers=context['max_grippers'])
        if policy is not None:
            source['task']['solver_settings']['physics_scene_policy'] = policy
        source['parts'] = {part: {'center_mass': [0, 0, 0]} for part in context['remaining_parts']}
        context['task_context'] = build_task_context(source)
        return context

    def assert_policy_rejected_before_physics(self, result):
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['error'], 'task_context_does_not_match_actual_input_or_run_conditions')
        self.assertEqual(result['path_check_calls'], 0)
        self.assertEqual(result['constraint_record_count'], 0)
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_legacy_fresh_task_provenance_rejects_reuse_execution(self):
        context = self.context_with_task_policy()
        self.assertNotIn('physics_scene_policy', context['task_context']['task']['solver_settings'])
        self.assert_policy_rejected_before_physics(self.action(context=context))

    def test_reuse_task_provenance_rejects_fresh_execution(self):
        context = self.context_with_task_policy('path_scene_reuse_v1')
        self.assert_policy_rejected_before_physics(self.action(context=context, fresh=True))

    def test_matching_reuse_task_provenance_is_accepted(self):
        result = self.action(context=self.context_with_task_policy('path_scene_reuse_v1'))
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['task_context_input_and_run_conditions_checked'])
        self.assertEqual(result['physics_scene_policy'], 'path_scene_reuse_v1')
        self.assertEqual(len(self.calls('controller')), 1)

    def test_matching_legacy_fresh_task_provenance_is_accepted(self):
        result = self.action(context=self.context_with_task_policy(), fresh=True)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['task_context_input_and_run_conditions_checked'])
        self.assertNotIn('physics_scene_policy', result)
        self.assertEqual(len(self.calls('controller')), 1)

    def test_direction_and_support_change_reuse_path_but_always_recheck_physics(self):
        first = self.action()
        choice = dict(self.choice, direction_id='zp', support_ids=['2'])
        second = self.action(choice=choice)
        self.assertEqual([first['status'], second['status']], ['PASS', 'PASS'])
        self.assertEqual(len(self.calls('path_constructor')), 1)
        self.assertEqual(len(self.calls('controller')), 2)
        self.assertEqual(len(self.calls('stability_constructor')), 2)
        self.assertEqual(len(self.calls('stability_check')), 2)
        self.assertEqual(self.calls('stability_constructor')[0][1], ['0'])
        self.assertEqual(self.calls('stability_constructor')[1][1], ['0', '2'])
        self.assertFalse(first['path_scene_reuse']['cache_hit'])
        self.assertTrue(second['path_scene_reuse']['cache_hit'])
        self.assertEqual(second['path_cache_io'], [])
        self.assertEqual(second['path_constructor_seconds'], 0.)
        for result in (first, second):
            self.assertEqual(result['path_check_calls'], 1)
            self.assertEqual(result['stability_check_calls'], 1)
            self.assertTrue(result['path_scene_reuse']['reset_state_verified'])
            self.assertFalse(result['path_scene_reuse']['outcome_reused'])
            self.assertFalse(result['path_scene_reuse']['stability_scene_reused'])
            self.assertTrue(result['path_scene_reuse']['retained_for_next_attempt'])
        self.assertEqual(self.session.summary()['reset_checks'], 2)

    def test_two_leg_control_shares_scene_but_runs_new_controller(self):
        self.action()
        choice = dict(self.choice, skill_id='turn_force_two_leg_v1', direction_id='xp',
                      second_direction_id='zp')
        result = self.action(choice=choice)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['path_scene_reuse']['cache_hit'])
        self.assertEqual(result['two_leg_controller_calls'], 1)
        self.assertEqual(len(self.calls('path_constructor')), 1)
        self.assertEqual(len(self.calls('controller')), 2)

    def test_state_pose_and_mover_changes_build_distinct_scenes(self):
        self.action()
        state = deepcopy(self.context); state['remaining_parts'] = ['0', '1', '2']
        state['removed_parts'] = ['3']
        second = self.action(context=state)
        posed = deepcopy(state); posed['pose_catalog'][0]['matrix'][0][3] = 0.25
        third = self.action(context=posed)
        fourth = self.action(context=posed, choice=dict(self.choice, part_id='2'))
        for result in (second, third, fourth):
            self.assertEqual(result['status'], 'PASS')
            self.assertFalse(result['path_scene_reuse']['cache_hit'])
        self.assertEqual(len(self.calls('path_constructor')), 4)
        self.assertEqual(self.session.summary()['cached_entries'], 1)

    def test_changed_input_or_source_between_calls_is_not_a_cache_hit(self):
        self.action()
        (self.assembly / '3.obj').write_text('different synthetic asset', encoding='utf-8')
        second = self.action()
        self.source.write_text('different synthetic source', encoding='utf-8')
        third = self.action()
        for result in (second, third):
            self.assertEqual(result['status'], 'PASS')
            self.assertFalse(result['path_scene_reuse']['cache_hit'])
        self.assertEqual(len(self.calls('path_constructor')), 3)

    def test_native_timeout_restored_on_hit_before_current_budget_is_applied(self):
        limited = dict(self.context, remaining_wall_seconds=5.)
        first = self.action(context=limited)
        second = self.action()
        self.assertEqual(first['status'], 'PASS'); self.assertEqual(second['status'], 'PASS')
        self.assertLessEqual(self.runtime.control_limits[0], 5.)
        self.assertEqual(self.runtime.control_limits[1], 60.)
        self.assertEqual(second['native_path_timeout_seconds'], 60.)
        self.assertTrue(second['path_scene_reuse']['cache_hit'])

    def test_budget_exhaustion_before_runtime_invalidates_previous_scene(self):
        self.action()
        result = self.action(context=dict(self.context, remaining_wall_seconds=0.))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['reason'], 'action_wall_budget_exhausted_before_runtime')
        self.assertEqual(result['path_check_calls'], 0)
        self.assertEqual(self.session.summary()['cached_entries'], 0)
        self.assertEqual(len(self.calls('controller')), 1)

    def test_budget_exhaustion_after_acquire_invalidates_scene_without_control(self):
        self.action()
        # First _remaining is before runtime; second follows acquire.
        with patch.object(executor, '_remaining', side_effect=[1., 0., 0.]):
            result = self.action(context=dict(self.context, remaining_wall_seconds=1.))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['reason'], 'action_wall_budget_exhausted_after_constructor')
        self.assertEqual(result['path_check_calls'], 0)
        self.assertTrue(result['path_scene_reuse']['cache_hit'])
        self.assertFalse(result['path_scene_reuse']['reset_state_verified'])
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_failed_path_is_retested_and_not_cached_as_a_verdict(self):
        self.runtime.path_success = False
        first = self.action()
        self.runtime.path_success = True
        second = self.action()
        self.assertEqual(first['status'], 'UNKNOWN')
        self.assertEqual(second['status'], 'PASS')
        self.assertTrue(second['path_scene_reuse']['cache_hit'])
        self.assertEqual(len(self.calls('controller')), 2)

    def test_controller_exception_invalidates_scene(self):
        self.action()
        self.runtime.raise_path = True
        result = self.action()
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertEqual(self.session.summary()['cached_entries'], 0)
        self.assertFalse(result['path_scene_reuse']['retained_for_next_attempt'])

    def test_configuration_mutation_during_control_is_error_and_discarded(self):
        self.runtime.mutate_configuration = True
        result = self.action()
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['error'], 'scene_identity_or_configuration_changed')
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_cached_configuration_mutation_rejected_before_next_controller(self):
        self.action()
        self.runtime.scenes[-1].force_mag = 123.
        result = self.action()
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['error'], 'cached_scene_mutated')
        self.assertEqual(len(self.calls('controller')), 1)
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_bad_reset_state_is_error_before_force(self):
        self.action()
        self.runtime.reset_bad = True
        result = self.action()
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['error'], 'scene_reset_did_not_restore_initial_state')
        self.assertEqual(len(self.calls('controller')), 1)
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_asset_mutation_during_action_is_error_and_cache_discarded(self):
        self.runtime.write_input = self.assembly / '2.obj'
        result = self.action()
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['reason'], 'input_or_solver_changed_during_execution')
        self.assertFalse(result['input_files_unchanged'])
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_source_mutation_during_action_is_error_and_cache_discarded(self):
        self.runtime.write_source = True
        result = self.action()
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['reason'], 'input_or_solver_changed_during_execution')
        self.assertFalse(result['source_files_unchanged'])
        self.assertEqual(self.session.summary()['cached_entries'], 0)

    def test_fresh_path_keeps_rebuilding_and_has_no_reuse_metadata(self):
        first = self.action(fresh=True); second = self.action(fresh=True)
        self.assertEqual([first['status'], second['status']], ['PASS', 'PASS'])
        self.assertEqual(len(self.calls('path_constructor')), 2)
        self.assertEqual(len(self.calls('controller')), 2)
        self.assertNotIn('path_scene_reuse', first); self.assertNotIn('physics_scene_policy', first)
        self.assertEqual(self.session.summary()['acquires'], 0)


if __name__ == '__main__': unittest.main()
