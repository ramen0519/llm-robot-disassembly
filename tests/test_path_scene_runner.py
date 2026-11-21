"""Opt-in scene wiring with injected worker/physics only; no API, WSL or solver."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
from tools import run_explicit_action_experiment as host
from tools import run_hierarchical_episode as hierarchy
from worker import run_explicit_actions as worker
from worker.transfer_protocol import validate_manifest
from tests.test_explicit_action_runner import fake_report
from tests.test_task_memory import features
import explicit_action_physics
import explicit_action_planner


class PathSceneRunnerTests(unittest.TestCase):
    def setUp(self):
        self.net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.net.start(); self.addCleanup(self.net.stop)
        self.key = patch('local_api_key.key_environment', side_effect=AssertionError('NO_KEY'))
        self.key.start(); self.addCleanup(self.key.stop)

    def test_host_and_hierarchy_parser_default_fresh_and_accept_opt_in(self):
        basic = ['--run-dir', str(ROOT / 'runs/not_created_scene_parser')]
        extra = ['--hierarchy-policy', 'unused', '--hierarchy-ledger', 'unused',
                 '--planner-ledger', 'unused', '--skill-library', 'unused']
        for parser, args in ((host.parser(), basic), (hierarchy.parser(), basic + extra)):
            self.assertEqual(parser.parse_args(args).physics_scene_policy, 'fresh_v1')
            self.assertEqual(parser.parse_args(args + ['--physics-scene-policy',
                'path_scene_reuse_v1']).physics_scene_policy, 'path_scene_reuse_v1')

    def test_invalid_programmatic_host_policy_rejected_before_directory_or_worker(self):
        with tempfile.TemporaryDirectory(prefix='scene_invalid_OFFLINE_', dir=ROOT / 'runs') as folder:
            output = Path(folder) / 'not_created'
            args = host.parser().parse_args(['--run-dir', str(output)])
            args.physics_scene_policy = 'unsupported'
            with patch.object(host, 'run_recorded', side_effect=AssertionError('NO_WORKER')) as runner:
                with self.assertRaisesRegex(ValueError, 'unknown_physics_scene_policy'):
                    host.execute(args)
            runner.assert_not_called(); self.assertFalse(output.exists())

    def run_host(self, policy=None):
        with tempfile.TemporaryDirectory(prefix='scene_host_OFFLINE_', dir=ROOT / 'runs') as folder:
            output = Path(folder) / 'episode'
            argv = ['--run-dir', str(output), '--mode', 'deterministic']
            if policy is not None:
                argv += ['--physics-scene-policy', policy]
            args = host.parser().parse_args(argv)
            commands = []
            def injected_worker(command, cwd, logdir):
                commands.append(command[:]); logdir.mkdir(parents=True, exist_ok=False)
                report = fake_report()
                report.update(actual_physics=False, worker_execution='INJECTED_NO_PHYSICS_NO_WSL')
                (output / 'worker_result.json').write_text(json.dumps(report), encoding='utf-8')
                return dict(exit_code=0, elapsed_seconds=0.001, fixture_only=True)
            with patch.object(host, 'run_recorded', side_effect=injected_worker), \
                 patch('subprocess.run', side_effect=AssertionError('NO_SUBPROCESS')), \
                 patch('builtins.print'):
                self.assertEqual(host.execute(args), 0)
            config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
            graph = json.loads((output / 'graph_state.json').read_text(encoding='utf-8'))
            self.assertFalse(graph['metrics']['actual_physics'])
            self.assertEqual(len(commands), 1)
            return commands[0], config

    def test_default_and_explicit_fresh_do_not_add_worker_flag(self):
        for policy in (None, 'fresh_v1'):
            with self.subTest(policy=policy):
                command, config = self.run_host(policy)
                self.assertNotIn('--physics-scene-policy', command)
                self.assertEqual(config['arguments']['physics_scene_policy'], 'fresh_v1')

    def test_host_passes_opt_in_through_real_graph_and_records_config(self):
        command, config = self.run_host('path_scene_reuse_v1')
        index = command.index('--physics-scene-policy')
        self.assertEqual(command[index + 1], 'path_scene_reuse_v1')
        self.assertEqual(config['arguments']['physics_scene_policy'], 'path_scene_reuse_v1')


class PathSceneManifestTests(unittest.TestCase):
    def manifest(self, schema='unseen_transfer_v1'):
        settings = dict(budget=16, state_action_budget=6, max_gripper=3, max_pose=128,
                        timeout=480, response_timeout=120, wall_timeout=600)
        args = SimpleNamespace(**settings, seed=0, base_part=None)
        value = dict(schema=schema, automatic_retries=0, settings=settings,
                     assembly_dirs={'/opt/asap-research/07009': {}}, path_limit=96, stability_limit=48)
        if schema == 'multitask_learning_v1':
            value.update(seed=0, purpose='training', training_ids=['07009', '02628', '01289'],
                         evaluation_ids=['02680', '02656'])
        return value, args

    def check(self, value, args):
        with tempfile.TemporaryDirectory(prefix='scene_manifest_OFFLINE_', dir=ROOT / 'runs') as folder:
            path = Path(folder) / 'manifest.json'
            path.write_text(json.dumps(value), encoding='utf-8')
            return validate_manifest(path, '/opt/asap-research/07009', args)

    def test_legacy_manifest_without_profile_remains_valid_for_fresh(self):
        for schema in ('unseen_transfer_v1', 'multitask_learning_v1'):
            with self.subTest(schema=schema):
                value, args = self.manifest(schema)
                self.assertEqual(self.check(value, args), value)
                args.physics_scene_policy = 'fresh_v1'
                self.assertEqual(self.check(value, args), value)

    def test_reuse_requires_explicit_matching_manifest_in_both_directions(self):
        for schema in ('unseen_transfer_v1', 'multitask_learning_v1'):
            with self.subTest(schema=schema):
                value, args = self.manifest(schema)
                args.physics_scene_policy = 'path_scene_reuse_v1'
                with self.assertRaisesRegex(ValueError, 'evaluation_physics_scene_policy_mismatch'):
                    self.check(value, args)
                value['settings']['physics_scene_policy'] = 'path_scene_reuse_v1'
                self.assertEqual(self.check(value, args), value)
                args.physics_scene_policy = 'fresh_v1'
                with self.assertRaisesRegex(ValueError, 'evaluation_physics_scene_policy_mismatch'):
                    self.check(value, args)

    def test_unknown_declared_profile_rejected(self):
        value, args = self.manifest()
        value['settings']['physics_scene_policy'] = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'evaluation_physics_scene_policy_mismatch'):
            self.check(value, args)

    def test_host_rejects_manifest_mismatch_before_worker_and_output_creation(self):
        with tempfile.TemporaryDirectory(prefix='scene_eval_OFFLINE_', dir=ROOT / 'runs') as folder:
            folder = Path(folder); output = folder / 'not_created'; path = folder / 'manifest.json'
            value, _ = self.manifest(); path.write_text(json.dumps(value), encoding='utf-8')
            args = host.parser().parse_args(['--run-dir', str(output), '--assembly-dir',
                '/opt/asap-research/07009', '--base-part', 'none', '--evaluation-manifest', str(path),
                '--physics-scene-policy', 'path_scene_reuse_v1'])
            with patch.object(host, 'run_recorded', side_effect=AssertionError('NO_WORKER')) as runner:
                with self.assertRaisesRegex(ValueError, 'evaluation_physics_scene_policy_mismatch'):
                    host.execute(args)
            runner.assert_not_called(); self.assertFalse(output.exists())


class WorkerSceneLifetimeTests(unittest.TestCase):
    def run_worker(self, policy=None, error_after_attempts=False):
        """Run the real worker main/control flow with all ASAP/physics injected."""
        with tempfile.TemporaryDirectory(prefix='scene_worker_OFFLINE_', dir=ROOT / 'runs') as folder:
            home = Path(folder); root = home / 'asap'; root.mkdir()
            assembly = home / 'assembly'; assembly.mkdir()
            (assembly / 'part.obj').write_text('synthetic, not a mesh', encoding='utf-8')
            output = home / 'episode'; sessions = []; calls = []
            class Session:
                def __init__(self):
                    self.closed = False; self.close_count = 0; sessions.append(self)
                def close(self):
                    self.closed = True; self.close_count += 1
                def summary(self):
                    return dict(fixture_only=True, max_entries=1, closed=self.closed,
                                close_count=self.close_count)
            class Planner:
                def __init__(self, **kwargs):
                    self.verify = kwargs['verifier_fn']
                def plan(self, **kwargs):
                    for index in range(2):
                        self.verify({'time_remaining_seconds': 17.0}, {'part_id': 'arm'},
                                    output / ('action' + str(index)))
                    if error_after_attempts:
                        raise RuntimeError('injected_planner_error')
                    return {}
                def report(self):
                    return dict(status='UNKNOWN', success=False, fixture_only=True)
                def get_stats(self, tree):
                    return {}
            def module(name, **members):
                result = ModuleType(name)
                for key, value in members.items(): setattr(result, key, value)
                return result
            modules = {
                'numpy': module('numpy', random=SimpleNamespace(seed=Mock())),
                'torch': module('torch', manual_seed=Mock()), 'trimesh': module('trimesh'),
                'assets': module('assets'),
                'assets.load': module('assets.load', load_part_ids=lambda _: ['arm', 'base', 'pin']),
                'assets.transform': module('assets.transform', transform_pt_by_matrix=Mock()),
                'plan_sequence': module('plan_sequence'),
                'plan_sequence.stable_pose': module('plan_sequence.stable_pose', get_combined_mesh=Mock()),
                'plan_sequence.planner': module('plan_sequence.planner'),
                'plan_sequence.planner.dfs': module('plan_sequence.planner.dfs', DFSSequencePlanner=object),
                'plan_sequence.planner.base': module('plan_sequence.planner.base', SequencePlanner=object),
                'plan_sequence.physics_planner': module('plan_sequence.physics_planner'),
                'path_scene_reuse': module('path_scene_reuse', PathSceneSession=Session),
            }
            def describe(root, assembly, base, settings):
                value = features(); value['task']['solver_settings'] = deepcopy(settings)
                return value
            def execute(*args, **kwargs):
                calls.append((deepcopy(args[0]), kwargs.copy(), len(args)))
                return dict(status='UNKNOWN', fixture_only=True)
            argv = ['--asap-root', str(root), '--assembly-dir', str(assembly), '--output-dir', str(output),
                    '--run-id', 'injected_scene_test', '--base-part', 'base']
            if policy is not None: argv += ['--physics-scene-policy', policy]
            with patch.dict(sys.modules, modules), \
                 patch.object(worker, 'checked_path', side_effect=lambda p, **kwargs: Path(p)), \
                 patch.object(worker, 'describe_task', side_effect=describe), \
                 patch.object(worker, 'ExplicitActionClient'), \
                 patch.object(worker.random, 'seed'), \
                 patch.object(worker.subprocess, 'check_output', return_value=worker.COMMIT), \
                 patch.object(worker.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                 patch.object(explicit_action_planner, 'make_explicit_action_planner', return_value=Planner), \
                 patch.object(explicit_action_physics, 'execute_action', side_effect=execute), \
                 patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK')):
                if error_after_attempts:
                    with self.assertRaisesRegex(RuntimeError, 'injected_planner_error'):
                        worker.main(argv)
                else:
                    self.assertEqual(worker.main(argv), 0)
            context = json.loads((output / 'context.json').read_text(encoding='utf-8'))
            result = json.loads((output / 'worker_result.json').read_text(encoding='utf-8')) if not error_after_attempts else None
            closed_summary = json.loads((output / 'path_scene_session.json').read_text(encoding='utf-8')) if sessions else None
            preservation = json.loads((output / 'preservation.json').read_text(encoding='utf-8'))
            self.assertTrue(preservation['input_unchanged'])
            return context, result, calls, sessions, closed_summary

    def test_worker_fresh_preserves_context_and_three_argument_call(self):
        default = self.run_worker()
        explicit = self.run_worker('fresh_v1')
        self.assertEqual(default[0], explicit[0])
        self.assertNotIn('physics_scene_policy', default[0]['task']['solver_settings'])
        for _, result, calls, sessions, closed in (default, explicit):
            self.assertEqual(sessions, []); self.assertIsNone(closed)
            self.assertNotIn('path_scene_session', result)
            self.assertEqual(len(calls), 2)
            for context, kwargs, argc in calls:
                self.assertNotIn('physics_scene_policy', context)
                self.assertEqual(kwargs, {}); self.assertEqual(argc, 3)

    def test_worker_owns_one_session_passes_it_and_closes_after_normal_plan(self):
        context, result, calls, sessions, closed = self.run_worker('path_scene_reuse_v1')
        self.assertEqual(context['task']['solver_settings']['physics_scene_policy'], 'path_scene_reuse_v1')
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(calls), 2)
        for request, kwargs, argc in calls:
            self.assertIs(kwargs['scene_session'], sessions[0]); self.assertEqual(argc, 3)
            self.assertEqual(request['physics_scene_policy'], 'path_scene_reuse_v1')
            self.assertEqual(request['remaining_wall_seconds'], 17.0)
            self.assertEqual(request['task_context'], context)
        self.assertEqual(result['path_scene_session']['max_entries'], 1)
        self.assertTrue(closed['closed']); self.assertEqual(closed['close_count'], 1)

    def test_worker_closes_and_records_session_when_plan_raises(self):
        _, result, calls, sessions, closed = self.run_worker('path_scene_reuse_v1', error_after_attempts=True)
        self.assertIsNone(result); self.assertEqual(len(calls), 2); self.assertEqual(len(sessions), 1)
        self.assertTrue(closed['closed']); self.assertEqual(closed['close_count'], 1)


if __name__ == '__main__': unittest.main()
