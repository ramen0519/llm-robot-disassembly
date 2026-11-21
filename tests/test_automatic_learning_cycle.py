"""CLI/lifecycle wiring tests with mocked subprocesses: NO API or physics.

Synthetic audit payloads deliberately exercise runner routing. Their PASS and
actual_physics fields are test inputs, never evidence of real ASAP execution.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_automatic_learning_cycle as runner


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


class AutomaticLearningCycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(runner.ROOT/'runs'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = json.loads((runner.ROOT/'configs/automatic_learning_cycle_v1.json').read_text(encoding='utf-8'))
        self.plan_path = put(self.root/'plan.json', self.plan)
        self.config = {'enabled': True, 'model': 'gpt-6-luna',
            'base_url': 'https://api.openai.com/v1', 'api_key_env': 'OPENAI_API_KEY',
            'max_calls': 39, 'max_usd': 0.5, 'input_usd_per_million': .1,
            'output_usd_per_million': .5, 'max_output_tokens': 2048,
            'reasoning_effort': 'none', 'request_overhead_tokens': 2048}
        self.config_path = put(self.root/'live_config.json', self.config)

    def args(self, mode='preview'):
        return ['--plan', str(self.plan_path), '--run-dir', str(self.root/'cycle'), '--mode', mode]

    def test_default_plan_is_deep_copied_and_count_bound_is_39(self):
        original = deepcopy(self.plan)
        validated = runner.validate_plan(self.plan)
        validated['source']['budget'] = 1
        self.assertEqual(self.plan, original)
        bounds = runner.limits(self.plan)
        self.assertEqual(bounds['max_api_attempts'], 39)
        self.assertEqual(bounds['max_episode_physical_actions'], 24)
        self.assertEqual(bounds['max_validation_physical_actions'], 18)
        self.assertEqual(bounds['automatic_http_retries'], 0)
        self.assertFalse(bounds['partial_solver_resume'])
        self.assertEqual(bounds['performance_comparison'], 'NOT_RUN')

    def test_source_library_adds_selector_fallback_upper_bound(self):
        plan = deepcopy(self.plan)
        plan['initial_library'] = str(self.root/'library.json')
        self.assertEqual(runner.limits(plan)['max_api_attempts'], 51)

    def test_task_bounds_reject_bool_float_and_excessive_work(self):
        for field, invalid in [('budget', True), ('seed', -1), ('seed', 2**32),
                ('budget', 17), ('state_action_budget', 17), ('max_gripper', 4),
                ('max_pose', 4), ('timeout', 241), ('wall_timeout', 301),
                ('response_timeout', 121), ('budget', 1.0), ('base_part', '../0')]:
            plan = deepcopy(self.plan)
            plan['source'][field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                runner.validate_plan(plan)
        for number in (0, 4, True):
            plan = deepcopy(self.plan)
            plan['max_proposals'] = number
            with self.assertRaises(ValueError):
                runner.validate_plan(plan)

    def test_unlisted_assembly_and_unknown_fields_rejected(self):
        for mutation in ('assembly', 'task_field', 'plan_field'):
            plan = deepcopy(self.plan)
            if mutation == 'assembly':
                plan['target']['assembly_dir'] = '/opt/asap-research/not-authorized'
            elif mutation == 'task_field':
                plan['source']['hidden_attempts'] = 100
            else:
                plan['unbounded'] = True
            with self.assertRaises(ValueError):
                runner.validate_plan(plan)

    def test_schema_version_and_purpose_are_strict(self):
        for key, value in [('schema_version', True), ('purpose', None), ('purpose', ' ')]:
            plan = {**self.plan, key: value}
            with self.assertRaises(ValueError):
                runner.validate_plan(plan)

    def test_two_admission_cases_are_required_for_physics_bound(self):
        original = json.loads((runner.ROOT/self.plan['admission_manifest']).read_text(encoding='utf-8'))
        admissions = [c for c in original['cases'] if c['phase'] == 'admission']
        for count in (0, 1, 3):
            template = {**original, 'cases': [admissions[i % 2] for i in range(count)]}
            path = put(self.root/('admissions_%d.json' % count), template)
            plan = {**self.plan, 'admission_manifest': str(path)}
            with self.assertRaisesRegex(ValueError, 'exactly_two_predeclared_admissions'):
                runner.validate_plan(plan)

    def test_e_drive_paths_rejected_before_filesystem_access(self):
        for field in ('assembly_dir', 'admission_manifest', 'initial_memory', 'initial_library'):
            plan = deepcopy(self.plan)
            if field == 'assembly_dir':
                plan['source'][field] = 'E:/must-not-read'
            else:
                plan[field] = 'E:/must-not-read'
            with self.subTest(field=field):
                # Only admission path may be safely inspected before the
                # later initial-memory/library path itself is checked.
                original = Path.lstat
                def guard(path, *args, **kwargs):
                    if str(path).replace('\\', '/').lower().startswith('e:'):
                        self.fail('forbidden path reached filesystem')
                    return original(path, *args, **kwargs)
                with patch.object(Path, 'lstat', guard), self.assertRaises(ValueError):
                    runner.validate_plan(plan)

    def test_preview_never_invokes_runner_key_or_subprocess(self):
        with patch.object(runner, 'execute_cycle', side_effect=AssertionError('must not execute')), \
                patch.object(runner, 'run_recorded', side_effect=AssertionError('must not launch')), \
                patch('local_api_key.key_environment', side_effect=AssertionError('must not read key')), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(self.args()), 0)
        preview = json.loads((self.root/'cycle/preview.json').read_text())
        self.assertEqual(preview['status'], 'PREVIEW')
        self.assertEqual(preview['api_calls'], 0)
        self.assertEqual(preview['physics_calls'], 0)
        self.assertFalse(preview['key_accessed'])

    def test_preview_output_cannot_overwrite(self):
        with redirect_stdout(io.StringIO()):
            runner.main(self.args())
            with self.assertRaises(FileExistsError):
                runner.main(self.args())

    def test_mode_api_flag_guards_prevent_execution(self):
        variants = [self.args('live'), self.args()+['--allow-live-api'],
            self.args('offline-fixture')+['--llm-config', str(self.config_path)],
            self.args('preview')+['--ledger', str(self.root/'ledger.sqlite')],
            self.args('live')+['--allow-live-api', '--llm-config', str(self.config_path)],
            self.args('preview')+['--source-fixture', str(self.root/'fixture.json')]]
        with patch.object(runner, 'execute_cycle') as execute:
            for argv in variants:
                with self.subTest(argv=argv), self.assertRaises(ValueError):
                    runner.main(argv)
            execute.assert_not_called()

    def test_live_configuration_is_pinned_before_execution(self):
        for field, value in [('model', 'another-model'), ('base_url', 'https://example.invalid'),
                             ('api_key_env', 'ANOTHER_KEY'), ('enabled', False)]:
            config = {**self.config, field: value}
            put(self.config_path, config)
            args = self.args('live')+['--allow-live-api', '--llm-config', str(self.config_path),
                                     '--ledger', str(self.root/'ledger.sqlite')]
            with patch.object(runner, 'execute_cycle') as execute, self.assertRaises(ValueError):
                runner.main(args)
            execute.assert_not_called()

    def test_pending_budget_refuses_new_external_execution(self):
        ledger = self.root/'ledger.sqlite'
        ledger.write_bytes(b'fixture ledger not a database')
        args = self.args('live')+['--allow-live-api', '--llm-config', str(self.config_path),
                                 '--ledger', str(ledger)]
        with patch.object(runner, 'budget_summary', return_value={'pending_requests': 1, 'unknown_outcomes': 1}), \
                patch.object(runner, 'execute_cycle') as execute, self.assertRaises(ValueError):
            runner.main(args)
        execute.assert_not_called()

    def test_completed_unknown_budget_also_refuses_new_external_execution(self):
        ledger = self.root/'ledger.sqlite'
        ledger.write_bytes(b'fixture ledger not a database')
        args = self.args('live')+['--allow-live-api', '--llm-config', str(self.config_path),
                                 '--ledger', str(ledger)]
        with patch.object(runner, 'budget_summary', return_value={'pending_requests': 0, 'unknown_outcomes': 1}), \
                patch.object(runner, 'execute_cycle') as execute, self.assertRaises(ValueError):
            runner.main(args)
        execute.assert_not_called()

    def test_direct_execute_rejects_invalid_mode_or_credentials(self):
        invalid = [{'mode': 'preview'}, {'mode': 'anything'}, {'mode': 'live'},
                   {'mode': 'offline-fixture', 'config_path': self.config_path},
                   {'mode': 'live', 'config_path': self.config_path,
                    'ledger': self.root/'ledger.sqlite', 'source_fixture': self.root/'fixture.json'}]
        with patch.object(runner, 'run_recorded') as execute:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    runner.execute_cycle(self.root/'cycle', self.plan, **kwargs)
            invalid_plan = {**self.plan, 'schema_version': True}
            with self.assertRaises(ValueError):
                runner.execute_cycle(self.root/'cycle', invalid_plan, mode='offline-fixture')
            execute.assert_not_called()

    def test_real_graph_forwards_fresh_memory_and_registered_library(self):
        self._mock_cycle(prepared=True)

    def test_no_eligible_experience_skips_generation_and_continues(self):
        self._mock_cycle(prepared=False)

    def test_failed_process_cannot_activate_its_admitted_result(self):
        self._mock_cycle(prepared=True, candidate_exit=1)

    def test_prepared_source_evidence_change_blocks_generation(self):
        self._mock_cycle(prepared=True, mutate_prepared=True)

    def _mock_cycle(self, *, prepared, candidate_exit=0, mutate_prepared=False):
        output = self.root/'cycle'
        fixture = put(self.root/'source_fixture.json', {'fixture_only': True})
        commands, preparation, memory_merges, library_merges = [], [], [], []

        def process(argv, cwd, command_dir):
            tool, args = Path(argv[3]).name, argv[4:]
            commands.append((tool, list(args)))
            folder = Path(args[args.index('--run-dir')+1])
            folder.mkdir(parents=True, exist_ok=False)
            Path(command_dir).mkdir(parents=True, exist_ok=False)
            if tool == 'run_skill_discovery.py':
                put(folder/'result.json', {'status': 'REGISTERED_SCOPED', 'candidate_id': 'new-fixture-skill',
                                          'admission': {'admitted': True}})
                put(folder/'source_preservation.json', {'unchanged': True})
                put(folder/'skills_frozen.json', {'entries': [{'fixture_program': True}]})
            else:
                target = folder.name == 'next_episode'
                put(folder/'worker_result.json', {'mock_unit_test_only': True})
                put(folder/'audit.json', {'validation_ok': True, 'actual_physics': True,
                    'plan_success': target, 'status': 'PASS' if target else 'UNKNOWN'})
                put(folder/'host_preservation.json', {'source_unchanged': True, 'memory_unchanged': True})
                put(folder/'action_memory_frozen.json', {'record_count': 3 if target else 2})
                if tool == 'run_generated_skills.py':
                    put(folder/'generated_skills_audit.json', {'library_unchanged': True,
                        'physical_attempts_of_generated_choices': 1,
                        'generated_choice_physical_passes': 1, 'actual_llm_skill_selections': 0,
                        'selected_skill_ids': ['new-fixture-skill']})
            return {'exit_code': candidate_exit if tool == 'run_skill_discovery.py' else 0}

        def prepare(source, admission, folder):
            preparation.append((Path(source), Path(admission), Path(folder)))
            Path(folder).mkdir(parents=True)
            originals = {str(Path(source)/'worker_result.json'): runner.sha(Path(source)/'worker_result.json'),
                         str(admission): runner.sha(admission)}
            if prepared:
                manifest = put(Path(folder)/'manifest.json', {'fixture_only': True})
                result = {'status': 'PREPARED', 'manifest_path': str(manifest)}
            else:
                result = {'status': 'NO_ELIGIBLE_EXPERIENCE', 'manifest_path': None}
            result['original_artifact_sha256'] = originals
            put(Path(folder)/'preparation.json', result)
            return result

        def memory_union(snapshots, observations, destination):
            memory_merges.append(([Path(p) for p in snapshots], list(observations), Path(destination)))
            counts = sum(json.loads(Path(p).read_text())['record_count'] for p in snapshots)
            return json.loads(put(destination, {'record_count': counts}).read_text())

        def library_union(snapshots, destination, **kwargs):
            library_merges.append(([Path(p) for p in snapshots], Path(destination), kwargs))
            value = {'entries': [{'fixture_program': True}] if snapshots else []}
            put(destination, value)
            return value

        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'run_recorded', side_effect=process))
            stack.enter_context(patch.object(runner, 'runtime_hashes', return_value={'fixture.py': 'a'*64}))
            stack.enter_context(patch.object(runner, 'load_snapshot', side_effect=lambda p: json.loads(Path(p).read_text())))
            stack.enter_context(patch.object(runner, 'retrieve', side_effect=lambda value, **kwargs: value['entries']))
            stack.enter_context(patch.object(runner, 'freeze_cumulative_memory', side_effect=memory_union))
            stack.enter_context(patch('prepare_episode_learning.prepare_episode_learning', side_effect=prepare))
            stack.enter_context(patch('orchestrator.skill_library_merge.freeze_skill_union', side_effect=library_union))
            stack.enter_context(patch('local_api_key.key_environment', side_effect=AssertionError('no API key access')))
            if mutate_prepared:
                from orchestrator.learning_cycle_workflow import LearningCycleWorkflow
                original_event = LearningCycleWorkflow._event
                def mutate_after_complete(flow, kind, stage=None, **kwargs):
                    original_event(flow, kind, stage, **kwargs)
                    if kind == 'stage_completed' and stage == 'prepare':
                        put(output/'source_episode/worker_result.json', {'changed_after_prepare': True})
                stack.enter_context(patch.object(LearningCycleWorkflow, '_event', mutate_after_complete))
                with self.assertRaisesRegex(RuntimeError, 'completed_artifact_hash_mismatch'):
                    runner.execute_cycle(output, self.plan, mode='offline-fixture', source_fixture=fixture)
                self.assertEqual(len(commands), 1)
                return
            first = runner.execute_cycle(output, self.plan, mode='offline-fixture', source_fixture=fixture)
            again = runner.execute_cycle(output, self.plan, mode='offline-fixture', source_fixture=fixture)

        self.assertEqual(first, again)
        self.assertEqual(first['mode'], 'offline-fixture')
        self.assertTrue(first['fixture_only'])
        self.assertNotEqual(first['status'], 'LIVE_LEARNING_CYCLE_OBSERVED')
        self.assertEqual(first['budget'], {'reserved_calls': 0})
        self.assertEqual(first['source_plan_success'], False)
        self.assertEqual(first['target_plan_success'], True)
        self.assertEqual(first['final_memory_records'], 5)
        self.assertEqual(len(preparation), 1)
        self.assertEqual(preparation[0][0], output/'source_episode')
        self.assertEqual(len(memory_merges), 2)
        self.assertEqual(memory_merges[0][0], [output/'source_episode/action_memory_frozen.json'])
        self.assertEqual(memory_merges[1][0], [output/'memory_for_next_episode.json',
                                               output/'next_episode/action_memory_frozen.json'])
        self.assertEqual(len(library_merges), 1)
        activated = prepared and candidate_exit == 0
        self.assertEqual(len(commands), 3 if prepared else 2)
        target_tool, target_args = commands[-1]
        self.assertEqual(target_tool, 'run_generated_skills.py' if activated else 'run_explicit_action_experiment.py')
        self.assertEqual(Path(target_args[target_args.index('--memory')+1]), output/'memory_for_next_episode.json')
        self.assertNotIn('--allow-live-api', target_args)
        self.assertEqual(target_args[target_args.index('--mode')+1], 'deterministic')
        self.assertEqual(first['new_skill_registered'], activated)
        if activated:
            self.assertEqual(Path(target_args[target_args.index('--skill-library')+1]), output/'library_for_next_episode.json')
            self.assertIn('--allow-fixture-library', target_args)
            self.assertEqual(library_merges[0][0], [output/'candidate_01/skills_frozen.json'])
        else:
            self.assertNotIn('--skill-library', target_args)
            self.assertEqual(library_merges[0][0], [])


if __name__ == '__main__':
    unittest.main()
