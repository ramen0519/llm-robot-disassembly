"""Bounded lifecycle wiring with synthetic results, never real API/physics."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_initial_contact_skill_cycle as runner


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


class InitialContactCycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runner.ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode = self.root / 'source_episode'
        self.episode.mkdir()
        self.source = put(self.episode / 'source.json', {'immutable': True})
        self.memory = put(self.root / 'memory.json', {'record_count': 29})
        self.library = put(self.root / 'library.json', {'entries': [{'strategy_id': 'old'}]})
        self.config = put(self.root / 'config.json', {'max_calls': 39, 'max_usd': .4})
        self.ledger = self.root / 'ledger.sqlite'
        self.ledger.write_bytes(b'mocked ledger')
        self.budget = dict(max_calls=39, max_usd='.4', pending_requests=0, unknown_outcomes=0,
                          remaining_calls=5, remaining_estimated_usd='.1254840', reserved_calls=34)
        self.manifest = {'synthesis_failure_scope': 'initial_disconnected', 'cases': [
            dict(case_id='case_%d' % index, phase=phase, source_task='07009', report='test_report',
                attempt_index=index, trigger_kind='stability_failed', expected_diagnostic_phase='initial_disconnected')
            for index, phase in ((1, 'debug'), (3, 'admission'), (7, 'admission'))]}

    def args(self):
        return ['--run-dir', str(self.root / 'preview'), '--episode-dir', str(self.episode),
                '--initial-memory', str(self.memory), '--initial-library', str(self.library)]

    def test_five_call_bounds_without_controlled_reuse(self):
        self.assertEqual(runner.LIMITS['max_api_attempts'], 1 + 2 * runner.TASK['budget'])
        self.assertEqual(runner.LIMITS['max_validation_physical_actions'], 6)
        self.assertEqual(runner.LIMITS['controlled_reuse_actions'], 0)
        self.assertEqual(runner.LIMITS['automatic_refinements'], 0)
        self.assertEqual(runner.TASK['state_action_budget'], 2)

    def test_manifest_rejects_dynamic_or_unscoped_conditions(self):
        runner.validate_manifest(self.manifest)
        for mutation in ('scope', 'count', 'phase'):
            value = deepcopy(self.manifest)
            if mutation == 'scope':
                value['synthesis_failure_scope'] = 'dynamic_stability_failed'
            elif mutation == 'count':
                value['cases'] = value['cases'][:2]
            else:
                value['cases'][0].pop('expected_diagnostic_phase')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.validate_manifest(value)

    def test_preview_is_offline_and_does_not_overwrite(self):
        with patch.object(runner, 'execute_cycle', side_effect=AssertionError('no execution')), \
                patch('local_api_key.key_environment', side_effect=AssertionError('no key')), \
                redirect_stdout(io.StringIO()):
            runner.main(self.args())
            with self.assertRaises(FileExistsError):
                runner.main(self.args())
        preview = json.loads((self.root / 'preview/preview.json').read_text())
        self.assertEqual(preview['limits']['max_api_attempts'], 5)
        self.assertEqual(preview['api_calls'], 0)
        self.assertFalse(preview['fixed_selection_injected'])

    def test_forbidden_path_guard_and_missing_authorization(self):
        args = self.args()
        args[args.index('--episode-dir') + 1] = 'E:/never-inspect'
        with patch.object(runner, 'checked', side_effect=AssertionError('filesystem')), self.assertRaises(ValueError):
            runner.main(args)
        with patch.object(runner, 'execute_cycle') as execute, self.assertRaises(ValueError):
            runner.main(self.args() + ['--mode', 'live'])
        execute.assert_not_called()

    def mock_cycle(self, *, admitted=True, executed=True, passed=True, available=5,
                   physical_calls=2, changed_input=False, charge_extra=False,
                   initial_prior=False, changed_prior=False):
        commands = []
        self.commands = commands
        budget = {**self.budget, 'remaining_calls': available, 'reserved_calls': 39 - available}
        prior_file = put(self.root / 'prior/generation_packet.json', {'original': True}) if initial_prior else None
        prior_hash = runner.sha(prior_file) if prior_file else None
        def prepare(episode, folder, *, failure_scope):
            self.assertEqual(failure_scope, 'initial_disconnected')
            manifest = put(Path(folder) / 'manifest.json', self.manifest)
            result = dict(status='PREPARED', memory_path=str(self.memory), manifest_path=str(manifest),
                original_artifact_sha256={str(self.source): runner.sha(self.source)})
            put(Path(folder) / 'preparation.json', result)
            if changed_input:
                put(self.memory, {'unexpected': True})
            if changed_prior:
                put(prior_file, {'unexpected': True})
            return result
        def process(command, cwd, command_dir):
            tool, args = Path(command[3]).name, command[4:]
            commands.append((tool, args))
            folder = Path(args[args.index('--run-dir') + 1])
            folder.mkdir()
            Path(command_dir).mkdir()
            if tool == 'run_skill_discovery.py':
                put(folder / 'result.json', {'status': 'REGISTERED_SCOPED' if admitted else 'NOT_REGISTERED',
                    'generation_origin': 'live_llm', 'candidate_id': 'new', 'admission': {'admitted': admitted}})
                put(folder / 'source_preservation.json', {'unchanged': True})
                put(folder / 'skills_frozen.json', {'entries': [{'strategy_id': 'new'}]})
                charged = 2 if charge_extra else 1
            else:
                self.assertEqual(tool, 'run_generated_skills.py')
                put(folder / 'audit.json', {'status': 'UNKNOWN', 'plan_success': False,
                    'validation_ok': True, 'actual_physics': True, 'verifier_calls': physical_calls,
                    'fallbacks': 0 if executed else 1, 'llm_choices': 2 if executed else 1})
                put(folder / 'host_preservation.json', {'source_unchanged': True, 'memory_unchanged': True})
                put(folder / 'generated_skills_audit.json', {'library_unchanged': True,
                    'actual_llm_selected_physically_executed_skill_ids': ['new'] if executed else ['old'],
                    'actual_llm_selected_physical_pass_skill_ids': ['new'] if executed and passed else ['old']})
                put(folder / 'action_memory_frozen.json', {'record_count': physical_calls})
                charged = min(4, budget['remaining_calls'])
            budget['reserved_calls'] += charged
            budget['remaining_calls'] -= charged
            return {'exit_code': 0}
        def memory_union(paths, observations, destination):
            result = {'record_count': sum(json.loads(Path(p).read_text()).get('record_count', 0) for p in paths)}
            put(destination, result)
            return result
        def library_union(paths, destination):
            result = {'entries': [row for p in paths for row in json.loads(Path(p).read_text())['entries']]}
            put(destination, result)
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'live_inputs', return_value=(self.config, self.ledger, {'max_calls': 39, 'max_usd': .4})))
            stack.enter_context(patch.object(runner, 'budget_summary', side_effect=lambda _: deepcopy(budget)))
            stack.enter_context(patch.object(runner, 'runtime_hashes', return_value={'fixture.py': 'a'*64}))
            stack.enter_context(patch.object(runner, 'load_snapshot', side_effect=lambda path: json.loads(Path(path).read_text())))
            stack.enter_context(patch.object(runner, 'retrieve', side_effect=lambda value, **kwargs: value['entries']))
            stack.enter_context(patch.object(runner, 'prepare_stability_learning', side_effect=prepare))
            stack.enter_context(patch.object(runner, 'run_recorded', side_effect=process))
            stack.enter_context(patch.object(runner, 'freeze_cumulative_memory', side_effect=memory_union))
            stack.enter_context(patch.object(runner, 'freeze_skill_union', side_effect=library_union))
            if initial_prior:
                stack.enter_context(patch.object(runner, 'refinement_inputs',
                    return_value=({str(prior_file): prior_hash}, deepcopy(self.manifest))))
            stack.enter_context(patch('local_api_key.key_environment', side_effect=AssertionError('no key')))
            return runner.execute_cycle(self.root / 'cycle', episode_dir=self.episode,
                initial_memory=self.memory, initial_library=self.library, config_path=self.config, ledger=self.ledger,
                refine_from=prior_file.parent if prior_file else None)

    def test_registered_skill_goes_directly_to_free_dfs_then_memory_union(self):
        result = self.mock_cycle()
        self.assertEqual([tool for tool, args in self.commands], ['run_skill_discovery.py', 'run_generated_skills.py'])
        args = self.commands[-1][1]
        self.assertEqual(args[args.index('--budget') + 1], '2')
        self.assertNotIn('--fixture', args)
        self.assertNotIn('--case-spec', args)
        self.assertEqual(result['status'], 'INITIAL_CONTACT_SKILL_REUSE_PASS')
        self.assertFalse(result['free_dfs_plan_success'])
        self.assertEqual(result['budget']['reserved_calls'], 39)
        self.assertFalse(result['controlled_reuse_executed'])
        self.assertFalse(result['target_is_held_out'])
        self.assertTrue((self.root / 'cycle/memory_after_reuse.json').is_file())

    def test_no_registration_means_no_refinement_or_free_dfs(self):
        result = self.mock_cycle(admitted=False)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(result['status'], 'NOT_REGISTERED')
        self.assertFalse(result['free_dfs_executed'])
        self.assertEqual(result['budget']['reserved_calls'], 35)

    def test_other_skill_success_does_not_count_as_new_skill_reuse(self):
        result = self.mock_cycle(executed=False)
        self.assertEqual(result['status'], 'NOT_OBSERVED_WITHIN_BUDGET')
        self.assertFalse(result['free_dfs_reuse_pass'])

    def test_new_execution_without_pass_is_reported_separately(self):
        result = self.mock_cycle(passed=False)
        self.assertEqual(result['status'], 'INITIAL_CONTACT_SKILL_EXECUTED')
        self.assertFalse(result['free_dfs_reuse_pass'])

    def test_insufficient_budget_fails_before_any_subprocess(self):
        with self.assertRaises(ValueError):
            self.mock_cycle(available=1)
        self.assertEqual(self.commands, [])
        self.assertFalse((self.root / 'cycle').exists())

    def test_changed_input_fails_before_any_subprocess(self):
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            self.mock_cycle(changed_input=True)
        self.assertEqual(self.commands, [])

    def test_more_than_two_physical_actions_fails_evidence_gate(self):
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            self.mock_cycle(physical_calls=3)
        self.assertFalse((self.root / 'cycle/cycle_result.json').exists())

    def test_generation_charge_above_one_stops_without_free_dfs(self):
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            self.mock_cycle(charge_extra=True)
        self.assertEqual(len(self.commands), 1)

    def test_four_remaining_calls_forward_prior_and_report_fallback_limitation(self):
        result = self.mock_cycle(available=4, initial_prior=True, executed=False)
        first = self.commands[0][1]
        self.assertEqual(Path(first[first.index('--refine-from') + 1]), self.root / 'prior')
        self.assertEqual(result['limits']['max_api_attempts'], 4)
        self.assertEqual(result['limits']['max_free_dfs_api_attempts'], 3)
        self.assertEqual(result['budget']['reserved_calls'], 39)
        self.assertEqual(result['status'], 'NOT_OBSERVED_WITHIN_BUDGET')
        self.assertTrue(result['explicit_prior_refinement'])
        self.assertTrue(result['api_exhaustion_may_execute_deterministic_fallback'])
        self.assertEqual(result['free_dfs_fallback_count'], 1)
        self.assertFalse(result['free_dfs_reuse_pass'])

    def test_changed_bound_prior_prevents_subprocess(self):
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            self.mock_cycle(available=4, initial_prior=True, changed_prior=True)
        self.assertEqual(self.commands, [])

    def test_initial_contact_refinement_binds_prior_source_and_debug_files(self):
        prior = self.root / 'prior'
        config = {'max_calls': 39, 'max_usd': .4}
        archived = put(prior / 'implementation/orchestrator/example.py', {'archived_fixture': True})
        original = {
            'provider_proposal.json': {'origin': 'actual_llm', 'status': 'proposed'},
            'result.json': {'status': 'NOT_REGISTERED', 'generation_origin': 'live_llm'},
            'precommitted_cases.json': self.manifest,
            'config.json': {'mode': 'live', 'api': config, 'ledger': str(self.ledger)},
            'source_preservation.json': {'unchanged': True}, 'generation_packet.json': {},
            'source_before.json': {'orchestrator/example.py': runner.sha(archived)}}
        for name, value in original.items():
            put(prior / name, value)
        debug = put(prior / 'cases/case_1/program_request.json', {'debug_only': True})
        with patch('run_skill_discovery.build_refinement_feedback',
                   return_value=[{'source_run': str(prior), 'case_id': 'case_1'}]):
            hashes, manifest = runner.refinement_inputs(prior, self.ledger, config)
        self.assertIn(str(archived), hashes)
        self.assertIn(str(debug), hashes)
        self.assertEqual(manifest, self.manifest)
        put(archived, {'mutated_source': True})
        with self.assertRaisesRegex(ValueError, 'prior_archived_source_changed'):
            runner.refinement_inputs(prior, self.ledger, config)

    def test_prior_other_ledger_is_rejected_before_debug_rehydration(self):
        prior = self.root / 'prior'
        config = {'max_calls': 39, 'max_usd': .4}
        files = {
            'provider_proposal.json': {'origin': 'actual_llm', 'status': 'proposed'},
            'result.json': {'status': 'NOT_REGISTERED', 'generation_origin': 'live_llm'},
            'precommitted_cases.json': self.manifest,
            'config.json': {'mode': 'live', 'api': config, 'ledger': str(self.root / 'wrong.sqlite')},
            'source_preservation.json': {'unchanged': True}, 'generation_packet.json': {}, 'source_before.json': {}}
        for name, value in files.items():
            put(prior / name, value)
        with patch('run_skill_discovery.build_refinement_feedback', side_effect=AssertionError('no feedback')), \
                self.assertRaisesRegex(ValueError, 'original_authorization'):
            runner.refinement_inputs(prior, self.ledger, config)


if __name__ == '__main__':
    unittest.main()
