"""Runner wiring with synthetic subprocess payloads; no API or physics."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_stability_skill_cycle as runner


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


class StabilitySkillCycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runner.ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episode = self.root / 'existing_episode'
        self.episode.mkdir()
        self.source = put(self.episode / 'source.json', {'immutable': True})
        self.memory = put(self.root / 'initial_memory.json', {'record_count': 24})
        self.library = put(self.root / 'initial_library.json', {'entries': [{'strategy_id': 'old'}]})
        self.config = put(self.root / 'config.json', {'max_calls': 39, 'max_usd': .40})
        self.ledger = self.root / 'ledger.sqlite'
        self.ledger.write_bytes(b'mocked ledger')
        self.budget = dict(max_calls=39, max_usd='0.4', pending_requests=0, unknown_outcomes=0,
            remaining_calls=13, remaining_estimated_usd='0.2', reserved_calls=26)
        self.manifest = {'synthesis_failure_scope': 'dynamic_stability_failed', 'cases': [
            dict(case_id='case_%d' % index, phase=phase, source_task='07009', report='test_report',
                 attempt_index=index, trigger_kind='stability_failed')
            for index, phase in ((2, 'debug'), (4, 'admission'), (5, 'admission'))]}

    def args(self):
        return ['--run-dir', str(self.root / 'preview'), '--episode-dir', str(self.episode),
                '--initial-memory', str(self.memory), '--initial-library', str(self.library)]

    def test_bounds(self):
        self.assertEqual(runner.LIMITS['max_api_attempts'], 3 + 2 + 2 * runner.TASK['budget'])
        self.assertEqual(runner.LIMITS['automatic_http_retries'], 0)
        self.assertEqual(runner.TASK['budget'], 4)
        self.assertEqual(runner.LIMITS['max_validation_physical_actions'], 18)
        self.assertEqual(runner.cycle_limits(), runner.LIMITS)
        self.assertEqual(runner.cycle_limits(2, 3)['max_api_attempts'], 10)
        for proposals, budget in ((0, 3), (4, 3), (2, 0), (2, 5), (True, 3), (2, 1.5)):
            with self.subTest(proposals=proposals, budget=budget), self.assertRaises(ValueError):
                runner.cycle_limits(proposals, budget)

    def test_preflight_requires_original_remaining_authorization(self):
        runner.check_budget(self.budget, required_calls=13)
        for field, value in (('max_calls', 40), ('max_usd', '0.41'), ('pending_requests', 1),
                             ('unknown_outcomes', 1), ('remaining_calls', 12),
                             ('remaining_estimated_usd', '0')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.check_budget({**self.budget, field: value}, required_calls=13)
        with self.assertRaisesRegex(ValueError, 'cycle_api_attempt_bound'):
            runner.check_budget({**self.budget, 'reserved_calls': 40}, before=self.budget)
        with self.assertRaisesRegex(ValueError, 'cycle_api_attempt_bound'):
            runner.check_budget({**self.budget, 'reserved_calls': 37}, before=self.budget, max_attempts=10)

    def test_controlled_case_copies_without_relabeling_as_held_out(self):
        before = deepcopy(self.manifest)
        case = runner.controlled_spec(self.manifest)
        self.assertEqual(case['phase'], 'transfer')
        self.assertEqual(case['attempt_index'], 2)
        self.assertEqual(case['trigger_kind'], 'stability_failed')
        self.assertEqual(self.manifest, before)
        for mutation in ({'cases': self.manifest['cases'][:2]}, {'synthesis_failure_scope': 'path_failed'}):
            with self.assertRaises(ValueError):
                runner.controlled_spec({**self.manifest, **mutation})

    def test_preview_never_launches_or_loads_key_and_refuses_overwrite(self):
        with patch.object(runner, 'execute_cycle', side_effect=AssertionError('no live')), \
                patch.object(runner, 'run_recorded', side_effect=AssertionError('no subprocess')), \
                patch('local_api_key.key_environment', side_effect=AssertionError('no key')), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(self.args()), 0)
            with self.assertRaises(FileExistsError):
                runner.main(self.args())
        result = json.loads((self.root / 'preview/preview.json').read_text())
        self.assertEqual(result['api_calls'], 0)
        self.assertFalse(result['target_is_held_out'])

    def test_e_drive_rejected_before_any_filesystem_inspection(self):
        args = self.args()
        args[args.index('--episode-dir') + 1] = 'E:/never-inspect'
        with patch.object(runner, 'checked', side_effect=AssertionError('no filesystem')), self.assertRaises(ValueError):
            runner.main(args)

    def test_live_flag_required(self):
        with patch.object(runner, 'execute_cycle') as execute, self.assertRaises(ValueError):
            runner.main(self.args() + ['--mode', 'live'])
        execute.assert_not_called()

    def mock_cycle(self, candidate_status='REGISTERED_SCOPED', selected=True,
                   max_proposals=3, episode_budget=4, initial_prior=False, mutate_prior=False):
        commands = []
        self.commands = commands
        budget = deepcopy(self.budget)
        budget['remaining_calls'] = runner.cycle_limits(max_proposals, episode_budget)['max_api_attempts']
        budget['reserved_calls'] = 39 - budget['remaining_calls']
        prior_file = put(self.root / 'prior/generation_packet.json', {'original': True}) if initial_prior else None
        prior_hash = runner.sha(prior_file) if prior_file else None
        def process(command, cwd, command_dir):
            tool = Path(command[3]).name
            args = command[4:]
            commands.append((tool, args))
            folder = Path(args[args.index('--run-dir') + 1])
            folder.mkdir()
            Path(command_dir).mkdir()
            if tool == 'run_skill_discovery.py':
                admitted = candidate_status == 'REGISTERED_SCOPED'
                put(folder / 'result.json', {'status': candidate_status, 'candidate_id': 'new',
                                            'admission': {'admitted': admitted}})
                put(folder / 'source_preservation.json', {'unchanged': True})
                put(folder / 'skills_frozen.json', {'entries': [{'strategy_id': 'new'}]})
                put(folder / 'provider_proposal.json', {'origin': 'actual_llm', 'status': 'proposed'})
                charged = 1
            elif tool == 'run_generated_skill_case.py':
                put(folder / 'case_report.json', {'status': 'RECOVERY_PASS', 'workflow_completed': True,
                    'generated_skill_id': 'new', 'generated_reuse_pass': True, 'llm_selection_used': selected,
                    'generated_selector_status': 'selected', 'memory_export': {'status': 'FROZEN'}})
                put(folder / 'memory_frozen.json', {'record_count': 2})
                put(folder / 'preservation.json', {'library_unchanged': True, 'source_unchanged': True,
                                                 'case_spec_unchanged': True})
                charged = 2
            else:
                put(folder / 'audit.json', {'status': 'UNKNOWN', 'plan_success': False,
                    'validation_ok': True, 'actual_physics': True})
                put(folder / 'host_preservation.json', {'source_unchanged': True, 'memory_unchanged': True})
                put(folder / 'generated_skills_audit.json', {'library_unchanged': True,
                    'actual_llm_selected_physically_executed_skill_ids': ['new'],
                    'actual_llm_selected_physical_pass_skill_ids': []})
                put(folder / 'action_memory_frozen.json', {'record_count': 4})
                charged = 2 * int(args[args.index('--budget') + 1])
            budget['reserved_calls'] += charged
            budget['remaining_calls'] -= charged
            return {'exit_code': 0}
        def prepare(episode, folder):
            manifest = put(Path(folder) / 'manifest.json', self.manifest)
            result = dict(status='PREPARED', memory_path=str(self.memory), manifest_path=str(manifest),
                original_artifact_sha256={str(self.source): runner.sha(self.source)})
            put(Path(folder) / 'preparation.json', result)
            if mutate_prior:
                put(prior_file, {'unexpected_mutation': True})
            return result
        def memory_union(snapshots, observations, destination):
            result = {'record_count': sum(json.loads(Path(path).read_text())['record_count'] for path in snapshots)}
            put(destination, result)
            return result
        def library_union(snapshots, destination):
            result = {'entries': [row for path in snapshots for row in json.loads(Path(path).read_text())['entries']]}
            put(destination, result)
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'run_recorded', side_effect=process))
            stack.enter_context(patch.object(runner, 'prepare_stability_learning', side_effect=prepare))
            stack.enter_context(patch.object(runner, 'live_inputs', return_value=(self.config, self.ledger, {'max_calls': 39, 'max_usd': .4})))
            stack.enter_context(patch.object(runner, 'budget_summary', side_effect=lambda _: deepcopy(budget)))
            stack.enter_context(patch.object(runner, 'runtime_hashes', return_value={'mock.py': 'a'*64}))
            stack.enter_context(patch.object(runner, 'load_snapshot', side_effect=lambda path: json.loads(Path(path).read_text())))
            stack.enter_context(patch.object(runner, 'retrieve', side_effect=lambda value, **kwargs: value['entries']))
            stack.enter_context(patch.object(runner, 'freeze_cumulative_memory', side_effect=memory_union))
            stack.enter_context(patch.object(runner, 'freeze_skill_union', side_effect=library_union))
            if initial_prior:
                stack.enter_context(patch.object(runner, 'refinement_inputs',
                    return_value=({str(prior_file): prior_hash}, deepcopy(self.manifest))))
            stack.enter_context(patch('local_api_key.key_environment', side_effect=AssertionError('no key')))
            result = runner.execute_cycle(self.root / 'cycle', episode_dir=self.episode,
                initial_memory=self.memory, initial_library=self.library, config_path=self.config, ledger=self.ledger,
                max_proposals=max_proposals, episode_budget=episode_budget,
                refine_from=prior_file.parent if prior_file else None)
        return result, commands

    def test_real_graph_wiring_registered_new_skill_reuse_and_frozen_memory(self):
        result, commands = self.mock_cycle()
        self.assertEqual([c[0] for c in commands], ['run_skill_discovery.py', 'run_generated_skill_case.py', 'run_generated_skills.py'])
        self.assertTrue(result['controlled_reuse_pass'])
        self.assertFalse(result['whole_episode_plan_success'])
        self.assertFalse(result['whole_episode_reuse_pass'])
        self.assertFalse(result['target_is_held_out'])
        self.assertEqual(result['registered_skill_ids'], ['new'])
        full_args = commands[-1][1]
        self.assertEqual(full_args[full_args.index('--budget') + 1], '4')
        self.assertEqual(Path(full_args[full_args.index('--memory') + 1]), self.root / 'cycle/memory_for_reuse.json')
        self.assertEqual(result['budget']['reserved_calls'], 37)

    def test_failed_admission_gets_at_most_two_explicit_refinements_then_no_reuse(self):
        result, commands = self.mock_cycle(candidate_status='NOT_REGISTERED')
        self.assertEqual(len(commands), 3)
        self.assertNotIn('--refine-from', commands[0][1])
        self.assertIn('--refine-from', commands[1][1])
        self.assertIn('--refine-from', commands[2][1])
        self.assertEqual(result['registered_skill_ids'], [])
        self.assertFalse(result['full_dfs_executed'])
        self.assertEqual(result['whole_episode_status'], 'NOT_RUN_NO_REGISTERED_STABILITY_SKILL')

    def test_invalid_generation_stops_without_retries_or_reuse(self):
        result, commands = self.mock_cycle(candidate_status='GENERATION_REJECTED')
        self.assertEqual(len(commands), 1)
        self.assertEqual(result['status'], 'WORKFLOW_COMPLETED_WITH_LIMITATIONS')

    def test_deterministic_response_never_counts_as_actual_llm_reuse(self):
        result, _ = self.mock_cycle(selected=False)
        self.assertFalse(result['controlled_reuse_pass'])

    def test_ten_call_continuation_forwards_prior_without_mutating_default_task(self):
        result, commands = self.mock_cycle(max_proposals=2, episode_budget=3, initial_prior=True)
        self.assertEqual(result['limits']['max_api_attempts'], 10)
        self.assertEqual(result['budget']['reserved_calls'], 38)
        first = commands[0][1]
        self.assertEqual(Path(first[first.index('--refine-from') + 1]), self.root / 'prior')
        full = commands[-1][1]
        self.assertEqual(full[full.index('--budget') + 1], '3')
        self.assertEqual(full[full.index('--state-action-budget') + 1], '3')
        self.assertEqual(runner.TASK['budget'], 4)
        self.assertEqual(runner.LIMITS['max_api_attempts'], 13)
        config = json.loads((self.root / 'cycle/cycle_config.json').read_text())
        self.assertIn(str(self.root / 'prior/generation_packet.json'), config['input_sha256'])

    def test_changed_prior_fails_before_paid_subprocess(self):
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            self.mock_cycle(max_proposals=2, episode_budget=3, initial_prior=True, mutate_prior=True)
        self.assertEqual(self.commands, [])
        self.assertFalse(json.loads((self.root / 'cycle/preservation.json').read_text())['inputs_unchanged'])

    def test_failed_continuation_uses_only_bounded_new_candidates(self):
        result, commands = self.mock_cycle(candidate_status='NOT_REGISTERED',
                                          max_proposals=2, episode_budget=3, initial_prior=True)
        self.assertEqual(len(commands), 2)
        self.assertEqual(result['budget']['reserved_calls'], 31)
        first, second = commands[0][1], commands[1][1]
        self.assertEqual(Path(first[first.index('--refine-from') + 1]), self.root / 'prior')
        self.assertEqual(Path(second[second.index('--refine-from') + 1]), self.root / 'cycle/candidate_01')

    def test_refinement_input_gate_checks_same_ledger_and_prior_completion(self):
        prior = self.root / 'prior'
        config = {'max_calls': 39, 'max_usd': .4}
        originals = {
            'provider_proposal.json': {'origin': 'actual_llm', 'status': 'proposed'},
            'result.json': {'status': 'NOT_REGISTERED'}, 'precommitted_cases.json': self.manifest,
            'config.json': {'mode': 'live', 'api': config, 'ledger': str(self.ledger)},
            'source_preservation.json': {'unchanged': True}, 'generation_packet.json': {}}
        for name, value in originals.items():
            put(prior / name, value)
        debug = prior / 'cases/case_2'
        put(debug / 'program_request.json', {'bound_debug': True})
        with patch('run_skill_discovery.build_refinement_feedback', return_value=[{'source_run': str(prior), 'case_id': 'case_2'}]):
            hashes, manifest = runner.refinement_inputs(prior, self.ledger, config)
            self.assertIn(str(debug / 'program_request.json'), hashes)
            self.assertEqual(manifest, self.manifest)
        for field, value in (('ledger', str(self.root / 'other.sqlite')), ('api', {'max_calls': 40})):
            put(prior / 'config.json', {**originals['config.json'], field: value})
            with patch('run_skill_discovery.build_refinement_feedback', side_effect=AssertionError('invalid scope')), \
                    self.assertRaises(ValueError):
                runner.refinement_inputs(prior, self.ledger, config)
        put(prior / 'config.json', originals['config.json'])
        put(prior / 'source_preservation.json', {'unchanged': False})
        with self.assertRaises(ValueError):
            runner.refinement_inputs(prior, self.ledger, config)


if __name__ == '__main__':
    unittest.main()
