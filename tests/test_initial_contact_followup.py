"""Synthetic subprocess wiring tests. No external API or actual physics."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_initial_contact_followup as runner


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


class InitialContactFollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runner.ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.memory = put(self.root / 'memory.json', {'record_count': 29})
        self.report = put(self.root / 'report.json', {'original': True})
        self.library = put(self.root / 'library.json', {'entries': [{'strategy_id': 'old'}]})
        self.old_config = {'max_calls': 39, 'max_usd': .4, 'model': 'gpt-6-luna', 'enabled': True}
        self.new_config = {**self.old_config, 'max_calls': 6, 'max_usd': .1}
        self.old_config_path = put(self.root / 'generation_config.json', self.old_config)
        self.new_config_path = put(self.root / 'reuse_config.json', self.new_config)
        self.old_ledger = self.root / 'old.sqlite'
        self.old_ledger.write_bytes(b'fixture old ledger')
        self.new_ledger = self.root / 'new.sqlite'
        self.prior_file = put(self.root / 'prior/generation_packet.json', {'debug_only': True})
        self.manifest = {'synthesis_failure_scope': 'initial_disconnected',
            'experience_report': str(self.report), 'experience_memory': str(self.memory),
            'cases': [dict(case_id='case_%d' % index, phase=phase, source_task='07009',
                report=str(self.report), attempt_index=index, trigger_kind='stability_failed',
                expected_diagnostic_phase='initial_disconnected')
                for index, phase in ((1, 'debug'), (3, 'admission'), (7, 'admission'))]}
        self.manifest_path = put(self.root / 'manifest.json', self.manifest)
        self.before = {
            'generation': dict(max_calls=39, max_usd='.4', reserved_calls=37, request_count=37,
                pending_requests=0, unknown_outcomes=0, remaining_calls=2, remaining_estimated_usd='.1'),
            'reuse': dict(max_calls=6, max_usd='.1', reserved_calls=0, request_count=0,
                pending_requests=0, unknown_outcomes=0, remaining_calls=6, remaining_estimated_usd='.1')}

    def args(self):
        return ['--run-dir', str(self.root / 'preview'), '--manifest', str(self.manifest_path),
                '--prior-candidate', str(self.prior_file.parent), '--initial-memory', str(self.memory),
                '--initial-library', str(self.library)]

    def test_authorization_policy_diff_is_only_approved_caps(self):
        runner.validate_policies(self.old_config, self.new_config)
        for key, value in (('max_calls', 7), ('max_usd', .11), ('model', 'other-model'), ('enabled', False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_policies(self.old_config, {**self.new_config, key: value})
        with self.assertRaises(ValueError):
            runner.validate_policies({**self.old_config, 'max_calls': 40}, self.new_config)

    def test_missing_reuse_ledger_is_not_created_by_preflight(self):
        value = runner.policy_budget(self.new_ledger, 'reuse', allow_missing=True, required_calls=6)
        self.assertEqual(value['reserved_calls'], 0)
        self.assertEqual(value['status'], 'NOT_STARTED')
        self.assertFalse(self.new_ledger.exists())

    def test_preview_does_not_open_key_budget_or_subprocess(self):
        with patch.object(runner, 'execute_followup', side_effect=AssertionError('no live')), \
                patch('local_api_key.key_environment', side_effect=AssertionError('no key')), \
                patch.object(runner, 'budget_summary', side_effect=AssertionError('no ledger')), \
                redirect_stdout(io.StringIO()):
            runner.main(self.args())
            with self.assertRaises(FileExistsError):
                runner.main(self.args())
        value = json.loads((self.root / 'preview/preview.json').read_text())
        self.assertEqual(value['limits']['generation_max_new_api_attempts'], 2)
        self.assertEqual(value['limits']['reuse_max_new_api_attempts'], 6)
        self.assertEqual(value['task']['budget'], 3)

    def test_forbidden_path_and_incomplete_live_authorizations(self):
        args = self.args()
        args[args.index('--prior-candidate') + 1] = 'E:/never-inspect'
        with patch.object(runner, 'checked', side_effect=AssertionError('filesystem')), self.assertRaises(ValueError):
            runner.main(args)
        with patch.object(runner, 'execute_followup') as execute, self.assertRaises(ValueError):
            runner.main(self.args() + ['--mode', 'live', '--allow-live-api'])
        execute.assert_not_called()

    def mock_followup(self, *, admission_at=1, generation_rejected=False, executed=True,
                      passed=True, mutate_other=False, changed_input=False, reused_ledger=False,
                      physical_calls=3, mismatch_manifest=False):
        commands = []
        self.commands = commands
        budgets = deepcopy(self.before)
        if reused_ledger:
            self.new_ledger.write_bytes(b'used reuse ledger')
            budgets['reuse'].update(reserved_calls=1, request_count=1, remaining_calls=5)
        prior_manifest = deepcopy(self.manifest)
        if mismatch_manifest:
            prior_manifest['cases'][0]['attempt_index'] = 9
        def live_inputs(config_path, ledger):
            return Path(config_path), Path(ledger), json.loads(Path(config_path).read_text())
        def summary(path):
            return deepcopy(budgets['generation' if Path(path) == self.old_ledger else 'reuse'])
        generation_count = 0
        def process(command, cwd, command_dir):
            nonlocal generation_count
            tool, args = Path(command[3]).name, command[4:]
            commands.append((tool, args))
            folder = Path(args[args.index('--run-dir') + 1])
            folder.mkdir()
            Path(command_dir).mkdir()
            if tool == 'run_skill_discovery.py':
                generation_count += 1
                self.assertEqual(Path(args[args.index('--ledger') + 1]), self.old_ledger)
                self.assertEqual(Path(args[args.index('--llm-config') + 1]), self.old_config_path)
                admitted = admission_at == generation_count
                status = 'GENERATION_REJECTED' if generation_rejected else 'REGISTERED_SCOPED' if admitted else 'NOT_REGISTERED'
                put(folder / 'result.json', {'status': status, 'candidate_id': 'new',
                    'generation_origin': 'live_llm', 'admission': {'admitted': admitted}})
                put(folder / 'source_preservation.json', {'unchanged': True})
                put(folder / 'skills_frozen.json', {'entries': [{'strategy_id': 'new'}]})
                put(folder / 'provider_proposal.json', {'origin': 'actual_llm', 'status': 'proposed'})
                kind, charged = 'generation', 1
                if mutate_other:
                    self.new_ledger.write_bytes(b'wrong generation budget target')
            else:
                self.assertEqual(tool, 'run_generated_skills.py')
                self.assertEqual(Path(args[args.index('--ledger') + 1]), self.new_ledger)
                self.assertEqual(Path(args[args.index('--llm-config') + 1]), self.new_config_path)
                self.new_ledger.write_bytes(b'new reuse ledger')
                put(folder / 'audit.json', {'status': 'UNKNOWN', 'plan_success': False,
                    'validation_ok': True, 'actual_physics': True, 'verifier_calls': physical_calls, 'fallbacks': 0})
                put(folder / 'host_preservation.json', {'source_unchanged': True, 'memory_unchanged': True})
                put(folder / 'generated_skills_audit.json', {'library_unchanged': True,
                    'actual_llm_selected_physically_executed_skill_ids': ['new'] if executed else ['old'],
                    'actual_llm_selected_physical_pass_skill_ids': ['new'] if executed and passed else ['old']})
                put(folder / 'action_memory_frozen.json', {'record_count': physical_calls})
                kind, charged = 'reuse', 6
            budgets[kind]['reserved_calls'] += charged
            budgets[kind]['request_count'] += charged
            budgets[kind]['remaining_calls'] -= charged
            return {'exit_code': 0}
        def packet(manifest, run_id):
            if changed_input:
                put(self.report, {'mutated': True})
            return {'evidence': [{'fixture': True}]}
        def memory_union(paths, observations, destination):
            result = {'record_count': sum(json.loads(Path(p).read_text()).get('record_count', 0) for p in paths)}
            put(destination, result)
            return result
        def library_union(paths, destination):
            result = {'entries': [row for p in paths for row in json.loads(Path(p).read_text())['entries']]}
            put(destination, result)
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'live_inputs', side_effect=live_inputs))
            stack.enter_context(patch.object(runner, 'budget_summary', side_effect=summary))
            stack.enter_context(patch.object(runner, 'runtime_hashes', return_value={'fixture.py': 'a'*64}))
            stack.enter_context(patch.object(runner, 'load_snapshot', side_effect=lambda p: json.loads(Path(p).read_text())))
            stack.enter_context(patch.object(runner, 'retrieve', side_effect=lambda value, **kwargs: value['entries']))
            stack.enter_context(patch.object(runner, 'refinement_inputs',
                return_value=({str(self.prior_file): runner.sha(self.prior_file)}, prior_manifest)))
            stack.enter_context(patch.object(runner, 'synthesis_packet', side_effect=packet))
            stack.enter_context(patch.object(runner, 'run_recorded', side_effect=process))
            stack.enter_context(patch.object(runner, 'freeze_cumulative_memory', side_effect=memory_union))
            stack.enter_context(patch.object(runner, 'freeze_skill_union', side_effect=library_union))
            stack.enter_context(patch('local_api_key.key_environment', side_effect=AssertionError('no key')))
            return runner.execute_followup(self.root / 'followup', manifest_path=self.manifest_path,
                prior_candidate=self.prior_file.parent, initial_memory=self.memory, initial_library=self.library,
                generation_config=self.old_config_path, generation_ledger=self.old_ledger,
                reuse_config=self.new_config_path, reuse_ledger=self.new_ledger)

    def test_registered_skill_routes_generation_old_reuse_new_and_preserves_origin(self):
        result = self.mock_followup()
        self.assertEqual([c[0] for c in self.commands], ['run_skill_discovery.py', 'run_generated_skills.py'])
        self.assertEqual(result['generation_new_api_calls'], 1)
        self.assertEqual(result['reuse_new_api_calls'], 6)
        self.assertEqual(result['status'], 'INITIAL_CONTACT_SKILL_REUSE_PASS')
        self.assertFalse(result['free_dfs_plan_success'])
        self.assertFalse(result['fixed_selection_injected'])
        args = self.commands[-1][1]
        self.assertEqual(args[args.index('--budget') + 1], '3')
        self.assertNotIn('--case-spec', args)
        self.assertNotIn('--fixture', args)

    def test_second_candidate_uses_first_debug_and_still_only_old_budget(self):
        result = self.mock_followup(admission_at=2)
        self.assertEqual(result['generation_new_api_calls'], 2)
        first, second = self.commands[0][1], self.commands[1][1]
        self.assertEqual(Path(first[first.index('--refine-from') + 1]), self.prior_file.parent)
        self.assertEqual(Path(second[second.index('--refine-from') + 1]), self.root / 'followup/candidate_01')
        self.assertEqual(result['budgets']['generation']['reserved_calls'], 39)
        self.assertEqual(result['budgets']['reuse']['reserved_calls'], 6)

    def test_both_candidates_fail_reuse_ledger_remains_absent(self):
        result = self.mock_followup(admission_at=None)
        self.assertEqual(len(self.commands), 2)
        self.assertFalse(self.new_ledger.exists())
        self.assertEqual(result['reuse_new_api_calls'], 0)
        self.assertEqual(result['budgets']['reuse']['status'], 'NOT_STARTED')
        self.assertEqual(result['status'], 'NOT_REGISTERED')

    def test_rejected_generation_stops_without_automatic_retry(self):
        result = self.mock_followup(admission_at=None, generation_rejected=True)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(result['generation_new_api_calls'], 1)
        self.assertFalse(self.new_ledger.exists())

    def test_no_new_skill_execution_cannot_borrow_old_skill_pass(self):
        result = self.mock_followup(executed=False)
        self.assertEqual(result['status'], 'NOT_OBSERVED_WITHIN_BUDGET')
        self.assertFalse(result['free_dfs_reuse_pass'])

    def test_wrong_stage_ledger_mutation_stops_flow(self):
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            self.mock_followup(mutate_other=True)
        self.assertEqual(len(self.commands), 1)

    def test_used_reuse_ledger_and_changed_predeclared_cases_fail_before_execution(self):
        with self.assertRaises(ValueError):
            self.mock_followup(reused_ledger=True)
        self.assertEqual(self.commands, [])

    def test_changed_manifest_case_fails_before_execution(self):
        with self.assertRaisesRegex(ValueError, 'must_remain_identical'):
            self.mock_followup(mismatch_manifest=True)
        self.assertEqual(self.commands, [])

    def test_input_mutation_fails_before_first_paid_subprocess(self):
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            self.mock_followup(changed_input=True)
        self.assertEqual(self.commands, [])

    def test_more_than_three_actual_actions_fails_result_gate(self):
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            self.mock_followup(physical_calls=4)
        self.assertFalse((self.root / 'followup/followup_result.json').exists())


if __name__ == '__main__':
    unittest.main()
