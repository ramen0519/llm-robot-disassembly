"""CLI -> LangGraph -> file IPC -> strategy provider -> DFS -> memory export.

Worker process and physics are injected. No WSL, API, real key or physics.
"""
import json
from copy import deepcopy
from pathlib import Path
import sys
import subprocess
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch

from tests.test_hierarchical_planner import policy, request
from tests.test_explicit_action_planner import make_planner, result_for
from tests.test_strategy_service import wire
from tools import run_hierarchical_episode as hierarchy
from worker.explicit_action_ipc import ExplicitActionClient

host = sys.modules[hierarchy.execute.__module__]

ROOT = Path(__file__).absolute().parents[1]
TRAIN = ROOT/'runs/support_learning_live_20261109_1155/revalidation_v2'


class StrategyRunnerTests(unittest.TestCase):
    def test_full_offline_runner_exports_fixture_actions_and_counts_no_live_calls(self):
        self._run_fixture()

    def test_opt_in_review_reaches_provider_journal_and_export_without_extra_calls(self):
        self._run_fixture('skill_selection_review_v1')

    def test_missing_reason_is_audited_without_extra_calls_or_changing_execution(self):
        self._run_fixture('skill_selection_review_v1', review_missing=True)

    def _run_fixture(self, skill_review_policy='legacy_v1', review_missing=False):
        with tempfile.TemporaryDirectory(prefix='strategy_runner_OFFLINE_', dir=ROOT/'runs') as folder:
            home = Path(folder); output = home/'episode'
            (home/'policy.json').write_text(json.dumps(policy()), encoding='utf-8')
            req = request(); req['remaining_parts'] = ['a', 'b', 'c']; req['candidate_ids'] = ['a', 'b', 'c']
            templates = dict(scout=[dict(steps=wire(req, parts=['a', 'b'])['steps'])], planner=[])
            if skill_review_policy != 'legacy_v1' and not review_missing:
                templates['scout'][0]['skill_review'] = []
            (home/'provider.json').write_text(json.dumps(templates), encoding='utf-8')
            argv = ['--run-dir', str(output), '--mode', 'deterministic',
                '--hierarchy-policy', str(home/'policy.json'), '--hierarchy-ledger', str(home/'global.sqlite'),
                '--planner-ledger', str(home/'planner.sqlite'), '--skill-library', str(TRAIN/'skills_frozen.json'),
                '--lesson-library', str(TRAIN/'lessons_frozen.json'), '--lesson-runtime',
                '--provider-fixture', str(home/'provider.json'), '--provider-contract', 'explicit_action_provider_v5',
                '--path-selection-policy', 'path_probe_round_v1', '--fallback-policy', 'path_stage_v1',
                '--plan-provider-contract', 'strategy_plan_v1', '--decision-failure-policy', 'halt_v1',
                '--recovery-support-policy', 'preserve_existing_v1',
                '--planning-feedback-policy', 'repair_evidence_v2',
                '--transport-failure-policy', 'latch_on_error_v1']
            if skill_review_policy != 'legacy_v1':
                argv += ['--skill-review-policy', skill_review_policy]
            def worker_process(command, cwd, logdir):
                logdir.mkdir(parents=True, exist_ok=False)
                self.assertEqual(command[command.index('--decision-failure-policy')+1], 'halt_v1')
                client = ExplicitActionClient(output/'ipc', timeout_seconds=20)
                def injected_verifier(context, choice, path):
                    return dict(result_for(context, choice), chosen=deepcopy(choice),
                        hidden_direction_search_calls=0, hidden_pose_search_calls=0,
                        hidden_support_search_calls=0, path_check_calls=1,
                        stability_check_calls=1, two_leg_controller_calls=0)
                worker = make_planner(decision=client, verifier=injected_verifier, path_selection_policy='path_probe_round_v1',
                    fallback_policy='path_stage_v1', decision_failure_policy='halt_v1')
                worker.run_id = command[command.index('--run-id')+1]
                worker.plan(budget=4)
                report = worker.report()
                report.update(actual_physics=False, worker_execution='INJECTED_NO_PHYSICS_NO_WSL')
                (output/'worker_result.json').write_text(json.dumps(report), encoding='utf-8')
                return dict(exit_code=0, elapsed_seconds=.01, fixture_only=True)
            process = Mock(side_effect=worker_process)
            forbidden = []
            local_vm_calls = []
            original_run = subprocess.run
            def only_local_vm(*args, **kwargs):
                command = args[0] if args else kwargs.get('args')
                if (type(command) is list and len(command) == 5 and command[0] == sys.executable
                        and command[1:4] == ['-I', '-S', '-B']
                        and Path(command[4]).absolute() == ROOT/'orchestrator/python_skill_child.py'):
                    local_vm_calls.append(True)
                    return original_run(*args, **kwargs)
                forbidden.append(''.join(traceback.format_stack()[-9:]))
                raise AssertionError('NO_SUBPROCESS')
            with patch.dict(hierarchy.execute.__globals__, run_recorded=process), \
                 patch.dict(ExplicitActionClient.__init__.__globals__['checked_path'].__globals__, IPC_ROOT=ROOT/'runs'), \
                 patch('local_api_key.key_environment', side_effect=AssertionError('NO_KEY')) as key, \
                 patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK')) as network, \
                 patch('subprocess.run', side_effect=only_local_vm) as subprocess_run, \
                 patch('builtins.print'):
                code = hierarchy.main(argv)
                diagnosis = {name: json.loads((output/name).read_text(encoding='utf-8'))
                    for name in ('graph_state.json', 'audit.json', 'host_preservation.json', 'hierarchy_audit.json')
                    if (output/name).exists()}
                if 'graph_state.json' in diagnosis:
                    diagnosis['graph_state.json'].pop('config', None)
                diagnosis['test_interception'] = dict(worker_calls=process.call_count,
                    local_vm_calls=len(local_vm_calls), forbidden_subprocess_calls=len(forbidden), stacks=forbidden)
                self.assertEqual(code, 0, diagnosis)
            key.assert_not_called(); network.assert_not_called(); self.assertEqual(process.call_count, 1)
            self.assertEqual(forbidden, [])
            report = json.loads((output/'hierarchy_audit.json').read_text())
            graph = json.loads((output/'graph_state.json').read_text())
            self.assertFalse(report['actual_physics'])
            self.assertTrue(report['provider_fixture'])
            self.assertEqual(report['budget']['live_api_attempts'], 0)
            self.assertEqual(report['physical_attempts'], 2)  # two injected verifier attempts
            self.assertEqual(report['admitted_plans'], 1)
            self.assertEqual(graph['status'], 'PASS')
            self.assertEqual(graph['metrics']['memory_records'], 2)
            self.assertEqual(graph['metrics']['fallbacks'], 0)
            reviews = json.loads((output/'skill_selection_reviews.json').read_text(encoding='utf-8'))
            decisions = json.loads((output/'hierarchy_decisions.json').read_text(encoding='utf-8'))
            plans = json.loads((output/'hierarchy_plans.json').read_text(encoding='utf-8'))
            self.assertEqual(len(plans), 1)
            self.assertEqual(reviews['policy'], skill_review_policy)
            self.assertFalse(reviews['selection_forced'])
            self.assertEqual(reviews['action_contexts'][1]['status'], 'no_new_model_choice')
            self.assertNotIn('skill_selection_review', decisions[1]['response']['metadata'])
            if review_missing:
                self.assertEqual(reviews['action_contexts'][0]['status'], 'invalid_model_review')
                error = decisions[0]['response']['metadata']['skill_selection_review_error']
                self.assertEqual(error['status'], 'invalid_or_missing')
                self.assertNotIn('skill_selection_review', decisions[0]['response']['metadata'])
                self.assertEqual(decisions[1]['response']['metadata']['plan_skill_selection_review_error'], error)
                self.assertEqual(reviews['plan_reviews'][0]['reason'], 'invalid_or_missing_review')
                self.assertEqual(reviews['plan_reviews'][0]['review_error'], error)
                self.assertEqual(report['skill_selection_reviews_recorded'], 0)
            elif skill_review_policy != 'legacy_v1':
                review = decisions[0]['response']['metadata']['skill_selection_review']
                self.assertEqual(review['reviews'], [])
                self.assertEqual(reviews['action_contexts'][0]['status'], 'new_model_review')
                self.assertEqual(report['skill_selection_reviews_recorded'], 1)
                self.assertEqual(reviews['plan_reviews'][0]['review'], review)
                self.assertEqual(decisions[1]['response']['metadata']['plan_skill_selection_review'], review)
                journal = [json.loads(p.read_text(encoding='utf-8'))
                    for p in (output/'hierarchy_journal').glob('*.json')]
                self.assertTrue(any(row.get('response', {}).get('metadata', {}).get(
                    'skill_selection_review') == review for row in journal))
            else:
                self.assertNotIn('skill_selection_review', decisions[0]['response']['metadata'])
                self.assertEqual(reviews['plan_reviews'][0]['reason'], 'legacy_not_requested')
                self.assertEqual(report['skill_selection_reviews_recorded'], 0)

    def test_strategy_cli_requires_opt_in_halt_and_guards_before_any_execution(self):
        argv = ['--run-dir', str(ROOT/'runs/not_created_strategy_guard'),
            '--hierarchy-policy', 'missing.json', '--hierarchy-ledger', 'missing.sqlite',
            '--planner-ledger', 'missing2.sqlite', '--skill-library', 'missing3.json',
            '--plan-provider-contract', 'strategy_plan_v1']
        with patch.object(hierarchy, 'execute', side_effect=AssertionError('NO_EXECUTION')) as execute:
            with self.assertRaisesRegex(ValueError, 'strategy_requires_halt'):
                hierarchy.main(argv)
            execute.assert_not_called()

    def test_review_cli_rejects_non_strategy_contract_before_any_file_or_execution(self):
        argv = ['--run-dir', str(ROOT/'runs/not_created_review_guard'),
            '--hierarchy-policy', 'missing.json', '--hierarchy-ledger', 'missing.sqlite',
            '--planner-ledger', 'missing2.sqlite', '--skill-library', 'missing3.json',
            '--skill-review-policy', 'skill_selection_review_v1']
        with patch.object(hierarchy, 'execute', side_effect=AssertionError('NO_EXECUTION')) as execute:
            with self.assertRaisesRegex(ValueError, 'skill_review_requires_strategy_plan_contract'):
                hierarchy.main(argv)
            execute.assert_not_called()


if __name__ == '__main__': unittest.main()
