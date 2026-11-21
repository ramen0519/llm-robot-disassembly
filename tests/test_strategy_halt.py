"""Opt-in halt contracts with injected DFS physics; no API, WSL or simulator."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.test_explicit_action_contract import request
from tests.test_explicit_action_planner import make_planner, response_for, result_for
from worker.decision_halt import halt_response, is_halt_response
from worker.explicit_action_contract import canonical_response, validate_response
from worker.path_probe_round import validate_path_round_response
from tools import run_explicit_action_experiment as runner
from worker import run_explicit_actions as worker_runner

ROOT = Path(__file__).absolute().parents[1]


class StrategyHaltTests(unittest.TestCase):
    def test_halt_is_complete_bound_unknown_not_action_space_exhaustion(self):
        req = request()
        original = deepcopy(req)
        response = halt_response(req, 'strategy_candidate_budget_exhausted')
        self.assertEqual(validate_response(req, response), response)
        self.assertTrue(is_halt_response(response))
        self.assertEqual(set(response['candidate_order']), set(req['candidate_ids']))
        self.assertIsNone(response['choice'])
        self.assertIsNone(response['action_id'])
        facts = response['metadata']['decision_halt_v1']
        self.assertEqual(facts['status'], 'UNKNOWN')
        self.assertIsNone(facts['physical_verdict'])
        self.assertFalse(facts['physical_attempt_executed'])
        self.assertFalse(facts['action_space_exhausted'])
        self.assertEqual(req, original)

    def test_halt_envelope_and_claims_are_strict(self):
        req = request()
        response = halt_response(req, 'strategy_unavailable')
        mutations = (
            lambda r: r.update(action_id='invented'),
            lambda r: r.update(choice=canonical_response(req, None)['choice']),
            lambda r: r.update(candidate_order=['1']),
            lambda r: r.update(state_id='stale'),
            lambda r: r.update(fallback_reason='other'),
            lambda r: r['metadata'].update(exhausted=True),
            lambda r: r['metadata'].update(candidate_pruning=True),
            lambda r: r['metadata'].update(physics_bypass=True),
            lambda r: r['metadata'].update(selection_source='canonical_fallback'),
            lambda r: r['metadata']['decision_halt_v1'].update(status='PASS'),
            lambda r: r['metadata']['decision_halt_v1'].update(physical_attempt_executed=True),
            lambda r: r['metadata']['decision_halt_v1'].update(physical_verdict='REJECTED'),
            lambda r: r['metadata']['decision_halt_v1'].update(action_space_exhausted=True),
            lambda r: r['metadata']['decision_halt_v1'].update(extra=True),
        )
        for mutate in mutations:
            value = deepcopy(response)
            mutate(value)
            with self.subTest(response=value), self.assertRaises(ValueError):
                validate_response(req, value)
        for reason in ('', ' leading', 'line\nbreak', 'x'*161, None):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                halt_response(req, reason)

    def test_path_round_accepts_halt_without_deferral_or_exhaustion_claim(self):
        req = request()
        req.update(attempted_actions=[], path_stage_observations=[])
        response = halt_response(req, 'need_new_strategy')
        self.assertEqual(validate_path_round_response(req, response), response)

    def test_opt_in_halt_never_dispatches_or_backtracks(self):
        physics = Mock(side_effect=AssertionError('NO_PHYSICS'))
        planner = make_planner(decision=lambda r: halt_response(r, 'no_valid_strategy'),
            verifier=physics, decision_failure_policy='halt_v1',
            path_selection_policy='path_probe_round_v1')
        planner.plan(budget=4, timeout=None)
        report = planner.report()
        physics.assert_not_called()
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertFalse(report['success'])
        self.assertEqual(report['stop_msg'], 'decision halted: no_valid_strategy')
        self.assertEqual(report['total_n_eval'], 0)
        self.assertEqual(planner.G_path, [['a', 'b', 'c']])
        self.assertEqual(planner.tree.edges, {})
        self.assertFalse(any(e['kind'] == 'backtrack' for e in planner.events))
        self.assertEqual(report['decisions'][0]['decision_source'], 'decision_halt')
        self.assertEqual(report['states'][0]['n_attempts'], 0)
        audit = runner.audit_report(report)
        self.assertTrue(audit['validation_ok'], audit)
        self.assertEqual(audit['decision_halts'], 1)
        self.assertEqual(audit['fallbacks'], 0)
        self.assertFalse(audit['actual_physics'])

    def test_callback_exception_halts_without_canonical_action(self):
        physics = Mock(side_effect=AssertionError('NO_PHYSICS'))
        def failed(_request):
            raise TimeoutError('private callback detail must not be copied')
        planner = make_planner(decision=failed, verifier=physics, decision_failure_policy='halt_v1')
        with patch('explicit_action_planner.canonical_response', side_effect=AssertionError('NO_FALLBACK')):
            planner.plan(budget=4, timeout=None)
        physics.assert_not_called()
        self.assertEqual(planner.stop_msg, 'decision halted: decision_exception:TimeoutError')
        self.assertNotIn('private callback', str(planner.report()))

    def test_bad_response_halts_without_canonical_path_action(self):
        physics = Mock(side_effect=AssertionError('NO_PHYSICS'))
        planner = make_planner(decision=lambda _r: {'wrong': True}, verifier=physics,
            decision_failure_policy='halt_v1', path_selection_policy='path_probe_round_v1')
        with patch('explicit_action_planner.canonical_path_round_response', side_effect=AssertionError('NO_FALLBACK')):
            planner.plan(budget=4, timeout=None)
        physics.assert_not_called()
        self.assertEqual(planner.stop_msg, 'decision halted: decision_exception:ValueError')

    def test_bare_null_is_not_accepted_as_action_exhaustion(self):
        def decide(req):
            value = canonical_response(req, None)
            value.update(choice=None, action_id=None)
            return value
        planner = make_planner(decision=decide, decision_failure_policy='halt_v1')
        planner.plan(budget=4, timeout=None)
        self.assertEqual(planner.report()['total_n_eval'], 0)
        self.assertTrue(planner.stop_msg.startswith('decision halted:'))
        self.assertFalse(any(e['kind'] == 'backtrack' for e in planner.events))

    def test_legacy_worker_rejects_halt_and_preserves_legacy_fallback(self):
        planner = make_planner(decision=lambda req: halt_response(req, 'not_opted_in'))
        planner.plan(budget=1, timeout=None)
        self.assertEqual(planner.report()['total_n_eval'], 1)
        self.assertFalse(is_halt_response(planner.decisions[0]['response']))
        self.assertEqual(planner.decisions[0]['decision_source'], 'canonical_fallback')
        self.assertEqual(planner.report()['decision_failure_policy'], 'legacy_v1')

    def test_halt_after_real_adapter_transition_preserves_partial_tree(self):
        requests = []
        def decide(req):
            requests.append(deepcopy(req))
            return response_for(req) if len(requests) == 1 else halt_response(req, 'terminal_strategy_unavailable')
        planner = make_planner(decision=decide, decision_failure_policy='halt_v1')
        planner.plan(budget=4, timeout=None)
        report = planner.report()
        self.assertEqual(report['total_n_eval'], 1)
        self.assertTrue(report['attempts'][0]['accepted_for_transition'])
        self.assertEqual(len(planner.tree.edges), 1)
        self.assertEqual(planner.G_path[-1], ['b', 'c'])
        self.assertEqual(report['final_pair_calls'], 0)
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertEqual(report['selected_plan'], [])
        self.assertFalse(any(e['kind'] == 'backtrack' for e in planner.events))

    def test_initial_terminal_pair_halt_does_not_choose_other_part(self):
        planner = make_planner(parts=['a', 'b'], decision=lambda req: halt_response(req, 'stop'),
            decision_failure_policy='halt_v1')
        planner.plan(budget=4, timeout=None)
        self.assertEqual(planner.report()['total_n_eval'], 0)
        self.assertEqual(planner.G_path, [['a', 'b']])

    def test_halt_policy_does_not_change_valid_action_success(self):
        planner = make_planner(decision=lambda req: response_for(req), decision_failure_policy='halt_v1')
        planner.plan(budget=2, timeout=None)
        self.assertTrue(planner.report()['success'])
        self.assertEqual(planner.report()['total_n_eval'], 2)
        self.assertEqual(planner.report()['final_pair_calls'], 1)

    def test_repeated_action_halts_instead_of_trying_another_action(self):
        planner = make_planner(decision=lambda req: response_for(req),
            verifier=lambda c, a, p: result_for(c, a, False), decision_failure_policy='halt_v1')
        planner.plan(budget=4, timeout=None)
        self.assertEqual(planner.report()['total_n_eval'], 1)
        self.assertEqual(planner.stop_msg, 'decision halted: decision_exception:ValueError')

    def test_auditor_rejects_halt_with_changed_policy_or_success(self):
        planner = make_planner(decision=lambda req: halt_response(req, 'stop'), decision_failure_policy='halt_v1')
        planner.plan(budget=2, timeout=None)
        for field, value in (('decision_failure_policy', 'legacy_v1'), ('stop_msg', 'solution found')):
            report = planner.report()
            report[field] = value
            self.assertFalse(runner.audit_report(report)['validation_ok'])

    def test_host_and_worker_cli_keep_legacy_default_and_accept_opt_in(self):
        host = ['--run-dir', str(ROOT/'runs/never_run_halt_parser')]
        worker = ['--asap-root', str(ROOT/'external/ASAP'), '--assembly-dir', str(ROOT/'tests'),
                  '--output-dir', str(ROOT/'runs/never_run_halt_parser'), '--run-id', 'parser-only']
        for parser, args in ((runner.parser(), host), (worker_runner.parser(), worker)):
            self.assertEqual(parser.parse_args(args).decision_failure_policy, 'legacy_v1')
            self.assertEqual(parser.parse_args(args+['--decision-failure-policy', 'halt_v1']).decision_failure_policy, 'halt_v1')
        with self.assertRaisesRegex(ValueError, 'unknown_decision_failure_policy'):
            make_planner(decision_failure_policy='unknown')

    def test_host_graph_forwards_halt_policy_and_exports_no_physics_observations(self):
        # Real host graph/audit/export with a fully injected worker subprocess.
        # The worker planner uses real DFS adapter logic and a forbidden verifier.
        with tempfile.TemporaryDirectory(prefix='strategy_halt_fixture_', dir=ROOT/'runs') as folder:
            output = Path(folder)/'run'
            args = runner.parser().parse_args(['--run-dir', str(output), '--mode', 'deterministic',
                                               '--decision-failure-policy', 'halt_v1'])
            physics = Mock(side_effect=AssertionError('NO_PHYSICS'))
            def fake_worker(command, cwd, logdir):
                self.assertEqual(command[command.index('--decision-failure-policy')+1], 'halt_v1')
                logdir.mkdir(parents=True, exist_ok=False)
                planner = make_planner(decision=lambda req: halt_response(req, 'fixture_strategy_exhausted'),
                    verifier=physics, decision_failure_policy='halt_v1')
                planner.plan(budget=2, timeout=None)
                report = planner.report()
                report.update(actual_physics=False, worker_execution='injected_fixture_no_subprocess')
                (output/'worker_result.json').write_text(json.dumps(report), encoding='utf-8')
                return {'exit_code': 0, 'elapsed_seconds': .001, 'fixture_only': True}
            with patch.object(runner, 'run_recorded', side_effect=fake_worker) as process:
                with patch('local_api_key.key_environment', side_effect=AssertionError('NO_KEY')) as key:
                    with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK')) as network:
                        with patch('builtins.print'):
                            self.assertEqual(runner.execute(args), 0)
            physics.assert_not_called()
            key.assert_not_called()
            network.assert_not_called()
            self.assertEqual(process.call_count, 1)
            graph = json.loads((output/'graph_state.json').read_text(encoding='utf-8'))
            snapshot = json.loads((output/'action_memory_frozen.json').read_text(encoding='utf-8'))
            self.assertEqual(graph['status'], 'UNKNOWN')
            self.assertEqual(graph['metrics']['decision_halts'], 1)
            self.assertEqual(graph['metrics']['fallbacks'], 0)
            self.assertEqual(graph['metrics']['memory_records'], 0)
            self.assertFalse(graph['metrics']['actual_physics'])
            self.assertEqual(snapshot['record_count'], 0)
            self.assertFalse((output/'batch_ledger.sqlite').exists())

    def test_worker_changes_remain_python38_compatible(self):
        for name in ('decision_halt.py', 'explicit_action_contract.py', 'explicit_action_planner.py', 'run_explicit_actions.py'):
            ast.parse((ROOT/'worker'/name).read_text(encoding='utf-8'), filename=name, feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
