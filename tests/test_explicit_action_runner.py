"""Host audit/IPC/authorization tests. No WSL, key read, API, or real physics.

Audit mutation tests must reject inconsistent saved artifacts; fixture success
is deliberately labelled fake physics and never demonstrates CAD feasibility.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'worker'))
sys.path.insert(0, str(ROOT/'tests'))
import run_explicit_action_experiment as runner
from explicit_action_ipc import ExplicitActionClient
from remote_rank_client import request_filename, write_json_new
from orchestrator.file_service import FileRankServer
from orchestrator.explicit_action_service import ExplicitActionService
from test_explicit_action_contract import request
from test_explicit_action_planner import make_planner, result_for
from worker.explicit_action_contract import canonical_response


def fake_report():
    def verifier(context, choice, path):
        raw = result_for(context, choice)
        raw.update(chosen=deepcopy(choice), hidden_direction_search_calls=0,
            hidden_pose_search_calls=0, hidden_support_search_calls=0,
            path_check_calls=1, stability_check_calls=1, two_leg_controller_calls=0)
        return raw
    planner = make_planner(verifier=verifier)
    planner.plan(budget=2, timeout=None)
    return planner.report()


class ExplicitActionAuditTests(unittest.TestCase):
    def rejected(self, report):
        audit = runner.audit_report(report)
        self.assertFalse(audit['validation_ok'], audit)

    def test_valid_fake_plan_not_reported_actual_physics_or_llm(self):
        audit = runner.audit_report(fake_report())
        self.assertTrue(audit['validation_ok'])
        self.assertFalse(audit['actual_physics']); self.assertEqual(audit['llm_choices'], 0)
        self.assertEqual(audit['sequence'], ['a', 'b'])

    def test_invalid_attempt_pass_rejected(self):
        report = fake_report(); report['attempts'][0]['raw']['path_success'] = False
        self.rejected(report)

    def test_hidden_support_search_rejected(self):
        report = fake_report(); report['attempts'][0]['raw']['hidden_support_search_calls'] = 1
        self.rejected(report)

    def test_late_pass_transition_rejected(self):
        report = fake_report(); report['attempts'][0]['deadline_exceeded'] = True
        self.rejected(report)

    def test_attempt_count_and_budget_checked(self):
        report = fake_report(); report['budget'] = 1
        self.rejected(report)
        report = fake_report(); report['total_n_eval'] = 3
        self.rejected(report)

    def test_selected_plan_must_reference_same_actual_attempt(self):
        report = fake_report()
        report['selected_plan'][0]['raw']['path_success'] = False
        self.rejected(report)

    def test_selected_attempt_cannot_invent_call_identity(self):
        report = fake_report(); report['selected_plan'][0]['call_id'] = 'unobserved'
        self.rejected(report)

    def test_selected_attempt_status_must_be_pass(self):
        report = fake_report(); report['selected_plan'][0]['status'] = 'UNKNOWN'
        self.rejected(report)

    def test_success_with_empty_plan_is_audited_error_not_exception(self):
        report = fake_report(); report['selected_plan'] = []
        self.rejected(report)

    def test_false_final_pair_count_rejected(self):
        report = fake_report(); report['final_pair_calls'] = 0
        self.rejected(report)

    def test_forged_terminal_pair_boolean_rejected(self):
        report = fake_report(); report['attempts'][0]['terminal_pair'] = True
        self.rejected(report)

    def test_pass_status_without_success_rejected(self):
        report = fake_report(); report['success'] = False
        self.rejected(report)

    def test_unknown_status_with_success_rejected(self):
        report = fake_report(); report['status'] = 'UNKNOWN'
        self.rejected(report)

    def test_declared_continuity_error_cannot_be_ignored(self):
        report = fake_report(); report['selected_plan_continuity_errors'] = ['missing_edge']
        self.rejected(report)

    def test_selected_action_id_must_match_contract_identity(self):
        report = fake_report(); report['attempts'][0]['action_id'] = 'action:'+'0'*64
        self.rejected(report)

    def test_legitimate_bounded_unknown_remains_valid_unknown(self):
        planner = make_planner(verifier=lambda c, a, p: result_for(c, a, False))
        planner.plan(budget=1, timeout=None)
        audit = runner.audit_report(planner.report())
        self.assertTrue(audit['validation_ok'], audit)
        self.assertEqual(audit['status'], 'UNKNOWN')
        self.assertFalse(audit['plan_success'])

    def test_selected_plan_cannot_start_after_omitting_initial_transition(self):
        report = fake_report(); report['selected_plan'] = report['selected_plan'][1:]
        self.rejected(report)

    def test_native_stats_sequence_must_match_selected_chain(self):
        report = fake_report(); report['stats']['sequence'] = ['b', 'a']
        self.rejected(report)

    def test_accepted_wrong_pose_action_or_support_is_rejected(self):
        # Mutate both copies identically, so only scope/physics condition checking
        # can reject it, rather than selected-plan object consistency alone.
        for field, value in (('pose', [[1,0,0,99],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
                             ('action', [0,1,0]), ('parts_fix', ['c'])):
            report = fake_report()
            report['attempts'][0]['raw'][field] = deepcopy(value)
            report['selected_plan'][0]['raw'][field] = deepcopy(value)
            self.rejected(report)


class ExplicitActionRunnerSafetyTests(unittest.TestCase):
    def args(self, *extra):
        output = ROOT/'runs'/('explicit_runner_safety_'+uuid.uuid4().hex)
        return runner.parser().parse_args(['--run-dir', str(output), *extra])

    def reject_without_external_work(self, args, configuration=None):
        with patch.object(runner, 'run_recorded', side_effect=AssertionError('WSL_forbidden')) as run:
            with patch.object(runner, 'read', return_value=configuration) as read:
                with patch('local_api_key.key_environment', side_effect=AssertionError('key_forbidden')) as key:
                    with self.assertRaises(ValueError): runner.execute(args)
                key.assert_not_called()
            run.assert_not_called()
        self.assertFalse(args.run_dir.exists())

    def test_live_mode_requires_explicit_flag(self):
        self.reject_without_external_work(self.args('--mode', 'llm'))

    def test_live_flag_cannot_enable_deterministic_mode(self):
        self.reject_without_external_work(self.args('--allow-live-api'))

    def test_offline_mode_rejects_nonzero_api_cap_before_key_or_worker(self):
        args = self.args('--llm-config', str(ROOT/'configs'/'unused.json'))
        self.reject_without_external_work(args, {'enabled': False, 'max_calls': 1, 'max_usd': 0})

    def test_live_incomplete_config_rejected(self):
        args = self.args('--mode', 'llm', '--allow-live-api', '--llm-config', str(ROOT/'configs'/'unused.json'))
        self.reject_without_external_work(args, {'enabled': True, 'max_calls': 1, 'max_usd': .01})

    def test_fixture_missing_file_argument_rejected(self):
        self.reject_without_external_work(self.args('--mode', 'fixture'))

    def test_unknown_assembly_and_invalid_budget_rejected(self):
        self.reject_without_external_work(self.args('--assembly-dir', '/opt/asap-research/unknown'))
        self.reject_without_external_work(self.args('--budget', '0'))

    def test_excluded_path_rejected_without_stat(self):
        args = self.args('--assembly-dir', 'E:/do-not-open')
        # checked(run-dir) legitimately examines the C workspace; the excluded
        # assembly string must then be rejected before directory enumeration.
        with patch.object(runner, 'run_recorded', side_effect=AssertionError('WSL_forbidden')):
            with self.assertRaises(ValueError): runner.execute(args)
        self.assertFalse(args.run_dir.exists())

    def test_adapter_rejects_outer_inner_request_identity_mismatch(self):
        r = request(); service = Mock(); service.timeout_seconds = 5
        service.handle.return_value = canonical_response(r, None)
        envelope = {'request': {'request_id': 'other', 'state_id': 's1', 'run_id': 'fixture',
            'assembled_parts': r['remaining_parts'], 'candidate_ids': r['candidate_ids'], 'base_part': r['base_part']},
            'features': {'explicit_action': r}, 'timeout_seconds': 5}
        with self.assertRaises(ValueError): runner.MailboxAdapter(service).rank(envelope)
        service.handle.assert_not_called()


class ExplicitActionIPCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='explicit_ipc_fixture_')
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)

    def test_full_file_roundtrip_with_labelled_deterministic_control(self):
        client = ExplicitActionClient(self.root/'ipc', timeout_seconds=3, allow_test_root=True)
        service = ExplicitActionService({'enabled': False}, self.root/'unused.sqlite', mode='deterministic')
        server = FileRankServer(self.root/'ipc', 'fixture', 'pipeline', lambda *a, **k: None,
            allow_test_root=True, rank_service=runner.MailboxAdapter(service))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client, request())
            deadline = time.monotonic()+2
            while not future.done() and time.monotonic() < deadline:
                server.serve_pending(); time.sleep(.01)
            response = future.result(timeout=3)
        self.assertEqual(response['metadata']['selection_source'], 'deterministic_control')
        self.assertFalse(response['metadata']['llm_live'])
        self.assertFalse((self.root/'unused.sqlite').exists())
        self.assertEqual(len(list((self.root/'ipc/requests').glob('*.json'))), 1)

    def test_outer_response_binding_mismatch_rejected(self):
        client = ExplicitActionClient(self.root/'ipc', timeout_seconds=1, allow_test_root=True)
        r = request(); outer = {'request_id': r['request_id'], 'state_id': r['state_id'],
            'metadata': {'envelope_sha256': 'bad', 'explicit_action_response': canonical_response(r, None)}}
        write_json_new(self.root/'ipc/responses'/request_filename(r['request_id']), outer, True)
        with self.assertRaisesRegex(ValueError, 'envelope_mismatch'): client(r)

    def test_timeout_does_not_invent_response(self):
        client = ExplicitActionClient(self.root/'ipc', timeout_seconds=.01, allow_test_root=True)
        with self.assertRaises(TimeoutError): client(request())


class ExplicitActionWorkflowExportTests(unittest.TestCase):
    def test_host_graph_exports_and_freezes_fake_observations_without_key_or_wsl(self):
        # The existing graph/checkpoint and memory code run for real. The entire
        # worker subprocess is injected, so this is not a physics execution.
        with tempfile.TemporaryDirectory(prefix='explicit_runner_fixture_', dir=ROOT/'runs') as folder:
            output = Path(folder)/'run'
            args = runner.parser().parse_args(['--run-dir', str(output), '--mode', 'deterministic'])
            def fake_worker(command, cwd, logdir):
                logdir.mkdir(parents=True, exist_ok=False)
                report = fake_report()
                report.update(actual_physics=False, worker_execution='injected_fixture_no_subprocess')
                (output/'worker_result.json').write_text(json.dumps(report), encoding='utf-8')
                return {'exit_code': 0, 'elapsed_seconds': .001, 'fixture_only': True}
            with patch.object(runner, 'run_recorded', side_effect=fake_worker) as process:
                with patch('local_api_key.key_environment', side_effect=AssertionError('key_forbidden')) as key:
                    with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden')) as network:
                        code = runner.execute(args)
                    network.assert_not_called()
                key.assert_not_called()
            self.assertEqual(code, 0); self.assertEqual(process.call_count, 1)
            graph = json.loads((output/'graph_state.json').read_text(encoding='utf-8'))
            snapshot = json.loads((output/'action_memory_frozen.json').read_text(encoding='utf-8'))
            self.assertEqual(graph['metrics']['memory_export'], 'PASS')
            self.assertEqual(snapshot['record_count'], 2)
            self.assertFalse(snapshot['physics_reuse_allowed'])
            self.assertFalse(graph['metrics']['actual_physics'])
            self.assertTrue(all(r['result']['verifier_origin'] != 'actual_ASAP_RedMax' for r in snapshot['records']))
            self.assertFalse((output/'batch_ledger.sqlite').exists())


if __name__ == '__main__': unittest.main()
