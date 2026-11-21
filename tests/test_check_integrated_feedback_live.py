"""Saved-evidence diagnostic tests: actual old files, injected network only."""
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from tools import check_integrated_feedback_live as diagnostic
from orchestrator.integrated_path_ranker import IntegratedRankService
from test_integrated_path_ranker import choices
from test_path_repair_ranker import provider


class SavedFeedbackDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='saved_feedback_test_', dir=ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.batch = ROOT / 'runs/integrated_repair_20261018'
        self.source = self.batch / 'natural_fixture'
        self.config = json.loads((self.batch / 'live_config.json').read_text(encoding='utf-8'))
        self.config_file = self.directory / 'config.json'
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        self.ledger = self.directory / 'ledger.sqlite'
        IntegratedRankService(self.config, self.ledger, transport=Mock())
        self.env = patch.dict(os.environ, {'OPENAI_API_KEY': 'fixture-not-a-real-provider-key'})
        self.env.start()
        self.addCleanup(self.env.stop)
        blocked = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network forbidden'))
        mock = blocked.start()
        self.addCleanup(blocked.stop)
        self.addCleanup(mock.assert_not_called)

    def args(self, live=False, suffix='output'):
        return diagnostic.parser().parse_args(['--source-run', str(self.source),
            '--run-dir', str(self.directory / suffix), '--llm-config', str(self.config_file),
            '--ledger', str(self.ledger), '--skill-memory',
            str(ROOT / 'runs/constraint_path_repair_20261018/frozen_path_skills.json'),
            '--allow-live-api' if live else '--prepare-only'])

    def transport(self, invalid=False):
        def choose(url, headers, body, timeout):
            value = choices(body)
            if invalid:
                value['candidate_ids'] = ['invented']
            return provider(value)
        return Mock(side_effect=choose)

    def test_real_F2_lineage_is_preserved_and_only_diagnostic_ids_change(self):
        requests, lineage, files = diagnostic.build_requests(self.source, 'diagnostic:fixture')
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]['request']['request_id'], 'diagnostic:fixture:path')
        self.assertEqual(requests[1]['request']['request_id'], 'diagnostic:fixture:candidate')
        self.assertTrue(all(v['state_id_preserved'] for v in lineage))
        self.assertFalse(lineage[0]['diagnostic_candidate_request_generated'])
        self.assertTrue(lineage[1]['diagnostic_candidate_request_generated'])
        candidate = requests[1]['features']
        self.assertEqual(len(candidate['episode_feedback']), 3)
        self.assertEqual(len(candidate['path_feedback']), 1)
        self.assertFalse(candidate['path_feedback'][0]['outcome']['path_success'])
        self.assertIsNone(candidate['path_feedback'][0]['outcome']['stability_success'])
        self.assertTrue(candidate['constraint_history'])
        self.assertEqual(candidate['episode_feedback'][0]['event_id'].split(':')[0], 'integrated-repair')

    def test_prepare_has_no_key_or_transport_access_and_preserves_shared_ledger(self):
        transport = self.transport()
        keys = Mock(side_effect=AssertionError('prepare must not read key'))
        before = diagnostic.sha(self.ledger)
        args = self.args()
        result = diagnostic.execute(args, transport=transport, key_context_factory=keys)
        self.assertEqual(result['status'], 'PREPARED')
        self.assertEqual(result['new_api_attempts'], 0)
        self.assertEqual(result['budget_delta_calls'], 0)
        self.assertEqual(before, diagnostic.sha(self.ledger))
        transport.assert_not_called()
        keys.assert_not_called()
        previews = diagnostic._jsonlines(args.run_dir / 'request_body_previews.jsonl')
        self.assertEqual(len(previews), 2)
        model = json.loads(previews[1]['body']['input'][0]['content'])['features']
        self.assertEqual(len(model['path_feedback']), 1)
        self.assertTrue(model['frozen_skill_evidence'])
        self.assertTrue(result['source_preserved'])

    def test_injected_two_requests_share_ledger_without_physics_or_DFS_consumption(self):
        transport = self.transport()
        args = self.args(live=True)
        result = diagnostic.execute(args, transport=transport, key_context_factory=lambda: nullcontext('injected'))
        self.assertEqual(result['status'], 'PARTIAL')  # Injected transport is not live Luna.
        self.assertEqual(result['response_count'], 2)
        self.assertEqual(result['transport_attempts_including_injected_tests'], 2)
        self.assertEqual(result['new_api_attempts'], 0)
        self.assertEqual(result['budget_delta_calls'], 2)
        self.assertEqual(transport.call_count, 2)
        self.assertTrue(result['requests_independent'])
        for field in ('new_physics_calls', 'selected_path_executions', 'candidate_orders_consumed_by_DFS'):
            self.assertEqual(result[field], 0)
        self.assertFalse(result['full_DFS'])
        self.assertFalse(result['path_response_caused_stored_outcome'])
        responses = diagnostic._jsonlines(args.run_dir / 'responses.jsonl')
        self.assertTrue(all(r['fallback_reason'] is None for r in responses))
        second_body = transport.call_args_list[1].args[2]
        second = json.loads(second_body['input'][0]['content'])['features']['path_feedback'][0]
        self.assertFalse(second['result']['path_success'])
        self.assertEqual(second['provenance']['repaired_evidence_id'],
            json.loads((self.source / 'integrated_feedback.json').read_text())['completed_path_feedback'][0]['outcome']['repaired_evidence_id'])

    def test_bad_responses_do_not_retry_or_become_physics_success(self):
        transport = self.transport(invalid=True)
        args = self.args(live=True)
        result = diagnostic.execute(args, transport=transport, key_context_factory=lambda: nullcontext('injected'))
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(result['live_valid_responses'], 0)
        self.assertEqual(result['new_physics_calls'], 0)
        self.assertTrue(all(r['fallback_reason'] == 'invalid_permutation'
                            for r in diagnostic._jsonlines(args.run_dir / 'responses.jsonl')))

    def test_timeout_is_not_retried_and_second_independent_request_is_bounded(self):
        transport = Mock(side_effect=TimeoutError('fixture error that must not be copied'))
        args = self.args(live=True)
        result = diagnostic.execute(args, transport=transport, key_context_factory=lambda: nullcontext('injected'))
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(result['budget_delta_calls'], 2)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertNotIn('fixture error', (args.run_dir / 'responses.jsonl').read_text())

    def test_exclusive_output_rejects_reentry_before_any_api(self):
        args = self.args()
        diagnostic.execute(args)
        transport = self.transport()
        args.prepare_only, args.allow_live_api = False, True
        with self.assertRaises(FileExistsError):
            diagnostic.execute(args, transport=transport, key_context_factory=lambda: nullcontext('injected'))
        transport.assert_not_called()

    def test_excluded_E_path_is_rejected_before_read(self):
        args = self.args()
        args.source_run = Path('E:/do-not-access')
        with patch.object(diagnostic, 'read', side_effect=AssertionError('no file read')) as reader:
            with self.assertRaises(ValueError):
                diagnostic.execute(args)
            reader.assert_not_called()

    def test_incorrect_batch_policy_is_rejected_without_key_access(self):
        changed = deepcopy(self.config)
        changed['max_calls'] = 13
        self.config_file.write_text(json.dumps(changed))
        keys = Mock(side_effect=AssertionError('not reached'))
        with self.assertRaises(ValueError):
            diagnostic.execute(self.args(live=True), key_context_factory=keys)
        keys.assert_not_called()


if __name__ == '__main__':
    unittest.main()
