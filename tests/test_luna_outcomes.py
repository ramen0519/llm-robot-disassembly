"""Synthetic CLI outcome collection only; never launch WSL, a model or physics."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_luna_experiment as runner


class LunaOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='asap_luna_outcomes_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'runs' / 'fixture'
        self.output.mkdir(parents=True)
        root = patch.object(runner, 'ROOT', self.root)
        root.start()
        self.addCleanup(root.stop)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        self.network = network.start()
        self.addCleanup(network.stop)
        self.addCleanup(self.network.assert_not_called)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def make_run(self, name):
        run = self.output / name
        run.mkdir()
        return run

    def test_validator_unknown_is_not_a_process_error(self):
        for status, exit_code, expected in (('PASS', 0, 'PASS'), ('UNKNOWN', 1, 'UNKNOWN'),
                                           ('ERROR', 1, 'ERROR'), ('PASS', 1, 'ERROR'),
                                           ('UNKNOWN', 0, 'ERROR'), ('UNKNOWN', 7, 'ERROR')):
            with self.subTest(status=status, exit_code=exit_code):
                self.assertEqual(runner.validation_status({'status': status}, exit_code), expected)
        self.assertEqual(runner.validation_status(None, 0), 'ERROR')

    def test_unknown_row_preserves_unfinished_plan_and_raw_metrics(self):
        run = self.make_run('native_off')
        self.write(run / 'parsed_result.json', dict(status='UNKNOWN', plan_success=False,
            raw_stats=dict(success=False, sequence=None, stop_msg='timeout', time=45.2),
            tree_fingerprint_sha256='a' * 64))
        self.write(run / 'execute/command.json', dict(exit_code=0, elapsed_seconds=46))
        events = [dict(kind='run_start'), dict(kind='verifier_result', status='UNKNOWN'),
                  dict(kind='run_end', status='returned_from_upstream')]
        (run / 'events.jsonl').write_text('\n'.join(json.dumps(item) for item in events))
        row, _, _, _ = runner.run_outcome(run, run.name, {'status': 'UNKNOWN', 'phase': 'END'})
        self.assertEqual(row['graph_status'], 'UNKNOWN')
        self.assertEqual(row['plan_status'], 'UNKNOWN')
        self.assertFalse(row['plan_success'])
        self.assertEqual(row['stop_msg'], 'timeout')
        self.assertEqual(row['verifier_calls'], 1)
        self.assertEqual(row['unknown_calls'], 1)
        self.assertIsNone(row['sequence'])
        self.assertFalse(row['partial'])
        self.assertEqual(row['artifact_issues'], [])

    def test_failed_run_without_parsed_result_remains_visible(self):
        run = self.make_run('native_on')
        self.write(run / 'execute/command.json', dict(exit_code=1, elapsed_seconds=3))
        events = [dict(kind='run_start'), dict(kind='verifier_result', status='ERROR'),
                  dict(kind='run_end', status='ERROR')]
        (run / 'events.jsonl').write_text('\n'.join(json.dumps(item) for item in events))
        row, _, _, _ = runner.run_outcome(run, run.name, {'status': 'ERROR', 'phase': 'END'})
        self.assertEqual(row['graph_status'], 'ERROR')
        self.assertIsNone(row['plan_status'])
        self.assertEqual(row['process_exit'], 1)
        self.assertEqual(row['error_calls'], 1)
        self.assertTrue(row['partial'])
        self.assertIn('parsed_result.json', [item['artifact'] for item in row['artifact_issues']])

    def test_incomplete_jsonl_retains_parseable_observations_and_issue(self):
        run = self.make_run('reverse_off')
        (run / 'events.jsonl').write_text(
            json.dumps(dict(kind='run_start')) + '\n' +
            json.dumps(dict(kind='verifier_result', status='PASS')) + '\n{"kind":')
        raw_before = (run / 'events.jsonl').read_bytes()
        row, _, _, _ = runner.run_outcome(run, run.name, {'status': 'ERROR', 'phase': None})
        self.assertEqual(row['verifier_calls'], 1)
        self.assertTrue(row['partial'])
        self.assertIsNone(row['run_end_status'])
        self.assertIn(3, [issue.get('line') for issue in row['artifact_issues']])
        self.assertEqual((run / 'events.jsonl').read_bytes(), raw_before)

    def test_empty_attempt_uses_null_metrics_not_fabricated_zero(self):
        run = self.make_run('reverse_on')
        row, _, _, _ = runner.run_outcome(run, run.name, {'status': 'ERROR', 'phase': None})
        self.assertTrue(row['partial'])
        self.assertIsNone(row['process_exit'])
        self.assertIsNone(row['rank_requests'])
        self.assertIsNone(row['verifier_calls'])

    def test_archive_selects_complete_pass_unknown_error_and_excludes_incomplete(self):
        roots = [self.make_run(name) for name in ('pass', 'unknown', 'incomplete', 'error')]
        for run in roots:
            (run / 'events.jsonl').write_text('fixture raw ' + run.name)
        hashes = {run.name: hashlib.sha256((run / 'events.jsonl').read_bytes()).hexdigest() for run in roots}
        def collect(run, context, event_ids, run_ids):
            if run.name == 'incomplete':
                event_ids.add('error:result')
                run_ids.add('error')
                raise ValueError('incomplete trace envelope')
            self.assertNotIn(run.name, run_ids)
            self.assertNotIn(run.name + ':result', event_ids)
            run_ids.add(run.name)
            event_ids.add(run.name + ':result')
            status = run.name.upper()
            return [dict(record_id=run.name + ':result', status=status)], dict(
                run_end_status='ERROR' if status == 'ERROR' else 'returned_from_upstream',
                status_counts={status: 1})
        archiver = Mock(return_value={'record_count': 3})
        selected = runner.archive_completed_runs(roots, {'fixture': True}, self.output, collect, archiver)
        self.assertEqual([Path(item['run_root']).name for item in selected['accepted']], ['pass', 'unknown', 'error'])
        self.assertEqual(selected['excluded'][0]['reason'], 'incomplete trace envelope')
        self.assertEqual(selected['excluded'][0]['error_type'], 'ValueError')
        self.assertEqual(archiver.call_args.args[0], [roots[0], roots[1], roots[3]])
        self.assertTrue((self.output / 'archive_selection.json').is_file())
        self.assertTrue((self.output / 'archive_result.json').is_file())
        self.assertEqual(hashes, {run.name: hashlib.sha256((run / 'events.jsonl').read_bytes()).hexdigest() for run in roots})

    def test_no_valid_episode_still_writes_exclusion_diagnostics(self):
        run = self.make_run('broken')
        collector = Mock(side_effect=FileNotFoundError('private-path'))
        archiver = Mock()
        selection = runner.archive_completed_runs([run], {}, self.output, collector, archiver)
        self.assertEqual(selection['accepted'], [])
        self.assertEqual(selection['excluded'][0]['reason'], 'FileNotFoundError')
        self.assertNotIn('private-path', json.dumps(selection))
        archiver.assert_not_called()
        self.assertTrue((self.output / 'archive_selection.json').is_file())

    def test_roundtrip_rejects_duplicate_missing_or_malformed_responses(self):
        decision = dict(request=dict(request_id='r1', candidate_ids=['1', '2']), applied_order=['2', '1'])
        response = dict(request_id='r1', candidate_ids=['2', '1'])
        self.assertTrue(runner.roundtrip_valid([decision], [response]))
        self.assertFalse(runner.roundtrip_valid([decision, decision], [response, response]))
        self.assertFalse(runner.roundtrip_valid([decision], []))
        self.assertFalse(runner.roundtrip_valid([{'kind': 'rank_decision'}], [response]))


if __name__ == '__main__':
    unittest.main()
