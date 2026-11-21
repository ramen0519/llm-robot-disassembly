"""Real LangGraph stages with local fixtures: zero API and physics calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from orchestrator.learning_cycle_workflow import LearningCycleWorkflow


class LearningCycleWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'cycle'
        self.identity = {'mode': 'offline_fixture', 'source_sha256': 'a'*64}
        self.calls = []

    def stages(self):
        def callback(name):
            def run(prior):
                self.calls.append(name)
                artifact = self.root/(name+'.json')
                artifact.write_text(json.dumps({'fixture': name}), encoding='utf-8')
                return {'status': 'FIXTURE_ONLY', 'prior_count': len(prior),
                        'artifacts': {'fixture': str(artifact)}}
            return run
        return {name: callback(name) for name in ('source', 'prepare', 'learn', 'target', 'finish')}

    def flow(self, stages=None, identity=None):
        return LearningCycleWorkflow(self.root, self.identity if identity is None else identity,
            self.stages() if stages is None else stages, allow_test_root=True)

    def test_actual_graph_order_and_completed_reuse(self):
        flow = self.flow()
        first = flow.run()
        self.assertEqual(first['status'], 'COMPLETE')
        self.assertEqual(self.calls, list(flow.stages))
        self.assertEqual(first['executed_stages'], list(flow.stages))
        self.assertEqual(first['completed_artifact_reuse'], [])
        self.assertEqual([r['prior_count'] for r in first['stage_results'].values()], list(range(5)))
        again = self.flow().run()
        self.assertEqual(first['stage_results'], again['stage_results'])
        self.assertEqual(self.calls, list(flow.stages))
        self.assertEqual(again['executed_stages'], [])
        self.assertEqual(again['completed_artifact_reuse'], list(flow.stages))
        self.assertFalse((flow.journal/'session.lock').exists())
        events = [json.loads(l) for l in (flow.journal/'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['kind']=='completed_artifact_reuse' for e in events), 5)
        self.assertEqual(len({e['event_id'] for e in events}), len(events))

    def test_callback_cannot_mutate_prior_results(self):
        def mutate(prior):
            prior['source']['status'] = 'CORRUPTED'
            prior['fabricated'] = {}
            return {'status': 'NO_ELIGIBLE'}
        stages = {'source': lambda _: {'status': 'UNKNOWN'}, 'target': mutate}
        result = self.flow(stages).run()
        self.assertEqual(result['stage_results']['source']['status'], 'UNKNOWN')
        self.assertNotIn('fabricated', result['stage_results'])

    def test_failure_statuses_are_values_not_plan_success_claims(self):
        stages = {name: lambda _, name=name: {'status': name}
                  for name in ('source', 'learn', 'target')}
        stages['learn'] = lambda _: {'status': 'NOT_REGISTERED'}
        result = self.flow(stages).run()
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['stage_results']['learn']['status'], 'NOT_REGISTERED')
        self.assertIn('target', result['stage_results'])

    def test_callback_exception_sanitized_and_never_retried(self):
        secret = 'SECRET_EXCEPTION_DO_NOT_STORE'
        def broken(_):
            self.calls.append('broken')
            raise ValueError(secret)
        flow = self.flow({'source': broken, 'target': lambda _: self.fail('must not run')})
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed') as caught:
            flow.run()
        self.assertNotIn(secret, str(caught.exception))
        for path in flow.journal.rglob('*.json*'):
            self.assertNotIn(secret, path.read_text(encoding='utf-8'))
        failed = json.loads((flow.journal/'source/stage_failed.json').read_text())
        self.assertEqual(failed['error_type'], 'ValueError')
        with self.assertRaisesRegex(RuntimeError, 'unsupported_partial_resume'):
            flow.run()
        self.assertEqual(self.calls, ['broken'])

    def test_changed_identity_or_order_rejected_before_callback(self):
        flow = self.flow()
        flow.run()
        for changed in ({**self.identity, 'mode': 'live'}, {'mode': 'offline_fixture'}):
            with self.assertRaisesRegex(ValueError, 'cycle_identity_mismatch'):
                self.flow(identity=changed).run()
        with self.assertRaisesRegex(ValueError, 'cycle_identity_mismatch'):
            self.flow(dict(reversed(list(self.stages().items())))).run()
        self.assertEqual(len(self.calls), 5)

    def test_changed_or_missing_artifact_prevents_reuse(self):
        flow = self.flow()
        flow.run()
        artifact = self.root/'source.json'
        artifact.write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'completed_artifact_hash_mismatch'):
            flow.run()
        artifact.unlink()
        with self.assertRaisesRegex(RuntimeError, 'completed_artifact_missing'):
            flow.run()
        self.assertEqual(len(self.calls), 5)

    def test_prior_artifact_modified_by_later_callback_fails(self):
        stages = self.stages()
        def corrupt(_):
            (self.root/'source.json').write_text('mutated')
            return {}
        stages['prepare'] = corrupt
        flow = self.flow(stages)
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            flow.run()
        self.assertFalse((flow.journal/'prepare/stage_completed.json').exists())

    def test_completed_record_tampering_rejected(self):
        flow = self.flow()
        flow.run()
        path = flow.journal/'source/stage_completed.json'
        value = json.loads(path.read_text())
        value['result']['status'] = 'FAKE_PASS'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'completed_stage_record_mismatch'):
            flow.run()

    def test_partial_json_and_unknown_files_fail_closed(self):
        flow = self.flow()
        flow.run()
        path = flow.journal/'source/stage_completed.json'
        path.write_text('{')
        with self.assertRaisesRegex(RuntimeError, 'invalid_or_partial_stage_journal'):
            flow.run()
        (flow.journal/'unknown').write_text('marker')
        with self.assertRaisesRegex(RuntimeError, 'unknown_partial_cycle_state'):
            flow.run()

    def test_stale_lock_never_cleared(self):
        flow = self.flow()
        flow.journal.mkdir(parents=True)
        lock = flow.journal/'session.lock'
        lock.write_text('stale')
        with self.assertRaisesRegex(RuntimeError, 'cycle_active_or_stale_session_lock'):
            flow.run()
        self.assertEqual(lock.read_text(), 'stale')
        self.assertEqual(self.calls, [])

    def test_partial_event_log_blocks_reuse(self):
        flow = self.flow()
        flow.run()
        with (flow.journal/'events.jsonl').open('ab') as stream:
            stream.write(b'{"partial":')
        with self.assertRaisesRegex(RuntimeError, 'invalid_or_partial_event_journal'):
            flow.run()
        self.assertEqual(len(self.calls), 5)

    def test_keyboard_interrupt_is_persisted_and_not_relaunched(self):
        def stop(_):
            self.calls.append('source')
            raise KeyboardInterrupt('secret interruption text')
        flow = self.flow({'source': stop})
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            flow.run()
        failed = json.loads((flow.journal/'source/stage_failed.json').read_text())
        self.assertEqual(failed['error_type'], 'KeyboardInterrupt')
        self.assertNotIn('secret', json.dumps(failed))
        with self.assertRaisesRegex(RuntimeError, 'unsupported_partial_resume'):
            flow.run()
        self.assertEqual(self.calls, ['source'])

    def test_concurrent_invocation_fails_without_second_callback(self):
        entered, release = threading.Event(), threading.Event()
        results = []
        def block(_):
            self.calls.append('source')
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test_timeout')
            return {}
        flow = self.flow({'source': block})
        thread = threading.Thread(target=lambda: results.append(flow.run()))
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(RuntimeError, 'cycle_active_or_stale_session_lock'):
                self.flow({'source': block}).run()
        finally:
            release.set()
            thread.join(5)
        self.assertEqual(self.calls, ['source'])
        self.assertEqual(len(results), 1)

    def test_complete_prefix_reused_without_reexecuting(self):
        flow = self.flow()
        original = flow._event
        def stop_after_completion(kind, stage=None, **fields):
            original(kind, stage, **fields)
            if kind == 'stage_completed' and stage == 'source':
                raise RuntimeError('test_process_boundary')
        with patch.object(flow, '_event', side_effect=stop_after_completion):
            with self.assertRaisesRegex(RuntimeError, 'test_process_boundary'):
                flow.run()
        result = self.flow().run()
        self.assertEqual(self.calls, ['source', 'prepare', 'learn', 'target', 'finish'])
        self.assertEqual(result['completed_artifact_reuse'], ['source'])

    def test_noncontiguous_or_empty_stage_state_rejected(self):
        flow = self.flow()
        flow.run()
        # Simulate missing earlier journal, keeping later evidence untouched.
        with patch.object(flow, '_complete', side_effect=lambda stage, prior: None if stage=='source' else {}):
            with self.assertRaisesRegex(RuntimeError, 'noncontiguous_stage_journal'):
                flow.run()
        other = self.root/'other'
        second = LearningCycleWorkflow(other, self.identity, {'source': lambda _: {}}, allow_test_root=True)
        second.journal.mkdir(parents=True)
        second._write_new(second.journal/'identity.json', second.binding)
        (second.journal/'source').mkdir()
        with self.assertRaisesRegex(RuntimeError, 'unsupported_partial_resume'):
            second.run()

    def test_paths_rejected_lexically_before_filesystem(self):
        for location in ('E:/forbidden', 'e:\\forbidden', '/mnt/e', '/mnt/e/probe', '//server/share', '../outside'):
            with self.subTest(location=location), patch.object(Path, 'lstat', side_effect=AssertionError('FS touched')):
                with self.assertRaises(ValueError):
                    LearningCycleWorkflow(location, self.identity, {'source': lambda _: {}}, allow_test_root=True)

    def test_artifact_forbidden_path_stops_before_stat(self):
        flow = self.flow({'source': lambda _: {'artifacts': {'bad': 'E:/forbidden'}}})
        with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
            flow.run()
        failed = json.loads((flow.journal/'source/stage_failed.json').read_text())
        self.assertEqual(failed['error_type'], 'ValueError')

    def test_read_only_project_config_can_be_bound_but_journal_stays_in_runs(self):
        config = Path(__file__).absolute().parents[1]/'configs/skill_discovery_cases_v1.json'
        flow = self.flow({'source': lambda _: {'artifacts': {'config': str(config)}}})
        result = flow.run()
        self.assertEqual(result['stage_results']['source']['artifacts']['config'], str(config))
        self.assertEqual(flow.run()['completed_artifact_reuse'], ['source'])
        with self.assertRaises(ValueError):
            LearningCycleWorkflow(config.parent/'forbidden_journal', self.identity, {'source': lambda _: {}})

    def test_invalid_finite_json_and_result_types(self):
        for identity in ({'nan': float('nan')}, {'object': object()}, {}, []):
            with self.subTest(identity_type=type(identity).__name__):
                with self.assertRaises(ValueError):
                    self.flow(identity=identity)
        for index, value in enumerate((None, [], {'nan': float('nan')}, {'artifacts': {'x': 1}})):
            flow = LearningCycleWorkflow(self.root/str(index), self.identity,
                {'source': lambda _, value=value: value}, allow_test_root=True)
            with self.assertRaisesRegex(RuntimeError, 'learning_cycle_stage_failed'):
                flow.run()

    def test_stage_name_and_constructor_copy(self):
        for stages in ({}, {'../escape': lambda _: {}}, {'a': None}, {'a.b': lambda _: {}}):
            with self.assertRaises(ValueError):
                self.flow(stages)
        identity = deepcopy(self.identity)
        stages = {'source': lambda _: {'status': 'NOT_RUN'}}
        flow = self.flow(stages, identity)
        identity['mode'] = 'changed'
        stages['source'] = lambda _: self.fail('external mutation')
        self.assertEqual(flow.run()['stage_results']['source']['status'], 'NOT_RUN')


if __name__ == '__main__':
    unittest.main()
