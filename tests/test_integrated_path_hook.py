"""Adapter contract tests; saved real evidence is read, physics/API are not run."""
from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
try:
    import numpy
except ImportError:
    numpy = None
if numpy is not None:
    import integrated_path_hook as module


@unittest.skipIf(numpy is None, 'NumPy unavailable on host; execute in ASAP Python3.8')
class IntegratedPathHookTests(unittest.TestCase):
    def setUp(self):
        run = ROOT / 'runs/constraint_path_repair_20261018/live_source'
        self.outcome = json.loads((run / 'path_outcomes.json').read_text())['outcomes'][0]
        records = json.loads((run / 'constraint_memory.json').read_text())['records']
        self.hook = module.IntegratedPathHook.__new__(module.IntegratedPathHook)
        self.hook.records = records
        self.hook.completed_feedback = []
        self.hook.outcomes = []
        self.hook.output = run  # write_new is patched in context tests.
        self.emitted = []
        self.hook.emit = lambda kind, **kw: self.emitted.append((kind, kw))
        self.hook.session = SimpleNamespace(context=deepcopy(self.outcome['scope']['task_context']))

    def test_success_is_published_with_all_candidates_and_source_constraints(self):
        self.hook._publish(deepcopy(self.outcome))
        item = self.hook.completed_feedback[0]
        self.assertEqual(item['outcome'], self.outcome)
        self.assertTrue(item['constraint_references'])
        self.assertTrue(all(r['origin'] == 'current_episode_history' for r in item['constraint_references']))
        self.assertTrue(all(r['whole_part_verdict'] == 'UNKNOWN' and not r['reuse_verdict']
                            for r in item['constraint_references']))

    def test_failed_path_keeps_stability_unexecuted(self):
        value = deepcopy(self.outcome)
        value['path_success'] = False
        self.hook._publish(value)
        saved = self.hook.completed_feedback[0]['outcome']
        self.assertFalse(saved['feasible'])
        self.assertIsNone(saved['stability_success'])
        self.assertIsNone(saved['parts_fix'])

    def test_unfinished_stability_not_published_as_pass(self):
        value = deepcopy(self.outcome)
        value['stability_success'] = None
        with self.assertRaises(ValueError):
            self.hook._publish(value)
        self.assertEqual(self.hook.completed_feedback, [])

    def test_copy_is_immutable_and_run_cap_is_enforced(self):
        value = deepcopy(self.outcome)
        self.hook._publish(value)
        value['feasible'] = False
        self.assertTrue(self.hook.completed_feedback[0]['outcome']['feasible'])
        with self.assertRaises(ValueError):
            self.hook._publish(deepcopy(self.outcome))

    def test_proxy_supplies_both_routes_and_preserves_cache(self):
        class Client:
            features = {}
            cache = {'a': 'existing'}
            def __call__(self, request):
                return deepcopy(self.features)
        client = Client()
        proxy = module.FeedbackClient(client, self.hook)
        self.hook._publish(deepcopy(self.outcome))
        for protocol in ('independent_rank_v1', 'constraint_path_repair_v1'):
            proxy.features = {'protocol': protocol}
            scope = self.outcome['scope']
            result = proxy(SimpleNamespace(request_id='test:' + protocol,
                assembled_parts=scope['assembled_parts'], candidate_ids=[p for p in scope['assembled_parts']
                    if p != self.hook.session.context['task']['base_part']]))
            self.assertEqual(result['protocol'], protocol)
            self.assertEqual(len(result['path_feedback']), 1)
            self.assertTrue(result['constraint_history'])
        self.assertIs(proxy.cache, client.cache)

    def test_constraints_survive_when_no_repair_was_executed(self):
        class Client:
            features = {'protocol': 'independent_rank_v1'}
            def __call__(self, request):
                return deepcopy(self.features)
        proxy = module.FeedbackClient(Client(), self.hook)
        scope = self.outcome['scope']
        result = proxy(SimpleNamespace(request_id='test:no_repair', assembled_parts=scope['assembled_parts'],
            candidate_ids=[p for p in scope['assembled_parts'] if p != scope['base_part']]))
        self.assertEqual(result['path_feedback'], [])
        self.assertTrue(result['constraint_history'])

    def test_wrapper_restores_client_and_simulate_after_exception(self):
        class Sequence:
            def _simulate(self, a, b=None):
                raise RuntimeError('synthetic failure')
        original = Sequence._simulate
        client = object()
        self.hook.session.client = client
        @contextmanager
        def installed(hook, planner):
            yield hook
        with patch.object(module.ConstraintPathHook, 'installed', installed), patch.object(module, 'write_new') as write:
            with self.assertRaises(RuntimeError):
                with self.hook.installed(Sequence):
                    self.assertEqual(inspect.signature(Sequence._simulate), inspect.signature(original))
                    Sequence()._simulate('x')
            self.assertIs(Sequence._simulate, original)
            self.assertIs(self.hook.session.client, client)
            self.assertEqual(write.call_count, 1)


if __name__ == '__main__':
    unittest.main()
