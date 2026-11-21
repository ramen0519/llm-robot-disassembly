"""Rule selection through the hook; fixtures are not physics validation."""
from pathlib import Path
import json
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker'))
from rank_hook import EventLog, installed_hooks
from test_rank_hook import FixtureGenerator, FixturePlanner, rng_digest


class RuleHookTests(unittest.TestCase):
    def test_selected_rule_reaches_hook_and_only_changes_order(self):
        initial_rng = random.getstate()
        try:
            for name in ('id_ascending', 'reverse_native'):
                with self.subTest(ranker=name), tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / 'events.jsonl'
                    log = EventLog(path, 'test-rule-' + name)
                    try:
                        with installed_hooks(FixtureGenerator, FixturePlanner, log,
                                             'rank', rng_digest, ranker_name=name):
                            generator = FixtureGenerator()
                            actual = list(generator.generate_candidate_part(['0', '1', '2', '3', '6']))
                    finally:
                        log.close()
                    events = [json.loads(line) for line in path.read_text().splitlines()]
                    generation = next(e for e in events if e['kind'] == 'generation')
                    decision = next(e for e in events if e['kind'] == 'rank_decision')
                    native = generation['native_order']
                    expected = sorted(native) if name == 'id_ascending' else list(reversed(native))
                    self.assertEqual(actual, expected)
                    self.assertEqual(generation['applied_order'], expected)
                    self.assertEqual(decision['ranker_name'], name)
                    self.assertEqual(decision['rng_before'], decision['rng_after'])
                    self.assertIsNone(decision['fallback_reason'])
                    self.assertEqual(generator.calls, 1)
        finally:
            random.setstate(initial_rng)

    def test_rule_cannot_silently_change_trace_or_noop_control(self):
        for mode in ('trace', 'noop'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                log = EventLog(Path(temp) / 'events.jsonl', 'test-guard')
                try:
                    with self.assertRaises(ValueError):
                        with installed_hooks(FixtureGenerator, FixturePlanner, log,
                                             mode, rng_digest, ranker_name='id_ascending'):
                            self.fail('invalid mode must not patch classes')
                finally:
                    log.close()


if __name__ == '__main__':
    unittest.main()
