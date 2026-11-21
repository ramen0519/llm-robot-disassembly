"""No provider or real credentials: new run isolation and explicit opt-in."""
import unittest
from unittest.mock import Mock, patch
from tools import start_skill_id_retest as retest
from tools import start_skill_two_episode as original
from tests import test_start_skill_two_episode as base_tests


class IdRetestLauncherTests(base_tests.TwoEpisodeLauncherTests):
    def setUp(self):
        super().setUp()
        self.output = self.root / 'runs' / retest.RUN_NAME
        self.original_launch = original.launch
        redirect = patch.object(original, 'launch', side_effect=lambda *a, **kw:
            self.original_launch(*a, **kw, run_name=retest.RUN_NAME))
        redirect.start()
        self.addCleanup(redirect.stop)

    def test_old_run_remains_unchanged_and_old_budget_is_not_reused(self):
        old = self.root / 'runs' / original.RUN_NAME
        old.mkdir()
        (old / 'sentinel.txt').write_text('preserved', encoding='utf-8')
        self.assertEqual(self.invoke(True), 0)
        self.assertEqual((old / 'sentinel.txt').read_text(), 'preserved')
        import json
        task = json.loads((self.output / 'task_start.json').read_text())
        self.assertFalse(task['prior_six_call_budget_reused'])
        self.assertEqual(self.original_launch(True, self.root, self.recorder), 3)


class IdRetestEntrypointTests(unittest.TestCase):
    def test_wrapper_binds_new_name_and_preserves_explicit_authorization(self):
        with patch.object(retest.base, 'launch', return_value=0) as runner:
            recorder = Mock()
            self.assertEqual(retest.launch(False, retest.ROOT, recorder), 0)
            runner.assert_called_once_with(False, retest.ROOT, recorder, run_name=retest.RUN_NAME)
            self.assertNotEqual(retest.RUN_NAME, original.RUN_NAME)

    def test_name_cannot_escape_runs_before_any_key_read(self):
        with patch.object(original, 'key_environment') as key:
            for name in ('../escape', 'C:/escape', '', 'name/child'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    original.launch(True, run_name=name)
            key.assert_not_called()


if __name__ == '__main__':
    unittest.main()
