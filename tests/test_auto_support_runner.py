"""Normal producer defaults and post-plan audit; injected physics returns only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker'))
from run_path_only_proposal import audit_selected_steps, parse_args


class AutoSupportRunnerTests(unittest.TestCase):
    def options(self, *extra):
        return parse_args(['--assembly-dir', 'cad', '--output-dir', 'out', *extra])

    def audit(self, cap, seconds=10):
        parts = ['5', '4', '3', '2', '0']
        proposal = {'steps': [dict(attempt_id=i+1, part_move=p, child=parts[i+1:], pose=None)
                              for i, p in enumerate(parts[:-1])]}
        original = deepcopy(proposal['steps'])
        calls = []
        def check(asset, cad, held, free, **kwargs):
            calls.append((list(held), list(free)))
            return (True, None) if not free else (False, free[:1])
        audit_selected_steps(proposal, support_cap=cap, audit_seconds=seconds, asset_folder='assets',
                             assembly_dir='cad', check_stable=check, emit=lambda *a, **kw: None)
        self.assertEqual(original, [{k:v for k,v in s.items() if k != 'stability'} for s in proposal['steps']])
        return [s['stability'] for s in proposal['steps']], calls

    def test_default_follows_the_remainder_and_starts_each_check_unheld(self):
        for args in ([], ['--support-cap', 'auto']):
            options = self.options(*args)
            self.assertEqual(options.support_cap, 'auto')
            rows, _ = self.audit(options.support_cap)
            self.assertEqual([r['effective_support_cap'] for r in rows], [4,3,2,1])
            self.assertEqual([len(r['held_parts']) for r in rows], [4,3,2,1])
            self.assertTrue(all(r['trials'][0]['parts_fix'] == [] for r in rows))

    def test_explicit_cap_keeps_previous_behavior(self):
        rows, _ = self.audit(self.options('--support-cap','2').support_cap)
        self.assertEqual([r['effective_support_cap'] for r in rows], [2,2,2,1])
        self.assertEqual([r['up_to_cap_status'] for r in rows], ['REJECTED','REJECTED','PASS','PASS'])
        zero, calls = self.audit(self.options('--support-cap','0').support_cap)
        self.assertTrue(all(r['up_to_cap_status']=='REJECTED' for r in zero))
        self.assertTrue(all(not fixed for fixed, _ in calls))

    def test_expired_audit_retains_cap_but_has_no_witness(self):
        rows, calls = self.audit(self.options().support_cap, 0)
        self.assertFalse(calls)
        self.assertEqual([r['effective_support_cap'] for r in rows], [4,3,2,1])
        self.assertTrue(all(r['up_to_cap_status']=='NOT_RUN' and r['held_parts'] is None for r in rows))
