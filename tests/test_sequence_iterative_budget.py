"""Pure budget arithmetic and stage separation; API/process/filesystem-free."""
from decimal import Decimal
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from orchestrator import sequence_iterative_budget as budget


class SequenceIterativeBudgetTests(unittest.TestCase):
    def setUp(self):
        for target in ('urllib.request.OpenerDirector.open','subprocess.Popen','pathlib.Path.open'):
            guard=patch(target,side_effect=AssertionError('PURE_BUDGET_NO_IO'))
            mock=guard.start();self.addCleanup(guard.stop);self.addCleanup(mock.assert_not_called)

    def test_initial_includes_learning_after_both_rounds(self):
        value=budget.stage_budget('initial')
        self.assertEqual(value['rounds'],[1,2])
        self.assertEqual(value['free_episodes'],10)
        self.assertEqual(value['learning_rounds'],10)
        self.assertEqual(value['roles']['sol']['max_calls'],250)
        self.assertEqual(value['roles']['luna']['max_calls'],40)
        self.assertEqual(value['max_calls'],290)
        self.assertEqual(Decimal(value['max_usd']),Decimal('41'))
        self.assertEqual(Decimal(value['conservative_reservation_total']),Decimal('39.5264'))
        self.assertEqual(value['physics']['max_validation_runs'],40)

    def test_continuation_is_separate_and_full_not_executable(self):
        continued=budget.stage_budget('continuation');full=budget.stage_budget('full')
        self.assertEqual(continued['rounds'],[3,4,5])
        self.assertEqual((continued['max_calls'],Decimal(continued['max_usd'])),(435,Decimal('61.5')))
        self.assertEqual((full['max_calls'],Decimal(full['max_usd'])),(725,Decimal('102.5')))
        self.assertEqual(full['roles']['luna']['max_calls'],100)
        self.assertEqual(full['physics']['max_validation_runs'],100)
        self.assertFalse(full['execution_stage'])
        with self.assertRaisesRegex(ValueError,'execution_stage'):
            budget.validate_stage_request('full',725,'102.5')

    def test_total_is_sum_of_separate_stages(self):
        first=budget.stage_budget('initial');second=budget.stage_budget('continuation');full=budget.stage_budget('full')
        for key in ('max_calls','free_episodes','learning_rounds'):
            self.assertEqual(full[key],first[key]+second[key])
        for key in ('max_usd','conservative_reservation_total'):
            self.assertEqual(Decimal(full[key]),Decimal(first[key])+Decimal(second[key]))
        for key in ('max_outer_seconds','max_worker_seconds','max_path_checks','max_stability_checks','max_composite_attempts'):
            self.assertEqual(full['physics'][key],first['physics'][key]+second['physics'][key])

    def test_exact_reservations_and_no_discount_or_retry_assumption(self):
        self.assertEqual(budget.role_reservation('sol'),Decimal('.155648'))
        self.assertEqual(budget.role_reservation('luna'),Decimal('.01536'))
        for stage in ('initial','continuation','full'):
            value=budget.stage_budget(stage)
            self.assertLess(Decimal(value['conservative_reservation_total']),Decimal(value['max_usd']))
            self.assertEqual(value['automatic_http_retries'],0)
            self.assertEqual(value['automatic_episode_retries'],0)
            self.assertFalse(value['cache_discounts_assumed'])
            self.assertFalse(value['live_enabled'])
            self.assertFalse(value['authorization_inferred'])
            self.assertFalse(value['continuation_is_automatic'])

    def test_physics_bounds_include_all_candidate_versions(self):
        first=budget.stage_budget('initial')['physics']
        self.assertEqual(first['max_outer_seconds'],65400)
        self.assertEqual(first['max_outer_hhmmss'],'18:10:00')
        self.assertEqual(first['max_worker_seconds'],60000)
        self.assertEqual(first['max_path_checks'],7680)
        self.assertEqual(first['max_stability_checks'],7680)
        full=budget.stage_budget('full')['physics']
        self.assertEqual(full['max_outer_hhmmss'],'45:25:00')
        self.assertEqual(full['max_worker_hhmmss'],'41:40:00')
        self.assertEqual(full['max_path_checks'],19200)
        self.assertEqual(first['settings']['validation']['outer_timeout_seconds'],660)

    def test_valid_numeric_contract_does_not_claim_authorization(self):
        result=budget.validate_stage_request('continuation',435,'61.50')
        self.assertFalse(result['authorization_inferred'])
        self.assertFalse(result['checkpoint_validated'])
        self.assertEqual(budget.validate_stage_request('initial',1,Decimal('.01'))['max_usd'],'0.01')

    def test_invalid_nonfinite_and_unknown_scalars_rejected(self):
        for value in (True,False,None,{},[],float('nan'),float('inf'),-float('inf'),
                      'NaN','Infinity','-Infinity','sNaN','unknown',0,-1,'-0.01'):
            with self.subTest(usd=value),self.assertRaises(ValueError):
                budget.validate_stage_request('initial',290,value)
        for calls in (True,False,0,-1,291,290.0,'290',None,{},float('inf')):
            with self.subTest(calls=calls),self.assertRaises(ValueError):
                budget.validate_stage_request('initial',calls,'41')
        with self.assertRaisesRegex(ValueError,'cost_cap_exceeded'):
            budget.validate_stage_request('initial',290,'41.00001')
        for stage in (None,[],{},True,'','auto','full_then_continue'):
            with self.subTest(stage=stage),self.assertRaises(ValueError):budget.stage_budget(stage)
        for role in (None,[],True,'astra','unknown'):
            with self.subTest(role=role),self.assertRaises(ValueError):budget.role_reservation(role)

    def test_returned_settings_are_copies(self):
        value=budget.stage_budget('initial')
        value['roles']['sol']['max_calls']=999
        value['physics']['settings']['free']['timeout_seconds']=999
        self.assertEqual(budget.stage_budget('initial')['roles']['sol']['max_calls'],250)
        self.assertEqual(budget.stage_budget('initial')['physics']['settings']['free']['timeout_seconds'],3600)

    def test_proposal_pins_existing_manifest_without_reading_it(self):
        result=budget.build_budget_proposal('C:/dev/disassembly_codex_ready_v2/runs/pinned.json','a'*64)
        self.assertEqual(result['dataset_manifest']['sha256'],'a'*64)
        self.assertEqual(result['active_proposed_stage'],'initial')
        self.assertTrue(result['full_summary_is_not_executable'])
        self.assertEqual(result['API_calls_executed'],0)
        self.assertEqual(result['physics_calls_executed'],0)
        for pin in ('','a'*63,'z'*64,None):
            with self.subTest(pin=pin),self.assertRaises(ValueError):
                budget.build_budget_proposal('known-path',pin)


if __name__=='__main__':unittest.main()
