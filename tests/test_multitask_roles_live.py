"""Offline runner/budget tests. Mock stage outcomes never count as model/physics success."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from unittest.mock import patch
from tools import run_multitask_roles as runner
from tools import remaining_split_budget as budgets
from tools.run_multitask_learning import learn
from tools.run_luna_experiment import read,write,sha

READY=runner.ROOT/'runs/multitask_roles_live_20261106/prepared'


class RolesLiveTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='roles_live_UNIT_',dir=runner.ROOT/'runs')
        self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.protocol=read(READY/'protocol.json');self.policy=read(READY/'policy_DISABLED.json')
        self.binding=self.protocol['budget_parent']

    def test_suballocation_is_bound_to_parent_without_modification(self):
        parent=Path(self.binding['parent_ledger']);before=sha(parent)
        value=budgets.validate_remaining_binding(self.binding,self.policy,restriction=True)
        self.assertEqual(value['policy']['max_calls'],23)
        self.assertEqual(self.policy['max_calls'],16)
        self.assertEqual(sha(parent),before)
        with self.assertRaises(ValueError):budgets.validate_remaining_binding(self.binding,self.policy)

    def test_restriction_cannot_change_prices_models_or_episode_budget(self):
        for key,value in [('input_usd_per_million','0'),('model','other'),('reasoning_effort','high')]:
            p=deepcopy(self.policy);p['roles']['planner'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                budgets.validate_remaining_binding(self.binding,p,restriction=True)
        p=deepcopy(self.policy);p['episode']['planner_calls']=3
        with self.assertRaises(ValueError):budgets.validate_remaining_binding(self.binding,p,restriction=True)

    def test_any_cap_increase_or_reallocation_rejected(self):
        for path,key,value in [(None,'max_usd','13'),(None,'max_calls',35),
                               ('learner','max_calls',5),('learner','max_usd','1')]:
            p=deepcopy(self.policy);(p if path is None else p['roles'][path])[key]=value
            with self.subTest(path=path,key=key),self.assertRaises(ValueError):
                budgets.validate_remaining_binding(self.binding,p,restriction=True)
        with self.assertRaises(ValueError):budgets.validate_remaining_binding(self.binding,self.policy,
            restriction=True,reallocation={'kind':'not_authorized'})

    def test_unknown_parent_cost_cannot_be_refunded_or_changed(self):
        b=dict(self.binding,parent_rows_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'parent_changed'):
            budgets.validate_remaining_binding(b,self.policy,restriction=True)

    def test_one_exclusive_claim_prevents_double_spend(self):
        policy=dict(self.policy,enabled=True)
        with patch.object(budgets,'CLAIM_ROOT',self.root/'claims'):
            c=budgets.claim_remaining_binding(self.binding,policy,self.root/'live',restriction=True)
            self.assertTrue(c['strictly_bounded_suballocation'])
            self.assertTrue(c['unknown_reservations_retained'])
            with self.assertRaises(FileExistsError):
                budgets.claim_remaining_binding(self.binding,policy,self.root/'again',restriction=True)

    def test_disabled_or_unapproved_cannot_claim_or_launch(self):
        with patch.object(budgets,'CLAIM_ROOT',self.root/'claims'):
            with self.assertRaises(ValueError):budgets.claim_remaining_binding(self.binding,self.policy,self.root/'live',restriction=True)
        with self.assertRaises(ValueError):runner.execute(READY,self.root/'live',False)
        self.assertFalse((self.root/'live').exists());self.assertFalse((self.root/'claims').exists())

    def test_preparation_is_bound_to_sources_and_fixed_evaluation(self):
        # Runtime hash is intentionally stale after the tiny summary-status edit;
        # data/policy validation is exercised with that one field substituted.
        with patch.object(runner,'runtime_hashes',return_value=self.protocol['runtime']):
            p,c=runner.validate(READY)
        self.assertEqual(p['evaluation_ids'],['02680','02656'])
        self.assertEqual(p['max_seconds'],5400);self.assertEqual(p['max_physics_attempts'],80)
        self.assertEqual(len(read(Path(p['memory']))['records']),32)
        self.assertFalse(c['enabled'])
        with patch.object(runner,'runtime_hashes',return_value={}):
            with self.assertRaisesRegex(ValueError,'runtime_changed'):runner.validate(READY)

    def simulate_learning(self,responses,guard=False):
        calls=[]
        def invoke(name,tool,args,limit):
            calls.append((name,args));out=Path(args[args.index('--run-dir')+1]);out.mkdir()
            value=responses[name];write(out/'result.json',value)
            return 0 if value['status'] in ('CANDIDATE','REGISTERED_SCOPED') else 1
        pool=SimpleNamespace(summary=lambda:dict(transport_guard=guard))
        result=learn(self.root,invoke,pool,learning_inputs=(Path(self.protocol['manifest']),Path(self.protocol['packet'])))
        return result,calls

    def test_hidden_healthy_failure_stops_without_revision_or_evaluation(self):
        r,c=self.simulate_learning({'lesson_extract':dict(status='CANDIDATE'),
            'candidate_01':dict(status='NONAPPLICABILITY_REJECTED')})
        self.assertEqual(len(c),2);self.assertEqual(r['evaluation'],'NOT_RUN')

    def test_second_training_product_debug_failure_drives_only_one_revision(self):
        r,c=self.simulate_learning({'lesson_extract':dict(status='CANDIDATE'),
            'candidate_01':dict(status='NOT_REGISTERED',cases=[dict(case_id='source_02628',outcome='PASS'),
                dict(case_id='source_01289',outcome='UNKNOWN')]),
            'candidate_02':dict(status='PREFLIGHT_REJECTED')})
        self.assertEqual(len(c),3);self.assertIn('--refine-from',c[2][1])
        self.assertEqual(r['status'],'NOT_REGISTERED')

    def test_yaw_admission_failure_is_hidden_no_revision(self):
        r,c=self.simulate_learning({'lesson_extract':dict(status='CANDIDATE'),
            'candidate_01':dict(status='NOT_REGISTERED',cases=[dict(case_id='source_02628',outcome='PASS'),
                dict(case_id='source_01289',outcome='PASS'),dict(case_id='source_01289_yaw90',outcome='UNKNOWN')])})
        self.assertEqual(len(c),2)

    def test_transport_guard_prevents_new_candidate(self):
        r,c=self.simulate_learning({'lesson_extract':dict(status='CANDIDATE'),
            'candidate_01':dict(status='GENERATION_REJECTED')},guard=True)
        self.assertEqual(len(c),2)

    def test_stage_gate_no_published_skill_means_no_evaluation(self):
        with patch.object(runner,'learn',return_value=dict(status='NOT_REGISTERED')), \
             patch.object(runner,'evaluate',side_effect=AssertionError('NO_EVAL')) as ev:
            r=runner.stages(self.root,self.protocol,None,None)
        ev.assert_not_called();self.assertEqual(r['evaluation']['status'],'NOT_RUN')

    def test_evaluation_stop_is_not_reported_as_completion(self):
        with patch.object(runner,'learn',return_value=dict(status='FROZEN')), \
             patch.object(runner,'evaluate',return_value=dict(status='STOPPED')):
            r=runner.stages(self.root,self.protocol,None,None)
        self.assertEqual(r['status'],'STOPPED')


if __name__=='__main__':unittest.main()
