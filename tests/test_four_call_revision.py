"""Offline reallocation/controller tests. Injected provider results are not live evidence."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from orchestrator.split_revision_budget import SplitRevisionBudget
from orchestrator.hierarchy_budget import HierarchyBudget,HierarchyBudgetError
from worker.task_memory import digest
from tools.remaining_split_budget import remaining_proposal,validate_remaining_binding,claim_remaining_binding
from tools.run_luna_experiment import read,write,sha
from tools import run_split_revision_cycle as cycle

ROOT=cycle.ROOT
PREVIOUS=ROOT/'runs/execution_revision_live_20261021/live_cycle'
PREP=ROOT/'runs/direction_grounding_fix_20261021/final_preparation'


class FourCallTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='four_calls_UNIT_',dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        n=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.network=n.start();self.addCleanup(n.stop);self.addCleanup(self.network.assert_not_called)
        self.remaining=remaining_proposal(PREVIOUS/'hierarchy.sqlite')
        self.parent={k:self.remaining[k] for k in ('parent_ledger','parent_ledger_sha256','parent_policy_sha256','parent_rows_sha256')}
        self.policy=deepcopy(self.remaining['policy']);self.policy['enabled']=True
        self.policy['roles']['learner']['max_calls']=2;self.policy['roles']['planner']['max_calls']=1
        self.policy['episode']['planner_calls']=1
        self.approval=dict(approved=True,scope='remaining_four_calls_reallocated_v1',budget_parent=self.parent,
            remaining_policy_sha256=digest(self.remaining['policy']),maximum_new_api_attempts=4,
            maximum_program_estimated_usd=self.policy['max_usd'],role_calls=dict(learner=2,scout=1,planner=1),
            automatic_http_retries=0,fixture_only=True)
        self.reallocation=self.declaration(self.approval,'approval')

    def declaration(self,approval,name):
        path=self.root/(name+'.json');write(path,approval)
        return dict(kind='single_candidate_4_v1',authorization_path=str(path),authorization_sha256=sha(path))

    def protocol(self):
        p=read(PREVIOUS.parent/'protocol.json')
        p.update(mode='single_candidate_4_v1',physical_revisions=1,contract_corrections=0,
            generation_contract_version=3,budget_parent=self.parent,budget_reallocation=self.reallocation,
            source_hashes=cycle.runtime_hashes())
        p['files']['parent']=dict(path=str(PREP/'historical_parent.json'),sha256=sha(PREP/'historical_parent.json'))
        return p

    def test_explicit_transfer_preserves_global_money_and_unknown_reservations(self):
        value=validate_remaining_binding(self.parent,self.policy,reallocation=self.reallocation)
        self.assertEqual(value,self.remaining)
        self.assertEqual(self.policy['max_usd'],'1.3182912')
        self.assertEqual(sha(PREVIOUS/'hierarchy.sqlite'),self.parent['parent_ledger_sha256'])

    def test_original_guard_still_rejects_unapproved_reallocation(self):
        with self.assertRaisesRegex(ValueError,'increase_or_reallocate'):
            validate_remaining_binding(self.parent,self.policy)

    def test_approval_changes_hash_and_unapproved_scope_fail(self):
        d=dict(self.reallocation,authorization_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'authorization_changed'):
            validate_remaining_binding(self.parent,self.policy,reallocation=d)
        for key,value in [('approved',False),('maximum_new_api_attempts',5),('role_calls',dict(learner=3,scout=1,planner=0)),
                          ('automatic_http_retries',1),('remaining_policy_sha256','0'*64)]:
            auth=dict(self.approval,**{key:value});decl=self.declaration(auth,key)
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_remaining_binding(self.parent,self.policy,reallocation=decl)

    def test_no_extra_calls_money_different_model_or_settings(self):
        for target,key,value in [('global','max_calls',5),('global','max_usd','1.35'),
                                 ('learner','max_calls',3),('planner','input_usd_per_million','1'),
                                 ('learner','max_output_tokens',2048),('episode','fallback_calls',1)]:
            bad=deepcopy(self.policy)
            (bad if target=='global' else bad['episode'] if target=='episode' else bad['roles'][target])[key]=value
            with self.subTest(target=target,key=key),self.assertRaises(ValueError):
                validate_remaining_binding(self.parent,bad,reallocation=self.reallocation)

    def test_stale_parent_and_excluded_auth_path_rejected(self):
        bad=dict(self.parent,parent_rows_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'parent_changed'):
            validate_remaining_binding(bad,self.policy,reallocation=self.reallocation)
        with patch('tools.remaining_split_budget.remaining_proposal',return_value=self.remaining),\
             patch.object(Path,'open',side_effect=AssertionError('NO_FILESYSTEM')):
            with self.assertRaisesRegex(ValueError,'excluded_path'):
                validate_remaining_binding(self.parent,self.policy,reallocation=dict(self.reallocation,authorization_path='E:/NO_ACCESS'))

    def test_reallocated_and_original_claim_share_same_exclusive_identity(self):
        with patch('tools.remaining_split_budget.CLAIM_ROOT',self.root/'claims'):
            claim=claim_remaining_binding(self.parent,self.policy,self.root/'one',reallocation=self.reallocation)
            self.assertEqual(claim['explicit_reallocation'],self.reallocation)
            original=deepcopy(self.remaining['policy']);original['enabled']=True
            with self.assertRaises(FileExistsError):claim_remaining_binding(self.parent,original,self.root/'two')
            with self.assertRaises(FileExistsError):
                claim_remaining_binding(self.parent,self.policy,self.root/'three',reallocation=self.reallocation)

    def test_disabled_policy_cannot_claim(self):
        disabled=dict(self.policy,enabled=False)
        with patch('tools.remaining_split_budget.CLAIM_ROOT',self.root/'claims'):
            with self.assertRaisesRegex(ValueError,'disabled'):
                claim_remaining_binding(self.parent,disabled,self.root/'one',reallocation=self.reallocation)
        self.assertFalse((self.root/'claims').exists())

    def test_real_four_call_protocol_validates_bound_inputs_without_API(self):
        slots,paths=cycle.validate_protocol(self.protocol(),self.policy)
        self.assertTrue(slots.single_candidate);self.assertEqual(slots.contract_corrections,0)
        self.assertEqual(paths['parent'],PREP/'historical_parent.json')

    def test_protocol_cannot_drop_authorization_or_restore_replanning(self):
        p=self.protocol()
        for key in ('mode','budget_reallocation','budget_parent'):
            bad=deepcopy(p);del bad[key]
            with self.subTest(key=key),self.assertRaises(ValueError):cycle.validate_protocol(bad,self.policy)
        for key,value in [('generation_contract_version',2),('contract_corrections',1),('max_physics_calls',99)]:
            with self.subTest(key=key),self.assertRaises(ValueError):cycle.validate_protocol(dict(p,**{key:value}),self.policy)

    def test_single_candidate_stops_after_every_possible_outcome(self):
        for outcome in ('contract_rejected','measured_debug_failure','registered','stop'):
            b=SplitRevisionBudget(1,0,single_candidate=True)
            self.assertEqual(b.allocate(),'physical_revision');b.observe(outcome)
            self.assertIsNone(b.allocate());self.assertEqual(b.used,dict(physical_revision=1,contract_correction=0))

    def test_legacy_zero_slots_are_still_invalid(self):
        with self.assertRaises(ValueError):SplitRevisionBudget(1,0)
        for values in ((2,0),(1,1),(True,0)):
            with self.assertRaises(ValueError):SplitRevisionBudget(*values,single_candidate=True)

    def test_loop_never_issues_second_generation_after_rejection_or_physics_failure(self):
        for index,outcome in enumerate(('contract_rejected','measured_debug_failure','stop','registered')):
            output=self.root/str(index);output.mkdir();calls=[]
            def invoke(label,tool,args):
                calls.append(args);folder=Path(args[args.index('--run-dir')+1]);folder.mkdir()
                write(folder/'result.json',dict(status='REGISTERED_SCOPED' if outcome=='registered' else 'PREFLIGHT_REJECTED'))
                return 0 if outcome=='registered' else 1
            with patch.object(cycle,'classify_candidate',return_value=outcome):
                result=cycle.run_candidates(output,dict(mode='single_candidate_4_v1',physical_revisions=1,contract_corrections=0,
                    generation_contract_version=3,generation_timeout_seconds=90),
                    {k:self.root/k for k in ('parent','manifest','draft_lesson','prior_library')},dict(learner='UNIT'),invoke)
            self.assertEqual(len(calls),1)
            self.assertEqual(result['admitted_folder'] is not None,outcome=='registered')
            self.assertEqual(calls[0][calls[0].index('--python-revision-version')+1],3)

    def test_injected_transport_cannot_spend_fifth_or_second_planner_call(self):
        pool=HierarchyBudget(self.root/'budget.sqlite',self.policy)
        backend=Mock(return_value={})
        def send(role,index):
            cfg=self.policy['roles'][role]
            body=dict(model=cfg['model'],store=False,reasoning=dict(effort=cfg['reasoning_effort']),
                      input=str(index),max_output_tokens=32)
            return pool.transport(role,backend)(cfg['base_url']+'/responses',{},body,1)
        send('planner',0)
        with self.assertRaisesRegex(HierarchyBudgetError,'call_or_cost'):send('planner',1)
        send('learner',0);send('learner',1);send('scout',0)
        with self.assertRaisesRegex(HierarchyBudgetError,'call_or_cost'):send('learner',2)
        self.assertEqual(backend.call_count,4);self.assertEqual(pool.summary()['live_api_attempts'],0)

    def test_timeout_not_refunded_or_retried(self):
        pool=HierarchyBudget(self.root/'unknown.sqlite',self.policy)
        backend=Mock(side_effect=TimeoutError('UNIT_TIMEOUT'))
        cfg=self.policy['roles']['learner']
        body=dict(model=cfg['model'],store=False,reasoning=dict(effort=cfg['reasoning_effort']),input='unit',max_output_tokens=32)
        call=pool.transport('learner',backend)
        with self.assertRaises(TimeoutError):call(cfg['base_url']+'/responses',{},body,1)
        with self.assertRaisesRegex(HierarchyBudgetError,'no_resend'):call(cfg['base_url']+'/responses',{},body,1)
        self.assertEqual(backend.call_count,1);self.assertEqual(pool.summary()['pending_or_unknown'],1)

if __name__=='__main__':unittest.main()
