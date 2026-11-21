"""Offline final-trial runner/budget tests; all provider results here are fixtures."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.hierarchy_budget import HierarchyBudget, HierarchyBudgetError, validate_policy
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.conditional_plan import validate_plan
from tests.test_hierarchical_planner import request, plan_value, feedback
from tools.remaining_split_budget import remaining_proposal,validate_remaining_binding,claim_remaining_binding
from tools.run_explicit_action_experiment import validate_live_launch_config
from tools.run_luna_experiment import read,write,sha
from tools import run_split_revision_cycle as cycle
from worker.task_memory import digest

ROOT=cycle.ROOT
PARENT=ROOT/'runs/direction_grounding_live_20261021/live_cycle'
PREP=ROOT/'runs/separation_feedback_fix_20261022/prepared_v4'


class ThreeCallTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='three_calls_UNIT_',dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        n=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.net=n.start();self.addCleanup(n.stop);self.addCleanup(self.net.assert_not_called)
        self.remaining=remaining_proposal(PARENT/'hierarchy.sqlite')
        self.binding={k:self.remaining[k] for k in ('parent_ledger','parent_ledger_sha256','parent_policy_sha256','parent_rows_sha256')}
        self.policy=deepcopy(self.remaining['policy']);self.policy['enabled']=True
        self.policy['roles']['learner']['max_calls']=2
        self.policy['roles']['scout'].update(max_calls=0,max_usd='0')
        self.policy['episode']['scout_calls']=0
        self.deadline=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat()
        self.approval=dict(approved=True,scope='remaining_three_calls_reallocated_v1',budget_parent=self.binding,
            remaining_policy_sha256=digest(self.remaining['policy']),maximum_new_api_attempts=3,
            maximum_program_estimated_usd=self.policy['max_usd'],role_calls=dict(learner=2,scout=0,planner=1),
            automatic_http_retries=0,no_revision_after_failure=True,deadline_utc=self.deadline,fixture_only=True)
        self.declaration=self.declare(self.approval,'approval')

    def declare(self,value,name):
        path=self.root/(name+'.json');write(path,value)
        return dict(kind='single_candidate_3_v1',authorization_path=str(path),authorization_sha256=sha(path))

    def protocol(self):
        p=read(PARENT.parent/'protocol.json')
        p.update(mode='single_candidate_3_v1',physical_revisions=1,contract_corrections=0,
            generation_contract_version=4,deadline_utc=self.deadline,budget_parent=self.binding,
            budget_reallocation=self.declaration,source_hashes=cycle.runtime_hashes())
        p['files']['parent']=dict(path=str(PREP/'historical_parent.json'),sha256=sha(PREP/'historical_parent.json'))
        return p

    def test_exact_approved_transfer_preserves_unknown_and_caps(self):
        self.assertEqual(validate_remaining_binding(self.binding,self.policy,reallocation=self.declaration),self.remaining)
        self.assertEqual(self.policy['max_usd'],'1.3103559')
        for role in ('learner','planner'):
            self.assertEqual(self.policy['roles'][role]['max_usd'],self.remaining['policy']['roles'][role]['max_usd'])
        self.assertEqual(sha(PARENT/'hierarchy.sqlite'),self.binding['parent_ledger_sha256'])

    def test_default_guard_rejects_implicit_transfer(self):
        with self.assertRaisesRegex(ValueError,'increase_or_reallocate'):
            validate_remaining_binding(self.binding,self.policy)

    def test_wrong_approval_or_hash_rejected(self):
        with self.assertRaisesRegex(ValueError,'authorization_changed'):
            validate_remaining_binding(self.binding,self.policy,reallocation=dict(self.declaration,authorization_sha256='0'*64))
        for key,value in [('no_revision_after_failure',False),('maximum_new_api_attempts',4),('automatic_http_retries',1),
                          ('role_calls',dict(learner=1,scout=1,planner=1)),('approved',False)]:
            d=self.declare(dict(self.approval,**{key:value}),key)
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_remaining_binding(self.binding,self.policy,reallocation=d)

    def test_money_rates_model_output_and_calls_cannot_change(self):
        for target,key,value in [('global','max_calls',4),('global','max_usd','1.35'),
                                 ('learner','max_calls',3),('learner','max_usd','.04'),
                                 ('planner','input_usd_per_million','1'),('learner','max_output_tokens',2048)]:
            bad=deepcopy(self.policy)
            (bad if target=='global' else bad['roles'][target])[key]=value
            with self.subTest(target=target,key=key),self.assertRaises(ValueError):
                validate_remaining_binding(self.binding,bad,reallocation=self.declaration)

    def test_claim_remains_exclusive_with_original_parent(self):
        with patch('tools.remaining_split_budget.CLAIM_ROOT',self.root/'claims'):
            claim_remaining_binding(self.binding,self.policy,self.root/'new',reallocation=self.declaration)
            with self.assertRaises(FileExistsError):
                claim_remaining_binding(self.binding,self.policy,self.root/'again',reallocation=self.declaration)
            old=deepcopy(self.remaining['policy']);old['enabled']=True
            with self.assertRaises(FileExistsError):claim_remaining_binding(self.binding,old,self.root/'old')

    def test_protocol_v4_and_zero_scout_bind_latest_debug_without_api(self):
        slots,paths=cycle.validate_protocol(self.protocol(),self.policy)
        self.assertTrue(slots.single_candidate);self.assertEqual(slots.contract_corrections,0)
        self.assertEqual(paths['parent'],PREP/'historical_parent.json')

    def test_protocol_cannot_add_correction_replan_or_drop_deadline(self):
        p=self.protocol()
        for key,value in [('generation_contract_version',3),('contract_corrections',1),('max_physics_calls',13),('deadline_utc','2000-01-01T00:00:00+00:00')]:
            with self.subTest(key=key),self.assertRaises(ValueError):cycle.validate_protocol(dict(p,**{key:value}),self.policy)
        del p['deadline_utc']
        with self.assertRaises(ValueError):cycle.validate_protocol(p,self.policy)

    def test_deadline_must_be_aware_unexpired_and_at_most_two_hours(self):
        for deadline in ('2000-01-01T00:00:00+00:00','2099-01-01T00:00:00+00:00','2099-01-01T00:00:00'):
            with self.assertRaises(ValueError):cycle.remaining_seconds(dict(mode='single_candidate_3_v1',deadline_utc=deadline))
        self.assertIsNone(cycle.remaining_seconds(dict(mode='single_candidate_4_v1')))

    def test_inactive_scout_cannot_have_budget_or_episode_fallback(self):
        validate_policy(self.policy)
        for target,key,value in [('scout','max_usd','.03'),('episode','scout_calls',1),('episode','fallback_calls',1)]:
            p=deepcopy(self.policy)
            (p['episode'] if target=='episode' else p['roles'][target])[key]=value
            with self.assertRaises(ValueError):validate_policy(p)

    def test_sol_only_launch_exception_is_narrow(self):
        cfg=self.policy['roles']['scout'];factory=Mock()
        metadata=dict(kind='luna_sol_hierarchy_v1',hierarchy_policy=self.policy)
        validate_live_launch_config(cfg,factory,metadata)
        for f,m in [(None,metadata),(factory,{}),(factory,dict(metadata,kind='other'))]:
            with self.assertRaises(ValueError):validate_live_launch_config(cfg,f,m)
        with self.assertRaises(ValueError):
            validate_live_launch_config(cfg,factory,dict(metadata,hierarchy_policy=dict(self.policy,enabled=False)))

    def test_transport_refuses_scout_fourth_call_and_second_sol(self):
        pool=HierarchyBudget(self.root/'budget.sqlite',self.policy);backend=Mock(return_value={})
        def send(role,n):
            cfg=self.policy['roles'][role]
            return pool.transport(role,backend)(cfg['base_url']+'/responses',{},dict(model=cfg['model'],store=False,
                reasoning=dict(effort=cfg['reasoning_effort']),input=str(n),max_output_tokens=32),1)
        with self.assertRaises(HierarchyBudgetError):send('scout',0)
        self.assertEqual(backend.call_count,0)
        send('learner',0);send('learner',1);send('planner',0)
        for role in ('learner','scout','planner'):
            with self.assertRaises(HierarchyBudgetError):send(role,2)
        self.assertEqual(backend.call_count,3)

    def test_same_unknown_request_is_not_resent(self):
        pool=HierarchyBudget(self.root/'unknown.sqlite',self.policy);backend=Mock(side_effect=TimeoutError('UNIT'))
        cfg=self.policy['roles']['learner'];body=dict(model=cfg['model'],store=False,reasoning=dict(effort='none'),input='unit',max_output_tokens=32)
        t=pool.transport('learner',backend)
        with self.assertRaises(TimeoutError):t(cfg['base_url']+'/responses',{},body,1)
        with self.assertRaisesRegex(HierarchyBudgetError,'no_resend'):t(cfg['base_url']+'/responses',{},body,1)
        self.assertEqual(backend.call_count,1);self.assertEqual(pool.summary()['pending_or_unknown'],1)

    def test_no_second_candidate_after_any_failure(self):
        for i,outcome in enumerate(('contract_rejected','measured_debug_failure','stop','registered')):
            output=self.root/str(i);output.mkdir();calls=[]
            def invoke(label,tool,args):
                calls.append(args);folder=Path(args[args.index('--run-dir')+1]);folder.mkdir()
                write(folder/'result.json',dict(status='REGISTERED_SCOPED' if outcome=='registered' else 'PREFLIGHT_REJECTED'))
                return 0 if outcome=='registered' else 1
            with patch.object(cycle,'classify_candidate',return_value=outcome):
                result=cycle.run_candidates(output,dict(mode='single_candidate_3_v1',physical_revisions=1,contract_corrections=0,
                    generation_contract_version=4,generation_timeout_seconds=90),
                    {k:self.root/k for k in ('parent','manifest','draft_lesson','prior_library')},dict(learner='UNIT'),invoke)
            self.assertEqual(len(calls),1)
            self.assertEqual(result['admitted_folder'] is not None,outcome=='registered')

    def test_first_decision_goes_directly_to_sol_and_code_reuses_plan(self):
        scout=Mock();scout.timeout_seconds=30;scout.handle.side_effect=AssertionError('NO_SCOUT')
        planner=Mock();planner.plan.side_effect=lambda r,*a,**kw:dict(status='planned',
            plan=validate_plan(plan_value(r),r,[]),metadata=dict(llm_live=False))
        with patch('orchestrator.hierarchical_service.retrieve',return_value=[]):
            svc=HierarchicalDecisionService(scout,planner,{},self.policy['episode'])
        r=request();a=svc.handle(r);n=feedback(r,a);svc.handle(n)
        self.assertEqual(planner.plan.call_count,1);scout.handle.assert_not_called()


if __name__=='__main__':unittest.main()
