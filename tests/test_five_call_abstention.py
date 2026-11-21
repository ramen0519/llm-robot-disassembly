"""Budget and process-boundary tests; provider/physics results are fixtures."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from tools import run_abstention_revision_cycle as cycle
from tools.remaining_split_budget import remaining_proposal, validate_remaining_binding, claim_remaining_binding
from tools.run_luna_experiment import read, write, sha
from worker.task_memory import digest
from orchestrator.hierarchy_budget import HierarchyBudget, HierarchyBudgetError

ROOT=cycle.ROOT
OLD=ROOT/'runs/stability_transition_learning_20261105'
PREP=ROOT/'runs/abstention_feedback_20261105'


class FiveCallTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='five_calls_UNIT_',dir=ROOT/'runs')
        self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        network=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.net=network.start();self.addCleanup(network.stop);self.addCleanup(self.net.assert_not_called)
        self.remaining=remaining_proposal(OLD/'live/hierarchy.sqlite')
        self.binding={k:self.remaining[k] for k in ('parent_ledger','parent_ledger_sha256','parent_policy_sha256','parent_rows_sha256')}
        self.policy=read(PREP/'five_call_reallocation_DISABLED.json');self.policy['enabled']=True
        self.approval=dict(approved=True,scope='remaining_five_calls_abstention_v1',budget_parent=self.binding,
            remaining_policy_sha256=digest(self.remaining['policy']),maximum_new_api_attempts=5,
            maximum_program_estimated_usd=self.policy['max_usd'],role_calls=dict(learner=2,scout=0,planner=3),
            automatic_http_retries=0,no_revision_after_failure=True,fixture_only=True)
        self.declaration=self.declare(self.approval,'approval')

    def declare(self,value,name):
        p=self.root/(name+'.json');write(p,value)
        return dict(kind='abstention_candidate_5_v1',authorization_path=str(p),authorization_sha256=sha(p))

    def protocol(self):
        files=dict(parent=PREP/'historical_preflight_parent.json',manifest=OLD/'prepared/manifest.json',
            draft_lesson=OLD/'live/lesson_initial/lesson.json',memory=OLD/'preflight_02/frozen/memory.json',
            prior_library=OLD/'preflight_02/frozen/skills.json',prior_lessons=OLD/'preflight_02/frozen/lessons.json')
        return dict(schema_version=1,budget_parent=self.binding,budget_reallocation=self.declaration,
            files={k:dict(path=str(v),sha256=sha(v)) for k,v in files.items()},source_hashes=cycle.runtime_hashes(),
            target=deepcopy(cycle.TARGET),max_physics_calls=22,total_timeout_seconds=2400,http_retries=0)

    def test_exact_reallocation_preserves_parent_and_money(self):
        self.assertEqual(validate_remaining_binding(self.binding,self.policy,reallocation=self.declaration),self.remaining)
        self.assertEqual(self.policy['max_usd'],'1.9707321')
        self.assertEqual(sha(OLD/'live/hierarchy.sqlite'),self.binding['parent_ledger_sha256'])
        for role in ('learner','planner'):
            self.assertEqual(self.policy['roles'][role]['max_usd'],self.remaining['policy']['roles'][role]['max_usd'])

    def test_implicit_transfer_and_wrong_authorization_rejected(self):
        with self.assertRaises(ValueError):validate_remaining_binding(self.binding,self.policy)
        for field,value in [('approved',False),('maximum_new_api_attempts',6),('no_revision_after_failure',False),
                            ('automatic_http_retries',1),('role_calls',dict(learner=3,scout=0,planner=2))]:
            d=self.declare(dict(self.approval,**{field:value}),field)
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_remaining_binding(self.binding,self.policy,reallocation=d)

    def test_cannot_change_rates_model_money_calls_or_restore_scout(self):
        for role,key,value in [('global','max_usd','2'),('global','max_calls',6),('learner','max_calls',3),
                ('learner','input_usd_per_million','.01'),('learner','max_output_tokens',2048),
                ('planner','max_calls',4),('planner','model','other'),('scout','max_calls',1)]:
            p=deepcopy(self.policy);(p if role=='global' else p['roles'][role])[key]=value
            with self.subTest(role=role,key=key),self.assertRaises(ValueError):
                validate_remaining_binding(self.binding,p,reallocation=self.declaration)

    def test_exclusive_claim_and_stale_parent(self):
        with patch('tools.remaining_split_budget.CLAIM_ROOT',self.root/'claims'):
            claim_remaining_binding(self.binding,self.policy,self.root/'live',reallocation=self.declaration)
            with self.assertRaises(FileExistsError):
                claim_remaining_binding(self.binding,self.policy,self.root/'again',reallocation=self.declaration)
        with self.assertRaises(ValueError):validate_remaining_binding(dict(self.binding,parent_rows_sha256='0'*64),
            self.policy,reallocation=self.declaration)

    def test_readonly_protocol_replays_actual_bound_abstention(self):
        paths=cycle.validate(self.protocol(),self.policy)
        self.assertEqual(set(paths),cycle.FILES)
        self.assertEqual(sha(OLD/'live/hierarchy.sqlite'),self.binding['parent_ledger_sha256'])

    def test_changed_physics_or_input_cannot_pass(self):
        p=self.protocol()
        for k,v in [('max_physics_calls',23),('total_timeout_seconds',2401),('http_retries',1)]:
            with self.subTest(k=k),self.assertRaises(ValueError):cycle.validate(dict(p,**{k:v}),self.policy)
        p['target']['max_gripper']=4
        with self.assertRaises(ValueError):cycle.validate(p,self.policy)
        p=self.protocol();p['files']['memory']['sha256']='0'*64
        with self.assertRaises(ValueError):cycle.validate(p,self.policy)

    def test_disabled_no_keys_no_output(self):
        p=self.root/'disabled.json';write(p,dict(self.policy,enabled=False))
        with self.assertRaises(ValueError):cycle.execute(self.root/'no_output',{},p,allow_live=True)
        self.assertFalse((self.root/'no_output').exists())

    def test_role_limits_and_transport_error_latch(self):
        pool=HierarchyBudget(self.root/'budget.sqlite',self.policy,transport_failure_policy='latch_on_error_v1')
        backend=Mock(return_value={})
        def send(role,i):
            cfg=self.policy['roles'][role]
            return pool.transport(role,backend)(cfg['base_url']+'/responses',{},dict(model=cfg['model'],store=False,
                reasoning=dict(effort=cfg['reasoning_effort']),input=str(i),max_output_tokens=32),1)
        with self.assertRaises(HierarchyBudgetError):send('scout',0)
        for i in range(2):send('learner',i)
        for i in range(3):send('planner',i)
        for role in ('learner','planner','scout'):
            with self.assertRaises(HierarchyBudgetError):send(role,9)
        self.assertEqual(backend.call_count,5)
        other=HierarchyBudget(self.root/'unknown.sqlite',self.policy,transport_failure_policy='latch_on_error_v1')
        cfg=self.policy['roles']['learner'];bad=Mock(side_effect=TimeoutError('UNIT'))
        body=dict(model=cfg['model'],store=False,reasoning=dict(effort='none'),input='timeout',max_output_tokens=32)
        with self.assertRaises(TimeoutError):other.transport('learner',bad)(cfg['base_url']+'/responses',{},body,1)
        with self.assertRaises(Exception):other.transport('learner',bad)(cfg['base_url']+'/responses',{},body,1)
        self.assertEqual(bad.call_count,1)

    def run_stages(self,candidate_ok=False,lesson_ok=False,accepted=False,used=True):
        output=self.root/('stages'+str(len(list(self.root.glob('stages*')))));output.mkdir()
        p=self.protocol();paths={k:Path(v['path']) for k,v in p['files'].items()};calls=[]
        final_lesson=output/'lesson_fixture.json';write(final_lesson,{})
        def invoke(name,tool,args,limit):
            calls.append((name,tool,args,limit));folder=Path(args[args.index('--run-dir')+1]);folder.mkdir()
            if name=='candidate_01':
                write(folder/'result.json',dict(status='REGISTERED_SCOPED' if candidate_ok else 'PREFLIGHT_REJECTED',
                    admission=dict(admitted=candidate_ok),actual_llm_generation=True,source_unchanged=True,candidate_id='new_UNIT'))
                return 0 if candidate_ok else 1
            if name=='lesson_finalize':
                write(folder/'result.json',dict(status='CANDIDATE' if lesson_ok else 'GENERATION_REJECTED',lesson=str(final_lesson)))
                return 0 if lesson_ok else 1
            write(folder/'audit.json',dict(validation_ok=True,actual_physics=True,status='PASS',plan_success=True))
            write(folder/'hierarchy_audit.json',dict(library_unchanged=True,lesson_library_unchanged=True,
                physics_bindings=[dict(skill_id='new_UNIT' if used else None,accepted=accepted)],
                memory_visible_requests=1,memory_cited_requests=0))
            write(folder/'host_preservation.json',dict(unchanged=True))
            return 0
        with patch('orchestrator.skill_library_merge.freeze_skill_union',return_value={}),\
             patch('orchestrator.lesson_memory.validate_library',return_value={'entries':[]}),\
             patch('orchestrator.lesson_memory.freeze_lessons',return_value={'entries':[]}),\
             patch('orchestrator.explicit_action_memory.freeze_cumulative_memory'),\
             patch('orchestrator.lesson_memory.freeze_bundle'):
            result=cycle.stages(output,p,paths,{k:output/(k+'.json') for k in ('learner','scout','planner')},invoke)
        return result,calls

    def test_single_failed_candidate_skips_all_followups(self):
        report,calls=self.run_stages()
        self.assertEqual(len(calls),1);self.assertEqual(report['target'],'NOT_RUN')
        self.assertIn('--abstention-feedback',calls[0][2]);self.assertIn('--prior-lessons',calls[0][2])
        self.assertIn('--transport-failure-policy',calls[0][2])

    def test_failed_lesson_skips_target(self):
        report,calls=self.run_stages(candidate_ok=True)
        self.assertEqual(len(calls),2);self.assertEqual(report['target'],'NOT_RUN')

    def test_registered_skill_pass_requires_actual_binding(self):
        for used,accepted in ((False,True),(True,False),(True,True)):
            report,calls=self.run_stages(True,True,accepted,used)
            self.assertEqual(len(calls),3)
            self.assertEqual(report['new_skill_passes'],int(used and accepted))
            self.assertEqual(report['new_skill_attempts'],int(used))
            self.assertEqual(report['status'],'COMPLETED')
            args=calls[-1][2]
            self.assertIn('conditional_plan_v4',args);self.assertIn('repair_evidence_v2',args)
            self.assertIn(cycle.TARGET['assembly_dir'],args)
            self.assertFalse(report['forced_skill_selection'])


if __name__=='__main__':unittest.main()
