"""No paid API or new physics: real saved inputs + labelled boundary mocks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.run_luna_experiment import read,write,sha
from tools.run_skill_discovery import source_case,run_case
from tools.python_source_budget import source_budget,validate_binding
from worker.skill_case_budget import worker_allowance,validate_source_context
from orchestrator.skill_case import physical_context,request_after_trigger

ROOT=Path(__file__).absolute().parents[1]
LIVE=ROOT/'runs/multitask_roles_live_20261106/live'
MANIFEST=ROOT/'runs/multitask_roles_20261105/prepared/manifest.json'


class SourceCaseBudgetTests(unittest.TestCase):
    def setUp(self):
        self.manifest=read(MANIFEST)
        self.items=[source_case(s) for s in self.manifest['cases']]
        temp=tempfile.TemporaryDirectory(prefix='source_budget_UNIT_',dir=ROOT/'runs')
        self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def test_original_allowance_and_thresholds_for_all_declared_poses(self):
        for item in self.items:
            bound=source_budget(item)
            self.assertGreater(bound['seconds'],400)
            ctx=physical_context(item['context'],item['request'],item['choice'],event_id='test',
                seconds=bound['seconds'],wall_limit=bound['seconds'],preserve_stability_policy=True)
            self.assertEqual(validate_binding(ctx,bound),bound['seconds'])
            self.assertEqual(ctx.get('stability_timeout'),item['context'].get('stability_timeout'))
            self.assertEqual(ctx['budget'],item['context']['budget'])
            self.assertEqual(worker_allowance(ctx,bound['source_request_path'],bound['source_request_sha256'],
                                             checked=lambda p:Path(p)),bound['seconds'])

    def test_mutated_clock_cannot_claim_actual_source(self):
        for key,value in [('remaining_wall_seconds',500),('budget',999),('stability_timeout',999),('seed',42)]:
            item=deepcopy(self.items[0]);item['context'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):source_budget(item)

    def test_changed_source_hash_is_rejected(self):
        item=self.items[0];bound=source_budget(item);bound['source_request_sha256']='0'*64
        with self.assertRaises(ValueError):validate_binding(item['context'],bound)

    def test_legacy_default_and_unbounded_values_stay_rejected(self):
        for seconds in (46,601,0,True,float('inf'),float('nan')):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                worker_allowance({'remaining_wall_seconds':seconds},checked=lambda p:Path(p))
        self.assertEqual(worker_allowance({'remaining_wall_seconds':45},checked=lambda p:Path(p)),45)
        p=self.items[0]
        with self.assertRaises(ValueError):physical_context(p['context'],p['request'],p['choice'],event_id='test',seconds=400)

    def test_e_path_is_rejected_before_read(self):
        from worker.remote_rank_client import checked_path
        with patch.object(Path,'read_bytes',side_effect=AssertionError('must not open')),self.assertRaises(ValueError):
            worker_allowance({},'E:/anything.json','0'*64,checked=checked_path)

    def test_feedback_clock_keeps_real_nonlate_and_late_separate(self):
        p=self.items[0];seconds=source_budget(p)['seconds']
        req=request_after_trigger(p['request'],p['raw'],p['choice'],run_id='test',event_id='test',
                                  wall_seconds=seconds,wall_limit=seconds)
        self.assertFalse(req['episode_feedback'][0]['deadline_exceeded'])
        raw=deepcopy(p['raw']);raw['elapsed_seconds']=seconds+1 # synthetic late fixture
        req=request_after_trigger(p['request'],raw,p['choice'],run_id='test',event_id='test',
                                  wall_seconds=seconds,wall_limit=seconds)
        self.assertTrue(req['episode_feedback'][0]['deadline_exceeded'])

    def test_same_binding_reaches_worker_and_fresh_feedback(self):
        import tools.run_skill_discovery as mod
        p=self.items[0];bound=source_budget(p);seen=[]
        def mocked(packet,folder,**kwargs):
            seen.append((packet,kwargs));(folder/'physics').mkdir(parents=True)
            write(folder/'physics/result.json',p['raw']);return deepcopy(p['raw'])
        case=dict(case_id='synthetic_boundary',source_task='02628',source_budget=bound,case_wall_seconds=bound['seconds'])
        with patch.object(mod,'actual_action',side_effect=mocked),patch.object(mod,'step_program',return_value={'response':None}):
            run_case(p,case,read(LIVE/'candidate_01/provider_proposal.json')['program'],self.root,'mock')
        self.assertEqual(seen[0][1]['source_budget'],bound)
        self.assertEqual(seen[0][0]['context']['remaining_wall_seconds'],bound['seconds'])
        req=read(self.root/'cases/synthetic_boundary/program_request.json')
        self.assertFalse(req['episode_feedback'][0]['deadline_exceeded'])

    def test_cumulative_budget_reserves_all_four_evaluations(self):
        from tools.run_multitask_roles import continuation_evaluation_allowed,continuation_physics_attempts
        for i in range(12):
            path=self.root/f'candidate_01/cases/c{i}/trigger/input.json'
            path.parent.mkdir(parents=True);write(path,{})
        self.assertEqual(continuation_physics_attempts(self.root),12)
        self.assertTrue(continuation_evaluation_allowed(self.root))
        path=self.root/'candidate_02/cases/c0/recovery/input.json'
        path.parent.mkdir(parents=True);write(path,{})
        self.assertFalse(continuation_evaluation_allowed(self.root))

    def test_runner_reuses_old_code_and_same_ledgers(self):
        from tools.run_multitask_learning import learn
        calls=[]
        def invoke(name,tool,args,limit):
            calls.append((args,limit));dest=Path(args[args.index('--run-dir')+1]);dest.mkdir()
            write(dest/'result.json',dict(status='NONAPPLICABILITY_REJECTED'));return 1
        result=learn(self.root,invoke,SimpleNamespace(summary=lambda:{}),
            learning_inputs=(MANIFEST,LIVE.parent/'prepared_v2/lesson_packet.json'),
            replay_from=LIVE/'candidate_01',budget_output=LIVE,source_case_clock=True)
        self.assertEqual(len(calls),1);self.assertIn('--source-case-clock',calls[0][0])
        self.assertIn('--replay-generated-from',calls[0][0])
        self.assertEqual(calls[0][0][calls[0][0].index('--ledger')+1],LIVE/'learner.sqlite')
        self.assertEqual(result['status'],'NOT_REGISTERED')

    def test_registry_rechecks_bound_source_budget(self):
        from tests.test_skill_validation import fixture_case,save
        from orchestrator.skill_validation import record_validation
        from orchestrator.skill_program import program_hash
        case,artifacts,program=fixture_case(self.root)
        raw_request=read(Path(artifacts['attempt_dir'])/'request.json')
        path=self.root/'synthetic_source_request.json';save(path,raw_request)
        seconds=raw_request['context']['remaining_wall_seconds']
        case.update(case_wall_seconds=seconds,source_budget=dict(
            source_request_path=str(path),source_request_sha256=sha(path),seconds=seconds))
        result=record_validation(case,artifacts,program_hash(program),True,True,program)
        self.assertEqual(result['verification_class'],'fixture_not_actual_physics')
        case['source_budget']['seconds']=seconds+1
        with self.assertRaisesRegex(ValueError,'source_budget'):
            record_validation(case,artifacts,program_hash(program),True,True,program)


if __name__=='__main__':unittest.main()
