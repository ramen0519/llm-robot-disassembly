from copy import deepcopy
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from tools import run_multitask_learning as live
from tools.run_luna_experiment import read,write
from tools.run_hierarchical_episode import parser
from worker.transfer_protocol import validate_manifest
from orchestrator.hierarchy_budget import validate_policy


class MultitaskLiveTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='multitask_live_UNIT_',dir=live.ROOT/'runs')
        self.addCleanup(t.cleanup);self.root=Path(t.name)

    def test_actual_commands_preserve_physics_seeds_and_roles(self):
        rows=read(live.PROPOSAL/'schedule_DISABLED.json')['episodes']
        for row in rows:
            args=parser().parse_args(live.episode_args(row,self.root,frozen=row['condition']=='frozen_training_memory_and_skills'))
            validate_manifest(args.evaluation_manifest,args.assembly_dir,args)
            self.assertTrue(args.allow_live_api);self.assertEqual(args.seed,row['seed'])
            self.assertEqual(args.budget,16);self.assertEqual(args.max_pose,128)
            self.assertEqual(args.max_skill_attempts,1)
            self.assertEqual(args.transport_failure_policy,'latch_on_error_v1')
            self.assertEqual('frozen_training' in str(args.memory),row['condition']=='frozen_training_memory_and_skills')

    def test_setup_new_exclusive_budget_never_adopts_prior_residual(self):
        out=self.root/'new';policy=live.setup(out)
        validate_policy(policy);self.assertTrue(policy['enabled'])
        self.assertEqual(policy['max_calls'],34);self.assertEqual(policy['max_usd'],'12.00')
        self.assertEqual(policy['roles']['learner']['max_calls'],4)
        self.assertEqual(policy['roles']['scout']['max_calls'],10)
        self.assertEqual(policy['roles']['planner']['max_calls'],20)
        self.assertFalse(read(out/'approval.json')['old_residual_reused'])
        live.verify_pins(out)
        with self.assertRaises(FileExistsError):live.setup(out)
        with patch.object(live,'runtime_hashes',return_value={}):
            with self.assertRaises(ValueError):live.verify_pins(out)

    def test_no_permission_means_no_directory_or_invocation(self):
        out=self.root/'forbidden'
        with self.assertRaises(ValueError):live.execute(out,'collect',False)
        self.assertFalse(out.exists())

    def test_timeout_unknown_error_never_common_failure(self):
        a=dict(status='UNKNOWN',deadline_exceeded=False,raw=dict(backend='ASAP_RedMax',path_check_calls=1,path_success=False))
        self.assertEqual(live.observed_family(a),'path_failed')
        for b in (dict(a,deadline_exceeded=True),dict(a,status='ERROR'),dict(a,raw=dict(a['raw'],backend='mock'))):
            self.assertIsNone(live.observed_family(b))
        b=dict(a,raw=dict(a['raw'],path_success=True,stability_success=False,stability_check_calls=1,
            stability_diagnostics={'phase':'dynamic_failure'}))
        self.assertEqual(live.observed_family(b),'dynamic_stability_failed')

    def test_source_selection_preserves_first_failure_not_later_pass(self):
        rows=[]
        a=dict(status='UNKNOWN',deadline_exceeded=False,raw=dict(backend='ASAP_RedMax',path_check_calls=1,path_success=False))
        for task in ('07009','02628','01289'):
            path=self.root/(task+'.json');write(path,dict(attempts=[a,dict(a,status='PASS')]))
            rows.append(dict(task_id=task,purpose='training',report=str(path),memory={'exists':'yes'}))
        chosen=live.choose_sources(dict(episodes=rows))
        self.assertEqual(chosen['selected_family'],'path_failed')
        self.assertTrue(all(r['attempt_index']==1 for r in chosen['selected'].values()))
        self.assertIsNone(live.choose_sources(dict(episodes=rows[:2]))['selected_family'])
        wrong=deepcopy(rows);wrong[0]['purpose']='evaluation'
        with self.assertRaises(ValueError):live.choose_sources(dict(episodes=wrong))

    def test_no_matching_family_stops_before_model_and_registration(self):
        out=self.root/'empty';out.mkdir();write(out/'collection.json',dict(status='COLLECTED',episodes=[]))
        def forbidden(*a):raise AssertionError('NO_API')
        result=live.learn(out,forbidden,None)
        self.assertEqual(result['status'],'NO_ELIGIBLE_SHARED_FAILURE')
        self.assertEqual(result['evaluation'],'NOT_RUN')

    def test_evaluation_cannot_start_without_registered_frozen_skill(self):
        write(self.root/'learn_result.json',dict(status='NOT_REGISTERED'))
        def forbidden(*a):raise AssertionError('NO_API')
        self.assertEqual(live.evaluate(self.root,forbidden,None)['status'],'NOT_RUN')

    def test_failed_lesson_stops_all_candidate_calls(self):
        write(self.root/'collection.json',dict(status='COLLECTED',episodes=[]))
        def invoke(name,tool,args,limit):
            self.assertEqual(name,'lesson_extract')
            out=self.root/'lesson_draft';out.mkdir();write(out/'result.json',dict(status='LESSON_REJECTED'));return 1
        with patch.object(live,'choose_sources',return_value=dict(selected_family='path_failed')),\
             patch.object(live,'learning_manifest',return_value=(self.root/'manifest',self.root/'packet')):
            # This control-flow fixture has no source artifacts. Exercise the
            # explicit legacy contract; source-bound input is covered separately.
            self.assertEqual(live.learn(self.root,invoke,None,observation_contract='legacy_v1')['status'],'LESSON_REJECTED')

    def run_learning_responses(self,responses):
        write(self.root/'collection.json',dict(status='COLLECTED',episodes=[]))
        calls=[]
        class Pool:
            def summary(self):return {'transport_guard':False}
        def invoke(name,tool,args,limit):
            calls.append((name,args));out=Path(args[args.index('--run-dir')+1]);out.mkdir()
            value=responses[name];write(out/'result.json',value)
            return 0 if value['status'] in ('CANDIDATE','REGISTERED_SCOPED') else 1
        with patch.object(live,'choose_sources',return_value=dict(selected_family='path_failed')),\
             patch.object(live,'learning_manifest',return_value=(self.root/'manifest',self.root/'packet')):
            result=live.learn(self.root,invoke,Pool(),observation_contract='legacy_v1')
        return result,calls

    def test_two_generation_slots_stop_without_implicit_third(self):
        result,calls=self.run_learning_responses({'lesson_extract':{'status':'CANDIDATE'},
            'candidate_01':{'status':'GENERATION_REJECTED'},'candidate_02':{'status':'PREFLIGHT_REJECTED'}})
        self.assertEqual([c[0] for c in calls],['lesson_extract','candidate_01','candidate_02'])
        self.assertIn('--generation-feedback-from',calls[2][1])
        self.assertEqual(result['status'],'NOT_REGISTERED')

    def test_admission_only_failure_cannot_drive_paid_revision(self):
        result,calls=self.run_learning_responses({'lesson_extract':{'status':'CANDIDATE'},
            'candidate_01':{'status':'NOT_REGISTERED','cases':[{'case_id':'source_07009','outcome':'PASS'},
                {'case_id':'source_02628','outcome':'PASS'},{'case_id':'source_01289','outcome':'FAIL'}]}})
        self.assertEqual(len(calls),2);self.assertEqual(result['status'],'NOT_REGISTERED')

    def test_debug_revision_then_failed_consolidation_cannot_publish(self):
        result,calls=self.run_learning_responses({'lesson_extract':{'status':'CANDIDATE'},
            'candidate_01':{'status':'PREFLIGHT_REJECTED'},
            'candidate_02':{'status':'REGISTERED_SCOPED','admission':{'admitted':True},
                'actual_llm_generation':True,'source_unchanged':True},
            'lesson_finalize':{'status':'LESSON_REJECTED'}})
        self.assertEqual(len(calls),4);self.assertIn('--abstention-feedback',calls[2][1])
        self.assertEqual(result['status'],'ADMITTED_LESSON_NOT_PUBLISHED')
        self.assertFalse((self.root/'frozen_training').exists())


if __name__=='__main__':unittest.main()
