"""Whole-cycle ordering and failure gates; injected stages are NOT physics."""
import unittest
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
from orchestrator.sequence_training_cycle import run_training_cycle
from run_sequence_training import validate_manifest, configs, NativeTrainingBackend


class Backend:
    def __init__(self, failure=None):
        self.halted=None; self.calls=[]; self.memory=[]; self.skills=[]
        self.failure=failure; self.candidates=0
    def event(self, kind, **kwargs): self.calls.append(kind)
    def freeze(self, name):
        self.calls.append(('freeze',name))
        return dict(memory=list(self.memory),skills=list(self.skills))
    def episode(self, task, seed, frozen):
        self.calls.append(('episode', task['assembly_id'], seed,deepcopy(frozen)))
        if self.failure=='network': self.halted='api_timeout'
        return dict(assembly_id=task['assembly_id'], seed=seed,reuse_status='NOT_SELECTED')
    def collect(self, result): self.memory.append(result); self.calls.append('collect')
    def packet(self,task,episode): return None if self.failure=='no_data' else {'fixture':True}
    def lesson(self,task,packet,**kwargs):
        self.calls.append('postlesson' if kwargs else 'lesson')
        return dict(ok=not (self.failure=='lesson' or (kwargs and self.failure=='postlesson')),lesson={'fixture':True})
    def candidate(self, task,packet,lesson,previous,feedback,revision):
        self.calls.append(('candidate',revision,previous,feedback)); self.candidates+=1
        return dict(ok=not(self.failure=='contract' and revision==0),candidate={'fixture':True,'version':revision})
    def validate_candidate(self,task,candidate,lesson):
        self.calls.append('physics_validation')
        return dict(ok=self.failure!='physics' and not(self.failure=='revision' and self.candidates==1),fixture=True)
    def register(self,candidate,lesson,proof):
        self.calls.append('register'); self.skills.append('fixture:skill')
        return {'skill_id':'fixture:skill'}


class CycleTests(unittest.TestCase):
    def run_case(self,failure=None):
        b=Backend(failure)
        r=run_training_cycle([dict(assembly_id='A')],b,held_out_ids=['E'])
        return b,r
    def test_complete_flow_freezes_between_learning_and_reuse(self):
        b,r=self.run_case()
        eps=[c for c in b.calls if isinstance(c,tuple) and c[0]=='episode']
        self.assertEqual(eps[0][3]['skills'],[])
        self.assertEqual(eps[1][3]['skills'],['fixture:skill'])
        self.assertEqual(eps[1][3]['memory'],[dict(assembly_id='A',seed=0,reuse_status='NOT_SELECTED')])
        self.assertLess(b.calls.index('postlesson'),b.calls.index('register'))
        self.assertEqual(r['tasks'][0]['reuse_status'],'NOT_SELECTED')
        self.assertEqual(r['performance_effect'],'UNVERIFIED')
    def test_contract_failure_reaches_new_candidate_with_feedback(self):
        b,r=self.run_case('contract')
        rev=[c for c in b.calls if isinstance(c,tuple) and c[0]=='candidate'][1]
        self.assertIsNotNone(rev[2]); self.assertFalse(rev[3]['ok'])
        self.assertEqual(b.calls.count('physics_validation'),1)
    def test_physics_failure_no_registration_no_false_success(self):
        b,r=self.run_case('physics')
        self.assertEqual(b.calls.count('physics_validation'),2)
        self.assertNotIn('register',b.calls); self.assertNotIn('postlesson',b.calls)
        self.assertIsNone(r['tasks'][0]['registered_skill'])
    def test_physics_revision_uses_feedback(self):
        b,r=self.run_case('revision'); self.assertIn('register',b.calls)
        rev=[c for c in b.calls if isinstance(c,tuple) and c[0]=='candidate'][1]
        self.assertFalse(rev[3]['ok'])
    def test_postlesson_failure_blocks_release(self):
        b,r=self.run_case('postlesson'); self.assertNotIn('register',b.calls)
    def test_network_failure_stops_other_tasks_and_paid_work(self):
        b=Backend('network')
        r=run_training_cycle([dict(assembly_id='A'),dict(assembly_id='B')],b)
        self.assertEqual(r['task_count'],1); self.assertNotIn('lesson',b.calls)
        self.assertTrue(r['stopped'])
    def test_missing_data_no_model_learning(self):
        b,r=self.run_case('no_data'); self.assertNotIn('lesson',b.calls)
    def test_heldout_and_duplicate_training_rejected(self):
        for tasks in ([dict(assembly_id='A')]*2,[dict(assembly_id='E')]):
            with self.assertRaises(ValueError):run_training_cycle(tasks,Backend(),held_out_ids=['E'])
    def test_cumulative_sources_cross_products(self):
        b=Backend();r=run_training_cycle([dict(assembly_id='A'),dict(assembly_id='B')],b)
        eps=[c for c in b.calls if isinstance(c,tuple) and c[0]=='episode']
        self.assertEqual(len(eps[2][3]['memory']),2)
        self.assertEqual(len(r['frozen']['memory']),4)
    def test_batch_caps_are_sum_of_finite_scoped_caps(self):
        sol,luna=configs()
        self.assertEqual(10*sol['max_calls']+5*luna['max_calls'],270)
        self.assertEqual(10*sol['max_usd']+5*luna['max_usd'],40.5)
    def test_forbidden_input_drive_rejected_before_io(self):
        tasks=[dict(assembly_id=str(i),part_count=5,assembly_dir='E:/not-accessed') for i in range(5)]
        with self.assertRaisesRegex(ValueError,'outside_allowed'):validate_manifest({'tasks':tasks})
    def test_offline_backend_requires_fake_services(self):
        # Before any file creation / credential retrieval.
        with self.assertRaisesRegex(ValueError,'explicit_injected'):
            NativeTrainingBackend(ROOT/'runs/uncreated_sequence_backend',[],live=False)


if __name__=='__main__': unittest.main()
