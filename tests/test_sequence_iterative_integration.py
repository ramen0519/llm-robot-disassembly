"""Real iterative backend + real IPC/learning contracts, injected physics/models.

No external API, WSL or actual physics is executed. Resumption uses immutable
checkpoint artifacts and the exact same production source/configuration pins.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tests'))
from test_sequence_training_integration import InjectedLearner,RecordedFixtureEpisodes,write,read
from run_sequence_iterative_training import IterativeBackend,IterativeLearningService,current_round_packet
from orchestrator.sequence_iterative_training import run_iterative_training
from orchestrator.sequence_iterative_budget import stage_budget
from orchestrator.sequence_training_checkpoint import make_checkpoint,load_checkpoint
from orchestrator.sequence_learning_service import make_learning_request
from orchestrator.sequence_learning import build_sequence_learning_packet
from run_sequence_training import configs
from worker.task_memory import digest


class IterativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='iterative_OFFLINE_',dir=ROOT/'runs')
        self.root=Path(self.temp.name);self.root.relative_to(ROOT/'runs');self.addCleanup(self.temp.cleanup)
        self.tasks=[dict(assembly_id='training-'+str(i),part_count=3,base_part='base',
            assembly_dir='/mnt/c/dev/disassembly_codex_ready_v2/runs/fixture-input/training-'+str(i)) for i in range(5)]
        self.held=['evaluation-X','00335']
        self.profiles={s:stage_budget(s) for s in ('initial','continuation')}
        self.manifest=self.root/'manifest.json';write(self.manifest,dict(tasks=self.tasks,
            evaluation_task_ids=['evaluation-X'],development_holdout_ids=['00335']))
        env=patch.dict(os.environ,OPENAI_API_KEY='offline-iterative-fixture');env.start();self.addCleanup(env.stop)
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('HTTP_FORBIDDEN'))
        proc=patch('subprocess.run',side_effect=AssertionError('SUBPROCESS_FORBIDDEN'))
        self.net=net.start();self.proc=proc.start();self.addCleanup(net.stop);self.addCleanup(proc.stop)
        self.addCleanup(self.net.assert_not_called);self.addCleanup(self.proc.assert_not_called)

    def backend(self,name,*,learner=None,episodes=None,**state):
        folder=self.root/name;folder.mkdir()
        model=learner or InjectedLearner();worker=episodes or RecordedFixtureEpisodes()
        backend=IterativeBackend(folder,self.tasks,self.held,live=False,learner_transport=model,
            episode_runner=worker,**state)
        return backend,model,worker

    def test_two_wave_checkpoint_then_three_more_no_episode_replay(self):
        first,model,worker=self.backend('first')
        result=run_iterative_training(self.tasks,first,held_out_ids=self.held,stop_round=5)
        self.assertEqual(result['status'],'REVIEW_REQUIRED',result)
        self.assertEqual(len(result['completed_rounds']),10)
        self.assertEqual([c['seed'] for c in worker.calls if not c['validation']],[0]*5+[1]*5)
        self.assertEqual(len(model.requests),21) # first 3, nine duplicate rounds *2
        self.assertEqual(len([c for c in worker.calls if c['validation']]),2)
        self.assertEqual(len(first.skills),1)
        self.assertEqual(sum(r['learning_status']=='DUPLICATE_PROGRAM' for r in result['completed_rounds']),9)
        self.assertTrue(all(r['initial_lesson_retained'] for r in result['completed_rounds']))
        checkpoint=self.root/'checkpoint.json'
        make_checkpoint(checkpoint,manifest_path=self.manifest,source_pins=first.initial_source,
            budget_profiles=self.profiles,frozen=result['frozen'],completed_rounds=result['completed_rounds'],
            held_out_ids=self.held,allow_fixture=True)
        restored=load_checkpoint(checkpoint,manifest_path=self.manifest,source_pins=first.initial_source,
            budget_profiles=self.profiles,allow_fixture=True)
        prior={str(p):p.read_bytes() for p in (first.output/'frozen').rglob('*.json')}
        second,model2,worker2=self.backend('second',snapshots=restored['snapshots'],
            lessons=restored['lessons'],skills=restored['skills'])
        resumed=run_iterative_training(self.tasks,second,start_round=3,stop_round=5,
            held_out_ids=self.held,prior_rounds=restored['completed_rounds'],review_approved=True)
        self.assertEqual(resumed['status'],'COMPLETED',resumed)
        self.assertEqual(len(resumed['completed_rounds']),25)
        self.assertEqual([c['seed'] for c in worker2.calls],[2]*5+[3]*5+[4]*5)
        self.assertEqual(len(model2.requests),30)
        self.assertEqual(len(second.skills),1)
        self.assertEqual(len(second.lessons),26)
        self.assertEqual({str(p):p.read_bytes() for p in (first.output/'frozen').rglob('*.json')},prior)
        self.assertEqual(resumed['performance_effect'],'UNVERIFIED')
        self.assertTrue(second.skills[0]['fixture'])

    def test_current_round_packet_and_claim_reject_stale_only_reference(self):
        backend,_,_=self.backend('fresh')
        one=run_iterative_training(self.tasks,backend,stop_round=1,held_out_ids=self.held)
        self.assertEqual(one['status'],'ROUND_LIMIT_REACHED')
        newest=backend.latest
        packet=current_round_packet(backend.snapshots,newest)
        current=set(packet['current_round_evidence_ids'])
        old=next(f['evidence_id'] for f in packet['facts'] if f['evidence_id'] not in current)
        self.assertTrue(all(f['evidence_id'] in current for f in packet['facts'][:len(current)]))
        transport=InjectedLearner()
        def stale(u,h,b,t):
            raw=transport(u,h,b,t)
            value=json.loads(raw['output'][0]['content'][0]['text'])
            value['lesson']['evidence_ids']=[old]
            raw['output'][0]['content'][0]['text']=json.dumps(value)
            return raw
        svc=IterativeLearningService(configs()[1],self.root/'stale.sqlite',transport=stale)
        request=make_learning_request('fixture','fixture:stale','lesson_initial',dict(packet=packet))
        response=svc.learn(request)
        self.assertFalse(response['ok'])
        self.assertEqual(response['error'],'initial_lesson_missing_current_round_evidence')

    def test_host_failure_stops_before_any_learning_or_next_product(self):
        backend,model,worker=self.backend('failed',episodes=RecordedFixtureEpisodes(host_error=True))
        result=run_iterative_training(self.tasks,backend,held_out_ids=self.held)
        self.assertEqual(result['status'],'HALTED')
        self.assertFalse(result['completed_rounds']);self.assertEqual(len(worker.calls),1)
        self.assertFalse(model.requests)

    def test_input_e_path_guard_before_any_stat(self):
        with patch.object(Path,'lstat',side_effect=AssertionError('forbidden IO')):
            with self.assertRaises(ValueError):IterativeBackend(Path('E:/forbidden'),self.tasks,self.held)


if __name__=='__main__':unittest.main()
