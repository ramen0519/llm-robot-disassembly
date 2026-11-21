"""Opt-in model upgrade, fake provider only. No API/physics success claims."""
from copy import deepcopy
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from tests.test_skill_discovery import config, synthesis, raw_response
from tests.test_lesson_pipeline import claim
from tests.test_hierarchical_planner import policy as old_policy
from orchestrator.hierarchy_budget import HierarchyBudget, validate_policy, estimate_request
from orchestrator.python_skill_discovery import PythonSkillDiscoveryService
from orchestrator.lesson_discovery import LessonDiscoveryService
from orchestrator.llm_ranker import RankService
from tools.run_luna_experiment import read, write

ROOT=Path(__file__).absolute().parents[1]
spec=importlib.util.spec_from_file_location('astra_runner',ROOT/'runs/astra_skill_learning_20261109_0918/run_learning.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def cfg(**kw):
    return config(model='gpt-6-astra',reasoning_effort='medium',max_output_tokens=25000,
        max_usd='15',input_usd_per_million='10',output_usd_per_million='50',**kw)

class AstraLearnerTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='astra_OFFLINE_',dir=ROOT/'runs')
        self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.net=net.start();self.addCleanup(net.stop);self.addCleanup(self.net.assert_not_called)
        env=patch.dict(os.environ,DISCOVERY_FIXTURE_KEY='OFFLINE_NOT_A_KEY')
        env.start();self.addCleanup(env.stop)

    def test_astra_25000_configuration_and_full_reservation(self):
        s=PythonSkillDiscoveryService(cfg(),self.root/'x.sqlite')
        self.assertIsNone(s._configuration()[0])
        p=synthesis();p['timeout_seconds']=600
        v=s.preview(p)
        self.assertEqual(v['body']['max_output_tokens'],25000)
        expected=(Decimal(v['body_bytes']+2048)*10+Decimal(25000)*50)/1000000
        self.assertEqual(Decimal(v['cost_estimate']['usd']),expected)
        self.assertEqual(RankService(cfg(),self.root/'rank.sqlite')._configuration()[0],'invalid_config')

    def test_legacy_caps_and_timeout_preserved(self):
        for model in ('gpt-6-luna','gpt-6.1-sol'):
            s=PythonSkillDiscoveryService(config(model=model),self.root/(model+'.sqlite'))
            p=synthesis();p['context']['python_output_tokens']=25000
            with self.assertRaisesRegex(ValueError,'output_cap'):s.preview(p)
            p=synthesis();p['timeout_seconds']=600
            with self.assertRaisesRegex(ValueError,'invalid_timeout'):s.preview(p)

    def test_astra_rates_reasoning_and_cap_fail_closed(self):
        for n,update in enumerate(({'input_usd_per_million':'.1'},{'output_usd_per_million':'.5'},
                       {'reasoning_effort':'none'},{'max_output_tokens':25001})):
            s=PythonSkillDiscoveryService(dict(cfg(),**update),self.root/('bad%d.sqlite'%n))
            self.assertIsNotNone(s._configuration()[0])
        p=synthesis();p['context']['python_output_tokens']=25001
        with self.assertRaises(ValueError):PythonSkillDiscoveryService(cfg(),self.root/'p.sqlite').preview(p)

    def test_learner_only_policy_preserves_old_models(self):
        old=old_policy();self.assertEqual(validate_policy(old),old)
        new=runner.policy();self.assertEqual(new['roles']['learner']['model'],'gpt-6-astra')
        self.assertEqual(sum(c['max_calls'] for c in new['roles'].values()),4)
        for role in ('scout','planner'):
            bad=deepcopy(new);bad['roles'][role]['model']='gpt-6-astra'
            with self.assertRaises(ValueError):validate_policy(bad)
        bad=deepcopy(new);bad['roles']['learner']['output_usd_per_million']='.5'
        with self.assertRaises(ValueError):validate_policy(bad)
        bad=deepcopy(new);bad['roles']['learner']['model']='gpt-6-luna'
        with self.assertRaises(ValueError):validate_policy(bad)

    def test_budget_immutable_and_role_cannot_be_used(self):
        p=runner.policy();pool=HierarchyBudget(self.root/'pool.sqlite',p)
        changed=deepcopy(p);changed['max_usd']='16'
        with self.assertRaises(ValueError):HierarchyBudget(self.root/'pool.sqlite',changed)
        send=Mock();call=pool.transport('planner',send)
        body=dict(model='gpt-6.1-sol',store=False,reasoning={'effort':'medium'},max_output_tokens=4096)
        with self.assertRaisesRegex(ValueError,'limit'):call('https://api.openai.com/v1/responses',{},body,10)
        send.assert_not_called()

    def test_shared_budget_stops_fifth_attempt(self):
        pool=HierarchyBudget(self.root/'budget.sqlite',runner.policy())
        send=Mock(return_value={'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}})
        call=pool.transport('learner',send)
        for n in range(5):
            body=dict(model='gpt-6-astra',store=False,reasoning={'effort':'medium'},
                      max_output_tokens=25000,input=[str(n)])
            if n<4:call('https://api.openai.com/v1/responses',{},body,600)
            else:
                with self.assertRaisesRegex(ValueError,'limit'):call('https://api.openai.com/v1/responses',{},body,600)
        self.assertEqual(send.call_count,4)
        self.assertEqual(pool.summary()['remaining_calls'],0)

    def test_timeout_latches_without_retry(self):
        pool=HierarchyBudget(self.root/'timeout.sqlite',runner.policy(),transport_failure_policy='latch_on_error_v1')
        send=Mock(side_effect=TimeoutError('fixture'));call=pool.transport('learner',send)
        body=dict(model='gpt-6-astra',store=False,reasoning={'effort':'medium'},max_output_tokens=25000)
        with self.assertRaises(TimeoutError):call('https://api.openai.com/v1/responses',{},body,600)
        with self.assertRaises(Exception):call('https://api.openai.com/v1/responses',{},dict(body,input=['new']),600)
        self.assertEqual(send.call_count,1);self.assertTrue(pool.summary()['transport_guard'])

    def test_fake_astra_lesson_parse_and_cache(self):
        p=synthesis();p['context']['lesson_stage']='extract';p['timeout_seconds']=600
        value=dict(request_id=p['request_id'],state_id=p['state_id'],lesson=claim(p))
        raw=dict(raw_response(value),model='gpt-6-astra');send=Mock(return_value=raw)
        s=LessonDiscoveryService(cfg(),self.root/'lesson.sqlite',send)
        a=s.propose(p);b=s.propose(p)
        self.assertEqual(a['status'],'proposed');self.assertEqual(b['status'],'proposed')
        self.assertEqual(send.call_count,1)
        self.assertEqual(a['metadata']['actual_cost_from_usage_usd'],'0.011')

    def test_incomplete_and_bad_lesson_not_admitted(self):
        p=synthesis();p['context']['lesson_stage']='extract'
        bad=dict(claim(p),evidence_ids=['invented'])
        for n,raw in enumerate([dict(raw_response({}),model='gpt-6-astra',status='incomplete'),
            dict(raw_response(dict(request_id=p['request_id'],state_id=p['state_id'],lesson=bad)),model='gpt-6-astra')]):
            s=LessonDiscoveryService(cfg(),self.root/(str(n)+'.sqlite'),Mock(return_value=raw))
            r=s.propose(p);self.assertNotEqual(r['status'],'proposed')
            self.assertFalse(r['metadata']['registered'])

    def test_extract_preview_keeps_astra_capacity_and_legacy_2048(self):
        from tools.run_lesson_discovery import extract
        for name,c in [('astra',dict(cfg(),enabled=False,max_calls=0,max_usd=0)),
                       ('luna',config(enabled=False,max_calls=0,max_usd=0))]:
            extract(synthesis(),self.root/name,c)
            packet=read(self.root/name/'packet.json');preview=read(self.root/name/'preview.json')
            self.assertEqual(preview['body']['max_output_tokens'],25000 if name=='astra' else 2048)
            self.assertEqual(packet.get('timeout_seconds',30),600 if name=='astra' else 30)

    def test_multitask_stage_arguments_keep_model_caps(self):
        from tools.run_multitask_learning import learn
        from types import SimpleNamespace
        out=self.root/'workflow';out.mkdir();write(out/'learner_config.json',cfg())
        calls=[]
        def invoke(name,tool,args,limit):
            calls.append((name,args,limit));dest=Path(args[args.index('--run-dir')+1]);dest.mkdir()
            if name=='lesson_extract':write(dest/'result.json',dict(status='CANDIDATE'));return 0
            write(dest/'result.json',dict(status='GENERATION_REJECTED'));return 1
        value=learn(out,invoke,SimpleNamespace(summary=lambda:{}),
            learning_inputs=(runner.INPUT/'manifest.json',runner.INPUT/'lesson_packet.json'))
        self.assertEqual(value['status'],'NOT_REGISTERED')
        self.assertEqual([c[0] for c in calls],['lesson_extract','candidate_01','candidate_02'])
        self.assertEqual(calls[0][2],660)
        for _,args,_ in calls[1:]:
            self.assertEqual(args[args.index('--generation-output-tokens')+1],'25000')
            self.assertEqual(args[args.index('--generation-timeout-seconds')+1],'600')

if __name__=='__main__':unittest.main()
