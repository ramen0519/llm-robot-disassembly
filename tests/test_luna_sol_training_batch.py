"""Batch boundary tests only; no model or physical execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from tools import run_luna_sol_training_batch as batch
from tools.run_luna_experiment import read, write


class TrainingBatchTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='training_batch_OFFLINE_',dir=batch.ROOT/'runs')
        self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        network=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_PAID_API'))
        network.start();self.addCleanup(network.stop)
        self.pool=SimpleNamespace(summary=lambda:{'transport_guard':None})

    def test_fixed_models_role_reservations_and_shared_caps(self):
        p=batch.policy()
        self.assertFalse(p['enabled']);self.assertEqual(p['max_calls'],80)
        self.assertEqual(p['max_usd'],'10.00')
        self.assertEqual(sum(v['max_calls'] for v in p['roles'].values()),80)
        self.assertEqual(p['roles']['planner']['model'],'gpt-6.1-sol')
        self.assertEqual(p['episode']['fallback_calls'],0)

    def test_prepare_pins_sources_and_only_six_training_inputs(self):
        folder=self.root/'prepared';p=batch.prepare(folder)
        self.assertEqual(batch.validate(folder),p)
        self.assertEqual(len(read(folder/'schedule.json')['episodes']),6)
        self.assertEqual(p['training_ids'],['07009','02628','01289'])
        self.assertTrue(p['evaluation'].startswith('NOT_RUN'))
        with self.assertRaises(FileExistsError):batch.prepare(folder)

    def test_changed_preparation_and_missing_live_authority_are_rejected(self):
        folder=self.root/'prepared';batch.prepare(folder)
        with self.assertRaisesRegex(ValueError,'authorization'):
            batch.execute(folder,self.root/'live',False)
        p=read(folder/'policy_DISABLED.json');p['max_calls']=81
        import json
        (folder/'policy_DISABLED.json').write_text(json.dumps(p),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'scope_changed'):batch.validate(folder)

    def test_six_collection_summaries_are_exclusive_and_pool_all_observations(self):
        rows=read(batch.PROPOSAL/'schedule_DISABLED.json')['episodes'][:6]
        old=read(batch.ROOT/'runs/multitask_learning_live_20261105/live/collection.json')['episodes']
        samples=[deepcopy(old[i if i<5 else 2]) for i in range(6)]
        calls=[]
        def invoke(name,tool,args,limit):
            calls.append(args)
            self.assertEqual(args[args.index('--knowledge-policy')+1],'agent_tools_v1')
            return 0
        with patch.object(batch,'episode_summary',side_effect=samples):
            result=batch.collect(self.root,invoke,self.pool,rows)
        self.assertEqual(result['status'],'COLLECTED');self.assertEqual(len(calls),6)
        self.assertGreater(result['memory_records'],0)
        self.assertEqual(read(self.root/'collection.json'),result)
        self.assertEqual(len(list(self.root.glob('collection_*.json'))),6)

    def test_transport_latch_stops_collection_after_one_and_preserves_partial_memory(self):
        rows=read(batch.PROPOSAL/'schedule_DISABLED.json')['episodes'][:6]
        value=read(batch.ROOT/'runs/multitask_learning_live_20261105/live/collection.json')['episodes'][0]
        pool=SimpleNamespace(summary=lambda:{'transport_guard':{'blocked':True}})
        calls=[]
        with patch.object(batch,'episode_summary',return_value=value):
            result=batch.collect(self.root,lambda *a:calls.append(a) or 0,pool,rows)
        self.assertEqual(len(calls),1);self.assertEqual(result['status'],'STOPPED')
        self.assertGreater(result['memory_records'],0)

    def test_no_eligible_failure_does_not_generate_or_evaluate(self):
        prepared=self.root/'prepared';batch.prepare(prepared)
        with patch.object(batch,'collect',return_value={'status':'COLLECTED'}), \
             patch('tools.prepare_multitask_roles.prepare',side_effect=ValueError('no_shared_failure')), \
             patch.object(batch,'learn') as learn:
            result=batch.stages(self.root,prepared,lambda *a:0,self.pool)
        self.assertEqual(result['status'],'NO_ELIGIBLE_TRAINING_ROLES');learn.assert_not_called()
        self.assertEqual(result['evaluation'],'NOT_RUN')

    def test_learning_uses_new_contract_and_never_launches_evaluation(self):
        prepared=self.root/'prepared';batch.prepare(prepared)
        manifest=self.root/'manifest.json'
        with patch.object(batch,'collect',return_value={'status':'COLLECTED'}), \
             patch('tools.prepare_multitask_roles.prepare',return_value=manifest), \
             patch.object(batch,'learn',return_value={'status':'FROZEN'}) as learn:
            result=batch.stages(self.root,prepared,lambda *a:0,self.pool)
        self.assertEqual(learn.call_args.kwargs['observation_contract'],'source_bound_v1')
        self.assertEqual(result['status'],'FROZEN');self.assertEqual(result['evaluation'],'NOT_RUN')


if __name__=='__main__':unittest.main()
