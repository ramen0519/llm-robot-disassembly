from copy import deepcopy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json,tempfile,unittest
from unittest.mock import patch
from tools import prepare_multitask_learning as prep
from tools.multitask_skill_evidence import build_packet,training_memory_sources,validate_roles,common_failure_families
from tools.run_luna_experiment import read,write,sha
from tools.run_python_skill_cycle import python_generation_packet
from orchestrator.lesson_discovery import LessonDiscoveryService
from tools.run_skill_discovery import preview_config
from worker.transfer_protocol import validate_manifest

ROOT=prep.ROOT;READY=ROOT/'runs/multitask_learning_prep_20261105/prepared_v2'
CHECK=ROOT/'runs/multitask_learning_prep_20261105/offline_evidence_v3'


class MultiTaskTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='multitask_UNIT_',dir=ROOT/'runs')
        self.addCleanup(t.cleanup);self.root=Path(t.name)

    def test_split_has_known_train_and_previously_unexposed_eval(self):
        split=read(READY/'split.json');exposure=read(READY/'exposure_audit.json')
        train=[x['id'] for x in split['training']];ev=[x['id'] for x in split['evaluation']]
        validate_roles(train,ev)
        self.assertEqual(train,list(prep.TRAIN));self.assertEqual(ev,['02680','02656'])
        self.assertFalse(set(ev)&set(exposure['exposed']))
        self.assertFalse(exposure['unreadable_or_oversize'])

    def test_selection_is_order_independent_and_excludes_related(self):
        inv=[dict(id=i,parts=n) for i,n in zip(prep.TRAIN,[5,7,9])]+[
            dict(id='11111',parts=6),dict(id='22222',parts=7),dict(id='33333',parts=10),dict(id='44444',parts=9)]
        sig=lambda i:Counter({i:5})
        a=prep.select(inv,{'22222'},sig);b=prep.select(list(reversed(inv)),{'22222'},sig)
        self.assertEqual(a,b)
        self.assertNotIn('22222',[x['id'] for x in a[1]])
        related=lambda i:Counter({'same':5}) if i in ('11111','07009') else sig(i)
        with self.assertRaises(ValueError):prep.select(inv,{'22222'},related)

    def test_obj_invalid_values_and_faces_rejected(self):
        good=b'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'
        self.assertEqual(prep.obj_check(good)['vertices'],3)
        for bad in (good.replace(b'1 2 3',b'1 2 4'),good.replace(b'1 0 0',b'nan 0 0'),b'v 0 0 0\n'):
            with self.assertRaises(ValueError):prep.obj_check(bad)

    def test_excluded_drive_rejected_before_metadata_access(self):
        with patch.object(Path,'exists',side_effect=AssertionError('NO_IO')):
            for path in ('E:/blocked','/mnt/e/blocked'):
                with self.assertRaises(ValueError):prep.safe(path)

    def test_budget_is_disabled_and_old_residual_not_adopted(self):
        p,c=prep.budget_policy()
        self.assertFalse(p['enabled']);self.assertFalse(c['old_residual_reused'])
        self.assertEqual(p['max_calls'],34);self.assertEqual(p['max_usd'],'12.00')
        self.assertEqual(c['worst_configured_reservation'],'11.7592064')

    def test_actual_two_sources_and_no_admission_read(self):
        manifest=read(CHECK/'integration_manifest_NOT_LIVE.json')
        p=build_packet(manifest,'offline-unit-multi')
        self.assertEqual({e['source_task'] for e in p['evidence']},{'07009','02628'})
        self.assertEqual(len(p['evidence']),30)
        self.assertIsNone(p['context']['multi_task_learning']['source_to_target_part_mapping'])
        self.assertFalse(p['context']['multi_task_learning']['evaluation_tasks_exposed'])
        self.assertEqual([e['summary']['trace_attempt_count'] for e in p['context']['automatically_extracted_same_condition_pairs']['sources']],[7,8])

    def test_evaluation_or_duplicate_source_rejected_before_source_read(self):
        m=read(CHECK/'integration_manifest_NOT_LIVE.json')
        for ident in ('02680','07009'):
            p=deepcopy(m);p['experience_sources'][1]['task_id']=ident
            with self.assertRaises(ValueError):build_packet(p,'invalid')
        p=deepcopy(m);p['cases'][2]['source_task']='02656'
        with self.assertRaises(ValueError):build_packet(p,'invalid')

    def test_modified_split_or_source_manifest_rejected(self):
        m=read(CHECK/'integration_manifest_NOT_LIVE.json')
        for k in ('split','source'):
            p=deepcopy(m)
            if k=='split':p['split']['sha256']='0'*64
            else:p['experience_sources'][0]['manifest_sha256']='0'*64
            with self.assertRaises(ValueError):build_packet(p,'invalid')

    def test_actual_provider_preview_keeps_both_contexts_and_all_facts(self):
        p=read(CHECK/'program_packet.json')
        service=LessonDiscoveryService(dict(preview_config(),max_output_tokens=4096),self.root/'preview.sqlite')
        try:r=service.preview(p)
        finally:service.close()
        body=json.loads(r['body']['input'][0]['content'])
        self.assertEqual(len(body['context']['runtime_contract']['examples']),2)
        self.assertEqual(len(body['evidence']),30)
        self.assertLess(r['body_bytes'],196608)
        self.assertGreater(r['body_bytes'],100000)
        self.assertFalse(body['context']['admission_inputs_supplied'])
        self.assertEqual(body['context']['multi_task_learning']['source_tasks'],['07009','02628'])

    def test_training_snapshot_identity_and_eval_rejection(self):
        split=read(READY/'split.json')
        path=ROOT/'runs/stability_support_review_20261105/live/episode/action_memory_frozen.json'
        source=dict(task_id='02628',purpose='training',path=str(path),sha256=sha(path))
        self.assertEqual(training_memory_sources(split,[source]),[path])
        for field,value in [('task_id','02680'),('purpose','evaluation'),('sha256','0'*64),('task_id','07009')]:
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                training_memory_sources(split,[dict(source,**{field:value})])

    def test_manifest_keeps_seed_settings_and_split_boundaries(self):
        settings=dict(budget=16,state_action_budget=6,max_gripper=3,max_pose=128,timeout=480,wall_timeout=600,response_timeout=120)
        a=SimpleNamespace(**settings,seed=1,base_part='none')
        assembly='/opt/asap-research/multitask_learning_20261105/07009'
        p=dict(schema='multitask_learning_v1',purpose='training',seed=1,automatic_retries=0,
            settings=settings,training_ids=list(prep.TRAIN),evaluation_ids=['02680','02656'],
            assembly_dirs={assembly:{}},path_limit=96,stability_limit=48)
        path=self.root/'protocol.json';write(path,p)
        self.assertEqual(validate_manifest(path,assembly,a),p)
        for field,value in [('seed',0),('max_gripper',4),('base_part','0')]:
            args=SimpleNamespace(**vars(a));setattr(args,field,value)
            with self.assertRaises(ValueError):validate_manifest(path,assembly,args)
        with self.assertRaises(ValueError):validate_manifest(path,assembly.replace('07009','02680'),a)
        bad=dict(p,purpose='evaluation');dest=self.root/'bad.json';write(dest,bad)
        with self.assertRaises(ValueError):validate_manifest(dest,assembly,a)

    def test_failure_gate_does_not_count_duplicate_memory_as_independent_trials(self):
        p=read(CHECK/'bound_packet.json');audit=common_failure_families(p)
        self.assertEqual(audit['independent_trace_observations'],15)
        self.assertFalse(audit['shared_cause_established'])
        q=deepcopy(p)
        for f in q['evidence']:
            f['deadline_exceeded']=True
        self.assertFalse(common_failure_families(q)['eligible_for_generation'])
        q=deepcopy(p)
        for f in q['evidence']:
            if f['source_task']=='02628':f['status']='PASS'
        self.assertFalse(common_failure_families(q)['eligible_for_generation'])
        q=deepcopy(p);q['evidence'][0]['source_task']='02680'
        with self.assertRaises(ValueError):common_failure_families(q)
        manifest=read(CHECK/'integration_manifest_NOT_LIVE.json')
        manifest['require_shared_failure_family']=True
        if audit['eligible_for_generation']:
            self.assertTrue(build_packet(manifest,'gate-test')['context']['shared_failure_family_gate']['eligible_for_generation'])
        else:
            with self.assertRaisesRegex(ValueError,'no_measured_shared_failure_family'):
                build_packet(manifest,'gate-test')

    def test_disabled_schedule_has_matched_conditions_and_no_live_permission(self):
        from tools.prepare_multitask_schedule import episode_order
        split=read(READY/'split.json');order=episode_order(split)
        self.assertEqual(len(order),10)
        self.assertEqual([(x['task'],x['seed']) for x in order[:6]],[(i,s) for s in (0,1) for i in prep.TRAIN])
        root=READY.parent/'final_proposal';schedule=read(root/'schedule_DISABLED.json')['episodes']
        self.assertEqual([x['condition'] for x in order[6:]],['empty_knowledge','frozen_training_memory_and_skills',
            'frozen_training_memory_and_skills','empty_knowledge'])
        self.assertEqual(len(schedule),10)
        manifests=[read(Path(x['manifest'])) for x in schedule]
        self.assertEqual(len({json.dumps(x['settings'],sort_keys=True) for x in manifests}),1)
        for x in schedule:
            self.assertFalse(x['enabled']);self.assertNotIn('--allow-live-api',x['argv'])
            self.assertEqual(sha(Path(x['manifest'])),x['manifest_sha256'])
        self.assertFalse(read(root/'policy_DISABLED.json')['enabled'])
        self.assertFalse(read(root/'empty_knowledge/skills.json')['entries'])
        self.assertFalse(read(root/'empty_knowledge/memory.json')['records'])


if __name__=='__main__':unittest.main()
