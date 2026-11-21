"""Offline checkpoint tests; all observations and model artifacts are fixtures."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator import sequence_training_checkpoint as cp
from orchestrator.sequence_knowledge import build_native_snapshot, make_fixture_sequence_skill
from orchestrator.sequence_learning import (build_sequence_learning_packet,
    validate_sequence_lesson, freeze_sequence_lessons, freeze_sequence_skills)
from worker.task_memory import PROJECT, digest
from tests.test_sequence_knowledge import native_attempt, REVERSE


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


class SequenceCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='sequence_checkpoint_FIXTURE_', dir=PROJECT/'runs')
        self.root = Path(self.tmp.name)
        self.root.relative_to(PROJECT/'runs')
        self.addCleanup(self.tmp.cleanup)
        self.old = self.root/'initial'
        self.old.mkdir()
        self.checkpoint = self.old/'review_checkpoint.json'
        self.manifest = self.root/'manifest.json'
        self.ids = ['training-'+str(i) for i in range(5)]
        self.held = ['00335', 'evaluation-A']
        write(self.manifest, dict(tasks=[dict(assembly_id=i,role='training') for i in self.ids],
             evaluation_task_ids=['evaluation-A'],development_holdout_ids=['00335']))
        self.pins = {'orchestrator/fixture.py':'a'*64}
        self.budget = dict(sol=dict(model='fixture-sol',max_calls=25,max_usd=4),
                           luna=dict(model='fixture-luna',max_calls=4,max_usd=.1))
        attempt = native_attempt()
        worker_path = self.old/'fixture-worker.json'
        write(worker_path, dict(profile='sequence_native_v1',actual_physics=False,fixture=True,attempts=[attempt]))
        self.snapshot = build_native_snapshot(dict(task=dict(assembly_id=self.ids[0],
            input_sha256='b'*64,solver_settings=dict(seed=0,fixture=True))), [attempt],
            str(worker_path), cp.artifact_ref(worker_path)['sha256'])
        packet = build_sequence_learning_packet(self.snapshot)
        claim = dict(title='Fixture only',hypothesis='Never an actual model or physics result.',
            when_to_apply='Offline wiring test.',when_not_to_apply='Performance claims.',
            ranking_guidance='Complete permutation.',evidence_ids=[packet['facts'][0]['evidence_id']],counterexample_ids=[])
        self.lesson = validate_sequence_lesson(packet,claim,dict(origin='injected_transport'))
        self.skill = make_fixture_sequence_skill(REVERSE)
        self.skill['source_task_ids'] = [self.ids[0]]
        self.skill['validation']['run_refs'] = [str(worker_path)]
        self.skill['validation']['source_hashes'] = [cp.artifact_ref(worker_path)['sha256']]
        self.skill['content_sha256'] = digest({k:v for k,v in self.skill.items() if k != 'content_sha256'})
        self.frozen = {key:str(self.old/(key+'.json')) for key in ('memory','lessons','skills')}
        write(Path(self.frozen['memory']),[self.snapshot])
        write(Path(self.frozen['lessons']),freeze_sequence_lessons([self.lesson]))
        write(Path(self.frozen['skills']),freeze_sequence_skills([self.skill]))
        self.rows = []
        for wave in (1,2):
            for identity in self.ids:
                host = self.old/(identity+'_round'+str(wave))/'result.json'
                write(host,dict(status='UNKNOWN' if wave==1 else 'PASS',fixture=True,
                    metrics=dict(process_exit=0,service_error=None,sources_and_frozen_inputs_preserved=True)))
                self.rows.append(dict(assembly_id=identity,round_index=wave,status='COMPLETED',
                    episode_status='UNKNOWN' if wave==1 else 'PASS',learning_status='REGISTERED',
                    episode_ref=cp.artifact_ref(host)))
        self.claim_root = self.root/'claims'
        patched = patch.object(cp,'CLAIM_ROOT',self.claim_root)
        patched.start(); self.addCleanup(patched.stop)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_API_ALLOWED'))
        transport = net.start(); self.addCleanup(net.stop); self.addCleanup(transport.assert_not_called)

    def inputs(self, **changes):
        args = dict(manifest_path=self.manifest,source_pins=self.pins,budget_profiles=self.budget,
                    allow_fixture=True)
        args.update(changes)
        return args

    def make(self, **changes):
        args = self.inputs(frozen=self.frozen,completed_rounds=self.rows,held_out_ids=self.held)
        args.update(changes)
        return cp.make_checkpoint(self.checkpoint,**args)

    def claim(self, **changes):
        args = self.inputs(new_run_dir=self.root/'continuation',continue_after_review=True,
                           claim_root=self.claim_root)
        args.update(changes)
        return cp.claim_continuation(self.checkpoint,**args)

    def test_immutable_checkpoint_loads_exact_knowledge_and_starts_only_round_three(self):
        original = self.make()
        loaded = cp.load_checkpoint(self.checkpoint,**self.inputs())
        self.assertEqual(loaded['checkpoint'],original)
        self.assertEqual(loaded['snapshots'],[self.snapshot])
        self.assertEqual(loaded['lessons'],[self.lesson])
        self.assertEqual(loaded['skills'],[self.skill])
        self.assertEqual(loaded['completed_rounds'],self.rows)
        self.assertEqual(loaded['next_round_by_task'],dict.fromkeys(self.ids,3))
        before = self.checkpoint.read_bytes()
        with self.assertRaises(FileExistsError): self.make()
        self.assertEqual(self.checkpoint.read_bytes(),before)

    def test_explicit_review_and_one_shot_claim_with_no_previous_result_mutation(self):
        self.make()
        before = {p:cp.artifact_ref(p)['sha256'] for p in self.old.rglob('*.json')}
        with self.assertRaisesRegex(ValueError,'explicit_review'):
            self.claim(continue_after_review=False)
        claimed = self.claim()
        self.assertEqual(claimed['claim']['first_round'],3)
        with self.assertRaisesRegex(ValueError,'already_claimed_or_uncertain'):
            self.claim(new_run_dir=self.root/'different-continuation')
        self.assertEqual({p:cp.artifact_ref(p)['sha256'] for p in before},before)

    def test_concurrent_claim_has_exactly_one_winner(self):
        self.make()
        def attempt(index):
            try:
                self.claim(new_run_dir=self.root/('continuation-'+str(index)))
                return 'claimed'
            except ValueError as error:
                return str(error)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(attempt,[1,2]))
        self.assertEqual(results.count('claimed'),1)
        self.assertIn('checkpoint_continuation_already_claimed_or_uncertain',results)

    def test_partial_uncertain_wrong_wave_duplicate_or_holdout_rows_rejected(self):
        cases=[]
        cases.append(self.rows[:-1])
        for key,value in [('status','INCOMPLETE'),('learning_status','TRANSPORT_ERROR'),
                          ('episode_status','ERROR'),('halt_required',True),('assembly_id','00335'),
                          ('round_index',3)]:
            rows=deepcopy(self.rows);rows[0][key]=value;cases.append(rows)
        rows=deepcopy(self.rows);rows[0],rows[1]=rows[1],rows[0];cases.append(rows)
        rows=deepcopy(self.rows);rows[1]['episode_ref']=rows[0]['episode_ref'];cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows[0]), self.assertRaises(ValueError): self.make(completed_rounds=rows)
        self.assertFalse(self.checkpoint.exists())

    def test_referenced_host_error_cannot_be_promoted_to_completed(self):
        bad=self.old/'badhost.json'
        write(bad,dict(status='ERROR',metrics=dict(process_exit=0,service_error=None,
                                               sources_and_frozen_inputs_preserved=True)))
        rows=deepcopy(self.rows);rows[0]['episode_ref']=cp.artifact_ref(bad)
        with self.assertRaisesRegex(ValueError,'host_not_accepted'): self.make(completed_rounds=rows)

    def test_source_budget_manifest_and_frozen_knowledge_mutation_all_block(self):
        self.make()
        for changes,error in [({'source_pins':{'orchestrator/fixture.py':'b'*64}},'source_pins_changed'),
                              ({'budget_profiles':dict(self.budget,extra_call=True)},'budget_profile_changed')]:
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,error):
                cp.load_checkpoint(self.checkpoint,**self.inputs(**changes))
        other=self.root/'changed_manifest.json'
        write(other,json.loads(self.manifest.read_text(encoding='utf8')))
        with self.assertRaisesRegex(ValueError,'manifest_changed'):
            cp.load_checkpoint(self.checkpoint,**self.inputs(manifest_path=other))
        memory=Path(self.frozen['memory'])
        with memory.open('a',encoding='utf8') as stream:stream.write(' ')
        with self.assertRaisesRegex(ValueError,'artifact_hash_changed'):
            cp.load_checkpoint(self.checkpoint,**self.inputs())

    def test_checkpoint_content_tampering_is_rejected(self):
        self.make()
        changed=self.root/'tampered.json'
        value=json.loads(self.checkpoint.read_text(encoding='utf8'));value['max_rounds']=6
        write(changed,value)
        with self.assertRaisesRegex(ValueError,'hash_or_state'):
            cp.load_checkpoint(changed,**self.inputs())

    def test_fixture_cannot_be_used_for_live_continuation(self):
        with self.assertRaisesRegex(ValueError,'fixture_knowledge_not_live'): self.make(allow_fixture=False)
        self.make()
        with self.assertRaisesRegex(ValueError,'fixture_checkpoint_not_live'):
            cp.load_checkpoint(self.checkpoint,**self.inputs(allow_fixture=False))

    def test_heldout_native_memory_and_skill_sources_cannot_enter_checkpoint(self):
        snapshot=deepcopy(self.snapshot)
        row=snapshot['records'][0];row['source_task_context']['task']['assembly_id']='00335'
        row['record_sha256']=digest({k:v for k,v in row.items() if k!='record_sha256'})
        snapshot['content_sha256']=digest(snapshot['records'])
        path=self.root/'heldout-memory.json';write(path,[snapshot])
        with self.assertRaisesRegex(ValueError,'task_leakage'):
            self.make(frozen=dict(self.frozen,memory=str(path)))
        skill=deepcopy(self.skill);skill['source_task_ids']=['evaluation-A']
        skill['content_sha256']=digest({k:v for k,v in skill.items() if k!='content_sha256'})
        path=self.root/'heldout-skills.json';write(path,freeze_sequence_skills([skill]))
        with self.assertRaisesRegex(ValueError,'task_leakage'):
            self.make(frozen=dict(self.frozen,skills=str(path)))

    def test_changed_original_observation_cannot_be_loaded(self):
        self.make()
        path=Path(self.snapshot['records'][0]['source_ref'])
        with path.open('a',encoding='utf8') as stream:stream.write(' ')
        with self.assertRaisesRegex(ValueError,'artifact_hash_changed'):
            cp.load_checkpoint(self.checkpoint,**self.inputs())

    def test_forbidden_paths_are_rejected_before_any_filesystem_probe(self):
        with patch.object(Path,'lstat',side_effect=AssertionError('MUST_NOT_PROBE')) as probe:
            for path in ('E:/never-touch.json','/mnt/e/never-touch.json','../elsewhere.json'):
                with self.subTest(path=path),self.assertRaises(ValueError):cp.artifact_ref(path)
            probe.assert_not_called()

    def test_same_run_or_alternate_claim_root_cannot_bypass_single_claim(self):
        self.make()
        with self.assertRaisesRegex(ValueError,'requires_new_run'):
            self.claim(new_run_dir=self.old/'nested')
        with self.assertRaisesRegex(ValueError,'claim_root_is_fixed'):
            self.claim(claim_root=self.root/'other-claims')


if __name__=='__main__':
    unittest.main()
