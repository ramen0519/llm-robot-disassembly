"""Synthetic file/transaction fixtures; no API, WSL or actual physics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orchestrator import knowledge_store as ks
from orchestrator.knowledge_index import build_index, observed_signature
from orchestrator.explicit_action_memory import append_action_observation, freeze_memory, RECORD_KEYS
from orchestrator.skill_registry import SkillRegistry
from tests.test_explicit_action_memory import outcome
from tests.test_explicit_action_contract import request
from worker.explicit_action_contract import canonical_response
from tools.run_generated_skills import resolve_knowledge_inputs


class StoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="knowledge_fixture_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store, self.memory, self.skills = self.root/'store', self.root/'memory.json', self.root/'skills.json'
        req = request()
        append_action_observation(self.root/'events.jsonl', req, canonical_response(req, None), outcome(), allow_test_root=True)
        freeze_memory(self.root/'events.jsonl', self.memory, allow_test_root=True)
        SkillRegistry(self.root/'registry', allow_test_root=True, fixture=True).freeze(self.skills)
        self.options = dict(allow_fixture=True, allow_test_root=True)

    def create(self, revision='rev_one', **changes):
        return ks.create_revision(self.store, revision, self.memory, self.skills,
            reason='synthetic unit fixture', **dict(self.options, **changes))

    def activate(self, revision='rev_one', expected=None):
        return ks.promote(self.store, revision, expected_active_sha256=expected,
                          reason='unit fixture only', **self.options)

    def test_original_bytes_and_thirteen_key_schema_unchanged(self):
        before = self.memory.read_bytes(), self.skills.read_bytes()
        manifest = self.create()
        loaded = ks.load_revision(self.store, 'rev_one', **self.options)
        self.assertEqual(Path(loaded['memory']).read_bytes(), before[0])
        self.assertEqual(Path(loaded['skills']).read_bytes(), before[1])
        self.assertEqual((self.memory.read_bytes(), self.skills.read_bytes()), before)
        self.assertEqual(set(json.loads(before[0])['records'][0]), RECORD_KEYS)
        self.assertFalse((self.store/'ACTIVE.json').exists())
        self.assertFalse(manifest['hard_pruning'])

    def test_no_revision_overwrite(self):
        self.create()
        with self.assertRaises(FileExistsError): self.create()

    def test_fixture_not_admitted_as_actual(self):
        self.create()
        with self.assertRaisesRegex(ValueError, 'fixture_bundle'):
            ks.load_revision(self.store, 'rev_one', allow_test_root=True)

    def test_first_and_second_activation_cas(self):
        self.create(); first=self.activate(); self.create('rev_two')
        with self.assertRaisesRegex(ValueError, 'compare_and_swap'):
            self.activate('rev_two')
        second=self.activate('rev_two', first['content_sha256'])
        self.assertEqual(second['previous_active_sha256'], first['content_sha256'])
        self.assertEqual(ks.active_pointer(self.store, allow_test_root=True)['revision'], 'rev_two')

    def test_old_pin_survives_new_active_revision(self):
        self.create(); first=self.activate()
        pin=ks.resolve_bundle(self.store, **self.options)['pin']
        self.create('rev_two'); self.activate('rev_two', first['content_sha256'])
        self.assertEqual(ks.verify_pin(pin, **self.options)['manifest']['revision'], 'rev_one')

    def test_evaluation_must_name_revision(self):
        self.create(); self.activate()
        with self.assertRaisesRegex(ValueError, 'explicit_revision'):
            ks.resolve_bundle(self.store, purpose='evaluation', **self.options)
        self.assertEqual(ks.resolve_bundle(self.store, 'rev_one', purpose='evaluation', **self.options)['pin']['purpose'], 'evaluation')

    def test_evaluation_revision_cannot_activate(self):
        self.create(purpose='evaluation')
        with self.assertRaisesRegex(ValueError, 'cannot_update_active'): self.activate()
        self.assertFalse((self.store/'ACTIVE.json').exists())

    def test_file_tamper_rejected(self):
        self.create(); path=self.store/'revisions/rev_one/memory.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'file_hash'):
            ks.load_revision(self.store, 'rev_one', **self.options)

    def test_partial_revision_not_readable(self):
        (self.store/'revisions/rev_one').mkdir(parents=True)
        with self.assertRaises(FileNotFoundError): ks.load_revision(self.store, 'rev_one', **self.options)

    def test_interpreter_change_blocks_execution_resolution(self):
        self.create()
        with patch.object(ks, 'source_fingerprints', return_value={'different':'version'}):
            with self.assertRaisesRegex(ValueError, 'interpreter_or_derivation_changed'):
                ks.resolve_bundle(self.store, 'rev_one', **self.options)

    def test_pin_tamper_rejected(self):
        self.create(); pin=ks.resolve_bundle(self.store, 'rev_one', **self.options)['pin']
        pin['revision']='rev_other'
        with self.assertRaisesRegex(ValueError, 'invalid_knowledge'): ks.verify_pin(pin, **self.options)

    def test_pointer_tamper_rejected(self):
        self.create(); self.activate()
        p=self.store/'ACTIVE.json'; value=json.loads(p.read_text()); value['revision']='rev_other'
        p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'invalid_knowledge'): ks.active_pointer(self.store, allow_test_root=True)

    def test_busy_lock_fails_without_removing_other_writer_lock(self):
        self.create(); lock=self.store/'.promotion.lock'; lock.write_text('another writer')
        with self.assertRaises(FileExistsError): self.activate()
        self.assertEqual(lock.read_text(),'another writer')

    def test_interrupted_promotion_preserves_prior_active(self):
        self.create(); first=self.activate(); self.create('rev_two')
        with patch.object(ks.os, 'replace', side_effect=OSError('injected interruption')):
            with self.assertRaises(OSError): self.activate('rev_two', first['content_sha256'])
        self.assertEqual(ks.active_pointer(self.store, allow_test_root=True),first)
        self.assertEqual(len(list((self.store/'history').glob('*_committed.json'))),1)

    def test_forbidden_drive_rejected_before_filesystem_access(self):
        with patch.object(Path,'lstat',side_effect=AssertionError('filesystem access')):
            with self.assertRaises(ValueError):
                ks.create_revision(self.store,'rev_one','E:/never-read.json',self.skills,reason='test',**self.options)

    def test_root_knowledge_does_not_expand_guard(self):
        with self.assertRaisesRegex(ValueError, 'within project runs'):
            ks.load_revision(ks.ROOT/'knowledge','rev_one')

    def test_traversal_revision_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid_knowledge_revision'): self.create('../rev_one')

    def test_duplicate_json_key_rejected(self):
        p=self.root/'duplicate.json';p.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError,'duplicate_json_key'):ks.read_json(p,True)

    def test_source_changes_leave_no_committed_manifest(self):
        original=ks.read_json; count=0
        def changed(path, allow_test_root=False):
            nonlocal count
            value,raw=original(path,allow_test_root)
            count+=1
            return value,raw+b' ' if count==3 else raw
        with patch.object(ks,'read_json',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'source_changed'):self.create()
        self.assertFalse((self.store/'revisions/rev_one/manifest.json').exists())


class IndexTests(unittest.TestCase):
    def test_unknown_is_not_collision_or_constraint(self):
        s=observed_signature(dict(status='UNKNOWN',path_success=False,stability_success=None,raw={}),{})
        self.assertEqual(s['observed_failure_stage'],'path')
        self.assertIsNone(s['failure_cause']);self.assertIsNone(s['blocker'])
        self.assertIsNone(s['deadline_exceeded']);self.assertFalse(s['hard_pruning'])

    def test_unknown_missing_stage_remains_null(self):
        s=observed_signature(dict(status='ERROR',path_success=None,stability_success=None,raw={}),{})
        self.assertIsNone(s['observed_failure_stage']);self.assertIsNone(s['stability_phase'])

    def test_invalid_diagnostics_not_invented(self):
        s=observed_signature(dict(status='UNKNOWN',path_success=True,stability_success=False,
                                 raw={'stability_diagnostics':{'phase':'collision'}}),{})
        self.assertEqual(s['diagnostic_error'],'invalid_stability_diagnostics')
        self.assertIsNone(s['stability_phase'])

    def test_two_leg_direction_preserved_without_frame_inference(self):
        s=observed_signature({'raw':{}},{'direction_id':'zp','second_direction_id':'xn','support_ids':[]})
        self.assertEqual((s['direction_id'],s['second_direction_id']),('zp','xn'))
        self.assertIsNone(s['direction_class'])


class RunnerInputTests(unittest.TestCase):
    def args(self,**changes):
        return SimpleNamespace(**dict(dict(knowledge_store=None,knowledge_pin=None,knowledge_revision=None,
            knowledge_purpose='development',skill_library=None,memory=None,allow_fixture_library=False),**changes))

    def test_legacy_explicit_paths_do_not_read_active(self):
        with patch('tools.run_generated_skills.resolve_bundle',side_effect=AssertionError('ACTIVE accessed')):
            self.assertIsNone(resolve_knowledge_inputs(self.args(skill_library=Path('existing.json'))))

    def test_bundle_paths_passed_to_existing_runner(self):
        args=self.args(knowledge_store=Path('runs/store'),knowledge_revision='rev_one')
        bundle=dict(memory='runs/store/memory.json',skills='runs/store/skills.json',pin={'immutable':'pin'})
        with patch('tools.run_generated_skills.resolve_bundle',return_value=bundle):
            self.assertEqual(resolve_knowledge_inputs(args),bundle['pin'])
        self.assertEqual(args.memory,Path(bundle['memory']))
        self.assertEqual(args.skill_library,Path(bundle['skills']))

    def test_ambiguous_input_rejected(self):
        with self.assertRaisesRegex(ValueError,'conflicts'):
            resolve_knowledge_inputs(self.args(knowledge_store=Path('runs/store'),memory=Path('old.json')))

    def test_no_library_rejected_before_execution(self):
        with self.assertRaisesRegex(ValueError,'required'):resolve_knowledge_inputs(self.args())

    def test_pinned_input_does_not_resolve_active(self):
        args=self.args(knowledge_pin=Path('runs/pin.json'))
        pin={'fixed':'revision'}
        with patch('tools.run_generated_skills.read_json',return_value=(pin,b'')), \
             patch('tools.run_generated_skills.verify_pin',return_value={'memory':'runs/m.json','skills':'runs/s.json'}), \
             patch('tools.run_generated_skills.resolve_bundle',side_effect=AssertionError('ACTIVE read')):
            self.assertEqual(resolve_knowledge_inputs(args),pin)


if __name__=='__main__':unittest.main()
