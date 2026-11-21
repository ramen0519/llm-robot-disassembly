import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from orchestrator.relational_knowledge import (
    context_from_geometry, digest, freeze_library, make_card, safe_path, validate_library)


class RelationalKnowledgeTests(unittest.TestCase):
    def geometry(self):
        return dict(parts={'a':dict(extents=[10,1,6],bounds_center=[0,0,0]),
                           'b':dict(extents=[10,3,1],bounds_center=[0,0,-3]),
                           'c':dict(extents=[1,1,1],bounds_center=[2,0,0])},
                    contact_edges=[['a','b'],['a','c']], frame='native CAD')

    def imported(self, kind='experience', record=None):
        return dict(kind=kind, source_path='runs/test/source.json', source_id='historical:1',
                    sha256='historical-file-hash', json_pointer='/records/0',
                    record=record or dict(result={'status':'UNKNOWN'}), record_id='one')

    def test_relative_centers_and_contacts_preserve_unknown_causes(self):
        context = context_from_geometry(self.geometry(), 'a')
        self.assertEqual(context['neighbors'][0]['center_delta_xyz'],[0,0,-3])
        self.assertEqual(context['topology']['component_delta'],1)
        self.assertEqual(context['topology']['remainder_isolated_nodes'],2)
        self.assertIsNone(context['neighbors'][0]['supports_target'])
        self.assertIsNone(context['neighbors'][0]['blocks_motion'])
        self.assertFalse(context['dynamic_contacts_observed'])

    def test_missing_contact_data_is_not_empty_contact_graph(self):
        geometry=self.geometry(); geometry.pop('contact_edges')
        missing=context_from_geometry(geometry,'a')
        self.assertIsNone(missing['topology']['contact_degree'])
        geometry['contact_edges']=[]
        observed=context_from_geometry(geometry,'a')
        self.assertEqual(observed['topology']['contact_degree'],0)
        self.assertEqual(observed['topology']['components_before'],3)

    def test_partial_state_filters_reference_graph_without_dynamic_claim(self):
        context=context_from_geometry(self.geometry(),'a',['a','c'])
        self.assertEqual([n['part_id'] for n in context['neighbors']],['c'])
        self.assertEqual(context['topology']['component_delta'],0)
        self.assertEqual(context['relation_scope'],'recorded_reference_geometry_restricted_to_remaining_parts')

    def test_unit_and_invalid_geometry_not_invented(self):
        geometry=self.geometry(); geometry['parts']['a']['extents']=[-1,2,3]
        context=context_from_geometry(geometry,'a')
        self.assertIsNone(context['shape']['extents_xyz'])
        self.assertIsNone(context['shape']['length_unit'])

    def test_forbidden_drive_rejected_before_any_filesystem_probe(self):
        with patch.object(Path,'lstat',side_effect=AssertionError('must not inspect')):
            with self.assertRaisesRegex(ValueError,'forbidden_drive'):
                safe_path('E:/do-not-touch/library.json')

    def test_junction_parent_is_checked_before_its_child(self):
        root=Path.cwd(); junction=root/'fictional_junction'; child=junction/'library.json'
        touched=[]
        def fake_lstat(path, *args, **kwargs):
            touched.append(path)
            if path==child:
                raise AssertionError('child must never be inspected')
            return SimpleNamespace(st_file_attributes=1024 if path==junction else 0,
                                   st_mode=0o040755)
        with patch.object(Path,'lstat',fake_lstat):
            with self.assertRaisesRegex(ValueError,'reparse'):
                safe_path(child,root)
        self.assertNotIn(child,touched)

    def test_registry_wrapper_keeps_python_and_author(self):
        original=dict(candidate=dict(program={'source':'def propose(ctx):\n    return None'},
                      generation={'returned_model':'original-author'},lifecycle='proposed'),
                      admission={'status':'rejected'},validations=[])
        card=make_card(self.imported('skill',original))
        self.assertEqual(card['content'],original)
        self.assertEqual(card['status'],'proposed')
        self.assertEqual(card['origin']['author_model'],'original-author')
        self.assertIsNotNone(card['original_program_sha256'])

    def test_author_code_and_candidate_status_are_not_promoted(self):
        source='def propose(ctx):\n    return 0'
        original=dict(source=source,status='CANDIDATE',author_model='recorded-model',
                      physics_validated=False,registered=False)
        card=make_card(self.imported('skill', original))
        self.assertEqual(card['content'],original)
        self.assertEqual(card['original_program_sha256'],hashlib.sha256(source.encode()).hexdigest())
        self.assertFalse(card['execution_authorized'])
        self.assertFalse(card['new_model_generation'])
        self.assertEqual(card['status'],'CANDIDATE')

    def test_hash_mismatch_and_unresolved_link_are_rejected(self):
        card=make_card(self.imported())
        library=freeze_library([card])
        library['cards'][0]['content']['result']['status']='PASS'
        with self.assertRaisesRegex(ValueError,'hash_mismatch'):
            validate_library(library)
        card['evidence_ids']=['missing']
        with self.assertRaisesRegex(ValueError,'dangling'):
            freeze_library([card])

    def test_library_is_not_skill_execution_authority(self):
        card=make_card(self.imported())
        card['execution_authorized']=True
        with self.assertRaisesRegex(ValueError,'cannot_authorize'):
            freeze_library([card])


if __name__ == '__main__':
    unittest.main()
