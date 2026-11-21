import tempfile
import unittest
from pathlib import Path
from tools.migrate_relational_knowledge import Migrator, resolve_versions


class RelationalMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='relational-migration-')
        self.addCleanup(self.temp.cleanup)
        self.m= Migrator(Path(self.temp.name))

    def project(self, row):
        return self.m.experience_context(dict(record=row,source_id='one',namespace='test'),{})[0]

    def geometry(self):
        return {p:dict(extents=[1,2,3],bounds_center=[0,0,0]) for p in ['0','1','2']}

    def test_legacy_removed_part_is_not_restored_to_query(self):
        context=self.project(dict(context={'geometry_features':self.geometry(),'initial_parts':['0','1','2']},
             request={'part_move':'0','parts_rest':['2'],'parts_removed':['1']}))
        self.assertEqual(context['remaining_parts'],['0','2'])
        self.assertEqual(context['target_part_id'],'0')

    def test_constraint_scope_preserves_target_and_state(self):
        context=self.project(dict(scope={'part_move':'1','assembled_parts':['1','2']},
                                  context={'geometry_features':self.geometry()}))
        self.assertEqual(context['target_part_id'],'1')
        self.assertEqual(context['remaining_parts'],['1','2'])

    def test_native_acceptance_and_deadline_are_not_dropped(self):
        context=self.project(dict(source_task_context={'geometry_features':self.geometry(),
            'task':{'assembly_id':'example'}},attempt={'part_move':'0','remaining_parts':['0','2'],
            'status':'PASS','verifier_domain':'ASAP_RedMax','accepted_for_transition':False,'deadline_exceeded':True}))
        self.assertFalse(context['observations']['accepted_for_transition'])
        self.assertTrue(context['observations']['deadline_exceeded'])
        self.assertEqual(context['conditions']['verifier_domain'],'ASAP_RedMax')

    def test_conflicting_versions_are_not_arbitrarily_joined(self):
        rows=[dict(origin={'source_path':'a'},content={'status':'PASS'}),
              dict(origin={'source_path':'b'},content={'status':'UNKNOWN'})]
        self.assertEqual(resolve_versions(rows,'c'),[])
        self.assertEqual(resolve_versions(rows,'a'),[rows[0]])

    def test_equivalent_pagination_versions_keep_all_origins(self):
        rows=[dict(origin={'source_path':'a'},content={'statement':'same','first_page':1}),
              dict(origin={'source_path':'b'},content={'statement':'same','occurrences':[]})]
        self.assertEqual(resolve_versions(rows,'c'),rows)


if __name__=='__main__':
    unittest.main()
