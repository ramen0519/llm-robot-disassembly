"""Offline contracts for T0–T5. Provider responses and case fixtures are explicit."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from caseplanner.astra import Astra, ModelError, order
from caseplanner.geometry import prepare,rigid
from caseplanner.images import Scene
from caseplanner.io import read,sha,write,location
from caseplanner.memory import Incidents,Library as CaseLibrary,normalized,validate_guidance,verified

def Library(*args,**kwargs):
    return CaseLibrary(*args,domain='offline_contract',**kwargs)

CFG=dict(model='gpt-6-astra',endpoint='https://api.openai.com/v1/responses',max_output_tokens=1024,
    input_usd_per_million='10',output_usd_per_million='50',max_request_context_tokens=60000,image_reserve_tokens_each=4096,
    reasoning_effort='low',response_timeout_seconds=60)
def step(node,nxt,p,before,after,serial=2):
    return dict(parent_uid=node,child_uid=nxt,part=p,before=before,after=after,physics_pass=True,
                accepted=True,attempt='a'+str(serial),attempt_index=serial)
G=dict(decision='create',applies_when='Given the same observed geometric context',strategy='Test the accessible adjacent candidate before reconsidering the target.',evidence_refs=['f'],skip_reason=None)
A=step('0','1','C',['T','C','B'],['T','B'])
B=step('1','2','T',['T','B'],['B'],3)
O=dict(stage='path',outcome='path_search_failed',evidence=None,check_id=None)
class TerminalInput(unittest.TestCase):
    def test_initial_two_part_node_owns_the_native_final_removal(self):
        from types import SimpleNamespace
        from caseplanner.run import Session

        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            args=SimpleNamespace(run=root,vision=root)
            scene=SimpleNamespace(meshes={'1':None,'0':None})
            session=Session(args,scene,None,None,None,'test-verifier')
            self.assertEqual(session.terminal[('0','1')],session.root)
            self.assertEqual(session.chains[session.root],[])


class Cases(unittest.TestCase):
    def setUp(self):
        self.inc=Incidents('run-one','physics-v1')
        self.event=self.inc.failed('0','T',dict(O),['a1'])
    def test_intermediate_is_not_resolution(self):
        self.assertEqual(self.inc.accepted([A]),[])
        self.inc.leave(set())
        self.assertEqual(self.event['state'],'unresolved')
    def test_continuous_local_solution(self):
        self.assertEqual(self.inc.accepted([A,B]),[self.event])
        self.assertEqual(self.event['state'],'resolved')
    def test_no_branch_splice(self):
        self.assertFalse(verified([dict(A,child_uid='other'),B],'0','T'))
        self.assertEqual(self.inc.accepted([dict(A,child_uid='other'),B]),[])
    def test_same_remaining_different_node(self):
        self.assertEqual(self.inc.accepted([dict(A,parent_uid='other'),B]),[])
    def test_late_pass_not_accepted(self):
        self.assertEqual(self.inc.accepted([A,dict(B,accepted=False)]),[])
    def test_mismatched_sets(self):
        self.assertFalse(verified([A,dict(B,before=['T','Z'],after=['Z'])],'0','T'))
    def test_earlier_edges_excluded(self):
        self.event['failure_attempt_index']=2
        self.assertFalse(self.inc.accepted([A,B]))
    def test_repeat_keeps_first_representative(self):
        again=self.inc.failed('0','T',dict(O,outcome='timeout'),['a2'])
        self.assertIs(again,self.event)
        self.assertEqual(again['representative']['outcome'],'path_search_failed')
        self.assertEqual(len(again['failures']),2)
    def test_unknown_evidence_rejected(self):
        with self.assertRaises(ValueError):validate_guidance(dict(G,evidence_refs=['invented']),{'f':1})
    def test_skip(self):
        self.assertFalse(validate_guidance(dict(decision='skip',evidence_refs=[],applies_when=None,strategy=None,skip_reason='No support'),{'f':1}))

class Search(unittest.TestCase):
    def test_frozen_dedup_filter_provenance_topk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=root/'run';run.mkdir()
            Image.new('RGB',(8,8)).save(run/'actual-test-input.png')
            vec=normalized(np.ones(384))
            np.save(run/'v.npy',vec)
            emb=dict(png='actual-test-input.png',png_sha256=sha(run/'actual-test-input.png'),vector='v.npy',
                     vector_sha256=sha(run/'v.npy'),version='synthetic-test-only')
            store=Library(root/'library')
            self.assertEqual(store.search(O,vec,emb['version']),[])
            for i in range(5):
                inc=Incidents('fixture'+str(i),'p')
                event=inc.failed('0','T',dict(O),[])
                with self.assertRaises(ValueError):store.register(event,G,{'f':1},'family',run,emb)
                inc.accepted([A,B])
                cid=store.register(event,G,{'f':1},'family',run,emb)
                self.assertEqual(cid,store.register(event,G,{'f':1},'family',run,emb))
                fresh=Library(root/'library')
                self.assertEqual(len(fresh.search(O,vec,emb['version'])),min(3,i+1))
            self.assertFalse(store.frozen)
            self.assertEqual(len(fresh.frozen),5)
            self.assertFalse(fresh.search(dict(O,stage='stability'),vec,emb['version']))
            self.assertFalse(fresh.search(dict(O,outcome='timeout'),vec,emb['version']))
            self.assertFalse(fresh.search(O,vec,'other camera/encoder'))
            self.assertFalse(Library(root/'library',['family']).frozen)
            self.assertFalse(Library(root/'library',enabled=False).frozen)
    def test_research_refuses_even_an_explicit_test_store_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            Library(Path(tmp))
            with self.assertRaises(ValueError): CaseLibrary(Path(tmp),domain='research')

    def test_invalid_frozen_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            store = Library(root/'store')
            write(root/'bad.json',dict(store.snapshot(), index='fake'))
            with self.assertRaises(ValueError):Library(root/'store',frozen_file=root/'bad.json')

    def test_pre_reset_database_is_rejected_with_bytes_preserved(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root/'index.sqlite'
            db = sqlite3.connect(str(database))
            try:
                db.execute('CREATE TABLE library_domain (singleton INTEGER PRIMARY KEY, name TEXT)')
                db.execute("INSERT INTO library_domain VALUES (1, 'research')")
                db.execute('CREATE TABLE cases (id TEXT PRIMARY KEY, family TEXT, data TEXT)')
                db.execute("INSERT INTO cases VALUES ('past', 'old-product', '{}')")
                db.commit()
            finally:
                db.close()
            before = sha(database)
            with self.assertRaisesRegex(ValueError, 'previous case collection'):
                CaseLibrary(root)
            self.assertEqual(sha(database), before)

    def test_previous_frozen_index_is_rejected_even_when_empty(self):
        from caseplanner.io import digest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pin = root/'old.json'
            write(pin, dict(index=digest([]), case_ids=[], records=[]))
            with self.assertRaisesRegex(ValueError, 'another collection'):
                CaseLibrary(root/'new', frozen_file=pin)
            self.assertEqual(CaseLibrary(root/'new').frozen, [])

    def test_current_fixture_snapshot_cannot_be_used_for_research(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Library(root/'fixture')
            research = CaseLibrary(root/'research')
            self.assertNotEqual(fixture.index, research.index)
            write(root/'fixture.json', fixture.snapshot())
            with self.assertRaisesRegex(ValueError, 'another collection or domain'):
                CaseLibrary(root/'research', frozen_file=root/'fixture.json')

    def test_snapshot_excludes_cases_added_between_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inc=Incidents('sample','geometry-v1')
            event=inc.failed('0','T',dict(O),[])
            inc.accepted([A,B])
            store=Library(root/'store')
            write(root/'before.json',store.snapshot())
            cid=store.register(event,G,{'f':1},'family',root,None)
            self.assertEqual(Library(root/'store',frozen_file=root/'before.json').frozen,[])
            self.assertEqual(Library(root/'store').frozen[0]['id'],cid)

    def test_must_cite_failed_and_resolved_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inc=Incidents('sample','geometry-v1')
            event=inc.failed('0','T',dict(O),[])
            inc.accepted([A,B])
            with self.assertRaises(ValueError):Library(root/'store').register(event,dict(G,evidence_refs=['failure']),{'failure':O,'a2':A,'a3':B},'family',root,None)

    def test_vector_shape_and_nonfinite(self):
        for invalid in ([1,2], np.zeros(384),np.full(384,np.nan)):
            with self.assertRaises(ValueError): normalized(invalid)

class Contract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.run=Path(self.temp.name)
        (self.run/'input').mkdir()
        for n in ('top.png','oblique.png'): Image.new('RGB',(8,8)).save(self.run/'input'/n)
        self.geom=dict(assembly_id='unit',coordinate_system={},parts=[],nearby_relations=[])
        self.state=dict(state_id='a',remaining_parts=['T','C','B'],removed_parts=[],last_transition=None)
        self.bodies=[]
    def tearDown(self):self.temp.cleanup()
    def response(self,body,endpoint,timeout):
        self.bodies.append(body)
        packet=json.loads(body['input'][0]['content'][0]['text'])
        response=dict(request_id=packet['request_id'],state_id=packet['state']['state_id'],
                      ranking=packet['valid_candidates'],applied_case_ids=[])
        return dict(id='resp_'+str(len(self.bodies)),status='completed',model=CFG['model'],usage={'input_tokens':100,'output_tokens':20},
                    output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(response))])])
    def test_first_images_then_delta_and_python_overrides_history(self):
        model=Astra(self.run,self.geom,CFG,3,'1',transport=self.response)
        self.assertEqual(model.rank(self.state,['T','C'],None,[]),(['T','C'],[]))
        next_state=dict(self.state,state_id='branch_after_backtracking')
        hits=[dict(case_id='case1',applies_when='A',strategy='B',similarity=.8)]
        model.rank(next_state,['C'],{'target_part':'T','result':'path/path_search_failed','evidence':None},hits)
        a,b=self.bodies
        self.assertEqual([x['type'] for x in a['input'][0]['content']],['input_text','input_image','input_image'])
        self.assertEqual(b['previous_response_id'],'resp_1')
        self.assertEqual(len(b['input'][0]['content']),1)
        pkt=json.loads(b['input'][0]['content'][0]['text'])
        self.assertNotIn('geometry',pkt)
        self.assertEqual(pkt['state'],next_state)
        self.assertEqual(set(pkt['guidance'][0]),{'case_id','applies_when','strategy'})
    def test_malformed_rank_does_not_dispatch_physics(self):
        def wrong(*a):
            body=self.response(*a);body['output'][0]['content'][0]['text']='{}';return body
        model=Astra(self.run,self.geom,CFG,2,'1',transport=wrong)
        with self.assertRaises(ModelError): model.rank(self.state,['T','C'],None,[])
        self.assertEqual(model.calls,2)
    def test_negative_rate_is_rejected(self):
        with self.assertRaises(ValueError):Astra(self.run,self.geom,dict(CFG,input_usd_per_million='-1'),1,1)

    def test_ledger_blocks_before_dispatch(self):
        for calls,money in ((0,'10'),(2,'.00001')):
            m=Astra(self.run,self.geom,CFG,calls,money,transport=self.response)
            with self.assertRaises(ModelError):m.rank(self.state,['T'],None,[])
        self.assertFalse(self.bodies)
    def test_connection_stops_without_independent_retry(self):
        def failure(*args):raise TimeoutError('connect')
        m=Astra(self.run,self.geom,CFG,5,1,transport=failure)
        with self.assertRaises(ModelError):m.rank(self.state,['T'],None,[])
        self.assertEqual(m.calls,1)
        self.assertGreater(m.booked,0)
    def test_duplicate_missing_external_stale_and_alias(self):
        good=dict(request_id='r',state_id='s',ranking=['a','b'],applied_case_ids=[])
        for bad in [dict(ranking=['a','a']),dict(ranking=['a']),dict(ranking=['x','b']),dict(state_id='old'),dict(applied_case_ids=['x'])]:
            with self.assertRaises(ValueError):order(dict(good,**bad),['a','b'],[],'r','s')

class Geometry(unittest.TestCase):
    def test_rotated_instances_and_exact_surface_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            mesh=trimesh.creation.box(extents=[2,4,6]);mesh.export(str(folder/'shape.obj'))
            pose=[[0,-1,0,12],[1,0,0,0],[0,0,1,0],[0,0,0,1]]
            parts=[dict(part_id='pin',mesh='shape.obj',geometry_id='repeat',assembly_matrix_mm=np.eye(4).tolist(),mesh_coordinate_frame='part_local'),
                   dict(part_id='jaw',mesh='shape.obj',geometry_id='repeat',assembly_matrix_mm=pose,mesh_coordinate_frame='part_local')]
            write(folder/'manifest.json',dict(unit='mm',parts=parts))
            g=prepare(folder/'manifest.json',folder/'out','two',tolerance=.1)
            jaw=g['parts'][1]
            np.testing.assert_allclose(jaw['bbox_center_mm'],[12,0,0])
            np.testing.assert_allclose(jaw['bbox_size_mm'],[4,2,6])
            self.assertAlmostEqual(jaw['volume_mm3'],48)
            self.assertEqual(jaw['source_part_id'],'jaw')
            self.assertEqual(g['nearby_relations'],[])
            self.assertAlmostEqual(read(folder/'out/surface_distances.json')[0]['min_mesh_surface_distance_mm'],9.)
            scene=Scene(folder/'out')
            self.assertEqual(len(scene.meshes),2)
    def test_missing_obj_units_and_invalid_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):prepare(Path(tmp),Path(tmp)/'out','missing')
        with self.assertRaises(ValueError):rigid(np.zeros((4,4)))
    def test_forbidden_path_rejected_lexically(self):
        for path in ('E:/forbidden','/mnt/e/forbidden','../escape'):
            with self.assertRaises(ValueError):location(path)
