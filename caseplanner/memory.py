"""Incident lineage, version-pinned image retrieval and run-start library snapshots."""
import json
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
import numpy as np
from PIL import Image
from . import CASE_COLLECTION
from .images import CAMERA
from .io import digest, location, read, sha, write

PAIRS={('path','path_search_failed'),('path','timeout'),('stability','stability_failed'),
       ('stability','initial_disconnected'),('stability','timeout')}
MODEL_SHA='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'


class Encoder:
    def __init__(self, folder):
        self.folder=location(folder)
        self.model=None

    def load(self):
        if self.model is not None: return
        sys.path.insert(0,str(self.folder.parent/'vf'))
        import torch
        import transformers
        from transformers import Dinov2Model
        import transformers.models.dinov2.modeling_dinov2 as implementation
        weight=sha(self.folder/'model.safetensors')
        if weight!=MODEL_SHA: raise ValueError('unexpected DINOv2 weights')
        self.model=Dinov2Model.from_pretrained(str(self.folder),local_files_only=True,use_safetensors=True).eval()
        self.processor=read(self.folder/'preprocessor_config.json')
        self.meta=dict(model='facebook/dinov2-small',revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
            weights=weight,code_sha256=sha(implementation.__file__),torch=torch.__version__,transformers=transformers.__version__,
            processor_sha256=sha(self.folder/'preprocessor_config.json'),config_sha256=sha(self.folder/'config.json'),
            resize='square RGB bicubic 224 full image',mean=self.processor['image_mean'],
            std=self.processor['image_std'],features='CLS last_hidden_state L2',camera=CAMERA)
        self.version=digest(self.meta)

    def encode(self, png):
        self.load()
        import torch
        with Image.open(location(png)) as im:
            if im.format!='PNG' or im.width!=im.height: raise ValueError('expected square PNG')
            rgb=np.asarray(im.convert('RGB').resize((224,224),Image.BICUBIC),dtype=np.float32)/255.
        rgb=(rgb-np.array(self.meta['mean'],dtype=np.float32))/np.array(self.meta['std'],dtype=np.float32)
        tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0)
        with torch.no_grad():
            vector=self.model(pixel_values=tensor).last_hidden_state[0,0].cpu().numpy()
        return normalized(vector)


def normalized(vector):
    vector=np.asarray(vector,dtype=np.float32)
    if vector.shape!=(384,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-8:
        raise ValueError('invalid DINO vector')
    return vector/np.linalg.norm(vector)


def validate_guidance(reply,facts):
    fields={'decision','applies_when','strategy','evidence_refs','skip_reason'}
    if not isinstance(reply,dict) or set(reply)!=fields: raise ValueError('guidance fields')
    refs=reply['evidence_refs']
    if not isinstance(refs,list) or not all(isinstance(r,str) and r in facts for r in refs):
        raise ValueError('invalid evidence reference')
    if reply['decision']=='skip':
        if reply['applies_when'] is not None or reply['strategy'] is not None or not isinstance(reply['skip_reason'],str) or not reply['skip_reason'].strip():
            raise ValueError('invalid skip')
        return False
    refs=reply['evidence_refs']
    if (reply['decision']!='create' or reply['skip_reason'] is not None or
        not all(isinstance(reply[k],str) and 0<len(reply[k].strip())<=700 for k in ('applies_when','strategy')) or
        not isinstance(refs,list) or not refs or not all(isinstance(r,str) and r in facts for r in refs)):
        raise ValueError('ungrounded guidance contract')
    return True


class Incidents:
    def __init__(self,run_id,settings):
        self.run_id=run_id
        self.settings=settings
        self.events={}

    def failed(self,node,target,obs,attempts):
        key=digest([self.run_id,node,target,self.settings])
        event=self.events.setdefault(key,dict(id=key,run_id=self.run_id,node=node,target=target,settings=self.settings,
                    state='open',representative=obs,failures=[],chain=[]))
        event['failures'].append(list(attempts))
        return event

    def leave(self,active_nodes):
        for event in self.events.values():
            if event['state']=='open' and event['node'] not in active_nodes:
                event['state']='unresolved'

    def accepted(self,chain):
        if not chain: return []
        solved=[]
        for e in self.events.values():
            if e['state']!='open' or chain[-1]['part']!=e['target']: continue
            for i,step in enumerate(chain):
                if step['parent_uid']!=e['node']: continue
                suffix=chain[i:]
                boundary=e.get('failure_attempt_index',0)
                if any(st.get('attempt_index',0)<=boundary for st in suffix) and boundary: break
                if not verified(suffix,e['node'],e['target']): break
                e.update(state='resolved',chain=list(suffix))
                solved.append(e)
                break
        return solved


def verified(chain,start,target):
    if not chain or chain[-1]['part']!=target: return False
    node=start
    remaining=None
    for step in chain:
        before,after=set(step['before']),set(step['after'])
        if (step['parent_uid']!=node or not step['accepted'] or not step['physics_pass'] or
            step['part'] not in before or before-{step['part']}!=after or (remaining is not None and before!=remaining)):
            return False
        node,remaining=step['child_uid'],after
    return True


class Library:
    def __init__(self,root,excluded=(),enabled=True,frozen_file=None,domain='research'):
        if domain not in ('research','offline_contract'): raise ValueError('invalid case domain')
        self.domain=domain
        self.root=location(root)
        self.root.mkdir(parents=True,exist_ok=True)
        database = location(self.root / 'index.sqlite')
        with closing(sqlite3.connect(str(database), timeout=30)) as db, db:
            # The transaction also serializes two first runs creating the same empty store.
            db.execute('BEGIN IMMEDIATE')
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone():
                db.execute('CREATE TABLE cases (id TEXT PRIMARY KEY, family TEXT NOT NULL, data TEXT NOT NULL)')
                db.execute('CREATE TABLE library_domain (singleton INTEGER PRIMARY KEY CHECK(singleton=1), '
                           'name TEXT NOT NULL, collection TEXT NOT NULL)')
                db.execute('INSERT INTO library_domain VALUES (1, ?, ?)', (domain, CASE_COLLECTION))
            columns = {row[1] for row in db.execute('PRAGMA table_info(library_domain)')}
            if 'collection' not in columns:
                raise ValueError('previous case collection: use the new default or a fresh --store directory')
            identity = db.execute('SELECT name, collection FROM library_domain WHERE singleton=1').fetchone()
            if identity is None or identity[1] != CASE_COLLECTION:
                raise ValueError('case collection does not match this research run')
            if identity[0] != domain:
                raise ValueError('test and research memories cannot mix')
            rows = db.execute('SELECT family, data FROM cases ORDER BY id').fetchall() if enabled else []
        self.frozen=[json.loads(data) for family,data in rows if family not in excluded]
        if frozen_file is not None:
            snapshot=read(frozen_file)
            if snapshot.get('collection') != CASE_COLLECTION or snapshot.get('domain') != domain:
                raise ValueError('pinned case index belongs to another collection or domain')
            records=snapshot['records']
            pairs=[[c['id'],c['bundle_sha256']] for c in records]
            ids = [cid for cid, _ in pairs]
            if (self._index_digest(pairs) != snapshot['index'] or snapshot.get('case_ids') != ids
                    or len(set(ids)) != len(ids)):
                raise ValueError('invalid pinned case index')
            if enabled:
                available={c['id']:c for c in self.frozen}
                if any(cid not in available or available[cid]['bundle_sha256']!=h for cid,h in pairs):
                    raise ValueError('pinned case missing/excluded/changed')
                self.frozen=[available[cid] for cid,h in pairs]
        self.index = self._index_digest([[c['id'], c['bundle_sha256']] for c in self.frozen])

    def _index_digest(self, pairs):
        return digest({'collection': CASE_COLLECTION, 'domain': self.domain, 'records': pairs})

    def snapshot(self):
        records = [dict(id=c['id'], source_run=c['run_id'], family=c['family'], bundle_sha256=c['bundle_sha256'])
                   for c in self.frozen]
        return dict(collection=CASE_COLLECTION, domain=self.domain, index=self.index,
                    case_ids=[c['id'] for c in records], records=records)

    def register(self,event,guide,facts,family,run,embedding):
        if event['state']!='resolved' or not verified(event['chain'],event['node'],event['target']):
            raise ValueError('unverified origin-to-target chain')
        if self.domain=='research' and event.get('guidance_provider')!='openai_responses':
            raise ValueError('research case requires real guidance provenance')
        if not validate_guidance(guide,facts): return None
        if 'failure' in facts:
            required={'failure',event['chain'][-1]['attempt']}
            if not required.issubset(guide['evidence_refs']): raise ValueError('must cite origin and target acceptance')
            if len(event['chain'])>1 and not any(s['attempt'] in guide['evidence_refs'] for s in event['chain'][:-1]):
                raise ValueError('no evidence for intervening removal')

        cid=event['id']
        obs=event['representative']
        bundle=dict(event=event,guidance=guide,facts=facts,source_run=event['run_id'],source_dir=str(run),
                    family=family,embedding=embedding,domain=self.domain,collection=CASE_COLLECTION)
        refs=set(event.get('failed_check_refs',[]))
        for step in event['chain']:
            refs.update(step.get('check_evidence',[]))
            if step.get('path'): refs.add(step['path'])
        for key in ('physics.json','settings.json','input/geometry.json','input/part_mapping.json'):
            if (run/key).exists(): refs.add(key)
        bundle['raw_artifacts']={p:sha(run/p) for p in sorted(refs)}
        manifest=dict(id=cid,collection=CASE_COLLECTION,family=family,run_id=event['run_id'],stage=obs['stage'],outcome=obs['outcome'],
                      guidance=guide,bundle_sha256=digest(bundle),embedding=embedding)
        with sqlite3.connect(str(self.root/'index.sqlite'),timeout=30) as db:
            db.execute('BEGIN IMMEDIATE')
            exists=db.execute('SELECT data FROM cases WHERE id=?',(cid,)).fetchone()
            if exists:
                if json.loads(exists[0])['bundle_sha256']!=manifest['bundle_sha256']:
                    raise ValueError('conflicting registration of same event')
                return cid
            # Run records and hashes remain permanent. Keep a self-contained query image and vector as well.
            dest=self.root/cid
            dest.mkdir(exist_ok=True)
            for relative,expected in bundle['raw_artifacts'].items():
                target=location(dest/'evidence'/relative)
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists(): shutil.copyfile(location(run/relative),target)
                if sha(target)!=expected: raise ValueError('archived raw evidence mismatch')
            if embedding:
                for item in ('png','vector'):
                    src=location(run/embedding[item])
                    if sha(src)!=embedding[item+'_sha256']: raise ValueError('source bytes changed')
                    target=dest/Path(src).name
                    if not target.exists(): shutil.copyfile(src,target)
                    if sha(target)!=embedding[item+'_sha256']: raise ValueError('archive bytes changed')
            if (dest/'case.json').exists():
                if digest(read(dest/'case.json'))!=manifest['bundle_sha256']: raise ValueError('case archive conflict')
            else: write(dest/'case.json',bundle)
            db.execute('INSERT INTO cases VALUES (?,?,?)',(cid,family,json.dumps(manifest,allow_nan=False)))
        return cid

    def search(self,observation,vector,version):
        stage,outcome=observation['stage'],observation['outcome']
        if vector is None or (stage,outcome) not in PAIRS: return []
        vector=normalized(vector)
        hits=[]
        for c in self.frozen:
            emb=c['embedding']
            if (c['stage'],c['outcome'])!=(stage,outcome) or not emb or emb['version']!=version: continue
            target=self.root/c['id']/Path(emb['vector']).name
            if sha(target)!=emb['vector_sha256'] or digest(read(self.root/c['id']/'case.json'))!=c['bundle_sha256']:
                raise ValueError('frozen case integrity error')
            score=float(np.clip(vector@normalized(np.load(target,allow_pickle=False)), -1.0, 1.0))
            hits.append(dict(case_id=c['id'],source_run=c['run_id'],similarity=score,
                             applies_when=c['guidance']['applies_when'],strategy=c['guidance']['strategy']))
        return sorted(hits,key=lambda hit:(-hit['similarity'],hit['case_id']))[:3]
