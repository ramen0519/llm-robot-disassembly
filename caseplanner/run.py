"""Wire the current candidate generator to the official serial DFS and both verifiers."""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .astra import Astra,Fixture,ModelError
from .io import append,digest,read,sha,write
from .memory import Encoder,Incidents,Library,PAIRS,validate_guidance
from .observe import Limit,Observer


def jsonable(obj):
    return json.loads(json.dumps(obj,default=lambda value:value.tolist(),allow_nan=False))


class Session:
    def __init__(self,args,scene,observer,model,library,physics_id):
        self.args,self.scene,self.obs,self.model,self.library=args,scene,observer,model,library
        self.run=args.run
        import uuid
        self.id=args.run.name+'-'+uuid.uuid4().hex[:16]
        self.encoder=Encoder(args.vision)
        self.settings=physics_id
        self.inc=Incidents(self.id,physics_id)
        self.attempts=[]
        self.active=[]
        self.chains={}
        self.best=[]
        self.failure=None
        self.pending=[]
        self.last_feedback={}
        self.embeddings={}
        self.lookups=[]
        self.registrations=[]
        self.selections=[]
        self.tree=None
        self.root=digest([self.id,sorted(scene.meshes),physics_id])
        self.chains[self.root]=[]
        self.active=[(tuple(sorted(scene.meshes)),self.root)]
        self.selected=self.root
        initial_parts=tuple(sorted(scene.meshes))
        self.terminal={initial_parts:self.root} if len(initial_parts)==2 else {}

    def event(self,kind,**data):
        append(self.run/'events.jsonl',dict(kind=kind,run_id=self.id,**jsonable(data)))

    def close_target(self):
        if not self.pending: return
        pending,self.pending=self.pending,[]
        if self.failure or time.monotonic() >= self.obs.deadline:
            self.event('subsearch_interrupted',attempts=[a['id'] for a in pending])
            return
        last=pending[-1]
        # Greedy support search can itself expire between two physics calls.
        # Only the last failed detector may describe its own snapshot.
        recs=[self.obs.checks[int(i.split('_')[1])-1] for i in last['checks']]
        stage='stability' if last['raw']['action'] is not None else 'path'
        rec=recs[-1] if recs and recs[-1]['stage']==stage and recs[-1].get('passed') is False else None
        obs=dict(stage=stage,outcome=rec['outcome'] if rec else 'unknown',target_part=last['part'],
                 evidence=rec.get('evidence') if rec else None,check_id=rec['id'] if rec else None)
        e=self.inc.failed(last['parent_uid'],last['part'],obs,[a['id'] for a in pending])
        e.setdefault('failure_attempt_index',int(last['id'].split('_')[1]))
        e.setdefault('failed_check_refs',[]).extend('checks/'+k+'.json' for a in pending for k in a['checks'])
        hits=[]
        if e['representative'] is obs and rec and rec.get('endpoint') and (stage,obs['outcome']) in PAIRS:
            try:
                poses=rec['endpoint']
                p=last['part']
                prior=next((s for s in reversed(recs) if s['stage']=='path' and s.get('first')),None)
                anchor=None
                if p not in poses and prior:
                    xyz=self.scene.world({p:prior['first'][p]})[p]
                    anchor=((xyz.min(axis=0)+xyz.max(axis=0))/2).tolist()
                # Derive CURRENT geometric neighbours using the recorded transforms and the same
                # triangle-surface test as the input, incl. the pre-removal pose only for framing.
                related=set((obs['evidence'] or {}).get('returned_part_ids',[]))
                related.update(self.neighbours(p,poses,prior))
                png='checks/'+rec['id']+'_target.png'
                self.scene.render(poses,self.run/png,target=p,related=related,anchor=anchor)
                obs.update(png=png,png_sha256=sha(self.run/png),present=sorted(poses),target_anchor_cm=anchor,
                           related=sorted(related),representative_attempt=last['id'])
                vec=self.encoder.encode(self.run/png)
                vector='checks/'+rec['id']+'_embedding.npy'
                np.save(self.run/vector,vec,allow_pickle=False)
                embedding=dict(png=png,png_sha256=sha(self.run/png),vector=vector,vector_sha256=sha(self.run/vector),
                               version=self.encoder.version,encoder=self.encoder.meta)
                self.embeddings[e['id']]=embedding
                hits=self.library.search(obs,vec,self.encoder.version)
            except Exception as error:
                self.event('media_or_lookup_error',incident=e['id'],error=str(error))
        elif e['id'] in self.embeddings:
            emb=self.embeddings[e['id']]
            hits=self.library.search(e['representative'],np.load(self.run/emb['vector'],allow_pickle=False),emb['version'])
        self.lookups.append(dict(incident=e['id'],index=self.library.index,hit_ids=[h['case_id'] for h in hits],hits=hits))
        self.last_feedback[last['parent_uid']]=(obs,hits)
        self.event('target_exhausted',incident=e['id'],representative=e['representative'],attempt_ids=e['failures'][-1],lookup=self.lookups[-1])

    def neighbours(self,p,poses,prior):
        import fcl
        from .geometry import collision_mesh
        subset=dict(poses)
        if p not in subset:
            if not prior: return []
            subset[p]=prior['first'][p]
        world=self.scene.world(subset)
        objs={k:collision_mesh(v,self.scene.meshes[k][1]) for k,v in world.items()}
        found=[]
        for k in objs:
            if k==p: continue
            dist=fcl.distance(objs[p],objs[k],fcl.DistanceRequest(),fcl.DistanceResult())
            if 0<=dist<=self.args.proximity_mm/10 or fcl.collide(objs[p],objs[k],fcl.CollisionRequest(),fcl.CollisionResult()):
                found.append(k)
        return found

    def select(self,planner,tree,parts):
        self.tree=tree
        path=[tuple(x) for x in planner.G_path]
        while len(self.active)>len(path): self.active.pop()
        if [n for n,uid in self.active]!=path:
            raise RuntimeError('native DFS lineage mismatch')
        self.selected=self.active[-1][1]
        self.inc.leave({uid for _,uid in self.active})
        if tuple(parts)!=self.active[-1][0]: raise RuntimeError('selected node mismatch')

    def rank(self,parts):
        try: self.obs.check()
        except Limit as exc:
            self.failure=str(exc)
            raise
        valid=sorted(p for p in parts if p!=self.args.base and not self.tree.has_edge(tuple(parts),tuple(q for q in parts if q!=p)))
        if not valid: raise RuntimeError('exhausted DFS node')
        chain=self.chains[self.selected]
        state=dict(state_id=self.selected,remaining_parts=parts,removed_parts=[s['part'] for s in chain],
                   last_transition={'part_id':chain[-1]['part'],'result':'PASS'} if chain else None)
        feedback,hits=self.last_feedback.get(self.selected,(None,[]))
        fb=None if feedback is None else dict(target_part=feedback['target_part'],result=feedback['stage']+'/'+feedback['outcome'],evidence=feedback['evidence'])
        try:
            ranking,used=self.model.rank(state,valid,fb,hits)
        except Exception as error:
            self.failure='model_error: '+str(error)
            raise
        row=dict(state=state,eligible=valid,ranking=ranking,feedback=fb,offered=[h['case_id'] for h in hits],applied=used,
                 model_response=getattr(self.model,'last_response',None),index=self.library.index,source=self.model.provider)
        self.selections.append(row)
        self.event('rank',**row)
        return ranking

    def simulate(self,original,planner,p,rest,removed,pose,**opts):
        self.obs.check()
        if len(self.attempts)>=self.args.max_attempts: raise Limit('composite_limit')
        before=sorted(rest+[p])
        parent=self.selected if len(before)>2 else self.terminal[tuple(before)]
        if self.pending and (self.pending[-1]['parent_uid'],self.pending[-1]['part'])!=(parent,p):
            self.close_target()
        if len(before)>2 and self.chains[parent] and self.chains[parent][-1]['after']!=before:
            raise RuntimeError('attempt lineage mismatch')
        a=dict(id='attempt_%04d'%(len(self.attempts)+1),part=p,parent_uid=parent,before=before,after=list(rest),
               pose=None if pose is None else np.asarray(pose).tolist(),checks=[],accepted=False,
               chooser='ASAP terminal mass/base' if len(before)==2 else self.model.provider)
        self.attempts.append(a)
        self.obs.active=a
        self.event('attempt_start',attempt=a['id'],part=p,parent=parent)
        try:
            result=original(planner,p,rest,removed,pose,**opts)
            a['raw']=jsonable(result)
            self.obs.check()  # A late PASS is never submitted to the native tree.
            if not result['feasible']: self.pending.append(a)
            else: self.pending=[]
            return result
        except Exception as exc:
            reason=str(exc) if isinstance(exc,Limit) else 'native_exception: '+type(exc).__name__+': '+str(exc)
            self.failure=self.failure or reason
            raise
        finally:
            self.event('attempt_return',attempt=a)

    def accept(self,parent,child,info):
        a=self.obs.active
        if not info['feasible']: return
        if a['accepted'] or a['before']!=list(parent) or a['after']!=list(child) or a['part']!=info['part_move']:
            raise RuntimeError('transition did not match actual native update')
        parent_uid=a['parent_uid']
        uid=digest([parent_uid,a['id'],list(child)])
        checkrows=[self.obs.checks[int(k.split('_')[1])-1] for k in a['checks']]
        chosen=[r for r in checkrows if r['stage']=='path' and r['passed'] and r['direction']==a['raw']['action']]
        support=[r['id'] for r in checkrows if r['stage']=='stability' and r['passed']]
        if not chosen or not support: raise RuntimeError('missing actual path/stability evidence')
        a['accepted']=True
        step=dict(parent_uid=parent_uid,child_uid=uid,part=a['part'],before=list(parent),after=list(child),accepted=True,
            physics_pass=True,attempt=a['id'],chosen_by=a['chooser'],pose=a['pose'],direction=a['raw']['action'],attempt_index=len(self.attempts),
            fixed_base=self.args.base,additional_support=a['raw']['parts_fix'],path_check=chosen[0]['id'],
            path=chosen[0]['trajectory'],stability_checks=support,check_evidence=['checks/'+s+'.json' for s in a['checks']])
        chain=self.chains[parent_uid]+[step]
        self.chains[uid]=chain
        if len(chain)>len(self.best): self.best=chain
        if len(child)>2: self.active.append((tuple(child),uid))
        if len(child)==2: self.terminal[tuple(child)]=uid
        self.event('accepted',transition=step)
        for event in self.inc.accepted(chain): self.learn(event)

    def learn(self,e):
        geom={p['part_id']:p for p in self.scene.geom['parts']}
        failure=e['representative']
        facts={'failure':dict(stage=failure['stage'],outcome=failure['outcome'],evidence=failure['evidence'],
               check=failure['check_id'],target_geometry=geom[e['target']],conditions=dict(verifier_sha=self.settings,
               max_seconds=self.args.seconds,max_grippers=self.args.max_grippers,max_poses=self.args.max_poses,
               fixed_base=self.args.base,path='single force direction per trial'),
               near=[p for p in self.scene.geom['nearby_relations'] if e['target'] in p['parts']])}
        for step in e['chain']:
            facts[step['attempt']]=dict(part=step['part'],geometry=geom[step['part']],path=step['path_check'],
                                        stability=step['stability_checks'],accepted=True,before=step['before'],after=step['after'])
        e['guidance_provider']=self.model.provider
        outcome=dict(incident=e['id'],state='resolved',case_id=None,provider=self.model.provider)
        try:
            reply=self.model.guidance(e,facts)
            outcome['guidance']=reply
            if validate_guidance(reply,facts) and not self.args.evaluation and self.args.mode=='astra':
                outcome['case_id']=self.library.register(e,reply,facts,self.args.family,self.run,self.embeddings.get(e['id']))
        except Exception as exc:
            outcome['guidance_error']=str(exc)
        self.registrations.append(outcome)
        self.event('local_resolution',facts=facts,**outcome)


def execute(args,scene,library=None,test_transport=None):
    sys.path.insert(0,str(args.asap_root))
    from plan_sequence import physics_planner as ph
    from plan_sequence.planner.dfs import DFSSequencePlanner
    from plan_sequence.planner.base import SequencePlanner
    from plan_sequence.generator.base import Generator
    import redmax_py
    from .deliver import finalize
    physics_files=['plan_sequence/physics_planner.py','plan_sequence/feasibility_check.py',
        'plan_sequence/planner/dfs.py','plan_sequence/planner/base.py','plan_sequence/sim_string.py']
    commit=subprocess.check_output(['git','-C',str(args.asap_root),'rev-parse','HEAD'],text=True).strip()
    if commit!='3727e716b57e0b508c4e9ecd9fefa6953e8f72da': raise ValueError('unreviewed ASAP revision')
    subprocess.run(['git','-C',str(args.asap_root),'diff','--exit-code','HEAD','--']+physics_files,check=True)
    settings=dict(asap=commit,redmax_sha256=sha(redmax_py.__file__),source={p:sha(args.asap_root/p) for p in physics_files},
        max_grippers=args.max_grippers,max_poses=args.max_poses,base=args.base,pose_reuse=0,
        path='six separate constant-force candidates',support='original serial greedy',native_unit='cm',pose_solver='native',
        seed=args.seed,composite_limit=args.max_attempts,path_limit=args.max_path,stability_limit=args.max_stability,
        elapsed_limit=args.seconds,python=platform.python_version())
    write(args.run/'physics.json',settings)
    if library is None: library=Library(args.store,set(args.exclude_family)|({args.family} if args.evaluation else set()),not args.memory_off)
    model=(Astra(args.run,scene.geom,read(args.config),args.max_calls,args.max_usd,args.rank_attempts,**({'transport':test_transport} if test_transport is not None else {}))
           if args.mode=='astra' else Fixture(args.fixture_order))
    obs=Observer(args.run,args,ph)
    session=Session(args,scene,obs,model,library,digest(settings))
    original=SequencePlanner._simulate
    original_update=SequencePlanner._update_tree
    class Ranked(Generator):
        def generate_candidate_part(self,assembled_parts): return session.rank(assembled_parts)
    class ObservedDFS(DFSSequencePlanner):
        def _select_node(self,tree):
            if time.monotonic()>=obs.deadline:
                session.failure='wall_time'
                raise Limit('wall_time')
            session.close_target()
            # Native DFS counts fixed base as a possible edge; it never is. Pop exhausted
            # non-root frames with no eligible child before delegating normal DFS selection.
            while len(self.G_path)>1:
                parts=self.G_path[-1]
                if any(p!=args.base and not tree.has_edge(tuple(parts),tuple(x for x in parts if x!=p)) for p in parts): break
                self.G_path.pop()
            selected=super()._select_node(tree)
            session.select(self,tree,selected)
            return selected
        def _simulate(self,part_move,parts_rest,parts_removed,pose,**kw):
            try: return session.simulate(original,self,part_move,parts_rest,parts_removed,pose,**kw)
            except Exception as exc:
                reason=str(exc) if isinstance(exc,Limit) else 'native_exception: '+type(exc).__name__+': '+str(exc)
                session.failure=session.failure or reason
                raise
        def _expand_leaf(self,*par,**kw):
            session.close_target()
            return super()._expand_leaf(*par,**kw)
    def update(planner,tree,parent,child,n_eval,sim_info):
        obs.check()
        original_update(planner,tree,parent,child,n_eval,sim_info)
        session.accept(parent,child,sim_info)
    # DrvFS flushes small native SDF writes very slowly. Stage exactly the verified input
    # bytes in Linux volatile memory for RedMax. Persistent artifacts always stay in the run.
    import tempfile
    import shutil
    scratch=tempfile.TemporaryDirectory(prefix='asap_cases_',dir='/dev/shm')
    material=Path(scratch.name)
    copied={}
    for path in sorted((args.run/'input/native').iterdir()):
        if path.suffix.lower() not in ('.obj','.json'): continue
        shutil.copyfile(path,material/path.name)
        copied[path.name]=sha(material/path.name)
        if copied[path.name]!=sha(path): raise ValueError('native input copy mismatch')
    write(args.run/'native_input.json',dict(sha256=copied,cache='run-private Linux RAM SDF, released after planning'))
    gen=Ranked(str(args.asap_root/'assets'),str(material),base_part=args.base,save_sdf=True)
    started=time.monotonic()
    try:
        planner=ObservedDFS(gen,num_proc=1,save_sdf=True)
        planner.seed(args.seed)
        obs.deadline=started+args.seconds
        model.deadline=obs.deadline
        if time.monotonic()>=obs.deadline: raise Limit('initialization_time_limit')
        with obs,patch.object(SequencePlanner,'_update_tree',update):
            tree=planner.plan(args.max_attempts,args.max_grippers,max_poses=args.max_poses,pose_reuse=0,
                early_term=True,timeout=max(.001,obs.deadline-time.monotonic()),debug=0)
        session.close_target()
        stop=session.failure or planner.stop_msg
        if planner.stop_msg=='exception' and not session.failure: stop='native_exception'
    except Exception as exc:
        stop=session.failure or type(exc).__name__+': '+str(exc)
    scratch.cleanup()
    elapsed=time.monotonic()-started
    session.inc.leave(set())
    return finalize(session,stop,elapsed)
