"""Serial process-local observers. Never step/reset/replay a simulator for evidence."""
from contextlib import ExitStack
import sys
import time
from unittest.mock import patch
import numpy as np
from .io import append,write


class Limit(RuntimeError): pass


def transforms(sim,ids):
    return {p:np.asarray(sim.get_body_E0j('part'+p)).copy().tolist() for p in ids}


def returned_locals(method,instance,args,kwargs):
    # Capture only the original checker return frame (including the time it measured).
    # No inference from post-return wall time. All wheel/source versions are pinned in run metadata.
    observed={}
    old=sys.gettrace()
    def returned(frame,event,arg):
        if event=='return':
            for key in ('t_plan','parts_disconnected','parts_fall','i'):
                if key in frame.f_locals: observed[key]=frame.f_locals[key]
        return returned
    def calls(frame,event,arg):
        if frame.f_code is method.__code__:
            frame.f_trace_lines=False
            return returned
        return None
    try:
        sys.settrace(calls)
        result=method(instance,*args,**kwargs)
    finally:
        sys.settrace(old)
    return result,observed


class Observer:
    def __init__(self,run,args,physics):
        self.run,self.args,self.physics=run,args,physics
        self.count={'path':0,'stability':0}
        self.active=None
        self.checks=[]
        self.recording=None
        self.deadline=float('inf')
        self.stack=ExitStack()

    def check(self):
        if time.monotonic()>=self.deadline: raise Limit('wall_time')

    def __enter__(self):
        ph=self.physics
        original_get=ph.MultiPartPathPlanner.get_state
        original_update=ph.StabilityChecker.update_status
        def path_sample(planner):
            result=original_get(planner)
            rec=self.recording
            if rec is not None and rec['stage']=='path':
                k=rec['samples']
                rec['samples']+=1
                if k==0:
                    rec['first']=transforms(planner.sim,[planner.part_move]+planner.parts_fix)
                if k%16==0:
                    rec['frames'].append(dict(index=k,poses=transforms(planner.sim,[planner.part_move])))
            return result
        def stable_sample(checker):
            result=original_update(checker)
            rec=self.recording
            if rec is not None and rec['stage']=='stability':
                rec['samples']+=1
                if rec['samples']==1 or rec['samples']%40==0:
                    rec['frames'].append(dict(index=rec['samples'],poses=transforms(checker.sim,checker.parts_move)))
            return result
        def wrap(original,stage):
            def checked(planner,*args,**kwargs):
                self.check()
                cap=self.args.max_path if stage=='path' else self.args.max_stability
                if self.count[stage]>=cap: raise Limit(stage+'_limit')
                self.count[stage]+=1
                cid='check_%05d'%(1+len(self.checks))
                row=dict(id=cid,attempt=self.active['id'],target=self.active['part'],stage=stage,
                         frames=[],samples=0,first=None)
                self.active['checks'].append(cid)
                self.checks.append(row)
                self.recording=row
                start=time.monotonic()
                ids=([planner.part_move]+planner.parts_fix if stage=='path' else list(planner.parts_fix)+list(planner.parts_move))
                try:
                    result,local=returned_locals(original,planner,args,kwargs)
                    passed=bool(result[0])
                    # Snapshot detached matrices while this exact simulator still holds the return state.
                    end=transforms(planner.sim,ids)
                    evidence=None
                    if stage=='path':
                        direction=np.asarray(args[0] if args else kwargs['action']).tolist()
                        row.update(direction=direction,trajectory='checks/'+cid+'.npz')
                        (self.run/'checks').mkdir(exist_ok=True)
                        np.savez_compressed(self.run/row['trajectory'],joint_q=np.asarray(result[1]))
                        outcome='pass' if passed else ('timeout' if local.get('t_plan',0)>planner.max_time else 'path_search_failed')
                    else:
                        row.update(fixed=list(planner.parts_fix),free=list(planner.parts_move))
                        if passed: outcome='pass'
                        elif local.get('parts_disconnected'): outcome='initial_disconnected'
                        elif result[1] is None: outcome='timeout'
                        else: outcome='stability_failed'
                        if result[1] is not None: evidence={'returned_part_ids':list(result[1])}
                    row.update(passed=passed,outcome=outcome,evidence=evidence,endpoint=end,
                               snapshot_source='RedMax state at original checker return before next reset',
                               elapsed_seconds=time.monotonic()-start)
                    if stage=='stability' and not row['samples']:
                        row['dynamic_frames']=False
                    write(self.run/('checks/'+cid+'.json'),row)
                    return result
                except Exception as e:
                    row.update(outcome='execution_error',passed=None,error_type=type(e).__name__)
                    record=self.run/('checks/'+cid+'.json')
                    if not record.exists():
                        write(record,row)
                    append(self.run/'observer_errors.jsonl',dict(check=cid,stage='runtime',outcome='execution_error',error_type=type(e).__name__))
                    raise
                finally:
                    self.recording=None
            return checked
        for cls,method,new in [(ph.MultiPartPathPlanner,'get_state',path_sample),
            (ph.StabilityChecker,'update_status',stable_sample),
            (ph.MultiPartPathPlanner,'check_success',wrap(ph.MultiPartPathPlanner.check_success,'path')),
            (ph.MultiPartStabilityPlanner,'check_success',wrap(ph.MultiPartStabilityPlanner.check_success,'stability'))]:
            self.stack.enter_context(patch.object(cls,method,new))
        return self

    def __exit__(self,*exc):
        return self.stack.__exit__(*exc)
