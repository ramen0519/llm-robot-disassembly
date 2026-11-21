"""Opt-in counters around original checkers, with no cached physics verdicts."""
from contextlib import contextmanager
from functools import wraps
import time


class EvaluationLimit(RuntimeError):
    pass


class _StepCounter:
    def __init__(self, sim):
        self.sim, self.steps = sim, 0

    def __getattr__(self, key):
        return getattr(self.sim, key)

    def forward(self, n, *args, **kwargs):
        result = self.sim.forward(n, *args, **kwargs)
        self.steps += int(n)
        return result


class PhysicsBudget:
    def __init__(self, path_limit=240, stability_limit=120, seconds=900, clock=time.monotonic):
        if any(type(x) is not int or x < 1 for x in (path_limit,stability_limit)) or seconds<=0:
            raise ValueError('invalid_evaluation_budget')
        self.clock, self.started = clock, clock()
        self.deadline=self.started+seconds
        self.limits={'path':path_limit,'stability':stability_limit}
        self.counts={'path':0,'stability':0}
        self.events=[]
        self.stop_reason=None

    def remaining(self):
        return max(0.,self.deadline-self.clock())

    def check(self, kind=None):
        reason='wall_budget' if self.remaining()<=0 else (kind+'_budget' if kind and self.counts[kind]>=self.limits[kind] else None)
        if reason:
            self.stop_reason=reason
            raise EvaluationLimit(reason)

    def report(self):
        return dict(counts=dict(self.counts), limits=dict(self.limits),stop_reason=self.stop_reason,
            elapsed_seconds=self.clock()-self.started,deadline_exceeded=self.remaining()<=0,
            physics_seconds=sum(x['seconds'] for x in self.events),
            physics_steps=sum(x['physics_steps'] for x in self.events),events=self.events,
            count_scope='native_check_success_including_terminal_not_composite_simulate')

    @contextmanager
    def installed(self, physics):
        changes=[]
        def install(cls,kind):
            original=cls.check_success
            @wraps(original)
            def check(planner,*args,**kwargs):
                self.check(kind)
                self.counts[kind]+=1
                row=dict(kind=kind,index=self.counts[kind],started=self.clock()-self.started,
                    part_move=getattr(planner,'part_move',None),parts_fix=list(getattr(planner,'parts_fix',[])),
                    raw_success=None,returned=False,error_type=None,physics_steps=0)
                self.events.append(row)
                sim=planner.sim; proxy=_StepCounter(sim); planner.sim=proxy
                old_time=getattr(planner,'max_time',None)
                if kind=='path':planner.max_time=min(old_time,self.remaining())
                else:
                    timeout=kwargs.get('timeout')
                    kwargs['timeout']=self.remaining() if timeout is None else min(timeout,self.remaining())
                start=self.clock()
                try:
                    value=original(planner,*args,**kwargs)
                    raw=value[0] if isinstance(value,tuple) else value
                    row.update(returned=True,raw_success=bool(raw))
                    return value
                except BaseException as error:
                    row['error_type']=type(error).__name__
                    raise
                finally:
                    row.update(seconds=self.clock()-start,physics_steps=proxy.steps,
                               deadline_exceeded=self.remaining()<=0)
                    planner.sim=sim
                    if kind=='path':planner.max_time=old_time
            changes.append((cls,original));cls.check_success=check
        install(physics.MultiPartPathPlanner,'path')
        install(physics.MultiPartStabilityPlanner,'stability')
        try:yield self
        finally:
            for cls,original in reversed(changes):cls.check_success=original
