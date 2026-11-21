"""Fixed, same-information rule control. No API, simulator, learned weights or pruning."""
from collections import Counter
from copy import deepcopy
from itertools import combinations
import math

from worker.explicit_action_contract import DIRECTIONS,SKILLS,action_id,validate_request,validate_response
from worker.path_probe_round import partition_path_round,semantic_path_key,validate_path_round_response
from orchestrator.skill_program import _scoped_last,step_program
from orchestrator.generated_skill_service import GeneratedSkillService


def signature(reference):
    """Only supplied scoped facts; never source-to-current part correspondence."""
    path,stable=reference.get('path_success'),reference.get('stability_success')
    phase='path_failed' if path is False else 'stability_failed' if path is True and stable is False else 'pass' if path is True and stable is True else 'unobserved'
    choice=reference.get('source_choice',reference.get('choice',{}))
    return dict(phase=phase,primitive=choice.get('skill_id'),support_count=len(choice.get('support_ids',[])),
        source_part_count=len(reference.get('source_remaining_parts',reference.get('remaining_parts',[]))),
        current_part_mapping=None,scope='relative_reference_not_current_verdict')


def memory_preferences(request):
    """Count comparable observed recovery templates once per development condition."""
    last=_scoped_last(request)
    stage=signature(last['feedback'])['phase'] if last else None
    refs={r['evidence_id']:r for r in request['memory_references']}
    counts=Counter(); ids={}; seen=set()
    for r in refs.values():
        parent=refs.get(r.get('parent_failure_evidence_id'))
        if (stage is None or parent is None or signature(parent)['phase']!=stage or r.get('status')!='PASS'
                or r.get('path_success') is not True or r.get('stability_success') is not True
                or r.get('verifier_origin') not in ('actual_ASAP_RedMax','ASAP_RedMax')):
            continue
        s=signature(r);key=(s['primitive'],s['support_count'])
        condition=(r.get('source_context_sha256'),tuple(r.get('source_remaining_parts',[])),r.get('source_action_id'))
        if condition in seen:continue
        seen.add(condition);counts[key]+=1;ids.setdefault(key,[]).extend([parent['evidence_id'],r['evidence_id']])
    return counts,ids


def _centers(request):
    parts=request['geometry'].get('parts',{})
    centers={p:list(parts.get(p,{}).get('bounds_center') or [0,0,0]) for p in request['candidate_ids']}
    mean=[sum(v[a] for v in centers.values())/max(1,len(centers)) for a in range(3)]
    return centers,mean


def choose_action(request):
    request=validate_request(request)
    partition=partition_path_round(request);deferred=set(partition['deferred_path_keys'])
    attempted=set(request['attempted_action_ids'])
    centers,mean=_centers(request)
    last=_scoped_last(request)
    unstable=(last['unstable']|last['initial_disconnected']) if last else set()
    counts,ids=memory_preferences(request)
    parts=request['geometry'].get('parts',{})
    # Fewer prior path failures is a soft priority, never removal of a part.
    failures=Counter(row['choice']['part_id'] for row in request['episode_feedback']
        if row.get('path_success') is False and row.get('remaining_parts')==request['remaining_parts'])
    order=sorted(request['candidate_ids'],key=lambda p:(failures[p],
        -sum((centers[p][a]-mean[a])**2 for a in range(3)),p))
    best=None;best_key=None
    for part in order:
        available=sorted(set(request['remaining_parts'])-{part,request['base_part']})
        # Support sets stay fully enumerable; priority comes from measured roles,
        # then few supports. Unknown diagnostics never become a fabricated role.
        supports=[list(c) for n in range(min(len(available),request['max_grippers']-1)+1)
                  for c in combinations(available,n)]
        support_orders={primitive:sorted(supports,key=lambda c:(-counts[(primitive,len(c))],
            -len(set(c)&unstable),len(c),c)) for primitive in SKILLS}
        vector=[centers[part][a]-mean[a] for a in range(3)]
        for pose in request['pose_catalog']:
            world=[sum(pose['matrix'][a][b]*vector[b] for b in range(3)) for a in range(3)]
            norm=max(1e-12,math.sqrt(sum(x*x for x in world)))
            for primitive in SKILLS:
                for direction,axis in DIRECTIONS.items():
                    seconds=[None] if primitive=='single_force_v1' else [k for k,v in DIRECTIONS.items() if sum(a*b for a,b in zip(axis,v))==0]
                    for second in seconds:
                        proto=dict(part_id=part,pose_id=pose['pose_id'],skill_id=primitive,direction_id=direction,
                                   second_direction_id=second,support_ids=[],evidence_ids=[])
                        if semantic_path_key(request,proto) in deferred:continue
                        for held in support_orders[primitive]:
                            choice=dict(proto,support_ids=held)
                            if action_id(request,choice) in attempted:continue
                            memory=counts[(primitive,len(held))]
                            key=(-memory,failures[part],-len(set(held)&unstable),
                                 -sum(a*b for a,b in zip(world,axis))/norm,
                                 primitive!='single_force_v1',len(held),
                                 -sum(x*x for x in vector),part,pose['pose_id'],direction,second or '',tuple(held))
                            if best_key is None or key<best_key:best,best_key=choice,key
                            # If no observations favor a different support count,
                            # the first legal set is optimal for this path.
                            break
    if best is None:
        from worker.path_probe_round import canonical_path_round_response
        return canonical_path_round_response(request,None)
    cited=ids.get((best['skill_id'],len(best['support_ids'])),[])
    best['evidence_ids']=sorted(set(cited))[:12]
    response=dict(schema_version=1,protocol=request['protocol'],request_id=request['request_id'],state_id=request['state_id'],
        candidate_order=[best['part_id']]+[p for p in order if p!=best['part_id']],choice=best,
        action_id=action_id(request,best),fallback_reason=None,
        metadata=dict(selection_source='deterministic_control',llm_live=False,attempted=False,
                      rule_policy='relative_evidence_outward_v1',memory_templates=len(counts),
                      candidate_pruning=False,physics_bypass=False,physics_verdict_reused=False))
    return validate_path_round_response(request,validate_response(request,response))


class TransferRuleActor:
    mode='deterministic'
    fallback_policy='path_stage_v1'
    def __init__(self,timeout_seconds=60):self.timeout_seconds=timeout_seconds;self.decisions=[]
    def handle(self,request):
        response=choose_action(request)
        self.decisions.append(dict(request=deepcopy(request),response=deepcopy(response)))
        return response
    def close(self):pass


class TransferRuleSkills(GeneratedSkillService):
    def _rows(self,request):
        rows=super()._rows(request)
        counts,_=memory_preferences(request)
        def priority(row):
            step=step_program(row['program'],request,max_actions=self.max_scan_actions)
            response=step['response']
            if response is None:return (1,0,0,row['strategy_id'])
            choice=response['choice']
            scope_count=len({str(v) for v in row.get('validation_scope',[])})
            return (0,-counts[(choice['skill_id'],len(choice['support_ids']))],-scope_count,row['strategy_id'])
        return sorted(rows,key=priority)
