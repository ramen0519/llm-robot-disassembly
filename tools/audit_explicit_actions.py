"""Read-only evidence audit and API cost preview; never calls API or WSL."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from run_explicit_action_experiment import audit_report, checked, read, write
from orchestrator.explicit_action_service import _ExplicitProvider
from orchestrator.explicit_action_memory import load_snapshot


def main():
    output = checked(sys.argv[1], True)
    output.mkdir(parents=True, exist_ok=False)
    checks, previews = [], []
    def check(name, valid, detail=None):
        checks.append({'name': name, 'pass': bool(valid), 'detail': detail})
    batch = ROOT/'runs/explicit_action_20261018'
    feedback = len(sys.argv) > 2 and sys.argv[2] == '--feedback'
    names = ('beam_feedback', '07009_feedback') if feedback else ('beam_fixture', '07009_fixture')
    for name in names:
        folder = batch/name
        report = read(folder/'worker_result.json')
        audit = audit_report(report)
        write(output/(name+'_audit.json'),audit)
        check(name+':audit',audit['validation_ok'] and audit['plan_success'],audit)
        check(name+':preservation', all(read(folder/'host_preservation.json').values())
              and read(folder/'preservation.json')['input_unchanged']
              and read(folder/'preservation.json')['memory_unchanged'])
        check(name+':API_zero',read(folder/'budget_after.json')['reserved_calls']==0 and audit['llm_choices']==0)
        snapshot = load_snapshot(folder/'action_memory_frozen.json')
        check(name+':memory_count',snapshot['record_count']==len(report['attempts']))
        for index, attempt in enumerate(report['attempts'],1):
            action = folder/'planner'/('attempt_%06d'%index)
            raw = read(action/'result.json')
            check(name+':raw_match:'+str(index), raw==attempt['raw'])
            for filename, hashkey in (('trajectory.json','trajectory_sha256'),('endpoint_evidence.json','evidence_sha256')):
                check(name+':'+filename+':'+str(index),hashlib.sha256((action/filename).read_bytes()).hexdigest()==raw[hashkey])
            check(name+':no_hidden_search:'+str(index),all(raw[k]==0 for k in
                ('hidden_direction_search_calls','hidden_pose_search_calls','hidden_support_search_calls')))
            if feedback:
                evidence = read(action/'endpoint_evidence.json')
                obs = raw.get('endpoint_observation') or {}
                pairs = [{k: p[k] for k in ('moving_part_id','rest_part_id','min_distance','separation_unsatisfied')}
                         for p in evidence['measurements']['pairs']]
                check(name+':measured_endpoint:'+str(index),obs.get('pairs')==pairs
                    and obs.get('min_sep')==evidence['measurements']['min_sep']
                    and obs.get('evidence_sha256')==raw['evidence_sha256']
                    and obs.get('blocker') is None and obs.get('all_paths_impossible') is False)
        provider = object.__new__(_ExplicitProvider)
        provider.config = {'model':'gpt-6-luna','max_output_tokens':768,'reasoning_effort':'none','request_overhead_tokens':2048}
        for d in report['decisions']:
            body = provider._body({'request':d['request']})
            encoded = json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            bound = len(encoded)+2048
            cost = (Decimal(bound)*Decimal('.10')+Decimal(768)*Decimal('.50'))/Decimal(1000000)
            previews.append({'source_run':name,'request_id':d['request']['request_id'],
                'input_byte_token_upper_estimate':bound,'output_cap':768,'reserved_cost_usd':str(cost)})
    beam = read(batch/names[0]/'worker_result.json')
    check('observed_failure_then_changed_path_pass',beam['attempts'][0]['status']=='UNKNOWN'
        and beam['attempts'][1]['status']=='PASS' and beam['attempts'][1]['raw']['repair_geometry_verified'] is True)
    target = read(batch/names[1]/'worker_result.json')
    source_ids = {r['evidence_id'] for r in load_snapshot(batch/names[0]/'action_memory_frozen.json')['records']}
    for index,d in enumerate(target['decisions']):
        refs=d['request']['memory_references']
        check('cross_task_reference:'+str(index),bool(refs) and all(r['evidence_id'] in source_ids
            and r['reference_only'] and not r['same_task'] and r['current_part_id_mapping'] is None for r in refs))
        if feedback:
            check('cross_task_measurements:'+str(index),all(r.get('measured_endpoint') is not None for r in refs))
    if feedback:
        first = beam['attempts'][0]['raw']['endpoint_observation']
        check('failed_endpoint_in_next_decision',beam['decisions'][1]['request']['episode_feedback'][0]['measured_endpoint']==first)
    write(output/'cost_preview.json',{'model':'gpt-6-luna','pricing':'standard_short_context',
        'input_usd_per_million':.10,'output_usd_per_million':.50,'request_previews':previews,
        'maximum_observed_request_reservation_usd':str(max(Decimal(p['reserved_cost_usd']) for p in previews)),
        'sixteen_calls_at_observed_max_usd':str(16*max(Decimal(p['reserved_cost_usd']) for p in previews)),
        'proposed_new_max_calls':16,'proposed_new_max_usd':.10,'approved':False,
        'paid_calls_performed':0,'key_accessed':False,
        'warning':'Estimate from recorded fixture payloads, not a provider-enforced billing cap. Live payloads differ; shared ledger refuses reservations beyond the cap.'})
    write(output/'audit.json',{'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL',
        'checks':checks,'count':len(checks),'paid_calls':0,'live_LLM_gate':'NOT_RUN_awaiting_new_approval',
        'performance_comparison':'NOT_RUN','feature_prototype_complete':False})
    print(json.dumps({'checks':len(checks),'passed':sum(c['pass'] for c in checks),
        'proposed16call_cost_at_observed_max':str(16*max(Decimal(p['reserved_cost_usd']) for p in previews))}))
    return 0 if all(c['pass'] for c in checks) else 1


if __name__=='__main__':
    raise SystemExit(main())
