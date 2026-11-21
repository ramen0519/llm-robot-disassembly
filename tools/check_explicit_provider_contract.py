"""Offline historical compatibility and request-cost preview; no transport/key.

This projects archived choices into a new wire format. It does not recover
rejected provider text, run a model, or re-execute the archived physics.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from run_explicit_action_experiment import checked, read, write
from orchestrator.explicit_action_service import (
    PROVIDER_CONTRACT, _ExplicitProvider, expand_provider_decision)
from worker.explicit_action_contract import validate_request, validate_response
from worker.explicit_action_planner import _selection_feedback


def augment_archived_request(request, report):
    """Reconstruct only from recorded dispatches, retaining original IDs/hash."""
    result = deepcopy(request)
    by_action = {a['action_id']: a for a in report['attempts']}
    by_request = {d['request']['request_id']:d for d in report['decisions']}
    result['attempted_actions'] = [dict(action_id=identifier,
        **{k:deepcopy(v) for k,v in by_action[identifier]['choice'].items() if k!='evidence_ids'})
        for identifier in result['attempted_action_ids']]
    by_evidence = {a['event_id']: a for a in report['attempts']}
    for row in result['episode_feedback']:
        attempt = by_evidence[row['evidence_id']]
        decision = by_request[attempt['request_id']]
        row['decision_source'] = attempt['decision_source']
        row['selection_feedback'] = _selection_feedback(decision['response'],decision['request'])
    return validate_request(result)


def inspect():
    provider = object.__new__(_ExplicitProvider)  # No ledger, env or HTTP initialization.
    provider.config = {'model':'gpt-6-luna','max_output_tokens':768,
        'reasoning_effort':'none','request_overhead_tokens':2048}
    runs = ['explicit_action_20261018/beam_feedback','explicit_action_20261018/07009_feedback',
            'explicit_action_live_20261018/beam','explicit_action_live_20261018/07009']
    checks, previews, summaries, input_hashes = [], [], [], {}
    for name in runs:
        path=checked(ROOT/'runs'/name/'worker_result.json')
        input_hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        report=read(path); converted=0; originals=0; skipped=0
        for decision in report['decisions']:
            original=decision['request']; response=decision['response']
            validate_response(original,response); originals+=1
            request=augment_archived_request(original,report)
            validate_response(request,response)  # Same action identity with readable history.
            checks.append({'source':name,'request_id':request['request_id'],
                           'internal_response_compatible':True,'attempt_table_exact':True})
            if response['choice'] and response['fallback_reason'] is None:
                wire={k:deepcopy(response[k]) for k in ('request_id','state_id','candidate_order','choice')}
                wire['choice'].pop('part_id')
                expanded=validate_response(request,expand_provider_decision(request,wire))
                assert expanded['action_id']==response['action_id']
                assert expanded['choice']==response['choice']
                converted+=1
            else:
                # No raw rejected body was recorded. Never fabricate its content.
                skipped+=1
            body=provider._body({'request':request})
            assert 'part_id' not in body['text']['format']['schema']['properties']['choice']['properties']
            encoded=json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
            bound=len(encoded)+provider.config['request_overhead_tokens']
            reservation=(Decimal(bound)*Decimal('.10')+Decimal(768)*Decimal('.50'))/Decimal(1000000)
            previews.append({'source':name,'request_id':request['request_id'],
                'input_byte_token_upper_estimate':bound,'output_cap':768,'reservation_usd':str(reservation)})
        summaries.append({'source':name,'old_internal_responses_valid':originals,
            'new_wire_projection_identical_action':converted,
            'invalid_provider_bodies_not_reconstructed':skipped})
    for name,h in input_hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    maximum=max(Decimal(p['reservation_usd']) for p in previews)
    return {'status':'PASS','provider_contract':PROVIDER_CONTRACT,'checks':checks,
        'runs':summaries,'input_sha256':input_hashes,'previous_artifacts_unchanged':True,
        'paid_API_calls':0,'actual_key_access':False,'new_physics_calls':0,
        'live_LLM_revalidation':'NOT_RUN','performance_comparison':'NOT_RUN',
        'limitations':['Offline projection of archived chosen actions, not new model output.',
            'Eight rejected provider bodies are unavailable and were not reconstructed.',
            'A valid action format does not establish physical feasibility or memory benefit.'],
        'cost_preview':{'request_previews':previews,
            'maximum_observed_new_payload_reservation_usd':str(maximum),
            'sixteen_calls_at_observed_max_usd':str(16*maximum),
            'proposed_max_calls':16,'proposed_program_estimated_cap_usd':'0.12',
            'approved':False,'automatic_retries':0,'per_assembly_physical_attempt_cap':8,
            'endpoint':'https://api.openai.com/v1','model':'gpt-6-luna',
            'pricing_checked_date':'2026-10-18','input_usd_per_million':'0.10','output_usd_per_million':'0.50',
            'pricing_source':'https://developers.openai.com/api/docs/pricing',
            'billing_cap_guaranteed':False,
            'scope':'Beam and 07009; freeze new Beam observations before 07009; new ledger required.',
            'warning':'Byte-based conservative estimate from recorded requests with new fields; future inputs may differ and the program must refuse reservations over its cap.'}}


def main():
    output=checked(sys.argv[1],True)
    output.mkdir(parents=True,exist_ok=False)
    # Any accidental network or key read fails this read-only diagnostic.
    with patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('network_forbidden')), \
         patch('orchestrator.llm_ranker.os.environ.get',side_effect=AssertionError('key_forbidden')):
        result=inspect()
    write(output/'audit.json',result)
    print(json.dumps({'status':result['status'],'internal_responses':len(result['checks']),
        'projected_choices':sum(x['new_wire_projection_identical_action'] for x in result['runs']),
        'proposed16call_reservation_usd':result['cost_preview']['sixteen_calls_at_observed_max_usd'],
        'paid_API_calls':0,'new_physics_calls':0}))
    return 0


if __name__=='__main__': raise SystemExit(main())
