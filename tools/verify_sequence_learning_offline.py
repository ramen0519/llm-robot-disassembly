"""Explicit fixture-model + real native Beam validation, never a paid trial.

Reads previously saved actual observations, tests both successful and rejected
responses with no HTTP, validates the candidate on two fresh Beam conditions,
and replays another product's request to check registered-card consumption.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_luna_experiment import read,write,safe
from run_sequence_training import NativeTrainingBackend
import run_sequence_native_experiment as native
from orchestrator.sequence_ranker import SequenceRankService
from orchestrator.sequence_learning import freeze_sequence_lessons


def injected_model(url, headers, body, timeout):
    request=json.loads(body['input'][0]['content'])
    # Service provider projection still has request identity and packet facts.
    data=request['input']; stage=request['stage']
    evidence=data['packet']['facts'][0]['evidence_id']
    result=dict(request_id=request['request_id'],state_id=request['state_id'])
    if stage.startswith('lesson_'):
        result['lesson']=dict(title='Offline wiring fixture',hypothesis='Fixture ordering; no effectiveness claim.',
            when_to_apply='Only this explicitly labelled technical validation.',
            when_not_to_apply='Do not treat this as an automatically learned general rule.',
            ranking_guidance='Preserve all candidates; the real verifier still checks every removal.',
            evidence_ids=[evidence],counterexample_ids=[])
    else:
        result['candidate']=dict(name='Fixture full permutation',description='Copies the complete input order for wiring validation only.',
            source="def propose(ctx):\n    return {'candidate_order': ctx['candidate_ids'][:], 'evidence_ids': []}\n",
            evidence_ids=[evidence])
    return dict(model='gpt-6-luna',status='completed',output=[dict(type='message',role='assistant',
        content=[dict(type='output_text',text=json.dumps(result))])],
        usage=dict(input_tokens=100,output_tokens=30,total_tokens=130))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();out=safe(args.run_dir,True);out.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'runs/sequence_native_prepare_20261110/beam_smoke_v2'
    task=dict(assembly_id='beam_assembly/original',part_count=5,assembly_dir=native.ASAP+'/assets/beam_assembly/original',base_part='6')
    # Dummy key is for injected RankService contract only; never read stored key.
    with patch.dict(os.environ,OPENAI_API_KEY='offline-fixture-not-a-secret'), patch(
            'urllib.request.OpenerDirector.open',side_effect=AssertionError('HTTP forbidden')):
        backend=NativeTrainingBackend(out,['00335'],live=False,learner_transport=injected_model,
            episode_runner=native.execute,validation_seconds=120)
        backend.last_episode[task['assembly_id']]=previous
        backend.collect(dict(path=str(previous)))
        packet=backend.packet(task,dict(path=str(previous)))
        lesson=backend.lesson(task,packet)
        if not lesson['ok']:raise ValueError('fixture_lesson_failed:'+str(lesson['error']))
        candidate=backend.candidate(task,packet,lesson['lesson'],None,None,0)
        if not candidate['ok']:raise ValueError('fixture_generation_failed:'+str(candidate['error']))
        validation=backend.validate_candidate(task,candidate['candidate'],lesson['lesson'])
        if not validation['ok']:
            write(out/'result.json',dict(status='VALIDATION_FAILED',validation=validation,actual_API_calls=0))
            return 1
        post=backend.lesson(task,packet,previous_lesson=lesson['lesson'],validation=validation,candidate=candidate['candidate'])
        if not post['ok']:raise ValueError('fixture_postlesson_failed:'+str(post['error']))
        card=backend.register(candidate['candidate'],post['lesson'],validation)
        frozen=backend.freeze('final')
        path=next((ROOT/'runs/sequence_native_00335_live_20261110/trial/episode/ipc/requests').glob('*.json'))
        envelope=read(path)
        def choose(url,headers,body,timeout):
            value=json.loads(body['input'][0]['content']);req=value['request']
            choices=value['knowledge']['sequence_skills']
            if not choices:raise AssertionError('registered card was not offered')
            result=dict(request_id=req['request_id'],state_id=req['state_id'],decision_kind='skill',
                candidate_ids=None,selected_skill_id=choices[0]['skill_id'],evidence_ids=[])
            return dict(status='completed',output=[dict(type='message',role='assistant',
                content=[dict(type='output_text',text=json.dumps(result))])],
                usage=dict(input_tokens=100,output_tokens=20,total_tokens=120))
        svc=SequenceRankService(backend.sol_config,out/'replay_ledger.sqlite',transport=choose,
            memory_records=backend.snapshots,skills=[card],lessons=freeze_sequence_lessons(backend.lessons),allow_fixture_skills=True)
        response=svc.rank(envelope)
        write(out/'cross_product_replay.json',response)
        success=response['fallback_reason'] is None and response['metadata']['skill_execution']['status']=='PROPOSED'
        write(out/'result.json',dict(status='PASS' if success else 'FAIL',actual_API_calls=0,
            model_generation='INJECTED_FIXTURE',physics_validation='ACTUAL_ASAP_REDMAX_TWO_BEAM_SEEDS',
            registration_fixture=card['fixture'],cross_product_reuse='STORED_REQUEST_REPLAY_ONLY',
            cross_product_physics='NOT_RUN',learned_effect='UNVERIFIED',frozen=frozen,
            postlesson_observation_count=len(post['lesson']['facts'])))
        return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
