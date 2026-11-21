"""Correct saved-observation replay clocks; preserve all historical artifacts."""
from copy import deepcopy
from math import isfinite
from pathlib import Path
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read,sha
from orchestrator.skill_registry import SkillRegistry
from worker.task_memory import digest


def recorded_seconds(prepared):
    spec=prepared['spec'];path=checked(spec['report'],True)
    if sha(path)!=prepared['source_report_sha256']:raise ValueError('source_report_changed')
    report=read(path);attempt=report['attempts'][spec['attempt_index']-1]
    seconds=prepared['context'].get('remaining_wall_seconds')
    elapsed=prepared['raw'].get('elapsed_seconds')
    if (type(seconds) not in (int,float) or not isfinite(seconds) or not 0<seconds<=600
            or type(elapsed) not in (int,float) or not isfinite(elapsed) or elapsed<0
            or attempt.get('deadline_exceeded') is not (elapsed>seconds)):
        raise ValueError('recorded_clock_binding_invalid')
    return seconds


def replay_request(prepared,run_id,event_id):
    """Historical input projection only; do not alter the fresh-case 45s cap."""
    from orchestrator.skill_case import request_after_trigger
    from worker.explicit_action_contract import validate_request
    seconds=recorded_seconds(prepared)
    request=request_after_trigger(prepared['request'],prepared['raw'],prepared['choice'],
        run_id=run_id,event_id=event_id,wall_seconds=min(45,seconds))
    # Both flags describe the bound original observation, not a new trial.
    # recorded_seconds checked this against the actual report and request.
    late=prepared['raw']['elapsed_seconds']>seconds
    request['episode_feedback'][-1]['deadline_exceeded']=late
    request['path_stage_observations'][-1]['deadline_exceeded']=late
    return validate_request(request)


def replay_candidate(parent,manifest,ledger,model):
    """Read an actual prior generated candidate; no provider request/retry."""
    parent=checked(parent,True)
    result=read(parent/'result.json');config=read(parent/'config.json')
    if (result.get('actual_llm_generation') is not True or result.get('status')!='PREFLIGHT_REJECTED'
            or result.get('physical_cases_not_run') is not True
            or read(parent/'precommitted_cases.json')!=manifest
            or checked(config['ledger'],True)!=checked(ledger,True)
            or config['api']['model']!=model):
        raise ValueError('bound_actual_unvalidated_candidate_required')
    candidate=SkillRegistry(parent/'registry').candidate(result['candidate_id'])
    generation=candidate['generation']
    response_path=checked(generation['provider_response_path'],True)
    proposal=read(response_path)
    if (generation['origin']!='live_llm' or sha(response_path)!=generation['provider_response_sha256']
            or proposal.get('origin')!='actual_llm' or proposal.get('status')!='proposed'
            or proposal.get('metadata',{}).get('llm_live') is not True
            or candidate['program']!=proposal['program']):
        raise ValueError('actual_generated_program_provenance_required')
    return deepcopy(proposal),deepcopy(generation),dict(parent=str(parent),
        generation_packet_sha256=sha(parent/'generation_packet.json'),
        program_sha256=digest(candidate['program']),new_api_calls=0,
        original_provider_response_sha256=sha(response_path),
        reason='host replay clock correction; original code unmodified')
