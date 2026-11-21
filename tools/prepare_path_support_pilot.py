"""Freeze measured 00561 training observations and prepare the next order request offline."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.llm_ranker import _json
from orchestrator.path_support_training import freeze, observed_proposal, root_decision
from orchestrator.sequence_iterative_budget import ROLES
from tools.rank_path_proposals import rank_runs
from tools.run_luna_experiment import read, safe, sha, write


CAD_ID = '00561'
PRIOR = ROOT/'runs/path_only_proposal_00561_20261111/physics'
RECHECK = ROOT/'runs/00561_auto_support_20261111/out'
PILOT = ROOT/'runs/00561_small_learning_pilot_20261111'
INITIAL_CAD = (ROOT/'runs/sequence_iterative_live_20261110/batch/round01_00561/00561_round1/episode'
               /'ipc/requests/2a1ce62c897b5e8f5a1eaa56c4983d43146c82ffce152540ce965e440e6dfb5f.json')


def check_trajectories(directory, steps):
    for step in steps:
        row = step['trajectory']
        name = row['file']
        if name != 'path_{:04d}.npy'.format(step['attempt_id']) or sha(directory/name) != row['sha256']:
            raise ValueError('trajectory provenance mismatch')


def reference():
    path = RECHECK/'stability_audit.json'
    audit = read(path)
    settings = read(PRIOR/'settings.json')
    original = read(PRIOR/'proposal.json')
    pins = audit['path_source']['file_sha256']
    if (audit['scope'] != 'fresh_native_gravity_on_frozen_path' or audit['support_cap'] != 'auto'
            or audit['fresh_path_checks'] != 0 or not all(audit['preserved'].values())
            or not all(pins[name] == sha(PRIOR/name) for name in ('result.json', 'settings.json', 'proposal.json'))
            or [{k:v for k,v in step.items() if k != 'stability'} for step in audit['steps']] != original['steps']):
        raise ValueError('frozen reference check failed')
    check_trajectories(PRIOR, audit['steps'])
    result = dict(original, steps=audit['steps'], support_score=audit['support_score'],
                  input_unchanged=audit['preserved']['cad'], asap_sources_unchanged=audit['preserved']['asap_sources'])
    return settings, observed_proposal(result, settings, '5', path.relative_to(ROOT).as_posix(),
                                     dict(pins, stability_audit=sha(path)), audit_seconds=audit['audit_budget_s'])


def intervention(first, expected_hashes):
    folder = PILOT/('first_'+first)
    physics = folder/'physics'
    settings = read(physics/'settings.json')
    result = read(physics/'result.json')
    condition = read(folder/'condition.json')
    verification = read(folder/'verified.json')
    source_path = physics/'result.json'
    pins = dict(result=sha(source_path), settings=sha(physics/'settings.json'),
                condition=sha(folder/'condition.json'), verified=sha(folder/'verified.json'))
    if (settings['parameters']['prescribed_first_part'] != first or settings['parameters']['support_cap'] != 'auto'
            or condition['first_part'] != first or verification['root_condition_verified'] is not True
            or verification['result_sha256'] != pins['result'] or settings['input_sha256'] != expected_hashes
            or result['settings_sha256'] != pins['settings']):
        raise ValueError('root intervention or CAD identity differs from the design')
    rank_runs([physics])  # Normal producer's run compatibility, native witness and score checks.
    check_trajectories(physics, result['steps'])
    return observed_proposal(result, settings, first, source_path.relative_to(ROOT).as_posix(), pins,
                             audit_seconds=settings['parameters']['audit_seconds'])


def prepare(output, *, first_parts=('0', '2')):
    output = safe(output, True)
    if output.exists():
        raise ValueError('use a new output directory')
    settings, baseline = reference()
    cad_hashes = settings['input_sha256']
    rows = [baseline] + [intervention(first, cad_hashes) for first in first_parts]
    memory = freeze(CAD_ID, cad_hashes, rows)
    body = root_decision(read(INITIAL_CAD), memory, split='training')
    size = len(_json(body).encode())
    cfg = ROLES['sol']
    cost = ((size+cfg['request_overhead_tokens'])*Decimal(cfg['input_usd_per_million'])
            + cfg['max_output_tokens']*Decimal(cfg['output_usd_per_million']))/Decimal(1000000)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'frozen_training_memory.json', memory)
    write(output/'next_sol_request.json', body)
    report = dict(status='PREPARED_OFFLINE', split='training', scope='same_CAD_initial_state',
                  frozen_sha256=memory['content_sha256'], comparison=memory['comparison'],
                  observations=len(rows), first_parts=[r['first_part'] for r in memory['observations']],
                  source_cad_envelope_sha256=sha(INITIAL_CAD), proposed_model=body['model'],
                  response_schema='full_permutation_and_optional_evidence_citations',
                  request_body_bytes=size, configured_reservation_usd=str(cost),
                  actual_api_calls=0, model_choice='NOT_RUN', learned_python_skill='NOT_RUN',
                  evaluation_data_used=False, next_round_physics='NOT_RUN')
    write(output/'preparation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False))
