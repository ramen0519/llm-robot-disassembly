"""Recorded offline evidence/contract replay. No API, credentials or simulator.

This command only writes a fresh run directory. Hypothetical behavior claims
are explicitly fixtures; they are not replayed provider responses or new PASS.
"""
from contextlib import ExitStack
from copy import deepcopy
from decimal import Decimal
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from run_integrated_repair import checked
from run_luna_experiment import read, write, sha
from run_skill_discovery import preview_config, executable_card
from orchestrator.skill_behavior_contract import behavior_examples, audit_behavior_claims, program_description
from orchestrator.skill_discovery import SkillDiscoveryService
from orchestrator.skill_evidence_packet import compact_generation_packet
from orchestrator.skill_program import step_program
from orchestrator.skill_refinement import audit_revision
from orchestrator.batch_budget import budget_summary


def replay_fixture(program, feedback):
    """Construct honest predictions locally for regression, never for live use."""
    requests = {row['current_observation_request']['request_id']:
                row['current_observation_request'] for row in feedback}
    rows = []
    for example in behavior_examples(feedback):
        step = step_program(program, requests[example['request_id']])
        choice = {k: v for k, v in step['response']['choice'].items() if k != 'evidence_ids'} if step['response'] else None
        passes = example['observed_pass_choices']
        relation = ('preserve_observed_pass' if choice in passes else 'change_observed_pass') if passes else 'no_observed_pass'
        rows.append(dict(request_id=example['request_id'], predicted_rule_index=step['rule_index'],
                         predicted_choice=choice, relation_to_observed_pass=relation))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    output = checked(args.run_dir, True)
    output.mkdir(parents=True, exist_ok=False)
    prior = ROOT / 'runs/initial_contact_skill_followup_20261019/live_01'
    summary, preserved = [], {}
    with ExitStack() as guards:
        for target in ('urllib.request.OpenerDirector.open', 'subprocess.Popen', 'local_api_key.key_environment'):
            guards.enter_context(patch(target, side_effect=AssertionError('offline_audit_external_execution_forbidden')))
        for name in ('candidate_01', 'candidate_02'):
            source = prior / name
            full = read(source / 'generation_packet.json')
            proposal = read(source / 'provider_proposal.json')
            preserved.update({str(source / file): sha(source / file)
                              for file in ('generation_packet.json', 'provider_proposal.json', 'api_preview.json')})
            current = deepcopy(full)
            current['context']['behavior_contract_version'] = 1
            service = SkillDiscoveryService(preview_config(), output / (name + '_unused.sqlite'))
            old_preview = service.preview(full)
            compact, audit = compact_generation_packet(current)
            new_preview = service.preview(current)
            cfg4096 = preview_config()
            cfg4096['max_output_tokens'] = 4096
            extended = SkillDiscoveryService(cfg4096, output / (name + '_4096_unused.sqlite')).preview(current)
            destination = output / name
            destination.mkdir()
            write(destination / 'provider_packet.json', compact)
            write(destination / 'compaction_audit.json', audit)
            write(destination / 'preview_4096.json', extended)
            claims = replay_fixture(proposal['program'], full['validation_feedback'])
            honest = audit_behavior_claims(proposal['program'], full['validation_feedback'], claims)
            false_claims = deepcopy(claims)
            for claim in false_claims:
                if claim['relation_to_observed_pass'] == 'change_observed_pass':
                    claim['relation_to_observed_pass'] = 'preserve_observed_pass'
            dishonest = audit_behavior_claims(proposal['program'], full['validation_feedback'], false_claims)
            write(destination / 'offline_claim_probe.json', dict(
                claim_origin='synthetic_regression_fixture_not_provider_response',
                program_origin='preserved_actual_Luna_response', honest_predictions=claims,
                honest_audit=honest, false_preservation_audit=dishonest,
                existing_revision_gate=audit_revision(proposal['program'], full['validation_feedback']),
                new_api_calls=0, new_physical_calls=0))
            write(destination / 'executable_description.json', program_description(proposal['program']))
            write(destination / 'canonical_card_preview.json', executable_card(proposal['program'], proposal['card']))
            original_body = read(source / 'api_preview.json')['body']
            summary.append(dict(candidate=name, legacy_body_unchanged=old_preview['body'] == original_body,
                full_packet_bytes=audit['full_bytes'], compact_packet_bytes=audit['provider_bytes'],
                old_body_bytes=old_preview['body_bytes'], new_body_bytes=new_preview['body_bytes'],
                body_reduction_percent=round(100*(1-new_preview['body_bytes']/old_preview['body_bytes']), 2),
                old_observed_provider_input_tokens=proposal['metadata']['usage']['input_tokens'],
                new_provider_input_tokens=None, no_new_provider_response=True,
                estimate_4096_usd=extended['cost_estimate']['usd'],
                correct_fixture_predictions_accepted=honest['accepted'],
                false_preservation_fixture_rejected=not dishonest['accepted']))
    if any(sha(checked(path)) != value for path, value in preserved.items()):
        raise ValueError('historical_evidence_changed')
    preview_budgets = {p.name: budget_summary(p) for p in output.glob('*.sqlite')}
    if any(b['request_count'] or b['reserved_calls'] for b in preview_budgets.values()):
        raise ValueError('offline_preview_reserved_an_attempt')
    report = dict(rows=summary, original_artifacts_unchanged=True,
        original_sha256=preserved, actual_api_calls=0, actual_physical_calls=0, key_accessed=False,
        ledger_created=bool(list(output.glob('*.sqlite'))),
        empty_local_preview_ledgers=preview_budgets,
        actual_Luna_improvement='NOT_RUN', new_registration='NOT_RUN', free_dfs_reuse='NOT_RUN',
        measured_max_request_estimate_usd=str(max(Decimal(row['estimate_4096_usd']) for row in summary)),
        claim_probe_is_synthetic=True, performance_effect='UNVERIFIED')
    write(output / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'original_sha256'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
