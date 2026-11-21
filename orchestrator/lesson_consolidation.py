"""Post-validation repair lessons bound to immutable code and DEBUG evidence.

Admission outcomes are not model input. The model cannot redefine executable
applicability or turn a repair correlation into a physical cause.
"""
from copy import deepcopy

from worker.task_memory import digest
from orchestrator.skill_validation import read_artifact, check_evidence
from orchestrator.skill_registry import retrieve
from orchestrator.lesson_runtime import CONTRACT


def repair_packet(draft_path, skills_path, run_id, *, allow_fixture=False):
    from orchestrator.lesson_memory import validate_lesson
    draft_file = read_artifact(draft_path, allow_fixture)
    skills_file = read_artifact(skills_path, allow_fixture)
    draft = validate_lesson(draft_file['value'])
    rows = retrieve(skills_file['value'], allow_fixture=allow_fixture)
    matches = [r for r in rows if r['card'].get('lesson_id') == draft['content_sha256']]
    if len(matches) != 1 or matches[0]['card'].get('runtime_contract') != CONTRACT:
        raise ValueError('consolidation_requires_one_admitted_runtime_program')
    row = matches[0]
    entry = next(e for e in skills_file['value']['entries']
                 if e['candidate']['program_hash'] == row['program_hash'])
    # Recheck raw files and interpreter outputs, not only cached PASS labels.
    records = [check_evidence(r, allow_test_root=allow_fixture) for r in entry['validations']]
    if any(r['case'].get('runtime_contract') != CONTRACT for r in records):
        raise ValueError('consolidation_runtime_validation_missing')
    debug = [r for r in records if r['case']['phase'] == 'debug']
    kinds = {r['case']['trigger_kind'] for r in records}
    if len(kinds) != 1 or not any(r['outcome'] == 'PASS' and r['natural_trigger'] for r in debug):
        raise ValueError('consolidation_requires_supported_debug_repair')
    trigger = next(iter(kinds))
    if draft['claim']['trigger_kind'] != trigger:
        raise ValueError('draft_trigger_does_not_match_validated_repair')
    evidence, repairs = [], []
    from orchestrator.recovery_sequence import debug_steps
    transitions = [step for record in debug for step in debug_steps(record)]
    for record in transitions:
        artifacts = record['artifacts']
        observed = {}
        for phase, key in (('trigger', 'trigger_attempt_dir'), ('recovery', 'attempt_dir')):
            directory = artifacts.get(key)
            if directory is None:
                continue
            from pathlib import Path
            raw = read_artifact(Path(directory) / 'result.json', allow_fixture)['value']
            identity = 'validated-debug:' + record['record_sha256'] + ':' + phase
            item = dict(evidence_id=identity, source_task=record['case']['source_task'],
                status=raw['status'], path_success=raw.get('path_success'),
                stability_success=raw.get('stability_success'), choice=deepcopy(raw.get('chosen')),
                validation_record_sha256=record['record_sha256'], condition_hash=record['condition_hash'],
                stage=phase, physical_causality_established=False,
                source_part_ids_are_target_ids=False)
            evidence.append(item); observed[phase] = item
        if 'trigger' in observed and 'recovery' in observed:
            before, after = observed['trigger']['choice'], observed['recovery']['choice']
            repairs.append(dict(case_id=record['case']['case_id'], outcome=record['outcome'],
                before=observed['trigger'], after=observed['recovery'],
                changed_fields=[k for k in before if k != 'evidence_ids' and before[k] != after[k]],
                causal_attribution=False, target_success_guaranteed=False))
    contract = dict(trigger_kind=trigger,
        when_to_apply='Only on current ' + trigger + '; the frozen Python predicate and legal-action checks decide applicability.',
        when_not_to_apply='Abstain without a bound current failure, outside the coded predicate, or without a legal untried proposal; no target success is guaranteed.')
    binding = dict(schema_version=1, runtime_contract=CONTRACT,
        draft_path=draft_file['path'], draft_sha256=draft_file['sha256'], draft_id=draft['content_sha256'],
        skills_path=skills_file['path'], skills_sha256=skills_file['sha256'],
        program_hash=row['program_hash'], debug_record_hashes=[r['record_sha256'] for r in debug],
        fixture=skills_file['value']['fixture'], contract=contract,
        repair_observations=repairs, claims_are_hypotheses=True)
    chains = [r for r in debug if r.get('recovery_sequence')]
    if chains:
        binding['bounded_recovery_costs'] = [dict(case_id=r['case']['case_id'],
            protocol=r['case']['recovery_protocol'],
            first_attempt_outcome=r['recovery_sequence']['first_attempt_outcome'],
            final_outcome=r['outcome'], recovery_attempts=r['recovery_sequence']['physical_recovery_attempts'],
            path_calls=r['path_calls'],stability_calls=r['stability_calls'],
            elapsed_seconds=r['elapsed_seconds']) for r in chains]
    return dict(run_id=run_id, request_id=run_id + ':consolidation', state_id='frozen-validated-repair',
        operation='synthesize', existing_skills=[], evidence=evidence,
        context=dict(lesson_stage='consolidate', python_output_tokens=2048,
            repair_binding=binding, validated_program=deepcopy(row['program']),
            initial_hypothesis=deepcopy(draft['claim']), admission_results_exposed=False))


def validate_claim_contract(claim, packet):
    contract = packet['context']['repair_binding']['contract']
    if any(claim.get(k) != contract[k] for k in ('trigger_kind', 'when_to_apply', 'when_not_to_apply')):
        raise ValueError('final_lesson_executable_applicability_mismatch')
    # At least one newly validated successful recovery must support this version.
    passing = {r['evidence_id'] for r in packet['evidence']
               if r.get('stage') == 'recovery' and r.get('status') == 'PASS'}
    if not passing.intersection(claim['evidence_ids']):
        raise ValueError('final_lesson_requires_validated_recovery_citation')


def make_final_lesson(claim, packet, provenance):
    from orchestrator.lesson_memory import make_lesson, sign
    validate_claim_contract(claim, packet)
    binding = packet['context']['repair_binding']
    rebuilt = repair_packet(binding['draft_path'], binding['skills_path'], packet['run_id'],
                            allow_fixture=binding['fixture'])
    if (rebuilt['context']['repair_binding'] != binding or rebuilt['evidence'] != packet['evidence']):
        raise ValueError('final_lesson_repair_evidence_changed')
    draft = read_artifact(binding['draft_path'], binding['fixture'])['value']
    lesson = make_lesson(claim, packet, provenance, previous=draft)
    lesson.pop('content_sha256')
    lesson['repair_binding'] = deepcopy(binding)
    return sign(lesson)


def validate_final_lesson(lesson):
    binding = lesson['repair_binding']
    if (type(binding) is not dict or type(binding.get('fixture')) is not bool
            or lesson.get('previous_version') != binding.get('draft_id')):
        raise ValueError('invalid_final_lesson_binding')
    packet = repair_packet(binding['draft_path'], binding['skills_path'], 'repair-recheck',
                           allow_fixture=binding['fixture'])
    if binding != packet['context']['repair_binding']:
        raise ValueError('final_lesson_repair_binding_changed')
    from orchestrator.lesson_memory import evidence_view
    if lesson['observed_facts'] != evidence_view(packet):
        raise ValueError('final_lesson_observations_changed')
    validate_claim_contract(lesson['claim'], packet)
    return lesson
