"""Import hash-bound historical debug evidence, never its API authorization."""
from pathlib import Path
from orchestrator.python_skill_preflight import audit_program
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read, sha


def historical_packet(packet, declaration, manifest):
    from tools.run_python_skill_cycle import refinement_packet
    from orchestrator.skill_registry import SkillRegistry
    required = {'schema_version', 'run_dir', 'result_sha256', 'manifest_sha256',
                'candidate_id', 'debug_record_sha256', 'authorization_inherited'}
    preflight_parent = set(declaration) == required | {'preflight_sha256'}
    if ((set(declaration) != required and not preflight_parent) or declaration['schema_version'] != 1
            or declaration['authorization_inherited'] is not False):
        raise ValueError('historical_debug_declaration_required')
    prior = checked(Path(declaration['run_dir']), True)
    if (sha(prior/'result.json') != declaration['result_sha256']
            or sha(prior/'precommitted_cases.json') != declaration['manifest_sha256']):
        raise ValueError('historical_parent_hash_mismatch')
    result, config = read(prior/'result.json'), read(prior/'config.json')
    if (config.get('mode') != 'live' or result.get('actual_llm_generation') is not True
            or result.get('status') != ('PREFLIGHT_REJECTED' if preflight_parent else 'NOT_REGISTERED')
            or result.get('candidate_id') != declaration['candidate_id']):
        raise ValueError('actual_failed_live_parent_required')
    registry = SkillRegistry(prior/'registry', fixture=False)
    candidate = registry.candidate(result['candidate_id'])
    records = [r for r in registry.validations(result['candidate_id']) if r['case']['phase'] == 'debug']
    if sorted(r['record_sha256'] for r in records) != sorted(declaration['debug_record_sha256']):
        raise ValueError('historical_debug_records_mismatch')
    if preflight_parent and (records or declaration['debug_record_sha256'] != []
            or result.get('preflight_sha256') != declaration['preflight_sha256']):
        raise ValueError('historical_preflight_parent_binding_mismatch')
    # This helper reads evidence only. It does not invoke a service, copy a
    # ledger, or change same-ledger validation for ordinary --refine-from.
    value = refinement_packet(packet, prior, manifest)
    if preflight_parent:
        # refinement_packet replayed the old exact code/input/audit, including its
        # saved hash. A code-level abstention is never relabeled as physics.
        checks = value['context']['development_preflight_feedback']['checks']
        if not any(c.get('runtime_reason') == 'python_abstained' for c in checks):
            raise ValueError('historical_preflight_abstention_required')
        value['context']['historical_debug_parent'] = dict(candidate_id=candidate['candidate_id'],
            result_sha256=declaration['result_sha256'], preflight_sha256=declaration['preflight_sha256'],
            debug_record_sha256=[], authorization_inherited=False,
            historical_admission_results_exposed=False, kind='preflight_abstention_not_physical_failure')
        return value
    audit = audit_program(candidate['program'], value['validation_feedback'])
    if not any(c['observed_failed_action'] is not None for c in audit['checks']):
        raise ValueError('historical_measured_debug_failure_required')
    value['context']['historical_debug_parent'] = dict(
        candidate_id=candidate['candidate_id'], result_sha256=declaration['result_sha256'],
        debug_record_sha256=declaration['debug_record_sha256'], authorization_inherited=False,
        historical_admission_results_exposed=False)
    return value


def classify_candidate(folder, result, ledger):
    """Recheck provenance before assigning another candidate slot."""
    from tools.run_python_skill_cycle import contract_rejection_feedback, refinement_packet
    status = result.get('status')
    if result.get('source_unchanged') is False:
        raise ValueError('learning_source_changed')
    if status == 'REGISTERED_SCOPED' and result.get('admission', {}).get('admitted') is True:
        return 'registered'
    if status == 'GENERATION_REJECTED':
        try:
            contract_rejection_feedback(folder, ledger)
        except (ValueError, OSError, KeyError, TypeError):
            return 'stop'
        return 'contract_rejected'
    if status not in ('PREFLIGHT_REJECTED', 'NOT_REGISTERED'):
        return 'stop'
    saved = read(folder/'generation_packet.json')
    value = refinement_packet(saved, folder, read(folder/'precommitted_cases.json'),
                              live=True, ledger=ledger)
    if status == 'PREFLIGHT_REJECTED':
        return 'contract_rejected'
    audit = audit_program(value['previous_program'], value['validation_feedback'])
    # Admission-only failure, timeout, abstention and infrastructure failures
    # are not measured debug failure and cannot authorize another revision.
    return ('measured_debug_failure' if any(c['observed_failed_action'] is not None
            for c in audit['checks']) else 'stop')
