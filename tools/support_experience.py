"""Bind complete, predeclared support probes as development observations only.

This importer never picks a preferred action, creates a skill, or promotes a
probe PASS into skill validation. Original results and unsuccessful rows remain.
"""
from copy import deepcopy
from itertools import combinations
from pathlib import Path

from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read, sha
from worker.task_memory import digest
from orchestrator.skill_validation import _actual_attempt
from orchestrator.skill_case import validate_trigger_scope


def support_experiences(manifest):
    from tools.run_skill_discovery import source_case
    from tools.python_source_budget import source_budget, observation_request
    from orchestrator.python_skill_program import build_context
    from orchestrator.lesson_runtime import extension_for
    declared = manifest.get('support_probe_sources', [])
    if type(declared) is not list or len(declared) > 4:
        raise ValueError('bounded_support_probe_sources_required')
    by_id = {c['case_id']: c for c in manifest['cases']}
    groups, evidence, seen = [], [], set()
    for ref in declared:
        spec = by_id.get(ref.get('case_id'))
        if not spec or spec['phase'] != 'debug' or spec['case_id'] in seen:
            raise ValueError('support_probe_requires_unique_declared_debug_case')
        seen.add(spec['case_id'])
        root = checked(ref['run_dir'], True)
        files = {'prepared': root/'probe_prepared.json',
                 'plan': root/'offline_v2/support_probe_DISABLED.json',
                 'result': root/'live_support_probe/result.json'}
        for name, path in files.items():
            if sha(checked(path, True)) != ref[name + '_sha256']:
                raise ValueError('support_probe_' + name + '_hash_mismatch')
        prepared, plan, result = (read(files[k]) for k in ('prepared', 'plan', 'result'))
        if (prepared['plan_sha256'] != ref['plan_sha256']
                or plan['early_stop_on_success'] is not False
                or result['api_calls'] != 0 or result['registered_skills'] != 0
                or result['status'] not in ('GRID_COMPLETE', 'INSUFFICIENT_TIME_FOR_UNCHANGED_CASE_BUDGET', 'ERROR_NO_RETRY')):
            raise ValueError('complete_declared_diagnostic_probe_required')
        source = source_case(spec)
        binding = source_budget(source)
        if plan['source_binding'] != binding:
            raise ValueError('support_probe_source_budget_mismatch')
        original = plan['source_packet']
        validate_trigger_scope(source['request'], source['raw'], source['choice'])
        if any(original['choice'][k] != source['choice'][k] for k in source['choice']
               if k not in ('support_ids', 'evidence_ids')):
            raise ValueError('support_probe_source_non_support_scope_mismatch')
        for key in ('task_context', 'remaining_parts', 'base_part', 'max_grippers', 'pose_catalog'):
            if plan['request'][key] != source['request'][key]:
                raise ValueError('support_probe_declared_condition_mismatch')
        eligible = sorted(set(source['request']['remaining_parts']) -
                          {original['choice']['part_id'], source['request']['base_part']})
        expected = [list(c) for n in range(source['request']['max_grippers']) for c in combinations(eligible, n)]
        choices = plan['choices']
        if (len(expected) > 256 or [r['support_ids'] for r in choices] != expected
                or [r['index'] for r in choices] != list(range(len(expected)))
                or len(prepared['packet_hashes']) != len(expected)
                or result['attempted'] != len(result['results'])
                or result['not_run'] != len(expected) - result['attempted']
                or [r['index'] for r in result['results']] != list(range(result['attempted']))):
            raise ValueError('support_probe_incomplete_or_reordered_denominator')
        request, _ = observation_request(source, 'support-development', 'support-development:' + spec['case_id'])
        context, _ = build_context(request, extension_for())
        rows, refs = [], [{'path': str(p), 'sha256': sha(p)} for p in files.values()]
        case = dict(input_hashes=source['raw']['input_hashes'], source_hashes=source['expected_source_hashes'],
                    source_budget=binding, case_wall_seconds=binding['seconds'])
        for choice in choices:
            index = choice['index']
            if any(choice['choice'][k] != original['choice'][k] for k in original['choice']
                   if k not in ('support_ids', 'evidence_ids')):
                raise ValueError('probe_changed_non_support_action')
            row = dict(index=index, support_ids=deepcopy(choice['support_ids']),
                       raw_status=None, outcome='NOT_RUN', path_success=None, stability_success=None)
            if index < result['attempted']:
                folder = checked(root/'live_support_probe'/('support_%02d' % index), True)
                saved = result['results'][index]
                if not (folder/'physics/result.json').is_file():
                    if saved['status'] != 'ERROR' or index != result['attempted'] - 1:
                        raise ValueError('missing_nonerror_support_result')
                    row.update(outcome='ERROR', physical_result_present=False)
                else:
                    item = _actual_attempt(folder/'physics', case, False, False)
                    raw, physical = item['result']['value'], item['request']['value']
                    packet = read(checked(folder/'input.json', True))
                    if digest(packet) != prepared['packet_hashes'][index] or packet != physical:
                        raise ValueError('support_probe_packet_binding_mismatch')
                    if (raw['chosen'] != choice['choice'] or saved['support_ids'] != choice['support_ids']
                            or any(saved[k] != raw[k] for k in ('status', 'path_success', 'stability_success'))):
                        raise ValueError('support_probe_summary_disagrees_with_raw')
                    validate_trigger_scope(source['request'], raw, choice['choice'])
                    observed = dict(evidence_id='support-probe:' + item['result']['sha256'],
                        source_task=spec['source_task'], status=raw['status'], choice=deepcopy(raw['chosen']),
                        path_success=raw['path_success'], stability_success=raw['stability_success'],
                        deadline_exceeded=item['late'], accepted_for_transition=False,
                        source_result_sha256=item['result']['sha256'],
                        elapsed_seconds=raw['elapsed_seconds'],
                        stability_diagnostics=deepcopy(raw.get('stability_diagnostics')),
                        origin='predeclared_diagnostic_grid_NOT_generated_skill', phase='debug',
                        reference_only=True, source_part_ids_are_not_target_ids=True,
                        physics_reuse_allowed=False, failure_cause=None)
                    evidence.append(observed)
                    row.update(evidence_id=observed['evidence_id'], raw_status=raw['status'],
                        outcome='LATE' if item['late'] else raw['status'],
                        path_success=raw['path_success'], stability_success=raw['stability_success'],
                        elapsed_seconds=raw['elapsed_seconds'], physical_result_present=True,
                        stability_diagnostics=deepcopy(raw.get('stability_diagnostics')),
                        source_result_sha256=item['result']['sha256'])
                    refs.extend({'path': r['path'], 'sha256': r['sha256']} for r in item['refs'])
            rows.append(row)
        groups.append(dict(case_id=spec['case_id'], source_task=spec['source_task'], phase='debug',
            source_runtime_example=context, fixed_choice=deepcopy(original['choice']),
            source_task_context_sha256=digest(request['task_context']),
            declared_count=len(choices), attempted_count=result['attempted'], not_run=result['not_run'],
            rows=rows, source_refs=refs, fixed_conditions_sha256=digest(original['context']),
            selection_policy='all_predeclared_support_sets_in_order',
            generalization='UNVERIFIED', diagnostic_origin=True, recommended_action=None,
            causal_effect_established=False, runtime_observation_injection=False))
    return dict(kind='support_development_experiences_v1', groups=groups, evidence=evidence,
                api_calls=0, new_physics_calls=0, registration_evidence=False)


def attach_support_experiences(packet, manifest):
    if not manifest.get('support_probe_sources'):
        return packet
    bundle = support_experiences(manifest)
    result = deepcopy(packet)
    result['context']['support_development_experiences'] = bundle
    identities = {e['evidence_id'] for e in result['evidence']}
    for row in bundle['evidence']:
        if row['evidence_id'] in identities:
            raise ValueError('duplicate_support_probe_evidence')
        result['evidence'].append(deepcopy(row)); identities.add(row['evidence_id'])
    return result
