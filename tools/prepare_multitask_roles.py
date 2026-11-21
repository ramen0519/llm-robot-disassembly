"""Prepare opt-in training roles from observed failures; no paid/physics calls."""
import argparse
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools.run_integrated_repair import checked
from tools.run_luna_experiment import read, write, sha
from tools.multitask_skill_evidence import validate_roles
from tools.run_multitask_learning import observed_family, FAMILIES
from tools.run_skill_discovery import source_case
from worker.task_memory import digest


def select_roles(collection, split):
    """Protocol amendment after collection, before generation/validation.

    Use all completed source observations, never recovery-test outcomes. Choose
    in declared training order, then collection order and first matching attempt.
    Training/evaluation identities never change. Require one all-PASS product.
    """
    train = [r['id'] for r in split['training']]
    validate_roles(train, [r['id'] for r in split['evaluation']])
    found = {f: {} for f in FAMILIES}
    healthy = {t: [] for t in train}
    unclean = set()
    for row in collection['episodes']:
        task = row['task_id']
        if row['purpose'] != 'training' or task not in train:
            raise ValueError('evaluation_in_training_collection')
        report = read(checked(row['report'], True))
        if not row.get('memory') or not report.get('attempts'):
            unclean.add(task)
            continue
        all_pass = all(a.get('status') == 'PASS' and a.get('accepted_for_transition') is True
                       and a.get('deadline_exceeded') is False for a in report['attempts'])
        if all_pass and report.get('success') is True:
            healthy[task].append(row)
        else:
            unclean.add(task)
        for index, attempt in enumerate(report['attempts'], 1):
            family = observed_family(attempt)
            if family:
                found[family].setdefault(task, dict(row=row, attempt_index=index))
    negative = [t for t in train if t not in unclean and len(healthy[t]) >= 2]
    if len(negative) != 1:
        raise ValueError('one_all_PASS_training_product_two_episodes_required')
    generators = [t for t in train if t not in negative]
    family = next((f for f in FAMILIES if all(t in found[f] for t in generators)), None)
    if family is None:
        raise ValueError('two_training_products_with_shared_measured_failure_required')
    return dict(family=family, generation_tasks=generators, nonapplicability_tasks=negative,
        selected=[found[family][t] for t in generators], healthy_rows=healthy[negative[0]][:2],
        coverage={f: list(v) for f, v in found.items()},
        selection_rule='fixed family priority, declared task order, first observed attempt; no recovery success selection',
        assignment_after_collection=True, shared_cause_established=False)


def child_manifest(row, spec):
    memory = checked(row['memory']['path'], True)
    if sha(memory) != row['memory']['sha256']:
        raise ValueError('collection_memory_changed')
    return dict(experience_task=row['task_id'], experience_report=spec['report'],
        experience_memory=str(memory), cases=[spec], frozen_episode_memory=dict(
            path=str(memory), file_sha256=sha(memory), content_sha256=read(memory)['content_sha256']))


def select_state_roles(collection, split):
    """Fixed task/family/trace order, no success-selected recovery outcomes."""
    from tools.skill_nonapplicability import post_pass_state
    train = [r['id'] for r in split['training']]
    validate_roles(train, [r['id'] for r in split['evaluation']])
    found = {f: {} for f in FAMILIES}; normals = []
    for task in train:
        for row in collection['episodes']:
            if row['purpose'] != 'training' or row['task_id'] not in train:
                raise ValueError('evaluation_in_training_collection')
            if row['task_id'] != task:
                continue
            if not row.get('validated') or not row.get('memory'):
                raise ValueError('validated_training_collection_required')
            report = read(checked(row['report'], True))
            for index, attempt in enumerate(report['attempts'], 1):
                family = observed_family(attempt)
                if family:
                    found[family].setdefault(task, dict(row=row, attempt_index=index))
                try:
                    provenance = post_pass_state(report, index)
                except ValueError:
                    continue
                normals.append(dict(row=row, attempt_index=index, state_provenance=provenance))
    family = next((f for f in FAMILIES if len(found[f]) >= 2), None)
    if family is None:
        raise ValueError('two_training_products_with_shared_measured_failure_required')
    # Two real no-trigger requests remain required; never synthesize healthy feedback.
    distinct = []; identities = set()
    for case in normals:
        identity = (case['row']['report'], case['state_provenance']['request_sha256'])
        if identity not in identities:
            distinct.append(case); identities.add(identity)
    if len(distinct) < 2:
        raise ValueError('two_observed_post_PASS_no_trigger_states_required')
    generators = [t for t in train if t in found[family]][:2]
    healthy = distinct[:2]
    return dict(family=family, generation_tasks=generators,
        nonapplicability_tasks=list(dict.fromkeys(c['row']['task_id'] for c in healthy)),
        selected=[found[family][t] for t in generators], healthy_cases=healthy,
        coverage={f: list(v) for f, v in found.items()}, assignment_after_collection=True,
        shared_cause_established=False, selection_rule='fixed family/task/trace order; first two actual post-PASS no-trigger requests; no recovery outcome selection')


def prepare(collection_path, split_path, output, *, role_policy='whole_product_v2'):
    collection_path, split_path, output = (checked(p, True) for p in (collection_path, split_path, output))
    collection, split = read(collection_path), read(split_path)
    if role_policy not in ('whole_product_v2', 'observed_states_v3'):
        raise ValueError('unknown_training_role_policy')
    state_roles = role_policy == 'observed_states_v3'
    choice = (select_state_roles if state_roles else select_roles)(collection, split)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'source_selection.json', dict(choice,
        collection_path=str(collection_path), collection_sha256=sha(collection_path),
        split_sha256=sha(split_path), api_calls=0, physical_calls=0))
    refs, cases, negatives = [], [], []
    for selected in choice['selected']:
        row = selected['row']; task = row['task_id']
        spec = dict(case_id='source_' + task, phase='debug', source_task=task,
            report=str(checked(row['report'], True)), attempt_index=selected['attempt_index'],
            trigger_kind='path_failed' if choice['family'] == 'path_failed' else 'stability_failed')
        if choice['family'] != 'path_failed':
            spec['expected_diagnostic_phase'] = ('dynamic_failure' if choice['family'] ==
                                                'dynamic_stability_failed' else 'initial_disconnected')
        child = child_manifest(row, spec)
        if choice['family'] != 'path_failed':
            child['python_failure_scope'] = 'remaining_body_stability_failed'
        path = output / ('source_' + task + '.json'); write(path, child)
        refs.append(dict(task_id=task, manifest_path=str(path), manifest_sha256=sha(path)))
        cases.append(spec)
    cases += [dict(c, case_id=c['case_id'] + '_yaw90', phase='admission', pose_yaw_degrees=90)
              for c in cases[:]]
    healthy_cases = choice['healthy_cases'] if state_roles else [dict(row=r, attempt_index=2) for r in choice['healthy_rows']]
    for index, healthy in enumerate(healthy_cases):
        row = healthy['row']
        task = row['task_id']; report = checked(row['report'], True)
        # The second actual removal request includes the updated state after
        # its first PASS. Never turn a PASS into a fabricated failure request.
        spec = dict(case_id='healthy_' + str(index), phase='negative_contract', source_task=task,
                    report=str(report), attempt_index=healthy['attempt_index'])
        path = output / (spec['case_id'] + '.json'); write(path, child_manifest(row, spec))
        negatives.append(dict(case_id=spec['case_id'], task_id=task, manifest_path=str(path),
            manifest_sha256=sha(path), report_sha256=sha(report),
            request_sha256=digest(source_case(spec)['request'])))
        if state_roles:
            negatives[-1]['state_provenance'] = healthy['state_provenance']
    manifest = dict(multi_task_schema='development_states_v3' if state_roles else 'development_roles_v2',
        split=dict(path=str(split_path), sha256=sha(split_path)), experience_sources=refs, cases=cases,
        require_shared_failure_family=True, nonapplicability_cases=negatives,
        role_assignment=dict(generation_tasks=choice['generation_tasks'],
            nonapplicability_tasks=choice['nonapplicability_tasks'],
            positive_admission_scope='same_products_yaw90_NOT_new_product_transfer',
            assignment_after_collection=True, evaluation_split_unchanged=True))
    if state_roles:
        manifest['role_assignment']['nonapplicability_scope'] = 'observed_post_PASS_states_NOT_held_out_products'
    from tools.run_python_skill_cycle import python_generation_packet, attach_debug_inputs
    from tools.skill_nonapplicability import bind_cases
    prepared = [source_case(c) for c in cases]
    bound = bind_cases(manifest)
    packet = python_generation_packet(manifest, 'multi-roles-offline:' + output.name)
    debug = attach_debug_inputs(deepcopy(packet), prepared, packet['run_id'])
    write(output / 'manifest.json', manifest)
    write(output / 'lesson_packet.json', packet)
    write(output / 'program_debug_packet.json', debug)
    write(output / 'healthy_cases_LOCAL_ONLY.json', bound)
    write(output / 'summary.json', dict(generation_tasks=choice['generation_tasks'],
        nonapplicability_tasks=choice['nonapplicability_tasks'], family=choice['family'],
        original_trace_observations=sum(e.get('evidence_origin') == 'episode_trace' for e in packet['evidence']),
        trace_and_memory_views=len(packet['evidence']), healthy_requests=len(bound),
        positive_cases=[dict(case_id=c['case_id'], phase=c['phase'], task=c['source_task'],
                            yaw=c.get('pose_yaw_degrees', 0)) for c in cases],
        evaluation_tasks=[r['id'] for r in split['evaluation']], evaluation_data_read=False,
        api_calls=0, physics_calls=0, generation='NOT_RUN', registration='NOT_RUN',
        yaw_validation='NOT_RUN', healthy_program_validation='NOT_RUN', evaluation='NOT_RUN'))
    return output / 'manifest.json'


def residual_proposal(ledger):
    """Disabled sub-budget, preserving unknown-cost reservation and old latch."""
    from tools.remaining_split_budget import remaining_proposal
    from orchestrator.hierarchy_budget import validate_policy
    residual = remaining_proposal(ledger)
    policy = deepcopy(residual['policy']); policy.update(enabled=False, max_calls=16, max_usd='4.80')
    estimates = {}
    for role, calls, cap in [('learner', 4, '.10'), ('scout', 4, '.12'), ('planner', 8, '4.58')]:
        cfg = policy['roles'][role]; old = residual['policy']['roles'][role]
        if calls > old['max_calls'] or Decimal(cap) > Decimal(old['max_usd']):
            raise ValueError('proposed_role_exceeds_remaining_approval')
        cfg.update(max_calls=calls, max_usd=cap)
        limit = 196608 if role == 'learner' else 262144
        amount = (Decimal(limit + cfg['request_overhead_tokens']) * Decimal(cfg['input_usd_per_million'])
                  + Decimal(cfg['max_output_tokens']) * Decimal(cfg['output_usd_per_million'])) / Decimal(1000000)
        estimates[role] = dict(calls=calls, max_body_bytes=limit, reservation=str(amount * calls))
        if amount * calls > Decimal(cap):
            raise ValueError('maximum_body_reservation_exceeds_role_cap')
    validate_policy(policy)
    if policy['max_calls'] > residual['policy']['max_calls'] or Decimal(policy['max_usd']) > Decimal(residual['policy']['max_usd']):
        raise ValueError('proposal_exceeds_remaining_approval')
    return dict(parent=residual, policy=policy, estimates=estimates,
        maximum_reservation=str(sum((Decimal(v['reservation']) for v in estimates.values()), Decimal(0))),
        automatic_retries=0, api_calls=0, current_provider_pricing_verified=False,
        actual_billing_cap_guaranteed=False, old_transport_latch_reset=False,
        continuation_runner='NOT_CONNECTED', enabled=False,
        proposed_physics_attempts=80, proposed_wall_minutes=90,
        execution_requires_explicit_protocol_continuation=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--budget-ledger', type=Path, required=True)
    args = parser.parse_args()
    target = prepare(args.collection, args.split, args.run_dir)
    write(target.parent / 'remaining_proposal_DISABLED.json', residual_proposal(args.budget_ledger))
    print(target)
