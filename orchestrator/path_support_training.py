"""Training-only root decisions using conditional, post-removal support observations."""
from copy import deepcopy
from pathlib import PurePosixPath

from orchestrator.llm_ranker import _json
from orchestrator.sequence_iterative_budget import ROLES
from orchestrator.sequence_ranker import SequenceRankService, output_schema
from worker.path_only_sequence_proposal import PROFILE
from worker.proposal_support_score import score_plan
from worker.task_memory import digest


MEMORY = 'path_support_training_memory/v1'
DECISION = 'path_support_training_root_order/v1'
INSTRUCTIONS = (
    'Choose a full order in which to try the currently available first parts. This is a fresh '
    'path-only training proposal for the same CAD. DFS and the simulator handle later choices, '
    'poses, directions and paths. Residual stability is measured after a complete or partial path. '
    'Among eligible complete proposals prefer lower witnessed peak supports, then lower total. '
    'Each row describes only its sampled chain and poses. Ties do not distinguish first parts, '
    'missing scores are unknown, and untried parts remain candidates. Never infer minimum support, '
    'robot feasibility, or a causal rule from a single chain. Use direct and return all candidates '
    'once, selected_skill_id null, exact request and state IDs, and zero or more supplied evidence IDs.'
)


def observed_proposal(result, settings, first_part, source, source_hashes, *, audit_seconds):
    """Recompute the score from the actual step witnesses, including partial/unknown rows."""
    if result['profile'] != PROFILE or settings['profile'] != PROFILE:
        raise ValueError('expected path-only proposal and settings')
    if result['input_unchanged'] is not True or result['asap_sources_unchanged'] is not True:
        raise ValueError('unverified source preservation')
    cad_ids = sorted(key[:-4] for key in settings['input_sha256'] if '/' not in key and key.endswith('.obj'))
    if sorted(result['initial_parts']) != cad_ids:
        raise ValueError('observations lack matching CAD part IDs')
    if first_part not in result['initial_parts'] or (
            result['disassembly_order'] and first_part != result['disassembly_order'][0]):
        raise ValueError('first part does not match the actual chain')
    score = score_plan(result, support_cap='auto')
    if score != result['support_score']:
        raise ValueError('score differs from step witnesses')
    settings_keys = ('base_part', 'pose_reuse', 'planner', 'terminal', 'cache_policy', 'asap_commit',
                     'redmax_binary_sha256', 'native_physics_constants', 'asap_path_sources')
    conditions = {key: settings[key] for key in settings_keys}
    for key in ('seed', 'budget', 'plan_seconds', 'max_poses'):
        conditions[key] = settings['parameters'][key]
    conditions.update(audit_seconds=audit_seconds, support_cap='auto')
    steps = []
    for step in result['steps']:
        stability = step.get('stability') or {}
        steps.append(dict(part=step['part_move'], remaining=step['child'], pose_sha256=digest(step['pose']),
                          no_support=stability.get('no_support_status'), held=stability.get('held_parts'),
                          support_status=stability.get('up_to_cap_status', 'NOT_RUN')))
    intervention = settings['parameters'].get('prescribed_first_part')
    if intervention is not None and intervention != first_part:
        raise ValueError('declared root intervention does not match the measurement')
    obs = dict(verifier_profile=PROFILE, first_part=first_part, settings=conditions,
               assembly_id=PurePosixPath(settings['assembly']).name,
               cad_digest=digest(settings['input_sha256']),
               root_policy='prespecified_first' if intervention is not None else 'unrestricted_heur_out',
               descendant_policy='native_heur_out',
               order=result['disassembly_order'], remaining=result['remaining_parts'], steps=steps,
               complete=result['complete'], score=score['score'], execution_status=result['execution_status'],
               planner_stop=result.get('planner_stop'), budget_stop=result.get('budget_stop'),
               eligible=score['eligible'], source=source, source_sha256=source_hashes)
    obs['evidence_id'] = 'path_support:' + digest(obs)
    return obs


def freeze(assembly_id, cad_hashes, observations):
    """One materialized record per intervention, stored in lexical ID order (not score order)."""
    rows = sorted(observations, key=lambda r: r['first_part'])
    if not rows or len({r['first_part'] for r in rows}) != len(rows):
        raise ValueError('one observation per sampled first part is required')
    for row in rows:
        if row['assembly_id'] != assembly_id or row['cad_digest'] != digest(cad_hashes):
            raise ValueError('observations from different training inputs cannot be pooled')
        if row['evidence_id'] != 'path_support:' + digest({k:v for k,v in row.items() if k != 'evidence_id'}):
            raise ValueError('changed source observation')
    eligible = [r for r in rows if r['eligible']]
    comparable = len({digest(r['settings']) for r in eligible}) <= 1
    scores = {tuple(r['score']) for r in eligible}
    if len(eligible) < 2:
        contrast = 'insufficient_scored_chains'
    elif not comparable:
        contrast = 'different_check_settings'
    elif len(scores) == 1:
        contrast = 'tied_on_scored_chains'
    else:
        contrast = 'different_scores_on_sampled_chains'
    summary = {
        'insufficient_scored_chains': 'Fewer than two sampled chains have an eligible complete score.',
        'different_check_settings': 'The eligible chains were measured under different settings.',
        'tied_on_scored_chains': 'All eligible sampled chains tie on peak and total witnessed supports.',
        'different_scores_on_sampled_chains': 'The eligible sampled chains have different witnessed support scores.',
    }[contrast]
    lesson = dict(origin='deterministic_observation_summary', observation=summary,
                  scope='same training CAD; each sampled whole chain at its recorded poses',
                  scored=[dict(first_part=r['first_part'], peak=r['score'][0], total=r['score'][1],
                               evidence_id=r['evidence_id']) for r in rows if r['eligible']],
                  unscored=[dict(first_part=r['first_part'], evidence_id=r['evidence_id'])
                            for r in rows if not r['eligible']],
                  evidence_ids=[r['evidence_id'] for r in rows],
                  first_part_causal_effect='UNVERIFIED', cross_product_generalization='UNVERIFIED',
                  python_skill='NOT_GENERATED')
    payload = dict(protocol=MEMORY, split='training', assembly_id=assembly_id, cad_sha256=cad_hashes,
                   observations=deepcopy(rows), comparison=contrast, scoped_lesson=lesson,
                   comparable_scored_count=len(eligible) if comparable else None,
                   conditional_count=len(rows)-len(eligible))
    return dict(payload, content_sha256=digest(payload))


def root_decision(source_envelope, memory, *, split):
    """Reuse the verified CAD at the initial state, never the native solver goal or feedback."""
    if split != 'training' or memory.get('split') != 'training' or memory.get('protocol') != MEMORY:
        raise ValueError('this product-specific memory is restricted to training')
    if memory['content_sha256'] != digest({k: v for k, v in memory.items() if k != 'content_sha256'}):
        raise ValueError('changed frozen training memory')
    SequenceRankService._validate_envelope(source_envelope)
    cad = source_envelope['features']['cad']
    task = cad['task']
    parts = sorted(cad['parts'])
    if (task['assembly_id'] != memory['assembly_id'] or task['input_sha256'] != memory['cad_sha256'] or
            parts != sorted(source_envelope['features']['memory_context']['initial_parts']) or
            parts != sorted(source_envelope['request']['assembled_parts']) or
            parts != source_envelope['request']['candidate_ids'] or
            source_envelope['features']['episode_feedback'] or task['base_part'] is not None):
        raise ValueError('requires matching initial training CAD and no episode feedback or fixed base')
    request = dict(run_id=DECISION, request_id='train:'+memory['content_sha256'],
                   state_id='initial:'+digest([task['assembly_id'], task['input_sha256'], parts]),
                   assembled_parts=parts, candidate_ids=parts, base_part=None)
    # Include all observations (also the unscored), never an inferred preference or heuristic rank.
    examples = []
    for row in memory['observations']:
        item = {key: row[key] for key in ('evidence_id', 'first_part', 'order', 'steps', 'root_policy',
                'descendant_policy', 'remaining', 'complete', 'eligible', 'score', 'execution_status', 'budget_stop')}
        item['limits'] = {k: row['settings'][k] for k in ('seed','budget','max_poses','plan_seconds','audit_seconds')}
        examples.append(item)
    payload = dict(protocol=DECISION, split=split, request=request, assembly_id=task['assembly_id'],
                   current_geometry=cad['parts'], current_relations=cad['relations'],
                   verifier_profile=PROFILE, support_cap='auto', score_fields=['peak', 'total'],
                   frozen_memory_sha256=memory['content_sha256'], observed_chains=examples,
                   comparison=memory['comparison'], scoped_lesson=deepcopy(memory['scoped_lesson']),
                   robot_execution_verified=False)
    knowledge = dict(evidence_ids=[r['evidence_id'] for r in examples], sequence_skills=[])
    cfg = ROLES['sol']
    body = dict(model=cfg['model'], store=False, instructions=INSTRUCTIONS,
                input=[dict(role='user', content=_json(payload))], max_output_tokens=cfg['max_output_tokens'],
                reasoning=dict(effort='low'), text=dict(format=dict(type='json_schema', strict=True,
                name='path_support_root_order', schema=output_schema(request, knowledge))))
    if len(_json(body).encode()) > cfg['max_body_bytes']:
        raise ValueError('training root input exceeds model input budget')
    return body
