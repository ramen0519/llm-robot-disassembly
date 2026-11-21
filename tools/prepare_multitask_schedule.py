"""Write a disabled, pinned collection/evaluation schedule. Never execute it."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.prepare_multitask_learning import budget_policy
from tools.prepare_transfer_pilot import safe,sha,write
from tools.run_luna_experiment import read
from tools.multitask_skill_evidence import validate_roles


def episode_order(split):
    train=[x['id'] for x in split['training']];evaluation=[x['id'] for x in split['evaluation']]
    validate_roles(train,evaluation)
    rows=[dict(task=i,seed=s,purpose='training',condition='empty_collection') for s in (0,1) for i in train]
    # Predeclared balanced order. Every evaluation starts from the initial CAD;
    # there is no knowledge update between either condition or either product.
    for n,i in enumerate(evaluation):
        conditions=['empty_knowledge','frozen_training_memory_and_skills']
        if n%2:conditions.reverse()
        rows.extend(dict(task=i,seed=20261105,purpose='evaluation',condition=c) for c in conditions)
    return rows


def prepare(prepared,output):
    prepared=safe(prepared);output=safe(output)
    output.mkdir(parents=True,exist_ok=False)
    split=read(prepared/'split.json');native=read(prepared/'prepared_inputs.json')
    protocol=read(prepared/'protocol_DISABLED.json')
    train=[x['id'] for x in split['training']];evaluation=[x['id'] for x in split['evaluation']]
    validate_roles(train,evaluation)
    policy,cost=budget_policy()
    write(output/'policy_DISABLED.json',policy);write(output/'cost_estimate.json',cost)
    write(output/'scout_config.json',policy['roles']['scout'])
    from orchestrator.skill_library_merge import freeze_skill_union
    from orchestrator.lesson_memory import freeze_lessons
    from orchestrator.explicit_action_memory import freeze_memory
    initial=output/'empty_knowledge';initial.mkdir()
    skills=freeze_skill_union([],initial/'skills.json')
    write(initial/'lessons.json',freeze_lessons([],skills))
    (initial/'observations.jsonl').write_text('',encoding='utf8')
    freeze_memory(initial/'observations.jsonl',initial/'memory.json')
    settings={k:protocol['same_physics'][k] for k in ('state_action_budget','max_gripper','max_pose','timeout','wall_timeout','response_timeout')}
    settings['budget']=protocol['max_physics_per_episode']
    commands=[]
    from tools.run_hierarchical_episode import parser
    from worker.transfer_protocol import validate_manifest
    for n,row in enumerate(episode_order(split)):
        tag=f"{n+1:02d}_{row['task']}_{row['seed']}_{row['condition']}"
        assembly=next(p for p in native['assembly_dirs'] if Path(p).name==row['task'])
        manifest=dict(schema='multitask_learning_v1',purpose=row['purpose'],seed=row['seed'],
            automatic_retries=0,settings=settings,training_ids=train,evaluation_ids=evaluation,
            assembly_dirs={assembly:native['assembly_dirs'][assembly]},path_limit=96,stability_limit=48,
            split_sha256=sha(prepared/'split.json'),input_manifest_sha256=sha(prepared/'prepared_inputs.json'))
        manifest_path=output/'manifests'/(tag+'.json');write(manifest_path,manifest)
        knowledge=output/'pending_frozen_training' if row['condition']=='frozen_training_memory_and_skills' else initial
        run=output/'pending_live'/tag
        # No --allow-live-api flag and a disabled hierarchy policy. These are
        # parse-checked plans, not a hidden launcher or authority to spend.
        args=['--run-dir',str(run),'--mode','llm','--assembly-dir',assembly,'--base-part','none',
            '--seed',str(row['seed']),'--evaluation-manifest',str(manifest_path),
            '--hierarchy-policy',str(output/'policy_DISABLED.json'),
            '--hierarchy-ledger',str(output/'pending_shared_budget.sqlite'),
            '--llm-config',str(output/'scout_config.json'),'--ledger',str(run.parent/(tag+'_scout.sqlite')),
            '--planner-ledger',str(run.parent/(tag+'_planner.sqlite')),
            '--skill-library',str(knowledge/'skills.json'),'--lesson-library',str(knowledge/'lessons.json'),
            '--memory',str(knowledge/'memory.json'),'--memory-policy','assembly_balanced_v1','--memory-max-records','8',
            '--lesson-runtime','--max-skill-attempts','1',
            '--provider-contract','explicit_action_provider_v5','--plan-provider-contract','conditional_plan_v4',
            '--fallback-policy','path_stage_v1','--path-selection-policy','path_probe_round_v1',
            '--recovery-support-policy','preserve_existing_v1','--planning-feedback-policy','repair_evidence_v2',
            '--transport-failure-policy','latch_on_error_v1','--pose-catalog-policy','complete_v1','--stability-diagnostics']
        for k,v in settings.items():args+=['--'+k.replace('_','-'),str(v)]
        parsed=parser().parse_args(args);validate_manifest(manifest_path,assembly,parsed)
        commands.append(dict(row,id=tag,argv=[str(ROOT/'.venv-orchestrator/Scripts/python.exe'),'-X','utf8','-B','-u',
            str(ROOT/'tools/run_hierarchical_episode.py'),*args],
            enabled=False,missing_authorization=True,manifest=str(manifest_path),manifest_sha256=sha(manifest_path),
            prerequisites=['explicit_new_budget_approval']+(['training_collection_complete','skill_admitted_and_lesson_updated',
                'training_only_memory_skill_lesson_bundle_frozen'] if row['purpose']=='evaluation' else [])))
    protocol.update(policy_path=str(output/'policy_DISABLED.json'),api_max=34,max_usd='12.00',
        supersedes_cost_only=str(prepared/'cost_estimate.json'),
        source_experience_selection='First matching observed failure in seed-major order per generation task; all runs retained.',
        actual_batch_launcher='NOT_ENABLED; bind fresh observations/compatible family before learning',
        generation_manifest='PENDING_NEW_OBSERVATIONS; saved historical replay is only an integration check',
        require_shared_failure_family=True,
        evaluation_order='task1 empty then frozen; task2 frozen then empty',
        learning_gate='At least two generation tasks with matching measured failure family plus reserved training admission case; otherwise stop.',
        repeated_collection='Two seeds per task, all start with empty long-term knowledge. Merge six training logs only after collection.',
        scope='One learned repair family; does not promise multiple registered skills or solve every product.')
    write(output/'protocol_DISABLED.json',protocol)
    write(output/'schedule_DISABLED.json',dict(episodes=commands,api_calls=0,path_calls=0,stability_calls=0,
        teacher_call_allocation=['lesson_extract','python_generate','optional_explicit_revision','lesson_consolidate'],
        new_authorization_required=True,old_residual_reused=False))
    print('DISABLED_SCHEDULE_READY: 6 collection + 4 evaluation; 34 API / configured USD 12 proposed; 0 executed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.prepared,a.output)
