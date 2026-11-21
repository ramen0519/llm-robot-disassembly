"""API/physics-free, precommitted 3-development/2-evaluation CAD preparation."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.prepare_transfer_pilot import safe,sha,write,normalized_geometry,ARCHIVE,ARCHIVE_SHA
from orchestrator.hierarchy_budget import validate_policy
from decimal import Decimal

TRAIN=('07009','02628','01289')
EVAL_BANDS=((5,7),(8,10))
SALT='multitask-learning-v1'
SKIP_DIRS={'before','implementation','original_sources','source_snapshot','sources','snapshot',
    'source_before','source_backup','source_backups','frozen_sources','__pycache__','node_modules','.git'}
SCAN_NAMES={'config.json','context.json','task_context.json','protocol.json','scope.json','task.json',
    'case_report.json','worker_result.json','dataset_selection.json','selected_assemblies.json',
    'features.json','library_for_reuse.json','skills.json','manifest.json','selection.json',
    'generation_packet.json','planner_report.json','action_memory_frozen.json','request.json'}


def exposure_audit(output):
    known={k:['prior_explicit_history'] for k in ('00561','07009','02628','01289','01256','02754','07544','00000')}
    examined,skipped=0,[]
    def visit(obj,ref,parent=''):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key in ('assembly_id','assembly','task_id','source_task','experience_task','selected_id') and isinstance(value,str) and re.fullmatch(r'\d{5}',value):
                    known.setdefault(value,[]).append(ref)
                if key in ('assembly_dir','assembly_path','report','experience_report') and isinstance(value,str):
                    for ident in re.findall(r'(?:[/\\])(\d{5})(?=[/\\]|$)',value):known.setdefault(ident,[]).append(ref)
                if key=='id' and parent in ('selection.json','selected_assemblies.json') and isinstance(value,str) and re.fullmatch(r'\d{5}',value):
                    known.setdefault(value,[]).append(ref)
                visit(value,ref,parent)
        elif isinstance(obj,list):
            for value in obj:visit(value,ref,parent)
    for here,dirs,files in os.walk(safe(ROOT/'runs'),followlinks=False):
        here=Path(here)
        dirs[:]=[d for d in dirs if d not in SKIP_DIRS and here/d!=output
            and not (here/d).is_symlink() and not (here/d).is_junction()]
        for name in files:
            if name not in SCAN_NAMES:continue
            p=safe(here/name)
            if p.stat().st_size>32*1024*1024:skipped.append(str(p.relative_to(ROOT)));continue
            try:value=json.loads(p.read_text(encoding='utf-8-sig'))
            except (ValueError,UnicodeError):skipped.append(str(p.relative_to(ROOT)));continue
            examined+=1;visit(value,p.relative_to(ROOT).as_posix(),name)
    return dict(examined_json_files=examined,unreadable_or_oversize=skipped,
        exposed={k:sorted(set(v)) for k,v in sorted(known.items())},
        scope='repository observed/declared tasks; NOT model pretraining audit',
        conservative_known_history_included=True)


def obj_check(payload):
    vertices=[];faces=[]
    for line in payload.decode('utf-8').splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':
            v=[float(x) for x in fields[1:4]]
            if len(v)!=3 or not all(math.isfinite(x) for x in v):raise ValueError('invalid_obj_vertex')
            vertices.append(v)
        elif fields[0]=='f':
            indices=[int(x.split('/')[0]) for x in fields[1:]]
            if len(indices)<3 or any(i==0 or (i<0 and -i>len(vertices)) for i in indices):
                raise ValueError('invalid_obj_face')
            faces.append(indices)
    if not vertices or not faces or any(i>len(vertices) for f in faces for i in f):raise ValueError('invalid_obj_topology')
    lo=[min(v[a] for v in vertices) for a in range(3)];hi=[max(v[a] for v in vertices) for a in range(3)]
    if max(hi[a]-lo[a] for a in range(3))<=0:raise ValueError('degenerate_obj')
    return dict(vertices=len(vertices),faces=len(faces),bounds=[lo,hi],finite=True,
        mesh_watertight='NOT_CHECKED',physics_feasible='NOT_RUN')


def select(inventory,exposed,signature):
    index={x['id']:x for x in inventory}
    if [index[i]['parts'] for i in TRAIN]!=[5,7,9]:raise ValueError('training_count_mismatch')
    train=[deepcopy(index[i]) for i in TRAIN]
    selected=[];excluded=[]
    known={i:signature(i) for i in sorted(set(exposed)|set(TRAIN)) if i in index}
    for lo,hi in EVAL_BANDS:
        choices=sorted((x for x in inventory if lo<=x['parts']<=hi and x['id'] not in exposed and x['id'] not in TRAIN),
            key=lambda x:hashlib.sha256((SALT+'|'+x['id']).encode()).hexdigest())
        for row in choices:
            ident=row['id'];sig=signature(ident)
            related=[i for i,s in known.items() if s and sum((sig&s).values())/min(sum(sig.values()),sum(s.values()))>=.5]
            if related:
                excluded.append(dict(id=ident,reason='at_least_half_parts_share_normalized_vertices',related=related));continue
            selected.append(dict(row,band=[lo,hi]));known[ident]=sig;break
        else:raise ValueError('no_unexposed_nonduplicate_candidate_in_band')
    return train,selected,excluded


def budget_policy():
    old=json.loads((ROOT/'runs/stability_transition_learning_20261105/preflight_02/policy_DISABLED.json').read_text())
    p=deepcopy(old);p.update(enabled=False,max_calls=34,max_usd='12.00')
    p['episode'].update(scout_calls=1,planner_calls=2,fallback_calls=0)
    for role,calls,usd in [('learner',4,'.10'),('scout',10,'.30'),('planner',20,'11.60')]:
        p['roles'][role].update(max_calls=calls,max_usd=usd)
    validate_policy(p)
    estimates={}
    for role,cfg in p['roles'].items():
        limit=196608 if role=='learner' else 262144
        single=(Decimal(limit+cfg['request_overhead_tokens'])*Decimal(cfg['input_usd_per_million'])+
                Decimal(cfg['max_output_tokens'])*Decimal(cfg['output_usd_per_million']))/Decimal(1000000)
        total=single*cfg['max_calls'];assert total<=Decimal(cfg['max_usd'])
        estimates[role]=dict(calls=cfg['max_calls'],body_bytes_limit=limit,reservation_per_call=str(single),reservation_total=str(total))
    total=sum((Decimal(x['reservation_total']) for x in estimates.values()),Decimal(0))
    assert total<Decimal(p['max_usd'])
    return p,dict(by_role=estimates,worst_configured_reservation=str(total),proposed_max_usd=p['max_usd'],
        new_api_authorized=False,live_enabled=False,current_provider_pricing_verified=False,billing_cap_guaranteed=False,
        basis='Existing configured rates, UTF8 byte ceiling + 2048 overhead + 4096 output; not invoice forecast.',
        old_residual_reused=False)


def prepare(output):
    output=safe(output)
    if output.exists():raise FileExistsError('new_preparation_directory_required')
    output.mkdir(parents=True)
    write(output/'selection_policy.json',dict(train_ids=list(TRAIN),train_part_counts=[5,7,9],
        training_rationale='reuse two developed small assemblies plus the previously timed-out nine-part case; no replacement by success',
        evaluation_bands=EVAL_BANDS,sort_rule='SHA256('+SALT+'|id)',
        exclude_prior_observed_or_declared=True,exclude_half_matching_part_vertices=True,
        no_selection_after_new_solver_outcomes=True,invalid_inputs_stop_no_silent_replacement=True,
        archive_sha256=ARCHIVE_SHA,api_calls=0,physics_calls=0))
    exposure=exposure_audit(output.parent)
    write(output/'exposure_audit.json',exposure)
    if exposure['unreadable_or_oversize']:raise ValueError('exposure_audit_incomplete_review_required')
    if sha(ARCHIVE)!=ARCHIVE_SHA:raise ValueError('official_archive_changed')
    inventory=json.loads((ROOT/'runs/assembly_rules_20261017/dataset_inventory.json').read_text())
    with zipfile.ZipFile(ARCHIVE) as z:
        allnames={}
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('unsafe_archive_member')
            allnames[info.filename]=info
        signatures={}
        def signature(ident):
            if ident not in signatures:
                signatures[ident]=Counter(normalized_geometry(z.read(name)) for name in sorted(allnames)
                    if name.startswith('test_assembly/'+ident+'/') and name.endswith('.obj'))
            return signatures[ident]
        train,evaluation,excluded=select(inventory,exposure['exposed'],signature)
        selection=dict(schema_version=1,training=train,evaluation=evaluation,exclusions=excluded,
            official_archive=str(ARCHIVE),official_archive_sha256=ARCHIVE_SHA,
            solver_results_used_for_evaluation_selection=False,evaluation_status='NOT_RUN',
            source_split='Our new development/evaluation split WITHIN official ASAP test archive',
            duplicate_check_limits='Part vertex fingerprint is translation/uniform-scale invariant; arbitrary rotation/topology/semantic duplicates are not certified.')
        write(output/'split.json',selection)  # committed before extraction/validation
        assets=[]
        for row in train+evaluation:
            prefix='test_assembly/'+row['id']+'/'
            target=output/'raw_assemblies'/row['id'];target.mkdir(parents=True)
            hashes={};checks={}
            for name in sorted(allnames):
                if not name.startswith(prefix) or allnames[name].is_dir():continue
                leaf=name[len(prefix):]
                if '/' in leaf or not (leaf.endswith('.obj') or leaf=='contact_graph.json'):
                    raise ValueError('unexpected_selected_asset_member')
                data=z.read(name)
                with (target/leaf).open('xb') as f:f.write(data)
                hashes[leaf]=hashlib.sha256(data).hexdigest()
                if leaf.endswith('.obj'):checks[leaf]=obj_check(data)
            if len(checks)!=row['parts']:raise ValueError('archive_part_count_mismatch')
            assets.append(dict(id=row['id'],parts=row['parts'],directory=str(target),files=hashes,
                obj_checks=checks,byte_exact_original_copy=True,transforms_applied=False))
    write(output/'input_checks.json',dict(assets=assets,api_calls=0,physics_calls=0,ASAP_loader='NOT_RUN'))
    p,cost=budget_policy();write(output/'policy_DISABLED.json',p);write(output/'cost_estimate.json',cost)
    protocol=dict(schema_version=1,enabled=False,split_sha256=sha(output/'split.json'),
        training_ids=list(TRAIN),evaluation_ids=[x['id'] for x in evaluation],
        generation_tasks=list(TRAIN[:2]),admission_task=TRAIN[2],
        training_seeds=[0,1],training_episode_count=6,
        evaluation_conditions=['empty_knowledge','frozen_training_memory_and_skills'],evaluation_seeds=[20261105],
        evaluation_episode_count=4,episode_api_max=3,learner_api_max=4,api_max=34,
        learner_stages=['lesson_extract:1','python_generate:1','conditional_explicit_revision:1','lesson_consolidate:1'],
        skill_family_priority=['dynamic_stability_failed','path_failed','initial_disconnected'],
        min_independent_generation_tasks=2,max_skill_candidates=2,
        no_eligible_multi_task_family='STOP_AND_REPORT_NO_NEW_SKILL; never fabricate or change held-out tasks',
        admission='original and declared yaw90 from reserved training task; never expose admission outcomes in generation',
        collect_first_then_synthesize=True,initial_knowledge='empty; exclude all historical pilot knowledge',
        evaluation_freeze_required=True,evaluation_to_training_feedback=False,
        max_physics_per_episode=16,max_physics_generation=16,max_physics_total=176,
        max_wall_seconds_per_episode=600,max_total_seconds=9000,
        same_physics=dict(max_gripper=3,max_pose=128,state_action_budget=6,pose_catalog_policy='complete_v1',
            base_part='none',timeout=480,wall_timeout=600,response_timeout=120),
        automatic_http_retries=0,old_api_residual_used=False,success_forced=False,
        performance_claim='functional pilot only; 2 evaluation tasks x 1 seed insufficient for general performance claim',
        ASAP_comparison='NOT_INCLUDED; later same verifier/config/budget and official GNN split audit',
        generated_skill_reuse_metrics=['available','applicable','selected','executed','physical_recovery_pass'],
        initial_plan='six collection episodes in seed-major fixed training order; do not exclude failed runs')
    write(output/'protocol_DISABLED.json',protocol)
    print(json.dumps(dict(training=[(x['id'],x['parts']) for x in train],evaluation=[(x['id'],x['parts']) for x in evaluation],
        exposed_tasks=list(exposure['exposed']),examined_json=exposure['examined_json_files'],
        exclusions=len(excluded),api_calls=0,physics_calls=0,proposed_api_calls=34,proposed_usd='12.00'),ensure_ascii=False))
    return selection


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    prepare(p.parse_args().run_dir)
