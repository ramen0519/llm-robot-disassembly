"""Offline, outcome-independent preparation for five order-only learning tasks.

This tool never loads credentials, starts WSL/physics, or calls a model. Prepared
raw CAD is not a physics-ready certificate. All proposed budgets remain disabled.
"""
import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from tools.prepare_transfer_pilot import safe, sha, write, normalized_geometry, ARCHIVE, ARCHIVE_SHA
from tools.prepare_multitask_learning import exposure_audit, obj_check

TRAIN_COUNTS = (5, 7, 9, 12, 15)
EVAL_BANDS = ((10, 12), (13, 15))
DEVELOPMENT_HOLDOUT = '00335'
SALT = 'sequence-native-learning-v1'
KNOWN_EXPOSED = ('00161', '00335', '00365', '00561', '07009', '02628',
                 '01289', '01256', '02680', '02656', '02754', '07544', '00000')
OBSERVED_RUN = ROOT/'runs/sequence_native_00335_live_20261110/analysis.json'
PRICING = {
    'gpt-6.1-sol': dict(input_usd_per_million='2', output_usd_per_million='10',
        source='https://developers.openai.com/api/docs/models/gpt-6.1-sol'),
    'gpt-6-luna': dict(input_usd_per_million='0.10', output_usd_per_million='0.50',
        source='https://developers.openai.com/api/docs/models/gpt-6-luna'),
}


def archive_inventory(bundle):
    grouped, names = {}, set()
    for item in bundle.infolist():
        path = PurePosixPath(item.filename)
        if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                or ':' in item.filename or stat.S_ISLNK(item.external_attr >> 16)
                or item.filename in names):
            raise ValueError('unsafe_or_duplicate_archive_member')
        names.add(item.filename)
        if item.is_dir():
            continue
        if len(path.parts) != 3 or path.parts[0] != 'test_assembly' or not re.fullmatch(r'\d{5}', path.parts[1]):
            raise ValueError('unexpected_official_archive_layout')
        if path.suffix == '.obj':
            grouped.setdefault(path.parts[1], []).append(item)
    return [dict(id=ident, archive_dir='test_assembly/'+ident, parts=len(items),
                 obj_bytes=sum(item.file_size for item in items)) for ident, items in sorted(grouped.items())]


def overlap(left, right):
    denominator = min(sum(left.values()), sum(right.values()))
    return sum((left & right).values()) / denominator if denominator else 0.


def select_split(inventory, exposed, signature):
    """Read inventory/geometry only; no episode outcomes enter selection."""
    index = {row['id']: row for row in inventory}
    if len(index) != len(inventory) or DEVELOPMENT_HOLDOUT not in index:
        raise ValueError('invalid_or_incomplete_inventory')
    training, evaluation, excluded, selected_signatures = [], [], [], {}
    for count in TRAIN_COUNTS:
        for row in sorted((r for r in inventory if r['parts'] == count), key=lambda r:r['id']):
            ident = row['id']; sig = signature(ident)
            related = [i for i,s in selected_signatures.items() if overlap(sig,s) >= .5]
            if related:
                excluded.append(dict(id=ident, phase='training', reason='half_shared_normalized_vertices', related=related))
                continue
            training.append(dict(deepcopy(row), role='training', prior_exposed=ident in exposed))
            selected_signatures[ident] = sig
            break
        else:
            raise ValueError('no_distinct_training_candidate_for_part_count:'+str(count))
    known = {i:signature(i) for i in sorted(set(exposed)|set(selected_signatures)|{DEVELOPMENT_HOLDOUT}) if i in index}
    for lo, hi in EVAL_BANDS:
        eligible = (r for r in inventory if lo <= r['parts'] <= hi and r['id'] not in known)
        candidates = sorted(eligible, key=lambda r:hashlib.sha256((SALT+'|'+r['id']).encode()).hexdigest())
        for row in candidates:
            ident = row['id']; sig = signature(ident)
            related = [i for i,s in known.items() if overlap(sig,s) >= .5]
            if related:
                excluded.append(dict(id=ident, phase='evaluation', reason='half_shared_normalized_vertices', related=related))
                continue
            evaluation.append(dict(deepcopy(row), role='future_unexposed_evaluation_metadata_only',
                prior_exposed=False, band=[lo,hi], status='NOT_RUN'))
            known[ident] = sig
            break
        else:
            raise ValueError('no_unexposed_nonduplicate_evaluation_candidate')
    return dict(training=training, evaluation=evaluation, exclusions=excluded,
        development_holdout=dict(deepcopy(index[DEVELOPMENT_HOLDOUT]), role='development_holdout',
            prior_exposed=True, pristine=False, reason='previous native GNN and live LLM experiments'),
        source_split='New within-official-test-archive training/evaluation split; not official GNN train split',
        solver_outcomes_used_for_selection=False,
        duplicate_check='Translation/uniform-scale invariant part vertices; >=50% overlap excluded. Rotation/topology/semantic identity is not certified.',
        evaluation_novelty='No exposure found in tracked repository audit; not a guarantee about external/model training data.')


def reservation(model, body_bytes, output_tokens):
    rate = PRICING[model]
    return (Decimal(body_bytes+2048)*Decimal(rate['input_usd_per_million'])
            + Decimal(output_tokens)*Decimal(rate['output_usd_per_million'])) / Decimal(1000000)


def budget_proposal(observed):
    sol = reservation('gpt-6.1-sol',65536,2048)
    luna = reservation('gpt-6-luna',131072,4096)
    api = observed['api']; usage = api['usage']; n = api['actual_attempts']
    observed_cost = (Decimal(usage['input_tokens'])*2+Decimal(usage['output_tokens'])*10)/Decimal(1000000)
    if type(n) is not int or n < 1:
        raise ValueError('invalid_saved_usage_count')
    return dict(status='PROPOSED_NOT_AUTHORIZED_NOT_RUN', live_enabled=False,
        endpoint='https://api.openai.com/v1/responses', pricing=PRICING, pricing_checked_date='2026-11-10',
        automatic_http_retries=0, max_calls=270, max_program_usd='40.50',
        sol=dict(model='gpt-6.1-sol', episodes=10, max_calls_each=25, max_calls=250,
            max_usd_each='4.00', max_usd='40.00', max_body_bytes=65536,
            max_output_tokens=2048, request_overhead_tokens=2048,
            reservation_each=str(sol), reservation_total=str(sol*250)),
        luna=dict(model='gpt-6-luna', products=5, max_calls_each=4, max_calls=20,
            max_usd_each='0.10', max_usd='0.50', initial_lesson=1,
            candidate_plus_feedback_revision_max=2, postvalidation_lesson=1,
            max_body_bytes=131072, max_output_tokens=4096, request_overhead_tokens=2048,
            reservation_each=str(luna), reservation_total=str(luna*20)),
        worst_configured_reservation_usd=str(sol*250+luna*20),
        usage_based_reference=dict(source_run='sequence_native_00335_live_20261110',
            saved_requests=n, saved_input_tokens=usage['input_tokens'],saved_output_tokens=usage['output_tokens'],
            saved_usage_cost_at_current_rates_usd=str(observed_cost),
            same_mean_250_sol_calls_usd=str(observed_cost/Decimal(n)*250),
            interpretation='An arithmetic reference from one 16-part run, not a forecast or spending guarantee.'),
        physics=dict(free_episodes=10, timeout_each=3600, outer_timeout_each=3900,
            composite_limit_each=256,path_limit_each=256,stability_limit_each=256,
            candidate_admission_cases_max=20, admission_timeout_each=600, admission_outer_timeout_each=660,
            admission_composite_limit_each=128,admission_path_limit_each=128,admission_stability_limit_each=128,
            max_total_path_calls=5120,max_total_stability_calls=5120,max_total_composite_attempts=5120,
            max_worker_seconds=48000,max_outer_seconds=52200,
            scope='Part-level ASAP/RedMax only; proposed count limits are independent per checker.'),
        time_expectation=dict(hours_range=[5,8],basis='Planning estimate including new mesh setup and skill validation; not measured.',
            strict_outer_hours=14.5, possible_early_stop=True),
        cache_discounts_assumed=False, tool_fees_assumed=False,
        actual_invoice_guaranteed=False, success_or_registration_guaranteed=False,
        earlier_approval_residual_reused=False)


def check_graph(payload, part_ids):
    graph = json.loads(payload)
    ids = [str(n['id']) for n in graph['nodes']]
    if len(set(ids)) != len(ids) or set(ids) != set(part_ids):
        raise ValueError('contact_graph_part_mismatch')
    for edge in graph['links']:
        if str(edge['source']) not in ids or str(edge['target']) not in ids:
            raise ValueError('contact_graph_unknown_endpoint')
    return dict(nodes=len(ids),links=len(graph['links']),part_ids_match=True)


def complete_large_exposure_audit(audit):
    """Read bounded reports omitted by the older 32 MiB metadata scanner."""
    result=deepcopy(audit); resolved=[]
    def visit(value, reference):
        if type(value) is dict:
            for key,item in value.items():
                if key in ('assembly_id','assembly','task_id','source_task','experience_task','selected_id') and type(item) is str and re.fullmatch(r'\d{5}',item):
                    result['exposed'].setdefault(item,[]).append(reference)
                if key in ('assembly_dir','assembly_path','report','experience_report') and type(item) is str:
                    for ident in re.findall(r'(?:[/\\])(\d{5})(?=[/\\]|$)',item):
                        result['exposed'].setdefault(ident,[]).append(reference)
                visit(item,reference)
        elif type(value) is list:
            for item in value:visit(item,reference)
    for relative in audit['unreadable_or_oversize']:
        path=safe(ROOT/relative)
        if path.stat().st_size>128*1024*1024:
            raise ValueError('exposure_report_exceeds_supplemental_bound')
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        visit(value,relative)
        resolved.append(dict(path=relative,bytes=path.stat().st_size,sha256=sha(path)))
    result['supplemental_reports']=resolved
    result['original_skipped_reports']=list(audit['unreadable_or_oversize'])
    result['unreadable_or_oversize']=[]
    result['exposed']={k:sorted(set(v)) for k,v in result['exposed'].items()}
    return result


def prepare(output):
    output = safe(output)
    if output.exists():
        raise FileExistsError('new_sequence_preparation_directory_required')
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('official_archive_sha256_mismatch')
    output.mkdir(parents=True)
    write(output/'selection_policy.json',dict(training_part_counts=TRAIN_COUNTS,
        training_sort='lexicographic ID; skip >=50% normalized-vertex duplicate of earlier training selection',
        evaluation_bands=EVAL_BANDS,evaluation_sort='SHA256('+SALT+'|id)',
        training_outcome_selection=False,invalid_input_policy='stop; no success-based replacement',
        API_calls=0,physics_calls=0,archive_sha256=ARCHIVE_SHA))
    audit = complete_large_exposure_audit(exposure_audit(output.parent))
    for ident in KNOWN_EXPOSED:
        audit['exposed'].setdefault(ident,[]).append('explicit_project_history')
    write(output/'exposure_audit.json',audit)
    if audit['unreadable_or_oversize']:
        raise ValueError('incomplete_exposure_audit_review_required')
    with zipfile.ZipFile(ARCHIVE) as bundle:
        inventory = archive_inventory(bundle)
        stored = json.loads(safe(ROOT/'runs/assembly_rules_20261017/dataset_inventory.json').read_text(encoding='utf8'))
        if inventory != stored:
            raise ValueError('official_inventory_changed')
        signatures = {}
        def signature(ident):
            if ident not in signatures:
                prefix='test_assembly/'+ident+'/'
                signatures[ident]=Counter(normalized_geometry(bundle.read(name)) for name in sorted(bundle.namelist())
                    if name.startswith(prefix) and name.endswith('.obj'))
            return signatures[ident]
        split = select_split(inventory,audit['exposed'],signature)
        split.update(schema_version=1,archive_sha256=ARCHIVE_SHA,source_archive=str(ARCHIVE),
            archive_source='https://people.csail.mit.edu/yunsheng/ASAP/dataset_2404/test_assembly.zip')
        write(output/'split.json',split)
        selected=[]; manifest={}
        for row in split['training']:
            target=output/'raw_assemblies'/row['id'];target.mkdir(parents=True)
            names=sorted(n for n in bundle.namelist() if n.startswith(row['archive_dir']+'/') and not n.endswith('/'))
            part_ids=[PurePosixPath(n).stem for n in names if n.endswith('.obj')]
            checks,files={},{}
            for name in names:
                leaf=PurePosixPath(name).name
                if leaf!='contact_graph.json' and not leaf.endswith('.obj'):
                    raise ValueError('unexpected_assembly_data')
                data=bundle.read(name)
                checks[leaf]=obj_check(data) if leaf.endswith('.obj') else check_graph(data,part_ids)
                with (target/leaf).open('xb') as stream:stream.write(data)
                files[leaf]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            if len(part_ids)!=row['parts'] or 'contact_graph.json' not in files:
                raise ValueError('prepared_input_count_or_contact_graph_missing')
            selected.append(dict(row,part_ids=part_ids,files=files,checks=checks))
            manifest[row['id']]=dict(raw_dir=str(target),files=files,
                normalized_vertex_multiset=dict(signature(row['id'])),status='RAW_DATA_VALIDATED_PHYSICS_NOT_RUN')
        write(output/'selected_assemblies.json',selected)
        tasks=[dict(assembly_id=row['id'],part_count=row['parts'],
            assembly_dir='/mnt/c/'+(output/'dense_assemblies'/row['id']).as_posix()[3:],
            raw_dir=manifest[row['id']]['raw_dir']) for row in split['training']]
        write(output/'manifest.json',dict(schema_version=1,archive_sha256=ARCHIVE_SHA,assemblies=manifest,tasks=tasks,
            evaluation_task_ids=[row['id'] for row in split['evaluation']],development_holdout_ids=[DEVELOPMENT_HOLDOUT],
            source_commit='3727e716b57e0b508c4e9ecd9fefa6953e8f72da',physics_preparation='NOT_RUN',
            required_before_live='Pinned official subdivision max_edge=0.5, mesh checks, SDF preparation, request-size preflight.',
            no_new_rescale=True,raw_coordinates_preserved=True))
    schedule=[]
    for i,row in enumerate(split['training']):
        for repeat in (1,2):
            schedule.append(dict(index=len(schedule)+1,product_index=i+1,assembly_id=row['id'],parts=row['parts'],
                repetition=repeat,seed=repeat-1,status='NOT_RUN',
                raw_dir=manifest[row['id']]['raw_dir'],
                skill_learning_after_episode=repeat==1,
                freeze_before_episode=True,
                prior_training_memory_allowed=True,same_product_previous_episode_allowed=repeat==2,
                purpose='experience_collection' if repeat==1 else 'free_reuse_after_provisional_admission'))
    write(output/'schedule.json',dict(schema_version=1,profile='sequence_native_v1',episodes=schedule,
        episode_order='Per product: episode1, lesson/candidate/admission/postlesson, freeze, episode2.',
        model_weights_updated=False,learning_is_external_memory_and_python_skill=True,
        evaluation_included=False,development_holdout=DEVELOPMENT_HOLDOUT,
        prior_knowledge_default='empty sequence-specific training library; no evaluation/old explicit-action records',
        candidate_failure_policy='Preserve failure; no forced registration or forced skill selection.',
        automatic_http_retries=0,live_enabled=False))
    observed=json.loads(safe(OBSERVED_RUN).read_text(encoding='utf8'))
    write(output/'api_proposal.json',budget_proposal(observed))
    write(output/'preparation_result.json',dict(status='PREPARED_OFFLINE',training_ids=[r['id'] for r in split['training']],
        future_evaluation_ids=[r['id'] for r in split['evaluation']],development_holdout=DEVELOPMENT_HOLDOUT,
        API_calls=0,physics_calls=0,model_usage='NOT_RUN',mesh_subdivision='NOT_RUN',
        raw_training_parts=sum(r['parts'] for r in split['training']),archive_hash_verified=True))
    return split


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    args=parser.parse_args()
    result=prepare(args.run_dir)
    print(json.dumps(dict(training=[r['id'] for r in result['training']],
        evaluation=[r['id'] for r in result['evaluation']],API_calls=0,physics_calls=0)))
