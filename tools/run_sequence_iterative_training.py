"""Up to five learning rounds, with a mandatory review after the first two.

The old two-episode runner and all previous results remain unchanged. Default
mode only prepares a plan and checks local inputs. Continuation is a new run,
using a verified checkpoint and a separate explicit live/review flag.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from run_luna_experiment import safe,read,write,sha
from run_sequence_training import NativeTrainingBackend,configs,validate_manifest,preflight_inputs,task_label
from run_sequence_native_experiment import source_hashes
from orchestrator.sequence_iterative_training import run_iterative_training
from orchestrator.sequence_iterative_budget import stage_budget
from orchestrator.sequence_learning import (build_sequence_learning_packet,validate_sequence_learning_packet,
    check_sequence_lesson,freeze_sequence_lessons,freeze_sequence_skills,validate_sequence_candidate)
from orchestrator.sequence_learning_service import SequenceLearningService,MAX_BODY_BYTES
from orchestrator.sequence_knowledge import execute_sequence_skill,execute_sequence_program
from orchestrator.llm_ranker import _json
from orchestrator.batch_budget import budget_summary
from worker.task_memory import digest


def current_round_packet(snapshots, latest):
    """Give new evidence half of the slots; retain exact facts and exclusions."""
    fresh=build_sequence_learning_packet(latest,max_records=12)
    prior=[s for s in snapshots if s['content_sha256']!=latest['content_sha256']]
    prior_facts=build_sequence_learning_packet(prior,max_records=12)['facts'] if any(s['record_count'] for s in prior) else []
    facts=fresh['facts']+prior_facts
    selected={f['evidence_id'] for f in facts}
    all_ids=[r['evidence_id'] for s in snapshots for r in s['records']]
    if len(all_ids)!=len(set(all_ids)):raise ValueError('duplicate_iterative_evidence')
    excluded=[i for i in all_ids if i not in selected]
    packet=deepcopy(fresh)
    packet.update(facts=facts,source_snapshot_sha256=[s['content_sha256'] for s in snapshots],
        source_fact_count=len(all_ids),selected_fact_count=len(facts),excluded_evidence_ids=excluded[:64],
        excluded_evidence_count=len(excluded),excluded_evidence_sha256=digest(excluded),
        excluded_ids_are_prefix=len(excluded)>64,
        current_round_evidence_ids=[f['evidence_id'] for f in fresh['facts']])
    packet.pop('content_sha256',None);packet['content_sha256']=digest(packet)
    return validate_sequence_learning_packet(packet)


class IterativeLearningService(SequenceLearningService):
    def _body(self,envelope):
        body=super()._body(envelope)
        request=envelope['request']
        if request['stage']=='lesson_initial':
            ids=request['input']['packet'].get('current_round_evidence_ids',[])
            if not ids:raise ValueError('current_round_evidence_required')
            body['instructions']+=' This round must learn from at least one of these fresh observed evidence IDs in lesson.evidence_ids: '+_json(ids)+'. Older facts are references, not the result of this round.'
        if len(_json(body).encode('utf8'))>MAX_BODY_BYTES:
            raise ValueError('sequence_learning_body_too_large')
        return body

    def _parse_response(self,raw,request,metadata):
        result,error=super()._parse_response(raw,request,metadata)
        if error is None and request['stage']=='lesson_initial':
            required=set(request['input']['packet'].get('current_round_evidence_ids',[]))
            if not required.intersection(result['lesson']['claim']['evidence_ids']):
                metadata['rejected_lesson']=result['lesson']
                return None,'initial_lesson_missing_current_round_evidence'
        return result,error


def condition_key(record):
    """Diagnostic similarity only: never a physical cache or pruning rule.

Seed/time/observed result are deliberately excluded to count repeated broad
conditions across rounds. Physics still runs anew. This is not equivalence.
"""
    context=record['source_task_context'];task=context.get('task',context)
    attempt=record['attempt'];req=attempt['request']
    return digest(dict(task=task['assembly_id'],input_sha256=task['input_sha256'],
        remaining=sorted(attempt['remaining_parts']),part=attempt['part_move'],
        pose=req.get('pose'),max_grippers=req.get('max_grippers'),
        domain=attempt['verifier_domain'],physics_binding=attempt.get('source_hashes')))


class IterativeBackend:
    def __init__(self,output,tasks,held_out_ids,*,live=True,learner_transport=None,episode_runner=None,
                 snapshots=(),lessons=(),skills=(),episode_seconds=3600,validation_seconds=600):
        self.output=safe(output,True);self.tasks=deepcopy(tasks);self.held_out_ids=list(held_out_ids)
        self.live=live;self.learner_transport=learner_transport;self.episode_runner=episode_runner
        self.episode_seconds,self.validation_seconds=episode_seconds,validation_seconds
        self.snapshots,self.lessons,self.skills=deepcopy(list(snapshots)),deepcopy(list(lessons)),deepcopy(list(skills))
        self.child=None;self.round_index=None;self.active_task=None;self.latest=None
        self.initial_source=source_hashes()
        self.free_refs=[];self.duplicate_events=[];self.stage_halt=None
        write(self.output/'source_pins.json',self.initial_source)

    @property
    def halted(self):
        return self.stage_halt or (self.child.halted if self.child else None)

    def event(self,kind,**fields):
        if source_hashes()!=self.initial_source:raise ValueError('iterative_source_changed')
        with (self.output/'events.jsonl').open('a',encoding='utf8') as stream:
            stream.write(_json(dict(kind=kind,**fields))+'\n')
        print(_json(dict(event=kind,task=fields.get('task'),round_index=fields.get('round_index'))),flush=True)

    def begin_round(self,task,round_index):
        if self.halted:raise ValueError('halted_iterative_batch')
        folder=self.output/('round%02d_%s'%(round_index,task_label(task['assembly_id'])))
        folder.mkdir(exist_ok=False)
        self.child=NativeTrainingBackend(folder,self.held_out_ids,live=self.live,
            learner_transport=self.learner_transport,episode_runner=self.episode_runner,
            allowed_task_ids=[t['assembly_id'] for t in self.tasks],
            episode_seconds=self.episode_seconds,validation_seconds=self.validation_seconds)
        self.child.snapshots,self.child.lessons,self.child.skills=self.snapshots,self.lessons,self.skills
        self.round_index,self.active_task=round_index,task['assembly_id']
        self.latest=None;self.episode_ref=None;self.duplicate_events=[]
        self.updated_post_lesson=None
        self.before=dict(observations=sum(s['record_count'] for s in self.snapshots),
            conditions={condition_key(r) for s in self.snapshots for r in s['records']},
            skills=len(self.skills),lessons=len(self.lessons))
        self.child.learners[self.active_task]=IterativeLearningService(self.child.luna_config,
            folder/(task_label(self.active_task)+'_luna_ledger.sqlite'),transport=self.learner_transport)

    def freeze(self,name):
        folder=self.output/'frozen'/name;folder.mkdir(parents=True,exist_ok=False)
        write(folder/'memory.json',self.snapshots)
        write(folder/'lessons.json',freeze_sequence_lessons(self.lessons))
        write(folder/'skills.json',freeze_sequence_skills(self.skills))
        paths={key:str(folder/(key+'.json')) for key in ('memory','lessons','skills')}
        write(folder/'pins.json',{path:sha(Path(path)) for path in paths.values()})
        return paths

    def episode(self,task,seed,frozen):
        result=self.child.episode(task,seed,frozen)
        path=safe(Path(result['path'])/'result.json',True)
        if path.is_file():self.episode_ref=dict(path=str(path),sha256=sha(path))
        self.current_episode=result
        return result

    def collect(self,episode):
        self.child.collect(episode)
        path=safe(Path(episode['path'])/'native_observations.json',True)
        self.latest=read(path) if path.is_file() else None

    def packet(self,task,episode):
        if not self.latest or not self.latest['record_count']:return None
        packet=current_round_packet(self.snapshots,self.latest)
        write(self.child.output/'current_round_learning_packet.json',packet)
        return packet

    def lesson(self,*args,**kwargs):return self.child.lesson(*args,**kwargs)
    def candidate(self,*args,**kwargs):return self.child.candidate(*args,**kwargs)
    def validate_candidate(self,*args,**kwargs):return self.child.validate_candidate(*args,**kwargs)
    def register(self,candidate,lesson,validation):
        card=self.child.register(candidate,lesson,validation)
        # Keep all historical entries; the bounded active window starts with
        # recently validated knowledge instead of permanently favoring round 1.
        self.skills.sort(key=lambda item:item['skill_id']!=card['skill_id'])
        self.lessons.sort(key=lambda item:item['content_sha256']!=lesson['content_sha256'])
        self.updated_post_lesson=lesson['content_sha256']
        return card

    def duplicate_candidate(self,candidate):
        validate_sequence_candidate(candidate)
        exact=[s['skill_id'] for s in self.skills if s['program_sha256']==candidate['program_sha256']]
        # Equal outputs on these few probes are recorded, not treated as proof
        # that programs are equivalent. Only byte-identical programs are skipped.
        probes=[]
        for ctx in self.child.runtime_contexts.get(self.active_task,[]):
            proposed=execute_sequence_program(candidate['program'],ctx['candidate_ids'],ctx)
            if proposed['status']!='PROPOSED':continue
            for card in self.skills:
                result=execute_sequence_skill(card,ctx['candidate_ids'],ctx,allow_fixture=not self.live)
                if result['status']=='PROPOSED' and proposed['candidate_order']==result['candidate_order']:
                    probes.append(dict(skill_id=card['skill_id'],context_sha256=digest(ctx),same_order=True))
        result=dict(duplicate=bool(exact),matching_skill_ids=exact,skill_id=exact[0] if exact else None,order_probe_matches=probes,
            equivalence_proven=False,validation_scope_expanded=False,
            reason='identical_program_already_registered' if exact else 'new_program_candidate')
        self.duplicate_events.append(result)
        write(self.child.output/('duplicate_'+candidate['content_sha256']+'.json'),result)
        return result

    def retain_lesson(self,lesson):
        lesson=check_sequence_lesson(lesson)
        if not any(old['content_sha256']==lesson['content_sha256'] for old in self.lessons):
            self.lessons.insert(1 if self.updated_post_lesson else 0,deepcopy(lesson))
        # A fresh observation-based lesson is a hypothesis, even if code failed.
        write(self.child.output/'retained_initial_lesson.json',lesson)

    def round_summary(self,task,round_index):
        new=[r for s in self.snapshots for r in s['records']][self.before['observations']:]
        conditions={condition_key(r) for r in new}
        novelty=dict(new_record_count=len(new),free_observations=self.latest['record_count'] if self.latest else 0,
            new_condition_count=len(conditions-self.before['conditions']),
            repeated_condition_count=len(conditions&self.before['conditions']),
            new_skill_count=len(self.skills)-self.before['skills'],new_lessons=len(self.lessons)-self.before['lessons'],
            duplicate_programs=sum(e['duplicate'] for e in self.duplicate_events),
            order_probe_matches=sum(len(e['order_probe_matches']) for e in self.duplicate_events),
            signature_use='diagnostic_only_not_physics_equivalence',effect='UNVERIFIED')
        accounting=[]
        for path in sorted(self.child.output.glob('*_luna_ledger.sqlite'))+sorted(self.child.output.glob('*_round*/api_ledger.sqlite')):
            accounting.append(dict(path=str(path),summary=budget_summary(path)))
        write(self.child.output/'round_accounting.json',accounting)
        summary=dict(episode_ref=self.episode_ref,novelty=novelty,accounting=accounting)
        write(self.child.output/'round_summary.json',summary)
        return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--stage',choices=('initial','continuation'),default='initial')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--continue-after-review',action='store_true')
    p.add_argument('--allow-live-api',action='store_true')
    p.add_argument('--use-registered-key',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if args.use_registered_key and not args.allow_live_api:raise ValueError('offline_cannot_read_key')
    if args.stage=='initial' and (args.checkpoint or args.continue_after_review):
        raise ValueError('initial_cannot_import_or_approve_continuation')
    if args.stage=='continuation' and not args.checkpoint:
        raise ValueError('continuation_requires_checkpoint')
    manifest_path=safe(args.manifest,True);manifest=read(manifest_path)
    tasks,held=validate_manifest(manifest);inputs=preflight_inputs(manifest)
    pins=source_hashes()
    profiles={stage:stage_budget(stage) for stage in ('initial','continuation')}
    restored=None
    if args.checkpoint:
        from orchestrator.sequence_training_checkpoint import load_checkpoint
        restored=load_checkpoint(safe(args.checkpoint,True),manifest_path=manifest_path,
                                  source_pins=pins,budget_profiles=profiles)
    if args.allow_live_api and args.stage=='continuation' and not args.continue_after_review:
        raise ValueError('review_required_before_continuation')
    output=safe(args.run_dir,True);output.mkdir(parents=True,exist_ok=False)
    plan=dict(stage=args.stage,budget=profiles[args.stage],tasks=tasks,held_out_ids=held,
              manifest_sha256=sha(manifest_path),maximum_rounds=5,review_after_round=2,
              source_pins=pins,API_status='LIVE_EXPLICIT_FLAG' if args.allow_live_api else 'NOT_RUN')
    write(output/'plan.json',plan);write(output/'input_preflight.json',inputs)
    if not args.allow_live_api:
        print('PREPARED_ONLY: API=0 physics=0; first two rounds require review before continuation.')
        return 0
    context=nullcontext()
    if args.use_registered_key:
        from local_api_key import key_environment
        context=key_environment()
    with context:
        if not os.environ.get('OPENAI_API_KEY','').strip():
            raise ValueError('missing_api_key_before_claim_or_execution')
        if restored:
            from orchestrator.sequence_training_checkpoint import claim_continuation
            restored=claim_continuation(args.checkpoint,claim_root=ROOT/'runs/sequence_iterative_claims',
                new_run_dir=output,continue_after_review=args.continue_after_review,
                manifest_path=manifest_path,source_pins=pins,budget_profiles=profiles)
        backend=IterativeBackend(output,tasks,held,
            snapshots=restored['snapshots'] if restored else (),lessons=restored['lessons'] if restored else (),
            skills=restored['skills'] if restored else ())
        try:
            result=run_iterative_training(tasks,backend,start_round=3 if restored else 1,
                stop_round=5 if restored else 2,held_out_ids=held,
                prior_rounds=restored['checkpoint']['completed_rounds'] if restored else (),
                review_approved=bool(restored and args.continue_after_review))
        except Exception as error:
            result=dict(status='ERROR',stopped=True,error_type=type(error).__name__,stop_reason=backend.halted,
                        performance_effect='UNVERIFIED')
        write(output/'result.json',result)
        if result.get('status')=='REVIEW_REQUIRED':
            from orchestrator.sequence_training_checkpoint import make_checkpoint
            try:
                checkpoint=make_checkpoint(output/'review_checkpoint.json',manifest_path=manifest_path,
                    source_pins=pins,budget_profiles=profiles,frozen=result['frozen'],
                    completed_rounds=result['completed_rounds'],held_out_ids=held)
                write(output/'checkpoint_status.json',dict(status='READY_FOR_REVIEW',
                    checkpoint_sha256=sha(output/'review_checkpoint.json'),continuation_authorized=False))
            except Exception as error:
                write(output/'checkpoint_status.json',dict(status='CHECKPOINT_FAILED',
                    error_type=type(error).__name__,continuation_authorized=False))
                return 1
    return 1 if result.get('status') in ('ERROR','HALTED') else 0


if __name__=='__main__':raise SystemExit(main())
