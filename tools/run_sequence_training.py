"""Sequence-only cumulative training. Preparation is the default; no implicit API.

Five selected tasks, two freely chosen episodes each. Between the two episodes:
grounded lesson -> at most two candidate versions -> two native validation
conditions -> revised lesson -> provisional registration -> frozen reuse.
Every failed candidate and run is retained. No automatic HTTP or episode retry.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
from run_luna_experiment import safe, read, write, sha
import run_sequence_native_experiment as native
from orchestrator.llm_ranker import DEFAULTS
from orchestrator.sequence_ranker import SequenceRankService
from orchestrator.sequence_training_cycle import run_training_cycle
from orchestrator.sequence_learning_runtime import CandidateValidationService
from orchestrator.batch_budget import budget_summary
from worker.task_memory import digest


def configs():
    sol = dict(DEFAULTS, enabled=True, model='gpt-6.1-sol', max_calls=25, max_usd=4,
               input_usd_per_million=2, output_usd_per_million=10,
               max_output_tokens=2048, reasoning_effort='low')
    luna = dict(DEFAULTS, enabled=True, model='gpt-6-luna', max_calls=4, max_usd=.10,
                input_usd_per_million=.10, output_usd_per_million=.50,
                max_output_tokens=4096, reasoning_effort='low')
    return sol, luna


def task_label(identity):
    # A task identity such as beam_assembly/original is data, never a path.
    return identity.replace('/','__').replace('\\','__').replace(':','_')


def validate_manifest(value):
    tasks = value.get('tasks')
    if type(tasks) is not list or len(tasks) != 5:
        raise ValueError('exactly_five_training_tasks_required')
    held = value.get('evaluation_task_ids', []) + value.get('development_holdout_ids', ['00335'])
    ids = [t['assembly_id'] for t in tasks]
    if len(set(ids)) != 5 or set(ids).intersection(held):
        raise ValueError('training_evaluation_overlap')
    for task in tasks:
        if type(task['part_count']) is not int or not 3 <= task['part_count'] <= 30:
            raise ValueError('invalid_training_part_count')
        path = task['assembly_dir'].replace('\\', '/')
        if '..' in path.split('/') or not path.startswith('/mnt/c/dev/disassembly_codex_ready_v2/runs/'):
            raise ValueError('training_input_outside_allowed_c_path')
    return deepcopy(tasks), sorted(set(held))


def preflight_inputs(manifest):
    """Pure local input gate, shared by prepare and live before key access."""
    tasks, held = validate_manifest(manifest)
    if manifest.get('mesh_preparation') != 'PASS':
        raise ValueError('prepared_mesh_manifest_required')
    verified = {}
    for task in tasks:
        local = ROOT/task['assembly_dir'].split('/mnt/c/dev/disassembly_codex_ready_v2/')[1]
        safe(local, True)
        pins = task.get('input_sha256')
        if type(pins) is not dict or not pins:
            raise ValueError('training_input_pins_required')
        if sum(name.endswith('.obj') for name in pins) != task['part_count']:
            raise ValueError('training_part_count_mismatch')
        for relative, expected in pins.items():
            if sha(safe(local/relative,True)) != expected:
                raise ValueError('training_input_hash_changed')
        verified[task['assembly_id']] = digest(pins)
    return dict(status='PASS',input_sha256=verified,held_out_ids=held,API_calls=0,physics_calls=0)


class NativeTrainingBackend:
    def __init__(self, output, held_out_ids, *, episode_seconds=3600, validation_seconds=600,
                 live=True, learner_transport=None, episode_runner=None, allowed_task_ids=None):
        self.output = safe(output, True)
        self.held_out_ids = held_out_ids
        self.allowed_task_ids = set(allowed_task_ids) if allowed_task_ids is not None else None
        self.policy = dict(exclude_current_task=False, excluded_task_ids=held_out_ids)
        self.snapshots, self.lessons, self.skills = [], [], []
        self.halted = None
        self.live = live
        if not live and (learner_transport is None or episode_runner is None):
            raise ValueError('offline_backend_requires_explicit_injected_services')
        self.learner_transport = learner_transport
        self.episode_runner = episode_runner or native.execute
        self.episode_seconds, self.validation_seconds = episode_seconds, validation_seconds
        self.sol_config, self.luna_config = configs()
        self.learners, self.runtime_contexts = {}, {}
        self.last_episode = {}
        self.stage_counter = 0
        self.source_pins = native.source_hashes()
        write(self.output/'source_pins.json', self.source_pins)

    def unchanged(self):
        if native.source_hashes() != self.source_pins:
            raise ValueError('training_source_changed')

    def event(self, kind, **fields):
        self.unchanged()
        with (self.output/'events.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(kind=kind, **fields), ensure_ascii=False) + '\n')
        print(json.dumps(dict(event=kind, task=fields.get('task')), ensure_ascii=False), flush=True)

    def freeze(self, name):
        from orchestrator.sequence_learning import freeze_sequence_lessons, freeze_sequence_skills
        folder = self.output/'frozen'/name
        folder.mkdir(parents=True, exist_ok=False)
        write(folder/'memory.json', self.snapshots)
        write(folder/'lessons.json', freeze_sequence_lessons(self.lessons))
        write(folder/'skills.json', freeze_sequence_skills(self.skills))
        files = {label: str(folder/(label+'.json')) for label in ('memory','lessons','skills')}
        write(folder/'pins.json', {path:sha(Path(path)) for path in files.values()})
        return files

    def _args(self, output, task, seed, frozen, *, validation=False):
        seconds = self.validation_seconds if validation else self.episode_seconds
        argv = ['--run-dir', str(output), '--assembly-dir', task['assembly_dir'],
                '--mode', 'fixture' if validation else 'llm', '--seed', str(seed),
                '--budget', '128' if validation else '256',
                '--path-limit', '128' if validation else '256',
                '--stability-limit', '128' if validation else '256',
                '--timeout', str(seconds), '--outer-timeout', str(seconds+(60 if validation else 300)),
                '--max-gripper','3','--max-pose','128']
        for label,path in frozen.items():
            argv += ['--'+label, path]
        if task.get('base_part') is not None:
            argv += ['--base-part', task['base_part']]
        if not validation:
            config_path = self.output/('sol_'+output.name+'.json')
            write(config_path, self.sol_config)
            argv += ['--llm-config', str(config_path), '--allow-live-api']
        return native.parser().parse_args(argv)

    def episode(self, task, seed, frozen):
        self.unchanged()
        path = self.output/('%s_round%d' % (task_label(task['assembly_id']), seed+1))
        code = self.episode_runner(self._args(path, task, seed, frozen), knowledge_limits=self.policy)
        self.last_episode[task['assembly_id']] = path
        if not (path/'metrics.json').is_file():
            self.halted = 'episode_result_missing'
            return dict(path=str(path), status='ERROR', reuse_status='NOT_RUN')
        metrics = read(path/'metrics.json')
        if not self.host_accepted(path, code):
            self.halted = 'episode_host_error'
        if (path/'api_ledger.sqlite').is_file():
            write(path/'batch_accounting.json', budget_summary(path/'api_ledger.sqlite'))
        # Known budget exhaustion is recorded and the batch can continue. Any
        # uncertain network/contract/source failure stops all subsequent calls.
        reasons = [read(p).get('fallback_reason') for p in (path/'episode/ipc/responses').glob('*.json')]
        if any(r and r not in ('call_budget_exhausted','cost_budget_exhausted','budget_exhausted') for r in reasons):
            self.halted = 'episode_decision_error'
        if metrics.get('worker_status') == 'ERROR':
            self.halted = 'episode_worker_error'
        count = metrics.get('sequence_skill_executions', 0)
        return dict(path=str(path), status=metrics.get('worker_status'), metrics=metrics,
                    expected_task_id=task['assembly_id'],
                    reuse_status='EXECUTED_EFFECT_UNVERIFIED' if count else 'NOT_SELECTED')

    @staticmethod
    def host_accepted(path, code):
        if code != 0 or not (path/'result.json').is_file():
            return False
        host = read(path/'result.json')
        metrics = host.get('metrics', {})
        return (host.get('status') in ('PASS','UNKNOWN')
            and metrics.get('process_exit') == 0 and metrics.get('service_error') is None
            and metrics.get('sources_and_frozen_inputs_preserved') is True)

    def collect(self, episode):
        path = safe(Path(episode['path'])/'native_observations.json', True)
        if path.is_file():
            from orchestrator.sequence_knowledge import validate_native_snapshot
            snapshot = validate_native_snapshot(read(path))
            actual = {r['source_task_context'].get('task',r['source_task_context']).get('assembly_id')
                      for r in snapshot['records']}
            if (actual.intersection(self.held_out_ids) or None in actual
                    or (self.allowed_task_ids is not None and not actual.issubset(self.allowed_task_ids))
                    or (episode.get('expected_task_id') is not None and actual - {episode['expected_task_id']})):
                self.halted = 'training_task_identity_mismatch'
                raise ValueError(self.halted)
            self.snapshots.append(snapshot)

    def packet(self, task, episode):
        from orchestrator.sequence_learning import build_sequence_learning_packet
        if not any(s['record_count'] for s in self.snapshots):
            return None
        packet = build_sequence_learning_packet(self.snapshots)
        write(self.output/(task_label(task['assembly_id'])+'_learning_packet.json'), packet)
        return packet

    def _learn(self, task, stage, packet, lesson=None, previous=None, feedback=None):
        from orchestrator.sequence_learning_service import SequenceLearningService, make_learning_request
        self.unchanged()
        identity = task['assembly_id']
        if identity not in self.learners:
            self.learners[identity] = SequenceLearningService(self.luna_config,
                self.output/(task_label(identity)+'_luna_ledger.sqlite'), transport=self.learner_transport)
        self.stage_counter += 1
        rid = self.output.name + ':learning:' + str(self.stage_counter)
        data = dict(packet=packet, lesson=lesson, runtime_contexts=self.runtime_contexts.get(identity, []),
                    previous_candidate=previous, validation_feedback=feedback)
        request = make_learning_request(self.output.name, rid, stage, data)
        prefix = '%s_%02d_%s' % (task_label(identity), self.stage_counter, stage)
        write(self.output/(prefix+'_request.json'), request)
        result = self.learners[identity].learn(request)
        write(self.output/(prefix+'_result.json'), result)
        if result.get('halt_required'):
            self.halted = result.get('error') or 'learning_transport_error'
        return result

    def lesson(self, task, packet, previous_lesson=None, validation=None, candidate=None):
        stage = 'lesson_initial' if previous_lesson is None else 'lesson_postvalidation'
        if validation is not None:
            from orchestrator.sequence_learning import build_sequence_learning_packet
            # Only this candidate's new validation observations form the post
            # packet; old facts remain in the candidate's original lesson.
            snapshots = []
            for ref in validation['native_snapshot_refs']:
                path = safe(Path(ref['path']), True)
                if sha(path) != ref['sha256']:
                    raise ValueError('validation_observation_hash_changed')
                snapshots.append(read(path))
            packet = build_sequence_learning_packet(snapshots)
        result = self._learn(task, stage, packet, previous_lesson, candidate, validation)
        if result.get('ok') and validation is not None:
            from orchestrator.sequence_learning import validate_sequence_lesson
            cited = set(result['lesson']['claim']['evidence_ids'])
            fresh = {f['evidence_id'] for f in packet['facts']}
            if not cited.intersection(fresh):
                result.update(ok=False, error='postlesson_missing_new_validation_evidence')
            else:
                provenance = dict(result['lesson']['provenance'],
                    validated_program_sha256=candidate['program_sha256'],
                    validated_candidate_sha256=candidate['content_sha256'],
                    validation_proof_sha256=digest(validation['proof']),
                    prior_lesson_sha256=previous_lesson['content_sha256'])
                result['lesson'] = validate_sequence_lesson(packet,result['lesson']['claim'],provenance)
            write(self.output/(task_label(task['assembly_id'])+'_postlesson_bound_'+candidate['content_sha256'][:16]+'.json'),result)
        return result

    def candidate(self, task, packet, lesson, previous, feedback, revision):
        # The same prepare() and exact VM context are used in generation,
        # candidate validation and later free-policy execution.
        from orchestrator.sequence_learning import freeze_sequence_lessons
        svc = SequenceRankService(dict(DEFAULTS), self.output/(task_label(task['assembly_id'])+'_context.sqlite'),
            memory_records=self.snapshots, lessons=freeze_sequence_lessons(self.lessons+[lesson]),
            skills=self.skills, knowledge_limits=self.policy,
            allow_fixture_skills=not self.live,
            transport=(lambda *a: (_ for _ in ()).throw(RuntimeError('offline_network_forbidden'))) if not self.live else None)
        paths = list((self.last_episode[task['assembly_id']]/'episode/ipc/requests').glob('*.json'))
        paths.sort(key=lambda p: (-len(read(p)['request']['assembled_parts']),read(p)['request']['request_id']))
        contexts = []
        for path in paths[:1]:
            envelope = read(path)
            contexts.append(svc.prepare(envelope)['features']['sequence_knowledge'])
        self.runtime_contexts[task['assembly_id']] = contexts
        return self._learn(task, 'python_generate' if revision == 0 else 'python_revision',
                           packet, lesson, previous, feedback)

    def validate_candidate(self, task, candidate, lesson):
        from orchestrator.sequence_learning import freeze_sequence_lessons, validate_sequence_registration
        root = self.output/('validation_'+candidate['content_sha256'][:16])
        root.mkdir(exist_ok=False)
        frozen = self.freeze('validation_'+candidate['content_sha256'][:16])
        lesson_path = root/'lessons.json'
        write(lesson_path, freeze_sequence_lessons(self.lessons+[lesson]))
        frozen['lessons'] = str(lesson_path)
        cases, feedback, validation_snapshots = [], [], []
        for seed in (0,1):
            run = root/('seed%d' % seed)
            holder = []
            def factory(service):
                policy = CandidateValidationService(service, candidate)
                holder.append(policy)
                return policy
            code = self.episode_runner(self._args(run, task, seed, frozen, validation=True),
                                service_factory=factory, knowledge_limits=self.policy,
                                allow_fixture_knowledge=not self.live)
            worker_path = run/'episode/worker_result.json'
            if not worker_path.is_file():
                feedback.append(dict(seed=seed, status='ERROR', reason='missing_worker_result'))
                break
            worker = read(worker_path)
            # Validation success AND failure are observations. They are kept in
            # training knowledge and are never mixed into the held-out tasks.
            self.collect(dict(path=str(run),expected_task_id=task['assembly_id']))
            if (run/'native_observations.json').is_file():
                snapshot_path = run/'native_observations.json'
                validation_snapshots.append(dict(path=str(snapshot_path),sha256=sha(snapshot_path)))
            if not self.host_accepted(run, code):
                self.halted = 'validation_host_error'
                feedback.append(dict(seed=seed,status='ERROR',reason=self.halted))
                break
            trace = holder[0].trace(sorted((run/'episode/ipc/responses').glob('*.json')), read, sha)
            write(run/'candidate_trace.json', trace)
            cases.append(dict(case_id='seed%d' % seed,
                worker_result=dict(path=str(worker_path),sha256=sha(worker_path)),
                decision_trace=dict(path=str(run/'candidate_trace.json'),sha256=sha(run/'candidate_trace.json'))))
            feedback.append(dict(seed=seed, status=worker['status'], success=worker['success'],
                connected_depth=worker['connected']['max_depth'], stop_reason=worker['stop_reason'],
                physics_counts=worker['budget']['counts'],
                observations=[dict(status=a['status'], part_move=a['part_move'],
                    remaining_parts=a['remaining_parts'], raw_sim_info=a.get('raw_sim_info'),
                    accepted_for_transition=a['accepted_for_transition'], deadline_exceeded=a['deadline_exceeded'])
                    for a in worker['attempts'][-6:]]))
            if not worker['success']:
                break
        proof = dict(protocol='sequence_validation_proof_v1', candidate_sha256=candidate['content_sha256'],
                     program_sha256=candidate['program_sha256'], cases=cases)
        write(root/'proof.json', proof)
        error = None
        try:
            validate_sequence_registration(candidate, proof, allow_fixture=not self.live)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        result = dict(ok=error is None and not self.halted, proof=proof, feedback=feedback, error=error,
                      native_snapshot_refs=validation_snapshots,
                      physics='ACTUAL' if cases else 'NOT_RUN_OR_NO_COMPLETE_PROOF', performance_effect='UNVERIFIED')
        write(root/'validation.json', result)
        return result

    def register(self, candidate, lesson, validation):
        from orchestrator.sequence_learning import validate_sequence_registration
        if lesson['provenance'].get('validated_program_sha256') != candidate['program_sha256'] or self.halted:
            raise ValueError('postlesson_or_host_registration_gate')
        admitted = validate_sequence_registration(candidate, validation['proof'], allow_fixture=not self.live)
        card = admitted['card']
        # Keep each proof even if the same Python source was learned before.
        # Existing card stays immutable; duplicates do not inflate library size.
        duplicate = any(old['skill_id'] == card['skill_id'] for old in self.skills)
        if not duplicate:
            self.skills.append(card)
        if not any(old['content_sha256'] == lesson['content_sha256'] for old in self.lessons):
            self.lessons.append(lesson)
        write(self.output/('admitted_'+candidate['content_sha256']+'.json'),
              dict(**admitted, library_duplicate=duplicate, postvalidation_lesson_sha256=lesson['content_sha256']))
        return card


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--allow-live-api', action='store_true')
    p.add_argument('--use-registered-key', action='store_true')
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--validation-seconds', type=int, default=600)
    args = p.parse_args()
    if args.use_registered_key and not args.allow_live_api:
        raise ValueError('offline_cannot_read_key')
    if not 1 <= args.episode_seconds <= 3600 or not 1 <= args.validation_seconds <= 600:
        raise ValueError('time_budget_exceeds_profile')
    manifest_path = safe(args.manifest, True)
    manifest = read(manifest_path)
    tasks, held = validate_manifest(manifest)
    inputs = preflight_inputs(manifest)
    out = safe(args.run_dir, True); out.mkdir(parents=True, exist_ok=False)
    sol, luna = configs()
    write(out/'plan.json', dict(manifest_sha256=sha(manifest_path), tasks=tasks, held_out_ids=held,
        API_call_cap=270, configured_USD_cap=40.5, sol=sol,luna=luna,
        API_status='AUTHORIZED_FLAG' if args.allow_live_api else 'NOT_RUN',
        automatic_retries=0, source_profile='sequence_native_v1',
        validation_additional_runs_max=20, episodes=10))
    write(out/'input_preflight.json',inputs)
    if not args.allow_live_api:
        print('PREPARED_ONLY: API=0 physics=0. Live execution requires separate authorization.')
        return 0
    ctx = nullcontext()
    if args.use_registered_key:
        from local_api_key import key_environment
        ctx = key_environment()
    with ctx:
        backend = NativeTrainingBackend(out, held, episode_seconds=args.episode_seconds,
                                         validation_seconds=args.validation_seconds,
                                         allowed_task_ids=[t['assembly_id'] for t in tasks])
        try:
            result = run_training_cycle(tasks, backend, held_out_ids=held)
        except Exception as error:
            # Preserve completed artifacts and stop; never launch an automatic
            # retry from an uncertain stage or lose the failure report.
            result = dict(protocol='sequence_training_cycle_v1',stopped=True,
                          stop_reason=backend.halted or type(error).__name__,
                          error_type=type(error).__name__,performance_effect='UNVERIFIED')
        accounting = []
        for path in sorted(out.glob('*_luna_ledger.sqlite')) + sorted(out.glob('*_round*/api_ledger.sqlite')):
            accounting.append(dict(path=str(path),summary=budget_summary(path)))
        write(out/'batch_accounting.json',dict(ledgers=accounting,
            configured_call_cap=270,configured_usd_cap=40.5,automatic_retries=0))
    write(out/'result.json', result)
    return 1 if result['stopped'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
