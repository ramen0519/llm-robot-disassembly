"""Actual training backend with injected model/worker boundaries; NOT physics.

File IPC, provider parsing, native-observation storage, candidate validation
traces, registration checks and freezing are real code. Every synthetic worker
result has actual_physics=False and fixture=True. HTTP/WSL are forbidden.
"""
from copy import deepcopy
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import run_sequence_training as training
from orchestrator.file_service import FileRankServer, atomic_new
from orchestrator.llm_ranker import DEFAULTS
from orchestrator.sequence_ranker import SequenceRankService, FixtureSequenceService
from orchestrator.sequence_knowledge import build_native_snapshot
from orchestrator.sequence_training_cycle import run_training_cycle
from worker.independent_features import describe_loaded, filter_features
from worker.task_memory import task_context, digest


PROGRAM = "def propose(ctx):\n    return {'candidate_order': ctx['candidate_ids'][:], 'evidence_ids': []}\n"


class MeshFixture:
    bounds = [[0, 0, 0], [1, 1, 1]]
    center_mass = [.5, .5, .5]
    volume = 1
    is_watertight = True
    is_winding_consistent = True


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False)


class InjectedLearner:
    def __init__(self, *, reject_first_python=False, stale_post_citation=False):
        self.requests = []
        self.reject_first_python = reject_first_python
        self.stale_post_citation = stale_post_citation
        self.initial_evidence = None

    def __call__(self, url, headers, body, timeout):
        request = json.loads(body['input'][0]['content'])
        self.requests.append(request)
        stage, material = request['stage'], request['input']
        evidence = material['packet']['facts'][0]['evidence_id']
        if self.initial_evidence is None:
            self.initial_evidence = evidence
        value = dict(request_id=request['request_id'], state_id=request['state_id'])
        if stage.startswith('lesson_'):
            if stage == 'lesson_postvalidation' and self.stale_post_citation:
                evidence = self.initial_evidence
            value['lesson'] = dict(title='Backend integration fixture', hypothesis='Fixture only; not an effectiveness claim.',
                when_to_apply='Explicit offline wiring validation.', when_not_to_apply='Any real effectiveness conclusion.',
                ranking_guidance='Preserve complete current candidates.', evidence_ids=[evidence], counterexample_ids=[])
        else:
            source = 'import os\ndef propose(ctx):\n    return None\n' if self.reject_first_python and stage=='python_generate' else PROGRAM
            value['candidate'] = dict(name='Fixture complete order', description='Copies current order only for integration testing.',
                source=source, evidence_ids=[evidence])
        return dict(status='completed', model='gpt-6-luna', id='injected-response',
            output=[dict(type='message', role='assistant', content=[dict(type='output_text', text=json.dumps(value))])],
            usage=dict(input_tokens=100, output_tokens=100, total_tokens=200))


class RecordedFixtureEpisodes:
    """Produce explicit no-physics fixtures through real FileRankServer services."""
    def __init__(self, *, host_error=False, validation_host_error=False, observed_task=None):
        self.calls = []
        self.host_error, self.validation_host_error = host_error, validation_host_error
        self.observed_task = observed_task

    def __call__(self, args, *, service_factory=None, knowledge_limits=None, allow_fixture_knowledge=False):
        out = Path(args.run_dir)
        out.mkdir(parents=True, exist_ok=False)
        validation = service_factory is not None
        self.calls.append(dict(path=out, validation=validation, seed=args.seed))
        identity = self.observed_task or args.assembly_dir.rstrip('/').split('/')[-1]
        features = describe_loaded({p: {'mesh': MeshFixture(), 'transform': 'final'} for p in ('1','2','base')},
            {p+'.obj': 'a'*64 for p in ('1','2','base')}, identity, 'c'*40, 'base',
            dict(seed=args.seed, fixture=True))
        run_id = 'fixture:' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
        req = dict(run_id=run_id, request_id=run_id+':r1', state_id='state:'+digest(['1','2','base']),
            assembled_parts=['1','2','base'], candidate_ids=['1','2'], base_part='base')
        envelope = dict(schema_version=1, kind='rank_request', timeout_seconds=4, request=req,
            features=dict(protocol='sequence_native_v1', cad=filter_features(features,req['assembled_parts']),
                memory_context=task_context(features), episode_feedback=[], validated_repair_skills=[]))
        service = SequenceRankService(dict(DEFAULTS), out/'fixture_ranker.sqlite',
            memory_records=read(args.memory), lessons=read(args.lessons), skills=read(args.skills),
            knowledge_limits=knowledge_limits, allow_fixture_skills=True,
            transport=lambda *a: (_ for _ in ()).throw(AssertionError('NO_PROVIDER_IN_WORKER_FIXTURE')))
        if validation:
            pipeline = service_factory(service)
        else:
            cards = read(args.skills).get('entries', [])
            pipeline = FixtureSequenceService(service, selected_skill_id=cards[0]['skill_id'] if cards else None)
        mailbox = out/'episode/ipc'
        server = FileRankServer(mailbox, run_id, 'pipeline', lambda *a, **k: None, rank_service=pipeline)
        filename = hashlib.sha256(req['request_id'].encode()).hexdigest()+'.json'
        atomic_new(mailbox/'requests'/filename, envelope)
        try:
            server.serve_pending()
        finally:
            server.close()
        response = read(mailbox/'responses'/filename)
        if response['fallback_reason'] is not None:
            raise AssertionError('Unexpected fixture IPC contract rejection: '+response['fallback_reason'])
        chain, attempts = [], []
        remaining = ['1','2','base']
        for index,mover in enumerate(response['candidate_ids'], 1):
            child = [p for p in remaining if p != mover]
            raw = dict(feasible=True, part_move=mover, action=[1,0,0], parts_fix=['base'], pose=None)
            chain.append(dict(parent=remaining[:], child=child[:], part_move=mover, sim_info=raw))
            attempts.append(dict(index=index, event_id=run_id+':attempt:'+str(index), call_id='v'+str(index),
                part_move=mover, remaining_parts=remaining[:], parts_rest=child[:],
                request=dict(part_move=mover, parts_rest=child[:], parts_removed=[p for p in ['1','2','base'] if p not in remaining],
                             pose=None, max_grippers=3, timeout=60), returned=True, feasible=True,
                status='PASS', accepted_for_transition=True, deadline_exceeded=False,
                verifier_domain='ASAP_RedMax', raw_sim_info=raw, source_hashes=dict(fixture='f'*64),
                selection_scope='ranked_candidate' if index==1 else 'upstream_terminal_completion'))
            remaining = child
        worker = dict(profile='sequence_native_v1', status='PASS', success=True, fixture=True,
            actual_physics=False, error=None, decision_halt_reason=None, stop_reason='fixture_complete',
            lower_level_policy='original_ASAP', hard_pruning=False, complete_permutation=True,
            heuristic_tripwire_calls=0, source_methods_restored=True, sources_unchanged=True,
            input_unchanged=True, full_robot_execution=False,
            physics_binding=dict(redmax_binary_sha256='a'*64,physics_source_sha256='b'*64),
            source_methods_sha256={k:'c'*64 for k in ('simulate','update_tree','path_check','stability_check')},
            connected=dict(success=True,complete_chain=chain,max_depth=2), selected_plan=chain, attempts=attempts,
            configuration=dict(seed=args.seed,automatic_retries=0,candidate_pruning=False,physics_verdict_reuse=False),
            rank_decisions=[dict(request=req,applied_order=response['candidate_ids'],full_candidate_set=True,
                heuristic_or_gnn_invoked=False,rng_before='fixture',rng_after='fixture')],
            budget=dict(counts=dict(path=0,stability=0)))
        worker_path = out/'episode/worker_result.json'
        write(worker_path, worker)
        write(out/'episode/context.json', task_context(features))
        native = build_native_snapshot(task_context(features), attempts, str(worker_path),
            hashlib.sha256(worker_path.read_bytes()).hexdigest())
        write(out/'native_observations.json', native)
        count = int(response['metadata'].get('skill_execution', {}).get('status') == 'PROPOSED') if response['metadata'].get('skill_execution') else 0
        write(out/'metrics.json', dict(worker_status='PASS', success=True, sequence_skill_executions=count,
            actual_API_attempts=0, fixture=True))
        failed = self.host_error or (validation and self.validation_host_error)
        write(out/'result.json', dict(status='ERROR' if failed else 'PASS', metrics=dict(
            process_exit=0, service_error=None, sources_and_frozen_inputs_preserved=not failed)))
        return int(failed)


class TrainingBackendIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sequence_backend_OFFLINE_', dir=ROOT/'runs')
        self.root = Path(self.temp.name)
        self.root.relative_to(ROOT/'runs')  # Cleanup is restricted to this owned test directory.
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, OPENAI_API_KEY='offline-backend-fixture-secret')
        env.start(); self.addCleanup(env.stop)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_HTTP'))
        self.net = net.start(); self.addCleanup(net.stop); self.addCleanup(self.net.assert_not_called)
        process = patch('subprocess.run', side_effect=AssertionError('NO_WSL_OR_PHYSICS'))
        self.process = process.start(); self.addCleanup(process.stop); self.addCleanup(self.process.assert_not_called)
        self.task = dict(assembly_id='training-A', part_count=3, base_part='base',
            assembly_dir='/mnt/c/dev/disassembly_codex_ready_v2/runs/fixture-input/training-A')

    def backend(self, learner=None, episodes=None):
        learner, episodes = learner or InjectedLearner(), episodes or RecordedFixtureEpisodes()
        backend = training.NativeTrainingBackend(self.root, ['evaluation-X','00335'], live=False,
            learner_transport=learner, episode_runner=episodes, allowed_task_ids=['training-A'])
        return backend,learner,episodes

    def test_real_backend_lifecycle_registers_card_and_freezes_before_reuse(self):
        backend, learner, episodes = self.backend()
        result = run_training_cycle([self.task],backend,held_out_ids=['evaluation-X','00335'])
        self.assertFalse(result['stopped'], result)
        self.assertEqual([r['stage'] for r in learner.requests], ['lesson_initial','python_generate','lesson_postvalidation'])
        self.assertEqual(len(episodes.calls),4)
        self.assertEqual([c['seed'] for c in episodes.calls if c['validation']], [0,1])
        self.assertEqual(len(backend.skills),1)
        self.assertEqual(backend.skills[0]['protocol'],'sequence_skill_v1')
        self.assertTrue(backend.skills[0]['fixture'])
        self.assertNotIn('card',backend.skills[0])  # Store admitted.card, not its wrapper.
        self.assertEqual(result['tasks'][0]['registered_skill'],backend.skills[0]['skill_id'])
        self.assertEqual(result['tasks'][0]['reuse_status'],'EXECUTED_EFFECT_UNVERIFIED')
        frozen=read(result['frozen']['skills'])
        self.assertEqual(frozen['entries'],backend.skills)
        self.assertTrue(frozen['fixture'])
        self.assertEqual(result['performance_effect'],'UNVERIFIED')
        post=backend.lessons[0]
        initial_ids={f['evidence_id'] for f in learner.requests[0]['input']['packet']['facts']}
        post_ids=set(post['claim']['evidence_ids'])
        self.assertFalse(post_ids.intersection(initial_ids))
        self.assertEqual(post['provenance']['validated_program_sha256'],backend.skills[0]['program_sha256'])

    def test_generation_contract_failure_gets_one_explicit_revision_and_preserved_source(self):
        backend, learner, episodes = self.backend(InjectedLearner(reject_first_python=True))
        result=run_training_cycle([self.task],backend)
        self.assertFalse(result['stopped'],result)
        self.assertEqual([r['stage'] for r in learner.requests],
            ['lesson_initial','python_generate','python_revision','lesson_postvalidation'])
        revision=learner.requests[2]['input']
        self.assertIn('import os',revision['previous_candidate']['program']['source'])
        self.assertEqual(revision['validation_feedback']['status'],'REJECTED')
        saved=list(self.root.glob('*python_generate_result.json'))
        rejected=read(saved[0])
        self.assertFalse(rejected['ok'])
        self.assertIn('import os',rejected['raw_candidate_source']['source_text'])
        self.assertEqual(len(episodes.calls),4)

    def test_host_error_with_worker_pass_stops_before_any_learning(self):
        backend, learner, episodes=self.backend(episodes=RecordedFixtureEpisodes(host_error=True))
        result=run_training_cycle([self.task],backend)
        self.assertTrue(result['stopped'])
        self.assertEqual(backend.halted,'episode_host_error')
        self.assertEqual(learner.requests,[])
        self.assertEqual(len(episodes.calls),1)
        self.assertEqual(backend.skills,[])

    def test_validation_host_error_blocks_postlesson_and_registration(self):
        backend,learner,episodes=self.backend(episodes=RecordedFixtureEpisodes(validation_host_error=True))
        result=run_training_cycle([self.task],backend)
        self.assertTrue(result['stopped'])
        self.assertEqual(backend.halted,'validation_host_error')
        self.assertEqual([r['stage'] for r in learner.requests],['lesson_initial','python_generate'])
        self.assertEqual(backend.skills,[])
        self.assertEqual(len(episodes.calls),2)

    def test_heldout_observations_cannot_enter_cumulative_memory(self):
        backend,learner,episodes=self.backend(episodes=RecordedFixtureEpisodes(observed_task='evaluation-X'))
        frozen=backend.freeze('initial')
        episode=backend.episode(self.task,0,frozen)
        with self.assertRaisesRegex(ValueError,'training_task_identity_mismatch'):
            backend.collect(episode)
        self.assertEqual(backend.snapshots,[])
        self.assertEqual(learner.requests,[])

    def test_luna_stage_cap_is_shared_by_real_backend_and_never_automatically_retries(self):
        backend,learner,episodes=self.backend()
        frozen=backend.freeze('initial')
        episode=backend.episode(self.task,0,frozen);backend.collect(episode)
        packet=backend.packet(self.task,episode)
        responses=[backend._learn(self.task,'lesson_initial',packet) for _ in range(5)]
        self.assertTrue(all(r['ok'] for r in responses[:4]))
        self.assertEqual(responses[-1]['error'],'call_budget_exhausted')
        self.assertEqual(len(learner.requests),4)
        self.assertEqual(backend.halted,'call_budget_exhausted')
        with closing(sqlite3.connect(self.root/'training-A_luna_ledger.sqlite')) as db:
            self.assertEqual(db.execute('select sum(reserved_call) from rank_requests').fetchone()[0],4)

    def test_postlesson_old_citation_cannot_register_new_candidate(self):
        backend,learner,episodes=self.backend(InjectedLearner(stale_post_citation=True))
        result=run_training_cycle([self.task],backend)
        self.assertFalse(result['stopped'])
        self.assertEqual(backend.skills,[])
        self.assertIsNone(result['tasks'][0]['registered_skill'])
        self.assertEqual(result['tasks'][0]['reuse_status'],'NOT_SELECTED')
        post=next(stage['result'] for stage in result['tasks'][0]['stages'] if stage['stage']=='postvalidation_lesson')
        self.assertFalse(post['ok'])
        self.assertEqual(post['error'],'sequence_lesson_unobserved_evidence')


if __name__=='__main__':
    unittest.main()
