"""Saved-input replay and synthetic boundary checks; API0 / new physics0."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

from tools.run_luna_experiment import read, write, sha
from tools.run_skill_discovery import source_case, run_case
from tools.python_source_budget import observation_binding, observation_request, source_budget
from tools.run_python_skill_cycle import attach_debug_inputs, python_generation_packet, main
from tools.run_multitask_learning import learn
from orchestrator.execution_observation import (action_result, from_attempt, validate_execution,
    export_episode_memory)
from orchestrator.explicit_action_memory import (append_action_observation, freeze_memory,
    load_snapshot, FrozenActionMemory)
from orchestrator.lesson_memory import evidence_view
from orchestrator.knowledge_tools import KnowledgeCatalog
from orchestrator.knowledge_context import KnowledgeContext
from tests.test_explicit_action_contract import request
from worker.explicit_action_contract import canonical_response
from worker.task_memory import digest

ROOT = Path(__file__).absolute().parents[1]
MANIFEST = ROOT/'runs/multitask_roles_20261105/prepared/manifest.json'
LIVE = ROOT/'runs/multitask_roles_live_20261106/live'
LIBRARY = ROOT/'runs/separation_final_live_20261022/live_cycle/skills_frozen.json'
LESSONS = ROOT/'runs/separation_final_live_20261022/live_cycle/lessons_frozen.json'


class OfflineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='observation_OFFLINE_', dir=ROOT/'runs')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('API_FORBIDDEN'))
        blocker = network.start()
        self.addCleanup(network.stop)
        self.addCleanup(blocker.assert_not_called)


class SourceObservationTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.manifest = read(MANIFEST)
        self.prepared = [source_case(s) for s in self.manifest['cases']]

    def test_original_41_second_failures_reach_generation_and_runtime(self):
        packet = python_generation_packet(self.manifest, 'offline')
        before = deepcopy(self.prepared)
        old = attach_debug_inputs(packet, self.prepared, 'offline')
        new = attach_debug_inputs(packet, self.prepared, 'offline', source_observation=True)
        for a, b in zip(old['context']['python_debug_requests'], new['context']['python_debug_requests']):
            self.assertTrue(a['current_observation_request']['episode_feedback'][0]['deadline_exceeded'])
            feedback = b['current_observation_request']['episode_feedback'][0]
            self.assertFalse(feedback['deadline_exceeded'])
            self.assertFalse(feedback['accepted_for_transition'])
            self.assertGreater(b['observation_binding']['original_allowance_seconds'], 400)
            self.assertFalse(b['observation_binding']['saved_observation_is_new_trial'])
            from orchestrator.python_skill_program import build_context as base_context
            self.assertIsNotNone(base_context(b['current_observation_request'])[1])
        self.assertEqual(self.prepared, before)

    def test_source_identity_tampering_fails_before_generation(self):
        for kind in ('raw', 'request', 'context'):
            p = deepcopy(self.prepared[0])
            if kind == 'raw': p['raw']['elapsed_seconds'] += 1
            elif kind == 'request': p['request']['request_id'] += '-wrong'
            else: p['context']['seed'] += 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): observation_binding(p)

    def test_true_late_synthetic_observation_stays_ineligible(self):
        p = deepcopy(self.prepared[0])
        original = read(Path(p['source_path'])/'request.json')
        p['context']['remaining_wall_seconds'] = 30
        original['context'] = deepcopy(p['context'])
        source = self.root/'synthetic_source'; source.mkdir()
        write(source/'request.json', original); write(source/'result.json', p['raw'])
        report = read(Path(p['spec']['report']))
        report['attempts'][p['spec']['attempt_index']-1]['deadline_exceeded'] = True
        report['attempts'][p['spec']['attempt_index']-1]['context'] = deepcopy(p['context'])
        path = self.root/'synthetic_report.json'; write(path, report)
        p['source_path'] = str(source); p['spec']['report'] = str(path); p['source_report_sha256'] = sha(path)
        req, binding = observation_request(p, 'late', 'late:event')
        self.assertTrue(req['episode_feedback'][0]['deadline_exceeded'])
        from orchestrator.python_skill_program import build_context
        self.assertIsNone(build_context(req)[1])
        self.assertEqual(binding['original_allowance_seconds'], 30)

    def test_source_budget_preserves_all_original_solver_and_stability_settings(self):
        from orchestrator.skill_case import physical_context
        from worker.skill_case_budget import validate_source_context
        for p in self.prepared:
            bound = observation_binding(p)['fresh_validation_budget']
            context = physical_context(p['context'], p['request'], p['choice'], event_id='offline',
                seconds=bound['seconds'], wall_limit=bound['seconds'], preserve_stability_policy=True)
            self.assertEqual(validate_source_context(context, read(Path(p['source_path'])/'request.json')['context']), bound['seconds'])

    def test_general_fresh_entrypoint_binds_first_generation_and_revision(self):
        calls = []
        def invoke(name, tool, args, limit):
            calls.append((name, tool, args, limit))
            out = Path(args[args.index('--run-dir')+1]); out.mkdir()
            if name == 'lesson_extract':
                write(out/'result.json', {'status':'CANDIDATE'})
                write(out/'lesson.json', read(LIVE/'lesson_draft/lesson.json'))
                return 0
            write(out/'result.json', {'status':'GENERATION_REJECTED'})
            return 1
        result = learn(self.root, invoke, SimpleNamespace(summary=lambda:{}),
            learning_inputs=(MANIFEST, LIVE.parent/'prepared_v2/lesson_packet.json'))
        self.assertEqual([c[0] for c in calls], ['lesson_extract','candidate_01','candidate_02'])
        self.assertEqual(result['status'], 'NOT_REGISTERED')
        for name, _, args, limit in calls[1:]:
            self.assertEqual(args[args.index('--observation-contract')+1], 'source_bound_v1')
            self.assertNotIn('--replay-generated-from', args)
            self.assertEqual(limit, 3700)
        manifest = read(self.root/'learning_observation_contract.json')
        self.assertEqual(len(manifest['cases']), 4)
        self.assertEqual(manifest['new_physics_calls_by_binding'],0)

    def test_new_cli_preview_emits_bound_packet_without_api_or_physics(self):
        output = self.root/'preview'
        with patch('tools.run_python_skill_cycle.run_case', side_effect=AssertionError('PHYSICS_FORBIDDEN')) as physics, redirect_stdout(StringIO()):
            code = main(['--run-dir', str(output), '--manifest', str(MANIFEST),
                '--lesson', str(LIVE/'lesson_draft/lesson.json'), '--mode','preview',
                '--observation-contract','source_bound_v1'])
        self.assertEqual(code, 0); physics.assert_not_called()
        self.assertEqual(read(output/'config.json')['fresh_validation_wall_seconds'], 'bound_source_per_case')
        self.assertEqual(len(read(output/'observation_bindings.json')), 4)
        for row in read(output/'generation_packet.json')['context']['python_debug_requests']:
            self.assertFalse(row['current_observation_request']['episode_feedback'][0]['deadline_exceeded'])
        self.assertFalse(read(output/'config.json')['api']['enabled'])

    def test_fresh_worker_boundary_uses_the_exact_bound_allowance(self):
        from tools import run_skill_discovery as runner
        p = self.prepared[0]; bound = observation_binding(p)['fresh_validation_budget']; seen=[]
        def physical(packet, folder, **kwargs):
            seen.append((deepcopy(packet),deepcopy(kwargs)))
            (folder/'physics').mkdir(parents=True)
            write(folder/'physics/result.json', p['raw'])
            return deepcopy(p['raw'])
        case = dict(case_id='synthetic',source_task='02628',source_budget=bound,case_wall_seconds=bound['seconds'])
        with patch.object(runner,'actual_action',side_effect=physical), patch.object(runner,'step_program',return_value={'response':None}):
            run_case(p,case,read(LIVE/'candidate_01/provider_proposal.json')['program'],self.root,'offline')
        self.assertEqual(seen[0][1]['source_budget'],bound)
        self.assertEqual(seen[0][0]['context']['remaining_wall_seconds'],bound['seconds'])
        self.assertFalse(read(self.root/'cases/synthetic/program_request.json')['episode_feedback'][0]['deadline_exceeded'])


class ExecutionMemoryTests(OfflineTest):
    def report(self, status='PASS', late=False, accepted=True):
        r = request(); response=canonical_response(r,None)
        raw = dict(status=status, backend='ASAP_RedMax', fixture=True,
            path_success=status=='PASS', stability_success=True if status=='PASS' else None,
            elapsed_seconds=41.0)
        attempt = dict(request_id=r['request_id'],state_id=r['state_id'],choice=response['choice'],
            status=status,raw=raw,elapsed_seconds=42.0,accepted_for_transition=accepted,
            deadline_exceeded=late,context={'time_remaining_seconds':40.0 if late else 400.0})
        return dict(decisions=[dict(request=r,response=response)],attempts=[attempt])

    def export(self, report):
        records = export_episode_memory(report,self.root,'fixture/planner_report.json',allow_test_root=True)
        return records,load_snapshot(self.root/'action_memory_frozen.json',True)

    def test_late_physical_pass_roundtrips_without_becoming_transition_success(self):
        report=self.report(late=True,accepted=False); before=deepcopy(report)
        records, snapshot=self.export(report)
        refs=FrozenActionMemory(snapshot).retrieve(report['decisions'][0]['request'])
        self.assertEqual(refs[0]['status'],'PASS')
        self.assertFalse(refs[0]['accepted_for_transition']);self.assertTrue(refs[0]['deadline_exceeded'])
        self.assertFalse(refs[0]['physics_reuse_allowed']);self.assertFalse(refs[0]['current_attempt_verified'])
        self.assertEqual(records[0]['result']['raw'],report['attempts'][0]['raw'])
        self.assertEqual(report,before)

    def test_accepted_pass_and_failed_attempt_preserve_distinct_meanings(self):
        for status,late,accepted in [('PASS',False,True),('UNKNOWN',False,False),('UNKNOWN',True,False)]:
            result=action_result(self.report(status,late,accepted)['attempts'][0],'fixture/events.json')
            facts=evidence_view({'evidence':[dict(evidence_id='a',source_task='fixture',result=result)]})
            self.assertEqual(facts[0]['status'],status)
            self.assertIs(facts[0]['accepted_for_transition'],accepted)
            self.assertIs(facts[0]['deadline_exceeded'],late)

    def test_missing_historical_values_remain_unknown(self):
        attempt=self.report()['attempts'][0]
        for key in ('context','accepted_for_transition','deadline_exceeded','elapsed_seconds'):attempt.pop(key)
        value=from_attempt(attempt)
        for key in ('accepted_for_transition','deadline_exceeded','remaining_wall_seconds','attempt_elapsed_seconds'):
            self.assertIsNone(value[key])

    def test_execution_metadata_reaches_actual_lesson_provider_body_without_api(self):
        from orchestrator.lesson_discovery import LessonDiscoveryService
        from tests.test_skill_discovery import synthesis, config
        packet=synthesis();packet['context']['lesson_stage']='extract'
        row=packet['evidence'][0]
        result=action_result(self.report(late=True,accepted=False)['attempts'][0],'fixture/report.json')
        packet['evidence']=[dict(evidence_id=row['evidence_id'],source_task='fixture',result=result)]
        service=LessonDiscoveryService(config(enabled=False,max_calls=0,max_usd=0),self.root/'preview.sqlite')
        self.addCleanup(service.close)
        body=service.preview(packet)['body']
        projected=json.loads(body['input'][0]['content'])
        fact=projected['observed_facts'][0]
        self.assertEqual(fact['status'],'PASS')
        self.assertTrue(fact['deadline_exceeded']);self.assertFalse(fact['accepted_for_transition'])

    def test_contradictory_acceptance_and_invalid_numbers_are_rejected(self):
        for status,late in [('PASS',True),('UNKNOWN',False)]:
            with self.assertRaises(ValueError):action_result(self.report(status,late,True)['attempts'][0],'fixture/x.json')
        value=from_attempt(self.report()['attempts'][0])
        for bad in (True,-1,float('nan'),float('inf'),'40'):
            broken=dict(value,attempt_elapsed_seconds=bad)
            with self.subTest(bad=bad),self.assertRaises(ValueError):validate_execution(broken,'PASS')

    def test_metadata_is_hashed_and_cannot_be_edited_in_frozen_memory(self):
        _,snapshot=self.export(self.report())
        snapshot['records'][0]['result']['execution']['accepted_for_transition']=False
        snapshot['content_sha256']=digest(snapshot['records'])
        with self.assertRaisesRegex(ValueError,'hash'):FrozenActionMemory(snapshot)

    def test_model_search_read_and_unified_context_keep_acceptance(self):
        report=self.report(late=True,accepted=False);records,snapshot=self.export(report)
        catalog=KnowledgeCatalog(read(LIBRARY),read(LESSONS),snapshot)
        doc=catalog._documents['memory:'+records[0]['evidence_id']]['content']
        self.assertFalse(doc['accepted_for_transition']);self.assertTrue(doc['deadline_exceeded'])
        self.assertFalse(doc['missing_acceptance_metadata'])
        r=report['decisions'][0]['request']
        r['memory_references']=FrozenActionMemory(snapshot).retrieve(r)
        r['evidence_ids']=[v['evidence_id'] for v in r['memory_references']]
        view=KnowledgeContext(read(LIBRARY),read(LESSONS)).query(r)
        self.assertEqual(view['audit']['observations_with_missing_acceptance'],[])

    def test_old_result_and_lesson_projection_remain_exactly_compatible(self):
        result=action_result(self.report()['attempts'][0],'fixture/report.json');result.pop('execution')
        r=request();append_action_observation(self.root/'old.jsonl',r,canonical_response(r,None),result,allow_test_root=True)
        snap=freeze_memory(self.root/'old.jsonl',self.root/'old.json',True)
        refs=FrozenActionMemory(snap).retrieve(r)
        self.assertNotIn('execution',refs[0]);self.assertNotIn('accepted_for_transition',refs[0])
        facts=evidence_view({'evidence':[dict(evidence_id='old',result=result)]})
        self.assertNotIn('execution',facts[0])
        catalog=KnowledgeCatalog(read(LIBRARY),read(LESSONS),snap)
        doc=catalog._documents['memory:'+snap['records'][0]['evidence_id']]['content']
        self.assertIsNone(doc['accepted_for_transition']);self.assertTrue(doc['missing_acceptance_metadata'])

    def test_same_action_late_and_accepted_results_are_not_collapsed_by_state_retrieval(self):
        r=request()
        for i,late in enumerate((False,True)):
            req=deepcopy(r);req['request_id'] += str(i)
            result=action_result(self.report(late=late,accepted=not late)['attempts'][0],'fixture/report.json')
            append_action_observation(self.root/'mixed.jsonl',req,canonical_response(req,None),result,allow_test_root=True)
        snap=freeze_memory(self.root/'mixed.jsonl',self.root/'mixed.json',True)
        refs=FrozenActionMemory(snap).retrieve(r,12,'state_diverse_v1')
        self.assertEqual(len(refs),2)
        self.assertEqual({v['accepted_for_transition'] for v in refs},{False,True})


if __name__=='__main__':unittest.main()
