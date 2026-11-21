"""API-free tests. Stored RedMax replay and injected workers are NOT new physics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.run_luna_experiment import read,write,sha
from worker.task_memory import digest
from tools.support_experience import support_experiences
from orchestrator.support_learning import compact_bundle,audit_support_program
from orchestrator.recovery_sequence import append_feedback,validate_protocol

PROBE=ROOT/'runs/support_failure_diagnosis_20261109_1001'
PARENT=ROOT/'runs/astra_skill_learning_20261109_0918/live/candidate_02'
PREPARED=ROOT/'runs/support_learning_improvements_20261109/prepared_v2'


class ProviderRepresentationTests(unittest.TestCase):
    def test_nested_shared_data_roundtrip_preserves_every_outcome(self):
        from orchestrator.support_learning import share_repeated_data
        common={'parts':[{'center':[1,2,3],'metadata':'a'*400}]}
        original=dict(evidence=[dict(evidence_id=str(i),status=s,geometry=deepcopy(common))
            for i,s in enumerate(('PASS','UNKNOWN','LATE','ERROR','NOT_RUN'))],
            nested={'geometry':deepcopy(common),'other':deepcopy(common)})
        view=share_repeated_data(original);pool=view.pop('shared_data');view.pop('shared_data_rule')
        def expand(node):
            if isinstance(node,dict):
                if set(node)=={'$support_shared'}:return expand(pool[node['$support_shared']])
                return {k:expand(v) for k,v in node.items()}
            return [expand(v) for v in node] if isinstance(node,list) else node
        self.assertEqual(expand(view),original)

    def test_duplicate_evidence_fields_expand_losslessly(self):
        from orchestrator.support_learning import deduplicate_observations
        original={'evidence':[dict(evidence_id='trace',choice={'part_id':'x'},
            remaining_parts=['x','y'],execution={'accepted_for_transition':False}),
            dict(evidence_id='memory',same_observation_as_event_id='trace',choice={'part_id':'x'},
            remaining_parts=['x','y'],execution={'accepted_for_transition':False},status='UNKNOWN')]}
        compact=deduplicate_observations(deepcopy(original))
        reference=compact['evidence'][1].pop('identical_fields_reference')
        for key in reference['fields']:
            compact['evidence'][1][key]=deepcopy(compact['evidence'][0][key])
        self.assertEqual(compact,original)

    def test_shared_runtime_subtrees_expand_exactly(self):
        from orchestrator.support_learning import share_runtime_examples
        original=dict(examples=[dict(runtime_input_example=dict(geometry={'parts':[1,2]},
            pose_catalog=[{'pose_id':'p'}],last={'part_id':str(i)})) for i in range(6)])
        compact=share_runtime_examples(deepcopy(original))
        self.assertEqual(len(compact['shared_input_subtrees']),2)
        for example in compact['examples']:
            for key,ref in example.pop('runtime_input_shared_fields').items():
                example['runtime_input_example'][key]=deepcopy(compact['shared_input_subtrees'][ref])
        self.assertEqual(compact['examples'],original['examples'])


@unittest.skipUnless(PREPARED.is_dir(),'requires preserved local research artifacts')
class SupportEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=read(PREPARED/'manifest.json')
        cls.bundle=support_experiences(cls.manifest)

    def test_preserves_all_pass_failure_and_not_run_rows(self):
        group=self.bundle['groups'][0]
        self.assertEqual([r['outcome'] for r in group['rows']].count('PASS'),5)
        self.assertEqual([r['outcome'] for r in group['rows']].count('UNKNOWN'),15)
        self.assertEqual([r['outcome'] for r in group['rows']].count('NOT_RUN'),2)
        self.assertEqual(len(self.bundle['evidence']),20)
        self.assertFalse(self.bundle['registration_evidence'])
        self.assertIsNone(group['recommended_action'])

    def test_admission_source_rejected_before_any_probe_io(self):
        m=deepcopy(self.manifest);m['support_probe_sources'][0]['case_id']='source_02628_yaw90'
        with self.assertRaisesRegex(ValueError,'declared_debug'):
            support_experiences(m)

    def test_changed_hash_rejected(self):
        m=deepcopy(self.manifest);m['support_probe_sources'][0]['result_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'hash_mismatch'):support_experiences(m)

    def test_e_drive_rejected_lexically(self):
        m=deepcopy(self.manifest);m['support_probe_sources'][0]['run_dir']='E:/do-not-access'
        with self.assertRaises((ValueError,PermissionError)):support_experiences(m)

    def test_projection_preserves_denominator_and_no_file_paths(self):
        view=compact_bundle(self.bundle);g=view['groups'][0]
        self.assertEqual(len(g['rows']),22)
        self.assertNotIn('source_refs',g)
        self.assertIn('geometry',g['source_observation'])
        self.assertIsNone(g['rows'][12]['failure_cause'])
        bad=deepcopy(self.bundle);bad['groups'][0]['rows'].pop()
        with self.assertRaisesRegex(ValueError,'complete_support'):compact_bundle(bad)

    def test_old_actual_astra_program_is_measured_counterexample(self):
        packet=read(PARENT/'generation_packet.json')
        packet['context']['support_development_experiences']=deepcopy(self.bundle)
        program=read(PARENT/'provider_proposal.json')['program']
        result=audit_support_program(program,packet)
        row=next(r for r in result['checks'] if r['case_id']=='source_02628')
        self.assertIn('repeats_measured_failed_support_in_exact_development_state',row['errors'])
        self.assertTrue(all(c['accepted'] for c in row['renamed_checks']))
        self.assertFalse(result['new_physical_success'])

    def test_different_solver_seed_is_not_exact_grid_evidence(self):
        packet=read(PARENT/'generation_packet.json')
        packet['context']['support_development_experiences']=deepcopy(self.bundle)
        rows=packet.get('validation_feedback') or packet['context']['python_debug_requests']
        for group in packet['context']['support_development_experiences']['groups']:
            task=deepcopy(next(r for r in rows if r['case_id']==group['case_id'])
                          ['current_observation_request']['task_context'])
            task['task']['solver_settings']['seed']=17
            group['source_task_context_sha256']=digest(task)
        result=audit_support_program(read(PARENT/'provider_proposal.json')['program'],packet)
        row=next(r for r in result['checks'] if r['case_id']=='source_02628')
        self.assertEqual(row['known_source_outcomes'],[])

    def test_direction_change_rejected_before_support_only_physics(self):
        packet=read(PARENT/'generation_packet.json')
        packet['context']['support_development_experiences']=deepcopy(self.bundle)
        packet['context']['recovery_protocol']=self.manifest['recovery_protocol']
        program=dict(language='python_policy_v1',source="""def propose(ctx):
    last = ctx['last']['choice']
    direction = 'xp' if last['direction_id'] != 'xp' else 'xn'
    out = {'part_id': last['part_id'], 'pose_id': last['pose_id'], 'skill_id': 'single_force_v1', 'direction_id': direction, 'second_direction_id': None, 'support_ids': [], 'evidence_ids': []}
    if eligible(out):
        return out
    return None
""")
        result=audit_support_program(program,packet)
        self.assertTrue(any('outside_declared_support_recovery_scope' in r['errors'] for r in result['checks']))
        self.assertFalse(result['accepted'])

    def test_chain_timeout_is_explicit_in_existing_learning_orchestrator(self):
        from tools.run_multitask_learning import learn
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(prefix='support_budget_fixture_',dir=ROOT/'runs') as directory:
            out=Path(directory)
            write(out/'learner_config.json',dict(model='gpt-6-astra',max_output_tokens=25000))
            calls=[]
            def invoke(name,tool,args,limit):
                calls.append((name,limit));dest=Path(args[args.index('--run-dir')+1]);dest.mkdir()
                write(dest/'result.json',dict(status='CANDIDATE' if name=='lesson_extract' else 'GENERATION_REJECTED'))
                return 0 if name=='lesson_extract' else 1
            result=learn(out,invoke,SimpleNamespace(summary=lambda:{}),
                learning_inputs=(PREPARED/'manifest.json',PREPARED/'lesson_packet.json'))
            self.assertEqual(result['status'],'NOT_REGISTERED')
            self.assertEqual(calls,[('lesson_extract',660),('candidate_01',8720),('candidate_02',8720)])


@unittest.skipUnless(PREPARED.is_dir(),'requires preserved local research artifacts')
class RecoverySequenceTests(unittest.TestCase):
    def setUp(self):
        from tools.run_skill_discovery import source_case
        from tools.python_source_budget import source_budget
        from orchestrator.skill_program import program_hash
        self.temp=tempfile.TemporaryDirectory(prefix='support_sequence_fixture_',dir=ROOT/'runs')
        assert Path(self.temp.name).resolve().is_relative_to((ROOT/'runs').resolve())
        self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name)
        spec=next(s for s in read(PREPARED/'manifest.json')['cases'] if s['case_id']=='source_02628')
        self.prepared=source_case(spec);self.binding=source_budget(self.prepared)
        # Human-authored test policy intentionally chooses one known failure then a stored PASS.
        # It is never described as generated, learned, or newly physically validated.
        self.program=dict(language='python_policy_v1',source='''def propose(ctx):
    if not ctx.get('failure'):
        return None
    ids = sorted(ctx['remaining_parts'])
    last = ctx['last']['choice']
    supports = [ids[2], ids[4]]
    if len(ctx['skill_history']) > 1:
        supports = [ids[1], ids[2]]
    out = {'part_id': last['part_id'], 'pose_id': last['pose_id'], 'skill_id': last['skill_id'], 'direction_id': last['direction_id'], 'second_direction_id': None, 'support_ids': supports, 'evidence_ids': []}
    if eligible(out):
        return out
    return None
''')
        self.case=dict(case_id='sequence_fixture',phase='debug',source_task='02628',
            program_hash=program_hash(self.program),input_hashes=self.prepared['raw']['input_hashes'],
            source_hashes=self.prepared['expected_source_hashes'],trigger_kind='stability_failed',
            expected_diagnostic_phase='dynamic_failure',runtime_contract='lesson_runtime_v2',
            case_wall_seconds=self.binding['seconds'],source_budget=self.binding,
            recovery_protocol=dict(version='bounded_support_recovery_v1',max_recovery_attempts=3,
                max_total_seconds=2000,action_scope='support_only'))
        self.elapsed=0;self.calls=[]

    def fake_worker(self,packet,folder,source_budget):
        self.assertEqual(source_budget,self.binding)
        supports=packet['choice']['support_ids'];key=tuple(sorted(supports))
        sources={('0','5'):PARENT/'cases/source_02628/trigger/physics',
            ('2','4'):PROBE/'live_support_probe/support_17/physics',
            ('1','2'):PROBE/'live_support_probe/support_12/physics'}
        source=sources[key];raw=read(source/'result.json')
        self.assertEqual(raw['chosen'],packet['choice'])
        dest=folder/'physics';dest.mkdir(parents=True)
        write(dest/'request.json',packet)
        for name in ('trajectory.json','endpoint_evidence.json'):
            (dest/name).write_bytes((source/name).read_bytes())
        raw['fixture']=True  # Actual registry must reject this injected/replayed worker.
        write(dest/'result.json',raw)
        (folder/'execute').mkdir()
        write(folder/'execute/command.json',dict(exit_code=0,launch_error=None,
            elapsed_seconds=raw['elapsed_seconds']+0.1,fixture=True))
        self.calls.append(key);self.elapsed+=raw['elapsed_seconds']+0.25
        return raw

    def run_chain(self):
        from tools.run_support_recovery_case import run_case
        return run_case(self.prepared,self.case,self.program,self.output,'fixture',
                        action_executor=self.fake_worker,clock=lambda:self.elapsed)

    def test_failure_feedback_continues_to_changed_action_and_all_costs(self):
        from orchestrator.skill_validation import record_validation,check_evidence
        artifacts=self.run_chain()
        record=record_validation(self.case,artifacts,self.case['program_hash'],True,True,self.program)
        self.assertEqual(record['outcome'],'PASS')
        self.assertEqual(record['recovery_sequence']['first_attempt_outcome'],'UNKNOWN')
        self.assertEqual(record['path_calls'],2)
        self.assertEqual(record['trigger_path_calls'],1)
        self.assertEqual(len(record['recovery_sequence']['steps']),2)
        self.assertTrue(record['fixture_mode'])
        self.assertEqual(record['classification_version'],4)
        self.assertFalse(record['execution_roles']['automatic_recovery']['allowed'])
        self.assertEqual(len(record['execution_roles']['automatic_recovery']['step_guards']),2)
        self.assertFalse(record['execution_roles']['automatic_recovery']['automatic_execution_verified'])
        self.assertEqual(check_evidence(record,True),record)
        legacy=record_validation(self.case,artifacts,self.case['program_hash'],True,True,self.program,
                                 _classification_version=3)
        self.assertNotIn('execution_roles',legacy)
        self.assertTrue(all('execution_roles' not in s for s in legacy['recovery_sequence']['steps']))
        self.assertEqual(check_evidence(legacy,True),legacy)
        with self.assertRaisesRegex(ValueError,'fixture_cannot'):
            record_validation(self.case,artifacts,self.case['program_hash'],True,False,self.program)

    def test_attempt_limit_stops_without_hidden_extra_call(self):
        from orchestrator.skill_validation import record_validation
        self.case['recovery_protocol']['max_recovery_attempts']=1
        artifacts=self.run_chain()
        record=record_validation(self.case,artifacts,self.case['program_hash'],True,True,self.program)
        self.assertEqual(record['outcome'],'UNKNOWN');self.assertEqual(len(self.calls),2)
        self.assertEqual(record['recovery_sequence']['stop_reason'],'ATTEMPT_LIMIT')

    def test_v2_supplies_measured_execution_at_every_step_and_replays(self):
        from orchestrator.skill_program import program_hash
        from orchestrator.skill_validation import record_validation,check_evidence
        self.case['recovery_protocol']['version']='bounded_support_recovery_v2'
        self.program['source']=self.program['source'].replace('def propose(ctx):',
            "def propose(ctx):\n    if not ctx['last']['feedback'].get('execution'):\n        return None")
        self.case['program_hash']=program_hash(self.program)
        artifacts=self.run_chain()
        record=record_validation(self.case,artifacts,self.case['program_hash'],True,True,self.program)
        self.assertEqual(record['outcome'],'PASS')
        for step in record['recovery_sequence']['steps']:
            selection=read(step['artifacts']['selection_path'])
            request=read(selection['program_request_path'])
            for feedback in request['episode_feedback']:
                execution=feedback['execution']
                self.assertEqual(execution['source'],'validation_attempt')
                self.assertEqual(execution['remaining_wall_seconds'],self.binding['seconds'])
                self.assertFalse(execution['accepted_for_transition'])
                self.assertGreater(execution['attempt_elapsed_seconds'],execution['physics_elapsed_seconds'])
        self.assertEqual(check_evidence(record,True),record)

    def test_v2_rejects_missing_or_unbound_host_time(self):
        from orchestrator.recovery_sequence import validation_execution
        raw=read(PARENT/'cases/source_02628/trigger/physics/result.json')
        for command in ({'exit_code':0},dict(exit_code=0,elapsed_seconds=0),
                        dict(exit_code=1,elapsed_seconds=999)):
            with self.assertRaisesRegex(ValueError,'command_not_bound'):
                validation_execution(raw,self.binding['seconds'],command)

    def test_changed_trace_and_omitted_failure_cannot_be_promoted(self):
        from orchestrator.skill_validation import record_validation
        artifacts=self.run_chain();file=Path(artifacts['recovery_sequence_path']);trace=read(file)
        trace['steps']=trace['steps'][1:];file.write_text(json.dumps(trace),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'initial_feedback'):
            record_validation(self.case,artifacts,self.case['program_hash'],True,True,self.program)

    def test_invalid_protocol_and_insufficient_budget_stop_before_worker(self):
        p=deepcopy(self.case['recovery_protocol']);p['max_recovery_attempts']=4
        with self.assertRaises(ValueError):validate_protocol(p)
        self.case['recovery_protocol']['max_total_seconds']=50
        with self.assertRaisesRegex(ValueError,'budget_cannot_cover'):self.run_chain()
        self.assertEqual(self.calls,[])

    def test_abstention_stops_after_trigger_without_physics_retry(self):
        from orchestrator.skill_program import program_hash
        from orchestrator.skill_validation import record_validation
        self.program['source']='def propose(ctx):\n    return None\n'
        self.case['program_hash']=program_hash(self.program)
        artifacts=self.run_chain()
        record=record_validation(self.case,artifacts,self.case['program_hash'],True,True,self.program)
        self.assertEqual(record['outcome'],'NOT_APPLICABLE')
        self.assertEqual(record['recovery_sequence']['physical_recovery_attempts'],0)
        self.assertEqual(len(self.calls),1)

    def test_total_deadline_is_not_overridden_by_final_pass(self):
        from orchestrator.skill_validation import record_validation
        artifacts=self.run_chain();file=Path(artifacts['recovery_sequence_path']);trace=read(file)
        trace['elapsed_seconds']=2001;file.write_text(json.dumps(trace),encoding='utf-8')
        record=record_validation(self.case,artifacts,self.case['program_hash'],True,True,self.program)
        self.assertEqual(record['outcome'],'LATE')

    def test_registry_keeps_all_steps_and_cannot_skip_chain_evidence(self):
        from orchestrator.skill_registry import SkillRegistry
        from orchestrator.recovery_sequence import debug_steps
        registry=SkillRegistry(self.output/'registry',True,fixture=True)
        candidate=registry.propose(self.program,dict(description='Replay fixture ONLY',
            runtime_contract='lesson_runtime_v2',recovery_protocol=self.case['recovery_protocol']),
            dict(origin='offline_fixture',generation_id='test'))
        admission=dict(self.case,case_id='unrun_admission',phase='admission')
        registry.set_manifests(candidate['candidate_id'],dict(phase='debug',cases=[self.case]),
                               dict(phase='admission',cases=[admission]))
        artifacts=self.run_chain()
        plain={k:v for k,v in artifacts.items() if k!='recovery_sequence_path'}
        with self.assertRaisesRegex(ValueError,'sequence_evidence_required'):
            registry.add_validation(candidate['candidate_id'],self.case,plain)
        record=registry.add_validation(candidate['candidate_id'],self.case,artifacts)
        self.assertEqual([r['outcome'] for r in debug_steps(record)],['UNKNOWN','PASS'])
        self.assertFalse(registry.admit(candidate['candidate_id'])['admitted'])
        # Exercise the real refinement builder after registry replay checking.
        # The final PASS must not hide the failed first repair from the model.
        write(self.output/'config.json',dict(mode='fixture',ledger=None))
        write(self.output/'result.json',dict(actual_llm_generation=False,
            candidate_id=candidate['candidate_id'],status='NOT_REGISTERED'))
        manifest=dict(cases=[self.case,admission])
        write(self.output/'precommitted_cases.json',manifest)
        from tools.run_python_skill_cycle import refinement_packet
        revised=refinement_packet(read(PREPARED/'lesson_packet.json'),self.output,manifest)
        self.assertEqual([r['outcome'] for r in revised['validation_feedback']],['UNKNOWN','PASS'])
        self.assertEqual([len(r['current_observation_request']['episode_feedback'])
            for r in revised['validation_feedback']],[1,2])

    def test_program_exception_stops_without_worker_retry(self):
        def broken(*args,**kwargs):raise TimeoutError('test transport/worker failure')
        from tools.run_support_recovery_case import run_case
        from unittest.mock import Mock
        backend=Mock(side_effect=broken)
        with self.assertRaises(TimeoutError):
            run_case(self.prepared,self.case,self.program,self.output,'fixture',
                     action_executor=backend,clock=lambda:self.elapsed)
        self.assertEqual(backend.call_count,1)


if __name__=='__main__':unittest.main()
