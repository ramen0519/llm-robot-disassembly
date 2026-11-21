"""API-free release gates; synthetic verification is never real physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from tests.test_skill_discovery import synthesis, config, raw_response
from tests.test_skill_validation import fixture_case, save
from tests.test_lesson_pipeline import claim
from tests.test_hierarchical_planner import policy
from tools.python_skill_fixture import PROGRAM
from orchestrator.lesson_runtime import CONTRACT, bind_execution, checked_execution, extension_for, execute_bound
from orchestrator.python_skill_program import step_program, build_context
from orchestrator.skill_program import program_hash
from orchestrator.skill_validation import record_validation, check_evidence
from orchestrator.skill_registry import SkillRegistry, retrieve, _contract_probe
from orchestrator.lesson_memory import make_lesson, validate_lesson, freeze_lessons, validate_library, published_card, sign, with_prior_knowledge
from orchestrator.lesson_consolidation import repair_packet, make_final_lesson, validate_claim_contract
from orchestrator.lesson_discovery import LessonDiscoveryService
from orchestrator.python_skill_evidence import generation_view
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.generated_skill_service import GeneratedSkillService


PROGRAM_V2 = dict(PROGRAM, source=PROGRAM['source'].replace('def propose(ctx):\n',
    'def propose(ctx):\n    if ctx.get("failure") is None:\n        return None\n', 1))


def v2_case(root, case_id='debug', phase='debug', condition='a'):
    case, artifacts, _ = fixture_case(root, case_id, phase, condition, policy=PROGRAM)
    selection_path=Path(artifacts['selection_path'])
    selected=json.loads(selection_path.read_text())
    req=json.loads(Path(selected['program_request_path']).read_text())
    case.update(program_hash=program_hash(PROGRAM_V2), runtime_contract=CONTRACT)
    selected.update(program_hash=program_hash(PROGRAM_V2), program_runtime=bind_execution(PROGRAM_V2,req))
    save(selection_path,selected)
    return case,artifacts,req,selected


class LessonReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='lesson_release_FIXTURE_')
        self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.network=net.start();self.addCleanup(net.stop);self.addCleanup(self.network.assert_not_called)
        env=patch.dict(os.environ,DISCOVERY_FIXTURE_KEY='NOT_A_REAL_KEY')
        env.start();self.addCleanup(env.stop)

    def admitted(self):
        packet=synthesis();packet['operation']='synthesize'
        draft=make_lesson(claim(packet),packet,dict(origin='manual_authored'))
        save(self.root/'draft.json',draft)
        registry=SkillRegistry(self.root/'registry',True,fixture=True)
        candidate=registry.propose(PROGRAM_V2,dict(description='fixture',lesson_id=draft['content_sha256'],
            runtime_contract=CONTRACT),dict(origin='fixture_authored'))
        cases=[v2_case(self.root,'debug','debug','a'),
            v2_case(self.root,'ADMISSION_HIDDEN_CASE1','admission','b'),
            v2_case(self.root,'ADMISSION_HIDDEN_CASE2','admission','c')]
        registry.set_manifests(candidate['candidate_id'],dict(phase='debug',cases=[cases[0][0]]),
                              dict(phase='admission',cases=[x[0] for x in cases[1:]]))
        for case,artifacts,_,_ in cases:registry.add_validation(candidate['candidate_id'],case,artifacts)
        self.assertTrue(registry.admit(candidate['candidate_id'])['admitted'])
        skills=registry.freeze(self.root/'skills.json')
        return draft,skills,cases[0][2]

    def consolidated(self):
        draft,skills,req=self.admitted()
        packet=repair_packet(self.root/'draft.json',self.root/'skills.json','fixture-final',allow_fixture=True)
        value=claim(packet);value.update(packet['context']['repair_binding']['contract'])
        value['evidence_ids']=[e['evidence_id'] for e in packet['evidence']]
        value['hypothesis']='Updated after measured debug repair; transfer remains unverified.'
        final=make_final_lesson(value,packet,dict(origin='manual_authored'))
        return draft,skills,req,packet,final

    def test_validation_replay_and_runtime_use_same_input(self):
        case,artifacts,req,selected=v2_case(self.root)
        self.assertIsNone(step_program(PROGRAM_V2,req)['response'])
        ext=checked_execution(PROGRAM_V2,req,selected,case)
        result=record_validation(case,artifacts,case['program_hash'],True,True,PROGRAM_V2)
        self.assertEqual(result['outcome'],'PASS')
        self.assertEqual(check_evidence(result,True),result)
        self.assertIsNotNone(execute_bound(PROGRAM_V2,req,ext)['response'])
        checks=_contract_probe(PROGRAM_V2,req,ext)
        self.assertTrue(all(checks[k] for k in ('not_applicable','observation_sensitivity','duplicate_action')))
        ctx,_=build_context(req,ext)
        self.assertEqual(ctx['failure']['kind'],'path_failed')
        self.assertEqual(ctx['lesson_references'],[])

    def test_context_binding_tamper_missing_and_downgrade_rejected(self):
        case,_,req,selected=v2_case(self.root)
        for field in ('context_sha256','request_sha256','program_sha256'):
            bad=deepcopy(selected);bad['program_runtime'][field]='0'*64
            with self.assertRaisesRegex(ValueError,'binding_mismatch'):
                checked_execution(PROGRAM_V2,req,bad,case)
        with self.assertRaisesRegex(ValueError,'contract_mismatch'):
            checked_execution(PROGRAM_V2,req,{k:v for k,v in selected.items() if k!='program_runtime'},case)
        with self.assertRaisesRegex(ValueError,'contract_mismatch'):
            checked_execution(PROGRAM_V2,req,selected,{k:v for k,v in case.items() if k!='runtime_contract'})

    def test_old_artifact_replay_keeps_old_contract(self):
        case,artifacts,program=fixture_case(self.root,policy=PROGRAM)
        record=record_validation(case,artifacts,case['program_hash'],True,True,program)
        self.assertEqual(check_evidence(record,True),record)
        self.assertNotIn('runtime_contract',record['case'])

    def test_draft_cannot_be_published_for_new_runtime(self):
        draft,skills,_=self.admitted()
        with self.assertRaisesRegex(ValueError,'post_validation'):
            freeze_lessons([draft],skills,allow_fixture=True)

    def test_post_validation_lesson_binds_new_repairs_and_final_code(self):
        draft,skills,_,packet,final=self.consolidated()
        self.assertNotEqual(final['content_sha256'],draft['content_sha256'])
        self.assertEqual(final['previous_version'],draft['content_sha256'])
        self.assertEqual(final['repair_binding']['program_hash'],program_hash(PROGRAM_V2))
        self.assertTrue(any(r['status']=='PASS' for r in final['observed_facts']))
        self.assertEqual(validate_lesson(final),final)
        library=freeze_lessons([final],skills,allow_fixture=True)
        self.assertEqual(validate_library(library,skills,allow_fixture=True),library)
        card=published_card(retrieve(skills,allow_fixture=True)[0],library)
        self.assertEqual(card['lesson_id'],final['content_sha256'])
        self.assertEqual(card['hypothesis'],final['claim']['hypothesis'])
        self.assertFalse(library['entries'][0]['causal_claim_verified'])

    def test_final_applicability_mismatch_and_missing_recovery_citation_rejected(self):
        _,_,_,packet,final=self.consolidated()
        for key,value in (('when_to_apply','Only stability failure'),('when_not_to_apply','Always use'),
                          ('trigger_kind','stability_failed')):
            bad=deepcopy(final['claim']);bad[key]=value
            with self.assertRaisesRegex(ValueError,'applicability_mismatch'):
                validate_claim_contract(bad,packet)
        bad=deepcopy(final['claim'])
        bad['evidence_ids']=[r['evidence_id'] for r in packet['evidence'] if r['stage']=='trigger']
        with self.assertRaisesRegex(ValueError,'recovery_citation'):
            validate_claim_contract(bad,packet)

    def test_finalization_cannot_hide_changed_validation_artifacts(self):
        _,_,_,_,final=self.consolidated()
        path=self.root/'debug/recovery/result.json'
        path.write_text(path.read_text()+' ',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'changed'):
            validate_lesson(final)

    def test_consolidation_provider_hides_admission_and_paths(self):
        _,_,_,packet,final=self.consolidated()
        service=LessonDiscoveryService(config(),self.root/'ledger.sqlite')
        self.addCleanup(service.close)
        body=service.preview(packet)['body']; sent=json.dumps(body)
        for forbidden in ('ADMISSION_HIDDEN_CASE1','ADMISSION_HIDDEN_CASE2','draft_path','skills_path'):
            self.assertNotIn(forbidden,sent)
        self.assertIn(PROGRAM_V2['source'],json.loads(body['input'][0]['content'])['context']['validated_program']['source'])
        self.assertIn('AFTER',body['instructions'])

    def test_bad_consolidation_response_rejects_without_retry(self):
        _,_,_,packet,final=self.consolidated()
        bad=deepcopy(final['claim']);bad['when_to_apply']='Always, including unobserved states'
        transport=Mock(return_value=raw_response(dict(request_id=packet['request_id'],
            state_id=packet['state_id'],lesson=bad)))
        service=LessonDiscoveryService(config(),self.root/'ledger.sqlite',transport)
        self.addCleanup(service.close)
        self.assertEqual(service.propose(packet)['status'],'rejected')
        self.assertEqual(service.propose(packet)['status'],'rejected')
        self.assertEqual(transport.call_count,1)

    def test_valid_consolidation_transport_and_prior_knowledge_are_bound(self):
        _,skills,_,packet,final=self.consolidated()
        transport=Mock(return_value=raw_response(dict(request_id=packet['request_id'],
            state_id=packet['state_id'],lesson=final['claim'])))
        service=LessonDiscoveryService(config(),self.root/'valid.sqlite',transport)
        self.addCleanup(service.close)
        response=service.propose(packet)
        self.assertEqual(response['status'],'proposed',response)
        self.assertEqual(response['origin'],'injected_transport')
        library=freeze_lessons([final],skills,allow_fixture=True)
        save(self.root/'lessons.json',library)
        newer=with_prior_knowledge(synthesis(),self.root/'skills.json',self.root/'lessons.json',allow_fixture=True)
        supplied=newer['context']['prior_knowledge']
        self.assertEqual(supplied['supplied_programs'],1)
        self.assertEqual(supplied['entries'][0]['lesson'],final['claim'])
        self.assertFalse(supplied['admission_results_exposed'])
        self.assertNotIn('ADMISSION_HIDDEN_CASE',json.dumps(supplied))

    def test_neutral_refinement_preserves_measured_geometry(self):
        _,_,req,_=v2_case(self.root)
        p=synthesis();p['operation']='refine';p['previous_program']=PROGRAM_V2
        p['context']['lesson_stage']='program'
        p['validation_feedback']=[dict(case_id='debug',phase='debug',current_observation_request=req)]
        projected=generation_view(p)
        geometry=projected['validation_feedback'][0]['current_observation_request']['geometry']
        self.assertEqual(set(geometry['parts']),set(req['geometry']['parts']))
        for part,features in geometry['parts'].items():
            for k,v in features.items():self.assertEqual(v,req['geometry']['parts'][part][k])
        self.assertNotIn('frame',geometry)
        self.assertNotIn('physics_success_implied',geometry)
        self.assertEqual(geometry['relations'],{})
        self.assertFalse(projected['validation_feedback'][0]['geometry_projection']['physics_success_implied'])
        p['validation_feedback'][0]['phase']='admission'
        with self.assertRaisesRegex(ValueError,'debug_feedback_only'):generation_view(p)

    def test_normal_hierarchy_and_deterministic_control_share_runtime(self):
        _,skills,req,_,final=self.consolidated()
        library=freeze_lessons([final],skills,allow_fixture=True)
        svc=HierarchicalDecisionService(Mock(),Mock(),skills,policy()['episode'],allow_fixture=True,
            lesson_library=library,lesson_runtime=True)
        cards=svc._cards(req,time.monotonic()+30)
        expected=step_program(PROGRAM_V2,req,extension=extension_for())['response']['choice']
        self.assertEqual(cards[0]['currently_proposed_choice'],expected)
        self.assertEqual(cards[0]['card']['lesson_id'],final['content_sha256'])
        fallback=Mock();fallback.mode='deterministic';fallback.timeout_seconds=60
        control=GeneratedSkillService(skills,fallback,mode='deterministic',allow_fixture=True,lesson_library=library)
        self.assertEqual(control.handle(req)['choice'],expected)
        fallback.handle.assert_not_called()
        with self.assertRaisesRegex(ValueError,'finalized_lesson'):
            GeneratedSkillService(skills,fallback,mode='deterministic',allow_fixture=True)

    def test_cycle_does_not_publish_technically_admitted_code_if_finalization_fails(self):
        """Synthetic subprocess outputs exercise the actual learn() publication gate.

        No model, child process, physics or fake PASS attribution is involved.
        """
        from types import SimpleNamespace
        from tools import run_hierarchical_cycle as runner
        draft,skills,_=self.admitted()
        with tempfile.TemporaryDirectory(prefix='lesson_release_gate_',dir=runner.ROOT/'runs') as tmp:
            base=Path(tmp)
            plan=json.loads((runner.ROOT/'configs/automatic_learning_cycle_v1.json').read_text(encoding='utf-8'))
            plan.update(initial_library=None,initial_memory=None)
            config=policy();config['enabled']=False
            save(base/'policy.json',config);save(base/'fixture.json',{})
            save(base/'manifest.json',{})
            for failure_mode in ('process_error','rejected_response'):
                output=base/failure_mode;calls=[]
                class LearningStageOnly:
                    def __init__(self,folder,identity,stages):self.stages=stages
                    def run(self):
                        result=self.stages['learn']({'prepare':dict(status='PREPARED',
                            manifest=str(base/'manifest.json'))})
                        save(output/'result.json',result)
                        return result
                def subprocess_fixture(command,cwd,log):
                    calls.append(log.name)
                    if log.name=='lesson_extract_command':
                        save(output/'lesson_candidate/result.json',dict(status='CANDIDATE',
                            lesson=str(self.root/'draft.json')))
                    elif log.name=='candidate_01_command':
                        save(output/'candidate_01/result.json',dict(status='REGISTERED',
                            candidate_id='synthetic_admitted',admission=dict(admitted=True),source_unchanged=True))
                        save(output/'candidate_01/skills_frozen.json',skills)
                    elif log.name=='lesson_consolidate_command':
                        if failure_mode=='process_error':return dict(exit_code=1)
                        save(output/'lesson_finalized/result.json',dict(status='REJECTED'))
                    else:self.fail('unexpected subprocess '+log.name)
                    return dict(exit_code=0)
                fake_module=SimpleNamespace(python_generation_packet=lambda *a:synthesis())
                with patch.object(runner,'LearningCycleWorkflow',LearningStageOnly), \
                     patch.object(runner,'run_recorded',side_effect=subprocess_fixture), \
                     patch.dict('sys.modules',{'run_python_skill_cycle':fake_module}):
                    result=runner.execute_cycle(output,plan,base/'policy.json',
                        source_fixture=base/'fixture.json',target_fixture=base/'fixture.json',lesson_learning=True)
                self.assertEqual(result['status'],'NO_NEW_SKILL')
                self.assertEqual(result['learning_error'],'post_validation_lesson_not_published')
                self.assertEqual(result['new_ids'],[])
                self.assertEqual(json.loads((output/'library_for_next_episode.json').read_text())['entries'],[])
                self.assertEqual(json.loads((output/'lessons_for_next_episode.json').read_text())['entries'],[])
                self.assertEqual(calls,['lesson_extract_command','candidate_01_command','lesson_consolidate_command'])

    def test_mismatched_admission_scope_skips_generation_and_preserves_observation(self):
        """A real prepare/learn routing test with synthetic source observations."""
        from tools import run_hierarchical_cycle as runner
        with tempfile.TemporaryDirectory(prefix='lesson_scope_gate_',dir=runner.ROOT/'runs') as tmp:
            base=Path(tmp);output=base/'cycle'
            plan=json.loads((runner.ROOT/'configs/automatic_learning_cycle_v1.json').read_text(encoding='utf-8'))
            plan.update(initial_library=None,initial_memory=None)
            config=policy();config['enabled']=False
            save(base/'policy.json',config);save(base/'fixture.json',{})
            manifest=base/'prepared.json'
            save(manifest,dict(cases=[dict(phase='admission',trigger_kind='path_failed')]))
            prep=dict(status='PREPARED',selected=dict(trigger_kind='stability_failed',evidence_id='synthetic-source'),
                manifest_path=str(manifest))
            def freeze_fixture(inputs,extra,dest):
                save(dest,dict(fixture=True,record_count=1))
                return dict(record_count=1)
            class PrepareAndLearnOnly:
                def __init__(self,folder,identity,stages):self.stages=stages
                def run(self):
                    source=dict(episode_dir=str(base),artifacts=dict(memory=str(base/'fixture.json')))
                    prepared=self.stages['prepare'](dict(source=source))
                    learned=self.stages['learn'](dict(prepare=prepared))
                    result=dict(prepared=prepared,learned=learned)
                    save(output/'result.json',result)
                    return result
            with patch.object(runner,'LearningCycleWorkflow',PrepareAndLearnOnly), \
                 patch.object(runner,'prepare_episode_learning',return_value=prep), \
                 patch.object(runner,'freeze_cumulative_memory',side_effect=freeze_fixture), \
                 patch.object(runner,'run_recorded',side_effect=AssertionError('NO_SUBPROCESS')) as command:
                result=runner.execute_cycle(output,plan,base/'policy.json',
                    source_fixture=base/'fixture.json',target_fixture=base/'fixture.json',lesson_learning=True)
            self.assertEqual(result['prepared']['status'],'UNSUPPORTED_VALIDATION_SCOPE')
            self.assertEqual(result['prepared']['selected'],prep['selected'])
            self.assertTrue(Path(result['prepared']['memory']).is_file())
            self.assertEqual(result['learned']['status'],'NO_NEW_SKILL')
            self.assertEqual(result['learned']['candidates'],[])
            command.assert_not_called()


if __name__=='__main__':unittest.main()
