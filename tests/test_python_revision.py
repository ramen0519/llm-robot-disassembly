"""Offline synthetic contracts and saved real-output replay; NOT new model/physics."""
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.test_python_preflight import policy, row
from tests.test_skill_discovery import synthesis, config, decision, raw_response
from tests.test_lesson_pipeline import claim
from tests.test_skill_validation import save
from orchestrator.lesson_memory import make_lesson
from orchestrator.lesson_discovery import LessonDiscoveryService
from orchestrator.python_skill_preflight import preflight_rows, audit_program, generation_contract
from orchestrator.python_skill_revision import (enabled, physical_choice, validate_predictions,
    compact_view, audit_revision, FLAG)
from worker.explicit_action_contract import digest
import tools.run_python_skill_cycle as cycle
import tools.run_hierarchical_cycle as hierarchy

ROOT = Path(__file__).absolute().parents[1]


def packet():
    p = synthesis()
    p.update(operation='refine', previous_program=policy(), validation_feedback=[row()])
    p['context'].update(lesson_stage='program',
        lesson=make_lesson(claim(p), p, dict(origin='manual_authored')), **{FLAG: 1})
    return p


def predictions(p, program):
    # Only a TEST oracle. Production NEVER fabricates predictions on behalf of a model.
    audit = audit_program(program, preflight_rows(p))
    return [dict(case_id=c['case_id'], predicted_action=physical_choice(c['proposed_action']),
        relationship='abstain' if c['proposed_action'] is None else
        'same_as_failed_repair' if c['repeated_failed_action'] else
        'changed_from_failed_repair' if c['observed_failed_action'] is not None else 'no_failed_repair')
        for c in audit['checks']]


class PythonRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='python_revision_OFFLINE_', dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        self.net = net.start(); self.addCleanup(net.stop); self.addCleanup(self.net.assert_not_called)
        self.p = packet()

    def service(self, transport=None, **overrides):
        s = LessonDiscoveryService(config(**overrides), self.root/(str(len(list(self.root.iterdir())))+'.sqlite'), transport)
        self.addCleanup(s.close)
        return s

    def test_opt_in_version_is_strict_and_old_body_unchanged_shape(self):
        p = deepcopy(self.p); del p['context'][FLAG]
        body = self.service(enabled=False,max_calls=0,max_usd=0).preview(p)['body']
        self.assertNotIn('behavior_checks',body['text']['format']['schema']['properties'])
        self.assertIn('python_runtime_contract',json.loads(body['input'][0]['content'])['context'])
        for bad in (True, 0, '1', 5, None):
            p['context'][FLAG]=bad
            with self.assertRaises(ValueError): enabled(p)

    def test_compact_input_preserves_original_trigger_and_failed_repair_separately(self):
        before=deepcopy(self.p)
        body=self.service(enabled=False,max_calls=0,max_usd=0).preview(self.p)['body']
        sent=json.loads(body['input'][0]['content'])
        c=sent['context']; case=c['revision_cases'][0]
        example=c['runtime_contract']['examples'][0]
        self.assertEqual(example['runtime_input_example']['last']['choice']['direction_id'],'xp')
        self.assertEqual(case['previous_repair']['chosen']['direction_id'],'xn')
        self.assertTrue(case['measured_failed_previous_repair'])
        self.assertEqual(sent['evidence'],self.p['evidence'])
        self.assertNotIn('validation_feedback',sent)
        self.assertEqual(self.p,before)
        self.assertEqual(example['runtime_input_example'],generation_contract(self.p)['examples'][0]['runtime_input_example'])

    def test_success_late_error_unmeasured_not_promoted_to_failed_repair(self):
        for outcome in ('PASS','LATE','ERROR'):
            p=deepcopy(self.p);p['validation_feedback'][0]['outcome']=outcome
            new=policy()
            result=audit_revision(new,p,predictions(p,new))
            self.assertTrue(result['accepted'])
            self.assertEqual(result['checks'][0]['actual_relationship'],'no_failed_repair')

    def test_truthful_repeated_action_rejected_even_with_renamed_source(self):
        new=policy('    renamed = 22\n',citations='[]')
        result=audit_revision(new,self.p,predictions(self.p,new))
        self.assertFalse(result['accepted'])
        self.assertIn('unchanged_failed_debug_action',result['checks'][0]['errors'])
        self.assertEqual(result['physical_calls'],0)

    def test_claimed_change_does_not_hide_actual_repeat(self):
        claims=predictions(self.p,policy(direction='yp'))
        result=audit_revision(policy(),self.p,claims)
        self.assertFalse(result['accepted'])
        self.assertIn('predicted_action_differs_from_execution',result['checks'][0]['errors'])
        self.assertIn('predicted_relationship_differs_from_execution',result['checks'][0]['errors'])

    def test_changed_action_passes_only_development_gate(self):
        new=policy(direction='yp');result=audit_revision(new,self.p,predictions(self.p,new))
        self.assertTrue(result['accepted'])
        self.assertFalse(result['new_physical_success']);self.assertFalse(result['candidate_pruning'])

    def test_abstain_and_runtime_error_never_register_success(self):
        for source in ('def propose(ctx):\n    return None\n','def propose(ctx):\n    return 1 / 0\n'):
            new=dict(language='python_policy_v1',source=source)
            result=audit_revision(new,self.p,predictions(self.p,new))
            self.assertFalse(result['accepted'])
            self.assertTrue(result['checks'][0]['prediction_matches_execution'])

    def test_missing_duplicate_unknown_and_illegal_predictions_fail_closed(self):
        good=predictions(self.p,policy(direction='yp'))
        for wrong in (None,[],good+good,[dict(good[0],case_id='not_declared')],
            [dict(good[0],predicted_action=dict(good[0]['predicted_action'],part_id='unknown'))]):
            self.assertFalse(audit_revision(policy(direction='yp'),self.p,wrong)['accepted'])

    def test_pose_alias_and_citation_changes_do_not_change_physical_identity(self):
        req=self.p['validation_feedback'][0]['current_observation_request']
        req['pose_catalog'].append(dict(pose_id='alias',matrix=deepcopy(req['pose_catalog'][0]['matrix'])))
        new=policy(pose='"alias"',citations='[]');claims=predictions(self.p,new)
        claims[0]['predicted_action']['pose_id']=req['pose_catalog'][0]['pose_id']
        result=audit_revision(new,self.p,claims)
        self.assertFalse(result['accepted']);self.assertTrue(result['checks'][0]['prediction_matches_execution'])

    def test_admission_inputs_rejected_before_model(self):
        self.p['validation_feedback'][0]['phase']='admission'
        with self.assertRaisesRegex(ValueError,'debug'):
            self.service(enabled=False,max_calls=0,max_usd=0).preview(self.p)

    def test_provider_contract_retains_predictions_and_does_not_execute(self):
        value=decision(self.p);value['program']=policy(direction='yp')
        value['behavior_checks']=predictions(self.p,value['program'])
        transport=Mock(return_value=raw_response(value))
        with patch.dict(os.environ,DISCOVERY_FIXTURE_KEY='fixture-not-a-real-key'):
            result=self.service(transport).propose(self.p)
        self.assertEqual(result['status'],'proposed',result)
        self.assertEqual(result['behavior_checks'],value['behavior_checks'])
        self.assertFalse(result['metadata']['python_executed']);transport.assert_called_once()

    def test_invalid_predictions_preserve_source_without_http_retry(self):
        value=decision(self.p);value['program']=policy();value['behavior_checks']=[]
        transport=Mock(return_value=raw_response(value))
        with patch.dict(os.environ,DISCOVERY_FIXTURE_KEY='fixture-not-a-real-key'):
            result=self.service(transport).propose(self.p)
        self.assertEqual(result['fallback_reason'],'invalid_revision_predictions')
        self.assertIn('python_rejected_source',result['metadata']);transport.assert_called_once()

    def test_actual_saved_code_repeats_despite_changed_source(self):
        prior=ROOT/'runs/lesson_lifecycle_live_20261021/live_cycle'
        p=json.loads((prior/'candidate_02/generation_packet.json').read_text(encoding='utf-8'))
        p['context'][FLAG]=1
        first=json.loads((prior/'candidate_01/provider_proposal.json').read_text(encoding='utf-8'))['program']
        second=json.loads((prior/'candidate_02/provider_proposal.json').read_text(encoding='utf-8'))['program']
        self.assertNotEqual(digest(first),digest(second))
        a=predictions(p,first);b=predictions(p,second)
        self.assertEqual(a[0]['predicted_action'],b[0]['predicted_action'])
        result=audit_revision(second,p,b)
        self.assertFalse(result['accepted'])
        self.assertTrue(result['checks'][0]['repeated_failed_action'])
        body=self.service(enabled=False,max_calls=0,max_usd=0).preview(p)['body']
        self.assertLess(len(json.dumps(body)),65000)

    def test_runner_prediction_failure_precedes_physics_and_binds_next_candidate(self):
        p=deepcopy(self.p);value=decision(p);value['program']=policy()
        value['behavior_checks']=predictions(p,policy(direction='yp'))
        proposal=dict(value,status='proposed',origin='actual_llm',metadata={'llm_live':True})
        manifest=dict(cases=[dict(case_id=k,phase=phase,source_task='fixture',trigger_kind='path_failed')
            for k,phase in [('synthetic_debug','debug'),('hidden1','admission'),('hidden2','admission')]])
        save(self.root/'manifest.json',manifest);save(self.root/'lesson.json',p['context']['lesson'])
        save(self.root/'config.json',dict(config(),api_key_env='OPENAI_API_KEY'))
        req=p['validation_feedback'][0]['current_observation_request'];out=self.root/'candidate'
        def prepared(spec):return dict(spec=spec,request=req,raw={'input_hashes':{}},expected_source_hashes={})
        with patch.object(cycle,'source_case',side_effect=prepared), \
             patch.object(cycle,'python_generation_packet',return_value=p), \
             patch.object(cycle,'attach_debug_inputs',side_effect=lambda p,*_:p), \
             patch.object(cycle,'require_live_config'), \
             patch('local_api_key.key_environment',return_value=nullcontext()), \
             patch.object(LessonDiscoveryService,'propose',return_value=proposal), \
             patch.object(cycle,'run_case',side_effect=AssertionError('NO_PHYSICS')) as physics:
            code=cycle.main(['--run-dir',str(out),'--manifest',str(self.root/'manifest.json'),
                '--lesson',str(self.root/'lesson.json'),'--mode','live','--allow-live-api',
                '--llm-config',str(self.root/'config.json'),'--ledger',str(self.root/'shared.sqlite'),
                '--python-revision-contract'])
        self.assertEqual(code,1);physics.assert_not_called()
        result=json.loads((out/'result.json').read_text());self.assertEqual(result['status'],'PREFLIGHT_REJECTED')
        refined=cycle.refinement_packet(synthesis(),out,manifest)
        self.assertIn('predicted_action_differs_from_execution',json.dumps(refined))
        self.assertNotIn('hidden1',json.dumps(refined))
        save(out/'provider_proposal.json',dict(proposal,behavior_checks=[]))
        with self.assertRaisesRegex(ValueError,'binding_mismatch|generation_response_changed'):
            cycle.refinement_packet(synthesis(),out,manifest)

    def test_lifecycle_routes_prediction_rejection_to_next_allocated_candidate(self):
        from tests.test_hierarchical_planner import policy as hierarchy_policy
        plan=json.loads((ROOT/'configs/automatic_learning_cycle_v1.json').read_text())
        plan.update(initial_library=None,initial_memory=None,max_proposals=2)
        save(self.root/'policy.json',hierarchy_policy(enabled=True))
        save(self.root/'manifest.json',{})
        out=self.root/'cycle';calls=[]
        # Reuse the SAVED real draft's bound provenance; no new generation.
        draft_result=json.loads((ROOT/'runs/lesson_lifecycle_live_20261021/live_cycle/lesson_candidate/result.json').read_text())
        saved_lesson=json.loads(Path(draft_result['lesson']).read_text(encoding='utf-8'))
        save(self.root/'lesson.json',saved_lesson)
        class LearnOnly:
            def __init__(self,folder,identity,stages):
                self.stages=stages
                assert identity['python_revision_contract_version']==1
            def run(self):
                result=self.stages['learn']({'prepare':{'status':'PREPARED','manifest':str(self_root/'manifest.json')}})
                save(out/'result.json',result);return result
        self_root=self.root
        def command(argv,cwd,logs):
            calls.append(argv);folder=Path(argv[argv.index('--run-dir')+1])
            if folder.name=='lesson_candidate':
                save(folder/'result.json',dict(status='CANDIDATE',lesson=str(self.root/'lesson.json')))
                return {'exit_code':0}
            save(folder/'generation_packet.json',self.p)
            save(folder/'config.json',dict(mode='live',ledger=str(out/'learner.sqlite')))
            save(folder/'provider_proposal.json',dict(status='rejected',fallback_reason='invalid_revision_predictions',
                metadata={'llm_live':True}))
            save(folder/'result.json',dict(status='GENERATION_REJECTED',physical_cases_not_run=True))
            return {'exit_code':1}
        with patch.object(hierarchy,'LearningCycleWorkflow',LearnOnly), \
             patch.object(hierarchy,'run_recorded',side_effect=command), \
             patch('run_python_skill_cycle.python_generation_packet',return_value=deepcopy(self.p)):
            result=hierarchy.execute_cycle(out,plan,self.root/'policy.json',live=True,
                lesson_learning=True,python_revision_contract=True)
        self.assertEqual(len(calls),3)
        self.assertNotIn('--python-revision-contract',calls[0])
        for argv in calls[1:]:self.assertIn('--python-revision-contract',argv)
        self.assertIn('--generation-feedback-from',calls[2])
        self.assertEqual(result['status'],'NO_NEW_SKILL')
        self.assertEqual(len(result['candidates']),2)
        for flag in ('--ledger','--hierarchy-ledger','--hierarchy-policy'):
            self.assertEqual(calls[1][calls[1].index(flag)+1],calls[2][calls[2].index(flag)+1])

    def test_revision_flags_do_not_enable_live_or_fixture_implicitly(self):
        with self.assertRaisesRegex(ValueError,'requires_live_lesson_cycle'):
            hierarchy.execute_cycle(self.root/'absent',{},self.root/'absent.json',python_revision_contract=True)
        for argv in ([],['--lesson','none','--mode','fixture'],['--lesson','none','--lesson-only']):
            with self.assertRaisesRegex(ValueError,'requires_regular_lesson'):
                cycle.main(['--run-dir',str(self.root/'not_created'),'--python-revision-contract',*argv])
        self.assertFalse((self.root/'not_created').exists())


if __name__=='__main__':unittest.main()
