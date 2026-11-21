"""Real saved input replay plus synthetic process-boundary tests; no new physics/API."""
from copy import deepcopy
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from tools.run_luna_experiment import read,write,sha
from tools.python_recorded_clock import recorded_seconds,replay_request,replay_candidate
from tools.run_skill_discovery import source_case
from tools.run_python_skill_cycle import python_generation_packet,attach_debug_inputs
from orchestrator.python_skill_preflight import preflight_rows
from orchestrator.python_skill_revision import audit_revision
from worker.task_memory import digest

ROOT=Path(__file__).absolute().parents[1]
LIVE=ROOT/'runs/multitask_roles_live_20261106/live'
MANIFEST=ROOT/'runs/multitask_roles_20261105/prepared/manifest.json'


class RecordedClockTests(unittest.TestCase):
    def setUp(self):
        self.manifest=read(MANIFEST)
        self.prepared=[source_case(s) for s in self.manifest['cases']]
        self.proposal=read(LIVE/'candidate_01/provider_proposal.json')
        t=tempfile.TemporaryDirectory(prefix='clock_UNIT_',dir=ROOT/'runs')
        self.addCleanup(t.cleanup);self.root=Path(t.name)

    def test_original_nonlate_observation_is_not_reclassified_by_30s_default(self):
        packet=python_generation_packet(self.manifest,'clock-test')
        old=attach_debug_inputs(packet,self.prepared,'clock-test')
        new=attach_debug_inputs(packet,self.prepared,'clock-test',recorded_clock=True)
        for a,b,item in zip(preflight_rows(old),preflight_rows(new),self.prepared):
            self.assertGreater(item['raw']['elapsed_seconds'],30)
            self.assertLess(item['raw']['elapsed_seconds'],recorded_seconds(item))
            self.assertTrue(a['current_observation_request']['episode_feedback'][0]['deadline_exceeded'])
            self.assertFalse(b['current_observation_request']['episode_feedback'][0]['deadline_exceeded'])
            self.assertFalse(b['current_observation_request']['path_stage_observations'][0]['deadline_exceeded'])
            self.assertEqual(a['original_trigger'],b['original_trigger'])

    def test_actual_unchanged_model_code_passes_corrected_execution_contract(self):
        packet=attach_debug_inputs(python_generation_packet(self.manifest,'clock'),self.prepared,'clock',recorded_clock=True)
        packet['context']['python_revision_contract_version']=2
        audit=audit_revision(self.proposal['program'],packet,None)
        self.assertTrue(audit['accepted'])
        self.assertTrue(all(c['proposed_action'] is not None for c in audit['checks']))

    def test_recorded_budget_must_match_historical_deadline_and_report_hash(self):
        for seconds in (0,30,601,float('nan'),True):
            p=deepcopy(self.prepared[0]);p['context']['remaining_wall_seconds']=seconds
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):recorded_seconds(p)
        p=deepcopy(self.prepared[0]);p['source_report_sha256']='0'*64
        with self.assertRaises(ValueError):recorded_seconds(p)

    def test_historically_late_observation_stays_late(self):
        p=deepcopy(self.prepared[0]);p['context']['remaining_wall_seconds']=30
        report=read(Path(p['spec']['report']));report['attempts'][p['spec']['attempt_index']-1]['deadline_exceeded']=True
        # Deliberate synthetic report fixture; never counted as real new evidence.
        path=self.root/'synthetic_report.json';write(path,report)
        p['spec']['report']=str(path);p['source_report_sha256']=sha(path)
        request=replay_request(p,'clock','clock:e')
        self.assertTrue(request['episode_feedback'][0]['deadline_exceeded'])

    def test_replay_checks_actual_provenance_same_manifest_and_same_ledger(self):
        proposal,generation,binding=replay_candidate(LIVE/'candidate_01',self.manifest,LIVE/'learner.sqlite','gpt-6-luna')
        self.assertEqual(proposal,self.proposal)
        self.assertEqual(binding['new_api_calls'],0)
        self.assertEqual(generation['origin'],'live_llm')
        for ledger,model,manifest in [(self.root/'other.sqlite','gpt-6-luna',self.manifest),
                (LIVE/'learner.sqlite','other',self.manifest),(LIVE/'learner.sqlite','gpt-6-luna',{})]:
            with self.assertRaises(ValueError):replay_candidate(LIVE/'candidate_01',manifest,ledger,model)

    def test_fresh_case_budget_remains_bounded_by_existing_worker(self):
        from tools.run_skill_discovery import run_case
        for seconds in (46,0,float('nan'),True):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                run_case(self.prepared[0],dict(case_wall_seconds=seconds),{},self.root,'test')

    def test_bounded_fresh_trigger_and_feedback_use_same_clock(self):
        import tools.run_skill_discovery as source
        from orchestrator.skill_case import request_after_trigger as original
        seen=[]
        def fake_physics(packet,folder):
            seen.append(packet);(folder/'physics').mkdir(parents=True)
            raw=deepcopy(self.prepared[0]['raw']);write(folder/'physics/result.json',raw);return raw
        case=dict(case_id='mock_clock',source_task='02628',case_wall_seconds=45)
        with patch.object(source,'actual_action',side_effect=fake_physics), \
             patch.object(source,'request_after_trigger',wraps=original) as feedback, \
             patch.object(source,'step_program',return_value={'response':None}):
            source.run_case(self.prepared[0],case,self.proposal['program'],self.root,'mock_clock')
        self.assertEqual(seen[0]['context']['remaining_wall_seconds'],45)
        self.assertEqual(feedback.call_args.kwargs['wall_seconds'],45)

    def test_negative_historical_status_never_used_as_model_error(self):
        # The earlier no-trigger is a host input bug, not evidence of poor LLM
        # selection. Preserve its artifacts and replay only the original program.
        value=read(LIVE/'candidate_01/development_preflight.json')
        self.assertTrue(all(c['runtime_reason']=='no_trigger' for c in value['checks']))
        self.assertFalse(value['accepted'])
        self.assertEqual(digest(self.proposal['program']),value['program_sha256'])

    def test_revalidation_reuses_one_ledger_skips_lesson_and_new_generation(self):
        from tools.run_multitask_learning import learn
        from types import SimpleNamespace
        calls=[]
        def invoke(name,tool,args,limit):
            calls.append(args);dest=Path(args[args.index('--run-dir')+1]);dest.mkdir()
            write(dest/'result.json',dict(status='NONAPPLICABILITY_REJECTED'));return 1
        result=learn(self.root,invoke,SimpleNamespace(summary=lambda:{}),
            learning_inputs=(MANIFEST,ROOT/'runs/multitask_roles_live_20261106/prepared_v2/lesson_packet.json'),
            replay_from=LIVE/'candidate_01',budget_output=LIVE)
        self.assertEqual(len(calls),1);self.assertIn('--replay-generated-from',calls[0])
        self.assertIn('--recorded-trigger-clock',calls[0])
        self.assertEqual(calls[0][calls[0].index('--ledger')+1],LIVE/'learner.sqlite')
        self.assertEqual(result['evaluation'],'NOT_RUN')

    def test_only_real_abstention_requests_special_feedback(self):
        from tools.run_multitask_learning import learn
        from types import SimpleNamespace
        for reason in ('no_trigger','python_abstained','python_execution_rejected'):
            dest=self.root/reason;dest.mkdir();calls=[]
            def invoke(name,tool,args,limit):
                calls.append(args);out=Path(args[args.index('--run-dir')+1]);out.mkdir()
                write(out/'result.json',dict(status='PREFLIGHT_REJECTED'))
                write(out/'development_preflight.json',dict(checks=[dict(runtime_reason=reason,runtime_error=None)]))
                return 1
            learn(dest,invoke,SimpleNamespace(summary=lambda:{}),
                learning_inputs=(MANIFEST,ROOT/'runs/multitask_roles_live_20261106/prepared_v2/lesson_packet.json'),
                replay_from=LIVE/'candidate_01',budget_output=LIVE)
            self.assertEqual(len(calls),2)
            self.assertEqual('--abstention-feedback' in calls[1],reason=='python_abstained')
            self.assertNotIn('--replay-generated-from',calls[1])

    def test_unreproduced_trigger_cannot_spend_a_code_revision(self):
        from tools.run_multitask_learning import learn
        from types import SimpleNamespace
        calls=[]
        def invoke(name,tool,args,limit):
            calls.append(name);out=Path(args[args.index('--run-dir')+1]);out.mkdir()
            write(out/'result.json',dict(status='NOT_REGISTERED',cases=[
                dict(case_id='source_02628',outcome='ERROR'),
                dict(case_id='source_01289',outcome='NOT_TRIGGERED')]))
            return 1
        learn(self.root,invoke,SimpleNamespace(summary=lambda:{}),
            learning_inputs=(MANIFEST,ROOT/'runs/multitask_roles_live_20261106/prepared_v2/lesson_packet.json'),
            replay_from=LIVE/'candidate_01',budget_output=LIVE)
        self.assertEqual(calls,['candidate_01'])


if __name__=='__main__':unittest.main()
