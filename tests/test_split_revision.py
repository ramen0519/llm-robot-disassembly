"""API-free allocation/provenance tests; saved physics is NOT new physics."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.split_revision_budget import SplitRevisionBudget
from orchestrator.python_skill_preflight import audit_program
from tools.python_revision_parent import historical_packet, classify_candidate
from tools.run_luna_experiment import read, write, sha
from tools import run_split_revision_cycle as cycle
from tools.run_python_skill_cycle import refinement_packet

ROOT = cycle.ROOT
OLD = ROOT/'runs/python_revision_live_20261021/live_cycle/candidate_02'
PREP = ROOT/'runs/split_revision_prep_20261021'


class BudgetTests(unittest.TestCase):
    def test_format_corrections_protect_second_physical_revision(self):
        b=SplitRevisionBudget()
        kinds=[]
        for outcome in ('contract_rejected','measured_debug_failure','contract_rejected','registered'):
            kinds.append(b.allocate()); b.observe(outcome)
        self.assertEqual(kinds,['physical_revision','contract_correction','physical_revision','contract_correction'])
        self.assertIsNone(b.allocate())
        self.assertEqual(b.used,dict(physical_revision=2,contract_correction=2))

    def test_two_physical_failures_cannot_spend_correction_slots(self):
        b=SplitRevisionBudget()
        for _ in range(2):
            self.assertEqual(b.allocate(),'physical_revision'); b.observe('measured_debug_failure')
        self.assertIsNone(b.allocate())
        self.assertEqual(b.used['contract_correction'],0)

    def test_repeated_contract_errors_stop_with_physical_slot_unspent(self):
        b=SplitRevisionBudget()
        for _ in range(3):
            self.assertIsNotNone(b.allocate()); b.observe('contract_rejected')
        self.assertIsNone(b.allocate())
        self.assertEqual(b.used['physical_revision'],1)

    def test_pending_candidate_cannot_duplicate_allocation(self):
        b=SplitRevisionBudget(); b.allocate()
        with self.assertRaises(ValueError): b.allocate()
        b.observe('stop'); self.assertIsNone(b.allocate())

    def test_invalid_counts_and_unsolicited_outcome(self):
        for n in (0,True,4,-1):
            with self.assertRaises(ValueError): SplitRevisionBudget(n,2)
        with self.assertRaises(ValueError): SplitRevisionBudget().observe('registered')


class SavedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.manifest=read(OLD/'precommitted_cases.json')
        self.declaration=read(PREP/'historical_parent.json')
        self.packet=read(OLD/'generation_packet.json')
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.network=net.start(); self.addCleanup(net.stop); self.addCleanup(self.network.assert_not_called)

    def test_actual_old_physical_failure_is_bound_and_repeated_action_rejected(self):
        packet=historical_packet(self.packet,self.declaration,self.manifest)
        self.assertEqual([r['case_id'] for r in packet['validation_feedback']],['episode_debug_000001'])
        row=packet['validation_feedback'][0]
        self.assertFalse(row['recovery']['path_success'])
        self.assertIsNone(row['recovery']['stability_success'])
        audit=audit_program(packet['previous_program'],packet['validation_feedback'])
        self.assertTrue(audit['checks'][0]['repeated_failed_action'])
        self.assertFalse(audit['accepted'])
        self.assertFalse(packet['context']['historical_debug_parent']['authorization_inherited'])

    def test_parent_and_validation_hashes_cannot_change(self):
        for name,value in [('result_sha256','0'*64),('manifest_sha256','0'*64),
                           ('debug_record_sha256',['0'*64]),('candidate_id','0'*64),
                           ('authorization_inherited',True)]:
            dec=deepcopy(self.declaration); dec[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):
                historical_packet(self.packet,dec,self.manifest)

    def test_e_drive_rejected_before_any_filesystem_read(self):
        dec=deepcopy(self.declaration); dec['run_dir']='E:/DO_NOT_ACCESS'
        with patch.object(Path,'open',side_effect=AssertionError('NO_FILESYSTEM_READ')):
            with self.assertRaisesRegex(ValueError,'excluded_path'):
                historical_packet(self.packet,dec,self.manifest)

    def test_original_same_ledger_guard_is_not_weakened(self):
        with self.assertRaisesRegex(ValueError,'same_approved_ledger'):
            refinement_packet(self.packet,OLD,self.manifest,live=True,ledger=PREP/'new_unused.sqlite')

    def test_measured_debug_failure_qualifies_but_admission_only_failure_does_not(self):
        ledger=Path(read(OLD/'config.json')['ledger'])
        result=read(OLD/'result.json')
        self.assertEqual(classify_candidate(OLD,result,ledger),'measured_debug_failure')
        packet=historical_packet(self.packet,self.declaration,self.manifest)
        packet['validation_feedback'][0]['outcome']='PASS'
        with patch('tools.run_python_skill_cycle.refinement_packet',return_value=packet):
            self.assertEqual(classify_candidate(OLD,result,ledger),'stop')

    def test_timeout_does_not_authorize_physical_revision(self):
        packet=historical_packet(self.packet,self.declaration,self.manifest)
        packet['validation_feedback'][0]['recovery']['deadline_exceeded']=True
        with patch('tools.run_python_skill_cycle.refinement_packet',return_value=packet):
            self.assertEqual(classify_candidate(OLD,read(OLD/'result.json'),Path('UNIT_LEDGER')),'stop')

    def test_declared_conditions_cannot_change(self):
        manifest=deepcopy(self.manifest);manifest['cases'][0]['attempt_index']=999
        with self.assertRaisesRegex(ValueError,'conditions_changed'):
            historical_packet(self.packet,self.declaration,manifest)

    def test_admission_outcomes_not_sent_in_actual_preview(self):
        preview=read(PREP/'configured_preview.json')
        body=str(preview['body'])
        self.assertIn('episode_debug_000001',body)
        self.assertNotIn('beam_admission_yn',body)
        self.assertNotIn('beam_admission_xn',body)
        self.assertIn('measured_failed_previous_repair',body)


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='split_UNIT_',dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup);self.output=Path(self.temp.name)
        self.protocol=dict(physical_revisions=2,contract_corrections=2)
        self.paths={k:Path('UNIT_'+k) for k in ('parent','manifest','draft_lesson','prior_library')}
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.network=net.start();self.addCleanup(net.stop);self.addCleanup(self.network.assert_not_called)

    def loop(self,statuses,outcomes):
        calls=[]
        def invoke(label,tool,args):
            calls.append(args)
            folder=Path(args[args.index('--run-dir')+1]);folder.mkdir()
            status=statuses[len(calls)-1]
            write(folder/'result.json',dict(status=status,source_unchanged=True))
            return 0 if status=='REGISTERED_SCOPED' else 1
        with patch.object(cycle,'classify_candidate',side_effect=outcomes):
            result=cycle.run_candidates(self.output,self.protocol,self.paths,dict(learner='UNIT_cfg'),invoke)
        return result,calls

    def test_controller_splits_slots_and_does_not_forget_physical_reference(self):
        result,calls=self.loop(['GENERATION_REJECTED','NOT_REGISTERED','PREFLIGHT_REJECTED','REGISTERED_SCOPED'],
            ['contract_rejected','measured_debug_failure','contract_rejected','registered'])
        self.assertEqual(len(calls),4)
        self.assertIn('--historical-debug-parent',calls[1])
        self.assertIn('--generation-feedback-from',calls[1])
        self.assertIn('--refine-from',calls[2]); self.assertIn('--refine-from',calls[3])
        self.assertEqual(result['allocation']['used'],dict(physical_revision=2,contract_correction=2))
        self.assertIsNotNone(result['admitted_folder'])

    def test_controller_exhausts_only_physical_budget_on_two_physics_failures(self):
        result,calls=self.loop(['NOT_REGISTERED']*2,['measured_debug_failure']*2)
        self.assertEqual(len(calls),2);self.assertIsNone(result['admitted_folder'])

    def test_controller_transport_failure_stops(self):
        result,calls=self.loop(['GENERATION_REJECTED'],['stop'])
        self.assertEqual(len(calls),1);self.assertIsNone(result['admitted_folder'])

    def test_real_classification_429_is_not_a_correction(self):
        folder=self.output/'rejected';folder.mkdir(); ledger=self.output/'ledger.sqlite'
        write(folder/'config.json',dict(mode='live',ledger=str(ledger)))
        write(folder/'provider_proposal.json',dict(status='rejected',fallback_reason='http_429',
              metadata=dict(llm_live=True)))
        result=dict(status='GENERATION_REJECTED',physical_cases_not_run=True)
        write(folder/'result.json',result)
        self.assertEqual(classify_candidate(folder,result,ledger),'stop')

    def test_missing_result_does_not_retry(self):
        calls=[]
        def invoke(*args):calls.append(args);return 2
        result=cycle.run_candidates(self.output,self.protocol,self.paths,dict(learner='UNIT_cfg'),invoke)
        self.assertEqual(len(calls),1);self.assertIsNone(result['admitted_folder'])

    def test_disabled_policy_prevents_all_commands(self):
        with patch.object(cycle,'run_recorded',side_effect=AssertionError('NO_COMMAND')):
            with self.assertRaisesRegex(ValueError,'explicit_live_permission'):
                cycle.execute(self.output/'unused',{},PREP/'policy_proposed_disabled.json',allow_live=True)

    def execute_fixture(self, admitted=True, finalize=True, new_pass=True):
        """Only control flow is injected; never label this as LLM/physics success."""
        from contextlib import ExitStack
        policy=read(PREP/'policy_proposed_disabled.json');policy['enabled']=True
        policy_path=self.output/'policy.json';write(policy_path,policy)
        folder=self.output/'admitted';folder.mkdir()
        write(folder/'result.json',dict(candidate_id='UNIT_PROGRAM'))
        final_lesson=self.output/'lesson.json';write(final_lesson,dict(unit=True))
        protocol=read(PREP/'protocol.json');protocol['source_hashes']=cycle.runtime_hashes()
        paths={k:Path(v['path']) for k,v in protocol['files'].items()}
        calls=[]
        def command(argv,cwd,logs):
            calls.append(argv);target=Path(argv[argv.index('--run-dir')+1]);target.mkdir(parents=True)
            if 'run_lesson_discovery.py' in argv[3]:
                write(target/'result.json',dict(status='CANDIDATE' if finalize else 'REJECTED',
                                               lesson=str(final_lesson)))
                return dict(exit_code=0 if finalize else 1)
            write(target/'audit.json',dict(validation_ok=True,actual_physics=True,status='PASS',plan_success=True))
            write(target/'hierarchy_audit.json',dict(library_unchanged=True,lesson_library_unchanged=True,
                physics_bindings=[dict(skill_id='UNIT_PROGRAM',accepted=new_pass),dict(skill_id=None,accepted=True)],
                memory_visible_requests=3,memory_cited_requests=1))
            write(target/'host_preservation.json',dict(unchanged=True))
            return dict(exit_code=0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(cycle,'validate_protocol',return_value=(SplitRevisionBudget(),paths)))
            stack.enter_context(patch.object(cycle,'run_candidates',return_value=dict(admitted_folder=str(folder) if admitted else None)))
            stack.enter_context(patch.object(cycle,'run_recorded',side_effect=command))
            stack.enter_context(patch('orchestrator.skill_library_merge.freeze_skill_union',return_value=dict(entries=[{}])))
            stack.enter_context(patch('orchestrator.lesson_memory.freeze_lessons',return_value=dict(entries=[{}])))
            stack.enter_context(patch('orchestrator.lesson_memory.freeze_bundle'))
            stack.enter_context(patch('orchestrator.explicit_action_memory.freeze_cumulative_memory'))
            result=cycle.execute(self.output/'executed',protocol,policy_path,allow_live=True)
        return result,calls

    def test_no_registration_skips_finalization_and_target(self):
        result,calls=self.execute_fixture(admitted=False)
        self.assertEqual(calls,[]);self.assertEqual(result['target'],'NOT_RUN')

    def test_failed_finalization_skips_target(self):
        result,calls=self.execute_fixture(finalize=False)
        self.assertEqual(len(calls),1);self.assertEqual(result['target'],'NOT_RUN')

    def test_published_skill_uses_free_target_and_counts_physics_acceptance(self):
        result,calls=self.execute_fixture()
        self.assertEqual(len(calls),2)
        self.assertEqual(result['new_skill_attempts'],1);self.assertEqual(result['new_skill_passes'],1)
        self.assertEqual(result['memory_cited_requests'],1)
        self.assertIn('--lesson-library',calls[-1]);self.assertIn('--memory',calls[-1])
        self.assertNotIn('--provider-fixture',calls[-1]);self.assertNotIn('--fixture',calls[-1])

    def test_plan_success_does_not_mean_skill_success(self):
        result,calls=self.execute_fixture(new_pass=False)
        self.assertTrue(result['target_plan_success']);self.assertEqual(result['new_skill_attempts'],1)
        self.assertEqual(result['new_skill_passes'],0)


if __name__=='__main__':unittest.main()
