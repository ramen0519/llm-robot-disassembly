"""Synthetic contract tests. No model, network, or physics calls."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from tests.test_hierarchical_planner import request, policy
from tests.test_explicit_action_contract import attach_attempts
from worker.explicit_action_contract import validate_request, canonical_response, action_id, path_signature
from worker.virtual_search_restore import FIELD, snapshot, bind, restored_reference
from worker.explicit_request_budget import bounded_memory_request, capacity_probe
from orchestrator.explicit_action_provider_v5 import project_request_v5, restore_request_v5
from orchestrator.strategy_plan import CONTRACT
from orchestrator.strategy_service import StrategyDecisionService
from orchestrator.strategy_candidates import _support_reference, _scope


def restored_fixture():
    req=validate_request(request()); choice=canonical_response(req,None)['choice']
    attach_attempts(req,[choice]); identity=req['attempted_action_ids'][0]
    event='fixture:accepted:1'
    req['path_stage_observations']=[dict(action_id=identity,source_event_id=event,
        path_success=True,status='PASS',deadline_exceeded=False,backend='ASAP_RedMax')]
    child=dict(state_id='child-fixture',remaining_parts=[p for p in req['remaining_parts'] if p!=choice['part_id']])
    edge=dict(state_id=req['state_id'],event_id=event,action_id=identity,status='PASS',
        accepted_for_transition=True,deadline_exceeded=False,remaining_parts=req['remaining_parts'],choice=choice,
        context=dict(run_id=req['run_id'],parts_rest=child['remaining_parts'],pose_matrix=req['pose_catalog'][0]['matrix']),
        raw=dict(path_success=True,stability_success=True,backend='ASAP_RedMax'))
    seed=snapshot(edge,child,1);req[FIELD]=bind(req,seed)
    return req,seed


class VirtualRestoreTests(unittest.TestCase):
    def setUp(self):
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NO_NETWORK'))
        self.net=net.start();self.addCleanup(net.stop);self.addCleanup(self.net.assert_not_called)

    def test_old_feedback_not_needed_and_no_new_evidence(self):
        req,_=restored_fixture(); before=deepcopy(req)
        self.assertEqual(validate_request(req),req)
        ref=restored_reference(req)
        self.assertEqual(_support_reference(req,ref),(ref['choice']['support_ids'],'worker_restored_pre_action',None))
        self.assertEqual(req['episode_feedback'],[]);self.assertEqual(req['evidence_ids'],[])
        self.assertEqual(req,before)
        legacy=deepcopy(req);legacy.pop(FIELD)
        self.assertEqual(_support_reference(legacy,None)[2],'support_reference_unavailable')

    def test_action_identity_history_and_path_signature_unchanged(self):
        req,_=restored_fixture();legacy=deepcopy(req);legacy.pop(FIELD)
        choice=req[FIELD]['source']['choice']
        self.assertEqual(action_id(req,choice),action_id(legacy,choice))
        self.assertEqual(path_signature(req,choice),path_signature(legacy,choice))
        self.assertEqual(canonical_response(req,None),canonical_response(legacy,None))

    def test_cross_request_state_run_and_pose_or_source_mutation_rejected(self):
        req,_=restored_fixture()
        mutations=[lambda r:r.update(run_id='other'),lambda r:r.update(request_id='other'),
            lambda r:r.update(state_id='other'),lambda r:r[FIELD].update(node_visit_revision=0),
            lambda r:r[FIELD].update(node_visit_revision=True),lambda r:r[FIELD].update(scope='robot'),
            lambda r:r[FIELD].update(returned_from_parts=[]),
            lambda r:r[FIELD]['source'].update(status='UNKNOWN'),
            lambda r:r[FIELD]['source'].update(accepted_for_transition=False),
            lambda r:r[FIELD]['source'].update(deadline_exceeded=True),
            lambda r:r[FIELD]['source'].update(event_id='other'),
            lambda r:r[FIELD]['source'].update(backend='MuJoCo'),
            lambda r:r['pose_catalog'][0]['matrix'][0].__setitem__(3,5)]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                changed=deepcopy(req);mutate(changed)
                with self.assertRaises(ValueError):validate_request(changed)

    def test_visible_original_pass_stays_accepted_and_must_match(self):
        req,_=restored_fixture();s=req[FIELD]['source']
        req['episode_feedback']=[dict(evidence_id=s['event_id'],action_id=s['action_id'],
            status='PASS',accepted_for_transition=True,deadline_exceeded=False,
            path_success=True,stability_success=True,remaining_parts=list(req['remaining_parts']),
            selected_pose_matrix=deepcopy(s['pose_matrix']),choice=deepcopy(s['choice']))]
        req['evidence_ids']=[s['event_id']]
        validate_request(req)
        req['episode_feedback'][0]['accepted_for_transition']=False
        with self.assertRaisesRegex(ValueError,'virtual_search_restore'):validate_request(req)

    def test_v5_and_request_budget_preserve_worker_context(self):
        req,_=restored_fixture()
        bundle,_=project_request_v5(req)
        self.assertEqual(restore_request_v5(bundle),req)
        bounded,_=bounded_memory_request(req,[])
        self.assertEqual(bounded[FIELD],req[FIELD])

    def test_synthetic_history_does_not_keep_real_restore_authority(self):
        req,_=restored_fixture()
        synthetic,audit=capacity_probe(req,attempted_count=4,episode_count=2,memory_count=0)
        self.assertNotIn(FIELD,synthetic)
        self.assertTrue(audit['synthetic_history'])
        self.assertIn(FIELD,req)

    def test_python_skill_context_gets_no_new_restore_authority(self):
        from orchestrator.python_skill_program import build_context
        req,_=restored_fixture();legacy=deepcopy(req);legacy.pop(FIELD)
        self.assertEqual(build_context(req),build_context(legacy))

    def test_file_mailbox_roundtrip_preserves_restore_context(self):
        import tempfile
        import time
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor
        from tests.test_explicit_action_runner import ExplicitActionClient, FileRankServer, runner
        req,_=restored_fixture();received=[]
        service=Mock(timeout_seconds=3)
        def select(value):
            received.append(deepcopy(value))
            return canonical_response(value,'fixture_only')
        service.handle.side_effect=select
        with tempfile.TemporaryDirectory(prefix='restore_ipc_fixture_') as tmp:
            root=Path(tmp)/'ipc'
            client=ExplicitActionClient(root,timeout_seconds=3,allow_test_root=True)
            server=FileRankServer(root,req['run_id'],'pipeline',lambda *a,**k:None,
                allow_test_root=True,rank_service=runner.MailboxAdapter(service))
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(client,req);deadline=time.monotonic()+2
                while not future.done() and time.monotonic()<deadline:
                    server.serve_pending();time.sleep(.01)
                response=future.result(timeout=3)
        self.assertEqual(received,[req])
        self.assertEqual(response,canonical_response(req,'fixture_only'))

    def test_reference_does_not_override_new_unknown_or_error_command(self):
        for status in ('UNKNOWN','ERROR'):
            req,seed=restored_fixture();new=canonical_response(req,None)['choice']
            old=deepcopy(req[FIELD]['source']['choice']);attach_attempts(req,[old,new])
            req['path_stage_observations'].append(dict(action_id=action_id(req,new),
                source_event_id='fixture:new',path_success=None,status=status,deadline_exceeded=True,backend='ASAP_RedMax'))
            req[FIELD]=bind(req,seed)
            validate_request(req)
            self.assertIsNone(restored_reference(req))
            self.assertEqual(_support_reference(req,None)[2],'support_reference_unavailable')

    def test_node_revision_invalidates_anchor_but_not_action_id(self):
        req,seed=restored_fixture();next_req=deepcopy(req);seed['node_visit_revision']=2
        next_req[FIELD]=bind(next_req,seed)
        self.assertNotEqual(_scope(req),_scope(next_req))
        choice=req[FIELD]['source']['choice']
        self.assertEqual(action_id(req,choice),action_id(next_req,choice))

    def test_service_rejects_stale_revision_conflicts_and_missing_context_before_provider(self):
        req,seed=restored_fixture();provider=Mock(provider_contract=CONTRACT)
        with patch('orchestrator.hierarchical_service.retrieve',return_value=[]):
            service=StrategyDecisionService(provider,provider,{},policy()['episode'],
                recovery_support_policy='preserve_existing_v1')
        with patch('orchestrator.hierarchical_service.HierarchicalDecisionService.handle',return_value='guard-only'):
            seed['node_visit_revision']=2;req[FIELD]=bind(req,seed);service.handle(req)
            stale=deepcopy(req);seed['node_visit_revision']=1;stale[FIELD]=bind(stale,seed)
            with self.assertRaisesRegex(ValueError,'stale_or_conflicting'):service.handle(stale)
            wrong=deepcopy(req);wrong[FIELD]['returned_from_state_id']='different-child'
            with self.assertRaisesRegex(ValueError,'stale_or_conflicting'):service.handle(wrong)
            missing=deepcopy(req);missing.pop(FIELD)
            with self.assertRaisesRegex(ValueError,'missing_virtual_restore'):service.handle(missing)
        provider.plan.assert_not_called()


if __name__=='__main__':unittest.main()
