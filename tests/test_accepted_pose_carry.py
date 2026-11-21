"""Saved actual observations and injected DFS fixtures; NO API/new physics."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.test_explicit_action_planner import Planner, GraphFixture, IDENTITY, result_for, response_for
from worker.accepted_pose_carry import (FIELD, POLICY, append_to_state, bind, entry_reference, resolve_pose, snapshot, validate)
from worker.explicit_action_contract import action_id, validate_request

ROOT = Path(__file__).absolute().parents[1]
SAVED = ROOT / 'runs/part_research_00161_prepare_20261109/episode'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def carry(request, edge, limit=128):
    current = deepcopy(request)
    state = dict(state_id=current['state_id'], pose_catalog=deepcopy(current['pose_catalog']),
        n_attempts=len(current['attempted_action_ids']), attempted_action_ids=deepcopy(current['attempted_action_ids']))
    proof = snapshot(edge, run_id=current['run_id'], task_context=current['task_context'],
        base_part=current['base_part'], max_grippers=current['max_grippers'], remaining_parts=current['remaining_parts'])
    identity = append_to_state(state, proof, limit)
    current['pose_catalog'] = deepcopy(state['pose_catalog'])
    current[FIELD] = bind(current, state)
    return validate_request(current), state, proof, identity


def bound_fixture_result(context, choice, feasible=True):
    """Provenance-shaped injected result, never a real RedMax observation."""
    result = result_for(context, choice, feasible)
    result.update(backend='ASAP_RedMax', task_context_input_and_run_conditions_checked=True,
        input_files_unchanged=True, source_files_unchanged=True,
        input_hashes=deepcopy(context['task_context']['task']['input_sha256']), chosen=deepcopy(choice),
        executed=dict(pose=deepcopy(context['pose_matrix']), pose_id=choice['pose_id'],
            part_id=choice['part_id'], support_ids=deepcopy(choice['support_ids']), path_fixed_parts=context['parts_rest']),
        scope=dict(task_context=deepcopy(context['task_context']), assembled_parts=context['remaining_parts'],
            parts_rest=context['parts_rest'], part_move=choice['part_id'], pose=deepcopy(context['pose_matrix']),
            base_part=context['base_part'], selected_action=deepcopy(choice)))
    return result


def fixture_worker(decide, verify, poses, parts=None):
    return Planner(parts=parts or ['a','b','c','d'], decision_fn=decide, verifier_fn=verify,
        pose_provider=poses, run_id='carry-fixture', graph_factory=GraphFixture,
        task_context=dict(fixture_only=True, task=dict(input_sha256={'fixture.obj':'1'*64})),
        pose_transition_policy=POLICY, path_selection_policy='path_probe_round_v1',
        fallback_policy='path_stage_v1', decision_failure_policy='halt_v1')


class AcceptedPoseCarryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = read(SAVED/'worker_result.json')
        cls.decisions = read(SAVED/'hierarchy_decisions.json')

    def setUp(self):
        for name in ('urllib.request.OpenerDirector.open','subprocess.Popen'):
            guard = patch(name, side_effect=AssertionError('NO_API_OR_NEW_PHYSICS'))
            mock = guard.start()
            self.addCleanup(mock.assert_not_called)
            self.addCleanup(guard.stop)

    def saved(self, request_number=5, attempt_number=4):
        return deepcopy(self.decisions[request_number-1]['request']), deepcopy(self.worker['attempts'][attempt_number-1])

    def test_both_actual_failures_recover_exact_original_transform_without_recenter(self):
        for number, attempt in ((5,4),(22,21)):
            with self.subTest(request=number):
                request, edge = self.saved(number,attempt)
                original = deepcopy(request)
                self.assertFalse(any(p['matrix'] == edge['context']['pose_matrix'] for p in request['pose_catalog']))
                projected,state,proof,identity = carry(request,edge)
                self.assertEqual(request,original)
                self.assertEqual(projected['pose_catalog'][:-1], request['pose_catalog'])
                self.assertEqual(projected['pose_catalog'][-1]['matrix'],edge['context']['pose_matrix'])
                self.assertEqual(resolve_pose(projected,projected['episode_feedback'][-1]),identity)
                self.assertEqual(projected['state_id'],request['state_id'])
                self.assertFalse(projected[FIELD]['physics_success_implied'])
                self.assertTrue(projected[FIELD]['requires_fresh_verification'])
                self.assertNotIn(proof['event_id'],projected['candidate_ids'])

    def test_same_rotation_different_translation_resolves_exact_carried_matrix(self):
        request,edge=self.saved()
        shifted=deepcopy(edge['context']['pose_matrix']); shifted[0][3]+=100
        request['pose_catalog'].append(dict(pose_id='same_rotation_wrong_translation_fixture',matrix=shifted))
        projected,_,_,identity=carry(request,edge)
        self.assertEqual(resolve_pose(projected,projected['episode_feedback'][-1]),identity)
        self.assertNotEqual(identity,'same_rotation_wrong_translation_fixture')

    def test_same_matrix_already_in_catalog_does_not_duplicate_pose(self):
        request,edge=self.saved()
        request['pose_catalog'].append(dict(pose_id='exact_geometry_fixture',matrix=deepcopy(edge['context']['pose_matrix'])))
        projected,state,_,identity=carry(request,edge)
        self.assertEqual(identity,'exact_geometry_fixture')
        self.assertEqual(len(projected['pose_catalog']),len(request['pose_catalog']))
        self.assertEqual(state['pose_catalog_revision'],0)

    def test_source_requires_actual_accepted_timely_subverdicts(self):
        for where,key,value in (('edge','accepted_for_transition',False),('edge','deadline_exceeded',True),
            ('edge','status','UNKNOWN'),('raw','backend','Mock'),('raw','path_success',False),
            ('raw','stability_success',False),('raw','source_files_unchanged',False),
            ('raw','input_files_unchanged',False),('raw','task_context_input_and_run_conditions_checked',False)):
            with self.subTest(where=where,key=key):
                request,edge=self.saved(); target=edge if where=='edge' else edge['raw']; target[key]=value
                with self.assertRaises(ValueError): carry(request,edge)

    def test_source_rejects_changed_run_mesh_scope_and_child(self):
        for mutation in ('run','mesh','task','child','selected','pose','scope'):
            with self.subTest(mutation=mutation):
                request,edge=self.saved()
                if mutation=='run': request['run_id']='other-run'
                if mutation=='mesh': edge['raw']['input_hashes']['0.obj']='0'*64
                if mutation=='task': edge['context']['task_context']['task']['solver_settings']['seed']=99
                if mutation=='child': edge['context']['parts_rest']=edge['context']['parts_rest'][:-1]
                if mutation=='selected': edge['raw']['chosen']['direction_id']='zn'
                if mutation=='pose': edge['raw']['pose'][0][3]+=1
                if mutation=='scope': edge['raw']['scope']['task_context']['task']['solver_settings']['seed']=99
                with self.assertRaises(ValueError): carry(request,edge)

    def test_full_request_binding_rejects_tampering(self):
        request,edge=self.saved(); projected,_,_,_=carry(request,edge)
        for mutation in ('matrix','part','task','proof','active','revision'):
            with self.subTest(mutation=mutation):
                altered=deepcopy(projected)
                if mutation=='matrix': altered['pose_catalog'][-1]['matrix'][0][3]+=1
                if mutation=='part': altered['remaining_parts'].remove('0')
                if mutation=='task': altered['task_context']['task']['solver_settings']['seed']=1
                if mutation=='proof': altered[FIELD]['active']['source']['verdict']['stability_success']=False
                if mutation=='active': altered[FIELD]['active']['pose_id']=altered['pose_catalog'][0]['pose_id']
                if mutation=='revision': altered[FIELD]['catalog_revision']+=1
                with self.assertRaises(ValueError): validate_request(altered)

    def test_visible_parent_observation_cannot_conflict(self):
        request,edge=self.saved(); projected,_,_,_=carry(request,edge)
        projected['episode_feedback'][-1]['accepted_for_transition']=False
        with self.assertRaises(ValueError): validate_request(projected)

    def test_compact_provenance_survives_parent_feedback_window(self):
        request,edge=self.saved(); projected,state,_,_=carry(request,edge)
        projected['episode_feedback']=[]
        projected['evidence_ids']=[m['evidence_id'] for m in projected['memory_references']]
        projected[FIELD]=bind(projected,state)
        self.assertIsNotNone(validate_request(projected))
        altered=deepcopy(projected); altered[FIELD]['active']['source']['task_context_sha256']='0'*64
        with self.assertRaises(ValueError): validate_request(altered)

    def test_pose_limit_fails_without_truncating_or_resetting_budget(self):
        request,edge=self.saved()
        with self.assertRaisesRegex(ValueError,'accepted_pose_catalog_overflow_no_truncation'):
            carry(request,edge,limit=len(request['pose_catalog']))

    def test_new_source_same_matrix_retains_catalog_budget_and_state_identity(self):
        request,edge=self.saved(); projected,state,proof,identity=carry(request,edge)
        state.update(n_attempts=7,attempted_action_ids=['fixture-attempt'])
        original=deepcopy(state['pose_catalog']); oldstate=state['state_id']
        append_to_state(state,proof)
        self.assertEqual(state['pose_catalog'],original)
        self.assertEqual(state['n_attempts'],7)
        self.assertEqual(state['attempted_action_ids'],['fixture-attempt'])
        self.assertEqual(state['state_id'],oldstate)
        self.assertEqual(state['pose_catalog_revision'],1)

    def test_reference_from_other_branch_is_not_current_pose_authority(self):
        request,edge=self.saved(); projected,_,_,_=carry(request,edge)
        reference=deepcopy(projected['episode_feedback'][-1]); reference['evidence_id']='other-branch'
        self.assertIsNone(resolve_pose(projected,reference))

    def test_reentry_with_prior_attempts_uses_new_source_then_latest_command(self):
        request,edge=self.saved(); projected,state,proof,identity=carry(request,edge)
        old = deepcopy(self.decisions[5]['request'])  # one observed child failure, request6
        state['attempted_action_ids']=deepcopy(old['attempted_action_ids'])
        state['n_attempts']=1
        state['virtual_restore_snapshot']={'stale_fixture':True}
        # Synthetic reentry: a later accepted parent action under another exact
        # parent catalog pose. This constructs no new physical success claim.
        newedge=deepcopy(edge)
        pose=deepcopy(edge['context']['pose_catalog'][0])
        newedge['event_id']=request['run_id']+':attempt:50'
        newedge['request_id']=request['run_id']+':request:50'
        newedge['context']['request_id']=newedge['request_id']
        newedge['choice']['pose_id']=pose['pose_id']
        parent=deepcopy(self.decisions[3]['request'])
        newid=action_id(parent,newedge['choice'])
        newedge['action_id']=newid; newedge['context']['action_id']=newid
        newedge['context']['pose_matrix']=deepcopy(pose['matrix'])
        newedge['raw']['pose']=deepcopy(pose['matrix'])
        newedge['raw']['chosen']=deepcopy(newedge['choice'])
        newedge['raw']['scope']['pose']=deepcopy(pose['matrix'])
        newedge['raw']['scope']['selected_action']=deepcopy(newedge['choice'])
        newedge['raw']['executed']['pose']=deepcopy(pose['matrix'])
        newedge['raw']['executed']['pose_id']=pose['pose_id']
        newproof=snapshot(newedge,run_id=request['run_id'],task_context=request['task_context'],
            base_part=request['base_part'],max_grippers=request['max_grippers'],remaining_parts=request['remaining_parts'])
        nextid=append_to_state(state,newproof)
        self.assertEqual(state['n_attempts'],1)
        self.assertEqual(state['pose_carry_entry_attempt_count'],1)
        self.assertEqual(state['pose_carry_activation_revision'],2)
        self.assertNotIn('virtual_restore_snapshot',state)
        old['pose_catalog']=deepcopy(state['pose_catalog']); old[FIELD]=bind(old,state)
        validate_request(old)
        fresh=entry_reference(old)
        self.assertEqual(fresh['evidence_id'],newedge['event_id'])
        self.assertEqual(resolve_pose(old,fresh),nextid)
        self.assertIsNone(resolve_pose(old,old['episode_feedback'][-1]))
        prior_history=deepcopy(old['attempted_action_ids'])
        # A new command after entry takes precedence even when its verdict fails.
        newchoice=dict(part_id='1',pose_id=nextid,skill_id='single_force_v1',direction_id='xp',
            second_direction_id=None,support_ids=[],evidence_ids=[])
        current_id=action_id(old,newchoice)
        old['attempted_action_ids'].append(current_id)
        old['attempted_actions'].append(dict(action_id=current_id,**{k:v for k,v in newchoice.items() if k!='evidence_ids'}))
        event_id=request['run_id']+':attempt:51'
        old['path_stage_observations'].append(dict(action_id=current_id,path_success=False,status='UNKNOWN',
            deadline_exceeded=False,backend='ASAP_RedMax',source_event_id=event_id))
        feedback=dict(evidence_id=event_id,action_id=current_id,choice=newchoice,
            selected_pose_matrix=pose['matrix'],remaining_parts=old['remaining_parts'],status='UNKNOWN',
            accepted_for_transition=False,deadline_exceeded=False,path_success=False,stability_success=None)
        old['episode_feedback'].append(feedback); old['evidence_ids'].append(event_id)
        old[FIELD]=bind(old,state)
        validate_request(old)
        self.assertIsNone(entry_reference(old))
        self.assertEqual(resolve_pose(old,feedback),nextid)
        self.assertEqual(old['attempted_action_ids'][:-1],prior_history)
        self.assertIsNone(resolve_pose(old,fresh))
        altered=deepcopy(old); altered[FIELD]['entry_attempt_count']=2
        with self.assertRaises(ValueError): validate_request(altered)

    def test_legacy_request_has_no_new_resolution(self):
        request,_=self.saved()
        self.assertIsNone(validate(request)); self.assertIsNone(resolve_pose(request,None))

    def test_worker_carries_only_after_verified_transition_and_rechecks_next_action(self):
        calls=[]; requests=[]
        shifted=deepcopy(IDENTITY); shifted[0][3]=25.0
        def poses(parts):
            return [dict(pose_id='root' if len(parts)==4 else 'recentered',
                         matrix=deepcopy(IDENTITY if len(parts)==4 else shifted))]
        def decide(request):
            requests.append(deepcopy(request))
            pose=next((p['pose_id'] for p in request['pose_catalog'] if p['matrix']==IDENTITY),request['pose_catalog'][0]['pose_id'])
            return response_for(request,part=request['candidate_ids'][0],pose=pose)
        def verify(context,choice,_):
            calls.append(deepcopy(context)); return bound_fixture_result(context,choice)
        worker=fixture_worker(decide,verify,poses)
        worker.plan(budget=3,timeout=None)
        report=worker.report()
        self.assertTrue(report['success'],report['planner_error'])
        self.assertEqual(len(calls),3)  # carrying a pose never bypasses fresh verification
        self.assertNotIn(FIELD,requests[0]); self.assertIn(FIELD,requests[1]); self.assertIn(FIELD,requests[2])
        self.assertTrue(all(c['pose_matrix']==IDENTITY for c in calls))
        self.assertEqual([len(r['remaining_parts']) for r in requests],[4,3,2])

    def test_worker_rejects_fixture_lacking_full_scope_in_opt_in_profile(self):
        worker=fixture_worker(lambda r:response_for(r,pose='p0'),lambda c,a,p:result_for(c,a),
            lambda parts:[dict(pose_id='p0',matrix=deepcopy(IDENTITY))])
        worker.plan(budget=3,timeout=None)
        self.assertEqual(worker.report()['status'],'ERROR')
        self.assertEqual(worker.report()['total_n_eval'],1)


if __name__=='__main__': unittest.main()
