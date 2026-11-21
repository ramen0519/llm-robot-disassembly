"""Offline adversarial provider tests: injected JSON, no real key/API/physics."""
from copy import deepcopy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import Mock
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_explicit_action_contract import request
from test_explicit_action_service import ExplicitActionServiceTests, provider, value, config
from orchestrator.explicit_action_service import PROVIDER_CONTRACT, _ExplicitProvider
from worker.explicit_action_contract import action_id, canonical_response, validate_response


class ProviderV2Tests(ExplicitActionServiceTests):
    # Inherit helpers, not a second copy of the existing test suite (see load_tests).
    def test_part_is_bound_once_from_noncanonical_order(self):
        r = request()
        result = self.service(self.transport(lambda v: v.update(candidate_order=['2', '1']))).handle(r)
        self.assertIsNone(result['fallback_reason'])
        self.assertEqual(result['choice']['part_id'], '2')
        self.assertEqual(result['metadata']['selected_part_source'], 'candidate_order_first')
        self.assertEqual(result['metadata']['provider_contract'], PROVIDER_CONTRACT)
        self.assertEqual(result['action_id'], action_id(r, result['choice']))

    def test_old_redundant_field_is_rejected_even_when_consistent(self):
        for part in ('1', '2', 'untrusted-secret-text'):
            with self.subTest(part=part):
                result = self.service(self.transport(lambda v: v['choice'].update(part_id=part))).handle(request())
                self.assertEqual(result['fallback_reason'], 'invalid_action_response')
                self.assertEqual(result['metadata']['decision_validation_error'], 'invalid_provider_choice_fields')
                self.assertTrue(result['metadata']['redundant_part_field_present'])
                self.assertNotIn('untrusted-secret-text', json.dumps(result))

    def test_empty_duplicate_foreign_and_nonstrings_never_index_or_bind(self):
        for order in ([], ['1', '1'], ['1'], ['1', 'foreign'], [True, '1'], [{}, '1'], '12', None):
            with self.subTest(order=order):
                transport = self.transport(lambda v: v.update(candidate_order=order))
                result = self.service(transport).handle(request())
                self.assertEqual(result['metadata']['decision_validation_error'], 'invalid_candidate_permutation')
                self.assertEqual(result['metadata']['selection_source'], 'deterministic_fallback')
                self.assertEqual(transport.call_count, 1)
                validate_response(request(), result)

    def test_binding_keeps_self_support_guard(self):
        result = self.service(self.transport(lambda v: v['choice'].update(support_ids=['1']))).handle(request())
        self.assertEqual(result['metadata']['decision_validation_error'], 'illegal_support_set_or_budget')
        self.assertEqual(result['metadata']['selection_source'], 'deterministic_fallback')

    def test_stale_identity_is_rejected_before_binding(self):
        result = self.service(self.transport(lambda v: v.update(state_id='private-provider-content'))).handle(request())
        self.assertFalse(result['metadata']['response_identity_matches'])
        self.assertNotIn('private-provider-content', json.dumps(result))
        self.assertEqual(result['metadata']['decision_validation_error'], 'stale_or_invalid_response_identity')

    def test_repeat_reports_exact_guard_and_never_recalls_provider(self):
        r = request(); old = value(r); initial = canonical_response(r, None)
        r['attempted_action_ids'] = [initial['action_id']]
        transport = Mock(return_value=provider(old))
        service = self.service(transport)
        result = service.handle(r); again = service.handle(r)
        self.assertTrue(result['metadata']['action_repeated'])
        self.assertEqual(result['metadata']['rejected_action_id'], initial['action_id'])
        self.assertEqual(result['metadata']['decision_validation_error'], 'action_already_attempted')
        self.assertNotEqual(result['action_id'], initial['action_id'])
        self.assertEqual(set(result['candidate_order']), set(r['candidate_ids']))
        self.assertEqual(result['choice']['part_id'], initial['choice']['part_id'])
        self.assertEqual(transport.call_count, 1)
        self.assertTrue(again['metadata']['cached'])

    def test_support_order_and_citation_changes_still_repeat(self):
        r = request(); r.update(remaining_parts=['1','2','3','base'], candidate_ids=['1','2','3'], max_grippers=3)
        r['episode_feedback'] = [{'status':'UNKNOWN','evidence_id':'episode:1'}]; r['evidence_ids']=['episode:1']
        v = value(r); v['choice']['support_ids']=['2','3']
        full = dict(v['choice'], part_id='1'); old_id = action_id(r,full)
        r['attempted_action_ids']=[old_id]
        v['choice'].update(support_ids=['3','2'],evidence_ids=['episode:1'])
        result = self.service(Mock(return_value=provider(v))).handle(r)
        self.assertTrue(result['metadata']['action_repeated'])
        self.assertEqual(result['metadata']['rejected_action_id'], old_id)

    def test_same_part_different_conditions_remain_valid(self):
        for changes in ({'direction_id':'zp'}, {'support_ids':['2']},
                        {'skill_id':'turn_force_two_leg_v1','second_direction_id':'zp'},
                        {'pose_id':'p1'}):
            with self.subTest(changes=changes):
                r=request(); p=deepcopy(r['pose_catalog'][0]); p['pose_id']='p1'; p['matrix'][2][3]=1
                r['pose_catalog'].append(p)
                v=value(r); initial=canonical_response(r,None)
                r['attempted_action_ids']=[initial['action_id']]; v['choice'].update(changes)
                result=self.service(Mock(return_value=provider(v))).handle(r)
                self.assertIsNone(result['fallback_reason'])
                self.assertEqual(result['choice']['part_id'],'1')
                self.assertFalse(result['metadata']['action_repeated'])
                self.assertNotEqual(result['action_id'],initial['action_id'])

    def test_same_pose_matrix_with_alias_is_still_repeat(self):
        r=request(); p=deepcopy(r['pose_catalog'][0]); p['pose_id']='alias'; r['pose_catalog'].append(p)
        v=value(r); v['choice']['pose_id']='p0'
        r['attempted_action_ids']=[action_id(r,dict(v['choice'],part_id='1'))]
        v['choice']['pose_id']='alias'
        result=self.service(Mock(return_value=provider(v))).handle(r)
        self.assertTrue(result['metadata']['action_repeated'])

    def test_cache_payload_contains_contract_version(self):
        payload=json.loads(_ExplicitProvider._validate_envelope({'request':request(),'timeout_seconds':60}))
        self.assertEqual(payload['provider_contract'],PROVIDER_CONTRACT)

    def test_old_cache_conflicts_without_paid_retry(self):
        transport=self.transport(); ledger=self.root/'old.sqlite'; service=self.service(transport,ledger=ledger)
        service.handle(request())
        with closing(sqlite3.connect(ledger)) as db, db:
            stored=json.loads(db.execute('SELECT payload_json FROM rank_requests').fetchone()[0])
            stored.pop('provider_contract')
            serialized=json.dumps(stored,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
            db.execute('UPDATE rank_requests SET payload_json=?,payload_hash=?',
                (serialized,hashlib.sha256(serialized.encode()).hexdigest()))
        result=self.service(transport,ledger=ledger).handle(request())
        self.assertEqual(result['fallback_reason'],'request_id_conflict')
        self.assertEqual(transport.call_count,1)

    def test_pending_outcome_is_not_reissued_after_restart(self):
        transport=self.transport(); ledger=self.root/'pending.sqlite'
        self.service(transport,ledger=ledger).handle(request())
        with closing(sqlite3.connect(ledger)) as db, db:
            db.execute("UPDATE rank_requests SET status='pending',response_json=NULL")
        result=self.service(transport,ledger=ledger).handle(request())
        self.assertEqual(result['fallback_reason'],'unknown_outcome')
        self.assertEqual(transport.call_count,1)

    def test_refusal_incomplete_and_nonjson_are_not_repaired(self):
        for raw in ({**provider(value(request())),'status':'incomplete'},
                    {**provider(value(request())),'output':[{'type':'message','role':'assistant','content':[{'type':'refusal'}]}]},
                    {'status':'completed','output':[]}):
            transport=Mock(return_value=raw)
            result=self.service(transport).handle(request())
            self.assertEqual(result['fallback_reason'],'invalid_action_response')
            self.assertFalse(result['metadata']['raw_response_recorded'])
            self.assertEqual(transport.call_count,1)

    def test_provider_planner_repeat_feedback_roundtrip_with_fake_physics(self):
        from test_explicit_action_planner import make_planner, result_for
        calls=[]
        def transport(url, headers, body, timeout):
            r=json.loads(body['input'][0]['content'])['request']; v=value(r)
            if len(calls)==1:
                # Same physical action, but this request has fresh correlation IDs.
                v['choice']=deepcopy(calls[0][1]['choice'])
            elif len(calls)==2:
                v['choice'].update(direction_id='zp', support_ids=[])
            calls.append((deepcopy(r),deepcopy(v)))
            return provider(v)
        service=self.service(Mock(side_effect=transport),config(max_calls=4))
        invocations=[]
        def verifier(context, choice, output):
            invocations.append(choice)
            return result_for(context,choice,feasible=len(invocations)>2)
        planner=make_planner(decision=service.handle,verifier=verifier)
        planner.plan(budget=4,timeout=None)
        report=planner.report()
        self.assertTrue(report['success'])  # Fixture physics, not CAD success.
        self.assertEqual(len(calls),4)
        self.assertEqual(report['final_pair_calls'],1)
        self.assertEqual([len(r['attempted_actions']) for r,v in calls],[0,1,2,0])
        self.assertEqual(len({a['action_id'] for a in report['attempts']}),4)
        self.assertEqual(report['decisions'][1]['response']['metadata']['decision_validation_error'],'action_already_attempted')
        feedback=calls[2][0]['episode_feedback'][1]
        self.assertEqual(feedback['decision_source'],'canonical_fallback')
        self.assertEqual(feedback['selection_feedback']['scope'],'selection_validation_not_physics')
        self.assertTrue(feedback['selection_feedback']['action_repeated'])
        self.assertEqual(feedback['selection_feedback']['rejected_action_id'],report['attempts'][0]['action_id'])
        self.assertEqual(feedback['choice'],report['attempts'][1]['choice'])
        self.assertFalse(report['hard_pruning'])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(ProviderV2Tests(name) for name in ProviderV2Tests.__dict__ if name.startswith('test_'))


if __name__=='__main__': unittest.main()
