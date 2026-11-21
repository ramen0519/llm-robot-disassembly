"""Offline error/guard tests. No real key, HTTP, DNS, WSL, or physical simulation."""
from copy import deepcopy
from contextlib import closing
import errno
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import ssl
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

from orchestrator.transport_diagnostics import diagnose, TransportCircuitOpen
from orchestrator.hierarchy_budget import HierarchyBudget
from orchestrator.llm_ranker import RankService, BASE_URL, _http_transport
from orchestrator.hierarchical_planner import SolPlanService
from orchestrator.hierarchical_service import HierarchicalDecisionService
from orchestrator.explicit_action_service import ExplicitActionService
from tests.test_llm_ranker import config, envelope, provider_response
from tests.test_hierarchical_planner import policy, request, feedback, plan_value
from tests.test_explicit_action_service import provider

SECRET = 'sk-fixture-PRIVATE-not-real'


def http_error(status=429, code=None, kind=None, body=None, headers=None):
    if body is None:
        body = json.dumps(dict(error=dict(message=SECRET, code=code, type=kind))).encode()
    stream = io.BytesIO(body)
    return urllib.error.HTTPError('https://private.invalid/' + SECRET, status, SECRET,
                                  headers or {}, stream)


class DiagnosticTests(unittest.TestCase):
    def test_typed_dns_and_nested_url_error_are_preserved_without_message(self):
        error = urllib.error.URLError(urllib.error.URLError(socket.gaierror(-2, SECRET)))
        result = diagnose(error)
        self.assertEqual(result['category'], 'dns_error')
        self.assertEqual(result['errno'], -2)
        self.assertEqual(result['root_error_type'], 'gaierror')
        self.assertNotIn(SECRET, json.dumps(result))

    def test_tls_certificate_code_and_tls_error(self):
        error = ssl.SSLCertVerificationError(1, SECRET)
        error.verify_code = 20
        result = diagnose(urllib.error.URLError(error))
        self.assertEqual(result['category'], 'tls_certificate')
        self.assertEqual(result['certificate_verify_code'], 20)
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertEqual(diagnose(ssl.SSLError(1, SECRET))['category'], 'tls_error')

    def test_os_and_windows_errors_are_not_reported_as_quota(self):
        cases = [(ConnectionRefusedError(errno.ECONNREFUSED, SECRET), 'connection_refused'),
                 (ConnectionResetError(errno.ECONNRESET, SECRET), 'connection_interrupted'),
                 (PermissionError(errno.EACCES, SECRET), 'network_permission'),
                 (OSError(10065, SECRET), 'network_unreachable'),
                 (OSError(10060, SECRET), 'transport_timeout'),
                 (TimeoutError(SECRET), 'transport_timeout')]
        for error, expected in cases:
            with self.subTest(expected=expected):
                result = diagnose(urllib.error.URLError(error))
                self.assertEqual(result['category'], expected)
                self.assertIsNone(result['http_status'])
                self.assertTrue(result['stop_paid_requests'])
                self.assertEqual(result['billing_status'], 'unknown')

    def test_unstructured_url_error_remains_unknown_and_cycle_is_bounded(self):
        error = urllib.error.URLError('CERTIFICATE_VERIFY_FAILED ' + SECRET)
        self.assertEqual(diagnose(error)['category'], 'url_error_unspecified')
        self.assertNotIn(SECRET, json.dumps(diagnose(error)))
        error.reason = error
        self.assertEqual(diagnose(error)['category'], 'url_error_unspecified')

    def test_http_429_requires_evidence_to_separate_quota_and_rate(self):
        for code, kind, expected in (
            ('insufficient_quota', None, 'http_quota'),
            ('project_spend_limit_exceeded', 'insufficient_quota', 'http_quota'),
            ('rate_limit_exceeded', None, 'http_rate_limit'),
            ('slow_down', 'rate_limit_error', 'http_rate_limit'),
            (None, None, 'http_429_unspecified')):
            result = diagnose(http_error(code=code, kind=kind))
            self.assertEqual(result['category'], expected)
            self.assertEqual(result['http_status'], 429)
            self.assertNotIn(SECRET, json.dumps(result))

    def test_http_auth_permission_and_server_have_separate_categories(self):
        for status, category in ((401,'http_authentication'), (403,'http_permission'), (503,'http_server'),(400,'http_error')):
            self.assertEqual(diagnose(http_error(status))['category'], category)

    def test_specific_http_code_takes_precedence_over_broad_type(self):
        result=diagnose(http_error(code='rate_limit_exceeded',kind='insufficient_quota'))
        self.assertEqual(result['category'],'http_rate_limit')

    def test_partial_certificate_and_unusual_url_reason_do_not_break_logging(self):
        self.assertIsNone(diagnose(ssl.SSLCertVerificationError(1,SECRET))['certificate_verify_code'])
        detail=diagnose(urllib.error.URLError(ValueError(SECRET)))
        self.assertTrue(detail['stop_paid_requests'])
        self.assertNotIn(SECRET,json.dumps(detail))

    def test_response_body_is_bounded_consumed_once_and_not_logged(self):
        error = http_error(body=b'x' * 12000)
        spy = Mock(wraps=error.read); error.read = spy
        first = diagnose(error); second = diagnose(error)
        self.assertEqual(first, second)
        spy.assert_called_once_with(8193)
        self.assertEqual(first['error_body_status'], 'too_large')
        self.assertIsNone(first['provider_error_code'])

    def test_malformed_and_secret_provider_values_headers_are_not_persisted(self):
        for body in (b'not json '+SECRET.encode(), json.dumps({'error':{'code':SECRET,'type':SECRET,'message':SECRET}}).encode()):
            result = diagnose(http_error(body=body, headers={'x-request-id':SECRET,'retry-after':SECRET}))
            self.assertNotIn(SECRET, json.dumps(result))
            self.assertIsNone(result['provider_request_id'])
            self.assertIsNone(result['provider_error_code'])

    def test_allowlisted_request_id_and_retry_delay_are_diagnostic_only(self):
        result = diagnose(http_error(headers={'x-request-id':'req_'+'a'*32,'retry-after':'12'}))
        self.assertEqual(result['retry_after_seconds'],12)
        self.assertEqual(result['provider_request_id'],'req_'+'a'*32)
        self.assertEqual(result['automatic_retries'],0)

    def test_local_schema_failure_does_not_latch_connection_guard(self):
        result = diagnose(ValueError(SECRET))
        self.assertFalse(result['stop_paid_requests'])
        self.assertEqual(result['category'], 'unknown_error')
        self.assertNotIn(SECRET,json.dumps(result))

    def test_real_transport_function_calls_opener_once_with_no_retry(self):
        error = urllib.error.URLError(socket.gaierror(-2, SECRET))
        opener = Mock();opener.open.side_effect=error
        with patch('orchestrator.llm_ranker.urllib.request.build_opener', return_value=opener):
            with self.assertRaises(urllib.error.URLError):
                _http_transport(BASE_URL+'/responses',{'Authorization':'Bearer '+SECRET},{'store':False},1)
        self.assertEqual(opener.open.call_count,1)


class GuardTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='transport_test_');self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        env=patch.dict(os.environ,OPENAI_API_KEY=SECRET,RANKER_TEST_KEY=SECRET)
        env.start();self.addCleanup(env.stop)
        net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('network_forbidden'))
        self.net=net.start();self.addCleanup(net.stop);self.addCleanup(self.net.assert_not_called)
    def budget(self, path=None, guarded=True):
        return HierarchyBudget(path or self.root/'global.sqlite',policy(),
            transport_failure_policy='latch_on_error_v1' if guarded else 'legacy_v1')
    def body(self,role='planner',text='first'):
        cfg=policy()['roles'][role]
        return dict(model=cfg['model'],store=False,instructions=text,max_output_tokens=16,
                    reasoning={'effort':cfg['reasoning_effort']})

    def test_failure_blocks_other_role_and_restart_without_new_reservation(self):
        budget=self.budget();bad=Mock(side_effect=urllib.error.URLError(socket.gaierror(-2,SECRET)))
        with self.assertRaises(urllib.error.URLError):budget.transport('scout',bad)(BASE_URL+'/responses',{},self.body('scout'),1)
        other=Mock(return_value={})
        with self.assertRaises(TransportCircuitOpen):budget.transport('planner',other)(BASE_URL+'/responses',{},self.body(),1)
        reopened=self.budget()
        self.assertEqual(reopened.transport_blocked()['category'],'dns_error')
        with self.assertRaises(TransportCircuitOpen):reopened.transport('planner',other)(BASE_URL+'/responses',{},self.body(text='second'),1)
        self.assertEqual(budget.summary()['transport_attempts'],1);other.assert_not_called()
        self.assertEqual(budget.summary()['roles']['planner'],0)
        self.assertIsNone(json.loads(budget.summary()['calls'][0]['metadata'])['actual_cost_from_usage_usd'])

    def test_policy_cannot_be_reset_or_retrofitted_onto_a_legacy_ledger(self):
        self.budget()
        with self.assertRaisesRegex(ValueError,'immutable_transport_policy'):self.budget(guarded=False)
        legacy_path=self.root/'legacy.sqlite';self.budget(legacy_path,guarded=False)
        with self.assertRaisesRegex(ValueError,'new_ledger_required'):self.budget(legacy_path)
        budget=self.budget();budget.transport_failure_policy='legacy_v1'
        with self.assertRaisesRegex(ValueError,'mutable_transport'):budget.transport_blocked()

    def test_pending_attempt_blocks_parallel_role_before_sending(self):
        budget=self.budget();other=Mock(return_value={})
        def pending(*args):
            with self.assertRaises(TransportCircuitOpen):budget.transport('planner',other)(BASE_URL+'/responses',{},self.body(),1)
            raise TimeoutError(SECRET)
        with self.assertRaises(TimeoutError):budget.transport('scout',pending)(BASE_URL+'/responses',{},self.body('scout'),1)
        other.assert_not_called();self.assertEqual(budget.summary()['transport_attempts'],1)

    def test_legacy_still_allows_new_request_after_network_error(self):
        budget=self.budget(guarded=False)
        transport=Mock(side_effect=[urllib.error.URLError(SECRET),{}])
        send=budget.transport('planner',transport)
        with self.assertRaises(urllib.error.URLError):send(BASE_URL+'/responses',{},self.body(),1)
        self.assertEqual(send(BASE_URL+'/responses',{},self.body(text='second'),1),{})
        self.assertEqual(transport.call_count,2);self.assertIsNone(budget.transport_blocked())

    def test_shared_and_role_ledger_have_same_redacted_error_and_cache_does_not_resend(self):
        budget=self.budget();backend=Mock(side_effect=http_error(code='insufficient_quota'))
        cfg=policy()['roles']['planner']
        svc=SolPlanService(cfg,self.root/'planner.sqlite',budget.transport('planner',backend))
        first=svc.plan(request(),[],{},64,30)
        detail=first['metadata']['transport_diagnostics']
        self.assertEqual(detail['category'],'http_quota')
        self.assertEqual(json.loads(budget.summary()['calls'][0]['metadata'])['transport_diagnostics'],detail)
        cached=svc.plan(request(),[],{},64,30)
        self.assertTrue(cached['metadata']['cached']);self.assertFalse(cached['metadata']['attempted'])
        new=deepcopy(request());new['request_id']+=':next'
        blocked=svc.plan(new,[],{},64,30)
        self.assertEqual(blocked['fallback_reason'],'transport_circuit_open')
        self.assertFalse(blocked['metadata']['attempted']);self.assertEqual(backend.call_count,1)
        with closing(sqlite3.connect(svc.ledger_path)) as conn:
            self.assertEqual(conn.execute('SELECT reserved_call,reserved_usd FROM rank_requests WHERE request_id=?',(new['request_id'],)).fetchone(),(0,'0'))
        for path in self.root.glob('*.sqlite'):
            self.assertNotIn(SECRET,path.read_bytes().decode('latin1'))

    def test_hierarchy_stops_paid_escalation_after_scout_connection_error(self):
        budget=self.budget();backend=Mock(side_effect=urllib.error.URLError(socket.gaierror(-2,SECRET)))
        scout=ExplicitActionService(policy()['roles']['scout'],self.root/'scout.sqlite',budget.transport('scout',backend),mode='llm')
        planner=Mock()
        with patch('orchestrator.hierarchical_service.retrieve',return_value=[]):
            svc=HierarchicalDecisionService(scout,planner,{},policy()['episode'],transport_guard=budget.transport_blocked)
        first=svc.handle(request())
        self.assertEqual(first['fallback_reason'],'transport_circuit_open')
        self.assertEqual(first['metadata']['hierarchy_role'],'deterministic_fallback')
        self.assertEqual(svc.planner_calls,0);planner.plan.assert_not_called()
        svc.handle(feedback(request(),first,False))
        self.assertEqual(backend.call_count,1);self.assertEqual(svc.scout_calls,1)
        self.assertTrue(any(e['kind']=='paid_api_blocked' for e in svc.events))

    def test_successful_plan_continues_with_fresh_feedback_and_no_extra_call(self):
        budget=self.budget();r=request();raw=provider(plan_value(r));raw['model']=policy()['roles']['planner']['model']
        backend=Mock(return_value=raw)
        planner=SolPlanService(policy()['roles']['planner'],self.root/'planner.sqlite',budget.transport('planner',backend))
        limits=policy()['episode'];limits['scout_calls']=0;scout=Mock()
        with patch('orchestrator.hierarchical_service.retrieve',return_value=[]):
            svc=HierarchicalDecisionService(scout,planner,{},limits,transport_guard=budget.transport_blocked)
        first=svc.handle(r);second=svc.handle(feedback(r,first,True))
        self.assertEqual(second['metadata']['hierarchy_role'],'plan_action')
        self.assertEqual(backend.call_count,1);self.assertIsNone(budget.transport_blocked())

    def test_invalid_response_is_not_connection_failure_and_does_not_latch(self):
        budget=self.budget();backend=Mock(return_value={'status':'completed','output':[]})
        svc=SolPlanService(policy()['roles']['planner'],self.root/'planner.sqlite',budget.transport('planner',backend))
        self.assertNotEqual(svc.plan(request(),[],{},64,30)['status'],'planned')
        self.assertIsNone(budget.transport_blocked())

    def test_cli_default_preserves_legacy_and_opt_in_is_explicit(self):
        from tools.run_hierarchical_episode import parser
        option=next(a for a in parser()._actions if a.dest=='transport_failure_policy')
        self.assertEqual(option.default,'legacy_v1')
        self.assertIn('latch_on_error_v1',option.choices)


if __name__=='__main__':unittest.main()
