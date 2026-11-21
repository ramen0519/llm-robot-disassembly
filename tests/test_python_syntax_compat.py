"""API-free bounded-language and inert rejection diagnostics regressions."""
from copy import deepcopy
from contextlib import closing, nullcontext
import json
import os
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.python_skill_vm import VM, parse, PolicyError, SYNTAX_DESCRIPTION
from orchestrator.python_skill_program import step_program, API_DESCRIPTION
from orchestrator.python_skill_discovery import PythonSkillDiscoveryService
from orchestrator.python_skill_diagnostics import (
    rejected_source, save_rejection_artifacts, validate_record, sha_text,
)
from tests.test_skill_discovery import synthesis, config, decision, raw_response
from tests.test_skill_program import request, append_observation
from tests.test_python_preflight import policy
from tests.test_skill_validation import save
from tools.run_python_skill_cycle import contract_rejection_feedback

ROOT = Path(__file__).absolute().parents[1]


def execute(body, data=None, helpers='', steps=50000):
    return VM('def propose(ctx):\n' + body + '\n' + helpers,
              lambda choice: True, max_steps=steps).call('propose', [data or {}])


class SyntaxTests(unittest.TestCase):
    def test_filtered_multiple_clauses_and_outer_scope(self):
        got = execute('    x = 99\n    r = [x+y for x in [1,2] if x > 0 for y in [3,4] if y == 4]\n    return [r,x]')
        self.assertEqual(got, [[5,6],99])

    def test_nested_comprehensions_and_dict_iteration(self):
        self.assertEqual(execute('    return [[x*y for y in [2,3]] for x in [4,5]]'), [[8,12],[10,15]])
        self.assertEqual(execute('    return [k for k in ctx]', {'a':1,'b':2}), ['a','b'])

    def test_comprehension_iterable_reads_outer_binding(self):
        self.assertEqual(execute('    x = [1,2]\n    y = [x+1 for x in x]\n    return [x,y]'), [[1,2],[2,3]])

    def test_comprehension_target_does_not_leak(self):
        with self.assertRaisesRegex(PolicyError, 'undefined_local'):
            execute('    r = [x for x in [1]]\n    return x')

    def test_named_helpers_and_dict_constructor(self):
        got = execute('    return dict(a=add(y=4,x=3), b=add(1,y=2))',
                      helpers='def add(x,y):\n    return x+y')
        self.assertEqual(got, {'a':7,'b':3})

    def test_helper_missing_and_duplicate_positional_arguments(self):
        for expr in ('f(x=1)', 'f(1,x=2,y=3)'):
            with self.subTest(expr=expr), self.assertRaisesRegex(PolicyError, 'arity'):
                execute('    return '+expr, helpers='def f(x,y):\n    return x+y')

    def test_sorted_reverse_lambda_and_stability(self):
        data = {'rows':[{'v':1,'id':'a'},{'v':2,'id':'b'},{'v':1,'id':'c'}]}
        self.assertEqual(execute('    return [p["id"] for p in sorted(ctx["rows"], key=lambda p:p["v"], reverse=True)]',data), ['b','a','c'])

    def test_key_helper_builtin_and_none(self):
        self.assertEqual(execute('    return sorted([1,3,2],key=neg)',helpers='def neg(x):\n    return -x'), [3,2,1])
        self.assertEqual(execute('    return sorted([-4,2,-1],key=abs)'), [-1,2,-4])
        self.assertEqual(execute('    return sorted([2,1],key=None)'), [1,2])

    def test_lambda_closure_local_data_and_named_helper(self):
        self.assertEqual(execute('    multiplier = -1\n    return sorted([1,2],key=lambda x:scale(x,m=multiplier))',
                                helpers='def scale(x,m):\n    return x*m'), [2,1])

    def test_min_max_default_and_numeric_keywords(self):
        self.assertEqual(execute('    return [min([],default=7), max([-3,2],key=abs), sum([1,2],start=4), round(1.234,ndigits=2), int("10",base=2)]'),
                         [7,-3,7,1.23,2])

    def test_unsupported_keywords_unpack_and_callback_escape_fail_closed(self):
        for expr in ('sorted([1],unknown=True)', 'dict(**ctx)', 'dict(a=1,a=2)',
                     'ctx.get(key="x")', 'sorted([1],key=eligible)',
                     'sorted([1],key=ctx["key"])', 'sorted([1],key=lambda x,y:x)',
                     'sorted([1],key=lambda x=1:x)', '(lambda x:x)(1)',
                     'lambda x:x', 'sorted([1],key=f)'):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                parse('def propose(ctx):\n    return '+expr+'\ndef f(x,y):\n    return x')

    def test_key_callback_cannot_return_host_function(self):
        with self.assertRaisesRegex(PolicyError, 'undefined_local'):
            execute('    return sorted([1],key=lambda x:abs)')

    def test_unsupported_comprehensions_and_host_access(self):
        for expr in ('{x:x for x in [1]}', '{x for x in [1]}', '(x for x in [1])',
                     '[x for x,y in [[1,2]]]', '[x async for x in ctx]',
                     '[x for x in [1] for y in [2] for z in [3] for w in [4]]',
                     '[open(x) for x in ["secret"]]',
                     'sorted([1],key=lambda x:x.__class__)',
                     'sorted([1],key=lambda x:eval("1"))'):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                parse('def propose(ctx):\n    return '+expr)

    def test_comprehension_output_and_complexity_limits(self):
        for expr, code in (('[x+y for x in range(65) for y in range(65)]','container_limit'),
                           ('[[0]*1000 for x in range(40)]','value_complexity')):
            with self.subTest(expr=expr), self.assertRaisesRegex(PolicyError,code):
                execute('    return '+expr)

    def test_filters_cannot_hide_unbounded_work(self):
        with self.assertRaisesRegex(PolicyError,'step_limit'):
            execute('    return [x for x in range(4096) for y in range(4096) if False]',steps=100)

    def test_key_callback_is_metered(self):
        with self.assertRaisesRegex(PolicyError,'step_limit'):
            execute('    return sorted(range(100),key=lambda x:sum([y for y in range(100)]))',steps=200)

    def test_key_helper_recursion_is_bounded(self):
        with self.assertRaisesRegex(PolicyError,'depth'):
            execute('    return sorted([1],key=recur)',helpers='def recur(x):\n    return recur(x)')

    def test_static_location_includes_node_and_line(self):
        with self.assertRaises(PolicyError) as found:
            parse('def propose(ctx):\n    return {x:x for x in ctx}')
        detail = found.exception.diagnostic
        self.assertEqual(detail['node_type'],'DictComp')
        self.assertEqual(detail['line'],2)
        self.assertEqual(detail['column'],11)

    def test_syntax_error_has_location_without_source_message(self):
        with self.assertRaises(PolicyError) as found:
            parse('def propose(ctx):\n    return [')
        self.assertEqual(str(found.exception),'python_syntax')
        self.assertEqual(found.exception.diagnostic['line'],2)

    def test_child_uses_new_syntax_and_preserves_current_action_guard(self):
        p = policy('    ids = [p for p in ctx["candidate_ids"] if p != ctx["base_part"]]\n'
                   '    ordered = sorted(ids,reverse=True)\n',direction='yp')
        for base in ('base',None):
            req = request()
            req['base_part'] = base
            if base is None:
                req['candidate_ids'] = list(req['remaining_parts'])
            append_observation(req)
            before = deepcopy(req)
            result = step_program(p,req,diagnose=True)
            self.assertIsNotNone(result['response'],result)
            self.assertEqual(result['physical_calls'],0)
            self.assertFalse(result['response']['metadata']['physics_bypass'])
            self.assertEqual(result['response']['metadata']['python_runtime'],'python_data_vm_v2')
            self.assertEqual(req,before)

    def test_input_audit_still_sees_misspelled_fields_in_new_syntax(self):
        for prefix in ('    bad = [ctx["geometry"].get("support_ids") for x in [1]]\n',
                       '    bad = sorted([1],key=lambda x:ctx["geometry"].get("support_ids"))\n'):
            out = step_program(policy(prefix,direction='yp'),append_observation(request()),diagnose=True)
            self.assertFalse(out['input_audit']['accepted'])

    def test_generator_uses_the_vm_syntax_contract(self):
        from orchestrator.python_skill_discovery import INSTRUCTIONS, STABILITY_INSTRUCTIONS
        self.assertIn(SYNTAX_DESCRIPTION,API_DESCRIPTION)
        self.assertNotIn('No comprehensions',API_DESCRIPTION)
        for instructions in (INSTRUCTIONS,STABILITY_INSTRUCTIONS):
            self.assertIn(SYNTAX_DESCRIPTION,instructions)
            self.assertNotIn('tuple unpacking, comprehensions',instructions)


class RejectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='syntax_TEST_',dir=ROOT/'runs')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        net = patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('NETWORK_FORBIDDEN'))
        self.net = net.start(); self.addCleanup(net.stop); self.addCleanup(self.net.assert_not_called)

    def rejected(self, source):
        packet = synthesis()
        packet['operation'] = 'synthesize'
        value = decision(packet)
        value['program'] = dict(language='python_policy_v1',source=source)
        raw = raw_response(value); raw['model'] = 'gpt-6-luna'
        service = PythonSkillDiscoveryService(config(enabled=False,max_calls=0,max_usd=0),self.folder/'ledger.sqlite')
        metadata = {}
        try:
            parsed, reason = service._parse_response(raw,packet,metadata)
        finally:
            service.close()
        self.assertIsNone(parsed)
        self.assertEqual(reason,'invalid_generated_python')
        return dict(status='rejected',fallback_reason=reason,metadata=metadata)

    def test_rejected_source_is_inert_and_persisted_with_location(self):
        source='def propose(ctx):\n    return {x:x for x in ctx}'
        proposal = self.rejected(source)
        record = proposal['metadata']['python_rejected_source']
        self.assertEqual(record['source_text'],source)
        self.assertEqual(record['diagnostic']['node_type'],'DictComp')
        self.assertFalse(record['executable']); self.assertFalse(record['registered'])
        self.assertFalse(proposal['metadata']['python_executed'])
        self.assertNotIn('program',proposal)
        save_rejection_artifacts(self.folder,proposal)
        self.assertEqual((self.folder/'rejected_source.txt').read_text(encoding='utf-8'),source)
        self.assertFalse((self.folder/'rejected_source.py').exists())
        detail=json.loads((self.folder/'rejected_source_diagnostic.json').read_text(encoding='utf-8'))
        self.assertEqual(detail['source_file'],'rejected_source.txt')
        self.assertEqual(detail['stored_sha256'],sha_text(source))

    def test_known_key_and_key_like_tokens_redacted_before_ledger_or_file(self):
        secret='UNIT_ACTIVE_SECRET_VALUE'
        source='def propose(ctx):\n    # '+secret+' sk-proj-UNIT_ONLY_123456789\n    return {x:x for x in ctx}'
        with patch.dict(os.environ,{'DISCOVERY_FIXTURE_KEY':secret}):
            proposal=self.rejected(source)
        serialized=json.dumps(proposal)
        self.assertNotIn(secret,serialized);self.assertNotIn('sk-proj-UNIT_ONLY',serialized)
        record=proposal['metadata']['python_rejected_source']
        self.assertTrue(record['redacted']); self.assertEqual(record['source_sha256'],sha_text(source))
        save_rejection_artifacts(self.folder,proposal)
        self.assertNotIn(secret,(self.folder/'rejected_source.txt').read_text(encoding='utf-8'))

    def test_unknown_exception_text_not_stored_as_diagnostic(self):
        record=rejected_source({'source':'text'},ValueError('untrusted secret error'))
        self.assertEqual(record['diagnostic']['code'],'python_validation_rejected')
        self.assertNotIn('untrusted secret error',json.dumps(record))

    def test_large_unicode_source_bounded_and_clearly_truncated(self):
        source='가'*10000
        record=rejected_source({'source':source},PolicyError('python_source_size'))
        self.assertLessEqual(len(record['source_text'].encode('utf-8')),16384)
        self.assertTrue(record['truncated']);validate_record(record)

    def test_empty_or_missing_program_source_is_handled(self):
        self.assertIsNone(rejected_source(None,ValueError()))
        self.assertIsNone(rejected_source({'source':None},ValueError()))
        self.assertEqual(rejected_source({'source':''},PolicyError('python_source_size'))['source_text'],'')

    def test_existing_artifacts_not_overwritten(self):
        proposal=self.rejected('import os')
        save_rejection_artifacts(self.folder,proposal)
        with self.assertRaises(FileExistsError):
            save_rejection_artifacts(self.folder,proposal)

    def test_hash_tampering_prevents_storage_or_forwarding(self):
        proposal=self.rejected('import os')
        proposal['metadata']['python_rejected_source']['source_text']='changed'
        with self.assertRaisesRegex(ValueError,'record_invalid'):
            save_rejection_artifacts(self.folder,proposal)
        self.assertFalse((self.folder/'rejected_source.txt').exists())

    def test_same_ledger_feedback_carries_rejected_source_not_valid_program(self):
        proposal=self.rejected('def propose(ctx):\n    return {x:x for x in ctx}')
        proposal['metadata']['llm_live']=True # routing fixture, NEVER a live claim
        ledger=self.folder/'learner.sqlite'
        save(self.folder/'config.json',dict(mode='live',ledger=str(ledger)))
        save(self.folder/'provider_proposal.json',proposal)
        feedback=contract_rejection_feedback(self.folder,ledger)
        self.assertIn('rejected_source_diagnostic',feedback)
        self.assertFalse(feedback['rejected_source_diagnostic']['executable'])
        self.assertNotIn('previous_program',feedback)
        self.assertNotIn('program',feedback)
        with self.assertRaisesRegex(ValueError,'bound_live'):
            contract_rejection_feedback(self.folder,self.folder/'wrong.sqlite')

    def test_legacy_rejections_without_source_still_route(self):
        ledger=self.folder/'learner.sqlite'
        save(self.folder/'config.json',dict(mode='live',ledger=str(ledger)))
        save(self.folder/'provider_proposal.json',dict(status='rejected',fallback_reason='invalid_generated_python',
             metadata=dict(llm_live=True,python_validation_error='python_forbidden_syntax:ListComp')))
        feedback=contract_rejection_feedback(self.folder,ledger)
        self.assertNotIn('rejected_source_diagnostic',feedback)
        self.assertEqual(feedback['python_validation_error'],'python_forbidden_syntax:ListComp')

    def test_injected_transport_ledger_contains_redacted_evidence_and_no_retry(self):
        secret='UNIT_KEY_FOR_LEDGER_ONLY'
        source='def propose(ctx):\n    # '+secret+'\n    return {x:x for x in ctx}'
        calls=[]
        def transport(url,headers,body,timeout):
            calls.append(1)
            packet=json.loads(body['input'][0]['content'])
            value=decision(packet);value['program']=dict(language='python_policy_v1',source=source)
            raw=raw_response(value);raw['model']=body['model'];return raw
        ledger=self.folder/'injected.sqlite'
        service=PythonSkillDiscoveryService(config(),ledger,transport)
        try:
            with patch.dict(os.environ,{'DISCOVERY_FIXTURE_KEY':secret}):
                result=service.propose(synthesis())
                cached=service.propose(synthesis())
        finally:service.close()
        self.assertEqual(len(calls),1)
        self.assertFalse(result['metadata']['llm_live'])
        self.assertEqual(result['metadata']['retries'],0)
        self.assertEqual(cached['metadata']['python_rejected_source'],result['metadata']['python_rejected_source'])
        with closing(sqlite3.connect(ledger)) as db:
            text=db.execute('SELECT response_json FROM rank_requests').fetchone()[0]
        self.assertNotIn(secret,text)
        self.assertIn('REDACTED_API_KEY',text)

    def test_actual_cycle_rejection_branch_writes_artifacts_before_any_physics(self):
        import tools.run_python_skill_cycle as cycle
        # Real runner branch, injected response only; no key access or transport.
        proposal=self.rejected('def propose(ctx):\n    return {x:x for x in ctx}')
        proposal.update(origin='injected_transport')
        proposal['metadata']['llm_live']=False
        packet=synthesis()
        manifest=dict(cases=[dict(case_id=k,phase=p,source_task='fixture',trigger_kind='path_failed')
            for k,p in [('d','debug'),('a1','admission'),('a2','admission')]])
        save(self.folder/'manifest.json',manifest)
        service=PythonSkillDiscoveryService(config(api_key_env='OPENAI_API_KEY'),self.folder/'cli.sqlite')
        self.addCleanup(service.close)
        save(self.folder/'api.json',service.config)
        out=self.folder/'candidate'
        with patch.object(cycle,'source_case',return_value={}), \
             patch.object(cycle,'python_generation_packet',return_value=packet), \
             patch.object(cycle,'PythonSkillDiscoveryService',return_value=service), \
             patch.object(service,'propose',return_value=proposal) as generate, \
             patch('local_api_key.key_environment',return_value=nullcontext()), \
             patch.object(cycle,'run_case',side_effect=AssertionError('NO_PHYSICS')) as physics:
            code=cycle.main(['--run-dir',str(out),'--manifest',str(self.folder/'manifest.json'),
                '--mode','live','--allow-live-api','--llm-config',str(self.folder/'api.json'),
                '--ledger',str(self.folder/'cli.sqlite')])
        self.assertEqual(code,1);generate.assert_called_once();physics.assert_not_called()
        result=json.loads((out/'result.json').read_text(encoding='utf-8'))
        self.assertFalse(result['actual_llm_generation'])
        self.assertTrue(result['physical_cases_not_run'])
        self.assertFalse(result['registered'])
        self.assertTrue((out/'rejected_source.txt').is_file())
        self.assertTrue((out/'rejected_source_diagnostic.json').is_file())
        self.assertFalse((out/'candidate.py').exists())


if __name__ == '__main__':
    unittest.main()
