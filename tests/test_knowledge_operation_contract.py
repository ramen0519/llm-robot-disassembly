"""Offline regression of actual read errors; no live API or new physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from orchestrator.knowledge_tools import (KnowledgeCatalog, OPERATION_CONTRACT,
    operation_schema, operation_instructions, validate_operation)
from orchestrator.knowledge_retrieval import KnowledgeRetrieval, output_schema
from tests import test_agent_knowledge as existing


ROOT = Path(__file__).resolve().parents[1]


class KnowledgeOperationContractTests(unittest.TestCase):
    owner = existing.AgentKnowledgeTests.owner
    transport = existing.AgentKnowledgeTests.transport

    def setUp(self):
        existing.AgentKnowledgeTests.setUp(self)
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NETWORK_FORBIDDEN'))
        guard = network.start()
        self.addCleanup(network.stop)
        self.addCleanup(guard.assert_not_called)

    def scripted(self, batches, *, rounds=2, cfg=None):
        sent = []
        def transport(url, headers, body, timeout):
            sent.append(deepcopy(body))
            batch = batches[len(sent)-1]
            if isinstance(batch, Exception):
                raise batch
            packet = json.loads(body['input'][0]['content'])
            raw = existing.provider(dict(request_id=packet['request_id'], state_id=packet['state_id'],
                finish=True, reason='OFFLINE contract regression', operations=deepcopy(batch)))
            raw['model'] = body['model']
            return raw
        agent = KnowledgeRetrieval(self.catalog, self.owner(transport, cfg=cfg), max_rounds=rounds)
        view = agent.retrieve(self.req, time.monotonic()+90)
        return view, sent, agent

    def test_actual_three_malformed_reads_now_accept_unchanged_typed_operations(self):
        source = ROOT/'runs/support_learning_live_20261109_1155/revalidation_v2'
        diagnosis = ROOT/'runs/support_free_reuse_20261109_1250/live_read_diagnosis.json'
        files = [source/'skills_frozen.json', source/'lessons_frozen.json', diagnosis]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
        skills, lessons, report = [json.loads(p.read_text(encoding='utf-8')) for p in files]
        session = KnowledgeCatalog(skills, lessons).session(self.req)
        self.assertEqual(len(report['malformed_reads']), 3)
        for failure in report['malformed_reads']:
            operation = failure['actual_operation']
            document = session.execute(operation)
            self.assertNotIn('error', document)
            self.assertEqual(document['document_id'], operation['document_id'])
            self.assertEqual(document['kind'], operation['kind'])
        self.assertEqual(len(session.read_documents), 2)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in files])

    def test_provider_schema_is_shared_and_distinguishes_read_from_search(self):
        schema = operation_schema()
        self.assertEqual(output_schema(self.req)['properties']['operations']['items'], schema)
        self.assertIn(OPERATION_CONTRACT, operation_instructions())
        self.assertIn(operation_instructions(), self.catalog.session(self.req).index()['operation_contract'])
        branches = schema['anyOf']
        self.assertEqual(branches[0]['properties']['tool']['enum'], ['search'])
        for branch in branches[1:]:
            fields = branch['properties']
            self.assertEqual(fields['tool']['enum'], ['read'])
            for key in ('query', 'trigger', 'offset', 'limit'):
                self.assertEqual(fields[key], {'type': 'null'})
        operation = dict(existing.read('skill:'+self.sid), kind='skill')
        self.assertEqual(validate_operation(operation), operation)

    def test_read_rejects_meaningless_search_fields_without_publishing_documents(self):
        for key, value in (('query', ''), ('trigger', 'all'), ('offset', 0), ('limit', 1)):
            with self.subTest(field=key):
                session = self.catalog.session(self.req)
                operation = dict(existing.read('skill:'+self.sid), **{key: value})
                with self.assertRaisesRegex(ValueError, 'read_cannot_search'):
                    session.execute(operation)
                self.assertEqual(session.read_documents, {})

    def test_read_kind_and_ids_are_checked_before_any_document_is_published(self):
        session = self.catalog.session(self.req)
        with self.assertRaisesRegex(ValueError, 'invalid_knowledge_read'):
            session.execute(dict(existing.read('skill:'+self.sid), kind='lesson'))
        with self.assertRaisesRegex(ValueError, 'invalid_knowledge_read'):
            session.execute(existing.read('../private-file'))
        self.assertEqual(session.execute(existing.read('skill:missing'))['error'], 'unknown_document_id')
        self.assertEqual(session.read_documents, {})
        self.assertEqual(session.execute(dict(existing.read('skill:'+self.sid), kind='skill'))['kind'], 'skill')

    def test_search_rejects_read_fields_null_arguments_and_boolean_offsets(self):
        for updates in ({'document_id': 'skill:'+self.sid}, {'query': None}, {'kind': None}, {'offset': True}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.catalog.session(self.req).execute(dict(existing.search(), **updates))

    def test_finish_error_gets_only_existing_bounded_correction_round(self):
        valid = dict(existing.read('skill:'+self.sid), kind='skill')
        invalid = dict(valid, query='irrelevant')
        view, sent, agent = self.scripted([[invalid], [valid]])
        self.assertEqual(len(sent), 2)
        self.assertEqual(view['audit']['retrieval_status'], 'ready')
        self.assertEqual(view['audit']['corrected_tool_error_count'], 1)
        self.assertEqual(view['audit']['correction_rounds'], 1)
        self.assertEqual(view['audit']['tool_error_codes'], ['read_cannot_search'])
        self.assertEqual(view['audit']['selected_skill_ids'], [self.sid])
        context = json.loads(sent[1]['input'][0]['content'])['context']
        self.assertEqual(context['rounds_remaining'], 1)
        self.assertEqual(context['unresolved_tool_errors'][0]['error'], 'read_cannot_search')
        self.assertTrue(all(call['response']['metadata'].get('retries', 0) == 0
            for call in view['retrieval_trace']['calls']))
        self.assertTrue(agent.retrieve(self.req, time.monotonic()+90)['audit']['retrieval_cache_hit'])
        self.assertEqual(len(sent), 2)

    def test_unresolved_error_is_rejected_not_ready_abstention(self):
        invalid = dict(existing.read('skill:'+self.sid), limit=1)
        view, sent, _ = self.scripted([[invalid]], rounds=1)
        self.assertEqual(len(sent), 1)
        self.assertEqual(view['audit']['retrieval_status'], 'rejected')
        self.assertEqual(view['audit']['stop_reason'], 'knowledge_tool_errors')
        self.assertTrue(view['audit']['knowledge_incomplete'])
        self.assertFalse(view['audit']['model_finished'])
        self.assertEqual(view['skills'], [])

    def test_partial_preserves_successful_reads_and_error_diagnostics(self):
        view, sent, _ = self.scripted([[existing.read('skill:'+self.sid), existing.read('skill:missing')]], rounds=1)
        self.assertEqual(view['audit']['retrieval_status'], 'partial')
        self.assertEqual(view['audit']['stop_reason'], 'knowledge_tool_errors')
        self.assertEqual(view['audit']['tool_error_codes'], ['unknown_document_id'])
        self.assertEqual(view['audit']['selected_skill_ids'], [self.sid])
        self.assertEqual(view['audit']['read_document_ids'], ['skill:'+self.sid])
        self.assertEqual(len(sent), 1)

    def test_empty_finish_cannot_erase_a_previous_tool_error(self):
        invalid = dict(existing.read('skill:'+self.sid), query='ignored?')
        view, sent, _ = self.scripted([[invalid], []])
        self.assertEqual(view['audit']['retrieval_status'], 'rejected')
        self.assertEqual(view['audit']['corrected_tool_error_count'], 0)
        self.assertEqual(len(view['audit']['unresolved_tool_errors']), 1)
        self.assertEqual(len(sent), 2)

    def test_successful_unrelated_search_does_not_claim_to_correct_missing_read(self):
        view, sent, _ = self.scripted([[existing.read('skill:missing')], [existing.search(kind='skill')]])
        self.assertEqual(view['audit']['retrieval_status'], 'rejected')
        self.assertEqual(view['audit']['stop_reason'], 'knowledge_tool_errors')
        self.assertEqual(view['audit']['corrected_tool_error_count'], 0)
        self.assertEqual(view['audit']['unresolved_tool_errors'][0]['error'], 'unknown_document_id')
        self.assertEqual(len(sent), 2)

    def test_reading_another_document_preserves_partial_status_and_original_missing_id(self):
        view, _, _ = self.scripted([[existing.read('skill:missing')], [existing.read('skill:'+self.sid)]])
        self.assertEqual(view['audit']['retrieval_status'], 'partial')
        self.assertEqual(view['audit']['selected_skill_ids'], [self.sid])
        self.assertEqual(view['audit']['corrected_tool_error_count'], 0)
        self.assertEqual(view['audit']['unresolved_tool_errors'][0]['operation']['document_id'], 'skill:missing')

    def test_transport_failure_during_correction_never_retries_or_reports_partial_success(self):
        invalid = dict(existing.read('skill:'+self.sid), query='bad')
        view, sent, _ = self.scripted([[invalid], TimeoutError('OFFLINE')])
        self.assertEqual(view['audit']['retrieval_status'], 'rejected')
        self.assertEqual(view['audit']['stop_reason'], 'api_timeout')
        self.assertEqual(len(sent), 2)

    def test_corrective_round_still_obeys_original_call_budget(self):
        cfg = existing.policy()['roles']['planner']; cfg['max_calls'] = 1
        invalid = dict(existing.read('skill:'+self.sid), query='bad')
        view, sent, _ = self.scripted([[invalid]], cfg=cfg)
        self.assertEqual(len(sent), 1)
        self.assertEqual(view['audit']['stop_reason'], 'call_budget_exhausted')
        self.assertEqual(view['audit']['retrieval_status'], 'rejected')

    def generated_case(self, partial):
        from tests.test_generated_skill_service import row, snapshot
        from tests.test_skill_program import append_observation, request
        from tests.test_skill_discovery import raw_response
        from orchestrator.generated_skill_service import GeneratedSkillService
        from orchestrator.explicit_action_service import ExplicitActionService
        from orchestrator.skill_discovery import SkillDiscoveryService
        from orchestrator.explicit_action_provider_v5 import restore_request_v5, encode_provider_decision_v5
        from worker.path_probe_round import canonical_path_round_response
        rows = [row()]; cfg = existing.policy()['roles']['scout']; cfg['max_calls'] = 4
        sent = []
        def transport(url, headers, body, timeout):
            sent.append(deepcopy(body)); packet = json.loads(body['input'][0]['content'])
            form = body['text']['format']['name']
            if form == 'knowledge_tool_requests':
                operations = [existing.read('skill:missing')]
                if partial:
                    operations.insert(0, dict(existing.read('skill:'+rows[0]['strategy_id']), kind='skill'))
                value = dict(request_id=packet['request_id'], state_id=packet['state_id'], finish=True,
                    reason='OFFLINE partial/rejected lookup', operations=operations)
            elif partial:
                self.assertEqual(packet['context']['retrieved_knowledge']['audit']['retrieval_status'], 'partial')
                value = dict(request_id=packet['request_id'], state_id=packet['state_id'],
                    selected_skill_id=rows[0]['strategy_id'], evidence_ids=[], explanation_hypothesis='OFFLINE')
            else:
                audit = packet['retrieved_knowledge']['audit']
                self.assertEqual(audit['retrieval_status'], 'rejected')
                self.assertEqual(audit['decision_context'], 'current_observation_only_after_retrieval_rejection')
                self.assertEqual(packet['retrieved_knowledge']['documents'], [])
                req = restore_request_v5({k: packet[k] for k in ('projection_version', 'request', 'shared_source_contexts')})
                value = encode_provider_decision_v5(req, canonical_path_round_response(req, None, fallback_policy='path_stage_v1'))
            raw = raw_response(value); raw['model'] = body['model']; return raw
        actor = ExplicitActionService(cfg, self.root/'generated.sqlite', transport,
            provider_contract='explicit_action_provider_v5', fallback_policy='path_stage_v1')
        selector = SkillDiscoveryService(cfg, actor.service.ledger_path, transport)
        with patch('orchestrator.generated_skill_service.retrieve', return_value=deepcopy(rows)), \
             patch('orchestrator.knowledge_context.retrieve', return_value=deepcopy(rows)):
            service = GeneratedSkillService(snapshot(), actor, selector,
                knowledge_policy='agent_tools_v1', knowledge_max_rounds=1)
            response = service.handle(append_observation(request()))
        self.assertEqual(len(sent), 2)
        self.assertEqual(response['metadata']['generated_program_executed'], partial)
        self.assertEqual(response['metadata']['knowledge_context']['retrieval_status'], 'partial' if partial else 'rejected')

    def test_generated_selector_can_use_valid_partial_documents_with_status_visible(self):
        self.generated_case(True)

    def test_generated_actor_receives_rejected_tool_error_as_current_observation_only(self):
        self.generated_case(False)


if __name__ == '__main__':
    unittest.main()
