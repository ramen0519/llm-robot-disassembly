"""Pure audit counterexamples; no API, secrets, physics, or historical writes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
from audit_live_skill_reuse import actual_selection, card_lineage, digest, reuse_demonstrated, unique


class LiveReuseAuditTests(unittest.TestCase):
    def setUp(self):
        network = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        mocked = network.start()
        self.addCleanup(network.stop)
        self.addCleanup(mocked.assert_not_called)
        self.first = 'new_trial/development_empty'
        context = {'seed': 0, 'setup': 'fixture'}
        self.records = {'e1': dict(record_id='e1', source={'run_id': self.first}, status='PASS',
            context=context, raw_result={'feasible': True}, request={'part_move': '1', 'parts_rest': ['0', '6']})}
        self.invocations = {'r1': {'request': {'run_id': self.first, 'base_part': '6', 'assembled_parts': ['0', '1', '6']}}}
        self.outcomes = {'e1': {'invocation_id': 'r1'}}
        self.card = dict(callable_id='promote_verified', lifecycle='provisional', source_evidence_ids=['e1'],
            source_invocation_id='r1', preconditions={'context_sha256': digest(context), 'assembled_parts': ['0', '1', '6'], 'base_part': '6'},
            parameters={'preferred_ids': ['1']})
        self.metadata = dict(attempted=True, llm_live=True, selected_by='llm', model_returned='gpt-6-luna',
            response_id='resp_fixture', usage={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2},
            llm_fallback_reason=None, skill_error_type=None)
        self.logged = dict(status='completed', reserved_call=1,
            response_json=json.dumps({'metadata': {**self.metadata, 'selected_skill_id': 'learned'}, 'fallback_reason': None}))
        self.detail = dict(variant='reuse_frozen', actual_luna_selection=True, callable_id='promote_verified',
            first_episode_card_lineage=True, execution_and_outcomes_valid=True)

    def lineage(self):
        return card_lineage(self.card, self.records, self.first, self.invocations, self.outcomes)

    def test_valid_positive_lineage_binds_source_state_and_invocation(self):
        self.assertTrue(self.lineage())

    def test_old_or_second_episode_evidence_rejected_even_with_same_id(self):
        for source in ('old_trial/development_empty', 'new_trial/reuse_frozen'):
            with self.subTest(source=source):
                self.records['e1']['source']['run_id'] = source
                self.assertFalse(self.lineage())

    def test_missing_or_duplicate_evidence_and_unknown_outcomes_rejected(self):
        original = deepcopy(self.card)
        for ids in ([], ['missing'], ['e1', 'e1']):
            self.card['source_evidence_ids'] = ids
            self.assertFalse(self.lineage())
        self.card = original
        for status in ('UNKNOWN', 'ERROR'):
            self.records['e1']['status'] = status
            self.assertFalse(self.lineage())

    def test_positive_label_without_raw_success_rejected(self):
        self.records['e1']['raw_result']['feasible'] = False
        self.assertFalse(self.lineage())

    def test_cross_state_context_or_outcome_binding_rejected(self):
        for field, value in (('context_sha256', 'different'), ('assembled_parts', ['1', '6']), ('base_part', '0')):
            original = deepcopy(self.card)
            self.card['preconditions'][field] = value
            self.assertFalse(self.lineage())
            self.card = original
        self.outcomes['e1']['invocation_id'] = 'wrong'
        self.assertFalse(self.lineage())

    def test_unsupported_preferred_candidate_rejected(self):
        self.card['parameters']['preferred_ids'] = ['0']
        self.assertFalse(self.lineage())

    def test_provider_response_bound_to_selected_skill_and_usage(self):
        self.assertTrue(actual_selection(self.logged, self.metadata, 'learned'))
        self.assertFalse(actual_selection(self.logged, self.metadata, 'other'))
        changed = {**self.metadata, 'usage': {'total_tokens': 9}}
        self.assertFalse(actual_selection(self.logged, changed, 'learned'))

    def test_fallback_offline_wrong_model_or_pending_never_live_success(self):
        for field, value in (('selected_by', 'deterministic_fallback'), ('offline_fixture', True),
                             ('model_returned', 'other'), ('attempted', False),
                             ('llm_fallback_reason', 'call_budget_exhausted'), ('skill_error_type', 'ValueError')):
            self.assertFalse(actual_selection(self.logged, {**self.metadata, field: value}, 'learned'))
        self.assertFalse(actual_selection({**self.logged, 'status': 'pending'}, self.metadata, 'learned'))
        self.assertFalse(actual_selection({**self.logged, 'reserved_call': 0}, self.metadata, 'learned'))

    def test_native_choice_is_not_learned_reuse(self):
        self.assertTrue(reuse_demonstrated([self.detail]))
        self.assertFalse(reuse_demonstrated([{**self.detail, 'callable_id': 'native_order'}]))

    def test_first_episode_or_broken_execution_cannot_demonstrate_reuse(self):
        for field, value in (('variant', 'development_empty'), ('actual_luna_selection', False),
                             ('first_episode_card_lineage', False), ('execution_and_outcomes_valid', False)):
            self.assertFalse(reuse_demonstrated([{**self.detail, field: value}]))
        self.assertFalse(reuse_demonstrated([]))

    def test_duplicate_artifact_identity_rejected(self):
        with self.assertRaises(ValueError):
            unique([{'id': 'same'}, {'id': 'same'}], 'id')


if __name__ == '__main__':
    unittest.main()
