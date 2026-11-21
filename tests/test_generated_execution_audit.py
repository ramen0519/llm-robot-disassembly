"""A mailbox timeout must never turn a generated choice into a false success."""
from copy import deepcopy
import unittest
from tools.run_generated_skills import execution_bindings, physical_usage_audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.decision = {'request': {'request_id': 'r'}, 'response': {
            'action_id': 'generated_action', 'choice': {'part_id': '0'},
            'metadata': {'generated_program_executed': True, 'generated_program_hash': 'hash',
                         'generated_skill_id': 'hash'}}}
        self.attempt = {'request_id': 'r', 'action_id': 'generated_action',
                        'choice': {'part_id': '0'}, 'raw': {'chosen': {'part_id': '0'}}}

    def test_exact_received_and_executed_choice_is_counted(self):
        good, bad = execution_bindings([self.decision], {
            'decisions': [self.decision], 'attempts': [self.attempt]})
        self.assertEqual((len(good), len(bad)), (1, 0))

    def test_same_request_fallback_after_timeout_is_not_generated_success(self):
        replacement = deepcopy(self.decision)
        replacement['response']['metadata'] = {'generated_program_executed': False}
        replacement['response']['action_id'] = 'fallback'
        good, bad = execution_bindings([self.decision], {
            'decisions': [replacement], 'attempts': [self.attempt]})
        self.assertEqual((len(good), len(bad)), (0, 1))

    def test_raw_physics_choice_must_match(self):
        self.attempt['raw']['chosen'] = {'part_id': '1'}
        good, bad = execution_bindings([self.decision], {
            'decisions': [self.decision], 'attempts': [self.attempt]})
        self.assertEqual((len(good), len(bad)), (0, 1))

    def test_worker_skill_id_must_match_even_with_same_program_and_choice(self):
        worker = deepcopy(self.decision)
        worker['response']['metadata']['generated_skill_id'] = 'different_skill'
        good, bad = execution_bindings([self.decision], {
            'decisions': [worker], 'attempts': [self.attempt]})
        self.assertEqual((len(good), len(bad)), (0, 1))

    def test_missing_skill_identity_is_not_counted(self):
        self.decision['response']['metadata'].pop('generated_skill_id')
        good, bad = execution_bindings([self.decision], {
            'decisions': [self.decision], 'attempts': [self.attempt]})
        self.assertEqual((len(good), len(bad)), (0, 1))

    def test_conflicting_duplicate_request_is_not_last_write_wins(self):
        other = deepcopy(self.decision)
        other['response']['metadata']['generated_skill_id'] = 'other'
        good, bad = execution_bindings([other, self.decision], {
            'decisions': [self.decision], 'attempts': [self.attempt]})
        self.assertEqual((len(good), len(bad)), (0, 1))

    def live_bundle(self, identifier='r', identity='new_skill', *, passed=True):
        decision = deepcopy(self.decision)
        decision['request'].update(request_id=identifier, run_id='unit_fixture', state_id='state')
        response = decision['response']
        response['action_id'] = 'action:' + identifier
        selection_response = dict(request_id='selector:' + identifier, state_id='state',
            operation='select', status='selected', origin='actual_llm', selected_skill_id=identity,
            metadata={'llm_live': True})
        response['metadata'].update(generated_skill_id=identity, generated_program_hash=identity,
            generated_skill_selection_origin='actual_llm',
            generated_selector_response=deepcopy(selection_response), generated_library_sha256='library')
        selection = {'packet': dict(request_id='selector:' + identifier, run_id='unit_fixture',
            state_id='state', operation='select', current_request=deepcopy(decision['request']),
            skills=[{'skill_id': identity, 'program_hash': identity}],
            context={'library_sha256': 'library'}), 'response': selection_response}
        attempt = deepcopy(self.attempt)
        attempt.update(request_id=identifier, action_id=response['action_id'],
                       status='PASS' if passed else 'UNKNOWN',
                       accepted_for_transition=passed, deadline_exceeded=False)
        return decision, selection, attempt

    def test_exact_live_selector_and_physics_are_attributed_to_same_skill(self):
        decision, selection, attempt = self.live_bundle()
        audit = physical_usage_audit([decision], [selection], {
            'decisions': [decision], 'attempts': [attempt]})
        row = audit['physical_usage_by_skill']['new_skill']
        self.assertEqual(row['physical_attempts'], 1)
        self.assertEqual(row['physical_passes'], 1)
        self.assertEqual(row['actual_llm_selected_physical_attempts'], 1)
        self.assertEqual(row['actual_llm_selected_physical_passes'], 1)
        self.assertEqual(row['actual_llm_selected_request_ids'], ['r'])
        self.assertEqual(audit['actual_llm_selected_physical_pass_skill_ids'], ['new_skill'])

    def test_new_selection_cannot_borrow_old_skill_physics(self):
        old, old_selection, old_attempt = self.live_bundle('old_request', 'old_skill')
        new, new_selection, new_attempt = self.live_bundle('new_request', 'new_skill')
        replacement = deepcopy(new)
        replacement['response']['metadata']['generated_program_executed'] = False
        audit = physical_usage_audit([old, new], [old_selection, new_selection], {
            'decisions': [old, replacement], 'attempts': [old_attempt, new_attempt]})
        self.assertEqual(audit['physically_executed_skill_ids'], ['old_skill'])
        self.assertEqual(audit['actual_llm_selected_physical_pass_skill_ids'], ['old_skill'])
        self.assertNotIn('new_skill', audit['physical_usage_by_skill'])

    def test_different_request_selector_cannot_lend_actual_origin(self):
        decision, selection, attempt = self.live_bundle()
        selection['packet']['current_request']['request_id'] = 'other_request'
        audit = physical_usage_audit([decision], [selection], {
            'decisions': [decision], 'attempts': [attempt]})
        self.assertEqual(audit['physically_executed_skill_ids'], ['new_skill'])
        self.assertEqual(audit['actual_llm_selected_physically_executed_skill_ids'], [])

    def test_mismatched_selector_program_hash_or_library_cannot_lend_origin(self):
        for field in ('program_hash', 'library_sha256'):
            with self.subTest(field=field):
                decision, selection, attempt = self.live_bundle()
                if field == 'program_hash':
                    selection['packet']['skills'][0][field] = 'other'
                else:
                    selection['packet']['context'][field] = 'other'
                audit = physical_usage_audit([decision], [selection], {
                    'decisions': [decision], 'attempts': [attempt]})
                self.assertEqual(audit['actual_llm_selected_physical_attempts'], 0)

    def test_injected_or_nonlive_selector_does_not_count_as_actual_llm(self):
        for change in ({'origin': 'injected_transport'}, {'metadata': {'llm_live': False}}):
            with self.subTest(change=change):
                decision, selection, attempt = self.live_bundle()
                selection['response'].update(change)
                decision['response']['metadata']['generated_selector_response'] = deepcopy(selection['response'])
                audit = physical_usage_audit([decision], [selection], {
                    'decisions': [decision], 'attempts': [attempt]})
                self.assertEqual(audit['actual_llm_selected_physical_attempts'], 0)
                self.assertEqual(audit['physical_usage_by_skill']['new_skill']['physical_attempts'], 1)

    def test_failure_is_executed_but_not_pass(self):
        decision, selection, attempt = self.live_bundle(passed=False)
        audit = physical_usage_audit([decision], [selection], {
            'decisions': [decision], 'attempts': [attempt]})
        self.assertEqual(audit['actual_llm_selected_physical_attempts'], 1)
        self.assertEqual(audit['actual_llm_selected_physical_passes'], 0)
        self.assertEqual(audit['physical_pass_skill_ids'], [])

    def test_late_or_unaccepted_raw_pass_is_not_transition_pass(self):
        for change in ({'deadline_exceeded': True}, {'accepted_for_transition': False},
                       {'status': 'UNKNOWN'}):
            with self.subTest(change=change):
                decision, selection, attempt = self.live_bundle()
                attempt.update(change)
                audit = physical_usage_audit([decision], [selection], {
                    'decisions': [decision], 'attempts': [attempt]})
                self.assertEqual(audit['actual_llm_selected_physical_attempts'], 1)
                self.assertEqual(audit['actual_llm_selected_physical_passes'], 0)

    def test_worker_selector_response_mismatch_prevents_live_attribution(self):
        decision, selection, attempt = self.live_bundle()
        worker = deepcopy(decision)
        worker['response']['metadata']['generated_selector_response']['selected_skill_id'] = 'other'
        audit = physical_usage_audit([decision], [selection], {
            'decisions': [worker], 'attempts': [attempt]})
        self.assertEqual(audit['physically_executed_skill_ids'], ['new_skill'])
        self.assertEqual(audit['actual_llm_selected_physical_attempts'], 0)

    def test_actual_origin_ledger_replay_remains_bound_to_same_request(self):
        decision, selection, attempt = self.live_bundle()
        selection['response']['metadata'] = {'llm_live': False, 'original_llm_live': True}
        decision['response']['metadata']['generated_selector_response'] = deepcopy(selection['response'])
        audit = physical_usage_audit([decision], [selection], {
            'decisions': [decision], 'attempts': [attempt]})
        self.assertEqual(audit['actual_llm_selected_physical_attempts'], 1)
