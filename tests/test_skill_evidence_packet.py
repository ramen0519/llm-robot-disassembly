"""API/physics-free projection tests and read-only replay of saved requests.

Synthetic backend labels only exercise contracts. Saved historical observations
are not new physics runs, and these tests make no performance claims.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from orchestrator.skill_evidence_packet import (
    compact_generation_packet, packet_size_components, restore_debug_request,
)
from orchestrator.skill_refinement import _bound_case
from tests.test_skill_program import program
from tests.test_skill_refinement import feedback


ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'runs/initial_contact_skill_followup_20261019/live_01'


def packet():
    items = feedback()
    items[0]['observed_program'] = program()
    event = items[0]['current_observation_request']['evidence_ids'][0]
    observation = dict(source_task='synthetic_only', remaining_parts=['b', 'a', 'base'],
        choice=deepcopy(items[0]['trigger']['chosen']), status='UNKNOWN',
        path_success=False, stability_success=None, failure_cause=None,
        deadline_exceeded=False, all_paths_impossible=False)
    return dict(run_id='unit', request_id='unit:generation', state_id='unit', operation='refine',
        context=dict(target_failure_scope={'kind': 'initial_disconnected'},
            frozen_episode_memory={'binding_checked': True}),
        evidence=[dict(observation, evidence_id=event, evidence_origin='episode_trace',
                       frozen_memory_evidence_id='memory:0'),
                  dict(observation, evidence_id='memory:0', evidence_origin='frozen_episode_memory',
                       source_event_id=event, same_observation_as_event_id=event)],
        existing_skills=[], previous_program=program(), validation_feedback=items)


class EvidencePacketTests(unittest.TestCase):
    def test_legacy_is_an_independent_unchanged_copy(self):
        full = packet()
        full['context']['target_failure_scope']['kind'] = 'dynamic_stability_failed'
        value, audit = compact_generation_packet(full)
        self.assertEqual(value, full)
        self.assertIsNot(value, full)
        self.assertFalse(audit['applied'])

    def test_original_unchanged_alias_citation_and_exact_local_roundtrip(self):
        full = packet()
        before = deepcopy(full)
        value, audit = compact_generation_packet(full)
        self.assertEqual(full, before)
        self.assertEqual({r['evidence_id'] for r in value['evidence']},
                         {r['evidence_id'] for r in full['evidence']})
        self.assertEqual(value['evidence'][1]['alias_of_evidence_id'], full['evidence'][0]['evidence_id'])
        request = _bound_case(full['validation_feedback'][0])[0]
        self.assertEqual(restore_debug_request(value, request['request_id'], audit), request)
        self.assertEqual(audit['new_api_calls'], 0)
        self.assertEqual(audit['new_physical_calls'], 0)
        self.assertFalse(audit['provider_request_is_lossless'])
        with self.assertRaisesRegex(ValueError, 'requires_local_audit'):
            restore_debug_request(value, request['request_id'])

    def test_local_request_mutation_detected(self):
        value, audit = compact_generation_packet(packet())
        request_id = next(iter(audit['local_debug_requests']))
        audit['local_debug_requests'][request_id]['candidate_ids'].append('fabricated')
        with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
            restore_debug_request(value, request_id, audit)

    def test_unknown_status_and_null_stability_are_not_relabelled(self):
        value, _ = compact_generation_packet(packet())
        row = value['debug_reasoning_table'][0]
        self.assertEqual(row['trigger']['status'], 'UNKNOWN')
        self.assertIsNone(row['trigger']['stability_success'])
        self.assertIsNone(row['trigger']['failure_cause'])
        self.assertFalse(row['trigger']['all_paths_impossible'])
        self.assertIsNone(row['recommendation'])
        self.assertEqual(value['context']['behavior_examples'][0]['observed_pass_choices'], [])

    def test_alias_disagreement_fails_closed(self):
        full = packet()
        full['evidence'][1]['path_success'] = True
        with self.assertRaisesRegex(ValueError, 'observation_disagrees'):
            compact_generation_packet(full)

    def test_alias_requires_bound_source_context(self):
        full = packet()
        full['context']['frozen_episode_memory']['binding_checked'] = False
        with self.assertRaisesRegex(ValueError, 'requires_bound_source_context'):
            compact_generation_packet(full)

    def test_alias_requires_reciprocal_trace_link(self):
        full = packet()
        del full['evidence'][0]['frozen_memory_evidence_id']
        with self.assertRaisesRegex(ValueError, 'unbound_memory_trace_alias'):
            compact_generation_packet(full)

    def test_unknown_citations_and_duplicate_evidence_rejected(self):
        full = packet()
        full['context']['evidence_ids'] = ['invented']
        with self.assertRaisesRegex(ValueError, 'unknown_citation'):
            compact_generation_packet(full)
        full = packet()
        full['evidence'].append(deepcopy(full['evidence'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate_evidence_id'):
            compact_generation_packet(full)

    def test_admission_and_transfer_never_projected(self):
        for phase in ('admission', 'transfer'):
            for location in ('feedback', 'evidence', 'context'):
                full = packet()
                if location == 'feedback':
                    full['validation_feedback'][0]['phase'] = phase
                elif location == 'evidence':
                    full['evidence'][0]['source_phase'] = phase
                else:
                    full['context']['unexpected_results'] = {'source_phase': phase}
                with self.subTest(phase=phase, location=location):
                    with self.assertRaisesRegex(ValueError, 'non_debug'):
                        compact_generation_packet(full)

    def test_invalid_interpreter_trace_rejected_before_projection(self):
        full = packet()
        full['validation_feedback'][0]['interpreter_trace']['program_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            compact_generation_packet(full)

    def test_request_id_conflict_rejected(self):
        full = packet()
        other = deepcopy(full['validation_feedback'][0])
        other['current_observation_request']['geometry']['parts']['a']['volume'] = 3.0
        full['validation_feedback'].append(other)
        with self.assertRaisesRegex(ValueError, 'request_identity_conflict'):
            compact_generation_packet(full)

    def test_size_bounds_and_nonfinite_values_fail_without_side_effects(self):
        for constant in ('MAX_FULL_BYTES', 'MAX_PROVIDER_BYTES', 'MAX_NODES'):
            with self.subTest(constant=constant):
                with patch('orchestrator.skill_evidence_packet.' + constant, 1):
                    with self.assertRaises(ValueError):
                        compact_generation_packet(packet())
        full = packet()
        full['context']['invalid'] = float('nan')
        with self.assertRaises(ValueError):
            compact_generation_packet(full)

    def test_component_measurement_does_not_double_count(self):
        full = packet()
        sizes = packet_size_components(full)
        expected = len(json.dumps(full, separators=(',', ':'), ensure_ascii=False,
                                  sort_keys=True).encode('utf-8'))
        self.assertEqual(sizes['total_bytes'], expected)
        self.assertTrue(sizes['nested_component_counts_overlap'])


@unittest.skipUnless((SAVED / 'candidate_02/generation_packet.json').is_file(),
                     'saved historical packet unavailable; synthetic contracts still run')
class SavedEvidencePacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = SAVED / 'candidate_02/generation_packet.json'
        cls.raw = cls.path.read_bytes()
        cls.full = json.loads(cls.raw)
        cls.value, cls.audit = compact_generation_packet(cls.full)

    def test_real_saved_requests_exact_locally_preserved_without_new_physics(self):
        for item in self.full['validation_feedback']:
            request = _bound_case(item)[0]
            self.assertEqual(restore_debug_request(self.value, request['request_id'], self.audit), request)
        self.assertEqual(self.audit['bound_debug_case_count'], 3)
        self.assertEqual(len(self.audit['same_observation_aliases']), 12)
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertLess(self.audit['provider_bytes'], self.audit['full_bytes'] * .45)

    def test_all_twelve_source_events_failure_denominator_and_unmatched_remain(self):
        before = self.full['context']['initial_contact_observed_transitions']
        after = self.value['context']['initial_contact_observed_transitions']
        for key in ('trace_attempt_count', 'phase_denominator', 'unmatched_initial_failures',
                    'total_pair_count', 'initial_failure_count', 'excluded_attempt_count'):
            self.assertEqual(after[key], before[key])
        self.assertEqual(len(after['observations']), 12)
        indexed = {r['evidence_id']: r for r in self.value['evidence']}
        for row in before['observations']:
            current = indexed[row['evidence_id']]
            self.assertEqual(current['recorded_condition']['attempt_index'], row['attempt_index'])
            self.assertEqual(current['path_success'], row['path_success'])
            self.assertEqual(current['stability_success'], row['stability_success'])
            self.assertEqual(current['choice']['part_id'], row['choice']['part_id'])
        self.assertEqual({row['evidence_id'] for row in self.value['evidence']},
                         {row['evidence_id'] for row in self.full['evidence']})

    def test_debug_pass_failure_and_semantic_context_examples_are_visible(self):
        rows = self.value['debug_reasoning_table']
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['attempted_recovery']['status'] for r in rows],
                         [r['recovery']['status'] for r in self.full['validation_feedback']])
        self.assertTrue(all(r['observed_same_context_pass_choices'] for r in rows))
        for row in rows:
            self.assertEqual(row['observed_same_context_pass_choices'][0]['part_id'], '4')
            self.assertEqual(row['observed_same_context_pass_choices'][0]['direction_id'], 'zp')
            self.assertIsNone(row['recommendation'])
        self.assertEqual(len(self.value['context']['behavior_examples']), 3)

    def test_exact_features_pose_and_diagnostic_types_retained(self):
        tables = self.value['tables']
        parts = self.full['validation_feedback'][0]['current_observation_request']['geometry']['parts']
        self.assertIn(parts, tables['part_features'].values())
        self.assertNotEqual(parts['2']['volume'], parts['3']['volume'])
        pose = self.full['validation_feedback'][0]['current_observation_request']['pose_catalog'][0]['matrix']
        self.assertIn(pose, tables['poses'].values())
        diagnostics = list(tables['stability_diagnostics'].values())
        self.assertTrue(any(r['phase'] == 'initial_disconnected' and r['physics_steps'] == 0 for r in diagnostics))
        self.assertTrue(any(r['phase'] == 'dynamic_failure' for r in diagnostics))

    def test_transition_forgery_rejected(self):
        full = deepcopy(self.full)
        full['context']['initial_contact_observed_transitions']['pairs'][0]['before']['choice']['part_id'] = '4'
        with self.assertRaisesRegex(ValueError, 'transition_pair_observation_mismatch'):
            compact_generation_packet(full)

    def test_source_choice_forgery_rejected(self):
        full = deepcopy(self.full)
        # Alter both duplicate views: alias consistency alone must not suffice.
        for row in full['evidence']:
            if row['evidence_id'] == full['evidence'][0]['evidence_id'] or row.get('same_observation_as_event_id') == full['evidence'][0]['evidence_id']:
                row['choice']['part_id'] = '4'
        with self.assertRaisesRegex(ValueError, 'source_transition_choice_mismatch'):
            compact_generation_packet(full)


if __name__ == '__main__':
    unittest.main()
