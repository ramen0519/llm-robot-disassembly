"""Append/freeze/retrieve observation tests with fake, labelled physics records."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_explicit_action_contract import request
from worker.explicit_action_contract import canonical_response
from orchestrator.explicit_action_memory import (append_action_observation, freeze_memory,
    load_observations, load_snapshot, FrozenActionMemory)


def outcome(status='PASS'):
    return {'status': status, 'path_success': status == 'PASS',
        'stability_success': True if status == 'PASS' else None,
        'verifier_origin': 'fixture_not_actual_physics', 'raw': {'fixture': True},
        'evidence_ref': 'fixture/events.jsonl#1'}


class ExplicitActionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='explicit_memory_fixture_')
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.log = self.root/'observations.jsonl'; self.frozen = self.root/'frozen.json'

    def add(self, r=None, status='PASS', parent=None):
        r = r or request()
        return append_action_observation(self.log, r, canonical_response(r, None), outcome(status),
                                          parent, allow_test_root=True)

    def freeze(self):
        return freeze_memory(self.log, self.frozen, allow_test_root=True)

    def test_exact_observation_and_raw_fixture_identity_preserved(self):
        record = self.add(); self.assertEqual(load_observations(self.log, True), [record])
        self.assertEqual(record['result']['verifier_origin'], 'fixture_not_actual_physics')

    def test_duplicate_same_observation_idempotent(self):
        original = self.add(); content = self.log.read_bytes()
        self.assertEqual(self.add(), original); self.assertEqual(self.log.read_bytes(), content)

    def test_conflicting_duplicate_rejected_without_overwrite(self):
        self.add(); content = self.log.read_bytes()
        with self.assertRaisesRegex(ValueError, 'conflicting'): self.add(status='UNKNOWN')
        self.assertEqual(self.log.read_bytes(), content)

    def test_snapshot_frozen_even_when_new_episode_appended(self):
        self.add(); snapshot = self.freeze(); content = self.frozen.read_bytes()
        r = request(); r['request_id'] = 'fixture:r2'; self.add(r, 'UNKNOWN')
        self.assertEqual(self.frozen.read_bytes(), content)
        self.assertEqual(len(FrozenActionMemory(snapshot).retrieve(request())), 1)
        self.assertEqual(len(load_observations(self.log, True)), 2)

    def test_snapshot_refuses_overwrite(self):
        self.add(); self.freeze()
        with self.assertRaises(FileExistsError): self.freeze()

    def test_same_conditions_still_require_new_physics(self):
        self.add(); memory = FrozenActionMemory(self.freeze()); refs = memory.retrieve(request())
        self.assertTrue(refs[0]['same_conditions']); self.assertFalse(refs[0]['reference_only'])
        self.assertFalse(refs[0]['physics_reuse_allowed']); self.assertFalse(refs[0]['hard_pruning'])
        self.assertFalse(refs[0]['current_attempt_verified'])

    def test_cross_task_never_maps_source_parts(self):
        self.add(); memory = FrozenActionMemory(self.freeze()); r = request()
        r['task_context']['assembly_id'] = 'other'; r['task_context']['input_sha256'] = 'b'*64
        refs = memory.retrieve(r)
        self.assertTrue(refs[0]['reference_only']); self.assertIsNone(refs[0]['current_part_id_mapping'])
        self.assertFalse(refs[0]['same_task']); self.assertEqual(refs[0]['source_choice']['part_id'], '1')

    def test_seed_setting_change_prevents_exact_condition_claim(self):
        self.add(); memory = FrozenActionMemory(self.freeze()); r = request()
        r['task_context']['solver_settings']['seed'] = 2
        self.assertTrue(memory.retrieve(r)[0]['reference_only'])

    def test_changed_remaining_parts_prevents_exact_condition_claim(self):
        self.add(); memory = FrozenActionMemory(self.freeze()); r = request()
        r['remaining_parts'] = ['1', 'base']; r['candidate_ids'] = ['1']
        ref = memory.retrieve(r)[0]
        self.assertTrue(ref['same_task']); self.assertFalse(ref['same_conditions'])

    def test_failures_retained_without_invented_cause(self):
        self.add(status='UNKNOWN'); ref = FrozenActionMemory(self.freeze()).retrieve(request())[0]
        self.assertEqual(ref['status'], 'UNKNOWN'); self.assertIsNone(ref['failure_cause'])
        self.assertIsNone(ref['blocker']); self.assertFalse(ref['hard_pruning'])

    def test_recovery_links_actual_recorded_failure_without_generalization(self):
        failed = self.add(status='UNKNOWN'); r = request(); r['request_id'] = 'fixture:r2'
        r['attempted_action_ids'] = [failed['source_action_id']]
        recovered = self.add(r, 'PASS', failed['evidence_id'])
        refs = FrozenActionMemory(self.freeze()).retrieve(request())
        ref = next(x for x in refs if x['evidence_id'] == recovered['evidence_id'])
        self.assertEqual(ref['repair_lifecycle'], 'provisional_observed_recovery_only')

    def test_nonexistent_failure_link_rejected(self):
        with self.assertRaisesRegex(ValueError, 'parent_must'): self.add(parent='action-evidence:'+'a'*64)

    def test_success_cannot_be_parent_failure(self):
        first = self.add(); r = request(); r['request_id'] = 'fixture:r2'
        with self.assertRaisesRegex(ValueError, 'parent_must'): self.add(r, parent=first['evidence_id'])

    def test_identical_action_repeat_is_not_changed_action_recovery(self):
        first = self.add(status='UNKNOWN'); r = request(); r['request_id'] = 'fixture:r2'
        with self.assertRaisesRegex(ValueError, 'changed_action_conditions'):
            self.add(r, parent=first['evidence_id'])

    def test_pass_without_stability_evidence_rejected(self):
        r = request(); result = outcome(); result['stability_success'] = None
        with self.assertRaisesRegex(ValueError, 'both_actual_stages'):
            append_action_observation(self.log, r, canonical_response(r, None), result, allow_test_root=True)

    def test_forged_record_and_snapshot_hash_rejected(self):
        self.add(); snapshot = self.freeze(); snapshot['records'][0]['result']['status'] = 'UNKNOWN'
        with self.assertRaises(ValueError): FrozenActionMemory(snapshot)

    def test_project_guard_rejects_excluded_path_before_stat(self):
        with patch('pathlib.Path.stat', side_effect=AssertionError('must_not_touch')):
            with self.assertRaises(ValueError): load_observations('E:/never/open.jsonl')

    def test_immutable_input_copy(self):
        self.add(); snapshot = self.freeze(); memory = FrozenActionMemory(snapshot)
        snapshot['records'].clear(); first = memory.retrieve(request()); first[0]['status'] = 'ERROR'
        self.assertEqual(memory.retrieve(request())[0]['status'], 'PASS')

    def test_empty_memory_is_valid(self):
        self.assertEqual(FrozenActionMemory().retrieve(request()), [])

    def test_frozen_file_load_validation(self):
        self.add(); snapshot = self.freeze(); self.assertEqual(load_snapshot(self.frozen, True), snapshot)

    def test_measured_endpoint_is_preserved_as_source_only_cross_task_evidence(self):
        r = request(); result = outcome('UNKNOWN')
        endpoint = {'scope': 'selected_trajectory_terminal_state_only', 'min_sep': .05,
            'pairs': [{'moving_part_id': '1', 'rest_part_id': '2', 'min_distance': .02, 'separation_unsatisfied': True}],
            'termination_reason': 'unknown', 'evidence_sha256': 'e'*64,
            'blocker': None, 'all_paths_impossible': False}
        result['raw']['endpoint_observation'] = deepcopy(endpoint)
        append_action_observation(self.log, r, canonical_response(r, None), result, allow_test_root=True)
        memory = FrozenActionMemory(self.freeze()); target = request()
        target['task_context']['assembly_id'] = 'another_assembly'
        ref = memory.retrieve(target)[0]
        self.assertEqual(ref['measured_endpoint'], endpoint)
        self.assertTrue(ref['reference_only']); self.assertIsNone(ref['current_part_id_mapping'])
        self.assertEqual(ref['measured_endpoint']['pairs'][0]['moving_part_id'], '1')
        self.assertIsNone(ref['failure_cause']); self.assertIsNone(ref['blocker'])
        self.assertFalse(ref['hard_pruning']); self.assertFalse(ref['physics_reuse_allowed'])
        ref['measured_endpoint']['pairs'][0]['rest_part_id'] = 'changed_by_caller'
        self.assertEqual(memory.retrieve(target)[0]['measured_endpoint'], endpoint)

    def test_raw_stability_parts_preserved_without_inferred_cause_or_id_mapping(self):
        r = request(); result = outcome('UNKNOWN')
        result.update(path_success=True, stability_success=False)
        result['raw']['raw_upstream'] = {'parts_fall': ['2']}
        append_action_observation(self.log, r, canonical_response(r, None), result, allow_test_root=True)
        target = request(); target['task_context']['assembly_id'] = 'another_assembly'
        ref = FrozenActionMemory(self.freeze()).retrieve(target)[0]
        self.assertEqual(ref['raw_stability_parts'], ['2'])
        self.assertIsNone(ref['current_part_id_mapping']); self.assertIsNone(ref['failure_cause'])
        self.assertIsNone(ref['measured_endpoint']); self.assertFalse(ref['current_attempt_verified'])

    def test_unprovided_endpoint_and_stability_ids_stay_null(self):
        self.add(status='UNKNOWN'); ref = FrozenActionMemory(self.freeze()).retrieve(request())[0]
        self.assertIsNone(ref['measured_endpoint']); self.assertIsNone(ref['raw_stability_parts'])


if __name__ == '__main__': unittest.main()
