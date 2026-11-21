"""Diagnostic persistence/retrieval with explicitly synthetic observations."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_explicit_action_contract import request
from test_explicit_action_memory import outcome
from worker.explicit_action_contract import canonical_response
from worker.stability_diagnostics import StabilityObservation, project_stability_diagnostics
from orchestrator.explicit_action_memory import (append_action_observation, freeze_memory,
                                               load_observations, FrozenActionMemory)


class StabilityMemoryProjectionTests(unittest.TestCase):
    def make_memory(self, root, diagnostic=None):
        req = request()
        result = outcome('UNKNOWN')
        result['path_success'] = True
        result['stability_success'] = False
        if diagnostic is not None:
            result['raw']['stability_diagnostics'] = deepcopy(diagnostic)
        log = root / 'observations.jsonl'
        record = append_action_observation(log, req, canonical_response(req, None), result, allow_test_root=True)
        frozen = root / 'frozen.json'
        snapshot = freeze_memory(log, frozen, allow_test_root=True)
        return req, record, log, frozen, FrozenActionMemory(snapshot)

    def diagnostic(self):
        observed = StabilityObservation()
        observed.checker_count = 1
        observed.data['initial_disconnected_parts'] = ['2']
        observed.returned(False, ['2'])
        return observed.data

    def test_raw_preserved_and_only_allowlisted_projection_retrieved(self):
        diagnostic = self.diagnostic()
        with tempfile.TemporaryDirectory() as directory:
            req, record, log, frozen, memory = self.make_memory(Path(directory), diagnostic)
            before = frozen.read_bytes()
            references = memory.retrieve(req)
            self.assertEqual(load_observations(log, True)[0]['result']['raw']['stability_diagnostics'], diagnostic)
            self.assertEqual(references[0]['stability_diagnostics'], project_stability_diagnostics(diagnostic))
            self.assertNotIn('initial_contact_graph', references[0]['stability_diagnostics'])
            self.assertEqual(frozen.read_bytes(), before)
            self.assertEqual(references[0]['verifier_origin'], 'fixture_not_actual_physics')
            self.assertFalse(references[0]['physics_reuse_allowed'])
            self.assertFalse(references[0]['hard_pruning'])

    def test_cross_task_diagnostic_keeps_source_ids_without_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            req, _, _, _, memory = self.make_memory(Path(directory), self.diagnostic())
            req['task_context']['assembly_id'] = 'different'
            req['task_context']['input_sha256'] = 'b'*64
            ref = memory.retrieve(req)[0]
            self.assertTrue(ref['reference_only'])
            self.assertIsNone(ref['current_part_id_mapping'])
            self.assertEqual(ref['stability_diagnostics']['initial_disconnected_parts'], ['2'])

    def test_legacy_observation_does_not_gain_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            req, _, _, _, memory = self.make_memory(Path(directory))
            self.assertNotIn('stability_diagnostics', memory.retrieve(req)[0])

    def test_malformed_diagnostic_is_not_promoted_to_evidence(self):
        diagnostic = self.diagnostic()
        diagnostic['phase'] = 'ALL_PATHS_IMPOSSIBLE'
        with tempfile.TemporaryDirectory() as directory:
            req, _, _, _, memory = self.make_memory(Path(directory), diagnostic)
            with self.assertRaisesRegex(ValueError, 'diagnostic_schema'):
                memory.retrieve(req)


if __name__ == '__main__':
    unittest.main()
