"""Offline bridge contracts plus replay of the two preserved registrations.

Replaying saved CAD descriptors is not a RedMax experiment or a live model test.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from orchestrator import scored_sequence_bridge as bridge
from orchestrator.scored_order_skill import feature_rows, run_skill


ROOT = Path(__file__).absolute().parents[1]
REGISTRIES = [ROOT / 'runs/scored_order_skill_pilot_20261111/registry/skill.json',
              ROOT / 'runs/scored_order_astra_author_20261112/registry/skill.json']


def geometry():
    part = dict(bounds=[[0, 0, 0], [1, 2, 3]], bounds_center=[0.5, 1, 1.5],
                extents=[1, 2, 3], volume=6,
                validity={'center_mass_status': 'uniform_density_estimate'})
    return dict(task={'assembly_id': 'unseen'}, parts={p: deepcopy(part) for p in 'abc'},
                relations={'contact_edges': [['a', 'b'], ['b', 'c']]})


def request(cad, base=None, **extra):
    parts = sorted(cad['parts'])
    return dict(assembled_parts=parts, candidate_ids=[p for p in parts if p != base],
                base_part=base, **extra)


class ScoredBridgePathTests(unittest.TestCase):
    def test_e_drive_rejected_before_filesystem_access(self):
        with patch.object(Path, 'lstat', side_effect=AssertionError('must not inspect path')):
            with self.assertRaisesRegex(ValueError, 'forbidden_drive'):
                bridge.build_bridge('E:/not-permitted/registry.json')

    def test_lexical_traversal_and_unc_are_rejected(self):
        for path in ('runs/../outside.json', '//host/share/registry.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                bridge.build_bridge(path)

    def test_unregistered_fixture_or_foreign_contract_are_rejected(self):
        for card in ({'protocol': bridge.REGISTRATION_PROTOCOL, 'status': 'CANDIDATE'},
                     {'protocol': bridge.REGISTRATION_PROTOCOL, 'status': 'REGISTERED_SCOPED', 'fixture': True},
                     {'protocol': 'sequence_skill_v1', 'status': 'REGISTERED_SCOPED'}):
            with patch.object(bridge, 'read_json', return_value=card):
                with self.assertRaisesRegex(ValueError, 'registered_nonfixture'):
                    bridge.build_bridge(ROOT / 'runs/not-read.json')


class ScoredSequenceBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(p.is_file() for p in REGISTRIES):
            raise unittest.SkipTest('preserved real registrations unavailable; no fixture physics substitution')
        cls.bridges = [bridge.build_bridge(p) for p in REGISTRIES]

    def test_real_generation_and_physics_registration_are_bound(self):
        for frozen, path in zip(self.bridges, REGISTRIES):
            original = json.loads(path.read_bytes())
            self.assertEqual(frozen['registration'], original)
            self.assertEqual(frozen['program']['source'].encode('utf8'),
                             original['candidate']['source'].encode('utf8'))
            self.assertEqual(frozen['source_profile'], 'path_only_sequence_proposal/v1')
            self.assertEqual(frozen['native_physics_validation'], 'NOT_RUN')
            self.assertEqual(frozen['authority'], 'candidate_order_only')
            self.assertFalse(frozen['physics_reuse_allowed'])
            self.assertEqual(len(frozen['source_file_hashes']), 24)
            self.assertTrue(set(bridge.RUNTIME_SOURCES) <= frozen['source_file_hashes'].keys())
            self.assertEqual(bridge.validate_bridge(frozen), frozen)

    def test_compact_card_binds_host_artifact_without_full_proof(self):
        for frozen in self.bridges:
            card = bridge.compact_bridge(frozen)
            self.assertEqual(card['skill_id'], frozen['skill_id'])
            self.assertEqual(card['content_sha256'], frozen['content_sha256'])
            self.assertEqual(card['original_registry_sha256'], frozen['original_registry_sha256'])
            self.assertEqual(card['program'], frozen['program'])
            self.assertNotIn('registration', card)
            self.assertEqual(card['validation']['profile'], bridge.SOURCE_PROFILE)
            self.assertEqual(card['native_physics_validation'], 'NOT_RUN')
            with self.assertRaises(ValueError):
                bridge.validate_bridge(card)

    def test_program_or_physics_authority_tampering_rejected_even_if_rehashed(self):
        changes = [('native_physics_validation', 'PASS'), ('fixture', True),
                   ('source_profile', 'sequence_native_v1'), ('physics_reuse_allowed', True),
                   ('source_task_ids', []), ('original_registry_sha256', '0' * 64)]
        for key, value in changes:
            bad = deepcopy(self.bridges[0]); bad[key] = value
            bad['content_sha256'] = bridge.digest({k: v for k, v in bad.items() if k != 'content_sha256'})
            with self.subTest(key=key), self.assertRaises(ValueError):
                bridge.validate_bridge(bad)
        bad = deepcopy(self.bridges[0]); bad['program']['source'] += '\n# edited'
        bad['content_sha256'] = bridge.digest({k: v for k, v in bad.items() if k != 'content_sha256'})
        with self.assertRaises(ValueError):
            bridge.validate_bridge(bad)

    def test_modified_source_file_fails_closed(self):
        with patch.object(bridge, '_read_bytes', return_value=b'changed bytes'):
            out = bridge.execute_bridge(self.bridges[0], geometry(), request(geometry()))
        self.assertEqual(out['status'], 'REJECTED')
        self.assertIn('bridge_source_hash_changed', str(out['diagnostics']))
        self.assertFalse(out['vm_executed'])

    def test_evidence_pins_cannot_be_removed_with_a_new_content_hash(self):
        bad = deepcopy(self.bridges[0])
        removed = next(p for p in bad['source_file_hashes'] if p.endswith('events.jsonl'))
        del bad['source_file_hashes'][removed]
        bad['content_sha256'] = bridge.digest({k: v for k, v in bad.items() if k != 'content_sha256'})
        with self.assertRaisesRegex(ValueError, 'pin_set_mismatch'):
            bridge.validate_bridge(bad)

    def test_fixture_candidate_cannot_be_relabelled_as_registered(self):
        bad = deepcopy(self.bridges[0]['registration'])
        bad['candidate']['provenance']['origin'] = 'fixture'
        with patch.object(bridge, 'read_json', return_value=bad):
            with self.assertRaisesRegex(ValueError, 'actual_model'):
                bridge.build_bridge(REGISTRIES[0])

    def test_full_state_features_include_base_but_output_excludes_it(self):
        cad = geometry()
        rows = feature_rows(dict(parts=cad['parts'], contact_edges=cad['relations']['contact_edges']), list(cad['parts']))
        for frozen in self.bridges:
            legacy = run_skill(frozen['program']['source'], rows)
            out = bridge.execute_bridge(frozen, cad, request(cad, base='b'))
            self.assertEqual(out['status'], 'PROPOSED', out)
            self.assertEqual(out['candidate_order'], [p for p in legacy['candidate_order'] if p != 'b'])
            self.assertEqual(out['vm_steps'], legacy['vm_steps'])
            self.assertEqual(out['full_feature_rows_sha256'], bridge.digest(rows))
            self.assertTrue(out['base_score_evaluated'])
            self.assertNotIn('b', out['priorities'])
            self.assertFalse(out['physics_executed'])
            self.assertFalse(out['current_attempt_verified'])

    def test_partial_duplicate_or_base_candidates_rejected(self):
        cad = geometry()
        for candidates in (['a'], ['a', 'a'], ['a', 'b', 'c']):
            req = request(cad, base='b'); req['candidate_ids'] = candidates
            out = bridge.execute_bridge(self.bridges[0], cad, req)
            self.assertEqual(out['status'], 'REJECTED')
            self.assertFalse(out['vm_executed'])
            self.assertIsNone(out['candidate_order'])
            self.assertTrue(out['fallback_required'])
            self.assertFalse(out['fallback_applied'])

    def test_source_product_exclusion_and_explicit_training_replay(self):
        cad = geometry(); cad['task']['assembly_id'] = '00561'
        out = bridge.execute_bridge(self.bridges[0], cad, request(cad))
        self.assertEqual(out['status'], 'DECLINED')
        self.assertEqual(out['diagnostics'], ['current_task_in_source_products'])
        self.assertFalse(out['vm_executed'])
        replay = bridge.execute_bridge(self.bridges[0], cad, request(cad, purpose='training_replay'))
        self.assertEqual(replay['status'], 'PROPOSED')
        self.assertFalse(replay['transfer_evaluation'])

    def test_task_identity_mismatch_and_native_terminal_deferred(self):
        cad = geometry()
        out = bridge.execute_bridge(self.bridges[0], cad, request(cad, assembly_id='different'))
        self.assertEqual(out['status'], 'REJECTED')
        cad['parts'].pop('c'); cad['relations']['contact_edges'] = [['a', 'b']]
        out = bridge.execute_bridge(self.bridges[0], cad, request(cad))
        self.assertEqual(out['status'], 'DECLINED')
        self.assertEqual(out['diagnostics'], ['native_terminal_rule_owns_last_two'])

    def test_missing_or_invalid_geometry_is_not_replaced_by_guesses(self):
        for kind in ('edges', 'part', 'nan', 'self_edge', 'task_type', 'relations_type'):
            cad = geometry(); req = request(cad)
            if kind == 'edges': cad['relations'] = {}
            if kind == 'part': cad['parts'].pop('a')
            if kind == 'nan': cad['parts']['a']['bounds'][0][0] = float('nan')
            if kind == 'self_edge': cad['relations']['contact_edges'] = [['a', 'a']]
            if kind == 'task_type': cad['task'] = None
            if kind == 'relations_type': cad['relations'] = None
            with self.subTest(kind=kind):
                out = bridge.execute_bridge(self.bridges[0], cad, req)
                self.assertEqual(out['status'], 'REJECTED')
                self.assertFalse(out['vm_executed'])

    def test_abstain_and_rejected_vm_do_not_choose_fallback(self):
        for status, expected in [('ABSTAIN', 'DECLINED'), ('REJECTED', 'REJECTED')]:
            raw = dict(status=status, candidate_order=None, priorities={}, diagnostics=['fixture_vm_result'], vm_steps=7)
            with patch.object(bridge, 'run_skill', return_value=raw):
                out = bridge.execute_bridge(self.bridges[0], geometry(), request(geometry()))
            self.assertEqual(out['status'], expected)
            self.assertIsNone(out['candidate_order'])
            self.assertTrue(out['fallback_required'])
            self.assertFalse(out['fallback_applied'])

    def test_invalid_vm_permutation_is_rejected(self):
        raw = dict(status='PROPOSED', candidate_order=['a', 'b'], priorities={'a': 1, 'b': 2}, diagnostics=[], vm_steps=3)
        with patch.object(bridge, 'run_skill', return_value=raw):
            out = bridge.execute_bridge(self.bridges[0], geometry(), request(geometry()))
        self.assertEqual(out['status'], 'REJECTED')
        self.assertIn('invalid_full_permutation', str(out['diagnostics']))

    def test_tied_candidates_preserve_legacy_id_tiebreak_and_diagnostic(self):
        cad = geometry(); cad['relations']['contact_edges'] = [['a', 'b'], ['b', 'c'], ['c', 'a']]
        for frozen in self.bridges:
            out = bridge.execute_bridge(frozen, cad, request(cad, base='b'))
            self.assertEqual(out['candidate_order'], ['a', 'c'])
            self.assertIn('all_candidate_priorities_tied', out['diagnostics'])

    def test_actual_cad_saved_descriptors_reproduce_original_vm_order(self):
        cases = [
            ('20014', 'runs/native_pair_20014_00709_20261119/episodes/20014/features.json', None),
            ('Challenge/H07', 'runs/remaining32_prep_timeout_retry_20261116/live/p26/episodes/gnn/p26-gnn/features.json', '11'),
            ('00561', 'runs/scored_order_skill_pilot_20261111/validation/00561/actual_snapshot.json', None),
            ('02628', 'runs/scored_order_skill_pilot_20261111/validation/02628/actual_snapshot.json', None),
        ]
        for identity, relative, base in cases:
            path = ROOT / relative
            if not path.is_file():
                self.skipTest('saved real CAD descriptor unavailable: ' + relative)
            cad = json.loads(path.read_bytes())
            cad.setdefault('task', {'assembly_id': identity})
            contacts = cad.get('contact_edges', cad.get('relations', {}).get('contact_edges'))
            rows = feature_rows(dict(parts=cad['parts'], contact_edges=contacts), sorted(cad['parts']))
            for frozen in self.bridges:
                with self.subTest(product=identity, skill=frozen['skill_id']):
                    legacy = run_skill(frozen['program']['source'], rows)
                    purpose = 'training_replay' if identity in frozen['source_task_ids'] else 'evaluation'
                    out = bridge.execute_bridge(frozen, cad, request(cad, base, purpose=purpose))
                    self.assertEqual(out['status'], 'PROPOSED', out)
                    self.assertEqual(out['candidate_order'], [p for p in legacy['candidate_order'] if p != base])
                    self.assertEqual(out['vm_steps'], legacy['vm_steps'])
                    self.assertEqual(out['full_feature_rows_sha256'], bridge.digest(rows))


if __name__ == '__main__':
    unittest.main()
