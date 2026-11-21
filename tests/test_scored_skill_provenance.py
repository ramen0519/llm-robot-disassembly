"""Stored-source replay only: no API calls, simulator, or historical writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from orchestrator.knowledge_import import import_sources
from orchestrator.scored_skill_provenance import (
    PACKET_PATH, REGISTRATIONS, Reader, load_verified_scored_provenance)
from tools.migrate_relational_knowledge import Migrator


ROOT = Path(__file__).resolve().parents[1]
PATHS = [PACKET_PATH] + [s['run'] + '/registry/skill.json' for s in REGISTRATIONS]


class ScoredSkillProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proof = load_verified_scored_provenance(ROOT)
        cls.imported = import_sources(ROOT, [dict(path=p, status='DISCOVERED') for p in PATHS])
        cls.before = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                      for p in cls.proof['source_hashes']}

    @classmethod
    def tearDownClass(cls):
        after = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in cls.before}
        if after != cls.before:
            raise AssertionError('historical source changed during stored replay')

    def test_two_real_registrations_and_all_packet_exposures_verified(self):
        proof = self.proof
        self.assertEqual(proof['issues'], [])
        self.assertEqual(len(proof['skills']), 2)
        self.assertEqual(len(proof['observations']), 39)
        for row in proof['skills'].values():
            self.assertEqual(row['status'], 'VERIFIED_HISTORICAL_REGISTRATION')
            self.assertEqual(len(row['exposure_evidence_ids']), 39)
            self.assertEqual(len(row['training_products']), 5)
            self.assertEqual(len(row['validation_proofs']), 2)
            self.assertEqual(row['validation_profile'], 'path_only_sequence_proposal/v1')
            self.assertEqual(row['native_sequence_profile_validation'], 'NOT_RUN')
            self.assertEqual(row['performance_effect'], 'UNVERIFIED')
            self.assertEqual(row['new_api_calls'], 0)
            self.assertEqual(row['new_physics_calls'], 0)

    def test_migration_preserves_program_and_status_but_restores_exact_links(self):
        original = {row['source_id']: row['record'] for row in self.imported['records']}
        migration = Migrator(ROOT)
        cards = migration.convert(deepcopy(self.imported))
        self.assertEqual(migration.issues, [])
        self.assertEqual(len(cards), 41)
        self.assertTrue(all(c['dataset_role'] == 'training' for c in cards))
        indexed = {c['card_id']: c for c in cards}
        for card in cards:
            self.assertEqual(card['content'], original[card['origin']['source_id']])
            self.assertFalse(card['execution_authorized'])
            self.assertFalse(card['new_model_generation'])
            if card['kind'] == 'skill':
                proof = card['provenance_validation']
                self.assertEqual(card['status'], 'REGISTERED_SCOPED')
                self.assertEqual(card['unresolved_evidence_ids'], [])
                self.assertEqual({indexed[x]['origin']['source_id'] for x in card['evidence_ids']},
                                 set(proof['cited_evidence_ids']))
                self.assertEqual(card['context']['source_products'], proof['training_products'])
                self.assertEqual(proof['separate_lesson'], 'NOT_PRESENT_IN_THIS_GENERATION_ARTIFACT')
        self.assertFalse(any(c['kind'] == 'lesson' for c in cards))

    def test_anonymous_observation_restores_geometry_and_source_ids_only_in_context(self):
        cards = Migrator(ROOT).convert(deepcopy(self.imported))
        for card in (c for c in cards if c['kind'] == 'experience'):
            source = self.proof['observations'][card['origin']['source_id']]
            ctx = card['context']
            self.assertEqual(ctx['product_id'], source['source_product'])
            self.assertEqual(ctx['source_geometry_status'], 'linked')
            self.assertTrue(ctx['shape']['extents_xyz'])
            self.assertFalse(ctx['dynamic_contacts_observed'])
            self.assertIsNone(ctx['observations']['detailed_failure_cause'])
            self.assertFalse(ctx['observations']['minimum_support_proven'])
            self.assertEqual(card['content']['product'], source['product_alias'])
            self.assertEqual(ctx['source_observation_records'], source['source_records'])

    def test_same_local_id_in_another_file_does_not_leak_into_training_or_links(self):
        imported = deepcopy(self.imported)
        record = next(r for r in imported['records'] if r['kind'] == 'experience')
        impostor = deepcopy(record)
        impostor.update(record_id='unrelated_local_id', source_path='runs/evaluation/observations.json',
                        namespace='file:runs/evaluation/observations.json', origins=[])
        imported['records'].append(impostor)
        cards = Migrator(ROOT).convert(imported)
        duplicate = next(c for c in cards if c['card_id'] == 'relational:unrelated_local_id')
        self.assertEqual(duplicate['dataset_role'], 'unclassified')
        self.assertNotIn('provenance_validation', duplicate)
        for skill in (c for c in cards if c['kind'] == 'skill'):
            self.assertNotIn(duplicate['card_id'], skill['evidence_ids'])

    def test_registration_without_imported_evidence_remains_unclassified(self):
        imported = deepcopy(self.imported)
        imported['records'] = [r for r in imported['records'] if r['kind'] == 'skill']
        cards = Migrator(ROOT).convert(imported)
        self.assertTrue(all(c['dataset_role'] == 'unclassified' for c in cards))
        self.assertTrue(all(c['unresolved_evidence_ids'] for c in cards))

    def test_matching_product_names_are_not_a_training_authorization(self):
        imported = deepcopy(self.imported)
        item = next(r for r in imported['records'] if r['kind'] == 'skill')
        item.update(source_path='runs/unverified/registry/skill.json', record_id='unverified_registry', origins=[])
        cards = Migrator(ROOT).convert(imported)
        card = next(c for c in cards if c['card_id'] == 'relational:unverified_registry')
        self.assertEqual(card['dataset_role'], 'unclassified')
        self.assertNotIn('provenance_validation', card)

    def test_imported_registration_content_must_match_pinned_original(self):
        imported = deepcopy(self.imported)
        item = next(r for r in imported['records'] if r['kind'] == 'skill')
        item['record']['candidate']['source'] = 'def propose(ctx):\n    return 1.0'
        cards = Migrator(ROOT).convert(imported)
        card = next(c for c in cards if c['card_id'] == 'relational:' + item['record_id'])
        self.assertEqual(card['dataset_role'], 'unclassified')
        self.assertNotIn('provenance_validation', card)

    def test_raw_observation_hash_failure_blocks_entire_packet(self):
        target = next(iter(self.proof['observations'].values()))['source_records'][0]['file']
        original = Reader.read
        def fail(reader, name, expected=None):
            if name == target:
                raise ValueError('source_hash_mismatch:' + name)
            return original(reader, name, expected)
        with patch.object(Reader, 'read', fail):
            result = load_verified_scored_provenance(ROOT)
        self.assertFalse(result['observations'])
        self.assertFalse(result['skills'])
        self.assertTrue(any('source_hash_mismatch' in i['reason'] for i in result['issues']))

    def test_changed_candidate_or_proof_fails_closed_for_affected_registration(self):
        for target in [REGISTRATIONS[0]['candidate_path'], REGISTRATIONS[0]['run'] + '/validation/00561/verified.json']:
            with self.subTest(target=target):
                def callback(name):
                    data = json.loads((ROOT / name).read_text(encoding='utf-8-sig'))
                    if name == target:
                        data['tampered'] = True
                    return data
                result = load_verified_scored_provenance(ROOT, callback)
                self.assertEqual(len(result['skills']), 1)
                self.assertEqual(next(iter(result['skills'].values()))['registration_path'],
                                 REGISTRATIONS[1]['run'] + '/registry/skill.json')
                self.assertTrue(any('source_changed_while_reading' in i['reason'] for i in result['issues']))

    def test_forbidden_or_escaping_path_rejected_before_probe(self):
        reader = Reader(ROOT)
        with patch.object(Path, 'lstat', side_effect=AssertionError('must not probe')):
            for name in ['E:/forbidden/source.json', '../outside.json']:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'repository_relative_path_required'):
                    reader.read(name)


if __name__ == '__main__':
    unittest.main()
