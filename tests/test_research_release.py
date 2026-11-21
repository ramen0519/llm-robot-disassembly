"""Gate invalidation and fail-closed tests; no external execution."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator import research_release as gate


class ResearchReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=gate.ROOT/'runs', prefix='release_gate_fixture_')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder/'certificate.json'
        self.log = self.folder/'tests.log'
        self.log.write_text('INJECTED_REPORT_FOR_VALIDATION_TEST_ONLY', encoding='utf-8')
        self.sources = {'worker/fixture.py': 'a'*64}
        self.replays = {'runs/fixture.json': 'b'*64}
        self.value = dict(schema=gate.SCHEMA, profile_id=gate.PROFILE_ID, status='PASS',
            test_modules=list(gate.TEST_MODULES), test_ids=[m+'.Fixture.test' for m in gate.TEST_MODULES],
            tests=dict(run=len(gate.TEST_MODULES), failures=0, errors=0, skipped=0,
                expected_failures=0, unexpected_successes=0),
            source_sha256=self.sources, replay_sha256=self.replays,
            source_and_replays_unchanged=True,
            test_log=self.log.relative_to(gate.ROOT).as_posix(), test_log_sha256=gate.sha(self.log),
            API_calls=0, physics_calls=0, authorizes_paid_execution=False, physical_performance='UNVERIFIED')

    def validate(self, value=None, sources=None, replays=None, profile_id=gate.PROFILE_ID):
        self.path.write_text(json.dumps(self.value if value is None else value), encoding='utf-8')
        with patch.object(gate, 'source_inventory', return_value=self.sources if sources is None else sources), \
             patch.object(gate, 'replay_inventory', return_value=self.replays if replays is None else replays), \
             patch.object(gate, 'gate_tests', return_value=dict.fromkeys(self.value['test_ids'])):
            return gate.validate_release_certificate(self.path, profile_id=profile_id)

    def test_capacity_requires_explicit_full_profile_coverage(self):
        with self.assertRaisesRegex(ValueError, 'release_does_not_cover_capacity_profile'):
            self.validate(profile_id='part_capacity_v1')
        value = dict(self.value, covered_profiles=list(gate.COVERED_PROFILES))
        self.validate(value, profile_id='part_capacity_v1')
        value['covered_profiles'] = [gate.PROFILE_ID]
        with self.assertRaisesRegex(ValueError, 'release_does_not_cover_capacity_profile'):
            self.validate(value, profile_id='part_capacity_v1')

    def test_valid_offline_report_does_not_authorize_api_or_claim_physics(self):
        result = self.validate()
        self.assertFalse(result['authorizes_paid_execution'])
        self.assertEqual(result['physical_performance'], 'UNVERIFIED')

    def test_any_changed_added_or_removed_source_invalidates_gate(self):
        for sources in ({}, {**self.sources, 'tools/new.py':'c'*64}, {'worker/fixture.py':'d'*64}):
            with self.subTest(sources=sources), self.assertRaisesRegex(ValueError, 'source_changed'):
                self.validate(sources=sources)

    def test_replay_change_cannot_be_silently_reused(self):
        with self.assertRaisesRegex(ValueError, 'replay_inputs_changed'):
            self.validate(replays={})

    def test_same_count_but_wrong_test_set_rejected(self):
        value = deepcopy(self.value); value['test_ids'][-1] = 'unrelated.Tests.test_pass'
        with self.assertRaisesRegex(ValueError, 'required_test_ids'):
            self.validate(value)

    def test_missing_certificate_fails_before_inventory(self):
        with patch.object(gate, 'source_inventory', side_effect=AssertionError('NO_ACCESS')):
            with self.assertRaisesRegex(ValueError, 'certificate_required'):
                gate.validate_release_certificate(None)

    def test_failed_skipped_incomplete_or_repeated_test_ids_rejected(self):
        for key in ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes'):
            value = deepcopy(self.value); value['tests'][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'tests_incomplete'):
                self.validate(value)
        value = deepcopy(self.value); value['test_ids'][-1] = value['test_ids'][0]
        with self.assertRaisesRegex(ValueError, 'tests_incomplete'):
            self.validate(value)

    def test_wrong_profile_subset_suite_and_false_physical_claim_rejected(self):
        for key, replacement in [('profile_id','legacy_v1'), ('test_modules',[]),
                ('authorizes_paid_execution',True), ('physical_performance','PASS'), ('status','FAIL')]:
            value = {**self.value, key:replacement}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'invalid_offline'):
                self.validate(value)

    def test_log_tamper_rejected(self):
        self.log.write_text('CHANGED', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'test_log_changed'):
            self.validate()

    def test_forbidden_paths_rejected_before_filesystem(self):
        with patch.object(Path, 'lstat', side_effect=AssertionError('NO_FILESYSTEM')):
            for path in ('E:/never-access', '/mnt/e/never-access', '//server/share', '../outside'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    gate.workspace_path(path)

    def test_junction_rejected_before_read(self):
        from types import SimpleNamespace
        with patch.object(Path, 'lstat', return_value=SimpleNamespace(st_mode=0, st_file_attributes=0x400)):
            with self.assertRaisesRegex(ValueError, 'reparse_path'):
                gate.workspace_path('runs/junction/data.json')


if __name__ == '__main__':
    unittest.main()
