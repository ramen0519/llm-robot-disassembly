"""Inspect saved actual evidence and API preview; no new physics or paid calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_skill_discovery as runner
from orchestrator.skill_discovery import SkillDiscoveryService


class StabilitySynthesisTests(unittest.TestCase):
    def setUp(self):
        episode = ROOT / 'runs/automatic_learning_cycle_20261019/live_01/next_episode'
        self.memory = episode / 'action_memory_frozen.json'
        self.report = episode / 'planner/planner_report.json'
        self.manifest = dict(experience_report=str(self.report), experience_task='07009',
            experience_memory=str(self.memory), synthesis_failure_scope='dynamic_stability_failed',
            frozen_episode_memory=dict(path=str(self.memory), file_sha256=runner.sha(self.memory),
                content_sha256=runner.read(self.memory)['content_sha256']))
        for target in ('urllib.request.OpenerDirector.open', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('no network or physics in unit test'))
            guard.start()
            self.addCleanup(guard.stop)

    def packet(self):
        return runner.synthesis_packet(self.manifest, 'unit:stability-evidence')

    def test_measured_facts_exposed_in_both_views_without_invented_cause(self):
        packet = self.packet()
        self.assertEqual(len(packet['evidence']), 24)
        scoped = packet['context']['target_failure_scope']
        self.assertEqual(len(scoped['evidence_ids']), 6)  # 3 observations, two ID views
        self.assertTrue(scoped['initial_disconnected_not_in_scope'])
        for item in packet['evidence']:
            self.assertIsNone(item['failure_cause'])
            self.assertIsNone(item['current_part_id_mapping'])
            diagnostic = item['stability_diagnostics']
            if item['evidence_id'] in scoped['evidence_ids']:
                self.assertEqual(diagnostic['phase'], 'dynamic_failure')
                self.assertEqual(diagnostic['extra_physics_calls'], 0)
                for part in diagnostic['dynamic_failure_parts']:
                    self.assertGreater(part['displacement'], part['displacement_threshold'])
                    self.assertEqual(part['part_id'], '4')
            elif diagnostic and diagnostic['phase'] == 'initial_disconnected':
                self.assertNotIn(item['evidence_id'], scoped['evidence_ids'])
        frozen = packet['context']['frozen_episode_memory']
        self.assertTrue(frozen['binding_checked'])
        self.assertEqual(frozen['additional_physical_observation_count'], 0)

    def test_new_scope_requires_frozen_bound_memory(self):
        self.manifest.pop('experience_memory')
        with self.assertRaisesRegex(ValueError, 'bound_frozen_memory'):
            self.packet()

    def test_diagnostic_alone_cannot_replace_executed_stability_checker(self):
        raw = runner.read(self.report)['attempts'][1]['raw']
        self.assertTrue(runner._checked_stability_failure_stage(raw))
        for field, value in (('path_check_calls', 0), ('stability_check_calls', 0),
                             ('stability_mode', 'unverified'), ('status', 'ERROR')):
            with self.subTest(field=field):
                broken = deepcopy(raw)
                broken[field] = value
                self.assertFalse(runner._checked_stability_failure_stage(broken))
        raw['raw_upstream']['stability_success'] = None
        self.assertFalse(runner._checked_stability_failure_stage(raw))

    def test_unknown_scope_rejected(self):
        self.manifest['synthesis_failure_scope'] = 'invented_collision'
        with self.assertRaisesRegex(ValueError, 'unsupported_synthesis_failure_scope'):
            self.packet()

    def test_original_raw_artifact_still_binds_projected_diagnostics(self):
        original_read = runner.read
        def altered(path):
            value = original_read(path)
            if Path(path) == self.report:
                value = deepcopy(value)
                value['attempts'][1]['raw']['stability_diagnostics']['dynamic_failure_parts'][0]['displacement'] += 1
            return value
        with patch.object(runner, 'read', side_effect=altered), self.assertRaisesRegex(ValueError, 'binding_mismatch'):
            self.packet()

    def test_malformed_diagnostics_never_become_llm_facts(self):
        original_read = runner.read
        def altered(path):
            value = original_read(path)
            if Path(path) == self.report:
                value['attempts'][1]['raw']['stability_diagnostics']['extra_physics_calls'] = 1
            return value
        with patch.object(runner, 'read', side_effect=altered), self.assertRaisesRegex(ValueError, 'must_not_query'):
            self.packet()

    def test_provider_preview_accepts_scope_and_measured_memory_without_api(self):
        packet = self.packet()
        with tempfile.TemporaryDirectory() as temporary:
            service = SkillDiscoveryService(runner.preview_config(), Path(temporary) / 'preview.sqlite')
            try:
                preview = service.preview(packet)
            finally:
                service.close()
        self.assertEqual(preview['api_calls'], 0)
        sent = json.loads(preview['body']['input'][0]['content'])
        self.assertEqual(sent['context']['target_failure_scope'], packet['context']['target_failure_scope'])
        self.assertIn('contains_observed_unstable', sent['context']['language_semantics']['support'])
        self.assertIn('MOVER', sent['context']['language_semantics']['part'])
        self.assertIn('HOLDS', sent['context']['language_semantics']['part'])

    def test_legacy_scope_is_not_relabelled(self):
        self.manifest.pop('synthesis_failure_scope')
        self.assertNotIn('target_failure_scope', self.packet()['context'])


if __name__ == '__main__':
    unittest.main()
