"""Frozen-memory synthesis binding using copied historical logs, no new physics.

All source mutations target owned temporary copies. Provider preview never
loads credentials or sends a request. These tests do not claim live learning.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import prepare_episode_learning as learning
import run_skill_discovery as discovery
from orchestrator import explicit_action_memory
from orchestrator.skill_discovery import SkillDiscoveryService
from worker.task_memory import _safe_path, digest

HISTORICAL = ROOT / 'runs/explicit_action_v2_live_20261018/beam'
TEMPLATE = ROOT / 'configs/skill_discovery_cases_v1.json'


class SkillMemoryPacketTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='memory_packet_saved_artifacts_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).absolute()
        self.episode = self.root / 'historical_artifact_copy'
        shutil.copytree(HISTORICAL, self.episode)
        self.memory_path = self.episode / 'action_memory_frozen.json'
        self.report_path = self.episode / 'planner/planner_report.json'
        snapshot = self.load(self.memory_path)
        for record in snapshot['records']:
            record['result']['evidence_ref'] = str(self.report_path)
        self.rehash_write(snapshot)
        prepared = learning.prepare_episode_learning(self.episode, TEMPLATE, self.root / 'prepared',
                                                     allow_test_root=True)
        self.manifest = self.load(Path(prepared['manifest_path']))
        original_checked = discovery.checked
        original_snapshot = explicit_action_memory.load_snapshot

        def checked(path, runs_only=False):
            value = Path(path).absolute()
            if self.root in value.parents:
                return _safe_path(path, True)
            return original_checked(path, runs_only)

        self.checked = checked
        for patcher in (
                patch.object(discovery, 'checked', side_effect=checked),
                patch.object(discovery, 'read', side_effect=lambda path: self.load(checked(path, True))),
                patch.object(discovery, 'sha', side_effect=lambda path: hashlib.sha256(checked(path).read_bytes()).hexdigest()),
                patch.object(explicit_action_memory, 'load_snapshot',
                             side_effect=lambda path: original_snapshot(path, allow_test_root=True)),
                patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('no real API permitted')),
                patch('subprocess.Popen', side_effect=AssertionError('no new physics permitted'))):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def load(path):
        return json.loads(path.read_text(encoding='utf-8'))

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def rehash_write(self, snapshot):
        for record in snapshot['records']:
            record['record_sha256'] = digest({k: v for k, v in record.items() if k != 'record_sha256'})
        snapshot['content_sha256'] = digest(snapshot['records'])
        self.write(self.memory_path, snapshot)

    def update_expected_hashes(self):
        snapshot = self.load(self.memory_path)
        self.manifest['frozen_episode_memory'].update(
            file_sha256=hashlib.sha256(self.memory_path.read_bytes()).hexdigest(),
            content_sha256=snapshot['content_sha256'])

    def packet(self):
        return discovery.synthesis_packet(self.manifest, 'offline:memory-binding')

    def test_memory_ids_are_citeable_and_bound_to_original_event_ids(self):
        packet = self.packet()
        snapshot = self.load(self.memory_path)
        report = self.load(self.report_path)
        frozen = [e for e in packet['evidence'] if e.get('evidence_origin') == 'frozen_episode_memory']
        trace = [e for e in packet['evidence'] if e.get('evidence_origin') == 'episode_trace']
        self.assertEqual([e['evidence_id'] for e in frozen], [r['evidence_id'] for r in snapshot['records']])
        self.assertEqual([e['evidence_id'] for e in trace], [a['event_id'] for a in report['attempts']])
        self.assertEqual(len(packet['evidence']), 16)
        self.assertEqual(len({e['evidence_id'] for e in packet['evidence']}), 16)
        for stored, event, record in zip(frozen, trace, snapshot['records']):
            self.assertEqual(stored['same_observation_as_event_id'], event['evidence_id'])
            self.assertEqual(event['frozen_memory_evidence_id'], stored['evidence_id'])
            self.assertEqual(stored['choice'], record['choice'])
            self.assertEqual(stored['status'], record['result']['status'])
            self.assertEqual(stored['path_success'], record['result']['path_success'])
            self.assertEqual(stored['stability_success'], record['result']['stability_success'])
            self.assertIsNone(stored['failure_cause'])
            self.assertIsNone(stored['current_part_id_mapping'])
            self.assertFalse(stored['all_paths_impossible'])
        provenance = packet['context']['frozen_episode_memory']
        self.assertEqual(provenance['record_count'], 8)
        self.assertEqual(provenance['exposed_record_count'], 8)
        self.assertEqual(provenance['file_sha256'], hashlib.sha256(self.memory_path.read_bytes()).hexdigest())
        self.assertEqual(provenance['content_sha256'], snapshot['content_sha256'])
        self.assertTrue(provenance['binding_checked'])
        self.assertTrue(provenance['duplicate_views_of_same_observations'])
        self.assertEqual(provenance['additional_physical_observation_count'], 0)

    def test_provider_preview_contains_memory_in_prompt_and_citation_enum_without_api(self):
        packet = self.packet()
        service = SkillDiscoveryService(discovery.preview_config(), self.root / 'preview.sqlite')
        try:
            preview = service.preview(packet)
        finally:
            service.close()
        self.assertEqual(preview['api_calls'], 0)
        body = preview['body']
        sent = json.loads(body['input'][0]['content'])
        self.assertEqual(sent['context']['frozen_episode_memory'], packet['context']['frozen_episode_memory'])
        schema = body['text']['format']['schema']
        cited = schema['properties']['card']['properties']['evidence_ids']['items']['enum']
        self.assertEqual(set(cited), {e['evidence_id'] for e in packet['evidence']})
        self.assertTrue(any(e.startswith('action-evidence:') for e in cited))

    def test_legacy_manifest_unchanged_without_memory_branch(self):
        self.manifest.pop('experience_memory')
        packet = self.packet()
        self.assertNotIn('frozen_episode_memory', packet['context'])
        self.assertEqual(len(packet['evidence']), 8)
        self.assertTrue(all('evidence_origin' not in e for e in packet['evidence']))

    def test_snapshot_file_hash_mismatch_rejected(self):
        self.manifest['frozen_episode_memory']['file_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'file_provenance'):
            self.packet()

    def test_snapshot_content_hash_mismatch_rejected(self):
        self.manifest['frozen_episode_memory']['content_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'content_provenance'):
            self.packet()

    def test_missing_snapshot_precommit_is_rejected(self):
        self.manifest.pop('frozen_episode_memory')
        with self.assertRaisesRegex(ValueError, 'file_provenance'):
            self.packet()

    def test_rehashed_wrong_raw_memory_still_rejected(self):
        snapshot = self.load(self.memory_path)
        snapshot['records'][0]['result']['raw']['reason'] = 'invented root cause'
        self.rehash_write(snapshot)
        self.update_expected_hashes()
        with self.assertRaisesRegex(ValueError, 'report_binding'):
            self.packet()

    def test_missing_memory_record_rejected(self):
        snapshot = self.load(self.memory_path)
        # Drop the last record so parent links of retained records remain valid.
        snapshot['records'].pop()
        snapshot['record_count'] -= 1
        self.rehash_write(snapshot)
        self.update_expected_hashes()
        with self.assertRaisesRegex(ValueError, 'episode_identity'):
            self.packet()

    def test_memory_pointer_to_another_report_rejected(self):
        snapshot = self.load(self.memory_path)
        snapshot['records'][0]['result']['evidence_ref'] = str(HISTORICAL / 'planner/planner_report.json')
        self.rehash_write(snapshot)
        self.update_expected_hashes()
        with self.assertRaisesRegex(ValueError, 'report_binding'):
            self.packet()

    def test_report_choice_tampering_rejected(self):
        report = self.load(self.report_path)
        report['attempts'][0]['choice']['direction_id'] = 'xn'
        self.write(self.report_path, report)
        with self.assertRaisesRegex(ValueError, 'report_binding'):
            self.packet()

    def test_raw_result_file_tampering_rejected(self):
        path = self.episode / 'planner/attempt_000001/result.json'
        raw = self.load(path)
        raw['reason'] = 'injected false cause'
        self.write(path, raw)
        with self.assertRaisesRegex(ValueError, 'actual_file_binding'):
            self.packet()

    def test_physical_request_action_tampering_rejected(self):
        path = self.episode / 'planner/attempt_000001/request.json'
        value = self.load(path)
        value['context']['action_id'] = 'stale:action'
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, 'actual_file_binding'):
            self.packet()

    def test_non_c_memory_rejected_before_snapshot_reader(self):
        self.manifest['experience_memory'] = 'E:/never-open/memory.json'
        with patch.object(explicit_action_memory, 'load_snapshot', side_effect=AssertionError('reader reached')):
            with self.assertRaises(ValueError):
                self.packet()


if __name__ == '__main__':
    unittest.main()
