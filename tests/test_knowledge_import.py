import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.knowledge_import import discover_sources, import_sources, safe_path


class HistoricalKnowledgeImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'runs').mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def put(self, name, value):
        path = self.root / 'runs' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def test_original_records_and_bytes_are_preserved(self):
        row = dict(evidence_id='e1', result=dict(path_success=False, blocker=None))
        path = self.put('a/memory.json', dict(records=[row]))
        before = path.read_bytes()
        data = import_sources(self.root)
        self.assertEqual(data['records'][0]['record'], row)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(data['records'][0]['sha256'], hashlib.sha256(before).hexdigest())

    def test_local_ids_do_not_join_across_unknown_namespaces(self):
        row = dict(id='e01', outcome='COMPLETE')
        self.put('a/memory.json', dict(observations=[row]))
        self.put('b/memory.json', dict(observations=[row]))
        self.assertEqual(len(import_sources(self.root)['records']), 2)

    def test_same_pinned_packet_preserves_multiple_origins(self):
        document = dict(packet_sha256='a'*64, observations=[dict(id='e01', outcome='COMPLETE')])
        self.put('a/common_start.json', document)
        self.put('b/common_start.json', document)
        records = import_sources(self.root)['records']
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]['origins']), 2)

    def test_registration_envelopes_are_not_collapsed(self):
        row = dict(skill_id='order-skill:'+'b'*64, source='def propose(ctx):\n    return 0')
        self.put('a/skills.json', dict(skills=[row], registered=False))
        self.put('b/skills.json', dict(skills=[row], registered=True))
        self.assertEqual(len(import_sources(self.root)['records']), 2)

    def test_fixture_is_preserved_without_promoting(self):
        self.put('fixture_case/memory.json', dict(records=[dict(evidence_id='x', result={})]))
        record = import_sources(self.root)['records'][0]
        self.assertIs(record['origins'][0]['fixture'], True)

    def test_raw_registry_admission_and_candidate_kept_together(self):
        entry = dict(candidate=dict(candidate_id='a'*64, program=dict(source='original')),
                     admission=dict(admitted=False, reasons=['physics_failed']), validations=[])
        self.put('x/skills_frozen.json', dict(entries=[entry], fixture=False))
        record = import_sources(self.root)['records'][0]
        self.assertEqual(record['kind'], 'skill')
        self.assertEqual(record['record'], entry)

    def test_jsonl_observations_imported(self):
        path = self.root / 'runs' / 'action_observations.jsonl'
        path.write_text('\n'.join(json.dumps(dict(evidence_id=str(i), result={})) for i in range(2)), encoding='utf-8')
        data = import_sources(self.root)
        self.assertEqual(len(data['records']), 2)
        self.assertEqual(data['records'][0]['origins'][0]['format'], 'jsonl_parsed_as_array')

    def test_prohibited_or_absolute_path_rejected_before_io(self):
        with patch('orchestrator.knowledge_import._regular') as inspect:
            for name in ('E:/not-accessed/memory.json', '../escape.json', 'C:/elsewhere/data.json'):
                with self.assertRaises(ValueError):
                    safe_path(self.root, name)
            inspect.assert_not_called()

    def test_credential_and_request_files_not_exported(self):
        self.put('a/memory_request.json', dict(records=[dict(result={})]))
        self.put('a/memory.json', dict(records=[dict(result={}, note='sk-proj-'+'A'*40)]))
        data = import_sources(self.root)
        self.assertEqual(data['records'], [])
        self.assertEqual({row['status'] for row in data['inventory']}, {'EXCLUDED', 'NOT_IMPORTED'})

    def test_duplicate_json_keys_rejected(self):
        path = self.root / 'runs' / 'memory.json'
        path.write_text('{"records":[],"records":[{"result":{}}]}', encoding='utf-8')
        data = import_sources(self.root)
        self.assertEqual(data['records'], [])
        self.assertEqual(data['inventory'][0]['status'], 'ERROR')

    def test_unsupported_and_empty_are_distinct(self):
        self.put('empty/skills.json', dict(entries=[]))
        self.put('old/knowledge_report.json', dict(note='not actual knowledge'))
        data = import_sources(self.root)
        self.assertEqual({row['status'] for row in data['inventory']}, {'EMPTY_SOURCE', 'NOT_IMPORTED'})

    def test_migration_output_excluded_from_input(self):
        self.put('relational_knowledge_migration_20261119/memory.json', dict(records=[dict(result={})]))
        self.assertEqual(discover_sources(self.root), [])

    def test_nested_native_sequence_memory_and_direct_entries(self):
        row = dict(evidence_id='native:'+ 'd'*64, observed=dict(status='PASS'))
        self.put('native/memory.json', [dict(protocol='sequence_native_observation_v1', records=[row])])
        self.put('native/lessons.json', dict(entries=[dict(claim={'title':'x'}, facts=[row], fixture=True)]))
        self.put('native/skills.json', dict(entries=[dict(skill_id='sequence:x', program={'source':'code'}, fixture=True)]))
        records = import_sources(self.root)['records']
        self.assertEqual({r['kind'] for r in records}, {'experience','lesson','skill'})
        experience = next(r for r in records if r['kind']=='experience')
        self.assertEqual(experience['json_pointer'], '/0/records/0')
        self.assertEqual(experience['record'], row)


if __name__ == '__main__':
    unittest.main()
