"""Offline record safety tests; fixture I/O only, no API or simulator."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import batch_events as subject


class BatchEventTests(unittest.TestCase):
    def test_equal_reserved_fields_do_not_duplicate_keywords(self):
        fields = {'kind': 'gate', 'unix_time': 12.0, 'protocol_version': 'v3', 'passed': False}
        before = dict(fields)
        event = subject.make_event('gate', fields, protocol_version='v3', unix_time=12)
        self.assertEqual(event, fields)
        self.assertEqual(fields, before)

    def test_conflicting_metadata_rejected_before_open(self):
        for fields in ({'kind': 'other'}, {'unix_time': 13}, {'protocol_version': 'v2'}):
            with self.subTest(fields=fields), patch.object(Path, 'open') as opened:
                with self.assertRaisesRegex(ValueError, 'event_metadata_conflict'):
                    subject.append_event('unused.jsonl', 'gate', fields,
                                         protocol_version='v3', unix_time=12)
                opened.assert_not_called()

    def test_nonfinite_payload_rejected_before_creating_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'gate.json'
            with self.assertRaises(ValueError):
                subject.write_new_json(target, {'value': float('nan')})
            self.assertFalse(target.exists())

    def test_existing_outcome_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'gate.json'
            subject.write_new_json(target, {'passed': False})
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                subject.write_new_json(target, {'passed': True})
            self.assertEqual(target.read_bytes(), before)

    def test_gate_survives_event_append_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'gate.json'
            with patch.object(subject, 'append_event', side_effect=OSError('fixture_only')) as append:
                with self.assertRaises(subject.EventWriteFailure) as caught:
                    subject.write_record_and_event(target, Path(folder) / 'events.jsonl', 'gate',
                        {'protocol_version': 'v3', 'passed': False}, protocol_version='v3')
            self.assertFalse(json.loads(target.read_text())['passed'])
            self.assertTrue(caught.exception.record_saved)
            append.assert_called_once()

    def test_error_record_keeps_original_error_type(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'error.json'
            with patch.object(subject, 'append_event', side_effect=OSError('fixture_only')):
                with self.assertRaises(subject.EventWriteFailure):
                    subject.write_record_and_event(target, Path(folder) / 'events.jsonl', 'error',
                        {'error_type': 'OriginalWorkerError', 'no_automatic_retry': True}, protocol_version='v3')
            self.assertEqual(json.loads(target.read_text())['error_type'], 'OriginalWorkerError')

    def test_append_preserves_previous_events(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'events.jsonl'
            subject.append_event(target, 'start', {}, protocol_version='v3', unix_time=1)
            before = target.read_bytes()
            subject.append_event(target, 'end', {'protocol_version': 'v3'}, protocol_version='v3', unix_time=2)
            self.assertTrue(target.read_bytes().startswith(before))
            self.assertEqual([json.loads(line)['kind'] for line in target.read_text().splitlines()], ['start', 'end'])

    def test_forbidden_paths_rejected_before_filesystem_access(self):
        for value in ('E:/never', 'e:never', '\\\\?\\E:\\never', '/mnt/e/never'):
            with self.subTest(value=value), patch.object(Path, 'open') as opened:
                with self.assertRaisesRegex(ValueError, 'forbidden_drive'):
                    subject.write_new_json(value, {})
                opened.assert_not_called()

    def test_boolean_timestamp_is_not_equal_numeric_timestamp(self):
        with self.assertRaises(ValueError):
            subject.make_event('gate', {'unix_time': True}, protocol_version='v3', unix_time=1)


if __name__ == '__main__':
    unittest.main()
