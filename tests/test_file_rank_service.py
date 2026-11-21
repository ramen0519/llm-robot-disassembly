import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from orchestrator.file_service import FileRankServer, atomic_new, guarded, read_message


def request():
    return dict(schema_version=1, kind='rank_request', request=dict(
        run_id='test', request_id='test:r1', state_id='state', assembled_parts=['1', '0', '6'],
        candidate_ids=['1', '0'], base_part='6'), features={}, timeout_seconds=1)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.events = []

    def tearDown(self):
        self.temp.cleanup()

    def service(self, backend='noop'):
        return FileRankServer(self.root, 'test', backend,
                              lambda kind, **data: self.events.append((kind, data)), True)

    def submit(self, value):
        name = hashlib.sha256(value['request']['request_id'].encode()).hexdigest() + '.json'
        atomic_new(self.root / 'requests' / name, value, True)
        return name

    def test_roundtrip_noop_and_reuse(self):
        server = self.service()
        name = self.submit(request())
        server.serve_pending()
        before = (self.root / 'responses' / name).read_bytes()
        server.serve_pending()
        self.service().serve_pending()
        self.assertEqual(before, (self.root / 'responses' / name).read_bytes())
        self.assertEqual(read_message(self.root / 'responses' / name, True)['candidate_ids'], ['1', '0'])
        self.assertEqual([event[0] for event in self.events], ['service_response', 'service_response_reused'])

    def test_reverse_is_complete_permutation(self):
        server = self.service('reverse_native')
        name = self.submit(request())
        server.serve_pending()
        self.assertEqual(read_message(self.root / 'responses' / name, True)['candidate_ids'], ['0', '1'])

    def test_no_live_backend_can_be_enabled_by_runner(self):
        with self.assertRaises(ValueError):
            self.service('live')
        with self.assertRaises(ValueError):
            self.service('llm')

    def test_llm_full_service_with_synthetic_transport_and_budget(self):
        calls = []
        def transport(url, headers, body, timeout):
            calls.append(body)
            payload = json.loads(body['input'][0]['content'])['request']
            result = dict(request_id=payload['request_id'], state_id=payload['state_id'],
                          candidate_ids=list(reversed(payload['candidate_ids'])))
            return dict(status='completed', model='fixture', output=[dict(type='message',
                role='assistant', content=[dict(type='output_text', text=json.dumps(result))])])
        config = dict(enabled=True, model='fixture', api_key_env='TEST_RANK_KEY',
                      max_calls=1, max_usd=1, input_usd_per_million=1,
                      output_usd_per_million=1)
        with patch.dict('os.environ', {'TEST_RANK_KEY': 'fixture-only-never-live'}):
            server = FileRankServer(self.root, 'test', 'llm',
                lambda kind, **data: self.events.append((kind, data)), True,
                llm_config=config, transport=transport)
            name = self.submit(request())
            server.serve_pending()
            response = read_message(self.root / 'responses' / name, True)
            self.assertEqual(response['candidate_ids'], ['0', '1'])
            self.assertIsNone(response['fallback_reason'])
            self.assertFalse(response['metadata']['llm_live'])
            second = request()
            second['request']['request_id'] = 'test:r2'
            second_name = self.submit(second)
            server.serve_pending()
            self.assertEqual(read_message(self.root / 'responses' / second_name, True)
                             ['fallback_reason'], 'call_budget_exhausted')
            self.assertEqual(len(calls), 1)
            server.close()

    def test_native_fallback_when_llm_disabled(self):
        server = self.service('llm_disabled')
        name = self.submit(request())
        server.serve_pending()
        response = read_message(self.root / 'responses' / name, True)
        self.assertEqual(response['candidate_ids'], ['1', '0'])
        self.assertEqual(response['fallback_reason'], 'llm_disabled')
        server.close()

    def test_filename_identity_mismatch_rejected(self):
        server = self.service()
        atomic_new(self.root / 'requests' / ('a' * 64 + '.json'), request(), True)
        with self.assertRaises(ValueError):
            server.serve_pending()
        self.assertFalse(list((self.root / 'responses').iterdir()))

    def test_candidate_injection_rejected(self):
        server = self.service()
        value = request()
        value['request']['candidate_ids'].append('foreign')
        self.submit(value)
        with self.assertRaises(ValueError):
            server.serve_pending()

    def test_new_write_cannot_replace_existing_bytes(self):
        path = self.root / 'file.json'
        atomic_new(path, {'one': 1}, True)
        with self.assertRaises(FileExistsError):
            atomic_new(path, {'two': 2}, True)
        self.assertEqual(read_message(path, True), {'one': 1})
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_excluded_path_rejected_before_filesystem(self):
        with patch.object(Path, 'lstat', side_effect=AssertionError('must not inspect path')):
            for name in ('E:/unused', '/mnt/e/unused', '/mnt/./e/unused', '/mnt/c/../e/unused'):
                with self.assertRaises(ValueError):
                    guarded(name, True)


if __name__ == '__main__':
    unittest.main()
