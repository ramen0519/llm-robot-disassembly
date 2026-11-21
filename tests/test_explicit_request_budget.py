"""Offline payload bounds and preflight isolation; never real API or physics."""
from copy import deepcopy
import builtins
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'worker'))

from worker.explicit_action_contract import validate_request
from worker.explicit_request_budget import (MAX_REQUEST_BYTES, RequestCoreTooLarge,
    bounded_memory_request, capacity_probe, request_bytes)


def request():
    parts = [str(i) for i in range(18)]
    return dict(schema_version=1, protocol='explicit_action_v1', run_id='offline',
        request_id='offline:r1', state_id='offline:s1', remaining_parts=parts,
        candidate_ids=parts[:], base_part=None, max_grippers=3,
        pose_catalog=[dict(pose_id='p'+str(i), matrix=[[1, 0, 0, i],
            [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) for i in range(3)],
        attempted_action_ids=[], attempted_actions=[], task_context=dict(assembly_id='fixture-18',
            input_sha256={p+'.obj': 'a'*64 for p in parts}, seed=0, budget=60),
        geometry=dict(parts={p: dict(bounds=[[0, 0, 0], [1, 2, 3]], extents=[1, 2, 3])
            for p in parts}), evidence_ids=[], episode_feedback=[], memory_references=[])


def reference(identity, parent=None, padding=''):
    return dict(evidence_id=identity, parent_failure_evidence_id=parent,
        reference_only=True, current_part_id_mapping=None,
        hard_pruning=False, physics_reuse_allowed=False,
        source_task_context=dict(assembly_id='source'), source_remaining_parts=['a', 'b'],
        source_choice=dict(part_id='a'), source_observation=padding)


class RequestBudgetTests(unittest.TestCase):
    def test_18_candidates_60_attempts_12_feedback_12_refs_preserved(self):
        original = request()
        result, audit = capacity_probe(original, memory_count=12)
        validate_request(result)
        self.assertEqual(result['candidate_ids'], original['candidate_ids'])
        self.assertEqual(result['pose_catalog'], original['pose_catalog'])
        self.assertEqual(len(result['attempted_actions']), 60)
        self.assertEqual(len(result['attempted_action_ids']), 60)
        self.assertEqual(len(result['episode_feedback']), 12)
        self.assertEqual(len(result['memory_references']), 12)
        self.assertLessEqual(audit['request_bytes'], MAX_REQUEST_BYTES)
        self.assertTrue(audit['synthetic_history'])
        self.assertFalse(audit['observed_physics'])
        self.assertEqual(original, request())

    def test_oversize_reference_skipped_without_mutating_any_core_field(self):
        core, _ = capacity_probe(request(), memory_count=0)
        refs = [reference('huge', padding='x'*MAX_REQUEST_BYTES), reference('small')]
        before = deepcopy(core)
        result, audit = bounded_memory_request(core, refs, True)
        for key in set(core) - {'memory_references', 'evidence_ids'}:
            self.assertEqual(result[key], core[key], key)
        self.assertEqual(result['memory_references'], [refs[1]])
        self.assertEqual(core, before)
        self.assertEqual(refs[0]['source_observation'], 'x'*MAX_REQUEST_BYTES)
        self.assertEqual(audit['excluded_references'][0]['reason'], 'request_byte_limit')
        self.assertEqual(audit['provided_evidence_ids'], ['small'])

    def test_oversize_core_fails_instead_of_trimming_geometry_or_attempts(self):
        core, _ = capacity_probe(request(), memory_count=0)
        core['geometry']['oversize'] = 'x'*MAX_REQUEST_BYTES
        before = deepcopy(core)
        with self.assertRaises(RequestCoreTooLarge) as caught:
            bounded_memory_request(core, [reference('small')], True)
        self.assertEqual(str(caught.exception), 'request_core_too_large')
        self.assertGreater(caught.exception.audit['core_bytes'], MAX_REQUEST_BYTES)
        self.assertFalse(caught.exception.audit['search_scope_changed'])
        self.assertEqual(core, before)

    def test_parent_chain_is_added_whole_even_when_child_listed_first(self):
        refs = [reference('child', 'parent'), reference('parent', 'ancestor'), reference('ancestor')]
        result, audit = bounded_memory_request(request(), refs, True)
        self.assertEqual([r['evidence_id'] for r in result['memory_references']],
                         ['ancestor', 'parent', 'child'])
        self.assertEqual(result['memory_references'], [refs[2], refs[1], refs[0]])
        self.assertEqual(audit['excluded_reference_count'], 0)

    def test_missing_parent_excludes_child_without_inventing_parent(self):
        result, audit = bounded_memory_request(request(), [reference('child', 'missing')], True)
        self.assertEqual(result['memory_references'], [])
        self.assertEqual(audit['excluded_references'][0]['reason'], 'parent_reference_not_available')

    def test_large_parent_excludes_both_without_truncating_source_scope(self):
        refs = [reference('child', 'parent'), reference('parent', padding='x'*MAX_REQUEST_BYTES)]
        result, audit = bounded_memory_request(request(), refs, True)
        self.assertEqual(result['memory_references'], [])
        self.assertEqual(audit['excluded_reference_count'], 2)

    def test_parent_can_stay_if_child_too_large(self):
        refs = [reference('parent'), reference('child', 'parent', 'x'*MAX_REQUEST_BYTES)]
        result, audit = bounded_memory_request(request(), refs, True)
        self.assertEqual(result['memory_references'], refs[:1])
        self.assertEqual(audit['excluded_reference_count'], 1)

    def test_utf8_bytes_not_character_count_define_boundary(self):
        core = request()
        core['geometry']['padding'] = ''
        available = MAX_REQUEST_BYTES - request_bytes(core)
        core['geometry']['padding'] = '가'*(available//3) + 'x'*(available%3)
        self.assertEqual(request_bytes(core), MAX_REQUEST_BYTES)
        result, _ = bounded_memory_request(core, [])
        self.assertEqual(result, core)
        core['geometry']['padding'] += 'x'
        with self.assertRaises(RequestCoreTooLarge):
            bounded_memory_request(core, [])

    def test_legacy_small_payload_is_byte_for_byte_unchanged(self):
        core = request()
        refs = [reference('child', 'not-retrieved-parent'), reference('other')]
        expected = deepcopy(core)
        expected['memory_references'] = deepcopy(refs)
        expected['evidence_ids'] = ['child', 'other']
        result, audit = bounded_memory_request(core, refs)
        self.assertEqual(result, expected)
        self.assertEqual(audit['excluded_reference_count'], 0)

    def test_malformed_safety_scope_or_hidden_priority_is_not_size_exclusion(self):
        for change in (dict(hard_pruning=True), dict(current_part_id_mapping={'a': '0'}),
                       dict(native_order=['0', '1'])):
            row = reference('bad'); row.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                bounded_memory_request(request(), [row], True)

    def test_duplicate_conflicting_and_cyclic_refs_rejected(self):
        for refs in ([reference('x'), reference('x')],
                     [reference('x', 'y'), reference('y', 'x')]):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                bounded_memory_request(request(), refs, True)
        core = request(); core['memory_references'] = [reference('x')]; core['evidence_ids'] = ['x']
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            bounded_memory_request(core, [reference('x', padding='different')], True)

    def test_row_limit_keeps_existing_rows_and_never_more_than_12(self):
        result, audit = bounded_memory_request(request(), [reference(str(i)) for i in range(15)], True)
        self.assertEqual(len(result['memory_references']), 12)
        self.assertEqual(audit['excluded_reference_count'], 3)
        self.assertTrue(all(r['reason'] == 'reference_count_limit' for r in audit['excluded_references']))

    def test_missing_candidate_stays_error_not_auto_repaired(self):
        core = request(); core['candidate_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'incomplete_candidate_set'):
            bounded_memory_request(core, [])

    def test_bad_probe_counts_rejected(self):
        for kwargs in (dict(memory_count=13), dict(episode_count=13),
                       dict(attempted_count=True), dict(attempted_count=0, episode_count=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                capacity_probe(request(), **kwargs)


class WorkerPreflightIsolationTests(unittest.TestCase):
    def test_worker_preflight_reaches_output_without_planner_physics_or_ipc(self):
        # Full worker control-flow test with explicit CAD/pose fixtures. Actual
        # 01256 geometry is checked separately by the WSL preflight command.
        worker = importlib.import_module('worker.run_explicit_actions')
        matrix = request()['pose_catalog'][0]['matrix']
        array = SimpleNamespace(tolist=lambda: deepcopy(matrix))
        numpy = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None),
                                eye=lambda _: array, round=lambda value, _: value)
        fake = {'numpy': numpy, 'torch': SimpleNamespace(manual_seed=lambda _: None),
            'trimesh': SimpleNamespace(), 'assets': SimpleNamespace(),
            'assets.load': SimpleNamespace(load_part_ids=lambda _: request()['remaining_parts']),
            'assets.transform': SimpleNamespace(transform_pt_by_matrix=lambda *_: None),
            'plan_sequence': SimpleNamespace(),
            'plan_sequence.stable_pose': SimpleNamespace(get_combined_mesh=lambda *_: None)}
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name in ('explicit_action_physics', 'explicit_action_planner',
                        'plan_sequence.planner.dfs', 'plan_sequence.planner.base', 'openai'):
                raise AssertionError('preflight imported execution backend: '+name)
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); assembly = folder / 'assembly'; assembly.mkdir()
            (assembly / '0.obj').write_text('offline fixture', encoding='utf-8')
            output = folder / 'output'
            with patch.dict(sys.modules, fake), patch('builtins.__import__', side_effect=guarded_import), \
                 patch.object(worker, 'checked_path', side_effect=lambda value, **_: Path(value)), \
                 patch.object(worker.subprocess, 'check_output', return_value=worker.COMMIT+'\n'), \
                 patch.object(worker.subprocess, 'run'), \
                 patch.object(worker, 'describe_task', return_value={'fixture': True}), \
                 patch.object(worker, 'task_context', return_value=request()['task_context']), \
                 patch.object(worker, 'filter_features', return_value=request()['geometry']), \
                 patch.object(worker, 'ExplicitActionClient', side_effect=AssertionError('no IPC allowed')):
                code = worker.main(['--asap-root', str(folder), '--assembly-dir', str(assembly),
                    '--output-dir', str(output), '--run-id', 'offline-preflight', '--base-part', '17',
                    '--memory-policy', 'state_diverse_v1', '--memory-max-records', '8',
                    '--request-preflight-only'])
            self.assertEqual(code, 0)
            self.assertTrue((output / 'request_preflight_report.json').is_file())
            self.assertTrue((output / 'preservation.json').is_file())
            self.assertFalse((output / 'ipc').exists())
            self.assertFalse((output / 'planner').exists())

    def test_cli_defaults_preserve_legacy_four_references(self):
        worker = importlib.import_module('worker.run_explicit_actions')
        args = worker.parser().parse_args(['--asap-root', 'x', '--assembly-dir', 'x',
                                          '--output-dir', 'x', '--run-id', 'offline'])
        self.assertEqual(args.memory_policy, 'legacy_v1')
        self.assertEqual(args.memory_max_records, 4)
        self.assertFalse(args.request_preflight_only)


if __name__ == '__main__':
    unittest.main()
