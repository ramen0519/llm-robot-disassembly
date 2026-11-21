"""Offline cumulative memory tests; all physics outcomes are labelled fixtures."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_explicit_action_contract import request
from test_explicit_action_memory import outcome
from worker.explicit_action_contract import (PROTOCOL, canonical, canonical_response,
    digest, enumerate_actions, action_id)
from orchestrator.explicit_action_memory import (append_action_observation,
    freeze_memory, freeze_cumulative_memory, load_snapshot, FrozenActionMemory)


class CumulativeActionMemoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='cumulative_memory_fixture_')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root / 'cumulative.json'
        self.serial = 0

    def add(self, filename='episode.jsonl', status='PASS', action_index=0,
            context=None, parent=None, run='fixture', state='s1'):
        self.serial += 1
        r = deepcopy(context) if context is not None else request()
        r.update(run_id=run, request_id=run + ':request:' + str(self.serial), state_id=state)
        choice = next(itertools.islice(enumerate_actions(r), action_index, None))
        response = canonical_response(r, None)
        response['choice'] = choice
        response['candidate_order'] = [choice['part_id']] + [
            part for part in sorted(r['candidate_ids']) if part != choice['part_id']]
        response['action_id'] = action_id(r, choice)
        return append_action_observation(self.root / filename, r, response, outcome(status),
            parent, allow_test_root=True)

    def write_rows(self, filename, records):
        path = self.root / filename
        path.write_text(''.join(canonical(record) + '\n' for record in records), encoding='utf-8')
        return path

    def memory(self, records):
        return FrozenActionMemory({'schema_version': 1, 'protocol': PROTOCOL, 'records': records,
            'record_count': len(records), 'content_sha256': digest(records), 'hard_pruning': False,
            'physics_reuse_allowed': False, 'frozen_utc': None})

    def merged(self, observations=(), snapshots=(), target=None):
        return freeze_cumulative_memory(list(snapshots), list(observations),
            target or self.target, allow_test_root=True)

    def test_union_preserves_all_original_records_and_file_hashes(self):
        first = self.add(filename='one.jsonl', status='UNKNOWN', run='run1')
        frozen = self.root / 'one.json'
        freeze_memory(self.root / 'one.jsonl', frozen, True)
        second = self.add(filename='two.jsonl', run='run2')
        before = {path: path.read_bytes() for path in [frozen, self.root/'one.jsonl', self.root/'two.jsonl']}
        result = self.merged([self.root/'two.jsonl'], [frozen])
        self.assertEqual(result['records'], [first, second])
        self.assertEqual(load_snapshot(self.target, True), result)
        self.assertEqual(result['record_count'], 2)
        for source in result['merge_provenance']['sources']:
            self.assertEqual(source['file_sha256'], hashlib.sha256(before[Path(source['path'])]).hexdigest())
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertFalse(result['hard_pruning'])
        self.assertFalse(result['physics_reuse_allowed'])

    def test_identical_evidence_deduplicates_across_sources_without_rewriting(self):
        record = self.add()
        snapshot = self.root/'first.json'
        freeze_memory(self.root/'episode.jsonl', snapshot, True)
        merged = self.merged([self.root/'episode.jsonl'], [snapshot])
        self.assertEqual(merged['records'], [record])
        self.assertEqual(merged['merge_provenance']['source_record_count'], 2)
        self.assertEqual(merged['merge_provenance']['duplicate_record_count'], 1)

    def test_identical_rows_inside_observation_source_deduplicate(self):
        record = self.add()
        path = self.write_rows('duplicates.jsonl', [record, record])
        result = self.merged([path])
        self.assertEqual(result['records'], [record])
        self.assertEqual(result['merge_provenance']['duplicate_record_count'], 1)

    def test_conflicting_same_identity_with_valid_hash_is_rejected(self):
        first = self.add()
        second = deepcopy(first)
        second['result'] = outcome('UNKNOWN')
        second['record_sha256'] = digest({k: v for k, v in second.items() if k != 'record_sha256'})
        path = self.write_rows('conflict.jsonl', [second])
        with self.assertRaisesRegex(ValueError, 'conflicting_cumulative'):
            self.merged([self.root/'episode.jsonl', path])
        self.assertFalse(self.target.exists())

    def test_hash_tampering_rejected_before_output(self):
        first = self.add()
        first['created_utc'] = 'changed-without-rehash'
        with self.assertRaisesRegex(ValueError, 'hash_mismatch'):
            self.merged([self.write_rows('tampered.jsonl', [first])])
        self.assertFalse(self.target.exists())

    def test_parent_can_be_in_another_input_file_but_must_exist_in_union(self):
        parent = self.add(status='UNKNOWN')
        child = self.add(action_index=1, parent=parent['evidence_id'])
        parent_path = self.write_rows('parent.jsonl', [parent])
        child_path = self.write_rows('child.jsonl', [child])
        merged = self.merged([child_path, parent_path])
        self.assertEqual(merged['records'], [child, parent])
        with self.assertRaisesRegex(ValueError, 'parent_must'):
            self.merged([child_path], target=self.root/'missing-parent.json')

    def test_missing_parent_in_loaded_snapshot_is_rejected(self):
        parent = self.add(status='UNKNOWN')
        child = self.add(action_index=1, parent=parent['evidence_id'])
        with self.assertRaisesRegex(ValueError, 'parent_must'):
            self.memory([child])

    def test_false_repair_links_to_different_scope_are_rejected(self):
        parent = self.add(status='UNKNOWN')
        modifications = ['run', 'state', 'task', 'remaining', 'base', 'grippers', 'part']
        for change in modifications:
            context = request()
            run, state, index = 'fixture', 's1', 1
            if change == 'run': run = 'other-run'
            if change == 'state': state = 'other-state'
            if change == 'task': context['task_context']['solver_settings']['seed'] = 99
            if change == 'remaining':
                context.update(remaining_parts=['1', '2', '3', 'base'], candidate_ids=['1', '2', '3'])
            if change == 'base': context.update(base_part=None, candidate_ids=['1', '2', 'base'])
            if change == 'grippers': context['max_grippers'] = 3
            if change == 'part':
                context.update(remaining_parts=['1', '2', 'base'], candidate_ids=['1', '2'])
                actions = list(enumerate_actions(context))
                index = next(i for i, choice in enumerate(actions) if choice['part_id'] == '2')
            child = self.add(filename=change+'.jsonl', context=context, run=run, state=state, action_index=index)
            child['parent_failure_evidence_id'] = parent['evidence_id']
            child['record_sha256'] = digest({k: v for k, v in child.items() if k != 'record_sha256'})
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, 'scope_mismatch'):
                    self.memory([parent, child])

    def test_parent_cannot_be_pass_error_or_identical_action(self):
        for status, index, error in [('PASS', 1, 'parent_must'), ('ERROR', 1, 'parent_must'),
                                     ('UNKNOWN', 0, 'changed_action')]:
            parent = self.add(filename=status+'.jsonl', status=status)
            child = self.add(filename='child-'+status+'.jsonl', action_index=index)
            child['parent_failure_evidence_id'] = parent['evidence_id']
            child['record_sha256'] = digest({k: v for k, v in child.items() if k != 'record_sha256'})
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, error): self.memory([parent, child])

    def test_cyclic_failure_links_rejected(self):
        first = self.add(status='UNKNOWN')
        second = self.add(status='UNKNOWN', action_index=1)
        first['parent_failure_evidence_id'] = second['evidence_id']
        second['parent_failure_evidence_id'] = first['evidence_id']
        for row in (first, second):
            row['record_sha256'] = digest({k: v for k, v in row.items() if k != 'record_sha256'})
        with self.assertRaisesRegex(ValueError, 'cyclic'): self.memory([first, second])

    def test_second_cumulative_snapshot_keeps_earlier_records_and_source_chain(self):
        first = self.add(filename='one.jsonl')
        one = self.merged([self.root/'one.jsonl'])
        two = self.add(filename='two.jsonl', run='run2', status='UNKNOWN')
        original = self.target.read_bytes()
        result = self.merged([self.root/'two.jsonl'], [self.target], self.root/'next.json')
        self.assertEqual(result['records'], [first, two])
        self.assertEqual(result['merge_provenance']['sources'][0]['file_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(load_snapshot(self.target, True), one)

    def test_existing_destination_and_sources_never_overwritten(self):
        self.add()
        result = self.merged([self.root/'episode.jsonl'])
        before = self.target.read_bytes()
        with self.assertRaises(FileExistsError): self.merged([self.root/'episode.jsonl'])
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(load_snapshot(self.target, True), result)

    def test_empty_union_is_valid_and_missing_source_is_not_silent(self):
        self.assertEqual(self.merged()['record_count'], 0)
        with self.assertRaises(FileNotFoundError):
            self.merged([self.root/'missing.jsonl'], target=self.root/'other.json')

    def test_source_path_container_must_be_explicit_list(self):
        with self.assertRaisesRegex(ValueError, 'path_lists'):
            freeze_cumulative_memory('not-a-list', [], self.target, True)

    def test_input_byte_limit_is_individual_and_aggregate(self):
        self.add(filename='one.jsonl')
        self.add(filename='two.jsonl', run='second')
        one, two = self.root/'one.jsonl', self.root/'two.jsonl'
        for sources, limit in [([one], one.stat().st_size - 1),
                ([one, two], max(one.stat().st_size, two.stat().st_size) + 1)]:
            with patch('orchestrator.explicit_action_memory.MEMORY_BYTE_LIMIT', limit):
                with self.assertRaisesRegex(ValueError, 'input_limit'): self.merged(sources)
            self.assertFalse(self.target.exists())

    def test_output_byte_limit_fails_before_creating_destination(self):
        self.add()
        source = self.root/'episode.jsonl'
        with patch('orchestrator.explicit_action_memory.MEMORY_BYTE_LIMIT', source.stat().st_size + 1):
            with self.assertRaisesRegex(ValueError, 'output_limit'): self.merged([source])
        self.assertFalse(self.target.exists())

    def test_excluded_drive_is_rejected_before_filesystem_lookup(self):
        with patch('pathlib.Path.lstat', side_effect=AssertionError('no_filesystem_lookup')):
            with self.assertRaises(ValueError):
                freeze_cumulative_memory(['E:/excluded.json'], [], self.target, True)

    def test_merge_provenance_counts_and_hash_format_validated(self):
        self.add()
        result = self.merged([self.root/'episode.jsonl'])
        for mutate in [lambda x: x['merge_provenance'].update(duplicate_record_count=2),
                lambda x: x['merge_provenance']['sources'][0].update(file_sha256='not-a-hash'),
                lambda x: x['merge_provenance']['sources'][0].update(record_count=8)]:
            changed = deepcopy(result)
            mutate(changed)
            with self.assertRaisesRegex(ValueError, 'merge'): FrozenActionMemory(changed)

    def test_legacy_retrieval_order_is_unchanged(self):
        rows = [self.add(status='UNKNOWN'), self.add(action_index=1), self.add(action_index=2)]
        refs = self.memory(rows).retrieve(request())
        self.assertEqual([ref['evidence_id'] for ref in refs], sorted(row['evidence_id'] for row in rows))

    def test_state_retrieval_preserves_parent_closure_and_honest_flags(self):
        parent = self.add(status='UNKNOWN')
        child = self.add(action_index=1, parent=parent['evidence_id'])
        other = self.add(action_index=2)
        memory = self.memory([other, child, parent])
        refs = memory.retrieve(request(), max_records=2, policy='state_diverse_v1')
        self.assertEqual([ref['evidence_id'] for ref in refs], [parent['evidence_id'], child['evidence_id']])
        self.assertEqual(refs[1]['repair_lifecycle'], 'provisional_observed_recovery_only')
        for ref in refs:
            self.assertFalse(ref['physics_reuse_allowed'])
            self.assertFalse(ref['hard_pruning'])
            self.assertFalse(ref['current_attempt_verified'])
            self.assertIsNone(ref['current_part_id_mapping'])
        one = memory.retrieve(request(), max_records=1, policy='state_diverse_v1')
        self.assertNotIn(child['evidence_id'], [ref['evidence_id'] for ref in one])

    def test_nested_parent_chain_is_kept_and_never_partially_exposed(self):
        first = self.add(status='UNKNOWN')
        second = self.add(status='UNKNOWN', action_index=1, parent=first['evidence_id'])
        third = self.add(action_index=2, parent=second['evidence_id'])
        memory = self.memory([third, second, first])
        refs = memory.retrieve(request(), 3, 'state_diverse_v1')
        self.assertEqual([ref['evidence_id'] for ref in refs], [r['evidence_id'] for r in [first, second, third]])
        refs = memory.retrieve(request(), 2, 'state_diverse_v1')
        self.assertNotIn(third['evidence_id'], [ref['evidence_id'] for ref in refs])
        for ref in refs:
            parent_id = ref['parent_failure_evidence_id']
            self.assertTrue(parent_id is None or parent_id in [row['evidence_id'] for row in refs])

    def test_duplicate_action_outcomes_choose_one_representative_but_keep_conflicting_outcomes(self):
        rows = [self.add(run='one'), self.add(run='two'), self.add(run='three', status='UNKNOWN')]
        refs = self.memory(rows).retrieve(request(), 12, 'state_diverse_v1')
        self.assertEqual(len(refs), 2)
        self.assertEqual({ref['status'] for ref in refs}, {'PASS', 'UNKNOWN'})

    def test_equal_relevance_balances_failure_and_success(self):
        rows = [self.add(action_index=i) for i in range(5)]
        failed = self.add(action_index=5, status='UNKNOWN')
        refs = self.memory(rows + [failed]).retrieve(request(), 2, 'state_diverse_v1')
        self.assertEqual([ref['status'] for ref in refs], ['UNKNOWN', 'PASS'])

    def test_current_conditions_take_priority_over_unrelated_success(self):
        unrelated = request()
        unrelated['task_context']['assembly_id'] = 'other'
        unrelated['task_context']['input_sha256'] = 'b'*64
        foreign = self.add(context=unrelated)
        failed = self.add(status='UNKNOWN')
        refs = self.memory([foreign, failed]).retrieve(request(), 1, 'state_diverse_v1')
        self.assertEqual(refs[0]['evidence_id'], failed['evidence_id'])

    def test_same_assembly_changed_seed_stays_reference_only(self):
        prior = request()
        prior['task_context']['solver_settings']['seed'] = 27
        related = self.add(context=prior, status='UNKNOWN')
        foreign_context = deepcopy(prior)
        foreign_context['task_context'].update(assembly_id='unrelated', input_sha256='b'*64)
        foreign = self.add(context=foreign_context)
        refs = self.memory([foreign, related]).retrieve(request(), 1, 'state_diverse_v1')
        self.assertEqual(refs[0]['evidence_id'], related['evidence_id'])
        self.assertFalse(refs[0]['same_task'])
        self.assertFalse(refs[0]['same_conditions'])
        self.assertTrue(refs[0]['reference_only'])
        self.assertIsNone(refs[0]['current_part_id_mapping'])

    def test_real_01256_nested_context_changed_seed_ranks_ahead_of_foreign_cad(self):
        artifact = Path(__file__).resolve().parents[1] / (
            'runs/assembly_01256_learning_20261018/actual_payload_preflight/preflight_initial_request.json')
        if not artifact.is_file():
            self.skipTest('real 01256 preflight artifact unavailable; nested fixture test remains active')
        current = json.loads(artifact.read_text(encoding='utf-8'))
        self.assertEqual(len(current['candidate_ids']), 18)
        self.assertEqual(current['task_context']['task']['assembly_id'], '01256')
        self._check_nested_related_priority(current)

    def test_nested_schema2_context_uses_task_identity_for_similarity(self):
        current = request()
        current['task_context'] = {'schema_version': 2, 'task': current['task_context'],
            'initial_parts': sorted(current['remaining_parts']), 'geometry_features': {}}
        self._check_nested_related_priority(current)

    def _check_nested_related_priority(self, current):
        prior = deepcopy(current)
        prior['task_context']['task']['solver_settings']['seed'] = 27
        related = self.add(context=prior, status='UNKNOWN', run='related-seed')
        foreign_context = deepcopy(prior)
        foreign_context['task_context']['task'].update(assembly_id='foreign-cad', input_sha256='b'*64)
        # Deliberately beat the related record in the fallback ID tie-break so
        # this checks actual nested identity relevance, not accidental ordering.
        foreign = None
        for index in range(64):
            candidate = self.add(filename='foreign.jsonl', context=foreign_context,
                                 status='UNKNOWN', run='foreign-'+str(index))
            if candidate['evidence_id'] < related['evidence_id']:
                foreign = candidate
                break
        self.assertIsNotNone(foreign, 'fixture must exercise the relevance rule before its ID tie-break')
        refs = self.memory([foreign, related]).retrieve(current, 1, 'state_diverse_v1')
        self.assertEqual(refs[0]['evidence_id'], related['evidence_id'])
        self.assertTrue(refs[0]['reference_only'])
        self.assertFalse(refs[0]['same_conditions'])
        self.assertIsNone(refs[0]['current_part_id_mapping'])

    def test_same_remaining_and_pose_rank_above_other_states(self):
        exact = self.add(status='UNKNOWN')
        fewer = request()
        fewer.update(remaining_parts=['1', 'base'], candidate_ids=['1'])
        removed = self.add(context=fewer)
        other_pose = request()
        other_pose['pose_catalog'][0]['matrix'][0][3] = 1
        posed = self.add(context=other_pose)
        refs = self.memory([removed, posed, exact]).retrieve(request(), 1, 'state_diverse_v1')
        self.assertEqual(refs[0]['evidence_id'], exact['evidence_id'])

    def test_new_policy_deterministic_under_input_reordering_and_returns_copies(self):
        rows = [self.add(action_index=i, status='UNKNOWN' if i % 2 else 'PASS') for i in range(6)]
        memory = self.memory(rows)
        first = memory.retrieve(request(), 4, 'state_diverse_v1')
        self.assertEqual(first, self.memory(list(reversed(rows))).retrieve(request(), 4, 'state_diverse_v1'))
        first[0]['status'] = 'CHANGED'
        self.assertNotEqual(memory.retrieve(request(), 4, 'state_diverse_v1')[0]['status'], 'CHANGED')

    def test_limits_and_unknown_policy_rejected(self):
        memory = self.memory([self.add()])
        self.assertEqual(memory.retrieve(request(), 0, 'state_diverse_v1'), [])
        for limit in [-1, 13, True, 1.5]:
            with self.assertRaisesRegex(ValueError, 'retrieval_limit'):
                memory.retrieve(request(), limit, 'state_diverse_v1')
        with self.assertRaisesRegex(ValueError, 'retrieval_policy'):
            memory.retrieve(request(), policy='unknown')


if __name__ == '__main__':
    unittest.main()
