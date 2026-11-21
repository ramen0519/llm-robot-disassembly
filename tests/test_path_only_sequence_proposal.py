"""Synthetic path-verdict fixtures for the sequence proposal contract."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker'))
from path_only_sequence_proposal import build_proposal


def edge(parent, child, part_move, path_pass=True):
    return {
        'parent': parent,
        'child': child,
        'part_move': part_move,
        'action': [1.0, 0.0, 0.0] if path_pass else None,
        'path_pass': path_pass,
    }


class PathOnlySequenceProposalTests(unittest.TestCase):
    def test_chosen_branch_reverses_to_full_candidate_starting_with_last_part(self):
        parts = ['lid', 'pin', 'shaft', 'base']
        dead_end = edge(parts, ['pin', 'shaft', 'base'], 'lid')
        chosen = [
            edge(parts, ['shaft', 'base', 'lid'], 'pin'),
            edge(['shaft', 'base', 'lid'], ['shaft', 'base'], 'lid'),
            edge(['shaft', 'base'], ['base'], 'shaft'),
        ]
        result = build_proposal(parts, [dead_end] + chosen)

        self.assertIs(result['complete'], True)
        self.assertEqual(result['disassembly_order'], ['pin', 'lid', 'shaft'])
        self.assertEqual(result['remaining_parts'], ['base'])
        self.assertEqual(result['assembly_order_candidate'], ['base', 'shaft', 'lid', 'pin'])
        self.assertEqual(len(result['steps']), len(chosen))
        for step, expected in zip(result['steps'], chosen):
            self.assertCountEqual(step['parent'], expected['parent'])
            self.assertCountEqual(step['child'], expected['child'])
            self.assertEqual(step['part_move'], expected['part_move'])

    def test_sibling_removals_and_orphaned_terminal_do_not_complete_partial_chain(self):
        parts = ['cap', 'pin', 'rail', 'base']
        result = build_proposal(parts, [
            edge(parts, ['pin', 'rail', 'base'], 'cap'),
            edge(parts, ['cap', 'rail', 'base'], 'pin'),
            edge(['pin', 'rail', 'base'], ['rail', 'base'], 'pin'),
            edge(['cap', 'base'], ['base'], 'cap'),
        ])

        self.assertIs(result['complete'], False)
        self.assertEqual(result['disassembly_order'], ['cap', 'pin'])
        self.assertCountEqual(result['remaining_parts'], ['rail', 'base'])
        self.assertIsNone(result['assembly_order_candidate'])
        self.assertEqual(len(result['steps']), 2)

    def test_failed_removal_cannot_connect_a_passing_descendant(self):
        parts = ['cap', 'pin', 'base']
        first = edge(parts, ['pin', 'base'], 'cap', path_pass=False)
        terminal = edge(['pin', 'base'], ['base'], 'pin')
        result = build_proposal(parts, [first, terminal])

        self.assertIs(result['complete'], False)
        self.assertEqual(result['disassembly_order'], [])
        self.assertCountEqual(result['remaining_parts'], parts)
        self.assertEqual(result['steps'], [])
        self.assertIsNone(result['assembly_order_candidate'])

    def test_failed_attempt_does_not_preclude_a_later_pass_for_same_removal(self):
        parts = ['pin', 'base']
        result = build_proposal(parts, [
            edge(parts, ['base'], 'pin', path_pass=False),
            edge(parts, ['base'], 'pin'),
        ])

        self.assertIs(result['complete'], True)
        self.assertEqual(result['assembly_order_candidate'], ['base', 'pin'])
        self.assertEqual(len(result['steps']), 1)

    def test_terminal_path_pass_after_deadline_does_not_complete_accepted_prefix(self):
        parts = ['cap', 'pin', 'base']
        first = edge(parts, ['pin', 'base'], 'cap')
        first['accepted_for_transition'] = True
        terminal = edge(['pin', 'base'], ['base'], 'pin')
        terminal['accepted_for_transition'] = False
        result = build_proposal(parts, [first, terminal])

        self.assertIs(result['complete'], False)
        self.assertEqual(result['disassembly_order'], ['cap'])
        self.assertCountEqual(result['remaining_parts'], ['pin', 'base'])
        self.assertIsNone(result['assembly_order_candidate'])
        self.assertEqual(len(result['steps']), 1)

    def test_malformed_passing_edges_are_rejected(self):
        parts = ['cap', 'pin', 'base']
        malformed = {
            'two parts removed': edge(parts, ['base'], 'cap'),
            'other part removed': edge(parts, ['pin', 'base'], 'pin'),
            'foreign child part': edge(parts, ['bolt', 'base'], 'cap'),
            'duplicate child id': edge(parts, ['base', 'base'], 'cap'),
        }
        for problem, record in malformed.items():
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                build_proposal(parts, [record])

    def test_other_verifier_profile_is_rejected(self):
        parts = ['pin', 'base']
        with self.assertRaises(ValueError):
            build_proposal(parts, [edge(parts, ['base'], 'pin')], profile='sequence_native_v1')


if __name__ == '__main__':
    unittest.main()
