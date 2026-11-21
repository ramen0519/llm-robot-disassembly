"""Offline CLI and provenance checks on synthetic run directories."""
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import rank_path_proposals as cli
from worker.path_only_sequence_proposal import build_proposal
from worker.proposal_stability import assess_step
from worker.proposal_support_policy import effective_support_cap, parse_support_cap


def synthetic_result(counts, support_cap=2, parts=('a', 'b', 'c')):
    parts = list(parts)
    edges = []
    for i, count in enumerate(counts):
        child = parts[i+1:]
        step = dict(parent=parts[i:], child=child, part_move=parts[i], path_pass=True, pose=None,
                    deadline_exceeded=False, accepted_for_transition=True, action=[1, 0, 0],
                    path_fixed_parts=child, attempt_id=i+1, finished_s=float(i+1), terminal=len(child)==1)

        def check_stable(asset, assembly, fixed, free, **kwargs):
            if count is None:
                return False, None
            if len(fixed) < count:
                return False, [free[0]]
            return True, None

        step['stability'] = assess_step(step, asset_folder='fixture', assembly_dir='fixture',
                                        support_cap=effective_support_cap(support_cap, len(child)),
                                        check_stable=check_stable, wall_time_s=100.0, clock=lambda: 0.0)
        edges.append(step)
    result = build_proposal(parts, edges)
    result.update(initial_parts=parts, candidate_policy='heur-out', api_calls=0, execution_status='OK',
                  input_unchanged=True, asap_sources_unchanged=True, path_physics={'limits': {'path': 192, 'stability': 1}})
    return result


class PathProposalRankingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=cli._lexical_path(tempfile.gettempdir()))
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        active_root = patch('worker.task_memory.PROJECT', self.workspace)
        active_root.start()
        self.addCleanup(active_root.stop)
        self.runs = self.workspace / 'runs'

    def run_fixture(self, name, seed, counts, support_cap=2, parts=('a', 'b', 'c')):
        folder = self.runs / name / 'physics'
        folder.mkdir(parents=True)
        assembly = '/opt/asap-research/ASAP/assets/example'
        sha = '1' * 64
        settings = dict(profile=cli.PROFILE, assembly=assembly, api_calls=0,
                        parameters=dict(assembly_dir=assembly, output_dir=str(folder), seed=seed, budget=32,
                                        max_poses=128, plan_seconds=420.0, audit_seconds=240.0, support_cap=support_cap),
                        generator='heur-out', planner='dfs', base_part=None, pose_reuse=0,
                        terminal='original_mass_rule', cache_policy='read_only_existing_SDF',
                        input_sha256={part+'.obj': sha for part in parts}, asap_commit='a'*40,
                        asap_path_sources={k: sha for k in cli.ASAP_SOURCES},
                        runner_sha256={k: sha for k in cli.RUNNER_SOURCES}, redmax_binary_sha256=sha,
                        native_physics_constants={k: 1 for k in cli.PHYSICS_CONSTANTS})
        (folder / 'settings.json').write_text(json.dumps(settings), encoding='utf-8')
        (folder / 'result.json').write_text(json.dumps(synthetic_result(counts, support_cap, parts)), encoding='utf-8')
        return folder

    def test_support_cap_parsing_and_step_limits(self):
        for value, policy in [('auto', 'auto'), ('0', 0), ('2', 2), (0, 0), (3, 3)]:
            with self.subTest(value=value):
                self.assertEqual(parse_support_cap(value), policy)
                expected = 4 if policy == 'auto' else min(policy, 4)
                self.assertEqual(effective_support_cap(value, 4), expected)
                self.assertEqual(effective_support_cap(value, 0), 0)
        for invalid in (True, False, None, -1, '-1', '1.5', '', 'all', 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_support_cap(invalid)
        for invalid in (True, -1, '2'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                effective_support_cap('auto', invalid)

    def test_auto_ranks_three_support_witnesses_and_preserves_policy(self):
        parts = list('abcde')
        counts = [3, 0, 2, 1]
        auto = self.run_fixture('auto', 0, counts, 'auto', parts)
        other = self.run_fixture('other', 1, [4, 0, 1, 0], 'auto', parts)
        report = cli.rank_runs([other, auto])
        self.assertEqual(report['comparison_settings']['parameters']['support_cap'], 'auto')
        self.assertEqual([(r['support_cap'], r['score']) for r in report['ranked']],
                         [('auto', [3, 6]), ('auto', [4, 5])])
        fixed = self.run_fixture('fixed', 0, counts, 4, parts)
        self.assertEqual(cli.rank_runs([fixed])['ranked'][0]['score'], [3, 6])
        two = self.run_fixture('two', 0, counts, 2, parts)
        blocked, = cli.rank_runs([two])['conditional']
        self.assertEqual(blocked['step_counts'], [None, 0, 2, 1])
        self.assertEqual(blocked['step_statuses'][0], 'REJECTED')
        self.assertIsNone(blocked['score'])
        with self.assertRaisesRegex(ValueError, 'incomparable settings'):
            cli.rank_runs([auto, fixed])

        path = auto / 'result.json'
        recorded = json.loads(path.read_bytes())
        recorded['steps'][0]['stability']['budget']['support_cap'] = 2
        path.write_text(json.dumps(recorded), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'step audit budget'):
            cli.rank_runs([auto])

    def test_output_orders_eligible_ties_and_preserves_conditional(self):
        more = self.run_fixture('more', 0, [2, 0])
        tied1 = self.run_fixture('tied1', 1, [1, 0])
        blocked = self.run_fixture('blocked', 2, [None, 0])
        tied2 = self.run_fixture('tied2', 3, [0, 1])
        output = self.runs / 'report' / 'rank.json'
        args = ['--output', str(output)] + [str(p) for p in (more, tied1, blocked, tied2)]
        self.assertEqual(cli.main(args), 0)
        saved = output.read_bytes()
        result = json.loads(saved)
        self.assertEqual([(r['seed'], r['score'], r['rank']) for r in result['ranked']],
                         [(1, [1, 1], 1), (3, [1, 1], 1), (0, [2, 2], 3)])
        row, = result['conditional']
        self.assertEqual((row['rank'], row['score'], row['stable_prefix'], row['step_counts']),
                         (None, None, 0, [None, 0]))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            cli.main(args)
        self.assertEqual(saved, output.read_bytes())

    def test_comparison_rejects_different_budgets_physics_and_input_hashes(self):
        a = self.run_fixture('a', 0, [0, 0])
        b = self.run_fixture('b', 1, [0, 0])
        path = b / 'settings.json'
        original = path.read_bytes()
        for group, key, value in (
                ('parameters', 'plan_seconds', 419), ('parameters', 'audit_seconds', 200),
                ('parameters', 'max_poses', 127), ('parameters', 'support_cap', 3),
                (None, 'pose_reuse', 1), ('input_sha256', 'a.obj', '9' * 64),
                ('asap_path_sources', 'plan_sequence/planner/base.py', '2' * 64),
                ('runner_sha256', 'proposal_stability.py', '3' * 64),
                ('native_physics_constants', 'CONTACT_EPS', 0.2),
                (None, 'redmax_binary_sha256', '4' * 64)):
            with self.subTest(field=key):
                changed = json.loads(original)
                (changed if group is None else changed[group])[key] = value
                path.write_text(json.dumps(changed), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'incomparable'):
                    cli.rank_runs([a, b])

    def test_rejects_mismatched_pair_and_missing_verifier_evidence(self):
        folder = self.run_fixture('pair', 0, [0, 0])
        path = folder / 'settings.json'
        original = path.read_bytes()
        for field, value in (('output_dir', str(folder.parent)), ('budget', 1)):
            changed = json.loads(original)
            changed['parameters'][field] = value
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.subTest(field=field), self.assertRaises(ValueError):
                cli.rank_runs([folder])
        changed = json.loads(original)
        changed['input_sha256']['x.obj'] = changed['input_sha256'].pop('a.obj')
        path.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'mesh IDs'):
            cli.rank_runs([folder])
        changed = json.loads(original)
        del changed['runner_sha256']['proposal_stability.py']
        path.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            cli.rank_runs([folder])

    def test_lexical_paths_rejected_before_lookup(self):
        with patch('tools.rank_path_proposals._safe_path', side_effect=AssertionError('unexpected lookup')):
            for value in ('E:/should-not-be-opened', '/mnt/e/blocked', 'runs/../outside',
                          'runs/previous.json:alternate', 'runs/nul.json'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    cli._run_path(value)

    def test_result_binds_to_saved_settings_and_score_when_present(self):
        folder = self.run_fixture('bound', 9, [0, 0])
        row, = cli.rank_runs([folder])['ranked']
        path = folder / 'result.json'
        result = json.loads(path.read_bytes())
        result['settings_sha256'] = row['settings_sha256']
        result['support_score'] = cli.score_plan(result, support_cap=2)
        path.write_text(json.dumps(result), encoding='utf-8')
        self.assertEqual(cli.rank_runs([folder])['ranked'][0]['score'], [0, 0])
        result['support_score']['score'] = [1, 1]
        path.write_text(json.dumps(result), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'saved support score'):
            cli.rank_runs([folder])
        result['settings_sha256'] = '9' * 64
        path.write_text(json.dumps(result), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'different settings.json'):
            cli.rank_runs([folder])


if __name__ == '__main__':
    unittest.main()
