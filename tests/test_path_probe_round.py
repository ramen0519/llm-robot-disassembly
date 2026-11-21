"""Pure scheduling/worker tests with injected physics; no API or simulator."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

from tests.test_explicit_action_contract import request, attach_attempts
from tests.test_explicit_action_planner import make_planner, result_for, response_for
from worker.explicit_action_contract import (action_id, canonical_response,
    enumerate_actions, validate_request, validate_response)
from worker.path_probe_round import (partition_path_round, semantic_path_key,
    validate_path_round_response, canonical_path_round_response)
from worker.explicit_request_budget import capacity_probe


def round_request():
    r = request()
    r.update(attempted_actions=[], path_stage_observations=[])
    return r


def remember(r, choices, values=None):
    attach_attempts(r, choices)
    if values is None:
        values = [False] * len(choices)
    r['path_stage_observations'] = [dict(action_id=identity, path_success=value,
        status='UNKNOWN', deadline_exceeded=False, backend='ASAP_RedMax',
        source_event_id='fixture:actual-stage:'+str(index))
        for index, (identity, value) in enumerate(zip(r['attempted_action_ids'], values))]
    return r


class PathProbeRoundContractTests(unittest.TestCase):
    def test_optional_observation_table_does_not_change_legacy_request(self):
        old = request()
        self.assertNotIn('path_stage_observations', validate_request(old))
        with self.assertRaisesRegex(ValueError, 'observations_required'):
            partition_path_round(old)
        canonical_response(old, 'offline')

    def test_observation_rows_require_ordered_one_to_one_exact_fields(self):
        r = round_request()
        choices = list(enumerate_actions(r))[:2]
        remember(r, choices)
        validate_request(r)
        for mutate in (
            lambda x: x['path_stage_observations'].pop(),
            lambda x: x['path_stage_observations'].reverse(),
            lambda x: x['path_stage_observations'][0].update(extra='unsupported'),
            lambda x: x['path_stage_observations'][0].update(path_success=0),
            lambda x: x['path_stage_observations'][0].update(deadline_exceeded=0),
            lambda x: x['path_stage_observations'][0].update(backend=[]),
            lambda x: x['path_stage_observations'][0].update(source_event_id=''),
            lambda x: x['path_stage_observations'][0].update(source_event_id=x['path_stage_observations'][1]['source_event_id']),
            lambda x: x.pop('attempted_actions'),
            lambda x: x['path_stage_observations'][0].update(action_id='wrong'),
        ):
            changed = deepcopy(r)
            mutate(changed)
            with self.subTest(request=changed):
                with self.assertRaises(ValueError): validate_request(changed)

    def test_provenance_ids_do_not_become_citation_permissions(self):
        r = round_request()
        remember(r, [canonical_response(r, None)['choice']])
        self.assertEqual(validate_request(r)['evidence_ids'], [])
        chosen = canonical_path_round_response(r, None)
        chosen['choice']['evidence_ids'] = [r['path_stage_observations'][0]['source_event_id']]
        with self.assertRaisesRegex(ValueError, 'unknown_evidence_id'):
            validate_response(r, chosen)

    def test_changed_task_or_state_scope_rejects_old_observations(self):
        r = round_request()
        remember(r, [canonical_response(r, None)['choice']])
        r['task_context']['solver_settings']['seed'] = 1
        with self.assertRaisesRegex(ValueError, 'attempted_action_scope_mismatch'):
            partition_path_round(r)

    def test_actual_false_defers_support_variants_but_preserves_full_partition(self):
        r = round_request()
        first = canonical_response(r, None)['choice']
        remember(r, [first])
        saved = deepcopy(r)
        state = partition_path_round(r)
        key = semantic_path_key(r, first)
        self.assertEqual(state['deferred_path_keys'], [key])
        self.assertEqual((state['all_path_count'], state['attempted_path_count'],
                          state['untried_path_count']), (60, 1, 59))
        active, deferred = set(), set()
        for choice in enumerate_actions(r):
            target = deferred if semantic_path_key(r, choice) in state['deferred_path_keys'] else active
            target.add(action_id(r, choice))
        self.assertTrue(active and deferred)
        self.assertFalse(active & deferred)
        self.assertEqual(active | deferred, {action_id(r, c) for c in enumerate_actions(r)})
        self.assertFalse(state['permanent_action_pruning'])
        self.assertFalse(state['physics_verdict_reused'])
        self.assertEqual(r, saved)

    def test_path_success_observation_cancels_deferral(self):
        r = round_request()
        first, alternate = list(enumerate_actions(r))[:2]
        remember(r, [first, alternate], [False, True])
        state = partition_path_round(r)
        self.assertEqual(len(state['observed_failed_path_keys']), 1)
        self.assertEqual(len(state['observed_success_path_keys']), 1)
        self.assertEqual(state['deferred_path_keys'], [])

    def test_error_unknown_late_non_actual_backend_never_prove_stage_failure(self):
        for mutate in (
            lambda row: row.update(status='ERROR'),
            lambda row: row.update(path_success=None),
            lambda row: row.update(deadline_exceeded=True),
            lambda row: row.update(backend=None),
            lambda row: row.update(backend='injected_fixture_not_actual'),
            lambda row: row.update(status='PASS', path_success=False),
        ):
            r = round_request()
            remember(r, [canonical_response(r, None)['choice']])
            mutate(r['path_stage_observations'][0])
            state = partition_path_round(r)
            self.assertEqual(state['deferred_path_keys'], [])
            self.assertEqual(state['ignored_observation_count'], 1)

    def test_all_semantic_paths_probed_releases_every_untried_support(self):
        r = round_request()
        representatives = [c for c in enumerate_actions(r) if not c['support_ids']]
        remember(r, representatives)
        state = partition_path_round(r)
        self.assertEqual(state['untried_path_count'], 0)
        self.assertEqual(state['round_index'], 1)
        self.assertEqual(state['release_reason'], 'untried_paths_exhausted')
        self.assertEqual(state['deferred_path_keys'], [])
        choice = canonical_path_round_response(r, 'offline')
        self.assertIsNotNone(choice['choice'])
        validate_path_round_response(r, choice)

    def test_ambiguous_same_path_cancels_deferral_independent_of_order(self):
        for ambiguous_first in (False, True):
            for ambiguous in ('ERROR', 'late', 'unknown', 'fixture'):
                r = round_request()
                choices = list(enumerate_actions(r))[:2]
                remember(r, choices)
                row = r['path_stage_observations'][0 if ambiguous_first else 1]
                if ambiguous == 'ERROR': row['status'] = 'ERROR'
                elif ambiguous == 'late': row['deadline_exceeded'] = True
                elif ambiguous == 'unknown': row['path_success'] = None
                else: row['backend'] = 'synthetic_fixture'
                state = partition_path_round(r)
                with self.subTest(kind=ambiguous, first=ambiguous_first):
                    self.assertEqual(state['observed_failed_path_count'], 1)
                    self.assertEqual(state['ambiguous_path_count'], 1)
                    self.assertEqual(state['deferred_path_keys'], [])

    def test_pose_alias_uses_same_semantic_path_and_catalog_count(self):
        r = round_request()
        r['pose_catalog'].append(dict(pose_id='alias', matrix=deepcopy(r['pose_catalog'][0]['matrix'])))
        first = canonical_response(r, None)['choice']
        alias = deepcopy(first)
        alias['pose_id'] = 'alias'
        self.assertEqual(semantic_path_key(r, first), semantic_path_key(r, alias))
        remember(r, [first])
        self.assertEqual(partition_path_round(r)['all_path_count'], 60)

    def test_signature_excludes_support_but_keeps_part_pose_force_schedule(self):
        r = round_request()
        first = canonical_response(r, None)['choice']
        key = semantic_path_key(r, first)
        alternate = deepcopy(first)
        alternate['support_ids'] = ['2']
        self.assertEqual(key, semantic_path_key(r, alternate))
        alternate['direction_id'] = 'yn'
        self.assertNotEqual(key, semantic_path_key(r, alternate))
        changed = deepcopy(r)
        changed['task_context']['solver_settings']['seed'] = 1
        self.assertEqual(key, semantic_path_key(changed, first))
        self.assertNotEqual(partition_path_round(r)['scope_digest'],
                            partition_path_round(changed)['scope_digest'])

    def test_guard_rejects_deferred_choice_but_fallback_finds_active_one(self):
        r = round_request()
        first = canonical_response(r, None)['choice']
        remember(r, [first])
        deferred = canonical_response(r, 'offline')
        validate_response(r, deferred)
        with self.assertRaisesRegex(ValueError, 'action_deferred_for_probe_round'):
            validate_path_round_response(r, deferred)
        for fallback in ('legacy_v1', 'path_stage_v1'):
            active = canonical_path_round_response(r, 'offline', fallback)
            validate_path_round_response(r, active)
            self.assertNotEqual(semantic_path_key(r, first), semantic_path_key(r, active['choice']))
            self.assertFalse(active['metadata']['llm_live'])
            self.assertFalse(active['metadata']['physics_bypass'])
            self.assertEqual(set(active['candidate_order']), set(r['candidate_ids']))

    def test_capacity_probe_labels_all_synthetic_stage_rows_unobserved(self):
        r = round_request()
        remember(r, [canonical_response(r, None)['choice']])
        result, audit = capacity_probe(r, attempted_count=13, episode_count=12, memory_count=0)
        self.assertEqual(len(result['path_stage_observations']), 13)
        self.assertTrue(all(row['path_success'] is None and row['backend'] is None
                            for row in result['path_stage_observations']))
        self.assertFalse(audit['path_stage_observations_are_real_physics'])
        self.assertEqual(partition_path_round(result)['deferred_path_keys'], [])
        old, old_audit = capacity_probe(request(), attempted_count=13, episode_count=12, memory_count=0)
        self.assertNotIn('path_stage_observations', old)
        self.assertNotIn('synthetic_path_stage_observations', old_audit)

    def test_python38_grammar_for_all_worker_changes(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('explicit_action_contract.py', 'path_probe_round.py', 'explicit_action_planner.py',
                     'run_explicit_actions.py', 'explicit_request_budget.py'):
            ast.parse((root/'worker'/name).read_text(encoding='utf-8'), feature_version=8)


class PathProbeRoundPlannerTests(unittest.TestCase):
    @staticmethod
    def fake_actual_stage(context, choice, output):
        # Unit injection only. This label exercises admission logic, not ASAP.
        result = result_for(context, choice, False)
        result['backend'] = 'ASAP_RedMax'
        return result

    def test_worker_uses_full_current_state_history_beyond_recent_feedback(self):
        captured = []
        def choose(r):
            captured.append(deepcopy(r))
            return canonical_path_round_response(r, None)
        planner = make_planner(choose, self.fake_actual_stage,
            path_selection_policy='path_probe_round_v1', fallback_policy='path_stage_v1')
        planner.plan(budget=14, state_action_budget=14, timeout=None)
        self.assertEqual(planner.report()['total_n_eval'], 14)
        self.assertEqual(len(captured[-1]['episode_feedback']), 12)
        self.assertEqual(len(captured[-1]['path_stage_observations']), 13)
        self.assertEqual([row['action_id'] for row in captured[-1]['path_stage_observations']],
                         captured[-1]['attempted_action_ids'])
        self.assertNotIn(captured[-1]['path_stage_observations'][0]['source_event_id'],
                         captured[-1]['evidence_ids'])

    def test_worker_guard_does_not_execute_or_record_deferred_proposal_as_attempt(self):
        proposed = []
        def choose(r):
            reply = response_for(r, part='a', support=[] if not r['attempted_action_ids'] else ['b'])
            proposed.append(reply)
            return reply
        planner = make_planner(choose, self.fake_actual_stage,
            path_selection_policy='path_probe_round_v1', fallback_policy='path_stage_v1')
        planner.plan(budget=2, state_action_budget=2, timeout=None)
        report = planner.report()
        self.assertEqual(report['total_n_eval'], 2)
        second = report['decisions'][1]
        self.assertEqual(second['decision_source'], 'canonical_fallback')
        self.assertTrue(second['response']['metadata']['path_selection_deferred'])
        self.assertNotEqual(report['attempts'][1]['action_id'], proposed[1]['action_id'])
        self.assertNotIn(proposed[1]['action_id'], report['states'][0]['attempted_action_ids'])
        self.assertEqual(len(report['attempts']), 2)

    def test_new_state_starts_with_empty_stage_table(self):
        captured = []
        def choose(r):
            captured.append(deepcopy(r))
            return canonical_path_round_response(r, None)
        planner = make_planner(choose, path_selection_policy='path_probe_round_v1')
        planner.plan(budget=2, timeout=None)
        self.assertEqual(len(captured), 2)
        self.assertTrue(all(row['path_stage_observations'] == [] for row in captured))

    def test_enricher_cannot_corrupt_observations_before_decision(self):
        calls = []
        def corrupt(r):
            r['path_stage_observations'] = [dict(action_id='fake')]
            return r
        planner = make_planner(lambda r: calls.append(r), self.fake_actual_stage,
            path_selection_policy='path_probe_round_v1', request_enricher=corrupt)
        planner.plan(budget=2, timeout=None)
        self.assertEqual(calls, [])
        self.assertEqual(planner.report()['total_n_eval'], 0)
        self.assertEqual(planner.report()['status'], 'ERROR')

    def test_legacy_planner_omits_table_and_unknown_policy_rejected(self):
        captured = []
        planner = make_planner(lambda r: (captured.append(deepcopy(r)) or canonical_response(r, None)))
        planner.plan(budget=1)
        self.assertNotIn('path_stage_observations', captured[0])
        with self.assertRaisesRegex(ValueError, 'unknown_path_selection_policy'):
            make_planner(path_selection_policy='unknown')


if __name__ == '__main__':
    unittest.main()
