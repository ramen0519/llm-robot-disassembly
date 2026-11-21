"""Sequence-native contract fixtures only: no paid API, CAD or RedMax execution."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker'))
from evaluation_budget import PhysicsBudget, EvaluationLimit
from independent_features import describe_loaded
from rank_contract import RankResponse
from remote_rank_client import RemoteRankingError
import remote_rank_client
from run_sequence_native import initialize_output, positive_seconds, positive_int
from sequence_native import (PROFILE, TERMINAL_POLICY, DecisionHalted,
    SequenceNativeSession, connected_path, installed_sequence, tree_snapshot)


class MeshFixture:
    bounds = [[0., 0., 0.], [1., 2., 3.]]
    center_mass = [.5, 1., 1.5]
    volume = 6.
    is_watertight = True
    is_winding_consistent = True


def features(base='base'):
    return describe_loaded({p: {'mesh': MeshFixture(), 'transform': 'final'}
        for p in ('base', 'a', 'b', 'c')}, {'fixture.obj': 'a'*64},
        'fixture_no_physics', 'c'*40, base, {'seed': 0, 'budget': 8})


class MemoryLog:
    def __init__(self):
        self.run_id = 'sequence-fixture'
        self.events = []

    def emit(self, kind, **fields):
        self.events.append(json.loads(json.dumps(dict(run_id=self.run_id,
            event_id=self.run_id+':e'+str(len(self.events)+1), kind=kind, **fields),
            allow_nan=False)))


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


class LocalClient:
    def __init__(self, mode='reverse'):
        self.mode = mode
        self.features = {}
        self.timeout_seconds = 120.
        self.requests, self.observations, self.timeouts = [], [], []
        self.callback = None

    def __call__(self, request):
        self.requests.append(asdict(request))
        self.observations.append(deepcopy(self.features))
        self.timeouts.append(self.timeout_seconds)
        if self.callback:
            self.callback()
        if self.mode == 'error':
            raise RuntimeError('fixture-error-no-fallback')
        if self.mode == 'fallback':
            raise RemoteRankingError('host_fallback_reason:transport_error')
        if self.mode == 'timeout':
            raise RemoteRankingError('response_timeout')
        if self.mode == 'not-response':
            return {'candidate_ids': list(request.candidate_ids)}
        order = tuple(reversed(request.candidate_ids))
        if self.mode == 'missing':
            order = order[:-1]
        if self.mode == 'duplicate':
            order = (order[0],)*len(order)
        if self.mode == 'invented':
            order = ('ghost',)+order[1:]
        if self.mode == 'list':
            order = list(order)
        rid = 'old' if self.mode == 'stale-request' else request.request_id
        sid = 'old' if self.mode == 'stale-state' else request.state_id
        return RankResponse(rid, sid, order)


class DataView(dict):
    def __call__(self, data=False):
        return list(self.items()) if data else list(self)


class GraphFixture:
    """Small networkx-shaped test tree; no native planner or physics."""
    def __init__(self, root=('a', 'b', 'base', 'c')):
        self.nodes = DataView({root: {'n_gripper': 1}})
        self.edges = DataView()

    def __contains__(self, node):
        return node in self.nodes

    def successors(self, node):
        return [b for a, b in self.edges if a == node]

    def add(self, parent, child, mover, feasible=True):
        self.nodes.setdefault(parent, {'n_gripper': 1})
        self.nodes[child] = {'n_gripper': 1 if feasible else None}
        self.edges[parent, child] = {'sim_info': dict(part_move=mover, feasible=feasible)}


def native_fixtures():
    class Generator:
        def __init__(self, *args, base_part=None, **kwargs):
            self.base_part = base_part

    class Legacy(Generator):
        def generate_candidate_part(self, assembled_parts):
            raise AssertionError('legacy heuristic used')

        def heuristic_func(self, part):
            raise AssertionError('legacy heuristic used')

    class Planner:
        def __init__(self):
            self.parts = ['a', 'b', 'base', 'c']
            self.received = []
            self.updates = 0
            self.callback = None
            self.error = None
            self.raw = dict(feasible=True, action=[1, 0, 0], parts_fix=['base'], part_move='a')

        def _simulate(self, part_move, parts_rest, parts_removed, pose, max_grippers,
                      timeout=None, grasp_planner=None, optimizer='L-BFGS-B', debug=0, render=False):
            self.received.append(dict(part_move=part_move, parts_rest=parts_rest,
                parts_removed=parts_removed, pose=pose, max_grippers=max_grippers,
                timeout=timeout, grasp_planner=grasp_planner, optimizer=optimizer,
                debug=debug, render=render))
            if self.callback:
                self.callback()
            if self.error:
                raise self.error
            return self.raw

        def _update_tree(self, tree, parts_parent, parts_child, n_eval, sim_info):
            self.updates += 1
            tree.add(tuple(parts_parent), tuple(parts_child), sim_info['part_move'], sim_info['feasible'])

    return Generator, Legacy, Planner


def session(mode='reverse', seconds=60, limit=8, base='base'):
    clock, client, log = Clock(), LocalClient(mode), MemoryLog()
    guard = PhysicsBudget(24, 12, seconds, clock)
    result = SequenceNativeSession(log, client, features(base), guard, limit,
        lambda: 'unchanged-rng', {'physics_sha256': 'f'*64}, clock)
    return result, clock


class RankingContractTests(unittest.TestCase):
    def test_only_part_order_changes_and_all_candidates_are_present(self):
        s, _ = session()
        self.assertEqual(list(s.generate(['c', 'a', 'base', 'b'], 'base')), ['c', 'b', 'a'])
        req, value = s.client.requests[0], s.client.observations[0]
        self.assertEqual(req['candidate_ids'], ('a', 'b', 'c'))
        self.assertEqual(value['protocol'], PROFILE)
        self.assertEqual(value['terminal_policy'], TERMINAL_POLICY)
        self.assertEqual(value['decision_scope'], 'complete_part_permutation_only')
        self.assertEqual(value['validated_repair_skills'], [])
        self.assertFalse(any(k in value for k in ('native_order', 'heuristic_rank', 'recommended_order')))
        self.assertEqual(s.guard.counts, dict(path=0, stability=0))

    def test_without_base_every_part_is_in_the_permutation(self):
        s, _ = session(base=None)
        self.assertEqual(set(s.generate(['a', 'b', 'base', 'c'], None)), {'a', 'b', 'base', 'c'})

    def test_invalid_response_halts_instead_of_canonical_fallback(self):
        for mode in ('missing', 'duplicate', 'invented', 'stale-request', 'stale-state',
                     'list', 'not-response', 'error', 'fallback', 'timeout'):
            with self.subTest(mode=mode):
                s, _ = session(mode)
                with self.assertRaises(DecisionHalted):
                    list(s.generate(['a', 'b', 'base'], 'base'))
                self.assertEqual(len(s.rank_decisions), 0)
                self.assertIsNotNone(s.halt_reason)
                with self.assertRaises(DecisionHalted):
                    list(s.generate(['a', 'b', 'base'], 'base'))
                self.assertEqual(len(s.client.requests), 1)

    def test_solver_rng_cannot_be_changed_by_ranking(self):
        s, _ = session()
        calls = iter(('before', 'after'))
        s.rng_digest = lambda: next(calls)
        with self.assertRaisesRegex(DecisionHalted, 'ranking_mutated_solver_rng'):
            list(s.generate(['a', 'b', 'base'], 'base'))

    def test_current_master_deadline_caps_and_restores_ipc_timeout(self):
        s, clock = session(seconds=30)
        clock.value = 7
        list(s.generate(['a', 'b', 'base'], 'base'))
        self.assertEqual(s.client.timeouts, [23])
        self.assertEqual(s.client.timeout_seconds, 120)

    def test_valid_response_received_after_deadline_never_yields_a_candidate(self):
        s, clock = session(seconds=30)
        s.client.callback = lambda: setattr(clock, 'value', 31)
        with self.assertRaises(EvaluationLimit):
            list(s.generate(['a', 'b', 'base'], 'base'))
        self.assertEqual(len(s.rank_decisions), 0)
        self.assertEqual(s.client.timeout_seconds, 120)

    def test_revisited_native_states_get_new_requests_not_verified_result_cache(self):
        s, _ = session()
        for _ in range(2):
            list(s.generate(['a', 'b', 'base'], 'base'))
        self.assertEqual(s.generation_count, 2)
        self.assertNotEqual(s.client.requests[0]['request_id'], s.client.requests[1]['request_id'])
        self.assertEqual(s.client.requests[0]['state_id'], s.client.requests[1]['state_id'])

    def test_feedback_is_bounded_and_snapshot_not_a_shared_mutable_reference(self):
        s, _ = session()
        s.feedback = [{'event_id': str(i), 'accepted_for_transition': False} for i in range(20)]
        list(s.generate(['a', 'b', 'base'], 'base'))
        s.feedback[-1]['accepted_for_transition'] = True
        supplied = s.client.observations[0]['episode_feedback']
        self.assertEqual(len(supplied), 12)
        self.assertEqual(supplied[0]['event_id'], '8')
        self.assertFalse(supplied[-1]['accepted_for_transition'])


class NativeObservationTests(unittest.TestCase):
    def setUp(self):
        self.s, self.clock = session()
        self.Generator, self.Legacy, self.Planner = native_fixtures()
        self.planner = self.Planner()
        self.tree = GraphFixture()

    def installed(self):
        return installed_sequence(self.Generator, {'heur-out': self.Legacy}, self.Planner, self.s)

    def simulate(self, rest=None):
        return self.planner._simulate('a', ['b', 'base', 'c'] if rest is None else rest,
            [], [[1, 0], [0, 1]], 3, timeout=25, optimizer='L-BFGS-B')

    def update(self, raw):
        self.planner._update_tree(self.tree, self.planner.parts, ['b', 'base', 'c'], 1, raw)

    def test_original_simulate_parameters_and_return_are_unchanged(self):
        with self.installed():
            raw = self.simulate()
        self.assertIs(raw, self.planner.raw)
        self.assertEqual(self.planner.received[0]['timeout'], 25)
        self.assertEqual(self.planner.received[0]['max_grippers'], 3)
        self.assertEqual(self.planner.received[0]['pose'], [[1, 0], [0, 1]])
        self.assertEqual(self.s.attempts[0]['status'], 'PASS')
        self.assertFalse(self.s.attempts[0]['accepted_for_transition'])

    def test_acceptance_is_recorded_only_after_actual_native_tree_update(self):
        with self.installed():
            raw = self.simulate()
            self.assertFalse(self.s.feedback[-1]['accepted_for_transition'])
            self.update(raw)
        self.assertEqual(self.planner.updates, 1)
        self.assertTrue(self.s.attempts[0]['accepted_for_transition'])
        self.assertTrue(self.s.feedback[-1]['accepted_for_transition'])
        self.assertEqual(self.s.feedback[-1]['source_hashes'], {'physics_sha256': 'f'*64})
        self.assertEqual(self.s.feedback[-1]['request']['pose'], [[1, 0], [0, 1]])
        self.assertEqual(connected_path(self.tree, self.planner.parts)['max_depth'], 1)

    def test_raw_pass_late_does_not_become_a_tree_transition(self):
        self.planner.callback = lambda: setattr(self.clock, 'value', 61)
        with self.installed(), self.assertRaises(EvaluationLimit):
            self.simulate()
        self.assertEqual(self.s.attempts[0]['status'], 'PASS')
        self.assertTrue(self.s.attempts[0]['feasible'])
        self.assertTrue(self.s.attempts[0]['deadline_exceeded'])
        self.assertFalse(self.s.attempts[0]['accepted_for_transition'])
        self.assertEqual(self.planner.updates, 0)

    def test_deadline_between_simulate_and_update_blocks_native_state_change(self):
        with self.installed():
            raw = self.simulate()
            self.clock.value = 61
            with self.assertRaises(EvaluationLimit):
                self.update(raw)
        self.assertEqual(self.planner.updates, 0)
        self.assertFalse(self.s.feedback[-1]['accepted_for_transition'])
        self.assertFalse(self.s.completed_in_budget())

    def test_reporting_time_does_not_revoke_a_completed_accepted_chain(self):
        self.planner.parts = ['a', 'base']
        self.tree = GraphFixture(('a', 'base'))
        with self.installed():
            raw = self.simulate(rest=['base'])
            self.clock.value = 59
            self.planner._update_tree(self.tree, ['a', 'base'], ['base'], 1, raw)
            self.clock.value = 100  # Expensive immutable evidence serialization afterwards.
        self.assertTrue(connected_path(self.tree, self.planner.parts)['success'])
        self.assertEqual(self.s.completion_elapsed_seconds, 59)
        self.assertTrue(self.s.completed_in_budget())
        self.assertFalse(self.s.transitions[-1]['deadline_exceeded'])
        self.assertEqual(self.s.transitions[-1]['accepted_at_elapsed_seconds'], 59)

    def test_update_without_observed_simulation_cannot_mutate_tree(self):
        with self.installed(), self.assertRaisesRegex(RuntimeError, 'without_observed_simulation'):
            self.update(self.planner.raw)
        self.assertEqual(self.planner.updates, 0)

    def test_failed_path_is_unknown_and_not_invented_collision(self):
        self.planner.raw.update(feasible=False, action=None, parts_fix=None)
        with self.installed():
            raw = self.simulate()
            self.update(raw)
        event = self.s.feedback[-1]
        self.assertEqual(event['status'], 'UNKNOWN')
        self.assertEqual(event['observed_stage'], 'path_search_returned_none')
        self.assertIsNone(event['blocker'])
        self.assertFalse(event['accepted_for_transition'])
        self.assertEqual(connected_path(self.tree, self.planner.parts)['max_depth'], 0)

    def test_stability_failure_distinct_from_path_search_failure(self):
        self.planner.raw.update(feasible=False, parts_fix=None)
        with self.installed():
            self.simulate()
        self.assertEqual(self.s.feedback[-1]['observed_stage'], 'stability_returned_none')

    def test_original_exception_is_recorded_and_propagated_without_fabricated_verdict(self):
        self.planner.error = ValueError('fixture-only')
        with self.installed(), self.assertRaises(ValueError):
            self.simulate()
        self.assertEqual(self.s.feedback[-1]['status'], 'ERROR')
        self.assertIsNone(self.s.feedback[-1]['raw'])
        self.assertEqual(self.s.feedback[-1]['observed_stage'], 'exception')

    def test_terminal_rule_is_explicit_and_counts_toward_composite_budget(self):
        self.s.composite_limit = 1
        with self.installed():
            self.simulate(rest=['base'])
            with self.assertRaisesRegex(EvaluationLimit, 'composite_budget'):
                self.simulate()
        self.assertEqual(len(self.planner.received), 1)
        self.assertEqual(self.s.attempts[0]['selection_scope'], 'upstream_terminal_completion')

    def test_heuristic_tripwire_and_all_patches_restore(self):
        before = (self.Planner._simulate, self.Planner._update_tree, self.Legacy.__init__,
            self.Legacy.generate_candidate_part, self.Legacy.heuristic_func)
        with self.installed() as Ranked:
            gen = Ranked(base_part='base')
            self.assertEqual(list(gen.generate_candidate_part(['a', 'b', 'base'])), ['b', 'a'])
            with self.assertRaises(DecisionHalted):
                self.Legacy(base_part='base')
        after = (self.Planner._simulate, self.Planner._update_tree, self.Legacy.__init__,
            self.Legacy.generate_candidate_part, self.Legacy.heuristic_func)
        self.assertEqual(before, after)
        self.assertNotIn('__init__', self.Legacy.__dict__)
        self.assertEqual(self.s.tripwire_calls, 1)


class ConnectedEvidenceTests(unittest.TestCase):
    def test_separate_search_branches_are_never_added_as_one_plan(self):
        t = GraphFixture(('a', 'b', 'c'))
        t.add(('a', 'b', 'c'), ('b', 'c'), 'a')
        t.add(('a', 'b', 'c'), ('a', 'c'), 'b')
        result = connected_path(t, ['a', 'b', 'c'])
        self.assertEqual(result['max_depth'], 1)
        self.assertFalse(result['success'])
        t.add(('a', 'c'), ('c',), 'a')
        result = connected_path(t, ['a', 'b', 'c'])
        self.assertTrue(result['success'])
        self.assertEqual([e['part_move'] for e in result['complete_chain']], ['b', 'a'])

    def test_malformed_edge_is_rejected(self):
        t = GraphFixture(('a', 'b', 'c'))
        t.add(('a', 'b', 'c'), ('c',), 'a')
        with self.assertRaisesRegex(ValueError, 'invalid_native_removal_edge'):
            connected_path(t, ['a', 'b', 'c'])

    def test_no_native_tree_is_honestly_zero_progress(self):
        result = connected_path(None, ['a', 'b', 'c'])
        self.assertFalse(result['success'])
        self.assertEqual(result['max_depth'], 0)


class OutputSafetyTests(unittest.TestCase):
    def test_host_can_precreate_empty_ipc_but_existing_results_are_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(remote_rank_client, 'IPC_ROOT', Path(tmp)):
            out = Path(tmp)/'new-run'
            (out/'ipc'/'requests').mkdir(parents=True)
            (out/'ipc'/'responses').mkdir()
            self.assertEqual(initialize_output(out), out)
            (out/'worker_result.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                initialize_output(out)

    def test_stale_response_or_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(remote_rank_client, 'IPC_ROOT', Path(tmp)):
            out = Path(tmp)/'old-run'
            (out/'ipc'/'responses').mkdir(parents=True)
            (out/'ipc'/'responses'/'old.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'previous_IPC_messages'):
                initialize_output(out)

    def test_nonfinite_and_nonpositive_budgets_are_rejected(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaises(Exception):
                positive_seconds(value)
        with self.assertRaises(Exception):
            positive_int('0')


if __name__ == '__main__':
    unittest.main()
