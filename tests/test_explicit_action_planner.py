"""DFS contract tests with fake physics; these do not establish CAD feasibility.

The graph is an in-memory fixture. ASAP's pinned source method bodies are loaded
by AST without importing simulation, Torch, or geometry dependencies on Windows.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
from explicit_action_contract import (DIRECTIONS, action_id, canonical_response,
                                      validate_response)
from explicit_action_planner import make_explicit_action_planner


IDENTITY = [[1., 0., 0., 0.], [0., 1., 0., 0.],
            [0., 0., 1., 0.], [0., 0., 0., 1.]]


class GraphFixture:
    """Minimal graph API used by the actual upstream transition/stat methods."""
    def __init__(self):
        self.nodes, self.edges = {}, {}

    def add_node(self, key, **fields):
        self.nodes[key] = fields

    def add_edge(self, parent, child, **fields):
        self.edges[(parent, child)] = fields

    def has_node(self, key):
        return key in self.nodes

    def has_edge(self, parent, child):
        return (parent, child) in self.edges

    def predecessors(self, node):
        return [a for a, b in self.edges if b == node]

    def in_degree(self, node):
        return len(self.predecessors(node))


def load_upstream_methods():
    """Compile only audited class methods; never execute upstream imports."""
    base = ast.parse((ROOT / 'external/ASAP/plan_sequence/planner/base.py').read_text())
    dfs = ast.parse((ROOT / 'external/ASAP/plan_sequence/planner/dfs.py').read_text())
    original_base = next(n for n in base.body if isinstance(n, ast.ClassDef)
                         and n.name == 'SequencePlanner')
    original_dfs = next(n for n in dfs.body if isinstance(n, ast.ClassDef)
                        and n.name == 'DFSSequencePlanner')
    names = {'_update_tree', 'find_sequence', 'check_success', 'get_stats'}
    base_class = ast.ClassDef(name='SequencePlanner', bases=[], keywords=[],
        body=[n for n in original_base.body if isinstance(n, ast.FunctionDef)
              and n.name in names], decorator_list=[])
    dfs_class = ast.ClassDef(name='DFSSequencePlanner',
        bases=[ast.Name(id='SequencePlanner', ctx=ast.Load())], keywords=[],
        body=[n for n in original_dfs.body if isinstance(n, ast.FunctionDef)
              and n.name == '_reset'], decorator_list=[])
    class ArrayFixture:
        @staticmethod
        def allclose(a, b):
            return a == b
    namespace = {'np': ArrayFixture}
    module = ast.Module(body=[base_class, dfs_class], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'actual_asap_method_fixture',
                 'exec'), namespace)
    return namespace['DFSSequencePlanner'], namespace['SequencePlanner']


DFS, Sequence = load_upstream_methods()
Planner = make_explicit_action_planner(DFS, Sequence)


def response_for(request, part=None, direction='xp', support=None, pose='p0',
                 second=None, evidence_ids=None):
    choice = dict(part_id=part or request['candidate_ids'][0], pose_id=pose,
        skill_id='single_force_v1' if second is None else 'turn_force_two_leg_v1',
        direction_id=direction, second_direction_id=second,
        support_ids=list(support or []), evidence_ids=list(evidence_ids or []))
    order = [choice['part_id']] + [p for p in request['candidate_ids']
                                   if p != choice['part_id']]
    response = dict(schema_version=1, protocol='explicit_action_v1',
        request_id=request['request_id'], state_id=request['state_id'],
        candidate_order=order, choice=choice, action_id=action_id(request, choice),
        fallback_reason=None, metadata={'selection_source': 'test_fixture',
                                         'llm_live': False})
    return validate_response(request, response)


def result_for(context, choice, feasible=True):
    return dict(status='PASS' if feasible else 'UNKNOWN', feasible=feasible,
        path_success=feasible, stability_success=True if feasible else None,
        action=DIRECTIONS[choice['direction_id']] if feasible else None,
        parts_fix=choice['support_ids'] if feasible else None,
        part_move=choice['part_id'], pose=context['pose_matrix'], grasp=None,
        fixture_only=True)


def make_planner(decision=None, verifier=None, parts=None, **kwargs):
    return Planner(parts=parts or ['a', 'b', 'c'],
        decision_fn=decision or (lambda r: canonical_response(r, 'test_fixture')),
        verifier_fn=verifier or (lambda c, a, p: result_for(c, a)),
        pose_provider=kwargs.pop('pose_provider',
            lambda parts: [dict(pose_id='p0', matrix=deepcopy(IDENTITY))]),
        run_id='test-fixture', task_context={'fixture_only': True},
        graph_factory=GraphFixture, **kwargs)


class ExplicitActionPlannerTests(unittest.TestCase):
    def test_actual_asap_subclass_and_successful_transition_methods_used(self):
        planner = make_planner()
        self.assertIsInstance(planner, DFS)
        tree = planner.plan(budget=2, timeout=None)
        self.assertTrue(planner.report()['success'])
        self.assertEqual(Sequence.get_stats(tree)['sequence'], ['a', 'b'])
        self.assertEqual(planner.report()['total_n_eval'], 2)
        self.assertEqual(planner.report()['final_pair_calls'], 1)
        self.assertFalse(planner.report()['unchanged_upstream_dfs'])
        self.assertTrue(planner.report()['selected_plan_continuity_checked'])
        self.assertEqual(planner.report()['selected_plan_continuity_errors'], [])

    def test_final_pair_uses_explicit_choice_not_mass(self):
        requests = []
        def choose(request):
            requests.append(deepcopy(request))
            return response_for(request, part='b', support=['a'])
        planner = make_planner(choose, parts=['a', 'b'])
        planner.plan(budget=1)
        report = planner.report()
        self.assertEqual(requests[0]['candidate_ids'], ['a', 'b'])
        self.assertEqual(report['stats']['sequence'], ['b'])
        self.assertEqual(report['terminal_parts'], ['a'])
        self.assertEqual(report['stats']['n_gripper'], 2)
        self.assertEqual(report['stats']['n_eval'], 1)
        self.assertFalse(hasattr(planner, 'part_mass'))

    def test_final_pair_failure_does_not_remove_other_part(self):
        requests = []
        def choose(request):
            requests.append(deepcopy(request))
            return response_for(request, part='a' if len(requests) == 1 else 'b')
        planner = make_planner(choose, lambda c, a, p:
            result_for(c, a, a['part_id'] == 'b'), parts=['a', 'b'])
        tree = planner.plan(budget=2)
        self.assertTrue(planner.report()['success'])
        self.assertEqual([r['candidate_ids'] for r in requests], [['a', 'b']] * 2)
        self.assertNotIn((('a', 'b'), ('b',)), tree.edges)

    def test_same_part_can_retry_new_direction_after_failure(self):
        requests = []
        def choose(request):
            requests.append(deepcopy(request))
            if len(request['remaining_parts']) == 3:
                return response_for(request, 'a',
                    'xp' if not request['attempted_action_ids'] else 'yp')
            return response_for(request, 'c')
        planner = make_planner(choose, lambda c, a, p:
            result_for(c, a, a['part_id'] != 'a' or a['direction_id'] == 'yp'))
        planner.plan(budget=3)
        self.assertTrue(planner.report()['success'])
        self.assertEqual([(a['choice']['part_id'], a['choice']['direction_id'])
                          for a in planner.attempts],
                         [('a', 'xp'), ('a', 'yp'), ('c', 'xp')])
        self.assertEqual(requests[1]['candidate_ids'], ['a', 'b', 'c'])
        self.assertEqual(requests[1]['episode_feedback'][0]['status'], 'UNKNOWN')
        self.assertEqual(len(requests[1]['evidence_ids']), 1)

    def test_same_part_can_change_pose_support_and_two_leg_skill(self):
        requests = []
        def choose(request):
            requests.append(deepcopy(request))
            if len(requests) == 1:
                return response_for(request, 'a', 'xp', pose='p0')
            return response_for(request, 'a', 'xp', ['b'], 'p1', 'zp')
        planner = make_planner(choose,
            lambda c, a, p: result_for(c, a, a['skill_id'] == 'turn_force_two_leg_v1'),
            parts=['a', 'b'], pose_provider=lambda parts: [
                dict(pose_id='p0', matrix=deepcopy(IDENTITY)),
                dict(pose_id='p1', matrix=[[1.,0.,0.,1.], [0.,1.,0.,0.],
                                        [0.,0.,1.,0.], [0.,0.,0.,1.]])])
        planner.plan(budget=2)
        self.assertTrue(planner.report()['success'])
        self.assertEqual(planner.attempts[1]['choice']['support_ids'], ['b'])
        self.assertEqual(planner.attempts[1]['context']['pose_matrix'][0][3], 1.)
        edge = planner.tree.edges[(('a', 'b'), ('b',))]
        self.assertFalse(edge['sim_info']['legacy_single_action_replayer_supported'])

    def test_global_budget_includes_terminal_pair(self):
        planner = make_planner()
        tree = planner.plan(budget=1)
        self.assertFalse(planner.report()['success'])
        self.assertFalse(Sequence.get_stats(tree)['success'])
        self.assertEqual(planner.report()['stop_msg'], 'budget reached')
        self.assertEqual(planner.report()['final_pair_calls'], 0)
        self.assertFalse(any(len(node) == 1 for node in tree.nodes))

    def test_zero_budget_never_calls_selector_or_physics(self):
        def prohibited(*args):
            raise AssertionError('must not execute')
        planner = make_planner(prohibited, prohibited)
        planner.plan(budget=0)
        self.assertEqual(planner.n_eval, 0)
        self.assertEqual(planner.decisions, [])

    def test_state_budget_backtracks_then_tries_other_root_part(self):
        def choose(request):
            parts = request['remaining_parts']
            if len(parts) == 3:
                part = 'a' if not request['attempted_action_ids'] else 'b'
                return response_for(request, part)
            return canonical_response(request, 'test_fixture')
        def verify(context, choice, output):
            return result_for(context, choice,
                context['remaining_parts'] != ['b', 'c'])
        planner = make_planner(choose, verify)
        planner.plan(budget=6, state_action_budget=2)
        report = planner.report()
        self.assertTrue(report['success'])
        self.assertEqual([a['choice']['part_id'] for a in report['selected_plan']],
                         ['b', 'a'])
        self.assertEqual(report['stats']['sequence'], ['b', 'a'])
        backtrack = [e for e in planner.events if e['kind'] == 'backtrack'][0]
        self.assertEqual(backtrack['remaining_parts'], ['b', 'c'])
        self.assertEqual(backtrack['verdict'], 'UNKNOWN')
        self.assertFalse(backtrack['proves_impossible'])

    def test_failed_actions_never_create_structural_edges(self):
        planner = make_planner(verifier=lambda c, a, p: result_for(c, a, False))
        tree = planner.plan(budget=3, state_action_budget=3)
        self.assertEqual(len(tree.edges), 0)
        self.assertEqual(len(tree.nodes), 1)
        self.assertEqual(len(planner.attempts), 3)

    def test_bad_reply_falls_back_without_candidate_deletion(self):
        requests = []
        def invalid(request):
            requests.append(deepcopy(request))
            response = response_for(request)
            response['candidate_order'].pop()
            return response
        planner = make_planner(invalid)
        planner.plan(budget=2)
        self.assertTrue(planner.report()['success'])
        self.assertTrue(all(a['decision_source'] == 'canonical_fallback'
                            for a in planner.attempts))
        self.assertEqual(requests[0]['candidate_ids'], ['a', 'b', 'c'])

    def test_repeated_action_reply_falls_back_to_next_untried_action(self):
        count = [0]
        def duplicate(request):
            # Deliberately return a now-stale action; do not prevalidate fixture.
            clean = deepcopy(request)
            clean['attempted_action_ids'] = []
            clean['attempted_actions'] = []
            return response_for(clean)
        def verify(context, choice, output):
            count[0] += 1
            return result_for(context, choice, count[0] == 2)
        planner = make_planner(duplicate, verify, parts=['a', 'b'])
        planner.plan(budget=2)
        self.assertTrue(planner.report()['success'])
        self.assertNotEqual(planner.attempts[0]['action_id'],
                            planner.attempts[1]['action_id'])
        self.assertEqual(planner.attempts[1]['decision_source'], 'canonical_fallback')

    def test_all_state_attempts_are_readable_after_episode_window_expires(self):
        planner = make_planner(verifier=lambda c, a, p: result_for(c, a, False))
        planner.plan(budget=14, state_action_budget=14)
        request = planner.decisions[-1]['request']
        self.assertEqual(len(request['episode_feedback']), 12)
        self.assertEqual(len(request['attempted_actions']), 13)
        self.assertEqual([row['action_id'] for row in request['attempted_actions']],
                         request['attempted_action_ids'])
        first = request['attempted_actions'][0]
        self.assertEqual(first['action_id'], planner.attempts[0]['action_id'])
        self.assertNotIn('evidence_ids', first)
        self.assertNotIn(planner.attempts[0]['event_id'], request['evidence_ids'])
        self.assertEqual(request['candidate_ids'], ['a', 'b', 'c'])
        self.assertEqual(len(planner.states[('a', 'b', 'c')]['attempted_actions']), 14)

    def test_readable_attempts_are_current_state_only_when_advancing_and_backtracking(self):
        def choose(request):
            if len(request['remaining_parts']) == 3:
                return response_for(request, 'a' if not request['attempted_actions'] else 'b')
            return canonical_response(request, 'test_fixture')
        planner = make_planner(choose, lambda c, a, p:
            result_for(c, a, c['remaining_parts'] != ['b', 'c']))
        planner.plan(budget=5, state_action_budget=2)
        root_first, child_first, child_second, root_return, other_child = [
            d['request'] for d in planner.decisions]
        self.assertEqual(root_first['attempted_actions'], [])
        self.assertEqual(child_first['attempted_actions'], [])
        self.assertEqual(len(child_second['attempted_actions']), 1)
        self.assertEqual(len(root_return['attempted_actions']), 1)
        self.assertEqual(root_return['attempted_actions'][0]['action_id'],
                         planner.attempts[0]['action_id'])
        self.assertEqual(other_child['attempted_actions'], [])
        self.assertTrue(planner.report()['success'])

    def test_readable_attempts_cannot_be_modified_by_enricher_or_selector(self):
        def bad_enricher(request):
            if request['attempted_actions']:
                request['attempted_actions'][0]['direction_id'] = 'yp'
            return request
        planner = make_planner(verifier=lambda c, a, p: result_for(c, a, False),
                               request_enricher=bad_enricher)
        planner.plan(budget=2)
        self.assertEqual(planner.stop_msg, 'exception')
        self.assertEqual(planner.n_eval, 1)
        self.assertEqual(planner.states[('a', 'b', 'c')]['attempted_actions'][0]
                         ['direction_id'], 'xp')

        def bad_selector(request):
            response = canonical_response(request, 'fixture')
            request['attempted_actions'].clear()
            return response
        planner = make_planner(bad_selector,
            lambda c, a, p: result_for(c, a, False))
        planner.plan(budget=2)
        self.assertEqual(len(planner.states[('a', 'b', 'c')]['attempted_actions']), 2)
        self.assertEqual(len(planner.decisions[1]['request']['attempted_actions']), 1)

    def test_repeated_valid_reply_feedback_is_separate_from_physics_failure(self):
        def duplicate(request):
            clean = deepcopy(request)
            clean['attempted_action_ids'] = []
            clean['attempted_actions'] = []
            return response_for(clean)
        planner = make_planner(duplicate, lambda c, a, p: result_for(c, a, False))
        planner.plan(budget=3)
        feedback = planner.decisions[2]['request']['episode_feedback'][1]
        diagnostic = feedback['selection_feedback']
        self.assertEqual(feedback['decision_source'], 'canonical_fallback')
        self.assertEqual(diagnostic['scope'], 'selection_validation_not_physics')
        self.assertEqual(diagnostic['decision_validation_error'], 'action_already_attempted')
        self.assertTrue(diagnostic['action_repeated'])
        self.assertEqual(diagnostic['rejected_action_id'], planner.attempts[0]['action_id'])
        self.assertEqual(feedback['status'], 'UNKNOWN')
        self.assertFalse(feedback['path_success'])
        self.assertNotEqual(feedback['action_id'], diagnostic['rejected_action_id'])

    def test_service_feedback_projects_fixed_fields_without_provider_prose(self):
        def provider(request):
            response = canonical_response(request, 'invalid_action_response')
            response['metadata'].update(provider_contract='explicit_action_provider_v2',
                selection_source='deterministic_fallback',
                decision_validation_error='invalid_candidate_permutation',
                rejected_action_id=None, action_repeated=False, raw_response_recorded=False,
                raw_provider_output='untrusted collision claim',
                explanation='all paths impossible')
            return response
        planner = make_planner(provider, lambda c, a, p: result_for(c, a, False))
        planner.plan(budget=2)
        feedback = planner.decisions[1]['request']['episode_feedback'][0]
        diagnostic = feedback['selection_feedback']
        self.assertEqual(diagnostic['provider_contract'], 'explicit_action_provider_v2')
        self.assertEqual(diagnostic['decision_validation_error'], 'invalid_candidate_permutation')
        self.assertFalse(diagnostic['raw_response_recorded'])
        self.assertNotIn('untrusted collision', json.dumps(feedback))
        self.assertNotIn('all paths impossible', json.dumps(feedback))

    def test_unknown_selection_metadata_is_sanitized_and_no_repeat_id_invented(self):
        def provider(request):
            response = canonical_response(request, None)
            response['metadata'].update(selection_source={'untrusted': 'provider'},
                provider_contract='untrusted', decision_validation_error='collision',
                rejected_action_id=response['action_id'], action_repeated=True,
                raw_response_recorded='text')
            return response
        planner = make_planner(provider, lambda c, a, p: result_for(c, a, False))
        planner.plan(budget=2)
        feedback = planner.decisions[1]['request']['episode_feedback'][0]
        diagnostic = feedback['selection_feedback']
        self.assertEqual(feedback['decision_source'], 'unknown')
        self.assertEqual(diagnostic['decision_validation_error'], 'invalid_response')
        self.assertIsNone(diagnostic['provider_contract'])
        self.assertIsNone(diagnostic['rejected_action_id'])
        self.assertIsNone(diagnostic['raw_response_recorded'])
        self.assertFalse(diagnostic['action_repeated'])

    def test_callback_error_cannot_fabricate_a_repeated_provider_action(self):
        def throw(request):
            raise ValueError('action_already_attempted')
        planner = make_planner(throw, lambda c, a, p: result_for(c, a, False))
        planner.plan(budget=2)
        diagnostic = planner.decisions[1]['request']['episode_feedback'][0]['selection_feedback']
        self.assertFalse(diagnostic['action_repeated'])
        self.assertIsNone(diagnostic['rejected_action_id'])
        self.assertEqual(len(planner.attempts), 2)

    def test_verifier_exception_still_appears_in_current_state_attempt_ledger(self):
        def broken(context, choice, output):
            raise RuntimeError('not a collision diagnosis')
        planner = make_planner(verifier=broken)
        planner.plan(budget=2)
        request = planner.decisions[1]['request']
        self.assertEqual(request['attempted_actions'][0]['action_id'],
                         planner.attempts[0]['action_id'])
        self.assertEqual(request['episode_feedback'][0]['status'], 'ERROR')
        self.assertIsNone(request['episode_feedback'][0]['selection_feedback']
                          ['decision_validation_error'])

    def test_fixed_base_is_preserved_through_terminal_pair(self):
        planner = make_planner(parts=['base', 'a'], base_part='base', max_grippers=1)
        planner.plan(budget=1)
        self.assertEqual(planner.report()['terminal_parts'], ['base'])
        self.assertEqual(planner.decisions[0]['request']['candidate_ids'], ['a'])
        self.assertEqual(planner.report()['stats']['n_gripper'], 1)

    def test_selector_cannot_mutate_owned_request_or_stack(self):
        def mutate(request):
            good = response_for(request)
            request['remaining_parts'].clear()
            request['pose_catalog'][0]['matrix'][0][0] = 123
            return good
        planner = make_planner(mutate, parts=['a', 'b'])
        planner.plan(budget=1)
        self.assertTrue(planner.report()['success'])
        self.assertEqual(planner.attempts[0]['remaining_parts'], ['a', 'b'])
        self.assertEqual(planner.attempts[0]['context']['pose_matrix'], IDENTITY)

    def test_verifier_cannot_mutate_owned_choice_or_context(self):
        def mutate(context, choice, output):
            good = result_for(context, choice)
            context['remaining_parts'].clear()
            choice['part_id'] = 'invented'
            return good
        planner = make_planner(verifier=mutate, parts=['a', 'b'])
        planner.plan(budget=1)
        self.assertEqual(planner.attempts[0]['choice']['part_id'], 'a')
        self.assertEqual(planner.attempts[0]['remaining_parts'], ['a', 'b'])

    def test_verifier_errors_are_counted_and_do_not_prune(self):
        calls = [0]
        def error(context, choice, output):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError('fixture exception')
            return result_for(context, choice)
        planner = make_planner(verifier=error, parts=['a', 'b'])
        planner.plan(budget=2)
        self.assertTrue(planner.report()['success'])
        self.assertEqual(planner.attempts[0]['status'], 'ERROR')
        self.assertEqual(planner.n_eval, 2)

    def test_inconsistent_pass_is_not_a_success(self):
        def inconsistent(context, choice, output):
            result = result_for(context, choice)
            result['path_success'] = False
            return result
        planner = make_planner(verifier=inconsistent, parts=['a', 'b'])
        tree = planner.plan(budget=1)
        self.assertFalse(planner.report()['success'])
        self.assertEqual(planner.attempts[0]['status'], 'ERROR')
        self.assertEqual(len(tree.edges), 0)

    def test_verifier_cannot_silently_change_selected_action_fields(self):
        wrong_pose = deepcopy(IDENTITY)
        wrong_pose[0][3] = 1.
        for field, value in [('parts_fix', ['b']), ('part_move', 'b'),
                             ('pose', wrong_pose), ('action', [0, 0, 1])]:
            with self.subTest(field=field):
                def changed(context, choice, output):
                    result = result_for(context, choice)
                    result[field] = value
                    return result
                planner = make_planner(verifier=changed, parts=['a', 'b'])
                planner.plan(budget=1)
                self.assertFalse(planner.report()['success'])
                self.assertEqual(planner.attempts[0]['status'], 'ERROR')
                self.assertEqual(planner.attempts[0]['raw']['raw_returned_result'][field],
                                 value)

    def test_deadline_expiring_during_decision_never_starts_physics(self):
        now = [0.]
        def late(request):
            now[0] = 11.
            return response_for(request)
        planner = make_planner(late, clock=lambda: now[0])
        planner.plan(budget=2, timeout=10.)
        self.assertEqual(planner.n_eval, 0)
        self.assertEqual(planner.stop_msg, 'timeout')

    def test_late_terminal_success_is_saved_but_not_accepted_within_budget(self):
        now = [0.]
        def late(context, choice, output):
            now[0] = 11.
            return result_for(context, choice)
        planner = make_planner(verifier=late, clock=lambda: now[0], parts=['a', 'b'])
        tree = planner.plan(budget=1, timeout=10.)
        self.assertFalse(planner.report()['success'])
        self.assertEqual(planner.attempts[0]['status'], 'PASS')
        self.assertTrue(planner.attempts[0]['deadline_exceeded'])
        self.assertFalse(planner.attempts[0]['accepted_for_transition'])
        self.assertEqual(len(tree.edges), 0)

    def test_memory_failures_cannot_prune_candidates(self):
        memory = [dict(evidence_id='past-failure', status='UNKNOWN',
            reference_only=True, current_part_id_mapping=None,
            hard_pruning=False, physics_reuse_allowed=False,
            source_part_id='a')]
        planner = make_planner(memory_references=memory)
        planner.plan(budget=2)
        self.assertEqual(planner.decisions[0]['request']['candidate_ids'],
                         ['a', 'b', 'c'])
        self.assertTrue(planner.report()['success'])

    def test_memory_enricher_can_add_reference_but_not_alter_scope(self):
        def enrich(request):
            request['memory_references'] = [dict(evidence_id='m1',
                reference_only=True, current_part_id_mapping=None,
                hard_pruning=False, physics_reuse_allowed=False)]
            request['evidence_ids'].append('m1')
            return request
        planner = make_planner(request_enricher=enrich, parts=['a', 'b'])
        planner.plan(budget=1)
        self.assertTrue(planner.report()['success'])
        self.assertIn('m1', planner.decisions[0]['request']['evidence_ids'])
        def malicious(request):
            request['candidate_ids'] = request['candidate_ids'][1:]
            return request
        planner = make_planner(request_enricher=malicious)
        planner.plan(budget=1)
        self.assertEqual(planner.stop_msg, 'exception')
        self.assertEqual(planner.n_eval, 0)

    def test_existing_output_is_never_overwritten_and_run_is_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'new_run'
            planner = make_planner(parts=['a', 'b'])
            planner.plan(budget=1, output_dir=target)
            report = json.loads((target / 'planner_report.json').read_text())
            self.assertTrue(report['success'])
            self.assertTrue((target / 'planner_events.jsonl').exists())
            with self.assertRaises(RuntimeError):
                planner.plan(budget=1)
            with self.assertRaises(FileExistsError):
                make_planner().plan(budget=1, output_dir=target)

    def test_forbidden_output_rejected_lexically(self):
        for path in ('E:/do-not-touch', '/mnt/e/do-not-touch',
                     '/mnt/c/../e/do-not-touch', 'relative-run'):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    make_planner().plan(budget=1, output_dir=path)

    def test_event_sink_receives_events_without_duplicate_kind_argument(self):
        calls = []
        planner = make_planner(parts=['a', 'b'],
            event_sink=lambda kind, **fields: calls.append((kind, fields)))
        planner.plan(budget=1)
        self.assertEqual(calls[0][0], 'planner_start')
        self.assertEqual(calls[-1][0], 'planner_stop')

    def test_nonfinite_timeout_rejected_before_callbacks(self):
        for timeout in (float('inf'), float('nan')):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    make_planner().plan(budget=1, timeout=timeout)

    def test_counter_ledger_distinguishes_dispatch_from_internal_checks(self):
        def counted(context, choice, output):
            result = result_for(context, choice)
            result.update(path_check_calls=1, stability_check_calls=0,
                two_leg_controller_calls=0, internal_stability_evaluations=0,
                geometry_queries=6, path_physics_steps=100)
            return result
        planner = make_planner(verifier=counted)
        planner.plan(budget=2)
        report = planner.report()
        self.assertEqual(report['total_n_eval'], 2)
        self.assertEqual(report['counters']['path_check_calls'], 2)
        self.assertEqual(report['counters']['stability_check_calls'], 0)
        self.assertEqual(report['counters']['geometry_queries'], 12)
        self.assertEqual(report['counters']['path_physics_steps'], 200)

    def test_missing_mock_counter_is_unknown_not_invented_zero(self):
        planner = make_planner(parts=['a', 'b'])
        planner.plan(budget=1)
        self.assertIsNone(planner.report()['counters']['path_check_calls'])

    def test_tampered_solution_continuity_never_reports_success(self):
        planner = make_planner()
        planner.plan(budget=2)
        planner._solution_choices[1]['remaining_parts'] = ['invented', 'c']
        report = planner.report()
        self.assertFalse(report['success'])
        self.assertEqual(report['status'], 'ERROR')
        self.assertIn('remaining_mismatch_at_1', report['selected_plan_continuity_errors'])

    def test_measured_endpoint_and_raw_stability_parts_reach_next_decision(self):
        endpoint = {'scope': 'selected_attempt_endpoint', 'min_sep': 0.01,
                    'pairs': [{'part_id': 'b', 'distance': 0.02}],
                    'all_paths_impossible': False, 'blocker': None}
        calls = [0]
        def measured(context, choice, output):
            calls[0] += 1
            result = result_for(context, choice, calls[0] > 1)
            if calls[0] == 1:
                result.update(path_success=True, stability_success=False,
                    endpoint_observation=endpoint,
                    raw_upstream={'path_success': True, 'stability_success': False,
                                  'parts_fall': ['b']},
                    unrelated_text='not a measured conclusion')
            return result
        planner = make_planner(verifier=measured, parts=['a', 'b'])
        planner.plan(budget=2)
        request = planner.decisions[1]['request']
        feedback = request['episode_feedback'][0]
        self.assertEqual(feedback['measured_endpoint'], endpoint)
        self.assertEqual(feedback['raw_stability_parts'], ['b'])
        self.assertIn('no_cause_inference', feedback['raw_stability_parts_semantics'])
        self.assertEqual(request['candidate_ids'], ['a', 'b'])
        self.assertNotIn('unrelated_text', feedback)
        feedback['measured_endpoint']['pairs'][0]['distance'] = 999.
        feedback['raw_stability_parts'].append('invented')
        self.assertEqual(planner.attempts[0]['raw']['endpoint_observation'], endpoint)
        self.assertEqual(planner.attempts[0]['raw']['raw_upstream']['parts_fall'], ['b'])

    def test_absent_measurements_remain_null_without_failure_cause_inference(self):
        planner = make_planner(verifier=lambda c, a, p: result_for(c, a, False),
                               parts=['a', 'b'])
        planner.plan(budget=2)
        feedback = planner.decisions[1]['request']['episode_feedback'][0]
        self.assertIsNone(feedback['measured_endpoint'])
        self.assertIsNone(feedback['raw_stability_parts'])
        self.assertEqual(feedback['status'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
