"""Offline capacity budget boundaries and one shared 25-call ledger."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.test_hierarchical_planner import policy as legacy_policy
from orchestrator.hierarchy_budget import HierarchyBudget, validate_policy


def capacity_policy():
    p = legacy_policy()
    p.update(schema_version=2, execution_profile='part_capacity_v1', max_calls=25, max_usd='15.00')
    for role, calls, usd in [('scout', 1, '.05'), ('planner', 24, '14.95'), ('learner', 0, '0')]:
        p['roles'][role].update(max_calls=calls, max_usd=usd)
    p['episode'].update(scout_calls=1, planner_calls=24, fallback_calls=0, max_plan_steps=8)
    return p


class CapacityBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='capacity_budget_')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'shared.sqlite'
        net = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network_forbidden'))
        self.net = net.start(); self.addCleanup(net.stop)
        self.addCleanup(self.net.assert_not_called)

    def test_legacy_six_limit_unchanged_and_new_profile_required(self):
        p = legacy_policy(); p['episode']['planner_calls'] = 7
        with self.assertRaisesRegex(ValueError, 'episode_limits'): validate_policy(p)
        for profile in (None, 'part_research_v1'):
            p = capacity_policy()
            if profile is None: p.pop('execution_profile')
            else: p['execution_profile'] = profile
            with self.subTest(profile=profile), self.assertRaises(ValueError): validate_policy(p)
        p = capacity_policy(); p['schema_version'] = 1
        with self.assertRaises(ValueError): validate_policy(p)
        self.assertEqual(validate_policy(capacity_policy())['episode']['planner_calls'], 24)

    def test_role_scope_and_global_caps_are_enforced(self):
        changes = [lambda p:p.update(max_calls=26), lambda p:p.update(max_usd='15.01'),
            lambda p:p['roles']['planner'].update(max_calls=25),
            lambda p:p['roles']['learner'].update(max_calls=1, max_usd='.01'),
            lambda p:p['episode'].update(fallback_calls=1),
            lambda p:p['episode'].update(max_plan_steps=9),
            lambda p:p['roles']['scout'].update(max_calls=2),
            lambda p:p['episode'].update(planner_calls=25),
            lambda p:p['roles']['planner'].update(max_calls=23),
            lambda p:p['roles']['learner'].update(model='gpt-6-astra', input_usd_per_million='10', output_usd_per_million='50')]
        for index, change in enumerate(changes):
            p = capacity_policy(); change(p)
            with self.subTest(index=index), self.assertRaises(ValueError): validate_policy(p)

    @staticmethod
    def body(p, role, index):
        c = p['roles'][role]
        return dict(model=c['model'], store=False, reasoning={'effort':c['reasoning_effort']},
            max_output_tokens=16, instructions='offline fixture '+str(index))

    def test_seventh_through_twenty_fourth_plans_share_one_quota(self):
        p = capacity_policy(); budget = HierarchyBudget(self.path, p, transport_failure_policy='latch_on_error_v1')
        backend = Mock(return_value={})
        for role, count in [('scout', 1), ('planner', 24)]:
            call = budget.transport(role, backend)
            for index in range(count):
                call(p['roles'][role]['base_url']+'/responses', {}, self.body(p, role, index), 1)
        self.assertEqual(backend.call_count, 25)
        report = budget.summary()
        self.assertEqual(report['roles'], {'scout':1, 'planner':24, 'learner':0})
        self.assertEqual(report['remaining_calls'], 0)
        self.assertEqual(report['live_api_attempts'], 0)
        self.assertEqual(report['automatic_http_retries'], 0)
        with self.assertRaisesRegex(ValueError, 'call_or_cost_limit'):
            budget.transport('planner', backend)(p['roles']['planner']['base_url']+'/responses', {}, self.body(p, 'planner', 25), 1)
        self.assertEqual(backend.call_count, 25)
        reopened = HierarchyBudget(self.path, p, transport_failure_policy='latch_on_error_v1')
        self.assertEqual(reopened.summary()['remaining_calls'], 0)

    def test_policy_cannot_reuse_or_reset_existing_ledger(self):
        p = capacity_policy(); HierarchyBudget(self.path, p)
        changed = deepcopy(p); changed['max_usd'] = '14.99'; changed['roles']['planner']['max_usd'] = '14.94'
        with self.assertRaisesRegex(ValueError, 'immutable_hierarchy_policy_conflict'):
            HierarchyBudget(self.path, changed)
        with self.assertRaisesRegex(ValueError, 'immutable_hierarchy_policy_conflict'):
            HierarchyBudget(self.path, legacy_policy())

    def test_timeout_latches_all_remaining_roles_without_retry(self):
        p = capacity_policy(); budget = HierarchyBudget(self.path, p, transport_failure_policy='latch_on_error_v1')
        backend = Mock(side_effect=TimeoutError('fixture timeout'))
        with self.assertRaises(TimeoutError):
            budget.transport('scout', backend)(p['roles']['scout']['base_url']+'/responses', {}, self.body(p, 'scout', 0), 1)
        with self.assertRaisesRegex(Exception, 'transport_circuit_open'):
            budget.transport('planner', backend)(p['roles']['planner']['base_url']+'/responses', {}, self.body(p, 'planner', 0), 1)
        self.assertEqual(backend.call_count, 1)
        self.assertEqual(budget.summary()['transport_attempts'], 1)
        self.assertTrue(budget.summary()['transport_guard']['blocked'])


if __name__ == '__main__':
    unittest.main()
