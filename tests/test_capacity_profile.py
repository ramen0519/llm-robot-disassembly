"""Explicit long-episode bounds and CLI/IPC fixtures; no paid or real physics."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from orchestrator import capacity_profile as capacity, research_profile as research
from tools import run_hierarchical_episode as episode, run_automatic_learning_cycle as automatic
from tests.test_research_profile import argv_base, task as short_task
from tests.test_hierarchical_planner import policy as old_policy

ROOT = Path(__file__).absolute().parents[1]


def task():
    return dict(short_task(), budget=256, state_action_budget=48,
                timeout=3600, wall_timeout=3900, response_timeout=120)


def policy(enabled=False):
    value = old_policy(enabled)
    value.update(schema_version=2, execution_profile=capacity.PROFILE_ID, max_calls=25, max_usd='15')
    value['episode'].update(scout_calls=1, planner_calls=24, fallback_calls=0,
                            max_plan_steps=8, recovery_per_step=1)
    for role, calls, dollars in [('scout', 1, '.05'), ('planner', 24, '14.95'), ('learner', 0, '0')]:
        value['roles'][role].update(max_calls=calls, max_usd=dollars)
    return value


class CapacityProfileTests(unittest.TestCase):
    def setUp(self):
        block = patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK'))
        network = block.start(); self.addCleanup(block.stop)
        self.addCleanup(network.assert_not_called)

    def test_capacity_requires_explicit_profile_and_reports_no_learning_authority(self):
        argsv = argv_base() + capacity.episode_flags() + automatic.task_argv(task())
        args = episode.parser().parse_args(argsv)
        value = capacity.apply_episode_profile(args, argsv)
        self.assertEqual(value['profile_id'], capacity.PROFILE_ID)
        self.assertEqual(value['task'], task())
        self.assertFalse(value['automatic_learning_enabled'])
        self.assertIsNone(value['learning_observation_contract'])
        self.assertEqual(value['learning_eligibility'], 'NOT_VALIDATED_FOR_AUTOMATIC_LEARNING')
        self.assertFalse(value['approval_inferred'])
        self.assertFalse(value['new_skill_generation'])
        self.assertEqual(value['extra_retrieval_api_calls'], 0)
        self.assertEqual(value['automatic_http_retries'], 0)
        self.assertEqual(value['policy'], research.EPISODE_SETTINGS)

    def test_normal_and_legacy_profiles_keep_old_caps_and_policies(self):
        self.assertEqual(research.LIMITS['timeout'], 600)
        self.assertEqual(research.LIMITS['budget'], 128)
        self.assertEqual(research.LIMITS['state_action_budget'], 24)
        with self.assertRaisesRegex(ValueError, 'research_task_bound_exceeded'):
            research.validate_task(task())
        args = episode.parser().parse_args(argv_base())
        self.assertIsNone(research.apply_episode_profile(args, argv_base()))
        self.assertEqual(args.physics_scene_policy, 'fresh_v1')
        with self.assertRaisesRegex(ValueError, 'explicit_capacity_profile_required'):
            capacity.apply_episode_profile(args, argv_base())

    def test_all_extended_bounds_are_checked_without_filesystem_access(self):
        for name, value in [('budget',257),('state_action_budget',49),('max_gripper',4),
                            ('max_pose',129),('timeout',3601),('wall_timeout',3901),
                            ('response_timeout',121),('seed',True),('seed',2**32),
                            ('assembly_dir','E:/not-opened'),('assembly_dir','/mnt/e/not-opened'),
                            ('assembly_dir','/opt/asap-research/../escape'),('base_part','../base')]:
            with self.subTest(name=name), patch.object(Path,'read_bytes',side_effect=AssertionError('NO_IO')):
                with self.assertRaises(ValueError):
                    capacity.validate_task(dict(task(), **{name:value}))

    def test_task_time_relationship_and_explicit_integer_contract(self):
        for changes in [dict(timeout=3600,wall_timeout=3600),dict(timeout=60,response_timeout=120),
                        dict(budget=30,state_action_budget=48),dict(budget=0),dict(timeout=3600.0),
                        dict(evaluation_manifest=33)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                capacity.validate_task(dict(task(),**changes))

    def test_conflicting_duplicate_and_missing_flags_rejected(self):
        base = argv_base()+['--research-profile',capacity.PROFILE_ID]+automatic.task_argv(task())
        missing = list(base); i=missing.index('--max-pose'); del missing[i:i+2]
        for argsv in [base+['--timeout','3600'],base+['--knowledge-policy','agent_tools_v1'],
                      base+['--physics-scene-policy','fresh_v1'],missing]:
            with self.subTest(argsv=argsv), self.assertRaises(ValueError):
                capacity.apply_episode_profile(episode.parser().parse_args(argsv),argsv)

    def test_capacity_policy_cannot_cross_to_short_or_legacy_profile(self):
        for profile_id in [None,research.PROFILE_ID]:
            with self.subTest(profile=profile_id), self.assertRaisesRegex(ValueError,'requires_explicit_capacity_profile'):
                capacity.validate_policy_binding(policy(),profile_id)
            self.assertIsNone(capacity.validate_policy_binding(old_policy(),profile_id))
        with self.assertRaisesRegex(ValueError,'requires_bound_capacity_policy'):
            capacity.validate_policy_binding(old_policy(),capacity.PROFILE_ID)
        self.assertIsNone(capacity.validate_policy_binding(policy(),capacity.PROFILE_ID))

    def test_policy_binding_preserves_episode_and_paid_scope(self):
        mutations=[lambda p:p.update(max_calls=26),lambda p:p.update(max_usd='15.01'),
                   lambda p:p['episode'].update(planner_calls=25),
                   lambda p:p['episode'].update(scout_calls=2),
                   lambda p:p['episode'].update(fallback_calls=1),
                   lambda p:p['episode'].update(max_plan_steps=9),
                   lambda p:p['episode'].update(recovery_per_step=2),
                   lambda p:p['roles']['learner'].update(max_calls=1),
                   lambda p:p['roles']['planner'].update(max_calls=25)]
        for mutate in mutations:
            value=policy();mutate(value)
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'policy_scope_exceeded'):
                capacity.validate_policy_binding(value,capacity.PROFILE_ID)

    def test_mismatched_runner_rejects_before_ledger_key_output_or_execution(self):
        for argsv in [argv_base(),argv_base()+research.episode_flags()+automatic.task_argv(short_task())]:
            with self.subTest(argsv=argsv), patch.object(episode,'read',return_value=policy()) as read, \
                 patch.object(episode,'HierarchyBudget',side_effect=AssertionError('NO_LEDGER')), \
                 patch.object(episode,'execute',side_effect=AssertionError('NO_PROCESS')), \
                 patch.object(episode,'write',side_effect=AssertionError('NO_OUTPUT')):
                with self.assertRaisesRegex(ValueError,'requires_explicit_capacity_profile'):
                    episode.main(argsv)
                self.assertEqual(read.call_count,1)

    def test_frozen_python_runtime_accepts_long_task_without_learning_contract(self):
        from orchestrator.lesson_runtime import extension_for
        from orchestrator.python_skill_program import build_context, evaluate
        # Initial state has no actions bound to the old timeout/attempt scope.
        # Historical action IDs must never be relabelled as a new physics run.
        original=json.loads((ROOT/'runs/support_free_reuse_20261109_1250/live/episode/hierarchy_decisions.json').read_text(encoding='utf-8'))[0]['request']
        self.assertEqual(original.get('attempted_action_ids'), [])
        snapshot=json.loads((ROOT/'runs/support_learning_live_20261109_1155/revalidation_v2/skills_frozen.json').read_text(encoding='utf-8'))
        program=snapshot['entries'][0]['candidate']['program']
        long=deepcopy(original)
        long['task_context']['task']['solver_settings'].update(timeout=3600,budget=256,state_action_budget=48)
        extension=extension_for()
        before,_=build_context(original,extension)
        after,_=build_context(long,extension)
        self.assertEqual(before,after)
        self.assertEqual(evaluate(program,original,extension=extension),
                         evaluate(program,long,extension=extension))

    def test_long_profile_cannot_enter_existing_automatic_learning_contract(self):
        with self.assertRaisesRegex(ValueError,'unknown_research_profile'):
            research.validate_profile(capacity.PROFILE_ID)
        with self.assertRaisesRegex(ValueError,'unknown_research_profile'):
            automatic.validate_plan({},research_profile=capacity.PROFILE_ID)

    def test_live_capacity_requires_certificate_before_policy_key_or_process(self):
        argsv=argv_base()+capacity.episode_flags()+automatic.task_argv(task())+['--mode','llm']
        with patch.object(episode,'read',side_effect=AssertionError('NO_READ')),\
             patch.object(episode,'execute',side_effect=AssertionError('NO_PROCESS')):
            with self.assertRaisesRegex(ValueError,'release_certificate_required'):
                episode.main(argsv)

    def test_certificate_check_is_bound_to_capacity_before_policy_read(self):
        argsv=argv_base()+capacity.episode_flags()+automatic.task_argv(task())+[
            '--mode','llm','--release-certificate','missing_certificate.json']
        with patch('orchestrator.research_release.validate_release_certificate',side_effect=ValueError('certificate_fixture_stop')) as validate,\
             patch.object(episode,'read',side_effect=AssertionError('NO_READ')):
            with self.assertRaisesRegex(ValueError,'certificate_fixture_stop'):
                episode.main(argsv)
        self.assertEqual(validate.call_args.kwargs['profile_id'],capacity.PROFILE_ID)

    def test_injected_provider_reads_shared_strategy_body_without_network(self):
        body=json.loads((ROOT/'runs/fifteen_request_fix_20261110/fixed_last_provider_body.json').read_text(encoding='utf-8'))
        envelope=json.loads((ROOT/'runs/fifteen_request_fix_20261110/reconstructed_last_envelope.json').read_text(encoding='utf-8'))
        fixture=dict(scout=[],planner=[dict(steps=[])])
        provider=episode.InjectedProvider(fixture)
        result=provider.transport('planner')('not-used',{},body,120)
        value=json.loads(result['output'][0]['content'][0]['text'])
        self.assertEqual(value['request_id'],envelope['request']['request_id'])
        self.assertEqual(value['state_id'],envelope['request']['state_id'])
        self.assertEqual(set(value['candidate_order']),set(envelope['request']['candidate_ids']))
        self.assertEqual(provider.counts['planner'],1)

    def test_capacity_profile_passes_existing_cli_graph_ipc_fixture(self):
        from tests import test_strategy_runner as runner
        original_main, original_wire = episode.main, runner.wire
        captured=[]
        def invoke(args):
            args=list(args)+['--research-profile',capacity.PROFILE_ID]+automatic.task_argv(task())
            result=original_main(args)
            output=Path(args[args.index('--run-dir')+1])
            audit=json.loads((output/'hierarchy_audit.json').read_text(encoding='utf-8'))
            captured.append(audit['research_profile'])
            return result
        def wire(*args,**kwargs):
            result=original_wire(*args,**kwargs)
            for step in result['steps']:step['stability_failover']=None
            return result
        with patch.object(episode,'main',side_effect=invoke),patch.object(runner,'wire',side_effect=wire),\
             patch.object(runner,'policy',side_effect=policy):
            runner.StrategyRunnerTests('test_full_offline_runner_exports_fixture_actions_and_counts_no_live_calls').test_full_offline_runner_exports_fixture_actions_and_counts_no_live_calls()
        self.assertEqual(len(captured),1)
        self.assertEqual(captured[0]['task'],task())
        self.assertEqual(captured[0]['profile_id'],capacity.PROFILE_ID)
        self.assertFalse(captured[0]['automatic_learning_enabled'])


if __name__ == '__main__':
    unittest.main()
