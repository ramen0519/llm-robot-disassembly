"""New policy boundaries and manifest declarations; no network/physics."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.research_profile import apply_episode_profile, episode_flags
from tests.test_research_profile import argv_base, task
from tools import run_hierarchical_episode as episode
from tools.run_automatic_learning_cycle import task_argv
from worker.transfer_protocol import validate_manifest


class PoseSwitchWiringTests(unittest.TestCase):
    def test_old_anchor_does_not_span_a_new_forward_visit(self):
        from orchestrator.strategy_candidates import _scope
        from tests.test_accepted_pose_carry import carry, read, SAVED
        original = read(SAVED/'hierarchy_decisions.json')[4]['request']
        edge = read(SAVED/'worker_result.json')['attempts'][3]
        current, _, _, _ = carry(original, edge)
        initial = _scope(current)
        # Scope-only fixture: a production request additionally needs a newly
        # worker-bound source proof, checked by the entry/reentry tests.
        changed = deepcopy(current)
        changed['accepted_pose_carry']['activation_revision'] += 1
        self.assertNotEqual(initial, _scope(changed))
        self.assertEqual(_scope(original), _scope(deepcopy(original)))

    def test_isolated_learning_validates_then_drops_old_visit_authority(self):
        from tests.test_accepted_pose_carry import carry, read, SAVED
        from orchestrator.skill_case import request_after_trigger
        from worker.explicit_action_contract import validate_request
        original = read(SAVED/'hierarchy_decisions.json')[4]['request']
        attempts = read(SAVED/'worker_result.json')['attempts']
        projected, _, _, _ = carry(original, attempts[3])
        before = deepcopy(projected)
        raw, choice = deepcopy(attempts[4]['raw']), deepcopy(attempts[4]['choice'])
        result = request_after_trigger(projected, raw, choice,
            source_choice=choice, run_id='isolated_replay_not_physics', event_id='replayed_failure',
            wall_seconds=600, wall_limit=600)
        self.assertNotIn('accepted_pose_carry', result)
        self.assertEqual(result['pose_catalog'], projected['pose_catalog'])
        self.assertEqual(result['task_context'], projected['task_context'])
        self.assertEqual(projected, before)
        self.assertIsNotNone(validate_request(result))
        broken = deepcopy(projected)
        broken['accepted_pose_carry']['active']['source']['verdict']['stability_success'] = False
        with self.assertRaises(ValueError):
            request_after_trigger(broken, raw, choice, source_choice=choice,
                run_id='isolated_replay_not_physics', event_id='replayed_failure',
                wall_seconds=600, wall_limit=600)

    def test_new_profile_pins_both_explicit_policies(self):
        argv = argv_base() + episode_flags() + task_argv(task())
        args = episode.parser().parse_args(argv)
        effective = apply_episode_profile(args, argv)
        self.assertEqual(effective['policy']['pose_transition_policy'], 'accepted_pose_carry_v1')
        self.assertEqual(effective['policy']['strategy_failure_policy'], 'model_ordered_failover_v1')
        for flag in ('--pose-transition-policy','--strategy-failure-policy'):
            conflicting = argv_base() + ['--research-profile','part_research_v1'] + task_argv(task()) + [flag,'legacy_v1']
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                apply_episode_profile(episode.parser().parse_args(conflicting), conflicting)

    def test_failover_cannot_attach_to_legacy_conditional_contract(self):
        argv = argv_base() + ['--strategy-failure-policy','model_ordered_failover_v1']
        with patch.object(episode, 'read', side_effect=AssertionError('NO_INPUT_OR_KEY')), \
             patch.object(episode, 'execute', side_effect=AssertionError('NO_EXECUTION')):
            with self.assertRaisesRegex(ValueError, 'failover_requires_strategy'):
                episode.main(argv)

    def test_manifest_declares_new_policies_without_reinterpreting_old_runs(self):
        args = SimpleNamespace(**task(), physics_scene_policy='fresh_v1',
            pose_transition_policy='legacy_v1', strategy_failure_policy='legacy_v1')
        manifest = dict(schema='unseen_transfer_v1', automatic_retries=0,
            assembly_dirs={args.assembly_dir:{}}, development_base_part=None,
            path_limit=16, stability_limit=16,
            settings={k:getattr(args,k) for k in ('budget','state_action_budget','max_gripper',
                'max_pose','timeout','wall_timeout','response_timeout')})
        with tempfile.TemporaryDirectory(prefix='manifest_policy_') as folder:
            path = Path(folder)/'manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(validate_manifest(path, args.assembly_dir, args), manifest)
            for field, policy in [('pose_transition_policy','accepted_pose_carry_v1'),
                                  ('strategy_failure_policy','model_ordered_failover_v1')]:
                trial_args = deepcopy(args)
                setattr(trial_args, field, policy)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field+'_mismatch'):
                    validate_manifest(path,args.assembly_dir,trial_args)
                declared = deepcopy(manifest); declared['settings'][field] = policy
                path.write_text(json.dumps(declared), encoding='utf-8')
                self.assertEqual(validate_manifest(path,args.assembly_dir,trial_args),declared)
                with self.assertRaisesRegex(ValueError, field+'_mismatch'):
                    validate_manifest(path,args.assembly_dir,args)
                path.write_text(json.dumps(manifest), encoding='utf-8')


if __name__=='__main__': unittest.main()
