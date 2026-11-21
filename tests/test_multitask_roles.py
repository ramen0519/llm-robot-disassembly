"""Offline input/contract tests. Handwritten probes are not model or physics success."""
from copy import deepcopy
from pathlib import Path
from decimal import Decimal
import json
import tempfile
import unittest
from unittest.mock import patch
from tools.run_luna_experiment import read, write, sha
from tools import prepare_multitask_roles as prep
from tools.multitask_skill_evidence import build_packet
from tools.skill_nonapplicability import bind_cases, audit
from orchestrator.lesson_runtime import extension_for
from orchestrator.python_skill_program import build_context
from worker.task_memory import digest

ROOT = prep.ROOT
READY = ROOT / 'runs/multitask_roles_20261105/prepared'
OLD = ROOT / 'runs/multitask_learning_live_20261105/live'


class MultiTaskRoleTests(unittest.TestCase):
    def setUp(self):
        self.manifest = read(READY / 'manifest.json')
        self.rows = read(READY / 'healthy_cases_LOCAL_ONLY.json')
        self.collection = read(OLD / 'collection.json')
        self.split = read(Path(self.manifest['split']['path']))
        tmp = tempfile.TemporaryDirectory(prefix='roles_OFFLINE_', dir=ROOT / 'runs')
        self.addCleanup(tmp.cleanup); self.temp = Path(tmp.name)

    def test_actual_failure_roles_and_complete_traces(self):
        selection = prep.select_roles(self.collection, self.split)
        self.assertEqual(selection['family'], 'dynamic_stability_failed')
        self.assertEqual(selection['generation_tasks'], ['02628', '01289'])
        self.assertEqual(selection['nonapplicability_tasks'], ['07009'])
        packet = build_packet(self.manifest, 'roles-test')
        self.assertEqual(len(packet['evidence']), 36)
        self.assertEqual({f['source_task'] for f in packet['evidence']}, {'02628', '01289'})
        self.assertEqual(sum(f.get('evidence_origin') == 'episode_trace' for f in packet['evidence']), 18)
        self.assertIsNone(packet['context']['multi_task_learning']['admission_task'])
        self.assertFalse(packet['context']['multi_task_learning']['evaluation_tasks_exposed'])

    def test_v2_without_healthy_cases_or_explicit_amendment_rejected(self):
        for key in ('nonapplicability_cases', 'role_assignment', 'require_shared_failure_family'):
            value = deepcopy(self.manifest); value.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError): build_packet(value, 'bad')

    def test_evaluation_cannot_be_used_as_generation_or_negative(self):
        for role in ('generation_tasks', 'nonapplicability_tasks'):
            value = deepcopy(self.manifest); value['role_assignment'][role][0] = '02680'
            with self.subTest(role=role), self.assertRaises(ValueError): build_packet(value, 'bad')
        value = deepcopy(self.manifest); value['nonapplicability_cases'][0]['task_id'] = '02656'
        with self.assertRaises(ValueError): bind_cases(value)

    def test_rotated_conditions_predeclared_and_no_trigger_overrides(self):
        for key, val in [('pose_yaw_degrees', 0), ('source_task', '07009'), ('attempt_index', 99)]:
            value = deepcopy(self.manifest); value['cases'][2][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError): build_packet(value, 'bad')
        value = deepcopy(self.manifest); value['cases'][0]['trigger_override'] = {'direction_id': 'zp'}
        with self.assertRaises(ValueError): build_packet(value, 'bad')

    def test_debug_source_cannot_be_changed_independently(self):
        value = deepcopy(self.manifest)
        value['cases'][0]['attempt_index'] += 1
        value['cases'][2]['attempt_index'] += 1
        with self.assertRaisesRegex(ValueError, 'generation_debug_source_binding'): build_packet(value, 'bad')

    def test_real_healthy_requests_preserve_post_removal_state(self):
        rows = bind_cases(self.manifest)
        self.assertEqual(rows, self.rows)
        for row in rows:
            request = row['request']; context, _ = build_context(request, extension_for())
            self.assertEqual(len(request['remaining_parts']), 4)
            self.assertEqual(len(request['episode_feedback'][0]['remaining_parts']), 5)
            self.assertTrue(request['episode_feedback'][0]['accepted_for_transition'])
            self.assertIsNone(context['failure']); self.assertIsNone(context['last'])
        self.assertEqual({r['task_id'] for r in rows}, {'07009'})

    def test_healthy_report_request_and_manifest_tampering_rejected(self):
        for key in ('report_sha256', 'request_sha256', 'manifest_sha256'):
            value = deepcopy(self.manifest); value['nonapplicability_cases'][0][key] = '0' * 64
            with self.subTest(key=key), self.assertRaises(ValueError): bind_cases(value)

    def test_duplicate_healthy_observation_cannot_count_twice(self):
        value = deepcopy(self.manifest)
        value['nonapplicability_cases'][1] = dict(value['nonapplicability_cases'][0], case_id='duplicate')
        with self.assertRaisesRegex(ValueError, 'duplicate_healthy_observation'): bind_cases(value)

    def test_failed_or_late_raw_observation_cannot_be_healthy(self):
        import tools.run_skill_discovery as source
        original = source.source_case
        def changed(spec):
            value = deepcopy(original(spec)); value['raw']['status'] = 'UNKNOWN'; return value
        with patch.object(source, 'source_case', side_effect=changed), self.assertRaisesRegex(ValueError, 'healthy_PASS'):
            bind_cases(self.manifest)

    def test_healthy_source_with_missing_memory_rejected(self):
        value = deepcopy(self.manifest); ref = value['nonapplicability_cases'][0]
        child = read(Path(ref['manifest_path'])); child.pop('frozen_episode_memory')
        path = self.temp / 'unbound.json'; write(path, child)
        ref.update(manifest_path=str(path), manifest_sha256=sha(path))
        with self.assertRaises(ValueError): bind_cases(value)

    def test_declining_is_accepted_but_typo_timeout_or_unneeded_action_is_not(self):
        none = dict(language='python_policy_v1', source='def propose(ctx):\n    return None\n')
        self.assertTrue(audit(none, self.rows)['accepted'])
        invalid = dict(none, source='def propose(ctx):\n    return ctx["does_not_exist"]\n')
        self.assertFalse(audit(invalid, self.rows)['accepted'])
        active = dict(none, source='def propose(ctx):\n    return {"part_id": ctx["candidate_ids"][0], '
            '"pose_id": ctx["pose_catalog"][0]["pose_id"], "skill_id": "single_force_v1", '
            '"direction_id": "xp", "second_direction_id": None, "support_ids": [], "evidence_ids": []}\n')
        value = audit(active, self.rows)
        self.assertFalse(value['accepted'])
        self.assertIsNotNone(value['checks'][0]['runtime']['response'])
        with patch('tools.skill_nonapplicability.step_program', return_value=dict(
                response=None, not_applicable_reason='python_wall_timeout')):
            self.assertFalse(audit(none, self.rows)['accepted'])

    def test_request_rebinding_rejects_runtime_changes(self):
        rows = deepcopy(self.rows); rows[0]['request']['request_id'] += '_changed'
        with self.assertRaisesRegex(ValueError, 'binding_changed'):
            audit(dict(language='python_policy_v1', source='def propose(ctx):\n    return None\n'), rows)

    def test_actual_preview_keeps_all_source_evidence_and_hides_negative_and_admission(self):
        saved = read(READY.parent / 'program_preview.json')
        body = json.loads(saved['body']['input'][0]['content'])
        self.assertEqual(len(body['evidence']), 36)
        self.assertLess(saved['body_bytes'], 196608)
        self.assertEqual(len(body['context']['runtime_contract']['examples']), 2)
        self.assertFalse(body['context']['admission_inputs_supplied'])
        text = json.dumps(body)
        self.assertNotIn('source_02628_yaw90', text)
        self.assertNotIn('healthy_0', text)
        self.assertNotIn('02680', body['context']['multi_task_learning']['source_tasks'])

    def test_residual_budget_preserves_unknown_reservation_without_unlock(self):
        ledger = OLD / 'hierarchy.sqlite'; before = sha(ledger)
        value = prep.residual_proposal(ledger)
        self.assertEqual(value['parent']['prior_calls'], 11)
        self.assertEqual(value['parent']['pending_or_unknown'], 1)
        self.assertEqual(value['parent']['policy']['max_calls'], 23)
        self.assertEqual(Decimal(value['parent']['policy']['max_usd']), Decimal('11.2188939'))
        self.assertFalse(value['policy']['enabled']); self.assertFalse(value['old_transport_latch_reset'])
        self.assertEqual(value['policy']['max_calls'], 16)
        self.assertEqual(value['maximum_reservation'], '4.7562752')
        self.assertEqual(sha(ledger), before)

    def test_insufficient_role_budget_rejected(self):
        from tools.remaining_split_budget import remaining_proposal
        value = remaining_proposal(OLD / 'hierarchy.sqlite')
        value['policy']['roles']['learner']['max_calls'] = 3
        with patch('tools.remaining_split_budget.remaining_proposal', return_value=value), self.assertRaises(ValueError):
            prep.residual_proposal(OLD / 'hierarchy.sqlite')

    def test_excluded_drive_blocked_before_access(self):
        value = deepcopy(self.manifest); value['nonapplicability_cases'][0]['manifest_path'] = 'E:/excluded.json'
        with self.assertRaises(ValueError): bind_cases(value)

    def test_entry_point_negative_gate_blocks_physics_and_registration(self):
        import tools.run_python_skill_cycle as cycle
        out = self.temp / 'candidate'
        with patch('tools.skill_nonapplicability.audit', return_value=dict(accepted=False,
                api_calls=0, physical_calls=0)), patch.object(cycle, 'run_case',
                side_effect=AssertionError('NO_PHYSICS')) as physics, patch.object(
                cycle.SkillRegistry, 'propose', side_effect=AssertionError('NO_REGISTRY')) as registry:
            code = cycle.main(['--mode', 'fixture', '--run-dir', str(out), '--manifest', str(READY / 'manifest.json'),
                               '--lesson', str(READY.parent / 'fixture_lesson_NOT_MODEL.json')])
        self.assertEqual(code, 1); physics.assert_not_called(); registry.assert_not_called()
        self.assertEqual(read(out / 'result.json')['status'], 'NONAPPLICABILITY_REJECTED')
        self.assertTrue((out / 'candidate.py').is_file())


if __name__ == '__main__': unittest.main()
