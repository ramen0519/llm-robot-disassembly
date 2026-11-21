"""Saved-artifact and synthetic contract tests; no new API or physics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tests'))
import run_skill_discovery as runner
from orchestrator.skill_case import matches_diagnostic_phase, is_reproduced_trigger
from orchestrator.skill_validation import record_validation, check_evidence
from orchestrator.skill_program import program_hash
from orchestrator.skill_discovery import (SkillDiscoveryService, INITIAL_CONTACT_INSTRUCTIONS,
                                         SYNTHESIS_INSTRUCTIONS)
from worker.stability_diagnostics import project_stability_diagnostics
from test_skill_validation import fixture_case, save

SOURCE = ROOT / 'runs/automatic_learning_cycle_20261019/live_01/next_episode'


class InitialContactScopeTests(unittest.TestCase):
    def setUp(self):
        for target in ('urllib.request.OpenerDirector.open', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('API and physics forbidden in unit test'))
            guard.start()
            self.addCleanup(guard.stop)
        self.report = runner.read(SOURCE / 'planner/planner_report.json')
        self.raw = deepcopy(self.report['attempts'][0]['raw'])

    def test_initial_phase_is_not_dynamic_and_old_scope_stays_compatible(self):
        self.assertTrue(matches_diagnostic_phase(self.raw, 'initial_disconnected'))
        self.assertFalse(matches_diagnostic_phase(self.raw, 'dynamic_failure'))
        self.assertTrue(matches_diagnostic_phase(self.raw, None))
        self.assertTrue(is_reproduced_trigger(self.raw, 'stability_failed',
                                             expected_diagnostic_phase='initial_disconnected'))
        dynamic = self.report['attempts'][1]['raw']
        self.assertTrue(matches_diagnostic_phase(dynamic, 'dynamic_failure'))
        self.assertFalse(matches_diagnostic_phase(dynamic, 'initial_disconnected'))

    def test_false_initial_diagnostic_is_not_a_natural_trigger(self):
        for mutate in (
            lambda r: r['stability_diagnostics'].update(physics_steps=1),
            lambda r: r['stability_diagnostics'].update(instrumentation_errors=['observer_error']),
            lambda r: r['executed'].update(stability_moving_parts=[]),
            lambda r: r['stability_diagnostics'].update(returned_parts=['foreign']),
            lambda r: r['raw_upstream'].update(parts_fall=[]),
            lambda r: r['stability_diagnostics'].update(returned_parts_semantics='not_observed'),
            lambda r: r.update(status='ERROR'),
            lambda r: r.update(elapsed_seconds=31),
            lambda r: r.update(stability_check_calls=0),
        ):
            raw = deepcopy(self.raw)
            mutate(raw)
            with self.subTest(mutate=mutate):
                self.assertFalse(is_reproduced_trigger(raw, 'stability_failed',
                    expected_diagnostic_phase='initial_disconnected'))

    def test_synthesis_exposes_initial_observations_as_six_not_twelve_trials(self):
        memory = SOURCE / 'action_memory_frozen.json'
        manifest = dict(experience_report=str(SOURCE / 'planner/planner_report.json'),
            experience_task='07009', experience_memory=str(memory),
            synthesis_failure_scope='initial_disconnected',
            frozen_episode_memory=dict(path=str(memory), file_sha256=runner.sha(memory),
                content_sha256=runner.read(memory)['content_sha256']))
        packet = runner.synthesis_packet(manifest, 'unit:initial')
        scope = packet['context']['target_failure_scope']
        self.assertEqual(len(scope['evidence_ids']), 12)  # two ID views of six trials
        self.assertFalse(scope['initial_disconnected_not_in_scope'])
        selected = [x for x in packet['evidence'] if x['evidence_id'] in scope['evidence_ids']]
        self.assertEqual(sum(x['evidence_origin'] == 'episode_trace' for x in selected), 6)
        for item in selected:
            self.assertEqual(item['stability_diagnostics']['physics_steps'], 0)
            self.assertIsNone(item['failure_cause'])
        self.assertIn('MOVER', packet['context']['language_semantics']['part'])
        self.assertIn('at least ONE', packet['context']['language_semantics']['support'])
        with tempfile.TemporaryDirectory() as temporary:
            service = SkillDiscoveryService(runner.preview_config(), Path(temporary) / 'preview.sqlite')
            try:
                initial_preview = service.preview(packet)
                self.assertEqual(initial_preview['api_calls'], 0)
                self.assertEqual(initial_preview['body']['instructions'], INITIAL_CONTACT_INSTRUCTIONS)
                manifest['synthesis_failure_scope'] = 'dynamic_stability_failed'
                dynamic_preview = service.preview(runner.synthesis_packet(manifest, 'unit:dynamic'))
                self.assertEqual(dynamic_preview['body']['instructions'], SYNTHESIS_INSTRUCTIONS)
            finally:
                service.close()

    def test_wrong_fresh_phase_stops_before_program_or_second_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            template = next(d['request'] for d in self.report['decisions']
                            if d['request']['request_id'] == self.report['attempts'][0]['request_id'])
            physical = runner.read(SOURCE / 'planner/attempt_000001/request.json')
            prepared = dict(request=template, context=physical['context'], choice=physical['choice'])
            case = dict(case_id='unit_initial', expected_diagnostic_phase='initial_disconnected')
            with patch.object(runner, 'actual_action', return_value=self.report['attempts'][1]['raw']) as actual:
                with self.assertRaisesRegex(ValueError, 'fresh_trigger_diagnostic_phase_mismatch'):
                    runner.run_case(prepared, case, {}, Path(temp), 'unit')
            self.assertEqual(actual.call_count, 1)

    def test_registration_replay_enforces_declared_initial_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            case, artifacts, policy = fixture_case(temp, stability_trigger=True)
            case['expected_diagnostic_phase'] = 'initial_disconnected'
            raw_path = Path(artifacts['trigger_attempt_dir']) / 'result.json'
            raw = json.loads(raw_path.read_text(encoding='utf-8'))
            # Synthetic fixture remains explicitly marked as such. Borrow only
            # diagnostic shape, then bind observed ID to the synthetic free part.
            diagnostic = deepcopy(self.raw['stability_diagnostics'])
            diagnostic.update(initial_disconnected_parts=['b'], returned_parts=['b'])
            raw['stability_diagnostics'] = diagnostic
            raw['raw_upstream']['parts_fall'] = ['b']
            selection_path = Path(artifacts['selection_path'])
            selected = json.loads(selection_path.read_text(encoding='utf-8'))
            selected['trigger_result_sha256'] = save(raw_path, raw)
            request_path = Path(selected['program_request_path'])
            request = json.loads(request_path.read_text(encoding='utf-8'))
            request['episode_feedback'][0].update(
                stability_diagnostics=project_stability_diagnostics(diagnostic), raw_stability_parts=['b'])
            selected['program_request_sha256'] = save(request_path, request)
            save(selection_path, selected)
            record = record_validation(case, artifacts, program_hash(policy), True, True, policy)
            self.assertEqual(record['verification_class'], 'fixture_not_actual_physics')
            self.assertEqual(check_evidence(record, True), record)
            changed = deepcopy(request)
            changed['episode_feedback'][0]['raw_stability_parts'] = []
            selected['program_request_sha256'] = save(request_path, changed)
            save(selection_path, selected)
            with self.assertRaisesRegex(ValueError, 'program_diagnostic_not_bound'):
                record_validation(case, artifacts, program_hash(policy), True, True, policy)
            selected['program_request_sha256'] = save(request_path, request)
            raw['stability_diagnostics']['physics_steps'] = 1
            selected['trigger_result_sha256'] = save(raw_path, raw)
            save(selection_path, selected)
            with self.assertRaisesRegex(ValueError, 'program_diagnostic_not_bound|natural_physical_failure_trigger_required'):
                record_validation(case, artifacts, program_hash(policy), True, True, policy)


if __name__ == '__main__':
    unittest.main()
