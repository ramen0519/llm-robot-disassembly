"""Shared active profile: CLI/IPC fixture + actual saved clock replay; API/physics0."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import traceback
import subprocess
import sys
import unittest
from unittest.mock import patch

from orchestrator.research_profile import (
    PROFILE_ID, EPISODE_SETTINGS, TASK_FIELDS, apply_episode_profile,
    episode_flags, learning_flags, validate_task)
from tools import run_hierarchical_episode as episode
from tools import run_hierarchical_cycle as cycle
from tools import run_automatic_learning_cycle as automatic

ROOT = Path(__file__).absolute().parents[1]


def task():
    return dict(assembly_dir='/opt/asap-research/ASAP/assets/beam_assembly/original',
        base_part='none', seed=0, budget=16, state_action_budget=8, max_gripper=3,
        max_pose=128, timeout=480, wall_timeout=600, response_timeout=120)


def argv_base():
    return ['--run-dir', str(ROOT/'runs/NO_CREATE_profile'), '--hierarchy-policy', 'missing.json',
        '--hierarchy-ledger', 'missing.sqlite', '--planner-ledger', 'missing2.sqlite',
        '--skill-library', 'missing3.json']


class ResearchProfileTests(unittest.TestCase):
    def test_legacy_defaults_stay_legacy(self):
        args = episode.parser().parse_args(argv_base())
        self.assertIsNone(apply_episode_profile(args, argv_base()))
        self.assertEqual(args.plan_provider_contract, 'conditional_plan_v1')
        self.assertEqual(args.knowledge_policy, 'legacy_v1')
        self.assertEqual(args.physics_scene_policy, 'fresh_v1')
        self.assertEqual(args.pose_transition_policy, 'legacy_v1')
        self.assertEqual(args.strategy_failure_policy, 'legacy_v1')

    def test_explicit_profile_has_shared_policy_and_no_paid_retrieval(self):
        argsv = argv_base() + episode_flags() + automatic.task_argv(task())
        args = episode.parser().parse_args(argsv)
        result = apply_episode_profile(args, argsv)
        self.assertEqual(result['profile_id'], PROFILE_ID)
        self.assertEqual(result['extra_retrieval_api_calls'], 0)
        self.assertFalse(result['approval_inferred'])
        for name, value in EPISODE_SETTINGS.items():
            self.assertEqual(getattr(args, name), value)
        self.assertEqual(learning_flags(), ['--observation-contract', 'source_bound_v1',
                                           '--transport-failure-policy', 'latch_on_error_v1'])

    def test_conflicts_duplicates_and_implicit_task_bounds_rejected(self):
        base = argv_base() + ['--research-profile', PROFILE_ID] + automatic.task_argv(task())
        cases = [base + ['--knowledge-policy', 'agent_tools_v1'],
                 base + ['--timeout', '600'],
                 base + ['--decision-failure-policy', 'legacy_v1']]
        # decision_failure enum uses legacy_fallback_v1; select an actual parser option.
        cases[-1] = base + ['--recovery-support-policy', 'legacy_v1']
        missing = list(base)
        index = missing.index('--max-pose'); del missing[index:index+2]
        cases.append(missing)
        for argsv in cases:
            with self.subTest(argsv=argsv), self.assertRaises(ValueError):
                apply_episode_profile(episode.parser().parse_args(argsv), argsv)

    def test_all_clock_and_budget_limits_checked_without_path_access(self):
        for key, value in [('timeout', 601), ('response_timeout', 121), ('max_pose', 129),
                ('budget', 129), ('state_action_budget', 25), ('seed', True),
                ('assembly_dir', 'E:/not-opened'), ('assembly_dir', '/mnt/e/not-opened'),
                ('assembly_dir', '/opt/asap-research/../escape'), ('wall_timeout', 480)]:
            invalid = dict(task(), **{key: value})
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('NO_IO')), \
                 self.subTest(key=key), self.assertRaises(ValueError):
                validate_task(invalid)

    def test_plan_version_opt_in_and_legacy_bounds_remain_distinct(self):
        plan = json.loads((ROOT/'configs/automatic_learning_cycle_v1.json').read_text(encoding='utf-8'))
        old = automatic.validate_plan(plan)
        self.assertEqual(old, plan)
        updated = dict(plan, schema_version=2, source=task(), target=task())
        self.assertEqual(automatic.validate_plan(updated, research_profile=PROFILE_ID), updated)
        with self.assertRaises(ValueError):
            automatic.validate_plan(updated)
        with self.assertRaises(ValueError):
            automatic.validate_plan(plan, research_profile=PROFILE_ID)
        updated['source']['assembly_dir'] = '/opt/asap-research/not-declared'
        with self.assertRaisesRegex(ValueError, 'predeclared_input_manifest'):
            automatic.validate_plan(updated, research_profile=PROFILE_ID)

    def test_live_profile_cannot_start_without_certificate_before_reading_policy(self):
        argsv = argv_base() + episode_flags() + automatic.task_argv(task()) + ['--mode', 'llm']
        with patch.object(episode, 'read', side_effect=AssertionError('NO_INPUT_READ')), \
             patch.object(episode, 'execute', side_effect=AssertionError('NO_PROCESS')):
            with self.assertRaisesRegex(ValueError, 'release_certificate_required'):
                episode.main(argsv)

    def test_cycle_profile_requires_lessons_and_certificate_before_reads(self):
        with patch.object(cycle, 'read', side_effect=AssertionError('NO_INPUT_READ')):
            with self.assertRaisesRegex(ValueError, 'requires_lesson_learning'):
                cycle.execute_cycle(ROOT/'runs/NO_CREATE_cycle', {}, 'missing.json', research_profile=PROFILE_ID)
            with self.assertRaisesRegex(ValueError, 'release_certificate_required'):
                cycle.execute_cycle(ROOT/'runs/NO_CREATE_cycle', {}, 'missing.json', research_profile=PROFILE_ID,
                                    lesson_learning=True, live=True)

    def test_actual_saved_observations_reach_learning_with_original_clock(self):
        manifest = json.loads((ROOT/'runs/multitask_roles_20261105/prepared/manifest.json').read_text(encoding='utf-8'))
        with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_API')), \
             patch('subprocess.run', side_effect=AssertionError('NO_PHYSICS')):
            packet, bindings = cycle.research_learning_packet(manifest, 'research_saved_replay')
        rows = packet['context']['python_debug_requests']
        self.assertTrue(rows)
        self.assertEqual(len(bindings), len(manifest['cases']))
        for row in rows:
            bound = bindings[row['case_id']]
            self.assertGreater(bound['original_allowance_seconds'], 400)
            feedback = row['current_observation_request']['episode_feedback'][-1]
            self.assertFalse(feedback['deadline_exceeded'])
            self.assertFalse(feedback['accepted_for_transition'])
            self.assertGreater(row['original_trigger']['elapsed_seconds'], 30)
            self.assertEqual(row['observation_binding'], bound)

    def test_new_profile_passes_existing_real_cli_graph_ipc_with_fixture_physics(self):
        from tests import test_strategy_runner as runner
        StrategyRunnerTests = runner.StrategyRunnerTests
        original = episode.main
        original_wire = runner.wire
        captured = []
        def invoke(args):
            args = list(args) + ['--research-profile', PROFILE_ID] + automatic.task_argv(task())
            captured.append(args)
            return original(args)
        def current_profile_wire(*args, **kwargs):
            result = original_wire(*args, **kwargs)
            # Explicit opt-out in this success-only integration fixture. The
            # provider contract must never silently add failover permission.
            for step in result['steps']:
                step['stability_failover'] = None
            return result
        # Reuse its entire actual runner/graph/IPC test: only worker physics and
        # provider answers are fixtures, not injected successful stage files.
        with patch.object(episode, 'main', side_effect=invoke), \
             patch.object(runner, 'wire', side_effect=current_profile_wire):
            StrategyRunnerTests('test_full_offline_runner_exports_fixture_actions_and_counts_no_live_calls').test_full_offline_runner_exports_fixture_actions_and_counts_no_live_calls()
        self.assertEqual(len(captured), 1)

    def test_failed_candidate_process_cannot_activate_claimed_admission(self):
        result = {'admission': {'admitted': True}, 'status': 'REGISTERED_SCOPED'}
        self.assertFalse(cycle.candidate_admitted(result, 1))
        self.assertTrue(cycle.candidate_admitted(result, 0))
        self.assertFalse(cycle.candidate_admitted({'admission': {'admitted': False}}, 0))

    def test_no_published_skill_or_failed_consolidation_skips_paid_target(self):
        cases = [dict(new_ids=[], learning_error=None, merge_conflict=None),
                 dict(new_ids=['claimed'], learning_error='post_validation_lesson_not_published'),
                 dict(new_ids=['claimed'], merge_conflict='new_library_provenance_conflicts_with_prior')]
        for result in cases:
            self.assertIsNotNone(cycle.target_skip_reason(result, PROFILE_ID))
            self.assertIsNone(cycle.target_skip_reason(result, None))
        self.assertIsNone(cycle.target_skip_reason(dict(new_ids=['published']), PROFILE_ID))

    def test_source_change_blocks_before_child_launch(self):
        with patch.object(cycle, 'sha', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'source_changed_before_command'):
                cycle.assert_cycle_sources({'orchestrator/research_profile.py': 'pinned'})

    def test_verifier_exception_cannot_become_paid_physical_revision(self):
        failed = dict(status='NOT_REGISTERED', cases=[dict(outcome='FAIL'), dict(outcome='ERROR')])
        self.assertEqual(cycle.candidate_infrastructure_error(failed), 'learning_case_infrastructure_error')
        failed['cases'] = [dict(outcome='FAIL')]
        self.assertIsNone(cycle.candidate_infrastructure_error(failed))
        self.assertIsNone(cycle.candidate_infrastructure_error(dict(status='GENERATION_REJECTED')))

    def test_source_citations_project_without_relaxing_physical_choice(self):
        from tools.run_skill_discovery import source_case
        from tools.python_source_budget import observation_request
        from orchestrator.skill_case import request_after_trigger
        spec = dict(case_id='real_cited_source', phase='debug', source_task='02628',
            report='runs/strategy_live_20261109_1631/episode/planner/planner_report.json',
            attempt_index=1, trigger_kind='stability_failed')
        prepared = source_case(spec)
        original = deepcopy(prepared)
        self.assertTrue(prepared['raw']['chosen']['evidence_ids'])
        request, binding = observation_request(prepared, 'source-citation-replay', 'replayed-observation')
        self.assertEqual(prepared, original)
        self.assertEqual(request['episode_feedback'][-1]['choice']['evidence_ids'], [])
        self.assertEqual(request['evidence_ids'], ['replayed-observation'])
        self.assertFalse(request['episode_feedback'][-1]['deadline_exceeded'])
        for field, value in [('direction_id', 'xn'), ('support_ids', ['2'])]:
            altered = dict(prepared['choice'], **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'choice_projection'):
                request_after_trigger(prepared['request'], prepared['raw'], altered,
                    run_id='test', event_id='test', wall_seconds=binding['original_allowance_seconds'],
                    wall_limit=binding['original_allowance_seconds'], source_choice=prepared['raw']['chosen'])
        corrupt = deepcopy(prepared['raw'])
        corrupt['scope']['selected_action']['support_ids'] = ['2']
        with self.assertRaisesRegex(ValueError, 'trigger_scope_mismatch'):
            request_after_trigger(prepared['request'], corrupt, prepared['choice'], run_id='test', event_id='test',
                wall_seconds=binding['original_allowance_seconds'], wall_limit=binding['original_allowance_seconds'],
                source_choice=prepared['raw']['chosen'])

    def test_isolated_learning_request_drops_only_validated_worker_restore_authority(self):
        from tools.run_skill_discovery import source_case
        from tools.python_source_budget import source_budget
        from orchestrator.skill_case import request_after_trigger
        from tests.test_explicit_action_contract import attach_attempts
        from worker.virtual_search_restore import bind, snapshot
        from worker.explicit_action_contract import validate_request
        spec = dict(case_id='synthetic_restore_projection', phase='debug', source_task='02628',
            report='runs/strategy_live_20261109_1631/episode/planner/planner_report.json',
            attempt_index=1, trigger_kind='stability_failed')
        prepared = source_case(spec)
        request = deepcopy(prepared['request']); chosen = deepcopy(prepared['raw']['chosen'])
        attach_attempts(request, [chosen])
        action = request['attempted_action_ids'][0]
        pose = next(p['matrix'] for p in request['pose_catalog'] if p['pose_id'] == chosen['pose_id'])
        event = 'synthetic_restore_pass_for_contract_test_NOT_physics'
        request['path_stage_observations'] = [dict(action_id=action, source_event_id=event,
            path_success=True, status='PASS', deadline_exceeded=False, backend='ASAP_RedMax')]
        child = dict(state_id='synthetic_child', remaining_parts=[p for p in request['remaining_parts'] if p != chosen['part_id']])
        edge = dict(state_id=request['state_id'], event_id=event, action_id=action, status='PASS',
            accepted_for_transition=True, deadline_exceeded=False, remaining_parts=request['remaining_parts'], choice=chosen,
            context=dict(run_id=request['run_id'], parts_rest=child['remaining_parts'], pose_matrix=pose),
            raw=dict(path_success=True, stability_success=True, backend='ASAP_RedMax'))
        request['virtual_search_restore'] = bind(request, snapshot(edge, child, 1))
        validate_request(request)
        original = deepcopy(request)
        seconds = source_budget(prepared)['seconds']
        isolated = request_after_trigger(request, prepared['raw'], prepared['choice'], run_id='isolated',
            event_id='isolated', wall_seconds=seconds, wall_limit=seconds, source_choice=chosen)
        self.assertNotIn('virtual_search_restore', isolated)
        self.assertEqual(request, original)
        corrupt = deepcopy(request); corrupt['virtual_search_restore']['node_visit_revision'] = 0
        with self.assertRaisesRegex(ValueError, 'virtual_search_restore'):
            request_after_trigger(corrupt, prepared['raw'], prepared['choice'], run_id='isolated',
                event_id='isolated', wall_seconds=seconds, wall_limit=seconds, source_choice=chosen)

    def test_real_cycle_graph_no_eligible_experience_skips_target_and_preserves_reason(self):
        from tests.test_hierarchical_planner import policy
        # Child episode evidence is deliberately a boundary fixture, NOT real
        # physics. The surrounding graph/freeze/target gating is actual code.
        with tempfile.TemporaryDirectory(prefix='research_cycle_BOUNDARY_', dir=ROOT/'runs') as folder:
            home = Path(folder)
            def put(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding='utf-8')
                return path
            template = json.loads((ROOT/'runs/multitask_roles_20261105/prepared/manifest.json').read_text(encoding='utf-8'))
            template['cases'] = [c for c in template['cases'] if c['phase'] == 'debug'] + [
                c for c in template['cases'] if c['phase'] == 'admission'][:2]
            plan = json.loads((ROOT/'configs/automatic_learning_cycle_v1.json').read_text(encoding='utf-8'))
            plan.update(schema_version=2, source=task(), target=task(), initial_library=None, initial_memory=None,
                        admission_manifest=str(put(home/'admissions.json', template)))
            fixture = put(home/'provider.json', {'label': 'unused boundary provider fixture'})
            policy_path = put(home/'policy.json', policy())
            seen = []
            def process(argv, cwd, command_dir):
                self.assertEqual(Path(argv[3]).name, 'run_hierarchical_episode.py')
                argsv = argv[4:]
                args = episode.parser().parse_args(argsv)
                effective = apply_episode_profile(args, argsv)
                self.assertEqual(effective['policy'], EPISODE_SETTINGS)
                seen.append(effective)
                destination = Path(args.run_dir)
                destination.mkdir()
                put(destination/'worker_result.json', dict(run_id='boundary_fixture_source'))
                put(destination/'audit.json', dict(status='UNKNOWN', plan_success=False,
                    validation_ok=True, actual_physics=True, evidence_label='INJECTED_BOUNDARY_ONLY'))
                put(destination/'host_preservation.json', dict(source_unchanged=True, memory_unchanged=True))
                put(destination/'hierarchy_audit.json', dict(status='UNKNOWN', library_unchanged=True,
                    lesson_library_unchanged=True, evidence_label='INJECTED_BOUNDARY_ONLY'))
                cycle.freeze_cumulative_memory([], [], destination/'action_memory_frozen.json')
                return dict(exit_code=0)
            def prepare(source, template, output):
                result = dict(status='NO_ELIGIBLE_EXPERIENCE', selected=None, manifest_path=None)
                put(Path(output)/'preparation.json', result)
                return result
            with patch.object(cycle, 'run_recorded', side_effect=process), \
                 patch.object(cycle, 'prepare_episode_learning', side_effect=prepare), \
                 patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_API')), \
                 patch('subprocess.run', side_effect=AssertionError('NO_PHYSICS')):
                result = cycle.execute_cycle(home/'cycle', plan, policy_path, source_fixture=fixture,
                    target_fixture=fixture, lesson_learning=True, research_profile=PROFILE_ID)
            self.assertEqual(len(seen), 1)
            self.assertEqual(result['status'], 'COMPLETED_WITH_TARGET_NOT_RUN')
            self.assertEqual(result['target_status'], 'NOT_RUN')
            self.assertEqual(result['target_not_run_reason'], 'no_new_published_skill')
            self.assertIsNone(result['target_plan_success'])
            self.assertEqual(result['new_skill_ids'], [])
            self.assertEqual(result['api_budget']['live_api_attempts'], 0)

    def test_cycle_routes_same_profile_and_freezes_only_successful_published_child(self):
        from tests.test_hierarchical_planner import policy
        from tools.prepare_episode_learning import prepare_episode_learning as real_prepare
        source = ROOT/'runs/strategy_live_20261109_1631/episode'
        library_source = ROOT/'runs/support_learning_live_20261109_1155/revalidation_v2'
        skill_snapshot = json.loads((library_source/'skills_frozen.json').read_text(encoding='utf-8'))
        candidate_id = skill_snapshot['entries'][0]['candidate']['candidate_id']
        lesson = json.loads((library_source/'lessons_frozen.json').read_text(encoding='utf-8'))['entries'][0]['lesson']
        for candidate_exit, consolidation_exit in ((0, 0), (1, 0), (0, 1)):
            with self.subTest(candidate_exit=candidate_exit, consolidation_exit=consolidation_exit), \
                 tempfile.TemporaryDirectory(prefix='research_cycle_REPLAY_BOUNDARY_', dir=ROOT/'runs') as folder:
                home = Path(folder)
                def put(path, value):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value), encoding='utf-8')
                    return path
                template = json.loads((ROOT/'runs/multitask_roles_20261105/prepared/manifest.json').read_text(encoding='utf-8'))
                template['cases'] = [c for c in template['cases'] if c['phase'] == 'admission'][:2]
                plan = json.loads((ROOT/'configs/automatic_learning_cycle_v1.json').read_text(encoding='utf-8'))
                plan.update(schema_version=2, source=task(), target=task(), initial_library=None,
                            initial_memory=None, admission_manifest=str(put(home/'admissions.json', template)))
                original_plan = deepcopy(plan)
                provider = put(home/'provider.json', {'label': 'boundary provider NOT actual new model'})
                policy_path = put(home/'policy.json', policy())
                commands, effective, preparations, stage_errors = [], [], [], []
                vm_calls = []
                original_run = subprocess.run
                def vm_only(command, *args, **kwargs):
                    if (type(command) is list and len(command) == 5 and command[0] == sys.executable
                            and command[1:4] == ['-I', '-S', '-B']
                            and Path(command[4]).absolute() == ROOT/'orchestrator/python_skill_child.py'):
                        vm_calls.append(True)
                        return original_run(command, *args, **kwargs)
                    raise AssertionError('NO_PHYSICS_OR_EXTERNAL_SUBPROCESS')
                graph_class = cycle.LearningCycleWorkflow
                def trace_graph(output, identity, stages):
                    traced = {}
                    for name, fn in stages.items():
                        def run(state, callback=fn, stage=name):
                            try:
                                return callback(state)
                            except Exception:
                                stage_errors.append(stage + ': ' + traceback.format_exc())
                                raise
                        traced[name] = run
                    return graph_class(output, identity, traced)
                def process(argv, cwd, command_dir):
                    tool, argsv = Path(argv[3]).name, argv[4:]
                    commands.append(tool)
                    destination = Path(argsv[argsv.index('--run-dir')+1]); destination.mkdir()
                    if tool == 'run_hierarchical_episode.py':
                        args = episode.parser().parse_args(argsv)
                        active = apply_episode_profile(args, argsv)
                        effective.append(active)
                        for name in ('worker_result.json', 'action_memory_frozen.json'):
                            (destination/name).write_bytes((source/name).read_bytes())
                        put(destination/'hierarchy_decisions.json', [])
                        put(destination/'audit.json', dict(status='UNKNOWN', plan_success=False,
                            validation_ok=True, actual_physics=True,
                            evidence_label='REPLAY_SAVED_SOURCE_NOT_NEW_PHYSICS'))
                        put(destination/'host_preservation.json', dict(source_unchanged=True, memory_unchanged=True))
                        put(destination/'hierarchy_audit.json', dict(status='UNKNOWN', library_unchanged=True,
                            lesson_library_unchanged=True, physics_bindings=[], memory_visible_requests=0,
                            evidence_label='REPLAY_BOUNDARY_NOT_NEW_MODEL'))
                        if len(effective) == 2:
                            # Target receives the newly frozen outputs, never the
                            # original empty library or raw mutable source log.
                            self.assertEqual(Path(args.skill_library), home/'cycle/library_for_next_episode.json')
                            self.assertEqual(Path(args.memory), home/'cycle/memory_for_next_episode.json')
                            self.assertEqual(len(json.loads(Path(args.skill_library).read_text())['entries']), 1)
                        return dict(exit_code=0)
                    if tool == 'run_lesson_discovery.py':
                        consolidated = '--consolidate-library' in argsv
                        lesson_path = put(destination/'lesson.json', lesson)
                        status = 'REJECTED' if consolidated and consolidation_exit else 'CANDIDATE'
                        put(destination/'result.json', dict(status=status, lesson=str(lesson_path),
                            evidence_label='MOCK_CHILD_WITH_STORED_ARTIFACT_NOT_NEW_GENERATION'))
                        return dict(exit_code=consolidation_exit if consolidated else 0)
                    self.assertEqual(tool, 'run_python_skill_cycle.py')
                    self.assertEqual(argsv[argsv.index('--observation-contract')+1], 'source_bound_v1')
                    self.assertEqual(argsv[argsv.index('--transport-failure-policy')+1], 'latch_on_error_v1')
                    put(destination/'skills_frozen.json', skill_snapshot)
                    put(destination/'result.json', dict(status='REGISTERED_SCOPED', candidate_id=candidate_id,
                        source_unchanged=True, admission={'admitted': True}, cases=[],
                        evidence_label='MOCK_CHILD_WITH_STORED_ADMISSION_NOT_NEW_VALIDATION'))
                    return dict(exit_code=candidate_exit)
                def prepare(_source, template_path, output):
                    # Use the real saved episode checker/selector, not a fake
                    # PREPARED response. The recorded source is explicitly replay.
                    result = real_prepare(source, template_path, output)
                    preparations.append(result)
                    return result
                with patch.object(cycle, 'run_recorded', side_effect=process), \
                     patch.object(cycle, 'prepare_episode_learning', side_effect=prepare), \
                     patch.object(cycle, 'LearningCycleWorkflow', side_effect=trace_graph), \
                     patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_API')), \
                     patch('subprocess.run', side_effect=vm_only):
                    try:
                        result = cycle.execute_cycle(home/'cycle', plan, policy_path, source_fixture=provider,
                            target_fixture=provider, lesson_learning=True, research_profile=PROFILE_ID)
                    except Exception:
                        self.fail('\n'.join(stage_errors))
                self.assertEqual(plan, original_plan)
                self.assertEqual(preparations[0]['new_api_calls'], 0)
                self.assertEqual(preparations[0]['new_physical_calls'], 0)
                self.assertTrue(preparations[0]['original_files_unchanged'])
                passed = candidate_exit == consolidation_exit == 0
                self.assertEqual(len(effective), 2 if passed else 1)
                if passed:
                    self.assertEqual(effective[0]['policy'], effective[1]['policy'])
                    self.assertEqual(result['new_skill_ids'], [candidate_id])
                    self.assertEqual(result['status'], 'COMPLETED')
                else:
                    self.assertEqual(result['target_status'], 'NOT_RUN')
                    self.assertEqual(result['new_skill_ids'], [])
                    self.assertIn(result['target_not_run_reason'], (
                        'failed_candidate_process_cannot_publish_admission', 'post_validation_lesson_not_published'))
                self.assertEqual(result['api_budget']['live_api_attempts'], 0)


if __name__ == '__main__':
    unittest.main()
