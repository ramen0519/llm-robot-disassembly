"""Offline regression certificate for the opt-in supported research profile.

This is a reproducibility gate, not API authorization or a claim of physical
success. Any source, test, or pinned replay input change invalidates the gate.
Legacy launch paths retain their previous behavior.
"""
import hashlib
import json
from pathlib import Path
import stat
import unittest

ROOT = Path(__file__).absolute().parents[1]
PROFILE_ID = 'part_research_v1'
COVERED_PROFILES = (PROFILE_ID, 'part_capacity_v1')
SCHEMA = 'offline_research_release/v1'
SOURCE_DIRS = ('orchestrator', 'worker', 'tools', 'tests')

# Explicit regression boundaries: do not replace this with a passing test count.
TEST_MODULES = (
    'tests.test_research_release', 'tests.test_research_profile', 'tests.test_research_saved_payloads',
    'tests.test_strategy_pose_preflight', 'tests.test_research_knowledge_feedback',
    'tests.test_strategy_plan', 'tests.test_strategy_candidates',
    'tests.test_strategy_skill_review', 'tests.test_strategy_skill_review_wiring',
    'tests.test_strategy_service', 'tests.test_strategy_runner',
    'tests.test_strategy_evidence', 'tests.test_strategy_followup',
    'tests.test_strategy_halt', 'tests.test_strategy_saved_skill',
    'tests.test_virtual_search_restore', 'tests.test_virtual_search_restore_integration',
    'tests.test_accepted_pose_carry', 'tests.test_strategy_failover', 'tests.test_pose_switch_wiring',
    'tests.test_pose_switch_integration',
    'tests.test_strategy_support_preflight', 'tests.test_strategy_support_replan',
    'tests.test_support_preflight_saved_replay',
    'tests.test_provider_request_errors', 'tests.test_strategy_payload_projection',
    'tests.test_strategy_payload_saved_replay', 'tests.test_strategy_reconsideration',
    'tests.test_capacity_budget', 'tests.test_capacity_profile',
    'tests.test_observation_contract', 'tests.test_recorded_trigger_clock',
    'tests.test_source_case_budget', 'tests.test_knowledge_context',
    'tests.test_knowledge_operation_contract', 'tests.test_runtime_skill_feedback',
    'tests.test_learning_cycle_workflow', 'tests.test_automatic_learning_cycle',
    'tests.test_episode_learning', 'tests.test_skill_case', 'tests.test_skill_validation',
    'tests.test_skill_registry', 'tests.test_skill_library_merge',
    'tests.test_skill_evidence_contract_integration', 'tests.test_lesson_pipeline', 'tests.test_lesson_release',
    'tests.test_hierarchical_planner', 'tests.test_transport_diagnostics', 'tests.test_llm_ranker',
)
REPLAY_INPUTS = (
    'runs/fifteen_request_fix_20261110/fixed_last_provider_body.json',
    'runs/fifteen_request_fix_20261110/reconstructed_last_envelope.json',
    'runs/fifteen_request_fix_20261110/saved_successful_envelopes.json',
    'runs/fifteen_part_prepare_20261110/episode/hierarchy_decisions.json',
    'runs/fifteen_part_prepare_20261110/episode/hierarchy_plans.json',
    'runs/fifteen_part_prepare_20261110/episode/hierarchy_events.json',
    'runs/fifteen_part_prepare_20261110/episode/knowledge_contexts.json',
    'runs/repeat_00161_prepare_20261110/trial_3/episode/hierarchy_decisions.json',
    'runs/repeat_00161_prepare_20261110/trial_3/episode/hierarchy_plans.json',
    'runs/repeat_00161_prepare_20261110/trial_3/episode/worker_result.json',
    'runs/part_research_00161_prepare_20261109/episode/hierarchy_decisions.json',
    'runs/part_research_00161_prepare_20261109/episode/hierarchy_plans.json',
    'runs/part_research_00161_prepare_20261109/episode/worker_result.json',
    'runs/capacity_10part_prepare_20261109/episode/hierarchy_decisions.json',
    'runs/capacity_10part_prepare_20261109/episode/hierarchy_plans.json',
    'runs/capacity_10part_prepare_20261109/episode/worker_result.json',
    'runs/dfs_restore_fix_20261109/final_replay/restored_request.json',
    'runs/support_learning_live_20261109_1155/revalidation_v2/skills_frozen.json',
    'runs/support_learning_live_20261109_1155/revalidation_v2/lessons_frozen.json',
    'runs/support_free_reuse_20261109_1250/live/episode/hierarchy_decisions.json',
    'runs/support_free_reuse_20261109_1250/live/episode/config.json',
    'runs/multitask_roles_20261105/prepared/manifest.json',
    'runs/multitask_roles_live_20261106/live/candidate_01/provider_proposal.json',
    'runs/separation_final_live_20261022/live_cycle/skills_frozen.json',
    'runs/separation_final_live_20261022/live_cycle/lessons_frozen.json',
    'runs/multitask_learning_live_20261105/live/episodes/02_02628_0_empty_collection/planner/planner_report.json',
    'runs/multitask_learning_live_20261105/live/episodes/03_01289_0_empty_collection/planner/planner_report.json',
)


def workspace_path(value, *, root=ROOT):
    raw = str(value).replace('\\', '/')
    if raw.lower().startswith(('e:', '/mnt/e', '//')) or '..' in raw.split('/'):
        raise ValueError('release_forbidden_path')
    root = Path(root).absolute()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError('release_workspace_only') from None
    cursor = root
    for segment in (None,) + relative.parts:
        if segment is not None:
            cursor = cursor / segment
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('release_reparse_path')
    return path


def sha(path, *, root=ROOT):
    return hashlib.sha256(workspace_path(path, root=root).read_bytes()).hexdigest()


def source_inventory(*, root=ROOT):
    pins = {}
    for folder in SOURCE_DIRS:
        for path in sorted(workspace_path(folder, root=root).iterdir()):
            path = workspace_path(path, root=root)
            if path.suffix == '.py' and path.is_file():
                pins[path.relative_to(root).as_posix()] = sha(path, root=root)
    return pins


def replay_inventory(*, root=ROOT):
    return {name: sha(name, root=root) for name in REPLAY_INPUTS}


def gate_tests():
    """Discover the exact supported suite without executing it; reject missing tests."""
    def flatten(suite):
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                yield from flatten(item)
            else:
                yield item
    tests = {}
    loader = unittest.TestLoader()
    for module in TEST_MODULES:
        found = list(flatten(loader.loadTestsFromName(module)))
        if not found or loader.errors:
            raise ValueError('release_test_module_missing_or_unloadable:' + module)
        for test in found:
            tests.setdefault(test.id(), test)
    return tests


def validate_release_certificate(path, *, root=ROOT, profile_id=PROFILE_ID):
    if profile_id not in COVERED_PROFILES:
        raise ValueError('unknown_release_execution_profile')
    if path is None:
        raise ValueError('offline_release_certificate_required')
    path = workspace_path(path, root=root)
    if path.relative_to(root).parts[0] != 'runs':
        raise ValueError('release_certificate_must_be_in_runs')
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if (value.get('schema') != SCHEMA or value.get('profile_id') != PROFILE_ID
            or value.get('status') != 'PASS'
            or value.get('test_modules') != list(TEST_MODULES)
            or value.get('API_calls') != 0 or value.get('physics_calls') != 0
            or value.get('source_and_replays_unchanged') is not True
            or value.get('authorizes_paid_execution') is not False
            or value.get('physical_performance') != 'UNVERIFIED'):
        raise ValueError('invalid_offline_release_certificate')
    if profile_id != PROFILE_ID and value.get('covered_profiles') != list(COVERED_PROFILES):
        raise ValueError('release_does_not_cover_capacity_profile')
    stats = value.get('tests', {})
    ids = value.get('test_ids', [])
    if (type(stats.get('run')) is not int or stats['run'] < len(TEST_MODULES)
            or any(stats.get(key) != 0 for key in ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes'))
            or type(ids) is not list or len(ids) != stats['run'] or len(set(ids)) != len(ids)):
        raise ValueError('release_tests_incomplete')
    if value.get('source_sha256') != source_inventory(root=root):
        raise ValueError('release_source_changed_rerun_offline_gate')
    if value.get('replay_sha256') != replay_inventory(root=root):
        raise ValueError('release_replay_inputs_changed')
    if set(ids) != set(gate_tests()):
        raise ValueError('release_required_test_ids_changed')
    log = workspace_path(value['test_log'], root=root)
    if log.parent != path.parent or sha(log, root=root) != value.get('test_log_sha256'):
        raise ValueError('release_test_log_changed')
    return value
