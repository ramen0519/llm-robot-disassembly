"""Runner policy and reporting tests. No WSL, physics, API or secret access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_prototype_completion import (APPROVED_EXT4_ASSEMBLY, assembly_validation_status,
                                      checked, decision_summary, parser, read_jsonl, validate_arguments)


def args(*extra):
    return parser().parse_args(['--run-dir', str(ROOT / 'runs' / ('runner_policy_' + uuid.uuid4().hex)), *extra])


class PrototypeRunnerTests(unittest.TestCase):
    def test_generic_validator_accepts_valid_unknown_with_exit_zero(self):
        self.assertEqual(assembly_validation_status(
            {'status': 'UNKNOWN', 'validation_ok': True, 'plan_success': False}, 0), 'UNKNOWN')

    def test_generic_validator_requires_explicit_valid_success_for_pass(self):
        self.assertEqual(assembly_validation_status(
            {'status': 'PASS', 'validation_ok': True, 'plan_success': True}, 0), 'PASS')
        for result, exit_code in (
            ({'status': 'PASS', 'validation_ok': False, 'plan_success': True}, 0),
            ({'status': 'PASS', 'validation_ok': True, 'plan_success': False}, 0),
            ({'status': 'PASS', 'validation_ok': True, 'plan_success': True}, 1),
            ({'status': 'UNKNOWN', 'validation_ok': True, 'plan_success': False}, 1),
            ({'status': 'UNKNOWN', 'validation_ok': True, 'plan_success': None}, 0),
            ({'status': 'ERROR', 'validation_ok': False, 'plan_success': None}, 1),
            ({'status': 'UNKNOWN', 'validation_ok': False, 'plan_success': None}, 1),
            ({}, 0), (None, 0)):
            self.assertEqual(assembly_validation_status(result, exit_code), 'ERROR')

    def test_default_fixture_has_zero_budget_and_new_ledger(self):
        parsed = args()
        output, assembly, base, memory, config_path, config, ledger, live = validate_arguments(parsed)
        self.assertFalse(live)
        self.assertEqual(config, {'enabled': False, 'max_calls': 0, 'max_usd': 0})
        self.assertEqual(ledger, output / 'batch_ledger.sqlite')
        self.assertIsNone(memory)
        self.assertEqual(base, '6')

    def test_none_base_and_positive_timeouts_accepted(self):
        values = validate_arguments(args('--base-part', 'none', '--timeout', '40', '--wall-timeout', '80'))
        self.assertIsNone(values[2])

    def test_E_rejected_before_any_filesystem_access(self):
        for path in ('E:/secret', 'e:\\unused', '/mnt/e/unused', '/mnt/./e/unused'):
            with patch.object(Path, 'lstat', side_effect=AssertionError('filesystem_access')):
                with self.assertRaisesRegex(ValueError, 'excluded_path'):
                    checked(path)

    def test_disallowed_assembly_path_rejected(self):
        for assembly in ('E:/assets', '/mnt/e/assets', '/opt/not-approved', '/opt/asap-research/ASAP/assets/../other'):
            with self.assertRaises(ValueError):
                validate_arguments(args('--assembly-dir', assembly))

    def test_exact_dedicated_ext4_input_is_allowed_without_widening_boundary(self):
        values = validate_arguments(args('--assembly-dir', APPROVED_EXT4_ASSEMBLY, '--base-part', 'none'))
        self.assertEqual(values[1], APPROVED_EXT4_ASSEMBLY)
        self.assertIsNone(values[2])
        for assembly in (APPROVED_EXT4_ASSEMBLY + '/child', APPROVED_EXT4_ASSEMBLY + '_other',
                         APPROVED_EXT4_ASSEMBLY.rsplit('/', 1)[0],
                         APPROVED_EXT4_ASSEMBLY.replace('07009', '00001')):
            with self.assertRaisesRegex(ValueError, 'approved_ASAP_asset'):
                validate_arguments(args('--assembly-dir', assembly))

    def test_existing_run_is_never_reused(self):
        parsed = args()
        parsed.run_dir = ROOT / 'runs'
        with self.assertRaises(FileExistsError):
            validate_arguments(parsed)

    def test_invalid_numeric_limits_rejected(self):
        for option, value in (('--budget', '0'), ('--max-gripper', '1'), ('--max-pose', '0'),
                              ('--wall-timeout', '0'), ('--timeout', '-1'), ('--response-timeout', '121')):
            with self.assertRaisesRegex(ValueError, 'invalid_bounded'):
                validate_arguments(args(option, value))

    def test_live_requires_both_explicit_flag_and_config(self):
        for extra in (('--mode', 'llm'), ('--mode', 'llm', '--allow-live-api'), ('--allow-live-api',)):
            with self.assertRaisesRegex(ValueError, 'explicit_API'):
                validate_arguments(args(*extra))

    def test_fixture_cannot_enable_paid_config(self):
        with tempfile.TemporaryDirectory(prefix='prototype_runner_', dir=ROOT / 'runs') as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'enabled': True, 'max_calls': 1, 'max_usd': 0.01}))
            with self.assertRaisesRegex(ValueError, 'fixture_requires_disabled'):
                validate_arguments(args('--llm-config', str(path)))

    def test_config_inline_key_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='prototype_runner_', dir=ROOT / 'runs') as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'enabled': False, 'api_key': 'fixture-only'}))
            with self.assertRaisesRegex(ValueError, 'no_inline_credentials'):
                validate_arguments(args('--llm-config', str(path)))

    def test_decision_summary_uses_nested_actual_pipeline_metadata(self):
        invocation = {'skill_id': 'retry_after_state_change_v1', 'invocation_id': 'fixture'}
        decisions = [
            {'response': {'metadata': {'ranking_source': 'controlled_physical_probe', 'offered_hint_ids': ['h1']}}},
            {'response': {'metadata': {'ranking_source': 'fixture_decision', 'offered_hint_ids': ['h1', 'h2'],
                                       'used_hint_ids': ['h2'], 'used_source_evidence_ids': ['source-e1'],
                                       'used_current_candidate_ids': ['different-id'], 'repair_invocation': invocation}}},
        ]
        summary = decision_summary(decisions)
        self.assertEqual(summary['transfer_exposure_count'], 2)
        self.assertEqual(summary['transfer_selection_count'], 1)
        self.assertEqual(summary['requests_citing_transfer'], 1)
        self.assertEqual(summary['selected_source_evidence_ids'], ['source-e1'])
        self.assertEqual(summary['selected_current_candidate_ids'], ['different-id'])
        self.assertEqual(summary['repair_invocations'], [invocation])
        self.assertEqual(summary['decision_sources'], {'controlled_physical_probe': 1, 'fixture_decision': 1})
        self.assertFalse(summary['cross_task_causal_benefit_demonstrated'])

    def test_exposure_without_citation_is_not_reuse(self):
        summary = decision_summary([{'response': {'metadata': {'offered_hint_ids': ['h'], 'used_hint_ids': []}}}])
        self.assertEqual(summary['transfer_exposure_count'], 1)
        self.assertEqual(summary['transfer_selection_count'], 0)
        self.assertEqual(summary['requests_citing_transfer'], 0)

    def test_partial_raw_log_remains_available_and_is_reported(self):
        with tempfile.TemporaryDirectory(prefix='prototype_runner_', dir=ROOT / 'runs') as directory:
            path = Path(directory) / 'events.jsonl'
            text = '{"kind":"valid_prefix"}\n{"truncated":'
            path.write_text(text)
            issues = []
            values = read_jsonl(path, issues)
            self.assertEqual(values, [{'kind': 'valid_prefix'}])
            self.assertEqual(issues[0]['line'], 2)
            self.assertEqual(path.read_text(), text)
            with self.assertRaises(ValueError):
                read_jsonl(path)


if __name__ == '__main__':
    unittest.main()
