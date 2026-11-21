"""Review provenance and opt-in wiring. Injected providers/verifier; API/WSL 0."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.strategy_plan import CONTRACT, compile_plan
from orchestrator.strategy_service import StrategyDecisionService
from tests.test_hierarchical_planner import request, policy
from tests.test_strategy_service import wire
from tests.test_explicit_action_planner import make_planner
from tools.run_hierarchical_cycle import execute_cycle

POLICY = 'skill_selection_review_v1'


class StrategySkillReviewWiringTests(unittest.TestCase):
    def test_review_audits_do_not_change_planning_feedback_or_displace_real_events(self):
        from orchestrator.hierarchical_service import HierarchicalDecisionService
        service = object.__new__(HierarchicalDecisionService)
        service.events = [dict(kind='physical_failure', sequence=1, reason='measured'),
            dict(kind='skill_selection_review_not_recorded', sequence=2),
            dict(kind='skill_selection_review_context', sequence=3),
            dict(kind='sol_requested', sequence=4)]
        before = deepcopy(service.events)
        self.assertEqual(service._planning_prior_events(), [
            dict(kind='physical_failure', sequence=1, reason='measured'),
            dict(kind='sol_requested', sequence=2)])
        self.assertEqual(service.events, before)

    def test_opt_in_requires_both_provider_contracts_to_match(self):
        scout = Mock(provider_contract=CONTRACT, skill_review_policy='legacy_v1')
        planner = Mock(provider_contract=CONTRACT, skill_review_policy=POLICY)
        with self.assertRaisesRegex(ValueError, 'skill_review_policy_provider_mismatch'):
            StrategyDecisionService(scout, planner, {}, policy()['episode'],
                skill_review_policy=POLICY, recovery_support_policy='preserve_existing_v1')

    def test_bad_policy_rejected(self):
        with self.assertRaises(ValueError):
            StrategyDecisionService(None, None, {}, {}, skill_review_policy='typo')

    def test_cycle_rejects_review_on_conditional_path_before_files(self):
        with patch('tools.run_hierarchical_cycle.checked', side_effect=AssertionError('NO_FILE')):
            with self.assertRaisesRegex(ValueError, 'skill_review_requires_strategy_research_cycle'):
                execute_cycle('unused', {}, 'unused', skill_review_policy=POLICY)

    def test_sol_review_survives_journal_once_and_execution_is_not_another_selection(self):
        reviews = []
        planner = Mock(provider_contract=CONTRACT, skill_review_policy=POLICY)
        scout = Mock(provider_contract=CONTRACT, skill_review_policy=POLICY)
        def plan(req, cards, escalation, **kwargs):
            compiled = compile_plan(wire(req), req, cards)
            # Review validation is covered in provider tests. Here an explicit
            # injected provider result exercises service/export provenance.
            review = dict(policy=POLICY, request_id=req['request_id'], state_id=req['state_id'],
                plan_id=compiled['plan_id'], reviews=[], model_self_report=True,
                selection_forced=False, physical_success_implied=False)
            reviews.append(deepcopy(review))
            return dict(status='planned', plan=compiled, metadata=dict(
                llm_live=False, skill_selection_review=review, skill_review_policy=POLICY))
        planner.plan.side_effect = plan
        limits = dict(policy()['episode'], scout_calls=0, planner_calls=1)
        with tempfile.TemporaryDirectory() as folder, \
             patch('orchestrator.hierarchical_service.retrieve', return_value=[]), \
             patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('NO_NETWORK')) as network:
            service = StrategyDecisionService(scout, planner, {}, limits,
                skill_review_policy=POLICY, recovery_support_policy='preserve_existing_v1',
                planning_feedback_policy='repair_evidence_v2', journal_dir=Path(folder)/'journal')
            worker = make_planner(decision=service.handle, path_selection_policy='path_probe_round_v1',
                fallback_policy='path_stage_v1', decision_failure_policy='halt_v1')
            worker.plan(budget=4)
            self.assertTrue(worker.report()['success'])
            self.assertEqual(planner.plan.call_count, 1)
            first, continued = [d['response']['metadata'] for d in service.decisions]
            self.assertEqual(first['skill_selection_review'], reviews[0])
            self.assertEqual(continued['plan_skill_selection_review'], reviews[0])
            self.assertNotIn('skill_selection_review', continued)
            self.assertEqual(continued['skill_selection_review_context']['status'], 'no_new_model_choice')
            events = [json.loads(line) for line in
                (Path(folder)/'journal/events.jsonl').read_text(encoding='utf-8').splitlines()]
            recorded = [e for e in events if e['kind']=='skill_selection_review_recorded']
            self.assertEqual(len(recorded), 1)
            self.assertEqual(recorded[0]['review'], reviews[0])
            self.assertEqual(recorded[0]['role'], 'planner')
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
