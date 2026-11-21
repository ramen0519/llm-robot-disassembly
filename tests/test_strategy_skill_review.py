"""Injected strategy review responses only: no network, API, VM or physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator import strategy_skill_review as review
from orchestrator.strategy_failover import POLICY as FAILOVER_POLICY
from orchestrator.strategy_plan import CONTRACT, compile_plan, schema
from orchestrator.strategy_planner import StrategyPlanService
from tests.test_explicit_action_service import provider
from tests.test_hierarchical_planner import request, policy
from tests.test_strategy_plan import value, card, registered_value
from worker.explicit_action_contract import digest


def row(selected=None, *, decision='not_selected', reason_code='prefer_other_procedure',
        reason='Use the explicitly selected base procedure for this plan.'):
    return dict(skill_id=(selected or card())['skill_id'], decision=decision,
                reason_code=reason_code, reason=reason)


class StrategySkillReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='strategy_review_fixture_')
        self.addCleanup(temp.cleanup)
        self.root, self.serial = Path(temp.name), 0
        env = patch.dict(os.environ, STRATEGY_REVIEW_FIXTURE_KEY='fixture-only-not-a-real-key')
        env.start(); self.addCleanup(env.stop)
        for target in ('urllib.request.OpenerDirector.open', 'socket.create_connection'):
            guard = patch(target, side_effect=AssertionError('actual_network_forbidden'))
            blocked = guard.start(); self.addCleanup(guard.stop); self.addCleanup(blocked.assert_not_called)

    def service(self, wire=None, *, selected_policy=review.POLICY, role='planner', ledger=None,
                failure_policy='legacy_v1'):
        cfg = dict(policy()['roles'][role], api_key_env='STRATEGY_REVIEW_FIXTURE_KEY')
        raw = provider(wire if wire is not None else dict(value(), skill_review=[]))
        raw['model'] = cfg['model']
        transport = Mock(return_value=raw)
        self.serial += 1
        path = ledger or self.root/('review_%d.sqlite' % self.serial)
        service = StrategyPlanService(cfg, path, transport, role=role,
            skill_review_policy=selected_policy, strategy_failure_policy=failure_policy)
        return service, transport

    def envelope(self, cards=None, *, selected_policy=review.POLICY):
        result = dict(request=request(), skills=cards or [], escalation={}, max_steps=64,
            timeout_seconds=30, plan_provider_contract=CONTRACT,
            planning_feedback_policy='repair_evidence_v2')
        if selected_policy != review.LEGACY:
            result['skill_review_policy'] = selected_policy
        return result

    def response(self, wire, cards=None, **options):
        service, transport = self.service(wire, **options)
        result = service.plan(request(), cards or [], {}, timeout_seconds=30)
        transport.assert_called_once()
        return result

    def test_legacy_contract_body_and_plan_remain_unchanged(self):
        service, _ = self.service(value(), selected_policy=review.LEGACY)
        body = service._body(self.envelope(selected_policy=review.LEGACY))
        self.assertNotIn('skill_selection_review', json.loads(body['input'][0]['content']))
        self.assertNotIn('skill_review', body['text']['format']['schema']['properties'])
        self.assertNotIn(review.POLICY, body['instructions'])
        result = service.plan(request(), [], {}, timeout_seconds=30)
        self.assertEqual(result['status'], 'planned')
        self.assertEqual(result['plan'], compile_plan(value(), request(), [], orientation_reference=None))
        self.assertNotIn('skill_selection_review', result['metadata'])
        self.assertNotIn('skill_review_policy', result['metadata'])

    def test_opt_in_body_requests_bounded_all_card_declarations_without_preference(self):
        selected = card(); selected['applicability'] = 'no_trigger'
        service, _ = self.service()
        before = self.envelope([selected]); original = deepcopy(before)
        body = service._body(before)
        self.assertEqual(before, original)
        context = json.loads(body['input'][0]['content'])
        self.assertEqual(context['skill_selection_review']['supplied_skills'], [dict(
            skill_id=selected['skill_id'], current_proposal_available=False, applicability='no_trigger')])
        self.assertFalse(context['skill_selection_review']['selection_forced'])
        self.assertIn('No skill is\npreferred or required', body['instructions'])
        self.assertIn('not a thinking transcript', body['instructions'])
        extension = body['text']['format']['schema']['properties']['skill_review']
        self.assertEqual((extension['minItems'], extension['maxItems']), (1, 1))
        self.assertEqual(extension['items']['properties']['reason']['maxLength'], 240)
        self.assertFalse(extension['items']['additionalProperties'])

    def test_empty_library_requires_empty_review_and_does_not_select_a_skill(self):
        result = self.response(dict(value(), skill_review=[]))
        self.assertEqual(result['status'], 'planned')
        self.assertEqual(result['metadata']['skill_selection_review']['reviews'], [])
        self.assertEqual(result['plan'], compile_plan(value(), request(), [], orientation_reference=None))
        base = schema(request(), []); old = deepcopy(base)
        extended = review.extend_schema(base, [])
        self.assertEqual(base, old)
        self.assertEqual(extended['properties']['skill_review']['maxItems'], 0)

    def test_selected_skill_without_current_proposal_is_allowed_conditionally(self):
        selected = card(); selected['applicability'] = 'no_trigger'
        declared = row(selected, decision='selected', reason_code='planned_procedure',
                       reason='Use the registered procedure if its runtime conditions hold.')
        result = self.response(dict(registered_value(selected=selected), skill_review=[declared]), [selected])
        self.assertEqual(result['status'], 'planned', result)
        facts = result['metadata']['skill_selection_review']
        self.assertEqual(facts['reviews'][0]['decision'], 'selected')
        self.assertFalse(facts['reviews'][0]['current_proposal_available'])
        self.assertEqual(facts['reviews'][0]['planned_uses'], [dict(
            step_index=0, alternative_index=None, part_id=value()['steps'][0]['part_id'])])
        self.assertEqual(facts['plan_id'], result['plan']['plan_id'])
        self.assertEqual(facts['request_sha256'], digest(request()))
        self.assertEqual(facts['skills_sha256'], digest([selected]))
        self.assertTrue(facts['model_self_report'])
        self.assertFalse(facts['physical_success_implied'])
        self.assertFalse(facts['selection_forced'])

    def test_available_skill_may_be_declined_without_rewriting_actions(self):
        selected = card()
        selected['currently_proposed_choice'] = {'part_id': value()['steps'][0]['part_id']}
        wire = dict(value(), skill_review=[row(selected)])
        original = deepcopy(wire)
        result = self.response(wire, [selected])
        self.assertEqual(wire, original)
        self.assertEqual(result['status'], 'planned', result)
        self.assertEqual(result['plan'], compile_plan(value(), request(), [selected], orientation_reference=None))
        self.assertEqual(result['metadata']['skill_selection_review']['reviews'][0]['planned_uses'], [])

    def test_no_trigger_and_no_current_proposal_match_supplied_facts(self):
        selected = card(); selected['applicability'] = 'no_trigger'
        for code in ('no_trigger', 'no_current_proposal'):
            with self.subTest(code=code):
                result = self.response(dict(value(), skill_review=[row(selected, reason_code=code)]), [selected])
                self.assertEqual(result['status'], 'planned', result)

    def test_false_no_trigger_or_no_proposal_claim_is_untrusted_without_changing_plan(self):
        cases = [(dict(card(), applicability='deadline'), 'no_trigger', 'trigger_fact_mismatch'),
                 (dict(card(), currently_proposed_choice={'part_id': '1'}),
                  'no_current_proposal', 'proposal_fact_mismatch'),
                 (dict(card(), currently_proposed_choice={'part_id': '1'}, applicability='no_trigger'),
                  'no_trigger', 'proposal_fact_mismatch')]
        for selected, code, expected in cases:
            with self.subTest(code=code, selected=selected):
                result = self.response(dict(value(), skill_review=[row(selected, reason_code=code)]), [selected])
                self.assertEqual(result['status'], 'planned', result)
                self.assertEqual(result['plan'], compile_plan(value(), request(), [selected], orientation_reference=None))
                self.assertEqual(result['metadata']['skill_selection_review_error']['code'], 'strategy_skill_review_'+expected)
                self.assertNotIn('skill_selection_review', result['metadata'])

    def test_missing_proposal_fact_stays_unknown_instead_of_no_proposal(self):
        selected = card(); selected.pop('currently_proposed_choice')
        self.assertIsNone(review.context([selected])['supplied_skills'][0]['current_proposal_available'])
        result = self.response(dict(value(), skill_review=[row(selected, reason_code='scope_uncertain')]), [selected])
        self.assertEqual(result['status'], 'planned')
        self.assertIsNone(result['metadata']['skill_selection_review']['reviews'][0]['current_proposal_available'])
        invalid = self.response(dict(value(), skill_review=[row(selected, reason_code='no_current_proposal')]), [selected])
        self.assertEqual(invalid['status'], 'planned')
        self.assertEqual(invalid['metadata']['skill_selection_review_error']['code'], 'strategy_skill_review_proposal_fact_mismatch')

    def test_missing_review_recorded_as_unknown_and_same_plan_cached_without_retry(self):
        service, transport = self.service(value())
        for _ in range(2):
            result = service.plan(request(), [], {}, timeout_seconds=30)
            self.assertEqual(result['status'], 'planned')
            self.assertEqual(result['plan'], compile_plan(value(), request(), [], orientation_reference=None))
            self.assertEqual(result['metadata']['skill_selection_review_error'], dict(
                code='missing_strategy_skill_review', status='invalid_or_missing', model_self_report=True))
            self.assertNotIn('skill_selection_review', result['metadata'])
            self.assertEqual(result['metadata']['retries'], 0)
        transport.assert_called_once()

    def test_every_supplied_skill_must_be_reviewed_once_and_ids_must_exist(self):
        first, second = card(), dict(card(), skill_id='another-supplied-card')
        cases = [([], 'invalid_strategy_skill_review_coverage'),
                 ([row(first)], 'invalid_strategy_skill_review_coverage'),
                 ([row(first), row(first)], 'duplicate_strategy_skill_review_id'),
                 ([row(first), row(dict(second, skill_id='unknown'))], 'unknown_strategy_skill_review_id')]
        for rows, expected in cases:
            with self.subTest(rows=rows):
                result = self.response(dict(value(), skill_review=rows), [first, second])
                self.assertEqual(result['status'], 'planned')
                self.assertEqual(result['metadata']['skill_selection_review_error']['code'], expected)
                self.assertNotIn('skill_selection_review', result['metadata'])
        result = self.response(dict(value(), skill_review=[row(second), row(first)]), [first, second])
        self.assertEqual(result['status'], 'planned', result)

    def test_review_selection_must_match_primary_plan_in_both_directions(self):
        selected = card()
        cases = [(value(), row(selected, decision='selected', reason_code='planned_procedure')),
                 (registered_value(), row(selected))]
        for wire, declaration in cases:
            with self.subTest(wire=wire):
                result = self.response(dict(wire, skill_review=[declaration]), [selected])
                self.assertEqual(result['status'], 'planned')
                self.assertEqual(result['plan'], compile_plan(wire, request(), [selected], orientation_reference=None))
                self.assertEqual(result['metadata']['skill_selection_review_error']['code'], 'strategy_skill_review_selection_mismatch')
                self.assertNotIn('skill_selection_review', result['metadata'])

    def test_failover_alternative_counts_as_conditional_selected_use_not_execution(self):
        selected = card()
        wire = value()
        for item in wire['steps']:
            item['stability_failover'] = None
        alternative = {k: deepcopy(v) for k, v in wire['steps'][1].items() if k != 'stability_failover'}
        alternative.update(procedure='registered_skill', registered_skill_id=selected['skill_id'])
        wire['steps'][0]['stability_failover'] = dict(after_failures=2, alternatives=[alternative])
        wire['skill_review'] = [row(selected, decision='selected', reason_code='planned_procedure')]
        result = self.response(wire, [selected], failure_policy=FAILOVER_POLICY)
        self.assertEqual(result['status'], 'planned', result)
        fact = result['metadata']['skill_selection_review']['reviews'][0]
        self.assertEqual(fact['planned_uses'], [dict(step_index=0, alternative_index=0,
                                                  part_id=alternative['part_id'])])
        wire['skill_review'] = [row(selected)]
        invalid = self.response(wire, [selected], failure_policy=FAILOVER_POLICY)
        self.assertEqual(invalid['status'], 'planned')
        self.assertEqual(invalid['plan'], result['plan'])
        self.assertEqual(invalid['metadata']['skill_selection_review_error']['code'], 'strategy_skill_review_selection_mismatch')

    def test_short_reason_is_not_an_execution_authority_or_plan_identity(self):
        results = []
        for text in ('Prefer the base procedure for this plan.', '현재 계획에서는 기본 절차를 선택한다.'):
            results.append(self.response(dict(value(), skill_review=[row(reason=text)]), [card()]))
        self.assertEqual(results[0]['plan'], results[1]['plan'])
        self.assertNotEqual(results[0]['metadata']['skill_selection_review']['review_sha256'],
                            results[1]['metadata']['skill_selection_review']['review_sha256'])
        self.assertNotIn('skill_review', results[0]['plan'])

    def test_malformed_long_multiline_or_wrong_kind_reason_is_not_trusted_or_replanned(self):
        changes = [dict(reason=''), dict(reason=' '), dict(reason=' leading'), dict(reason='line\nbreak'),
                   dict(reason='a'*241), dict(reason=42), dict(reason_code='invented'),
                   dict(reason_code='planned_procedure'), dict(decision='unavailable'), dict(extra='field')]
        for change in changes:
            with self.subTest(change=change):
                invalid = dict(row(), **change)
                result = self.response(dict(value(), skill_review=[invalid]), [card()])
                self.assertEqual(result['status'], 'planned', result)
                self.assertEqual(result['plan'], compile_plan(value(), request(), [card()], orientation_reference=None))
                self.assertNotIn('skill_selection_review', result['metadata'])
                self.assertIn('skill_selection_review_error', result['metadata'])

    def test_both_roles_use_one_provider_call_and_cached_review(self):
        for role in ('scout', 'planner'):
            service, transport = self.service(dict(value(), skill_review=[row()]), role=role)
            result = service.plan(request(), [card()], {}, timeout_seconds=30)
            cached = service.plan(request(), [card()], {}, timeout_seconds=30)
            self.assertEqual(result['status'], 'planned', result)
            self.assertEqual(result['metadata']['skill_selection_review'], cached['metadata']['skill_selection_review'])
            self.assertTrue(cached['metadata']['cached'])
            self.assertFalse(cached['metadata']['attempted'])
            transport.assert_called_once()

    def test_policy_change_cannot_silently_reuse_or_retry_an_old_ledger_request(self):
        ledger = self.root/'shared.sqlite'
        old, old_transport = self.service(value(), selected_policy=review.LEGACY, ledger=ledger)
        self.assertEqual(old.plan(request(), [], {}, timeout_seconds=30)['status'], 'planned')
        new, new_transport = self.service(ledger=ledger)
        result = new.plan(request(), [], {}, timeout_seconds=30)
        self.assertEqual(result['fallback_reason'], 'request_id_conflict')
        old_transport.assert_called_once(); new_transport.assert_not_called()

    def test_policy_mismatch_is_rejected_before_transport(self):
        service, transport = self.service()
        result = service.rank(self.envelope(selected_policy=review.LEGACY))
        self.assertEqual(result['fallback_reason'], 'invalid_request')
        transport.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'policy_mismatch'):
            service._body(self.envelope(selected_policy=review.LEGACY))
        with self.assertRaisesRegex(ValueError, 'unknown_strategy_skill_review_policy'):
            self.service(selected_policy='other')

    def test_legacy_response_with_new_field_is_still_rejected(self):
        result = self.response(dict(value(), skill_review=[]), selected_policy=review.LEGACY)
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['metadata']['plan_contract_error'], 'stale_or_invalid_strategy_plan_identity')

    def test_review_never_makes_an_invalid_physical_action_contract_valid(self):
        wire = dict(value(), skill_review=[row()])
        wire['steps'][0]['support_ids'] = []
        result = self.response(wire, [card()])
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['metadata']['plan_contract_error'], 'invalid_strategy_step_fields')
        self.assertNotIn('skill_selection_review', result['metadata'])


if __name__ == '__main__':
    unittest.main()
