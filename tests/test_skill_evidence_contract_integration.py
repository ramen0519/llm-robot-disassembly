"""Local replay and injected response checks; no API, key or new physics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.skill_discovery import SkillDiscoveryService, response_schema
from orchestrator.batch_budget import budget_summary
from orchestrator.skill_behavior_contract import behavior_examples, audit_behavior_claims
from orchestrator.skill_program import step_program
from orchestrator.skill_refinement import audit_revision
from tools.run_skill_discovery import executable_card, preview_config
from tests.test_skill_discovery import raw_response

ROOT = Path(__file__).absolute().parents[1]
PRIOR = ROOT / 'runs/initial_contact_skill_followup_20261019/live_01/candidate_02'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fixture_claims(program, feedback):
    """Test-only calculated predictions, NOT an actual model response."""
    by_id = {item['current_observation_request']['request_id']:
             item['current_observation_request'] for item in feedback}
    claims = []
    for example in behavior_examples(feedback):
        step = step_program(program, by_id[example['request_id']])
        choice = ({k: v for k, v in step['response']['choice'].items() if k != 'evidence_ids'}
                  if step['response'] else None)
        passes = example['observed_pass_choices']
        relation = ('preserve_observed_pass' if choice in passes else 'change_observed_pass') if passes else 'no_observed_pass'
        claims.append(dict(request_id=example['request_id'], predicted_rule_index=step['rule_index'],
            predicted_choice=choice, relation_to_observed_pass=relation))
    return claims


class EvidenceContractIntegrationTests(unittest.TestCase):
    def setUp(self):
        for target in ('urllib.request.OpenerDirector.open', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('external execution forbidden'))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.ledger = Path(self.temp.name) / 'never_created.sqlite'
        self.service = SkillDiscoveryService(preview_config(), self.ledger)
        self.packet = read(PRIOR / 'generation_packet.json')
        self.program = read(PRIOR / 'provider_proposal.json')['program']

    def contracted(self):
        packet = deepcopy(self.packet)
        packet['context']['behavior_contract_version'] = 1
        return packet

    def assert_no_external_attempt(self):
        # Existing RankService initializes an empty local SQLite schema even
        # for disabled previews. File existence is not an API request.
        budget = budget_summary(self.ledger)
        self.assertEqual(budget['request_count'], 0)
        self.assertEqual(budget['reserved_calls'], 0)

    def response(self, packet, claims):
        value = {key: packet[key] for key in ('request_id', 'state_id')}
        value.update(program=self.program, card=deepcopy(read(PRIOR / 'provider_proposal.json')['card']),
                     behavior_claims=claims)
        return raw_response(value)

    def test_legacy_saved_body_remains_identical(self):
        self.assertEqual(self.service.preview(self.packet)['body'], read(PRIOR / 'api_preview.json')['body'])
        self.assertNotIn('behavior_claims', response_schema(self.packet)['required'])
        self.assert_no_external_attempt()

    def test_versioned_body_is_smaller_with_complete_prediction_schema(self):
        packet = self.contracted()
        before = deepcopy(packet)
        old = self.service.preview(self.packet)
        new = self.service.preview(packet)
        self.assertLess(new['body_bytes'], old['body_bytes'])
        self.assertEqual(packet, before)
        schema = new['body']['text']['format']['schema']
        self.assertIn('behavior_claims', schema['required'])
        self.assertNotIn('current_observation_request', new['body']['input'][0]['content'])
        self.assert_no_external_attempt()
        self.assertFalse(new['key_accessed'])

    def test_truthful_prediction_can_pass_but_known_failed_action_still_rejected(self):
        packet = self.contracted()
        claims = fixture_claims(self.program, packet['validation_feedback'])
        metadata = {}
        value, reason = self.service._parse_response(self.response(packet, claims), packet, metadata)
        self.assertIsNone(reason)
        self.assertIsNotNone(value)
        self.assertTrue(metadata['behavior_contract_audit']['accepted_for_fresh_validation'])
        self.assertFalse(audit_revision(self.program, packet['validation_feedback'])['accepted'])
        self.assert_no_external_attempt()

    def test_claim_of_preserving_pass_is_rejected_when_empty_rank_selects_failed_part(self):
        packet = self.contracted()
        claims = fixture_claims(self.program, packet['validation_feedback'])
        target = next(c for c in claims if c['relation_to_observed_pass'] == 'change_observed_pass')
        target['relation_to_observed_pass'] = 'preserve_observed_pass'
        metadata = {}
        value, reason = self.service._parse_response(self.response(packet, claims), packet, metadata)
        self.assertIsNone(value)
        self.assertEqual(reason, 'behavior_prediction_mismatch')
        self.assertFalse(metadata['behavior_contract_audit']['accepted'])

    def test_missing_claims_fail_closed_before_physics(self):
        packet = self.contracted()
        raw = self.response(packet, [])
        value, reason = self.service._parse_response(raw, packet, {})
        self.assertIsNone(value)
        self.assertEqual(reason, 'behavior_prediction_mismatch')

    def test_new_card_comes_from_program_and_raw_prose_remains_unchanged(self):
        original = deepcopy(read(PRIOR / 'provider_proposal.json')['card'])
        before = deepcopy(original)
        new = executable_card(self.program, original)
        self.assertEqual(original, before)
        self.assertIn('executable predicates', new['applicability'])
        self.assertNotEqual(new['applicability'], original['applicability'])
        self.assertNotIn(original['hypothesis'], new['description'])
        self.assertEqual(new['executable_description']['rule_count'], 2)
        self.assertEqual(new['evidence_ids'], original['evidence_ids'])

    def test_unknown_contract_version_cannot_silently_use_legacy_path(self):
        packet = self.contracted()
        for value in (None, False, 0, True, 2, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                packet['context']['behavior_contract_version'] = value
                self.service.preview(packet)


if __name__ == '__main__':
    unittest.main()
