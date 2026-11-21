"""Offline round/schema/codec/service checks; no live API, solver or WSL."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_explicit_provider_v3 import fixture_request, remember, schema_accepts
from test_explicit_action_service import ExplicitActionServiceTests, config, provider
from orchestrator.explicit_action_service import ExplicitActionService, build_provider_body
from orchestrator.explicit_action_provider_v3 import build_domain, response_schema_v3, PROVIDER_CONTRACT_V3
from orchestrator.explicit_action_provider_v4 import (PROVIDER_CONTRACT_V4, build_domain_v4,
    response_schema_v4, provider_input_v4, encode_provider_decision_v4, expand_provider_decision_v4)
from orchestrator.llm_ranker import DEFAULTS
from worker.explicit_action_contract import canonical_response, enumerate_actions, validate_response
from worker.path_probe_round import (partition_path_round, semantic_path_key,
    canonical_path_round_response, validate_path_round_response)


def round_request(parts=3, poses=2):
    request = fixture_request(parts, poses)
    request['path_stage_observations'] = []
    return request


def observed(request, choice, path_success=False, status='UNKNOWN', late=False, backend='ASAP_RedMax'):
    remember(request, choice)
    request['path_stage_observations'].append(dict(action_id=request['attempted_action_ids'][-1],
        path_success=path_success, status=status, deadline_exceeded=late, backend=backend,
        source_event_id=request['run_id']+':attempt:'+str(len(request['attempted_action_ids']))))


def rectangle_tuples(rectangles):
    tuples = set()
    for row in rectangles:
        expanded = set(product([row['part_id']], row['pose_slots'], row['force_slots'], row['support_slots']))
        if tuples & expanded:
            raise AssertionError('overlapping_provider_rectangles')
        tuples.update(expanded)
    return tuples


class ProviderV4CodecTests(unittest.TestCase):
    def test_initial_round_matches_v3_without_changing_request(self):
        request = round_request(); before = deepcopy(request)
        domain = build_domain_v4(request)
        self.assertEqual(rectangle_tuples(domain['rectangles']), rectangle_tuples(build_domain(request)['rectangles']))
        self.assertEqual(domain['deferred_action_serializations'], 0)
        self.assertEqual(response_schema_v4(request), response_schema_v3(request))
        self.assertEqual(request, before)

    def test_exhaustive_small_domain_active_plus_deferred_is_entire_untried_set(self):
        request = round_request()
        old = canonical_response(request, None)
        observed(request, old['choice'])
        domain, schema = build_domain_v4(request), response_schema_v4(request)
        complete = rectangle_tuples(build_domain(request)['rectangles'])
        active, delayed = rectangle_tuples(domain['rectangles']), rectangle_tuples(domain['deferred_rectangles'])
        self.assertFalse(active & delayed)
        self.assertEqual(active | delayed, complete)
        self.assertEqual(len(delayed), 3)
        represented = set()
        for values in product(request['candidate_ids'], range(2), range(30), range(len(domain['support_catalog']))):
            decision = dict(zip(('part_id', 'pose_slot', 'force_slot', 'support_slot'), values))
            if schema_accepts(schema['properties']['decision'], decision):
                represented.add(values)
        self.assertEqual(represented, active)
        self.assertEqual(domain['active_action_serializations'] + domain['deferred_action_serializations'], len(complete))

    def test_deferred_support_variant_rejected_without_part_pruning(self):
        request = round_request(); old = canonical_response(request, None)
        observed(request, old['choice']); variant = deepcopy(old)
        variant['choice']['support_ids'] = ['1']
        wire = encode_provider_decision_v4(request, variant)
        self.assertFalse(schema_accepts(response_schema_v4(request), wire))
        with self.assertRaisesRegex(ValueError, 'action_deferred_for_probe_round'):
            expand_provider_decision_v4(request, wire)
        replacement = canonical_path_round_response(request, None)
        self.assertEqual(set(replacement['candidate_order']), set(request['candidate_ids']))
        self.assertNotEqual(semantic_path_key(request, replacement['choice']), semantic_path_key(request, old['choice']))

    def test_success_and_mixed_outcomes_do_not_defer_support_variants(self):
        for mixed in (False, True):
            request = round_request(); old = canonical_response(request, None)
            if mixed:
                observed(request, old['choice'])
                old['choice']['support_ids'] = ['1']
            observed(request, old['choice'], path_success=True)
            variant = deepcopy(old); variant['choice']['support_ids'] = ['2']
            wire = encode_provider_decision_v4(request, variant)
            self.assertTrue(schema_accepts(response_schema_v4(request), wire))
            validate_path_round_response(request, expand_provider_decision_v4(request, wire))
            self.assertEqual(build_domain_v4(request)['deferred_action_serializations'], 0)

    def test_ambiguous_observation_alone_never_defers(self):
        for options in (dict(path_success=None), dict(status='ERROR'), dict(late=True), dict(backend='mock')):
            request = round_request(); old = canonical_response(request, None)
            observed(request, old['choice'], **options)
            domain = build_domain_v4(request)
            self.assertEqual(domain['deferred_action_serializations'], 0)

    def test_ambiguous_observation_for_same_path_cancels_prior_failure_deferral(self):
        for options in (dict(path_success=None), dict(status='ERROR'), dict(late=True), dict(backend='mock')):
            request = round_request(); old = canonical_response(request, None)
            observed(request, old['choice'])
            changed = deepcopy(old['choice']); changed['support_ids'] = ['1']
            observed(request, changed, **options)
            variant = deepcopy(old); variant['choice']['support_ids'] = ['2']
            self.assertEqual(build_domain_v4(request)['deferred_action_serializations'], 0)
            self.assertTrue(schema_accepts(response_schema_v4(request), encode_provider_decision_v4(request, variant)))

    def test_pose_aliases_cannot_bypass_round_but_all_catalog_entries_remain(self):
        request = round_request(3, 1)
        alias = deepcopy(request['pose_catalog'][0]); alias['pose_id'] = 'alias'
        request['pose_catalog'].append(alias)
        old = canonical_response(request, None); observed(request, old['choice'])
        domain = build_domain_v4(request)
        self.assertEqual(len(domain['pose_catalog']), 2)
        self.assertEqual(domain['forbidden_action_serializations'], 2)
        self.assertEqual(domain['deferred_action_serializations'], 6)
        wire = encode_provider_decision_v4(request, old)
        wire['decision']['pose_slot'] = 1-wire['decision']['pose_slot']
        wire['decision']['support_slot'] = domain['support_catalog'].index(['1'])
        self.assertFalse(schema_accepts(response_schema_v4(request), wire))

    def test_all_paths_probed_releases_every_remaining_support_variant(self):
        request = round_request(2, 1)
        all_choices = list(enumerate_actions(request))
        for choice in all_choices:
            if not choice['support_ids']:
                observed(request, choice)
        domain = build_domain_v4(request)
        self.assertEqual(domain['round_state']['release_reason'], 'untried_paths_exhausted')
        self.assertEqual(domain['deferred_action_serializations'], 0)
        self.assertGreater(domain['available_action_serializations'], 0)
        self.assertEqual(rectangle_tuples(domain['rectangles']), rectangle_tuples(build_domain(request)['rectangles']))
        validate_path_round_response(request, canonical_path_round_response(request, None))

    def test_part_with_all_paths_deferred_remains_in_complete_tail(self):
        request = round_request(3, 1)
        for choice in list(enumerate_actions(request)):
            if choice['part_id'] == '0' and not choice['support_ids']:
                observed(request, choice)
        lookup, schema, audit = provider_input_v4(request)
        self.assertEqual(lookup['candidate_address_list'], ['0', '1', '2'])
        self.assertEqual(audit['active_first_candidate_ids'], ['1', '2'])
        self.assertFalse(audit['all_candidates_have_active_first_choice'])
        response = canonical_path_round_response(request, None)
        wire = encode_provider_decision_v4(request, response)
        self.assertTrue(schema_accepts(schema, wire))
        decoded = expand_provider_decision_v4(request, wire)
        self.assertEqual(set(decoded['candidate_order']), {'0', '1', '2'})
        self.assertNotEqual(decoded['candidate_order'][0], '0')

    def test_v3_still_allows_untried_support_variants(self):
        request = round_request(); old = canonical_response(request, None)
        observed(request, old['choice']); old['choice']['support_ids'] = ['1']
        wire = encode_provider_decision_v4(request, old)
        self.assertTrue(schema_accepts(response_schema_v3(request), wire))
        self.assertFalse(schema_accepts(response_schema_v4(request), wire))

    def test_18_parts_6_8_128_poses_60_history_partition_and_schema_limits(self):
        for pose_count in (6, 8, 128):
            request = round_request(18, pose_count)
            forces = build_domain(request)['force_catalog']
            for i in range(60):
                choice = dict(part_id=str(i%18), pose_id='p'+str((i//18)%pose_count),
                    **forces[(7*i)%30], support_ids=[], evidence_ids=[])
                observed(request, choice)
            _lookup, _schema, audit = provider_input_v4(request)
            self.assertEqual(audit['full_action_serializations'], 18*pose_count*30*154)
            self.assertEqual(audit['forbidden_action_serializations'], 60)
            self.assertEqual(audit['deferred_action_serializations'], 60*153)
            self.assertTrue(audit['active_plus_deferred_equals_untried'])
            self.assertTrue(audit['all_candidates_have_active_first_choice'])
            self.assertTrue(audit['active_pool_nonempty_while_untried_paths_remain'])
            for field, limit in (('property_count',5000),('enum_values',1000),
                    ('string_characters',120000),('max_object_depth',10),('schema_bytes',196608)):
                self.assertLessEqual(audit[field], limit)

    def test_body_describes_round_constraints_without_v3_full_availability_claim(self):
        request = round_request(); observed(request, canonical_response(request, None)['choice'])
        body = build_provider_body(DEFAULTS, request, PROVIDER_CONTRACT_V4)
        self.assertEqual(body['text']['format']['name'], PROVIDER_CONTRACT_V4)
        self.assertNotIn('All legal untried combinations remain representable.', body['instructions'])
        self.assertNotIn('Every complete candidate permutation is expressible', body['instructions'])
        self.assertIn('temporarily deferred', body['instructions'])
        payload = json.loads(body['input'][0]['content'])
        self.assertEqual(payload['request'], request)
        self.assertIn('provider_v4_lookup', payload)
        self.assertNotIn('attempted_path_keys', payload['provider_v4_lookup']['path_probe_round'])

    def test_missing_or_inconsistent_observations_cannot_form_schema(self):
        request = round_request(); del request['path_stage_observations']
        with self.assertRaisesRegex(ValueError, 'path_stage_observations_required'):
            provider_input_v4(request)
        request = round_request(); remember(request, canonical_response(request, None)['choice'])
        with self.assertRaisesRegex(ValueError, 'path_stage_observations_must_match_attempts'):
            provider_input_v4(request)


class ProviderV4ServiceTests(ExplicitActionServiceTests):
    def v4_service(self, transport, ledger=None):
        self.serial += 1
        return ExplicitActionService(config(), ledger or self.root/('v4_%d.sqlite'%self.serial),
            transport=transport, provider_contract=PROVIDER_CONTRACT_V4, fallback_policy='path_stage_v1')

    def v4_transport(self):
        def call(url, headers, body, timeout):
            request = json.loads(body['input'][0]['content'])['request']
            return provider(encode_provider_decision_v4(request, canonical_path_round_response(request, None)))
        return Mock(side_effect=call)

    def test_valid_response_one_call_cached_and_round_audit_preserved(self):
        request = round_request(); observed(request, canonical_response(request, None)['choice'])
        transport = self.v4_transport(); service = self.v4_service(transport)
        response = service.handle(request); cached = service.handle(request)
        self.assertEqual(transport.call_count, 1)
        self.assertTrue(cached['metadata']['cached'])
        self.assertEqual(response['metadata']['provider_domain_audit']['deferred_action_serializations'], 3)
        self.assertEqual(response['metadata']['provider_contract'], PROVIDER_CONTRACT_V4)
        validate_path_round_response(request, response)

    def test_injected_deferred_response_rejected_with_one_call_and_safe_active_fallback(self):
        request = round_request(); prior = canonical_response(request, None)
        observed(request, prior['choice']); proposal = deepcopy(prior)
        proposal['choice']['support_ids'] = ['1']
        transport = Mock(return_value=provider(encode_provider_decision_v4(request, proposal)))
        response = self.v4_service(transport).handle(request)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(response['metadata']['decision_validation_error'], 'action_deferred_for_probe_round')
        self.assertEqual(response['metadata']['selection_source'], 'deterministic_fallback')
        self.assertFalse(response['metadata']['llm_live'])
        self.assertNotEqual(response['choice'], proposal['choice'])
        validate_path_round_response(request, response)
        self.assertEqual(len(request['attempted_actions']), 1)

    def test_exact_duplicate_still_has_explicit_duplicate_diagnostic(self):
        request = round_request(); prior = canonical_response(request, None)
        observed(request, prior['choice'])
        transport = Mock(return_value=provider(encode_provider_decision_v4(request, prior)))
        response = self.v4_service(transport).handle(request)
        self.assertEqual(response['metadata']['decision_validation_error'], 'action_already_attempted')
        self.assertTrue(response['metadata']['action_repeated'])
        validate_path_round_response(request, response)

    def test_missing_table_raises_before_any_paid_transport(self):
        request = round_request(); del request['path_stage_observations']
        transport = self.v4_transport()
        with self.assertRaisesRegex(ValueError, 'path_stage_observations_required'):
            self.v4_service(transport).handle(request)
        transport.assert_not_called()

    def test_v3_cache_cannot_be_reused_under_v4(self):
        request = round_request(); ledger = self.root/'version_conflict.sqlite'
        first_transport = self.v4_transport()
        first = ExplicitActionService(config(), ledger, transport=first_transport,
            provider_contract=PROVIDER_CONTRACT_V3, fallback_policy='path_stage_v1')
        first.handle(request)
        transport = self.v4_transport(); response = self.v4_service(transport, ledger).handle(request)
        transport.assert_not_called()
        self.assertEqual(response['fallback_reason'], 'request_id_conflict')
        validate_path_round_response(request, response)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(ProviderV4CodecTests))
    suite.addTests(ProviderV4ServiceTests(name) for name in ProviderV4ServiceTests.__dict__ if name.startswith('test_'))
    return suite


if __name__ == '__main__':
    unittest.main()
