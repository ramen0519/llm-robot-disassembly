"""Offline schema/codec/external-boundary regression; never real API/physics."""
from copy import deepcopy
from itertools import permutations, product
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_explicit_action_contract import request
from test_explicit_action_service import ExplicitActionServiceTests, config, provider
from orchestrator.explicit_action_service import ExplicitActionService, build_provider_body
from orchestrator.llm_ranker import DEFAULTS
from orchestrator.explicit_action_provider_v3 import (PROVIDER_CONTRACT_V3, build_domain,
    provider_input_v3, response_schema_v3, schema_statistics, expand_provider_decision_v3,
    encode_provider_decision_v3)
from worker.explicit_action_contract import (action_id, canonical_response,
    enumerate_actions, path_signature, validate_response)


def schema_accepts(schema, value):
    """Independent validator of the small supported subset emitted by v3.

    This checks JSON-schema membership, not provider availability or live use.
    Unexpected keywords fail rather than silently accepting an unsupported one.
    """
    allowed = {'type', 'properties', 'required', 'additionalProperties', 'anyOf',
               'enum', 'minimum', 'maximum'}
    if set(schema) - allowed:
        raise AssertionError('unreviewed_schema_keyword')
    if 'anyOf' in schema:
        return any(schema_accepts(option, value) for option in schema['anyOf'])
    kind = schema.get('type')
    if kind == 'object':
        return (type(value) is dict and set(value) == set(schema['properties'])
            and set(schema['required']) == set(schema['properties'])
            and schema['additionalProperties'] is False
            and all(schema_accepts(item, value[key]) for key, item in schema['properties'].items()))
    if kind == 'integer':
        return type(value) is int and schema.get('minimum', value) <= value <= schema.get('maximum', value)
    if kind == 'boolean':
        return type(value) is bool
    if kind == 'string':
        return type(value) is str and value in schema['enum']
    raise AssertionError('unreviewed_schema_type')


def fixture_request(parts=3, poses=2):
    r = request()
    r.update(remaining_parts=[str(i) for i in range(parts)], candidate_ids=[str(i) for i in range(parts)],
             base_part=None, max_grippers=3, attempted_action_ids=[], attempted_actions=[])
    initial = deepcopy(r['pose_catalog'][0])
    r['pose_catalog'] = []
    for i in range(poses):
        pose = deepcopy(initial)
        pose['pose_id'], pose['matrix'][0][3] = 'p'+str(i), i
        r['pose_catalog'].append(pose)
    return r


def remember(r, choice):
    identifier = action_id(r, choice)
    r['attempted_action_ids'].append(identifier)
    r['attempted_actions'].append(dict(action_id=identifier,
        **{k: deepcopy(v) for k, v in choice.items() if k != 'evidence_ids'}))


class ProviderV3CodecTests(unittest.TestCase):
    def test_small_domain_exhaustive_matches_original_action_enumerator(self):
        r = fixture_request()
        prior = list(enumerate_actions(r))
        for choice in prior[::71]:
            remember(r, choice)
        domain, schema = build_domain(r), response_schema_v3(r)
        expected = {action_id(r, choice) for choice in enumerate_actions(r)}
        represented = set()
        for part, pose, force, support in product(r['candidate_ids'], range(2), range(30), range(len(domain['support_catalog']))):
            wire = dict(part_id=part, pose_slot=pose, force_slot=force, support_slot=support)
            if schema_accepts(schema['properties']['decision'], wire):
                choice = dict(part_id=part, pose_id='p'+str(pose), **domain['force_catalog'][force],
                              support_ids=domain['support_catalog'][support], evidence_ids=[])
                represented.add(action_id(r, choice))
        self.assertEqual(represented, expected)
        self.assertEqual(len(represented), domain['available_action_serializations'])

    def test_every_four_part_permutation_roundtrips_without_tiebreak(self):
        r = fixture_request(parts=4, poses=1)
        base = canonical_response(r, None)
        encoded = set()
        for order in permutations(r['candidate_ids']):
            response = deepcopy(base)
            response['candidate_order'] = list(order)
            response['choice']['part_id'] = order[0]
            wire = encode_provider_decision_v3(r, response)
            self.assertTrue(schema_accepts(response_schema_v3(r), wire))
            decoded = expand_provider_decision_v3(r, wire)
            self.assertEqual(decoded['candidate_order'], list(order))
            validate_response(r, decoded)
            encoded.add(json.dumps(wire, sort_keys=True))
        self.assertEqual(len(encoded), 24)

    def test_exact_repeat_blocked_but_path_equivalent_support_variant_retained(self):
        r = fixture_request()
        old = canonical_response(r, None)
        remember(r, old['choice'])
        wire = encode_provider_decision_v3(r, old)
        schema = response_schema_v3(r)
        self.assertFalse(schema_accepts(schema, wire))
        changed = deepcopy(old)
        changed['choice']['support_ids'] = ['1']
        wire = encode_provider_decision_v3(r, changed)
        self.assertTrue(schema_accepts(schema, wire))
        validate_response(r, expand_provider_decision_v3(r, wire))

    def test_same_matrix_aliases_do_not_bypass_exact_action_guard(self):
        r = fixture_request()
        alias = deepcopy(r['pose_catalog'][0]); alias['pose_id'] = 'alias'
        r['pose_catalog'].append(alias)
        old = canonical_response(r, None)
        remember(r, old['choice'])
        alternate = deepcopy(old); alternate['choice']['pose_id'] = 'alias'
        self.assertFalse(schema_accepts(response_schema_v3(r), encode_provider_decision_v3(r, alternate)))
        alternate['choice']['pose_id'] = 'p1'
        self.assertTrue(schema_accepts(response_schema_v3(r), encode_provider_decision_v3(r, alternate)))
        self.assertEqual(len(build_domain(r)['pose_catalog']), 3)

    def test_exhausted_part_stays_in_full_candidate_permutation(self):
        r = fixture_request(parts=2, poses=1)
        for choice in list(enumerate_actions(r)):
            if choice['part_id'] == '0':
                remember(r, choice)
        response = canonical_response(r, None)
        wire = encode_provider_decision_v3(r, response)
        self.assertTrue(schema_accepts(response_schema_v3(r), wire))
        self.assertEqual(expand_provider_decision_v3(r, wire)['candidate_order'], ['1', '0'])
        self.assertEqual(build_domain(r)['candidate_address_list'], ['0', '1'])

    def test_base_and_moving_part_never_legal_supports(self):
        r = fixture_request(); r['base_part'] = '2'; r['candidate_ids'] = ['0', '1']
        domain, schema = build_domain(r), response_schema_v3(r)
        self.assertTrue(all('2' not in support for support in domain['support_catalog']))
        for i, support in enumerate(domain['support_catalog']):
            wire = dict(part_id='0', pose_slot=0, force_slot=0, support_slot=i)
            self.assertEqual(schema_accepts(schema['properties']['decision'], wire), '0' not in support)

    def test_complete_force_catalog_has_only_valid_single_or_orthogonal_two_leg(self):
        r = fixture_request()
        forces = build_domain(r)['force_catalog']
        self.assertEqual(len(forces), 30)
        self.assertEqual(sum(f['skill_id']=='single_force_v1' for f in forces), 6)
        for force in forces:
            response = canonical_response(r, None); response['choice'].update(force)
            validate_response(r, expand_provider_decision_v3(r, encode_provider_decision_v3(r, response)))

    def test_18_parts_6_8_and_128_poses_with_60_history_fit_schema_limits(self):
        for pose_count in (6, 8, 128):
            with self.subTest(poses=pose_count):
                r = fixture_request(18, pose_count)
                forces = build_domain(r)['force_catalog']
                for i in range(60):
                    choice = dict(part_id=str(i%18), pose_id='p'+str((i//18)%pose_count),
                        **forces[(7*i)%30], support_ids=[], evidence_ids=[])
                    remember(r, choice)
                lookup, schema, stats = provider_input_v3(r)
                self.assertEqual(stats['full_action_serializations'], 18*pose_count*30*154)
                self.assertEqual(stats['available_action_serializations'], 18*pose_count*30*154-60)
                self.assertEqual(stats['forbidden_action_serializations'], 60)
                self.assertEqual(len(lookup['candidate_address_list']), 18)
                self.assertEqual(len(lookup['pose_slots']), pose_count)
                self.assertLessEqual(stats['property_count'], 5000)
                self.assertLessEqual(stats['enum_values'], 1000)
                self.assertLessEqual(stats['string_characters'], 120000)
                self.assertLessEqual(stats['max_object_depth'], 10)
                self.assertLessEqual(stats['schema_bytes'], 196608)

    def test_invalid_wire_values_not_repaired(self):
        r = fixture_request()
        original = encode_provider_decision_v3(r, canonical_response(r, None))
        for field, value in [('pose_slot', True), ('pose_slot', -1), ('force_slot', 30),
                             ('support_slot', 10000), ('part_id', 'foreign')]:
            wire = deepcopy(original); wire['decision'][field] = value
            self.assertFalse(schema_accepts(response_schema_v3(r), wire))
            with self.assertRaises(ValueError): expand_provider_decision_v3(r, wire)
        wire = deepcopy(original); wire['candidate_tail']['pick_1'] = 2
        with self.assertRaises(ValueError): expand_provider_decision_v3(r, wire)

    def test_evidence_selection_is_unique_exact_and_rejects_missing_flags(self):
        r = fixture_request()
        r['episode_feedback'] = [dict(evidence_id='ep1', status='UNKNOWN'), dict(evidence_id='ep2', status='PASS')]
        r['evidence_ids'] = ['ep1', 'ep2']
        response = canonical_response(r, None); response['choice']['evidence_ids'] = ['ep2']
        wire = encode_provider_decision_v3(r, response)
        self.assertEqual(wire['evidence_selection'], {'e0': False, 'e1': True})
        self.assertEqual(expand_provider_decision_v3(r, wire)['choice']['evidence_ids'], ['ep2'])
        del wire['evidence_selection']['e0']
        with self.assertRaises(ValueError): expand_provider_decision_v3(r, wire)

    def test_opaque_history_or_huge_support_domain_fail_before_schema_generation(self):
        r = fixture_request(); old = canonical_response(r, None)
        r['attempted_action_ids'] = [old['action_id']]; del r['attempted_actions']
        with self.assertRaisesRegex(ValueError, 'readable_attempted_actions'): build_domain(r)
        r = fixture_request(18); r['max_grippers'] = 18
        with self.assertRaisesRegex(ValueError, 'support_domain_too_large'): build_domain(r)

    def test_body_has_no_contradictory_v2_wire_instructions_or_priority_fields(self):
        r = fixture_request()
        body = build_provider_body(DEFAULTS, r, PROVIDER_CONTRACT_V3)
        for text in ('in candidate_order.', 'candidate_order[0]', 'choice contains its action'):
            self.assertNotIn(text, body['instructions'])
        self.assertEqual(body['text']['format']['name'], PROVIDER_CONTRACT_V3)
        data = json.loads(body['input'][0]['content'])
        self.assertEqual(data['request'], r)
        self.assertIn('Path fixes all other remaining parts', data['physics_scope'])
        for key in ('native_order', 'fallback_order', 'heuristic_score', 'recommended_order'):
            self.assertNotIn('"'+key+'"', json.dumps(body))


class ProviderV3ServiceTests(ExplicitActionServiceTests):
    def v3_service(self, transport, ledger=None, fallback_policy='legacy_v1'):
        self.serial += 1
        return ExplicitActionService(config(), ledger or self.root/('v3_%d.sqlite'%self.serial),
            transport=transport, provider_contract=PROVIDER_CONTRACT_V3, fallback_policy=fallback_policy)

    def v3_transport(self):
        def call(url, headers, body, timeout):
            r = json.loads(body['input'][0]['content'])['request']
            return provider(encode_provider_decision_v3(r, canonical_response(r, None)))
        return Mock(side_effect=call)

    def test_v3_transport_one_call_cache_and_decoded_worker_contract(self):
        transport = self.v3_transport(); service = self.v3_service(transport)
        r = fixture_request(); first = service.handle(r); second = service.handle(r)
        validate_response(r, first)
        self.assertEqual(transport.call_count, 1)
        self.assertTrue(second['metadata']['cached'])
        self.assertEqual(first['metadata']['provider_contract'], PROVIDER_CONTRACT_V3)
        self.assertEqual(first['metadata']['selected_part_source'], 'provider_decision_part')
        self.assertEqual(first['metadata']['provider_domain_audit']['candidate_parts_preserved'], 3)

    def test_injected_repeated_output_rejected_even_if_provider_ignores_schema(self):
        r = fixture_request(); old = canonical_response(r, None); remember(r, old['choice'])
        transport = Mock(return_value=provider(encode_provider_decision_v3(r, old)))
        result = self.v3_service(transport, fallback_policy='path_stage_v1').handle(r)
        self.assertEqual(transport.call_count, 1)
        self.assertTrue(result['metadata']['action_repeated'])
        self.assertEqual(result['metadata']['decision_validation_error'], 'action_already_attempted')
        self.assertEqual(result['metadata']['fallback_policy'], 'path_stage_v1')
        self.assertNotEqual(path_signature(r, result['choice']), path_signature(r, old['choice']))
        validate_response(r, result)

    def test_v2_cache_cannot_be_reused_as_v3_without_new_request_identity(self):
        ledger = self.root/'cross_version.sqlite'; transport = self.transport()
        self.service(transport, ledger=ledger).handle(request())
        v3_transport = self.v3_transport()
        result = self.v3_service(v3_transport, ledger=ledger).handle(request())
        self.assertEqual(result['fallback_reason'], 'request_id_conflict')
        v3_transport.assert_not_called()

    def test_opaque_attempts_rejected_before_paid_reservation(self):
        r = fixture_request(); old = canonical_response(r, None)
        r['attempted_action_ids'] = [old['action_id']]; del r['attempted_actions']
        transport = self.v3_transport(); result = self.v3_service(transport).handle(r)
        transport.assert_not_called()
        self.assertFalse(result['metadata']['attempted'])
        self.assertEqual(result['metadata']['provider_preflight_error'], 'provider_requires_readable_attempted_actions')


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(ProviderV3CodecTests))
    suite.addTests(ProviderV3ServiceTests(name) for name in ProviderV3ServiceTests.__dict__ if name.startswith('test_'))
    return suite


if __name__ == '__main__':
    unittest.main()
