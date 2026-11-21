"""Offline evidence projection/codec/service checks; network and physics absent."""
from copy import deepcopy
from contextlib import closing
import json
import sqlite3
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_explicit_provider_v3 import schema_accepts
from test_explicit_provider_v4 import round_request, observed
from test_explicit_action_service import ExplicitActionServiceTests, config, provider
from orchestrator.explicit_action_service import ExplicitActionService, build_provider_body
from orchestrator.explicit_action_provider_v4 import (PROVIDER_CONTRACT_V4,
    provider_input_v4, encode_provider_decision_v4)
from orchestrator.explicit_action_provider_v5 import (PROVIDER_CONTRACT_V5, REVIEW_VALUES,
    project_request_v5, restore_request_v5, provider_input_v5, response_schema_v5,
    decision_brief_v5, same_mesh_id_correspondence, review_ids_v5,
    encode_provider_decision_v5, expand_provider_decision_v5)
from orchestrator.llm_ranker import DEFAULTS
from worker.explicit_action_contract import canonical, digest, canonical_response
from worker.path_probe_round import canonical_path_round_response, validate_path_round_response


def with_evidence(parts=3, poses=2, memory_count=2):
    r = round_request(parts, poses)
    r['task_context']['task'] = {'input_sha256': {p+'.obj': digest(p) for p in r['remaining_parts']}}
    r['task_context']['fixture_context'] = 'x'*2500
    for i in range(memory_count):
        r['memory_references'].append(dict(evidence_id='memory:'+str(i),
            source_task_context=deepcopy(r['task_context']), source_context_sha256=digest(r['task_context']),
            source_remaining_parts=list(r['remaining_parts']), source_pose_matrix=r['pose_catalog'][0]['matrix'],
            source_choice=canonical_response(r, None)['choice'],
            same_task=True, same_conditions=True, reference_only=False, current_part_id_mapping=None,
            current_attempt_verified=False, hard_pruning=False, physics_reuse_allowed=False,
            parent_failure_evidence_id=None if i == 0 else 'memory:0',
            status='UNKNOWN', path_success=True, stability_success=False, raw_stability_parts=['1']))
        r['evidence_ids'].append('memory:'+str(i))
    r['episode_feedback'] = [dict(evidence_id='episode:0', status='UNKNOWN',
        choice=canonical_response(r, None)['choice'], path_success=False,
        stability_success=None, raw_stability_parts=None)]
    r['evidence_ids'].append('episode:0')
    return r


def raw_from_body(body):
    payload = json.loads(body['input'][0]['content'])
    return restore_request_v5({k: payload[k] for k in ('projection_version','request','shared_source_contexts')})


class ProviderV5ProjectionTests(unittest.TestCase):
    def test_lossless_current_context_dedupe_preserves_every_other_field_and_parent(self):
        r = with_evidence(); original = deepcopy(r)
        bundle, audit = project_request_v5(r)
        self.assertEqual(restore_request_v5(bundle), original)
        self.assertEqual(r, original)
        self.assertEqual(audit['source_context_occurrences'], 2)
        self.assertEqual(audit['unique_source_contexts'], 1)
        self.assertEqual(audit['shared_contexts_using_current_task'], 1)
        self.assertGreater(audit['bytes_saved'], 4000)
        for before, after in zip(r['memory_references'], bundle['request']['memory_references']):
            self.assertEqual({k:v for k,v in before.items() if k != 'source_task_context'},
                             {k:v for k,v in after.items() if k != 'source_task_context'})

    def test_different_contexts_and_missing_legacy_source_context_restore(self):
        r = with_evidence(memory_count=3)
        ref = r['memory_references'][1]; ref['source_task_context']['task']['seed'] = 9
        ref['source_context_sha256'] = digest(ref['source_task_context'])
        del r['memory_references'][2]['source_task_context']
        bundle, audit = project_request_v5(r)
        self.assertEqual(restore_request_v5(bundle), r)
        self.assertEqual(audit['unique_source_contexts'], 2)
        self.assertEqual(audit['shared_contexts_using_current_task'], 1)
        self.assertEqual(sum('value' in v for v in bundle['shared_source_contexts'].values()), 1)

    def test_context_digest_mismatch_is_rejected_not_repaired(self):
        r = with_evidence(); r['memory_references'][0]['source_context_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source_context_digest_mismatch'):
            project_request_v5(r)

    def test_restoration_rejects_tampered_context_unknown_or_unused_ref(self):
        for change in ('tamper', 'unknown', 'unused'):
            bundle, _ = project_request_v5(with_evidence())
            if change == 'tamper': bundle['request']['task_context']['fixture_context'] = 'different'
            elif change == 'unknown':
                bundle['request']['memory_references'][0]['source_task_context']['shared_source_context_sha256'] = '0'*64
            else: bundle['shared_source_contexts']['0'*64] = {'value': {}}
            with self.assertRaises(ValueError): restore_request_v5(bundle)

    def test_empty_memory_is_valid_without_citations_or_context_invention(self):
        r = round_request(); bundle, audit = project_request_v5(r)
        self.assertEqual(bundle['shared_source_contexts'], {})
        self.assertEqual(restore_request_v5(bundle), r)
        self.assertEqual(decision_brief_v5(r)['evidence'], [])
        wire = encode_provider_decision_v5(r, canonical_response(r, None))
        self.assertEqual(wire['evidence_review'], {})
        self.assertEqual(expand_provider_decision_v5(r, wire)['choice']['evidence_ids'], [])

    def test_hash_proof_is_same_id_geometry_only_and_does_not_rewrite_safety_flags(self):
        r = with_evidence(); ref = r['memory_references'][0]; before = deepcopy(ref)
        proof = same_mesh_id_correspondence(r, ref)
        self.assertEqual(set(proof['parts']), set(r['remaining_parts']))
        for part, pair in proof['parts'].items():
            self.assertEqual(pair['current_part_id'], part)
            self.assertEqual(pair['obj_sha256'], r['task_context']['task']['input_sha256'][part+'.obj'])
        self.assertFalse(proof['physics_reuse_allowed'])
        self.assertFalse(proof['current_attempt_verified'])
        self.assertEqual(ref, before); self.assertIsNone(ref['current_part_id_mapping'])
        ref['source_task_context']['task']['input_sha256']['0.obj'] = 'a'*64
        self.assertNotIn('0', same_mesh_id_correspondence(r, ref)['parts'])
        ref['source_task_context']['task']['input_sha256'] = {'new_id.obj': digest('0')}
        self.assertIsNone(same_mesh_id_correspondence(r, ref))

    def test_scope_and_missing_diagnostic_are_not_upgraded_to_fact(self):
        r = with_evidence(); ref = r['memory_references'][0]
        ref.update(reference_only=True, same_conditions=False)
        brief = decision_brief_v5(r)
        item = next(row for row in brief['evidence'] if row['evidence_id'] == ref['evidence_id'])
        source = item['observations'][0]
        self.assertTrue(source['scope']['reference_only'])
        self.assertFalse(source['scope']['same_conditions'])
        self.assertIsNone(source['stability_diagnostics'])
        self.assertEqual(source['raw_stability_parts'], ['1'])
        self.assertNotIn('failure_cause', source)
        self.assertEqual(brief['review_semantics'], 'model_self_report_not_proof_of_attention_or_causal_effect')

    def test_compact_diagnostic_is_allowlisted_and_invalid_facts_fail_closed(self):
        from worker.stability_diagnostics import StabilityObservation, project_stability_diagnostics
        r = with_evidence(); diagnostic = deepcopy(StabilityObservation().data)
        diagnostic['untrusted_extra_text'] = 'not a diagnostic fact'
        r['episode_feedback'][0]['stability_diagnostics'] = diagnostic
        brief = decision_brief_v5(r)['evidence'][0]['observations'][0]
        self.assertEqual(brief['stability_diagnostics'],project_stability_diagnostics(diagnostic))
        self.assertNotIn('untrusted_extra_text',brief['stability_diagnostics'])
        diagnostic['extra_physics_calls'] = 1
        with self.assertRaisesRegex(ValueError,'stability_observer_must_not_query'):
            decision_brief_v5(r)
        with self.assertRaisesRegex(ValueError,'stability_observer_must_not_query'):
            provider_input_v5(r)

    def test_gravity_identity_and_ninety_degree_pose_is_algebra_only(self):
        r = round_request(3, 2)
        r['pose_catalog'][1]['matrix'] = [[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]
        gravity = decision_brief_v5(r)['pose_gravity']
        self.assertEqual(gravity[0]['gravity_unit_in_input_frame'], [0,0,-1])
        self.assertEqual(gravity[1]['gravity_unit_in_input_frame'], [0,1,0])
        self.assertEqual([p['pose_id'] for p in gravity], [p['pose_id'] for p in r['pose_catalog']])

    def test_body_remains_lossless_bounded_and_old_v4_unchanged(self):
        r = with_evidence(memory_count=8); original = deepcopy(r)
        cfg = dict(DEFAULTS, max_output_tokens=1536)
        old = build_provider_body(cfg, r, PROVIDER_CONTRACT_V4)
        new = build_provider_body(cfg, r, PROVIDER_CONTRACT_V5)
        self.assertEqual(raw_from_body(new), r)
        self.assertEqual(r, original)
        self.assertEqual(new['max_output_tokens'], 1536)
        self.assertLess(len(canonical(new)), len(canonical(old)))
        self.assertEqual(build_provider_body(cfg, r, PROVIDER_CONTRACT_V4), old)
        self.assertIn('No citation or review is mandatory', new['instructions'])
        self.assertNotIn('evidence_selection', new['instructions'])
        self.assertNotIn('All legal untried combinations remain representable.', new['instructions'])
        self.assertTrue(new['input'][0]['content'].startswith('{"decision_brief":'))


class ProviderV5CodecTests(unittest.TestCase):
    def test_decision_schema_and_candidate_tail_identical_to_v4(self):
        r = with_evidence(); observed(r, canonical_response(r, None)['choice'])
        lookup4, schema4, audit4 = provider_input_v4(r)
        lookup5, schema5, audit5 = provider_input_v5(r)
        self.assertEqual(lookup4, lookup5)
        for key in ('request_id', 'state_id', 'decision', 'candidate_tail'):
            self.assertEqual(schema4['properties'][key], schema5['properties'][key])
        for key in ('active_action_serializations','deferred_action_serializations','available_action_serializations'):
            self.assertEqual(audit4[key], audit5[key])
        self.assertNotIn('evidence_selection', schema5['properties'])

    def test_three_review_states_decode_only_used_and_keep_action_identity(self):
        r = with_evidence(); response = canonical_response(r, None)
        identities = sorted(r['evidence_ids']); response['choice']['evidence_ids'] = identities[:1]
        review = {'e'+str(i): status for i, status in enumerate(REVIEW_VALUES)}
        wire = encode_provider_decision_v5(r, response, review)
        self.assertTrue(schema_accepts(response_schema_v5(r), wire))
        decoded = expand_provider_decision_v5(r, wire)
        self.assertEqual(decoded['choice']['evidence_ids'], identities[:1])
        self.assertEqual(decoded['action_id'], response['action_id'])
        self.assertEqual(review_ids_v5(r, review)['reviewed_not_used'], identities[1:2])

    def test_every_individual_slot_maps_exact_id(self):
        r = with_evidence(memory_count=8)
        for i, identity in enumerate(sorted(r['evidence_ids'])):
            wire = encode_provider_decision_v5(r, canonical_response(r, None))
            wire['evidence_review']['e'+str(i)] = 'used_for_choice'
            self.assertEqual(expand_provider_decision_v5(r, wire)['choice']['evidence_ids'], [identity])

    def test_invalid_review_missing_extra_boolean_and_unknown_values_rejected(self):
        r = with_evidence()
        for bad in ({}, {'e0':'used_for_choice'}, {'e0':True,'e1':'not_reviewed','e2':'not_reviewed'},
                    {'e0':'inferred','e1':'not_reviewed','e2':'not_reviewed'},
                    {'e0':'not_reviewed','e1':'not_reviewed','e2':'not_reviewed','e3':'not_reviewed'}):
            wire = encode_provider_decision_v5(r, canonical_response(r, None)); wire['evidence_review'] = bad
            self.assertFalse(schema_accepts(response_schema_v5(r), wire))
            with self.assertRaisesRegex(ValueError,'invalid_provider_evidence_review'):
                expand_provider_decision_v5(r, wire)

    def test_review_does_not_make_exact_or_deferred_action_legal(self):
        r = with_evidence(); old = canonical_response(r, None); observed(r, old['choice'])
        wire = encode_provider_decision_v5(r, old); wire['evidence_review']['e0'] = 'used_for_choice'
        with self.assertRaisesRegex(ValueError, 'action_already_attempted'):
            expand_provider_decision_v5(r, wire)
        old['choice']['support_ids'] = ['1']; wire = encode_provider_decision_v5(r, old)
        with self.assertRaisesRegex(ValueError, 'action_deferred_for_probe_round'):
            expand_provider_decision_v5(r, wire)

    def test_largest_18_part_128_pose_history_with_evidence_stays_schema_bounded(self):
        r = with_evidence(18,128,8)
        for i in range(60):
            action = canonical_path_round_response(r, None)['choice']; observed(r,action)
        _, _, audit = provider_input_v5(r)
        self.assertEqual(audit['candidate_parts_preserved'],18)
        self.assertEqual(audit['pose_entries_preserved'],128)
        self.assertTrue(audit['active_plus_deferred_equals_untried'])
        for field, limit in (('property_count',5000),('enum_values',1000),('string_characters',120000),
                             ('max_object_depth',10),('schema_bytes',196608)):
            self.assertLessEqual(audit[field],limit)


class ProviderV5ServiceTests(ExplicitActionServiceTests):
    def v5_service(self, transport, ledger=None):
        self.serial += 1
        return ExplicitActionService(config(max_output_tokens=1536), ledger or self.root/('v5_%d.sqlite'%self.serial),
            transport=transport, provider_contract=PROVIDER_CONTRACT_V5, fallback_policy='path_stage_v1')

    def v5_transport(self, change=None):
        def call(url,headers,body,timeout):
            r = raw_from_body(body); wire = encode_provider_decision_v5(r,canonical_path_round_response(r,None))
            if change: change(wire)
            return provider(wire)
        return Mock(side_effect=call)

    def test_validated_wire_is_exact_stored_value_and_review_is_self_report(self):
        r = with_evidence()
        transport = self.v5_transport(lambda wire:wire['evidence_review'].update(e0='used_for_choice',e1='reviewed_not_used'))
        service = self.v5_service(transport); response = service.handle(r); cached = service.handle(r)
        self.assertEqual(transport.call_count,1)
        self.assertTrue(cached['metadata']['cached'])
        meta = response['metadata']
        self.assertTrue(meta['validated_provider_wire_recorded'])
        self.assertFalse(meta['raw_response_recorded'])
        self.assertEqual(meta['evidence_review_kind'],'model_self_report')
        self.assertEqual(meta['evidence_review_ids']['used_for_choice'],response['choice']['evidence_ids'])
        decoded = expand_provider_decision_v5(r,meta['validated_provider_wire'])
        self.assertEqual(decoded['choice'],response['choice'])
        self.assertEqual(decoded['candidate_order'],response['candidate_order'])
        self.assertEqual(service.decisions[0]['request'],r)

    def test_all_not_reviewed_is_successful_valid_zero_citation_response(self):
        r = with_evidence(); response = self.v5_service(self.v5_transport()).handle(r)
        self.assertIsNone(response['fallback_reason'])
        self.assertEqual(response['choice']['evidence_ids'],[])
        self.assertEqual(set(response['metadata']['evidence_review'].values()),{'not_reviewed'})

    def test_bad_review_one_transport_safe_fallback_no_stored_invalid_wire(self):
        r = with_evidence(); observed(r,canonical_response(r,None)['choice'])
        transport = self.v5_transport(lambda wire:wire['evidence_review'].update(e0=True))
        response = self.v5_service(transport).handle(r)
        self.assertEqual(transport.call_count,1)
        self.assertEqual(response['metadata']['decision_validation_error'],'invalid_provider_evidence_review')
        self.assertEqual(response['metadata']['selection_source'],'deterministic_fallback')
        self.assertFalse(response['metadata']['validated_provider_wire_recorded'])
        self.assertNotIn('validated_provider_wire',response['metadata'])
        self.assertNotIn('evidence_review',response['metadata'])
        validate_path_round_response(r,response)

    def test_exact_repeat_keeps_specific_diagnostic_and_does_not_store_wire(self):
        r = with_evidence(); old = canonical_response(r,None); observed(r,old['choice'])
        transport = Mock(return_value=provider(encode_provider_decision_v5(r,old)))
        response = self.v5_service(transport).handle(r)
        self.assertTrue(response['metadata']['action_repeated'])
        self.assertEqual(response['metadata']['decision_validation_error'],'action_already_attempted')
        self.assertNotIn('validated_provider_wire',response['metadata'])

    def test_bad_context_digest_and_missing_observation_reject_before_transport_or_reservation(self):
        for bad in ('digest','observations'):
            r = with_evidence(); transport = self.v5_transport(); ledger = self.root/(bad+'.sqlite')
            if bad == 'digest': r['memory_references'][0]['source_context_sha256'] = '0'*64
            else: del r['path_stage_observations']
            with self.assertRaises(ValueError): self.v5_service(transport,ledger).handle(r)
            transport.assert_not_called()
            # RankService initializes an empty ledger in its constructor, but
            # malformed preparation must never create a paid reservation row.
            with closing(sqlite3.connect(str(ledger))) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM rank_requests').fetchone()[0],0)

    def test_v4_request_cache_conflicts_instead_of_being_reinterpreted(self):
        r = with_evidence(); ledger = self.root/'v4_v5.sqlite'
        old_transport = Mock(return_value=provider(encode_provider_decision_v4(r,canonical_response(r,None))))
        old = ExplicitActionService(config(max_output_tokens=1536),ledger,transport=old_transport,
                                    provider_contract=PROVIDER_CONTRACT_V4,fallback_policy='path_stage_v1')
        old.handle(r)
        new_transport = self.v5_transport(); response = self.v5_service(new_transport,ledger).handle(r)
        self.assertEqual(response['fallback_reason'],'request_id_conflict'); new_transport.assert_not_called()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(ProviderV5ProjectionTests))
    suite.addTests(loader.loadTestsFromTestCase(ProviderV5CodecTests))
    suite.addTests(ProviderV5ServiceTests(name) for name in ProviderV5ServiceTests.__dict__ if name.startswith('test_'))
    return suite


if __name__ == '__main__': unittest.main()
