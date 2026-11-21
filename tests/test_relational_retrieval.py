"""Offline relevance and provenance checks; no physical validation claims."""
from copy import deepcopy
import unittest

from orchestrator.relational_retrieval import search_cards


def context(product='A', target='p0', *, extents=(10, 1, 6), degree=6,
            delta=1, isolated=1, profile='path-only', unit='mm'):
    return dict(product_id=product, target_part_id=target,
                shape=dict(extents_xyz=list(extents), length_unit=unit, frame='native CAD'),
                neighbors=[dict(part_id='near', shape=dict(extents_xyz=[10,3,.9]),
                                center_delta_xyz=[0,0,-3], contact=True)],
                topology=dict(contact_degree=degree, degree_fraction=degree/11,
                              component_delta=delta, remainder_isolated_nodes=isolated),
                conditions=dict(verifier_domain='RedMax', verifier_profile=profile,
                                support_policy='auto', fixed_base_present=False, pose=None))


def card(identity, ctx=None, kind='experience', evidence=(), **content):
    return dict(card_id=identity, kind=kind, context=ctx or {}, evidence_ids=list(evidence),
                origin=dict(path='observations.json', record_pointer='/'+identity, source_hash='hash'),
                content=content, status='CANDIDATE' if kind!='experience' else 'RECORDED')


class RelationalRetrievalTests(unittest.TestCase):
    def test_shape_relation_match_without_part_count_or_ids(self):
        cards=[card('far', context('same-product', 'same-id', extents=(2,2,2), degree=1, delta=0)),
               card('near', context('other-product', 'different-id', extents=(10,1.2,5.8)))]
        result=search_cards(cards, context('same-product', 'same-id'))
        self.assertEqual(result['results'][0]['card_id'],'near')
        self.assertFalse(result['results'][0]['execution_authorized'])
        self.assertFalse(result['audit']['outcome_used_for_ranking'])

    def test_numeric_tie_scores_do_not_depend_on_ids_or_outcome(self):
        a=card('a',context('p','part1'), outcome='PASS')
        b=card('b',context('different','part2'), outcome='FAILED')
        result=search_cards([a,b],context('q','target'))['results']
        self.assertEqual(result[0]['score'],result[1]['score'])
        a['card_id']='z'; a['context']['target_part_id']='renamed'
        other=search_cards([a,b],context('q','renamed'))['results']
        self.assertEqual([r['score'] for r in result],[r['score'] for r in other])

    def test_axis_permutation_and_known_unit_conversion(self):
        a=context(extents=(10,1,6),unit='mm')
        b=context(extents=(.6,1,.1),unit='cm')
        item=search_cards([card('b',b)],a)['results'][0]
        self.assertAlmostEqual(item['channels']['shape']['distance'],0.)

    def test_unknown_units_do_not_make_absolute_size_match(self):
        a=context(unit='input_CAD_unit_not_confirmed_mm')
        b=context(unit='input_CAD_unit_not_confirmed_mm',extents=(100,10,60))
        item=search_cards([card('b',b)],a)['results'][0]
        self.assertEqual(item['channels']['shape']['matched_fields'],['bbox_sorted_ratio'])
        self.assertIn('shape.absolute_size_comparable_units',item['missing_features'])

    def test_null_is_not_a_condition_or_numeric_match(self):
        empty=card('empty',dict(shape={},conditions=dict(verifier_domain=None),topology={}))
        self.assertEqual(search_cards([empty],{})['results'],[])
        result=search_cards([card('known',context()),empty],context())
        self.assertEqual([r['card_id'] for r in result['results']],['known'])

    def test_condition_mismatch_visible_but_does_not_prune_experience(self):
        b=context(profile='native')
        b['conditions']['fixed_base_present']=True
        result=search_cards([card('b',b)],context())['results'][0]
        differences={r['field'] for r in result['condition_differences']}
        self.assertEqual(differences,{'verifier_profile','fixed_base_present'})
        self.assertFalse(result['execution_authorized'])

    def test_base_id_equality_not_required_or_used(self):
        a,b=context(),context()
        del a['conditions']['fixed_base_present'];del b['conditions']['fixed_base_present']
        a['conditions']['fixed_base']='base-a';b['conditions']['fixed_base']='base-b'
        self.assertEqual(search_cards([card('b',b)],a)['results'][0]['condition_differences'],[])

    def test_source_family_not_three_independent_observations(self):
        e=card('e',context())
        lesson=card('l',kind='lesson',evidence=['e'],statement='conditional claim')
        skill=card('s',kind='skill',evidence=['l'],description='code candidate')
        result=search_cards([e,lesson,skill],context())
        self.assertEqual(len(result['results']),1)
        self.assertEqual(result['results'][0]['card_id'],'e')
        self.assertEqual(set(result['results'][0]['related_card_ids']),{'l','s'})
        self.assertEqual(result['audit']['returned_unique_evidence_groups'],1)
        direct=search_cards([e,lesson,skill],dict(context=context(),kind='skill'))
        self.assertEqual(direct['results'][0]['card_id'],'s')
        self.assertEqual(direct['results'][0]['matched_evidence_card_id'],'e')

    def test_distinct_steps_share_evidence_count_but_keep_their_contexts(self):
        parent=card('episode',context())
        step=card('step',context(extents=(1,1,1)),evidence=['episode'])
        parent['evidence_group_id']=step['evidence_group_id']='one_episode'
        result=search_cards([parent,step],context())
        self.assertEqual(len(result['results']),2)
        self.assertEqual(result['audit']['returned_unique_evidence_groups'],1)
        by_id={r['card_id']:r for r in result['results']}
        self.assertGreater(by_id['step']['channels']['shape']['distance'],0)
        self.assertEqual(by_id['step']['matched_evidence_card_id'],'step')

    def test_empty_conditions_and_numeric_labels_do_not_match(self):
        cards=[card('a06',dict(conditions=dict(support_policy={})),description='a06 e12')]
        self.assertEqual(search_cards(cards,dict(text='a06',conditions=dict(support_policy={})))['results'],[])

    def test_diversity_does_not_drop_later_states_or_promote_fixture(self):
        a,b,c=card('a',context()),card('b',context()),card('c',context(extents=(10,2,6)))
        a['evidence_group_id']=b['evidence_group_id']='same_episode'
        c['evidence_group_id']='other_episode'
        fixture=card('fixture',context());fixture['fixture']=True
        rows=search_cards([a,b,c,fixture],context(),limit=2)
        self.assertEqual([v['card_id'] for v in rows['results']],['a','c'])
        tail=search_cards([a,b,c,fixture],context(),limit=2,offset=2)
        self.assertEqual([v['card_id'] for v in tail['results']],['b'])

    def test_more_citations_do_not_inflate_score(self):
        cards=[card('e1',context()),card('e2',context()),
               card('one',kind='skill',evidence=['e1']),card('two',kind='skill',evidence=['e1','e2'])]
        results=search_cards(cards,dict(context=context(),kind='skill'))['results']
        self.assertEqual(results[0]['score'],results[1]['score'])

    def test_excluded_product_cannot_reenter_through_skill_evidence(self):
        cards=[card('e',context('held-out')),card('s',kind='skill',evidence=['e'])]
        result=search_cards(cards,context(),exclude_products=['held-out'])
        self.assertEqual(result['results'],[])
        self.assertEqual(result['audit']['excluded_count'],2)

    def test_mixed_evidence_lesson_is_excluded_as_a_whole(self):
        cards=[card('held',context('held-out')),card('train',context('training')),
               card('lesson',kind='lesson',evidence=['held','train'],statement='held-out answer'),
               card('skill',kind='skill',evidence=['lesson'])]
        result=search_cards(cards,context(),exclude_products=['held-out'])
        self.assertEqual([r['card_id'] for r in result['results']],['train'])
        self.assertEqual(result['audit']['excluded_count'],3)

    def test_explicit_no_base_is_different_from_unknown_and_fixed_base(self):
        a,b=context(),context()
        for ctx,base in ((a,'none'),(b,'base')):
            del ctx['conditions']['fixed_base_present']
            ctx['conditions'].update(fixed_base=base,known_fields=['fixed_base'])
        result=search_cards([card('b',b)],a)['results'][0]
        self.assertIn('fixed_base_present',{v['field'] for v in result['condition_differences']})

    def test_allowed_cards_applies_to_resolved_evidence(self):
        cards=[card('e',context()),card('s',kind='skill',evidence=['e'])]
        self.assertEqual(search_cards(cards,context(),allowed_card_ids=['s'])['results'],[])

    def test_cycles_broken_and_missing_links_not_invented(self):
        cards=[card('a',kind='lesson',evidence=['b']),card('b',kind='skill',evidence=['a','absent'])]
        self.assertEqual(search_cards(cards,context())['results'],[])

    def test_keyword_search_does_not_scan_raw_ids_or_source_code(self):
        cards=[card('secret',kind='skill',description='support recovery',source='secret_code_xyz')]
        self.assertEqual(search_cards(cards,dict(text='secret_code_xyz'))['results'],[])
        self.assertEqual(search_cards(cards,dict(text='support'))['results'][0]['card_id'],'secret')

    def test_success_and_failure_from_same_part_are_preserved(self):
        cards=[card('pass',context(),outcome='PASS'),card('fail',context(),outcome='UNKNOWN')]
        self.assertEqual(len(search_cards(cards,context())['results']),2)

    def test_input_immutable_and_result_independent(self):
        cards=[card('e',context())]; query=context(); before=deepcopy((cards,query))
        result=search_cards(cards,query)
        result['results'][0]['channels']['shape']['distance']=999
        self.assertEqual((cards,query),before)
        self.assertNotEqual(search_cards(cards,query)['results'][0]['channels']['shape']['distance'],999)

    def test_pagination_and_invalid_contract(self):
        cards=[card(str(i),context()) for i in range(7)]
        first=search_cards(cards,context(),limit=3)
        second=search_cards(cards,context(),limit=3,offset=first['next_offset'])
        self.assertFalse({r['card_id'] for r in first['results']} & {r['card_id'] for r in second['results']})
        for bad in (0,33,False):
            with self.assertRaises(ValueError): search_cards(cards,context(),limit=bad)
        with self.assertRaises(ValueError): search_cards([cards[0],dict(cards[0],status='different')],context())

    def test_model_can_change_search_channels_without_losing_condition_differences(self):
        a=card('shape_match',context(degree=1,delta=0,isolated=0,profile='other-profile'))
        b=card('relation_match',context(extents=(10,4,6)))
        shape=search_cards([a,b],dict(context=context(),channels=['shape']))
        relation=search_cards([a,b],dict(context=context(),channels=['relations']))
        self.assertEqual(shape['results'][0]['card_id'],'shape_match')
        self.assertEqual(relation['results'][0]['card_id'],'relation_match')
        self.assertEqual(list(shape['results'][0]['channels']),['shape'])
        self.assertIn('verifier_profile',{d['field'] for d in shape['results'][0]['condition_differences']})
        empty=search_cards([card('text_only',description='support')],dict(text='support',channels=['shape']))
        self.assertEqual(empty['results'],[])
        for bad in ([],['shape','shape'],['invalid'],'shape',None):
            with self.assertRaises(ValueError):
                search_cards([a],dict(context=context(),channels=bad))


if __name__=='__main__':
    unittest.main()
