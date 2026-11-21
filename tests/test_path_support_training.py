"""Order-training packet fixtures. No model, verifier, or source CAD files are used."""
from copy import deepcopy
import json
import unittest

from orchestrator.path_support_training import observed_proposal, freeze, root_decision
from orchestrator.sequence_ranker import PROTOCOL
from tests.test_independent_ranker import MeshFixture
from tests.test_proposal_support_score import PARTS, proposal
from worker.independent_features import describe_loaded, filter_features
from worker.proposal_support_score import score_plan
from worker.task_memory import task_context


CAD = {part + '.obj': 'a'*64 for part in PARTS}


def root():
    geometry = describe_loaded({part: {'mesh': MeshFixture(), 'transform': 'final'} for part in PARTS},
                               CAD, 'training-product', 'b'*40, None, {'seed': 0})
    return dict(schema_version=1, kind='rank_request', timeout_seconds=4,
                request=dict(run_id='fixture', request_id='old', state_id='old:state', assembled_parts=PARTS,
                             candidate_ids=list(PARTS), base_part=None),
                features=dict(protocol=PROTOCOL, cad=filter_features(geometry, PARTS), memory_context=task_context(geometry),
                              episode_feedback=[], validated_repair_skills=[]))


def record(first='a', counts=(4, 3, 2, 1), seed=0):
    # A bijection only changes which numeric/letter ID the fixture starts with.
    p = proposal(list(counts), cap=5)
    def relabel(item):
        if isinstance(item, dict):
            return {k: relabel(v) for k,v in item.items()}
        if isinstance(item, list):
            return [relabel(v) for v in item]
        if item == 'a': return first
        if item == first: return 'a'
        return item
    p = relabel(p)
    p['remaining_parts'].sort()
    p['support_score'] = score_plan(p, support_cap='auto')
    params = dict(seed=seed, budget=32, max_poses=128, plan_seconds=420, audit_seconds=480)
    if first != 'a':
        params['prescribed_first_part'] = first
    settings = dict(profile=p['profile'], assembly='/fixture/training-product', input_sha256=CAD,
                    base_part=None, pose_reuse=0, planner='dfs', terminal='original_mass_rule',
                    cache_policy='readonly', asap_commit='b'*40, redmax_binary_sha256='c'*64,
                    native_physics_constants={}, asap_path_sources={}, parameters=params)
    return observed_proposal(p, settings, first, 'fixture', {'result': 'd'*64}, audit_seconds=480)


class PathTrainingTests(unittest.TestCase):
    def test_memory_and_body_keep_observations_for_all_branches_without_native_verdicts(self):
        rows = [record('b'), record('a'), record('c', counts=())]
        frozen = freeze('training-product', CAD, rows)
        self.assertEqual(frozen['comparison'], 'tied_on_scored_chains')
        self.assertEqual(frozen['conditional_count'], 1)
        body = root_decision(root(), frozen, split='training')
        data = json.loads(body['input'][0]['content'])
        self.assertEqual(data['request']['candidate_ids'], list(PARTS))
        self.assertNotEqual(data['request']['request_id'], 'old')
        cases = data['observed_chains']
        self.assertEqual([row['first_part'] for row in cases], ['a','b','c'])
        self.assertEqual([row['score'] for row in cases], [[4,10],[4,10],None])
        self.assertFalse(cases[2]['complete'])
        self.assertEqual(cases[0]['root_policy'], 'unrestricted_heur_out')
        self.assertEqual(cases[1]['root_policy'], 'prespecified_first')
        self.assertEqual(cases[1]['limits']['seed'], 0)
        self.assertEqual(data['scoped_lesson'], frozen['scoped_lesson'])
        self.assertEqual([r['first_part'] for r in data['scoped_lesson']['scored']], ['a','b'])
        self.assertEqual(data['scoped_lesson']['unscored'],[{'first_part':'c','evidence_id':cases[2]['evidence_id']}])
        self.assertEqual(data['scoped_lesson']['python_skill'], 'NOT_GENERATED')
        self.assertEqual(data['scoped_lesson']['evidence_ids'], [r['evidence_id'] for r in cases])
        self.assertNotIn('native_full_disassembly', body['input'][0]['content'])
        self.assertNotIn('solver_settings', body['input'][0]['content'])
        schema = body['text']['format']['schema']['properties']
        self.assertEqual(schema['evidence_ids']['items']['enum'], [r['evidence_id'] for r in cases])
        self.assertEqual(schema['decision_kind']['enum'], ['direct'])

    def test_different_settings_and_partial_are_not_counted_as_objective_improvement(self):
        for other, expected in [(record('b', (2,1,0,0)), 'different_scores_on_sampled_chains'),
                                (record('b', (2,1,0,0),seed=1), 'different_check_settings'),
                                (record('b', ('UNKNOWN',3,2,1)), 'insufficient_scored_chains')]:
            frozen = freeze('training-product', CAD, [record(), other])
            self.assertEqual(frozen['comparison'], expected)

    def test_evaluation_and_other_cad_cannot_reuse_training_product_ids(self):
        memory = freeze('training-product', CAD, [record()])
        with self.assertRaises(ValueError):
            root_decision(root(), memory, split='evaluation')
        with self.assertRaises(ValueError):
            freeze('other-product', CAD, [record()])
        different = {**CAD, 'a.obj': '9'*64}
        with self.assertRaises(ValueError):
            freeze('training-product', different, [record()])
        mixed = deepcopy(memory)
        mixed['observations'][0]['score'] = [0,0]
        with self.assertRaises(ValueError):
            root_decision(root(), mixed, split='training')

    def test_partial_or_preloaded_root_rejected(self):
        memory = freeze('training-product', CAD, [record()])
        for attr in ('candidate_ids', 'assembled_parts', 'episode_feedback'):
            req = root()
            if attr == 'episode_feedback':
                req['features'][attr] = [{'event_id':'old'}]
            else:
                req['request'][attr] = PARTS[:-1]
            with self.assertRaises(ValueError):
                root_decision(req, memory, split='training')
