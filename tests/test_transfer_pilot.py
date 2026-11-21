"""Synthetic unit tests only; neither API nor real RedMax runs here."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_skill_program import request,append_observation,choice
from worker.explicit_action_contract import validate_response
from worker.path_probe_round import validate_path_round_response
from worker.evaluation_budget import PhysicsBudget,EvaluationLimit
from worker.transfer_retrieval import select_references
from orchestrator.transfer_policy import choose_action,memory_preferences
from orchestrator.transfer_services import compact_selection,restore_selection,bounded_body
from orchestrator.transfer_services import pack_packet,unpack_packet


class Sim:
    def forward(self,n,*a,**kw):return n


class PathChecker:
    max_time=60
    def __init__(self):self.sim=Sim();self.parts_fix=[];self.part_move='a'
    def check_success(self,*a,**kw):self.sim.forward(7);return True,[[0,0,0]]


class StabilityChecker(PathChecker):
    def check_success(self,*a,**kw):self.sim.forward(4);return False,['a']


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.net=patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('API_FORBIDDEN'));self.net.start();self.addCleanup(self.net.stop)

    def test_rule_is_pure_repeatable_and_complete(self):
        r=request();before=deepcopy(r);a=choose_action(r);b=choose_action(r)
        self.assertEqual(a,b);self.assertEqual(r,before);validate_response(r,a)
        self.assertEqual(set(a['candidate_order']),set(r['candidate_ids']))

    def test_packed_fields_ids_and_order_roundtrip(self):
        r=append_observation(request());value={'request':r,'again':deepcopy(r),'ids':['x'*64,'x'*64],
            'rows':[{'a':1,'b':None},{'b':False,'a':2}], 'nonuniform':[{'a':1},{'b':2}]}
        before=deepcopy(value);packed=pack_packet(value)
        self.assertEqual(unpack_packet(packed),value);self.assertEqual(value,before)

    def test_packing_rejects_reserved_keys(self):
        with self.assertRaises(ValueError):pack_packet({'$shared':'impostor'})

    def test_unpack_rejects_cycle(self):
        with self.assertRaises(ValueError):unpack_packet({'encoding':'lossless_tables_v1',
            'data':{'$shared':'v0'},'shared_values':{'v0':{'$shared':'v0'}}})

    def test_rule_reads_geometry_instead_of_first_id(self):
        r=request();r['geometry']['parts']['a']['bounds_center']=[0,0,0];r['geometry']['parts']['b']['bounds_center']=[3,0,0]
        a=choose_action(r);self.assertIn(a['choice']['direction_id'],('xp','xn'))

    def test_failed_path_not_repeated_but_part_not_removed(self):
        r=append_observation(request());a=choose_action(r);validate_path_round_response(r,a)
        self.assertNotIn(a['action_id'],r['attempted_action_ids']);self.assertIn('a',a['candidate_order'])

    def test_memory_unknown_is_not_pass(self):
        r=append_observation(request());self.assertEqual(memory_preferences(r)[0],{})

    def test_terminal_rule_keeps_both_candidates(self):
        r=request();r.update(remaining_parts=['a','b'],candidate_ids=['a','b'],base_part=None)
        a=choose_action(r);self.assertEqual(set(a['candidate_order']),{'a','b'})

    def test_memory_closure_and_no_target_id_mapping(self):
        a=dict(evidence_id='p',source_remaining_parts=['other'],source_choice=dict(skill_id='single_force_v1',support_ids=[]),parent_failure_evidence_id=None,path_success=False,stability_success=None,status='UNKNOWN')
        b=dict(a,evidence_id='c',parent_failure_evidence_id='p',path_success=True,stability_success=True,status='PASS')
        refs=select_references([b,a],request(),2)
        self.assertEqual([r['evidence_id'] for r in refs],['p','c'])
        self.assertTrue(all(r['relative_observation']['current_part_id_mapping'] is None for r in refs))
        self.assertEqual(len(select_references([b,a],request(),1)),1)

    def test_missing_parent_is_error(self):
        row=dict(evidence_id='c',parent_failure_evidence_id='missing',source_remaining_parts=[])
        with self.assertRaises(ValueError):select_references([row],request(),2)

    def test_selector_projection_is_exact(self):
        r=append_observation(request());p=dict(operation='select',current_request=r,evidence=deepcopy(r['episode_feedback']))
        compact=compact_selection(p);self.assertEqual(restore_selection(compact),p)

    def test_selector_rejects_extra_evidence(self):
        p=dict(operation='select',current_request=request(),evidence=[dict(evidence_id='madeup')])
        with self.assertRaises(ValueError):compact_selection(p)

    def test_synthesis_disabled(self):
        with self.assertRaises(ValueError):compact_selection(dict(operation='synthesize'))

    def test_oversize_fails_before_transport(self):
        with self.assertRaises(ValueError):bounded_body({'value':'a'*100000})

    def test_small_body_unchanged(self):
        b={'x':1};self.assertIs(bounded_body(b),b)

    def physics(self):return SimpleNamespace(MultiPartPathPlanner=PathChecker,MultiPartStabilityPlanner=StabilityChecker)

    def test_budget_counts_internal_checkers_and_steps(self):
        g=PhysicsBudget(2,1);p=PathChecker();s=StabilityChecker()
        with g.installed(self.physics()):
            self.assertEqual(p.check_success(),(True,[[0,0,0]]));self.assertEqual(s.check_success(),(False,['a']))
        self.assertEqual(g.counts,{'path':1,'stability':1});self.assertEqual(g.report()['physics_steps'],11)

    def test_budget_never_calls_over_limit(self):
        g=PhysicsBudget(1,1)
        with g.installed(self.physics()):
            PathChecker().check_success()
            with self.assertRaises(EvaluationLimit):PathChecker().check_success()
        self.assertEqual(g.counts['path'],1)

    def test_deadline_prevents_call(self):
        now=[0.];g=PhysicsBudget(seconds=1,clock=lambda:now[0]);now[0]=2
        with g.installed(self.physics()):
            with self.assertRaises(EvaluationLimit):PathChecker().check_success()
        self.assertEqual(g.counts['path'],0)

    def test_wrapper_restores_checker_and_sim(self):
        orig=PathChecker.check_success;p=PathChecker();sim=p.sim
        with PhysicsBudget().installed(self.physics()):p.check_success()
        self.assertIs(orig,PathChecker.check_success);self.assertIs(sim,p.sim);self.assertEqual(p.max_time,60)


if __name__=='__main__':unittest.main()
