"""Offline tests against the motor CAD occurrence inventory (no physics or model)."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
from fastener_sequence import BoltedJoint, bind_released_scene, plan_joint
from run_sequence_native import main as planner_main

SPEC = ROOT / 'configs/lab_motor_rear_fasteners.json'
BOM = ROOT / 'runs/lab_total_structure_v1/bom/bom.json'


class MotorJointTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(SPEC.read_text(encoding='utf-8'))
        self.bom = json.loads(BOM.read_text(encoding='utf-8'))
        self.joint = BoltedJoint.from_bom(self.spec, self.bom)
        self.events = dict(joint_id=self.joint.id, source_step_sha256=self.joint.step_hash, events=[])

    def test_bound_to_four_distinct_motor_bolts(self):
        bolts = {(p['path'], p['label']) for p in self.bom['placements'] if p['product'] == 'M5×L15'}
        self.assertEqual({(b.path, b.label) for b in self.joint.bolts}, bolts)
        self.assertEqual(len(bolts), 4)

    def test_all_81_reachable_states_and_removal_gate(self):
        bolts = [b.id for b in self.joint.bolts]
        for stages in itertools.product(range(3), repeat=4):
            with self.subTest(stages=stages):
                state = self.joint.initial()
                for bolt, stage in zip(bolts, stages):
                    for action in ('unfasten', 'recover')[:stage]:
                        state = state.apply(action, bolt, 'CONFIRMED')
                snap = state.snapshot()
                expected = {b for b, s in zip(bolts, stages) if s != 2}
                self.assertEqual(set(state.blockers()), expected)
                self.assertEqual(set(state.remaining()), expected | {'frame', 'housing'})
                engaged = {b for b, s in zip(bolts, stages) if s == 0}
                self.assertEqual(set(snap['rigid_group']), engaged | {'frame','housing'} if engaged else set())
                self.assertEqual(state.removal_candidates(), () if expected else ('housing',))
                self.assertEqual({(a['part_id'], a['action']) for a in snap['bolt_action_candidates']},
                                 {(b, ('unfasten','recover')[s]) for b,s in zip(bolts,stages) if s != 2})
                if expected:
                    with self.assertRaises(ValueError):
                        state.removal_request('housing')
                else:
                    request = state.removal_request('housing')
                    self.assertEqual(request['occurrence_path'], self.joint.removable.path)
                    self.assertEqual(request['physics_status'], 'NOT_RUN')
                for part in ('frame', 'm5_1', 'unknown'):
                    with self.assertRaises(ValueError):
                        state.removal_request(part)

    def test_each_bolt_can_be_last_and_branch_state_is_independent(self):
        initial = self.joint.initial()
        for order in itertools.permutations(b.id for b in self.joint.bolts):
            state = initial
            for i, bolt in enumerate(order):
                state = state.apply('unfasten', bolt, 'CONFIRMED')
                self.assertFalse(state.removal_candidates())
                state = state.apply('recover', bolt, 'CONFIRMED')
                self.assertEqual(bool(state.removal_candidates()), i == 3)
        self.assertEqual(initial.statuses, ('engaged',)*4)

    def test_invalid_or_unconfirmed_events_never_grant_removal(self):
        state = self.joint.initial()
        for action, bolt in [('recover','m5_1'), ('unfasten','ghost'), ('remove','housing')]:
            with self.assertRaises(ValueError):
                state.apply(action, bolt, 'CONFIRMED')
        for stage in ('unfasten','recover'):
            for outcome in ('UNKNOWN','FAILED'):
                self.assertIs(state.apply(stage, 'm5_1', outcome), state)
            with self.assertRaises(ValueError):
                state.apply(stage, 'm5_1', 'maybe')
            state = state.apply(stage, 'm5_1', 'CONFIRMED')
            with self.assertRaises(ValueError):
                state.apply(stage, 'm5_1', 'CONFIRMED')

    def test_configuration_rejects_removed_duplicate_or_wrong_occurrences(self):
        for field, value in [('source_step_sha256', 'b'*64), ('bolts', []),
                             ('bolts', self.spec['bolts'][:-1]),
                             ('bolts', self.spec['bolts'][:1]*2),
                             ('retained', self.spec['removable']),
                             ('removable', dict(self.spec['removable'], label='wrong')),
                             ('removable', dict(self.spec['removable'], path='#unknown'))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                BoltedJoint.from_bom(dict(self.spec, **{field:value}), self.bom)

    def test_replayed_request_is_idempotent_conflicts_and_stale_revisions_rejected(self):
        event = dict(request_id='op1', expected_revision=0, action='unfasten', part_id='m5_1',
                     outcome='CONFIRMED')
        self.events['events'] = [event, deepcopy(event)]
        report = plan_joint(self.joint, self.events)
        self.assertEqual([r['revision'] for r in report['trace']], [0,1])
        for change in (dict(part_id='m5_2'), dict(request_id='op2')):
            self.events['events'] = [event, dict(event, **change)]
            with self.assertRaises(ValueError):
                plan_joint(self.joint, self.events)
        for change in (dict(joint_id='other'), dict(source_step_sha256='c'*64)):
            with self.assertRaises(ValueError):
                plan_joint(self.joint, dict(self.events, events=[], **change))

    def test_sequence_worker_entry_point_emits_post_recovery_candidate(self):
        for bolt in self.joint.bolts:
            for action in ('unfasten','recover'):
                revision = len(self.events['events'])
                self.events['events'].append(dict(request_id='t'+str(revision), expected_revision=revision,
                    action=action, part_id=bolt.id, outcome='CONFIRMED'))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'events.json'
            path.write_text(json.dumps(self.events), encoding='utf-8')
            args = ['run_sequence_native.py','fasteners','--joint',str(SPEC),'--bom',str(BOM),'--events',str(path)]
            out = io.StringIO()
            with patch.object(sys,'argv',args), redirect_stdout(out):
                self.assertEqual(planner_main(), 0)
        report = json.loads(out.getvalue())
        self.assertEqual([row['state']['removal_candidates'] for row in report['trace']], [[]]*8+[['housing']])
        self.assertEqual(report['candidate_requests'][0]['occurrence_path'], '#58087/#56568')
        self.assertEqual(report['robot_execution'], 'NOT_RUN')

    def test_native_entry_rejects_locked_joint_before_asap_or_output(self):
        args = ['run_sequence_native.py', '--asap-root','unused','--assembly-dir','unused',
                '--output-dir','unused','--run-id','test','--joint',str(SPEC),'--bom',str(BOM)]
        with patch.object(sys, 'argv', args), patch('run_sequence_native.initialize_output') as output:
            with self.assertRaisesRegex(ValueError, 'still blocks'):
                planner_main()
            output.assert_not_called()

    def test_native_handoff_checks_remaining_cad_and_mesh_fingerprints(self):
        excluded = {b.path for b in self.joint.bolts}
        parts = [p['path'] for p in self.bom['placements'] if p['leaf'] and p['path'] not in excluded]
        self.assertEqual(len(parts), 68)
        rows = {path: dict(obj=str(i)+'.obj', sha256=format(i, '064x')) for i,path in enumerate(parts)}
        mapping = dict(source_step_sha256=self.joint.step_hash, placements=rows)
        hashes = {r['obj']:r['sha256'] for r in rows.values()}
        actual = bind_released_scene(self.joint, self.bom, mapping, hashes, rows[self.joint.retained.path]['obj'][:-4])
        self.assertEqual(actual, rows[self.joint.removable.path]['obj'][:-4])
        for changed in (dict(hashes, **{'remaining_bolt.obj':'0'*64}), dict(hashes, **{'0.obj':'f'*64})):
            with self.assertRaises(ValueError):
                bind_released_scene(self.joint, self.bom, mapping, changed, None)
        with self.assertRaises(ValueError):
            bind_released_scene(self.joint, self.bom, mapping, hashes, 'wrong')
        with self.assertRaises(ValueError):
            bind_released_scene(self.joint, self.bom, dict(mapping, placements={}), hashes, None)
