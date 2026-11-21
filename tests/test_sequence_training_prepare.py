"""Selection and budget preparation checks: no paid calls, WSL or physics."""
from collections import Counter
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import prepare_sequence_training as prep


def inventory():
    return [dict(id=ident,parts=count,archive_dir='test_assembly/'+ident,obj_bytes=100)
        for ident,count in [('00001',5),('00002',7),('00003',9),('00004',12),('00005',15),
                            ('00006',10),('00007',14),('00335',16)]]


def signature(ident):
    return Counter({ident:10})


class SequenceTrainingPrepareTests(unittest.TestCase):
    def setUp(self):
        for target in ('urllib.request.OpenerDirector.open','subprocess.Popen'):
            blocker=patch(target,side_effect=AssertionError('NO_API_OR_SUBPROCESS'))
            mock=blocker.start();self.addCleanup(blocker.stop);self.addCleanup(mock.assert_not_called)

    def test_fixed_counts_and_metadata_holdouts_without_outcome_input(self):
        split=prep.select_split(inventory(),{'00335':[]},signature)
        self.assertEqual([r['parts'] for r in split['training']],[5,7,9,12,15])
        self.assertEqual([r['id'] for r in split['training']],['00001','00002','00003','00004','00005'])
        self.assertEqual({r['id'] for r in split['evaluation']},{'00006','00007'})
        self.assertTrue(split['development_holdout']['prior_exposed'])
        self.assertFalse(split['development_holdout']['pristine'])
        self.assertFalse(split['solver_outcomes_used_for_selection'])
        self.assertTrue(all(r['status']=='NOT_RUN' for r in split['evaluation']))

    def test_selection_is_inventory_order_independent(self):
        self.assertEqual(prep.select_split(inventory(),{},signature),
                         prep.select_split(list(reversed(inventory())),{},signature))

    def test_missing_count_stops_without_replacement(self):
        rows=[r for r in inventory() if r['parts']!=9]
        with self.assertRaisesRegex(ValueError,'no_distinct_training_candidate_for_part_count:9'):
            prep.select_split(rows,{},signature)

    def test_duplicate_training_geometry_excluded_by_predeclared_rule(self):
        rows=inventory()+[dict(id='00008',parts=7,archive_dir='test_assembly/00008',obj_bytes=100)]
        split=prep.select_split(rows,{},lambda i:signature('00001' if i=='00002' else i))
        self.assertEqual(split['training'][1]['id'],'00008')
        self.assertEqual(split['exclusions'][0]['reason'],'half_shared_normalized_vertices')

    def test_exposed_or_duplicate_evaluation_rejected(self):
        for exposed,sig in [({'00006':[]},signature),({},lambda i:signature('00001' if i=='00006' else i))]:
            with self.subTest(exposed=exposed),self.assertRaisesRegex(ValueError,'no_unexposed_nonduplicate'):
                prep.select_split(inventory(),exposed,sig)

    def test_archive_inventory_checks_counts_and_hostile_names(self):
        for names,valid in [(['test_assembly/00001/0.obj','test_assembly/00001/1.obj'],True),
                            (['test_assembly/../0.obj'],False),(['E:/x.obj'],False),
                            (['test_assembly/00001/0.obj']*2,False)]:
            data=io.BytesIO()
            with zipfile.ZipFile(data,'w') as z:
                for name in names:z.writestr(name,'v 0 0 0\n')
            data.seek(0)
            with self.subTest(names=names),zipfile.ZipFile(data) as z:
                if valid:self.assertEqual(prep.archive_inventory(z)[0]['parts'],2)
                else:
                    with self.assertRaises(ValueError):prep.archive_inventory(z)

    def test_contact_graph_cannot_silently_change_parts(self):
        graph=dict(nodes=[dict(id='0'),dict(id='1')],links=[dict(source=0,target=1)])
        self.assertEqual(prep.check_graph(json.dumps(graph),['0','1'])['nodes'],2)
        with self.assertRaisesRegex(ValueError,'part_mismatch'):
            prep.check_graph(json.dumps(graph),['0','2'])
        graph['links'][0]['target']=9
        with self.assertRaisesRegex(ValueError,'unknown_endpoint'):
            prep.check_graph(json.dumps(graph),['0','1'])

    def test_budget_is_disabled_with_separate_usage_reference_and_reservation(self):
        observed=dict(api=dict(actual_attempts=16,usage=dict(input_tokens=325621,output_tokens=3847)))
        p=prep.budget_proposal(observed)
        self.assertEqual(p['max_calls'],270)
        self.assertEqual(p['max_program_usd'],'40.50')
        self.assertEqual(Decimal(p['worst_configured_reservation_usd']),Decimal('39.2192'))
        self.assertEqual(Decimal(p['usage_based_reference']['same_mean_250_sol_calls_usd']),Decimal('10.77675'))
        self.assertFalse(p['live_enabled'])
        self.assertEqual(p['automatic_http_retries'],0)
        self.assertEqual(p['physics']['max_worker_seconds'],48000)
        self.assertEqual(p['physics']['max_outer_seconds'],52200)

    def test_forbidden_path_rejected_lexically_and_existing_output_preserved(self):
        with patch.object(Path,'exists',side_effect=AssertionError('NO_PATH_PROBE')):
            with self.assertRaisesRegex(ValueError,'forbidden_path'):
                prep.prepare('E:/must-not-be-probed')
        with patch.object(prep,'sha',side_effect=AssertionError('NO_ARCHIVE_READ')):
            with self.assertRaisesRegex(FileExistsError,'new_sequence_preparation'):
                prep.prepare(prep.ROOT/'runs')

    def test_supplemental_report_is_read_and_bound_instead_of_silently_skipped(self):
        with tempfile.TemporaryDirectory(prefix='sequence_exposure_fixture_',dir=prep.ROOT/'runs') as folder:
            path=Path(folder)/'report.json'
            path.write_text(json.dumps(dict(task=dict(assembly_id='00335'),
                nested=[dict(assembly_dir='/opt/asap-research/ASAP/assets/test/00161')])) ,encoding='utf8')
            relative=path.relative_to(prep.ROOT).as_posix()
            audit=dict(exposed={},unreadable_or_oversize=[relative])
            result=prep.complete_large_exposure_audit(audit)
            self.assertEqual(set(result['exposed']),{'00335','00161'})
            self.assertEqual(result['unreadable_or_oversize'],[])
            self.assertEqual(result['supplemental_reports'][0]['sha256'],prep.sha(path))
            self.assertEqual(audit['unreadable_or_oversize'],[relative])


if __name__=='__main__':unittest.main()
