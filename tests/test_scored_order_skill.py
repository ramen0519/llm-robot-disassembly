import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestrator.scored_order_generation import GenerationService, generate, make_candidate, preview, verify_live_candidate
from orchestrator.scored_order_skill import feature_rows, run_skill


SOURCE = "def propose(ctx):\n    return ctx['contact_degree']\n"


def sample():
    part = dict(bounds=[[0,0,0],[1,2,3]], bounds_center=[0.5,1,1.5], extents=[1,2,3], volume=6,
                validity={'center_mass_status':'uniform_density_estimate'})
    geometry = dict(parts={p:part for p in ('x','y','z')}, contact_edges=[['x','y'],['y','z']])
    return feature_rows(geometry, list(geometry['parts']))


def packet():
    return dict(protocol='scored_order_observations/v1',training_products=['A'],objective='fixture',
                examples=[dict(evidence_id='event-A',candidates=sample(),case='case',score=[2,3])])


def provider(source=SOURCE):
    text = dict(name='fixture',description='fixture arithmetic',source=source,evidence_ids=['event-A'])
    return dict(status='completed', model='gpt-6-luna', id='fixture-response',
                usage=dict(input_tokens=100,output_tokens=100,total_tokens=200),
                output=[dict(type='message',role='assistant',status='completed',
                             content=[dict(type='output_text',text=json.dumps(text))])])


class ScoredOrderSkillTests(unittest.TestCase):
    def test_id_independent_numeric_program_and_complete_ties(self):
        rows=sample()
        self.assertEqual(run_skill(SOURCE,rows)['candidate_order'],['y','x','z'])
        renamed={'p':rows['z'],'q':rows['y'],'r':rows['x']}
        self.assertEqual(run_skill(SOURCE,renamed)['candidate_order'],['q','p','r'])
        flat=run_skill("def propose(ctx):\n return 1",rows)
        self.assertEqual(flat['candidate_order'],['x','y','z'])
        self.assertEqual(flat['diagnostics'],['all_candidate_priorities_tied'])
        self.assertEqual(run_skill("def propose(ctx):\n return None",rows)['status'],'ABSTAIN')
        self.assertEqual(rows['y']['remainder_isolated_nodes'],2)

    def test_no_io_no_part_ids_no_shapes_leak(self):
        for source in ("def propose(ctx):\n return ctx['part_id']",
                       "def propose(ctx):\n import os\n return 0",
                       "def propose(ctx):\n return ['x']",
                       "def propose(ctx):\n return ctx",
                       "def propose(ctx):\n return 1 / 0"):
            self.assertEqual(run_skill(source,sample())['status'],'REJECTED')
        bad = sample()
        bad['x']['score']=[2,1]
        self.assertEqual(run_skill(SOURCE,bad)['status'],'REJECTED')

    def test_preview_size_limits_and_generator_has_no_rule_or_example_program(self):
        p=packet()
        prepared=preview(p)
        self.assertEqual(prepared['max_calls'],1)
        self.assertEqual(prepared['body']['store'],False)
        self.assertNotIn(SOURCE, str(prepared['body']))
        p['examples'][0]['human_hypothesis']='choose x'
        with self.assertRaises(ValueError):
            preview(p)

    def test_fixture_durable_limit_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,OPENAI_API_KEY='test-only-fake'):
            transport=Mock(return_value=provider())
            dest=Path(temp)/'ledger.sqlite'
            s=GenerationService(packet(),dest,transport)
            result=s.rank(s.envelope())
            self.assertIsNone(result['fallback_reason'],result)
            cand=result['metadata']['candidate']
            self.assertEqual(cand['provenance']['origin'],'fixture')
            self.assertEqual(cand['source'],SOURCE)
            self.assertFalse(cand['registered'])
            self.assertTrue(s.rank(s.envelope())['metadata']['cached'])
            p2=packet()
            p2['examples'][0]['score']=[2,4]
            other=GenerationService(p2,dest,transport)
            self.assertEqual(other.rank(other.envelope())['fallback_reason'],'call_budget_exhausted')
            self.assertEqual(transport.call_count,1)

    def test_invalid_source_and_output_retained_not_accepted(self):
        code="def propose(ctx):\n import os\n return 1"
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,OPENAI_API_KEY='fixture-secret'):
            s=GenerationService(packet(),Path(temp)/'ledger.sqlite',Mock(return_value=provider(code)))
            response=s.rank(s.envelope())
            self.assertEqual(response['fallback_reason'],'candidate_rejected')
            self.assertEqual(response['metadata']['rejected_source']['source_text'],code)
            self.assertNotIn('candidate',response['metadata'])
        value=dict(name='fixture',description='d',source=SOURCE,evidence_ids=['not-in-packet'])
        with self.assertRaises(ValueError):
            make_candidate(value,packet(),dict(origin='fixture'))

    def test_one_shot_host_recovers_ledger_and_rejects_tampered_local_source(self):
        # HTTP is substituted at the transport boundary. All files are temporary.
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, OPENAI_API_KEY='fixture-secret'), \
                patch('tools.run_luna_experiment.ROOT', Path(temp)), \
                patch('orchestrator.llm_ranker._http_transport',return_value=provider()) as http:
            run=Path(temp)/'runs'/'fixture'
            run.mkdir(parents=True)
            (run/'training_observations.json').write_text(json.dumps(packet()),encoding='utf8')
            self.assertEqual(generate(run)['status'],'PREPARED_OFFLINE')
            http.assert_not_called()
            self.assertEqual(generate(run,True)['status'],'CANDIDATE')
            candidate_file=run/'luna/candidate.json'
            candidate_file.unlink()
            self.assertEqual(generate(run,True)['status'],'CANDIDATE')
            self.assertEqual(http.call_count,1)
            candidate=verify_live_candidate(run)
            candidate['source']=SOURCE+'# edited'
            candidate_file.write_text(json.dumps(candidate),encoding='utf8')
            with self.assertRaises(ValueError):
                verify_live_candidate(run)
