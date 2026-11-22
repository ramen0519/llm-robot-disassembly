"""Offline Responses transport + actual RedMax + actual encoder. Never sends HTTP."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT))
from run_disassembly import main
from caseplanner.io import read,write

class Double:
    def __init__(self): self.n=0
    def __call__(self,body,endpoint,timeout):
        self.n+=1
        if body['text']['format']['name']=='candidate_order':
            pkt=json.loads(body['input'][0]['content'][0]['text'])
            reply=dict(request_id=pkt['request_id'],state_id=pkt['state']['state_id'],ranking=sorted(pkt['valid_candidates']),
                       applied_case_ids=[c['case_id'] for c in pkt['guidance']])
        else:
            pkt=json.loads(body['input'][0]['content'])
            refs=list(pkt['facts'])
            reply=dict(decision='create',applies_when='In this diagnostic, a supported pair loses its initial connections when the carrier is tried.',
                strategy='Try an eligible supported component first, then reassess the carrier after the removal is accepted.',
                evidence_refs=refs,skip_reason=None)
        return dict(id='test_response_%d'%self.n,model='offline-test-double',status='completed',
            output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(reply))])],
            usage=dict(input_tokens=100,output_tokens=40))
if __name__=='__main__':
    out=Path(sys.argv[1]).absolute()
    out.mkdir(parents=True,exist_ok=False)
    common=['--mode','astra','--input',str(ROOT/'evidence/diagnostic_tray_meshes'),'--assembled-obj','--unit','mm',
        '--family','diagnostic_tray','--assembly','diagnostic_tray','--base','0','--max-attempts','6','--max-path','36',
        '--max-stability','16','--seconds','120','--max-calls','8','--max-usd','3','--store',str(out/'isolated_test_store')]
    for role in ('collect','query'):
        args=common+['--run',str(out/role)]
        main(args,test_transport=Double())
        result=read(out/role/'result.json')
        assert result['api_calls']==0 and result['rank_source']=='offline_responses_double'
        assert result['status']=='COMPLETE' and result['resolved']==1
    train,query=(read(out/k/'result.json') for k in ('collect','query'))
    cid=train['new_cases'][0]
    frozen=read(out/'query/frozen_index.json')
    retrieval=read(out/'query/retrieval.json')
    assert frozen['case_ids']==[cid] and read(out/'collect/frozen_index.json')['case_ids']==[]
    assert any(cid in r['hit_ids'] for r in retrieval['lookups'])
    assert any(cid in r['applied'] for r in retrieval['decisions'])
    assert query['new_cases'][0] not in frozen['case_ids']
    write(out/'verification.json',dict(physics='actual RedMax',images='actual PNG + DINOv2',provider='offline responses double',
        guidance='explicit fixture text',model_selection='explicit fixture citation',live_model_calls=0,
        collection=train['run_id'],source_case=cid,query=query['run_id'],frozen_index=frozen['index'],
        local_resolution=True,registered_in_test_namespace=True,retrieved=True,fixture_citation=True,
        new_query_case_excluded=True,unseen_product_evaluation=False))
