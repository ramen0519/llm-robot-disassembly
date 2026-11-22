"""Inject transport failure through the same default pipeline. No HTTP is performed."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
from run_disassembly import main
from caseplanner.io import read
if __name__=='__main__':
    dest=Path(sys.argv[1]).absolute()
    def deny(request,endpoint,timeout):
        raise TimeoutError('controlled offline connection failure')
    code=main(['--input',str(ROOT/'evidence/diagnostic_tray_meshes'),'--unit','mm','--assembled-obj','--base','0',
               '--assembly','diagnostic_tray','--family','diagnostic_tray','--run',str(dest),'--max-calls','3',
               '--max-usd','1','--store',str(dest.parent/'error_contract_store')],test_transport=deny)
    r=read(dest/'result.json')
    assert code==2 and r['accepted_removals']==r['api_calls']==0 and r['offline_model_messages']==1
    assert r['counts']=={'path':0,'stability':0} and not r['new_cases']
    assert read(dest/'export_v1.json')['assembly_candidate'] is None
    print('PASS: no second dispatch, native fallback, evidence registration or accepted transition')
