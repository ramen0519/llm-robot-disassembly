"""Bounded known-Beam regression before any holdout planning."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_unseen_transfer import RUN,WSL,PYTHON,linux,native_command,read,write
from run_recorded import run_recorded

def main():
    steps=[('prepare_development',WSL+[PYTHON,'-B','-u',linux(ROOT/'worker/prepare_transfer_development.py')])]
    assembly='/opt/asap-research/unseen_transfer_pilot_20261019/development_beam'
    for name,command in steps:
        if run_recorded(command,ROOT,RUN/'commands'/name)['exit_code']:raise RuntimeError(name)
    for name,unmetered in [('native_unmetered',True),('native_metered',False)]:
        if run_recorded(native_command(assembly,RUN/'development'/name,RUN/'development_protocol.json',unmetered),ROOT,RUN/'commands'/name)['exit_code']:raise RuntimeError(name)
    old=read(RUN/'development/native_unmetered/native_result.json');new=read(RUN/'development/native_metered/native_result.json')
    def facts(v):
        return dict(success=v['success'],sequence=v['native_stats'].get('sequence'),depth=v['max_depth'],
            attempts=[{k:r[k] for k in ('part_move','parts_rest','feasible','returned')} for r in v['attempts']])
    same=facts(old)==facts(new)
    write(RUN/'development/native_regression.json',dict(same_observed_plan_and_attempts=same,old=facts(old),new=facts(new),
        actual_physics=True,metered_counts=new['budget']['counts'],api_calls=0))
    if not same or not new['success']:raise RuntimeError('native_regression_failed')
    if run_recorded([sys.executable,'-B','-u',str(ROOT/'tools/run_unseen_transfer.py'),'dev_rule'],ROOT,RUN/'commands/dev_rule')['exit_code']:raise RuntimeError('rule_development_failed')
    print('DEVELOPMENT_CHECK_COMPLETE',flush=True)

if __name__=='__main__':main()
