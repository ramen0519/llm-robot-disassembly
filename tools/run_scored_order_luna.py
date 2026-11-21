"""Preview one scored-order generation request; live only with its explicit option."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
from orchestrator.scored_order_generation import generate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/scored_order_skill_pilot_20261111')
    parser.add_argument('--execute',action='store_true')
    options=parser.parse_args()
    print(json.dumps(generate(options.run_dir,options.execute),ensure_ascii=False))
