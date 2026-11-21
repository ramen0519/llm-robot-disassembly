"""New copy of the known Beam for bounded regression, not evaluation learning."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'worker'))
from remote_rank_client import checked_path

def main():
    run=checked_path(ROOT/'runs/unseen_transfer_pilot_20261019')
    root=checked_path(Path('/opt/asap-research/ASAP'),False)
    source=root/'assets/beam_assembly/original'
    target=checked_path(Path('/opt/asap-research/unseen_transfer_pilot_20261019/development_beam'),False)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    shutil.copytree(str(source),str(target))
    sys.path.insert(0,str(root))
    from assets.load import load_part_ids
    from plan_sequence.physics_planner import get_body_mass
    get_body_mass(str(root/'assets'),str(target),load_part_ids(str(target)),save_sdf=True)
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    spec=json.loads((run/'protocol.json').read_text())
    spec.update(purpose='development',development_base_part='6',order=[],
        assembly_dirs={str(target):{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()}})
    spec['settings'].update(budget=8,state_action_budget=8,timeout=120,wall_timeout=150)
    with (run/'development_protocol.json').open('x') as f:json.dump(spec,f,indent=2)
    print('development_copy_ready; source_unchanged; api_calls=0')

if __name__=='__main__':main()
