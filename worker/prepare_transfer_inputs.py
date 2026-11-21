"""Official subdivision, shared SDF preparation, and input-only checks."""
import os
os.environ['OMP_NUM_THREADS']='1'
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'worker'))
from remote_rank_client import checked_path


def main():
    run=checked_path(PROJECT/'runs/unseen_transfer_pilot_20261019')
    root=checked_path(Path('/opt/asap-research/ASAP'),False)
    dest=checked_path(Path('/opt/asap-research/unseen_transfer_pilot_20261019'),False)
    dest.mkdir(exist_ok=False)
    sys.path.insert(0,str(root))
    import numpy as np
    from assets.subdivide import subdivide_assembly
    from assets.load import load_assembly
    from plan_sequence.physics_planner import get_body_mass
    selection=json.loads((run/'selection.json').read_text())
    rows=[];inputs={}
    for item in selection['selected']:
        ident=item['id'];start=time.monotonic();source=run/'raw_assemblies'/ident;target=dest/ident
        print('PREPARE '+ident,flush=True)
        subdivide_assembly(str(source),str(target),max_edge=.5)
        old,new=load_assembly(str(source)),load_assembly(str(target))
        checks={'same_ids':sorted(old)==sorted(new),'part_count':len(new)==item['parts']}
        for part in old:
            checks[part+'_bounds']=bool(np.allclose(old[part]['mesh'].bounds,new[part]['mesh'].bounds,atol=1e-6,rtol=1e-6))
            checks[part+'_volume']=bool(np.isclose(old[part]['mesh'].volume,new[part]['mesh'].volume,atol=1e-5,rtol=1e-5))
        if not all(checks.values()):raise ValueError('subdivision_invariant_failed:'+ident)
        # SDF/geometry construction only: no path or gravity rollout, no outcome-based selection.
        get_body_mass(str(root/'assets'),str(target),sorted(new),save_sdf=True)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()}
        inputs[str(target)]=hashes
        rows.append(dict(id=ident,checks=checks,seconds=time.monotonic()-start,files=hashes,
                         watertight={k:bool(v['mesh'].is_watertight) for k,v in new.items()}))
        print(json.dumps({'id':ident,'input_status':'PASS','seconds':rows[-1]['seconds']}),flush=True)
    with (run/'prepared_inputs.json').open('x') as f:json.dump(dict(rows=rows,assembly_dirs=inputs,method='official_subdivide_max_edge_0.5_shared_SDF_no_rollout',path_calls=0,stability_calls=0),f,indent=2)


if __name__=='__main__':main()
