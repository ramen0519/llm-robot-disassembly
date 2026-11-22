"""Four-part diagnostic with two independent posts resting on a removable tray."""
import sys
from pathlib import Path
import trimesh

if __name__=='__main__':
    root=Path(sys.argv[1])
    root.mkdir(parents=True,exist_ok=False)
    boxes=[('0',(60,35,5),(0,0,2.5)),('1',(52,30,10),(0,0,10)),
           ('2',(10,10,18),(-15,0,24)),('3',(10,10,18),(15,0,24))]
    for pid,size,center in boxes:
        mesh=trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        mesh.export(str(root/(pid+'.obj')))
