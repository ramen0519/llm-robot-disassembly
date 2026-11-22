"""Explicit synthetic three-part CAD fixture. Only RedMax decides outcomes."""
from pathlib import Path
import sys
import numpy as np
import trimesh

def main(root):
    root.mkdir(parents=True,exist_ok=False)
    # One watertight open-top cup, a removable insert and a separate lid. Millimetres.
    def ring(half,z):
        return [[-half,-half,z],[half,-half,z],[half,half,z],[-half,half,z]]
    vertices=ring(18,0)+ring(18,26)+ring(12,26)+ring(12,4)
    polygons=[[3,2,1,0],[12,13,14,15]]
    for i in range(4):
        j=(i+1)%4
        polygons.extend([[i,j,j+4,i+4],[i+4,j+4,j+8,i+8],[i+8,j+8,j+12,i+12]])
    faces=[tri for a,b,c,d in polygons for tri in ([a,b,c],[a,c,d])]
    cup=trimesh.Trimesh(vertices=vertices,faces=faces)
    assert cup.is_volume
    cup.export(str(root/'0.obj'))
    for name,extents,height in [('1',[23.,23.,21.8],14.9),('2',[36.,36.,4.],28.)]:
        block=trimesh.creation.box(extents=extents)
        block.apply_translation([0,0,height])
        block.export(str(root/(name+'.obj')))
if __name__=='__main__':
    main(Path(sys.argv[1]))
