"""Orthographic CPU z-buffer. Snapshot poses are mesh-local to simulation world."""
import hashlib
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from .geometry import vertices_after
from .io import read

CAMERA='orthographic-z-upY-center-target-768-margin10-zbuffer/v1'
COLORS=np.array([[61,132,191],[224,139,65],[90,162,118],[190,98,109],[151,126,184],[113,151,158]])

class Scene:
    def __init__(self, folder):
        self.meshes={}
        self.assembled={}
        self.geom=read(folder/'geometry.json')
        for m in read(folder/'part_mapping.json').values():
            p=m['part_id']
            mesh=trimesh.load_mesh(str(folder/m['native_mesh']),process=False,maintain_order=True,skip_materials=True)
            self.meshes[p]=(np.asarray(mesh.vertices),np.asarray(mesh.faces))
            mat=np.array(m['assembly_matrix_mm'])
            mat[:3,3]/=10
            self.assembled[p]=mat

    def world(self, poses):
        return {p:vertices_after(self.meshes[p][0],np.asarray(t)) for p,t in poses.items()}

    def camera(self, world, direction=(0,0,1), target=None, related=None, anchor=None):
        normal=np.array(direction,dtype=float)
        normal/=np.linalg.norm(normal)
        up=np.array([0.,1.,0.]) if direction==(0,0,1) else np.array([0.,0.,1.])
        right=np.cross(up,normal)
        right/=np.linalg.norm(right)
        basis=np.stack((right,np.cross(normal,right),normal)).T
        focus=[p for p in world if related is None or p in related or p==target]
        if not focus:
            focus=list(world)
        xyz=np.concatenate([world[p] for p in focus])@basis
        if target is None:
            ctr=(xyz.min(axis=0)+xyz.max(axis=0))/2
        else:
            if target in world:
                pts=world[target]
                anchor=(pts.min(axis=0)+pts.max(axis=0))/2
            if anchor is None:
                raise ValueError('missing pre-removal target anchor')
            ctr=np.array(anchor)@basis
        half=max(.001,np.max(np.abs(xyz[:,:2]-ctr[:2]))*1.1)
        return basis,ctr,half

    def render(self, poses, path=None, camera=None, direction=(0,0,1), size=768, **framing):
        world=self.world(poses)
        basis,ctr,half=camera or self.camera(world,direction,**framing)
        im=np.full((size,size,3),247,dtype=np.uint8)
        depth=np.full((size,size),-np.inf)
        for p,points in world.items():
            coord=(points@basis-ctr)
            coord[:,:2]*=size/(2*half)
            coord[:,0]+=size/2
            coord[:,1]=size/2-coord[:,1]
            color=COLORS[sorted(self.meshes).index(p)%len(COLORS)]
            from .raster import backend,draw
            raster=backend()
            if raster is not None:
                draw(raster,coord,points,self.meshes[p][1],color,depth,im)
                continue
            for face in self.meshes[p][1]:
                tri=coord[face]
                x0,y0=np.maximum([0,0],np.floor(tri[:,:2].min(axis=0))).astype(int)
                x1,y1=np.minimum([size-1,size-1],np.ceil(tri[:,:2].max(axis=0))).astype(int)
                if x1<x0 or y1<y0: continue
                a,b,c=tri
                ab=b-a
                ac=c-a
                den=ab[0]*ac[1]-ac[0]*ab[1]
                if abs(den)<1e-10: continue
                yy,xx=np.ogrid[y0:y1+1,x0:x1+1]
                dx,dy=xx+.5-a[0],yy+.5-a[1]
                u=(dx*ac[1]-dy*ac[0])/den
                v=(ab[0]*dy-ab[1]*dx)/den
                z=a[2]+u*ab[2]+v*ac[2]
                sub=depth[y0:y1+1,x0:x1+1]
                test=(u>=0)&(v>=0)&(u+v<=1)&(z>sub)
                sub[test]=z[test]
                normal=np.cross(points[face[1]]-points[face[0]],points[face[2]]-points[face[0]])
                shade=.57+.43*abs(float(normal@np.array([.28,.39,.877])))/max(np.linalg.norm(normal),1e-12)
                im[y0:y1+1,x0:x1+1][test]=np.clip(color*shade,0,255)
        img=Image.fromarray(im)
        if path is not None: img.save(path)
        return img
