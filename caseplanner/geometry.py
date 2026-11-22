"""Produce local meshes, assembled geometry and native RedMax centimetre input."""
import itertools
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh
import fcl
from scipy.spatial.transform import Rotation
from .io import digest, location, read, sha, write

SCALE = {'mm':1., 'cm':10., 'm':1000., 'inch':25.4}


def rigid(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4,4) or not np.isfinite(matrix).all() or
        not np.allclose(matrix[3], [0,0,0,1]) or
        not np.allclose(matrix[:3,:3].T @ matrix[:3,:3],np.eye(3),atol=1e-7) or
        not np.isclose(np.linalg.det(matrix[:3,:3]),1.)):
        raise ValueError('expected rigid local-to-assembly transform')
    return matrix


def vertices_after(vertices, matrix):
    return np.asarray(vertices) @ matrix[:3,:3].T + matrix[:3,3]


def native_manifest(folder, unit, already_assembled):
    if unit not in SCALE:
        raise ValueError('OBJ needs declared length unit')
    config = read(folder/'config.json') if (folder/'config.json').exists() else None
    if config and already_assembled:
        raise ValueError('ambiguous already-applied final pose')
    if not config and not already_assembled:
        raise ValueError('OBJ needs explicit assembled frame or occurrence manifest')
    rows = []
    for p in sorted(folder.glob('*.obj')):
        mat = np.eye(4)
        if config:
            values = config[p.stem]['final_state']
            if len(values) not in (3,6):
                raise ValueError('invalid ASAP final_state')
            mat[:3,3] = np.array(values[:3])*SCALE[unit]
            if len(values)==6:
                mat[:3,:3] = Rotation.from_euler('xyz',values[3:]).as_matrix()
        rows.append(dict(part_id=p.stem, mesh=p.name, cad_name=None, assembly_matrix_mm=mat.tolist(),
            mesh_coordinate_frame='assembly' if already_assembled else 'part_local'))
    return dict(unit=unit,parts=rows)


def objwrite(path, vertices, faces):
    with path.open('x') as stream:
        np.savetxt(stream,vertices,fmt='v %.15g %.15g %.15g')
        np.savetxt(stream,faces+1,fmt='f %d %d %d')


def collision_mesh(world, faces):
    model=fcl.BVHModel()
    model.beginModel(len(world),len(faces))
    model.addSubModel(world,faces)
    model.endModel()
    return fcl.CollisionObject(model)


def prepare(source, out, assembly, unit=None, already_assembled=False, tolerance=.1,
            step_python='/usr/bin/python3'):
    source=location(source)
    if not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('invalid proximity tolerance')
    out.mkdir(parents=True,exist_ok=False)
    raw=out/'source'
    raw.mkdir()
    sim=out/'native'
    sim.mkdir()
    original={}
    if source.suffix.lower() in ('.stp','.step'):
        original[source.name]=sha(source)
        shutil.copyfile(source,raw/source.name)
        subprocess.run([step_python,'-B',str(Path(__file__).with_name('step_xcaf.py')),str(raw/source.name),str(raw)],check=True)
        manifest=read(raw/'occurrences.json')
        folder=raw
    elif source.is_dir():
        folder=source
        manifest=native_manifest(folder,unit,already_assembled)
        if (folder/'config.json').is_file():
            original['config.json']=sha(folder/'config.json')
            shutil.copyfile(folder/'config.json',raw/'config.json')
    else:
        folder=source.parent
        manifest=read(source)
        original[source.name]=sha(source)
        shutil.copyfile(source,raw/source.name)
    unit=manifest['unit']
    if unit not in SCALE:
        raise ValueError('unknown original units')
    parts, mappings, objects,config = {},{},{},{}
    source_ids=[row['part_id'] for row in manifest['parts']]
    if (not all(isinstance(p,str) and p and not any(ord(c)<32 for c in p) for p in source_ids)
        or len(set(source_ids))!=len(source_ids)): raise ValueError('invalid or repeated source part id')
    # Upstream ASAP indexes its palette with int(id). Give named OBJ occurrences stable native IDs.
    used={p for p in source_ids if re.fullmatch(r'[0-9]+',p)}
    available=(str(n) for n in itertools.count() if str(n) not in used)
    aliases={p:(p if p in used else next(available)) for p in sorted(source_ids)}
    for row in manifest['parts']:
        original_id=row['part_id']
        pid=aliases[original_id]
        filename=row['mesh']
        if Path(filename).name!=filename or not filename.lower().endswith('.obj'):
            raise ValueError('mesh must be a sibling OBJ')
        meshpath=location(folder/filename)
        rawhash=sha(meshpath)
        if folder != raw:
            original[filename]=rawhash
            if not (raw/filename).exists():
                shutil.copyfile(meshpath,raw/filename)
        matrix=rigid(row['assembly_matrix_mm'])
        frame=row['mesh_coordinate_frame']
        if frame not in ('assembly','part_local'):
            raise ValueError('unknown mesh frame')
        mesh=trimesh.load_mesh(str(meshpath),process=False,maintain_order=True,skip_materials=True)
        if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError('invalid part mesh')
        mm=np.asarray(mesh.vertices,dtype=float)*SCALE[unit]
        if not np.isfinite(mm).all():
            raise ValueError('nonfinite mesh')
        local=vertices_after(mm,np.linalg.inv(matrix)) if frame=='assembly' else mm
        world=vertices_after(local,matrix)
        clean=trimesh.Trimesh(vertices=local,faces=mesh.faces,process=True)
        vol=float(clean.volume)
        good=clean.is_watertight and clean.is_winding_consistent and math.isfinite(vol) and vol>0
        low,high=world.min(axis=0),world.max(axis=0)
        rotation=Rotation.from_matrix(matrix[:3,:3])
        config[pid]={'final_state':(matrix[:3,3]/10).tolist()+rotation.as_euler('xyz').tolist()}
        # Preserve native OBJ bytes for already-native local cm meshes.
        if SCALE[unit]==10 and frame=='part_local':
            shutil.copyfile(meshpath,sim/(pid+'.obj'))
        else:
            objwrite(sim/(pid+'.obj'),local/10,mesh.faces)
        objects[pid]=collision_mesh(world,np.asarray(mesh.faces,dtype=np.int32))
        parts[pid]=dict(part_id=pid,source_part_id=original_id,geometry_id=row.get('geometry_id') or 'mesh-'+rawhash[:20],
            cad_name=row.get('cad_name'),bbox_size_mm=(high-low).tolist(),bbox_center_mm=((high+low)/2).tolist(),
            volume_mm3=vol if good else None,volume_status='closed_oriented_mesh' if good else 'unverified_mesh_volume',
            assembly_pose=dict(translation_mm=matrix[:3,3].tolist(),rotation_xyzw=rotation.as_quat().tolist()))
        mappings[pid]=dict(part_id=pid,source_part_id=original_id,geometry_id=parts[pid]['geometry_id'],source_mesh='source/'+filename,
            source_sha256=rawhash,mesh_coordinate_frame=frame,assembly_pose_applied_to_source_mesh=frame=='assembly',
            assembly_matrix_mm=matrix.tolist(),native_mesh='native/'+pid+'.obj',native_mesh_sha256=sha(sim/(pid+'.obj')),
            native_mesh_frame='part_local',occurrence_label=row.get('occurrence_label'),definition_label=row.get('definition_label'))
    if len(parts)<2:
        raise ValueError('at least two distinct occurrences required')
    write(sim/'config.json',config)
    pairs=[]
    for a,b in itertools.combinations(sorted(parts),2):
        hit=fcl.collide(objects[a],objects[b],fcl.CollisionRequest(),fcl.CollisionResult())
        distance=0. if hit else float(fcl.distance(objects[a],objects[b],fcl.DistanceRequest(),fcl.DistanceResult()))
        valid=math.isfinite(distance) and distance>=0
        pairs.append(dict(parts=[a,b],min_mesh_surface_distance_mm=distance if valid else None,
                          within_tolerance=bool(distance<=tolerance) if valid else None,
                          method='FCL triangle BVH intersection/distance',intersecting_triangles=bool(hit)))
    output=dict(assembly_id=assembly,source_hash=digest(original),source_files=original,source_unit=unit,
        step_declared_units=manifest.get('step_declared_length_units'),coordinate_system=dict(reference_frame='assembly',
        length_unit='mm',rotation='xyzw',assembly_mm_to_native_cm=np.diag([.1,.1,.1,1]).tolist()),
        tolerance_mm=tolerance,parts=list(parts.values()),nearby_relations=[p for p in pairs if p['within_tolerance']])
    write(out/'geometry.json',output)
    write(out/'part_mapping.json',mappings)
    write(out/'surface_distances.json',pairs)
    return output
