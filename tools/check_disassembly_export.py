"""Verify this transfer bundle without importing RedMax. Optionally pack it for assembly."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from caseplanner.io import digest,location,read,sha


def check(file):
    file=location(file)
    root=file.parent
    d=read(file)
    if d['schema']!='disassembly_export/v1' or d['robot_verified'] is not False:
        raise ValueError('unsupported export')
    hashes=d['artifacts']
    if digest(hashes)!=d['run_hash']: raise ValueError('run hash mismatch')
    for item,expected in hashes.items():
        p=Path(item)
        if p.is_absolute() or '..' in p.parts or ':' in item or chr(92) in item or sha(root/p)!=expected:
            raise ValueError('broken relative artifact: '+item)
    geometry=read(root/'input/geometry.json')
    if digest(geometry['source_files'])!=d['source_hash']: raise ValueError('source inventory')
    for k,expected in geometry['source_files'].items():
        if hashes['input/source/'+k]!=expected: raise ValueError('source hash changed')
    for pid,row in d['part_mapping'].items():
        if pid!=row['part_id'] or hashes[row['native_mesh']]!=row['native_mesh_sha256'] or hashes[row['source_mesh']]!=row['source_sha256']:
            raise ValueError('broken part mapping')
    if d.get('evidence_scope') == 'all_native_checks_and_media/v1':
        def require_artifact(name):
            if name not in hashes:
                raise ValueError('missing packaged evidence: ' + str(name))
        require_artifact('input/surface_distances.json')
        for attempt in read(root/'attempts.json'):
            for cid in attempt['checks']:
                record='checks/'+cid+'.json'
                require_artifact(record)
                trajectory=read(root/record).get('trajectory')
                if trajectory:
                    require_artifact(trajectory)
        for event in read(root/'incidents.json'):
            rep=event['representative']
            if rep.get('png'):
                require_artifact(rep['png'])
                if rep['png_sha256']!=hashes[rep['png']]:
                    raise ValueError('representative image is not the recorded failure')
        media=read(root/'media.json')
        clips=[media]+[item['clip'] for item in media.get('incident_clips',[]) if 'clip' in item]
        for clip in clips:
            if not clip.get('video'):
                continue
            name=clip['video']
            require_artifact(name)
            if clip['sha256']!=hashes[name]:
                raise ValueError('media provenance mismatch')
            stem=Path(name).stem
            require_artifact(stem+'_frames.json')
            ledger=read(root/(stem+'_frames.json'))
            if len(ledger)!=clip['frames']:
                raise ValueError('video frame count mismatch')
            for entry in ledger:
                require_artifact('checks/'+entry['check']+'.json')
            if ledger:
                require_artifact(stem+'_frame_000.png')
                require_artifact(stem+'_frame_%03d.png'%(len(ledger)-1))
    remain=set(d['initial_assembled_parts'])
    uid=None; sequence=[]
    for step in d['verified_transitions']:
        p=step['part']
        if not step['accepted'] or not step['physics_pass'] or set(step['before'])!=remain or p not in remain:
            raise ValueError('unaccepted transition')
        if uid is not None and step['parent_uid']!=uid: raise ValueError('branch discontinuity')
        remain.remove(p);sequence.append(p);uid=step['child_uid']
        if set(step['after'])!=remain: raise ValueError('wrong successor')
        path=read(root/('checks/'+step['path_check']+'.json'))
        if path['target']!=p or not path['passed'] or path['trajectory']!=step['path'] or step['path'] not in hashes:
            raise ValueError('missing successful native trajectory')
        for s in step['stability_checks']:
            support=read(root/('checks/'+s+'.json'))
            if support['stage']!='stability' or not support['passed'] or support['attempt']!=step['attempt']:
                raise ValueError('missing successful stability check')
        if path['attempt']!=step['attempt'] or not step['stability_checks']: raise ValueError('cross-attempt evidence')
    if set(d['remaining_parts'])!=remain or list(reversed(sequence))!=d['reversed_local_removals']:
        raise ValueError('residual/reverse mismatch')
    if len(remain)==1 and len(sequence)==len(d['initial_assembled_parts'])-1:
        if d['assembly_candidate']!=dict(initial_base=next(iter(remain)),add_parts=sequence[::-1]):
            raise ValueError('invalid complete assembly proposal')
    elif d['assembly_candidate'] is not None: raise ValueError('partial promoted to full assembly candidate')
    return dict(verified_files=len(hashes),run_id=d['run_id'],removals=sequence,residual=sorted(remain),robot_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('export')
    p.add_argument('--zip',type=Path)
    a=p.parse_args()
    export=location(a.export)
    result=check(export)
    if a.zip:
        dest=location(a.zip)
        with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
            archive.write(export,'export_v1.json')
            for name in read(export)['artifacts']:
                archive.write(export.parent/name,name)
        with zipfile.ZipFile(dest) as archive:
            expected=dict(read(export)['artifacts'])
            expected['export_v1.json']=sha(export)
            if len(expected)!=len(archive.infolist()) or set(expected)!=set(archive.namelist()):
                raise ValueError('ZIP inventory mismatch')
            for name,wanted in expected.items():
                checksum=hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1048576), b''):
                        checksum.update(block)
                if checksum.hexdigest()!=wanted:
                    raise ValueError('ZIP checksum failed for '+name)
        result['zip_sha256']=sha(dest)
    print(json.dumps(result,ensure_ascii=False))
