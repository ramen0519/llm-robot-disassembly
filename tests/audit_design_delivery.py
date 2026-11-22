"""Local verification of actual research artifacts; takes input paths explicitly."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from caseplanner.io import read,sha,write
from check_disassembly_export import check

def main(out, release=False):
    parent=ROOT/'evidence'
    run_names=['beam_release','battery_one_target','diagnostic_tray_01','contract_pair_01/collect','contract_pair_01/query']
    checks={}
    for name in run_names:
        run=parent/name
        checks[name]=check(run/'export_v1.json')
        meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(run/'execution.mp4')]))
        stream=next(v for v in meta['streams'] if v['codec_type']=='video')
        media=read(run/'media.json')
        assert int(stream['nb_frames'])==media['frames'] and sha(run/media['video'])==media['sha256']
        checks[name].update(codec=stream['codec_name'],frames=media['frames'],seconds=read(run/'result.json')['seconds'])
        for view in ('top','oblique'):
            with Image.open(run/('input/'+view+'.png')) as im: assert im.size==(768,768)
    pre=parent/'battery_step_02/input'
    geom=read(pre/'geometry.json')
    independent=read(Path('/mnt/f/disassembly_research/battery_pack_asap_straight_20261120_v2/export.json'))
    assert len(geom['parts'])==len(independent['parts'])==13
    delta=[]
    for part,other in zip(geom['parts'],independent['parts']):
        b=np.asarray(other['bounds_mm'])
        delta.extend(abs(b.mean(axis=0)-part['bbox_center_mm']))
        delta.extend(abs(b[1]-b[0]-part['bbox_size_mm']))
    assert max(delta)<1.e-5
    checks['battery_step']=dict(occurrences=13,geometry_count=len(set(p['geometry_id'] for p in geom['parts'])),
       maximum_independent_bbox_delta_mm=float(max(delta)),surface_pairs=len(read(pre/'surface_distances.json')),
       near_pairs=len(geom['nearby_relations']),source_sha256=geom['source_files'],unknown_mesh_volumes=[
       p['part_id'] for p in geom['parts'] if p['volume_mm3'] is None])
    for view in ('top','oblique'):
        a=np.asarray(Image.open(parent/'beam_input/input'/(view+'.png')),dtype=int)
        b=np.asarray(Image.open(parent/'beam_release/input'/(view+'.png')),dtype=int)
        checks['renderer_'+view]=dict(max_channel_delta=int(np.abs(a-b).max()),
            changed_pixel_fraction=float(np.any(a!=b,axis=2).mean()))
        assert checks['renderer_'+view]['changed_pixel_fraction']<.001
    e=read(parent/'battery_one_target/incidents.json')[0]
    obs=e['representative']
    rec=read(parent/'battery_one_target/checks'/(obs['check_id']+'.json'))
    assert obs['stage']=='stability' and obs['target_part'] not in rec['endpoint'] and obs['present']==sorted(rec['endpoint'])
    assert e['state']=='unresolved' and not read(parent/'battery_one_target/result.json')['new_cases']
    checks['battery_rep']=dict(check=obs['check_id'],part=obs['target_part'],present=obs['present'],
         outcome=obs['outcome'],png_sha256=obs['png_sha256'],index_hits=read(parent/'battery_one_target/retrieval.json')['lookups'][0]['hit_ids'])
    checks['paired_contract']=read(parent/'contract_pair_01/verification.json')
    if release:
        import hashlib
        import sqlite3
        import zipfile
        for name,archive_name in [('beam_transfer_final','beam_transfer_complete.zip'),
                                  ('contract_pair_transfer/query','tray_case_transfer.zip'),
                                  ('battery_transfer_final','battery_transfer_complete.zip')]:
            run=parent/name
            export=read(run/'export_v1.json')
            verified=check(run/'export_v1.json')
            with zipfile.ZipFile(parent/archive_name) as archive:
                expected=dict(export['artifacts'])
                expected['export_v1.json']=sha(run/'export_v1.json')
                assert set(archive.namelist())==set(expected) and len(archive.infolist())==len(expected)
                for member,wanted in expected.items():
                    h=hashlib.sha256()
                    with archive.open(member) as stream:
                        for block in iter(lambda:stream.read(1048576),b''):
                            h.update(block)
                    assert h.hexdigest()==wanted,member
                assert 'input/surface_distances.json' in expected
            verified.update(zip_members=len(expected),zip_sha256=sha(parent/archive_name))
            checks[name]=verified
        checks['two_part_terminal']=check(parent/'two_part_terminal/export_v1.json')
        terminal=read(parent/'two_part_terminal/result.json')
        assert terminal['accepted_removals']==1 and terminal['chain'][0]['chosen_by']=='ASAP terminal mass/base'
        assert terminal['api_calls']==0
        case=read(parent/'contract_pair_transfer/verification.json')
        assert case['retrieved'] and case['new_query_case_excluded'] and not case['live_model_calls']
        checks['latest_pair']=case
        with sqlite3.connect('file:'+str(ROOT/'.cases/index.sqlite')+'?mode=ro',uri=True) as db:
            checks['research_case_count']=db.execute('select count(*) from cases').fetchone()[0]
        assert checks['research_case_count']==0
    write(out,checks)
    print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--release',action='store_true')
    args=parser.parse_args()
    main(args.output,args.release)
