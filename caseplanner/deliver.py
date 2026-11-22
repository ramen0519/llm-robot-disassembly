"""Render recorded transforms and export the connected chain as a relocatable bundle."""
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from .io import digest,read,sha,write
from .memory import verified


def frames(check):
    if check['stage']=='path':
        fixed=check.get('first') or {}
    else: fixed={p:check['endpoint'][p] for p in check.get('fixed',[])}
    result=[]
    for sample in check.get('frames',[]):
        result.append((sample['index'],dict(fixed,**sample['poses'])))
    if check.get('endpoint'):
        result.append((check.get('samples',0),check['endpoint']))
    return result


def video(session,checks,custom_clips=None,name='execution'):
    run,scene=session.run,session.scene
    clips=[]
    for step in session.best:
        c=checks[step['path_check']]
        clips.append((c,'Accepted '+step['part']+' | '+step['attempt']))
    if not clips and session.obs.checks:
        c=next((c for c in reversed(session.obs.checks) if c.get('endpoint')),None)
        if c: clips.append((c,c['stage']+' | '+c['outcome']+' | '+c['attempt']))
    if custom_clips is not None: clips=custom_clips
    if not clips: return dict(status='NO_PHYSICAL_FRAMES',video=None)
    samples=[]
    for c,label in clips:
        fs=frames(c)
        count=min(16,len(fs))
        if not count: continue
        for idx in np.unique(np.linspace(0,len(fs)-1,count,dtype=int)):
            k,poses=fs[idx]
            samples.append((c['id'],k,poses,label))
    if not samples: return dict(status='NO_PHYSICAL_FRAMES',video=None)
    # A single fixed camera includes the initial assembly and every actual displayed pose.
    all_points=[np.concatenate(list(scene.world(scene.assembled).values()))]
    for _,_,poses,_ in samples: all_points.append(np.concatenate(list(scene.world(poses).values())))
    cam=scene.camera({'extent':np.concatenate(all_points)},(1,1,1))
    target=run/(name+'.mp4')
    command=['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','480x480',
             '-r','8','-i','-','-an','-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    ledger=[]
    try:
        for index,(cid,k,poses,label) in enumerate(samples):
            image=scene.render(poses,camera=cam,size=480)
            draw=ImageDraw.Draw(image)
            draw.rectangle((0,451,479,479),fill=(28,42,54))
            draw.text((10,458),label+' | '+cid,fill='white')
            if index in (0,len(samples)-1): image.save(run/((name+'_frame_%03d.png')%index))
            proc.stdin.write(image.tobytes())
            ledger.append(dict(video_frame=index,check=cid,sample_index=k,present=sorted(poses)))
        proc.stdin.close()
        err=proc.stderr.read()
        if proc.wait(): raise RuntimeError('ffmpeg: '+err.decode(errors='replace')[:300])
    finally:
        if proc.poll() is None: proc.kill();proc.wait()
    write(run/(name+'_frames.json'),ledger)
    return dict(status='RECORDED_TRANSFORMS_RENDERED',video=target.name,sha256=sha(target),frames=len(ledger),
        capture='native physics transforms sampled in this run, displayed at 8 fps; cuts between attempts',
        dynamic=any(c.get('samples',0)>1 for c,_ in clips))


def finalize(session,stop,elapsed):
    s=session
    run=s.run
    all_ids=sorted(s.scene.meshes)
    chain=s.best
    if chain and not verified(chain,s.root,chain[-1]['part']): raise RuntimeError('export disconnected')
    remainder=chain[-1]['after'] if chain else all_ids
    complete=len(remainder)==1 and len(chain)==len(all_ids)-1
    attempts=s.attempts
    write(run/'attempts.json',attempts)
    write(run/'incidents.json',list(s.inc.events.values()))
    write(run/'retrieval.json',dict(frozen_index=s.library.index,lookups=s.lookups,decisions=s.selections,registration=s.registrations))
    checks={c['id']:c for c in s.obs.checks}
    media_start=time.monotonic()
    try:
        media=video(s,checks)
        media['incident_clips']=[]
        for e in s.inc.events.values():
            cid=e['representative']['check_id']
            if not cid or cid not in checks: continue
            clips=[(checks[cid],e['representative']['stage']+' | '+e['representative']['outcome'])]
            if e['state']=='resolved':
                clips.extend((checks[st['path_check']],'Accepted '+st['part']+' | '+st['attempt']) for st in e['chain'])
            if len(clips)==1 and not checks[cid].get('samples'):
                media['incident_clips'].append(dict(incident=e['id'],status='STATIC_DETECTION_NO_DYNAMIC_FRAMES',png=e['representative'].get('png')))
            else:
                media['incident_clips'].append(dict(incident=e['id'],clip=video(s,checks,clips,'incident_'+e['id'][:12])))
    except Exception as exc: media=dict(status='MEDIA_ERROR',detail=str(exc),video=None)
    media['render_seconds']=time.monotonic()-media_start
    write(run/'media.json',media)
    score=[len(step['additional_support']) for step in chain]
    result=dict(status='COMPLETE' if complete else 'PARTIAL' if chain else 'NO_REMOVAL',
        observation_status=('error' if str(stop).startswith(('model_error','native_exception')) else 'recorded'),
        run_id=s.id,rank_source=s.model.provider,stop=stop,accepted_removals=len(chain),required_removals=len(all_ids)-1,
        chain=chain,remaining=remainder,seconds=elapsed,counts=s.obs.count,composite_attempts=len(attempts),
        max_additional_support=max(score) if score else None,cumulative_additional_support=sum(score) if score else None,
        support_steps=score,api_calls=s.model.calls if s.model.provider=='openai_responses' else 0,
        offline_model_messages=s.model.calls if s.model.provider=='offline_responses_double' else 0,
        api_estimated_reported_usd=str(s.model.reported) if s.model.provider=='openai_responses' else '0',
        api_charged_or_reserved_estimate_usd=str(s.model.booked) if s.model.provider=='openai_responses' else '0',incidents=len(s.inc.events),
        resolved=sum(e['state']=='resolved' for e in s.inc.events.values()),new_cases=[r['case_id'] for r in s.registrations if r['case_id']],
        retrieval_selections=sum(len(r['applied']) for r in s.selections),media_status=media['status'])
    write(run/'result.json',result)
    parts=read(run/'input/part_mapping.json')
    refs=['result.json','settings.json','physics.json','input/geometry.json','input/part_mapping.json','input/views.json',
          'input/top.png','input/oblique.png','input/native/config.json','attempts.json','incidents.json','retrieval.json','media.json']
    refs.extend(p for p in ('code_versions.json','native_input.json','frozen_index.json') if (run/p).is_file())
    for m in parts.values():
        for key in ('source_mesh','native_mesh'):
            m[key]='input/'+m[key]
            refs.append(m[key])
    for step in chain: refs.extend([step['path']]+step['check_evidence'])
    refs+=['input/source/'+name for name in s.scene.geom['source_files']]
    if (run/'events.jsonl').exists(): refs.append('events.jsonl')
    # Include failed and discarded checks referenced by the run log as well as the accepted path.
    for pattern in ('input/*.json', 'checks/check_*.json', 'checks/check_*.npz',
                    'checks/check_*_target.png', 'checks/check_*_embedding.npy',
                    'execution.mp4', 'execution_frames.json', 'execution_frame_*.png', 'observer_errors.jsonl',
                    'incident_*.mp4', 'incident_*_frames.json', 'incident_*_frame_*.png'):
        refs.extend(path.relative_to(run).as_posix() for path in run.glob(pattern))
    hashes={p:sha(run/p) for p in sorted(set(refs))}
    product=dict(schema='disassembly_export/v1',run_id=s.id,source_hash=s.scene.geom['source_hash'],
        run_hash=digest(hashes),artifacts=hashes,evidence_scope='all_native_checks_and_media/v1',coordinate_system=s.scene.geom['coordinate_system'],part_mapping=parts,
        initial_assembled_parts=all_ids,verified_transitions=chain,remaining_parts=remainder,completion=result['status'],
        reversed_local_removals=[st['part'] for st in reversed(chain)],assembly_candidate=(
          dict(initial_base=remainder[0],add_parts=[st['part'] for st in reversed(chain)]) if complete else None),
        trajectory_convention='ASAP joint_q in native cm, RedMax free3d-exp joint state; use recorded mesh-local-to-world matrices for display',
        robot_verified=False)
    write(run/'export_v1.json',product)
    print('result',result['status'],'removals',len(chain),'stop',stop,'physics',s.obs.count,'media',media['status'],flush=True)
    if s.failure and str(s.failure).startswith('model_error'): return 2
    if str(stop).startswith('native_exception') or media['status']=='MEDIA_ERROR': return 2
    return 0 if complete else 3
