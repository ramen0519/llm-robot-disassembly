"""Prepare a held-out evaluation without solver outcomes or paid API calls."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import zipfile

ROOT = Path(__file__).absolute().parents[1]
RUN = ROOT / 'runs/unseen_transfer_pilot_20261019'
ARCHIVE = ROOT / 'runs/assembly_rules_20261017/test_assembly.zip'
ARCHIVE_SHA = 'c89ea240b1962e3230b00487bf23f3fcbb8b5cd7b6ce1a6816e043a31cac09c2'


def safe(path):
    raw = str(path).replace('\\', '/')
    if raw.lower().startswith(('e:', '/mnt/e', '//')) or '..' in raw.split('/'):
        raise ValueError('forbidden_path')
    p = Path(path).absolute()
    p.relative_to(ROOT)
    for parent in [p] + list(p.parents):
        if parent == ROOT.parent:
            break
        if parent.exists() and (parent.is_symlink() or getattr(parent.stat(follow_symlinks=False), 'st_file_attributes', 0) & 0x400):
            raise ValueError('linked_path')
    return p


def sha(p):
    h = hashlib.sha256()
    with safe(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    p = safe(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def walk(folder):
    for here, dirs, files in os.walk(safe(folder), followlinks=False):
        for name in dirs + files:
            safe(Path(here) / name)
        for name in files:
            yield Path(here) / name


def preserve():
    result = {}
    for p in walk(ROOT / 'runs'):
        if RUN in p.parents:
            continue
        result[p.relative_to(ROOT).as_posix()] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    return result


def exposures():
    known = {k: ['prior_explicit_user_history'] for k in ('00561','02628','01289','07009','01256')}
    names = {'config.json','context.json','task_context.json','protocol.json','scope.json',
             'case_report.json','worker_result.json','dataset_selection.json','selected_assemblies.json',
             'features.json','library_for_reuse.json','skills.json','manifest.json'}
    examined = 0
    def visit(obj, source, parent=''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ('assembly_id','assembly','task_id','selected_id') and isinstance(value,str) and re.fullmatch(r'\d{5}',value):
                    known.setdefault(value, []).append(source)
                if key in ('assembly_dir','assembly_path') and isinstance(value,str):
                    for ident in re.findall(r'(?:/|\\)(\d{5})(?=/|\\|$)',value):
                        known.setdefault(ident, []).append(source)
                if key in ('assemblies','selected_assemblies') and isinstance(value,list):
                    for ident in value:
                        if isinstance(ident,str) and re.fullmatch(r'\d{5}',ident):
                            known.setdefault(ident, []).append(source)
                if key == 'id' and parent == 'selected_assemblies.json' and isinstance(value,str) and re.fullmatch(r'\d{5}',value):
                    known.setdefault(value, []).append(source)
                visit(value,source,parent)
        elif isinstance(obj,list):
            for row in obj:
                visit(row,source,parent)
    for p in walk(ROOT/'runs'):
        if RUN in p.parents or p.name not in names or p.stat().st_size > 16*1024*1024:
            continue
        if any(s in p.parts for s in ('implementation','source_snapshot','sources','snapshot')):
            continue
        try:
            value=json.loads(p.read_text(encoding='utf-8-sig'))
        except (ValueError,UnicodeError):
            continue
        examined += 1
        visit(value,p.relative_to(ROOT).as_posix(),p.name)
    return {'examined_json_files':examined,'exposed':{k:sorted(set(v)) for k,v in sorted(known.items())}}


def normalized_geometry(payload):
    """Translation/uniform-scale invariant vertex fingerprint; not topology proof."""
    vertices=[]
    for line in payload.decode('utf-8').splitlines():
        if line.startswith('v '):
            vertices.append(tuple(float(v) for v in line.split()[1:4]))
    if not vertices or any(len(v)!=3 or not all(__import__('math').isfinite(x) for x in v) for v in vertices):
        raise ValueError('invalid_vertices')
    lo=[min(v[a] for v in vertices) for a in range(3)]
    hi=[max(v[a] for v in vertices) for a in range(3)]
    scale=max(hi[a]-lo[a] for a in range(3))
    if scale<=0:
        raise ValueError('degenerate_vertices')
    normalized=sorted(set(tuple(round((v[a]-lo[a])/scale,7) for a in range(3)) for v in vertices))
    return hashlib.sha256(json.dumps(normalized,separators=(',',':')).encode()).hexdigest()


def prepare():
    safe(RUN).mkdir(exist_ok=False)
    write(RUN/'authorization.json',dict(max_calls=360,max_usd=5,retries=0,
        scope='three unseen assemblies, four conditions, one seed; no skill synthesis',
        new_user_approval=True,old_01256_ledger_used=False))
    print('HASHING_PREVIOUS_RESULTS',flush=True)
    write(RUN/'preserved_before.json',preserve())
    originals={}
    for folder in ('worker','orchestrator','tools','tests'):
        for p in sorted((ROOT/folder).glob('*.py')):
            originals[p.relative_to(ROOT).as_posix()]=sha(p)
            target=RUN/'original_sources'/p.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
    write(RUN/'original_sources.json',originals)
    exposure=exposures()
    write(RUN/'exposure_audit.json',exposure)
    if sha(ARCHIVE)!=ARCHIVE_SHA:
        raise ValueError('archive_sha_mismatch')
    inventory=json.loads((ROOT/'runs/assembly_rules_20261017/dataset_inventory.json').read_text())
    with zipfile.ZipFile(ARCHIVE) as z:
        signatures={}
        def signature(ident):
            if ident not in signatures:
                names=sorted(n for n in z.namelist() if n.startswith('test_assembly/'+ident+'/') and n.endswith('.obj'))
                signatures[ident]=Counter(normalized_geometry(z.read(n)) for n in names)
            return signatures[ident]
        development={k:signature(k) for k in exposure['exposed']}
        chosen=[]; exclusion=[]
        for lo,hi in ((10,12),(13,16),(17,20)):
            candidates=sorted((x for x in inventory if lo<=x['parts']<=hi and x['id'] not in exposure['exposed']),
                key=lambda x:hashlib.sha256(('unseen-transfer-v1|'+x['id']).encode()).hexdigest())
            for item in candidates:
                ident=item['id']; sig=signature(ident)
                related=[other for other,known in development.items() if known and sum((sig & known).values())/min(sum(sig.values()),sum(known.values()))>=0.5]
                if related:
                    exclusion.append({'id':ident,'reason':'at_least_half_parts_share_normalized_vertices','with':related});continue
                dest=RUN/'raw_assemblies'/ident;dest.mkdir(parents=True)
                hashes={}
                for name in sorted(z.namelist()):
                    if not name.startswith('test_assembly/'+ident+'/') or name.endswith('/'):
                        continue
                    leaf=name.split('/')[-1]
                    if leaf not in ('contact_graph.json',) and not leaf.endswith('.obj'):
                        raise ValueError('unexpected_archive_member')
                    if name!='test_assembly/'+ident+'/'+leaf:
                        raise ValueError('unsafe_archive_member')
                    data=z.read(name)
                    with (dest/leaf).open('xb') as f:f.write(data)
                    hashes[leaf]=hashlib.sha256(data).hexdigest()
                chosen.append(dict(item,files=hashes,band=[lo,hi],selection_score=hashlib.sha256(('unseen-transfer-v1|'+ident).encode()).hexdigest(),normalized_vertex_multiset=dict(sig)))
                development[ident]=sig
                break
            else:raise ValueError('empty_eligible_band')
    write(RUN/'selection.json',dict(selected=chosen,excluded=exclusion,archive_sha256=ARCHIVE_SHA,
        rule='first SHA256(unseen-transfer-v1|id) per band after provenance/geometry exclusions',
        solver_results_used=False,near_duplicate_limit='translation/uniform scale invariant vertices only; arbitrary rotations not certified',
        evaluation_status='not_started'))
    print(json.dumps({'selected':[(x['id'],x['parts']) for x in chosen],'exposed':list(exposure['exposed']), 'previous_files':len(json.loads((RUN/'preserved_before.json').read_text()))}),flush=True)


if __name__=='__main__':
    prepare()
