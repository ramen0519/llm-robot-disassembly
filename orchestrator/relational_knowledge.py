"""Versioned, read-only projections of historical evidence, lessons and skills.

No model, generated-code execution, geometry engine or physical verdicts. Missing
measurements remain missing; graph features describe the recorded reference CAD.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re

ROOT = Path(__file__).absolute().parents[1]
PROTOCOL = 'relational_knowledge/v1'
MAX_LIBRARY_BYTES = 256 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def safe_path(value, root=None):
    """Reject forbidden drives lexically, then every junction/symlink component."""
    root = Path(root or ROOT).absolute()
    text = str(value)
    if (re.search(r'(?i)(?:^|[/\\])e:', text) or PureWindowsPath(text).drive.upper() == 'E:'
            or PureWindowsPath(str(root)).drive.upper() == 'E:'):
        raise ValueError('forbidden_drive')
    if text.startswith(('\\\\', '//')) or '..' in text.replace('\\', '/').split('/'):
        raise ValueError('unsafe_knowledge_path')
    path = Path(text)
    path = path if path.is_absolute() else root / path
    path = path.absolute()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError('knowledge_path_outside_workspace') from None
    # Ancestors first: never inspect a child through an unchecked junction.
    for ancestor in reversed([path, *path.parents]):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        if ancestor.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('knowledge_reparse_path')
    return path


def read_json(path, root=None):
    path = safe_path(path, root)
    with path.open('rb') as stream:
        raw = stream.read(MAX_LIBRARY_BYTES + 1)
    if len(raw) > MAX_LIBRARY_BYTES:
        raise ValueError('knowledge_file_too_large')
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate_knowledge_key')
            out[key] = value
        return out
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_knowledge')))


def validate_library(value):
    if not isinstance(value, dict) or value.get('protocol') != PROTOCOL or not isinstance(value.get('cards'), list):
        raise ValueError('invalid_relational_library')
    if value.get('content_sha256') != digest({k:v for k,v in value.items() if k != 'content_sha256'}):
        raise ValueError('relational_library_hash_mismatch')
    identities = set()
    for card in value['cards']:
        identity = card.get('card_id')
        if not isinstance(identity, str) or identity in identities or card.get('kind') not in ('experience', 'lesson', 'skill'):
            raise ValueError('invalid_or_duplicate_relational_card')
        identities.add(identity)
        if not isinstance(card.get('content'), dict) or not isinstance(card.get('origin'), dict):
            raise ValueError('invalid_relational_content')
        if card.get('execution_authorized') is not False or card.get('physics_reuse_allowed') is not False:
            raise ValueError('relational_card_cannot_authorize_execution')
        if not isinstance(card.get('evidence_ids'), list):
            raise ValueError('invalid_relational_links')
    for card in value['cards']:
        if not set(card['evidence_ids']) <= identities:
            raise ValueError('dangling_relational_evidence_link')
    return value


def load_library(path):
    return validate_library(read_json(path))


def freeze_library(cards, **metadata):
    value = dict(protocol=PROTOCOL, cards=cards, metadata=metadata,
                 original_records_immutable=True, new_learning=False,
                 hard_pruning=False, physics_reuse_allowed=False)
    value['content_sha256'] = digest(value)
    return validate_library(value)


def _vector(value):
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(
            type(x) in (int, float) and math.isfinite(x) for x in value):
        return list(value)
    return None


def shape_from_part(part, unit=None, frame=None):
    part = part if isinstance(part, dict) else {}
    bounds = part.get('bounds')
    lo = hi = None
    if isinstance(bounds, list) and len(bounds) == 2:
        lo, hi = _vector(bounds[0]), _vector(bounds[1])
    extents = _vector(part.get('extents', part.get('extents_xyz')))
    if extents is None and lo is not None and hi is not None:
        extents = [b-a for a,b in zip(lo, hi)]
    if extents is not None and (min(extents) < 0 or max(extents) <= 0):
        extents = None
    center = _vector(part.get('bounds_center'))
    if center is None and lo is not None and hi is not None:
        center = [(a+b)/2 for a,b in zip(lo, hi)]
    return dict(extents_xyz=extents,
                ratio_xyz=[x/max(extents) for x in extents] if extents else None,
                ratio_sorted=sorted((x/max(extents) for x in extents), reverse=True) if extents else None,
                center_xyz=center, length_unit=unit, frame=frame,
                representation='axis_aligned_bounding_box',
                invariance='axis_permutation_only_not_arbitrary_rotation',
                exact_surface_shape_known=False)


def _component_count(nodes, edges):
    adjacent = {n:set() for n in nodes}
    for a,b in edges:
        adjacent[a].add(b); adjacent[b].add(a)
    unseen = set(nodes)
    count = 0
    while unseen:
        count += 1
        todo = [unseen.pop()]
        while todo:
            fresh = adjacent[todo.pop()] & unseen
            unseen -= fresh; todo.extend(fresh)
    return count


def context_from_geometry(geometry, target_part_id, remaining_parts=None, conditions=None):
    geometry = geometry if isinstance(geometry, dict) else {}
    parts = geometry.get('parts', geometry.get('geometry_features', {}))
    parts = parts if isinstance(parts, dict) else {}
    target = str(target_part_id) if target_part_id is not None else None
    remaining = list(map(str, remaining_parts)) if remaining_parts is not None else list(parts)
    known = set(remaining)
    unit = geometry.get('length_unit', geometry.get('units'))
    frame = geometry.get('frame', geometry.get('coordinate_frame'))
    shape = shape_from_part(parts.get(target), unit, frame)
    source_edges = geometry.get('contact_edges')
    if source_edges is None and isinstance(geometry.get('relations'), dict):
        source_edges = geometry['relations'].get('contact_edges')
    valid_edges = isinstance(source_edges, list) and all(
        isinstance(e, (list, tuple)) and len(e) == 2 and str(e[0]) != str(e[1]) for e in source_edges)
    edges = sorted({tuple(sorted(map(str, e))) for e in (source_edges or [])
                    if valid_edges and set(map(str, e)) <= known}) if valid_edges else None
    neighbors = []
    if edges is not None and target in known:
        for other in sorted({b if a == target else a for a,b in edges if target in (a,b)}):
            other_shape = shape_from_part(parts.get(other), unit, frame)
            center = other_shape['center_xyz']; origin = shape['center_xyz']
            neighbors.append(dict(part_id=other, shape=other_shape,
                center_delta_xyz=[a-b for a,b in zip(center, origin)] if center is not None and origin is not None else None,
                contact=True, relation='recorded_reference_contact',
                contact_position=None, contact_normal=None, contact_area=None,
                fastening=None, supports_target=None, blocks_motion=None))
    topology = {key: None for key in ('parent_count', 'contact_degree', 'degree_fraction',
                  'components_before', 'components_if_removed', 'component_delta', 'remainder_isolated_nodes')}
    if remaining:
        topology['parent_count'] = len(remaining)
    if edges is not None and target in known:
        child = known - {target}; child_edges = [e for e in edges if target not in e]
        before = _component_count(known, edges); after = _component_count(child, child_edges)
        connected = set(x for e in child_edges for x in e)
        topology.update(contact_degree=len(neighbors), degree_fraction=len(neighbors)/max(1, len(known)-1),
                        components_before=before, components_if_removed=after, component_delta=after-before,
                        remainder_isolated_nodes=len(child-connected))
    return dict(product_id=geometry.get('product_id'), target_part_id=target,
                remaining_parts=remaining, shape=shape, neighbors=neighbors, topology=topology,
                conditions=deepcopy(conditions or {}),
                relation_scope='recorded_reference_geometry_restricted_to_remaining_parts',
                dynamic_contacts_observed=False, contact_source=geometry.get('contact_source'),
                contact_eps=geometry.get('contact_eps'),
                evidence_status=dict(shape='derived' if shape['extents_xyz'] else 'missing',
                                     contacts='observed_reference' if edges is not None else 'missing',
                                     topology='derived_static_graph' if edges is not None else 'missing',
                                     fastening='missing', support_relation='missing', blocker='missing'),
                missing_features=[k for k,v in {'shape':shape['extents_xyz'], 'contact_edges':edges,
                    'length_unit':unit, 'frame':frame, 'contact_surface':None, 'dynamic_response':None}.items() if v is None])


def make_card(imported, context=None):
    """Preserve the author's text/program, status and all original observations."""
    record = deepcopy(imported['record'])
    kind = imported['kind']
    effective = record.get('candidate', record.get('lesson', record))
    effective = effective if isinstance(effective, dict) else record
    provenance = effective.get('provenance') or effective.get('generation') or {}
    provenance = provenance if isinstance(provenance, dict) else {}
    origin = dict(source_path=imported['source_path'], json_pointer=imported.get('json_pointer'),
                  source_id=imported.get('source_id'), source_sha256=imported.get('sha256'),
                  aliases=deepcopy(imported.get('origins', imported.get('aliases', []))),
                  author_model=first_model(effective, provenance))
    identity = imported.get('record_id') or digest(dict(kind=kind, origin=origin, content=record))
    program = effective.get('source')
    if program is None and isinstance(effective.get('program'), dict):
        program = effective['program'].get('source')
    status = record.get('status', effective.get('status', effective.get('lifecycle', 'HISTORICAL_RECORD')))
    return dict(card_id='relational:'+identity, kind=kind, origin=origin,
                content=record, context=context or {}, evidence_ids=[], unresolved_evidence_ids=[],
                envelope=deepcopy(imported.get('envelope', {})), status=status,
                dataset_role='unclassified', execution_authorized=False, physics_reuse_allowed=False,
                new_model_generation=False, causal_claim_verified=False,
                original_program_sha256=hashlib.sha256(program.encode()).hexdigest() if isinstance(program, str) else None,
                original_status_preserved=True)


def first_model(effective, provenance):
    return effective.get('author_model') or effective.get('author') or provenance.get('returned_model') or provenance.get('model')
