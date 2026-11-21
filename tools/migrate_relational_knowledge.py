"""Additive offline migration of all discovered historical knowledge formats."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.knowledge_import import import_sources
from orchestrator.relational_knowledge import (context_from_geometry, digest, freeze_library,
                                               make_card, read_json, safe_path)
from orchestrator.scored_skill_provenance import (PACKET_PATH as SCORED_PACKET_PATH,
                                                  REGISTRATIONS as SCORED_REGISTRATIONS,
                                                  load_verified_scored_provenance)


def at(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first(*values):
    return next((v for v in values if v is not None), None)


def resolve_versions(candidates, source_path):
    """Prefer a reference in the same source bundle; otherwise retain equal versions.

Only pagination bookkeeping may differ. Conflicting facts/status/provenance stay
unresolved instead of selecting an arbitrary historical version.
"""
    same_source=[c for c in candidates if c['origin']['source_path']==source_path or
                 any(o.get('source_path')==source_path for o in c['origin'].get('aliases',[]) if isinstance(o,dict))]
    if len(same_source)==1:
        return same_source
    ignored={'first_page','occurrences','slot','duplicate_of_slot'}
    fingerprints={digest({k:v for k,v in c['content'].items() if k not in ignored}) for c in candidates}
    return candidates if len(fingerprints)==1 else []


class Migrator:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.read_hashes = {}
        self.cache = {}
        self.issues = []
        self.alias_manifest = self.read('runs/three_authors_60_20261114/packet_manifest.json', optional=True) or {}
        self.packet_namespace = 'packet:' + self.alias_manifest.get('packet_sha256', '')
        self.scored_provenance = dict(skills={}, observations={}, source_hashes={}, issues=[])

    def scored_observation(self, imported):
        proof = self.scored_provenance['observations'].get(imported.get('source_id'))
        if (proof and imported['source_path'] == SCORED_PACKET_PATH and
                imported.get('sha256') == proof['packet_sha256'] and
                digest(imported['record']) == proof['example_sha256']):
            return proof
        return None

    def scored_registration(self, imported):
        proof = self.scored_provenance['skills'].get(imported.get('source_id'))
        if (proof and imported['source_path'] == proof['registration_path'] and
                imported.get('sha256') == proof['registration_sha256'] and
                digest(imported['record']) == proof['registration_content_sha256']):
            return proof
        return None

    def read(self, name, optional=False):
        if not isinstance(name, str):
            return None
        if name in self.cache:
            return self.cache[name]
        try:
            path = safe_path(name, self.root)
            value = read_json(path, self.root)
            self.read_hashes[str(path.relative_to(self.root)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.cache[name] = value
            return value
        except (ValueError, OSError, json.JSONDecodeError) as err:
            if not optional:
                self.issues.append(dict(source=name, reason=type(err).__name__ + ':' + str(err)[:160]))
            self.cache[name] = None
            return None

    def experience_context(self, imported, original_by_observation):
        row = imported['record']
        linked_original = None
        if imported.get('namespace') == self.packet_namespace:
            source = self.alias_manifest.get('evidence_sources', {}).get(imported.get('source_id'))
            linked_original = original_by_observation.get(source)
        basis = linked_original['record'] if linked_original else row
        scored = self.scored_observation(imported)
        if scored:
            # Restore the exact source CAD/part identities only in the relational
            # context. The anonymous historical model input remains untouched.
            basis = dict(row, product=scored['source_product'], first=scored['source_mover'],
                         remaining_parts=scored['source_parent'],
                         initial_candidate_features=scored['unaliased_features'],
                         geometry=self.read(scored['geometry_source']['source']),
                         source_files=[scored['geometry_source']['source']],
                         protocol=scored['protocol'])
        scope = basis.get('scope') if isinstance(basis.get('scope'), dict) else {}
        task_context = first(scope.get('task_context'), basis.get('source_task_context'), basis.get('task_context'), basis.get('context'), {})
        task_context = task_context if isinstance(task_context,dict) else {}
        task = first(at(task_context, 'task'), basis.get('task'), {})
        if not isinstance(task, dict):
            task = {}
        target = first(basis.get('first'), basis.get('mover'), basis.get('part_id'),
                       at(basis, 'choice', 'part_id'), at(scope, 'action', 'part_id'),
                       at(basis, 'action', 'part_id'), at(basis,'request','part_move'),
                       at(basis,'request','part_id'), at(basis,'source_state','part_move'),
                       at(basis,'attempt','part_move'), at(basis,'attempt','part_id'), scope.get('part_move'))
        product = first(basis.get('product'), basis.get('assembly_id'), task.get('assembly_id'))
        remaining = first(scope.get('remaining_parts'), basis.get('parent_parts'), basis.get('assembled_parts'),
                          at(basis,'request','parts_assembled'), at(basis,'request','assembled_parts'),
                          at(basis,'source_state','remaining_parts'), at(basis,'attempt','parent'),
                          scope.get('assembled_parts'),at(basis,'attempt','remaining_parts'))
        if remaining is None and isinstance(at(basis,'request','parts_rest'),list) and target is not None:
            remaining=list(dict.fromkeys([str(target)]+list(map(str,basis['request']['parts_rest']))))
        geometry = None
        geom_features = first(task_context.get('geometry_features'), basis.get('geometry_features'))
        if isinstance(geom_features, dict):
            geometry = dict(parts=geom_features, relations=task_context.get('relations'),
                            frame=task_context.get('frame'), length_unit=task_context.get('length_unit'))
        condition = {}
        source_files = basis.get('source_files', [])
        for name in source_files if isinstance(source_files, list) else []:
            if isinstance(name, dict):
                name = name.get('path', name.get('source_path'))
            if not isinstance(name, str):
                continue
            if name.endswith(('/preaction_geometry.json','/snapshot.json')):
                possible = self.read(name)
                if isinstance(possible,dict) and isinstance(possible.get('parts'),dict):
                    geometry = possible
            elif name.endswith('/condition.json') or name.endswith('/settings.json'):
                value = self.read(name)
                if isinstance(value, dict):
                    condition.update(value)
        if geometry is None and isinstance(basis.get('geometry'), dict) and 'parts' in basis['geometry']:
            geometry = basis['geometry']
        features = basis.get('initial_candidate_features')
        if remaining is None and isinstance(features, dict):
            remaining = list(features)
        if remaining is None:
            remaining = first(basis.get('remaining_parts'), task_context.get('initial_parts'))
        raw = first(at(basis, 'result', 'raw'), basis.get('raw_result'), basis.get('native_result'),
                    at(basis,'attempt','raw_sim_info'), {})
        raw = raw if isinstance(raw,dict) else {}
        verifier = first(at(basis, 'result', 'verifier_origin'), raw.get('backend'),
                         at(basis,'attempt','verifier_domain'))
        if source_files and geometry is not None:
            verifier = verifier or 'ASAP_RedMax'
        domain = 'ASAP_RedMax' if verifier in ('actual_ASAP_RedMax', 'ASAP_RedMax') else verifier
        conditions = dict(verifier_domain=domain,
            verifier_profile=first(condition.get('profile'), scope.get('protocol'), basis.get('protocol')),
            pose=first(scope.get('pose_matrix'), at(raw, 'executed', 'pose'), basis.get('pose')),
            support_policy=first(condition.get('support_cap'), condition.get('support_policy'),
                                 at(task, 'solver_settings', 'support_budget')),
            seed=first(condition.get('seed'), at(task, 'solver_settings', 'seed')),
            source_input_sha256=task.get('input_sha256'), solver_settings=task.get('solver_settings'))
        if basis.get('observation_id') in self.alias_manifest.get('evidence_sources', {}).values() and source_files:
            conditions['verifier_profile']='path_only_sequence_proposal/v1'
        base_parent = next((v for v in (scope, task, condition, task_context) if any(k in v for k in ('base_part','fixed_base'))), None)
        if base_parent is not None:
            base = base_parent.get('base_part', base_parent.get('fixed_base'))
            conditions.update(fixed_base=str(base) if base is not None else 'none',
                              fixed_base_present=base is not None)
        conditions['known_fields'] = [k for k,v in conditions.items() if v is not None]
        context = context_from_geometry(geometry or {}, target, remaining, conditions)
        context['product_id'] = product
        feature_row = first(at(features, str(target)) if features else None, basis.get('first_features'),
                            at(basis, 'prechoice_features', str(target)), at(basis,'candidates',str(target)))
        if isinstance(feature_row, dict):
            for key in context['topology']:
                if context['topology'][key] is None:
                    context['topology'][key] = feature_row.get(key)
            context['stored_prechoice_features'] = deepcopy(feature_row)
        context['outcome'] = first(basis.get('outcome'), at(basis, 'result', 'status'), basis.get('status'),
                                   at(basis,'attempt','status'))
        context['trigger'] = first(at(basis,'observed_signature','observed_failure_stage'),
                                   at(basis,'stability_diagnostics','failure_phase'))
        context['observations'] = dict(path_success=at(basis, 'result', 'path_success'),
            stability_success=first(at(basis, 'result', 'stability_success'), basis.get('stability_success')),
            accepted_for_transition=first(basis.get('accepted_for_transition'), at(raw,'accepted_for_transition'),
                                          at(basis,'attempt','accepted_for_transition')),
            deadline_exceeded=first(basis.get('deadline_exceeded'), at(raw,'deadline_exceeded'),
                                     at(basis,'attempt','deadline_exceeded')),
            endpoint_observation=deepcopy(raw.get('endpoint_observation')),
            stability_diagnostics=deepcopy(first(raw.get('stability_diagnostics'),basis.get('stability_diagnostics'))),
            detailed_failure_cause=None, minimum_support_proven=False)
        context['source_geometry_status'] = 'linked' if geometry else 'NOT_AVAILABLE_IN_IMPORTED_RECORD'
        return context, linked_original

    def convert(self, imported):
        records = imported['records']
        scored_paths = {SCORED_PACKET_PATH} | {s['run'] + '/registry/skill.json' for s in SCORED_REGISTRATIONS}
        if any(record['source_path'] in scored_paths for record in records):
            self.scored_provenance = load_verified_scored_provenance(self.root, self.read)
            self.read_hashes.update(self.scored_provenance['source_hashes'])
            self.issues.extend(self.scored_provenance['issues'])
        originals = {r['record'].get('observation_id'):r for r in records
                     if r['kind']=='experience' and 'initial_candidate_features' in r['record']}
        cards = []; record_cards = {}; canonical_ids = defaultdict(list); namespace_ids = defaultdict(list)
        for record in records:
            try:
                context, linked = self.experience_context(record, originals) if record['kind']=='experience' else ({}, None)
            except (TypeError, KeyError, AttributeError, ValueError) as err:
                self.issues.append(dict(source=record['source_path'], record=record.get('source_id'),
                                        reason='context_projection:' + type(err).__name__ + ':' + str(err)[:160]))
                context, linked = {'projection_status':'MISSING_UNSUPPORTED_SOURCE', 'product_id':None}, None
            card = make_card(record, context)
            card['origin']['namespace'] = record.get('namespace')
            card['record_scope'] = 'episode' if 'selected_stages' in record['record'] else 'historical_record'
            scored_observation = self.scored_observation(record)
            scored_registration = self.scored_registration(record)
            if scored_observation:
                card['dataset_role'] = 'training'
                card['provenance_validation'] = dict(scored_observation,
                    status='VERIFIED_PACKET_AND_RAW_SOURCES', evidence_id=record['source_id'])
                card['context']['source_observation_records'] = deepcopy(scored_observation['source_records'])
            if scored_registration:
                card['provenance_validation'] = deepcopy(scored_registration)
                card['context']['source_products'] = list(scored_registration['training_products'])
            if ((record['source_path']=='runs/training15_20261114/training_experiences.json'
                 and record.get('source_id') in self.alias_manifest.get('evidence_sources',{}).values()) or linked):
                card['dataset_role'] = 'training'
            record_fixture = first(record['record'].get('fixture'), record['record'].get('fixture_registry'),
                                   at(record['record'],'candidate','fixture_registry'))
            card['fixture'] = bool(record_fixture or record.get('envelope', {}).get('fixture') or
                                   any(o.get('fixture') for o in record.get('origins',[])))
            card['evidence_group_id'] = first(record['record'].get('source_request_id'),
                record['record'].get('observation_id'), record.get('source_id'), card['card_id'])
            cards.append(card); record_cards[record['record_id']] = card
            sid = record.get('source_id')
            if sid:
                canonical_ids[sid].append(card); namespace_ids[(record.get('namespace'), sid)].append(card)
            if linked:
                card['_original_record_id'] = linked['record_id']
        # Only exact, scoped IDs are joined; short aliases require their original packet mapping.
        for record, card in zip(records, cards):
            original_id = card.pop('_original_record_id', None)
            if original_id:
                anchor = record_cards[original_id]
                card['evidence_ids'].append(anchor['card_id'])
                card['evidence_group_id'] = anchor['evidence_group_id']
            raw_refs = set()
            def refs(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in ('evidence_ids','lesson_ids','source_evidence_ids') and isinstance(item, list):
                            raw_refs.update(x for x in item if isinstance(x,str))
                        elif key == 'evidence_id' and isinstance(item,str):
                            raw_refs.add(item)
                        elif key not in ('geometry_features','initial_candidate_features','prechoice_features'):
                            refs(item)
                elif isinstance(value,list):
                    for item in value:
                        refs(item)
            refs(record['record'])
            raw_refs.discard(record.get('source_id'))
            for source_id in sorted(raw_refs):
                candidates = namespace_ids.get((record.get('namespace'), source_id), [])
                scoped_registration = self.scored_registration(record)
                if not candidates and scoped_registration and source_id in scoped_registration['cited_evidence_ids']:
                    candidates = [c for c in namespace_ids.get(('file:' + SCORED_PACKET_PATH, source_id), [])
                                  if c.get('provenance_validation', {}).get('status') == 'VERIFIED_PACKET_AND_RAW_SOURCES']
                if not candidates and record.get('namespace') == self.packet_namespace:
                    original_name = self.alias_manifest.get('evidence_sources', {}).get(source_id)
                    original_record = originals.get(original_name)
                    if original_record:
                        candidates = [record_cards[original_record['record_id']]]
                if not candidates and ':' in source_id:
                    candidates = canonical_ids.get(source_id, [])
                resolved=resolve_versions(candidates,record['source_path']) if len(candidates)>1 else candidates
                if resolved:
                    card['evidence_ids'].extend(c['card_id'] for c in resolved if c['card_id']!=card['card_id'])
                else:
                    card['unresolved_evidence_ids'].append(dict(source_id=source_id,
                        reason='ambiguous_id' if len(candidates)>1 else 'not_in_imported_sources'))
            card['evidence_ids'] = sorted(set(card['evidence_ids']))
        # Existing final lessons associate programs by exact hashes, not invented prose.
        program_lessons = defaultdict(list)
        for card in cards:
            if card['kind']=='lesson':
                for program_hash in card['content'].get('program_hashes',[]):
                    if isinstance(program_hash,str):
                        program_lessons[program_hash].append(card['card_id'])
        for card in cards:
            if card['kind']=='skill':
                effective=card['content'].get('candidate',card['content'])
                hashes=[effective.get('program_hash'),effective.get('program_sha256'),effective.get('source_sha256')]
                for value in hashes:
                    card['evidence_ids'].extend(program_lessons.get(value,[]))
                card['evidence_ids']=sorted(set(card['evidence_ids']))
        # Preserve per-stage facts separately from whole-episode costs. No new verdicts.
        stage_cards = []
        for record in records:
            if record['kind'] != 'experience' or 'selected_stages' not in record['record']:
                continue
            parent_card = record_cards[record['record_id']]
            for stage in record['record']['selected_stages']:
                if not isinstance(stage,dict):
                    continue
                stage_record = deepcopy(record)
                stage_record['record'] = dict(stage, product=record['record'].get('product'),
                    source_files=record['record'].get('source_files', []),
                    first=stage.get('mover'), initial_candidate_features=stage.get('prechoice_features',
                      record['record'].get('initial_candidate_features') if stage.get('stage')==1 else None))
                stage_record['source_id'] = str(record.get('source_id')) + '/stage/' + str(stage.get('stage'))
                stage_record['record_id'] = digest(dict(parent=parent_card['card_id'], stage=stage))
                stage_record['json_pointer'] = str(record.get('json_pointer')) + '/selected_stages/' + str(stage.get('stage',1)-1)
                context, _ = self.experience_context(stage_record, originals)
                child = make_card(stage_record, context)
                child.update(record_scope='selected_stage', dataset_role=parent_card['dataset_role'], fixture=parent_card['fixture'],
                    evidence_ids=[parent_card['card_id']], evidence_group_id=parent_card['evidence_group_id'],
                    derived_record=True, source_runtime_request_id=None, episode_score_not_assigned_to_stage=True)
                stage_cards.append(child)
        cards.extend(stage_cards)
        by_id={card['card_id']:card for card in cards}
        for _ in range(4):
            for card in cards:
                if (card['kind'] in ('lesson','skill') and card['evidence_ids'] and not card['unresolved_evidence_ids']
                    and all(by_id[key]['dataset_role']=='training' for key in card['evidence_ids'])):
                    card['dataset_role']='training'
        return cards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = safe_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    imported = import_sources(ROOT)
    migration = Migrator()
    cards = migration.convert(imported)
    # Before authorizing any runtime retrieval the caller pins this exact file/hash.
    library = freeze_library(cards, purpose='historical_archive_development_only',
                             evaluation_requires_explicit_source_split=True,
                             derived_sources_sha256=migration.read_hashes,
                             import_summary=imported['summary'], new_api_calls=0, new_physics_calls=0)
    eligible={c['card_id'] for c in cards if c['dataset_role']=='training' and c['fixture'] is False}
    while True:
        reduced={c['card_id'] for c in cards if c['card_id'] in eligible and set(c['evidence_ids'])<=eligible}
        if reduced==eligible: break
        eligible=reduced
    training_library=freeze_library([c for c in cards if c['card_id'] in eligible],
        purpose='training_runtime', source_archive_sha256=library['content_sha256'],
        training_packet_sha256=migration.alias_manifest.get('packet_sha256'),
        excludes_unclassified_and_fixtures=True, evaluation_requires_excluded_target_products=True)
    summary = dict(imported['summary'], total_cards=len(cards),
        cards_by_kind=dict(Counter(c['kind'] for c in cards)),
        linked_cards=sum(bool(c['evidence_ids']) for c in cards),
        experiences_with_shape=sum(bool(c.get('context',{}).get('shape',{}).get('extents_xyz')) for c in cards if c['kind']=='experience'),
        experiences_with_contacts=sum(c.get('context',{}).get('evidence_status',{}).get('contacts')=='observed_reference' for c in cards if c['kind']=='experience'),
        unresolved_link_count=sum(len(c['unresolved_evidence_ids']) for c in cards),
        sources_checked=len(migration.read_hashes), issues=migration.issues,
        runtime_model_validation='NOT_RUN', physical_effect='UNVERIFIED',
        training_runtime_cards=len(training_library['cards']),
        verified_scored_registrations=len(migration.scored_provenance['skills']),
        verified_scored_observations=len(migration.scored_provenance['observations']),
        fixture_cards=sum(c['fixture'] for c in cards))
    protected = {item['path']:item['sha256'] for item in imported['inventory'] if item.get('sha256')}
    protected.update(migration.read_hashes)
    changed = [name for name,sha in protected.items() if hashlib.sha256(safe_path(name).read_bytes()).hexdigest()!=sha]
    if changed:
        raise RuntimeError('source_changed_during_migration:' + str(changed[:5]))
    for name,value in [('library.json',library), ('training_runtime_library.json',training_library), ('inventory.json',imported['inventory']),
                       ('summary.json',summary), ('preservation.json',dict(unchanged=len(protected), sources_sha256=protected))]:
        with (output/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
