"""Read-only provenance recovery for two historical scored-order registrations.

This does not register programs, rerun physics, or authorize a different verifier
profile. Pins identify the already-reviewed source artifacts; product names alone
never establish training membership. Local observation IDs are scoped to the exact
39-example packet, with reversible CAD/part aliases and hashed raw observations.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from orchestrator.knowledge_import import safe_path
from orchestrator.relational_knowledge import digest
from orchestrator.scored_order_skill import feature_rows, validate_source


PILOT = 'runs/scored_order_skill_pilot_20261111'
ASTRA = 'runs/scored_order_astra_author_20261112'
PACKET_PATH = PILOT + '/training_observations.json'
PROVENANCE_PATH = PILOT + '/training_provenance.json'
PACKET_HASH = 'bc09107dd04e53565f51024c529d00d1ae7348dc1952a76de335d28d25b0a5ee'
PROVENANCE_HASH = 'a4d4bafffdb9194a68740f412021c118812fd3817ca64b38981d8e58439431c8'
REGISTRATIONS = (
    dict(run=PILOT, model='gpt-6-luna',
         registration_sha256='d438ef51827281b05294d087133821c41b2e7e5914dace35fd334bdb2843a728',
         candidate_path=PILOT + '/revision/candidate.json',
         candidate_sha256='57d7ee5d05487ee98131822c5edad3d654a9d432c069a83ce46209a8e527fb03'),
    dict(run=ASTRA, model='gpt-6-astra',
         registration_sha256='87224496b012e0a0443338fcbcbb4632702a382751b39e966996202540671df0',
         candidate_path=ASTRA + '/generation/candidate.json',
         candidate_sha256='47c2c21c5e83514ebb84f31fd45dfeb63f84360055bbf502cc013a7de263e44e'),
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class Reader:
    def __init__(self, root, read_json=None):
        self.root = Path(root)
        self.read_json = read_json
        self.hashes = {}

    def read(self, name, expected=None):
        path = safe_path(self.root, name)
        blob = path.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        require(expected is None or sha == expected, 'source_hash_mismatch:' + name)
        self.hashes[name] = sha
        value = json.loads(blob.decode('utf-8-sig'))
        if self.read_json is not None:
            require(value == self.read_json(name), 'source_changed_while_reading:' + name)
        return value


def _observations(reader):
    packet = reader.read(PACKET_PATH, PACKET_HASH)
    provenance = reader.read(PROVENANCE_PATH, PROVENANCE_HASH)
    require(provenance['observations_sha256'] == PACKET_HASH and
            packet['protocol'] == 'scored_order_observations/v1', 'packet_provenance_mismatch')
    products = provenance['products']
    require(len(set(products.values())) == len(products) and
            set(products.values()) == set(packet['training_products']) and
            provenance['transfer_product_excluded'] not in products, 'training_alias_or_split_mismatch')
    geometries = {product: reader.read(pin['source'], pin['sha256'])
                  for product, pin in provenance['source_geometry'].items()}
    observations = {}
    require(len(packet['examples']) == len(provenance['examples']), 'missing_observation_provenance')
    for index, example in enumerate(packet['examples']):
        eid = example['evidence_id']
        require(eid == example['case'] and eid not in observations, 'duplicate_observation_alias')
        identity = {k: v for k, v in example.items() if k not in ('evidence_id', 'case')}
        require(eid == 'obs_' + digest(identity)[:16], 'observation_identity_mismatch')
        proof = provenance['examples'][eid]
        product = proof['product']
        mapping = provenance['per_product_alias'][product]
        require(len(mapping) == len(set(mapping.values())) and
                products[product] == example['product'], 'product_alias_mismatch')
        parent = proof['root_or_parent']
        mover = proof['first']
        rows = feature_rows(geometries[product], parent)
        require({mapping[p]: value for p, value in rows.items()} == example['candidates'] and
                mapping[mover] == example['first'], 'observation_geometry_or_part_alias_mismatch')
        raw_refs = proof['records']
        require(bool(raw_refs), 'missing_raw_observation')
        for raw in raw_refs:
            reader.read(raw['file'], raw['sha256'])
        geometry_ref = provenance['source_geometry'][product]
        observations[eid] = dict(
            packet_path=PACKET_PATH, packet_sha256=PACKET_HASH,
            json_pointer='/examples/' + str(index), example_sha256=digest(example),
            source_product=product, source_parent=list(parent), source_mover=mover,
            source_records=deepcopy(raw_refs), geometry_source=deepcopy(geometry_ref),
            unaliased_features=rows, alias_mapping=deepcopy(mapping),
            protocol='path_only_sequence_proposal/v1', evidence_scope=example['scope'],
            product_alias=example['product'], no_new_physics=True)
    return packet, observations


def _registration(reader, spec, packet, observations):
    registration_path = spec['run'] + '/registry/skill.json'
    card = reader.read(registration_path, spec['registration_sha256'])
    candidate = reader.read(spec['candidate_path'], spec['candidate_sha256'])
    require(card['protocol'] == 'scored_path_order_python_registration/v1' and
            card['status'] == 'REGISTERED_SCOPED' and card['candidate'] == candidate and
            candidate['skill_id'] == card['skill_id'], 'registration_candidate_mismatch')
    require(candidate['packet_sha256'] == digest(packet) and
            candidate['provenance']['origin'] == 'actual_llm' and
            candidate['provenance']['model'].startswith(spec['model']) and
            hashlib.sha256(candidate['source'].encode('utf-8')).hexdigest() == candidate['source_sha256'],
            'generation_or_source_hash_mismatch')
    validate_source(candidate['source'])
    require(set(candidate['evidence_ids']) <= observations.keys() and
            bool(candidate['evidence_ids']), 'unresolved_generation_evidence')
    # All examples were presented during generation, not only the cited subset.
    training = sorted({item['source_product'] for item in observations.values()})
    require(sorted(card['training_products']) == training and
            set(card['validation_products']) <= set(training), 'registration_training_scope_mismatch')
    require(len(card['validation_summary']) == len(card['validation_products']), 'missing_validation_proof')
    proofs = []
    for row in card['validation_summary']:
        product = row['product']
        folder = spec['run'] + '/validation/' + product
        proof_path, result_path = folder + '/verified.json', folder + '/physics/result.json'
        proof = reader.read(proof_path)
        result = reader.read(result_path, row['result_sha256'])
        intent = reader.read(folder + '/intent.json', row['intent_sha256'])
        reader.read(folder + '/physics/settings.json', result['settings_sha256'])
        require(proof == row and proof['skill_id'] == card['skill_id'] and
                proof['candidate_digest'] == digest(candidate) and
                proof['qualifies'] is True and proof['complete'] is True and
                proof['python_executions'] > 0 and proof['domain'] == 'path_only_sequence_proposal/v1' and
                result['profile'] == proof['domain'] and result['complete'] is True and
                result['input_unchanged'] is True and result['asap_sources_unchanged'] is True and
                result['disassembly_order'] == proof['order'] and result['remaining_parts'] == proof['last'] and
                intent['candidate_digest'] == digest(candidate), 'physics_proof_mismatch')
        proofs.append(dict(product=product, proof_path=proof_path,
                           proof_sha256=reader.hashes[proof_path], result_path=result_path,
                           result_sha256=reader.hashes[result_path], profile=proof['domain'],
                           score=deepcopy(proof['score']), python_executions=proof['python_executions']))
    return dict(status='VERIFIED_HISTORICAL_REGISTRATION', registration_path=registration_path,
                registration_sha256=spec['registration_sha256'], candidate_path=spec['candidate_path'],
                registration_content_sha256=digest(card),
                source_sha256=candidate['source_sha256'], skill_id=card['skill_id'],
                candidate_sha256=spec['candidate_sha256'], candidate_digest=digest(candidate),
                packet_path=PACKET_PATH, packet_sha256=PACKET_HASH,
                packet_content_sha256=digest(packet), cited_evidence_ids=list(candidate['evidence_ids']),
                exposure_evidence_ids=sorted(observations), training_products=training,
                validation_products=list(card['validation_products']), validation_proofs=proofs,
                validation_profile='path_only_sequence_proposal/v1',
                native_sequence_profile_validation='NOT_RUN',
                separate_lesson='NOT_PRESENT_IN_THIS_GENERATION_ARTIFACT',
                performance_effect=card['performance_effect'], scope=card['scope'],
                new_api_calls=0, new_physics_calls=0)


def load_verified_scored_provenance(root, read_json=None):
    """Return verified registrations/observations, or explicit fail-closed issues."""
    reader = Reader(root, read_json)
    output = dict(skills={}, observations={}, source_hashes=reader.hashes, issues=[])
    try:
        packet, observations = _observations(reader)
        output['observations'] = observations
    except (OSError, ValueError, TypeError, KeyError) as error:
        output['issues'].append(dict(source=PACKET_PATH, reason=str(error)))
        return output
    for spec in REGISTRATIONS:
        try:
            proof = _registration(reader, spec, packet, observations)
            output['skills'][proof['skill_id']] = proof
        except (OSError, ValueError, TypeError, KeyError) as error:
            output['issues'].append(dict(source=spec['run'] + '/registry/skill.json', reason=str(error)))
    return output
