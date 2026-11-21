"""Registered read-only procedures and append-only development experience.

Only reviewed local callables execute. Learned cards parameterize that code;
they are bounded, provisional observations, not generated programs or proofs.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .episode_archive import guarded, EvidenceStore, export_snapshot, _collect_run
from evidence_memory import FrozenMemory, canonical, digest


def module_hash() -> str:
    return hashlib.sha256(guarded(Path(__file__).absolute()).read_bytes()).hexdigest()


def native_order(request: dict, parameters: dict) -> list[str]:
    return list(request['candidate_ids'])


def promote_verified(request: dict, parameters: dict) -> list[str]:
    offered = request['candidate_ids']
    preferred = parameters['preferred_ids']
    return [p for p in preferred if p in offered] + [p for p in offered if p not in preferred]


CALLABLES = {'native_order': native_order, 'promote_verified': promote_verified}


def base_card() -> dict:
    return dict(skill_id='native_order_v1', version=1, callable_id='native_order',
        description='Preserve the current ASAP heur-out candidate order.',
        code_sha256=module_hash(), backend='ASAP_3D', lifecycle='validated_for_contract',
        validation_scope='Candidate permutation only; no performance claim.',
        preconditions=None, parameters={}, source_evidence_ids=[], source_invocation_id=None,
        input_contract='Read-only current request', output_contract='Complete candidate permutation',
        state_mutations=False, internal_verifier_calls=0,
        origin='independently_written_reviewed_template')


def applicable(card: dict, context: dict, request: dict, evidence: dict) -> bool:
    if card.get('code_sha256') != module_hash() or card.get('callable_id') not in CALLABLES:
        return False
    if card['skill_id'] == 'native_order_v1':
        return card == base_card()
    scope = card.get('preconditions')
    if (card.get('lifecycle') != 'provisional' or type(scope) is not dict
            or scope.get('context_sha256') != digest(context)
            or scope.get('assembled_parts') != sorted(request['assembled_parts'])
            or scope.get('base_part') != request['base_part']):
        return False
    positives = card.get('parameters', {}).get('preferred_ids')
    return (type(positives) is list and bool(positives) and len(set(positives)) == len(positives)
            and set(positives).issubset(request['candidate_ids'])
            and set(positives).issubset(evidence['positive_candidate_ids'])
            and set(card['source_evidence_ids']).issubset(evidence['evidence_ids']))


def execute_skill(card: dict, context: dict, request: dict, evidence: dict) -> list[str]:
    if not applicable(card, context, request, evidence):
        raise ValueError('skill_not_applicable')
    before = canonical(request)
    order = CALLABLES[card['callable_id']](deepcopy(request), deepcopy(card['parameters']))
    if (canonical(request) != before or type(order) is not list
            or len(order) != len(request['candidate_ids'])
            or len(set(order)) != len(order) or set(order) != set(request['candidate_ids'])):
        raise ValueError('skill_contract_violation')
    return order


class SkillMemory:
    """One development store. Evaluation opens frozen files, not this store as hints."""
    def __init__(self, path: Path):
        self.path = guarded(path)
        for suffix in ('-wal', '-shm', '-journal'):
            guarded(Path(str(self.path) + suffix))
        self.evidence = EvidenceStore(self.path)
        self.db = self.evidence.connection
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS skill_entries (
                kind TEXT NOT NULL, entry_id TEXT NOT NULL, payload TEXT NOT NULL,
                sha256 TEXT NOT NULL, PRIMARY KEY(kind, entry_id));
            CREATE TRIGGER IF NOT EXISTS skill_entries_no_update BEFORE UPDATE ON skill_entries
                BEGIN SELECT RAISE(ABORT, 'skill entries are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS skill_entries_no_delete BEFORE DELETE ON skill_entries
                BEGIN SELECT RAISE(ABORT, 'skill entries are append-only'); END;
        ''')

    def get(self, kind: str, entry_id: str) -> dict | None:
        row = self.db.execute('SELECT payload,sha256 FROM skill_entries WHERE kind=? AND entry_id=?',
                              (kind, entry_id)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if digest(value) != row[1]:
            raise ValueError('skill_entry_hash_mismatch')
        return value

    def put(self, kind: str, entry_id: str, value: dict) -> bool:
        previous = self.get(kind, entry_id)
        if previous is not None:
            if canonical(previous) != canonical(value):
                raise ValueError('skill_entry_id_conflict')
            return False
        with self.db:
            self.db.execute('INSERT INTO skill_entries VALUES(?,?,?,?)',
                            (kind, entry_id, canonical(value), digest(value)))
        return True

    def entries(self, kind: str) -> list[dict]:
        ids = [row[0] for row in self.db.execute(
            'SELECT entry_id FROM skill_entries WHERE kind=? ORDER BY entry_id', (kind,))]
        return [self.get(kind, entry_id) for entry_id in ids]

    def freeze(self, directory: Path) -> dict:
        directory = guarded(directory)
        directory.mkdir(exist_ok=False)
        evidence = export_snapshot(self.evidence, guarded(directory / 'memory.json'))
        cards = [base_card()] + self.entries('registered_skill')
        library = dict(schema_version=1, cards=cards, content_sha256=digest(cards),
                       source_code_sha256=module_hash(), evaluation_updates_allowed=False)
        with guarded(directory / 'library.json').open('x', encoding='utf-8') as stream:
            stream.write(canonical(library))
        return dict(memory_count=evidence['record_count'], skill_count=len(cards),
                    memory_sha256=evidence['content_sha256'], library_sha256=library['content_sha256'])

    def collect_completed(self, run: Path, context: dict) -> dict:
        """Bind actual worker results to skill invocations after process completion."""
        run = guarded(run)
        records, source = _collect_run(run, context, set(), set())
        previous = self.get('episode_collection', source['run_id'])
        if previous is not None:
            if previous['source'] != source:
                raise ValueError('episode_source_changed')
            return {**previous, 'inserted': {'inserted': 0, 'duplicates': len(records)}, 'reused': True}
        inserted = self.evidence.insert_many(records)
        events = [json.loads(line) for line in guarded(run / 'events.jsonl').read_text().splitlines()]
        decisions = {e['generation_id']: e for e in events if e['kind'] == 'rank_decision'}
        starts = {e['call_id']: e for e in events if e['kind'] == 'verifier_start'}
        grouped = {}
        unassociated = []
        for record in records:
            start = starts[record['source']['call_id']]
            decision = decisions.get(start.get('generation_id'))
            if decision is None:
                unassociated.append(record['record_id'])
                continue
            rid = decision['request']['request_id']
            invocation = self.get('invocation', rid)
            if invocation is None:
                raise ValueError('missing_skill_invocation')
            if (invocation['request'] != decision['request']
                    or invocation['applied_order'] != decision['applied_order']):
                raise ValueError('worker_did_not_apply_skill_proposal')
            grouped.setdefault(rid, []).append(record)
            self.put('outcome', record['record_id'], dict(record_id=record['record_id'],
                invocation_id=rid, skill_id=invocation['skill_id'], status=record['status'],
                raw_source=record['source'], interpretation='bounded observation, not causal skill success'))
        proposals = []
        for rid, observations in grouped.items():
            invocation = self.get('invocation', rid)
            positive = [r for r in observations if r['status'] == 'PASS']
            if not positive:
                continue
            preferred = [p for p in invocation['applied_order']
                         if any(r['request']['part_move'] == p for r in positive)]
            scope = dict(context_sha256=digest(context), assembled_parts=sorted(invocation['request']['assembled_parts']),
                         base_part=invocation['request']['base_part'])
            sid = 'observed_positive_' + digest(dict(scope=scope, preferred=preferred))[:24]
            if self.get('registered_skill', sid) is not None:
                continue
            card = dict(skill_id=sid, version=1, callable_id='promote_verified',
                description='Prioritize previously PASS candidates under this exact frozen context/state; reverify every action.',
                code_sha256=module_hash(), backend='ASAP_3D', lifecycle='proposed',
                validation_scope='Known beam/context revisit only; no generalization or causal improvement claim.',
                preconditions=scope, parameters={'preferred_ids': preferred},
                source_evidence_ids=sorted(r['record_id'] for r in positive), source_invocation_id=rid,
                input_contract='Read-only current request and exact evidence',
                output_contract='Complete candidate permutation', state_mutations=False,
                internal_verifier_calls=0, origin='observed_parameters_of_reviewed_local_template')
            self.put('proposal', sid, card)
            proposals.append(sid)
        report = dict(source=source, inserted=inserted, proposals=proposals,
                      associated_outcomes=sum(map(len, grouped.values())), unassociated_evidence_ids=unassociated)
        self.put('episode_collection', source['run_id'], {**report, 'inserted': {'total': len(records)}})
        return report

    def review_and_register(self, skill_id: str) -> dict:
        """Offline template/provenance checks; never label a card generally validated."""
        old = self.get('registered_skill', skill_id)
        if old is not None:
            return old
        card = self.get('proposal', skill_id)
        if card is None or card['callable_id'] != 'promote_verified' or card['code_sha256'] != module_hash():
            raise ValueError('unreviewed_skill_template')
        records = {r['record_id']: r for r in self.evidence.records()}
        linked = [records.get(rid) for rid in card['source_evidence_ids']]
        invocation = self.get('invocation', card['source_invocation_id'])
        if not linked or invocation is None or any(r is None or r['status'] != 'PASS' for r in linked):
            raise ValueError('skill_requires_verified_source')
        scope = card['preconditions']
        for record in linked:
            request = record['request']
            if (digest(record['context']) != scope['context_sha256']
                    or sorted(request['parts_rest'] + [request['part_move']]) != scope['assembled_parts']
                    or request['pose'] is not None or request['max_grippers'] != 2
                    or record['source']['run_id'] != invocation['request']['run_id']
                    or self.get('outcome', record['record_id'])['invocation_id'] != card['source_invocation_id']):
                raise ValueError('skill_provenance_mismatch')
        if set(card['parameters']['preferred_ids']) != {r['request']['part_move'] for r in linked}:
            raise ValueError('unsupported_skill_parameters')
        registered = {**card, 'lifecycle': 'provisional'}
        evidence = dict(positive_candidate_ids=card['parameters']['preferred_ids'], evidence_ids=card['source_evidence_ids'])
        execute_skill(registered, linked[0]['context'], invocation['request'], evidence)
        self.put('review', skill_id, dict(skill_id=skill_id, proposal_sha256=digest(card),
            review_method='offline_allowlisted_template_contract_and_source_verification',
            human_review_claimed=False, generalization_validated=False))
        self.put('registered_skill', skill_id, registered)
        return registered

    def close(self) -> None:
        self.evidence.close()


class FrozenSkills:
    def __init__(self, directory: Path):
        directory = guarded(directory)
        self.memory = FrozenMemory(guarded(directory / 'memory.json'))
        value = json.loads(guarded(directory / 'library.json').read_text(encoding='utf-8'))
        cards = value['cards']
        if (value['content_sha256'] != digest(cards) or value['source_code_sha256'] != module_hash()
                or len({c['skill_id'] for c in cards}) != len(cards) or cards[0] != base_card()):
            raise ValueError('invalid_frozen_library')
        self._cards = canonical(cards)
        self.library_sha256 = value['content_sha256']

    def retrieve(self, context: dict, request: dict) -> tuple[list[dict], dict]:
        evidence = self.memory.lookup(context, tuple(request['assembled_parts']), request['base_part'])
        cards = [c for c in json.loads(self._cards) if applicable(c, context, request, evidence)]
        return cards, evidence
