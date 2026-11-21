"""Planning states for a declared bolted joint; no simulation or robot motion."""
from dataclasses import dataclass
from typing import Tuple

from independent_rank import enumerate_candidates


@dataclass(frozen=True)
class Occurrence:
    id: str
    path: str
    label: str


@dataclass(frozen=True)
class BoltedJoint:
    id: str
    step_hash: str
    retained: Occurrence
    removable: Occurrence
    bolts: Tuple[Occurrence, ...]

    @classmethod
    def from_bom(cls, spec, bom):
        digest = spec['source_step_sha256']
        if (spec['schema'] != 'bolted_joint/v1' or not spec['joint_id'] or
                len(digest) != 64 or digest != bom['summary']['source_sha256']):
            raise ValueError('invalid joint or different source STEP')
        placements = [p for p in bom['placements'] if p['leaf']]
        leaves = {p['path']: p['label'] for p in placements}

        def occurrence(row):
            item = Occurrence(**row)
            if not item.id or not item.label or leaves.get(item.path) != item.label:
                raise ValueError('unknown or mismatched CAD occurrence: ' + str(item))
            return item

        joint = cls(spec['joint_id'], digest, occurrence(spec['retained']),
                    occurrence(spec['removable']), tuple(occurrence(b) for b in spec['bolts']))
        members = (joint.retained, joint.removable) + joint.bolts
        if (not joint.bolts or len({p.path for p in members}) != len(members)
                or len({p.id for p in members}) != len(members)):
            raise ValueError('empty bolt list or duplicate joint member')
        group = spec.get('bom_bolt_group')
        if group is not None:
            expected = {p['path'] for p in placements if p['product'] == group['product']
                        and p['path'].rpartition('/')[0] == group['parent_path']}
            if {b.path for b in joint.bolts} != expected:
                raise ValueError('declared bolts do not match the CAD occurrence group')
        return joint

    def initial(self):
        return FastenerState(self, ('engaged',) * len(self.bolts))


NEXT = {'unfasten': ('engaged', 'released_in_place'),
        'recover': ('released_in_place', 'recovered')}


@dataclass(frozen=True)
class FastenerState:
    joint: BoltedJoint
    statuses: Tuple[str, ...]

    def __post_init__(self):
        if (len(self.statuses) != len(self.joint.bolts) or
                any(s not in ('engaged', 'released_in_place', 'recovered') for s in self.statuses)):
            raise ValueError('invalid bolt state')

    def bolt_states(self):
        return {bolt.id: status for bolt, status in zip(self.joint.bolts, self.statuses)}

    def remaining(self):
        return ((self.joint.retained.id, self.joint.removable.id) +
                tuple(b for b, s in self.bolt_states().items() if s != 'recovered'))

    def blockers(self):
        return {bolt: s for bolt, s in self.bolt_states().items() if s != 'recovered'}

    def removal_candidates(self):
        # Reuse the native planner's enumeration. A bolt has its own two actions;
        # a generic translation must never substitute for either of them.
        if self.blockers():
            return ()
        return enumerate_candidates(self.remaining(), self.joint.retained.id)

    def removal_request(self, part):
        """Issue a candidate for downstream path/stability checking. No removal is committed."""
        if part not in self.removal_candidates():
            raise ValueError('part is not eligible for separate removal: ' + part)
        return dict(part_id=part, occurrence_path=self.joint.removable.path,
                    retained_occurrence=self.joint.retained.path, joint_id=self.joint.id,
                    source_step_sha256=self.joint.step_hash, physics_status='NOT_RUN')

    def apply(self, action, bolt_id, outcome):
        states = self.bolt_states()
        if action not in NEXT or bolt_id not in states:
            raise ValueError('unknown bolt or bolt action')
        before, after = NEXT[action]
        if states[bolt_id] != before:
            raise ValueError('invalid bolt transition: {} {} from {}'.format(action, bolt_id, states[bolt_id]))
        if outcome in ('FAILED', 'UNKNOWN'):
            return self
        if outcome != 'CONFIRMED':
            raise ValueError('invalid operation outcome')
        return FastenerState(self.joint, tuple(after if b == bolt_id else s for b, s in states.items()))

    def snapshot(self):
        states = self.bolt_states()
        engaged = [bolt for bolt, status in states.items() if status == 'engaged']
        return dict(bolt_states=states, remaining_joint_parts=self.remaining(),
                    rigid_group=([self.joint.retained.id, self.joint.removable.id] + engaged
                                 if engaged else []),
                    removal_blockers=self.blockers(),
                    bolt_action_candidates=[dict(action=action, part_id=b)
                        for b, s in states.items() for action, (before, _) in NEXT.items() if s == before],
                    removal_candidates=self.removal_candidates())


def plan_joint(joint, events):
    """Replay an ordered planning branch. Request IDs and revisions guard against stale events."""
    if events['joint_id'] != joint.id or events['source_step_sha256'] != joint.step_hash:
        raise ValueError('events refer to a different joint or CAD revision')
    state = joint.initial()
    revision = 0
    trace = [dict(revision=revision, state=state.snapshot())]
    seen = {}
    for event in events['events']:
        rid = event['request_id']
        if not isinstance(rid, str) or not rid:
            raise ValueError('missing request id')
        if rid in seen:
            if seen[rid] != event:
                raise ValueError('conflicting duplicate request: ' + rid)
            continue
        if type(event['expected_revision']) is not int or event['expected_revision'] != revision:
            raise ValueError('stale joint revision: ' + rid)
        new_state = state.apply(event['action'], event['part_id'], event['outcome'])
        revision += int(new_state != state)
        state = new_state
        seen[rid] = event
        trace.append(dict(event=event, revision=revision, state=state.snapshot()))
    return dict(profile='fastener_sequence/v1', joint_id=joint.id, source_step_sha256=joint.step_hash,
                evidence_domain='symbolic_planning_replay', trace=trace,
                candidate_requests=[state.removal_request(p) for p in state.removal_candidates()],
                physics_status='NOT_RUN', robot_execution='NOT_RUN')


def bind_released_scene(joint, bom, placement_map, input_hashes, base_part):
    """Bind a separately exported, post-recovery assembly to the original CAD occurrences."""
    if placement_map['source_step_sha256'] != joint.step_hash:
        raise ValueError('post-recovery OBJ map belongs to another CAD revision')
    recovered = {b.path for b in joint.bolts}
    remaining = {p['path'] for p in bom['placements'] if p['leaf']} - recovered
    mapped = placement_map['placements']
    if set(mapped) != remaining:
        raise ValueError('post-recovery scene must contain every other CAD occurrence exactly once')
    files = {row['obj']: row['sha256'] for row in mapped.values()}
    objects = {name: digest for name, digest in input_hashes.items() if name.endswith('.obj')}
    if len(files) != len(mapped) or files != objects or any('/' in f or '\\' in f for f in files):
        raise ValueError('post-recovery OBJ identity or contents do not match the placement map')
    if base_part is not None and mapped[joint.retained.path]['obj'] != base_part + '.obj':
        raise ValueError('base part differs from retained joint member')
    return mapped[joint.removable.path]['obj'][:-4]
