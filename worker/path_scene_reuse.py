"""One-episode, one-entry RedMax path scene cache (Python 3.8).

Only construction is reused. There are no cached outcomes or trajectories.
The original check_success resets before applying each force; its reset is
observed before any step. Stability planners are deliberately outside this cache.
"""
from copy import deepcopy
import hashlib
import json
import math
import time

POLICY = 'path_scene_reuse_v1'
SUPPORTED_BACKEND = {
    'physics_source_sha256': 'a703d55d010786f0d51272ff3737f118000be492d0e35a61eb8765ebde987ea1',
    'sim_string_sha256': '718d4f5f46323613a6a85c6d29131f6237ef277817d7e8280980f3f763c8a41a',
    'redmax_binary_sha256': '639fffe9d986866a8de5ee6d78298a28bb8d49a5239f83891785708142afb091',
}
# This exact binary's reset clears q/qdot, external forces, contact and integrator
# histories. A new native build must be reviewed and physically revalidated.


def _plain(value):
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _encoded(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _signature(planner):
    values = {name: getattr(planner, name) for name in (
        'parts_fix', 'part_move', 'parts_removed', 'pose', 'force_mag',
        'frame_skip', 'min_sep', 'contact_eps', 'col_th_base', '_evidence_col_th')}
    values['timestep'] = planner.sim.options.h
    return _encoded(values)


def _state(planner):
    state = planner.get_state()
    value = _plain({'q': state.q, 'qdot': state.qdot})
    for name in ('q', 'qdot'):
        if (not isinstance(value[name], list) or len(value[name]) not in (3, 6)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in value[name])):
            raise ValueError('scene_initial_state_invalid')
    return value


def validate_backend(hashes):
    if any(hashes.get(name) != digest for name, digest in SUPPORTED_BACKEND.items()):
        raise ValueError('scene_reuse_native_backend_not_validated')


class PathSceneLease:
    def __init__(self, owner, entry, audit):
        self.owner, self.entry, self.audit = owner, entry, audit
        self.planner = entry['planner']
        self.finished = False

    def after_reset(self, planner):
        """Called inside the controller's native reset, before applying force."""
        if (self.finished or self.owner._closed or not self.owner._busy
                or planner is not self.planner or self.audit['reset_state_verified']):
            raise ValueError('invalid_scene_reset_observation')
        # _ScheduledSimulation is a transparent proxy around the stored native
        # scene. Compare actual state, not the end position of the preceding run.
        current = _state(planner)
        if current != self.entry['initial_state']:
            raise ValueError('scene_reset_did_not_restore_initial_state')
        if _signature(planner) != self.entry['signature']:
            raise ValueError('scene_physics_configuration_changed')
        self.audit['reset_state_verified'] = True
        self.owner._counts['reset_checks'] += 1

    def finish(self, reusable):
        if self.finished:
            return
        self.finished = True
        try:
            if (self.planner.sim is not self.entry['simulation']
                    or _signature(self.planner) != self.entry['signature']):
                self.audit['discard_reason'] = 'scene_identity_or_configuration_changed'
                raise ValueError('scene_identity_or_configuration_changed')
            if not reusable or not self.audit['reset_state_verified']:
                self.owner.invalidate(self.audit.get('discard_reason', 'error_deadline_or_unexecuted'))
                self.audit['retained_for_next_attempt'] = False
            else:
                self.audit['retained_for_next_attempt'] = self.owner._entry is self.entry
        except Exception:
            self.owner.invalidate('scene_release_validation_error')
            self.audit['retained_for_next_attempt'] = False
            raise
        finally:
            self.owner._busy = False


class PathSceneSession:
    """Own a single path scene; never share across episodes or threads."""
    def __init__(self, max_entries=1):
        if type(max_entries) is not int or max_entries != 1:
            raise ValueError('scene_session_requires_one_entry')
        self._entry = None
        self._busy = False
        self._closed = False
        self._counts = dict(acquires=0, builds=0, hits=0, reset_checks=0,
                            evictions=0, invalidations=0, stochastic_builds=0)
        self._last_invalidation = None

    def acquire(self, key, factory, rng_digest):
        if self._closed or self._busy:
            raise ValueError('scene_session_closed_or_busy')
        started = time.perf_counter()
        self._busy = True
        self._counts['acquires'] += 1
        try:
            validate_backend(key['source_hashes'])
            fingerprint = hashlib.sha256(_encoded(key).encode()).hexdigest()
            hit = self._entry is not None and self._entry['key'] == fingerprint
            if hit:
                entry = self._entry
                if (entry['planner'].sim is not entry['simulation']
                        or _signature(entry['planner']) != entry['signature']):
                    raise ValueError('cached_scene_mutated')
                self._counts['hits'] += 1
                constructor_seconds = 0.0
            else:
                if self._entry is not None:
                    self._counts['evictions'] += 1
                self._entry = None
                before_rng = rng_digest()
                construction_started = time.perf_counter()
                planner, cache_io = factory()
                constructor_seconds = time.perf_counter()-construction_started
                constructor_rng_unchanged = before_rng == rng_digest()
                self._counts['builds'] += 1
                native_max_time = float(planner.max_time)
                if not math.isfinite(native_max_time) or native_max_time <= 0:
                    raise ValueError('invalid_native_path_time_limit')
                entry = dict(key=fingerprint, planner=planner, simulation=planner.sim,
                    initial_state=_state(planner), signature=_signature(planner),
                    native_max_time=native_max_time, cache_io=deepcopy(cache_io),
                    constructor_seconds=constructor_seconds,
                    constructor_rng_unchanged=constructor_rng_unchanged)
                if constructor_rng_unchanged:
                    self._entry = entry
                else:
                    # Execute this fresh scene normally; never roll the global
                    # RNG back or skip a future stochastic constructor.
                    self._counts['stochastic_builds'] += 1
            entry['planner'].max_time = entry['native_max_time']
            audit = dict(policy=POLICY, cache_hit=hit, key_sha256=fingerprint,
                constructor_calls=int(not hit), constructor_seconds=constructor_seconds,
                acquire_seconds=time.perf_counter()-started,
                original_constructor_seconds=entry['constructor_seconds'],
                constructor_rng_unchanged=entry['constructor_rng_unchanged'],
                global_rng_rewound=False, native_path_timeout_seconds=entry['native_max_time'],
                reset_state_verified=False, outcome_reused=False,
                stability_scene_reused=False, initial_state=deepcopy(entry['initial_state']),
                construction_cache_io=deepcopy(entry['cache_io']),
                construction_evidence_scope='original_construction_not_current_disk_load',
                retained_for_next_attempt=False)
            return PathSceneLease(self, entry, audit)
        except Exception:
            self.invalidate('acquire_error')
            self._busy = False
            raise

    def invalidate(self, reason):
        self._entry = None
        self._counts['invalidations'] += 1
        self._last_invalidation = str(reason)

    def close(self):
        self._entry = None
        self._busy = False
        self._closed = True

    def summary(self):
        return dict(policy=POLICY, max_entries=1, closed=self._closed,
                    cached_entries=int(self._entry is not None), busy=self._busy,
                    last_invalidation=self._last_invalidation, **self._counts)
