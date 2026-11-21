"""Post-hoc checks of a proposed remainder in its recorded ASAP pose (Python 3.8).

The caller supplies the native check_stable adapter. Its two return values do
not identify the physics branch, or show whether an individual part fell.
"""
import math
import time

try:
    from .explicit_action_physics import _ids, _pose
except ImportError:  # Worker CLI adds worker/ to sys.path.
    from explicit_action_physics import _ids, _pose


def _trial_outcome(success, parts_fall, free, singleton_no_support):
    if type(success) is not bool:
        return 'ERROR', 'invalid_native_success'
    if success:
        if parts_fall is not None and parts_fall != []:
            return 'ERROR', 'unexpected_parts_on_native_success'
        return 'PASS', 'native_success'
    if parts_fall is None:
        return 'UNKNOWN', 'native_timeout'
    if (type(parts_fall) is not list or not parts_fall or
            any(type(part) is not str or part not in free for part in parts_fall) or
            len(set(parts_fall)) != len(parts_fall)):
        return 'ERROR', 'invalid_native_parts_fall'
    if singleton_no_support:
        return 'REJECTED', 'native_singleton_parts_reported_initial_connectivity_unconfirmed'
    return 'REJECTED', 'native_parts_reported_failure_phase_unobserved'


def assess_step(step, *, asset_folder, assembly_dir, support_cap, check_stable,
                wall_time_s, clock=time.monotonic):
    """Try zero supports, then only the native serial greedy extensions.

    The step has a part_move, child part IDs and an explicit native JSON pose
    (4x4 or null). Fixed IDs refer to the child and never include the mover.
    The adapter accepts the native argument names and converts the JSON pose
    to a matrix for the physics backend. No base or implicit fixture is fixed.
    """
    started = clock()
    result = dict(
        no_support_status='UNKNOWN', no_support_reason='not_attempted',
        up_to_cap_status='UNKNOWN', reason=None, held_parts=None,
        all_remaining_fixed=None, trials=[], elapsed_s=0.0,
        budget=dict(support_cap=support_cap, wall_time_s=wall_time_s),
        observation_coverage=dict(
            scope='selected_remainder_at_recorded_pose', fixed_base=None,
            support_search='native_serial_greedy',
            source='check_stable_return_only', ground_contact_observed=False,
            failure_phase_observed=False))

    try:
        if type(support_cap) is not int or support_cap < 0:
            raise ValueError('invalid_support_cap')
        if (type(wall_time_s) not in (float, int) or not math.isfinite(wall_time_s)
                or wall_time_s < 0):
            raise ValueError('invalid_wall_time_s')
        child = _ids(step['child'], 'child_parts', empty=False)
        mover = _ids([step['part_move']], 'part_move')[0]
        if mover in child:
            raise ValueError('part_move_in_child')
        pose = _pose(step['pose'])
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        result.update(no_support_status='ERROR', no_support_reason='invalid_input',
                      up_to_cap_status='ERROR', reason='invalid_input',
                      error=str(error), elapsed_s=clock() - started)
        return result

    fixed = []
    deadline = started + wall_time_s
    while True:
        free = [part for part in child if part not in fixed]
        before = clock()
        remaining = deadline - before
        if remaining <= 0:
            if not result['trials']:
                result['no_support_reason'] = 'wall_time_exhausted'
            result.update(up_to_cap_status='UNKNOWN', reason='wall_time_exhausted',
                          elapsed_s=before - started)
            break

        trial = dict(parts_fix=list(fixed), parts_move=list(free),
                     all_remaining_fixed=not free, timeout_s=remaining,
                     success=None, parts_fall=None)
        result['trials'].append(trial)
        try:
            success, parts_fall = check_stable(
                asset_folder, assembly_dir, list(fixed), list(free),
                pose=pose, timeout=remaining)
        except Exception as error:
            after = clock()
            status, reason = 'ERROR', 'native_check_error'
            trial['error'] = dict(type=type(error).__name__, message=str(error))
        else:
            after = clock()
            trial.update(success=success, parts_fall=parts_fall)
            status, reason = _trial_outcome(success, parts_fall, free,
                                           len(child) == 1 and not fixed)
            if after >= deadline and status != 'ERROR':
                status, reason = 'UNKNOWN', 'wall_time_exhausted'
        trial.update(status=status, reason=reason, elapsed_s=after - before,
                     deadline_exhausted=after >= deadline)
        if not fixed:
            result.update(no_support_status=status, no_support_reason=reason)
        result.update(up_to_cap_status=status, reason=reason,
                      elapsed_s=after - started)
        if status == 'PASS':
            result.update(held_parts=list(fixed), all_remaining_fixed=not free)
            break
        if status != 'REJECTED':
            break
        proposed = fixed + parts_fall
        if len(proposed) > support_cap:
            result['reason'] = 'greedy_support_cap_exceeded'
            break
        fixed = proposed
    return result
