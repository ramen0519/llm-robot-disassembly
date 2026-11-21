"""Bound an opt-in replay allowance to a preserved actual source request.

No solver threshold changes. Default standalone case allowance remains 45s.
This is operator-selected metadata, never generated-program input/control.
"""
import hashlib
import json
import math


def finite_allowance(value, maximum=600):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError('invalid_bound_case_allowance')
    return value


def validate_source_context(context, source):
    seconds = finite_allowance(source.get('remaining_wall_seconds'))
    # New action IDs, directions and declared pose transforms are expected.
    # Everything governing geometry, state, seed and physical policy is fixed.
    variable = {'event_id', 'action_id', 'pose_catalog', 'pose_matrix', 'parts_rest',
                'time_remaining_seconds'}
    if ({k:v for k,v in context.items() if k not in variable} !=
            {k:v for k,v in source.items() if k not in variable}):
        raise ValueError('source_case_physical_policy_changed')
    return seconds


def worker_allowance(context, source_path=None, source_sha256=None, *, checked):
    if source_path is None and source_sha256 is None:
        return finite_allowance(context.get('remaining_wall_seconds'), 45)
    if not source_path or not source_sha256:
        raise ValueError('source_request_and_hash_required')
    # checked rejects E/escapes before filesystem IO.
    data = checked(source_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != source_sha256:
        raise ValueError('source_request_hash_changed')
    return validate_source_context(context, json.loads(data)['context'])
