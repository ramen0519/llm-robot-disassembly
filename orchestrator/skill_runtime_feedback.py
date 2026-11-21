"""Bounded code observations for selection, never new physical evidence."""
from copy import deepcopy
import json

from worker.explicit_action_contract import digest

MAX_FEEDBACK_BYTES = 2048
TOTAL_FEEDBACK_BYTES = 16384
INTERPRETATION = (
    "Observed code conditions/values explain this execution only. Missing values "
    "are unknown; short-circuited operands were not evaluated. Recent conditions "
    "are history, not proven causes. This is not physics failure or impossibility. "
    "Do not force skill selection, change capacity, or prune candidates.")


def size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def trace_from_vm(vm, program, request, context):
    """Observe the SAME invocation; do not replay code/conditions for a reason."""
    site = deepcopy(vm.returns[-1]) if vm.returns else None
    trace = dict(schema_version=1, kind='runtime_skill_abstention_v1',
        program_sha256=digest(program), request_sha256=digest(request),
        context_sha256=digest(context), return_site=site,
        implicit_return=site is None,
        recent_conditions=deepcopy(vm.recent[-4:]) if site is None else [],
        coverage='bounded_executed_conditions_not_all_paths',
        code_observation_only=True, physical_failure_observed=False,
        physical_calls=0, api_calls=0, truncated=False)
    # Keep the nearest enclosing guard first when a large trace needs clipping.
    # Never truncate a scalar/comparison into a different apparent measurement.
    while size(trace) > MAX_FEEDBACK_BYTES:
        trace['truncated'] = True
        groups = ([site['recent_conditions'], site['conditions']] if site else
                  [trace['recent_conditions']])
        group = next((g for g in groups if g), None)
        if group is None:
            break
        if len(group) > 1:
            group.pop(0)
        elif group[0]['comparisons']:
            group[0]['comparisons'].pop(0)
        else:
            group.pop(0)
    return trace


class FeedbackBudget:
    """One aggregate limit for all skill diagnostics in a model decision."""
    def __init__(self, count=1):
        if type(count) is not int or not 0 <= count <= 32:
            raise ValueError('runtime_feedback_skill_count')
        # Reserve a small explicit omission record for every remaining skill.
        self.remaining = TOTAL_FEEDBACK_BYTES - count * 384

    def take(self, step):
        if step is not None and step.get('response') is not None:
            return None
        reason = step.get('not_applicable_reason') if step else 'deadline'
        value = dict(reason=reason, code_observation_only=True,
                     physical_failure_observed=False)
        trace = deepcopy(step.get('runtime_abstention')) if step else None
        if trace is not None:
            value['trace'] = trace
        else:
            value['trace_unavailable'] = ('execution_error_or_limit' if reason in
                ('python_execution_rejected', 'python_wall_timeout', 'python_input_byte_limit',
                 'deadline') else 'not_recorded')
        extra = max(0, size(value) - 384)
        if extra > self.remaining:
            value.pop('trace', None)
            value['trace_unavailable'] = 'decision_diagnostic_byte_limit'
            extra = 0
        self.remaining -= extra
        return value
