"""Opt-in, bounded execution evidence for a Python policy's abstention.

The original VM and historical preflight hashes are unchanged. This observer
runs in a disposable process and never executes native generated Python.
"""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from worker.explicit_action_contract import digest
from orchestrator.python_skill_vm import VM, _Return

FLAG = 'preflight_abstention_feedback'
INSTRUCTIONS = """
Some previous_preflight entries include a bound abstention_diagnostic from host
execution of the OLD Python on this exact development input. Read its return
line, taken conditions and evaluated operand values. Short-circuited operands
were NOT evaluated. This is a code-level abstention, NOT a failed physical repair.
For example, a computed number of requested supports above the current capacity
only makes that proposal infeasible; it does not prove that every pose, support
subset, mover or path is impossible. Do not increase the real capacity or copy
source IDs to manufacture success. Revise the algorithm for the observed case
using actual geometry and feedback, or retain an honest scope limitation. An
unchanged abstention remains a failed development gate, not a registered skill.
Never turn these diagnostics into physics evidence or an instruction to prune.
"""


def _text(node):
    value = ast.unparse(node)
    return dict(expression=value[:300], expression_truncated=len(value) > 300)


def _value(value):
    """Retain useful scalars without dumping the input geometry/context."""
    if value is None or type(value) in (bool, int, float):
        return dict(value=value, truncated=False)
    if type(value) is str:
        return dict(value=value[:120], truncated=len(value) > 120)
    if type(value) in (list, tuple):
        return dict(value=[v if v is None or type(v) in (bool, int, float)
                           else v[:80] if type(v) is str else {'type': type(v).__name__}
                           for v in value[:8]], length=len(value), truncated=len(value) > 8)
    if type(value) is dict:
        return dict(type='dict', keys=[str(k)[:80] for k in list(value)[:8]],
                    length=len(value), truncated=True, values_omitted=True)
    raise ValueError('abstention_observer_requires_data_values')


class ObservedVM(VM):
    """Observe values already evaluated by VM; never re-evaluate a condition."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guards, self.recent, self.frames, self.returns = [], [], [], []

    def call(self, name, values, keywords=None):
        generated = name in self.functions
        if generated:
            self.frames.append(name)
        try:
            return super().call(name, values, keywords)
        finally:
            if generated:
                self.frames.pop()

    def expr(self, node, env):
        value = super().expr(node, env)
        for frame in self.guards[-4:]:
            if id(node) not in frame['node_ids'] or frame['depth'] != len(self.frames):
                continue
            # Only operand roots need values; other subtrees are not retained.
            if id(node) in frame['operand_ids']:
                frame['values'][id(node)] = _value(value)
            if isinstance(node, ast.Compare) and len(frame['public']['comparisons']) < 4:
                operands = [node.left, *node.comparators]
                frame['public']['comparisons'].append(dict(**_text(node), result=bool(value),
                    operands=[dict(**_text(n), evaluated=id(n) in frame['values'],
                        **deepcopy(frame['values'].get(id(n), {}))) for n in operands]))
            if node is frame['test']:
                frame['public'].update(condition_truth=bool(value), branch='then' if value else 'else')
        return value

    def block(self, nodes, env):
        for node in nodes:
            frame = None
            if isinstance(node, ast.If):
                members = list(ast.walk(node.test))
                frame = dict(test=node.test, node_ids={id(n) for n in members}, values={}, depth=len(self.frames),
                    operand_ids={id(n) for c in members if isinstance(c, ast.Compare)
                                 for n in [c.left, *c.comparators]},
                    public=dict(line=node.lineno, **_text(node.test),
                                condition_truth=None, branch=None, comparisons=[]))
                self.guards.append(frame)
            try:
                # Exactly the original evaluator's statement and tick logic.
                super().block([node], env)
            except _Return as returned:
                if isinstance(node, ast.Return) and self.frames[-1:] == ['propose']:
                    self.returns.append(dict(line=node.lineno, **_text(node),
                        returns_none=returned.value is None,
                        conditions=deepcopy([g['public'] for g in self.guards[-4:]]),
                        conditions_truncated=len(self.guards) > 4,
                        recent_conditions=deepcopy(self.recent[-4:]) if not self.guards else []))
                    self.returns[:] = self.returns[-1:]
                raise
            finally:
                if frame is not None:
                    self.guards.pop()
                    self.recent.append(deepcopy(frame['public']))
                    self.recent[:] = self.recent[-4:]


def evaluate_observed(program, request, extension):
    """Trusted child entry, also testable against the uninstrumented evaluator."""
    from orchestrator import python_skill_program as engine
    observed = []
    class Capture(ObservedVM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append(self)
    original = engine.VM
    try:
        engine.VM = Capture
        result = engine.evaluate(program, request, extension=extension, diagnose=True)
    finally:
        engine.VM = original
    vm = observed[0]
    context, _ = engine.build_context(request, extension)
    return dict(schema_version=1, kind='python_abstention_execution_v1',
        program_sha256=digest(program), request_sha256=digest(request), context_sha256=digest(context),
        abstained=result['choice'] is None and result['diagnostic_runtime_error'] is None,
        runtime_error=result['diagnostic_runtime_error'], input_audit=result['input_audit'],
        vm_steps=result['steps'], eligibility_checks=result['eligibility_checks'],
        return_site=deepcopy(vm.returns[-1]) if vm.returns else None,
        implicit_return=result['choice'] is None and not vm.returns and result['diagnostic_runtime_error'] is None,
        recent_conditions=deepcopy(vm.recent[-4:]) if not vm.returns else [],
        code_observation_only=True, physical_failure_observed=False,
        api_calls=0, physical_calls=0, hard_pruning=False)


def trace_abstention(program, request, *, timeout_seconds=5):
    from orchestrator.python_skill_program import validate_program, MAX_INPUT_BYTES
    from worker.explicit_action_contract import validate_request
    from orchestrator.lesson_runtime import extension_for
    validate_program(program)
    validate_request(request)
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 10:
        raise ValueError('invalid_abstention_trace_timeout')
    packet = dict(program=program, request=request, extension=extension_for())
    payload = json.dumps(packet, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError('abstention_trace_input_limit')
    env = {key: os.environ[key] for key in ('SystemRoot', 'WINDIR') if key in os.environ}
    child = subprocess.run([sys.executable, '-I', '-S', '-B', str(Path(__file__).absolute())],
        input=payload, capture_output=True, timeout=timeout_seconds, env=env,
        cwd=str(Path(__file__).absolute().parent), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if child.returncode or len(child.stdout) > 32768:
        raise ValueError('abstention_trace_child_failed')
    value = json.loads(child.stdout)
    if (value.get('program_sha256') != digest(program) or value.get('request_sha256') != digest(request)
            or value.get('kind') != 'python_abstention_execution_v1'):
        raise ValueError('abstention_trace_binding_mismatch')
    return value


def attach_feedback(packet):
    """Enrich verified old preflight feedback without changing old audit hashes."""
    from orchestrator.python_skill_preflight import preflight_rows
    from orchestrator.python_skill_program import build_context
    from orchestrator.lesson_runtime import extension_for
    program = packet.get('previous_program')
    previous = packet.get('context', {}).get('development_preflight_feedback', {})
    if not program or previous.get('program_sha256') != digest(program):
        raise ValueError('bound_previous_program_and_preflight_required')
    checks = {c['case_id']: c for c in previous.get('checks', [])}
    rows = []
    for row in preflight_rows(packet):
        old = checks.get(row['case_id'])
        request = row['current_observation_request']
        context, _ = build_context(request, extension_for())
        if (not old or old.get('request_sha256') != digest(request)
                or old.get('context_sha256') != digest(context)):
            raise ValueError('abstention_preflight_input_binding_mismatch')
        if old.get('runtime_reason') != 'python_abstained' or old.get('runtime_error') is not None:
            continue
        trace = trace_abstention(program, request)
        if not trace['abstained'] or trace['runtime_error'] is not None:
            raise ValueError('abstention_replay_disagrees_with_preflight')
        rows.append(dict(case_id=row['case_id'], phase='debug', diagnostic=trace,
                         diagnostic_sha256=digest(trace)))
    if not rows:
        raise ValueError('no_bound_preflight_abstention')
    value = deepcopy(packet)
    value['context'][FLAG] = dict(schema_version=1, previous_program_sha256=digest(program),
        source_preflight_sha256=digest(previous), rows=rows, diagnostics_not_physics=True,
        api_calls=0, physical_calls=0)
    return value


def feedback_for_case(packet, row):
    bundle = packet.get('context', {}).get(FLAG)
    if bundle is None:
        return None
    program = packet.get('previous_program')
    old = packet['context'].get('development_preflight_feedback', {})
    if (bundle.get('schema_version') != 1 or not program
            or bundle.get('previous_program_sha256') != digest(program)
            or bundle.get('source_preflight_sha256') != digest(old)
            or bundle.get('diagnostics_not_physics') is not True):
        raise ValueError('abstention_feedback_provenance_mismatch')
    rows = bundle.get('rows')
    from orchestrator.python_skill_preflight import preflight_rows
    allowed = {r['case_id'] for r in preflight_rows(packet)}
    if (type(rows) is not list or not 1 <= len(rows) <= 8
            or any(type(r) is not dict for r in rows)
            or len({r['case_id'] for r in rows}) != len(rows)
            or any(r.get('phase') != 'debug' or r.get('case_id') not in allowed for r in rows)):
        raise ValueError('abstention_debug_only_feedback_required')
    matching = [r for r in rows if r['case_id'] == row['case_id']]
    if not matching:
        return None
    trace = matching[0]['diagnostic']
    if matching[0].get('diagnostic_sha256') != digest(trace):
        raise ValueError('abstention_diagnostic_hash_mismatch')
    from orchestrator.python_skill_program import build_context
    from orchestrator.lesson_runtime import extension_for
    context, _ = build_context(row['current_observation_request'], extension_for())
    if (trace.get('program_sha256') != digest(program)
            or trace.get('request_sha256') != digest(row['current_observation_request'])
            or trace.get('context_sha256') != digest(context)
            or trace.get('kind') != 'python_abstention_execution_v1'
            or trace.get('abstained') is not True or trace.get('runtime_error') is not None
            or trace.get('code_observation_only') is not True
            or trace.get('physical_failure_observed') is not False
            or trace.get('hard_pruning') is not False
            or trace.get('physical_calls') != 0 or trace.get('api_calls') != 0):
        raise ValueError('abstention_feedback_current_input_mismatch')
    return deepcopy(trace)


if __name__ == '__main__':
    from orchestrator.python_skill_program import MAX_INPUT_BYTES
    payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError('abstention_trace_input_limit')
    item = json.loads(payload)
    result = evaluate_observed(item['program'], item['request'], item['extension'])
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(encoded) > 32768:
        raise ValueError('abstention_trace_output_limit')
    sys.stdout.buffer.write(encoded)
