"""Opt-in v4: measured per-pair semantics, then executable use, then physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from orchestrator.python_skill_program import MAX_INPUT_BYTES, build_context
from orchestrator.lesson_runtime import extension_for
from orchestrator.python_separation_probe import unmet

INSTRUCTIONS = """
Separation-use contract v4 (a measured-endpoint repair hypothesis, not all skills).
Also implement pure unseparated_parts(ctx, part_id, pose_id). Return sorted unique
rest_part_id values whose min_distance < measured_endpoint.min_sep. Equality is
satisfied. Check each pair separately; neither a sum/average nor distance < 0
implements ASAP's condition. Return [] if every measured pair is satisfied.
Return None if last is absent, last.choice mover/pose differs, endpoint is missing
or scope is not selected_trajectory_terminal_state_only, pairs are incomplete or
duplicated, a moving ID mismatches, or separation_unsatisfied disagrees with the
distance predicate. Complete pairs cover remaining_parts minus the mover. Do not
call eligible/select_directions from this helper, recurse, or change ctx.
The host already binds ctx.last to the CURRENT task/state/configuration. The pure
select_directions helper must call unseparated_parts with that same full ctx and
selected mover/pose, then actually use its returned IDs with their geometry to
choose signed world directions. None is unknown, not an empty successful set.
Tests check numeric thresholds and pair identity independently of last force;
changing which pair is unsatisfied while holding the distance multiset and force
fixed must demonstrate concrete directional use. A dead helper or merely reading
flags is insufficient. Symmetric/insufficient inputs may fail to demonstrate use;
this is a scoped development contract, not proof that every other skill is bad.
Geometry bounds/centers are INITIAL assembled CAD coordinates, NOT the measured
terminal positions. Transform relative CAD vectors by the selected pose rotation
before comparing to world direction vectors; signed projections retain orientation.
Measured SDF distances include collision thresholds; they do NOT provide contact
normals, minimum escape vectors, or universal blockers. Center-based ranking is a
hypothesis only. No winning direction is provided. Keep actual failure UNKNOWN.
The ORIGINAL trigger calls the skill; previous_repair is a different observed
failed action used to revise the algorithm, not an extra runtime field. Use both
without hard-coded source IDs or substituting admission cases. Synthetic tests
are never memory facts. Path success also requires actual simulator execution;
stability is tested separately afterwards. Fresh admission remains mandatory.
"""


def generation_view(packet, projected):
    from orchestrator.python_direction_grounding import generation_view as direction_view
    from orchestrator.python_skill_preflight import preflight_rows
    value = direction_view(packet, projected)
    value['context'].update(python_revision_contract_version=4,
        separation_use_contract=dict(helper='unseparated_parts(ctx, part_id, pose_id)',
            predicate='each min_distance < min_sep; equality is satisfied',
            interpretation='selected trajectory endpoint only; not blocker or impossibility',
            geometry_frame='initial assembled CAD; pose rotation maps relative vectors to world',
            synthetic_diagnostics_are_not_evidence=True, no_prescribed_winning_direction=True))
    previous = {r['case_id']:r.get('separation_grounding') for r in packet.get('context', {}).get(
        'development_preflight_feedback', {}).get('checks', [])}
    for row, case in zip(preflight_rows(packet), value['context']['revision_cases']):
        ctx, _ = build_context(row['current_observation_request'], extension_for())
        last = ctx.get('last')
        # Deterministic interpretation of already supplied observations, never
        # new evidence. Raw threshold/distances remain in the exact runtime input.
        case['original_endpoint_interpretation'] = dict(
            unmet_part_ids=None if not last else unmet(ctx, last['choice']['part_id'], last['choice']['pose_id']),
            derived_only=True, runtime_field_added=False, not_a_blocker_list=True)
        if previous.get(case['case_id']):
            case['previous_separation_check'] = deepcopy(previous[case['case_id']])
    return value


def check_separation_use(program, request, choice):
    packet = json.dumps(dict(program=program, request=request, choice=choice),
        ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(packet) > MAX_INPUT_BYTES:
        return dict(accepted=False, errors=['python_input_byte_limit'])
    runner = Path(__file__).absolute().with_name('python_separation_child.py')
    environment = {k:os.environ[k] for k in ('SystemRoot','WINDIR') if k in os.environ}
    try:
        child = subprocess.run([sys.executable,'-I','-S','-B',str(runner)], input=packet,
            capture_output=True, timeout=8, cwd=str(runner.parent), env=environment,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if child.returncode or len(child.stdout) > 65536:
            raise ValueError('python_separation_child_failed')
        result = json.loads(child.stdout)
        if type(result.get('accepted')) is not bool or type(result.get('errors')) is not list:
            raise ValueError('python_separation_child_contract')
        return result
    except subprocess.TimeoutExpired:
        return dict(accepted=False, errors=['python_separation_diagnostic_timeout'])
    except (ValueError, TypeError, OSError) as error:
        return dict(accepted=False, errors=['python_separation_diagnostic_failed:'+type(error).__name__])


def audit_separation_revision(program, packet):
    from orchestrator.python_direction_grounding import audit_direction_revision
    from orchestrator.python_skill_preflight import preflight_rows
    result = audit_direction_revision(program, packet)
    result['version'] = 'python_separation_revision_preflight_v4'
    for row, check in zip(preflight_rows(packet), result['checks']):
        if not check['accepted']:
            check['separation_grounding'] = dict(accepted=False, errors=['base_preflight_not_passed'], not_run=True)
            continue
        audit = check_separation_use(program, row['current_observation_request'], check['proposed_action'])
        check['separation_grounding'] = audit
        if not audit['accepted']:
            check['errors'].append('separation_use_not_verified')
        check['accepted'] = not check['errors']
    result['accepted'] = all(c['accepted'] for c in result['checks'])
    return result
