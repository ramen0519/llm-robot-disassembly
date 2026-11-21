"""Opt-in v3: test executable direction dependence, never predict physics."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from orchestrator.python_skill_program import MAX_INPUT_BYTES
from orchestrator.python_direction_probe import MAX_HELPER_CALLS
from worker.task_memory import digest

INSTRUCTIONS = f"""
Direction-use contract v3. In addition to propose(ctx), implement the pure helper
select_directions(ctx, part_id, pose_id), returning a list of one or two current
direction IDs, or None. propose MUST call it on the FULL unchanged current ctx
and selected mover/pose and use that returned sequence in the actual action.
One direction is single_force_v1; two is turn_force_two_leg_v1. Do not call
eligible from this helper. Eligibility checks belong outside directional reasoning.
Do not recurse into this helper; at most {MAX_HELPER_CALLS} calls per propose evaluation.
Use the selected mover's spatial relation to remaining geometry, transformed by
the selected pose, AND conditional observed failure details in choosing signed
world directions. Extents used only to choose a mover, dead feature reads,
abs(dot) that erases positive/negative direction, and first-axis tie selection
do not demonstrate this. Account for unavailable geometry/evidence by abstaining;
do not invent a blocker, successful direction, or another selectable pose.
The host executes your helper with mover/pose fixed, compares synthetic geometry
position interventions and failure force/endpoint interventions separately, and
checks independence from dict serialization order. Both evidence families must
demonstrably affect concrete direction choices, not only switch abstention on/off.
These diagnostics are NOT physically observed outcomes. Never cite them as data,
identify test cases/part IDs, or manufacture branches just to pass. Supply your
own geometry/feedback-based algorithm; no host-prescribed direction is provided.
Sensitivity only establishes behavior on these inputs, not physical correctness;
fresh RedMax admission remains mandatory. Do not output another predicted-action
JSON or natural-language explanation as a substitute for executable logic.
"""

def source_diagnostics(program):
    if not program:return None
    tree=ast.parse(program['source'])
    unused=[]
    for function in tree.body:
        if not isinstance(function,ast.FunctionDef):continue
        stores={n.id for n in ast.walk(function) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        loads={n.id for n in ast.walk(function) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        unused.extend(dict(function=function.name,name=name) for name in sorted(stores-loads))
    return dict(program_sha256=digest(program),unused_assigned_locals=unused,
        direction_helper_present=any(isinstance(n,ast.FunctionDef) and n.name=='select_directions' for n in tree.body),
        static_diagnostic_NOT_failure_cause=True)

def generation_view(packet, projected):
    from orchestrator.python_execution_revision import execution_view
    value=execution_view(packet,projected)
    value['context'].update(python_revision_contract_version=3,
        direction_use_contract=dict(helper='select_directions(ctx, part_id, pose_id)',
            maximum_helper_calls_per_evaluation=MAX_HELPER_CALLS,
            output='one_or_two_direction_ids_or_None',counterfactuals_are_not_evidence=True,
            no_prescribed_winning_direction=True,required_sources=['geometry','measured_failure_details']),
        previous_direction_source_diagnostics=source_diagnostics(packet.get('previous_program')))
    previous={r['case_id']:r.get('direction_grounding') for r in
              packet.get('context',{}).get('development_preflight_feedback',{}).get('checks',[])}
    for case in value['context']['revision_cases']:
        audit=previous.get(case['case_id'])
        if audit:
            case['previous_direction_check']={k:deepcopy(audit[k]) for k in
                ('accepted','errors','actual_helper_bound_to_choice','geometry_direction_sensitive',
                 'failure_direction_sensitive','map_order_invariant','probe_count','scope') if k in audit}
    return value

def check_direction_use(program,request,choice):
    packet=json.dumps(dict(program=program,request=request,choice=choice),ensure_ascii=False,allow_nan=False).encode('utf-8')
    if len(packet)>MAX_INPUT_BYTES:
        return dict(accepted=False,errors=['python_input_byte_limit'])
    runner=Path(__file__).absolute().with_name('python_direction_child.py')
    environment={k:os.environ[k] for k in ('SystemRoot','WINDIR') if k in os.environ}
    try:
        child=subprocess.run([sys.executable,'-I','-S','-B',str(runner)],input=packet,
            capture_output=True,timeout=8,cwd=str(runner.parent),env=environment,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if child.returncode or len(child.stdout)>65536:raise ValueError('python_direction_child_failed')
        value=json.loads(child.stdout)
        if type(value.get('accepted')) is not bool or type(value.get('errors')) is not list:
            raise ValueError('python_direction_child_contract')
        return value
    except subprocess.TimeoutExpired:
        return dict(accepted=False,errors=['python_direction_diagnostic_timeout'])
    except (ValueError,TypeError,OSError) as error:
        return dict(accepted=False,errors=['python_direction_diagnostic_failed:'+type(error).__name__])

def audit_direction_revision(program,packet):
    from orchestrator.python_execution_revision import audit_execution_revision
    from orchestrator.python_skill_preflight import preflight_rows
    result=audit_execution_revision(program,packet)
    result['version']='python_direction_revision_preflight_v3'
    for row,check in zip(preflight_rows(packet),result['checks']):
        if not check['accepted']:
            check['direction_grounding']=dict(accepted=False,errors=['base_preflight_not_passed'],not_run=True)
            continue
        audit=check_direction_use(program,row['current_observation_request'],check['proposed_action'])
        check['direction_grounding']=audit
        if not audit['accepted']:check['errors'].append('direction_use_not_verified')
        check['accepted']=not check['errors']
    result.update(accepted=all(c['accepted'] for c in result['checks']),
        synthetic_diagnostics_not_physical_evidence=True,physical_calls=0,api_calls=0,
        new_physical_success=False)
    return result
