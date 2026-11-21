"""Development-only diagnostics for executed reads of the real Python input.

No wrappers change input types/values. Unknown fields in closed records are
errors; dynamic ID maps and optional observed diagnostics are not hard failures.
This is branch coverage, not a proof about every possible execution.
"""
from copy import deepcopy

DESCRIPTION = """Runtime input contract (not a recommendation):
Support choices are computed from remaining_parts minus selected part and
base_part, with at most max_grippers-1 supports. There is NO geometry.support_ids
or precomputed support recommendation. A legal support set need not be stable.
base_part may be None: it means NO fixed base, not missing geometry. Do not
require geometry.parts[base_part] or invent a base. Derive supported alternatives
from current remaining parts, pose and observed failure, or explicitly abstain.
The predeclared development applicability examples include required contexts;
abstention is safe but is NOT a successful repair on those examples.
ctx.failure contains only kind, evidence_id, physical_attempt_executed,
path_success, stability_success. observed_facts.failure_cause in a lesson is NOT
a ctx.failure field. Read executable fields from input_schema only.
geometry.parts is a map keyed by current part ID; each record may contain bounds,
extents, bounds_center, center_mass, volume, watertight, validity. Check missing or
invalid measurements. geometry.relations.frame is the geometry frame (not
geometry.frame). Apply the selected pose matrix when comparing world directions.
last.feedback.measured_endpoint may be absent/null; pairs may include
moving_part_id, rest_part_id, min_distance, separation_unsatisfied. These describe
an observed endpoint, not a proven blocker or a successful alternative.
runtime_input_examples below are subsets of build_context output: shown field
paths have exactly their executable nesting. Missing optional data are unknown.
Use input_schema for fields, not invented field names. Do not copy example IDs.
Refinement must change the physical action on a bound failed debug input, or
abstain explicitly. Renaming variables, reordering supports, pose aliases or
changing evidence citations is not a behavioral change. This development gate
does not prove impossibility or reuse a physics verdict. New actions still need
fresh physical admission. No extra model call is made by this gate.
"""

# Only records whose meaning is defined by the executable contract are closed.
# The remainder are logged as missing observations, not invented schema errors.
CLOSED = {
    ('ctx',): set('candidate_ids remaining_parts base_part max_grippers pose_catalog geometry memory_references last directions primitive_ids failure plan_guard skill_history lesson_references'.split()),
    ('ctx','geometry'): set('schema_version kind task parts relations role verifier_calls prior_plan_or_verdicts_used input_integrity state'.split()),
    ('ctx','geometry','parts','*'): set('bounds extents bounds_center center_mass volume watertight validity'.split()),
    ('ctx','geometry','relations'): set('contact_edges contact_source contact_graph_recomputed contact_graph_frame_verified contact_is_blocker_proof frame units known_precedence_constraints precedence_source'.split()),
    ('ctx','last'): set('choice feedback unstable initial_disconnected pose_matrix'.split()),
    ('ctx','last','feedback'): set('evidence_id status action_id remaining_parts choice decision_source selected_pose_matrix path_success stability_success measured_endpoint raw_stability_parts accepted_for_transition deadline_exceeded stability_diagnostics'.split()),
    ('ctx','last','feedback','measured_endpoint'): set('all_paths_impossible blocker evidence_sha256 min_sep pairs scope termination_reason'.split()),
    ('ctx','last','feedback','measured_endpoint','pairs','[]'): set('min_distance moving_part_id rest_part_id separation_unsatisfied'.split()),
    ('ctx','pose_catalog','[]'): {'pose_id','matrix'},
    ('ctx','failure'): set('kind evidence_id physical_attempt_executed path_success stability_success'.split()),
}
CHOICE = set('part_id pose_id skill_id direction_id second_direction_id support_ids evidence_ids'.split())


def pattern(path):
    if path[:3] == ('ctx','geometry','parts') and len(path) > 3:
        return path[:3] + ('*',) + path[4:]
    return path


class ReadAudit:
    def __init__(self, context):
        self.paths, self.findings = {}, []
        self.copies = []
        self._seen = set()
        self._index(context, ('ctx',))

    def _index(self, value, path):
        if type(value) not in (dict, list):
            return
        self.paths[id(value)] = path
        if type(value) is dict:
            for key, child in value.items():
                self._index(child, path + (str(key),))
        else:
            for child in value:
                self._index(child, path + ('[]',))

    def copied(self, source, target):
        if id(source) in self.paths:
            if len(self.copies) >= 512:
                raise ValueError('python_diagnostic_copy_limit')
            # Keep copied records alive so a recycled Python object ID cannot
            # accidentally mark a later local dict as external input.
            self.copies.append(target)
            self.paths[id(target)] = self.paths[id(source)]

    def read(self, target, key, line, method):
        if type(target) is not dict or id(target) not in self.paths or key in target:
            return
        path = self.paths[id(target)]
        allowed = CHOICE if path[-1:] == ('choice',) else CLOSED.get(pattern(path))
        code = 'unknown_input_field' if allowed is not None and key not in allowed else 'missing_observation'
        identity = (path, str(key), line, method)
        if identity in self._seen:
            return
        self._seen.add(identity)
        if len(self.findings) >= 48:
            return
        self.findings.append(dict(code=code, path='.'.join(path), key=str(key)[:100],
            line=line, method=method, available_keys=sorted(str(k) for k in target)[:40]))

    def report(self):
        return dict(accepted=not any(x['code']=='unknown_input_field' for x in self.findings),
            findings=deepcopy(self.findings), coverage='executed_reads_only',
            physics_calls=0, api_calls=0)


def input_description(context):
    """Actual fields, compact dynamic-ID maps; never admission observations."""
    schema = {}
    def walk(value, path):
        name = '.'.join(pattern(path))
        if type(value) is dict:
            if path == ('ctx','geometry','parts'):
                schema[name] = 'map[current_part_id]'
            else:
                schema.setdefault(name, set()).update(value)
            for key, child in value.items():
                if path in (('ctx','geometry','task'), ('ctx','geometry','input_integrity')):
                    continue
                walk(child, path + (str(key),))
        elif type(value) is list:
            for child in value[:1]:
                walk(child, path + ('[]',))
    walk(context, ('ctx',))
    example = deepcopy(context)
    # Omit only complete subtrees; never rename/move fields as the old projection did.
    for key in ('task','input_integrity'):
        example['geometry'].pop(key, None)
    example.pop('memory_references', None)
    return dict(input_schema={k: sorted(v) if isinstance(v,set) else v for k,v in schema.items()},
                runtime_input_example=example, omitted_subtrees=['geometry.task','geometry.input_integrity','memory_references'])
