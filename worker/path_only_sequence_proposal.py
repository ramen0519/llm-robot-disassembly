"""Build an ordered candidate from accepted part-level path checks (Python 3.8+)."""
import math

PROFILE = "path_only_sequence_proposal/v1"


def _state(ids):
    if not isinstance(ids, (list, tuple)) or not ids or any(type(x) is not str or not x for x in ids):
        raise ValueError("expected a non-empty sequence of part identifiers")
    state = frozenset(ids)
    if len(state) != len(ids):
        raise ValueError("duplicate part identifiers")
    return state


def build_proposal(parts, edges, *, profile=PROFILE):
    """Find the longest root-connected, chronologically accepted chain.

    DFS can backtrack. The first witnessed parent of each reachable subset is
    enough for an order proposal. Per-edge poses remain explicit conditions.
    """
    if profile != PROFILE:
        raise ValueError("unexpected verifier profile")
    root = _state(parts)
    if len(root) < 2:
        raise ValueError("a multi-part input is required")
    reached = {root: None}
    best = root
    for edge in edges:
        parent, child = _state(edge['parent']), _state(edge['child'])
        mover = edge['part_move']
        if not parent <= root or child != parent - {mover} or mover not in parent:
            raise ValueError("invalid removal transition")
        passed = edge['path_pass']
        accepted = edge.get('accepted_for_transition', passed)
        if type(passed) is not bool or type(accepted) is not bool or accepted and not passed:
            raise ValueError("invalid path verdict or transition acceptance")
        if passed:
            vector = edge['action']
            if (not isinstance(vector, (list, tuple)) or len(vector) != 3 or
                    not all(type(x) in (int, float) and math.isfinite(x) for x in vector) or not any(vector)):
                raise ValueError("a passing path requires a finite nonzero direction")
        if accepted and parent in reached and child not in reached:
            reached[child] = edge
            if len(child) < len(best):
                best = child
    cursor, steps = best, []
    while cursor != root:
        edge = reached[cursor]
        steps.append(dict(edge))
        cursor = frozenset(edge['parent'])
    steps.reverse()
    remaining = sorted(best)
    order = [edge['part_move'] for edge in steps]
    complete = len(best) == 1
    return dict(profile=profile, complete=complete, disassembly_order=order,
                remaining_parts=remaining, steps=steps,
                assembly_order_candidate=remaining + order[::-1] if complete else None,
                completion_scope='part_level_path_with_other_parts_fixed',
                robot_execution_validated=False, reorientation_validated=False)
