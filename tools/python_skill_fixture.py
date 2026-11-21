"""HANDWRITTEN numerical/conditional integration fixture; not LLM discovery."""

SOURCE = '''def dot(a, b):
    total = 0
    for i in range(3):
        total += a[i] * b[i]
    return total

def propose(ctx):
    last = ctx["last"]
    if last is None or last["feedback"]["path_success"] is not False:
        return None
    # Rotate a locally upward vector into world force coordinates using the
    # currently exposed pose. This computes; it does not certify feasibility.
    pose = last["pose_matrix"]
    upward = [pose[0][2], pose[1][2], pose[2][2]]
    best = None
    score = None
    for direction in ctx["directions"]:
        alignment = dot(upward, ctx["directions"][direction])
        for second in ctx["directions"]:
            if dot(ctx["directions"][direction], ctx["directions"][second]) != 0:
                continue
            action = {
                "part_id": last["choice"]["part_id"],
                "pose_id": last["choice"]["pose_id"],
                "skill_id": "turn_force_two_leg_v1",
                "direction_id": direction,
                "second_direction_id": second,
                "support_ids": last["choice"]["support_ids"],
                "evidence_ids": [last["feedback"]["evidence_id"]]
            }
            if eligible(action) and (score is None or alignment > score):
                best = action
                score = alignment
    return best
'''

PROGRAM = {"language": "python_policy_v1", "source": SOURCE}
