"""Predeclared replay conditions for one recorded plan guard; no search state."""
from copy import deepcopy
from orchestrator.conditional_plan import task_scope
from orchestrator.pose_guard_recovery import make_guard
from orchestrator.python_skill_pose import multiply, transform_vector, closest_axis
from worker.explicit_action_contract import DIRECTIONS, digest, validate_request

YAW = {
    0: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    90: [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    180: [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
}


def prepare(source, run_id, yaw=0):
    if type(yaw) is not int or yaw not in YAW:
        raise ValueError("undeclared_guard_yaw")
    req, plan = deepcopy(source["request"]), deepcopy(source["plan"])
    req.update(run_id=run_id, request_id=run_id + ":initial")
    # All source history is preserved as historical information. There are no
    # attempted actions at this state before the recorded guard.
    if req["attempted_action_ids"]:
        raise ValueError("source_guard_replay_requires_unattempted_state")
    if yaw:
        for pose in req["pose_catalog"]:
            pose["matrix"] = multiply(YAW[yaw], pose["matrix"])
            pose["pose_id"] = "pose_" + digest(pose["matrix"])[:16]
        for key, rotation in plan["source_poses"].items():
            plan["source_poses"][key] = [[sum(YAW[yaw][i][k] * rotation[k][j]
                for k in range(3)) for j in range(3)] for i in range(3)]
        for step in plan["steps"]:
            for key in ("direction", "second_direction"):
                if step[key] is not None:
                    step[key] = closest_axis(transform_vector(YAW[yaw], DIRECTIONS[step[key]]))
        # The old episode belongs to the original world frame, not this case.
        # Preserve it in the source artifact, never relabel as new observations.
        req["episode_feedback"] = []
        req["evidence_ids"] = [r["evidence_id"] for r in req["memory_references"]]
    validate_request(req)
    plan["task_scope"] = task_scope(req)
    guard = make_guard(plan, source["step_index"], req)
    return dict(request=req, plan=plan, index=source["step_index"], guard=guard,
        declared_world_yaw_degrees=yaw, source_plan_rebound_for_replay=True,
        new_llm_plan=False, unseen_assembly=False, repositioning_trajectory_verified=False)


def validate_replay(bundle, source):
    expected = prepare(source, bundle["request"]["run_id"],
                       bundle["declared_world_yaw_degrees"])
    if any(bundle.get(k) != v for k, v in expected.items()):
        raise ValueError("guard_replay_not_declared_source_transform")
    return expected
