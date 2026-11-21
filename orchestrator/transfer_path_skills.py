"""Cross-assembly procedure references from frozen, replayed path skills.

This read-only module never maps source part IDs/coordinates onto a target,
selects a target direction, runs physics, calls an API, or caches a verdict.
Source PASS means PASS only in its recorded source scope. Target selection and
fresh path/stability verification remain separate responsibilities.
"""
from copy import deepcopy
import math

from orchestrator import path_skill_memory as source_memory
from worker.constraint_memory import validate_scope
from worker.independent_features import FRAME as CAD_FRAME
from worker.task_memory import _sha, _validate_context, canonical, digest


POLICY = "cross_assembly_procedure_reference_no_coordinate_mapping_v1"
AXES = {"+X": [1, 0, 0], "-X": [-1, 0, 0], "+Y": [0, 1, 0],
        "-Y": [0, -1, 0], "+Z": [0, 0, 1], "-Z": [0, 0, -1]}
_COPY_KEYS = ("skill_evidence_id", "skill_id", "controller_sha256", "source_run_id",
              "source_request_id", "failed_constraint_ids", "source_scope", "source_scope_sha256",
              "validated_schedule", "source_trajectory_sha256", "validation", "lifecycle",
              "generalized", "force_new_physics", "hard_pruning", "reuse_verdict")


def load(path, allow_test_root=False):
    """Read the original format and verify every archived source-file hash."""
    return source_memory.load(path, allow_test_root=allow_test_root)


def _axis(action):
    if (type(action) is not list or len(action) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in action)):
        raise ValueError("source_action_requires_finite_unit_axis")
    for name, vector in AXES.items():
        if vector == action:
            return name
    raise ValueError("source_action_requires_finite_unit_axis")


def _source_schedule(schedule):
    if (type(schedule) is not list or len(schedule) != 2
            or any(type(item) is not dict or set(item) != {"step", "action"} for item in schedule)
            or type(schedule[0]["step"]) is not int or schedule[0]["step"] != 0
            or type(schedule[1]["step"]) is not int or not 1 <= schedule[1]["step"] <= 1000000):
        raise ValueError("source_requires_registered_two_leg_schedule")
    first, second = [_axis(item["action"]) for item in schedule]
    if sum(a * b for a, b in zip(AXES[first], AXES[second])) != 0:
        raise ValueError("source_turn_must_be_perpendicular")
    return first, second, schedule[1]["step"]


def _current(current):
    if type(current) is not dict:
        raise ValueError("transfer_target_must_be_object")
    context, scope, moving, controller, cad = None, None, None, None, None
    first_axis, allowed, switch = None, None, None
    if "memory_context" in current:
        context = current["memory_context"]
        moving = current.get("moving_part_id")
        registered = current.get("registered_skill")
        if registered is not None:
            if (type(registered) is not dict or registered.get("skill_id") != source_memory.SKILL
                    or not _sha(registered.get("code_sha256"))):
                raise ValueError("target_registered_controller_invalid")
            controller = registered["code_sha256"]
        refs = current.get("constraint_references", [])
        if type(refs) is not list or any(type(ref) is not dict for ref in refs):
            raise ValueError("target_constraint_references_invalid")
        scopes = [ref.get("source_scope") for ref in refs if ref.get("origin") == "current_episode_measurement"]
        if scopes:
            if any(canonical(value) != canonical(scopes[0]) for value in scopes):
                raise ValueError("conflicting_current_failure_scopes")
            scope = scopes[0]
            if type(scope) is not dict:
                raise ValueError("current_failure_scope_missing")
        cad = current.get("cad")
        first_axis, allowed, switch = [current.get(k) for k in ("first_axis", "allowed_second_axes", "switch_step")]
    elif "task_context" in current:
        scope, context, moving = current, current["task_context"], current.get("part_move")
    else:
        context = current
    _validate_context(context)
    if moving is not None and (type(moving) is not str or moving not in context["initial_parts"]
                               or moving == context["task"]["base_part"]):
        raise ValueError("target_moving_part_outside_candidate_set")
    assembled = None
    if scope is not None:
        validate_scope(scope, context)
        if moving is not None and moving != scope["part_move"]:
            raise ValueError("target_moving_part_disagrees_with_scope")
        moving, assembled = scope["part_move"], list(scope["assembled_parts"])
        measured_controller = scope["physics"].get("source_and_binary_hashes", {}).get("controller_sha256")
        if measured_controller is not None and not _sha(measured_controller):
            raise ValueError("target_measured_controller_invalid")
        if controller is not None and measured_controller is not None and controller != measured_controller:
            raise ValueError("target_controller_disagrees_with_measured_scope")
        if controller is None:
            controller = measured_controller
    if cad is not None:
        if (type(cad) is not dict or canonical(cad.get("task")) != canonical(context["task"])
                or type(cad.get("parts")) is not dict or not cad["parts"]
                or not set(cad["parts"]).issubset(context["initial_parts"])
                or any(canonical(value) != canonical(context["geometry_features"][part])
                       for part, value in cad["parts"].items())):
            raise ValueError("target_CAD_disagrees_with_context")
        observed = sorted(cad["parts"])
        if assembled is not None and set(observed) != set(assembled):
            raise ValueError("target_CAD_disagrees_with_current_state")
        assembled = observed
    if moving is not None and assembled is not None and moving not in assembled:
        raise ValueError("target_moving_part_not_assembled")
    if any(value is not None for value in (first_axis, allowed, switch)):
        if (type(first_axis) is not str or first_axis not in AXES
                or type(allowed) is not list or not allowed
                or any(type(axis) is not str or axis not in AXES for axis in allowed)
                or len(allowed) != len(set(allowed))
                or any(sum(a * b for a, b in zip(AXES[first_axis], AXES[axis])) != 0 for axis in allowed)
                or type(switch) is not int or not 1 <= switch <= 1000000):
            raise ValueError("target_action_domain_invalid")
        if scope is not None and scope["action_schedule"] != [{"step": 0, "action": AXES[first_axis]}]:
            raise ValueError("target_initial_axis_disagrees_with_measured_attempt")
    return {"context": context, "scope": scope, "moving": moving, "assembled": assembled,
            "controller": controller, "first_axis": first_axis,
            "allowed_second_axes": allowed, "switch_step": switch}


def matching_cards(snapshot, current, max_records=4):
    """Retrieve cross-assembly procedure references, never target parameters.

    Accept path/candidate features containing memory_context, a full measured
    scope, or a bare task context. Missing current scope/controller is explicit.
    A supplied controller hash must match the registered source code. Ordering
    is stable evidence-ID order, not a geometric score or a recommended skill.
    Same-assembly references stay in path_skill_memory.matching_cards.
    """
    if type(max_records) is not int or not 0 <= max_records <= 4:
        raise ValueError("transfer_reference_limit_must_be_zero_to_four")
    source_memory._validate_snapshot(snapshot)
    target = _current(current)
    context, current_scope = target["context"], target["scope"]
    cards = []
    for record in snapshot["records"]:
        source = record["source_scope"]
        source_task = source["task_context"]["task"]
        if source_task["assembly_id"] == context["task"]["assembly_id"]:
            continue
        if target["controller"] is not None and target["controller"] != record["controller_sha256"]:
            continue
        first, second, step = _source_schedule(record["validated_schedule"])
        card = {key: deepcopy(record[key]) for key in _COPY_KEYS}
        differences = ["different_assembly", "unverified_part_correspondence", "unverified_coordinate_correspondence"]
        if source_task["input_sha256"] != context["task"]["input_sha256"]:
            differences.append("different_input_assets")
        if current_scope is None:
            differences.append("current_scope_not_supplied")
        else:
            differences.extend("scope_" + key for key in source_memory._same_conditions(source, current_scope))
        if target["controller"] is None:
            differences.append("current_controller_not_supplied")
        observed = None if current_scope is None else deepcopy(current_scope["action_schedule"])
        # These IDs and geometry are explicitly SOURCE fields. No mapping is
        # inferred even if source and target happen to contain the same ID text.
        card["source_geometry"] = {
            "source_part_id": source["part_move"],
            "observations": deepcopy(source["task_context"]["geometry_features"][source["part_move"]]),
            "cad_frame": CAD_FRAME, "simulation_pose": deepcopy(source["pose"]),
            "path_state_frame": source.get("frame"), "units": "input_CAD_units_not_verified"}
        moving = target["moving"]
        card["target_context"] = {
            "task": deepcopy(context["task"]), "task_context_sha256": digest(context),
            "initial_parts": list(context["initial_parts"]), "assembled_parts": deepcopy(target["assembled"]),
            "moving_part_id": moving,
            "moving_geometry_observations": None if moving is None else deepcopy(context["geometry_features"][moving]),
            "cad_frame": CAD_FRAME, "simulation_pose": None if current_scope is None else deepcopy(current_scope["pose"]),
            "simulation_scope_supplied": current_scope is not None,
            "path_state_frame": None if current_scope is None else current_scope.get("frame"),
            "scope_sha256": None if current_scope is None else digest(current_scope),
            "registered_controller_sha256": target["controller"],
            "observed_action_schedule": observed,
            "first_axis": target["first_axis"], "allowed_second_axes": deepcopy(target["allowed_second_axes"]),
            "switch_step": target["switch_step"], "units": "input_CAD_units_not_verified"}
        card["transfer_semantics"] = {
            "policy": POLICY, "source_and_target_geometry_kept_separate": True,
            "current_part_id_mapping": None, "coordinate_transform": None, "axis_mapping": None,
            "axis_labels_have_same_geometric_role": None, "unit_scale_mapping": None,
            "action_kind": "registered_external_force_direction_command_not_a_waypoint_or_displacement",
            "action_frame": "upstream_simulation_world_axes",
            "source_axis_names": [first, second], "source_switch_step": step,
            "source_force_mag": source["physics"].get("force_mag"),
            "source_timestep": source["physics"].get("timestep"),
            "source_schedule_role": "source_validation_only_not_target_instruction",
            "procedure_pattern": "one_continuous_simulation_switch_to_a_perpendicular_force_without_reset",
            "target_parameter_selection": "required_from_current_observations_and_current_registered_domain",
            "target_selected_schedule": None, "target_axis_recommendation": None,
            "trajectory_copy_allowed": False, "past_pass_reused": False,
            "source_failed_constraint_ids_are_target_constraints": False,
            "target_path_verdict": "UNKNOWN", "target_stability_verdict": "UNKNOWN",
            "target_generalization_verified": False,
            "fresh_validation_required": ["registered_controller_version", "path_separation", "support_stability"]}
        card["source_provenance"] = {
            "snapshot_content_sha256": snapshot["content_sha256"], "record_sha256": record["record_sha256"],
            "source_parent_call_id": record["source_parent_call_id"],
            "file_sha256": {key: entry["sha256"] for key, entry in record["source_files"].items()}}
        card.update(reference_only=True, current_attempt_verified=False,
                    current_part_id_mapping=None, applicability="cross_assembly_procedure_reference",
                    condition_mismatches=differences, current_observed_action_schedule=observed,
                    action_schedule_matches_current_observation=observed == record["validated_schedule"],
                    transfer_policy=POLICY, source_validation_applies_to_target=False)
        canonical(card)
        cards.append(card)
        if len(cards) == max_records:
            break
    return cards if max_records else []
