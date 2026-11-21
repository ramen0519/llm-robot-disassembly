"""Compact projection of bound development evidence for Python synthesis.

Original packets and raw physics artifacts remain in each exclusive run.
No direction recommendation or physical conclusion is added here.
"""
from copy import deepcopy
from worker.stability_diagnostics import project_stability_diagnostics


def generation_view(packet):
    value = deepcopy(packet)
    if 'support_development_experiences' in value['context']:
        from orchestrator.support_learning import compact_bundle
        value['context']['support_development_experiences'] = compact_bundle(
            value['context']['support_development_experiences'])
        # Full measured diagnostics occur once in the complete grid above; keep
        # every top-level citation and outcome without a second graph/trace copy.
        for item in value.get('evidence', []):
            if item.get('origin') == 'predeclared_diagnostic_grid_NOT_generated_skill':
                item.pop('stability_diagnostics', None)
                item['diagnostics_reference'] = 'support_development_experiences.rows by evidence_id'
    if 'multi_source_initial_contact_transitions' in value['context']:
        # Apply the same lossless-pair projection independently to each task;
        # raw geometry already lives in bound runtime examples and local files.
        rows=[]
        for source in value['context']['multi_source_initial_contact_transitions']:
            projected=generation_view(dict(context=dict(python_failure_scope='remaining_body_stability_failed',
                initial_contact_observed_transitions=source['summary'])))
            rows.append(dict(task_id=source['task_id'],summary=projected['context'].get('initial_contact_observed_transitions')))
        value['context']['multi_source_initial_contact_transitions']=rows
    # Full requests remain local for exact-input preflight. The model gets the
    # common runtime projection, not task hashes and duplicated raw envelopes.
    value['context'].pop('python_debug_requests', None)
    value["context"].pop("language_semantics", None)  # legacy JSON IR, not Python
    value["context"]["evidence_projection"] = "python_debug_evidence_v1"
    transitions = value["context"].get("initial_contact_observed_transitions")
    if value["context"].get("python_failure_scope") and transitions is not None:
        # Keep every pair and denominator, while removing duplicated geometry,
        # hashes and raw observations already carried in the evidence array.
        compact = {key: deepcopy(transitions[key]) for key in (
            "kind", "trace_attempt_count", "phase_denominator", "eligible_observation_count",
            "excluded_attempt_count", "excluded_attempts", "initial_failure_count",
            "total_pair_count", "emitted_pair_count", "omitted_pair_count",
            "unmatched_initial_failures", "selection_rule",
            "source_ids_are_current_bindings", "prescriptive_policy_generated") if key in transitions}
        compact["pairs"] = []
        for pair in transitions["pairs"]:
            row = {key: deepcopy(pair[key]) for key in (
                "failure_evidence_id", "success_evidence_id", "changed_choice_fields",
                "same_recorded_state_and_pose", "same_remaining_parts", "same_rng_state",
                "success_mover_in_observed_initial_disconnected_set",
                "different_mover_and_possible_direction_or_support_are_confounders",
                "failure_cause", "causal_effect_established", "generalization_established",
                "reference_only", "current_part_id_mapping", "physics_reuse_allowed",
                "recommended_action")}
            for stage in ("before", "after"):
                row[stage] = {key: deepcopy(pair[stage][key]) for key in (
                    "choice", "pose_matrix", "remaining_parts", "base_part",
                    "max_grippers", "initial_disconnected_parts", "diagnostic_phase",
                    "path_success", "stability_success")}
            row["observed_role_relationships"] = dict(
                after_support_contains_before_mover=(
                    pair["before"]["choice"]["part_id"] in pair["after"]["choice"]["support_ids"]),
                after_mover_equals_before_mover=(
                    pair["before"]["choice"]["part_id"] == pair["after"]["choice"]["part_id"]),
                physical_causality_established=False)
            compact["pairs"].append(row)
        value["context"]["initial_contact_observed_transitions"] = compact
    projected = []
    for record in value.get("validation_feedback", []):
        if record.get("phase") != "debug":
            raise ValueError("python_generation_requires_debug_feedback_only")
        row = {key: deepcopy(record.get(key)) for key in (
            "case_id", "phase", "outcome", "validation_record_sha256")}
        if "origin" in record:
            row["origin"] = record["origin"]
            row["retain_successful_probe"] = record.get("retain_successful_probe", False)
        request = record.get("current_observation_request", {})
        row["current_observation_request"] = {
            key: deepcopy(request[key]) for key in (
                "candidate_ids", "remaining_parts", "base_part", "max_grippers",
                "pose_catalog", "episode_feedback") if key in request}
        if (value['context'].get('lesson_stage') or value["context"].get("stability_coordinate_contract")) and "geometry" in request:
            # Actual bound CAD observations needed to express a current-role
            # hypothesis. Keep source IDs as labels, never a target mapping.
            row["current_observation_request"]["geometry"] = {
                "parts": {part: {key: deepcopy(feature[key]) for key in
                    ("extents", "volume", "bounds_center", "watertight", "validity") if key in feature}
                    for part, feature in request["geometry"].get("parts", {}).items()},
                "frame": request["geometry"].get("relations", {}).get("frame"),
                "physics_success_implied": False}
            if value['context'].get('lesson_stage'):
                geometry = row['current_observation_request']['geometry']
                # Preserve executable nesting for new lesson generation. The
                # historical legacy projection above remains readable unchanged.
                geometry.pop('frame')
                geometry.pop('physics_success_implied')
                geometry['relations'] = {key: deepcopy(request['geometry']['relations'][key])
                    for key in ('frame', 'units') if key in request['geometry'].get('relations', {})}
                row['geometry_projection'] = dict(partial=True, physics_success_implied=False)
        for stage in ("original_trigger", "recovery"):
            raw = record.get(stage)
            row[stage] = None if raw is None else {
                key: deepcopy(raw.get(key)) for key in (
                    "status", "backend", "chosen", "pose", "path_success", "stability_success",
                    "elapsed_seconds", "path_check_calls", "stability_check_calls",
                    "path_physics_steps", "controller", "skill_fully_executed",
                    "endpoint_observation", "all_paths_impossible", "blocker", "reason",
                    "evidence_sha256")}
            if raw is not None and raw.get("stability_diagnostics") is not None:
                row[stage]["stability_diagnostics"] = project_stability_diagnostics(
                    raw["stability_diagnostics"])
                row[stage]["stability_mode"] = raw.get("stability_mode")
        projected.append(row)
    if "validation_feedback" in value:
        value["validation_feedback"] = projected
    if 'support_development_experiences' in value['context']:
        from orchestrator.support_learning import deduplicate_observations
        value = deduplicate_observations(value)
    return value
