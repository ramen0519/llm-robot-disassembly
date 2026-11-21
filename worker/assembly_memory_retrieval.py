"""Same-CAD contrastive observation retrieval; no inference of feasibility."""
from copy import deepcopy
from worker.explicit_action_contract import canonical


def select_references(references, request, max_records):
    """Balance scoped successes/failures and show changed support conditions.

    Source seed/budget/config remain in every returned row. Repeated physical
    choices are represented once per observed outcome in the display, not merged
    in storage or treated as statistically independent evidence. Parent chains
    are kept whole, even when this requires displaying a repeated condition.
    """
    if type(max_records) is not int or not 0 <= max_records <= 12:
        raise ValueError("invalid_retrieval_limit")
    by_id = {r["evidence_id"]: r for r in references}
    if len(by_id) != len(references):
        raise ValueError("duplicate_reference")

    def task(context):
        return context["task"] if type(context.get("task")) is dict else context

    current = task(request["task_context"])
    remaining = set(request["remaining_parts"])
    poses = [p["matrix"] for p in request["pose_catalog"]]

    def same_cad(row):
        source = task(row["source_task_context"])
        return (bool(current.get("assembly_id")) and bool(current.get("input_sha256"))
                and source.get("assembly_id") == current["assembly_id"]
                and source.get("input_sha256") == current["input_sha256"])

    def priority(row):
        matched = same_cad(row)
        distance = (len(remaining.symmetric_difference(row["source_remaining_parts"]))
                    if matched else abs(len(remaining) - len(row["source_remaining_parts"])))
        return (not matched, not (matched and row["source_choice"]["part_id"] in request["candidate_ids"]),
                distance, not (matched and row["source_pose_matrix"] in poses))

    def family(row):
        source = task(row["source_task_context"])
        c = row["source_choice"]
        return canonical(dict(assembly=source.get("assembly_id"), input=source.get("input_sha256"),
            remaining=sorted(row["source_remaining_parts"]), pose=row["source_pose_matrix"],
            part=c["part_id"], primitive=c["skill_id"], direction=c["direction_id"],
            second_direction=c["second_direction_id"]))

    def display_key(row):
        context = task(row["source_task_context"])
        return canonical(dict(family=family(row), supports=sorted(row["source_choice"]["support_ids"]),
            base=context.get("base_part"),
            grippers=context.get("solver_settings", {}).get("max_grippers"),
            status=row["status"], path=row["path_success"], stability=row["stability_success"]))

    selected, selected_ids, shown, positive_families = [], set(), set(), set()

    def closure(row):
        chain, visiting = [], set()
        while row["evidence_id"] not in selected_ids:
            identity = row["evidence_id"]
            if identity in visiting:
                raise ValueError("cyclic_reference")
            visiting.add(identity)
            chain.append(row)
            parent = row.get("parent_failure_evidence_id")
            if parent is None:
                break
            if parent not in by_id:
                raise ValueError("missing_parent_reference")
            row = by_id[parent]
        return list(reversed(chain))

    def include(row):
        if display_key(row) in shown:
            return False
        chain = closure(row)
        if len(selected) + len(chain) > max_records:
            return False
        for item in chain:
            selected.append(deepcopy(item))
            selected_ids.add(item["evidence_id"])
            shown.add(display_key(item))
            if item["status"] == "PASS":
                positive_families.add(family(item))
        return True

    # Match CAD/state before comparing conditions. Do not let an exact-setting
    # failed observation hide a nearby successful one of equal CAD/state scope.
    buckets = {}
    for row in references:
        buckets.setdefault(priority(row), []).append(row)
    for key in sorted(buckets):
        rows = buckets[key]
        order = lambda row: (not row["same_conditions"], not row["same_task"],
                             len(closure(row)), row["evidence_id"])
        positives = sorted((r for r in rows if r["status"] == "PASS"), key=order)
        negatives = sorted((r for r in rows if r["status"] != "PASS"), key=order)
        while positives or negatives:
            if positives:
                include(positives.pop(0))
            # Prefer a contrast with the same mover/pose/path and another hold.
            # An outcome is still just an observed result under its source scope.
            negatives.sort(key=lambda row: (family(row) not in positive_families, order(row)))
            if negatives:
                include(negatives.pop(0))
    return selected
