"""Synthetic event-contract fixtures; these tests do not run physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from audit_assembly_events import audit_events


PARTS = ["10", "2", "3", "fixed"]


def fixture(mode="rank", ranker="reverse_native", base_part="fixed"):
    events = []
    def emit(kind, **values):
        events.append(dict(kind=kind, run_id="fixture", event_id="e" + str(len(events)), **values))
    emit("run_start", mode=mode, ranker_name=ranker if mode == "rank" else None,
         live_llm_calls=0, base_part=base_part)
    if mode == "native":
        emit("run_end", status="returned_from_upstream")
        return events
    native = [part for part in PARTS if part != base_part]
    applied = (list(reversed(native)) if ranker == "reverse_native" else sorted(native)
               if ranker == "id_ascending" else native) if mode == "rank" else native
    state = hashlib.sha256(json.dumps(sorted(PARTS)).encode()).hexdigest()
    if mode == "rank":
        emit("rank_decision", generation_id="g1", ranker_name=ranker, fallback_reason=None,
             rng_before="unchanged", rng_after="unchanged", applied_order=applied,
             request=dict(run_id="fixture", request_id="r1", state_id=state, assembled_parts=PARTS,
                          candidate_ids=native, base_part=base_part))
    emit("generation", generation_id="g1", state_id=state, assembled_parts=PARTS,
         native_order=native, applied_order=applied)
    emit("candidate_yield", generation_id="g1", part=applied[0])
    moving = applied[0]
    rest = [part for part in PARTS if part != moving]
    # Two bounded poses for the same yielded candidate are valid distinct calls.
    for index, status in enumerate(("UNKNOWN", "PASS")):
        pose = [[1, 0, index], [0, 1, 0], [0, 0, 1]]
        emit("verifier_start", call_id="v" + str(index), generation_id="g1",
             request=dict(part_move=moving, parts_rest=rest, parts_removed=[], pose=pose, max_grippers=2))
        emit("verifier_result", call_id="v" + str(index), status=status, blocker=None, path=None,
             raw=dict(feasible=status == "PASS", part_move=moving, base_part=base_part, pose=pose,
                      action=[0, 0, 1] if status == "PASS" else None,
                      parts_fix=[] if status == "PASS" else None))
    # The final two-to-one simulation is outside the candidate generator.
    terminal = base_part if base_part is not None else PARTS[-1]
    leaf_move = next(part for part in PARTS if part not in (moving, terminal))
    emit("verifier_start", call_id="leaf", generation_id=None,
         request=dict(part_move=leaf_move, parts_rest=[terminal],
                      parts_removed=[part for part in PARTS if part not in (leaf_move, terminal)],
                      pose=None, max_grippers=2))
    emit("verifier_result", call_id="leaf", status="PASS", blocker=None, path=None,
         raw=dict(feasible=True, part_move=leaf_move, base_part=base_part, pose=None,
                  action=[0, 1, 0], parts_fix=[]))
    emit("run_end", status="returned_from_upstream")
    return events


class AssemblyEventTests(unittest.TestCase):
    def audit(self, trace=None, mode="rank", ranker="reverse_native", base_part="fixed"):
        return audit_events(fixture(mode, ranker, base_part) if trace is None else trace,
                            mode, ranker, PARTS, base_part)

    def test_rules_trace_and_optional_base_accept_all_candidates_and_multiple_poses(self):
        for mode, ranker, base in (("rank", "noop", "fixed"), ("rank", "id_ascending", "fixed"),
                                   ("rank", "reverse_native", "fixed"), ("trace", "noop", "fixed"),
                                   ("rank", "reverse_native", None), ("trace", None, None)):
            with self.subTest(mode=mode, ranker=ranker, base=base):
                result = self.audit(mode=mode, ranker=ranker, base_part=base)
                self.assertEqual(result["status"], "PASS", result["checks"])
                self.assertEqual(result["metrics"]["verifier_calls"], 3)
                self.assertEqual(result["metrics"]["verifier_status_counts"], {"PASS": 2, "UNKNOWN": 1, "ERROR": 0})

    def test_native_has_no_measured_verifier_count(self):
        result = self.audit(mode="native", ranker=None)
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(result["metrics"]["verifier_calls"])
        trace = fixture("native")
        trace.insert(1, dict(kind="generation", run_id="fixture", event_id="extra"))
        self.assertEqual(self.audit(trace, "native", None)["status"], "ERROR")

    def test_faithfully_logged_error_remains_distinct_from_integrity_error(self):
        trace = fixture()
        error = next(event for event in trace if event.get("kind") == "verifier_result")
        error.update(status="ERROR", raw=None, error_type="RuntimeError")
        trace[-1].update(status="ERROR", error_type="RuntimeError")
        result = self.audit(trace)
        self.assertEqual(result["status"], "PASS", result["checks"])
        self.assertEqual(result["metrics"]["execution_status"], "ERROR")
        self.assertFalse(result["metrics"]["run_completed_normally"])
        self.assertEqual(result["metrics"]["verifier_errors_reported"], 1)

    def test_duplicate_ids_and_incomplete_envelopes_rejected(self):
        for mutation in ("event", "generation", "call", "missing_end"):
            trace = fixture()
            if mutation == "event":
                trace[2]["event_id"] = trace[1]["event_id"]
            elif mutation == "generation":
                duplicate = deepcopy(next(event for event in trace if event["kind"] == "generation"))
                duplicate["event_id"] = "extra"
                trace.insert(3, duplicate)
            elif mutation == "call":
                next(event for event in trace if event.get("call_id") == "v1")["call_id"] = "v0"
            else:
                trace.pop()
            with self.subTest(mutation=mutation):
                self.assertEqual(self.audit(trace)["status"], "ERROR")

    def test_candidate_deletion_and_changed_rule_are_rejected(self):
        for field in ("native_order", "applied_order"):
            trace = fixture()
            event = next(event for event in trace if event["kind"] == "generation")
            event[field] = event[field][:-1]
            with self.subTest(field=field):
                self.assertEqual(self.audit(trace)["status"], "ERROR")
        trace = fixture()
        for event in trace:
            if "applied_order" in event:
                event["applied_order"] = sorted(event["applied_order"])
        self.assertFalse(self.audit(trace)["checks"]["expected_candidate_order"])

    def test_rank_request_rng_and_base_contracts_are_rejected_when_changed(self):
        for mutation in ("rng", "request", "base", "mode", "memory"):
            trace = fixture()
            rank = next(event for event in trace if event["kind"] == "rank_decision")
            if mutation == "rng":
                rank["rng_after"] = "changed"
            elif mutation == "request":
                rank["request"]["candidate_ids"] = ["3"]
            elif mutation == "base":
                rank["request"]["base_part"] = None
            elif mutation == "mode":
                trace[0]["mode"] = "trace"
            else:
                rank["memory"] = {"mode": "off"}
            with self.subTest(mutation=mutation):
                self.assertEqual(self.audit(trace)["status"], "ERROR")

    def test_nonprefix_yield_or_mismatched_pose_part_state_is_rejected(self):
        for mutation in ("yield", "pose", "part", "state", "leaf"):
            trace = fixture()
            if mutation == "yield":
                next(event for event in trace if event["kind"] == "candidate_yield")["part"] = "2"
            elif mutation in ("pose", "part"):
                raw = next(event for event in trace if event["kind"] == "verifier_result")["raw"]
                raw["pose" if mutation == "pose" else "part_move"] = None
            elif mutation == "state":
                next(event for event in trace if event["kind"] == "verifier_start")["request"]["parts_removed"] = ["10"]
            else:
                next(event for event in trace if event["kind"] == "verifier_start")["generation_id"] = None
            with self.subTest(mutation=mutation):
                self.assertEqual(self.audit(trace)["status"], "ERROR")

    def test_unknown_is_not_relabelled_collision_and_error_requires_null_raw(self):
        for mutation in ("status", "blocker", "error_raw"):
            trace = fixture()
            event = next(event for event in trace if event["kind"] == "verifier_result")
            if mutation == "status":
                event["status"] = "COLLISION"
            elif mutation == "blocker":
                event["blocker"] = "2"
            else:
                event.update(status="ERROR", error_type="RuntimeError")
            with self.subTest(mutation=mutation):
                self.assertFalse(self.audit(trace)["checks"]["raw_verifier_results_preserved"])

    def test_result_without_start_is_an_integrity_failure(self):
        trace = fixture()
        trace.remove(next(event for event in trace if event["kind"] == "verifier_start"))
        self.assertFalse(self.audit(trace)["checks"]["serial_verifier_pairs"])

    def test_nonzero_llm_and_malformed_event_are_rejected_without_physics(self):
        trace = fixture()
        trace[0]["live_llm_calls"] = 1
        self.assertEqual(self.audit(trace)["status"], "ERROR")
        self.assertEqual(self.audit([None])["status"], "ERROR")

    def test_malformed_nested_fields_report_integrity_error_instead_of_crashing(self):
        for mutation in ("kind", "generation_id", "raw_request"):
            trace = fixture()
            if mutation == "kind":
                trace[1]["kind"] = []
            elif mutation == "generation_id":
                trace[1]["generation_id"] = []
            else:
                event = next(event for event in trace if event["kind"] == "verifier_start")
                event["request"]["max_grippers"] = "bad"
            with self.subTest(mutation=mutation):
                self.assertEqual(self.audit(trace)["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
