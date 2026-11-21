"""Synthetic contract tests; no ASAP physics and no provider/API calls."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from worker import constraint_memory as memory
from worker.task_memory import digest


def context():
    return {"schema_version": 2, "usage": "soft_ranking_only",
            "task": {"assembly_id": "fixture-a",
                     "input_sha256": {"a.obj": "a" * 64, "base.obj": "b" * 64},
                     "source_commit": "c" * 40, "base_part": "base",
                     "solver_settings": {"seed": 0, "budget": 100},
                     "goal": {"type": "native_full_disassembly"}},
            "initial_parts": ["a", "b", "base"],
            "geometry_features": {"a": {}, "b": {}, "base": {}}}


def scope():
    return {"task_context": context(), "assembled_parts": ["a", "b", "base"],
            "part_move": "a", "parts_rest": ["b", "base"], "parts_removed": [],
            "base_part": "base", "pose": None,
            "support_policy": {"kind": "native_support_search_after_path"},
            "action_schedule": [{"action": [1, 0, 0], "frame_count": 100}],
            "physics": {"min_sep": 0.5, "force_mag": 100,
                        "parts_removed_in_path_simulation": []},
            "solver_fingerprint": "d" * 64, "seed": 0, "budget": 100,
            "rng_before": "e" * 64, "initial_q": [0, 0, 0, 0, 0, 0],
            "collector_version": "endpoint_fixture_v1", "frame": "fixture_model_frame"}


def event(identity="fixture:m1"):
    return {"schema_version": 1, "event_id": identity,
            "kind": "path_endpoint_measurement", "scope": scope(),
            "raw_success": False,
            "trajectory": {"sha256": "f" * 64, "relative_path": "paths/m1.json",
                           "states": 101, "frame": "fixture_model_frame"},
            "terminal_q": [0.1, 0, 0, 0, 0, 0],
            "measurements": {"min_sep": 0.5, "pairs": [
                {"rest_part_id": "b", "moving_part_id": "a",
                 "d_rest_move": 0.2, "d_move_rest": 0.3, "min_distance": 0.2,
                 "separation_unsatisfied": True},
                {"rest_part_id": "base", "moving_part_id": "a",
                 "d_rest_move": 0.8, "d_move_rest": 0.9, "min_distance": 0.8,
                 "separation_unsatisfied": False}],
                "contact_history": [], "geometry_queries": 8, "measurement_seconds": 0.002},
            "blocker": None, "termination_reason": "unknown"}


def records(events=None):
    return memory.records_from_evidence(context(), [event()] if events is None else events,
                                        "fixture/measurements.jsonl", "1" * 64)


class ConstraintMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "constraints.json"

    def retrieve(self, frozen=None, current=None, request_scope=None, **kwargs):
        return (memory.FrozenConstraintMemory(records()) if frozen is None else frozen).retrieve(
            context() if current is None else current,
            kwargs.pop("assembled_parts", ["a", "b", "base"]),
            kwargs.pop("candidate_ids", ["a", "b"]), request_scope=request_scope, **kwargs)

    def test_constraint_is_only_measured_endpoint_predicate(self):
        value = records()[0]
        self.assertEqual(value["kind"], "required_final_separation_unsatisfied")
        self.assertEqual(value["verdict"], "REJECTED")
        self.assertEqual(value["verdict_scope"], "measured_endpoint_predicate_only")
        self.assertEqual(value["whole_part_verdict"], "UNKNOWN")
        self.assertIsNone(value["blocker"])
        for key in ("all_paths_impossible", "hard_pruning", "reuse_verdict"):
            self.assertFalse(value[key])
        self.assertEqual(value["predicate"]["rest_part_id"], "b")
        self.assertEqual(value["scope"], scope())
        self.assertEqual(value["scope_sha256"], digest(scope()))
        self.assertEqual(value["evidence_sha256"], digest(event()))

    def test_false_bool_old_observation_is_never_upgraded(self):
        for value in (False, {"feasible": False},
                      {"kind": "verifier_result", "status": "UNKNOWN", "raw": {"feasible": False}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                records([value])

    def test_successful_separation_yields_no_constraint(self):
        value = event()
        value["raw_success"] = True
        value["termination_reason"] = "success"
        value["measurements"]["pairs"][0].update(d_rest_move=0.5, d_move_rest=0.5,
                                                 min_distance=0.5, separation_unsatisfied=False)
        self.assertEqual(records([value]), [])

    def test_bare_failed_path_with_satisfied_endpoint_yields_no_constraint(self):
        value = event()
        value["measurements"]["pairs"][0].update(d_rest_move=0.5, d_move_rest=0.5,
                                                 min_distance=0.5, separation_unsatisfied=False)
        self.assertFalse(value["raw_success"])
        self.assertEqual(records([value]), [])

    def test_success_with_unsatisfied_endpoint_rejected(self):
        value = event()
        value["raw_success"] = True
        value["termination_reason"] = "success"
        with self.assertRaisesRegex(ValueError, "successful path contradicts"):
            records([value])

    def test_one_record_per_unsatisfied_pair(self):
        value = event()
        value["measurements"]["pairs"][1].update(d_rest_move=-0.1, d_move_rest=0.2,
                                                 min_distance=-0.1, separation_unsatisfied=True)
        actual = records([value])
        self.assertEqual(len(actual), 2)
        self.assertEqual({item["predicate"]["rest_part_id"] for item in actual}, {"base", "b"})
        self.assertEqual(len({item["record_id"] for item in actual}), 2)

    def test_directional_distances_and_threshold_must_agree(self):
        for change in ({"min_distance": 0.3}, {"separation_unsatisfied": False},
                       {"d_rest_move": "0.2"}, {"min_distance": True},
                       {"moving_part_id": "b"}, {"rest_part_id": "not_present"}):
            value = event()
            value["measurements"]["pairs"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                records([value])

    def test_nonfinite_measurements_and_poses_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            for mutate in (lambda x: x["measurements"].update(min_sep=bad),
                           lambda x: x["measurements"]["pairs"][0].update(min_distance=bad),
                           lambda x: x["terminal_q"].__setitem__(0, bad),
                           lambda x: x["scope"]["physics"].update(force_mag=bad)):
                value = event()
                mutate(value)
                with self.assertRaises(ValueError):
                    records([value])

    def test_missing_duplicate_and_extra_pair_rejected(self):
        cases = [event(), event(), event()]
        cases[0]["measurements"]["pairs"].pop()
        cases[1]["measurements"]["pairs"].append(deepcopy(cases[1]["measurements"]["pairs"][0]))
        cases[2]["measurements"]["pairs"][0]["claimed_blocker"] = "b"
        for value in cases:
            with self.assertRaises(ValueError):
                records([value])

    def test_measurement_counts_times_and_threshold_validated(self):
        for change in ({"geometry_queries": 3}, {"geometry_queries": 7}, {"geometry_queries": True},
                       {"measurement_seconds": -1}, {"min_sep": -0.1},
                       {"min_sep": 0.6}, {"contact_history": None},
                       {"contact_history": ["partb", "partb"]}):
            value = event()
            value["measurements"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                records([value])

    def test_no_inferred_blocker_or_termination_reason_allowed(self):
        for change in ({"blocker": "b"}, {"termination_reason": "collision"},
                       {"termination_reason": "success"},
                       {"raw_success": 0}, {"schema_version": True}, {"event_id": ""}):
            value = event()
            value.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                records([value])

    def test_scope_must_be_complete_and_match_context(self):
        for key in memory.SCOPE_KEYS:
            value = event()
            del value["scope"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                records([value])
        value = event()
        value["scope"]["task_context"]["task"]["assembly_id"] = "other"
        with self.assertRaisesRegex(ValueError, "context disagrees"):
            records([value])

    def test_invalid_state_and_solver_identity_rejected(self):
        for change in ({"assembled_parts": ["a", "base"]}, {"parts_rest": ["base"]},
                       {"parts_removed": ["b"]}, {"base_part": "b"},
                       {"part_move": "base"}, {"seed": True}, {"budget": 0},
                       {"rng_before": "bad"}, {"solver_fingerprint": "bad"},
                       {"support_policy": {}}, {"action_schedule": []},
                       {"physics": {}}, {"collector_version": ""},
                       {"initial_q": [0, 0]}, {"pose": [0, 0, 0]}):
            value = event()
            value["scope"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                records([value])

    def test_search_removed_parts_are_separate_from_path_simulation_parts(self):
        value = event()
        value["scope"].update(assembled_parts=["a", "base"], parts_rest=["base"], parts_removed=["b"])
        value["measurements"]["pairs"] = [value["measurements"]["pairs"][1]]
        value["measurements"]["pairs"][0].update(d_rest_move=0.1, d_move_rest=0.2,
                                                 min_distance=0.1, separation_unsatisfied=True)
        observed = records([value])[0]
        self.assertEqual(observed["scope"]["parts_removed"], ["b"])
        self.assertEqual(observed["scope"]["physics"]["parts_removed_in_path_simulation"], [])

    def test_real_collector_contract_accepts_detached_fixture_measurements(self):
        # The collector implementation is real, but this simulator is a test
        # fixture. Passing this test never establishes actual RedMax physics.
        from worker.path_evidence import collect_endpoint

        class Sim:
            def get_joint_qm(self, body):
                return [0.1, 0, 0, 0, 0, 0]

            def get_body_distance(self, left, right):
                return 0.2 if "partb" in (left, right) else 0.8

            def body_in_contact(self, left, right, threshold):
                return self.get_body_distance(left, right) < threshold

            def get_contact_bodies(self, body):
                return ["partb"]

        class Planner:
            sim = Sim()
            part_move, parts_fix, parts_removed = "a", ["b", "base"], []
            min_sep = 0.5
            _evidence_col_th = {"a": 0.01, "b": 0.01, "base": 0.01}

        declared = scope()
        declared.pop("frame")
        observed = collect_endpoint(Planner(), False, [[0.1, 0, 0, 0, 0, 0]],
                                    declared, "fixture:collector", "paths/collector.json", "f" * 64)
        normalized = records([observed])
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["scope"]["frame"], "world_translation_xyz_euler_xyz")
        self.assertEqual(normalized[0]["evidence"]["measurements"]["geometry_queries"], 8)

    def test_trajectory_and_source_pointers_validated_without_reads(self):
        with patch.object(Path, "open", side_effect=AssertionError("no pointer reads")):
            self.assertEqual(len(records()), 1)
        for bad in ("E:/forbidden.json", "../escape.json", "/absolute/path.json", "C:/absolute.json"):
            value = event()
            value["trajectory"]["relative_path"] = bad
            with patch.object(Path, "open", side_effect=AssertionError("no pointer reads")):
                with self.assertRaises(ValueError):
                    records([value])
        for change in ({"sha256": "bad"}, {"states": 0}, {"states": True}):
            value = event()
            value["trajectory"].update(change)
            with self.assertRaises(ValueError):
                records([value])
        with self.assertRaises(ValueError):
            memory.records_from_evidence(context(), [event()], "E:/forbidden.log", "1" * 64)
        with self.assertRaises(ValueError):
            memory.records_from_evidence(context(), [event()], "fixture/log.jsonl", "bad")

    def test_identical_duplicate_event_is_idempotent_and_conflict_fails(self):
        value = event()
        self.assertEqual(records([value, deepcopy(value)]), records([value]))
        conflict = deepcopy(value)
        conflict["terminal_q"][0] = 0.2
        with self.assertRaisesRegex(ValueError, "conflicting measurement event"):
            records([value, conflict])

    def test_mutation_of_inputs_and_outputs_cannot_change_frozen_memory(self):
        source_events = [event()]
        values = records(source_events)
        frozen = memory.FrozenConstraintMemory(values)
        source_events[0]["scope"]["action_schedule"].clear()
        values[0]["predicate"]["measured_min_distance"] = -100
        first = self.retrieve(frozen)
        first["records"][0]["source_scope"]["parts_rest"].clear()
        second = self.retrieve(frozen)
        self.assertEqual(second["records"][0]["observed_predicate"]["measured_min_distance"], 0.2)
        self.assertEqual(second["records"][0]["source_scope"]["parts_rest"], ["b", "base"])
        with self.assertRaises(FrozenInstanceError):
            frozen.content_sha256 = "mutated"

    def test_same_task_state_without_request_scope_is_reference_only(self):
        value = self.retrieve()["records"][0]
        self.assertEqual(value["applicability"], "same_task_state_reference")
        self.assertTrue(value["reference_only"])
        self.assertFalse(value["current_attempt_verified"])

    def test_exact_requires_whole_explicit_scope_and_still_cannot_reuse_verdict(self):
        value = self.retrieve(request_scope=scope())["records"][0]
        self.assertEqual(value["applicability"], "exact_scope_observation")
        self.assertFalse(value["reference_only"])
        self.assertFalse(value["reuse_verdict"])
        self.assertFalse(value["hard_pruning"])

    def test_pose_support_action_seed_budget_rng_and_solver_changes_not_exact(self):
        identity_pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        for change in ({"pose": identity_pose}, {"support_policy": "different_policy"},
                       {"action_schedule": [[0, 1, 0]]}, {"seed": 1}, {"budget": 200},
                       {"rng_before": "2" * 64}, {"solver_fingerprint": "3" * 64},
                       {"physics": {"min_sep": 0.6}}, {"collector_version": "v2"},
                       {"initial_q": [0.1, 0, 0, 0, 0, 0]}, {"frame": "other_frame"}):
            changed = scope()
            changed.update(change)
            value = self.retrieve(request_scope=changed)["records"][0]
            with self.subTest(change=change):
                self.assertEqual(value["applicability"], "same_task_state_reference")
                self.assertTrue(value["reference_only"])

    def test_other_mesh_or_task_context_is_source_reference_with_no_id_transfer(self):
        for change in ({"assembly_id": "other"},
                       {"input_sha256": {"a.obj": "4" * 64, "base.obj": "b" * 64}},
                       {"solver_settings": {"seed": 4, "budget": 5}},
                       {"goal": "selective_removal"}):
            current = context()
            current["task"].update(change)
            requested = scope()
            requested["task_context"] = current
            value = self.retrieve(current=current, request_scope=requested)["records"][0]
            with self.subTest(change=change):
                self.assertEqual(value["applicability"], "source_conditions_reference")
                self.assertIsNone(value["current_part_id_mapping"])
                self.assertEqual(value["source_scope"]["task_context"], context())

    def test_changed_remaining_state_is_source_reference(self):
        value = self.retrieve(assembled_parts=["a", "base"], candidate_ids=["a"])["records"][0]
        self.assertEqual(value["applicability"], "source_conditions_reference")

    def test_retrieval_requires_all_candidates_and_consistent_request_scope(self):
        for change in ({"candidate_ids": ["a"]}, {"candidate_ids": ["a", "a", "b"]},
                       {"candidate_ids": ["a", "b", "base"]},
                       {"assembled_parts": ["a", "b"]}, {"max_records": True},
                       {"max_records": -1}, {"max_records": 13}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.retrieve(**change)
        with self.assertRaises(ValueError):
            self.retrieve(assembled_parts=["a", "base"], candidate_ids=["a"], request_scope=scope())

    def test_empty_or_limited_memory_never_returns_candidate_order(self):
        empty = self.retrieve(memory.FrozenConstraintMemory([]))
        self.assertEqual(empty["records"], [])
        self.assertEqual(empty["record_ids"], [])
        for key in ("candidate_ids", "candidate_order", "recommended_order"):
            self.assertNotIn(key, empty)
        limited = self.retrieve(max_records=0)
        self.assertTrue(limited["truncated"])
        self.assertEqual(limited["records"], [])

    def test_record_provenance_or_claim_mutation_rejected(self):
        mutations = [lambda r: r.update(whole_part_verdict="REJECTED"),
                     lambda r: r.update(blocker="b"), lambda r: r.update(hard_pruning=True),
                     lambda r: r.update(reuse_verdict=True),
                     lambda r: r.update(scope_sha256="0" * 64),
                     lambda r: r.update(evidence_event_id="forged"),
                     lambda r: r["predicate"].update(measured_min_distance=-2),
                     lambda r: r["source"].update(sha256="invalid")]
        for mutate in mutations:
            value = records()
            mutate(value[0])
            with self.assertRaises(ValueError):
                memory.FrozenConstraintMemory(value)

    def test_frozen_file_is_exclusive_and_roundtrips(self):
        values = records()
        saved = memory.save_snapshot(values + deepcopy(values), self.path, allow_test_root=True)
        self.assertEqual(saved["record_count"], 1)
        before = self.path.read_bytes()
        self.assertEqual(memory.load_snapshot(self.path, allow_test_root=True), saved)
        with self.assertRaises(FileExistsError):
            memory.save_snapshot([], self.path, allow_test_root=True)
        self.assertEqual(before, self.path.read_bytes())

    def test_conflicting_record_id_rejected(self):
        first = records()[0]
        other = deepcopy(first)
        other["source"]["ref"] = "fixture/other_log.jsonl"
        with self.assertRaisesRegex(ValueError, "conflicting constraint record"):
            memory.FrozenConstraintMemory([first, other])

    def test_snapshot_count_policy_hash_and_record_tampering_rejected(self):
        original = memory.save_snapshot(records(), self.path, allow_test_root=True)
        for mutate in (lambda x: x.update(record_count=2),
                       lambda x: x.update(content_sha256="0" * 64),
                       lambda x: x.update(hard_pruning=True),
                       lambda x: x.update(reuse_verdict=True),
                       lambda x: x.update(policy="other"),
                       lambda x: x["records"][0].update(all_paths_impossible=True)):
            value = deepcopy(original)
            mutate(value)
            self.path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                memory.load_snapshot(self.path, allow_test_root=True)

    def test_forbidden_snapshot_path_rejected_before_filesystem_access(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("forbidden path inspected")):
            for path in ("E:/never_touch.json", "/mnt/e/never_touch.json", "../escaped.json"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    memory.save_snapshot([], Path(path), allow_test_root=True)


if __name__ == "__main__":
    unittest.main()
