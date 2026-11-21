"""Synthetic JSON contract tests; no API calls or ASAP simulation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "worker"))
import task_memory
from task_memory import (FrozenTaskMemory, canonical, digest, from_legacy_records,
                         load_snapshot, project_for_model, records_from_events,
                         save_snapshot, task_context)


def features(assembly="assembly-a"):
    return {"schema_version": 2,
            "task": {"assembly_id": assembly, "input_sha256": {"arm.obj": "a" * 64, "base.obj": "b" * 64},
                     "source_commit": "b" * 40, "base_part": "base",
                     "solver_settings": {"seed": 0, "budget": 100, "max_grippers": 2,
                                         "optimizer": "L-BFGS-B", "physics_binary_sha256": "c" * 64},
                     "goal": "full_disassembly"},
            "parts": {"base": {"center_mass": [0, 0, 0]},
                      "arm": {"center_mass": [1, 0, 0]},
                      "pin": {"center_mass": [0, 1, 0]}}}


def event_pair(tag="1", status="PASS", part="arm", assembled=None, pose=None):
    assembled = ["arm", "base", "pin"] if assembled is None else assembled
    request = {"part_move": part, "parts_rest": [p for p in assembled if p != part],
               "parts_removed": [p for p in ["arm", "base", "pin"] if p not in assembled],
               "pose": pose, "max_grippers": 2, "timeout": None,
               "optimizer": "L-BFGS-B", "debug": 1, "render": False}
    start = {"kind": "verifier_start", "run_id": "source", "call_id": "v" + tag,
             "event_id": "source:s" + tag, "request": request}
    raw = {"feasible": status == "PASS", "part_move": part, "base_part": "base",
           "action": [0, 0, 1] if status == "PASS" else None,
           "parts_fix": ["base"] if status == "PASS" else None, "pose": pose}
    result = {"kind": "verifier_result", "run_id": "source", "call_id": "v" + tag,
              "event_id": "source:r" + tag, "status": status,
              "raw": None if status == "ERROR" else raw, "blocker": None, "path": None}
    if status == "ERROR":
        result["error_type"] = "RuntimeError"
    return start, result


def record(context=None, **kwargs):
    return records_from_events(task_context(features()) if context is None else context,
                               event_pair(**kwargs), "source/events.jsonl", "d" * 64)[0]


class TaskMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = task_context(features())

    def freeze(self, records):
        path = self.root / "frozen.json"
        save_snapshot(records, path, allow_test_root=True)
        return FrozenTaskMemory(path, allow_test_root=True), path

    def retrieve(self, memory, context=None, **kwargs):
        return memory.retrieve(self.context if context is None else context,
                               kwargs.pop("assembled_parts", ["arm", "base", "pin"]),
                               kwargs.pop("candidate_ids", ["arm", "pin"]), **kwargs)

    def test_context_copies_complete_geometry_and_solver_identity(self):
        source = features()
        context = task_context(source)
        source["task"]["solver_settings"]["seed"] = 7
        source["parts"].clear()
        self.assertEqual(context["initial_parts"], ["arm", "base", "pin"])
        self.assertEqual(context["task"]["solver_settings"]["seed"], 0)
        self.assertEqual(len(context["geometry_features"]), 3)

    def test_input_file_hash_map_and_aggregate_hash_are_supported(self):
        value = features()
        self.assertEqual(len(task_context(value)["task"]["input_sha256"]), 2)
        value["task"]["input_sha256"] = "a" * 64
        self.assertEqual(task_context(value)["task"]["input_sha256"], "a" * 64)
        for invalid in ({}, {"part.obj": "not-a-hash"}, {"part.obj": None}, "bad"):
            value["task"]["input_sha256"] = invalid
            with self.assertRaises(ValueError):
                task_context(value)

    def test_missing_identity_invalid_base_and_nonfinite_geometry_rejected(self):
        cases = []
        for field in ("input_sha256", "source_commit", "solver_settings", "goal"):
            value = features()
            del value["task"][field]
            cases.append(value)
        value = features()
        value["task"]["base_part"] = "missing"
        cases.append(value)
        value = features()
        value["parts"]["arm"]["center_mass"] = [float("nan"), 0, 0]
        cases.append(value)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                task_context(case)

    def test_free_base_is_supported(self):
        value = features()
        value["task"]["base_part"] = None
        context = task_context(value)
        start, result = event_pair()
        result["raw"]["base_part"] = None
        records = records_from_events(context, [start, result], "events.jsonl", "d" * 64)
        memory, _ = self.freeze(records)
        found = self.retrieve(memory, context)
        self.assertEqual(len(found["exact_observations"]), 1)

    def test_pass_unknown_error_preserve_conditional_raw_evidence(self):
        events = []
        for i, status in enumerate(("PASS", "UNKNOWN", "ERROR")):
            start, result = event_pair(str(i), status)
            if status == "UNKNOWN":
                result["stage_feedback"] = {"path_found": True, "support_search_succeeded": False}
            events.extend([start, result])
        records = records_from_events(self.context, events, "events.jsonl", "d" * 64)
        self.assertEqual([r["status"] for r in records], ["PASS", "UNKNOWN", "ERROR"])
        self.assertEqual(records[1]["stage_feedback"], events[3]["stage_feedback"])
        for value in records:
            self.assertEqual(value["constraints"], [])
            self.assertFalse(value["allow_hard_pruning"])
            self.assertIsNone(value["blocker"])
        self.assertFalse(records[1]["raw_result"]["feasible"])
        self.assertIsNone(records[2]["raw_result"])
        self.assertEqual(records[0]["support"], ["base"])

    def test_duplicate_events_idempotent_conflicts_fail(self):
        start, result = event_pair()
        self.assertEqual(len(records_from_events(self.context, [start, start, result, result],
                                                "events.jsonl", "d" * 64)), 1)
        conflict = deepcopy(result)
        conflict["status"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "conflicting event"):
            records_from_events(self.context, [start, result, conflict], "events.jsonl", "d" * 64)

    def test_unpaired_duplicate_call_and_wrong_run_are_rejected(self):
        start, result = event_pair()
        other = deepcopy(start)
        other["event_id"] = "another"
        wrong = deepcopy(result)
        wrong["run_id"] = "other"
        for events in ([start], [result, start], [start, other, result], [start, wrong]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                records_from_events(self.context, events, "events.jsonl", "d" * 64)

    def test_status_pose_support_and_membership_conflicts_fail(self):
        for mutate in (lambda a, b: b.update(status="UNKNOWN"),
                       lambda a, b: b["raw"].update(pose=[1, 2, 3]),
                       lambda a, b: b["raw"].update(parts_fix=["missing"]),
                       lambda a, b: a["request"].update(parts_removed=["pin"]),
                       lambda a, b: b.update(blocker="pin"),
                       lambda a, b: b["raw"].update(action=[0, 0, 0])):
            start, result = event_pair()
            mutate(start, result)
            with self.assertRaises(ValueError):
                records_from_events(self.context, [start, result], "events.jsonl", "d" * 64)

    def test_source_ref_is_pointer_never_opened(self):
        with patch.object(Path, "open", side_effect=AssertionError("no reads allowed")):
            records = records_from_events(self.context, event_pair(), "missing/log.jsonl", "d" * 64)
        self.assertEqual(records[0]["source"]["ref"], "missing/log.jsonl")

    def test_input_mutation_does_not_change_records(self):
        start, result = event_pair()
        records = records_from_events(self.context, [start, result], "events.jsonl", "d" * 64)
        start["request"]["parts_rest"].clear()
        result["raw"]["action"].clear()
        self.context["task"]["solver_settings"]["seed"] = 4
        self.assertEqual(records[0]["action"], [0, 0, 1])
        self.assertEqual(records[0]["request"]["parts_rest"], ["base", "pin"])
        self.assertEqual(records[0]["context"]["task"]["solver_settings"]["seed"], 0)

    def test_snapshot_is_exclusive_and_identical_records_are_deduplicated(self):
        memory, path = self.freeze([record(), record()])
        self.assertEqual(len(memory._records), 1)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_snapshot([], path, allow_test_root=True)
        self.assertEqual(path.read_bytes(), before)

    def test_snapshot_rejects_conflicting_ids(self):
        first, conflict = record(), record()
        conflict["source"]["ref"] = "other/events.jsonl"
        with self.assertRaisesRegex(ValueError, "conflicting observation"):
            self.freeze([first, conflict])
        self.assertFalse((self.root / "frozen.json").exists())

    def test_snapshot_tampering_hash_and_duplicate_ids_fail(self):
        _, path = self.freeze([record()])
        original = json.loads(path.read_text(encoding="utf-8"))
        for operation in (lambda x: x.update(content_sha256="0" * 64),
                          lambda x: x.update(record_count=9),
                          lambda x: x["records"].append(deepcopy(x["records"][0])),
                          lambda x: x.update(allow_hard_pruning=True)):
            value = deepcopy(original)
            operation(value)
            test_path = self.root / (str(len(list(self.root.iterdir()))) + ".json")
            test_path.write_text(canonical(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_snapshot(test_path, allow_test_root=True)

    def test_evaluation_retrieval_does_not_write_or_mutate_frozen_memory(self):
        source = record()
        memory, path = self.freeze([source])
        frozen_bytes = path.read_bytes()
        source["status"] = "UNKNOWN"
        first = self.retrieve(memory)
        first["exact_observations"][0]["source_context"]["task"]["goal"] = "changed"
        with patch.object(Path, "open", side_effect=AssertionError("retrieval cannot reopen file")):
            second = self.retrieve(memory)
        self.assertEqual(second["exact_observations"][0]["source_context"]["task"]["goal"], "full_disassembly")
        self.assertEqual(path.read_bytes(), frozen_bytes)
        with self.assertRaises(FrozenInstanceError):
            memory.content_sha256 = "changed"

    def test_stale_mesh_goal_seed_budget_support_are_not_exact(self):
        memory, _ = self.freeze([record()])
        changes = [("input_sha256", "f" * 64), ("assembly_id", "other"), ("goal", "recover_pin")]
        for field, value in changes:
            context = deepcopy(self.context)
            context["task"][field] = value
            self.assert_reference_only(self.retrieve(memory, context))
        for field, value in (("seed", 2), ("budget", 10), ("max_grippers", 3),
                             ("physics_binary_sha256", "f" * 64)):
            context = deepcopy(self.context)
            context["task"]["solver_settings"][field] = value
            self.assert_reference_only(self.retrieve(memory, context))

    def assert_reference_only(self, result):
        self.assertEqual(result["exact_observations"], [])
        self.assertEqual(len(result["cross_task_references"]), 1)
        reference = result["cross_task_references"][0]
        self.assertTrue(reference["reference_only"])
        self.assertIsNone(reference["current_part_id_mapping"])
        self.assertFalse(reference["current_attempt_verified"])
        self.assertEqual(reference["constraints"], [])
        self.assertFalse(result["physics_reuse_allowed"])

    def test_same_state_pass_is_observation_not_physics_cache_or_constraint(self):
        memory, _ = self.freeze([record()])
        result = self.retrieve(memory)
        self.assertEqual(len(result["exact_observations"]), 1)
        view = result["exact_observations"][0]
        self.assertFalse(view["current_attempt_verified"])
        self.assertFalse(result["physics_reuse_allowed"])
        self.assertEqual(result["constraints"], [])
        self.assertNotIn("positive_candidate_ids", result)

    def test_different_state_or_candidate_stays_reference_only(self):
        memory, _ = self.freeze([record()])
        self.assert_reference_only(self.retrieve(memory, candidate_ids=["pin"]))
        self.assert_reference_only(self.retrieve(memory, assembled_parts=["base", "pin"], candidate_ids=["pin"]))

    def test_unknown_and_error_never_remove_candidates(self):
        memory, _ = self.freeze([record(status="UNKNOWN"), record(tag="2", status="ERROR", part="pin")])
        result = self.retrieve(memory)
        self.assertEqual(result["current_candidate_ids"], ["arm", "pin"])
        self.assertEqual(result["status_counts"], {"PASS": 0, "UNKNOWN": 1, "ERROR": 1})
        self.assertEqual(result["constraints"], [])
        self.assertFalse(result["allow_hard_pruning"])

    def test_model_projection_excludes_recommendations_raw_extra_and_full_geometry(self):
        start, result = event_pair()
        result["raw"]["recommendation"] = "DO_NOT_INCLUDE_RAW_RECOMMENDATION"
        result["raw"]["preferred_order"] = ["pin", "arm"]
        result["stage_feedback"] = {"path_found": True, "reason": "UNVERIFIED_REASON",
                                    "preferred_order": ["pin"], "stability_passed": "invented"}
        records = records_from_events(self.context, [start, result], "events.jsonl", "d" * 64)
        memory, _ = self.freeze(records)
        projected = project_for_model(self.retrieve(memory))
        serialized = canonical(projected)
        for forbidden in ("DO_NOT_INCLUDE_RAW_RECOMMENDATION", "preferred_order", "UNVERIFIED_REASON",
                          "geometry_features", "center_mass", "invented"):
            self.assertNotIn(forbidden, serialized)
        observed = projected["exact_observations"][0]
        self.assertEqual(observed["source_part_id"], "arm")
        self.assertEqual(observed["source_action"], [0, 0, 1])
        self.assertEqual(observed["stage_feedback"], {"path_found": True})
        self.assertFalse(projected["physics_reuse_allowed"])

    def test_model_projection_cross_task_retains_source_without_mapping(self):
        memory, _ = self.freeze([record()])
        other = task_context(features("other"))
        projected = project_for_model(self.retrieve(memory, other))
        observation = projected["cross_task_references"][0]
        self.assertEqual(observation["source_identity"]["assembly_id"], "assembly-a")
        self.assertTrue(observation["reference_only"])
        self.assertIsNone(observation["current_part_id_mapping"])
        self.assertEqual(projected["exact_observations"], [])

    def test_exact_records_have_priority_and_shared_limit(self):
        foreign = task_context(features("foreign"))
        memory, _ = self.freeze([record(foreign, tag="0"), record(tag="1"), record(tag="2", status="UNKNOWN")])
        result = self.retrieve(memory, max_records=1)
        self.assertEqual(result["evidence_ids"], ["source:r1"])
        self.assertEqual(result["cross_task_references"], [])
        self.assertTrue(result["truncated"])
        self.assertEqual(self.retrieve(memory, max_records=0)["evidence_ids"], [])

    def test_invalid_candidates_and_limit_rejected(self):
        memory, _ = self.freeze([])
        for kwargs in ({"candidate_ids": ["arm", "arm"]}, {"candidate_ids": ["base"]},
                       {"candidate_ids": ["foreign"]}, {"max_records": True},
                       {"max_records": -1}, {"max_records": 101},
                       {"assembled_parts": ["arm", "pin"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.retrieve(memory, **kwargs)

    def test_legacy_import_preserves_original_beam_scope_and_never_maps_ids(self):
        from evidence_memory import beam_context, record_from_pair
        context = beam_context({"asset_hash": "old"}, {"optimizer": "L-BFGS-B"})
        start, result = event_pair(part="1")
        start["request"].update(parts_rest=["0", "2", "3", "6"], parts_removed=[])
        result["raw"].update(base_part="6", parts_fix=["6"])
        legacy = record_from_pair(context, start, result, "old/events.jsonl", "e" * 64)
        imported = from_legacy_records([legacy])
        self.assertEqual(imported[0]["legacy_record"], legacy)
        self.assertEqual(imported[0]["context"]["scope"], "official_beam_original")
        memory, _ = self.freeze(imported)
        result = self.retrieve(memory)
        self.assert_reference_only(result)
        self.assertEqual(result["cross_task_references"][0]["source_request"]["part_move"], "1")

    def test_non_C_and_E_paths_rejected_before_filesystem_calls(self):
        for name in ("E:/unused/snapshot.json", "e:\\unused\\snapshot.json", "/mnt/e/unused/snapshot.json",
                     "D:/snapshot.json", "\\\\server\\share\\snapshot.json"):
            with self.subTest(name=name), patch.object(Path, "lstat", side_effect=AssertionError("filesystem touched")), \
                    patch.object(Path, "open", side_effect=AssertionError("filesystem touched")):
                with self.assertRaises(ValueError):
                    load_snapshot(name, allow_test_root=True)
                with self.assertRaises(ValueError):
                    records_from_events(self.context, event_pair(), name, "d" * 64)

    def test_default_path_guard_rejects_test_temp_outside_runs(self):
        with self.assertRaises(ValueError):
            save_snapshot([], self.root / "outside.json")


if __name__ == "__main__":
    unittest.main()
