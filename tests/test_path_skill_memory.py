"""Synthetic artifact-contract tests only; no provider call or actual physics.

The fixture deliberately models recorded PASS fields without executing ASAP.
It tests registration gates, integrity and scoped retrieval, not physics truth.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from orchestrator import path_skill_memory as memory
from worker import constraint_memory
from worker.task_memory import canonical, digest
from test_constraint_memory import context, event, scope


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticRun:
    """All artifacts in this class are fabricated test fixtures, not runs."""

    def __init__(self, root):
        self.root = root
        self.run = root / "synthetic_source"
        self.replay_file = root / "synthetic_replay.json"
        self.output = root / "frozen.json"
        self.run_id = "synthetic-contract-only"
        self.context = context()
        controller_file = self.run / "implementation/worker/path_repair.py"
        controller_file.parent.mkdir(parents=True)
        controller_file.write_text("# Synthetic provenance bytes, never executed.\n", encoding="utf-8")
        self.code_sha = sha(controller_file)
        self.scope = scope()
        self.actions = [{"step": 0, "action": [1, 0, 0]},
                        {"step": 50, "action": [0, 0, 1]}]
        self.scope["action_schedule"] = deepcopy(self.actions)
        self.scope["physics"]["source_and_binary_hashes"] = {"controller_sha256": self.code_sha}
        self.failed_scope = deepcopy(self.scope)
        self.failed_scope["action_schedule"] = deepcopy(self.actions[:1])
        self.path = [[0, 0, 0, 0, 0, 0], [0.01, 0, 0, 0, 0, 0],
                     [0.01, 0, 0.02, 0, 0, 0]]
        controller = {"skill_id": "turn_force_two_leg_v1", "version": 1,
                      "reset_count": 1, "switch_count": 1, "intermediate_resets": 0,
                      "physics_steps": 100, "switch_q": self.path[1], "actions": self.actions}
        self.trajectory = {"raw_success": True, "path": self.path, "controller": controller,
                           "geometry": {"noncollinear": True}}
        write_json(self.run / "paths/success.json", self.trajectory)
        write_json(self.run / "paths/failure.json", {"raw_success": False, "path": self.path[:2]})
        failure = event(self.run_id + ":path1")
        failure["scope"] = deepcopy(self.failed_scope)
        failure["trajectory"].update(relative_path="paths/failure.json", states=2,
                                      sha256=sha(self.run / "paths/failure.json"))
        failure["terminal_q"] = self.path[1]
        success = event(self.run_id + ":path2")
        success.update(scope=deepcopy(self.scope), raw_success=True,
                       termination_reason="success", terminal_q=self.path[-1])
        success["trajectory"].update(relative_path="paths/success.json", states=3,
                                      sha256=sha(self.run / "paths/success.json"))
        success["measurements"]["pairs"][0].update(d_rest_move=0.6, d_move_rest=0.7,
                                                    min_distance=0.6, separation_unsatisfied=False)
        failure_file = self.run / "paths/failure_evidence.json"
        write_json(failure_file, failure)
        write_json(self.run / "paths/success_evidence.json", success)
        self.constraints = constraint_memory.records_from_evidence(
            self.context, [failure], str(failure_file), sha(failure_file))
        constraint_memory.save_snapshot(self.constraints, self.run / "constraint_memory.json")
        self.constraint_ids = [item["record_id"] for item in self.constraints]
        decision = {"skill_id": "turn_force_two_leg_v1", "second_axis": "+Z",
                    "switch_step": 50, "used_constraint_ids": self.constraint_ids}
        self.outcome = {"request_id": self.run_id + ":path_request1", "parent_call_id": "v1",
                        "feasible": True, "path_success": True, "stability_success": True,
                        "scope": self.scope, "controller": controller,
                        "geometry": deepcopy(self.trajectory["geometry"]), "parts_fix": [],
                        "trajectory_file": "paths/success.json",
                        "trajectory_sha256": sha(self.run / "paths/success.json"),
                        "failed_evidence_id": failure["event_id"], "repaired_evidence_id": success["event_id"],
                        "failed_path": "paths/failure.json", "decision": decision,
                        "legacy_single_action_replayer_supported": False}
        write_json(self.run / "path_outcomes.json", {"outcomes": [self.outcome]})
        sequence = ["a", "b"]
        write_json(self.run / "result.json", {"status": "PASS", "plan_success": True,
                    "run_id": self.run_id, "sequence": sequence,
                    "verifier_backend": "actual_ASAP_RedMax", "model_decision_fixture": False})
        write_json(self.run / "parsed_result.json", {"status": "PASS", "plan_success": True,
                    "validation_ok": True, "trusted_local_tree": True, "raw_stats": {"sequence": sequence}})
        write_json(self.run / "context.json", self.context)
        write_json(self.run / "events.jsonl", {"kind": "run_start", "run_id": self.run_id})
        write_json(self.run / "remote_audit.jsonl", {"request_id": self.outcome["request_id"],
                    "outcome": "accepted", "fallback_reason": None,
                    "response_metadata": {"llm_live": True, "attempted": True, "path_skill": decision}})
        self.tree = self.run / "native/dfs-independent/s0/fixture-a/tree.pkl"
        self.tree.parent.mkdir(parents=True)
        self.tree.write_bytes(b"Synthetic tree provenance only; not a pickle, never loaded.\n")
        check = {"status": "PASS", "selected_plan_edge": True, "path_success": True,
                 "stability_success": True, "support_set_equal": True,
                 "source_binary_controller_hashes_equal": True, "source_sdf_cache_hashes_equal": True,
                 "task_context_equal": True, "request_id": self.outcome["request_id"],
                 "plan_binding": {"sequence_index": 0, "parent": ["a", "b", "base"],
                                  "child": ["b", "base"], "parent_call_id": "v1",
                                  "selected_plan_edge": True, "extension_sha256": digest(self.outcome)},
                 "replayed_trajectory": deepcopy(self.trajectory),
                 "comparison": {"saved_path_hash": digest(self.path), "replayed_path_hash": digest(self.path),
                                "path_allclose": True, "path_shape_equal": True,
                                "start_matches_scoped_initial_state": True, "end_matches_measured_final_state": True,
                                "switch_state_present_in_path": True, "continuous_single_simulation": True,
                                "noncollinear": True, "switch_count": 1}}
        replay = {"status": "PASS", "actual_physics": True, "trusted_local_tree": True,
                  "input_files_unchanged": True, "sdf_cache_unchanged": True,
                  "source_and_binary_unchanged": True, "saved_outcomes_unchanged": True,
                  "saved_selected_plan_unchanged": True, "selected_sequence": sequence,
                  "source_outcomes_sha256": sha(self.run / "path_outcomes.json"),
                  "parsed_result_sha256": sha(self.run / "parsed_result.json"),
                  "tree_file_sha256": sha(self.tree), "replays": [check]}
        write_json(self.replay_file, replay)

    def freeze(self):
        return memory.freeze_validated_skills(self.run, self.replay_file, self.output)

    def current(self):
        return {"memory_context": deepcopy(self.context), "moving_part_id": "a",
                "registered_skill": {"code_sha256": self.code_sha},
                "constraint_references": [{"origin": "current_episode_measurement",
                                            "source_scope": deepcopy(self.failed_scope)}]}

    def mutate_replay(self, mutation):
        value = read_json(self.replay_file)
        mutation(value)
        write_json(self.replay_file, value)


class PathSkillMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_path_skill_", dir=ROOT / "runs")
        self.addCleanup(self.temp.cleanup)
        self.fixture = SyntheticRun(Path(self.temp.name))

    def test_freeze_load_references_only(self):
        with patch("pickle.load", side_effect=AssertionError("host must not unpickle")):
            snapshot = self.fixture.freeze()
            self.assertEqual(memory.load(self.fixture.output), snapshot)
        record = snapshot["records"][0]
        self.assertEqual(snapshot["record_count"], 1)
        self.assertEqual(record["failed_constraint_ids"], self.fixture.constraint_ids)
        self.assertEqual(record["validated_schedule"], self.fixture.actions)
        self.assertEqual(len(record["source_files"]), 14)
        self.assertEqual(record["lifecycle"], "validated_for_recorded_scope_only")
        self.assertTrue(record["force_new_physics"])
        for key in ("generalized", "hard_pruning", "reuse_verdict"):
            self.assertFalse(record[key])

    def test_same_conditions_still_require_new_physics(self):
        snapshot = self.fixture.freeze()
        cards = memory.matching_cards(snapshot, self.fixture.current())
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card["applicability"], "same_task_state_conditions_reference")
        self.assertEqual(card["condition_mismatches"], [])
        self.assertFalse(card["action_schedule_matches_current_observation"])
        self.assertFalse(card["current_attempt_verified"])
        self.assertTrue(card["reference_only"])
        self.assertTrue(card["force_new_physics"])
        self.assertNotIn("source_files", card)

    def test_full_scope_and_changed_conditions_are_explicit(self):
        snapshot = self.fixture.freeze()
        current = deepcopy(self.fixture.scope)
        card = memory.matching_cards(snapshot, current)[0]
        self.assertTrue(card["action_schedule_matches_current_observation"])
        self.assertFalse(card["reuse_verdict"])
        current["physics"]["force_mag"] = 50
        card = memory.matching_cards(snapshot, current)[0]
        self.assertEqual(card["condition_mismatches"], ["physics"])
        self.assertEqual(card["applicability"], "source_conditions_reference")

    def test_bare_task_context_is_incomplete_reference(self):
        card = memory.matching_cards(self.fixture.freeze(), self.fixture.context)[0]
        self.assertEqual(card["condition_mismatches"], ["current_scope_not_supplied", "current_controller_not_supplied"])
        self.assertIsNone(card["current_observed_action_schedule"])
        self.assertFalse(card["current_attempt_verified"])

    def test_different_task_controller_or_mover_not_reused(self):
        snapshot = self.fixture.freeze()
        changed = deepcopy(self.fixture.context)
        changed["task"]["input_sha256"]["a.obj"] = "9" * 64
        self.assertEqual(memory.matching_cards(snapshot, changed), [])
        changed = self.fixture.current()
        changed["registered_skill"]["code_sha256"] = "8" * 64
        self.assertEqual(memory.matching_cards(snapshot, changed), [])
        changed = deepcopy(self.fixture.scope)
        changed.update(part_move="b", parts_rest=["a", "base"])
        self.assertEqual(memory.matching_cards(snapshot, changed), [])

    def test_conflicting_current_scopes_rejected(self):
        snapshot = self.fixture.freeze()
        current = self.fixture.current()
        second = deepcopy(current["constraint_references"][0])
        second["source_scope"]["budget"] += 1
        current["constraint_references"].append(second)
        with self.assertRaisesRegex(ValueError, "conflicting_current_failure_scopes"):
            memory.matching_cards(snapshot, current)

    def test_scope_mover_contradiction_rejected(self):
        current = self.fixture.current()
        current["moving_part_id"] = "b"
        with self.assertRaisesRegex(ValueError, "current_skill_moving_part_mismatch"):
            memory.matching_cards(self.fixture.freeze(), current)

    def test_cards_are_deep_copies(self):
        snapshot = self.fixture.freeze()
        before = canonical(snapshot)
        card = memory.matching_cards(snapshot, self.fixture.current())[0]
        card["validated_schedule"][1]["action"][2] = -1
        card["source_scope"]["budget"] += 1
        self.assertEqual(canonical(snapshot), before)

    def test_existing_output_never_overwritten(self):
        self.fixture.output.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.fixture.freeze()
        self.assertEqual(self.fixture.output.read_text(), "preserve")

    def test_changed_source_detected_after_freeze(self):
        self.fixture.freeze()
        self.fixture.tree.write_bytes(b"changed fixture bytes")
        with self.assertRaisesRegex(ValueError, "frozen_skill_source_file_changed"):
            memory.load(self.fixture.output)

    def test_invalid_snapshot_and_record_hashes(self):
        snapshot = self.fixture.freeze()
        changed = deepcopy(snapshot)
        changed["records"][0]["source_run_id"] += "changed"
        write_json(self.fixture.output, changed)
        with self.assertRaises(ValueError):
            memory.load(self.fixture.output)
        changed["content_sha256"] = digest(changed["records"])
        write_json(self.fixture.output, changed)
        with self.assertRaisesRegex(ValueError, "skill_evidence_ID_mismatch"):
            memory.load(self.fixture.output)

    def test_generalization_pruning_and_cached_verdict_forbidden(self):
        snapshot = self.fixture.freeze()
        for key, value in (("generalized", True), ("hard_pruning", True),
                           ("reuse_verdict", True), ("force_new_physics", False)):
            changed = deepcopy(snapshot)
            changed["records"][0][key] = value
            changed["content_sha256"] = digest(changed["records"])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unverified_or_generalized_skill_claim"):
                memory.matching_cards(changed, self.fixture.current())

    def test_failed_mock_or_untrusted_source_not_registered(self):
        path = self.fixture.run / "result.json"
        original = read_json(path)
        for change in ({"status": "FAIL"}, {"plan_success": False},
                       {"verifier_backend": "mock"}, {"model_decision_fixture": True}):
            write_json(path, dict(original, **change))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "requires_actual_success"):
                self.fixture.freeze()
            self.assertFalse(self.fixture.output.exists())

    def test_all_replay_preservation_gates_required(self):
        original = read_json(self.fixture.replay_file)
        for key in ("input_files_unchanged", "sdf_cache_unchanged", "source_and_binary_unchanged",
                    "saved_outcomes_unchanged", "saved_selected_plan_unchanged"):
            write_json(self.fixture.replay_file, dict(original, **{key: False}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "did_not_preserve"):
                self.fixture.freeze()

    def test_all_actual_replay_conditions_required(self):
        original = read_json(self.fixture.replay_file)
        for key in ("path_success", "stability_success", "support_set_equal",
                    "source_binary_controller_hashes_equal", "source_sdf_cache_hashes_equal", "task_context_equal"):
            changed = deepcopy(original)
            changed["replays"][0][key] = False
            write_json(self.fixture.replay_file, changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "replay_not_fully_verified"):
                self.fixture.freeze()

    def test_off_plan_success_is_not_a_selected_skill(self):
        self.fixture.mutate_replay(lambda value: value["replays"][0].update(selected_plan_edge=False))
        with self.assertRaisesRegex(ValueError, "invalid_frozen_skill_snapshot"):
            self.fixture.freeze()

    def test_selected_edge_binding_mismatch_rejected(self):
        original = read_json(self.fixture.replay_file)
        for change in ({"parent_call_id": "v2"}, {"parent": ["a", "base"]},
                       {"child": ["a", "base"]}, {"extension_sha256": "0" * 64},
                       {"selected_plan_edge": False}):
            changed = deepcopy(original)
            changed["replays"][0]["plan_binding"].update(change)
            write_json(self.fixture.replay_file, changed)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "not_bound_to_selected_plan_edge"):
                self.fixture.freeze()

    def test_changed_tree_or_manifest_not_registered(self):
        self.fixture.tree.write_bytes(b"edited synthetic tree")
        with self.assertRaisesRegex(ValueError, "replay_bound_source_hash_mismatch"):
            self.fixture.freeze()

    def test_comparison_booleans_do_not_replace_path_comparison(self):
        self.fixture.mutate_replay(lambda value: value["replays"][0]["replayed_trajectory"]["path"][-1].__setitem__(2, 100))
        with self.assertRaisesRegex(ValueError, "fresh_replay_path_differs"):
            self.fixture.freeze()

    def test_replay_path_hash_and_all_comparison_gates_required(self):
        original = read_json(self.fixture.replay_file)
        for change in ({"saved_path_hash": "0" * 64}, {"replayed_path_hash": "1" * 64},
                       {"path_allclose": False}, {"start_matches_scoped_initial_state": False},
                       {"end_matches_measured_final_state": False}, {"switch_state_present_in_path": False},
                       {"continuous_single_simulation": False}, {"noncollinear": False}, {"switch_count": 0}):
            changed = deepcopy(original)
            changed["replays"][0]["comparison"].update(change)
            write_json(self.fixture.replay_file, changed)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "trajectory_comparison_not_verified"):
                self.fixture.freeze()

    def test_replay_must_execute_two_leg_switch(self):
        self.fixture.mutate_replay(lambda value: value["replays"][0]["replayed_trajectory"]["controller"].update(switch_count=0))
        with self.assertRaisesRegex(ValueError, "incomplete_two_leg_controller"):
            self.fixture.freeze()

    def test_changed_controller_code_not_registered(self):
        path = self.fixture.run / "implementation/worker/path_repair.py"
        path.write_text("# changed fixture", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "archived_controller_code_hash_mismatch"):
            self.fixture.freeze()

    def test_changed_successful_trajectory_not_registered(self):
        path = self.fixture.run / "paths/success.json"
        trajectory = read_json(path)
        trajectory["path"][-1][2] += 1
        write_json(path, trajectory)
        with self.assertRaisesRegex(ValueError, "successful_trajectory_hash_mismatch"):
            self.fixture.freeze()

    def test_changed_failed_trajectory_not_registered(self):
        write_json(self.fixture.run / "paths/failure.json", {"raw_success": False, "path": []})
        with self.assertRaisesRegex(ValueError, "measurement_trajectory_file_changed"):
            self.fixture.freeze()

    def test_changed_failure_evidence_not_registered(self):
        path = self.fixture.run / "paths/failure_evidence.json"
        value = read_json(path)
        value["measurements"]["measurement_seconds"] += 0.001
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "failed_constraint_provenance_mismatch"):
            self.fixture.freeze()

    def test_live_choice_must_be_unique_accepted_and_attempted(self):
        path = self.fixture.run / "remote_audit.jsonl"
        original = read_json(path)
        for change in ({"llm_live": False}, {"attempted": False}, {"path_skill": {}}):
            changed = deepcopy(original)
            changed["response_metadata"].update(change)
            write_json(path, changed)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "not_verified_live_choice"):
                self.fixture.freeze()
        path.write_text((canonical(original) + "\n") * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not_uniquely_accepted"):
            self.fixture.freeze()

    def test_missing_failure_constraints_rejected(self):
        # This rewrite is a synthetic fixture mutation, never a real snapshot.
        path = self.fixture.run / "constraint_memory.json"
        value = read_json(path)
        value.update(records=[], record_count=0, content_sha256=digest([]))
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "missing_measured_failure_constraints"):
            self.fixture.freeze()

    def test_forbidden_paths_are_rejected_before_filesystem_access(self):
        with patch.object(Path, "lstat", side_effect=AssertionError("no filesystem access")), \
             patch.object(Path, "open", side_effect=AssertionError("no filesystem access")):
            for path in ("E:/never-access", "/mnt/e/never-access", "C:/dev/../never-access"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    memory.load(path)

    def test_provenance_traversal_rejected(self):
        for value in ("../file", "/file", "C:/file", "E:/never-access", "paths//file"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                memory._relative(self.fixture.run, value)


if __name__ == "__main__":
    unittest.main()
