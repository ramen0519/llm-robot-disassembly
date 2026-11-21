"""Offline sequence knowledge contracts; fixtures are not physics/model results."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_explicit_action_contract import request
from worker.explicit_action_contract import action_scope, canonical_response, digest
from orchestrator.explicit_action_memory import FrozenActionMemory
from orchestrator.sequence_knowledge import (build_sequence_knowledge, execute_sequence_skill,
    make_fixture_sequence_skill, validate_sequence_skill, build_native_snapshot, validate_native_snapshot)
from orchestrator.lesson_memory import make_lesson, sign


REVERSE = "def propose(ctx):\n    return {'candidate_order': ctx['candidate_ids'][::-1], 'evidence_ids': []}\n"


def envelope(task="target"):
    return dict(request=dict(candidate_ids=["1", "2"], state_id="state", request_id="request"),
        features=dict(cad=dict(task=dict(assembly_id=task, input_sha256="b" * 64),
            parts={"1": dict(volume=4, extents=[1, 2, 2]), "2": dict(volume=2, extents=[1, 1, 2])})))


def record(task="source", status="PASS", request_id="fixture:r1", execution=None):
    req = request()
    req["task_context"]["assembly_id"] = task
    req["request_id"] = request_id
    response = canonical_response(req, None)
    result = dict(status=status, path_success=status == "PASS",
        stability_success=True if status == "PASS" else None,
        verifier_origin="fixture_not_actual_physics", raw={}, evidence_ref="fixture/events.jsonl#1")
    if execution is not None:
        result["execution"] = execution
    row = dict(schema_version=1, protocol=req["protocol"], source_run_id=req["run_id"],
        source_request_id=req["request_id"], source_state_id=req["state_id"],
        source_action_id=response["action_id"], scope=action_scope(req, response["choice"]),
        choice=response["choice"], result=result, parent_failure_evidence_id=None,
        created_utc="fixture")
    row["evidence_id"] = "action-evidence:" + digest({k: row[k] for k in
        ("source_run_id", "source_request_id", "source_action_id")})
    row["record_sha256"] = digest(row)
    return row


def snapshot(rows):
    return dict(schema_version=1, protocol="explicit_action_v1", records=rows,
        record_count=len(rows), content_sha256=digest(rows), hard_pruning=False,
        physics_reuse_allowed=False, frozen_utc="fixture")


def lessons(task="source", source_as_name=False):
    evidence_id = "fixture:lesson_fact"
    packet = dict(evidence=[dict(evidence_id=evidence_id,
        source_task=task if source_as_name else dict(assembly_id=task, input_sha256="a" * 64),
        result=dict(status="UNKNOWN", path_success=False, stability_success=None))])
    claim = dict(title="Fixture hypothesis", hypothesis="A hypothesis, not a constraint.",
        when_to_apply="Inspect current parts first.", when_not_to_apply="Unknown task roles.",
        evidence_ids=[evidence_id], counterexample_ids=[], roles=["source role"], trigger_kind="path_failed")
    lesson = make_lesson(claim, packet, dict(origin="injected_transport"))
    return sign(dict(schema_version=1, kind="frozen_lessons_v1", entries=[dict(lesson=lesson)],
                     hard_pruning=False, physics_reuse_allowed=False))


def native_attempt(*, status="PASS", accepted=True, deadline=False):
    return dict(event_id="fixture-native:attempt:1", part_move="1", remaining_parts=["1", "2"],
        request=dict(part_move="1", parts_rest=["2"], parts_removed=[], pose=None,
                     max_grippers=3, timeout=10), returned=True, status=status,
        accepted_for_transition=accepted, deadline_exceeded=deadline, verifier_domain="ASAP_RedMax",
        raw_sim_info=dict(feasible=status == "PASS", action=[1, 0, 0], parts_fix=["2"]),
        source_hashes=dict(commit="fixture"), selection_scope="ranked_candidate")


class SequenceNativeMemoryTests(unittest.TestCase):
    def freeze(self, attempt=None, task="source"):
        return build_native_snapshot(dict(task=dict(assembly_id=task, input_sha256="c" * 64,
            solver_settings=dict(seed=0, fixture=True))), [attempt or native_attempt()],
            "fixture/worker_result.json", "d" * 64)

    def test_native_final_attempt_preserved_and_readable_next_task(self):
        frozen = self.freeze()
        original = deepcopy(frozen)
        self.assertEqual(validate_native_snapshot(frozen), frozen)
        built = build_sequence_knowledge(envelope(), frozen)
        ref = built["context"]["memory_references"][0]
        self.assertEqual(ref["source_protocol"], "sequence_native_observation_v1")
        self.assertTrue(ref["path_success"])
        self.assertTrue(ref["stability_success"])
        self.assertTrue(ref["accepted_for_transition"])
        self.assertEqual(ref["source_choice"]["native_support_ids"], ["2"])
        self.assertIsNone(ref["current_part_id_mapping"])
        self.assertFalse(ref["physics_reuse_allowed"])
        self.assertEqual(frozen, original)

    def test_late_native_pass_not_accepted(self):
        frozen = self.freeze(native_attempt(accepted=False, deadline=True))
        ref = build_sequence_knowledge(envelope(), frozen)["context"]["memory_references"][0]
        self.assertEqual(ref["status"], "PASS")
        self.assertFalse(ref["accepted_for_transition"])
        self.assertTrue(ref["deadline_exceeded"])

    def test_native_unknown_acceptance_stays_unknown(self):
        frozen = self.freeze(native_attempt(accepted=None, deadline=None))
        ref = build_sequence_knowledge(envelope(), frozen)["context"]["memory_references"][0]
        self.assertIsNone(ref["accepted_for_transition"])
        self.assertIsNone(ref["deadline_exceeded"])

    def test_native_failed_path_not_collision(self):
        attempt = native_attempt(status="UNKNOWN", accepted=False)
        attempt["raw_sim_info"]["action"] = None
        ref = build_sequence_knowledge(envelope(), self.freeze(attempt))["context"]["memory_references"][0]
        self.assertFalse(ref["path_success"])
        self.assertIsNone(ref["stability_success"])
        self.assertIsNone(ref["failure_cause"])
        self.assertIsNone(ref["blocker"])

    def test_native_exception_does_not_imply_path_failed(self):
        attempt = native_attempt(status="ERROR", accepted=False)
        attempt.update(returned=False)
        del attempt["raw_sim_info"]
        ref = build_sequence_knowledge(envelope(), self.freeze(attempt))["context"]["memory_references"][0]
        self.assertIsNone(ref["path_success"])
        self.assertIsNone(ref["stability_success"])

    def test_native_and_legacy_snapshots_combined_without_retyping(self):
        built = build_sequence_knowledge(envelope(), [self.freeze(), snapshot([record()])])
        self.assertEqual(len(built["context"]["memory_references"]), 2)

    def test_native_target_excluded_by_default(self):
        built = build_sequence_knowledge(envelope(), self.freeze(task="00335"))
        self.assertEqual(built["context"]["memory_references"], [])
        self.assertEqual(built["audit"]["excluded_memory"][0]["reason"], "evaluation_task_excluded")

    def test_native_tamper_rejected(self):
        frozen = self.freeze()
        frozen["records"][0]["attempt"]["accepted_for_transition"] = False
        with self.assertRaises(ValueError):
            validate_native_snapshot(frozen)

    def test_native_late_acceptance_rejected(self):
        with self.assertRaisesRegex(ValueError, "timely_pass"):
            self.freeze(native_attempt(accepted=True, deadline=True))

    def test_native_per_file_input_hashes_supported(self):
        frozen = build_native_snapshot(dict(task=dict(assembly_id="source",
            input_sha256={"0.obj": "a" * 64, "1.obj": "b" * 64})),
            [native_attempt()], "fixture/worker_result.json", "d" * 64)
        self.assertEqual(validate_native_snapshot(frozen), frozen)


class SequenceKnowledgeTests(unittest.TestCase):
    def test_memory_preserves_scope_and_unknown_acceptance(self):
        source = record()
        built = build_sequence_knowledge(envelope(), [source])
        ref = built["context"]["memory_references"][0]
        self.assertEqual(ref["source_task_context"], source["scope"]["task_context"])
        self.assertEqual(ref["source_choice"], source["choice"])
        self.assertEqual(ref["source_pose_matrix"], source["scope"]["pose_matrix"])
        self.assertIsNone(ref["accepted_for_transition"])
        self.assertIsNone(ref["deadline_exceeded"])
        self.assertIsNone(ref["current_part_id_mapping"])
        self.assertIsNone(ref["failure_cause"])
        self.assertFalse(ref["current_attempt_verified"])
        self.assertTrue(ref["reference_only"])
        self.assertEqual(built["audit"]["missing_acceptance_ids"], [source["evidence_id"]])

    def test_actual_acceptance_is_preserved_but_not_current_pass(self):
        execution = dict(version="planner_execution_v1", source="planner_attempt",
            accepted_for_transition=False, deadline_exceeded=True, attempt_elapsed_seconds=3,
            physics_elapsed_seconds=2, remaining_wall_seconds=0,
            allowance_source="context.time_remaining_seconds")
        ref = build_sequence_knowledge(envelope(), [record(execution=execution)])["context"]["memory_references"][0]
        self.assertEqual(ref["status"], "PASS")
        self.assertFalse(ref["accepted_for_transition"])
        self.assertTrue(ref["deadline_exceeded"])
        self.assertFalse(ref["physics_reuse_allowed"])

    def test_no_pruning_or_exact_pass_reuse(self):
        rows = [record(status="UNKNOWN"), record(request_id="fixture:r2")]
        context = build_sequence_knowledge(envelope(), rows)["context"]
        self.assertEqual(context["candidate_ids"], ["1", "2"])
        self.assertEqual({r["status"] for r in context["memory_references"]}, {"PASS", "UNKNOWN"})
        self.assertFalse(context["hard_pruning"])
        self.assertFalse(context["physics_reuse_allowed"])
        self.assertTrue(all(not r["conditions_equivalent"] for r in context["memory_references"]))

    def test_default_excludes_00335_and_current_task(self):
        rows = [record("00335"), record("target"), record("source")]
        result = build_sequence_knowledge(envelope(), rows)
        self.assertEqual(len(result["context"]["memory_references"]), 1)
        self.assertEqual(len(result["audit"]["excluded_memory"]), 2)

    def test_sources_are_unchanged(self):
        e, rows = envelope(), [record()]
        before = deepcopy((e, rows))
        built = build_sequence_knowledge(e, rows)
        built["context"]["memory_references"][0]["source_choice"]["part_id"] = "changed"
        self.assertEqual((e, rows), before)

    def test_memory_off_does_not_load_data(self):
        context = build_sequence_knowledge(envelope(), {"broken": "ignored"},
            limits=dict(memory_enabled=False), lessons={"broken": "ignored"})["context"]
        self.assertEqual(context["memory_references"], [])
        self.assertEqual(context["lesson_references"], [])
        self.assertEqual(context["evidence_ids"], [])

    def test_snapshot_and_memory_instance_supported(self):
        frozen = snapshot([record()])
        expected = build_sequence_knowledge(envelope(), frozen)
        self.assertEqual(expected, build_sequence_knowledge(envelope(), FrozenActionMemory(frozen)))

    def test_tampered_memory_rejected(self):
        rows = [record()]
        rows[0]["result"]["raw"]["invented"] = "collision"
        with self.assertRaisesRegex(ValueError, "hash"):
            build_sequence_knowledge(envelope(), rows)

    def test_memory_budget_is_reported(self):
        built = build_sequence_knowledge(envelope(), [record(), record(request_id="fixture:r2")],
                                         limits=dict(max_records=1))
        self.assertEqual(len(built["context"]["memory_references"]), 1)
        self.assertEqual(built["audit"]["excluded_memory"][0]["reason"], "record_limit")

    def test_citation_bytes_count_before_whole_memory_object_packing(self):
        built = build_sequence_knowledge(envelope(), [record()])
        full_size = built["audit"]["context_bytes"]
        without_ids = deepcopy(built["context"])
        without_ids["evidence_ids"] = []
        uncounted_size = len(json.dumps(without_ids, ensure_ascii=False, allow_nan=False,
            separators=(",", ":")).encode("utf-8"))
        self.assertLess(uncounted_size, full_size)
        # The prior algorithm accepted this object then failed when IDs were
        # appended. The corrected algorithm removes the whole reference.
        limit = max(1024, full_size - 1)
        trimmed = build_sequence_knowledge(envelope(), [record()], limits=dict(max_context_bytes=limit))
        self.assertEqual(trimmed["context"]["memory_references"], [])
        self.assertEqual(trimmed["context"]["evidence_ids"], [])
        self.assertLessEqual(trimmed["audit"]["context_bytes"], limit)
        self.assertEqual(trimmed["audit"]["excluded_memory"][-1]["reason"], "context_byte_limit")

    def test_exact_citation_inclusive_byte_limit_keeps_reference(self):
        first = build_sequence_knowledge(envelope(), [record()])
        built = build_sequence_knowledge(envelope(), [record()],
            limits=dict(max_context_bytes=first["audit"]["context_bytes"]))
        self.assertEqual(built["context"], first["context"])

    def test_removing_lesson_also_removes_its_citation(self):
        first = build_sequence_knowledge(envelope(), lessons=lessons())
        limit = max(1024, first["audit"]["context_bytes"] - 1)
        built = build_sequence_knowledge(envelope(), lessons=lessons(), limits=dict(max_context_bytes=limit))
        self.assertEqual(built["context"]["lesson_references"], [])
        self.assertEqual(built["context"]["evidence_ids"], [])
        self.assertLessEqual(built["audit"]["context_bytes"], limit)

    def test_byte_budget_trims_memory_before_lesson_and_skill(self):
        card = make_fixture_sequence_skill(REVERSE)
        full = build_sequence_knowledge(envelope(), [record()], [card],
            limits=dict(allow_fixture_skills=True), lessons=lessons())
        limit = full["audit"]["context_bytes"] - 1
        built = build_sequence_knowledge(envelope(), [record()], [card],
            limits=dict(allow_fixture_skills=True, max_context_bytes=limit), lessons=lessons())
        self.assertEqual(built["context"]["memory_references"], [])
        self.assertEqual(len(built["context"]["lesson_references"]), 1)
        self.assertEqual(len(built["context"]["sequence_skills"]), 1)
        self.assertEqual(len(built["context"]["evidence_ids"]), 1)
        self.assertTrue(built["context"]["evidence_ids"][0].startswith("lesson:"))
        self.assertEqual(built["context"]["current_features"], full["context"]["current_features"])
        self.assertLessEqual(built["audit"]["context_bytes"], limit)

    def test_current_features_are_allowlisted(self):
        e = envelope()
        e["features"]["cad"]["parts"]["1"]["heuristic_score"] = 999
        context = build_sequence_knowledge(e)["context"]
        self.assertNotIn("heuristic_score", context["current_features"]["1"])
        self.assertEqual(context["current_features"]["1"]["volume"], 4)

    def test_legacy_action_programs_incompatible(self):
        built = build_sequence_knowledge(envelope(), skills=[dict(strategy_id="legacy:move", program={})])
        self.assertEqual(built["context"]["sequence_skills"], [])
        self.assertTrue(built["audit"]["excluded_skills"][0]["do_not_execute"])
        self.assertEqual(built["audit"]["actual_sequence_skill_status"], "NOT_YET_GENERATED")

    def test_fixture_not_enabled_by_default(self):
        card = make_fixture_sequence_skill(REVERSE)
        built = build_sequence_knowledge(envelope(), skills=[card])
        self.assertEqual(built["context"]["sequence_skills"], [])
        result = execute_sequence_skill(card, ["1", "2"], built["context"])
        self.assertEqual(result["status"], "REJECTED")

    def test_fixture_enabled_explicitly_does_not_claim_model_generated(self):
        built = build_sequence_knowledge(envelope(), skills=[make_fixture_sequence_skill(REVERSE)],
                                         limits=dict(allow_fixture_skills=True))
        self.assertEqual(len(built["context"]["sequence_skills"]), 1)
        self.assertEqual(built["audit"]["actual_sequence_skill_status"], "NOT_YET_GENERATED")

    def test_lessons_are_scoped_hypotheses(self):
        result = build_sequence_knowledge(envelope(), lessons=lessons())
        row = result["context"]["lesson_references"][0]
        self.assertTrue(row["claims_are_hypotheses"])
        self.assertFalse(row["target_verified"])
        self.assertIsNone(row["current_part_id_mapping"])
        self.assertEqual(row["provenance_check"], "hash_checked_source_files_not_reopened")
        self.assertEqual(result["context"]["evidence_ids"], ["lesson:" + row["lesson_id"]])

    def test_eval_lessons_excluded(self):
        built = build_sequence_knowledge(envelope(), lessons=lessons("00335"))
        self.assertEqual(built["context"]["lesson_references"], [])
        self.assertEqual(built["audit"]["excluded_lessons"][0]["reason"], "evaluation_task_excluded")

    def test_legacy_lesson_source_task_name_is_explicit_scope(self):
        built = build_sequence_knowledge(envelope(), lessons=lessons("07009", source_as_name=True))
        row = built["context"]["lesson_references"][0]
        self.assertEqual(row["source_task_ids"], ["07009"])
        self.assertEqual(row["observed_facts"][0]["source_task"], "07009")
        self.assertFalse(row["target_verified"])
        self.assertIsNone(row["current_part_id_mapping"])

    def test_legacy_lesson_source_task_name_still_excludes_evaluation(self):
        built = build_sequence_knowledge(envelope(), lessons=lessons("00335", source_as_name=True))
        self.assertEqual(built["context"]["lesson_references"], [])
        self.assertEqual(built["audit"]["excluded_lessons"][0]["reason"], "evaluation_task_excluded")

    def test_unknown_legacy_lesson_task_is_not_invented(self):
        for name in (None, "", " "):
            with self.subTest(source_task=name):
                built = build_sequence_knowledge(envelope(), lessons=lessons(name, source_as_name=True))
                self.assertEqual(built["context"]["lesson_references"], [])
                self.assertEqual(built["audit"]["excluded_lessons"][0]["reason"], "source_task_unknown")


class SequenceSkillTests(unittest.TestCase):
    def run_code(self, code, context=None):
        context = context or build_sequence_knowledge(envelope())["context"]
        return execute_sequence_skill(make_fixture_sequence_skill(code), context["candidate_ids"],
                                      context, allow_fixture=True)

    def test_full_reverse_program(self):
        result = self.run_code(REVERSE)
        self.assertEqual(result["status"], "PROPOSED")
        self.assertEqual(result["candidate_order"], ["2", "1"])
        self.assertFalse(result["physics_executed"])
        self.assertGreater(result["vm_steps"], 0)

    def test_current_geometry_program(self):
        code = "def propose(ctx):\n    return {'candidate_order': sorted(ctx['candidate_ids'], key=lambda p: ctx['current_features'][p]['volume']), 'evidence_ids': []}\n"
        result = self.run_code(code)
        self.assertEqual(result["status"], "PROPOSED")
        self.assertEqual(result["candidate_order"], ["2", "1"])

    def test_decline(self):
        self.assertEqual(self.run_code("def propose(ctx):\n    return None\n")["status"], "DECLINED")

    def test_unknown_duplicate_missing_and_hardcoded_ids(self):
        for order in (["1", "foreign"], ["1", "1"], ["1"], [], ["2", "1"]):
            with self.subTest(order=order):
                code = "def propose(ctx):\n    return {'candidate_order': %r, 'evidence_ids': []}\n" % order
                self.assertEqual(self.run_code(code)["status"], "REJECTED")

    def test_direction_support_pose_and_status_rejected(self):
        for key in ("direction", "support_ids", "pose", "status", "path_success"):
            with self.subTest(key=key):
                code = "def propose(ctx):\n    return {'candidate_order': ctx['candidate_ids'], 'evidence_ids': [], '%s': None}\n" % key
                self.assertEqual(self.run_code(code)["status"], "REJECTED")

    def test_unknown_evidence_rejected(self):
        code = "def propose(ctx):\n    return {'candidate_order': ctx['candidate_ids'], 'evidence_ids': ['invented']}\n"
        self.assertEqual(self.run_code(code)["status"], "REJECTED")

    def test_existing_evidence_allowed(self):
        context = build_sequence_knowledge(envelope(), [record()])["context"]
        code = "def propose(ctx):\n    return {'candidate_order': ctx['candidate_ids'], 'evidence_ids': ctx['evidence_ids']}\n"
        result = self.run_code(code, context)
        self.assertEqual(result["status"], "PROPOSED")
        self.assertEqual(result["evidence_ids"], context["evidence_ids"])

    def test_foreign_memory_id_not_current_part(self):
        context = build_sequence_knowledge(envelope(), [record()])["context"]
        # Source ID is coincidentally a current ID. The renaming probe keeps
        # historical source IDs fixed, so this still fails the current-ID contract.
        code = "def propose(ctx):\n    p = ctx['memory_references'][0]['source_choice']['part_id']\n    return {'candidate_order': [p, '2'], 'evidence_ids': []}\n"
        self.assertEqual(self.run_code(code, context)["status"], "REJECTED")

    def test_mutation_is_rejected_without_changing_host(self):
        context = build_sequence_knowledge(envelope())["context"]
        before = deepcopy(context)
        code = "def propose(ctx):\n    ctx['candidate_ids'].append('other')\n    return None\n"
        result = self.run_code(code, context)
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(context, before)

    def test_files_network_imports_and_host_execution_rejected(self):
        snippets = ("import os\ndef propose(ctx):\n    return None\n",
            "def propose(ctx):\n    return open('secret')\n",
            "def propose(ctx):\n    return __import__('socket')\n",
            "def propose(ctx):\n    return eval('1+1')\n",
            "def propose(ctx):\n    return ctx.__class__\n")
        for source in snippets:
            with self.subTest(source=source):
                self.assertEqual(self.run_code(source)["status"], "REJECTED")

    def test_metered_loop_rejected(self):
        source = "def propose(ctx):\n    for i in range(4096):\n        for j in range(4096):\n            a = i + j\n    return None\n"
        self.assertEqual(self.run_code(source)["status"], "REJECTED")

    def test_program_changed_after_card_signature_rejected(self):
        card = make_fixture_sequence_skill(REVERSE)
        card["program"]["source"] = "def propose(ctx):\n    return None\n"
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_sequence_skill(card, allow_fixture=True)

    def test_unregistered_card_rejected(self):
        card = make_fixture_sequence_skill(REVERSE)
        card["validation"]["status"] = "CANDIDATE"
        card["content_sha256"] = digest({k: v for k, v in card.items() if k != "content_sha256"})
        with self.assertRaisesRegex(ValueError, "registration"):
            validate_sequence_skill(card, allow_fixture=True)


if __name__ == "__main__":
    unittest.main()
