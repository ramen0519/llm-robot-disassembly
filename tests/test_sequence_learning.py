"""Offline fixtures: no model or physics call; provenance and binding checks."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sequence_knowledge import native_attempt
from orchestrator.sequence_knowledge import (build_native_snapshot, build_sequence_knowledge,
    execute_sequence_skill, make_fixture_sequence_skill)
from orchestrator.sequence_learning import (build_sequence_learning_packet,
    validate_sequence_learning_packet, validate_sequence_lesson, check_sequence_lesson,
    make_sequence_candidate, validate_sequence_candidate, execute_sequence_candidate,
    validate_sequence_registration, write_sequence_artifact, freeze_sequence_lessons,
    freeze_sequence_skills, validate_sequence_lesson_library, validate_sequence_skill_library,
    PROOF_PROTOCOL, TRACE_PROTOCOL)
from worker.task_memory import digest


SOURCE = "def propose(ctx):\n    return {'candidate_order': ctx['candidate_ids'][::-1], 'evidence_ids': []}\n"


def native_memory(task="training-A", event="fixture:attempt:1"):
    attempt = native_attempt()
    attempt["event_id"] = event
    return build_native_snapshot(dict(task=dict(assembly_id=task, input_sha256={"1.obj":"a"*64},
        solver_settings=dict(seed=0))), [attempt], "fixture/worker_result.json", "b"*64)


def packet():
    return build_sequence_learning_packet(native_memory())


def lesson(p=None, origin="injected_transport"):
    p = p or packet()
    claim = dict(title="Reference order hypothesis", hypothesis="Test a context-dependent order.",
        when_to_apply="Current geometric roles can be compared.",
        when_not_to_apply="Source part labels have no mapping.", ranking_guidance="Preserve all candidates.",
        evidence_ids=[p["facts"][0]["evidence_id"]], counterexample_ids=[])
    provenance = dict(origin=origin)
    if origin == "actual_llm":
        provenance.update(model="fixture-not-actual-model", request_id="fixture:model", response_sha256="c"*64)
    return validate_sequence_lesson(p, claim, provenance)


def candidate(source=SOURCE, origin="injected_transport"):
    learned = lesson(origin=origin)
    return make_sequence_candidate(learned, dict(name="Reverse fixture", description="Offline fixture only.",
        source=source, evidence_ids=[learned["facts"][0]["evidence_id"]]), learned["provenance"])


def envelope(seed=0, task="training-B"):
    return dict(request=dict(run_id="fixture", request_id="fixture:r1", state_id="fixture:state",
        assembled_parts=["1", "2", "3"], candidate_ids=["1", "2", "3"], base_part=None),
        features=dict(cad=dict(task=dict(assembly_id=task, input_sha256={"1.obj":"d"*64},
            solver_settings=dict(seed=seed)), parts={str(p):dict(volume=p) for p in (1,2,3)})))


class LearningContractsTests(unittest.TestCase):
    def test_native_packet_preserves_facts_not_inferred_causes(self):
        p = packet()
        self.assertEqual(validate_sequence_learning_packet(p), p)
        fact = p["facts"][0]
        self.assertTrue(fact["observed"]["accepted_for_transition"])
        self.assertEqual(fact["source_task"]["assembly_id"], "training-A")
        self.assertIsNone(fact["failure_cause"])
        self.assertIsNone(fact["blocker"])
        self.assertFalse(p["hard_pruning"])

    def test_source_part_geometry_is_preserved_as_source_observation(self):
        features=dict(bounds=[[0,0,0],[1,2,3]],extents=[1,2,3],center_mass=[.5,1,1.5],
                      volume=6,watertight=True,validity=dict(volume_finite=True))
        source=dict(task=dict(assembly_id="training-A",input_sha256="a"*64),
                    geometry_features={"1":features,"2":dict(volume=1000)})
        memory=build_native_snapshot(source,[native_attempt()],"fixture/result.json","b"*64)
        learned=build_sequence_learning_packet(memory)
        fact=learned["facts"][0]
        self.assertEqual(fact["source_state"]["part_move"],"1")
        self.assertEqual(fact["source_part_features"],features)
        self.assertEqual(fact["source_part_features_status"],"AVAILABLE")
        self.assertNotEqual(fact["source_part_features"],source["geometry_features"]["2"])
        fact["source_part_features"]["volume"]=123
        self.assertEqual(memory["records"][0]["source_task_context"]["geometry_features"]["1"]["volume"],6)

    def test_missing_source_geometry_stays_unknown(self):
        fact=packet()["facts"][0]
        self.assertIsNone(fact["source_part_features"])
        self.assertEqual(fact["source_part_features_status"],"NOT_AVAILABLE")

    def test_prior_packet_without_geometry_fields_remains_valid(self):
        p=packet()
        for fact in p["facts"]:
            fact.pop("source_part_features")
            fact.pop("source_part_features_status")
        p["content_sha256"]=digest({k:v for k,v in p.items() if k!="content_sha256"})
        self.assertEqual(validate_sequence_learning_packet(p),p)
        learned=lesson(p)
        self.assertEqual(check_sequence_lesson(learned),learned)

    def test_geometry_availability_cannot_disguise_missing_values(self):
        p=packet()
        p["facts"][0]["source_part_features_status"]="AVAILABLE"
        p["content_sha256"]=digest({k:v for k,v in p.items() if k!="content_sha256"})
        with self.assertRaisesRegex(ValueError,"availability_mismatch"):
            validate_sequence_learning_packet(p)

    def test_packet_merges_multiple_source_products(self):
        snapshots = [native_memory(), native_memory("training-B", "other:attempt:1")]
        p = build_sequence_learning_packet(snapshots)
        self.assertEqual({f["source_task"]["assembly_id"] for f in p["facts"]}, {"training-A", "training-B"})
        self.assertEqual(p["source_fact_count"], 2)

    def test_duplicate_native_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_sequence_learning_packet([native_memory(), native_memory()])

    def test_bounded_packet_reports_omitted_facts(self):
        p = build_sequence_learning_packet([native_memory(), native_memory("B", "b:1")], max_records=1)
        self.assertEqual(p["source_fact_count"], 2)
        self.assertEqual(p["selected_fact_count"], 1)
        self.assertEqual(len(p["excluded_evidence_ids"]), 1)

    def test_large_accumulation_keeps_bounded_exclusion_manifest(self):
        attempts=[]
        for index in range(300):
            item=native_attempt()
            item["event_id"]="fixture:attempt:"+str(index)
            attempts.append(item)
        memory=build_native_snapshot(dict(task=dict(assembly_id="training-A",input_sha256="a"*64)),
            attempts,"fixture/worker_result.json","b"*64)
        p=build_sequence_learning_packet(memory,max_records=2)
        self.assertEqual(p["excluded_evidence_count"],298)
        self.assertEqual(len(p["excluded_evidence_ids"]),64)
        self.assertTrue(p["excluded_ids_are_prefix"])
        self.assertEqual(validate_sequence_learning_packet(p),p)

    def test_model_lesson_stays_hypothesis(self):
        row = lesson()
        self.assertEqual(check_sequence_lesson(row), row)
        self.assertTrue(row["claims_are_hypotheses"])
        self.assertEqual(row["usefulness"], "UNVERIFIED")
        self.assertEqual(row["facts"], row["packet"]["facts"])

    def test_unobserved_lesson_evidence_rejected(self):
        row = lesson()
        row["claim"]["evidence_ids"] = ["made-up"]
        with self.assertRaisesRegex(ValueError, "unobserved"):
            validate_sequence_lesson(packet(), row["claim"], dict(origin="injected_transport"))

    def test_lesson_physical_fact_field_not_accepted(self):
        row = lesson()
        row["claim"]["blocker"] = "invented"
        with self.assertRaisesRegex(ValueError, "claim_fields"):
            validate_sequence_lesson(packet(), row["claim"], dict(origin="injected_transport"))

    def test_actual_generation_requires_model_request_hash(self):
        row = lesson()
        with self.assertRaises(ValueError):
            validate_sequence_lesson(packet(), row["claim"], dict(origin="actual_llm"))

    def test_rejected_python_preserved_as_data(self):
        source = "import os\ndef propose(ctx):\n    return None\n"
        draft = candidate(source)
        self.assertEqual(draft["status"], "REJECTED")
        self.assertEqual(draft["raw_candidate_response"]["source"], source)
        self.assertIsNotNone(draft["rejection"])
        self.assertEqual(validate_sequence_candidate(draft, require_executable=False), draft)
        with self.assertRaises(ValueError):
            execute_sequence_candidate(draft, ["1"], {})

    def test_candidate_not_registered_and_validation_executes(self):
        draft = candidate()
        e = envelope()
        context = build_sequence_knowledge(e)["context"]
        result = execute_sequence_candidate(draft, e["request"]["candidate_ids"], context)
        self.assertEqual(result["status"], "PROPOSED")
        self.assertEqual(result["candidate_order"], ["3", "2", "1"])
        self.assertFalse(result["registered"])
        self.assertEqual(result["validation_role"], "sequence_candidate")
        self.assertFalse(result["physics_executed"])

    def test_old_physical_action_output_is_rejected(self):
        draft = candidate("def propose(ctx):\n    return {'part_id':'1','direction_id':'xp','support_ids':[]}\n")
        context = build_sequence_knowledge(envelope())["context"]
        self.assertEqual(execute_sequence_candidate(draft, context["candidate_ids"], context)["status"], "REJECTED")

    def test_full_permutation_and_foreign_scope_enforced(self):
        context = build_sequence_knowledge(envelope())["context"]
        for order in (["1"], ["1","1","2"], ["1","2","3"], ["foreign","2","3"]):
            with self.subTest(order=order):
                draft = candidate("def propose(ctx):\n    return {'candidate_order': %r, 'evidence_ids': []}\n" % order)
                self.assertEqual(execute_sequence_candidate(draft, context["candidate_ids"], context)["status"], "REJECTED")

    def test_new_lesson_library_reaches_ordering_context(self):
        frozen = freeze_sequence_lessons([lesson()])
        self.assertEqual(validate_sequence_lesson_library(frozen), frozen)
        built = build_sequence_knowledge(envelope(), lessons=frozen, limits=dict(allow_fixture_skills=True))
        self.assertEqual(len(built["context"]["lesson_references"]), 1)
        self.assertEqual(len(built["context"]["evidence_ids"]), 1)
        self.assertEqual(built["context"]["lesson_references"][0]["source_task_ids"], ["training-A"])

    def test_large_lesson_keeps_cited_facts_and_four_candidate_programs(self):
        attempts=[]
        for index in range(24):
            item=native_attempt()
            item["event_id"]="large:attempt:"+str(index)
            attempts.append(item)
        source=dict(task=dict(assembly_id="training-A",
            input_sha256={str(i)+".obj":"a"*64 for i in range(18)},
            solver_settings=dict(seed=0,fixture_description="fixture"*300)),
            geometry_features={"1":dict(extents=[1,2,3],volume=6)})
        p=build_sequence_learning_packet(build_native_snapshot(source,attempts,"fixture/large.json","b"*64))
        learned=lesson(p)
        frozen=freeze_sequence_lessons([learned])
        original=deepcopy(frozen)
        cards=[make_fixture_sequence_skill(SOURCE,skill_id="fixture:program:"+str(i)) for i in range(4)]
        built=build_sequence_knowledge(envelope(),skills=cards,lessons=frozen,
            limits=dict(allow_fixture_skills=True))
        self.assertEqual(len(built["context"]["lesson_references"]),1)
        row=built["context"]["lesson_references"][0]
        self.assertEqual(row["source_fact_count"],24)
        self.assertEqual(row["provided_fact_count"],1)
        self.assertEqual(row["omitted_fact_count"],23)
        self.assertEqual(row["observed_facts"],[p["facts"][0]])
        self.assertEqual(len(built["context"]["sequence_skills"]),4)
        self.assertLessEqual(built["audit"]["context_bytes"],49152)
        self.assertEqual(frozen,original)

    def test_counterexamples_are_not_dropped_from_lesson_projection(self):
        p=build_sequence_learning_packet([native_memory(),native_memory("training-B","second:1")])
        learned=lesson(p)
        claim=deepcopy(learned["claim"])
        claim["counterexample_ids"]=[p["facts"][1]["evidence_id"]]
        learned=validate_sequence_lesson(p,claim,learned["provenance"])
        built=build_sequence_knowledge(envelope(task="evaluation-C"),lessons=freeze_sequence_lessons([learned]),
            limits=dict(allow_fixture_skills=True))
        row=built["context"]["lesson_references"][0]
        self.assertEqual(len(row["observed_facts"]),2)
        self.assertEqual(row["omitted_fact_count"],0)

    def test_fixture_lesson_cannot_silently_enter_live(self):
        with self.assertRaisesRegex(ValueError, "fixture"):
            build_sequence_knowledge(envelope(), lessons=freeze_sequence_lessons([lesson()]))

    def test_new_lesson_cannot_leak_current_evaluation_task(self):
        frozen = freeze_sequence_lessons([lesson()])
        built = build_sequence_knowledge(envelope(task="training-A"), lessons=frozen,
                                          limits=dict(allow_fixture_skills=True))
        self.assertEqual(built["context"]["lesson_references"], [])


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sequence_learning_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.draft = candidate()

    def write(self, relative, data):
        return write_sequence_artifact(self.root/relative, data, allow_test_root=True)

    def case(self, seed=0, name=None, *, alter=None):
        name = name or str(seed)
        e = envelope(seed)
        req = e["request"]
        req["run_id"] = "fixture:" + name
        req["request_id"] = req["run_id"] + ":r1"
        context = build_sequence_knowledge(e)["context"]
        result = execute_sequence_candidate(self.draft, req["candidate_ids"], context)
        response = dict(request_id=req["request_id"], state_id=req["state_id"],
            candidate_ids=result["candidate_order"], fallback_reason=None, metadata=dict(
                validation_role="sequence_candidate", candidate_sha256=self.draft["content_sha256"],
                program_sha256=self.draft["program_sha256"], context_sha256=digest(context)))
        parents = [["1","2","3"], ["1","2"]]
        children = [["1","2"], ["1"]]
        chain, attempts = [], []
        for index,(parent,child,mover) in enumerate(zip(parents,children,["3","2"])):
            raw = dict(feasible=True, part_move=mover, action=[1,0,0], parts_fix=child, pose=None)
            chain.append(dict(parent=parent, child=child, part_move=mover, sim_info=raw))
            attempts.append(dict(event_id=req["run_id"]+":attempt:"+str(index), remaining_parts=parent,
                part_move=mover, accepted_for_transition=True, deadline_exceeded=False, status="PASS",
                verifier_domain="ASAP_RedMax", raw_sim_info=raw))
        worker = dict(profile="sequence_native_v1", status="PASS", success=True, actual_physics=False,
            fixture=True, error=None, decision_halt_reason=None, lower_level_policy="original_ASAP",
            hard_pruning=False, complete_permutation=True, heuristic_tripwire_calls=0,
            source_methods_restored=True, sources_unchanged=True, input_unchanged=True,
            full_robot_execution=False, physics_binding=dict(redmax_binary_sha256="a"*64,physics_source_sha256="b"*64),
            source_methods_sha256={k:"c"*64 for k in ("simulate","update_tree","path_check","stability_check")},
            connected=dict(success=True, complete_chain=chain), selected_plan=chain, attempts=attempts,
            configuration=dict(seed=seed,automatic_retries=0,candidate_pruning=False,physics_verdict_reuse=False),
            rank_decisions=[dict(request=deepcopy(req), applied_order=result["candidate_order"],
                full_candidate_set=True,heuristic_or_gnn_invoked=False,rng_before="rng",rng_after="rng")])
        if alter:
            alter(worker,response,context)
        response_ref = self.write(name+"/response.json",response)
        trace = dict(protocol=TRACE_PROTOCOL,candidate_sha256=self.draft["content_sha256"],
            program_sha256=self.draft["program_sha256"], calls=[dict(request=req,context=context,response_ref=response_ref)])
        return dict(case_id=name, worker_result=self.write(name+"/worker.json",worker),
                    decision_trace=self.write(name+"/trace.json",trace))

    def proof(self, cases):
        return dict(protocol=PROOF_PROTOCOL, candidate_sha256=self.draft["content_sha256"],
                    program_sha256=self.draft["program_sha256"], cases=cases)

    def test_complete_bound_fixture_registers_only_fixture(self):
        proof = self.proof([self.case(0),self.case(1)])
        result = validate_sequence_registration(self.draft,proof,allow_fixture=True,allow_test_root=True)
        self.assertTrue(result["card"]["fixture"])
        self.assertEqual(result["admission"]["causal_benefit"],"NOT_TESTED")
        self.assertEqual(len(result["admission"]["cases"]),2)
        library=freeze_sequence_skills([result["card"]])
        self.assertEqual(validate_sequence_skill_library(library),library)
        e=envelope(task="evaluation-C")
        context=build_sequence_knowledge(e,skills=library,limits=dict(allow_fixture_skills=True))["context"]
        outcome=execute_sequence_skill(context["sequence_skills"][0],context["candidate_ids"],context,allow_fixture=True)
        self.assertEqual(outcome["status"],"PROPOSED")
        self.assertEqual(outcome["candidate_order"],["3","2","1"])

    def test_fixture_generation_not_registered_as_live(self):
        with self.assertRaisesRegex(ValueError,"fixture_sequence_generation"):
            validate_sequence_registration(self.draft,self.proof([self.case(0),self.case(1)]),allow_test_root=True)

    def test_bare_pass_is_not_registration_evidence(self):
        with self.assertRaises(ValueError):
            validate_sequence_registration(self.draft,dict(status="PASS",success=True),allow_fixture=True)

    def test_one_case_not_enough(self):
        with self.assertRaises(ValueError):
            validate_sequence_registration(self.draft,self.proof([self.case(0)]),allow_fixture=True,allow_test_root=True)

    def test_duplicate_conditions_rejected(self):
        proof=self.proof([self.case(0,"one"),self.case(0,"two")])
        with self.assertRaisesRegex(ValueError,"distinct_conditions"):
            validate_sequence_registration(self.draft,proof,allow_fixture=True,allow_test_root=True)

    def test_different_run_labels_do_not_make_independent_conditions(self):
        def add_label(w,r,c):
            w["configuration"].update(run_id="different",output_dir="different/path")
            c["task"]["run_id"]="different"
            r["metadata"]["context_sha256"]=digest(c)
        proof=self.proof([self.case(0,"original"),self.case(0,"relabeled",alter=add_label)])
        with self.assertRaisesRegex(ValueError,"distinct_conditions"):
            validate_sequence_registration(self.draft,proof,allow_fixture=True,allow_test_root=True)

    def test_wrong_program_or_context_response_binding_rejected(self):
        for key in ("program_sha256","context_sha256","candidate_sha256"):
            with self.subTest(key=key):
                first=self.case(0,key+"_one",alter=lambda w,r,c:r["metadata"].update({key:"e"*64}))
                second=self.case(1,key+"_two")
                with self.assertRaisesRegex(ValueError,"execution_binding"):
                    validate_sequence_registration(self.draft,self.proof([first,second]),allow_fixture=True,allow_test_root=True)

    def test_changed_artifact_bytes_rejected(self):
        one,two=self.case(0),self.case(1)
        Path(one["worker_result"]["path"]).write_text("{}",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"artifact_hash"):
            validate_sequence_registration(self.draft,self.proof([one,two]),allow_fixture=True,allow_test_root=True)

    def test_disconnected_and_unaccepted_physics_rejected(self):
        mutations=(lambda w,r,c:w["connected"].update(complete_chain=[]),
                   lambda w,r,c:w["attempts"][0].update(accepted_for_transition=False))
        for i,mutate in enumerate(mutations):
            with self.subTest(i=i):
                proof=self.proof([self.case(0,str(i)+"a",alter=mutate),self.case(1,str(i)+"b")])
                with self.assertRaises(ValueError):
                    validate_sequence_registration(self.draft,proof,allow_fixture=True,allow_test_root=True)

    def test_no_execution_trace_rejected(self):
        one,two=self.case(0),self.case(1)
        one["decision_trace"]=self.write("bad_trace.json",dict(protocol=TRACE_PROTOCOL,calls=[]))
        with self.assertRaisesRegex(ValueError,"trace_required"):
            validate_sequence_registration(self.draft,self.proof([one,two]),allow_fixture=True,allow_test_root=True)

    def test_native_candidate_pruning_is_not_admitted(self):
        def pruned(w,r,c):
            w["rank_decisions"][0]["request"]["candidate_ids"]=["1","2"]
        first,second=self.case(0,"pruned",alter=pruned),self.case(1,"good")
        with self.assertRaises(ValueError):
            validate_sequence_registration(self.draft,self.proof([first,second]),allow_fixture=True,allow_test_root=True)

    def test_exclusive_artifact_write_preserves_rejection(self):
        rejected=candidate("import os\ndef propose(ctx):\n    return None\n")
        ref=self.write("rejected.json",rejected)
        before=Path(ref["path"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.write("rejected.json",{})
        self.assertEqual(Path(ref["path"]).read_bytes(),before)

    def test_forbidden_drive_ref_rejected_before_read(self):
        proof=self.proof([self.case(0),self.case(1)])
        proof["cases"][0]["worker_result"]={"path":"E:/forbidden.json","sha256":"a"*64}
        with patch.object(Path,"read_bytes",side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError,"non-C"):
                validate_sequence_registration(self.draft,proof,allow_fixture=True,allow_test_root=True)


if __name__=="__main__":
    unittest.main()
