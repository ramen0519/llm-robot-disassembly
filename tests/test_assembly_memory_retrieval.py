"""Offline retrieval tests: no provider, no physical feasibility assertions."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from worker.assembly_memory_retrieval import select_references
from orchestrator.explicit_action_memory import FrozenActionMemory
from worker.explicit_request_budget import bounded_memory_request

ROOT=Path(__file__).absolute().parents[1]


def request():
    return dict(task_context={"task":{"assembly_id":"cad", "input_sha256":{"0.obj":"a"}}},
        remaining_parts=["0","1","2","3"],candidate_ids=["0","1","2","3"],
        pose_catalog=[dict(matrix=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])])


def ref(identity,status="UNKNOWN",supports=None,same=False):
    r=request()
    return dict(evidence_id=identity,source_task_context=deepcopy(r["task_context"]),
        source_remaining_parts=list(r["remaining_parts"]),
        source_pose_matrix=deepcopy(r["pose_catalog"][0]["matrix"]),
        source_choice=dict(part_id="3",skill_id="single_force_v1",direction_id="zp",
            second_direction_id=None,support_ids=supports or []),
        status=status,path_success=True,stability_success=status=="PASS",
        parent_failure_evidence_id=None,same_conditions=same,same_task=same,
        reference_only=not same,hard_pruning=False,physics_reuse_allowed=False,
        current_part_id_mapping=None,current_attempt_verified=False)


class AssemblyRetrievalTests(unittest.TestCase):
    def test_many_exact_failures_do_not_hide_nearby_scoped_success(self):
        rows=[ref("fail"+str(i),supports=[str(i%3)],same=True) for i in range(20)]
        positive=ref("success",status="PASS",supports=["0","1"])
        positive["source_task_context"]["task"]["solver_settings"]={"seed":1,"budget":2}
        rows.append(positive)
        result=select_references(rows,request(),4)
        self.assertEqual({r["status"] for r in result},{"PASS","UNKNOWN"})
        success=next(r for r in result if r["status"]=="PASS")
        self.assertTrue(success["reference_only"])
        self.assertFalse(success["same_conditions"])
        self.assertEqual(success["source_task_context"]["task"]["solver_settings"]["budget"],2)

    def test_contradictions_are_retained_and_duplicate_failures_do_not_fill_window(self):
        rows=[ref("f"+str(i),supports=["0"]) for i in range(20)]
        rows += [ref("success",status="PASS",supports=["0"]),ref("other",supports=["1"])]
        result=select_references(rows,request(),4)
        self.assertEqual(len(result),3)
        self.assertEqual(sum(r["source_choice"]["support_ids"]==["0"] for r in result),2)

    def test_prefers_support_contrast_with_same_mover_pose_path(self):
        success=ref("p","PASS",["0"])
        unrelated=ref("a",supports=["1"]);unrelated["source_choice"]["part_id"]="2"
        contrast=ref("z",supports=["1"])
        result=select_references([unrelated,success,contrast],request(),2)
        self.assertEqual([r["evidence_id"] for r in result],["p","z"])

    def test_same_cad_failure_precedes_foreign_success(self):
        own=ref("own")
        foreign=ref("foreign","PASS")
        foreign["source_task_context"]["task"]["input_sha256"]={"0.obj":"b"}
        self.assertEqual(select_references([foreign,own],request(),1)[0]["evidence_id"],"own")

    def test_parent_chain_atomic_and_not_invented(self):
        parent=ref("f",supports=["0"])
        child=ref("p","PASS",["0","1"]);child["parent_failure_evidence_id"]="f"
        one=select_references([parent,child],request(),1)
        self.assertEqual([r["evidence_id"] for r in one],["f"])
        two=select_references([parent,child],request(),2)
        self.assertEqual([r["evidence_id"] for r in two],["f","p"])
        with self.assertRaisesRegex(ValueError,"missing_parent"):
            select_references([child],request(),2)

    def test_cycle_rejected(self):
        a=ref("a");b=ref("b");a["parent_failure_evidence_id"]="b";b["parent_failure_evidence_id"]="a"
        with self.assertRaisesRegex(ValueError,"cyclic"):
            select_references([a,b],request(),4)

    def test_removed_part_is_not_preferred_and_never_remapped(self):
        old=ref("old","PASS");r=request();r["remaining_parts"].remove("3");r["candidate_ids"].remove("3")
        current=ref("current");current["source_choice"]["part_id"]="1"
        result=select_references([old,current],r,1)
        self.assertEqual(result[0]["evidence_id"],"current")
        self.assertIsNone(result[0]["current_part_id_mapping"])

    def test_input_immutable_and_physics_never_reused(self):
        rows=[ref("a"),ref("b","PASS",["0"])];r=request();before=deepcopy((rows,r))
        for n in (0,1,2,12):
            result=select_references(rows,r,n)
            self.assertLessEqual(len(result),n)
            for row in result:
                self.assertFalse(row["hard_pruning"])
                self.assertFalse(row["physics_reuse_allowed"])
                self.assertFalse(row["current_attempt_verified"])
        self.assertEqual((rows,r),before)

    def test_bad_limit_and_duplicate_identity_rejected(self):
        for n in (-1,13,True):
            with self.assertRaises(ValueError):select_references([],request(),n)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            select_references([ref("a"),ref("a")],request(),4)

    def test_actual_frozen_observations_reach_input_with_success_and_failure(self):
        directory=ROOT/"runs/01256_frozen_experience_20261021"
        path=directory/"memory_for_revalidation_frozen.json"
        before=path.read_bytes()
        memory=FrozenActionMemory(json.loads(before))
        episode=ROOT/"runs/hierarchy_01256_v2_live_20261021/episode_01/worker_result.json"
        r=deepcopy(json.loads(episode.read_text())["decisions"][0]["request"])
        r["memory_references"]=[]
        r["evidence_ids"]=[f["evidence_id"] for f in r["episode_feedback"]]
        refs=memory.retrieve(r,max_records=8,policy="assembly_balanced_v1")
        packet,audit=bounded_memory_request(r,refs,require_parent_pairs=True)
        self.assertTrue(any(ref["status"]=="PASS" for ref in packet["memory_references"]))
        self.assertTrue(any(ref["status"]=="UNKNOWN" for ref in packet["memory_references"]))
        self.assertTrue(all(ref["source_task_context"]["task"]["assembly_id"]=="01256"
                            for ref in packet["memory_references"]))
        self.assertEqual(packet["candidate_ids"],r["candidate_ids"])
        self.assertEqual(packet["attempted_action_ids"],r["attempted_action_ids"])
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(audit["excluded_reference_count"],0)


if __name__=="__main__":
    unittest.main()
