"""Pure fixture coordinator checks: zero API, files, physics or real results."""
from copy import deepcopy
import unittest

from orchestrator.sequence_iterative_training import run_iterative_training, LEARNING_STATUSES


TASKS = [dict(assembly_id="task"+str(i)) for i in range(1,6)]


class Backend:
    def __init__(self, *, failure=None, duplicate_after_first=True, unknown=False):
        self.halted=None
        self.calls=[]
        self.skills=set()
        self.lessons=[]
        self.observations=[]
        self.failure=failure
        self.duplicate_after_first=duplicate_after_first
        self.unknown=unknown
        self.current=None

    def log(self, method, **fields):
        self.calls.append(dict(method=method,current=self.current,**deepcopy(fields)))

    def begin_round(self, task, number):
        self.current=(task["assembly_id"],number)
        self.log("begin_round")

    def event(self, kind, **fields):
        self.log("event",kind=kind,fields=fields)

    def freeze(self, name):
        self.log("freeze",name=name)
        return dict(name=name,observations=len(self.observations),skills=sorted(self.skills))

    def episode(self, task, seed, frozen):
        self.log("episode",task=task["assembly_id"],seed=seed,frozen=frozen)
        if self.failure=="network" and self.current==("task1",1):
            return dict(status="UNKNOWN",halt_required=True,error="api_timeout")
        if self.failure=="host_error":
            return dict(status="ERROR",reuse_status="NOT_RUN")
        return dict(path="fixture/episode",status="UNKNOWN" if self.unknown else "PASS",
                    reuse_status="EXECUTED_EFFECT_UNVERIFIED" if task["assembly_id"] in self.skills else "NOT_SELECTED")

    def collect(self, episode):
        self.log("collect")
        self.observations.append((self.current,episode))

    def packet(self, task, episode):
        self.log("packet")
        if self.failure=="no_observations":
            return None
        return dict(fixture=True,current=self.current,observation_count=len(self.observations))

    def lesson(self, task, packet, previous_lesson=None, validation=None, candidate=None):
        self.log("initial_lesson" if previous_lesson is None else "postlesson")
        if previous_lesson is None and self.failure=="lesson":
            return dict(ok=False,error="invalid_claim")
        if previous_lesson is not None and self.failure=="postlesson":
            return dict(ok=False,error="invalid_postclaim")
        return dict(ok=True,lesson=dict(fixture=True,round=self.current,post=previous_lesson is not None))

    def candidate(self, task, packet, lesson, previous, feedback, revision):
        self.log("candidate",revision=revision,previous=previous,feedback=feedback)
        if self.failure=="candidate" or self.failure=="candidate_revision" and revision==0:
            return dict(ok=False,error="invalid_program",candidate=dict(source="rejected fixture"))
        if self.failure=="candidate_transport":
            return dict(ok=False,halt_required=True,error="api_connection_error")
        return dict(ok=True,candidate=dict(program_sha256=task["assembly_id"],
            content_sha256=task["assembly_id"]+":"+str(self.current[1])+":"+str(revision),fixture=True))

    def duplicate_candidate(self, candidate):
        self.log("duplicate_check")
        if self.failure=="ambiguous_duplicate":
            return dict(status="maybe")
        return dict(duplicate=self.duplicate_after_first and candidate["program_sha256"] in self.skills,
                    skill_id=candidate["program_sha256"])

    def validate_candidate(self, task, candidate, lesson):
        self.log("validation")
        if self.failure=="validation":
            return dict(ok=False,error="native_plan_failed",fixture=True)
        if self.failure=="validation_revision" and candidate["content_sha256"].endswith(":0"):
            return dict(ok=False,error="native_plan_failed",fixture=True)
        return dict(ok=True,fixture=True,proof="fixture:not-physical")

    def register(self, candidate, lesson, validation):
        self.log("register")
        if self.failure=="registration_exception":
            raise ValueError("do not propagate secret provider contents")
        self.skills.add(candidate["program_sha256"])
        return dict(skill_id=candidate["program_sha256"],fixture=True)

    def retain_lesson(self, lesson):
        self.log("retain_lesson")
        self.lessons.append(deepcopy(lesson))

    def round_summary(self, task, number):
        self.log("round_summary")
        result=dict(episode_ref=dict(path="fixture/"+task["assembly_id"]+"/round"+str(number)+".json",sha256="a"*64),
            new_record_count=2,new_condition_count=1,repeated_condition_count=1,
            new_skill_count=1 if number==1 else 0,duplicate_programs=0 if number==1 else 1,
            behavior_equivalence="NOT_TESTED")
        if self.failure=="missing_episode_ref":
            result.pop("episode_ref")
        if self.failure=="summary_override":
            result["learning_status"]="FICTITIOUS_SUCCESS"
        return result


class IterativeTrainingTests(unittest.TestCase):
    def run_fixture(self, backend=None, **kwargs):
        backend=backend or Backend()
        return backend,run_iterative_training(TASKS,backend,**kwargs)

    def test_two_complete_waves_and_mandatory_review(self):
        b,result=self.run_fixture(stop_round=5)
        self.assertEqual(result["status"],"REVIEW_REQUIRED")
        self.assertEqual(result["effective_stop_round"],2)
        self.assertEqual(result["next_round"],3)
        self.assertEqual(len(result["completed_rounds"]),10)
        episodes=[c for c in b.calls if c["method"]=="episode"]
        self.assertEqual([(r["current"][0],r["current"][1]) for r in episodes],
                         [(t["assembly_id"],r) for r in (1,2) for t in TASKS])
        self.assertEqual([r["seed"] for r in episodes],[0]*5+[1]*5)
        self.assertTrue(all(r["status"]=="COMPLETED" for r in result["completed_rounds"]))

    def test_round_two_also_learns_but_duplicates_skip_physics_and_post_api(self):
        b,result=self.run_fixture()
        count=lambda method:sum(c["method"]==method for c in b.calls)
        self.assertEqual(count("initial_lesson"),10)
        self.assertEqual(count("candidate"),10)
        self.assertEqual(count("validation"),5)
        self.assertEqual(count("postlesson"),5)
        self.assertEqual(count("retain_lesson"),10)
        self.assertEqual([r["learning_status"] for r in result["completed_rounds"]],
                         ["REGISTERED"]*5+["DUPLICATE_PROGRAM"]*5)
        self.assertTrue(all(r["initial_lesson_retained"] for r in result["completed_rounds"]))

    def test_unregistered_false_dictionary_does_not_skip_validation(self):
        b,result=self.run_fixture(Backend(duplicate_after_first=False))
        self.assertEqual(sum(c["method"]=="validation" for c in b.calls),10)
        self.assertEqual([r["learning_status"] for r in result["completed_rounds"]],["REGISTERED"]*10)

    def test_initial_invocation_cannot_preapprove_unseen_round_three(self):
        b,result=self.run_fixture(stop_round=5,review_approved=True)
        self.assertEqual(result["status"],"REVIEW_REQUIRED")
        self.assertFalse(any(c["current"] and c["current"][1]>=3 for c in b.calls))

    def test_resume_requires_explicit_approval_before_any_backend_call(self):
        _,first=self.run_fixture()
        b=Backend()
        with self.assertRaisesRegex(ValueError,"explicit_review"):
            run_iterative_training(TASKS,b,start_round=3,stop_round=5,prior_rounds=first["completed_rounds"])
        self.assertEqual(b.calls,[])

    def test_approved_resume_continues_up_to_five_with_prior_rows_unchanged(self):
        b,first=self.run_fixture()
        prior=deepcopy(first["completed_rounds"])
        before=deepcopy(prior)
        second=run_iterative_training(TASKS,b,start_round=3,stop_round=5,
            prior_rounds=prior,review_approved=True)
        self.assertEqual(second["status"],"COMPLETED")
        self.assertEqual(len(second["completed_rounds"]),25)
        self.assertEqual(second["new_completed_round_count"],15)
        self.assertEqual(prior,before)
        self.assertIsNone(second["next_round"])

    def test_resume_rejects_incomplete_duplicate_or_wrong_product_history(self):
        _,first=self.run_fixture()
        for change in (lambda p:p.pop(),lambda p:p.append(deepcopy(p[0])),
                       lambda p:p[0].update(assembly_id="unknown"),
                       lambda p:p[0].update(status="INCOMPLETE")):
            prior=deepcopy(first["completed_rounds"]); change(prior); b=Backend()
            with self.assertRaises(ValueError):
                run_iterative_training(TASKS,b,start_round=3,stop_round=5,prior_rounds=prior,review_approved=True)
            self.assertEqual(b.calls,[])

    def test_resume_keeps_preregistered_wave_order(self):
        _,first=self.run_fixture()
        b=Backend()
        with self.assertRaisesRegex(ValueError,"wave_order"):
            run_iterative_training(TASKS,b,start_round=3,stop_round=5,
                prior_rounds=list(reversed(first["completed_rounds"])),review_approved=True)
        self.assertEqual(b.calls,[])

    def test_unknown_episode_is_not_falsely_called_physics_success(self):
        _,result=self.run_fixture(Backend(unknown=True))
        self.assertEqual(result["status"],"REVIEW_REQUIRED")
        self.assertTrue(all(r["episode_status"]=="UNKNOWN" for r in result["completed_rounds"]))
        self.assertEqual(result["performance_effect"],"UNVERIFIED")

    def test_network_and_host_error_halt_all_remaining_work(self):
        for reason in ("network","host_error"):
            with self.subTest(reason=reason):
                b,result=self.run_fixture(Backend(failure=reason))
                self.assertEqual(result["status"],"HALTED")
                self.assertEqual(result["completed_rounds"],[])
                self.assertEqual(sum(c["method"]=="episode" for c in b.calls),1)
                self.assertFalse(any(c["method"]=="candidate" for c in b.calls))

    def test_learning_transport_halt_preserves_lesson_without_more_api(self):
        b,result=self.run_fixture(Backend(failure="candidate_transport"))
        self.assertEqual(result["status"],"HALTED")
        self.assertEqual(sum(c["method"]=="candidate" for c in b.calls),1)
        self.assertEqual(len(b.lessons),1)
        self.assertFalse(any(c["method"] in ("validation","postlesson") for c in b.calls))

    def test_learning_terminal_outcomes_are_separate_from_episode_success(self):
        outcomes={"no_observations":"NO_OBSERVATIONS","lesson":"LESSON_REJECTED",
            "candidate":"CANDIDATE_REJECTED","validation":"VALIDATION_FAILED",
            "postlesson":"POSTLESSON_REJECTED"}
        for failure,status in outcomes.items():
            with self.subTest(failure=failure):
                _,result=self.run_fixture(Backend(failure=failure),stop_round=1)
                self.assertEqual(result["status"],"ROUND_LIMIT_REACHED")
                self.assertEqual([r["learning_status"] for r in result["completed_rounds"]],[status]*5)
                self.assertTrue(all(r["episode_status"]=="PASS" for r in result["completed_rounds"]))
                self.assertIn(status,LEARNING_STATUSES)

    def test_rejected_generation_feedback_reaches_one_new_candidate(self):
        b,result=self.run_fixture(Backend(failure="candidate_revision"),stop_round=1)
        candidates=[c for c in b.calls if c["method"]=="candidate"]
        self.assertEqual(len(candidates),10)
        for call in candidates[1::2]:
            self.assertEqual(call["revision"],1)
            self.assertEqual(call["previous"],dict(source="rejected fixture"))
            self.assertEqual(call["feedback"]["error"],"invalid_program")
        self.assertTrue(all(r["learning_status"]=="REGISTERED" for r in result["completed_rounds"]))

    def test_physics_feedback_reaches_one_revision(self):
        b,result=self.run_fixture(Backend(failure="validation_revision"),stop_round=1)
        second=[c for c in b.calls if c["method"]=="candidate" and c["revision"]==1]
        self.assertEqual(len(second),5)
        self.assertTrue(all(c["feedback"]["error"]=="native_plan_failed" for c in second))
        self.assertTrue(all(r["learning_status"]=="REGISTERED" for r in result["completed_rounds"]))

    def test_ambiguous_duplicate_and_proof_error_fail_closed(self):
        for failure in ("ambiguous_duplicate","registration_exception"):
            b,result=self.run_fixture(Backend(failure=failure))
            self.assertEqual(result["status"],"HALTED")
            self.assertEqual(result["completed_rounds"],[])
            self.assertEqual(result["stop_reason"],"stage_exception:ValueError")
            self.assertNotIn("secret",str(result))

    def test_summary_requires_evidence_and_cannot_invent_learning_success(self):
        for failure in ("missing_episode_ref","summary_override"):
            b,result=self.run_fixture(Backend(failure=failure))
            self.assertEqual(result["status"],"HALTED")
            self.assertEqual(result["completed_rounds"],[])

    def test_knowledge_is_frozen_before_each_episode_and_lesson_retained_after_validation(self):
        b,result=self.run_fixture()
        for index,call in enumerate(b.calls):
            if call["method"]=="episode":
                self.assertEqual(b.calls[index-1]["method"],"freeze")
        first=[c["method"] for c in b.calls if c["current"]==("task1",1)]
        self.assertLess(first.index("validation"),first.index("retain_lesson"))
        self.assertLess(first.index("register"),first.index("retain_lesson"))

    def test_novelty_is_reported_without_automatic_low_novelty_stop(self):
        _,result=self.run_fixture()
        self.assertEqual(result["status"],"REVIEW_REQUIRED")
        self.assertEqual(result["completed_rounds"][-1]["novelty"]["new_skill_count"],0)
        self.assertEqual(result["completed_rounds"][-1]["reuse_status"],"EXECUTED_EFFECT_UNVERIFIED")

    def test_split_and_round_caps_are_checked_before_backend(self):
        invalid=[dict(held_out_ids=["task1"]),dict(max_rounds=6),dict(start_round=0),dict(stop_round=6),
                 dict(start_round=True),dict(review_approved="yes")]
        for kwargs in invalid:
            b=Backend()
            with self.assertRaises(ValueError):
                run_iterative_training(TASKS,b,**kwargs)
            self.assertEqual(b.calls,[])
        b=Backend()
        with self.assertRaises(ValueError):
            run_iterative_training(TASKS[:4],b)
        self.assertEqual(b.calls,[])

    def test_task_labels_cannot_inject_freeze_paths(self):
        for label in ("../outside","E:forbidden","folder\\task",".."):
            tasks=deepcopy(TASKS); tasks[0]["assembly_id"]=label; b=Backend()
            with self.assertRaisesRegex(ValueError,"path_free"):
                run_iterative_training(tasks,b)
            self.assertEqual(b.calls,[])


if __name__=="__main__":
    unittest.main()
