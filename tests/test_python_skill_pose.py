"""Synthetic coordinate tests only. No API or ASAP physics is run here."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_skill_program import request, choice, append_observation, IDENTITY
from orchestrator.python_skill_pose import (
    apply_declared_yaw, check_pose_contract, closest_axis, cube_rotations,
    multiply, source_contract, transform_vector,
)

HANDWRITTEN_TEST_ONLY = {"language": "python_policy_v1", "source": """
def axis(vector, directions):
    best = None
    score = -2
    for name in directions:
        v = directions[name]
        dot = v[0]*vector[0]+v[1]*vector[1]+v[2]*vector[2]
        if dot > score:
            best = name
            score = dot
    return best
def propose(ctx):
    last = ctx["last"]
    if last is None or last["feedback"]["path_success"] is not False:
        return None
    m = last["pose_matrix"]
    a = axis([m[0][2],m[1][2],m[2][2]],ctx["directions"])
    b = axis([m[0][0],m[1][0],m[2][0]],ctx["directions"])
    old = last["choice"]
    c = {"part_id":old["part_id"],"pose_id":old["pose_id"],
         "skill_id":"turn_force_two_leg_v1","direction_id":a,"second_direction_id":b,
         "support_ids":old["support_ids"],"evidence_ids":[last["feedback"]["evidence_id"]]}
    if eligible(c):
        return c
    return None
"""}


def packet(matrix=None, first="zp", second="xp"):
    return {"context": {"automatically_extracted_same_condition_pairs": {"pairs": [{
        "same_state": True, "same_part": True,
        "success_evidence_id": "synthetic_source",
        "after_outcome": {"accepted_for_transition": True},
        "after": {"pose_matrix": matrix or IDENTITY, "skill_id": "turn_force_two_leg_v1",
                  "direction_id": first, "second_direction_id": second},
    }]}}}


def template():
    return append_observation(request(), choice(direction_id="yp", support_ids=["b"]))


class PoseContractTests(unittest.TestCase):
    def test_24_distinct_proper_rotations(self):
        matrices = list(cube_rotations())
        self.assertEqual(len(matrices), 24)
        self.assertEqual(len({json.dumps(m) for m in matrices}), 24)

    def test_source_inverse_is_not_current_inverse(self):
        source = [[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]
        original = packet(source, "xp", "zn")
        before = deepcopy(original)
        contract = source_contract(original)
        self.assertEqual(contract["source_cad_vectors"], [[0,0,1],[1,0,0]])
        self.assertEqual(original, before)
        self.assertFalse(contract["physics_success_implied"])

    def test_correct_handwritten_fixture_passes_all_rotations_not_physics(self):
        req = template()
        old = deepcopy(req)
        result = check_pose_contract(HANDWRITTEN_TEST_ONLY, req, source_contract(packet()))
        self.assertEqual(result["passed"], 24)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["physical_calls"], 0)
        self.assertTrue(result["synthetic"])
        self.assertEqual(req, old)

    def test_fixed_world_policy_rejected_by_coordinate_gate(self):
        code = deepcopy(HANDWRITTEN_TEST_ONLY)
        code["source"] = code["source"].replace(
            'a = axis([m[0][2],m[1][2],m[2][2]],ctx["directions"])', 'a = "zp"').replace(
            'b = axis([m[0][0],m[1][0],m[2][0]],ctx["directions"])', 'b = "xp"')
        result = check_pose_contract(code, template(), source_contract(packet()))
        self.assertEqual(result["passed"], 1)
        self.assertFalse(result["accepted"])

    def test_no_matching_source_pair_is_not_invented(self):
        value = packet()
        value["context"]["automatically_extracted_same_condition_pairs"]["pairs"] = []
        with self.assertRaisesRegex(ValueError, "one_bound_source_pair"):
            source_contract(value)

    def test_unbound_feedback_rejected(self):
        with self.assertRaisesRegex(ValueError, "bound_path_failure"):
            check_pose_contract(HANDWRITTEN_TEST_ONLY, request(), source_contract(packet()))

    def test_yaw_rotates_pose_translation_and_force_without_mutation(self):
        req = template()
        req["pose_catalog"][0]["matrix"][0][3] = 2
        prepared = {"request": req, "choice": choice(direction_id="yp"),
                    "context": {"kept": "solver settings"}, "raw": {"kept": "historical physics"}}
        before = deepcopy(prepared)
        new = apply_declared_yaw(prepared, 90)
        m = new["request"]["pose_catalog"][0]["matrix"]
        self.assertEqual([m[0][3], m[1][3], m[2][3]], [0,2,0])
        self.assertEqual(new["choice"]["direction_id"], "xn")
        self.assertEqual(new["context"]["kept"], "solver settings")
        self.assertEqual(new["request"]["episode_feedback"], [])
        self.assertEqual(new["raw"], prepared["raw"])
        self.assertEqual(prepared, before)

    def test_invalid_yaw_and_nonaxis_vector_rejected(self):
        for value in (0, 45, True, "90"):
            with self.assertRaisesRegex(ValueError, "declared_yaw"):
                apply_declared_yaw({}, value)
        with self.assertRaisesRegex(ValueError, "axis_aligned"):
            closest_axis([0.7, 0.7, 0])


if __name__ == "__main__":
    unittest.main()
