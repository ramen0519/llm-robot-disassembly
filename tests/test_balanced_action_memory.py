"""Synthetic evidence-window regression; no policy performance or physics claim."""
from copy import deepcopy
import unittest

from orchestrator.explicit_action_memory import _state_diverse_references


class BalancedActionMemoryTests(unittest.TestCase):
    def fixture(self):
        context = dict(assembly_id="fixture", input_sha256="synthetic")
        req = dict(task_context=context, remaining_parts=["a", "b"],
                   candidate_ids=["a", "b"], pose_catalog=[dict(matrix=None)])
        rows = []
        for i in range(5):
            rows.append(dict(evidence_id="f"+str(i), same_conditions=True, same_task=True,
                source_task_context=deepcopy(context), source_remaining_parts=["a", "b"],
                source_choice=dict(part_id="a"), source_pose_matrix=None,
                source_action_id="action"+str(i), status="UNKNOWN",
                path_success=True, stability_success=False,
                parent_failure_evidence_id="f"+str(i-1) if i else None))
        rows.append(dict(rows[0], evidence_id="success", source_action_id="different",
                         status="PASS", stability_success=True, parent_failure_evidence_id=None))
        return req, rows

    def test_failure_chains_no_longer_starve_equal_relevance_success(self):
        req, rows = self.fixture()
        before = deepcopy(rows)
        legacy = _state_diverse_references(rows, req, 4)
        new = _state_diverse_references(rows, req, 4, balanced=True)
        self.assertTrue(all(r["status"] == "UNKNOWN" for r in legacy))
        self.assertEqual({r["status"] for r in new}, {"PASS", "UNKNOWN"})
        self.assertEqual(rows, before)
        self.assertEqual(len(new), 4)
        identities = {r["evidence_id"] for r in new}
        self.assertTrue(all(r["parent_failure_evidence_id"] in identities
                            for r in new if r["parent_failure_evidence_id"]))

    def test_no_success_fabricated_and_whole_chain_never_exceeds_limit(self):
        req, rows = self.fixture()
        for count in range(7):
            result = _state_diverse_references(rows[:-1], req, count, balanced=True)
            self.assertLessEqual(len(result), count)
            self.assertTrue(all(r["status"] == "UNKNOWN" for r in result))
            ids = {r["evidence_id"] for r in result}
            self.assertTrue(all(r["parent_failure_evidence_id"] in ids
                                for r in result if r["parent_failure_evidence_id"]))

    def test_success_with_different_state_cannot_outrank_exact_state_evidence(self):
        req, rows = self.fixture()
        rows[-1].update(same_conditions=False, same_task=False, source_remaining_parts=["a"])
        result = _state_diverse_references(rows, req, 4, balanced=True)
        self.assertTrue(all(r["status"] == "UNKNOWN" for r in result))


if __name__ == "__main__":
    unittest.main()
